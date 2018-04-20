---
layout: post
title:  "Spectral Fits with PyMC3"
date:   2018-03-21T23:51:06+00:00
categories: pymc3 fitting
---


In this post, we'll explore some basic implementations of a mixture model in PyMC3. Namely, we write out binned and unbinned fitting routines for a set of data drawn from two gaussian processes.

To start, we imagine an experiment that repeatedly observes one random variable $X$. For any one observation, the realized value $x$ of this $X$ could have originated from one of several— say, $n$— processes. In this way, the probability distribution $f(x &#124; \theta)$, for $X$, is given by the combination of several sub-distributions $\{p_i\}$, with relative weights $\{w_i\}$.

$$ X \sim f(x|\theta) = \sum_{i=1}^{n} w_i p_i(x|\theta_i) $$

For our experiment, let's have $x$ be a scalar drawn from the mixture distribution,

$$ X \sim f(x|\theta) = w_1 \text{Normal}(x|\mu_1, \sigma_1) + w_2 \text{Normal}(x|\mu_2, \sigma_2). $$

In this case, $\theta = \{\theta_1, \theta_2\} = \{w_1, \mu_1, \sigma_1, w_2, \mu_2, \sigma_2\}$ is the set of parameters upon which $f$ is conditioned.

# Generating some data

We simulate 3000 observations for this experiment as follows.


{% highlight python linenos %}
# import numpy for numerical operations on arrays and scipy for random number generation
import numpy as np
import scipy.stats as stats

# draw the observations from two separate Normal distributions
mu1, mu2 = -1., 5.
sd1, sd2 = 2., 2.
nObs1 = 1000
nObs2 = 2 * nObs1
obsSource1 = stats.norm(loc = mu1, scale = sd1)
obsSource2 = stats.norm(loc = mu2, scale = sd2)
data = np.append(obsSource1.rvs(size = nObs1), obsSource2.rvs(size = nObs2))
nObs = len(data)
print(nObs)

>> 3000
{% endhighlight %}

We can histogram the observations in `data` to visualize what was collected by our imaginary experiment.


{% highlight python linenos %}
# import matplotlib for visualizations
import matplotlib.pyplot as plt
%matplotlib inline

# set the horizontal range of plots
rangeX = (-10, 10)

# histogram the data, with 'rice' binning proportional to cubed root of data size
plt.figure()
plt.title('Data')
plt.xlabel('x')
plt.ylabel('counts')
plt.xlim(rangeX)
h_bin_heights, h_bin_edges = plt.hist(data, bins = 'rice', range = rangeX, density=False)[:2]

# get some info about the binning
nBinsX = len(h_bin_heights)
binWidX = h_bin_edges[1] - h_bin_edges[0]

# we will later want the bin centers, so lets calculate them
h_bin_centers = h_bin_edges[:-1] + binWidX/2.
{% endhighlight %}

![png](/assets/2018-03-21-binned-unbinned-spectral-fits-pymc3_files/fig_7_0.png)

# Modeling the data

We now propose a mixture model to explain the data. For simplicity, we claim to have a good idea of the location and spread of each gaussian. So, we posit

$$ X \sim f(x|w_1,w_2) = w_1 \text{Normal}(-1, 2) + w_2 \text{Normal}(5, 2). $$

We are only uncertain of the weights of the two distributions in our mixture model, which we now estimate through maximum likelihood methods, with the help of PyMC3.

# A binned fit

First we implement a binned maximum likelihood (ML) fit where the goal is to find the weights ($w_1$, $w_2$) that most accurately predict the heights of each bin in the histogrammed data. The fit proceeds by iteratively proposing weights, plugging them into a model for the bin heights, and assessing how closely the observed bin heights match the modeled bin heights.

We assume the observations that populate the bins of the histogram were Poisson distributed over the duration of the experiment. As such, we evaluate the agreement between the modeled and observed bin heights through a Poisson likelihood function. Below is our model for the `nBinsX` bin heights.

The predicted height of the $k^{\text{th}}$ bin:
$$ m_k = w_1 \text{Normal}(x_k|-1, 2) + w_2 \text{Normal}(x_k|5, 2) $$

The likelihood that the set of observed bin heights $\{h_k\}$ matches the set $\{m_k\}$ predicted by $w_1$ and $w_2$:

$$ L = \prod_{k=1}^{\texttt{nBinsX}} \text{Poisson}(h_k|m_k) $$

Note that, since there are two unknown parameters, we are essentially searching for a maximum in the two-dimensional space of the likelihood function. We could search through this space, using brute force, by setting up a grid of $w_1$ and $w_2$ values and evaluating $L$ at everypoint, but this strategy would become inconvenient for problems with higher-dimensional likelihood functions.

Instead, we can intellegently sample points from the likelihood function using Markov Chain Monte Carlo (MCMC) methods. Setting this intellegent sampling up in PyMC3, we have the following.


{% highlight python linenos %}
# import pymc for likelihood modeling and MCMC
import pymc3 as pm

# define the mixture model
def mix(x, w1, w2):
    norm1 = obsSource1.pdf(x)
    norm2 = obsSource2.pdf(x)
    return w1 * norm1 + w2 * norm2

# setup a pymc3 model instance
with pm.Model() as model:
    # define unknown model parameters that will be floated, each with a uniform prior
    w1 = pm.Uniform('w1', lower = 0., upper = len(data), testval = 10)
    w2 = pm.Uniform('w2', lower = 0., upper = len(data), testval = 10)

    # define the likelihood function
    # the observed values here are the bin heights, h_bin_heights
    L = pm.Poisson('L', mu = mix(h_bin_centers, w1, w2), observed = h_bin_heights, testval = h_bin_heights)

    # set the sampler and feed the floated parameters
    step = pm.Metropolis([w1, w2])

    # draw samples
    trace = pm.sample(draws = 10000, step = step, chains = 1)

>> Sequential sampling (1 chains in 1 job)
>>    CompoundStep
>> >Metropolis: [w2_interval__]
>> >Metropolis: [w1_interval__]
>> 100% ... 10500/10500 [00:03<00:00, 2886.57it/s]
{% endhighlight %}

Let's use the trace object from PyMC3 to check out the results of the MCMC search.

{% highlight python linenos %}
# import pandas to readout the trace object
import pandas as pd

# view a summary of the MCMC trace
df_summary = pm.summary(trace)
print(df_summary)

>>            mean         sd  mc_error      hpd_2.5     hpd_97.5
>> w1   527.041567  18.323054  0.381805   490.273632   561.412043
>> w2  1099.525432  25.340886  0.544281  1050.710203  1149.119203
{% endhighlight %}

We see that the ratio of weights is $w2/w1 \approx 2$, as we'd expect from our generating of the data; we drew twice as many events from `obsSource2` as we did from `obsSource1`.

From the trace, we can also view the MCMC steps for each weight and histogram those steps to obtain each weight's
marginal likelihood distribution (or marginal *posterior* distribution, if we had used non-uniform priors).


{% highlight python linenos %}
# have pymc3 plot the chains and marginal likelihood distributions
pm.traceplot(trace);
{% endhighlight %}


![png](/assets/2018-03-21-binned-unbinned-spectral-fits-pymc3_files/fig_18_0.png)


# Binned fit results

Let's view the joint likelihood distribution.


{% highlight python linenos %}
# create a hist2d of the joint likelihood distribution
plt.figure()
plt.title('Joint Likelihood from Binned Fit')
plt.xlabel('w1')
plt.ylabel('w2')
h2_image = plt.hist2d(trace['w1'], trace['w2'], bins=100, cmap = 'viridis')[3]
plt.colorbar(h2_image).set_label('Joint Likelihood')
{% endhighlight %}


![png](/assets/2018-03-21-binned-unbinned-spectral-fits-pymc3_files/fig_21_0.png)


Taking the mean of each weight's marginal likelihood as a reasonable guess for the true weight, we can evaluate how closely the ML model matches the data.


{% highlight python linenos %}
# pull out the mean of the weight results
w1 = df_summary.loc['w1']['mean']
w2 = df_summary.loc['w2']['mean']

# incorporating the guessed weights, see if the mixture model predicts the correct number of observations
nObs1_model = (1/binWidX) * w1 * 1 # 1 represents the integral of the nObs1 PDF
nObs2_model = (1/binWidX) * w2 * 1 # 1 represents the integral of the nObs2 PDF
print('predicted nObs = %.2f + %.2f = %.2f' % (nObs1_model, nObs2_model, nObs1_model + nObs2_model))
print('actual nObs = %d' % nObs)

>> predicted nObs = 975.03 + 2034.12 = 3009.15
>> actual nObs = 3000
{% endhighlight %}


{% highlight python linenos %}
# prepare visualizations of the scaled mixture model
linspaceX = np.linspace(rangeX[0], rangeX[1], nBinsX)
model1 = w1 * obsSource1.pdf(linspaceX)
model2 = w2 * obsSource2.pdf(linspaceX)

# plot the scaled mixture model against the data
plt.figure()
plt.title('Data and Scaled Mixture Model')
plt.xlabel('x')
plt.ylabel('counts')
plt.xlim(rangeX)
plt.hist(data, bins = 'rice', range = rangeX, density=False, label = 'data')
plt.plot(linspaceX, model1, color = 'xkcd:orange', linewidth=3, dashes = [2, 2, 2, 2],label = '1 Model')
plt.plot(linspaceX, model2, color = 'xkcd:blue', linewidth=3, dashes = [2, 2, 2, 2], label = '2 Model')
plt.plot(linspaceX, model1 + model2, color = 'xkcd:red', label = 'Total Model')
plt.legend(loc = 'upper left');
{% endhighlight %}


![png](/assets/2018-03-21-binned-unbinned-spectral-fits-pymc3_files/fig_24_0.png)


# An unbinned fit

We can once more estimate the weights that allow our mixture model to best match the data. This time, however,
we will not first reduce the data by histrogramming; we will instead use each of the individual observations. This procedure is known as an unbinned fit.

Our likelihood function in this unbinned case will not make use of bin heights. Instead, the unbinned ML fit proceeds by interatively proposing weights, plugging them into the mixture model, and then evaluating how well the observed values $x$ align with the mixture model. For instance, if many of the $\{x_i\}$ lie in the tails of the mixture model (where $f(x &#124; w_1,w_2)$ is small), there is poor alignment. Likewise, if few of the $\{x_i\}$ lie in the bulk of the mixture model (where $f(x &#124; w_1,w_2)$ is large), there is poor alignment. We're looking for the happy medium.

The likelihood function assessing the match between the model and the 3000 observations is

$$ L = \prod_i^{3000} f(x_i|w_1,w_2) $$

We can set this up in PyMC3 as follows.


{% highlight python linenos %}
# setup pymc3 model and sampling
with pm.Model() as model:
    # define the weights of the mixture model components, each having a flat prior
    # a two-dimensional flat Dirichlet distribution (alpha = 1) is used
    w = pm.Dirichlet('w', a = np.array([1., 1.]))

    # define the mixture model components
    comp_dists = [
        pm.Normal('norm1', mu = mu1, sd = sd1, shape = 1, testval = mu1).distribution,
        pm.Normal('norm2', mu = mu2, sd = sd2, shape = 1, testval = mu2).distribution
    ]

    # define the likelihood function using pm.mixture
    # the observed values here are the 3000 observations, data
    L = pm.Mixture(
        'L',
        w = w,
        comp_dists = comp_dists,
        observed = data
        )

    # instantiate sampler
    step = pm.Metropolis(w)

    # draw samples
    trace = pm.sample(draws = 10000, step = step, chains = 1)

>> Sequential sampling (1 chains in 1 job)
>> CompoundStep
>> >Metropolis: [w_stickbreaking__]
>> >NUTS: [norm2, norm1]
>> 100% ... 10464/10500 [00:22<00:00, 463.85it/s]
>> 100% ... 10500/10500 [00:22<00:00, 463.57it/s]
{% endhighlight %}

We use the trace object from PyMC3 to check out the results of the MCMC search.

{% highlight python linenos %}
# view a summary of the MCMC trace
df_summary = pm.summary(trace)
print(df_summary)

# checkout the ratio of the w2 and w1 means
print('w2/w1 = %.2f' % (df_summary.loc['w__1']['mean'] / df_summary.loc['w__0']['mean']))

# have pymc3 plot the chains and marginal likelihood distributions
pm.traceplot(trace);

>>               mean        sd  mc_error   hpd_2.5  hpd_97.5
>> norm1__0 -0.983890  1.978366  0.018363 -4.940810  2.797312
>> norm2__0  5.012095  1.972506  0.019112  1.221044  8.958915
>> w__0      0.325619  0.009638  0.000227  0.306269  0.344305
>> w__1      0.674381  0.009638  0.000227  0.655695  0.693731
>> w2/w1 = 2.07
{% endhighlight %}


![png](/assets/2018-03-21-binned-unbinned-spectral-fits-pymc3_files/fig_29_1.png)


Again we see that the ratio of weights is $w2/w1 ≈ 2$, as we expect. The `traceplot` displays histograms of MCMC samples from the two Normal components of the mixture models (labeled as `norm1` and `norm2`), and the marginal likelihood distributions for each weight (with $w2$ in orange and $w1$ in blue).

# Unbinned fit results

We can view the joint likelihood distribution for the weights. The distribution is narrow and highly correlated because the `pm.mixture` environment constrains the weight samples to obey $w1 + w2 = 1$ (see `pymc3/mixture.py::logp()`).


{% highlight python linenos %}
# create a hist2d of the joint likelihood distribution
plt.figure()
plt.title('Joint Likelihood from Unbinned Fit')
plt.xlabel('w1')
plt.ylabel('w2')
h2_image = plt.hist2d(trace['w'][:,0], trace['w'][:,1], bins=100, cmap = 'viridis')[3]
plt.colorbar(h2_image).set_label('Joint Likelihood')
{% endhighlight %}


![png](/assets/2018-03-21-binned-unbinned-spectral-fits-pymc3_files/fig_33_0.png)


Taking the mean of each weight's marginal likelihood as a reasonable guess for the true weight, we can evaluate how closely the ML model matches the data.


{% highlight python linenos %}
# pull out the mean of the weight results
w1 = df_summary.loc['w__0']['mean']
w2 = df_summary.loc['w__1']['mean']

# incorporating the estimated weights, we see the mixture model is normalized to one
nObs1_model = w1 * 1 # 1 represents the integral of the nObs1 PDF
nObs2_model = w2 * 1 # 1 represents the integral of the nObs2 PDF
print('predicted nObs = %.2f + %.2f = %.2f' % (nObs1_model, nObs2_model, nObs1_model + nObs2_model))

# so we scale by the known number of observations to scale the model to the data
print('scaled predicted nObs = %d (%.2f + %.2f) = %.2f'
      % (nObs, nObs1_model, nObs2_model, nObs*(nObs1_model + nObs2_model)))
print('actual nObs = %d' % nObs)

>> predicted nObs = 0.33 + 0.67 = 1.00
>> scaled predicted nObs = 3000 (0.33 + 0.67) = 3000.00
>> actual nObs = 3000
{% endhighlight %}

To visualize the model against the histogram of the data, we can either scale the model up to match the histogram— as we did in the previous cell—, or scale the histogram down to match the model. This scaling is needed, since our model would not be able to match the original bin heights— it knows nothing of them. The model knows only of the *density* of the data. Below, we scale the histogram to unit integral. We also plot a subset of the 3000 individual observations to convey how those observations are distributed.


{% highlight python linenos %}
# a normalized histogram of the data
plt.figure()
plt.title('Normed Data, Individual Observations, Mixture Model')
plt.xlabel('x')
plt.ylabel('density')
plt.xlim(rangeX)
plt.hist(data, bins = 'rice', range = rangeX, density = True, label = 'normed data')

# plot the mixture model distribution against the normalized data
linspaceX = np.linspace(rangeX[0], rangeX[1], nBinsX)
model1 = w1 * obsSource1.pdf(linspaceX) # binWidX * nObs *
model2 = w2 * obsSource2.pdf(linspaceX)
plt.plot(linspaceX, model1, color = 'xkcd:orange', linewidth=3, dashes = [2, 2],label = '1 Model')
plt.plot(linspaceX, model2, color = 'xkcd:blue', linewidth=3, dashes = [2, 2], label = '2 Model')
plt.plot(linspaceX, model1 + model2, color = 'xkcd:red', label = 'Total Model')

# plot some of the observations along the horizontal axis
data_subset = data[::30] # take every 30th observation
plt.plot(data_subset, np.zeros(len(data_subset)), 'y|', ms=20, label = 'observations')
plt.legend(loc = 'upper left');
{% endhighlight %}


![png](/assets/2018-03-21-binned-unbinned-spectral-fits-pymc3_files/fig_37_0.png)


# Discussion

We've seen here that there are multiple strategies for determining ML model parameters. One can match the model to the binned data, or one can match the model to the individual observations.

In these examples, we held the location and spread of the mixture components to be fixed. This was for the sake of simplicity, and additional parameters can be floated in principle. Since we floated only the weights, we saw that the ML values of $w1$ and $w2$ can vary with fit procedure, and in the binned case, with bin width. However, the *relative* ML values of $w1$ and $w2$ remained the same regardless of approach.

We've also seen that there is a tradeoff between binned and unbinned procedures; just look at the likelihood functions. In the unbinned case, each evaluation of the likelihood function (for a proposed $w1$ and $w2$) required a product of 3000 terms, with one term for each observation. In the binned case, a product of only `nBinsX` terms was needed, with one term for each bin. The histogramming acted to reduce the data to a lesser number of effective observations. So, a binned approach can save on computation, but may be less sensitive to finer details in the distribution of the observations.

We can also take a look at the MCMC output for the binned and unbinned scenarios. In each case, the chains underwent the same number of steps. However, in the binned case, the MCMC process took `00:03 s` at `2886.57 iterations/s` while for the unbinned case, the process took `00:22 s` at `463.57 iterations/s`.
