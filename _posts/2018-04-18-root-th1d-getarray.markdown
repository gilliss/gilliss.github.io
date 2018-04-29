---
layout: post
title:  "An Old ROOT Question"
date:   2018-04-18 16:24:29 -0800
categories: root pyroot th1d c++
---

I recently needed to pull some data from a C++ [ROOT][root-site] file into a Python analysis. Unfortunately, the data was stored in a [TH1D][root-th1d] and I didn't want to resort to a loop over bins, querying each bin for its contents. Using [PyROOT][pyroot-site], this would be

{% highlight python %}
# this snippet pulls out the contents of bins 1 through nBins,
# but ignores the underflow and overflow bins
nBins = aTH1D.GetNBinsX()
for bin in range(1, nBins + 1):
  arrayOfData.append(aTH1D.GetBinContent(bin))
{% endhighlight %}

but there ought to be a method that returns all of the bin contents in one command. In searching, I came across [this old post][root-post] from the RootTalk forum, where it seems members were discussing the same topic. The last reply in the thread points out a `GetArray()` method that returns a pointer to a data member of the TH1D class, presumably an array of the bin contents. The `GetArray()` method seemed promising, but the reply left open the following questions.

```
"Of course I am leaving for roottalk and for ROOT team the question
how one can figure out that [the returned] Float_t array is those bins. What about
overflow and underflow bins. Are they present in this array or not,
If yes where ?

  Does Int_t   TH1F::GetNbinsX()  returns the same number as
       Int_t   TArray::GetSize();"
(replace floats with doubles for relevance to TH1D)
```

In short, the reply asks if `GetArray()` indeed returns the bin contents we are looking for, and if those bins include the underflow and overflow data. We can do some surfing through the source code to find these answers and finally lay to rest this question, which appears to have gone unanswered since 1998.

# GetArray()

Our first stop is the [TH1D documentation][root-th1d], where we see that TH1D does indeed have the public member function `GetArray()`, inherited from TArrayD. This inheritance is established in the [class definition of TH1D][root-th1d-classdef] where TH1D inherits the [TArrayD public data member][root-tarrayd-farray], `Double_t *fArray`, and its corresponding [public "get" method][root-tarrayd-getarray], `Double_t *GetArray()`.

# The shape of fArray

The next question is what data `fArray` points to in the context of a TH1D. To answer this, we turn to the TH1D constructor, which, upon instantiation of a TH1D object, would presumably associate a suggestive shape or placeholder data with `fArray`. The TH1D constructor, at [TH1.cxx:9485][root-th1d-constructor], does just that with `TArrayD::Set(fNcells)`. What happens here is,

1. The user instantiates the TH1D with some number of bins `nbins` (which does not include under- and overflow bins), and a histogram range, from `xlow` to `xup`.
2. `nbins` and the range get passed into the inherited [TH1 constructor][root-th1-constructor], which loads its protected `TAxis fXaxis` member with those values. In particular `fNbins = nbins`.
3. The TH1 constructor queries `fAxis` for the number of bins and sets the TH1 integer data member `fNcells` to `fNbins + 2`. This is `fNcells = fXaxis.GetNbins() + 2`.
4. The value of `fNcells` is then used in the [TH1D constructor][root-th1d-constructor] to set the length of `fArray` via `TArrayD::Set(fNcells)`.

What is `fNcells`? We know its value is the number of user-set bins plus two. This is consistent with the [comment near its declaration][root-th1-fncells], "number of bins(1D) ... +U/Overflows," which reveals that the two extra bins are for under- and overflow data.

So, we've established that `GetArray()` returns `fArray` which includes both the middling, and the under- and overflow bins.

# What's in fArray?

Can we confirm that `fArray` indeed holds the contents of the histogram and not some other data? Let's check the command for setting a histogram's bin contents, and see where it puts the data. TH1D inherits its `SetBinContent()` command from [`TH1::SetBinContent()`][root-th1-setbincontent], which calls on [`TH1::UpdateBinContent()`][root-th1-updatebincontent]. This second method is [overridden by TH1D][root-th1d-updatebincontent] to set `fArray[bin] = content`. So, `fArray` is indeed the object that gets filled with and holds the histogram's bin content.

# What's the difference between all these "get" methods?

Now to answer the remaining question.

```
"Does Int_t   TH1F::GetNbinsX()  returns the same number as
     Int_t   TArray::GetSize();"
```

From the TH1D documentation, we see that `GetNbinsX()` is a public member function inherited from TH1. The function returns `fXaxis.GetNbins()`, which itself returns `fNbins`. Recall the exploration of the TH1 constructor, above, which revealed that `fNbins = nbins = fNcells - 2`.

The `GetSize()` method is a public member function of TH1D which it inherits from the TArray class. The method returns TArray's integer data member `fN` which corresponds to the number of elements in `fArray`. We encountered `fN` earlier-- it was the value that got set by `TArrayD::Set()`; `Set(fNcells)` yields `fN = fNcells`.

# Summary

In summary, the three base classes TArray, TH1, and TAxis, each retain a data member describing the binning of a histogram. TArray holds `fN` which equals the number of bins, including under- and overflow. TH1 holds `fNcells`, which is equivalent to `fN`. Finally, TAxis holds `fNbins` which equals the number of bins, excluding under- and overflow. With this information, we can see that `TH1D::GetNbinsX()` returns the user defined number of bins, which is two less than the total number of bins in the data array. The total number of bins, which includes the two extra bins for under- and overflow, is returned by `TArray::GetSize()`, or equivalently by `TH1D::GetNcells()`.

After searching the source code to produce the information above, I believe a brief clarification of these topics is warranted in the ROOT documentation. I would
suggest adding to the [preamble of the TH1 class][root-th1-preamble], in the section titled "Convention for numbering bins." The added text could read,

```
Bin contents for a histogram object, including under- and overflow, are stored in a TArray.
The size of this array is recorded in the data members `TArray::fN` and TH1::fNcells.
The number of bins displayed on the histogram axis, which excludes under- and overflow,
is recorded in the histogram's inherited data member TAxis::fNbins.
The relationship between the three values should be fN = fNcells = fNbins + 2,
and all three values are accessible from the histogram object via the inherited methods GetSize(), GetNcells(), and GetNBinsX().
The three methods return fN, fNcells, and fNbins, respectively.
```

[root-site]: https://root.cern.ch/
[pyroot-site]: https://root.cern.ch/pyroot
[root-post]: https://root.cern.ch/root/roottalk/roottalk98/2318.html
[root-th1d]: https://root.cern.ch/doc/master/classTH1D.html
[root-th1d-classdef]: https://root.cern.ch/doc/master/TH1_8h_source.html#l00610
[root-th1d-constructor]: https://root.cern.ch/doc/master/TH1_8cxx_source.html#l09485
[root-th1d-updatebincontent]: https://root.cern.ch/doc/master/TH1_8h_source.html#l00640
[root-tarrayd-farray]: https://root.cern.ch/doc/master/TArrayD_8h_source.html#l00030
[root-tarrayd-getarray]: https://root.cern.ch/doc/master/TArrayD_8h_source.html#l00044
[root-th1-constructor]: https://root.cern.ch/doc/master/TH1_8cxx_source.html#l00639
[root-th1-fncells]: https://root.cern.ch/doc/master/TH1_8h_source.html#l00086
[root-th1-setbincontent]: https://root.cern.ch/doc/master/TH1_8cxx_source.html#l08497
[root-th1-updatebincontent]: https://root.cern.ch/doc/master/TH1_8cxx_source.html#l08725
[root-th1-preamble]: https://root.cern.ch/doc/master/TH1_8cxx_source.html#l00181
