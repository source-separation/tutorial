Introduction to Classic Approaches
==================================

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/FTQbiNvZqaY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

```{dropdown} Video not working?
If you are having access issues, here is a OneDrive mirror to the full video.
<a href=""></a>

Alternatively the text on this page covers the same material as the video.
```


Now that we know the basics of spectrograms, masking, and phase, we have all of
the building blocks for diving into source separation.

To get a feel for the mechanics of source separation, we will begin learning
some of the "classic" algorithms.[^fn1]

This serves a few purposes:

  1. The classic algorithms do not suffer from the same lack of explainability
     that neural nets do (_i.e._, neural nets are often considered
     "black-boxes"), and therefore we will explore the general
     mechanics of source separation using them.
  2. These methods do not require heaps of data and tons of computing power to
     produce satisfactory results when their assumptions are met. Unlike deep
     nets, you can run them relatively quickly using commodity hardware.
  3. Despite these methods not having the same buzz surrounding them as deep
     nets do, we believe they are still useful in many situations. As we will
     see, when the assumptions of the algroithm matches the contents of our
     mix the results can sound quite good.

While it may be great to use the latest and greatest, sometimes the simpler
tool is the right one for the job. Don't discount these methods, especially
when many of them are easy to use, as we will demonstrate. 

One thing we will stress here and throughout the tutorial is the importantance
of understanding the assumptions (or inductive biases) that source separation
approaches make. With these classic algorithms, the assumptions are easy to
understand. Based on our assumptions about the types of mixtures we will see,
we can make very explicit design decisions that affect how mixtures are
processed to get out sources. As we will see later, this is also true for
deep nets, but the assumptions are a little bit harder to parse. More on that
later...

### Now, let's dive in to source separation!


 
[^fn1]: We understand that some people might take issue with us declaring all 
  non-deep learning source separation approaches "classic", but that's where
  we are now. ¯\_(ツ)_/¯