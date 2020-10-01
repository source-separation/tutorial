Overview
========

In plain language, training a deep separation model comprises of the following steps:

1. Grab some data for which you know the mixture and sources.
2. Pass the mixture through the deep model.
3. Compare the output of the model (an estimated source) with the ground truth known isolated source.
4. Adjust the parameters of the deep model so that it gets better at separation.
5. Go back to 1 and repeat until the deep model is deemed "converged".

To those unfamiliar with deep learning, or with audio, the steps above may 
raise more questions than they answer:

- How do I get data where I know the mixture and the sources?
- What does it mean to pass a "mixture" through a deep model? What is the actual data?
- How do I compare the output of the model to the actual isolated source?
- How do I make the model better, given that information?
- How do I know when I'm done?!

If all or some of these questions popped into your head, no worries! 
That's what this section is for! If you already know the answers, then go ahead
and skip ahead.

## Data

In the last section about handling data, we already looked at how we can use tools like
Scaper to construct complex mixtures on the fly with ground truth sources. Almost all
deep learning based methods in source separation are trained with _synthetic_ mixtures, with
some exceptions {cite}`wisdom2020unsupervised,seetharaman2019bootstrapping,seetharaman2020bootstrapping,tzinis2019unsupervised,drude2019unsupervised`.
Synthetic mixtures are created by taking known isolated sources and mixing them
together. As we saw in the previous section, this creates a mixture where the sources that go into it
are known.

The next major question is how does a mixture get represented as the _input_ to a deep learning
model? The choice of representation is very important - think of it as the very first
bottleneck in your pipeline. If the representation you choose is _lossy_, that is to say
that information about the original audio mixture is lost in the transformation, then any
problem that requires the lost information will not be solvable by the deep network.
Let's look at an illustrative example.

(TODO: Build example)

This mixture contains two sources - a synthetic click and a sine tone. Let's represent it using
a time-frequency based representation, which we covered in (TODO). A time-frequency representation
represents the audio via the energy that it has in each frequency band over time. The most common
implementation is via the Short-Time Fourier Transform, or STFT. The STFT has three main parameters:
the window length, the hop length, and the window function. For now, let's look at just the first two. 
These control the resolution in time (hop length) and the
resolution in frequency (window length) for the STFT. If you choose these parameters poorly, 
separation can become either very difficult or even impossible. 

```{note} Non-STFT based representations
Representations other than the STFT exist that can be helpful in analyzing many different types
of audio. Namely: the common fate transform {cite}`stoter2016common`, the _multi-resolution- 
common fate transform {cite}`pishdadian2018multi`, the deep scattering transform (TODO),  
wavelet transforms (TODO), and the Constant-Q transform (TODO), to name a few. Additionally,
post-processing techniques also exist, such as PCEN {cite}`lostanlen2018per`.
```

Let's look at how separation performance varies given different STFT parameters for separating these two
sources. For this experiment, we'll keep our window length very large at 8192, and vary our hop length from 1024
up to 8192.

