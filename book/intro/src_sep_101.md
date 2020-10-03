What is Source Separation?
==========================

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/FTQbiNvZqaY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

```{dropdown} Video not working?
If you are having access issues, here is a OneDrive mirror to the full video.
<a href=""></a>

Alternatively the text on this page covers the same material as the video.
```

Source Separation is the process of isolating individual sounds in an auditory
mixture of multiple sounds. {cite}`vincent2018audio` We call each sound heard
in a mixture a _source_.
For example, we might want to isolate a singer from the background music to make
a karaoke version of a song or isolate the bass guitar from the from the rest of
the band so a musician can learn the part. Put another way, given a mixture of
multiple sources, how can we recover only the source signals we're interested in?


Mathematically, we assume that a mixture signal $y(t)$ is composed of $N$ sources, 
$x_n(t)$, for $n=1...N$, such that

$$
y(t) = \displaystyle\sum_{i=1}^{N} x_i(t).
$$

Given only $y(t)$, the goal of a source separation system is to recover one or
more $x(t)$'s. 

It is often the case that there are more sources within
the mixture than there are mixture signals. Because of this, source separation
is considered an _underdetermined problem_, meaning that there are fewer observations
(_i.e._, the mixture) than there are required outcomes (_i.e.,_ the desires
source(s)). For example, if a stereo mixture contains a recording of a piano quartet
(_e.g._ a piano, violin, viola, and cell) , for any desired source in the
mixture we only have two observations (each channel of the stereo mix),
therefore source separation would be a useful tool to isolate one of the
sources (_e.g._, just the piano).


In this tutorial we will be focusing on music separation, or the process of 
isolating at least one musical instrument or singer from a musical mixture that
contains one or more other musical instruments or singers. Music is seen as a
distinct problem from other types of source separation because there are many
factors that make it uniquely challenging {cite}`cano2018musical`:

- Sources in music are _highly correlated_, meaning that all of the sources usually 
  change together at the same time. For example, in a rock band if the bass guitar
  changes its note at the start of a new measure, it is likely that the other
  instruments will change as well.
  
- Music is mixed and processed in ways that are aphysical and non-linear. 
  Contemporary recording practices are such that any given source in a mixture
  might never occur naturally in the real world. Reverb, filtering, and other
  non-linear signal processing techniques all make music separation difficult,
  and yet these are tools that recording engineers and musicians routinely use
  to create music. This is a problem because we rarely, if ever, know what
  processing was applied to any source or the whole mixture.
  
- If the result of a music source separation system is used in an end-user system,
  the bar for quality is much higher. As we will see, there are now many systems
  for musicians and sound engineers that incorporate source separation. As opposed
  to source separation being used as an intermediate step for another auditory
  process, source separation is the end goal and being listened to by users that
  might expect high quality results. Therefore it is paramount that the results
  of the system sound good enough for those users.[^fn1]


```{note}
What specific sounds constitute a source is highly dependent on the application
and desired output. Some the approaches make explicit assumptions about what a
source actually _is_, whereas others do not. 
For instance, a dog barking might be background noise in one scenario, and therefore
ignored, but for a sound event detection system it might be a source of interest.
Source separation research has largely assumed that sources and source types are
known a priori.
```


## Why Use Source Separation?

There are many reasons to study source separation. One might be interested in
using existing methods to enhance a downstream task, or one might be interested
in source separation as a pursuit in itself. 

There are many demonstrated uses for music source separation within the field of
Music Information Retrieval (MIR). In many scenarios, researchers have discovered
that it is easier to process a isolated sources than mixtures of those sources.
For example, source separation has been used to enhance:

- automatic music transcription {cite}`plumbley2002automatic,manilow2020simultaneous`, 
- lyric and music alignment {cite}`fujihara2006automatic`, 
- musical instrument detection {cite}`heittola2009musical`, 
- lyric recognition {cite}`mesaros2010automatic`, 
- automatic singer identification {cite}`weninger2011automatic,hu2015separation,sharma2019importance`, 
- vocal activity detection {cite}`stoller2018jointly`, 
- fundamental frequency estimation {cite}`jansson2019joint`, and 
- understanding the predictions of black-box audio models. {cite}`haunschmid2020towards,haunschmid2020audiolime`

Additionally, source separation has long been seen as an inherently worthwhile
endeavor on its own merits, with many thousands of research papers appearing over
the past few decades and more appearing every year.

Whether you plan to create new source separation research or use existing methods
to advance your own work, we hope this tutorial will provide you with a solid
foundation understanding this field.

## Related Fields of Research

### Speech Separation and Enhancement
By far the most popular application of Source Separation technology is for
separating two or more people talking at the same time. This is called
_Speech Separation_, and many of the technologies we discuss in this tutorial
were initially developed for speech and later expanded to music.

A similar thread of research is _Speech
Enhancement_ where the goal is to isolate and clean up speech from background
noise. The problem of can Speech Enhancement be formulated as a source
separation problem where one of the sources is always a speaker and the other
source is always noise (the "noise" source can either be modeled or ignored
completely).

These days, almost any advances from Speech Separation or Speech Enhancement
can be used for music source separation with slight modifications.


### Beamforming

[Beamforming](https://en.wikipedia.org/wiki/Beamforming) is a method of using
the spatial orientation of an array of microphones to separate sources. The core
idea is to take advantage of the interference patterns in signals to isolate
specific sources. Beamforming is commonly used to detect human speech in smart
speaker products like Amazon's Alexa, but it has musical applications in
[iZotopes Spire](https://www.izotope.com/en/products/spire-studio.html).

While this field is beyond the expertise of the authors, we will be touching on
some methods that do incorporate spatial features for source separation.


## A Historical Perspective

Source Separation is a problem that has been studied for decades. Although we won't
be able to cover everything in detail, this tutorial will provide a brief overview
of methods that we think provide a representative demonstration of important concepts and
provide practical value.

In general, modern source separation approaches fall into two broad categories: blind
source separation approaches and data-driven approaches. Blind source separation
approaches are algorithms that make explicit assumptions about the auditory scene
upon which they operate


[^fn1]: Some creative applications might not have such strict demands; when using
 source separation to create remixes for instance, the artifacts might be masked
 by other sources in the mix.
 