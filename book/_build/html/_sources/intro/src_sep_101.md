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
mixture of multiple sounds. We call each sound heard in a mixture a _source_.
For example, we might want to isolate a singer from the background music to make
a karaoke version of a song or isolate the bass guitar from the from the rest of
the band so a musician can learn the part. Put another way, given a mixture of
multiple sources, how can we recover only the source signals we're interested in?


Mathematically, we assume that a mixture signal $y(t)$ is composed of $N$ sources, 
$s_n(t)$, with corresponding mixing coefficients, $a_n$, for $n=1...N$, such that

$$
y(t) = \displaystyle\sum_{i=1}^{N} a_i s_i(t).
$$

Given only $y(t)$, the goal of a source separation system is to recover one or
more $s(t)$'s. 

TODO: Keep stuff about mixing coefficients?
We note that although the mixing coefficient, $a_i$ is required
to create the mixture $y(t)$, it is not necessary to recover $a_i$ along with
$s_i(t)$; the mixing coefficients are 

The mixture signal $y(t)$ might be stereo, meaning that it is actually _two_
signals that are related. However, what makes a mixture signal a good candidate
for a source separation approach is that there are more desired sources within
the mixture than there are mixture signals. Because of this, source separation
is considered an _underdetermined problem_, meaning that there are fewer observations
than there are required outcomes. For example, if a stereo mixture contains
a recording of a piano quartet, for any desired source in the mixture we only have
two observations (each channel of the stereo mix), therefore source separation
would be a useful tool to isolate one of the sources.


In this tutorial we will be focusing on music separation, or the process of 
isolating at least one musical instrument or singer from a musical mixture that
contains one or more other musical instruments or singers. Music is seen as a
distinct problem from other types of source separation because there are many
factors that make it uniquely challenging:

- Sources in music are _highly correlated_, meaning that all of the sources usually 
  change together at the same time. For example, in a rock band if the bass guitar
  changes its note at the start of a new measure, it is likely that the other
  instruments will change as well.
  
- Music is mixed and processed in ways that are aphysical. Contemporary recording
  practices are such that any given source in a mixture might never occur naturally
  in the real world. A famous, early example is the crooning singing style of
  Frank Sinatra or Bing Crosby, both of whom no longer had
  to belt out over a 20-piece big band to be heard after the invention of condenser
  microphone allowed them to sound like they were gently singing in your 
  ear.{cite}`lockheart2003history`
  Nowadays, recording engineers and musicians routinely create sounds that would
  never occur naturally.
  
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


### Beamforming


## A Historical Perspective

Source Separation is a problem that has been studied for decades. Although we won't
be able to cover anything except the most recent developments in great detail,



[^fn1]: Some creative applications might not have such strict demands; when using
 source separation to create remixes for instance, the artifacts might be masked
 by other sources in the mix.
 