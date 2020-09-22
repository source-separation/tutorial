---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

Phase
=====


<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/FTQbiNvZqaY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

```{dropdown} Video not working?
Here is a <a href="">OneDrive link</a> to the full video.
<a href=""></a>
```

```{code-cell} ipython3
---
other:
  more: true
tags: [hide-output, hide-input]
---
# Imports for this notebook

# from ipywidgets import interact
import numpy as np

# from bokeh.io import push_notebook, show, output_notebook
# from bokeh.plotting import figure
# output_notebook()

```

While many source separation papers mainly focus on their approaches for creating
better mask estimates, which only apply to the magnitude components of a
{term}`TF Representation`, the other crucial aspect of sound is its phase.

In this section, we will discuss the important problem of how to turn a magnitude
spectrogram of an estimated source back into a waveform so that we may listen to
the source estimation.


## Phase - A Quick Primer

[[PHASE IMAGE]]

An audio signal, $y(t)$, composed of exactly one sine wave,

$$
y(t) = A \sin (2 \pi f t + \phi)
$$

can be completely described by the parameters $t, A, f$ and $\phi$, where
$t$ represents time in seconds, $A$ is the wave's amplitude (unit-less), $f$ is
its frequency in Hz, and $\phi$ is its phase offset in radians (_i.e.,_ where
in the cycle the wave is at $t=0$). If $\phi \ne 0$, then the sine wave appears
shifted in time by $\frac{\phi}{2 \pi f}$, where negative values "delay" the
signal and positive values "advance" it. Recall that a sine wave is the same
value every $2 \pi f$ seconds,
_i.e._

$$
\sin (0) = \sin(2 \pi f) = \sin(2k \pi f) ~ \forall ~ k \in \mathbb{Z}.
$$

This is inherent the periodicity of the sine wave, and the point where the phase
"wraps around" or essentially restarts at 0 every $2 \pi f$ seconds.

Let's scale this up. We'll be a little hand-wavy here, but our old pal Fourier
told us that
[any sound can be represented as an infinite summation of sine waves](https://en.wikipedia.org/wiki/Fourier_transform)
each with their own amplitudes, frequencies, and phase offsets. So now any sound
we hear can be represented as many, many tuples of $(t, A, f, \phi)$.
Let's think back
to the section about time-frequency representations: each bin is index by time
along the x-axis and frequency along the y-axis. A {term}`TF bin` is a snapshot
of the sound at that particular time and at that particular frequency component.
In a magnitude spectrogram, power spectrogram, log spectrogram, etc, each value
represents the sound's energy for that frequency at that time. So, if you're keeping
track at home, spectrogram has entries for $t, A$ and $f$, but no representation
for the phase $\phi$.

## Why We Don't Model Phase

[[REAL PHASE & GAUSSIAN NOISE]]

Let's do a little experiment. Above are two images. One of the two images shows
the phase component of an audio signal, the other shows gaussian noise. Can you
guess which is which?

Therein lies the problem: there is a complicated interplay between how the
{term}`DFT` captures the signal at each time step, how the frequency is captured
and how the phase is captured. Let's build some intuition for why this is.
For the same time step, the lower frequency components of the signal change less
quickly than the higher frequency components. This means that for any two adjacent
time steps, the time difference is the same but the amount of change of any frequency
might not be the same. The phase wraparound happens much quicker at the higher frequencies
than at the lower frequencies.

For instance, let's say we have a sound wave that is composed
of two sine waves that have frequency $440$ Hz and $523.25$ Hz, which are A440 and
the C note above A440 respectively. Both start at the origin at time $t = 0$.
Let's look at the value of each of these sine waves at different time intervals:

```{code-cell} ipython3
---
other:
  more: true
---

time = np.linspace(0.0, 0.05, 2000)
f1 = 440.0   # A440
f2 = 523.25  # C above A440
sin1 = np.sin(2 * np.pi * f1 * time)

offset = 3
sin2 = np.sin(2 * np.pi * f2 * time) + offset

# for i, t in enumerate(time):
#     print(f'sin1({t:.1f}) = {sin1[i]:+.3f},\tsin2({t:.1f}) = {sin2[i]:+.3f}')

# p = figure(title="Phase Example", plot_height=400, plot_width=700, y_range=(-1.5, 4.5),
#            background_fill_color='#efefef')
# r1 = p.line(time, sin1, color="#8888cc", line_width=1.5, alpha=0.8)
# r2 = p.line(time, sin2, color="#cc88cc", line_width=1.5, alpha=0.8)
#
# def update(f2_=523.25, phi_=0.0):
#     sin2_ = np.sin(2 * np.pi * f2_ * time + phi_) + offset
#     r2.data_source.data['y'] = sin2_
#     push_notebook()
#
# show(p, notebook_handle=True)

```

```{code-cell} ipython3
---
other:
  more: true
---
# interact(update, f2_=(440.0, 880.0), phi_=(0, 4 * np.pi, 0.05))
```


When we take snapshots of each sine wave, it's difficult to find a pattern between
the two (other than, y'know, the sine wave we drew them from).
Another difficulty is that humans do not perceive phase offsets,, _i.e._,
if one of the phases at $t /ne 0$,

This is all to say that getting phase right is _hard_. That being said, there are ways
to estimate phase, but few if any source separation approaches
(or sound generation models) attempt to explicitly model phase to the same extent
that magnitude information is modeled. We will discuss some of these phase estimation
techniques below.

## How to Deal with Phase



### The Easy Way

All of that being said, there is an easy and very common way to deal with phase
for source separation: **copy it from the mixture!** This isn't perfect, but
researchers have discovered that it works surprisingly well, and when things go
wrong, it's usually not the fault of the phase.

So now, with this in place, we finally have our first strategy to convert our
source estimate back to a waveform. Assume we have a mixture STFT,
$Y \in \mathbb{C}^{T \times F}$, and estimated mask
$\hat{M}_i \in [0.0, 1.0]^{T \times F}$ for Source $i$.
Recall that we can apply the mask to a magnitude spectrogram of $Y$ like so:

$$
\hat{X}_i = \hat{M}_i \odot |Y|
$$

where $\hat{X}_i \in \mathbb{R}^{T \times F}$ represents a magnitude spectrogram of
our source estimate. Note that this equation looks similar if we want to apply
a mask to a power spectrogram ($|Y|^2$), log spectrogram ($\log |Y|$), etc.

Now we can just copy the phase from the mixture over to the magnitude spectrogram
of our source estimate, $\hat{X_i}$:

$$
\tilde{X}_i = \hat{X}_i \odot e^{j \cdot \angle Y}
$$

where we use $j = \sqrt{-1}$, "$\angle$" to represent the angle of the complex-valued
{term}`STFT` of $Y$, and $\tilde{X}_i \in \mathbb{C}^{T \times F}$ to indicate
that the estimate for Source $i$ is now complex-valued similar to an {term}`STFT`.

Putting it all together it looks like:

$$
\tilde{X_i} = (\hat{M}_i \odot |Y|) \odot e^{j \cdot \angle Y}.
$$

In code, this looks like:

```{code-cell} ipython3
---
other:
  more: true
---

# TODO: Get mix stft, and mask

# mix_magnitude, mix_phase = np.abs(mix_stft), np.angle(mix_stft)
# src_magnitude = mix_mag * mask
# src_stft = src_magnitude * np.exp(1j * mix_phase)

```


### The Hard Way, Part 1: Phase Estimation

It is possible to estimate the phase once the estimated mask is applied to the
mixture spectrogram. One popular way is the Griffin-Lim algorithm, which can
iteratively reconstruct the phase component of a spectrogram by...

MISI

It's worth noting that the STFT & iSTFT computations, and these phase estimation
algorithms are all differentiable and researchers have made models that


### The Hard Way, Part 2:  Waveform Estimation

Finally, a recent way that researchers have been tackling the phase problem is
by side-stepping any explicit representations of it at all. Recently, many
deep learning-based models have been proposed that are "end-to-end", meaning that
the model's input and output are all waveforms. In this case, the model decides
how to represent phase.



[^fn1]: The amplitude, loudness, and energy of a sound are all calculated differently
 but still related. Here we will use "amplitude" as a stand-in for whichever one you
 choose.


