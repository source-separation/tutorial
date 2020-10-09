
(opensrcmap)=
# Map of Open-Source Source Separation Projects

The open source world is filled with lots of projects for source separation! Here
we want to compile a list of some of these projects to provide an overview of the
landscape. This is not an exhaustive list, but it should serve as a good starting
point.

Some of the terms and evaluation results (_e.g._ "SDR") discussed on this page
may be unfamiliar to you at the moment, but by the end of the tutorial you we
hope that you will understand everything on this page. We encourage you to
come back to this page at the end of the tutorial.

All projects outlined below are primarily written in the `python` programming
language, unless otherwise noted.


(opensrc:audioprojects)=
## Useful Audio Projects

These projects are useful for any type of `python` audio project you might want to
do.


(opensrc:librosa)=
#### `librosa`

`librosa` is the de-facto standard library for all things music information retrieval.
Containing implementations of many, many common and useful algorithms in this space,
it is hard to imaging the field without it. Ignore it at your own peril!

[Github Link](https://github.com/librosa/librosa)


(opensrc:soundfile)=
#### `SoundFile`

`SoundFile` is one of many ways to load audio files in python, but we find that
it can handle almost any type of audio file without any problems. Use `ffmpeg`
(below) to give `SoundFile` super powers. `librosa`, `Asteroid`, `nussl`, and
many more open source projects use it under the hood.

[Github Link](https://github.com/bastibe/SoundFile)


(opensrc:ffmpeg)=
#### `ffmpeg`

`ffmpeg` is a command line application written in `C` that can read and manipulate
many different types of media files. Use this with `librosa` or `SoundFile` to
read many different audio formats (_e.g._ `.mp3`, `.ogg`, etc) without any hassle.

[Project Link](https://ffmpeg.org/)


(opensrc:scaper)=
#### `Scaper`

`Scaper` was originally intended for soundscape creation with the intention of
creating training data for sound event detection. But some researchers realized
the augmentation strategies implemented in `Scaper` are useful for source separation!
In this tutorial, we will discuss `Scaper` in detail. Stay tuned!

[Github Link](https://github.com/justinsalamon/scaper)


(opensrc:sourceseparationprojects)=
## Source Separation Projects

These projects are more tailored to doing source separation, as opposed to the
general projects we discussed above.


(opensrc:sourceseparationinfra)=
### Source Separation Infrastructure

Before we talk about projects that contain source separation approaches, let's
talk about some projects that help with source separation.


(opensrc:mireval)=
#### `mir_eval`

`mir_eval` is a python library that contains reference implementations for 
many MIR tasks, source separation included. While this is a good starting point,
we do not recommend using the source separation metrics that are implemented
here as they are obsolete at this point.

[Github Link](https://github.com/craffel/mir_eval)


(opensrc:museval)=
#### `mus_eval`

`mus_eval` has a more updated version of the source separation metrics, which
fixes some of the issues that `mir_eval` has. Some recent developments like
(SI-SDR) are missing from this library though.

[Github Link](https://github.com/sigsep/sigsep-mus-eval)


(opensrc:architectures)=
### Architecture-Specific Projects

It has become increasingly common for researchers to release code accompanying
their research papers. In the era of deep learning, the trained models are also
sometimes released. Here is a non-exhaustive list of some recent open source
projects. We have prioritized open source projects with code and downloadable
trained models by the original authors of the research papers described.

We will discuss some of these architectures in more detail in later sections,
but here we will provide some highlights and links to their Github repositories,
in alphabetical order.


(opensrc:convtasnet)=
#### `ConvTasnet`

See our overview here: {ref}`architectures:demucs`   
This implementation is from the {ref}`opensrc:demucs` github page.  
[Github Link](https://github.com/facebookresearch/demucs)  

| Weights Avail? | License | Training set      | Framework     | Vocal SDR |
|----------------|---------|-------------------|---------------|-----------|
| Yes            | MIT     | MUSDB             | pytorch       | 5.7       |
| Yes            | MIT     | MUSDB + 150 songs | pytorch       | 6.3       |




(opensrc:demucs)=
#### `Demucs`

See our overview here: {ref}`architectures:convtasnet`   
[Github Link](https://github.com/facebookresearch/demucs)


| Weights Avail? | License | Training set      | Framework     | Vocal SDR |
|----------------|---------|-------------------|---------------|-----------|
| Yes            | MIT     | MUSDB             | pytorch       | 5.6       |
| Yes            | MIT     | MUSDB + 150 songs | pytorch       | 6.3       |


(opensrc:openunmix)=
#### `Open-Unmix`

See our overview here: {ref}`architectures:openunmix`  
[Github Link (pytorch)](https://github.com/sigsep/open-unmix-pytorch)  
[Github Link (nnabla)](https://github.com/sigsep/open-unmix-nnabla)  

| Weights Avail? | License | Training set      | Framework     | Vocal SDR |
|----------------|---------|-------------------|---------------|-----------|
| Yes            | MIT     | MUSDB             | pytorch       | 5.3       |




(opensrc:spleeter)=
#### `Spleeter`

Spleeter is based on a `U-Net` architecture. See more here: {ref}`architectures:unets`   
[Github Link](https://github.com/deezer/spleeter)


| Weights Avail? | License | Training set      | Framework     | Vocal SDR |
|----------------|---------|-------------------|---------------|-----------|
| Yes            | MIT     | MUSDB + others    | tensorflow v1 | 5.9       |


(opensrc:waveunet)=
#### `Wave-U-Net`

See our overview here: {ref}`architectures:waveunet`   
[Github Link (pytorch)](https://github.com/f90/Wave-U-Net-Pytorch)  
[Github Link (tesnorflow v1)](https://github.com/f90/Wave-U-Net)  


| Weights Avail? | License | Training set      | Framework     | Vocal SDR |
|----------------|---------|-------------------|---------------|-----------|
| Yes            | MIT     | MUSDB             | pytorch       | 3.2       |
| Yes            | MIT     | MUSDB             | tensorflow v1 | 3.2       |




### Multi-Architecture Projects

A few existing projects aim to provide an entrypoint for multiple source separation
approaches. Here we will outline some of them.


#### `Asteroid`

`Asteroid` is a library that enables fast prototyping of deep learning-based 
source separation approaches. It contains implementations of 6 recent deep
learning-based source separation approaches and support for 6 speech datasets.
Support for more approaches and datasets are coming in the near future,
including support for MUSDB18.

`Asteroid` contains support for training and evaluating models, as well as
pretrained models that are ready to download through
[Zenodo](https://zenodo.org/communities/asteroid-models/search?page=1&size=20).
So far `Asteroid` only has available pre-trained models for speech separation,
but they will soon have code available to train music models. All of the code is based
on the [Pytorch](https://pytorch.org/) deep learning framework.

[Github Link](https://github.com/mpariente/asteroid)  


#### `nussl`

`nussl` is an object-oriented library for source separation that contains many
classic and deep learning-based algorithms, hooks for datasets, and evaluation
metrics all in the box. We will be discussing `nussl` throughout this tutorial
and exploring source separation. So we will be learning more about it very soon.

[Github Link](https://github.com/nussl/nussl)  


## What You'll Need for this Tutorial

In this tutorial, you will only need `nussl` and `Scaper`. These two projects
have all of the components needed to use off-the-shelf source separation or
develop new research.

As mentioned, we've included `requirements.txt` and `environment.yml` files
to get set up with all of the dependencies for these projects, should you find
them useful outside of this tutorial. (Recall that the coding portion of the
tutorial is all through Colab or Mybinder.org).


## Why We're Teaching `nussl` and `Scaper` in this Tutorial

There are so many wonderful open source projects out there, and we would love to
do a deep dive into all of them. But there's only so much time, so we choose
`nussl` and `Scaper` for a few reasons:

- **The combination of `nussl`+`Scaper` provides a solid foundation
  for all source separation projects.** These projects provide solutions for
  networks, data, evaluation, and interaction all in one.
  
- **The lessons we will explore in these two projects extend beyond them.**
  There are many common themes and design patterns in this area of research,
  and as we progress you will start to see the themes again and again. We
  believe that `nussl`+`Scaper` is a good way to explore these themes so that
  you can understand the structure of modern source separation systems.
  
- **We are the primary developers for these projects.** We _really_ understand
  these tools because, well, we built them! That means we are well equipped to
  answer any questions that come up. (And we can also be held accountable for
  mistakes you find!)
  
  
## Let's Start the Tutorial!

In the next section we'll start digging the conceptual foundations of source
separation.
  
