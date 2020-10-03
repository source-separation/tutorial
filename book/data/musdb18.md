(data:musdb18)=
# The MUSDB18 dataset

## Overview
The information on this page is based on the [MUSB18 dataset page](https://sigsep.github.io/datasets/musdb.html). Here we have edited down the content 
to focus on the details relevant to this tutorial while keeping it concise. For more details about the datataset pleasecan consult the dataset page.

MUSDB18 is a dataset of 150 full length music tracks (~10h total duration) of varying genres. For each track it provides a mixture along with the isolated stems for the 
drums, bass, vocals, and others. As its name suggests, the "others" stem contains all other sources in the mix that are not the drums, bass or vocals (labeled as 
"accompaniment" in the diagram below):

<img src=https://sigsep.github.io/assets/img/musheader.41c6bf29.png><br/>
Image source: https://sigsep.github.io/

All audio signals in the dataset are stereo and encoded at a sampling rate of 44.1 kHz. The mixture signal is identical to the sum of the stems.

The data in MUSDB18 is compiled from multiple sources: 
the [DSD100 dataset](dsd100.md), 
the [MedleyDB dataset](http://medleydb.weebly.com), 
the [Native Instruments stems pack](https://www.native-instruments.com/en/specials/stems-for-all/free-stems-tracks/),
and the [The Easton Ellises - heise stems remix competition](https://www.heise.de/ct/artikel/c-t-Remix-Wettbewerb-The-Easton-Ellises-2542427.html#englisch).

```{note{
MUSDB18 can be used academic purposes only, with multiple of its tracks licensed under a Creative Commons Non-Commercial Share Alike license (BY-NC-SA).
```

The full dataset is divided into train and test folders with 100 and 50 songs respectively. As their names suggest, the former should be used for model training and 
the latter for model evaluation.

```{note}
You do not need to download the full MUSDB18 dataset to complete this tutorial. For simplicity, we'll be using short excerpts (clips) from this dataset which 
we will download via python code later in the tutorial.  
```

## Downloading and inspecting MUSDB18 clips for this tutorial

TODO


## Acknowledgement
MUSDB18 was created by: Zafar Rafii, Antoine Liutkus, Fabian-Robert Stöter, Stylianos Ioannis Mimilakis, and Rachel Bittner. When using the dataset in your work, 
please be sure to cite it as:

```
@misc{musdb18,
  author       = {Rafii, Zafar and
                  Liutkus, Antoine and
                  Fabian-Robert St{\"o}ter and
                  Mimilakis, Stylianos Ioannis and
                  Bittner, Rachel},
  title        = {The {MUSDB18} corpus for music separation},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1117372},
  url          = {https://doi.org/10.5281/zenodo.1117372}
}
```
