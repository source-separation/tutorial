(data:datasets)=
# Datasets

## Overview
Here's a quick overview of existing datasets for Music Source Separation:

| **Dataset** | **Year** |  **Instrument categories** | **Tracks** | **Avgerage duration (s)** | **Full songs** | **Stereo** |
| ----------  | -------- |  ------------------------- | ---------- | ------------------------- | -------------- | ---------- |
| [MASS](http://www.mtg.upf.edu/download/datasets/mass) | 2008 | N/A | 9 | 16 $\pm$ 7 | ❌ | ✅️ |
| [MIR-1K](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) | 2010 | N/A | 1,000 | 8 $\pm$ 8 | ❌ | ❌ |
| [QUASI](http://www.tsi.telecom-paristech.fr/aao/en/2012/03/12/quasi/) | 2011 | N/A | 5 | 206 $\pm$ 21 | ✅ | ✅ |
| [ccMixter](http://www.loria.fr/~aliutkus/kam/)  | 2014 | N/A | 50 | 231 $\pm$ 77 | ✅ | ✅ |
| [MedleyDB](http://medleydb.weebly.com/) | 2014 | 82 | 63 | 206 $\pm$ 121 | ✅ | ✅ |
| [iKala](http://mac.citi.sinica.edu.tw/ikala/)  | 2015 |  2  | 206 | 30 | ❌ | ❌ |
| [DSD100](/datasets/dsd100.md)| 2015 | 4 | 100 | 251 $\pm$ 60 | ✅ | ✅ |
| [MUSDB18](https://sigsep.github.io/datasets/musdb.html) | 2017 | 4 | 150 | 236 $\pm$ 95 | ✅ | ✅ | 
| [Slakh2100](http://www.slakh.com/) | 2019 | 34 | 2100 | 249 | ✅ | ❌ |  
This extended table is based on: [SigSep/datasets](https://sigsep.github.io/datasets/), and reproduced with permission.

<!--- | [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) | 2019 | ? | ? | 150 | 236 $\pm$ 95 | ✅ | ✅ |)  # omitted since almost identical to MUSDB18 --->

The columns of the table indicate the key characteristics we must consider when choosing or curating a dataset for music source separation:
* **Number of tracks**: Generally speaking, the more the better. But quantity isn't enough! We need quality and variability too, as captured by 
the other columns.
* **Musical content**: Our models are unlikely to generalize well to mixtures that are very different from those used during training. Musical genre, while inherently a 
fuzzy concept, is a reasonable proxy for the types of mixtures we can expect in terms of instrumentation and arrangement, playing styles, and mixing characteristics. 
If we want our trained model to generalize well to music of a certain genre, it is important for that genre to be well represented in our training data.     
* **Instruments**: Which ones? How many? A model is unlikely to successfully separate an instrument it hasn't seen during training. Similarly, a model trained on sparse 
mixtures in terms of the number of instruments is unlikely to successfully separate dense mixtures.
* **Duration**: Does the dataset provide full-length songs, or just excerpts (clips) from songs? The former is a richer data source.
* **Format**: Are the tracks provided as stereo audio or mono? What's the sampling rate? Typically a source separation algorithm will output audio in the same format
with which it was trained, so if your goal is to separate stereo audio recorded at 44.1 kHz, ideally you will trained your model on audio in the same format. 

As we can see, earlier datasets were smaller in terms of the number of tracks, sometimes only providing short clips rather than full songs, and often focused on 
separating vocals from the accompaniment without providing access to all isolated stems (instruments) comprising the mix. More recent datasets typically include 
full duration songs recorded in stereo and provide all isolated stems, allowing us to choose which source separation problem we want to address, whether it's 
separating a specific instrument or voice from the mix, separating into groups such as harmonic versus percussive instruments, or separating a mix into all of 
its constituent sources.

In this tutorial we'll be using the MUSDB18 dataset. More specifically, we'll use short clips from this dataset. There's no need to download the dataset, we will provide
code for obtaining the clips later on in the tutorial. We'll discuss this dataset in more detail in the next section.


## Always listen to your data

A common pitfall in machine learning is not inspecting your data before you start training models with it. Inspecting the data is critical, as it gives us 
a much better understanding of the content, for example: what types of samples does it contain? Is it comprehensive or narrow? Are the annotations correct? 
Are there outliers? Etc. Inspecting the data is key to ensure we don't model it based on incorrect assumptions, which can lead to sub-optimal performance, 
generalization issues, bias, and other problems.

When working with audio, we have the unique privilege of being able to *listen* to our data. Listening to our data allows us to quickly get an idea for the type 
of music it includes, the quality (is it amateur musicians or professional?), the frequency range, and various other characteristics that can inform our design choices  
later on such as model selection, data pre-processing and post-processing, optimization algorithm, etc.

```{tip}
Always listen to your data before you start working with it. It will inform design choices, help prevent bias and generalization issues, and help identify what 
to look for when you perform error analysis on your trained model.  
``` 
