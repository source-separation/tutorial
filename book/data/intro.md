# Introduction

In this chapter we'll cover the key aspects we need to know about data for source separation: what do data for source 
separation look like, relevant datasets, input representations for training models, and, importantly, how to programatically
generate training and evaluation data to minimize the time we spend data wrangling and maximize the performance we can squeeze
out of our data.

## Data for music source separation

The inputs and outputs of source separation model look like this:

PLACEHOLDER: image showing mixture --> model --> stems

For this tutorial, we will assume the inputs and outputs were created in the following way:
1. Each instrument or voice is recorded in isolation into a separate audio track, called a "stem". The stem may be 
processed with effects such as compression, reverb, etc.
2. The mixture is obtained by summing the processed stems.

PLACEHOLDER: diagram of simplified mixing process 

```{note}
This is a simplified view of music creation. In practice, the mixture (musicians refer to this as the "mix") typically 
goes through a "mastering" step which includes the application of multiple non-linear transformations to the mixture signal
to produce the "master", which is rarely a simple sum of the stems. Nonetheless, this simplified view (no mastering) allows 
us to train models that produce compelling results, as we shall see throughout this tutorial. 
```

Since the master is not a linear sum of the stems, for training a source separation model we instead create our own mixture 
by summing the stems directly to create a linear mixture. When training a source separation model, we provide this mixture 
as input, the model outputs the estimated stems, and we compare these to the original stems that were used to create the 
mixture. The difference between the estimated stems and the original stems is used to update the model parameters during 
training:

PLACEHOLDER: block diagram of training

The difference between the estimated stems and original stems is also used to *evaluate* a trained source separation model,
as we shall see later on.

In summary, from a data standpoint, to train a music source separation model we need:
1. The isolated stems of all instruments/voices that comprise a music recording. This is commonly referred to as a
"multi-track recording", since each instrument is recorded on a separate track of a digital audio workstation (DAW).
2. The ability to programatically create mixtures from these stems for training and evaluation.


## Data is a key component

## Data for source separation hard to obtain

## Chapter outline
