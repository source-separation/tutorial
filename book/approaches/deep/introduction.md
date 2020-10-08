Introduction
============


We've now gotten some understanding of the general mechanics of source separation,
now let's turn to Neural Network methods. Neural Network-based methods are
commonly referred to as deep learning or deep net methods.

```{image} ../images/data/source_separation_training.png
:align: center
```

Deep nets are currently the state-of-the-art source separation technology.
In a nutshell, they work by training on a large amount of mixtures and
isolated source data. The network produces an output for a source, then
the network's output is compared against known, ground truth isolated source.
This comparison is used to update the network so that the next time
it produces an estimate for that source, the network's estimate is
closer to the true source. This process is called
[back propagation](https://en.wikipedia.org/wiki/Backpropagation).

The process we just described requires access to ground truth isolated
source data. This requirement means that many source separation systems are
[supervised machine learning](https://en.wikipedia.org/wiki/Supervised_learning)
systems. In order to achieve the impressive performance that deep nets
are famous for, **a large amount of isolated source data is required.**

Deep nets are also tricky to get right. Neural networks are highly
complex systems with millions of trainable parameters, sometimes
called _weights_. The choice of how to setup those weights
(_i.e._, the number and structure of the weights and connections, how
they are trained and updated, etc) is a difficult process. Each
of these setup choices is called a _hyperparameter_. The choice of
just one hyperparameter setting might be the difference between
getting amazing results and horrible results. This tutorial will
give you the tools to know which hyperparameter settings are the
right ones.


## The Road Ahead

The rest of this tutorial is dedicated to deep nets for source separation
and overcoming the problems required to get them working. On the next page,
we will discuss the neural network building blocks that are commonly used
in source separation. On the following page we will see how some prominent
open-source source separation projects that we discussed earlier use those
building blocks to create and train neural networks. These next two pages
will all be conceptual overviews.

In the next section, we will discuss how to get data and use it 
successfully for a source separation network. In the following section
we will put all of these pieces together, showing how, in code, everything
fits together.