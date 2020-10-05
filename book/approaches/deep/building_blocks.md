Building Blocks
===============


In this section we will do a quick overview of neural network components
at a conceptual level,
paying special attention to how they are used within modern source
separation systems.

Neural networks, sometimes called _deep learning_ (or some variant of
"neural networks" and "deep learning" like "deep nets"), are a type of
machine learning algorithm that have become very popular in the past few
years. While we will cover gradient descent in a later section, we will
not fully dive into the mathematics neural networks.

There are many, many resources for learning about neural networks,
so we won't repeat everything here, but we do want to leave you with
some intuition about how the work and importantly, **how they work within the
source separation context.**


On this page we will discuss the most common types of neural network
components used in source separation systems.




## Neural Network Components

### Layers

Neural networks are composed of _layers_, each of which has a set
of weights that get updated through gradient descent.

When we talk
about network layers, we usually talk about _forward_ and _backward_
passes through the network. During the forward pass, each layer
receives input from a previous layer or input data and transforms
the data by multiplying each component of the input by a set of
_weights_, the result of which is the output. Usually the way the
inputs are ingested and outputs calculated is a little more complicated
that this, but what's important to know is that as an example input
gets pumped through a neural network it goes through many discrete
transformations, each of which is a layer.

It's important to know that almost all deep net source separation
systems are trained with some variant of
[Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent),
which means that data is passed through each layer in _minibatches_ 
(sometimes referred to as just _batches_).

```{tip} 
An important first step for debugging neural networks is understanding
the input and output dimensionality at each layer. In many frameworks
(_e.g._, [pytorch](https://pytorch.org/), or [tensorflow](https://www.tensorflow.org/)),
the dimensionality is referred to as each layer's `shape`.

In many cases, getting the shapes set up right will get you 90% of the
way there to a working system.
```


#### Fully Connected Layers


```{figure} ../../images/deep_approaches/fully_connected.png
---
scale: 50%
alt: A fully connected layer.
name: fully-connected
---
An input layer, a fully connected hidden layer, and an output layer.
[Image Source](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg)
```


In a fully connected layer, every node is connected to every input node
and is connected to every output node. These are sometimes called _linear_
layers because, without an activation function, they apply a linear transform
to the input data.

Fully connected layers have many uses, but a common use is to expand or
compress the dimensionality of the input for the next layer. In this
sense, they can be thought of as a kind of "glue" between other components.

##### Masking

Within source separation, fully connected layers are usually used to
create masks, changing the dimensionality of the previous layer so
that it matches the dimensionality of the output (_e.g._, the number of
frequency components in a spectrogram or number of samples in a waveform).
The fully connected layers usually have an activation function when used
as a mask, discussed further below.


##### Reshaping

As we mentioned, linear layers are well suited to expand or compress the
dimensionality of the input data. There are many ways we can take advantage
of this property within source separation.

Here's a simple example: Let's say we have a source separation system
that applies a mask to a spectrogram that has 513 frequency components and
400 time steps. When we include the batch dimension, the output shape of 
our fully connected layer might look like `(16, 400, 513)`, where the
first dimension is the batch (with 16 examples). But if we want this
system to make _two_ masks, we can change the output dimensionality of
this fully connected layer to be `(16, 400, 1026)`,
where now we've doubled the frequency dimensions ($513 \times 2 = 1026$)
to indicate that we have two sources. Then we can reshape the output
such that it looks like `(16, 400, 513, 2)`, where we've expanded the frequency
out into the last dimension to take care of both of our sources.
The network will learn that this means there are two sources.

A similar thing happens with Deep Clustering (covered on the next page).
We use a linear layer to produce a high-dimensional embedding space
and then reshape it so that we can make sense of it.

#### Recurrent Layers

```{figure} ../../images/deep_approaches/rnn.png
---
alt: An unfolded recurrent node
name: rnn
---
Recurrent layers have self-connected loops. When the loops are
"unfolded" they show how each loop represents a different time step.
[Image Source](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg)
```

Recurrent layers are layers that have connections to themselves, as well
as to the next layer. In other words, some of the outputs of the nodes
are plugged back into the inputs of the same nodes, making loops so that
information to persist.

Because of these loops, recurrent layers are
particularly well suited for learning data that varies over time, like
audio. Each loop can ingest the next set of data in steps. The way that
recurrent layers are used in source separation is that they ingest the
audio as it changes along time. So, for example, if we have a spectrogram
that is input into a recurrent layer, the recurrent layer will ingest
one column of the spectrogram at a time (_i.e._, the whole spectrum at
that time step).

The number of units recurrent layers have defines their
_receptive field_ or the amount of time steps the layer can see at any
given time. So for instance, if a spectrogram has 1000 time steps
but the recurrent layer only has 400 units, only 400 time steps of
the spectrogram will be processed by the recurrent layers at a time.
It will start with the first time step, and work its way across every
one of the 1000 time steps, advancing by one step at a time. Because
of this, if you train a recurrent layer that only has 400 time steps, 
it can accept spectrograms of any length. This is not the case for
convolutional layers, which we will cover shortly.


````{panels}
:container: container-fluid 
:column: col-lg-6 col-md-6 col-sm-6 col-xs-12 
:card: shadow-none border-0

```{figure} ../../images/deep_approaches/lstm.png
:width: 100%
:name: lstm

LSTM cell [Image Source](https://commons.wikimedia.org/wiki/File:The_LSTM_cell.png)
```

---

```{figure} ../../images/deep_approaches/gru.png
:width: 100%
:name: gru

GRU [Image Source](https://commons.wikimedia.org/wiki/File:Gated_Recurrent_Unit,_base_type.svg)
```

````

The two most common types of recurrent layers are 
**[Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory)** layers 
and **[Gated Recurrent Unit (GRU)](https://en.wikipedia.org/wiki/Gated_recurrent_unit)** layers.
Although there are 
slight differences between the two, what's important to know is that
these layers are able to retain information in a way that is stable,
whereas naive recurrent layers like those shown in {numref}`rnn` are
_not_ stable.

For source separation, researcher mostly use **LSTM** layers. We are unaware of
any reason that source separation researchers have settled on LSTMs over
GRUs, but it is surely the case that GRUs are much less common in the literature
than LSTMs.


For more details on how LSTMs and GRUs work, see
[Christopher Olah's terrific blog post.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

##### Bidirectional or Unidirectional?

It is very common to see recurrent layers that are _bidirectional_, meaning
that they actually contain two sets of units: one that advances forward
in time and another that advances backward in time. The bidirectional
case can thus "see the future" because it starts at the end of an audio
clip. Because of this you should **not** use bidirectional recurrent
layers if you need a real time system.

Unidirectional recurrent layers
are called _causal_ and bidirectional recurrent are called _noncausal_.
If an LSTM or GRU is bidirectional, it is denoted as BLSTM or BGRU,
respectively.


```{admonition} Watch out!
There is some sloppiness when research papers report the details of
their BLSTM/BGRU layers. For instance, if a paper says they used a
"BLSTM with 600 units", does that mean 600 units in each direction
or 300 in each direction (600 total)?

We assume that "600 units" means 600 in _each direction_, because
[that's how the pytorch API configures it](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM).
But beware!
```

#### Convolutional Layers

````{panels}
:container: container-fluid 
:column: col-lg-6 col-md-6 col-sm-6 col-xs-12 
:card: shadow-none border-0

```{figure} ../../images/deep_approaches/conv1.gif
:width: 75%
:name: conv1

Convolution with 2D input with a 3x3 kernel and stride 1.
```

---

```{figure} ../../images/deep_approaches/conv2.gif
:width: 75%
:name: conv2

Convolution with 2D input with a 3x3 kernel and stride 2.
```
[Image Sources](https://www.cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html)
````


Convolutional layers are similar to fully connected layers shown above,
except that now each node is only connected to a small set of nodes 
from the previous layers. Reducing the amount of connections makes
the network less prone to overfitting to the training data. Convolutional
layers also have the property that they are
[translationally invariant](https://en.wikipedia.org/wiki/Translational_symmetry).

Convolutional layers are related to the mathematical/signal processing
concept of [convolutions](https://en.wikipedia.org/wiki/Convolution) in
that convolutional layers learn a _filter_ from a sliding window
of the input layer. This sliding window is the receptive field of the 
convolutional layer. A depiction of two convolutional layers are
down in {numref}`conv1` and {numref}`conv2`.

There are two main parameters that affect the output shape of a 
convolutional layer: the kernel size and the stride. The kernel
size dictates the number and shape of nodes from the previous layer that
nodes at the current layer see (the shape of the window), and stride
dictates the distance that the distance that the window moves between
adjacent nodes.

In source separation, convolutions have been used to great effect in
the waveform and time-frequency domains. In the waveform domain,
1D convolutions are used to input and output waveforms, and in the 
time-frequency domain 2D convolutions are used to input spectrograms
and output masks.

Unlike recurrent layers that can process one time step at a time,
convolutional layers have to have a full example with the exact input
shape in order to process data. For example, if we have a spectrogram
with 512 frequency bins and 1000 time steps above, but our first convolutional
layer requires an input shape of `(512, 128)`, we have to split our
spectrogram into 8 windows of size exactly 128 and include the necessary
padding on the last window.


Convolutional layers can sometimes have a hard time with edge effects.
For instance, when predicting a waveform it is possible that a
convolutional neural network might learn discontinuities,
which might lead to audible artifacts. One way around this is to
output overlapping windows similar to how an STFT is computed. Going
back to the above example with a spectrogram, we might instead 
split it into 16 overlapping windows.


For further reading, see the
[Wikipedia article on Convolutional Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network),
or [Stanford University's course webpage for CS231](https://cs231n.github.io/convolutional-networks/).

### Activation Functions

```{figure} ../../images/deep_approaches/activation_fn.png
---
scale: 50%
alt: How an activation function is applied in a network.
name: activation
---
How an activation function is applied in a network. Each node has a
"weight", which get combined and passed through the activation function.
[Image Source](https://commons.wikimedia.org/wiki/File:ArtificialNeuronModel_english.png)
```

Activation functions change how much one layer influences the next layer.
For each node in a layer, the activation function "decides" whether 
that node should be "on" or "off". Activation functions are usually
non-linear and always [differentiable](https://en.wikipedia.org/wiki/Differentiable_function).

It is usually important to have at least one layer with a non-linear
activation function. A list of many activation function can be seen
on this [Wikipedia article](https://en.wikipedia.org/wiki/Activation_function).
The most common ones seen in source separation
are discussed below:


#### Sigmoid

```{figure} ../../images/deep_approaches/sigmoid.png
---
scale: 50%
alt: A plot of the sigmoid activation function.
name: sigmoid
---
A plot of the Sigmoid activation function.
[Image Source](https://commons.wikimedia.org/wiki/File:Logistic-curve.svg)
```

The `sigmoid` activation function is shown in {numref}`sigmoid` and
is used very commonly as the output of a neural net that creates masks.
Because it is bounded in the range $[0.0, 1.0]$ it is perfectly
suited to make {ref}`masks:softmasks`. It is sometimes denoted $\sigma$.



#### Tanh

```{figure} ../../images/deep_approaches/tanh.png
---
scale: 50%
alt: A plot of the tanh activation function.
name: tanh
---
A plot of the tanh activation function.
[Image Source](https://commons.wikimedia.org/wiki/File:Activation_tanh.svg)
```

The `tanh`, or the hyperbolic tangent, activation function is
shown in {numref}`tanh`. Tanh is similar to sigmoid, but it is bounded
in the range $[-1.0, 1.0]$, instead of $[0.0, 1.0]$ like sigmoid.
For this reason it is more common to find a `tanh` in the
middle of a source separation network than at the ends.


#### ReLU & Friends

```{figure} ../../images/deep_approaches/relu.png
---
scale: 50%
alt: A plot of the ReLU activation function.
name: relu
---
A plot of the ReLU activation function.
[Image Source](https://commons.wikimedia.org/wiki/File:Activation_rectified_linear.svg)
```

The Rectified Linear Unit, or `relu` activation function is shown
above in {numref}`relu`. It is $0.0$ if the input is less than $0.0$
and is linear with slope $1$ otherwise:


![relu_eq](../../images/deep_approaches/relu_eq.svg)

ReLUs are sometimes used to
make masks, and some systems use them to output waveforms as well.


```{figure} ../../images/deep_approaches/leaky_relu.png
---
scale: 50%
alt: A plot of the PReLU activation function.
name: leaky-relu
---
A plot of the PReLU activation function.
[Image Source](https://commons.wikimedia.org/wiki/File:Activation_prelu.svg)
```

Two related activation functions to ReLUs are the `Leaky ReLU` and `PReLU`.

The Leaky ReLU is similar to the regular ReLU, but instead of 
the output being $0.0$ below $x=0.0$ it is ever so slightly above $0.0$:

![leaky_relu_eq](../../images/deep_approaches/leaky_relu_eq.svg)

And PReLU has a learnable network parameter $\alpha$ that defines the slope
below $x=0.0$:

![prelu_eq](../../images/deep_approaches/prelu_eq.svg)

These are sometimes used in middle layers or as an activation for the final
layer when outputting a waveform.


### Normalization

### Dropout

Dropout is a regularization technique that improves a network's
ability to generalize to unseen data. This is a simple technique
whereby at each training step some percentage of the nodes are
set to 0. This is very widely used in source separation systems
and is essential to making them work well.

### Spectrogram Components

- Calculating Spectrograms ahead of time
- Computing Spectrograms on the fly
- Mel Spectrograms

### Learned Filter Banks


## Loss Functions and Targets

### Spectrogram Losses

L1 & MSE
MSA, PSA


### Clustering Losses


### Waveform Losses

SI-SDR loss
