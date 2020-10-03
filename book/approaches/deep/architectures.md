Deep Learning Architectures
===========================


We now have the conceptual building blocks in place to understand
many of the modern deep learning systems for source separation.
In this section, we will outline a few of the recent systems.


## Mask-Based Systems


(architectures:unets)=
### U-Nets

```
```

U-Nets {cite}`jansson2017singing` are a very popular architecture
for music source separation systems. They input a spectrogram and
perform a series of 2D convolutions, each of which producing an encoding of
a smaller and smaller representation of the input. The small representation
at the center is then scaled back up by decoding with the same number of
2D deconvolutional layers, each of which corresponds to the shape of
one of the convolutional encoding layers. Each of the encoding layers
is concatenated to the corresponding decoding layers.

The original U-Net paper {cite}`jansson2017singing` has 6 strided 2D
convolutional encoder layers with 5x5 kernel sizes and strides of 2. 
After each encoder layer was a batch norm followed by a ReLU activation.
A Dropout of 50% is applied to the first three encoder layers.
After the 6th encoder layer, 5 decoder layers with the same kernel and
stride sizes, also with batch norm and ReLU activations. The final layer
has a sigmoid activation function that makes a mask.

The final mask is multiplied by the input mixture and the loss is taken
between the ground truth source spectrogram and mixture spectrogram
with the estimated mask applied, as per usual.

Many variants of the U-Net architecture have been proposed, some which add
conditioning on the latent space to control or enhance the output


(architectures:openunmix)=
### Open-Unmix

Open-Unmix is a 


(architectures:maskinference)=
### Mask Inference

Although many deep learning systems make masks, the term _Mask
Inference_ typically refers to a specific type of source separation
architecture. Mask Inference networks have a handful of recurrent
neural network layers that are connected to a fully connected layer
that outputs the mask. It is common to use Bidirectional 
Long Short-term Memory (BLSTM) networks as the recurrent layers. As
mentioned, this gives the network twice as many trainable parameters;
one set going forward in time and another going backward in time. The
fully connected layer then converts the output of the BLSTM to the
shape of the spectrogram to make a mask. It is typical to use sigmoid
activation on the fully connected layer to create the mask.

Batch norm, whitening? Dropout


(architectures:deepclustering)=
### Deep Clustering

Deep Clustering maps each {term}`TF-bin` to a high-dimensional
embedding space such that TF-bins dominated by the same source
are close and those dominated by different sources are far apart.
We say that a TF-bin is _dominated_ by some Source $S_i$ if most
of the energy in that source is from $S_i$.

Deep Clustering has the same basic network architecture as a
Mask Inference network: a set of BLSTM layers to to a
fully connected layer with a sigmoid activation function that makes 
masks.


(architectures:chimera)=
### Chimera

Chimera combines the Mask Inference and Deep Clustering architectures
into a multi-task neural network. The net is trained to optimize both
loss functions simultaneously. It does this by having a separate
"head" for each loss. Each of these heads is its own fully connected
layer and activation with only one loss applied from the


## Waveform Systems

(architectures:convtasnet)=
### ConvTasnet


(architectures:waveunet)=
### Wave-U-Net

(architectures:demucs)=
### Demucs
