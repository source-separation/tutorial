Intro
=====

Training deep networks in any field can be a challenging endeavor. It requires patience,
luck, and a lot of debugging. The goal of this section is to provide an overview
of both the theory and the practice of training deep networks for audio source
separation. We will cover the following topics:

- Gradient-descent based optimization
- Inputs and outputs for audio separation networks
- The building blocks of modern deep nets for separation
- Training a model with a straightforward script based on Scaper and nussl.
- How to diagnose and fix common bugs

As we proceed through this section, we will gradually build up a script that can be used
and modified to train and evaluate a deep audio source separation
network.
