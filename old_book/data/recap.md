(data:recap)=
# Recap

In this Chapter we covered:

## {ref}`Introduction <data:introduction>`

* What are the inputs/outputs of a source separation model
* How mixtures are created
* How data are used to train a source separation model (high level)
* Why data are key for music source separation, and why they are hard to obtain 

## {ref}`Datasets <data:datasets>`

* Overview of existing datasets
* Important dataset characteristics to look out for
* Why it's important to always listen to your data!

## {ref}`MUSDB18 <data:musdb18>`

* Overview of the MUSDB18 dataset we will use in this tutorial
* How to download the dataset preview clips using `nussl`
* How to inspect our data via audition and visualization
 
## {ref}`Generating mixtures with Scaper <data:scaper>`

* Why use Scaper rather than write ad-hoc mixing code
* Overview of Scaper
* How to prepare your source material (stems) for processing with Scaper
* How to generate randomly augmented mixtures
* The difference between incoherent and coherent mixtures and how to generate both
* How to plug our Scaper mixing code into `nussl`
