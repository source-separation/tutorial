
## Open-Source Tools & Data for Music Source Separation: A Pragmatic Guide for the MIR Practitioner

**By Ethan Manilow, Prem Seetharaman, and Justin Salamon**

This is the code repository for our [ISMIR 2020](https://ismir.github.io/ISMIR2020/)
tutorial about Open Source tools for Source Separation. This repo contains the
code to build the jupyter book website where the tutorial content is hosted.


[Click here to visit the tutorial!](https://source-separation.github.io/tutorial/)

## Building the book

To build the book, do the following:

```
pip install -e .
python -m common.data --run.cmd='download'
jb build --all book/
```

## Running an experiment

To run a basic mask estimation experiment with a Chimera network,
do the following:

```
python -m common.data --run.cmd='prepare_musdb'
python -m common.exp.chimera --run.cmd='train'
python -m common.exp.chimera --run.cmd='evaluate'
```

## Questions? Comments? Typos? Bugs? Issues?

Open a github issue [here](https://github.com/source-separation/tutorial/issues/new)



