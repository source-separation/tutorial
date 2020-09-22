---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

Evaluation
==========


<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/FTQbiNvZqaY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

```{dropdown} Video not working?
Here is a <a href="">OneDrive link</a> to the full video.
<a href=""></a>
```

Measuring the results of a source separation approach is a challenging problem.
Generally, there are two main categories for evaluating the outputs of a source
separation approach: objective and subjective. Objective measures rate
separation quality by performing a set of calculations that compare the output 
signals of a separation system to the ground truth isolated sources. Subjective
measures involve having human raters give scores for the source separation
system's output.

Objective and subjective measures both have benefits and drawbacks. Objective
measures struggle because there are many aspects of human perception that are
extremely difficult capture by computational means alone. However, compared to
subjective measures, they are much faster and cheaper to obtain. On the other
hand, subjective measures are expensive, time-consuming, and subject to
variability of human raters, but they can be more reliable than objective
measures because humans are involved in t


Objective measures are, by far, much more popular than subjective measures, but
we feel it is worth understanding them both to some extent.
 


## Objective Measures


### BSS-Eval & Friends

For the past 


## Subjective Measures

Having a human or set of humans evaluate a separation result is the gold standard
for measuring the quality of a system. However, this is rarely done due to how
difficult it is to get reliable evaluation data.

Ultimately, if 

### something here