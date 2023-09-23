# I hate machine learning classification, so I use Hypothesis testing for classifying MNIST data.

## Introduction
As the name suggests, I hate the overrate of the use of machine learning to classifying MNIST data. Like, why do we need to use those fancy algorithm for this simple task? So, I propose to use the more traditional way that can be taught to first year student who are taking basic statistics. Introducing hypothesis testing classification algorithm.

## Side/technical note
This is just a metric learning. Here, we use Hotelling's $T^2$ statistic to perform a $F$-test whether the mean difference between the new observation $x$ and known data $X_c$ is significant or not. If it is significant, then $x$ probably not in class $c$. We iterate all until we run out of class to iterate and we done. More technical detail would be

### Statistical Modelling

1. **Compute Class Statistics**:
    - For each digit class $c$, calculate the mean vector $\mu_c$ and the precision matrix $S_c^{-1}$.

2. **Hotelling's $T^2$ Test**:
    - For a given test observation $x$, compute the Hotelling's $T^2$ statistic for each class: $$ T^2_c = (x - \mu_c)^T S_c^{-1} (x - \mu_c) $$
    - Convert the $T^2$ statistic to an $F$-value: $$F_c = \frac{n - p}{p \times n} \times T^2_c$$
where $n$ is the number of samples in class $c$ and $p$ is the dimensionality of the data.
    - Derive the $p$-value for this $F$-value using the CDF of the $F$-distribution.

### Classification Decision

- Assign the test observation $x$ to the class $c^*$ with the highest $p$-value:

$$
c^* = \arg\max_c p_c
$$

- If all $p$-values are below a significance threshold (e.g., $0.05$), classify $x$ as "unknown".

## Result

We got accuracy of 90.43% on 50% test data. Here is the visualization of what we got.

![pval](https://github.com/aukkawut/IHateMLClassification/blob/main/pval1.png)

## Why no one do this?

Ok, I will let you evaluate time complexity of the said algorithm as your homework.
