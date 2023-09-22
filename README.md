# I hate machine learning classification, so I use Hypothesis testing for classifying MNIST data.

## Introduction
As the name suggests, I hate the overrate of the use of machine learning to classifying MNIST data. Like, why do we need to use those fancy algorithm for this simple task? So, I propose to use the more traditional way that can be taught to first year student who are taking basic statistics. Introducing hypothesis testing classification algorithm.

## Side/technical note
This is just a metric learning. Here, we use Mahalanobis distance as our metric and we want to test whether the difference (distance from new observation $x$ to the class vector $X_c$) in said metric is significant or not. If it is significant, then $x$ probably not in class $c$. We iterate all until we run out of class to iterate and we done. In case of tie, we can just randomly pick one for a sacrifice of tiny bit of accuracy.

