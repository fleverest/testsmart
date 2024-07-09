# The Framework

The general structure of sequential hypothesis tests in the package is provided by `SeqHypothesisTest`.
It defines a simplistic, cyclic testing interface; sample, decide, repeat.

## Decisions

Central to this theme is the concept of a **decision**. In sequential tests at large, we can either reject the null,
accept the null or continue sampling. This is reflected in the elements of the `Decision` enum:

```{eval-rst}
.. autoenum:: testsmart.hypothesis.Decision
```


## The base classes `HypothesisTest` and `SeqHypothesisTest`


A framework for the traditional hypothesis test is provided alongside the sequential framework. In fact, the sequential
testing framework inherits directly from the hypothesis testing framework, providing only the means to "update"
decisions by observing new data. In general, one should be careful to ensure any extension of `HypothesisTest` should
never yield a decision to continue sampling unless no data has been observed.

```{eval-rst}
.. autoclass:: testsmart.hypothesis.HypothesisTest
    :members:
```

The only addition to this for `SeqHypothesisTest` is the ability to update your decision sequentially. 

```{eval-rst}
.. autoclass:: testsmart.hypothesis.SeqHypothesisTest
    :show-inheritance:
    :members:
```