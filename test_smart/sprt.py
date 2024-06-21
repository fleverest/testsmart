"""
This file implements Wald's Sequential Probability Ratio Test using the
`SeqHypothesisTest` framework defined in hypothesis.py
"""

from abc import ABC, abstractmethod

from test_smart.hypothesis import SeqHypothesisTest, Decision

import numpy as np


class LogLikelihood(ABC):
    """
    A base class for implementing log-likelihood functions for use by SPRT.
    """

    @abstractmethod
    def ll(self, x: np.ndarray, theta: np.floating):
        """
        Log-likelihood ratio for the simple-vs-simple hypothesis test defined
        by theta0 and theta1.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Defines the representation of the Likelihood function for display
        in summaries.
        """
        pass


class ExponentialLogLikelihood(LogLikelihood):
    """
    Implements the likelihood function for exponential distributions.
    """

    def ll(self, x: np.ndarray, theta: np.floating):
        return np.log(theta) - x * theta

    def __repr__(self):
        return "<Exponential(theta) log-Likelihood>"


class NormalLogLikelihood(LogLikelihood):
    """
    Implements the log-likelihood function for normal distributions relative to a fixed
    scale parameter sigma.
    """

    def __init__(self, sigma: np.floating = 1):
        self.sigma = sigma

    def ll(self, x: np.ndarray, theta: np.floating):
        # fmt: off
        return (x - theta) ** 2 / (2 * self.sigma**2)
        # fmt: on

    def __repr__(self):
        return f"<Normal(theta, {self.sigma}) log-Likelihood>"


class SPRT(SeqHypothesisTest):
    """
    Wald's Sequential Probability Ratio Test for testing the simple hypothesis
    H_0: theta = theta0 versus H_1: theta = theta1, where theta1 > theta0
    for the mean theta of a population with given distribution.

    Currently, SPRT is implemented for testing simple-versus-simple hypotheses
    for the Exponential scale and Normal location parameters. When testing the Normal
    mean, you may optionally specify the standard deviation `sigma` during
    construction.  If left unspecified, it is assumed to be 1.
    """

    def __init__(
        self,
        alpha: np.floating,
        beta: np.floating,
        theta0: np.floating,
        theta1: np.floating,
        loglikelihood: LogLikelihood,
        **kwargs,
    ):
        super().__init__(alpha)
        self.beta = beta
        self.theta0 = theta0
        self.theta1 = theta1
        self.a = np.log(beta / (1 - alpha))
        self.b = np.log((1 - alpha) / beta)
        self.S = np.array([0])
        self.loglikelihood = loglikelihood

    def observe(self, x: np.ndarray) -> Decision:
        super().observe(x)
        self.S = np.append(
            self.S,
            np.cumsum(
                self.loglikelihood.ll(x, self.theta0)
                - self.loglikelihood.ll(x, self.theta1)
            )
            + self.S[-1],
        )
        if self.S[-1] <= self.a:
            self.decision = Decision.ACCEPT
        elif self.S[-1] >= self.b:
            self.decision = Decision.REJECT
        else:
            self.decision = Decision.CONTINUE
        return self.decision

    def summary(self):
        return {
            "null": f"theta = {self.theta0}",
            "alternative": f"theta = {self.theta1}",
            "loglikelihood": self.loglikelihood,
            "decision": self.decision,
            "n": len(self.observations),
        }
