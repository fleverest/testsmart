"""
This file defines a generic structure for hypothesis tests, from which a variety
of testing procedures can be built.
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Decision(Enum):
    """
    Decisions for statistical hypothesis tests.
    See "Sequential Tests of Statistical Hypotheses" by A. Wald.
    """

    ACCEPT = "Accept the null hypothesis"
    REJECT = "Reject the null hypothesis"
    CONTINUE = "Continue testing"

    def __str__(self):
        return self.value


class HypothesisTest(ABC):
    """
    A base class for implementing hypothesis tests.
    """

    alpha: np.floating
    decision: Decision

    def __init__(self, alpha: np.floating) -> None:
        self.alpha = alpha
        self.decision = Decision.CONTINUE

    @abstractmethod
    def observe(self, x: np.ndarray) -> Decision:
        """
        The `observe` method takes as input some data, updates the internal state of
        the test and returns the testing decision.
        """
        pass

    @abstractmethod
    def pval(self) -> np.floating:
        """
        The `pval` method returns the p-value for the statistical test.
        """
        pass

    @abstractmethod
    def summary(self) -> dict:
        """
        The `summary` method takes no input and should return some summary data,
        e.g. descriptions, test decisions, p-values or statistics related to the
        hypothesis being tested.
        """
        pass


class SeqHypothesisTest(HypothesisTest):
    """
    A base class for implementing sequential hypothesis tests.
    """

    # The current set of observations
    observations: np.ndarray

    def __init__(self, alpha: np.floating, n_total: np.integer) -> None:
        super().__init__(alpha)
        self.observations = np.array([])
        self.n_total = n_total

    def observe(self, x: np.ndarray) -> Decision:
        self.observations = np.append(self.observations, x)
        pass
    
    def stopped(self) -> bool:
        return self.decision != Decision.CONTINUE


class TooManySamplesError(Exception):
    """
    An error raised when the number of samples observed exceeds the prespecified
    population size.
    """
    pass
