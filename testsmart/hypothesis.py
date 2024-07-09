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

    alpha: float
    decision: Decision

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.decision = Decision.CONTINUE
        self._pval = None

    @abstractmethod
    def update(self, x: np.ndarray) -> Decision:
        """
        The `update` method takes as input some data, updates the internal state
        of the test and returns the testing decision.

        :param x: The observed data.
        :type x: np.ndarray
        :return: The resulting decision; for a standard hypothesis test this is
                 either to accept or reject the null hypothesis
        :rtype: Decision
        """
        pass

    def summary(self) -> dict:
        """
        Important summaries for the test.

        :return: A dictionary including the decision, the significance level of
                 the test, and the final p-value.
        :rtype: dict
        """
        return {"alpha": self.alpha, "p": self._pval, "decision": self.decision}


class SeqHypothesisTest(HypothesisTest):
    """
    A base class for implementing sequential hypothesis tests.
    """

    def __init__(self, alpha: float, n_total: int | np.floating) -> None:
        """
        Instantiate a new SeqHypothesisTest

        :param alpha: The significance level for the test.
        :type alpha: float
        :param n_total: The size of the population being sampled from, or `numpy.inf`
                        if sampling with replacement.
        :type n_total: int
        """
        super().__init__(alpha)
        self._observations = []
        self.n_total = n_total
        self.finite = np.isfinite(n_total)

    @property
    def observations(self) -> np.ndarray:
        return np.array(self.observations)

    @property
    def stopped(self) -> bool:
        """
        Indicates whether a sequential test is stopped or not. A test is considered
        stopped if its' decision is not to continue sampling.

        :return: True if the test is stopped, False otherwise.
        :rtype: bool
        """
        return self.decision != Decision.CONTINUE

    def update(self, x: np.ndarray) -> Decision:
        self._observations.extend(x)
        pass

    def summary(self) -> dict:
        """
        Important summaries for the test.

        :return: A dictionary including the decision, the significance level of the
                 test, the final p-value, the number of observations taken and the
                 total size of the population.
        :rtype: dict
        """
        return dict(
            super().summary(),
            **{"n_observations": len(self._observations), "n_total": self.n_total},
        )
