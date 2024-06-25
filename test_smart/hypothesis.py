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
        The `update` method takes as input some data, updates the internal state of
        the test and returns the testing decision.
        """
        pass

    @abstractmethod
    def pval(self) -> float | None:
        """
        The `pval` method returns the p-value for the statistical test.
        """
        return self._pval

    def summary(self) -> dict:
        """
        The `summary` method takes no input and should return some summary data,
        e.g. descriptions, test decisions, p-values or statistics related to the
        hypothesis being tested.
        """
        return {"alpha": self.alpha, "p": self.pval, "decision": self.decision}


class SeqHypothesisTest(HypothesisTest):
    """
    A base class for implementing sequential hypothesis tests.
    """

    def __init__(self, alpha: float, n_total: int) -> None:
        super().__init__(alpha)
        self._observations = []
        self.n_total = n_total
        self.finite = np.isfinite(n_total)

    @property
    def observations(self) -> np.ndarray:
        return np.array(self.observations)

    def update(self, x: np.ndarray) -> Decision:
        self._observations.extend(x)
        pass

    def stopped(self) -> bool:
        return self.decision != Decision.CONTINUE

    def summary(self) -> dict:
        return dict(
            super().summary(),
            **{"n_observations": len(self._observations), "n_total": self.n_total},
        )
