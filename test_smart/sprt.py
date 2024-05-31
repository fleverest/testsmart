"""
This file implements Wald's Sequential Probability Ratio Test using the
`SeqHypothesisTest` framework defined in hypothesis.py
"""

from test_smart.hypothesis import SeqHypothesisTest, Decision
from test_smart.utils import QueryEnum

import numpy as np


class Distribution(QueryEnum):
    """
    The distributions implemented for SPRT.
    """

    EXPONENTIAL = "Exponential"
    NORMAL = "Normal"


class UnknownDistributionError(ValueError):
    def __init__(self):
        super().__init__(
            f"Unknown Distribution. "
            f"Expected one of {[d.value for d in Distribution]}"
        )


class SPRT(SeqHypothesisTest):
    """
    Wald's Sequential Probability Ratio Test for testing the simple hypothesis
    H_0: theta = theta0 versus H_1: theta = theta1, where theta1 > theta0
    for the mean theta of a population with given distribution.

    Currently, SPRT is implemented for testing simple-versus-simple hypotheses
    for the Exponential Normal location parameters. When testing the Normal
    mean, you may optionally specify the standard deviation `sigma` during
    construction.  If left unspecified, it is assumed to be 1.
    """

    def __init__(
        self,
        alpha: np.floating,
        beta: np.floating,
        theta0: np.floating,
        theta1: np.floating,
        dist: Distribution | str,
        **kwargs,
    ):
        super().__init__(alpha)
        self.beta = beta
        self.theta0 = theta0
        self.theta1 = theta1
        self.a = np.log(beta / (1 - alpha))
        self.b = np.log((1 - alpha) / beta)
        self.S = np.array([0])
        if isinstance(dist, str):
            self.dist = Distribution.from_str(dist)
        elif isinstance(dist, Distribution):
            self.dist = dist
        else:
            raise UnknownDistributionError()
        self.__parse_kwargs(**kwargs)

    def __parse_kwargs(self, **kwargs):
        """
        Loads additional positional arguments.
        """
        if self.dist == Distribution.NORMAL:
            # Load standard deviation for Normal SPRT
            self.sigma = kwargs.pop("sigma", 1)

    def test(self, x: np.ndarray) -> Decision:
        self.observations = np.append(self.observations, x)
        self.S = np.append(self.S, np.cumsum(self.llr(x)) + self.S[-1])
        if self.S[-1] <= self.a:
            self.decision = Decision.ACCEPT
        elif self.S[-1] >= self.b:
            self.decision = Decision.REJECT
        else:
            self.decision = Decision.CONTINUE
        return self.decision

    def llr(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the log-likelihood ratio for each observation in `x`.
        """
        match self.dist:
            case Distribution.EXPONENTIAL:
                return self.exponential_llr(x)
            case Distribution.NORMAL:
                return self.normal_llr(x)

    def exponential_llr(self, x: np.ndarray) -> np.ndarray:
        return np.vectorize(
            lambda xi: -np.log(self.theta1 / self.theta0)
            + xi * (self.theta1 - self.theta0) / (self.theta0 * self.theta1)
        )(x)

    def normal_llr(self, x: np.ndarray) -> np.ndarray:
        return np.vectorize(
            lambda xi: (xi - self.theta0) ** 2 / (2 * self.sigma**2)
            - (xi - self.theta1) ** 2 / (2 * self.sigma**2)
        )(x)

    def summary(self):
        return {"decision": self.decision.value, "N": len(self.observations)}
