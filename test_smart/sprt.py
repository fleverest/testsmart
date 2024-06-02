"""
This file implements Wald's Sequential Probability Ratio Test using the
`SeqHypothesisTest` framework defined in hypothesis.py
"""

from test_smart.hypothesis import SeqHypothesisTest, Decision

from typing import Callable

import numpy as np


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
        llr: Callable[[np.floating, np.floating, np.ndarray], np.ndarray],
        **kwargs,
    ):
        super().__init__(alpha)
        self.beta = beta
        self.theta0 = theta0
        self.theta1 = theta1
        self.a = np.log(beta / (1 - alpha))
        self.b = np.log((1 - alpha) / beta)
        self.S = np.array([0])
        self.llr = llr(theta0, theta1)

    def test(self, x: np.ndarray) -> Decision:
        self.observations = np.append(self.observations, x)
        self.S = np.append(
            self.S, np.cumsum(self.llr(x)) + self.S[-1]
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
            "decision": self.decision.value,
            "N": len(self.observations),
        }


def exponential_llr() -> Callable[[np.floating, np.floating], np.ndarray]:
    return lambda theta0, theta1: np.vectorize(
        lambda xi: -np.log(theta1 / theta0) + xi * (theta1 - theta0) / (theta0 * theta1)
    )


def normal_llr(
    sigma: np.floating = 1,
) -> Callable[[np.floating, np.floating, np.ndarray], np.ndarray]:
    return lambda theta0, theta1: np.vectorize(
        lambda xi: (xi - theta0) ** 2 / (2 * sigma**2)
        - (xi - theta1) ** 2 / (2 * sigma**2)
    )
