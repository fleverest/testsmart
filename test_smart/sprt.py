"""
This file implements Wald's Sequential Probability Ratio Test using the
`SeqHypothesisTest` framework defined in hypothesis.py
"""

from test_smart.hypothesis import SeqHypothesisTest, Decision

import numpy as np

class ExponentialSPRT(SeqHypothesisTest):
  """
  Wald's Sequential Probability Ratio Test for testing the simple hypothesis
  H_0: theta = theta0 versus H_1: theta = theta1, where theta1 > theta0
  for an exponential(theta) population.
  """

  def __init__(
      self,
      alpha: np.floating,
      beta: np.floating,
      theta0: np.floating,
      theta1: np.floating
  ):
    super().__init__(alpha)
    self.beta = beta
    self.theta0 = theta0
    self.theta1 = theta1
    self.a = np.log(beta / (1 - alpha))
    self.b = np.log((1 - alpha) / beta)
    self.S = np.array([0])

  def test(self, x: np.ndarray) -> Decision:
    self.observations = np.append(self.observations, x)
    self.S = np.append(
      self.S,
      np.cumsum(self.loglik_ratio(x)) + self.S[-1]
    )
    if self.S[-1] <= self.a:
      self.decision = Decision.ACCEPT
      return self.decision
    elif self.S[-1] >= self.b:
      self.decision = Decision.REJECT
      return self.decision
    else:
      self.decision = Decision.CONTINUE
      return self.decision

  def loglik_ratio(self, x: np.ndarray) -> np.ndarray:
    """
    Calculates the log-likelihood ratio for each observation in `x`.
    """
    llr = np.vectorize(
      lambda xi: -np.log(self.theta1 / self.theta0) + xi * \
                  (self.theta1 - self.theta0) / (self.theta0 * self.theta1)
    )
    return llr(x)

  def summary(self):
    return {
      "decision": self.decision.value,
      "N": len(self.observations)
    }