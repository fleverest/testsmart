"""
This file defines a generic structure for hypothesis tests, from which a variety
of testing procedures can be built.
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

_DEFAULT_P_THRESHOLD = np.float64(0.05)

class Decision(Enum):
  """
  Outcomes for a statistical hypothesis test.
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

  def __init__(self, alpha: np.floating = _DEFAULT_P_THRESHOLD) -> None:
    self.alpha = alpha
    self.decision = Decision.CONTINUE

  @abstractmethod
  def test(self, x: np.ndarray) -> Decision:
    """
    The `test` method takes as input some data and outputs a test decision.
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

  # A history of computed p-values
  p_history: np.ndarray
  # The current set of observations
  observations: np.ndarray

  def __init__(self, alpha: np.floating = _DEFAULT_P_THRESHOLD) -> None:
    super().__init__(alpha)
    self.observations = np.array([])
    self.p_history = np.array([])