import numpy as np

from abc import ABC, abstractmethod

from testsmart.hypothesis import SeqHypothesisTest, Decision
from testsmart.utils import RunningSummaries, FPRunningSummaries


def bet_to_estimate(lam: float, mu: float, u: float = 1):
    return mu * (1 + lam * (u - mu))


class Estimator(ABC):
    """
    A base class for implementing restricted estimators for the mean of a bounded,
    non-negative random variable.
    """

    def __init__(self, n_total: int = np.inf, u: float = 1, t: float = 1 / 2):
        self.u = u
        self.t = t
        self.n_total = n_total

    @abstractmethod
    def estim(self) -> float:
        """
        Returns an estimate of the mean for a NonnegMean test.
        """
        pass

    def set_u_t_n(self, u: float, t: float, n_total: float) -> None:
        self.u = u
        self.t = t
        self.n_total = n_total

    def override_summaries(
        self, summaries: RunningSummaries | FPRunningSummaries
    ) -> None:
        self.summaries = summaries


class ShrinkTrunc(Estimator):

    def __init__(
        self,
        u: float = 1,
        t: float = 1 / 2,
        n_total: int = np.inf,
        eta0: float | None = None,
        c: float | None = None,
        d: int | None = 100,
        f: float | None = 0,
        minsd: float | None = 1e-6,
    ):
        super().__init__(n_total, u, t)
        if np.isfinite(n_total):
            self.summaries = FPRunningSummaries(n_total, t)
        else:
            self.summaries = RunningSummaries()
        self.eta0 = eta0 if eta0 else u * (1 - np.finfo(float).eps)
        self.c = c if c else (self.eta0 - self.t) / 2
        self.d = d
        self.f = f
        self.minsd = minsd

    def estim(self) -> float:
        """
        Calculate a shrinkage truncation estimate for the mean of a bounded,
        nonnegative data stream.
        """
        # Sum before the last data point
        prev_sum = self.summaries.prev_sum
        if np.isnan(prev_sum):
            prev_sum = 0  # Zero for first samples
        # SD before the last data point
        prev_sd = np.sqrt(self.summaries.prev_var)
        if np.isnan(prev_sd) or prev_sd == 0:
            prev_sd = 1  # Replace with 1 if nan or zero, to avoid division by zero
        # The remaining out-of-sample mean
        m = self.summaries.oos_mean if np.isfinite(self.n_total) else self.t
        count = self.summaries.count
        # Shrinkage estimate, shrunk towards eta0
        shrunk = (self.d * self.eta0 + prev_sum) / (self.d + count - 1)
        # Reshrunk shrinkage estimate, shrunk towards u
        reshrunked = (shrunk + self.u * self.f / prev_sd) / (1 + self.f / prev_sd)
        # Truncated reshrunk estimate
        return np.minimum(
            self.u * (1 - np.finfo(float).eps),
            np.maximum(reshrunked, m + self.c / np.sqrt(self.d + count - 1)),
        )


class Bet(ABC):
    """
    A base class for implementing bets for betting martingale tests on the mean
    for non-negative, bounded random variables.
    """

    def __init__(self, n_total: int = np.inf, u: float = 1, t: float = 1 / 2):
        self.u = u
        self.t = t
        self.n_total = n_total

    @abstractmethod
    def bet(self) -> float:
        """
        Returns a 'bet': a fraction of the total remaining wealth.
        """
        pass

    def set_u_t_n(self, u: float, t: float, n_total: float) -> None:
        self.u = u
        self.t = t
        self.n_total = n_total

    def override_summaries(
        self, summaries: RunningSummaries | FPRunningSummaries
    ) -> None:
        self.summaries = summaries


class FixedBet(Bet):

    def __init__(self, lam: float = 0.5):
        # We don't need super().__init__(...) here, bet just returns a constant.
        self.lam = lam

    def bet(self) -> float:
        return self.lam


class AGRAPA(Bet):

    def __init__(
        self,
        u: float = 1,
        t: float = 1 / 2,
        n_total: int = np.inf,
        lam0: float = 0.5,
        c0: float = 1 - np.finfo(float).eps,
        c_max: float = 1 - np.finfo(float).eps,
        c_grow: float = 0,
    ):
        super().__init__(n_total, u, t)
        if np.isfinite(n_total):
            self.summaries = FPRunningSummaries(n_total, t)
        else:
            self.summaries = RunningSummaries()
        self.lam0 = lam0
        self.c0 = c0
        self.c_max = c_max
        self.c_grow = c_grow

    def bet(self) -> float:
        # mean before the last data pont
        prev_mean = self.summaries.prev_mean
        # var before the last data point
        prev_var = self.summaries.prev_var
        m = self.summaries.oos_mean if np.isfinite(self.n_total) else self.t
        lam = (prev_mean - m) / (prev_var + (prev_mean - m) ** 2)
        if np.isnan(lam):
            lam = self.lam0
        c = self.c0 + (self.c_max - self.c_grow) * (
            1 - 1 / (1 + self.c_grow * np.sqrt(self.summaries.count))
        )
        return np.maximum(0, np.minimum(c / m, lam))


class NonNegMeanTest(SeqHypothesisTest):
    """
    A base class for implementing sequential tests for the hypothesis that a bounded,
    non-negative random variable in [0, u] has mean less than or equal to t, where
    0 < t < u.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_total: float = np.inf,
        u: float = 1,
        t: float = 1 / 2,
    ):
        super().__init__(alpha, n_total)
        self.u = u
        self.t = t
        if self.finite:
            self.summaries = FPRunningSummaries(n_total, t)
        else:
            self.summaries = RunningSummaries()


class AlphaMart(NonNegMeanTest):
    """
    The ALPHA martingale.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_total: float = np.inf,
        u: float = 1,
        t: float = 1 / 2,
        estim: Estimator | Bet | None = None,
        atol: float = np.finfo(float).eps,
        rtol: float = 10**-6,
    ):
        super().__init__(alpha, n_total, u, t)
        if estim:
            self.estim = estim
        else:
            self.estim = ShrinkTrunc(u, t, n_total)
        self.estim.override_summaries(self.summaries)
        self.atol = atol
        self.rtol = rtol
        self._eta_history = []
        self.e_hist = []
        self.e_process = []
        self.p_history = []

    @property
    def pval(self) -> float:
        return self.p_history[-1] if self.p_history else np.nan

    def update(self, x: list[float]) -> None:
        super().update(x)
        # Loop through them and calculate p-values one at a time.
        for xi in x:
            # Update summaries
            self.summaries.add([xi])
            # Compute out-of-sample mean
            m = self.summaries.oos_mean if self.finite else self.t
            # Estimate eta
            if isinstance(self.estim, Estimator):
                eta = self.estim.estim()  # Do not pass x to avoid updating twice
            elif isinstance(self.estim, Bet):
                eta = bet_to_estimate(self.estim.bet(), m, self.u)
            self._eta_history.append(eta)
            # compute e-value of sample xi
            if m > self.u:
                e_i = 0  # True mean certainly less than hypothesised
            elif m < 0:
                e_i = np.inf  # True mean certainly greater than hypothesised
            elif np.isclose(0, m, atol=self.atol) or np.isclose(
                self.u, m, atol=self.atol, rtol=self.rtol
            ):
                e_i = 1  # Ignore
            else:
                e_i = (
                    xi * eta / m + (self.u - xi) * (self.u - eta) / (self.u - m)
                ) / self.u
            self.e_hist.append(e_i)

            # Add to e_process (in this case it is just the cumulative product
            # of e-values)
            self.e_process.append(e_i * self.e_process[-1] if self.e_process else e_i)

            # Compute p-value and add to history: p is the smallest 1/e seen so far
            p = min(
                1, self.pval if not np.isnan(self.pval) else 1, 1 / self.e_process[-1]
            )
            self.p_history.append(p)

        if self.p_history[-1] < self.alpha:
            self.decision = Decision.REJECT
        else:
            self.decision = Decision.CONTINUE
        return self.decision
