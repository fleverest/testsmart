import numpy as np


class TooManySamplesError(Exception):
    """
    An error raised when the number of samples observed exceeds the prespecified
    population size.
    """

    def __init__(self, new_size, old_size, population_size):
        super().__init__(
            f"Exceeded population size: cannot add further observations. Tried to add"
            f" {new_size} observations when {old_size}/{population_size} observations"
            f" had already been taken."
        )


class RunningSummaries:
    """
    A class for calculating some summaries of a stream of data:
      1. Count,
      2. cumulative sum,
      3. running mean (Welford's algorithm),
      4. running (population) variance (Welford's algorithm),
    These summaries are useful for a variety of inferential tasks.
    """

    def __init__(self):
        self._data = []
        self._count = 0
        self._sums = []
        self._means = []
        self._vars = []

    @property
    def hist_sums(self) -> list[float]:
        """
        The historical sums of the data stream, up to the current time point.
        """
        return self._sums

    @property
    def prev_sum(self) -> float:
        """
        The previous sum of the data stream, i.e. at the second-to-last time point.
        """
        return self._sums[-2] if len(self._sums) > 1 else 0

    @property
    def sum(self) -> float:
        """
        The current total sum of the data stream.
        """
        return self._sums[-1] if self._sums else 0

    @property
    def hist_means(self) -> list[float]:
        """
        The historical running mean of the data stream, up to the current time.
        """
        return self._means

    @property
    def prev_mean(self) -> float:
        """
        The previous mean of the data stream, i.e. at the second-to-last time point.
        """
        return self._means[-2] if len(self._means) > 1 else np.nan

    @property
    def mean(self) -> float:
        """
        The current mean of the data stream.
        """
        return self._means[-1] if self._means else np.nan

    @property
    def hist_vars(self) -> list[float]:
        """
        The historical running (population) variance of the data stream, up to the
        current time.
        """
        return self._vars

    @property
    def prev_var(self) -> float:
        """
        The previous variance of the data stream, i.e. at the second-to-last time point.
        """
        return self._vars[-2] if len(self._vars) > 1 else np.nan

    @property
    def var(self) -> float:
        """
        The current (population) variance of the data stream.
        """
        return self._vars[-1] if self._vars else np.nan

    @property
    def count(self) -> int:
        """
        The current number of observations.
        """
        return self._count

    def add(self, x: list[float]) -> None:
        if self.count == 0:
            # Starting values
            self._data.append(x[0])
            self._count = 1
            self._sums.append(x[0])
            self._means.append(x[0])
            self._vars.append(0)
            x = x[1:]
        for xn in x:
            # Updates
            self._data.append(xn)
            # Increment counter
            self._count = self._count + 1
            # Add to cumulative sum
            self._sums.append(self.sum + xn)
            # Welford's online running mean and (population) variance
            self._means.append(self.sum / self.count)
            self._vars.append(
                self.var
                + ((xn - self._means[-2]) * (xn - self._means[-1]) - self.var)
                / self.count
            )


class FPRunningSummaries(RunningSummaries):
    """
    Just like RunningSummaries but with extras for finite-populations:
    Currently, also calculates out-of-sample means, given an initial reference mean.
    """

    def __init__(self, pop_size: int, pop_mean: float):
        super().__init__()
        self._pop_size = pop_size
        self._pop_mean = pop_mean
        self._oos_means = [self.pop_mean]
        self._oos_sums = [self.pop_mean * self.pop_size]

    @property
    def pop_mean(self):
        """
        The mean of the full population.
        """
        return self._pop_mean

    @property
    def pop_size(self):
        """
        The size of the full population.
        """
        return self._pop_size

    @property
    def oos_count(self):
        """
        The number of samples remaining in the population which haven't been sampled.
        """
        return self.pop_size - self.count

    @property
    def hist_oos_means(self):
        """
        The historical out-of-sample means of the data stream, up to the current time.
        """
        return self._oos_means

    @property
    def prev_oos_mean(self):
        """
        The previous out-of-sample mean of the data stream, i.e. at the second-to-last
        time point.
        """
        return self._oos_means[-2] if len(self._oos_means) > 1 else np.nan

    @property
    def oos_mean(self):
        """
        The current out-of-sample mean of the data stream.
        """
        return self._oos_means[-1] if self._oos_means else np.nan

    @property
    def hist_oos_sums(self):
        """
        The historical out-of-sample sums of the data stream, up to the current time.
        """
        return self._oos_sums

    @property
    def prev_oos_sum(self):
        """
        The previous out-of-sample sum of the data stream, i.e. at the second-to-last
        time point.
        """
        return self._oos_sums[-2] if len(self._oos_sums) > 1 else np.nan

    @property
    def oos_sum(self):
        """
        The current out-of-sample sum of the data stream.
        """
        return self._oos_sums[-1]

    def add(self, x: list[float]) -> None:
        if len(x) > self.oos_count:
            raise TooManySamplesError(len(x), self.count, self.pop_size)
        super().add(x)
        new_oos_sums = [self._oos_sums[0] - s for s in self._sums[-len(x) :]]
        self._oos_sums.extend(new_oos_sums)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._oos_means.extend(
                new_oos_sums
                / np.arange(self.oos_count + len(x) - 1, self.oos_count - 1, -1)
            )
