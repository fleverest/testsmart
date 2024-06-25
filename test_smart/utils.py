import numpy as np


class RunningSummaries:
    """
    A class for calculating some summaries of a stream of data:
      1. Count
      2. Cumulative sum
      3. Running mean (Welford's algorithm)
      4. Running (population) variance (Welford's algorithm)
    These summaries are useful for a variety of inferential tasks.
    """

    def __init__(self):
        self._data = []
        self._count = 0
        self._cumsums = []
        self._means = []
        self._vars = []

    @property
    def hist_sums(self) -> np.ndarray:
        """
        The historical sums of the data stream, up to the current time.
        """
        return np.array(self._cumsums[:-1])

    @property
    def sum(self) -> float:
        """
        The current total sum of the data stream.
        """
        return self._cumsums[-1] if self._cumsums else 0

    @property
    def hist_means(self) -> np.ndarray:
        """
        The historical running mean of the data stream, up to the current time.
        """
        return np.array(self._means[:-1])

    @property
    def mean(self) -> float:
        """
        The current mean of the data stream.
        """
        return self._means[-1] if self._means else np.nan

    @property
    def hist_vars(self) -> np.ndarray:
        """
        The historical running (population) variance of the data stream, up to the
        current time.
        """
        return np.array(self._vars[:-1])

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

    def add(self, x: np.ndarray) -> None:
        if self.count == 0:
            # Starting values
            self._data.append(x[0])
            self._count = 1
            self._cumsums.append(x[0])
            self._means.append(x[0])
            self._vars.append(0)
            x = x[1:]
        for xn in x:
            # Updates
            self._data.append(xn)
            # Increment counter
            self._count = self._count + 1
            # Add to cumulative sum
            self._cumsums.append(self.sum + xn)
            # Welford's online running mean and (population) variance
            self._means.append(self.sum / self.count)
            self._vars.append(
                self.var
                + ((xn - self._means[-2]) * (xn - self._means[-1]) - self.var)
                / self.count
            )
