import numpy as np

from test_smart.utils import RunningSummaries


class TestRunningSummaries:
    """
    Tests for the various summaries for updated
    """

    x = np.array([1.5, 2.5, 1.7, 1.0, 1.2, 2.1, 1.4, 1.8])

    def test_mean_is_full_mean(self):
        rs = RunningSummaries()
        rs.add(self.x)
        assert np.isclose(rs.mean, np.mean(self.x))

    def test_historical_means_are_partial_means(self):
        rs = RunningSummaries()
        rs.add(self.x)
        assert np.isclose(
            rs.hist_means, (np.cumsum(self.x) / np.arange(1, len(self.x) + 1))[:-1]
        ).all()
