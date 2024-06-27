import numpy as np

from test_smart.utils import RunningSummaries, FPRunningSummaries


class TestRunningSummaries:
    """
    Tests for RunningSummaries and FPRunningSummaries.
    """

    x = np.array([1.5, 2.5, 1.7, 1.0, 1.2, 2.1, 1.4, 1.8])

    def test_mean_is_full_mean(self):
        rs = RunningSummaries()
        rs.add(self.x)
        assert np.isclose(rs.mean, np.mean(self.x))

    def test_historical_means_are_partial_means(self):
        rs = RunningSummaries()
        rs.add(self.x)
        assert np.allclose(
            rs.hist_means, (np.cumsum(self.x) / np.arange(1, len(self.x) + 1))
        )

    def test_final_oos_mean_is_full_mean(self):
        rs = FPRunningSummaries(pop_size=len(self.x), pop_mean=np.mean(self.x))
        rs.add(self.x[:-1])
        assert np.isclose(self.x[-1], rs.oos_mean)

    def test_final_oos_count_is_zero(self):
        rs = FPRunningSummaries(pop_size=len(self.x), pop_mean=np.mean(self.x))
        rs.add(self.x)
        assert rs.oos_count == 0
