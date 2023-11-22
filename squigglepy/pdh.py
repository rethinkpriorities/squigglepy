from abc import ABC, abstractmethod
import numpy as np
from scipy import optimize, stats
import sortednp as snp
from typing import Optional
import warnings

from .distributions import LognormalDistribution, lognorm
from .samplers import sample


class PDHBase(ABC):
    def __len__(self):
        return self.num_bins

    def histogram_mean(self):
        """Mean of the distribution, calculated using the histogram data (even
        if the exact mean is known)."""
        return np.sum(self.masses * self.values)

    def mean(self):
        """Mean of the distribution. May be calculated using a stored exact
        value or the histogram data."""
        if self.exact_mean is not None:
            return self.exact_mean
        return self.histogram_mean()

    def histogram_sd(self):
        """Standard deviation of the distribution, calculated using the
        histogram data (even if the exact SD is known)."""
        mean = self.mean()
        return np.sqrt(np.sum(self.masses * (self.values - mean) ** 2))

    def sd(self):
        """Standard deviation of the distribution. May be calculated using a
        stored exact value or the histogram data."""
        return self.histogram_sd()

    @classmethod
    def _fraction_of_ev(cls, values: np.ndarray, masses: np.ndarray, x: np.ndarray | float):
        """Return the approximate fraction of expected value that is less than
        the given value.
        """
        if isinstance(x, np.ndarray):
            return np.array([cls._fraction_of_ev(values, masses, xi) for xi in x])
        mean = np.sum(masses * values)
        return np.sum(masses * values * (values <= x)) / mean

    @classmethod
    def _inv_fraction_of_ev(
        cls, values: np.ndarray, masses: np.ndarray, fraction: np.ndarray | float
    ):
        if isinstance(fraction, np.ndarray):
            return np.array([cls.inv_fraction_of_ev(values, masses, xi) for xi in fraction])
        if fraction <= 0:
            raise ValueError("fraction must be greater than 0")
        mean = np.sum(masses * values)
        fractions_of_ev = np.cumsum(masses * values) / mean
        epsilon = 1e-10  # to avoid floating point rounding issues
        index = np.searchsorted(fractions_of_ev, fraction - epsilon)
        return values[index]

    def fraction_of_ev(self, x: np.ndarray | float):
        """Return the approximate fraction of expected value that is less than
        the given value.
        """
        return self._fraction_of_ev(self.values, self.masses, x)

    def inv_fraction_of_ev(self, fraction: np.ndarray | float):
        """Return the value such that `fraction` of the contribution to
        expected value lies to the left of that value.
        """
        return self._inv_fraction_of_ev(self.values, self.masses, fraction)

    def __add__(x, y):
        extended_values = np.add.outer(x.values, y.values).flatten()
        res = x.binary_op(y, extended_values, ev=x.mean() + y.mean())
        if x.exact_mean is not None and y.exact_mean is not None:
            res.exact_mean = x.exact_mean + y.exact_mean
        if x.exact_sd is not None and y.exact_sd is not None:
            res.exact_sd = np.sqrt(x.exact_sd**2 + y.exact_sd**2)
        return res

    def __mul__(x, y):
        extended_values = np.outer(x.values, y.values).flatten()
        res = x.binary_op(y, extended_values, ev=x.mean() * y.mean(), is_mul=True)
        if x.exact_mean is not None and y.exact_mean is not None:
            res.exact_mean = x.exact_mean * y.exact_mean
        if x.exact_sd is not None and y.exact_sd is not None:
            res.exact_sd = np.sqrt(
                (x.exact_sd * y.exact_mean) ** 2
                + (y.exact_sd * x.exact_mean) ** 2
                + (x.exact_sd * y.exact_sd) ** 2
            )
        return res


class ScaledBinHistogram(PDHBase):
    """PDH with exponentially growing bin widths."""

    def __init__(
        self,
        left_bound: float,
        right_bound: float,
        bin_scale_rate: float,
        bin_densities: np.ndarray,
        exact_mean: Optional[float] = None,
        exact_sd: Optional[float] = None,
    ):
        # TODO: currently only supports positive-everywhere distributions
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.bin_scale_rate = bin_scale_rate
        self.bin_densities = bin_densities
        self.exact_mean = exact_mean
        self.exact_sd = exact_sd
        self.num_bins = len(bin_densities)
        self.bin_edges = self.get_bin_edges(
            self.left_bound, self.right_bound, self.bin_scale_rate, self.num_bins
        )
        self.values = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_widths = np.diff(self.bin_edges)
        self.masses = self.bin_densities * self.bin_widths

    @staticmethod
    def get_bin_edges(left_bound: float, right_bound: float, bin_scale_rate: float, num_bins: int):
        num_scaled_bins = int(num_bins / 2)
        num_fixed_bins = num_bins - num_scaled_bins
        min_width = (right_bound - left_bound) / (
            num_fixed_bins + sum(bin_scale_rate**i for i in range(num_scaled_bins))
        )
        bin_widths = np.concatenate(
            (
                [min_width for _ in range(num_fixed_bins)],
                [min_width * bin_scale_rate**i for i in range(num_scaled_bins)],
            )
        )
        return np.cumsum(np.concatenate(([left_bound], bin_widths)))

    def bin_density(self, index: int) -> float:
        return self.bin_densities[index]

    def binary_op(x, y, extended_values, ev, is_mul=False):
        # Note: This implementation is not nearly as well-optimized as
        # ProbabilityMassHistogram.
        extended_masses = np.outer(x.masses, y.masses).flatten()

        # Sort the arrays so product values are in order
        sorted_indexes = extended_values.argsort()
        extended_values = extended_values[sorted_indexes]
        extended_masses = extended_masses[sorted_indexes]

        num_bins = max(len(x), len(y))
        outer_ev = 1 / num_bins / 2

        left_bound = PDHBase._inv_fraction_of_ev(extended_values, extended_masses, outer_ev)
        right_bound = PDHBase._inv_fraction_of_ev(extended_values, extended_masses, 1 - outer_ev)
        bin_scale_rate = np.sqrt(x.bin_scale_rate * y.bin_scale_rate)
        bin_edges = ScaledBinHistogram.get_bin_edges(
            left_bound, right_bound, bin_scale_rate, num_bins
        )

        # Split masses into bins with bin_edges as delimiters
        split_masses = np.split(extended_masses, np.searchsorted(extended_values, bin_edges))[1:-1]

        bin_densities = []
        for i, elem_masses in enumerate(split_masses):
            mass = np.sum(elem_masses)
            density = mass / (bin_edges[i + 1] - bin_edges[i])
            bin_densities.append(density)

        return ScaledBinHistogram(left_bound, right_bound, bin_scale_rate, np.array(bin_densities))

    @classmethod
    def from_distribution(cls, dist, num_bins=1000, bin_scale_rate=None):
        if not isinstance(dist, LognormalDistribution):
            raise ValueError("Only LognormalDistributions are supported")

        left_bound = dist.inv_fraction_of_ev(1 / num_bins / 2)
        right_bound = dist.inv_fraction_of_ev(1 - 1 / num_bins / 2)

        def compute_bin_densities(bin_scale_rate):
            bin_edges = cls.get_bin_edges(left_bound, right_bound, bin_scale_rate, num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            edge_densities = stats.lognorm.pdf(
                bin_edges, dist.norm_sd, scale=np.exp(dist.norm_mean)
            )
            center_densities = stats.lognorm.pdf(
                bin_centers, dist.norm_sd, scale=np.exp(dist.norm_mean)
            )
            # Simpson's rule
            bin_densities = (edge_densities[:-1] + 4 * center_densities + edge_densities[1:]) / 6
            return bin_densities

        def loss(bin_scale_rate_arr):
            bin_scale_rate = bin_scale_rate_arr[0]
            bin_densities = compute_bin_densities(bin_scale_rate)
            hist = cls(left_bound, right_bound, bin_scale_rate, bin_densities)
            mean_error = (hist.mean() - dist.lognorm_mean) ** 2
            sd_error = (hist.sd() - dist.lognorm_sd) ** 2
            return mean_error

        if bin_scale_rate is None and num_bins == 1000:
            bin_scale_rate = 1.04
        elif bin_scale_rate is None:
            bin_scale_rate = optimize.minimize(loss, bin_scale_rate, bounds=[(1, 2)]).x[0]
        bin_edges = cls.get_bin_edges(left_bound, right_bound, bin_scale_rate, num_bins)
        bin_densities = compute_bin_densities(bin_scale_rate)

        return cls(
            left_bound,
            right_bound,
            bin_scale_rate,
            bin_densities,
            exact_mean=dist.lognorm_mean,
            exact_sd=dist.lognorm_sd,
        )


class ProbabilityMassHistogram(PDHBase):
    """Represent a probability distribution as an array of x values and their
    probability masses. Like Monte Carlo samples except that values are
    weighted by probability, so you can effectively represent many times more
    samples than you actually have values."""

    def __init__(
        self,
        values: np.ndarray,
        masses: np.ndarray,
        exact_mean: Optional[float] = None,
        exact_sd: Optional[float] = None,
    ):
        assert len(values) == len(masses)
        self.values = values
        self.masses = masses
        self.num_bins = len(values)
        self.exact_mean = exact_mean
        self.exact_sd = exact_sd

    def binary_op(x, y, extended_values, ev, is_mul=False):
        extended_masses = np.ravel(np.outer(x.masses, y.masses))
        num_bins = max(len(x), len(y))
        len_per_bin = int(len(extended_values) / num_bins)
        ev_per_bin = ev / num_bins
        extended_evs = extended_masses * extended_values

        # Cut boundaries between bins such that each bin has equal contribution
        # to expected value.
        if is_mul:
            # When multiplying, the values of extended_evs are all equal. x and
            # y both have the property that every bin contributes equally to
            # EV, which means the outputs of their outer product must all be
            # equal. We can use this fact to avoid a relatively slow call to
            # `cumsum` (which can also introduce floating point rounding errors
            # for extreme values).
            bin_boundaries = np.arange(1, num_bins) * len_per_bin
        else:
            cumulative_evs = np.cumsum(extended_evs)
            bin_boundaries = np.searchsorted(cumulative_evs, np.arange(ev_per_bin, ev, ev_per_bin))

        # Partition the arrays so every value in a bin is smaller than every
        # value in the next bin, but don't sort within bins. (Partitioning is
        # about 10% faster than mergesort.)
        sorted_indexes = extended_values.argpartition(bin_boundaries)
        extended_values = extended_values[sorted_indexes]
        extended_masses = extended_masses[sorted_indexes]

        bin_values = []
        bin_masses = []
        if is_mul:
            # Take advantage of the fact that all bins contain the same number
            # of elements
            bin_masses = extended_masses.reshape((num_bins, -1)).sum(axis=1)
            bin_values = ev_per_bin / bin_masses
        else:
            bin_boundaries = np.concatenate(([0], bin_boundaries, [len(extended_evs)]))
            for i in range(len(bin_boundaries) - 1):
                start = bin_boundaries[i]
                end = bin_boundaries[i+1]
                mass = np.sum(extended_masses[start:end])
                value = np.sum(extended_evs[start:end]) / mass
                bin_values.append(value)
                bin_masses.append(mass)

        return ProbabilityMassHistogram(np.array(bin_values), np.array(bin_masses))

    @classmethod
    def from_distribution(cls, dist, num_bins=1000):
        """Create a probability mass histogram from the given distribution. The
        histogram covers the full distribution except for the 1/num_bins/2
        expectile on the left and right tails. The boundaries are based on the
        expectile rather than the quantile to better capture the tails of
        fat-tailed distributions, but this can cause computational problems for
        very fat-tailed distributions.
        """
        if not isinstance(dist, LognormalDistribution):
            raise ValueError("Only LognormalDistributions are supported")

        assert num_bins % 100 == 0, "num_bins must be a multiple of 100"
        boundary = 1 / num_bins
        edge_values = np.concatenate(
            (
                [0],
                dist.inv_fraction_of_ev(np.linspace(boundary, 1 - boundary, num_bins - 1)),
                [np.inf],
            )
        )

        # How much each bin contributes to total EV.
        contribution_to_ev = dist.lognorm_mean / num_bins

        # We can compute the exact mass of each bin as the difference in
        # CDF between the left and right edges.
        masses = np.diff(
            stats.lognorm.cdf(edge_values, dist.norm_sd, scale=np.exp(dist.norm_mean)),
        )

        # Assume the value exactly equals the bin's contribution to EV
        # divided by its mass. This means the values will not be exactly
        # centered, but it guarantees that the expected value of the
        # histogram exactly equals the expected value of the distribution
        # (modulo floating point rounding).
        values = contribution_to_ev / masses

        # For sufficiently large values, CDF rounds to 1 which makes the
        # mass 0.
        #
        # Note: It would make logical sense to remove zero values, but it
        # messes up the binning algorithm for products which expects the number
        # of values to be a multiple of the number of bins.
        # values = values[masses > 0]
        # masses = masses[masses > 0]
        values = np.where(masses == 0, 0, values)

        if any(masses == 0):
            num_zeros = np.sum(masses == 0)
            warnings.warn(f"{num_zeros} values greater than {values[-1]} had CDFs of 1.", RuntimeWarning)

        return cls(
            np.array(values),
            np.array(masses),
            exact_mean=dist.lognorm_mean,
            exact_sd=dist.lognorm_sd,
        )
