from abc import ABC, abstractmethod
import numpy as np
from scipy import optimize, stats
from typing import Optional

from .distributions import LognormalDistribution, lognorm
from .samplers import sample


class PDHBase(ABC):
    def histogram_mean(self):
        """Mean of the distribution, calculated using the histogram data."""
        return np.sum(self.masses * self.values)

    def mean(self):
        """Mean of the distribution. May be calculated using a stored exact
        value or the histogram data."""
        return self.histogram_mean()

    def std(self):
        """Standard deviation of the distribution."""
        mean = self.mean()
        return np.sqrt(np.sum(self.masses * (self.values - mean) ** 2))

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
        epsilon = 1e-6  # to avoid floating point rounding issues
        index = np.searchsorted(fractions_of_ev, fraction - epsilon)
        return values[index]

    def fraction_of_ev(self, x: np.ndarray | float):
        """Return the approximate fraction of expected value that is less than
        the given value.
        """
        return self._fraction_of_ev(self.values, self.masses, fraction)

    def inv_fraction_of_ev(self, fraction: np.ndarray | float):
        """Return the value such that `fraction` of the contribution to
        expected value lies to the left of that value.
        """
        return self._inv_fraction_of_ev(self.values, self.masses, fraction)


class ScaledBinHistogram(PDHBase):
    """PDH with exponentially growing bin widths."""

    def __init__(
        self,
        left_bound: float,
        right_bound: float,
        bin_scale_rate: float,
        bin_densities: np.ndarray,
    ):
        # TODO: currently only supports positive-everywhere distributions
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.bin_scale_rate = bin_scale_rate
        self.bin_densities = bin_densities
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

    def __len__(self):
        return self.num_bins

    def bin_density(self, index: int) -> float:
        return self.bin_densities[index]

    def __mul__(x, y):
        extended_masses = np.outer(
            x.bin_densities * x.bin_widths, y.bin_densities * y.bin_widths
        ).flatten()
        extended_values = np.outer(x.values, y.values).flatten()
        return x.binary_op(y, extended_masses, extended_values)

    def binary_op(x, y, extended_masses, extended_values):
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

    def mean(self):
        """Mean of the distribution."""
        return np.sum(self.values * self.masses)

    def std(self):
        return np.sqrt(np.sum(self.masses * (self.values - self.mean()) ** 2))

    @classmethod
    def from_distribution(cls, dist, num_bins=1000):
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
            sd_error = (hist.std() - dist.lognorm_sd) ** 2
            return mean_error

        bin_scale_rate = 1.04
        if num_bins != 1000:
            bin_scale_rate = optimize.minimize(loss, bin_scale_rate, bounds=[(1, 2)]).x[0]
        bin_edges = cls.get_bin_edges(left_bound, right_bound, bin_scale_rate, num_bins)
        bin_densities = compute_bin_densities(bin_scale_rate)

        return cls(left_bound, right_bound, bin_scale_rate, bin_densities)

    def fraction_of_ev(self, x: np.ndarray | float):
        """Return the approximate fraction of expected value that is less than
        the given value.
        """
        if isinstance(x, np.ndarray):
            return np.array([self.fraction_of_ev(xi) for xi in x])
        return np.sum(self.masses * self.values * (self.values <= x)) / self.mean()

    def inv_fraction_of_ev(self, fraction: np.ndarray | float):
        """Return the value such that `fraction` of the contribution to
        expected value lies to the left of that value.
        """
        if isinstance(fraction, np.ndarray):
            return np.array([self.inv_fraction_of_ev(xi) for xi in fraction])
        if fraction <= 0:
            raise ValueError("fraction must be greater than 0")
        epsilon = 1e-6  # to avoid floating point rounding issues
        index = np.searchsorted(self.fraction_of_ev(self.values), fraction - epsilon)
        return self.values[index]


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
    ):
        assert len(values) == len(masses)
        self.values = values
        self.masses = masses
        self.num_bins = len(values)
        self.exact_mean = exact_mean

    def __len__(self):
        return len(self.values)

    def __mul__(x, y):
        extended_values = np.outer(x.values, y.values).flatten()
        return x.binary_op(
            y,
            extended_values,
            exact_mean=x.exact_mean * y.exact_mean
            if x.exact_mean is not None and y.exact_mean is not None
            else None,
        )

    def binary_op(x, y, extended_values, exact_mean=None):
        extended_masses = np.outer(x.masses, y.masses).flatten()

        # Sort the arrays so product values are in order
        sorted_indexes = extended_values.argsort()
        extended_values = extended_values[sorted_indexes]
        extended_masses = extended_masses[sorted_indexes]

        # Squash the x values into a shorter array such that each x value has
        # equal contribution to expected value
        elem_evs = extended_masses * extended_values
        ev = sum(elem_evs)
        num_bins = max(len(x), len(y))
        ev_per_bin = ev / num_bins

        # Cut boundaries between bins such that each bin has equal contribution
        # to expected value
        cumulative_evs = np.cumsum(elem_evs)
        bin_boundaries = np.searchsorted(cumulative_evs, np.arange(ev_per_bin, ev, ev_per_bin))

        # Split elem_evs and extended_masses into bins
        split_element_evs = np.split(elem_evs, bin_boundaries)
        split_extended_masses = np.split(extended_masses, bin_boundaries)

        bin_values = []
        bin_masses = []
        for elem_evs, elem_masses in zip(split_element_evs, split_extended_masses):
            # TODO: could optimize this further by using cumulative_evs and
            # creating an equivalent for masses. might not even need to use np.split
            mass = np.sum(elem_masses)
            value = np.sum(elem_evs) / mass
            bin_values.append(value)
            bin_masses.append(mass)

        res = ProbabilityMassHistogram(
            np.array(bin_values), np.array(bin_masses), exact_mean=exact_mean
        )
        return res

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
        edge_values = []
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
        # mass 0. In that case, ignore the value.
        values = np.where(masses == 0, 0, values)

        return cls(np.array(values), np.array(masses))
