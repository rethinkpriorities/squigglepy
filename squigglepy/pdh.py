from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from typing import Optional

from .distributions import LognormalDistribution, lognorm
from .samplers import sample


class PDHBase(ABC):
    @abstractmethod
    def bin_density(self, index: int) -> float:
        ...

    @abstractmethod
    def bin_width(self, index: int) -> float:
        ...

    @abstractmethod
    def group_masses(self, num_bins: int, masses: np.ndarray):
        """Group the given masses into the given number of bins."""
        ...

    def old__mul__(self, other):
        """Multiply two PDHs together."""
        new_num_bins: int = max(self.num_bins, other.num_bins)
        masses_per_bin: int = (self.num_bins * other.num_bins) // new_num_bins
        masses = []
        for i in range(self.num_bins):
            for j in range(other.num_bins):
                xval = self.bin_edges[i] * other.bin_edges[j]

                mass = (
                    self.bin_density(i)
                    * self.bin_width(i)
                    * other.bin_density(j)
                    * other.bin_width(j)
                )
                masses.push(xval, mass)

        masses.sort(key=lambda pair: pair[0])
        return self.group_masses(new_num_bins, masses)

    def __mul__(self, other):
        """Multiply two PDHs together."""
        x = self
        y = other

        # Create lists representing all n^2 bins over a double integral
        prod_edges = np.outer(x.bin_edges, y.bin_edges).flatten()
        prod_densities = np.outer(x.edge_densities, y.edge_densities).flatten()

        # TODO: this isn't quite right, we want the product of the x and y
        # values in the middle of the bins, not the edges
        xy_product = np.outer(x.bin_edges, y.bin_edges).flatten()

        # Sort the arrays so bin edges are in order
        bin_data = np.column_stack((prod_edges, prod_densities, xy_product))
        bin_data = bin_data[bin_data[:, 0].argsort()]

        new_num_bins: int = max(self.num_bins, other.num_bins)
        return self.group_masses(new_num_bins, bin_data)


class PHDArbitraryBins(PDHBase):
    """A probability density histogram (PDH) is a numerical representation of
    a probability density function. A PDH is defined by a set of bin edges and
    a set of bin densities. The bin edges are the boundaries of the bins, and
    the bin densities are the probability densities. Bins do not necessarily
    have uniform width.
    """

    def __init__(self, bin_edges: np.ndarray, edge_densities: np.ndarray):
        assert len(bin_edges) == len(edge_densities)
        self.bin_edges = bin_edges
        self.edge_densities = edge_densities
        self.num_bins = len(bin_edges) - 1

    def scale(self, scale_factor: float):
        """Scale the PDF by the given scale factor."""
        self.bin_edges *= scale_factor

    def group_masses(self, num_bins: int, bin_data: np.ndarray) -> np.ndarray:
        """Group masses such that each bin has equal contribution to expected
        value."""
        masses = bin_data[:, 0] * bin_data[:, 1]
        # formula for expected value is density * bin width * bin center
        fraction_of_ev = TODO
        ev = sum(fraction_of_ev)
        target_ev_per_bin = ev / num_bins
        # TODO: how to pick the left and right bounds?


class ProbabilityDensityHistogram(PDHBase):
    """PDH with exponentially growing bin widths."""

    def __init__(
        self,
        left_bound: float,
        right_bound: float,
        bin_growth_rate: float,
        bin_densities: np.ndarray,
    ):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.bin_growth_rate = bin_growth_rate
        self.bin_densities = bin_densities
        self.num_bins = len(bin_densities)

    def bin_density(self, index: int) -> float:
        return self.bin_densities[index]

    def _weighted_error(
        self,
        left_bound: float,
        right_bound: float,
        bin_growth_rate: float,
        masses: np.ndarray,
    ) -> float:
        raise NotImplementedError

    def group_masses(self, num_bins: int, masses: np.ndarray) -> np.ndarray:
        # Use gradient descent to choose bounds and growth rate that minimize
        # weighted error
        pass


class ProbabilityMassHistogram:
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
        return x.binary_op(y, extended_values)

    def binary_op(x, y, extended_values):
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

        bin_values = []
        bin_masses = []

        cumulative_evs = np.cumsum(elem_evs)

        bin_boundaries = np.searchsorted(
            cumulative_evs, np.arange(ev_per_bin, ev, ev_per_bin)
        )

        # Split elem_evs and extended_masses into bins
        split_element_evs = np.split(elem_evs, bin_boundaries)
        split_extended_masses = np.split(extended_masses, bin_boundaries)

        for elem_evs, elem_masses in zip(split_element_evs, split_extended_masses):
            total_mass = np.sum(elem_masses)
            weighted_value = np.sum(elem_evs) / total_mass
            bin_values.append(weighted_value)
            bin_masses.append(total_mass)

        res = ProbabilityMassHistogram(
            np.array(bin_values), np.array(bin_masses)
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
        if isinstance(dist, LognormalDistribution):
            assert num_bins % 100 == 0, "num_bins must be a multiple of 100"
            edge_values = []
            boundary = 1 / num_bins

            edge_values = np.concatenate(
                (
                    [0],
                    dist.inv_fraction_of_ev(
                        np.linspace(boundary, 1 - boundary, num_bins - 1)
                    ),
                    [np.inf],
                )
            )

            # How much each bin contributes to total EV.
            contribution_to_ev = dist.lognorm_mean / num_bins

            # We can compute the exact mass of each bin as the difference in
            # CDF between the left and right edges.
            masses = np.diff(
                stats.lognorm.cdf(
                    edge_values, dist.norm_sd, scale=np.exp(dist.norm_mean)
                ),
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

    def histogram_mean(self):
        """Mean of the distribution, calculated using the histogram data."""
        return np.sum(self.values * self.masses)

    def mean(self):
        """Mean of the distribution. May be calculated using a stored exact
        value or the histogram data."""
        return self.histogram_mean()

    def std(self):
        """Standard deviation of the distribution."""
        mean = self.mean()
        return np.sqrt(np.sum(self.masses * (self.values - mean)**2))

    def fraction_of_ev(self, x: np.ndarray | float):
        """Return the approximate fraction of expected value that is less than
        the given value.
        """
        if isinstance(x, np.ndarray):
            return np.array([self.fraction_of_ev(xi) for xi in x])
        return (
            np.sum(self.masses * self.values * (self.values <= x))
            / self.mean()
        )

    def inv_fraction_of_ev(self, fraction: np.ndarray | float):
        """Return the value such that `fraction` of the contribution to
        expected value lies to the left of that value.
        """
        if isinstance(fraction, np.ndarray):
            return np.array([self.inv_fraction_of_ev(xi) for xi in fraction])
        if fraction <= 0:
            raise ValueError("fraction must be greater than 0")
        epsilon = 1e-6  # to avoid floating point rounding issues
        index = np.searchsorted(
            self.fraction_of_ev(self.values), fraction - epsilon
        )
        return self.values[index]
