from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from scipy import optimize, stats
import sortednp as snp
from typing import Literal, Optional
import warnings

from .distributions import NormalDistribution, LognormalDistribution
from .samplers import sample


class BinSizing(Enum):
    ev = "ev"
    mass = "mass"
    uniform = "uniform"


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
    def _contribution_to_ev(
        cls, values: np.ndarray, masses: np.ndarray, x: np.ndarray | float, normalized=True
    ):
        """Return the approximate fraction of expected value that is less than
        the given value.
        """
        if isinstance(x, np.ndarray) and x.ndim == 0:
            x = x.item()
        elif isinstance(x, np.ndarray):
            return np.array(
                [cls._contribution_to_ev(values, masses, xi, normalized) for xi in x]
            )

        contributions = np.squeeze(np.sum(masses * values * (values <= x)))
        if normalized:
            mean = np.sum(masses * values)
            return contributions / mean
        return contributions

    @classmethod
    def _inv_contribution_to_ev(
        cls, values: np.ndarray, masses: np.ndarray, fraction: np.ndarray | float
    ):
        if isinstance(fraction, np.ndarray):
            return np.array(
                [cls._inv_contribution_to_ev(values, masses, xi) for xi in list(fraction)]
            )
        if fraction <= 0:
            raise ValueError("fraction must be greater than 0")
        mean = np.sum(masses * values)
        fractions_of_ev = np.cumsum(masses * values) / mean
        epsilon = 1e-10  # to avoid floating point rounding issues
        index = np.searchsorted(fractions_of_ev, fraction - epsilon)
        return values[index]

    def contribution_to_ev(self, x: np.ndarray | float):
        """Return the approximate fraction of expected value that is less than
        the given value.
        """
        return self._contribution_to_ev(self.values, self.masses, x)

    def inv_contribution_to_ev(self, fraction: np.ndarray | float):
        """Return the value such that `fraction` of the contribution to
        expected value lies to the left of that value.
        """
        return self._inv_contribution_to_ev(self.values, self.masses, fraction)

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


class ProbabilityMassHistogram(PDHBase):
    """Represent a probability distribution as an array of x values and their
    probability masses. Like Monte Carlo samples except that values are
    weighted by probability, so you can effectively represent many times more
    samples than you actually have values."""

    def __init__(
        self,
        values: np.ndarray,
        masses: np.ndarray,
        bin_sizing: Literal["ev", "quantile", "uniform"],
        exact_mean: Optional[float] = None,
        exact_sd: Optional[float] = None,
    ):
        assert len(values) == len(masses)
        self.values = values
        self.masses = masses
        self.num_bins = len(values)
        self.bin_sizing = BinSizing(bin_sizing)
        self.exact_mean = exact_mean
        self.exact_sd = exact_sd

    def binary_op(x, y, extended_values, ev, is_mul=False):
        assert (
            x.bin_sizing == y.bin_sizing
        ), f"Can only combine histograms that use the same bin sizing method (cannot combine {x.bin_sizing} and {y.bin_sizing})"
        bin_sizing = x.bin_sizing
        extended_masses = np.ravel(np.outer(x.masses, y.masses))
        num_bins = max(len(x), len(y))
        len_per_bin = int(len(extended_values) / num_bins)
        ev_per_bin = ev / num_bins
        extended_evs = extended_masses * extended_values

        # Cut boundaries between bins such that each bin has equal contribution
        # to expected value.
        if is_mul or bin_sizing == BinSizing.uniform:
            # When multiplying, the values of extended_evs are all equal. x and
            # y both have the property that every bin contributes equally to
            # EV, which means the outputs of their outer product must all be
            # equal. We can use this fact to avoid a relatively slow call to
            # `cumsum` (which can also introduce floating point rounding errors
            # for extreme values).
            bin_boundaries = np.arange(1, num_bins) * len_per_bin
        else:
            if bin_sizing == BinSizing.ev:
                cumulative_evs = np.cumsum(extended_evs)
                bin_boundaries = np.searchsorted(
                    cumulative_evs, np.arange(ev_per_bin, ev, ev_per_bin)
                )
            elif bin_sizing == BinSizing.mass:
                cumulative_masses = np.cumsum(extended_masses)
                bin_boundaries = np.searchsorted(
                    cumulative_masses, np.arange(1, num_bins) / num_bins
                )

        # Partition the arrays so every value in a bin is smaller than every
        # value in the next bin, but don't sort within bins. (Partition is
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
            if bin_sizing == BinSizing.ev:
                bin_values = ev_per_bin / bin_masses
            elif bin_sizing == BinSizing.mass:
                bin_values = extended_values.reshape((num_bins, -1)).mean(axis=1)
        else:
            bin_boundaries = np.concatenate(([0], bin_boundaries, [len(extended_evs)]))
            for i in range(len(bin_boundaries) - 1):
                start = bin_boundaries[i]
                end = bin_boundaries[i + 1]
                mass = np.sum(extended_masses[start:end])

                if bin_sizing == BinSizing.ev:
                    value = np.sum(extended_evs[start:end]) / mass
                elif bin_sizing == BinSizing.mass:
                    value = np.sum(extended_values[start:end] * extended_masses[start:end]) / mass
                bin_values.append(value)
                bin_masses.append(mass)

        return ProbabilityMassHistogram(np.array(bin_values), np.array(bin_masses), bin_sizing)

    @classmethod
    def _construct_bins(
        cls, num_bins, total_contribution_to_ev, support, dist, cdf, ppf, bin_sizing
    ):
        """Construct a list of bin masses and values. Helper function for
        :func:`from_distribution`, do not call this directly."""
        if num_bins == 0:
            return (np.array([]), np.array([]))

        if bin_sizing == BinSizing.ev:
            get_edge_value = dist.inv_contribution_to_ev
        elif bin_sizing == BinSizing.mass:
            get_edge_value = ppf
        else:
            raise ValueError(f"Unsupported bin sizing: {bin_sizing}")

        left_prop = dist.contribution_to_ev(support[0])
        right_prop = dist.contribution_to_ev(support[1])
        width = (right_prop - left_prop) / num_bins

        # Don't call get_edge_value on the left and right edges because it's
        # undefined for 0 and 1
        edge_values = np.concatenate(
            (
                [support[0]],
                np.atleast_1d(get_edge_value(
                    np.linspace(left_prop + width, right_prop - width, num_bins - 1)
                )) if num_bins > 1 else [],
                [support[1]],
            )
        )
        edge_cdfs = cdf(edge_values)
        masses = np.diff(edge_cdfs)

        # Assume the value exactly equals the bin's contribution to EV
        # divided by its mass. This means the values will not be exactly
        # centered, but it guarantees that the expected value of the
        # histogram exactly equals the expected value of the distribution
        # (modulo floating point rounding).
        if bin_sizing == BinSizing.ev:
            ev_contribution_per_bin = total_contribution_to_ev / num_bins
            values = ev_contribution_per_bin / masses
        elif bin_sizing == BinSizing.mass:
            # TODO: this might not work for negative values
            midpoints = (edge_cdfs[:-1] + edge_cdfs[1:]) / 2
            raw_values = ppf(midpoints)
            estimated_mean = np.sum(raw_values * masses)
            values = raw_values * total_contribution_to_ev / estimated_mean
        else:
            raise ValueError(f"Unsupported bin sizing: {bin_sizing}")

        # For sufficiently large values, CDF rounds to 1 which makes the
        # mass 0.
        if any(masses == 0):
            values = np.where(masses == 0, 0, values)
            num_zeros = np.sum(masses == 0)
            warnings.warn(
                f"{num_zeros} values greater than {values[-1]} had CDFs of 1.", RuntimeWarning
            )

        return (masses, values)

    @classmethod
    def from_distribution(cls, dist, num_bins=100, bin_sizing="ev"):
        """Create a probability mass histogram from the given distribution.

        Create a probability mass histogram from the given distribution. The
        histogram covers the full distribution except for the 1/num_bins/2
        expectile on the left and right tails. The boundaries are based on the
        expectile rather than the quantile to better capture the tails of
        fat-tailed distributions, but this can cause computational problems for
        very fat-tailed distributions.

        bin_sizing="ev" arranges values into bins such that:
            * The negative side has the correct negative contribution to EV and the
              positive side has the correct positive contribution to EV.
            * Every negative bin has equal contribution to EV and every positive bin
              has equal contribution to EV.
            * The number of negative and positive bins are chosen such that the
              absolute contribution to EV for negative bins is as close as possible
              to the absolute contribution to EV for positive bins.

        This binning method means that the distribution EV is exactly preserved
        and there is no bin that contains the value zero. However, the positive
        and negative bins do not necessarily have equal contribution to EV, and
        the magnitude of the error can be at most 1 / num_bins / 2. There are
        alternative binning implementations that exactly preserve both the EV
        and the contribution to EV per bin, but they are more complicated, and
        I considered this error rate acceptable. For example, if num_bins=100,
        the error after 16 multiplications is at most 8.3%. For
        one-sided distributions, the error is zero.
        """
        if isinstance(dist, LognormalDistribution):
            ppf = lambda p: stats.lognorm.ppf(p, dist.norm_sd, scale=np.exp(dist.norm_mean))
            cdf = lambda x: stats.lognorm.cdf(x, dist.norm_sd, scale=np.exp(dist.norm_mean))
            exact_mean = dist.lognorm_mean
            exact_sd = dist.lognorm_sd
            support = (0, np.inf)
        elif isinstance(dist, NormalDistribution):
            ppf = lambda p: stats.norm.ppf(p, loc=dist.mean, scale=dist.sd)
            cdf = lambda x: stats.norm.cdf(x, loc=dist.mean, scale=dist.sd)
            exact_mean = dist.mean
            exact_sd = dist.sd
            support = (-np.inf, np.inf)
        else:
            raise ValueError(f"Unsupported distribution type: {type(dist)}")

        assert num_bins % 100 == 0, "num_bins must be a multiple of 100"

        total_contribution_to_ev = dist.contribution_to_ev(np.inf, normalized=False)
        neg_contribution = dist.contribution_to_ev(0, normalized=False)
        pos_contribution = total_contribution_to_ev - neg_contribution

        # Divide up bins such that each bin has as close as possible to equal
        # contribution to EV.
        num_neg_bins = int(np.round(num_bins * neg_contribution / total_contribution_to_ev))
        num_pos_bins = num_bins - num_neg_bins

        # If one side is very small but nonzero, we must ensure that it gets at
        # least one bin.
        if neg_contribution > 0:
            num_neg_bins = max(1, num_neg_bins)
            num_pos_bins = num_bins - num_neg_bins
        if pos_contribution > 0:
            num_pos_bins = max(1, num_pos_bins)
            num_neg_bins = num_bins - num_pos_bins

        # All negative bins have exactly equal contribution to EV, and all
        # positive bins have exactly equal contribution to EV.
        neg_masses, neg_values = cls._construct_bins(
            num_neg_bins, -neg_contribution, (support[0], 0), dist, cdf, ppf, BinSizing(bin_sizing)
        )
        pos_masses, pos_values = cls._construct_bins(
            num_pos_bins, pos_contribution, (0, support[1]), dist, cdf, ppf, BinSizing(bin_sizing)
        )
        masses = np.concatenate((neg_masses, pos_masses))
        values = np.concatenate((neg_values, pos_values))

        return cls(
            np.array(values),
            np.array(masses),
            bin_sizing=bin_sizing,
            exact_mean=exact_mean,
            exact_sd=exact_sd,
        )
