"""
A numerical representation of a probability distribution as a histogram.
"""


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
    """An enum for the different methods of sizing histogram bins.

    Attributes
    ----------
    ev : str
        This method divides the distribution into bins such that each bin has
        equal contribution to expected value (see
        :func:`squigglepy.distributions.BaseDistribution.contribution_to_ev`).
        It works by first computing the bin edge values that equally divide up
        contribution to expected value, then computing the probability mass of
        each bin, then setting the value of each bin such that value * mass =
        contribution to expected value (rather than, say, setting value to the
        average value of the two edges).
    mass : str
        This method divides the distribution into bins such that each bin has
        equal probability mass.
    uniform : str
        This method divides the support of the distribution into bins of equal
        width.

    Pros and cons of bin sizing methods
    -----------------------------------
    The "ev" method is the most accurate for most purposes, and it has the
    important property that the histogram's expected value always exactly
    equals the true expected value of the distribution (modulo floating point
    rounding errors).

    The "ev" method differs from a standard trapezoid-method histogram in how
    it sizes bins and how it assigns values to bins. A trapezoid histogram
    divides the support of the distribution into bins of equal width, then
    assigns the value of each bin to the average of the two edges of the bin.
    The "ev" method of setting values naturally makes the histogram's expected
    value more accurate (the values are set specifically to make E[X] correct),
    but it also makes higher moments more accurate.

    Compared to a trapezoid histogram, an "ev" histogram must make the absolute
    value of the value in each bin larger: larger values within a bin get more
    weight in the expected value, so choosing the center value (or the average
    of the two edges) systematically underestimates E[X].

    It is possible to define the variance of a random variable X as

    .. math::
       E[X^2] - E[X]^2

    Similarly to how the trapezoid method underestimates E[X], the "ev" method
    necessarily underestimates E[X^2] (and therefore underestimates the
    variance/standard deviation) because E[X^2] places even more weight on
    larger values. But an alternative method that accurately estimated variance
    would necessarily *over*estimate E[X]. And however much the "ev" method
    underestimates E[X^2], the trapezoid method must underestimate it to a
    greater extent.

    The tradeoff is that the trapezoid method more accurately measures the
    probability mass in the vicinity of a particular value, whereas the "ev"
    method overestimates it. However, this is usually not as important as
    accurately measuring the expected value and variance.

    Implementation for two-sided distributions
    ------------------------------------------
    The interpretation of "ev" bin-sizing is slightly non-obvious for two-sided
    distributions because we must decide how to interpret bins with negative EV.

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
    and the contribution to EV per bin, but they are more complicated[1], and
    I considered this error rate acceptable. For example, if num_bins=100,
    the error after 16 multiplications is at most 8.3%. For
    one-sided distributions, the error is zero.

    [1] For example, we could exactly preserve EV contribution per bin in
    exchange for some inaccuracy in the total EV, and maintain a scalar error
    term that we multiply by whenever computing the EV. Or we could allow bins
    to cross zero, but this would require handling it as a special case.

    """

    ev = "ev"
    mass = "mass"
    uniform = "uniform"


class PDHBase(ABC):
    def __len__(self):
        return self.num_bins

    def positive_everywhere(self):
        """Return True if the distribution is positive everywhere."""
        return self.zero_bin_index == 0

    def negative_everywhere(self):
        """Return True if the distribution is negative everywhere."""
        return self.zero_bin_index == self.num_bins

    def is_one_sided(self):
        """Return True if the histogram contains only positive or negative values."""
        return self.positive_everywhere() or self.negative_everywhere()

    def is_two_sided(self):
        """Return True if the histogram contains both positive and negative values."""
        return not self.is_one_sided()

    def num_neg_bins(self):
        """Return the number of bins containing negative values."""
        return self.zero_bin_index

    def _check_bin_sizing(x, y):
        if x.bin_sizing != y.bin_sizing:
            raise ValueError(
                f"Can only multiply histograms that use the same bin sizing method (cannot multiply {x.bin_sizing} and {y.bin_sizing})"
            )

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
        return self.exact_sd

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
            return np.array([cls._contribution_to_ev(values, masses, xi, normalized) for xi in x])

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
        """Return the value such that ``fraction`` of the contribution to
        expected value lies to the left of that value.
        """
        return self._inv_contribution_to_ev(self.values, self.masses, fraction)

    def __add__(x, y):
        x._check_bin_sizing(y)
        cls = x
        num_bins = max(len(x), len(y))

        # Add every pair of values and find their joint masses.
        extended_values = np.add.outer(x.values, y.values).reshape(-1)
        extended_masses = np.outer(x.masses, y.masses).reshape(-1)

        is_sorted = False
        if (x.negative_everywhere() and y.negative_everywhere()) or (
            x.positive_everywhere() and y.positive_everywhere()
        ):
            # If both distributions are negative/positive everywhere, we don't
            # have to sort the extended values. This provides a ~10%
            # performance improvement.
            zero_index = 0 if x.positive_everywhere() else len(extended_values)
        else:
            # Sort so we can split the values into positive and negative sides.
            # Use timsort (called 'mergesort' by the numpy API) because
            # ``extended_values`` contains many sorted runs.
            sorted_indexes = extended_values.argsort(kind='mergesort')
            extended_values = extended_values[sorted_indexes]
            extended_masses = extended_masses[sorted_indexes]
            zero_index = np.searchsorted(extended_values, 0)
            is_sorted = True

        # Find how much of the EV contribution is on the negative side vs. the
        # positive side.
        neg_ev_contribution = (
            -np.sum(extended_values[:zero_index] * extended_masses[:zero_index])
        )
        pos_ev_contribution = (x.mean() + y.mean()) + neg_ev_contribution

        # Set the number of bins per side to be approximately proportional to
        # the EV contribution, but make sure that if a side has nonzero EV
        # contribution, it gets at least one bin.
        num_neg_bins = int(
            np.round(
                num_bins * neg_ev_contribution / (neg_ev_contribution + pos_ev_contribution)
            )
        )
        num_pos_bins = num_bins - num_neg_bins
        if zero_index > 0:
            num_neg_bins = max(1, num_neg_bins)
            num_pos_bins = num_bins - num_neg_bins
        if zero_index < len(extended_values):
            num_pos_bins = max(1, num_pos_bins)
            num_neg_bins = num_bins - num_pos_bins

        # Collect extended_values and extended_masses into the correct number
        # of bins. Make ``extended_values`` positive because ``resize_bins``
        # can only operate on non-negative values. Making them positive means
        # they're now reverse-sorted, so reverse them.
        neg_values, neg_masses = cls.resize_bins(
            extended_values=np.flip(-extended_values[:zero_index]),
            extended_masses=np.flip(extended_masses[:zero_index]),
            num_bins=num_neg_bins,
            ev=neg_ev_contribution,
            bin_sizing=x.bin_sizing,
            is_sorted=is_sorted,
        )

        # ``resize_bins`` returns positive values, so negate and flip them.
        neg_values = np.flip(-neg_values)
        neg_masses = np.flip(neg_masses)

        pos_values, pos_masses = cls.resize_bins(
            extended_values=extended_values[zero_index:],
            extended_masses=extended_masses[zero_index:],
            num_bins=num_pos_bins,
            ev=pos_ev_contribution,
            bin_sizing=x.bin_sizing,
            is_sorted=is_sorted,
        )

        values = np.concatenate((neg_values, pos_values))
        masses = np.concatenate((neg_masses, pos_masses))
        res = ProbabilityMassHistogram(
            values=values,
            masses=masses,
            zero_bin_index=zero_index,
            bin_sizing=x.bin_sizing,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
        )

        if x.exact_mean is not None and y.exact_mean is not None:
            res.exact_mean = x.exact_mean + y.exact_mean
        if x.exact_sd is not None and y.exact_sd is not None:
            res.exact_sd = np.sqrt(x.exact_sd**2 + y.exact_sd**2)
        return res

    def __mul__(x, y):
        cls = x
        x._check_bin_sizing(y)
        bin_sizing = x.bin_sizing
        num_bins = max(len(x), len(y))

        # If x+ is the positive part of x and x- is the negative part, then
        # result+ = (x+ * y+) + (x- * y-) and result- = (x+ * y-) + (x- *
        # y+). Multiply two-sided distributions by performing these steps:
        #
        # 1. Perform the four multiplications of one-sided distributions,
        #    producing n^2 bins.
        # 2. Add the two positive results and the two negative results into
        #    an array of positive values and an array of negative values.
        # 3. Run the binning algorithm on both arrays to compress them into
        #    a total of n bins.
        # 4. Join the two arrays into a new histogram.
        xneg_values = x.values[: x.zero_bin_index]
        xneg_masses = x.masses[: x.zero_bin_index]
        xpos_values = x.values[x.zero_bin_index :]
        xpos_masses = x.masses[x.zero_bin_index :]
        yneg_values = y.values[: y.zero_bin_index]
        yneg_masses = y.masses[: y.zero_bin_index]
        ypos_values = y.values[y.zero_bin_index :]
        ypos_masses = y.masses[y.zero_bin_index :]
        extended_neg_values = np.concatenate(
            (
                np.outer(xneg_values, ypos_values).flatten(),
                np.outer(xpos_values, yneg_values).flatten(),
            )
        )
        extended_neg_masses = np.concatenate(
            (
                np.outer(xneg_masses, ypos_masses).flatten(),
                np.outer(xpos_masses, yneg_masses).flatten(),
            )
        )
        extended_pos_values = np.concatenate(
            (
                np.outer(xneg_values, yneg_values).flatten(),
                np.outer(xpos_values, ypos_values).flatten(),
            )
        )
        extended_pos_masses = np.concatenate(
            (
                np.outer(xneg_masses, yneg_masses).flatten(),
                np.outer(xpos_masses, ypos_masses).flatten(),
            )
        )
        neg_ev_contribution = (
            x.neg_ev_contribution * y.pos_ev_contribution
            + x.pos_ev_contribution * y.neg_ev_contribution
        )
        pos_ev_contribution = (
            x.neg_ev_contribution * y.neg_ev_contribution
            + x.pos_ev_contribution * y.pos_ev_contribution
        )
        num_neg_bins = int(
            np.round(
                num_bins * neg_ev_contribution / (neg_ev_contribution + pos_ev_contribution)
            )
        )
        num_pos_bins = num_bins - num_neg_bins
        if neg_ev_contribution > 0:
            num_neg_bins = max(1, num_neg_bins)
            num_pos_bins = num_bins - num_neg_bins
        if pos_ev_contribution > 0:
            num_pos_bins = max(1, num_pos_bins)
            num_neg_bins = num_bins - num_pos_bins

        # resize_bins expects positive values, so negate them
        neg_values, neg_masses = cls.resize_bins(
            -extended_neg_values,
            extended_neg_masses,
            num_neg_bins,
            ev=neg_ev_contribution,
            bin_sizing=bin_sizing,
        )
        # the result will be positive and sorted ascending, so negate and
        # flip it
        neg_values = np.flip(-neg_values)
        neg_masses = np.flip(neg_masses)

        pos_values, pos_masses = cls.resize_bins(
            extended_pos_values,
            extended_pos_masses,
            num_pos_bins,
            ev=pos_ev_contribution,
            bin_sizing=bin_sizing,
        )
        values = np.concatenate((neg_values, pos_values))
        masses = np.concatenate((neg_masses, pos_masses))
        zero_bin_index = len(neg_values)
        res = ProbabilityMassHistogram(
            values,
            masses,
            zero_bin_index,
            bin_sizing,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
        )

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
        zero_bin_index: int,
        bin_sizing: Literal["ev", "quantile", "uniform"],
        neg_ev_contribution: float,
        pos_ev_contribution: float,
        exact_mean: Optional[float] = None,
        exact_sd: Optional[float] = None,
    ):
        """Create a probability mass histogram. You should usually not call
        this constructor directly; instead use :func:`from_distribution`.

        Parameters
        ----------
        values : np.ndarray
            The values of the distribution.
        masses : np.ndarray
            The probability masses of the values.
        zero_bin_index : int
            The index of the smallest bin that contains positive values (0 if all bins are positive).
        bin_sizing : Literal["ev", "quantile", "uniform"]
            The method used to size the bins.
        neg_ev_contribution : float
            The (absolute value of) contribution to expected value from the negative portion of the distribution.
        pos_ev_contribution : float
            The contribution to expected value from the positive portion of the distribution.
        exact_mean : Optional[float]
            The exact mean of the distribution, if known.
        exact_sd : Optional[float]
            The exact standard deviation of the distribution, if known.

        """
        assert len(values) == len(masses)
        self.values = values
        self.masses = masses
        self.num_bins = len(values)
        self.zero_bin_index = zero_bin_index
        self.bin_sizing = BinSizing(bin_sizing)
        self.neg_ev_contribution = neg_ev_contribution
        self.pos_ev_contribution = pos_ev_contribution
        self.exact_mean = exact_mean
        self.exact_sd = exact_sd

    @classmethod
    def resize_bins(
        cls,
        extended_values,
        extended_masses,
        num_bins,
        ev,
        bin_sizing,
        is_sorted=False,
    ):
        """Given two arrays of values and masses representing the result of a
        binary operation on two positive-everywhere distributions, compress the
        arrays down to ``num_bins`` bins and return the new values and masses of
        the bins.

        Parameters
        ----------
        extended_values : np.ndarray
            The values of the distribution. The values must all be non-negative.
        extended_masses : np.ndarray
            The probability masses of the values.
        num_bins : int
            The number of bins to compress the distribution into.
        ev : float
            The expected value of the distribution.
        bin_sizing : Literal["ev", "mass", "uniform"]
            The method used to size the bins.
        is_sorted : bool
            If True, assume that ``extended_values`` and ``extended_masses`` are
            already sorted in ascending order. This provides a significant
            performance improvement (~3x).

        Returns
        -------
        values : np.ndarray
            The values of the bins.
        masses : np.ndarray
            The probability masses of the bins.

        """
        if num_bins == 0:
            return (np.array([]), np.array([]))
        ev_per_bin = ev / num_bins
        items_per_bin = len(extended_values) // num_bins

        if bin_sizing == BinSizing.ev:
            if len(extended_masses) % num_bins > 0:
                # Increase the number of bins such that we can fit
                # extended_masses into them at items_per_bin each
                num_bins = int(np.ceil(len(extended_masses) / items_per_bin))

                # Fill any empty space with zeros
                extra_zeros = np.zeros(num_bins * items_per_bin - len(extended_masses))

                extended_values = np.concatenate((extended_values, extra_zeros))
                extended_masses = np.concatenate((extended_masses, extra_zeros))
                ev_per_bin = ev / num_bins

            if not is_sorted:
                # Partition such that the values in one bin are all less than
                # or equal to the values in the next bin. Values within bins
                # don't need to be sorted, and partitioning is ~10% faster than
                # timsort.
                boundary_bins = np.arange(0, num_bins + 1) * items_per_bin
                partitioned_indexes = extended_values.argpartition(boundary_bins[1:-1])
                extended_values = extended_values[partitioned_indexes]
                extended_masses = extended_masses[partitioned_indexes]

            # Take advantage of the fact that all bins contain the same number
            # of elements.
            extended_evs = extended_values * extended_masses
            masses = extended_masses.reshape((num_bins, -1)).sum(axis=1)

            # only works if all bins have equal contribution to EV
            # values = ev_per_bin / masses
            values = extended_evs.reshape((num_bins, -1)).sum(axis=1) / masses
            return (values, masses)

        raise ValueError(f"Unsupported bin sizing: {bin_sizing}")

    @classmethod
    def construct_bins(
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

        # Don't call get_edge_value on the left and right edges because it's
        # undefined for 0 and 1
        left_prop = dist.contribution_to_ev(support[0])
        right_prop = dist.contribution_to_ev(support[1])
        edge_values = np.concatenate(
            (
                [support[0]],
                np.atleast_1d(
                    get_edge_value(np.linspace(left_prop, right_prop, num_bins + 1)[1:-1])
                )
                if num_bins > 1
                else [],
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

        Parameters
        ----------
        dist : BaseDistribution
        num_bins : int
        bin_sizing : str (default "ev")
            See :ref:`squigglepy.pdh.BinSizing` for a list of valid options and a description of their behavior.

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
        neg_masses, neg_values = cls.construct_bins(
            num_neg_bins, -neg_contribution, (support[0], 0), dist, cdf, ppf, BinSizing(bin_sizing)
        )
        pos_masses, pos_values = cls.construct_bins(
            num_pos_bins, pos_contribution, (0, support[1]), dist, cdf, ppf, BinSizing(bin_sizing)
        )
        masses = np.concatenate((neg_masses, pos_masses))
        values = np.concatenate((neg_values, pos_values))

        return cls(
            np.array(values),
            np.array(masses),
            zero_bin_index=num_neg_bins,
            bin_sizing=bin_sizing,
            neg_ev_contribution=neg_contribution,
            pos_ev_contribution=pos_contribution,
            exact_mean=exact_mean,
            exact_sd=exact_sd,
        )
