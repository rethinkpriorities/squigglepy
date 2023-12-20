from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from numbers import Real
import numpy as np
from scipy import optimize, stats
from scipy.interpolate import PchipInterpolator
from typing import Callable, Literal, Optional, Tuple, Union
import warnings

from .distributions import (
    BaseDistribution,
    BernoulliDistribution,
    BetaDistribution,
    ChiSquareDistribution,
    ComplexDistribution,
    ConstantDistribution,
    ExponentialDistribution,
    GammaDistribution,
    LognormalDistribution,
    MixtureDistribution,
    NormalDistribution,
    ParetoDistribution,
    PERTDistribution,
    UniformDistribution,
)
from .version import __version__


class BinSizing(str, Enum):
    """An enum for the different methods of sizing histogram bins. A histogram
    with finitely many bins can only contain so much information about the
    shape of a distribution; the choice of bin sizing changes what information
    NumericDistribution prioritizes.
    """

    uniform = "uniform"
    """Divides the distribution into bins of equal width. For distributions
    with infinite support (such as normal distributions), it chooses a
    total width to roughly minimize total error, considering both intra-bin
    error and error due to the excluded tails."""

    log_uniform = "log-uniform"
    """Divides the distribution into bins with exponentially increasing width,
     so that the logarithms of the bin edges are uniformly spaced. For example,
     if you generated a NumericDistribution from a log-normal distribution with
     log-uniform bin sizing, and then took the log of each bin, you'd get a
     normal distribution with uniform bin sizing.
    """

    ev = "ev"
    """Divides the distribution into bins such that each bin has equal
    contribution to expected value (see
    :any:`IntegrableEVDistribution.contribution_to_ev`)."""

    mass = "mass"
    """Divides the distribution into bins such that each bin has equal
    probability mass. This method is generally not recommended
    because it puts too much probability mass near the center of the
    distribution, where precision is the least useful."""

    fat_hybrid = "fat-hybrid"
    """A hybrid method designed for fat-tailed distributions. Uses mass bin
    sizing close to the center and log-uniform bin siding on the right
    tail. Empirically, this combination provides the best balance for the
    accuracy of fat-tailed distributions at the center and at the tails."""

    bin_count = "bin-count"
    """Shortens a vector of bins by merging every ``len(vec)/num_bins`` bins
    together. Can only be used for resizing existing NumericDistributions, not
    for initializing new ones.
    """


def _bump_indexes(indexes, length):
    """Given an ordered list of indexes, ensure that every index is unique. If
    any index is not unique, increment or decrement it until it's unique.

    This function does not guarantee that every index gets moved to the closest
    unique index, but it does guarantee that every index gets moved to the
    closest unique index in a certain direction.
    """
    for i in range(1, len(indexes)):
        if indexes[i] <= indexes[i - 1]:
            indexes[i] = min(length - 1, indexes[i - 1] + 1)

    for i in reversed(range(len(indexes) - 1)):
        if indexes[i] >= indexes[i + 1]:
            indexes[i] = max(0, indexes[i + 1] - 1)

    return indexes


def _support_for_bin_sizing(dist, bin_sizing, num_bins):
    """Return where to set the bounds for a bin sizing method with fixed
    bounds, or None if the given dist/bin sizing does not require finite
    bounds.
    """
    # For norm/lognorm, wider domain increases error within each bin, and
    # narrower domain increases error at the tails. Inter-bin error is
    # proportional to width^3 / num_bins^2 and tail error is proportional to
    # something like exp(-width^2). Setting width using the formula below
    # balances these two sources of error. ``scale`` has an upper bound because
    # excessively large values result in floating point rounding errors.
    if isinstance(dist, NormalDistribution) and bin_sizing == BinSizing.uniform:
        scale = max(6.5, 4.5 + np.log(num_bins) ** 0.5)
        return (dist.mean - scale * dist.sd, dist.mean + scale * dist.sd)
    if isinstance(dist, LognormalDistribution) and bin_sizing == BinSizing.log_uniform:
        scale = max(6.5, 4.5 + np.log(num_bins) ** 0.5)
        return np.exp(
            (dist.norm_mean - scale * dist.norm_sd, dist.norm_mean + scale * dist.norm_sd)
        )

    # Uniform bin sizing is not gonna be very accurate for a lognormal
    # distribution no matter how you set the bounds.
    if isinstance(dist, LognormalDistribution) and bin_sizing == BinSizing.uniform:
        scale = 6
        return np.exp(
            (dist.norm_mean - scale * dist.norm_sd, dist.norm_mean + scale * dist.norm_sd)
        )

    # Compute the upper bound numerically because there is no good closed-form
    # expression (that I could find) that reliably captures almost all of the
    # mass without making the bins overly wide.
    if isinstance(dist, GammaDistribution) and bin_sizing == BinSizing.uniform:
        upper_bound = stats.gamma.ppf(1 - 1e-9, dist.shape, scale=dist.scale)
        return (0, upper_bound)
    if isinstance(dist, GammaDistribution) and bin_sizing == BinSizing.log_uniform:
        lower_bound = stats.gamma.ppf(1e-10, dist.shape, scale=dist.scale)
        upper_bound = stats.gamma.ppf(1 - 1e-10, dist.shape, scale=dist.scale)
        return (lower_bound, upper_bound)

    return None


DEFAULT_BIN_SIZING = {
    BetaDistribution: BinSizing.mass,
    ChiSquareDistribution: BinSizing.ev,
    ExponentialDistribution: BinSizing.ev,
    GammaDistribution: BinSizing.ev,
    LognormalDistribution: BinSizing.ev,
    NormalDistribution: BinSizing.ev,
    ParetoDistribution: BinSizing.ev,
    PERTDistribution: BinSizing.mass,
    UniformDistribution: BinSizing.uniform,
}
"""
Default bin sizing method for each distribution type. Chosen based on
empirical tests of which method best balances the accuracy across summary
statistics and across operations (addition, multiplication, left/right clip,
etc.).
"""

DEFAULT_NUM_BINS = {
    BetaDistribution: 50,
    ChiSquareDistribution: 200,
    ExponentialDistribution: 200,
    GammaDistribution: 200,
    LognormalDistribution: 200,
    MixtureDistribution: 200,
    NormalDistribution: 200,
    ParetoDistribution: 200,
    PERTDistribution: 100,
    UniformDistribution: 50,
}
"""
Default number of bins for each distribution type. The default is 200 for
most distributions, which provides a good balance of accuracy and speed. Some
distributions use a smaller number of bins because they are sufficiently narrow
and well-behaved that not many bins are needed for high accuracy.
"""

CACHED_LOGNORM_CDFS = {}
CACHED_LOGNORM_PPFS = {}


def cached_lognorm_cdfs(num_bins):
    if num_bins in CACHED_LOGNORM_CDFS:
        return CACHED_LOGNORM_CDFS[num_bins]
    support = _support_for_bin_sizing(
        LognormalDistribution(norm_mean=0, norm_sd=1), BinSizing.log_uniform, num_bins
    )
    values = np.exp(np.linspace(np.log(support[0]), np.log(support[1]), num_bins + 1))
    cdfs = stats.lognorm.cdf(values, 1)
    CACHED_LOGNORM_CDFS[num_bins] = cdfs
    return cdfs


def cached_lognorm_ppf_zscore(num_bins):
    if num_bins in CACHED_LOGNORM_PPFS:
        return CACHED_LOGNORM_PPFS[num_bins]
    cdfs = np.linspace(0, 1, num_bins + 1)
    ppfs = _log(stats.lognorm.ppf(cdfs, 1))
    CACHED_LOGNORM_PPFS[num_bins] = (cdfs, ppfs)
    return (cdfs, ppfs)


def _narrow_support(
    support: Tuple[float, float], new_support: Tuple[Optional[float], Optional[float]]
):
    """Narrow the support to the intersection of ``support`` and ``new_support``."""
    if new_support[0] is not None:
        support = (max(support[0], new_support[0]), support[1])
    if new_support[1] is not None:
        support = (support[0], min(support[1], new_support[1]))
    return support


def _log(x):
    with np.errstate(divide="ignore"):
        return np.log(x)


class BaseNumericDistribution(ABC):
    """BaseNumericDistribution

    An abstract base class for numeric distributions. For more documentation,
    see :class:`NumericDistribution` and :class:`ZeroNumericDistribution`.

    """

    def __repr__(self):
        return f"<{type(self).__name__}(mean={self.mean()}, sd={self.sd()}, num_bins={len(self)}, bin_sizing={self.bin_sizing}) at {hex(id(self))}>"

    def __str__(self):
        return f"{type(self).__name__}(mean={self.mean()}, sd={self.sd()})"

    def mean(self, axis=None, dtype=None, out=None):
        """Mean of the distribution. May be calculated using a stored exact
        value or the histogram data.

        Parameters
        ----------
        None of the parameters do anything, they're only there so that
        ``numpy.mean()`` can be called on a ``BaseNumericDistribution``.
        """
        if self.exact_mean is not None:
            return self.exact_mean
        return self.est_mean()

    def sd(self):
        """Standard deviation of the distribution. May be calculated using a
        stored exact value or the histogram data."""
        if self.exact_sd is not None:
            return self.exact_sd
        return self.est_sd()

    def std(self, axis=None, dtype=None, out=None):
        """Standard deviation of the distribution. May be calculated using a
        stored exact value or the histogram data.

        Parameters
        ----------
        None of the parameters do anything, they're only there so that
        ``numpy.std()`` can be called on a ``BaseNumericDistribution``.
        """
        return self.sd()

    def quantile(self, q):
        """Estimate the value of the distribution at quantile ``q`` by
        interpolating between known values.

        The accuracy at different quantiles depends on the bin sizing method
        used. :any:`BinSizing.mass` will produce bins that are evenly spaced
        across quantiles. :any:``BinSizing.ev`` for fat-tailed distributions
        will be very inaccurate at lower quantiles in exchange for greater
        accuracy on the right tail.

        Parameters
        ----------
        q : number or array_like
            The quantile or quantiles for which to determine the value(s).

        Returns
        -------
        quantiles: number or array-like
            The estimated value at the given quantile(s).

        """
        return self.ppf(q)

    @abstractmethod
    def ppf(self, q):
        """Percent point function/inverse CD. An alias for :any:`quantile`."""
        ...

    def percentile(self, p):
        """Estimate the value of the distribution at percentile ``p``. See
        :any:`quantile` for notes on this function's accuracy.
        """
        return np.squeeze(self.ppf(np.asarray(p) / 100))

    def condition_on_success(
        self,
        event: Union["BaseNumericDistribution", float],
        failure_outcome: Optional[Union["BaseNumericDistribution", float]] = 0,
    ):
        """``event`` is a probability distribution over a probability for some
        binary outcome. If the event succeeds, the result is the random
        variable defined by ``self``. If the event fails, the result is zero.
        Or, if ``failure_outcome`` is provided, the result is
        ``failure_outcome``.

        This function's return value represents the probability
        distribution over outcomes in this scenario.

        The return value is equivalent to the result of this procedure:

        1. Generate a probability ``p`` according to the distribution defined
           by ``event``.
        2. Generate a Bernoulli random variable with probability ``p``.
        3. If success, generate a random outcome according to the distribution
           defined by ``self``.
        4. Otherwise, generate a random outcome according to the distribution
           defined by ``failure_outcome``.

        """
        if failure_outcome != 0:
            # TODO: you can't just do a sum. I think what you want to do is
            # scale the masses and then smush the bins together
            raise NotImplementedError
        if isinstance(event, Real):
            p_success = event
        elif isinstance(event, BaseNumericDistribution):
            p_success = event.mean()
        else:
            raise TypeError(f"Cannot condition on type {type(event)}")
        return ZeroNumericDistribution.wrap(self, 1 - p_success)

    def __ne__(x, y):
        return not (x == y)

    def __radd__(x, y):
        return x + y

    def __sub__(x, y):
        return x + (-y)

    def __rsub__(x, y):
        return -x + y

    def __rmul__(x, y):
        return x * y

    def __truediv__(x, y):
        if isinstance(y, Real):
            return x.scale_by(1 / y)
        return x * y.reciprocal()

    def __rtruediv__(x, y):
        return y * x.reciprocal()


class NumericDistribution(BaseNumericDistribution):
    """NumericDistribution

    A numerical representation of a probability distribution as a histogram of
    values along with the probability mass near each value.

    A ``NumericDistribution`` is functionally equivalent to a Monte Carlo
    simulation where you generate infinitely many samples and then group the
    samples into finitely many bins, keeping track of the proportion of samples
    in each bin (a.k.a. the probability mass) and the average value for each
    bin.

    Compared to a Monte Carlo simulation, ``NumericDistribution`` can represent
    information much more densely by grouping together nearby values (although
    some information is lost in the grouping). The benefit of this is most
    obvious in fat-tailed distributions. In a Monte Carlo simulation, perhaps 1
    in 1000 samples account for 10% of the expected value, but a
    ``NumericDistribution`` (with the right bin sizing method, see
    :any:`BinSizing`) can easily track the probability mass of large values.

    For more, see :doc:`/numeric_distributions`.

    """

    def __init__(
        self,
        values: np.ndarray,
        masses: np.ndarray,
        zero_bin_index: int,
        neg_ev_contribution: float,
        pos_ev_contribution: float,
        exact_mean: Optional[float],
        exact_sd: Optional[float],
        bin_sizing: Optional[BinSizing] = None,
        min_bins_per_side: Optional[int] = 2,
        richardson_extrapolation_enabled: Optional[bool] = True,
    ):
        """Create a probability mass histogram. You should usually not call
        this constructor directly; instead, use :func:`from_distribution`.

        Parameters
        ----------
        values : np.ndarray
            The values of the distribution.
        masses : np.ndarray
            The probability masses of the values.
        zero_bin_index : int
            The index of the smallest bin that contains positive values (0 if all bins are positive).
        bin_sizing : :any:`BinSizing`
            The method used to size the bins.
        neg_ev_contribution : float
            The (absolute value of) contribution to expected value from the negative portion of the distribution.
        pos_ev_contribution : float
            The contribution to expected value from the positive portion of the distribution.
        exact_mean : Optional[float]
            The exact mean of the distribution, if known.
        exact_sd : Optional[float]
            The exact standard deviation of the distribution, if known.
        bin_sizing : Optional[BinSizing]
            The bin sizing method used to construct the distribution, if any.
        richardson_extrapolation_enabled : Optional[bool] = True
            If True, use Richardson extrapolation over the number of bins to
            improve the accuracy of unary and binary operations.
        """
        assert len(values) == len(masses)
        self._version = __version__
        self.values = values
        self.masses = masses
        self.num_bins = len(values)
        self.zero_bin_index = zero_bin_index
        self.neg_ev_contribution = neg_ev_contribution
        self.pos_ev_contribution = pos_ev_contribution
        self.exact_mean = exact_mean
        self.exact_sd = exact_sd
        self.bin_sizing = bin_sizing
        self.min_bins_per_side = min_bins_per_side
        self.richardson_extrapolation_enabled = richardson_extrapolation_enabled

        # These are computed lazily
        self.interpolate_cdf = None
        self.interpolate_ppf = None
        self.interpolate_cev = None
        self.interpolate_inv_cev = None

    @classmethod
    def _construct_edge_values(
        cls,
        num_bins,
        support,
        max_support,
        dist,
        cdf,
        ppf,
        bin_sizing,
    ):
        """Construct a list of bin edge values. Helper function for
        :func:`from_distribution`; do not call this directly.

        Parameters
        ----------
        num_bins : int
            The number of bins to use.
        support : Tuple[float, float]
            The support of the distribution.
        max_support : Tuple[float, float]
            The maximum support of the distribution, after clipping but before
            narrowing due to limitations of certain bin sizing methods. Namely,
            uniform and log-uniform bin sizing is undefined for infinite bounds,
            so ``support`` is narrowed to finite bounds, but ``max_support`` is
            not.
        dist : BaseDistribution
            The distribution to convert to a NumericDistribution.
        cdf : Callable[[np.ndarray], np.ndarray]
            The CDF of the distribution.
        ppf : Callable[[np.ndarray], np.ndarray]
            The inverse CDF of the distribution.
        bin_sizing : BinSizing
            The bin sizing method to use.

        Returns
        -------
        edge_values : np.ndarray
            The value of each bin edge.
        edge_cdfs : Optional[np.ndarray]
            The CDF at each bin edge. Only provided as a performance
            optimization if the CDFs are either required to determine bin edge
            values or if they can be pulled from a cache. Otherwise, the parent
            caller is responsible for calculating the CDFs.

        """
        edge_cdfs = None
        if bin_sizing == BinSizing.uniform:
            edge_values = np.linspace(support[0], support[1], num_bins + 1)

        elif bin_sizing == BinSizing.log_uniform:
            log_support = (_log(support[0]), _log(support[1]))
            log_edge_values = np.linspace(log_support[0], log_support[1], num_bins + 1)
            edge_values = np.exp(log_edge_values)
            if (
                isinstance(dist, LognormalDistribution)
                and dist.lclip is None
                and dist.rclip is None
            ):
                # Edge CDFs are the same regardless of the mean and SD of the
                # distribution, so we can cache them
                edge_cdfs = cached_lognorm_cdfs(num_bins)

        elif bin_sizing == BinSizing.ev:
            if not hasattr(dist, "inv_contribution_to_ev"):
                raise ValueError(
                    f"Bin sizing {bin_sizing} requires an inv_contribution_to_ev method, but {type(dist)} does not have one."
                )
            left_prop = dist.contribution_to_ev(support[0])
            right_prop = dist.contribution_to_ev(support[1])
            edge_values = np.concatenate(
                (
                    # Don't call inv_contribution_to_ev on the left and right
                    # edges because it's undefined for 0 and 1
                    [support[0]],
                    np.atleast_1d(
                        dist.inv_contribution_to_ev(
                            np.linspace(left_prop, right_prop, num_bins + 1)[1:-1]
                        )
                    )
                    if num_bins > 1
                    else [],
                    [support[1]],
                )
            )

        elif bin_sizing == BinSizing.mass:
            if (
                isinstance(dist, LognormalDistribution)
                and dist.lclip is None
                and dist.rclip is None
            ):
                edge_cdfs, edge_zscores = cached_lognorm_ppf_zscore(num_bins)
                edge_values = np.exp(dist.norm_mean + dist.norm_sd * edge_zscores)
            else:
                edge_cdfs = np.linspace(cdf(support[0]), cdf(support[1]), num_bins + 1)
                edge_values = ppf(edge_cdfs)

        elif bin_sizing == BinSizing.fat_hybrid:
            # Use a combination of mass and log-uniform
            logu_support = _support_for_bin_sizing(dist, BinSizing.log_uniform, num_bins)
            logu_support = _narrow_support(support, logu_support)
            logu_edge_values, logu_edge_cdfs = cls._construct_edge_values(
                num_bins, logu_support, max_support, dist, cdf, ppf, BinSizing.log_uniform
            )
            mass_edge_values, mass_edge_cdfs = cls._construct_edge_values(
                num_bins, support, max_support, dist, cdf, ppf, BinSizing.mass
            )

            if logu_edge_cdfs is not None and mass_edge_cdfs is not None:
                edge_cdfs = np.where(
                    logu_edge_values > mass_edge_values, logu_edge_cdfs, mass_edge_cdfs
                )
            edge_values = np.where(
                logu_edge_values > mass_edge_values, logu_edge_values, mass_edge_values
            )

        else:
            raise ValueError(f"Unsupported bin sizing method: {bin_sizing}")

        return (edge_values, edge_cdfs)

    @classmethod
    def _construct_bins(
        cls,
        num_bins,
        support,
        max_support,
        dist,
        cdf,
        ppf,
        bin_sizing,
        warn,
        is_reversed,
    ):
        """Construct a list of bin masses and values. Helper function for
        :func:`from_distribution`; do not call this directly.

        Parameters
        ----------
        num_bins : int
            The number of bins to use.
        support : Tuple[float, float]
            The support of the distribution.
        max_support : Tuple[float, float]
            The maximum support of the distribution, after clipping but before
            narrowing due to limitations of certain bin sizing methods. Namely,
            uniform and log-uniform bin sizing is undefined for infinite bounds,
            so ``support`` is narrowed to finite bounds, but ``max_support`` is
            not.
        dist : BaseDistribution
            The distribution to convert to a NumericDistribution.
        cdf : Callable[[np.ndarray], np.ndarray]
            The CDF of the distribution.
        ppf : Callable[[np.ndarray], np.ndarray]
            The inverse CDF of the distribution.
        bin_sizing : BinSizing
            The bin sizing method to use.
        warn : bool
            If True, raise warnings about bins with zero mass.

        Returns
        -------
        masses : np.ndarray
            The probability mass of each bin.
        values : np.ndarray
            The value of each bin.
        """
        if num_bins <= 0:
            return (np.array([]), np.array([]))

        edge_values, edge_cdfs = cls._construct_edge_values(
            num_bins, support, max_support, dist, cdf, ppf, bin_sizing
        )

        # Avoid re-calculating CDFs if we can because it's really slow.
        if edge_cdfs is None:
            edge_cdfs = cdf(edge_values)

        masses = np.diff(edge_cdfs)

        # Note: Re-calculating this for BinSize.ev appears to add ~zero
        # performance penalty. Perhaps Python is caching the result somehow?
        edge_ev_contributions = dist.contribution_to_ev(edge_values, normalized=False)
        bin_ev_contributions = np.diff(edge_ev_contributions)

        # For sufficiently large edge values, CDF rounds to 1 which makes the
        # mass 0. Values can also be 0 due to floating point rounding if
        # support is very small. Remove any 0s.
        mass_zeros = [i for i in range(len(masses)) if masses[i] == 0]
        ev_zeros = [i for i in range(len(bin_ev_contributions)) if bin_ev_contributions[i] == 0]
        non_monotonic = []

        if len(mass_zeros) == 0:
            # Set the value of each bin to equal the average value within the
            # bin.
            values = bin_ev_contributions / masses

            # Values can be non-monotonic if there are rounding errors when
            # calculating EV contribution. Look at the bottom and top separately
            # because on the bottom, the lower value will be the incorrect one, and
            # on the top, the upper value will be the incorrect one.
            sign = -1 if is_reversed else 1
            bot_diffs = sign * np.diff(values[: (num_bins // 10)])
            top_diffs = sign * np.diff(values[-(num_bins // 10) :])
            non_monotonic = [i for i in range(len(bot_diffs)) if bot_diffs[i] < 0] + [
                i + 1 + num_bins - len(top_diffs)
                for i in range(len(top_diffs))
                if top_diffs[i] < 0
            ]

        bad_indexes = set(mass_zeros + ev_zeros + non_monotonic)

        if len(bad_indexes) > 0:
            good_indexes = [i for i in range(num_bins) if i not in set(bad_indexes)]
            bin_ev_contributions = bin_ev_contributions[good_indexes]
            masses = masses[good_indexes]
            values = bin_ev_contributions / masses

            messages = []
            if len(mass_zeros) > 0:
                messages.append(f"{len(mass_zeros) + 1} neighboring values had equal CDFs")
            if len(ev_zeros) == 1:
                messages.append(
                    f"1 bin had zero expected value, most likely because it was too small"
                )
            elif len(ev_zeros) > 1:
                messages.append(
                    f"{len(ev_zeros)} bins had zero expected value, most likely because they were too small"
                )
            if len(non_monotonic) > 0:
                messages.append(f"{len(non_monotonic) + 1} neighboring values were non-monotonic")
            joint_message = "; and".join(messages)

            if warn:
                warnings.warn(
                    f"When constructing NumericDistribution, {joint_message}.",
                    RuntimeWarning,
                )

        return (masses, values)

    @classmethod
    def from_distribution(
        cls,
        dist: Union[BaseDistribution, BaseNumericDistribution, Real],
        num_bins: Optional[int] = None,
        bin_sizing: Optional[str] = None,
        warn: bool = True,
    ):
        """Create a probability mass histogram from the given distribution.

        Parameters
        ----------
        dist : BaseDistribution | BaseNumericDistribution | Real
            A distribution from which to generate numeric values. If the
            provided value is a :any:`BaseNumericDistribution`, simply return
            it.
        num_bins : Optional[int] (default = ref:``DEFAULT_NUM_BINS``)
            The number of bins for the numeric distribution to use. The time to
            construct a NumericDistribution is linear with ``num_bins``, and
            the time to run a binary operation on two distributions with the
            same number of bins is approximately quadratic. 100 bins provides a
            good balance between accuracy and speed. 1000 bins provides greater
            accuracy and is fast for small models, but for large models there
            will be some noticeable slowdown in binary operations.
        bin_sizing : Optional[str]
            The bin sizing method to use, which affects the accuracy of the
            bins. If none is given, a default will be chosen from
            :any:`DEFAULT_BIN_SIZING` based on the distribution type of
            ``dist``. It is recommended to use the default bin sizing method
            most of the time. See
            :any:`squigglepy.numeric_distribution.BinSizing` for a list of
            valid options and explanations of their behavior. warn :
            Optional[bool] (default = True) If True, raise warnings about bins
            with zero mass.
        warn : Optional[bool] (default = True)
            If True, raise warnings about bins with zero mass.

        Returns
        -------
        result : NumericDistribution | ZeroNumericDistribution
            The generated numeric distribution that represents ``dist``.

        """

        # ----------------------------
        # Handle special distributions
        # ----------------------------

        if isinstance(dist, BaseNumericDistribution):
            return dist
        if isinstance(dist, ConstantDistribution) or isinstance(dist, Real):
            x = dist if isinstance(dist, Real) else dist.x
            return cls(
                values=np.array([x]),
                masses=np.array([1]),
                zero_bin_index=0 if x >= 0 else 1,
                neg_ev_contribution=0 if x >= 0 else -x,
                pos_ev_contribution=x if x >= 0 else 0,
                exact_mean=x,
                exact_sd=0,
                bin_sizing=bin_sizing,
            )
        if isinstance(dist, BernoulliDistribution):
            return cls.from_distribution(1, num_bins, bin_sizing, warn).condition_on_success(
                dist.p
            )
        if isinstance(dist, MixtureDistribution):
            return cls.mixture(
                dist.dists,
                dist.weights,
                lclip=dist.lclip,
                rclip=dist.rclip,
                num_bins=num_bins,
                bin_sizing=bin_sizing,
                warn=warn,
            )
        if isinstance(dist, ComplexDistribution):
            left = dist.left
            right = dist.right
            if isinstance(left, BaseDistribution):
                left = cls.from_distribution(left, num_bins, bin_sizing, warn)
            if isinstance(right, BaseDistribution):
                right = cls.from_distribution(right, num_bins, bin_sizing, warn)
            if right is None:
                return dist.fn(left).clip(dist.lclip, dist.rclip)
            return dist.fn(left, right).clip(dist.lclip, dist.rclip)

        # ------------
        # Basic checks
        # ------------

        if type(dist) not in DEFAULT_BIN_SIZING:
            raise ValueError(f"Unsupported distribution type: {type(dist)}")

        num_bins = num_bins or DEFAULT_NUM_BINS[type(dist)]
        bin_sizing = BinSizing(bin_sizing or DEFAULT_BIN_SIZING[type(dist)])

        if num_bins % 2 != 0:
            raise ValueError(f"num_bins must be even, not {num_bins}")

        # ------------------------------------------------------------------
        # Handle distributions that are special cases of other distributions
        # ------------------------------------------------------------------

        if isinstance(dist, ChiSquareDistribution):
            return cls.from_distribution(
                GammaDistribution(shape=dist.df / 2, scale=2, lclip=dist.lclip, rclip=dist.rclip),
                num_bins=num_bins,
                bin_sizing=bin_sizing,
                warn=warn,
            )

        if isinstance(dist, ExponentialDistribution):
            return cls.from_distribution(
                GammaDistribution(shape=1, scale=dist.scale, lclip=dist.lclip, rclip=dist.rclip),
                num_bins=num_bins,
                bin_sizing=bin_sizing,
                warn=warn,
            )
        if isinstance(dist, PERTDistribution):
            # PERT is a generalization of Beta. We can generate a PERT by
            # generating a Beta and then scaling and shifting it.
            if dist.lclip is not None or dist.rclip is not None:
                raise ValueError("PERT distribution with lclip or rclip is not supported.")
            r = dist.right - dist.left
            alpha = 1 + dist.lam * (dist.mode - dist.left) / r
            beta = 1 + dist.lam * (dist.right - dist.mode) / r
            beta_dist = cls.from_distribution(
                BetaDistribution(
                    a=alpha,
                    b=beta,
                ),
                num_bins=num_bins,
                bin_sizing=bin_sizing,
                warn=warn,
            )
            # Note: There are formulas for the exact mean/SD of a PERT, but
            # scaling/shifting will correctly produce the exact mean/SD so we
            # don't need to set them manually.
            return dist.left + r * beta_dist

        # -------------------------------------------------------------------
        # Set up required parameters based on dist type and bin sizing method
        # -------------------------------------------------------------------

        max_support = {
            # These are the widest possible supports, but they maybe narrowed
            # later by lclip/rclip or by some bin sizing methods.
            BetaDistribution: (0, 1),
            GammaDistribution: (0, np.inf),
            LognormalDistribution: (0, np.inf),
            NormalDistribution: (-np.inf, np.inf),
            ParetoDistribution: (1, np.inf),
            UniformDistribution: (dist.x, dist.y),
        }[type(dist)]
        support = max_support
        ppf = {
            BetaDistribution: lambda p: stats.beta.ppf(p, dist.a, dist.b),
            GammaDistribution: lambda p: stats.gamma.ppf(p, dist.shape, scale=dist.scale),
            LognormalDistribution: lambda p: stats.lognorm.ppf(
                p, dist.norm_sd, scale=np.exp(dist.norm_mean)
            ),
            NormalDistribution: lambda p: stats.norm.ppf(p, loc=dist.mean, scale=dist.sd),
            ParetoDistribution: lambda p: stats.pareto.ppf(p, dist.shape),
            UniformDistribution: lambda p: stats.uniform.ppf(p, loc=dist.x, scale=dist.y - dist.x),
        }[type(dist)]
        cdf = {
            BetaDistribution: lambda x: stats.beta.cdf(x, dist.a, dist.b),
            GammaDistribution: lambda x: stats.gamma.cdf(x, dist.shape, scale=dist.scale),
            LognormalDistribution: lambda x: stats.lognorm.cdf(
                x, dist.norm_sd, scale=np.exp(dist.norm_mean)
            ),
            NormalDistribution: lambda x: stats.norm.cdf(x, loc=dist.mean, scale=dist.sd),
            ParetoDistribution: lambda x: stats.pareto.cdf(x, dist.shape),
            UniformDistribution: lambda x: stats.uniform.cdf(x, loc=dist.x, scale=dist.y - dist.x),
        }[type(dist)]

        # -----------
        # Set support
        # -----------

        dist_bin_sizing_supported = False
        new_support = _support_for_bin_sizing(dist, bin_sizing, num_bins)

        if new_support is not None:
            support = _narrow_support(support, new_support)
            dist_bin_sizing_supported = True
        elif bin_sizing == BinSizing.uniform:
            if isinstance(dist, BetaDistribution) or isinstance(dist, UniformDistribution):
                dist_bin_sizing_supported = True
        elif bin_sizing == BinSizing.log_uniform:
            if isinstance(dist, BetaDistribution):
                dist_bin_sizing_supported = True
        elif bin_sizing == BinSizing.ev:
            dist_bin_sizing_supported = True
        elif bin_sizing == BinSizing.mass:
            dist_bin_sizing_supported = True
        elif bin_sizing == BinSizing.fat_hybrid:
            if isinstance(dist, GammaDistribution) or isinstance(dist, LognormalDistribution):
                dist_bin_sizing_supported = True

        if not dist_bin_sizing_supported:
            raise ValueError(f"Unsupported bin sizing method {bin_sizing} for {type(dist)}.")

        # ----------------------------
        # Adjust support based on clip
        # ----------------------------

        support = _narrow_support(support, (dist.lclip, dist.rclip))

        # ---------------------------
        # Set exact_mean and exact_sd
        # ---------------------------

        if dist.lclip is None and dist.rclip is None:
            if isinstance(dist, BetaDistribution):
                exact_mean = stats.beta.mean(dist.a, dist.b)
                exact_sd = stats.beta.std(dist.a, dist.b)
            elif isinstance(dist, GammaDistribution):
                exact_mean = stats.gamma.mean(dist.shape, scale=dist.scale)
                exact_sd = stats.gamma.std(dist.shape, scale=dist.scale)
            elif isinstance(dist, LognormalDistribution):
                exact_mean = dist.lognorm_mean
                exact_sd = dist.lognorm_sd
            elif isinstance(dist, NormalDistribution):
                exact_mean = dist.mean
                exact_sd = dist.sd
            elif isinstance(dist, ParetoDistribution):
                if dist.shape <= 1:
                    raise ValueError(
                        "NumericDistribution does not support Pareto distributions with shape <= 1 because they have infinite mean."
                    )
                # exact_mean = 1 / (dist.shape - 1)  # Lomax
                exact_mean = dist.shape / (dist.shape - 1)
                if dist.shape <= 2:
                    exact_sd = np.inf
                else:
                    # exact_sd = np.sqrt(dist.shape / ((dist.shape - 1) ** 2 * (dist.shape - 2)))  # Lomax
                    exact_sd = np.sqrt(dist.shape / ((dist.shape - 1) ** 2 * (dist.shape - 2)))
            elif isinstance(dist, UniformDistribution):
                exact_mean = (dist.x + dist.y) / 2
                exact_sd = np.sqrt(1 / 12) * (dist.y - dist.x)
        else:
            if (
                isinstance(dist, BetaDistribution)
                or isinstance(dist, GammaDistribution)
                or isinstance(dist, LognormalDistribution)
            ):
                # For one-sided distributions without a known formula for
                # truncated mean, compute the mean using
                # ``contribution_to_ev``.
                contribution_to_ev = dist.contribution_to_ev(
                    support[1], normalized=False
                ) - dist.contribution_to_ev(support[0], normalized=False)
                mass = cdf(support[1]) - cdf(support[0])
                exact_mean = contribution_to_ev / mass
                exact_sd = None  # unknown
            elif isinstance(dist, NormalDistribution):
                a = (support[0] - dist.mean) / dist.sd
                b = (support[1] - dist.mean) / dist.sd
                exact_mean = stats.truncnorm.mean(a, b, dist.mean, dist.sd)
                exact_sd = stats.truncnorm.std(a, b, dist.mean, dist.sd)
            elif isinstance(dist, UniformDistribution):
                exact_mean = (support[0] + support[1]) / 2
                exact_sd = np.sqrt(1 / 12) * (support[1] - support[0])

        # -----------------------------------------------------------------
        # Split dist into negative and positive sides and generate bins for
        # each side
        # -----------------------------------------------------------------

        total_ev_contribution = dist.contribution_to_ev(
            support[1], normalized=False
        ) - dist.contribution_to_ev(support[0], normalized=False)
        neg_ev_contribution = max(
            0,
            dist.contribution_to_ev(max(0, support[0]), normalized=False)
            - dist.contribution_to_ev(support[0], normalized=False),
        )
        pos_ev_contribution = total_ev_contribution - neg_ev_contribution

        if bin_sizing == BinSizing.uniform:
            if support[0] > 0:
                neg_prop = 0
                pos_prop = 1
            elif support[1] < 0:
                neg_prop = 1
                pos_prop = 0
            else:
                width = support[1] - support[0]
                neg_prop = -support[0] / width
                pos_prop = support[1] / width
        elif bin_sizing == BinSizing.log_uniform:
            neg_prop = 0
            pos_prop = 1
        elif bin_sizing == BinSizing.ev:
            neg_prop = neg_ev_contribution / total_ev_contribution
            pos_prop = pos_ev_contribution / total_ev_contribution
        elif bin_sizing == BinSizing.mass:
            neg_mass = max(0, cdf(0) - cdf(support[0]))
            pos_mass = max(0, cdf(support[1]) - cdf(0))
            total_mass = neg_mass + pos_mass
            neg_prop = neg_mass / total_mass
            pos_prop = pos_mass / total_mass
        elif bin_sizing == BinSizing.fat_hybrid:
            neg_prop = 0
            pos_prop = 1
        else:
            raise ValueError(f"Unsupported bin sizing method: {bin_sizing}")

        # Divide up bins such that each bin has as close as possible to equal
        # contribution. If one side has very small but nonzero contribution,
        # still give it two bins.
        num_neg_bins, num_pos_bins = cls._num_bins_per_side(num_bins, neg_prop, pos_prop, 2)
        neg_masses, neg_values = cls._construct_bins(
            num_neg_bins,
            (support[0], min(0, support[1])),
            (max_support[0], min(0, max_support[1])),
            dist,
            cdf,
            ppf,
            bin_sizing,
            warn,
            is_reversed=True,
        )
        neg_values = -neg_values
        pos_masses, pos_values = cls._construct_bins(
            num_pos_bins,
            (max(0, support[0]), support[1]),
            (max(0, max_support[0]), max_support[1]),
            dist,
            cdf,
            ppf,
            bin_sizing,
            warn,
            is_reversed=False,
        )

        # Resize in case some bins got removed due to having zero mass/EV
        if len(neg_values) < num_neg_bins:
            neg_ev_contribution = abs(np.sum(neg_masses * neg_values))
            num_neg_bins = len(neg_values)
        if len(pos_values) < num_pos_bins:
            pos_ev_contribution = np.sum(pos_masses * pos_values)
            num_pos_bins = len(pos_values)

        masses = np.concatenate((neg_masses, pos_masses))
        values = np.concatenate((neg_values, pos_values))

        # Normalize masses to sum to 1 in case the distribution is clipped, but
        # don't do this until after setting values because values depend on the
        # mass relative to the full distribution, not the clipped distribution.
        masses /= np.sum(masses)

        return cls(
            values=np.array(values),
            masses=np.array(masses),
            zero_bin_index=num_neg_bins,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            exact_mean=exact_mean,
            exact_sd=exact_sd,
            bin_sizing=bin_sizing,
        )

    @classmethod
    def mixture(
        cls, dists, weights, lclip=None, rclip=None, num_bins=None, bin_sizing=None, warn=True
    ):
        # This function replicates how MixtureDistribution handles lclip/rclip:
        # it clips the sub-distributions based on their own lclip/rclip, then
        # takes the mixture sample, then clips the mixture sample based on the
        # mixture lclip/rclip.
        if num_bins is None:
            mixture_num_bins = DEFAULT_NUM_BINS[MixtureDistribution]
        dists = [d for d in dists]  # create new list to avoid mutating

        # Convert any Squigglepy dists into NumericDistributions
        for i in range(len(dists)):
            dists[i] = NumericDistribution.from_distribution(dists[i], num_bins, bin_sizing)

        value_vectors = [d.values for d in dists]
        weighted_mass_vectors = [d.masses * w for d, w in zip(dists, weights)]
        extended_values = np.concatenate(value_vectors)
        extended_masses = np.concatenate(weighted_mass_vectors)

        sorted_indexes = np.argsort(extended_values, kind="mergesort")
        extended_values = extended_values[sorted_indexes]
        extended_masses = extended_masses[sorted_indexes]
        zero_index = np.searchsorted(extended_values, 0)
        neg_ev_contribution = sum(d.neg_ev_contribution * w for d, w in zip(dists, weights))
        pos_ev_contribution = sum(d.pos_ev_contribution * w for d, w in zip(dists, weights))

        mixture = cls._resize_bins(
            extended_neg_values=extended_values[:zero_index],
            extended_neg_masses=extended_masses[:zero_index],
            extended_pos_values=extended_values[zero_index:],
            extended_pos_masses=extended_masses[zero_index:],
            num_bins=num_bins or mixture_num_bins,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            bin_sizing=BinSizing.ev,
            is_sorted=True,
        )
        mixture.bin_sizing = bin_sizing
        if all(d.exact_mean is not None for d in dists):
            mixture.exact_mean = sum(d.exact_mean * w for d, w in zip(dists, weights))
        if all(d.exact_sd is not None and d.exact_mean is not None for d in dists):
            second_moment = sum(
                (d.exact_mean**2 + d.exact_sd**2) * w for d, w in zip(dists, weights)
            )
            mixture.exact_sd = np.sqrt(second_moment - mixture.exact_mean**2)

        return mixture.clip(lclip, rclip)

    def given_value_satisfies(self, condition: Callable[float, bool]):
        """Return a new distribution conditioned on the value of the random
        variable satisfying ``condition``.

        Parameters
        ----------
        condition : Callable[float, bool]
        """
        good_indexes = np.where(np.vectorize(condition)(self.values))
        values = self.values[good_indexes]
        masses = self.masses[good_indexes]
        masses /= np.sum(masses)
        zero_bin_index = np.searchsorted(values, 0, side="left")
        neg_ev_contribution = -np.sum(masses[:zero_bin_index] * values[:zero_bin_index])
        pos_ev_contribution = np.sum(masses[zero_bin_index:] * values[zero_bin_index:])
        return NumericDistribution(
            values=values,
            masses=masses,
            zero_bin_index=zero_bin_index,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            exact_mean=None,
            exact_sd=None,
        )

    def probability_value_satisfies(self, condition: Callable[float, bool]):
        """Return the probability that the random variable satisfies
        ``condition``.

        Parameters
        ----------
        condition : Callable[float, bool]
        """
        return np.sum(self.masses[np.where(np.vectorize(condition)(self.values))])

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

    def est_mean(self):
        """Mean of the distribution, calculated using the histogram data (even
        if the exact mean is known)."""
        return np.sum(self.masses * self.values)

    def est_sd(self):
        """Standard deviation of the distribution, calculated using the
        histogram data (even if the exact SD is known)."""
        mean = self.mean()
        return np.sqrt(np.sum(self.masses * (self.values - mean) ** 2))

    def _init_interpolate_cdf(self):
        if self.interpolate_cdf is None:
            # Subtracting 0.5 * masses because eg the first out of 100 values
            # represents the 0.5th percentile, not the 1st percentile
            cum_mass = np.cumsum(self.masses) - 0.5 * self.masses
            self.interpolate_cdf = PchipInterpolator(self.values, cum_mass, extrapolate=True)

    def _init_interpolate_ppf(self):
        if self.interpolate_ppf is None:
            cum_mass = np.cumsum(self.masses) - 0.5 * self.masses

            # Mass diffs can be 0 if a mass is very small and gets rounded off.
            # The interpolator doesn't like this, so remove these values.
            nonzero_indexes = [i for (i, d) in enumerate(np.diff(cum_mass)) if d > 0]
            cum_mass = cum_mass[nonzero_indexes]
            values = self.values[nonzero_indexes]
            self.interpolate_ppf = PchipInterpolator(cum_mass, values, extrapolate=True)

    def cdf(self, x):
        """Estimate the proportion of the distribution that lies below ``x``."""
        self._init_interpolate_cdf()
        return self.interpolate_cdf(x)

    def ppf(self, q):
        self._init_interpolate_ppf()
        return self.interpolate_ppf(q)

    def clip(self, lclip, rclip):
        """Return a new distribution clipped to the given bounds.

        It is strongly recommended that, whenever possible, you construct a
        ``NumericDistribution`` by supplying a ``Distribution`` that has
        lclip/rclip defined on it, rather than calling
        ``NumericDistribution.clip``. ``NumericDistribution.clip`` works by
        deleting bins, which can greatly decrease accuracy.

        Parameters
        ----------
        lclip : Optional[float]
            The new lower bound of the distribution, or None if the lower bound
            should not change.
        rclip : Optional[float]
            The new upper bound of the distribution, or None if the upper bound
            should not change.

        Returns
        -------
        clipped : NumericDistribution
            A new distribution clipped to the given bounds.

        """
        if lclip is None and rclip is None:
            return NumericDistribution(
                values=self.values,
                masses=self.masses,
                zero_bin_index=self.zero_bin_index,
                neg_ev_contribution=self.neg_ev_contribution,
                pos_ev_contribution=self.pos_ev_contribution,
                exact_mean=self.exact_mean,
                exact_sd=self.exact_sd,
                bin_sizing=self.bin_sizing,
            )

        if lclip is None:
            lclip = -np.inf
        if rclip is None:
            rclip = np.inf

        if lclip >= rclip:
            raise ValueError(f"lclip ({lclip}) must be less than rclip ({rclip})")

        # bounds are inclusive
        start_index = np.searchsorted(self.values, lclip, side="left")
        end_index = np.searchsorted(self.values, rclip, side="right")

        new_values = np.array(self.values[start_index:end_index])
        new_masses = np.array(self.masses[start_index:end_index])
        clipped_mass = np.sum(new_masses)
        new_masses /= clipped_mass
        zero_bin_index = max(0, self.zero_bin_index - start_index)
        neg_ev_contribution = -np.sum(new_masses[:zero_bin_index] * new_values[:zero_bin_index])
        pos_ev_contribution = np.sum(new_masses[zero_bin_index:] * new_values[zero_bin_index:])

        return NumericDistribution(
            values=new_values,
            masses=new_masses,
            zero_bin_index=zero_bin_index,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            exact_mean=None,
            exact_sd=None,
            bin_sizing=self.bin_sizing,
        )

    def sample(self, n=1):
        """Generate ``n`` random samples from the distribution. The samples are generated by interpolating between bin values in the same manner as :any:`ppf`."""
        return self.ppf(np.random.uniform(size=n))

    def contribution_to_ev(self, x: Union[np.ndarray, float]):
        if self.interpolate_cev is None:
            bin_evs = self.masses * abs(self.values)
            fractions_of_ev = (np.cumsum(bin_evs) - 0.5 * bin_evs) / np.sum(bin_evs)
            self.interpolate_cev = PchipInterpolator(self.values, fractions_of_ev)
        return self.interpolate_cev(x)

    def inv_contribution_to_ev(self, fraction: Union[np.ndarray, float]):
        """Return the value such that ``fraction`` of the contribution to
        expected value lies to the left of that value.
        """
        if self.interpolate_inv_cev is None:
            bin_evs = self.masses * abs(self.values)
            fractions_of_ev = (np.cumsum(bin_evs) - 0.5 * bin_evs) / np.sum(bin_evs)
            self.interpolate_inv_cev = PchipInterpolator(fractions_of_ev, self.values)
        return self.interpolate_inv_cev(fraction)

    def plot(self, scale="linear"):
        import matplotlib
        from matplotlib import pyplot as plt

        # matplotlib.use('GTK3Agg')
        # matplotlib.use('Qt5Agg')
        values_for_widths = np.concatenate(([0], self.values))
        widths = np.diff(values_for_widths[1:])
        densities = self.masses / widths
        values, densities, widths = zip(
            *[
                (v, d, w)
                for v, d, w in zip(list(values_for_widths), list(densities), list(widths))
                if d > 0.001
            ]
        )
        if scale == "log":
            plt.xscale("log")
        plt.bar(values, densities, width=widths, align="edge")
        plt.savefig("/tmp/plot.png")
        plt.show()

    def richardson(r: float, correct_ev: bool = True):
        """A decorator that applies Richardson extrapolation to a
        NumericDistribution method to improve its accuracy.

        This decorator uses the following procedure (this procedure assumes a
        binary operation, but it works the same for any number of arguments):

        1. Evaluate ``z = func(x, y)`` where ``x`` and ``y`` are
           ``NumericDistribution`` objects.
        2. Construct a new ``x2`` and ``y2`` which are identical to ``x``
           and ``y`` except that they use half as many bins.
        3. Evaluate ``z2 = func(x2, y2)``.
        4. Apply Richardson extrapolation: ``res = (2^r * z - z2) / (2^r - 1)``
           for some constant exponent ``r``, chosen to maximize accuracy.

        Parameters
        ----------
        r : float
            The (positive) exponent to use in Richardson extrapolation. This
            should equal the rate at which the error shrinks as the number of
            bins increases. A higher ``r`` results in slower extrapolation.
        correct_ev : bool = True
            If True, adjust the negative and positive EV contributions to be
            exactly correct.

        Returns
        -------
        res : Callable
            A decorator function that takes a function ``func`` and returns a
            new function that applies Richardson extrapolation to ``func``.

        """

        def sum_pairs(arr):
            """Sum every pair of values in ``arr``."""
            return arr.reshape(-1, 2).sum(axis=1)

        def decorator(func):
            def inner(*hists):
                if not all(x.richardson_extrapolation_enabled for x in hists):
                    return func(*hists)

                # Richardson extrapolation often runs into issues due to wonky
                # bin sizing if the inputs don't have the same number of bins.
                if len(set(x.num_bins for x in hists)) != 1:
                    return func(*hists)

                # Empirically, BinSizing.ev and BinSizing.mass error shrinks at
                # a consistent rate r for all bins (except for the outermost
                # bins, which are more unpredictable). BinSizing.log_uniform
                # and BinSizing.uniform error growth rate isn't consistent
                # across bins, so Richardson extrapolation doesn't work well
                # (and often makes the result worse).
                if not all(
                    x.bin_sizing in [BinSizing.ev, BinSizing.mass, BinSizing.uniform]
                    for x in hists
                ):
                    return func(*hists)

                # Construct half_hists as identical to hists but with half as
                # many bins
                half_hists = []
                for x in hists:
                    if len(x.masses) % 2 != 0:
                        # If the number of bins is odd, we can't halve the
                        # number of bins, so just return the original result.
                        return func(*hists)

                    halfx_masses = sum_pairs(x.masses)
                    halfx_evs = sum_pairs(x.values * x.masses)
                    halfx_values = halfx_evs / halfx_masses
                    zero_bin_index = np.searchsorted(halfx_values, 0)
                    halfx = NumericDistribution(
                        values=halfx_values,
                        masses=halfx_masses,
                        zero_bin_index=zero_bin_index,
                        neg_ev_contribution=np.sum(
                            halfx_masses[:zero_bin_index] * -halfx_values[:zero_bin_index]
                        ),
                        pos_ev_contribution=np.sum(
                            halfx_masses[zero_bin_index:] * halfx_values[zero_bin_index:]
                        ),
                        exact_mean=x.exact_mean,
                        exact_sd=x.exact_sd,
                        min_bins_per_side=1,
                    )
                    half_hists.append(halfx)

                half_res = func(*half_hists)
                full_res = func(*hists)
                if 2 * half_res.zero_bin_index != full_res.zero_bin_index:
                    # In some edge cases, full_res has very small negative mass
                    # and half_res has no negative mass (ex: norm(mean=0, sd=2)
                    # + norm(mean=9, sd=1)), so the bins aren't lined up
                    # correctly for Richardson extrapolation. These edge cases
                    # are rare, so it's not a big deal to bail out on
                    # Richardson extrapolation in those cases.
                    return full_res

                paired_full_masses = sum_pairs(full_res.masses)
                paired_full_evs = sum_pairs(full_res.values * full_res.masses)
                paired_full_values = paired_full_evs / paired_full_masses
                k = 2 ** (-r)
                richardson_masses = (k * half_res.masses - paired_full_masses) / (k - 1)
                if any(richardson_masses < 0):
                    # TODO: delete me when you're confident that this doesn't
                    # happen anymore
                    return full_res

                mass_adjustment = np.repeat(richardson_masses / paired_full_masses, 2)
                new_masses = full_res.masses * np.where(
                    np.isnan(mass_adjustment), 1, mass_adjustment
                )

                # method 1: adjust EV
                # full_evs = full_res.values * full_res.masses
                # half_evs = half_res.values * half_res.masses
                # richardson_evs = (k * half_evs - paired_full_evs) / (k - 1)
                # ev_adjustment = np.repeat(richardson_evs / paired_full_evs, 2)
                # new_evs = full_evs * np.where(np.isnan(ev_adjustment), 1, ev_adjustment)
                # new_values = new_evs / new_masses

                # method 2: adjust values directly...
                richardson_values = (k * half_res.values - paired_full_values) / (k - 1)
                value_adjustment = np.repeat(richardson_values / paired_full_values, 2)
                new_values = full_res.values * np.where(
                    np.isnan(value_adjustment), 1, value_adjustment
                )
                # ...then adjust EV to be exactly correct
                if correct_ev:
                    new_neg_ev_contribution = np.sum(
                        new_masses[: full_res.zero_bin_index]
                        * -new_values[: full_res.zero_bin_index]
                    )
                    new_pos_ev_contribution = np.sum(
                        new_masses[full_res.zero_bin_index :]
                        * new_values[full_res.zero_bin_index :]
                    )
                    new_values[: full_res.zero_bin_index] *= (
                        full_res.neg_ev_contribution / new_neg_ev_contribution
                    )
                    new_values[full_res.zero_bin_index :] *= (
                        full_res.pos_ev_contribution / new_pos_ev_contribution
                    )

                full_res.masses = new_masses
                full_res.values = new_values
                full_res.zero_bin_index = np.searchsorted(new_values, 0)
                if len(np.unique([x.bin_sizing for x in hists])) == 1:
                    full_res.bin_sizing = hists[0].bin_sizing
                return full_res

            return inner

        return decorator

    @classmethod
    def _num_bins_per_side(
        cls, num_bins, neg_contribution, pos_contribution, min_bins_per_side, allowance=0
    ):
        """Determine how many bins to allocate to the positive and negative
        sides of the distribution.

        The negative and positive sides will get a number of bins approximately
        proportional to `neg_contribution` and `pos_contribution` respectively.

        Ordinarily, a domain gets its own bin if it represents greater than `1
        / num_bins / 2` of the total contribution. But if one side of the
        distribution has less than that, it will still get one bin if it has
        greater than `allowance * 1 / num_bins / 2` of the total contribution.
        `allowance = 0` means both sides get a bin as long as they have any
        contribution.

        If one side has less than that but still nonzero contribution, that
        side will be allocated zero bins and that side's contribution will be
        dropped, which means `neg_contribution` and `pos_contribution` may need
        to be adjusted.

        Parameters
        ----------
        num_bins : int
            Total number of bins across the distribution.
        neg_contribution : float
            The total contribution of value from the negative side, using
            whatever measure of value determines bin sizing.
        pos_contribution : float
            The total contribution of value from the positive side.
        allowance = 0.5 : float
            The fraction

        Returns
        -------
        (num_neg_bins, num_pos_bins) : (int, int)
            Number of bins assigned to the negative/positive side of the
            distribution.

        """
        min_prop_cutoff = min_bins_per_side * allowance * 1 / num_bins / 2
        total_contribution = neg_contribution + pos_contribution
        num_neg_bins = min_bins_per_side * int(
            np.round(num_bins * neg_contribution / total_contribution / min_bins_per_side)
        )
        num_pos_bins = num_bins - num_neg_bins

        if neg_contribution / total_contribution > min_prop_cutoff:
            num_neg_bins = max(min_bins_per_side, num_neg_bins)
            num_pos_bins = num_bins - num_neg_bins
        else:
            num_neg_bins = 0
            num_pos_bins = num_bins

        if pos_contribution / total_contribution > min_prop_cutoff:
            num_pos_bins = max(min_bins_per_side, num_pos_bins)
            num_neg_bins = num_bins - num_pos_bins
        else:
            num_pos_bins = 0
            num_neg_bins = num_bins

        return (num_neg_bins, num_pos_bins)

    @classmethod
    def _resize_pos_bins(
        cls,
        extended_values,
        extended_masses,
        num_bins,
        ev,
        bin_sizing=BinSizing.bin_count,
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
        # TODO: This whole method is messy, it could use a refactoring
        if num_bins == 0:
            return (np.array([]), np.array([]))

        bin_evs = None
        masses = None

        if bin_sizing == BinSizing.bin_count:
            if len(extended_values) == 1 and num_bins == 2:
                extended_values = np.repeat(extended_values, 2)
                extended_masses = np.repeat(extended_masses, 2) / 2
            elif len(extended_values) < num_bins:
                return (extended_values, extended_masses)

            boundary_indexes = np.round(np.linspace(0, len(extended_values), num_bins + 1)).astype(
                int
            )

            if not is_sorted:
                # Partition such that the values in one bin are all less than
                # or equal to the values in the next bin. Values within bins
                # don't need to be sorted, and partitioning is ~10% faster than
                # timsort.
                partitioned_indexes = extended_values.argpartition(boundary_indexes[1:-1])
                extended_values = extended_values[partitioned_indexes]
                extended_masses = extended_masses[partitioned_indexes]

            extended_evs = extended_values * extended_masses
            if len(extended_masses) % num_bins == 0:
                # Vectorize when possible for better performance
                bin_evs = extended_evs.reshape((num_bins, -1)).sum(axis=1)
                masses = extended_masses.reshape((num_bins, -1)).sum(axis=1)

        elif bin_sizing == BinSizing.ev:
            if not is_sorted:
                sorted_indexes = extended_values.argsort(kind="mergesort")
                extended_values = extended_values[sorted_indexes]
                extended_masses = extended_masses[sorted_indexes]

            extended_evs = extended_values * extended_masses
            cumulative_evs = np.concatenate(([0], np.cumsum(extended_evs)))

            # Using cumulative_evs[-1] as the upper bound can create rounding
            # errors. For example, if there are 100 bins with equal EV,
            # boundary_evs will be slightly smaller than cumulative_evs until
            # near the end, which will duplicate the first bin and skip a bin
            # near the end. Slightly increasing the upper bound fixes this.
            upper_bound = cumulative_evs[-1] * (1 + 1e-6)

            boundary_evs = np.linspace(0, upper_bound, num_bins + 1)
            boundary_indexes = np.searchsorted(cumulative_evs, boundary_evs, side="right") - 1
            # Fix bin boundaries where boundary[i] == boundary[i+1]
            if any(boundary_indexes[:-1] == boundary_indexes[1:]):
                boundary_indexes = _bump_indexes(boundary_indexes, len(extended_values))

        elif bin_sizing == BinSizing.log_uniform:
            # ``bin_count`` puts too much mass in the bins on the left and
            # right tails, but it's still more accurate than log-uniform
            # sizing, I don't know why.
            assert num_bins % 2 == 0
            assert len(extended_values) == num_bins**2

            use_pyramid_method = False
            if use_pyramid_method:
                # method 1: size bins in a pyramid shape. this preserves
                # log-uniform bin sizing but it makes the bin widths unnecessarily
                # large
                ascending_indexes = 2 * np.array(range(num_bins // 2 + 1)) ** 2
                descending_indexes = np.flip(num_bins**2 - ascending_indexes)
                boundary_indexes = np.concatenate((ascending_indexes, descending_indexes[1:]))

            else:
                # method 2: size bins by going out a fixed number of log-standard
                # deviations in each direction
                log_mean = np.average(np.log(extended_values), weights=extended_masses)
                log_sd = np.sqrt(
                    np.average((np.log(extended_values) - log_mean) ** 2, weights=extended_masses)
                )
                scale = 6.5
                log_left_bound = log_mean - scale * log_sd
                log_right_bound = log_mean + scale * log_sd
                log_boundary_values = np.linspace(log_left_bound, log_right_bound, num_bins + 1)
                boundary_values = np.exp(log_boundary_values)

                if not is_sorted:
                    # TODO: log-uniform can maybe avoid sorting. bin edges are
                    # calculated in advance, so scan once over
                    # extended_values/masses and add the mass to each bin. but
                    # need a way to find the right bin in O(1)
                    sorted_indexes = extended_values.argsort(kind="mergesort")
                    extended_values = extended_values[sorted_indexes]
                    extended_masses = extended_masses[sorted_indexes]

                boundary_indexes = np.searchsorted(extended_values, boundary_values)

            # Compute sums one at a time instead of using ``cumsum`` because
            # ``cumsum`` produces non-trivial rounding errors.
            extended_evs = extended_values * extended_masses
        else:
            raise ValueError(f"resize_pos_bins: Unsupported bin sizing method: {bin_sizing}")

        if bin_evs is None:
            bin_evs = np.array(
                [
                    np.sum(extended_evs[i:j])
                    for (i, j) in zip(boundary_indexes[:-1], boundary_indexes[1:])
                ]
            )
        if masses is None:
            masses = np.array(
                [
                    np.sum(extended_masses[i:j])
                    for (i, j) in zip(boundary_indexes[:-1], boundary_indexes[1:])
                ]
            )

        values = bin_evs / masses
        return (values, masses)

    @classmethod
    def _resize_bins(
        cls,
        extended_neg_values: np.ndarray,
        extended_neg_masses: np.ndarray,
        extended_pos_values: np.ndarray,
        extended_pos_masses: np.ndarray,
        num_bins: int,
        neg_ev_contribution: float,
        pos_ev_contribution: float,
        bin_sizing: Optional[BinSizing] = BinSizing.bin_count,
        min_bins_per_side: Optional[int] = 2,
        is_sorted: Optional[bool] = False,
    ):
        """Given two arrays of values and masses representing the result of a
        binary operation on two distributions, compress the arrays down to
        ``num_bins`` bins and return the new values and masses of the bins.

        Parameters
        ----------
        extended_neg_values : np.ndarray
            The values of the negative side of the distribution. The values must
            all be negative.
        extended_neg_masses : np.ndarray
            The probability masses of the negative side of the distribution.
        extended_pos_values : np.ndarray
            The values of the positive side of the distribution. The values must
            all be positive.
        extended_pos_masses : np.ndarray
            The probability masses of the positive side of the distribution.
        num_bins : int
            The number of bins to compress the distribution into.
        neg_ev_contribution : float
            The expected value of the negative side of the distribution.
        pos_ev_contribution : float
            The expected value of the positive side of the distribution.
        is_sorted : bool
            If True, assume that ``extended_neg_values``,
            ``extended_neg_masses``, ``extended_pos_values``, and
            ``extended_pos_masses`` are already sorted in ascending order. This
            provides a significant performance improvement (~3x).

        Returns
        -------
        values : np.ndarray
            The values of the bins.
        masses : np.ndarray
            The probability masses of the bins.

        """
        if True:
            # TODO: Lol
            num_neg_bins, num_pos_bins = cls._num_bins_per_side(
                num_bins, neg_ev_contribution, pos_ev_contribution, min_bins_per_side
            )
        elif bin_sizing == BinSizing.bin_count:
            num_neg_bins, num_pos_bins = cls._num_bins_per_side(
                num_bins, len(extended_neg_masses), len(extended_pos_masses), min_bins_per_side
            )
        elif bin_sizing == BinSizing.ev:
            num_neg_bins, num_pos_bins = cls._num_bins_per_side(
                num_bins, neg_ev_contribution, pos_ev_contribution, min_bins_per_side
            )
        else:
            raise ValueError(f"resize_bins: Unsupported bin sizing method: {bin_sizing}")

        total_ev = pos_ev_contribution - neg_ev_contribution
        if num_neg_bins == 0:
            neg_ev_contribution = 0
            pos_ev_contribution = total_ev
        if num_pos_bins == 0:
            neg_ev_contribution = -total_ev
            pos_ev_contribution = 0

        # Collect extended_values and extended_masses into the correct number
        # of bins. Make ``extended_values`` positive because ``_resize_bins``
        # can only operate on non-negative values. Making them positive means
        # they're now reverse-sorted, so reverse them.
        neg_values, neg_masses = cls._resize_pos_bins(
            extended_values=np.flip(-extended_neg_values),
            extended_masses=np.flip(extended_neg_masses),
            num_bins=num_neg_bins,
            ev=neg_ev_contribution,
            bin_sizing=bin_sizing,
            is_sorted=is_sorted,
        )

        # ``_resize_bins`` returns positive values, so negate and reverse them.
        neg_values = np.flip(-neg_values)
        neg_masses = np.flip(neg_masses)

        # Collect extended_values and extended_masses into the correct number
        # of bins, for the positive values this time.
        pos_values, pos_masses = cls._resize_pos_bins(
            extended_values=extended_pos_values,
            extended_masses=extended_pos_masses,
            num_bins=num_pos_bins,
            ev=pos_ev_contribution,
            bin_sizing=bin_sizing,
            is_sorted=is_sorted,
        )

        # Construct the resulting ``NumericDistribution`` object.
        values = np.concatenate((neg_values, pos_values))
        masses = np.concatenate((neg_masses, pos_masses))
        return NumericDistribution(
            values=values,
            masses=masses,
            zero_bin_index=len(neg_masses),
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            exact_mean=None,
            exact_sd=None,
        )

    def __eq__(x, y):
        return x.values == y.values and x.masses == y.masses

    # @richardson(r=2)  # TODO
    def __add__(x, y):
        if isinstance(y, Real):
            return x.shift_by(y)
        elif isinstance(y, ZeroNumericDistribution):
            return y.__radd__(x)
        elif not isinstance(y, NumericDistribution):
            raise TypeError(f"Cannot add types {type(x)} and {type(y)}")

        cls = x
        num_bins = max(len(x), len(y))

        # Add every pair of values and find the joint probabilty mass for every
        # sum.
        extended_values = np.add.outer(x.values, y.values).reshape(-1)
        extended_masses = np.outer(x.masses, y.masses).reshape(-1)

        # Sort so we can split the values into positive and negative sides.
        # Use timsort (called 'mergesort' by the numpy API) because
        # ``extended_values`` contains many sorted runs. And then pass
        # `is_sorted` down to `_resize_bins` so it knows not to sort again.
        sorted_indexes = extended_values.argsort(kind="mergesort")
        extended_values = extended_values[sorted_indexes]
        extended_masses = extended_masses[sorted_indexes]
        zero_index = np.searchsorted(extended_values, 0)
        is_sorted = True

        # Find how much of the EV contribution is on the negative side vs. the
        # positive side.
        neg_ev_contribution = -np.sum(extended_values[:zero_index] * extended_masses[:zero_index])
        pos_ev_contribution = np.sum(extended_values[zero_index:] * extended_masses[zero_index:])

        res = cls._resize_bins(
            extended_neg_values=extended_values[:zero_index],
            extended_neg_masses=extended_masses[:zero_index],
            extended_pos_values=extended_values[zero_index:],
            extended_pos_masses=extended_masses[zero_index:],
            num_bins=num_bins,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            is_sorted=is_sorted,
            min_bins_per_side=x.min_bins_per_side,
        )

        if x.exact_mean is not None and y.exact_mean is not None:
            res.exact_mean = x.exact_mean + y.exact_mean
        if x.exact_sd is not None and y.exact_sd is not None:
            res.exact_sd = np.sqrt(x.exact_sd**2 + y.exact_sd**2)
        return res

    def shift_by(self, scalar):
        """Shift the distribution over by a constant factor."""
        values = self.values + scalar
        zero_bin_index = np.searchsorted(values, 0)
        return NumericDistribution(
            values=values,
            masses=self.masses,
            zero_bin_index=zero_bin_index,
            neg_ev_contribution=-np.sum(values[:zero_bin_index] * self.masses[:zero_bin_index]),
            pos_ev_contribution=np.sum(values[zero_bin_index:] * self.masses[zero_bin_index:]),
            exact_mean=self.exact_mean + scalar if self.exact_mean is not None else None,
            exact_sd=self.exact_sd,
        )

    def __neg__(self):
        return NumericDistribution(
            values=np.flip(-self.values),
            masses=np.flip(self.masses),
            zero_bin_index=len(self.values) - self.zero_bin_index,
            neg_ev_contribution=self.pos_ev_contribution,
            pos_ev_contribution=self.neg_ev_contribution,
            exact_mean=-self.exact_mean if self.exact_mean is not None else None,
            exact_sd=self.exact_sd,
        )

    def __abs__(self):
        values = abs(self.values)
        masses = self.masses
        sorted_indexes = np.argsort(values, kind="mergesort")
        values = values[sorted_indexes]
        masses = masses[sorted_indexes]
        return NumericDistribution(
            values=values,
            masses=masses,
            zero_bin_index=0,
            neg_ev_contribution=0,
            pos_ev_contribution=np.sum(values * masses),
            exact_mean=None,
            exact_sd=None,
        )

    def __mul__(x, y):
        if isinstance(y, Real):
            return x.scale_by(y)
        elif isinstance(y, ZeroNumericDistribution):
            return y.__rmul__(x)
        elif not isinstance(y, NumericDistribution):
            raise TypeError(f"Cannot add types {type(x)} and {type(y)}")

        return x._inner_mul(y)

    @richardson(r=1.75)
    def _inner_mul(x, y):
        cls = x
        num_bins = max(len(x), len(y))

        # If xpos is the positive part of x and xneg is the negative part, then
        # resultpos = (xpos * ypos) + (xneg * yneg) and resultneg = (xpos * yneg) + (xneg *
        # ypos). We perform this calculation by running these steps:
        #
        # 1. Multiply the four pairs of one-sided distributions,
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

        # Calculate the four products.
        extended_neg_values = np.concatenate(
            (
                np.outer(xneg_values, ypos_values).reshape(-1),
                np.outer(xpos_values, yneg_values).reshape(-1),
            )
        )
        extended_neg_masses = np.concatenate(
            (
                np.outer(xneg_masses, ypos_masses).reshape(-1),
                np.outer(xpos_masses, yneg_masses).reshape(-1),
            )
        )
        extended_pos_values = np.concatenate(
            (
                np.outer(xneg_values, yneg_values).reshape(-1),
                np.outer(xpos_values, ypos_values).reshape(-1),
            )
        )
        extended_pos_masses = np.concatenate(
            (
                np.outer(xneg_masses, yneg_masses).reshape(-1),
                np.outer(xpos_masses, ypos_masses).reshape(-1),
            )
        )

        # Set the number of bins per side to be approximately proportional to
        # the EV contribution, but make sure that if a side has non-trivial EV
        # contribution, it gets at least one bin, even if it's less
        # contribution than an average bin.
        neg_ev_contribution = (
            x.neg_ev_contribution * y.pos_ev_contribution
            + x.pos_ev_contribution * y.neg_ev_contribution
        )
        pos_ev_contribution = (
            x.neg_ev_contribution * y.neg_ev_contribution
            + x.pos_ev_contribution * y.pos_ev_contribution
        )

        res = cls._resize_bins(
            extended_neg_values=extended_neg_values,
            extended_neg_masses=extended_neg_masses,
            extended_pos_values=extended_pos_values,
            extended_pos_masses=extended_pos_masses,
            num_bins=num_bins,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            min_bins_per_side=x.min_bins_per_side,
            is_sorted=False,
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

    def __pow__(x, y):
        """Raise the distribution to a power.

        Note: x * x does not give the same result as x ** 2 because
        multiplication assumes that the two distributions are independent.
        """
        if isinstance(y, Real) or isinstance(y, NumericDistribution):
            return (x.log() * y).exp()
        else:
            raise TypeError(f"Cannot compute x**y for types {type(x)} and {type(y)}")

    def __rpow__(x, y):
        # Compute y**x
        if isinstance(y, Real):
            return (x * np.log(y)).exp()
        else:
            raise TypeError(f"Cannot compute x**y for types {type(x)} and {type(y)}")

    def scale_by(self, scalar):
        """Scale the distribution by a constant factor."""
        if scalar < 0:
            return -self * -scalar
        return NumericDistribution(
            values=self.values * scalar,
            masses=self.masses,
            zero_bin_index=self.zero_bin_index,
            neg_ev_contribution=self.neg_ev_contribution * scalar,
            pos_ev_contribution=self.pos_ev_contribution * scalar,
            exact_mean=self.exact_mean * scalar if self.exact_mean is not None else None,
            exact_sd=self.exact_sd * scalar if self.exact_sd is not None else None,
        )

    def reciprocal(self):
        """Return the reciprocal of the distribution.

        Warning: The result can be very inaccurate for certain distributions
        and bin sizing methods. Specifically, if the distribution is fat-tailed
        and does not use log-uniform bin sizing, the reciprocal will be
        inaccurate for small values. Most bin sizing methods on fat-tailed
        distributions maximize information at the tails in exchange for less
        accuracy on [0, 1], which means the reciprocal will contain very little
        information about the tails. Log-uniform bin sizing is most accurate
        because it is invariant with reciprocation.
        """
        values = 1 / self.values
        sorted_indexes = values.argsort()
        values = values[sorted_indexes]
        masses = self.masses[sorted_indexes]

        # Re-calculate EV contribution manually.
        neg_ev_contribution = np.sum(values[: self.zero_bin_index] * masses[: self.zero_bin_index])
        pos_ev_contribution = np.sum(values[self.zero_bin_index :] * masses[self.zero_bin_index :])

        return NumericDistribution(
            values=values,
            masses=masses,
            zero_bin_index=self.zero_bin_index,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            # There is no general formula for the mean and SD of the
            # reciprocal of a random variable.
            exact_mean=None,
            exact_sd=None,
        )

    def exp(self):
        """Return the exponential of the distribution."""
        # Note: This code naively sets the average value within each bin to
        # e^x, which is wrong because we want E[e^X], not e^E[X]. An
        # alternative method, which you might expect to be more accurate, is to
        # interpolate the edges of each bin and then set each bin's value to
        # the result of the integral
        #
        #     .. math::
        #     \int_{lb}^{ub} \frac{1}{ub - lb} exp(x) dx
        #
        # However, this method turns out to be less accurate overall (although
        # it's more accurate in the tails of the distribution).
        #
        # Where the underlying distribution is normal, the naive e^E[X] method
        # systematically underestimates expected value (by something like 0.1%
        # with num_bins=200), and the integration method overestimates expected
        # value by about 3x as much. Both methods mis-estimate the standard
        # deviation in the same direction as they mis-estimate the mean but by
        # a somewhat larger margin, with the naive method again having the
        # better estimate.
        #
        # Another method would be not to interpolate the edge values, and
        # instead record the true edge values when the numeric distribution is
        # generated and carry them through mathematical operations by
        # re-calculating them. But this method would be much more complicated,
        # and we'd need to lazily compute the edge values to avoid a ~2x
        # performance penalty.
        values = np.exp(self.values)
        return NumericDistribution(
            values=values,
            masses=self.masses,
            zero_bin_index=0,
            neg_ev_contribution=0,
            pos_ev_contribution=np.sum(values * self.masses),
            exact_mean=None,
            exact_sd=None,
        )

    def log(self):
        """Return the natural log of the distribution."""
        # See :any:`exp`` for some discussion of accuracy. For ``log` on a
        # log-normal distribution, both the naive method and the integration
        # method tend to overestimate the true mean, but the naive method
        # overestimates it by less.
        if self.zero_bin_index != 0:
            raise ValueError("Cannot take the log of a distribution with non-positive values")

        values = np.log(self.values)
        return NumericDistribution(
            values=values,
            masses=self.masses,
            zero_bin_index=np.searchsorted(values, 0),
            neg_ev_contribution=np.sum(
                values[: self.zero_bin_index] * self.masses[: self.zero_bin_index]
            ),
            pos_ev_contribution=np.sum(
                values[self.zero_bin_index :] * self.masses[self.zero_bin_index :]
            ),
            exact_mean=None,
            exact_sd=None,
        )

    def __hash__(self):
        return hash(repr(self.values) + "," + repr(self.masses))


class ZeroNumericDistribution(BaseNumericDistribution):
    """
    A :any:`NumericDistribution` with a point mass at zero.
    """

    def __init__(self, dist: NumericDistribution, zero_mass: float):
        if not isinstance(dist, NumericDistribution):
            raise TypeError(f"dist must be a NumericDistribution, got {type(dist)}")
        if not isinstance(zero_mass, Real):
            raise TypeError(f"zero_mass must be a Real, got {type(zero_mass)}")

        self._version = __version__
        self.dist = dist
        self.zero_mass = zero_mass
        self.nonzero_mass = 1 - zero_mass
        self.exact_mean = None
        self.exact_sd = None
        self.exact_2nd_moment = None

        if dist.exact_mean is not None:
            self.exact_mean = dist.exact_mean * self.nonzero_mass
            if dist.exact_sd is not None:
                nonzero_moment2 = dist.exact_mean**2 + dist.exact_sd**2
                moment2 = self.nonzero_mass * nonzero_moment2
                variance = moment2 - self.exact_mean**2
                self.exact_sd = np.sqrt(variance)

        self._neg_mass = np.sum(dist.masses[: dist.zero_bin_index]) * self.nonzero_mass

        # To be computed lazily
        self.interpolate_ppf = None

    @classmethod
    def wrap(cls, dist: BaseNumericDistribution, zero_mass: float):
        if isinstance(dist, ZeroNumericDistribution):
            return cls(dist.dist, zero_mass + dist.zero_mass * (1 - zero_mass))
        return cls(dist, zero_mass)

    def given_value_satisfies(self, condition):
        nonzero_dist = self.dist.given_value_satisfies(condition)
        if condition(0):
            nonzero_mass = np.sum(
                [x for i, x in enumerate(self.dist.masses) if condition(self.dist.values[i])]
            )
            zero_mass = self.zero_mass
            total_mass = nonzero_mass + zero_mass
            scaled_zero_mass = zero_mass / total_mass
            return ZeroNumericDistribution(nonzero_dist, scaled_zero_mass)
        return nonzero_dist

    def probability_value_satisfies(self, condition):
        return (
            self.dist.probability_value_satisfies(condition) * self.nonzero_mass
            + condition(0) * self.zero_mass
        )

    def est_mean(self):
        return self.dist.est_mean() * self.nonzero_mass

    def est_sd(self):
        mean = self.mean()
        nonzero_variance = (
            np.sum(self.dist.masses * (self.dist.values - mean) ** 2) * self.nonzero_mass
        )
        zero_variance = self.zero_mass * mean**2
        variance = nonzero_variance + zero_variance
        return np.sqrt(variance)

    def ppf(self, q):
        if not isinstance(q, Real):
            return np.array([self.ppf(x) for x in q])

        if q < 0 or q > 1:
            raise ValueError(f"q must be between 0 and 1, got {q}")

        if q <= self._neg_mass:
            return self.dist.ppf(q / self.nonzero_mass)
        elif q < self._neg_mass + self.zero_mass:
            return 0
        else:
            return self.dist.ppf((q - self.zero_mass) / self.nonzero_mass)

    def __eq__(x, y):
        return x.zero_mass == y.zero_mass and x.dist == y.dist

    def __add__(x, y):
        if isinstance(y, NumericDistribution):
            return x + ZeroNumericDistribution(y, 0)
        elif isinstance(y, Real):
            return x.shift_by(y)
        elif not isinstance(y, ZeroNumericDistribution):
            raise ValueError(f"Cannot add types {type(x)} and {type(y)}")
        nonzero_sum = (x.dist + y.dist) * x.nonzero_mass * y.nonzero_mass
        extra_x = x.dist * x.nonzero_mass * y.zero_mass
        extra_y = y.dist * x.zero_mass * y.nonzero_mass
        zero_mass = x.zero_mass * y.zero_mass
        return ZeroNumericDistribution(nonzero_sum + extra_x + extra_y, zero_mass)

    def shift_by(self, scalar):
        old_zero_index = self.dist.zero_bin_index
        shifted_dist = self.dist.shift_by(scalar)
        scaled_masses = shifted_dist.masses * self.nonzero_mass
        values = np.insert(shifted_dist.values, old_zero_index, scalar)
        masses = np.insert(scaled_masses, old_zero_index, self.zero_mass)
        exact_mean = None
        if self.exact_mean is not None:
            exact_mean = self.exact_mean + scalar
        exact_sd = self.exact_sd

        return NumericDistribution(
            values=values,
            masses=masses,
            zero_bin_index=shifted_dist.zero_bin_index,
            neg_ev_contribution=shifted_dist.neg_ev_contribution * self.nonzero_mass
            + min(0, -scalar) * self.zero_mass,
            pos_ev_contribution=shifted_dist.pos_ev_contribution * self.nonzero_mass
            + min(0, scalar) * self.zero_mass,
            exact_mean=exact_mean,
            exact_sd=exact_sd,
        )

    def __neg__(self):
        return ZeroNumericDistribution(-self.dist, self.zero_mass)

    def __abs__(self):
        return ZeroNumericDistribution(abs(self.dist), self.zero_mass)

    def exp(self):
        # TODO: exponentiate the wrapped dist, then do something like shift_by
        # to insert a 1 into the bins
        return NotImplementedError

    def log(self):
        raise ValueError("Cannot take the log of a distribution with non-positive values")

    def __mul__(x, y):
        if isinstance(y, NumericDistribution):
            return x * ZeroNumericDistribution(y, 0)
        if isinstance(y, Real):
            return x.scale_by(y)
        dist = x.dist * y.dist
        nonzero_mass = x.nonzero_mass * y.nonzero_mass
        return ZeroNumericDistribution(dist, 1 - nonzero_mass)

    def scale_by(self, scalar):
        return ZeroNumericDistribution(self.dist.scale_by(scalar), self.zero_mass)

    def reciprocal(self):
        raise ValueError(
            "Reciprocal is undefined for probability distributions with non-infinitesimal mass at zero"
        )

    def __hash__(self):
        return 33 * hash(repr(self.zero_mass)) + hash(self.dist)


def numeric(
    dist: Union[BaseDistribution, BaseNumericDistribution],
    num_bins: Optional[int] = None,
    bin_sizing: Optional[str] = None,
    warn: bool = True,
):
    """Create a probability mass histogram from the given distribution.

    Parameters
    ----------
    dist : BaseDistribution | BaseNumericDistribution
        A distribution from which to generate numeric values. If the
        provided value is a :any:`BaseNumericDistribution`, simply return
        it.
    num_bins : Optional[int] (default = :any:``DEFAULT_NUM_BINS``)
        The number of bins for the numeric distribution to use. The time to
        construct a NumericDistribution is linear with ``num_bins``, and
        the time to run a binary operation on two distributions with the
        same number of bins is approximately quadratic with ``num_bins``.
        100 bins provides a good balance between accuracy and speed.
    bin_sizing : Optional[str]
        The bin sizing method to use, which affects the accuracy of the bins.
        If none is given, a default will be chosen from
        :any:`DEFAULT_BIN_SIZING` based on the distribution type of ``dist``.
        It is recommended to use the default bin sizing method most of the
        time. See :any:`BinSizing` for a list of valid options and explanations
        of their behavior. warn : Optional[bool] (default = True) If True,
        raise warnings about bins with zero mass.
    warn : Optional[bool] (default = True)
        If True, raise warnings about bins with zero mass.

    Returns
    -------
    result : NumericDistribution | ZeroNumericDistribution
        The generated numeric distribution that represents ``dist``.

    """
    return NumericDistribution.from_distribution(dist, num_bins, bin_sizing, warn)
