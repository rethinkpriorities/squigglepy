from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from numbers import Real
import numpy as np
from scipy import optimize, stats
from scipy.interpolate import PchipInterpolator
from typing import Literal, Optional, Tuple, Union
import warnings

from .distributions import (
    BaseDistribution,
    BetaDistribution,
    ChiSquareDistribution,
    ComplexDistribution,
    ExponentialDistribution,
    GammaDistribution,
    LognormalDistribution,
    MixtureDistribution,
    NormalDistribution,
    ParetoDistribution,
    UniformDistribution,
)
from .version import __version__


class BinSizing(Enum):
    """An enum for the different methods of sizing histogram bins. A histogram
    with finitely many bins can only contain so much information about the
    shape of a distribution; the choice of bin sizing changes what information
    NumericDistribution prioritizes.

    Attributes
    ----------
    uniform : str
        Divides the distribution into bins of equal width. For distributions
        with infinite support (such as normal distributions), it chooses a
        total width to roughly minimize total error, considering both intra-bin
        error and error due to the excluded tails.
    log-uniform : str
        Divides the logarithm of the distribution into bins of equal width. For
        example, if you generated a NumericDistribution from a log-normal
        distribution with log-uniform bin sizing, and then took the log of each
        bin, you'd get a normal distribution with uniform bin sizing.
    ev : str
        Divides the distribution into bins such that each bin has equal
        contribution to expected value (see
        :func:`squigglepy.distributions.IntegrableEVDistribution.contribution_to_ev`).
        It works by first computing the bin edge values that equally divide up
        contribution to expected value, then computing the probability mass of
        each bin, then setting the value of each bin such that value * mass =
        contribution to expected value (rather than, say, setting value to the
        average value of the two edges).
    mass : str
        Divides the distribution into bins such that each bin has equal
        probability mass. This method is generally not recommended
        because it puts too much probability mass near the center of the
        distribution, where precision is the least useful.
    fat-hybrid : str
        A hybrid method designed for fat-tailed distributions. Uses mass bin
        sizing close to the center and log-uniform bin siding on the right
        tail. Empirically, this combination provides the best balance for the
        accuracy of fat-tailed distributions at the center and at the tails.
    bin-count : str
        Shorten a vector of bins by merging every (1/len) bins together. Can
        only be used for resizing an existing NumericDistribution, not for
        initializing a new one.

    Interpretation for two-sided distributions
    ------------------------------------------
    The interpretation of the EV bin-sizing method is slightly non-obvious
    for two-sided distributions because we must decide how to interpret bins
    with negative expected value.

    The EV method arranges values into bins such that:
        * The negative side has the correct negative contribution to EV and the
          positive side has the correct positive contribution to EV.
        * Every negative bin has equal contribution to EV and every positive bin
          has equal contribution to EV.
        * If a side has nonzero probability mass, then it has at least one bin,
          regardless of how small its probability mass.
        * The number of negative and positive bins are chosen such that the
          absolute contribution to EV for negative bins is as close as possible
          to the absolute contribution to EV for positive bins given the above
          constraints.

    This binning method means that the distribution EV is exactly preserved
    and there is no bin that contains the value zero. However, the positive
    and negative bins do not necessarily have equal contribution to EV, and
    the magnitude of the error is at most 1 / num_bins / 2.

    """

    uniform = "uniform"
    log_uniform = "log-uniform"
    ev = "ev"
    mass = "mass"
    fat_hybrid = "fat-hybrid"
    bin_count = "bin-count"


def _support_for_bin_sizing(dist, bin_sizing, num_bins):
    """Return where to set the bounds for a bin sizing method with fixed
    bounds, or None if the given dist/bin sizing does not require finite
    bounds.
    """
    # For norm/lognorm, wider domain increases error within each bin, and
    # narrower domain increases error at the tails. Inter-bin error is
    # proportional to width^3 / num_bins^2 and tail error is proportional to
    # something like exp(-width^2). Setting width using the formula below
    # balances these two sources of error. These scale coefficients means that
    # a histogram with 100 bins will cover 6.6 standard deviations in each
    # direction which leaves off less than 1e-10 of the probability mass.
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
        scale = 7
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
    LognormalDistribution: BinSizing.fat_hybrid,
    NormalDistribution: BinSizing.uniform,
    ParetoDistribution: BinSizing.ev,
    UniformDistribution: BinSizing.uniform,
}

DEFAULT_NUM_BINS = {
    BetaDistribution: 50,
    ChiSquareDistribution: 200,
    ExponentialDistribution: 200,
    GammaDistribution: 200,
    LognormalDistribution: 200,
    MixtureDistribution: 200,
    NormalDistribution: 200,
    ParetoDistribution: 200,
    UniformDistribution: 50,
}

CACHED_NORM_CDFS = {}
CACHED_LOGNORM_CDFS = {}
CACHED_LOGNORM_PPFS = {}


def cached_norm_cdfs(num_bins):
    if num_bins in CACHED_NORM_CDFS:
        return CACHED_NORM_CDFS[num_bins]
    support = _support_for_bin_sizing(
        NormalDistribution(mean=0, sd=1), BinSizing.uniform, num_bins
    )
    values = np.linspace(support[0], support[1], num_bins + 1)
    cdfs = stats.norm.cdf(values)
    CACHED_NORM_CDFS[num_bins] = cdfs
    return cdfs


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
    return np.where(x == 0, -np.inf, np.log(x))


class BaseNumericDistribution(ABC):
    def quantile(self, q):
        """Estimate the value of the distribution at quantile ``q`` by
        interpolating between known values.

        Warning: This function is not very accurate in certain cases. Namely,
        fat-tailed distributions put much of their probability mass in the
        smallest bins because the difference between (say) the 10th percentile
        and the 20th percentile is inconsequential for most purposes. For these
        distributions, small quantiles will be very inaccurate, in exchange for
        greater accuracy in quantiles close to 1--this function can often
        reliably distinguish between (say) the 99.8th and 99.9th percentiles
        for fat-tailed distributions.

        The accuracy at different quantiles depends on the bin sizing method
        used. :ref:``BinSizing.mass`` will produce bins that are evenly spaced
        across quantiles. ``BinSizing.ev`` and ``BinSizing.log_uniform`` for
        fat-tailed distributions will lose accuracy at lower quantiles in
        exchange for greater accuracy on the right tail.

        Parameters
        ----------
        q : number or array_like
            The quantile or quantiles for which to determine the value(s).

        Return
        ------
        quantiles: number or array-like
            The estimated value at the given quantile(s).

        """
        return self.ppf(q)

    @abstractmethod
    def ppf(self, q):
        """Percent point function/inverse CD. An alias for :ref:``quantile``."""
        ...

    def percentile(self, p):
        """Estimate the value of the distribution at percentile ``p``. See
        :ref:``quantile`` for notes on this function's accuracy.
        """
        return np.squeeze(self.ppf(np.asarray(p) / 100))

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
    :ref:``BinSizing``) can easily track the probability mass of large values.

    Implementation Details
    ======================

    On setting values within bins
    -----------------------------
    Whenever possible, NumericDistribution assigns the value of each bin as the
    average value between the two edges (weighted by mass). You can think of
    this as the result you'd get if you generated infinitely many Monte Carlo
    samples and grouped them into bins, setting the value of each bin as the
    average of the samples. You might call this the expected value (EV) method,
    in contrast to two methods described below.

    The EV method guarantees that, whenever the histogram width covers the full
    support of the distribution, the histogram's expected value exactly equals
    the expected value of the true distribution (modulo floating point rounding
    errors).

    There are some other methods we could use, which are generally worse:

    1. Set the value of each bin to the average of the two edges (the
    "trapezoid rule"). The purpose of using the trapezoid rule is that we don't
    know the probability mass within a bin (perhaps the CDF is too hard to
    evaluate) so we have to estimate it. But whenever we *do* know the CDF, we
    can calculate the probability mass exactly, so we don't need to use the
    trapezoid rule.

    2. Set the value of each bin to the center of the probability mass (the
    "mass method"). This is equivalent to generating infinitely many Monte
    Carlo samples and grouping them into bins, setting the value of each bin as
    the **median** of the samples. This approach does not particularly help us
    because we don't care about the median of every bin. We might care about
    the median of the distribution, but we can calculate that near-exactly
    regardless of what value-setting method we use by looking at the value in
    the bin where the probability mass crosses 0.5. And the mass method will
    systematically underestimate (the absolute value of) EV because the
    definition of expected value places larger weight on larger (absolute)
    values, and the mass method does not.

    Although the EV method perfectly measures the expected value of a
    distribution, it systematically underestimates the variance. To see this,
    consider that it is possible to define the variance of a random variable X
    as

    .. math::
        E[X^2] - E[X]^2

    The EV method correctly estimates ``E[X]``, so it also correctly estimates
    ``E[X]^2``. However, it systematically underestimates E[X^2] because E[X^2]
    places more weight on larger values. But an alternative method that
    accurately estimated variance would necessarily *over*estimate E[X].

    """

    def __init__(
        self,
        values: np.ndarray,
        masses: np.ndarray,
        zero_bin_index: int,
        neg_ev_contribution: float,
        pos_ev_contribution: float,
        exact_mean: Optional[float] = None,
        exact_sd: Optional[float] = None,
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
        bin_sizing : :ref:``BinSizing``
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
        self._version = __version__
        self.values = values
        self.masses = masses
        self.num_bins = len(values)
        self.zero_bin_index = zero_bin_index
        self.neg_ev_contribution = neg_ev_contribution
        self.pos_ev_contribution = pos_ev_contribution
        self.exact_mean = exact_mean
        self.exact_sd = exact_sd

        # These are computed lazily
        self.interpolate_cdf = None
        self.interpolate_ppf = None

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

        Return
        ------
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

        Return
        ------
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
        dist: BaseDistribution | BaseNumericDistribution,
        num_bins: Optional[int] = None,
        bin_sizing: Optional[str] = None,
        warn: bool = True,
    ):
        """Create a probability mass histogram from the given distribution.

        Parameters
        ----------
        dist : BaseDistribution | BaseNumericDistribution
            A distribution from which to generate numeric values. If the
            provided value is a :ref:``BaseNumericDistribution``, simply return
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
            :ref:``DEFAULT_BIN_SIZING`` based on the distribution type of
            ``dist``. It is recommended to use the default bin sizing method
            most of the time. See
            :ref:`squigglepy.numeric_distribution.BinSizing` for a list of
            valid options and explanations of their behavior. warn :
            Optional[bool] (default = True) If True, raise warnings about bins
            with zero mass.

        Return
        ------
        result : NumericDistribution | ZeroNumericDistribution
            The generated numeric distribution that represents ``dist``.

        """

        # --------------------------------------------------
        # Handle special distributions (Mixture and Complex)
        # --------------------------------------------------

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
            return dist.fn(left, right).clip(dist.lclip, dist.rclip)

        # ------------
        # Basic checks
        # ------------

        if isinstance(dist, BaseNumericDistribution):
            return dist

        if type(dist) not in DEFAULT_BIN_SIZING:
            raise ValueError(f"Unsupported distribution type: {type(dist)}")

        num_bins = num_bins or DEFAULT_NUM_BINS[type(dist)]
        bin_sizing = BinSizing(bin_sizing or DEFAULT_BIN_SIZING[type(dist)])

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
            ParetoDistribution: (0, np.inf),
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
                    exact_sd = np.sqrt(
                        dist.shape / ((dist.shape - 1) ** 2 * (dist.shape - 2))
                    )
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
            dist.contribution_to_ev(0, normalized=False)
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
        # still give it one bin.
        num_neg_bins, num_pos_bins = cls._num_bins_per_side(
            num_bins, neg_prop, pos_prop, allowance=0
        )
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
            np.array(values),
            np.array(masses),
            zero_bin_index=num_neg_bins,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            exact_mean=exact_mean,
            exact_sd=exact_sd,
        )

    @classmethod
    def mixture(
        cls, dists, weights, lclip=None, rclip=None, num_bins=None, bin_sizing=None, warn=True
    ):
        if num_bins is None:
            mixture_num_bins = DEFAULT_NUM_BINS[MixtureDistribution]
        # This replicates how MixtureDistribution handles lclip/rclip: it
        # clips the sub-distributions based on their own lclip/rclip, then
        # takes the mixture sample, then clips the mixture sample based on
        # the mixture lclip/rclip.
        dists = [d for d in dists]  # create new list to avoid mutating

        # Convert any Squigglepy dists into NumericDistributions
        for i in range(len(dists)):
            if isinstance(dists[i], BaseDistribution):
                dists[i] = NumericDistribution.from_distribution(dists[i], num_bins, bin_sizing)
            elif not isinstance(dists[i], BaseNumericDistribution):
                raise ValueError(f"Cannot create a mixture with type {type(dists[i])}")

        value_vectors = [d.values for d in dists]
        weighted_mass_vectors = [d.masses * w for d, w in zip(dists, weights)]
        extended_values = np.concatenate(value_vectors)
        extended_masses = np.concatenate(weighted_mass_vectors)

        sorted_indexes = np.argsort(extended_values, kind="mergesort")
        extended_values = extended_values[sorted_indexes]
        extended_masses = extended_masses[sorted_indexes]
        zero_index = np.searchsorted(extended_values, 0)

        neg_ev_contribution = -np.sum(extended_masses[:zero_index] * extended_values[:zero_index])
        pos_ev_contribution = np.sum(extended_masses[zero_index:] * extended_values[zero_index:])

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

        return mixture.clip(lclip, rclip)

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
        if self.exact_sd is not None:
            return self.exact_sd
        return self.histogram_sd()

    def _init_interpolate_cdf(self):
        if self.interpolate_cdf is None:
            # Subtracting 0.5 * masses because eg the first out of 100 values
            # represents the 0.5th percentile, not the 1st percentile
            self._cum_mass = np.cumsum(self.masses) - 0.5 * self.masses
            self.interpolate_cdf = PchipInterpolator(self.values, self._cum_mass, extrapolate=True)

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
        """Return a new distribution clipped to the given bounds. Does not
        modify the current distribution.

        It is strongly recommended that, whenever possible, you construct a
        ``NumericDistribution`` by supplying a ``Distribution`` that has
        lclip/rclip defined on it, rather than clipping after the fact.
        Clipping after the fact can greatly decrease accuracy.

        Parameters
        ----------
        lclip : Optional[float]
            The new lower bound of the distribution, or None if the lower bound
            should not change.
        rclip : Optional[float]
            The new upper bound of the distribution, or None if the upper bound
            should not change.

        Return
        ------
        clipped : NumericDistribution
            A new distribution clipped to the given bounds.

        """
        if lclip is None and rclip is None:
            return NumericDistribution(
                self.values,
                self.masses,
                self.zero_bin_index,
                self.neg_ev_contribution,
                self.pos_ev_contribution,
                self.exact_mean,
                self.exact_sd,
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
        )

    def sample(self, n):
        """Generate ``n`` random samples from the distribution."""
        # TODO: Do interpolation instead of returning the same values repeatedly.
        # Could maybe simplify by calling self.quantile(np.random.uniform(size=n))
        return np.random.choice(self.values, size=n, p=self.masses)

    @classmethod
    def _contribution_to_ev(
        cls, values: np.ndarray, masses: np.ndarray, x: np.ndarray | float, normalized=True
    ):
        if isinstance(x, np.ndarray) and x.ndim == 0:
            x = x.item()
        elif isinstance(x, np.ndarray):
            return np.array([cls._contribution_to_ev(values, masses, xi, normalized) for xi in x])

        contributions = np.squeeze(np.sum(masses * abs(values) * (values <= x)))
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
        fractions_of_ev = np.cumsum(masses * abs(values)) / mean
        epsilon = 1e-10  # to avoid floating point rounding issues
        index = np.searchsorted(fractions_of_ev, fraction - epsilon)
        return values[index]

    def contribution_to_ev(self, x: np.ndarray | float):
        return self._contribution_to_ev(self.values, self.masses, x)

    def inv_contribution_to_ev(self, fraction: np.ndarray | float):
        """Return the value such that ``fraction`` of the contribution to
        expected value lies to the left of that value.
        """
        return self._inv_contribution_to_ev(self.values, self.masses, fraction)

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

    @classmethod
    def _num_bins_per_side(cls, num_bins, neg_contribution, pos_contribution, allowance=0):
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

        Return
        ------
        (num_neg_bins, num_pos_bins) : (int, int)
            Number of bins assigned to the negative/positive side of the
            distribution.

        """
        min_prop_cutoff = allowance * 1 / num_bins / 2
        total_contribution = neg_contribution + pos_contribution
        num_neg_bins = int(np.round(num_bins * neg_contribution / total_contribution))
        num_pos_bins = num_bins - num_neg_bins

        if neg_contribution / total_contribution > min_prop_cutoff:
            num_neg_bins = max(1, num_neg_bins)
            num_pos_bins = num_bins - num_neg_bins
        else:
            num_neg_bins = 0
            num_pos_bins = num_bins

        if pos_contribution / total_contribution > min_prop_cutoff:
            num_pos_bins = max(1, num_pos_bins)
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

        Return
        -------
        values : np.ndarray
            The values of the bins.
        masses : np.ndarray
            The probability masses of the bins.

        """
        if num_bins == 0:
            return (np.array([]), np.array([]))

        if bin_sizing == BinSizing.bin_count:
            items_per_bin = len(extended_values) // num_bins
            if len(extended_masses) % num_bins > 0:
                # Increase the number of bins such that we can fit
                # extended_masses into them at items_per_bin each
                num_bins = int(np.ceil(len(extended_masses) / items_per_bin))

                # Fill any empty space with zeros
                extra_zeros = np.zeros(num_bins * items_per_bin - len(extended_masses))
                extended_values = np.concatenate((extra_zeros, extended_values))
                extended_masses = np.concatenate((extra_zeros, extended_masses))
            boundary_indexes = np.arange(0, num_bins + 1) * items_per_bin
        elif bin_sizing == BinSizing.ev:
            # TODO: I think this is wrong, you have to sort/partition the values first
            extended_evs = extended_values * extended_masses
            cumulative_evs = np.concatenate(([0], np.cumsum(extended_evs)))
            boundary_values = np.linspace(0, cumulative_evs[-1], num_bins + 1)
            boundary_indexes = np.searchsorted(cumulative_evs, boundary_values, side="right") - 1
            # remove bin boundaries where boundary[i] == boundary[i+1]
            old_boundary_bins = boundary_indexes
            boundary_indexes = np.concatenate(
                (boundary_indexes[:-1][np.diff(boundary_indexes) > 0], [boundary_indexes[-1]])
            )
        elif bin_sizing == BinSizing.log_uniform:
            # ``bin_count`` puts too much mass in the bins on the left and
            # right tails, but it's still more accurate than log-uniform
            # sizing, I don't know why.
            assert num_bins % 2 == 0
            assert len(extended_values) == num_bins**2

            # method 1: size bins in a pyramid shape. this preserves
            # log-uniform bin sizing but it makes the bin widths unnecessarily
            # large>
            # ascending_indexes = 2 * np.array(range(num_bins // 2 + 1))**2
            # descending_indexes = np.flip(num_bins**2 - ascending_indexes)
            # boundary_indexes = np.concatenate((ascending_indexes, descending_indexes[1:]))

            # method 2: size bins by going out a fixed number of log-standard
            # deviations in each direction
            log_mean = np.average(np.log(extended_values), weights=extended_masses)
            log_sd = np.sqrt(np.average((np.log(extended_values) - log_mean)**2, weights=extended_masses))
            log_left_bound = log_mean - 6.5 * log_sd
            log_right_bound = log_mean + 6.5 * log_sd
            log_boundary_values = np.linspace(log_left_bound, log_right_bound, num_bins + 1)
            boundary_values = np.exp(log_boundary_values)

            sorted_indexes = extended_values.argsort(kind="mergesort")
            extended_values = extended_values[sorted_indexes]
            extended_masses = extended_masses[sorted_indexes]
            is_sorted = True

            boundary_indexes = np.searchsorted(extended_values, boundary_values)
        else:
            raise ValueError(f"resize_pos_bins: Unsupported bin sizing method: {bin_sizing}")

        if not is_sorted:
            # Partition such that the values in one bin are all less than
            # or equal to the values in the next bin. Values within bins
            # don't need to be sorted, and partitioning is ~10% faster than
            # timsort.
            partitioned_indexes = extended_values.argpartition(boundary_indexes[1:-1])
            extended_values = extended_values[partitioned_indexes]
            extended_masses = extended_masses[partitioned_indexes]

        if bin_sizing == BinSizing.bin_count:
            # Take advantage of the fact that all bins contain the same number
            # of elements.
            extended_evs = extended_values * extended_masses
            masses = extended_masses.reshape((num_bins, -1)).sum(axis=1)
            bin_evs = extended_evs.reshape((num_bins, -1)).sum(axis=1)
        elif bin_sizing == BinSizing.ev:
            # Calculate the expected value of each bin
            bin_evs = np.diff(cumulative_evs[boundary_indexes])
            cumulative_masses = np.concatenate(([0], np.cumsum(extended_masses)))
            masses = np.diff(cumulative_masses[boundary_indexes])
        elif bin_sizing == BinSizing.log_uniform:
            # Compute sums one at a time instead of using ``cumsum`` because
            # ``cumsum`` produces non-trivial rounding errors.
            extended_evs = extended_values * extended_masses
            bin_evs = np.array([np.sum(extended_evs[i:j]) for (i, j) in zip(boundary_indexes[:-1], boundary_indexes[1:])])
            masses = np.array([np.sum(extended_masses[i:j]) for (i, j) in zip(boundary_indexes[:-1], boundary_indexes[1:])])
        else:
            raise ValueError(f"resize_pos_bins: Unsupported bin sizing method: {bin_sizing}")

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

        Return
        -------
        values : np.ndarray
            The values of the bins.
        masses : np.ndarray
            The probability masses of the bins.

        """
        if bin_sizing == BinSizing.bin_count:
            num_neg_bins, num_pos_bins = cls._num_bins_per_side(
                num_bins, len(extended_neg_masses), len(extended_pos_masses)
            )
        elif bin_sizing == BinSizing.ev:
            num_neg_bins, num_pos_bins = cls._num_bins_per_side(
                num_bins, neg_ev_contribution, pos_ev_contribution
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
        )

    def __eq__(x, y):
        return x.values == y.values and x.masses == y.masses

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
        sum_mean = x.mean() + y.mean()
        # This `max` is a hack to deal with a problem where, when mean is
        # negative and almost all contribution is on the negative side,
        # neg_ev_contribution can sometimes be slightly less than abs(mean),
        # apparently due to rounding issues, which makes pos_ev_contribution
        # negative.
        pos_ev_contribution = max(0, sum_mean + neg_ev_contribution)

        res = cls._resize_bins(
            extended_neg_values=extended_values[:zero_index],
            extended_neg_masses=extended_masses[:zero_index],
            extended_pos_values=extended_values[zero_index:],
            extended_pos_masses=extended_masses[zero_index:],
            num_bins=num_bins,
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
            is_sorted=is_sorted,
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
            exact_mean=-self.exact_mean,
            exact_sd=self.exact_sd,
        )

    def __mul__(x, y):
        if isinstance(y, Real):
            return x.scale_by(y)
        elif isinstance(y, ZeroNumericDistribution):
            return y.__rmul__(x)
        elif not isinstance(y, NumericDistribution):
            raise TypeError(f"Cannot add types {type(x)} and {type(y)}")

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

    def scale_by_probability(self, p):
        return ZeroNumericDistribution(self, 1 - p)

    def condition_on_success(
        self,
        event: BaseNumericDistribution,
        failure_outcome: Optional[Union[BaseNumericDistribution, float]] = 0,
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
        # TODO: generalize this to accept point probabilities
        p_success = event.mean()
        return ZeroNumericDistribution(self, 1 - p_success)

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
        # See :ref:``exp`` for some discussion of accuracy. For ``log`` on a
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
    A :ref:``NumericDistribution`` with a point mass at zero.
    """

    def __init__(self, dist: NumericDistribution, zero_mass: float):
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

    def histogram_mean(self):
        return self.dist.histogram_mean() * self.nonzero_mass

    def mean(self):
        return self.dist.mean() * self.nonzero_mass

    def histogram_sd(self):
        mean = self.mean()
        nonzero_variance = (
            np.sum(self.dist.masses * (self.dist.values - mean) ** 2) * self.nonzero_mass
        )
        zero_variance = self.zero_mass * mean**2
        variance = nonzero_variance + zero_variance
        return np.sqrt(variance)

    def sd(self):
        if self.exact_sd is not None:
            return self.exact_sd
        return self.histogram_sd()

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
        # TODO: test this
        warnings.warn("ZeroNumericDistribution.shift_by is untested, use at your own risk")
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

    def exp(self):
        # TODO: exponentiate the wrapped dist, then do something like shift_by
        # to insert a 1 into the bins
        return NotImplementedError

    def log(self):
        raise ValueError("Cannot take the log of a distribution with non-positive values")

    def __mul__(x, y):
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
    dist: BaseDistribution,
    num_bins: Optional[int] = None,
    bin_sizing: Optional[str] = None,
    warn: bool = True,
):
    """Create a probability mass histogram from the given distribution.

    Parameters
    ----------
    dist : BaseDistribution | BaseNumericDistribution
        A distribution from which to generate numeric values. If the
        provided value is a :ref:``BaseNumericDistribution``, simply return
        it.
    num_bins : Optional[int] (default = ref:``DEFAULT_NUM_BINS``)
        The number of bins for the numeric distribution to use. The time to
        construct a NumericDistribution is linear with ``num_bins``, and
        the time to run a binary operation on two distributions with the
        same number of bins is approximately quadratic with ``num_bins``.
        100 bins provides a good balance between accuracy and speed.
    bin_sizing : Optional[str]
        The bin sizing method to use, which affects the accuracy of the
        bins. If none is given, a default will be chosen from
        :ref:``DEFAULT_BIN_SIZING`` based on the distribution type of
        ``dist``. It is recommended to use the default bin sizing method
        most of the time. See
        :ref:`squigglepy.numeric_distribution.BinSizing` for a list of
        valid options and explanations of their behavior. warn :
        Optional[bool] (default = True) If True, raise warnings about bins
        with zero mass.

    Return
    ------
    result : NumericDistribution | ZeroNumericDistribution
        The generated numeric distribution that represents ``dist``.
    """
    return NumericDistribution.from_distribution(dist, num_bins, bin_sizing, warn)
