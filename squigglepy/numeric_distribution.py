from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
import numpy as np
from scipy import optimize, stats
from typing import Literal, Optional, Tuple
import warnings

from .distributions import (
    BaseDistribution,
    LognormalDistribution,
    MixtureDistribution,
    NormalDistribution,
    UniformDistribution,
)


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
        probability mass. This maximizes the accuracy of uniformly-distributed
        quantiles; for example, with 100 bins, it ensures that every bin value
        falls between two percentiles. This method is generally not recommended
        because it puts too much probability mass near the center of the
        distribution, where precision is the least useful.

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


DEFAULT_BIN_SIZING = {
    NormalDistribution: BinSizing.uniform,
    LognormalDistribution: BinSizing.log_uniform,
    UniformDistribution: BinSizing.uniform,
}

DEFAULT_NUM_BINS = 100


def _narrow_support(
    support: Tuple[float, float], new_support: Tuple[Optional[float], Optional[float]]
):
    """Narrow the support to the intersection of ``support`` and ``new_support``."""
    if new_support[0] is not None:
        support = (max(support[0], new_support[0]), support[1])
    if new_support[1] is not None:
        support = (support[0], min(support[1], new_support[1]))
    return support


class NumericDistribution:
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
        self.neg_ev_contribution = neg_ev_contribution
        self.pos_ev_contribution = pos_ev_contribution
        self.exact_mean = exact_mean
        self.exact_sd = exact_sd

    @classmethod
    def _construct_bins(
        cls,
        num_bins,
        support,
        dist,
        cdf,
        ppf,
        bin_sizing,
        warn,
    ):
        """Construct a list of bin masses and values. Helper function for
        :func:`from_distribution`, do not call this directly."""
        if num_bins <= 0:
            return (np.array([]), np.array([]))

        if bin_sizing == BinSizing.uniform:
            edge_values = np.linspace(support[0], support[1], num_bins + 1)

        elif bin_sizing == BinSizing.log_uniform:
            log_support = (np.log(support[0]), np.log(support[1]))
            log_edge_values = np.linspace(log_support[0], log_support[1], num_bins + 1)
            edge_values = np.exp(log_edge_values)

        elif bin_sizing == BinSizing.ev:
            # Don't call get_edge_value on the left and right edges because it's
            # undefined for 0 and 1
            left_prop = dist.contribution_to_ev(support[0])
            right_prop = dist.contribution_to_ev(support[1])
            edge_values = np.concatenate(
                (
                    [support[0]],
                    np.atleast_1d(
                        dist.inv_contribution_to_ev(np.linspace(left_prop, right_prop, num_bins + 1)[1:-1])
                    )
                    if num_bins > 1
                    else [],
                    [support[1]],
                )
            )

        elif bin_sizing == BinSizing.mass:
            left_cdf = cdf(support[0])
            right_cdf = cdf(support[1])
            edge_cdfs = np.linspace(left_cdf, right_cdf, num_bins + 1)
            edge_values = ppf(edge_cdfs)

        elif bin_sizing == BinSizing.fat_hybrid:
            # Use a combination of ev and log-uniform
            scale = 1 + np.log(num_bins)
            lu_support = _narrow_support((np.log(support[0]), np.log(support[1])), (dist.norm_mean - scale * dist.norm_sd, dist.norm_mean + scale * dist.norm_sd))
            lu_edge_values = np.linspace(lu_support[0], lu_support[1], num_bins + 1)[:-1]
            lu_edge_values = np.exp(lu_edge_values)
            ev_left_prop = dist.contribution_to_ev(support[0])
            ev_right_prop = dist.contribution_to_ev(support[1])
            ev_edge_values = np.concatenate(
                (
                    [support[0]],
                    np.atleast_1d(
                        dist.inv_contribution_to_ev(np.linspace(ev_left_prop, ev_right_prop, num_bins + 1)[1:-1])
                    )
                    if num_bins > 1
                    else [],
                )
            )
            edge_values = np.where(lu_edge_values > ev_edge_values, lu_edge_values, ev_edge_values)
            edge_values = np.concatenate((edge_values, [support[1]]))

        else:
            raise ValueError(f"Unsupported bin sizing method: {bin_sizing}")

        # Avoid re-calculating CDFs if we can because it's really slow.
        if bin_sizing != BinSizing.mass:
            edge_cdfs = cdf(edge_values)

        masses = np.diff(edge_cdfs)

        # Set the value for each bin equal to its average value. This is
        # equivalent to generating infinitely many Monte Carlo samples and
        # grouping them into bins, and it has the nice property that the
        # expected value of the histogram will exactly equal the expected value
        # of the distribution.
        edge_ev_contributions = dist.contribution_to_ev(edge_values, normalized=False)
        bin_ev_contributions = np.diff(edge_ev_contributions)

        # For sufficiently large edge values, CDF rounds to 1 which makes the
        # mass 0. Values can also be 0 due to floating point rounding if
        # support is very small. Remove any 0s.
        if any(masses == 0) or any(bin_ev_contributions == 0):
            mass_zeros = len([x for x in masses if x == 0])
            ev_zeros = len([x for x in bin_ev_contributions if x == 0])
            nonzero_indexes = [
                i for i in range(len(masses)) if masses[i] != 0 and bin_ev_contributions[i] != 0
            ]
            bin_ev_contributions = bin_ev_contributions[nonzero_indexes]
            masses = masses[nonzero_indexes]
            if mass_zeros == 1:
                mass_zeros_message = f"1 value >= {edge_values[-1]} had a CDF of 1"
            else:
                mass_zeros_message = (
                    f"{mass_zeros} values >= {edge_values[-mass_zeros]} had CDFs of 1"
                )
            if ev_zeros == 1:
                ev_zeros_message = (
                    f"1 bin had zero expected value, most likely because it was too small"
                )
            else:
                ev_zeros_message = f"{ev_zeros} bins had zero expected value, most likely because they were too small"
            if mass_zeros > 0 and ev_zeros > 0:
                joint_message = f"{mass_zeros_message}; and {ev_zeros_message}"
            elif mass_zeros > 0:
                joint_message = mass_zeros_message
            else:
                joint_message = ev_zeros_message
            if warn:
                warnings.warn(
                    f"When constructing NumericDistribution, {joint_message}.",
                    RuntimeWarning,
                )

        values = bin_ev_contributions / masses
        return (masses, values)

    @classmethod
    def from_distribution(
        cls,
        dist: BaseDistribution,
        num_bins: Optional[int] = None,
        bin_sizing: Optional[str] = None,
        warn: bool = True,
    ):
        """Create a probability mass histogram from the given distribution.

        Parameters
        ----------
        dist : BaseDistribution
            A distribution from which to generate numeric values.
        num_bins : Optional[int] (default = 100)
            The number of bins for the numeric distribution to use. The time to
            construct a NumericDistribution is linear with ``num_bins``, and
            the time to run a binary operation on two distributions with the
            same number of bins is approximately quadratic with ``num_bins``.
            100 bins provides a good balance between accuracy and speed.
        bin_sizing : Optional[str]
            The bin sizing method to use. If none is given, a default will be
            chosen from :ref:``DEFAULT_BIN_SIZING`` based on the distribution
            type of ``dist``. It is recommended to use the default bin sizing
            method most of the time. See
            :ref:`squigglepy.numeric_distribution.BinSizing` for a list of
            valid options and explanations of their behavior.
        warn : Optional[bool] (default = True)
            If True, raise warnings about bins with zero mass.

        """
        if num_bins is None:
            num_bins = DEFAULT_NUM_BINS

        if isinstance(dist, MixtureDistribution):
            # This replicates how MixtureDistribution handles lclip/rclip: it
            # clips the sub-distributions based on their own lclip/rclip, then
            # takes the mixture sample, then clips the mixture sample based on
            # the mixture lclip/rclip.
            sub_dists = [cls.from_distribution(d, num_bins, bin_sizing, warn) for d in dist.dists]
            mixture = reduce(
                lambda acc, d: acc + d, [w * d for w, d in zip(dist.weights, sub_dists)]
            )
            return mixture.clip(dist.lclip, dist.rclip)
        if type(dist) not in DEFAULT_BIN_SIZING:
            raise ValueError(f"Unsupported distribution type: {type(dist)}")

        # -------------------------------------------------------------------
        # Set up required parameters based on dist type and bin sizing method
        # -------------------------------------------------------------------

        bin_sizing = BinSizing(bin_sizing or DEFAULT_BIN_SIZING[type(dist)])
        support = {
            # These are the widest possible supports, but they maybe narrowed
            # later by lclip/rclip or by some bin sizing methods
            LognormalDistribution: (0, np.inf),
            NormalDistribution: (-np.inf, np.inf),
            UniformDistribution: (dist.x, dist.y),
        }[type(dist)]
        ppf = {
            LognormalDistribution: lambda p: stats.lognorm.ppf(
                p, dist.norm_sd, scale=np.exp(dist.norm_mean)
            ),
            NormalDistribution: lambda p: stats.norm.ppf(p, loc=dist.mean, scale=dist.sd),
            UniformDistribution: lambda p: stats.uniform.ppf(p, loc=dist.x, scale=dist.y - dist.x),
        }[type(dist)]
        cdf = {
            LognormalDistribution: lambda x: stats.lognorm.cdf(
                x, dist.norm_sd, scale=np.exp(dist.norm_mean)
            ),
            NormalDistribution: lambda x: stats.norm.cdf(x, loc=dist.mean, scale=dist.sd),
            UniformDistribution: lambda x: stats.uniform.cdf(x, loc=dist.x, scale=dist.y - dist.x),
        }[type(dist)]

        # -----------
        # Set support
        # -----------

        dist_bin_sizing_supported = False
        new_support = None
        if bin_sizing == BinSizing.uniform:
            if isinstance(dist, LognormalDistribution):
                # Uniform bin sizing is not gonna be very accurate for a lognormal
                # distribution no matter how you set the bounds.
                new_support = (0, np.exp(dist.norm_mean + 7 * dist.norm_sd))
            elif isinstance(dist, NormalDistribution):
                # Wider domain increases error within each bin, and narrower
                # domain increases error at the tails. Inter-bin error is
                # proportional to width^3 / num_bins^2 and tail error is
                # proportional to something like exp(-width^2). Setting width
                # proportional to log(num_bins) balances these two sources of
                # error. These scale coefficients means that a histogram with
                # 100 bins will cover 7.1 standard deviations in each direction
                # which leaves off less than 1e-12 of the probability mass.
                scale = 2.5 + np.log(num_bins)
                new_support = (
                    dist.mean - dist.sd * scale,
                    dist.mean + dist.sd * scale,
                )
            elif isinstance(dist, UniformDistribution):
                new_support = support

        elif bin_sizing == BinSizing.log_uniform:
            if isinstance(dist, LognormalDistribution):
                scale = 2 + np.log(num_bins)
                new_support = (
                    np.exp(dist.norm_mean - dist.norm_sd * scale),
                    np.exp(dist.norm_mean + dist.norm_sd * scale),
                )

        elif bin_sizing == BinSizing.ev:
            dist_bin_sizing_supported = True

        elif bin_sizing == BinSizing.mass:
            dist_bin_sizing_supported = True

        elif bin_sizing == BinSizing.fat_hybrid:
            if isinstance(dist, LognormalDistribution):
                # Set a left bound but not a right bound because the right tail
                # will use ev bin sizing
                scale = 1 + np.log(num_bins)
                new_support = (
                    np.exp(dist.norm_mean - dist.norm_sd * scale),
                    support[1],
                )

        if new_support is not None:
            support = _narrow_support(support, new_support)
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
            if isinstance(dist, LognormalDistribution):
                exact_mean = dist.lognorm_mean
                exact_sd = dist.lognorm_sd
            elif isinstance(dist, NormalDistribution):
                exact_mean = dist.mean
                exact_sd = dist.sd
            elif isinstance(dist, UniformDistribution):
                exact_mean = (dist.x + dist.y) / 2
                exact_sd = np.sqrt(1 / 12) * (dist.y - dist.x)
        else:
            if isinstance(dist, LognormalDistribution):
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
            dist,
            cdf,
            ppf,
            bin_sizing,
            warn,
        )
        neg_values = -neg_values
        pos_masses, pos_values = cls._construct_bins(
            num_pos_bins,
            (max(0, support[0]), support[1]),
            dist,
            cdf,
            ppf,
            bin_sizing,
            warn,
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
        return self.exact_sd

    def cdf(self, x):
        """Estimate the proportion of the distribution that lies below ``x``.
        Uses linear interpolation between known values.
        """
        cum_mass = np.cumsum(self.masses) - 0.5 * self.masses
        return np.interp(x, self.values, cum_mass)

    def quantile(self, q):
        """Estimate the value of the distribution at quantile ``q`` using
        linear interpolation between known values.

        This function is not very accurate in certain cases:

        1. Fat-tailed distributions put much of their probability mass in the
        smallest bins because the difference between (say) the 10th percentile
        and the 20th percentile is inconsequential for most purposes. For these
        distributions, small quantiles will be very inaccurate, in exchange for
        greater accuracy in quantiles close to 1.

        2. For values with CDFs very close to 1, the values in bins may not be
        strictly ordered, in which case ``quantile`` may return an incorrect
        result. This will only happen if you request a quantile very close
        to 1 (such as 0.9999999).

        Parameters
        ----------
        q : number or array_like
            The quantile or quantiles for which to determine the value(s).

        Return
        ------
        quantiles: number or array-like
            The estimated value at the given quantile(s).
        """
        # Subtracting 0.5 * masses because eg the first out of 100 values
        # represents the 0.5th percentile, not the 1st percentile
        cum_mass = np.cumsum(self.masses) - 0.5 * self.masses
        return np.interp(q, cum_mass, self.values)

    def ppf(self, q):
        """An alias for :ref:``quantile``."""
        return self.quantile(q)

    def percentile(self, p):
        """Estimate the value of the distribution at percentile ``p``. See
        :ref:``quantile`` for notes on this function's accuracy.
        """
        return np.squeeze(self.quantile(np.asarray(p) / 100))

    def clip(self, lclip, rclip):
        """Return a new distribution clipped to the given bounds. Does not
        modify the current distribution.

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
    def _resize_bins(
        cls,
        extended_values,
        extended_masses,
        num_bins,
        ev,
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
        if num_bins == 0:
            return (np.array([]), np.array([]))
        items_per_bin = len(extended_values) // num_bins

        if len(extended_masses) % num_bins > 0:
            # Increase the number of bins such that we can fit
            # extended_masses into them at items_per_bin each
            num_bins = int(np.ceil(len(extended_masses) / items_per_bin))

            # Fill any empty space with zeros
            extra_zeros = np.zeros(num_bins * items_per_bin - len(extended_masses))
            extended_values = np.concatenate((extra_zeros, extended_values))
            extended_masses = np.concatenate((extra_zeros, extended_masses))

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
        bin_evs = extended_evs.reshape((num_bins, -1)).sum(axis=1)
        values = bin_evs / masses
        return (values, masses)

    def __eq__(x, y):
        return x.values == y.values and x.masses == y.masses

    def __ne__(x, y):
        return not (x == y)

    def __add__(x, y):
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

        # Set the number of bins per side to be approximately proportional to
        # the EV contribution, but make sure that if a side has nonzero EV
        # contribution, it gets at least one bin.
        num_neg_bins, num_pos_bins = cls._num_bins_per_side(
            num_bins, zero_index, len(extended_masses) - zero_index
        )
        if num_neg_bins == 0:
            neg_ev_contribution = 0
            pos_ev_contribution = sum_mean
        if num_pos_bins == 0:
            neg_ev_contribution = -sum_mean
            pos_ev_contribution = 0

        # Collect extended_values and extended_masses into the correct number
        # of bins. Make ``extended_values`` positive because ``_resize_bins``
        # can only operate on non-negative values. Making them positive means
        # they're now reverse-sorted, so reverse them.
        neg_values, neg_masses = cls._resize_bins(
            extended_values=np.flip(-extended_values[:zero_index]),
            extended_masses=np.flip(extended_masses[:zero_index]),
            num_bins=num_neg_bins,
            ev=neg_ev_contribution,
            is_sorted=is_sorted,
        )

        # ``_resize_bins`` returns positive values, so negate and reverse them.
        neg_values = np.flip(-neg_values)
        neg_masses = np.flip(neg_masses)

        # Collect extended_values and extended_masses into the correct number
        # of bins, for the positive values this time.
        pos_values, pos_masses = cls._resize_bins(
            extended_values=extended_values[zero_index:],
            extended_masses=extended_masses[zero_index:],
            num_bins=num_pos_bins,
            ev=pos_ev_contribution,
            is_sorted=is_sorted,
        )

        # Construct the resulting ``ProbabiltyMassHistogram`` object.
        values = np.concatenate((neg_values, pos_values))
        masses = np.concatenate((neg_masses, pos_masses))
        res = NumericDistribution(
            values=values,
            masses=masses,
            zero_bin_index=np.searchsorted(values, 0),
            neg_ev_contribution=neg_ev_contribution,
            pos_ev_contribution=pos_ev_contribution,
        )

        if x.exact_mean is not None and y.exact_mean is not None:
            res.exact_mean = x.exact_mean + y.exact_mean
        if x.exact_sd is not None and y.exact_sd is not None:
            res.exact_sd = np.sqrt(x.exact_sd**2 + y.exact_sd**2)
        return res

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

    def __sub__(x, y):
        return x + (-y)

    def __mul__(x, y):
        if isinstance(y, int) or isinstance(y, float):
            return x.scale_by(y)
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
        product_mean = x.mean() * y.mean()
        num_neg_bins, num_pos_bins = cls._num_bins_per_side(
            num_bins, len(extended_neg_masses), len(extended_pos_masses)
        )
        if num_neg_bins == 0:
            neg_ev_contribution = 0
            pos_ev_contribution = product_mean
        if num_pos_bins == 0:
            neg_ev_contribution = -product_mean
            pos_ev_contribution = 0

        # Collect extended_values and extended_masses into the correct number
        # of bins. Make ``extended_values`` positive because ``_resize_bins``
        # can only operate on non-negative values. Making them positive means
        # they're now reverse-sorted, so reverse them.
        neg_values, neg_masses = cls._resize_bins(
            -extended_neg_values,
            extended_neg_masses,
            num_neg_bins,
            ev=neg_ev_contribution,
        )

        # ``_resize_bins`` returns positive values, so negate and reverse them.
        neg_values = np.flip(-neg_values)
        neg_masses = np.flip(neg_masses)

        # Collect extended_values and extended_masses into the correct number
        # of bins, for the positive values this time.
        pos_values, pos_masses = cls._resize_bins(
            extended_pos_values,
            extended_pos_masses,
            num_pos_bins,
            ev=pos_ev_contribution,
        )

        # Construct the resulting ``ProbabiltyMassHistogram`` object.
        values = np.concatenate((neg_values, pos_values))
        masses = np.concatenate((neg_masses, pos_masses))
        zero_bin_index = len(neg_values)
        res = NumericDistribution(
            values=values,
            masses=masses,
            zero_bin_index=zero_bin_index,
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

    def __radd__(x, y):
        return x + y

    def __rsub__(x, y):
        return -x + y

    def __rmul__(x, y):
        return x * y

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

    def __truediv__(x, y):
        if isinstance(y, int) or isinstance(y, float):
            return x.scale_by(1 / y)
        return x * y.reciprocal()

    def __rtruediv__(x, y):
        return y * x.reciprocal()

    def __floordiv__(x, y):
        raise NotImplementedError

    def __rfloordiv__(x, y):
        raise NotImplementedError

    def __hash__(self):
        return hash(repr(self.values) + "," + repr(self.masses))


def numeric(dist, n=10000):
    # ``n`` is not directly meaningful, this is written as a drop-in
    # replacement for ``sq.sample``
    return NumericDistribution.from_distribution(dist, num_bins=max(100, int(np.ceil(np.sqrt(n)))))
