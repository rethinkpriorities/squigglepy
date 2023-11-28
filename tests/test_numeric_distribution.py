from functools import reduce
from hypothesis import assume, example, given, settings
import hypothesis.strategies as st
import numpy as np
from pytest import approx
from scipy import integrate, stats
import sys
import warnings

from ..squigglepy.distributions import (
    LognormalDistribution,
    NormalDistribution,
    UniformDistribution,
)
from ..squigglepy.numeric_distribution import NumericDistribution
from ..squigglepy import samplers

# There are a lot of functions testing various combinations of behaviors with
# no obvious way to order them. These functions are written basically in order
# of when I implemented them, with helper functions at the top, then
# construction and arithmetic operations, then non-arithmetical functions.


def relative_error(x, y):
    if x == 0 and y == 0:
        return 0
    if x == 0:
        return -1
    if y == 0:
        return np.inf
    return max(x / y, y / x) - 1


def print_accuracy_ratio(x, y, extra_message=None):
    ratio = relative_error(x, y)
    if extra_message is not None:
        extra_message += " "
    else:
        extra_message = ""
    direction_off = "small" if x < y else "large"
    if ratio > 1:
        print(f"{extra_message}Ratio: {direction_off} by a factor of {ratio:.1f}")
    else:
        print(f"{extra_message}Ratio: {direction_off} by {100 * ratio:.3f}%")


def get_mc_accuracy(exact_sd, num_samples, dists, operation):
    # Run multiple trials because NumericDistribution should usually beat MC,
    # but sometimes MC wins by luck. Even though NumericDistribution wins a
    # large percentage of the time, this test suite does a lot of runs, so the
    # chance of MC winning at least once is fairly high.
    mc_abs_error = []
    for i in range(10):
        mcs = [samplers.sample(dist, num_samples) for dist in dists]
        mc = reduce(operation, mcs)
        mc_abs_error.append(abs(np.std(mc) - exact_sd))

    mc_abs_error.sort()

    # Small numbers are good. A smaller index in mc_abs_error has a better
    # accuracy
    return mc_abs_error[-5]


def fix_uniform(a, b):
    """
    Check that a and b are ordered correctly and that they're not tiny enough
    to mess up floating point calculations.
    """
    if a > b:
        a, b = b, a
    assume(a != b)
    assume(((b - a) / (50 * (abs(a) + abs(b)))) ** 2 > sys.float_info.epsilon)
    assume(a == 0 or abs(a) > sys.float_info.epsilon)
    assume(b == 0 or abs(b) > sys.float_info.epsilon)
    return a, b


@given(
    norm_mean1=st.floats(min_value=-1e5, max_value=1e5),
    norm_mean2=st.floats(min_value=-1e5, max_value=1e5),
    norm_sd1=st.floats(min_value=0.1, max_value=100),
    norm_sd2=st.floats(min_value=0.001, max_value=1000),
)
@settings(max_examples=100)
def test_norm_sum_exact_summary_stats(norm_mean1, norm_mean2, norm_sd1, norm_sd2):
    """Test that the formulas for exact moments are implemented correctly."""
    dist1 = NormalDistribution(mean=norm_mean1, sd=norm_sd1)
    dist2 = NormalDistribution(mean=norm_mean2, sd=norm_sd2)
    hist1 = NumericDistribution.from_distribution(dist1)
    hist2 = NumericDistribution.from_distribution(dist2)
    hist_prod = hist1 + hist2
    assert hist_prod.exact_mean == approx(
        stats.norm.mean(norm_mean1 + norm_mean2, np.sqrt(norm_sd1**2 + norm_sd2**2))
    )
    assert hist_prod.exact_sd == approx(
        stats.norm.std(
            norm_mean1 + norm_mean2,
            np.sqrt(norm_sd1**2 + norm_sd2**2),
        )
    )


@given(
    norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    norm_sd1=st.floats(min_value=0.1, max_value=3),
    norm_sd2=st.floats(min_value=0.001, max_value=3),
)
@settings(max_examples=100)
def test_lognorm_product_exact_summary_stats(norm_mean1, norm_mean2, norm_sd1, norm_sd2):
    """Test that the formulas for exact moments are implemented correctly."""
    dist1 = LognormalDistribution(norm_mean=norm_mean1, norm_sd=norm_sd1)
    dist2 = LognormalDistribution(norm_mean=norm_mean2, norm_sd=norm_sd2)
    hist1 = NumericDistribution.from_distribution(dist1)
    hist2 = NumericDistribution.from_distribution(dist2)
    hist_prod = hist1 * hist2
    assert hist_prod.exact_mean == approx(
        stats.lognorm.mean(
            np.sqrt(norm_sd1**2 + norm_sd2**2), scale=np.exp(norm_mean1 + norm_mean2)
        )
    )
    assert hist_prod.exact_sd == approx(
        stats.lognorm.std(
            np.sqrt(norm_sd1**2 + norm_sd2**2), scale=np.exp(norm_mean1 + norm_mean2)
        )
    )


@given(
    mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    sd=st.floats(min_value=0.001, max_value=100),
)
@example(mean=0, sd=1)
def test_norm_basic(mean, sd):
    dist = NormalDistribution(mean=mean, sd=sd)
    hist = NumericDistribution.from_distribution(dist, bin_sizing="uniform")
    assert hist.histogram_mean() == approx(mean)
    assert hist.histogram_sd() == approx(sd, rel=0.01)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=3),
    bin_sizing=st.sampled_from(["uniform", "log-uniform", "ev", "mass"]),
)
def test_lognorm_mean(norm_mean, norm_sd, bin_sizing):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = NumericDistribution.from_distribution(dist, bin_sizing=bin_sizing)
    if bin_sizing == "ev":
        tolerance = 1e-6
    elif bin_sizing == "log-uniform":
        tolerance = 1e-2
    else:
        tolerance = 0.01 if dist.norm_sd < 3 else 0.1
    assert hist.histogram_mean() == approx(
        stats.lognorm.mean(dist.norm_sd, scale=np.exp(dist.norm_mean)),
        rel=tolerance,
    )


def test_norm_sd_bin_sizing_accuracy():
    # Accuracy order is ev > uniform > mass
    dist = NormalDistribution(mean=0, sd=1)
    ev_hist = NumericDistribution.from_distribution(dist, bin_sizing="ev")
    mass_hist = NumericDistribution.from_distribution(dist, bin_sizing="mass")
    uniform_hist = NumericDistribution.from_distribution(dist, bin_sizing="uniform")

    assert relative_error(ev_hist.histogram_sd(), dist.sd) < relative_error(uniform_hist.histogram_sd(), dist.sd)
    assert relative_error(uniform_hist.histogram_sd(), dist.sd) < relative_error(mass_hist.histogram_sd(), dist.sd)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.01, max_value=3),
)
def test_lognorm_sd(norm_mean, norm_sd):
    test_edges = False
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = NumericDistribution.from_distribution(dist, bin_sizing="log-uniform")
        ev_hist = NumericDistribution.from_distribution(dist, bin_sizing="ev")
        mass_hist = NumericDistribution.from_distribution(dist, bin_sizing="mass")
        uniform_hist = NumericDistribution.from_distribution(dist, bin_sizing="uniform")

    def true_variance(left, right):
        return integrate.quad(
            lambda x: (x - dist.lognorm_mean) ** 2
            * stats.lognorm.pdf(x, dist.norm_sd, scale=np.exp(dist.norm_mean)),
            left,
            right,
        )[0]

    def observed_variance(left, right):
        return np.sum(
            hist.masses[left:right] * (hist.values[left:right] - hist.histogram_mean()) ** 2
        )

    if test_edges:
        # Note: For bin_sizing=ev, adding more bins increases accuracy overall,
        # but decreases accuracy on the far right tail.
        midpoint = hist.values[int(hist.num_bins * 9 / 10)]
        expected_left_variance = true_variance(0, midpoint)
        expected_right_variance = true_variance(midpoint, np.inf)
        midpoint_index = int(len(hist) * hist.contribution_to_ev(midpoint))
        observed_left_variance = observed_variance(0, midpoint_index)
        observed_right_variance = observed_variance(midpoint_index, len(hist))
        print("")
        print_accuracy_ratio(observed_left_variance, expected_left_variance, "Left   ")
        print_accuracy_ratio(observed_right_variance, expected_right_variance, "Right  ")
        print_accuracy_ratio(hist.histogram_sd(), dist.lognorm_sd, "Overall")

    assert hist.histogram_sd() == approx(dist.lognorm_sd, rel=0.2)



def test_lognorm_sd_bin_sizing_accuracy():
    # For narrower distributions (eg lognorm_sd=lognorm_mean), the accuracy order is
    # log-uniform > ev > mass > uniform
    # For wider distributions, the accuracy order is
    # log-uniform > ev > uniform > mass
    dist = LognormalDistribution(lognorm_mean=1e6, lognorm_sd=1e7)
    log_uniform_hist = NumericDistribution.from_distribution(dist, bin_sizing="log-uniform")
    ev_hist = NumericDistribution.from_distribution(dist, bin_sizing="ev")
    mass_hist = NumericDistribution.from_distribution(dist, bin_sizing="mass")
    uniform_hist = NumericDistribution.from_distribution(dist, bin_sizing="uniform")

    assert relative_error(log_uniform_hist.histogram_sd(), dist.lognorm_sd) < relative_error(ev_hist.histogram_sd(), dist.lognorm_sd)
    assert relative_error(ev_hist.histogram_sd(), dist.lognorm_sd) < relative_error(uniform_hist.histogram_sd(), dist.lognorm_sd)
    assert relative_error(uniform_hist.histogram_sd(), dist.lognorm_sd) < relative_error(mass_hist.histogram_sd(), dist.lognorm_sd)


@given(
    mean1=st.floats(min_value=-1000, max_value=0.01),
    mean2=st.floats(min_value=0.01, max_value=1000),
    sd1=st.floats(min_value=0.1, max_value=10),
    sd2=st.floats(min_value=0.1, max_value=10),
    bin_sizing=st.sampled_from(["ev", "mass", "uniform"]),
)
def test_noncentral_norm_product(mean1, mean2, sd1, sd2, bin_sizing):
    dist1 = NormalDistribution(mean=mean1, sd=sd1)
    dist2 = NormalDistribution(mean=mean2, sd=sd2)
    mean_tolerance = 1e-5
    sd_tolerance = 0.2 if bin_sizing == "uniform" else 1
    hist1 = NumericDistribution.from_distribution(dist1, num_bins=25, bin_sizing=bin_sizing)
    hist2 = NumericDistribution.from_distribution(dist2, num_bins=25, bin_sizing=bin_sizing)
    hist_prod = hist1 * hist2
    assert hist_prod.histogram_mean() == approx(dist1.mean * dist2.mean, rel=mean_tolerance, abs=1e-10)
    assert hist_prod.histogram_sd() == approx(
        np.sqrt(
            (dist1.sd**2 + dist1.mean**2) * (dist2.sd**2 + dist2.mean**2)
            - dist1.mean**2 * dist2.mean**2
        ),
        rel=sd_tolerance,
    )


@given(
    mean=st.floats(min_value=-10, max_value=10),
    sd=st.floats(min_value=0.001, max_value=100),
    num_bins=st.sampled_from([25, 100]),
    bin_sizing=st.sampled_from(["ev", "mass", "uniform"]),
)
@settings(max_examples=100)
def test_norm_mean_error_propagation(mean, sd, num_bins, bin_sizing):
    dist = NormalDistribution(mean=mean, sd=sd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = NumericDistribution.from_distribution(
            dist, num_bins=num_bins, bin_sizing=bin_sizing
        )
        hist_base = NumericDistribution.from_distribution(
            dist, num_bins=num_bins, bin_sizing=bin_sizing
        )
    tolerance = 1e-10 if bin_sizing == "ev" else 1e-5

    for i in range(1, 17):
        true_mean = mean**i
        true_sd = np.sqrt((dist.sd**2 + dist.mean**2) ** i - dist.mean ** (2 * i))
        if true_sd > 1e15:
            break
        assert hist.histogram_mean() == approx(
            true_mean, abs=tolerance ** (1 / i), rel=tolerance ** (1 / i)
        ), f"On iteration {i}"
        hist = hist * hist_base


@given(
    mean1=st.floats(min_value=-100, max_value=100),
    mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    sd1=st.floats(min_value=0.001, max_value=100),
    sd2=st.floats(min_value=0.001, max_value=3),
    num_bins1=st.sampled_from([25, 100]),
    num_bins2=st.sampled_from([25, 100]),
)
def test_norm_lognorm_product(mean1, mean2, sd1, sd2, num_bins1, num_bins2):
    dist1 = NormalDistribution(mean=mean1, sd=sd1)
    dist2 = LognormalDistribution(norm_mean=mean2, norm_sd=sd2)
    hist1 = NumericDistribution.from_distribution(dist1, num_bins=num_bins1)
    hist2 = NumericDistribution.from_distribution(dist2, num_bins=num_bins2)
    hist_prod = hist1 * hist2
    assert all(np.diff(hist_prod.values) >= 0), hist_prod.values
    assert hist_prod.histogram_mean() == approx(hist_prod.exact_mean, abs=1e-5, rel=1e-5)

    # SD is pretty inaccurate
    sd_tolerance = 1 if num_bins1 == 100 and num_bins2 == 100 else 2
    assert hist_prod.histogram_sd() == approx(hist_prod.exact_sd, rel=sd_tolerance)


@given(
    norm_mean=st.floats(min_value=np.log(1e-9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=3),
    num_bins=st.sampled_from([25, 100]),
    bin_sizing=st.sampled_from(["ev", "log-uniform"]),
)
@example(norm_mean=0.0, norm_sd=1.0, num_bins=25, bin_sizing="ev").via(
    "discovered failure"
)
def test_lognorm_mean_error_propagation(norm_mean, norm_sd, num_bins, bin_sizing):
    assume(not (num_bins == 10 and bin_sizing == "log-uniform"))
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = NumericDistribution.from_distribution(dist, num_bins=num_bins, bin_sizing=bin_sizing)
    hist_base = NumericDistribution.from_distribution(
        dist, num_bins=num_bins, bin_sizing=bin_sizing
    )
    inv_tolerance = 1 - 1e-12 if bin_sizing == "ev" else 0.98

    for i in range(1, 13):
        true_mean = stats.lognorm.mean(np.sqrt(i) * norm_sd, scale=np.exp(i * norm_mean))
        if bin_sizing == "ev":
            # log-uniform can have out-of-order values due to the masses at the
            # end being very small
            assert all(np.diff(hist.values) >= 0), f"On iteration {i}: {hist.values}"
        assert hist.histogram_mean() == approx(true_mean, rel=1 - inv_tolerance**i), f"On iteration {i}"
        hist = hist * hist_base


@given(bin_sizing=st.sampled_from(["ev"]))
def test_lognorm_sd_error_propagation(bin_sizing):
    verbose = False
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    num_bins = 100
    hist = NumericDistribution.from_distribution(dist, num_bins=num_bins, bin_sizing=bin_sizing)
    abs_error = []
    rel_error = []

    if verbose:
        print("")
    for i in [1, 2, 4, 8, 16, 32]:
        true_mean = stats.lognorm.mean(np.sqrt(i))
        true_sd = hist.exact_sd
        abs_error.append(abs(hist.histogram_sd() - true_sd))
        rel_error.append(relative_error(hist.histogram_sd(), true_sd))
        if verbose:
            print(f"n = {i:2d}: {rel_error[-1]*100:4.1f}% from SD {hist.histogram_sd():.3f}")
        hist = hist * hist

    expected_error_pcts = (
        [0.9, 2.8, 9.9, 40.7, 211, 2678]
        if bin_sizing == "ev"
        else [12, 26.3, 99.8, 733, 32000, 1e9]
    )

    for i in range(len(expected_error_pcts)):
        assert rel_error[i] < expected_error_pcts[i] / 100


@given(
    norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_mean2=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd1=st.floats(min_value=0.1, max_value=3),
    norm_sd2=st.floats(min_value=0.1, max_value=3),
)
def test_lognorm_product(norm_mean1, norm_sd1, norm_mean2, norm_sd2):
    dists = [
        LognormalDistribution(norm_mean=norm_mean1, norm_sd=norm_sd1),
        LognormalDistribution(norm_mean=norm_mean2, norm_sd=norm_sd2),
    ]
    dist_prod = LognormalDistribution(
        norm_mean=norm_mean1 + norm_mean2, norm_sd=np.sqrt(norm_sd1**2 + norm_sd2**2)
    )
    pmhs = [NumericDistribution.from_distribution(dist) for dist in dists]
    pmh_prod = reduce(lambda acc, hist: acc * hist, pmhs)

    # Lognorm width grows with e**norm_sd**2, so error tolerance grows the same way
    tolerance = 1.05 ** (1 + (norm_sd1 + norm_sd2) ** 2) - 1
    assert pmh_prod.histogram_mean() == approx(dist_prod.lognorm_mean)
    assert pmh_prod.histogram_sd() == approx(dist_prod.lognorm_sd, rel=tolerance)


@given(
    norm_mean1=st.floats(-1e5, 1e5),
    norm_mean2=st.floats(min_value=-1e5, max_value=1e5),
    norm_sd1=st.floats(min_value=0.001, max_value=1e5),
    norm_sd2=st.floats(min_value=0.001, max_value=1e5),
    num_bins1=st.sampled_from([25, 100]),
    num_bins2=st.sampled_from([25, 100]),
    bin_sizing=st.sampled_from(["ev", "uniform"]),
)
# TODO: This example has rounding issues where -neg_ev_contribution > mean, so
# pos_ev_contribution ends up negative. neg_ev_contribution should be a little
# bigger
@example(
    norm_mean1=0,
    norm_mean2=-3,
    norm_sd1=0.5,
    norm_sd2=0.5,
    num_bins1=25,
    num_bins2=25,
    bin_sizing="uniform",
)
def test_norm_sum(norm_mean1, norm_mean2, norm_sd1, norm_sd2, num_bins1, num_bins2, bin_sizing):
    dist1 = NormalDistribution(mean=norm_mean1, sd=norm_sd1)
    dist2 = NormalDistribution(mean=norm_mean2, sd=norm_sd2)
    hist1 = NumericDistribution.from_distribution(dist1, num_bins=num_bins1, bin_sizing=bin_sizing)
    hist2 = NumericDistribution.from_distribution(dist2, num_bins=num_bins2, bin_sizing=bin_sizing)
    hist_sum = hist1 + hist2

    # The further apart the means are, the less accurate the SD estimate is
    distance_apart = abs(norm_mean1 - norm_mean2) / hist_sum.exact_sd
    sd_tolerance = 2 + 0.5 * distance_apart

    assert all(np.diff(hist_sum.values) >= 0)
    assert hist_sum.histogram_mean() == approx(hist_sum.exact_mean, abs=1e-10, rel=1e-5)
    assert hist_sum.histogram_sd() == approx(hist_sum.exact_sd, rel=sd_tolerance)


@given(
    norm_mean1=st.floats(min_value=-np.log(1e6), max_value=np.log(1e6)),
    norm_mean2=st.floats(min_value=-np.log(1e6), max_value=np.log(1e6)),
    norm_sd1=st.floats(min_value=0.1, max_value=3),
    norm_sd2=st.floats(min_value=0.01, max_value=3),
    num_bins1=st.sampled_from([25, 100]),
    num_bins2=st.sampled_from([25, 100]),
)
def test_lognorm_sum(norm_mean1, norm_mean2, norm_sd1, norm_sd2, num_bins1, num_bins2):
    dist1 = LognormalDistribution(norm_mean=norm_mean1, norm_sd=norm_sd1)
    dist2 = LognormalDistribution(norm_mean=norm_mean2, norm_sd=norm_sd2)
    hist1 = NumericDistribution.from_distribution(dist1, num_bins=num_bins1)
    hist2 = NumericDistribution.from_distribution(dist2, num_bins=num_bins2)
    hist_sum = hist1 + hist2
    assert all(np.diff(hist_sum.values) >= 0), hist_sum.values
    assert hist_sum.histogram_mean() == approx(hist_sum.exact_mean)

    # SD is very inaccurate because adding lognormals produces some large but
    # very low-probability values on the right tail and the only approach is to
    # either downweight them or make the histogram much wider.
    assert hist_sum.histogram_sd() > min(hist1.histogram_sd(), hist2.histogram_sd())
    assert hist_sum.histogram_sd() == approx(hist_sum.exact_sd, rel=2)


@given(
    mean1=st.floats(min_value=-100, max_value=100),
    mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    sd1=st.floats(min_value=0.001, max_value=100),
    sd2=st.floats(min_value=0.001, max_value=3),
    num_bins1=st.sampled_from([25, 100]),
    num_bins2=st.sampled_from([25, 100]),
)
# TODO: the top bin "should" be no less than 445 (extended_values[-100:] ranges
# from 445 to 459) but it's getting squashed down to 1.9. why? looks like there
# are actually only 3 bins and 1013 items per bin on the positive side. maybe
# we shouldn't be trying to size each side by contribution to EV
@example(mean1=-21.0, mean2=0.0, sd1=1.0, sd2=1.5, num_bins1=100, num_bins2=100).via(
    "discovered failure"
)
def test_norm_lognorm_sum(mean1, mean2, sd1, sd2, num_bins1, num_bins2):
    dist1 = NormalDistribution(mean=mean1, sd=sd1)
    dist2 = LognormalDistribution(norm_mean=mean2, norm_sd=sd2)
    hist1 = NumericDistribution.from_distribution(dist1, num_bins=num_bins1)
    hist2 = NumericDistribution.from_distribution(dist2, num_bins=num_bins2)
    hist_sum = hist1 + hist2
    sd_tolerance = 0.5
    assert all(np.diff(hist_sum.values) >= 0), hist_sum.values
    assert hist_sum.histogram_mean() == approx(hist_sum.exact_mean, abs=1e-6, rel=1e-6)
    assert hist_sum.histogram_sd() == approx(hist_sum.exact_sd, rel=sd_tolerance)


def test_norm_product_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 8 distributions together.

    Note: With more multiplications, MC has a good chance of being more
    accurate, and is significantly more accurate at 16 multiplications.
    """
    # Time complexity for binary operations is roughly O(n^2) for PMH and O(n)
    # for MC, so let MC have num_bins^2 samples.
    num_bins = 100
    num_samples = 100**2
    dists = [NormalDistribution(mean=i, sd=0.5 + i / 4) for i in range(9)]
    hists = [NumericDistribution.from_distribution(dist, num_bins=num_bins) for dist in dists]
    hist = reduce(lambda acc, hist: acc * hist, hists)
    dist_abs_error = abs(hist.histogram_sd() - hist.exact_sd)

    mc_abs_error = get_mc_accuracy(hist.exact_sd, num_samples, dists, lambda acc, mc: acc * mc)
    assert dist_abs_error < mc_abs_error


def test_lognorm_product_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 16 distributions together."""
    num_bins = 100
    num_samples = 100**2
    dists = [LognormalDistribution(norm_mean=i, norm_sd=0.5 + i / 4) for i in range(17)]
    hists = [NumericDistribution.from_distribution(dist, num_bins=num_bins) for dist in dists]
    hist = reduce(lambda acc, hist: acc * hist, hists)
    dist_abs_error = abs(hist.histogram_sd() - hist.exact_sd)

    mc_abs_error = get_mc_accuracy(hist.exact_sd, num_samples, dists, lambda acc, mc: acc * mc)
    assert dist_abs_error < mc_abs_error


def test_norm_sum_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 8 distributions together.

    Note: With more multiplications, MC has a good chance of being more
    accurate, and is significantly more accurate at 16 multiplications.
    """
    num_bins = 1000
    num_samples = num_bins**2
    dists = [NormalDistribution(mean=i, sd=0.5 + i / 4) for i in range(9)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hists = [
            NumericDistribution.from_distribution(dist, num_bins=num_bins, bin_sizing="uniform")
            for dist in dists
        ]
    hist = reduce(lambda acc, hist: acc + hist, hists)
    dist_abs_error = abs(hist.histogram_sd() - hist.exact_sd)

    mc_abs_error = get_mc_accuracy(hist.exact_sd, num_samples, dists, lambda acc, mc: acc + mc)
    assert dist_abs_error < mc_abs_error


def test_lognorm_sum_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 16 distributions together."""
    num_bins = 100
    num_samples = 100**2
    dists = [LognormalDistribution(norm_mean=i, norm_sd=0.5 + i / 4) for i in range(17)]
    hists = [NumericDistribution.from_distribution(dist, num_bins=num_bins) for dist in dists]
    hist = reduce(lambda acc, hist: acc + hist, hists)
    dist_abs_error = abs(hist.histogram_sd() - hist.exact_sd)

    mc_abs_error = get_mc_accuracy(hist.exact_sd, num_samples, dists, lambda acc, mc: acc + mc)
    assert dist_abs_error < mc_abs_error


@given(
    norm_mean=st.floats(min_value=-1e6, max_value=1e6),
    norm_sd=st.floats(min_value=0.001, max_value=3),
    num_bins=st.sampled_from([25, 100]),
    bin_sizing=st.sampled_from(["ev", "uniform"]),
)
def test_norm_negate(norm_mean, norm_sd, num_bins, bin_sizing):
    dist = NormalDistribution(mean=0, sd=1)
    hist = NumericDistribution.from_distribution(dist)
    neg_hist = -hist
    assert neg_hist.histogram_mean() == approx(-hist.histogram_mean())
    assert neg_hist.histogram_sd() == approx(hist.histogram_sd())


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=3),
    num_bins=st.sampled_from([25, 100]),
    bin_sizing=st.sampled_from(["ev", "uniform"]),
)
def test_lognorm_negate(norm_mean, norm_sd, num_bins, bin_sizing):
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    hist = NumericDistribution.from_distribution(dist)
    neg_hist = -hist
    assert neg_hist.histogram_mean() == approx(-hist.histogram_mean())
    assert neg_hist.histogram_sd() == approx(hist.histogram_sd())


@given(
    a=st.floats(min_value=-100, max_value=100),
    b=st.floats(min_value=-100, max_value=100),
)
@example(a=99.99999999988448, b=100.0)
@example(a=-1, b=1)
def test_uniform_basic(a, b):
    a, b = fix_uniform(a, b)
    dist = UniformDistribution(x=a, y=b)
    with warnings.catch_warnings():
        # hypothesis generates some extremely tiny input params, which
        # generates warnings about EV contributions being 0.
        warnings.simplefilter("ignore")
        hist = NumericDistribution.from_distribution(dist)
    assert hist.histogram_mean() == approx((a + b) / 2, 1e-6)
    assert hist.histogram_sd() == approx(np.sqrt(1 / 12 * (b - a) ** 2), rel=1e-3)


@given(
    dist2_type=st.sampled_from(["norm", "lognorm"]),
    mean1=st.floats(min_value=-1e6, max_value=1e6),
    mean2=st.floats(min_value=-100, max_value=100),
    sd1=st.floats(min_value=0.001, max_value=1000),
    sd2=st.floats(min_value=0.1, max_value=5),
    num_bins=st.sampled_from([30, 100]),
    bin_sizing=st.sampled_from(["ev", "uniform"]),
)
def test_sub(dist2_type, mean1, mean2, sd1, sd2, num_bins, bin_sizing):
    dist1 = NormalDistribution(mean=mean1, sd=sd1)

    if dist2_type == "norm":
        dist2 = NormalDistribution(mean=mean2, sd=sd2)
        neg_dist = NormalDistribution(mean=-mean2, sd=sd2)
    elif dist2_type == "lognorm":
        dist2 = LognormalDistribution(norm_mean=mean2, norm_sd=sd2)
        # We can't negate a lognormal distribution by changing the params
        neg_dist = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist1 = NumericDistribution.from_distribution(
            dist1, num_bins=num_bins, bin_sizing=bin_sizing
        )
        hist2 = NumericDistribution.from_distribution(
            dist2, num_bins=num_bins, bin_sizing=bin_sizing
        )
    hist_diff = hist1 - hist2
    backward_diff = hist2 - hist1
    assert not any(np.isnan(hist_diff.values))
    assert all(np.diff(hist_diff.values) >= 0)
    assert hist_diff.histogram_mean() == approx(-backward_diff.histogram_mean(), rel=0.01)
    assert hist_diff.histogram_sd() == approx(backward_diff.histogram_sd(), rel=0.05)

    if neg_dist:
        neg_hist = NumericDistribution.from_distribution(
            neg_dist, num_bins=num_bins, bin_sizing=bin_sizing
        )
        hist_sum = hist1 + neg_hist
        assert hist_diff.histogram_mean() == approx(hist_sum.histogram_mean(), rel=0.01)
        assert hist_diff.histogram_sd() == approx(hist_sum.histogram_sd(), rel=0.05)


def test_uniform_sum_basic():
    # The sum of standard uniform distributions is also known as an Irwin-Hall
    # distribution:
    # https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution
    dist = UniformDistribution(0, 1)
    hist1 = NumericDistribution.from_distribution(dist)
    hist_sum = NumericDistribution.from_distribution(dist)
    hist_sum += hist1
    assert hist_sum.exact_mean == approx(1)
    assert hist_sum.exact_sd == approx(np.sqrt(2 / 12))
    assert hist_sum.histogram_mean() == approx(1)
    assert hist_sum.histogram_sd() == approx(np.sqrt(2 / 12), rel=1e-3)
    hist_sum += hist1
    assert hist_sum.histogram_mean() == approx(1.5)
    assert hist_sum.histogram_sd() == approx(np.sqrt(3 / 12), rel=1e-3)
    hist_sum += hist1
    assert hist_sum.histogram_mean() == approx(2)
    assert hist_sum.histogram_sd() == approx(np.sqrt(4 / 12), rel=1e-3)


@given(
    # I originally had both dists on [-1000, 1000] but then hypothesis would
    # generate ~90% of cases with extremely tiny values that are too small for
    # floating point operations to handle, so I forced most of the values to be
    # at least a little away from 0.
    a1=st.floats(min_value=-1000, max_value=0.001),
    b1=st.floats(min_value=0.001, max_value=1000),
    a2=st.floats(min_value=0, max_value=1000),
    b2=st.floats(min_value=1, max_value=10000),
    flip2=st.booleans(),
)
def test_uniform_sum(a1, b1, a2, b2, flip2):
    if flip2:
        a2, b2 = -b2, -a2
    a1, b1 = fix_uniform(a1, b1)
    a2, b2 = fix_uniform(a2, b2)
    dist1 = UniformDistribution(x=a1, y=b1)
    dist2 = UniformDistribution(x=a2, y=b2)
    with warnings.catch_warnings():
        # hypothesis generates some extremely tiny input params, which
        # generates warnings about EV contributions being 0.
        warnings.simplefilter("ignore")
        hist1 = NumericDistribution.from_distribution(dist1)
        hist2 = NumericDistribution.from_distribution(dist2)

    hist_sum = hist1 + hist2
    assert hist_sum.histogram_mean() == approx(hist_sum.exact_mean)
    assert hist_sum.histogram_sd() == approx(hist_sum.exact_sd, rel=0.01)


@given(
    a1=st.floats(min_value=-1000, max_value=0.001),
    b1=st.floats(min_value=0.001, max_value=1000),
    a2=st.floats(min_value=0, max_value=1000),
    b2=st.floats(min_value=1, max_value=10000),
    flip2=st.booleans(),
)
def test_uniform_prod(a1, b1, a2, b2, flip2):
    if flip2:
        a2, b2 = -b2, -a2
    a1, b1 = fix_uniform(a1, b1)
    a2, b2 = fix_uniform(a2, b2)
    dist1 = UniformDistribution(x=a1, y=b1)
    dist2 = UniformDistribution(x=a2, y=b2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist1 = NumericDistribution.from_distribution(dist1)
        hist2 = NumericDistribution.from_distribution(dist2)
    hist_prod = hist1 * hist2
    assert hist_prod.histogram_mean() == approx(hist_prod.exact_mean, abs=1e-6, rel=1e-6)
    assert hist_prod.histogram_sd() == approx(hist_prod.exact_sd, rel=0.01)


@given(
    a=st.floats(min_value=-1000, max_value=0.001),
    b=st.floats(min_value=0.001, max_value=1000),
    norm_mean=st.floats(np.log(0.001), np.log(1e6)),
    norm_sd=st.floats(0.1, 2),
)
def test_uniform_lognorm_prod(a, b, norm_mean, norm_sd):
    a, b = fix_uniform(a, b)
    dist1 = UniformDistribution(x=a, y=b)
    dist2 = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist1 = NumericDistribution.from_distribution(dist1)
    hist2 = NumericDistribution.from_distribution(dist2)
    hist_prod = hist1 * hist2
    assert hist_prod.histogram_mean() == approx(hist_prod.exact_mean)
    assert hist_prod.histogram_sd() == approx(hist_prod.exact_sd, rel=0.5)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=1, max_value=99),
)
def test_numeric_dist_contribution_to_ev(norm_mean, norm_sd, bin_num):
    fraction = bin_num / 100
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = NumericDistribution.from_distribution(dist)
    assert hist.contribution_to_ev(dist.inv_contribution_to_ev(fraction)) == approx(fraction)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=2, max_value=98),
)
def test_numeric_dist_inv_contribution_to_ev(norm_mean, norm_sd, bin_num):
    # The nth value stored in the PMH represents a value between the nth and n+1th edges
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = NumericDistribution.from_distribution(dist)
    fraction = bin_num / hist.num_bins
    prev_fraction = fraction - 1 / hist.num_bins
    next_fraction = fraction
    assert hist.inv_contribution_to_ev(fraction) > dist.inv_contribution_to_ev(prev_fraction)
    assert hist.inv_contribution_to_ev(fraction) < dist.inv_contribution_to_ev(next_fraction)


@given(
    mean=st.floats(min_value=100, max_value=100),
    sd=st.floats(min_value=0.01, max_value=100),
    percent=st.integers(min_value=1, max_value=99),
)
def test_quantile_uniform(mean, sd, percent):
    dist = NormalDistribution(mean=mean, sd=sd)
    hist = NumericDistribution.from_distribution(dist, num_bins=200, bin_sizing="uniform")
    assert hist.quantile(0) == hist.values[0]
    assert hist.quantile(1) == hist.values[-1]
    assert hist.percentile(percent) == approx(stats.norm.ppf(percent / 100, loc=mean, scale=sd), rel=0.25)


@given(
    norm_mean=st.floats(min_value=-5, max_value=5),
    norm_sd=st.floats(min_value=0.1, max_value=2),
    percent=st.integers(min_value=1, max_value=99),
)
def test_quantile_log_uniform(norm_mean, norm_sd, percent):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = NumericDistribution.from_distribution(dist, num_bins=200, bin_sizing="log-uniform")
    assert hist.quantile(0) == hist.values[0]
    assert hist.quantile(1) == hist.values[-1]
    assert hist.percentile(percent) == approx(stats.lognorm.ppf(percent / 100, norm_sd, scale=np.exp(norm_mean)), rel=0.1)


@given(
    norm_mean=st.floats(min_value=-5, max_value=5),
    norm_sd=st.floats(min_value=0.1, max_value=2),
    # Don't try smaller percentiles because the smaller bins have a lot of
    # probability mass
    percent=st.integers(min_value=40, max_value=99),
)
def test_quantile_ev(norm_mean, norm_sd, percent):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = NumericDistribution.from_distribution(dist, num_bins=200, bin_sizing="ev")
    assert hist.quantile(0) == hist.values[0]
    assert hist.quantile(1) == hist.values[-1]
    assert hist.percentile(percent) == approx(stats.lognorm.ppf(percent / 100, norm_sd, scale=np.exp(norm_mean)), rel=0.1)


@given(
    mean=st.floats(min_value=100, max_value=100),
    sd=st.floats(min_value=0.01, max_value=100),
    percent=st.integers(min_value=0, max_value=100),
)
@example(mean=0, sd=1, percent=1)
def test_quantile_mass(mean, sd, percent):
    dist = NormalDistribution(mean=mean, sd=sd)
    hist = NumericDistribution.from_distribution(dist, num_bins=200, bin_sizing="mass")

    # It's hard to make guarantees about how close the value will be, but we
    # should know for sure that the cdf of the value is very close to the
    # percent.
    assert 100 * stats.norm.cdf(hist.percentile(percent), mean, sd) == approx(percent, abs=0.5)


@given(
    mean=st.floats(min_value=100, max_value=100),
    sd=st.floats(min_value=0.01, max_value=100),
)
def test_cdf_mass(mean, sd):
    dist = NormalDistribution(mean=mean, sd=sd)
    hist = NumericDistribution.from_distribution(dist, num_bins=200, bin_sizing="mass")

    assert hist.cdf(mean) == approx(0.5, abs=0.005)
    assert hist.cdf(mean - sd) == approx(stats.norm.cdf(-1), abs=0.005)
    assert hist.cdf(mean + 2 * sd) == approx(stats.norm.cdf(2), abs=0.005)

@given(
    mean=st.floats(min_value=100, max_value=100),
    sd=st.floats(min_value=0.01, max_value=100),
    percent=st.integers(min_value=0, max_value=100),
)
def test_cdf_inverts_quantile(mean, sd, percent):
    dist = NormalDistribution(mean=mean, sd=sd)
    hist = NumericDistribution.from_distribution(dist, num_bins=200, bin_sizing="mass")
    assert 100 * hist.cdf(hist.percentile(percent)) == approx(percent, abs=0.5)


@given(
    mean1=st.floats(min_value=100, max_value=100),
    mean2=st.floats(min_value=100, max_value=100),
    sd1=st.floats(min_value=0.01, max_value=100),
    sd2=st.floats(min_value=0.01, max_value=100),
    percent=st.integers(min_value=1, max_value=99),
)
def test_quantile_mass_after_sum(mean1, mean2, sd1, sd2, percent):
    dist1 = NormalDistribution(mean=mean1, sd=sd1)
    dist2 = NormalDistribution(mean=mean2, sd=sd2)
    hist1 = NumericDistribution.from_distribution(dist1, num_bins=200, bin_sizing="mass")
    hist2 = NumericDistribution.from_distribution(dist2, num_bins=200, bin_sizing="mass")
    hist_sum = hist1 + hist2
    assert hist_sum.percentile(percent) == approx(stats.norm.ppf(percent / 100, mean1 + mean2, np.sqrt(sd1**2 + sd2**2)), rel=0.1)


def test_plot():
    return None
    hist = NumericDistribution.from_distribution(
        LognormalDistribution(norm_mean=0, norm_sd=1)
    ) * NumericDistribution.from_distribution(NormalDistribution(mean=0, sd=5))
    # hist = NumericDistribution.from_distribution(LognormalDistribution(norm_mean=0, norm_sd=2))
    hist.plot(scale="linear")


def test_performance():
    return None
    # Note: I wrote some C++ code to approximate the behavior of distribution
    # multiplication. On my machine, distribution multiplication (with profile
    # = False) runs in 15s, and the equivalent C++ code (with -O3) runs in 11s.
    # The C++ code is not well-optimized, the most glaring issue being it uses
    # std::sort instead of something like argpartition (the trouble is that
    # numpy's argpartition can partition on many values simultaneously, whereas
    # C++'s std::partition can only partition on one value at a time, which is
    # far slower).
    dist1 = NormalDistribution(mean=0, sd=1)
    dist2 = LognormalDistribution(norm_mean=0, norm_sd=1)

    profile = True
    if profile:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    for i in range(10000):
        hist1 = NumericDistribution.from_distribution(dist1, num_bins=100, bin_sizing="mass")
        hist2 = NumericDistribution.from_distribution(dist2, num_bins=100, bin_sizing="mass")
        hist1 = hist1 * hist2

    if profile:
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
