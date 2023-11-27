from functools import reduce
from hypothesis import assume, example, given, settings
import hypothesis.strategies as st
import numpy as np
from pytest import approx
from scipy import integrate, stats

from ..squigglepy.distributions import LognormalDistribution, NormalDistribution
from ..squigglepy.pdh import ProbabilityMassHistogram
from ..squigglepy import samplers


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
    hist1 = ProbabilityMassHistogram.from_distribution(dist1)
    hist2 = ProbabilityMassHistogram.from_distribution(dist2)
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
    hist1 = ProbabilityMassHistogram.from_distribution(dist1)
    hist2 = ProbabilityMassHistogram.from_distribution(dist2)
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
@example(mean=1.0, sd=0.375).via("discovered failure")
def test_norm_basic(mean, sd):
    dist = NormalDistribution(mean=mean, sd=sd)
    hist = ProbabilityMassHistogram.from_distribution(dist, bin_sizing="ev")
    assert hist.histogram_mean() == approx(mean)
    assert hist.histogram_sd() == approx(sd, rel=0.01)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=5),
    bin_sizing=st.sampled_from(["ev", "uniform"]),
)
@example(norm_mean=-12.0, norm_sd=5.0, bin_sizing="uniform").via("discovered failure")
def test_lognorm_mean(norm_mean, norm_sd, bin_sizing):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist, bin_sizing=bin_sizing)
    tolerance = 1e-6 if bin_sizing == "ev" else (0.01 if dist.norm_sd < 3 else 0.1)
    assert hist.histogram_mean() == approx(
        stats.lognorm.mean(dist.norm_sd, scale=np.exp(dist.norm_mean)),
        rel=tolerance,
    )


@given(
    # norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    # norm_sd=st.floats(min_value=0.01, max_value=5),
    norm_mean=st.just(0),
    norm_sd=st.just(1),
)
# @example(norm_mean=0, norm_sd=3)
def test_lognorm_sd(norm_mean, norm_sd):
    # TODO: The margin of error on the SD estimate is pretty big, mostly
    # because the right tail is underestimating variance. But that might be an
    # acceptable cost. Try to see if there's a way to improve it without compromising the fidelity of the EV estimate.
    #
    # Note: Adding more bins increases accuracy overall, but decreases accuracy
    # on the far right tail.
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist, bin_sizing="ev")

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

    midpoint = hist.values[int(hist.num_bins * 9 / 10)]
    expected_left_variance = true_variance(0, midpoint)
    expected_right_variance = true_variance(midpoint, np.inf)
    midpoint_index = int(len(hist) * hist.contribution_to_ev(midpoint))
    observed_left_variance = observed_variance(0, midpoint_index)
    observed_right_variance = observed_variance(midpoint_index, len(hist))
    # print("")
    # print_accuracy_ratio(observed_left_variance, expected_left_variance, "Left   ")
    # print_accuracy_ratio(observed_right_variance, expected_right_variance, "Right  ")
    # print_accuracy_ratio(hist.histogram_sd(), dist.lognorm_sd, "Overall")
    assert hist.histogram_sd() == approx(dist.lognorm_sd, rel=0.5)


@given(
    mean1=st.floats(min_value=-1000, max_value=0.01),
    mean2=st.floats(min_value=0.01, max_value=1000),
    sd1=st.floats(min_value=0.1, max_value=10),
    sd2=st.floats(min_value=0.1, max_value=10),
    bin_sizing=st.sampled_from(["ev", "uniform"])
)
def test_noncentral_norm_product(mean1, mean2, sd1, sd2, bin_sizing):
    dist1 = NormalDistribution(mean=mean1, sd=sd1)
    dist2 = NormalDistribution(mean=mean2, sd=sd2)
    tolerance = 1e-9 if bin_sizing == "ev" else 1e-5
    hist1 = ProbabilityMassHistogram.from_distribution(
        dist1, num_bins=25, bin_sizing=bin_sizing
    )
    hist2 = ProbabilityMassHistogram.from_distribution(
        dist2, num_bins=25, bin_sizing=bin_sizing
    )
    hist_prod = hist1 * hist2
    assert hist_prod.histogram_mean() == approx(dist1.mean * dist2.mean, rel=tolerance, abs=1e-10)
    assert hist_prod.histogram_sd() == approx(
        np.sqrt(
            (dist1.sd**2 + dist1.mean**2) * (dist2.sd**2 + dist2.mean**2)
            - dist1.mean**2 * dist2.mean**2
        ),
        rel=1,
    )


@given(
    mean=st.floats(min_value=-10, max_value=10),
    sd=st.floats(min_value=0.001, max_value=100),
    num_bins=st.sampled_from([25, 100]),
    bin_sizing=st.sampled_from(["ev", "uniform"]),
)
@settings(max_examples=100)
def test_norm_mean_error_propagation(mean, sd, num_bins, bin_sizing):
    dist = NormalDistribution(mean=mean, sd=sd)
    hist = ProbabilityMassHistogram.from_distribution(
        dist, num_bins=num_bins, bin_sizing=bin_sizing
    )
    hist_base = ProbabilityMassHistogram.from_distribution(
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
    hist1 = ProbabilityMassHistogram.from_distribution(dist1, num_bins=num_bins1)
    hist2 = ProbabilityMassHistogram.from_distribution(dist2, num_bins=num_bins2)
    hist_prod = hist1 * hist2
    assert all(hist_prod.values[:-1] <= hist_prod.values[1:]), hist_prod.values
    assert hist_prod.histogram_mean() == approx(hist_prod.exact_mean, abs=1e-5, rel=1e-5)

    # SD is pretty inaccurate
    sd_tolerance = 1 if num_bins1 == 100 and num_bins2 == 100 else 2
    assert hist_prod.histogram_sd() == approx(hist_prod.exact_sd, rel=sd_tolerance)


@given(
    norm_mean=st.floats(min_value=np.log(1e-9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=3),
    num_bins=st.sampled_from([10, 25, 100]),
    bin_sizing=st.sampled_from(["ev"]),
)
def test_lognorm_mean_error_propagation(norm_mean, norm_sd, num_bins, bin_sizing):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(
        dist, num_bins=num_bins, bin_sizing=bin_sizing
    )
    hist_base = ProbabilityMassHistogram.from_distribution(
        dist, num_bins=num_bins, bin_sizing=bin_sizing
    )

    for i in range(1, 13):
        true_mean = stats.lognorm.mean(np.sqrt(i) * norm_sd, scale=np.exp(i * norm_mean))
        assert all(hist.values[:-1] <= hist.values[1:]), f"On iteration {i}: {hist.values}"
        assert hist.histogram_mean() == approx(true_mean), f"On iteration {i}"
        hist = hist * hist_base


@given(bin_sizing=st.sampled_from(["ev"]))
def test_lognorm_sd_error_propagation(bin_sizing):
    verbose = False
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    num_bins = 100
    hist = ProbabilityMassHistogram.from_distribution(
        dist, num_bins=num_bins, bin_sizing=bin_sizing
    )
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
    pmhs = [ProbabilityMassHistogram.from_distribution(dist) for dist in dists]
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
@example(norm_mean1=0, norm_mean2=-3, norm_sd1=0.5, norm_sd2=0.5, num_bins1=25, num_bins2=25, bin_sizing='uniform')
def test_norm_sum(norm_mean1, norm_mean2, norm_sd1, norm_sd2, num_bins1, num_bins2, bin_sizing):
    dist1 = NormalDistribution(mean=norm_mean1, sd=norm_sd1)
    dist2 = NormalDistribution(mean=norm_mean2, sd=norm_sd2)
    hist1 = ProbabilityMassHistogram.from_distribution(
        dist1, num_bins=num_bins1, bin_sizing=bin_sizing
    )
    hist2 = ProbabilityMassHistogram.from_distribution(
        dist2, num_bins=num_bins2, bin_sizing=bin_sizing
    )
    hist_sum = hist1 + hist2

    # The further apart the means are, the less accurate the SD estimate is
    distance_apart = abs(norm_mean1 - norm_mean2) / hist_sum.exact_sd
    sd_tolerance = 2 + 0.5 * distance_apart

    assert all(hist_sum.values[:-1] <= hist_sum.values[1:])
    assert hist_sum.histogram_mean() == approx(hist_sum.exact_mean, abs=1e-10, rel=1e-5)
    assert hist_sum.histogram_sd() == approx(hist_sum.exact_sd, rel=sd_tolerance)


@given(
    norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    norm_sd1=st.floats(min_value=0.1, max_value=3),
    norm_sd2=st.floats(min_value=0.01, max_value=3),
    num_bins1=st.sampled_from([25, 100]),
    num_bins2=st.sampled_from([25, 100]),
)
def test_lognorm_sum(norm_mean1, norm_mean2, norm_sd1, norm_sd2, num_bins1, num_bins2):
    dist1 = LognormalDistribution(norm_mean=norm_mean1, norm_sd=norm_sd1)
    dist2 = LognormalDistribution(norm_mean=norm_mean2, norm_sd=norm_sd2)
    hist1 = ProbabilityMassHistogram.from_distribution(dist1, num_bins=num_bins1)
    hist2 = ProbabilityMassHistogram.from_distribution(dist2, num_bins=num_bins2)
    hist_sum = hist1 + hist2
    assert all(hist_sum.values[:-1] <= hist_sum.values[1:]), hist_sum.values
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
    hist1 = ProbabilityMassHistogram.from_distribution(dist1, num_bins=num_bins1)
    hist2 = ProbabilityMassHistogram.from_distribution(dist2, num_bins=num_bins2)
    hist_sum = hist1 + hist2
    sd_tolerance = 0.5
    assert all(hist_sum.values[:-1] <= hist_sum.values[1:]), hist_sum.values
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
    hists = [ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins) for dist in dists]
    hist = reduce(lambda acc, hist: acc * hist, hists)
    dist_abs_error = abs(hist.histogram_sd() - hist.exact_sd)

    mc_abs_error = []
    for i in range(10):
        mcs = [samplers.sample(dist, num_samples) for dist in dists]
        mc = reduce(lambda acc, mc: acc * mc, mcs)
        mc_abs_error.append(abs(np.std(mc) - hist.exact_sd))

    mc_abs_error.sort()

    # dist should be more accurate than at least 7 out of 10 Monte Carlo runs.
    # it's often more accurate than 10/10, but MC sometimes wins a few due to
    # random variation
    assert dist_abs_error < mc_abs_error[7]


def test_lognorm_product_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 16 distributions together."""
    num_bins = 100
    num_samples = 100**2
    dists = [LognormalDistribution(norm_mean=i, norm_sd=0.5 + i / 4) for i in range(17)]
    hists = [ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins) for dist in dists]
    hist = reduce(lambda acc, hist: acc * hist, hists)
    dist_abs_error = abs(hist.histogram_sd() - hist.exact_sd)

    mc_abs_error = []
    for i in range(10):
        mcs = [samplers.sample(dist, num_samples) for dist in dists]
        mc = reduce(lambda acc, mc: acc * mc, mcs)
        mc_abs_error.append(abs(np.std(mc) - hist.exact_sd))

    mc_abs_error.sort()

    # dist should be more accurate than at least 8 out of 10 Monte Carlo runs
    assert dist_abs_error < mc_abs_error[8]


@given(bin_sizing=st.sampled_from(["ev", "uniform"]))
def test_norm_sum_sd_accuracy_vs_monte_carlo(bin_sizing):
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 8 distributions together.

    Note: With more multiplications, MC has a good chance of being more
    accurate, and is significantly more accurate at 16 multiplications.
    """
    num_bins = 100
    num_samples = 100**2
    dists = [NormalDistribution(mean=i, sd=0.5 + i / 4) for i in range(9)]
    hists = [
        ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins, bin_sizing=bin_sizing)
        for dist in dists
    ]
    hist = reduce(lambda acc, hist: acc + hist, hists)
    dist_abs_error = abs(hist.histogram_sd() - hist.exact_sd)

    mc_abs_error = []
    for i in range(10):
        mcs = [samplers.sample(dist, num_samples) for dist in dists]
        mc = reduce(lambda acc, mc: acc + mc, mcs)
        mc_abs_error.append(abs(np.std(mc) - hist.exact_sd))

    mc_abs_error.sort()

    # dist should be more accurate than at least 8 out of 10 Monte Carlo runs
    assert dist_abs_error < mc_abs_error[8]


def test_lognorm_sum_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 16 distributions together."""
    num_bins = 100
    num_samples = 100**2
    dists = [LognormalDistribution(norm_mean=i, norm_sd=0.5 + i / 4) for i in range(17)]
    hists = [ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins) for dist in dists]
    hist = reduce(lambda acc, hist: acc + hist, hists)
    dist_abs_error = abs(hist.histogram_sd() - hist.exact_sd)

    mc_abs_error = []
    for i in range(10):
        mcs = [samplers.sample(dist, num_samples) for dist in dists]
        mc = reduce(lambda acc, mc: acc + mc, mcs)
        mc_abs_error.append(abs(np.std(mc) - hist.exact_sd))

    mc_abs_error.sort()

    # dist should be more accurate than at least 8 out of 10 Monte Carlo runs
    assert dist_abs_error < mc_abs_error[8]


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=1, max_value=99),
)
def test_pmh_contribution_to_ev(norm_mean, norm_sd, bin_num):
    fraction = bin_num / 100
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist)
    assert hist.contribution_to_ev(dist.inv_contribution_to_ev(fraction)) == approx(fraction)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=2, max_value=98),
)
def test_pmh_inv_contribution_to_ev(norm_mean, norm_sd, bin_num):
    # The nth value stored in the PMH represents a value between the nth and n+1th edges
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist)
    fraction = bin_num / hist.num_bins
    prev_fraction = fraction - 1 / hist.num_bins
    next_fraction = fraction
    assert hist.inv_contribution_to_ev(fraction) > dist.inv_contribution_to_ev(prev_fraction)
    assert hist.inv_contribution_to_ev(fraction) < dist.inv_contribution_to_ev(next_fraction)


def test_plot():
    return None
    hist = ProbabilityMassHistogram.from_distribution(
        LognormalDistribution(norm_mean=0, norm_sd=1)
    ) * ProbabilityMassHistogram.from_distribution(
        NormalDistribution(mean=0, sd=5)
    )
    # hist = ProbabilityMassHistogram.from_distribution(LognormalDistribution(norm_mean=0, norm_sd=2))
    hist.plot(scale="linear")


def test_performance():
    return None  # don't accidentally run this test because it's really slow
    import cProfile
    import pstats
    import io

    dist1 = NormalDistribution(mean=0, sd=1)
    dist2 = NormalDistribution(mean=0, sd=1)

    pr = cProfile.Profile()
    pr.enable()

    for i in range(100):
        hist1 = ProbabilityMassHistogram.from_distribution(dist1, num_bins=1000)
        hist2 = ProbabilityMassHistogram.from_distribution(dist2, num_bins=1000)
        for _ in range(4):
            hist1 = hist1 + hist2

    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
