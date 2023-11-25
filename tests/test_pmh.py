from functools import reduce
from hypothesis import assume, given, settings
import hypothesis.strategies as st
import numpy as np
from pytest import approx
from scipy import integrate, stats

from ..squigglepy.distributions import LognormalDistribution, NormalDistribution
from ..squigglepy.pdh import ProbabilityMassHistogram
from ..squigglepy import samplers


def relative_error(x, y):
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
    norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    norm_sd1=st.floats(min_value=0.1, max_value=3),
    norm_sd2=st.floats(min_value=0.001, max_value=3),
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
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=5),
    bin_sizing=st.sampled_from(["ev", "mass"]),
)
def test_lognorm_mean(norm_mean, norm_sd, bin_sizing):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist, bin_sizing=bin_sizing)
    assert hist.histogram_mean() == approx(
        stats.lognorm.mean(dist.norm_sd, scale=np.exp(dist.norm_mean))
    )


@given(
    mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    sd=st.floats(min_value=0.001, max_value=100),
)
def test_norm_with_ev_bins(mean, sd):
    dist = NormalDistribution(mean=mean, sd=sd)
    hist = ProbabilityMassHistogram.from_distribution(dist, bin_sizing="ev")
    assert hist.histogram_mean() == approx(mean)
    assert hist.histogram_sd() == approx(sd, rel=0.01)


@given(
    mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    sd=st.floats(min_value=0.001, max_value=100),
)
def test_norm_with_mass_bins(mean, sd):
    dist = NormalDistribution(mean=mean, sd=sd)
    hist = ProbabilityMassHistogram.from_distribution(dist, bin_sizing="mass")
    assert hist.histogram_mean() == approx(mean)
    assert hist.histogram_sd() == approx(sd, rel=0.01)


@given(
    # norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    # norm_sd=st.floats(min_value=0.01, max_value=5),
    norm_mean=st.just(0),
    norm_sd=st.just(1),
)
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
    print("")
    print_accuracy_ratio(observed_left_variance, expected_left_variance, "Left   ")
    print_accuracy_ratio(observed_right_variance, expected_right_variance, "Right  ")
    print_accuracy_ratio(hist.histogram_sd(), dist.lognorm_sd, "Overall")
    assert hist.histogram_sd() == approx(dist.lognorm_sd, rel=0.05)


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


def test_noncentral_norm_product():
    dist_pairs = [
        (NormalDistribution(mean=0, sd=1), NormalDistribution(mean=0, sd=1)),
        (NormalDistribution(mean=2, sd=1), NormalDistribution(mean=-1, sd=2)),
    ]

    for dist1, dist2 in dist_pairs:
        hist1 = ProbabilityMassHistogram.from_distribution(dist1, num_bins=25)
        hist2 = ProbabilityMassHistogram.from_distribution(dist2, num_bins=25)
        hist_prod = hist1 * hist2
        assert hist_prod.histogram_mean() == approx(dist1.mean * dist2.mean)
        assert hist_prod.histogram_sd() == approx(
            np.sqrt(
                (dist1.sd**2 + dist1.mean**2) * (dist2.sd**2 + dist2.mean**2)
                - dist1.mean**2 * dist2.mean**2
            ),
            rel=0.25,
        )


@given(
    mean=st.floats(min_value=-10, max_value=10),
    sd=st.floats(min_value=0.001, max_value=100),
    num_bins=st.sampled_from([25, 100]),
    # "mass" sizing is just really bad given how it's currently implemented. it
    # does weird stuff like with mean=-20, sd=13, after only a few
    # multiplications, most bin values are 0
    bin_sizing=st.sampled_from(["ev"]),
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
    tolerance = 1e-12

    for i in range(1, 17):
        true_mean = mean**i
        true_sd = np.sqrt((dist.sd**2 + dist.mean**2) ** i - dist.mean ** (2 * i))
        if true_sd > 1e15:
            break
        assert hist.histogram_mean() == approx(
            true_mean, abs=tolerance ** (1 / i), rel=tolerance ** (1 / i)
        ), f"On iteration {i}"
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


# 2300 extended values, 93 bins
@given(
    mean1=st.floats(min_value=-100, max_value=100),
    mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    sd1=st.floats(min_value=0.001, max_value=100),
    sd2=st.floats(min_value=0.001, max_value=3),
    num_bins1=st.sampled_from([25, 100]),
    num_bins2=st.sampled_from([25, 100]),
)
def test_norm_lognorm_product(mean1, mean2, sd1, sd2, num_bins1, num_bins2):
    def callback(err, flag):
        import ipdb; ipdb.set_trace()
    np.seterrcall(callback)
    dist1 = NormalDistribution(mean=mean1, sd=sd1)
    dist2 = LognormalDistribution(norm_mean=mean2, norm_sd=sd2)
    hist1 = ProbabilityMassHistogram.from_distribution(dist1, num_bins=num_bins1)
    hist2 = ProbabilityMassHistogram.from_distribution(dist2, num_bins=num_bins2)
    hist_prod = hist1 * hist2
    assert all(hist_prod.values[:-1] <= hist_prod.values[1:]), hist_prod.values
    assert hist_prod.histogram_mean() == approx(hist_prod.exact_mean, abs=1e-6, rel=1e-6)

    # SD is pretty inaccurate
    assert relative_error(hist_prod.histogram_sd(), hist_prod.exact_sd) < 2


def test_norm_sd_accuracy_vs_monte_carlo():
    """Test that PMH SD is more accurate than Monte Carlo SD both for initial
    distributions and when multiplying up to 8 distributions together.

    Note: With more multiplications, MC has a good chance of being more
    accurate, and is significantly more accurate at 16 multiplications.
    """
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

    # dist should be more accurate than at least 8 out of 10 Monte Carlo runs
    assert dist_abs_error < mc_abs_error[8]


def test_lognorm_sd_accuracy_vs_monte_carlo():
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
    norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    norm_sd1=st.floats(min_value=0.1, max_value=3),
    norm_sd2=st.floats(min_value=0.001, max_value=3),
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
    # norm_mean1=st.floats(-1e6, 1e6),
    # norm_mean2=st.floats(min_value=-1e6, max_value=1e6),
    # norm_sd1=st.floats(min_value=0.1, max_value=100),
    # norm_sd2=st.floats(min_value=0.001, max_value=100),
    # num_bins1=st.sampled_from([25, 100]),
    # num_bins2=st.sampled_from([25, 100]),
    norm_mean1=st.just(0),
    norm_mean2=st.just(-218),
    norm_sd1=st.just(1),
    norm_sd2=st.just(1),
    num_bins1=st.just(25),
    num_bins2=st.just(25),
)
def test_norm_sum(norm_mean1, norm_mean2, norm_sd1, norm_sd2, num_bins1, num_bins2):
    dist1 = NormalDistribution(mean=norm_mean1, sd=norm_sd1)
    dist2 = NormalDistribution(mean=norm_mean2, sd=norm_sd2)
    hist1 = ProbabilityMassHistogram.from_distribution(dist1, num_bins=num_bins1)
    hist2 = ProbabilityMassHistogram.from_distribution(dist2, num_bins=num_bins2)
    hist_sum = hist1 + hist2
    assert all(hist_sum.values[:-1] <= hist_sum.values[1:])
    assert hist_sum.histogram_mean() == approx(hist_sum.exact_mean)
    assert hist_sum.histogram_sd() == approx(hist_sum.exact_sd, rel=2)


@given(
    # mean1=st.floats(min_value=-100, max_value=100),
    # mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    # sd1=st.floats(min_value=0.001, max_value=100),
    # sd2=st.floats(min_value=0.001, max_value=3),
    # num_bins1=st.sampled_from([25, 100]),
    # num_bins2=st.sampled_from([25, 100]),
    mean1=st.just(1.5),
    mean2=st.just(7),
    sd1=st.just(100),
    sd2=st.just(1),
    num_bins1=st.just(100),
    num_bins2=st.just(100),
)
def test_norm_lognorm_sum(mean1, mean2, sd1, sd2, num_bins1, num_bins2):
    dist1 = NormalDistribution(mean=mean1, sd=sd1)
    dist2 = LognormalDistribution(norm_mean=mean2, norm_sd=sd2)
    hist1 = ProbabilityMassHistogram.from_distribution(dist1, num_bins=num_bins1)
    hist2 = ProbabilityMassHistogram.from_distribution(dist2, num_bins=num_bins2)
    hist_sum = hist1 + hist2
    assert all(hist_sum.values[:-1] <= hist_sum.values[1:]), hist_sum.values
    assert hist_sum.histogram_mean() == approx(hist_sum.exact_mean, abs=1e-6, rel=1e-6)
    assert hist_sum.histogram_sd() == approx(hist_sum.exact_sd, rel=1)


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


def test_performance():
    return None  # so we don't accidentally run this while running all tests
    import cProfile
    import pstats
    import io

    dist = LognormalDistribution(norm_mean=0, norm_sd=1)

    pr = cProfile.Profile()
    pr.enable()

    for i in range(100):
        hist = ProbabilityMassHistogram.from_distribution(dist, num_bins=1000)
        for _ in range(4):
            hist = hist * hist

    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())