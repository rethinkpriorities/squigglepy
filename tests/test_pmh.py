from functools import reduce
from hypothesis import assume, given, settings
import hypothesis.strategies as st
import numpy as np
from pytest import approx
from scipy import integrate, stats

from ..squigglepy.distributions import LognormalDistribution
from ..squigglepy.pdh import ProbabilityMassHistogram, ScaledBinHistogram
from ..squigglepy import samplers


def print_accuracy_ratio(x, y, extra_message=None):
    ratio = max(x / y, y / x) - 1
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
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=5),
)
def test_pmh_mean(norm_mean, norm_sd):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist)
    assert hist.histogram_mean() == approx(stats.lognorm.mean(dist.norm_sd, scale=np.exp(dist.norm_mean)))


@given(
    # norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    # norm_sd=st.floats(min_value=0.01, max_value=5),
    norm_mean=st.just(0),
    norm_sd=st.just(1),
)
def test_pmh_sd(norm_mean, norm_sd):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist)

    def true_variance(left, right):
        return integrate.quad(
            lambda x: (x - dist.lognorm_mean) ** 2
            * stats.lognorm.pdf(x, dist.norm_sd, scale=np.exp(dist.norm_mean)),
            left,
            right,
        )[0]

    def observed_variance(left, right):
        return np.sum(hist.masses[left:right] * (hist.values[left:right] - hist.histogram_mean()) ** 2)

    midpoint = hist.values[990]
    expected_left_variance = true_variance(0, midpoint)
    expected_right_variance = true_variance(midpoint, np.inf)
    midpoint_index = int(len(hist) * hist.fraction_of_ev(midpoint))
    observed_left_variance = observed_variance(0, midpoint_index)
    observed_right_variance = observed_variance(midpoint_index, len(hist))
    print_accuracy_ratio(observed_left_variance, expected_left_variance, "Left ")
    print_accuracy_ratio(observed_right_variance, expected_right_variance, "Right")
    assert hist.histogram_sd() == approx(dist.lognorm_sd)


def test_sd_error_propagation(verbose=True):
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    num_bins = 1000
    hist = ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins)
    hist_base = ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins)
    abs_error = []
    rel_error = []

    if verbose:
        print("")
    for i in [1, 2, 4, 8, 16, 32, 64]:
        true_mean = stats.lognorm.mean(np.sqrt(i))
        true_sd = hist.exact_sd
        abs_error.append(abs(hist.histogram_sd() - true_sd))
        rel_error.append(np.exp(abs(np.log(hist.histogram_sd() / true_sd))) - 1)
        if verbose:
            print(f"n = {i:2d}: {rel_error[-1]*100:4.1f}% from {hist.histogram_sd():.3f}, mean {hist.histogram_mean():.3f}")
        hist = hist * hist

    expected_error_pcts = [0.2, 0.6, 2.5, 13.2, 77.2, 751]
    for i in range(len(expected_error_pcts)):
        assert rel_error[i] < expected_error_pcts[i] / 100


@given(
    norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    norm_sd1=st.floats(min_value=0.1, max_value=3),
    norm_sd2=st.floats(min_value=0.001, max_value=3),
)
@settings(max_examples=100)
def test_exact_moments(norm_mean1, norm_mean2, norm_sd1, norm_sd2):
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
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=1, max_value=999),
)
def test_pmh_fraction_of_ev(norm_mean, norm_sd, bin_num):
    fraction = bin_num / 1000
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist)
    assert hist.fraction_of_ev(dist.inv_fraction_of_ev(fraction)) == approx(fraction)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=2, max_value=998),
)
def test_pmh_inv_fraction_of_ev(norm_mean, norm_sd, bin_num):
    # The nth value stored in the PMH represents a value between the nth and n+1th edges
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist)
    fraction = bin_num / hist.num_bins
    prev_fraction = fraction - 1 / hist.num_bins
    next_fraction = fraction
    assert hist.inv_fraction_of_ev(fraction) > dist.inv_fraction_of_ev(prev_fraction)
    assert hist.inv_fraction_of_ev(fraction) < dist.inv_fraction_of_ev(next_fraction)


# TODO: uncomment
# @given(
#     norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
#     norm_mean2=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
#     norm_sd1=st.floats(min_value=0.1, max_value=3),
#     norm_sd2=st.floats(min_value=0.1, max_value=3),
# )
# def test_lognorm_product_summary_stats(norm_mean1, norm_sd1, norm_mean2, norm_sd2):
def test_lognorm_product_summary_stats():
    # norm_means = np.repeat([0, 1, 1, 100], 4)
    # norm_sds = np.repeat([1, 0.7, 2, 0.1], 4)
    norm_means = np.repeat([0], 2)
    norm_sds = np.repeat([1], 2)
    dists = [
        LognormalDistribution(norm_mean=norm_means[i], norm_sd=norm_sds[i])
        for i in range(len(norm_means))
    ]
    dist_prod = LognormalDistribution(
        norm_mean=np.sum(norm_means), norm_sd=np.sqrt(np.sum(norm_sds**2))
    )
    pmhs = [ProbabilityMassHistogram.from_distribution(dist) for dist in dists]
    pmh_prod = reduce(lambda acc, hist: acc * hist, pmhs)
    print_accuracy_ratio(pmh_prod.histogram_sd(), dist_prod.lognorm_sd)
    assert pmh_prod.histogram_mean() == approx(dist_prod.lognorm_mean)
    assert pmh_prod.histogram_sd() == approx(dist_prod.lognorm_sd)


def test_lognorm_sample():
    # norm_means = np.repeat([0, 1, -1, 100], 4)
    # norm_sds = np.repeat([1, 0.7, 2, 0.1], 4)
    norm_means = np.repeat([0], 2)
    norm_sds = np.repeat([1], 2)
    dists = [
        LognormalDistribution(norm_mean=norm_means[i], norm_sd=norm_sds[i])
        for i in range(len(norm_means))
    ]
    dist_prod = LognormalDistribution(
        norm_mean=np.sum(norm_means), norm_sd=np.sqrt(np.sum(norm_sds**2))
    )
    num_samples = 1e6
    sample_lists = [samplers.sample(dist, num_samples) for dist in dists]
    samples = np.product(sample_lists, axis=0)
    print_accuracy_ratio(np.std(samples), dist_prod.lognorm_sd)
    assert np.std(samples) == approx(dist_prod.lognorm_sd)


def test_scaled_bin():
    for repetitions in [1, 4, 8, 16]:
        norm_means = np.repeat([0], repetitions)
        norm_sds = np.repeat([1], repetitions)
        dists = [
            LognormalDistribution(norm_mean=norm_means[i], norm_sd=norm_sds[i])
            for i in range(len(norm_means))
        ]
        dist_prod = LognormalDistribution(
            norm_mean=np.sum(norm_means), norm_sd=np.sqrt(np.sum(norm_sds**2))
        )
        hists = [ScaledBinHistogram.from_distribution(dist) for dist in dists]
        hist_prod = reduce(lambda acc, hist: acc * hist, hists)
        print("")
        print_accuracy_ratio(hist_prod.histogram_mean(), dist_prod.lognorm_mean, "Mean")
        print_accuracy_ratio(hist_prod.histogram_sd(), dist_prod.lognorm_sd, "SD  ")


def test_accuracy_scaled_vs_flexible():
    for repetitions in [1, 4, 8, 16]:
        norm_means = np.repeat([0], repetitions)
        norm_sds = np.repeat([1], repetitions)
        dists = [
            LognormalDistribution(norm_mean=norm_means[i], norm_sd=norm_sds[i])
            for i in range(len(norm_means))
        ]
        dist_prod = LognormalDistribution(
            norm_mean=np.sum(norm_means), norm_sd=np.sqrt(np.sum(norm_sds**2))
        )
        scaled_hists = [ScaledBinHistogram.from_distribution(dist) for dist in dists]
        scaled_hist_prod = reduce(lambda acc, hist: acc * hist, scaled_hists)
        flexible_hists = [ProbabilityMassHistogram.from_distribution(dist) for dist in dists]
        flexible_hist_prod = reduce(lambda acc, hist: acc * hist, flexible_hists)
        scaled_mean_error = abs(scaled_hist_prod.histogram_mean() - dist_prod.lognorm_mean)
        flexible_mean_error = abs(flexible_hist_prod.histogram_mean() - dist_prod.lognorm_mean)
        scaled_sd_error = abs(scaled_hist_prod.histogram_sd() - dist_prod.lognorm_sd)
        flexible_sd_error = abs(flexible_hist_prod.histogram_sd() - dist_prod.lognorm_sd)
        assert scaled_mean_error > flexible_mean_error
        assert scaled_sd_error > flexible_sd_error
        print("")
        print(
            f"Mean error: scaled = {scaled_mean_error:.3f}, flexible = {flexible_mean_error:.3f}"
        )
        print(f"SD   error: scaled = {scaled_sd_error:.3f}, flexible = {flexible_sd_error:.3f}")


def test_performance():
    import cProfile
    import pstats
    import io

    dist = LognormalDistribution(norm_mean=0, norm_sd=1)

    pr = cProfile.Profile()
    pr.enable()

    for i in range(10):
        hist = ProbabilityMassHistogram.from_distribution(dist)
        for _ in range(4):
            hist = hist * hist

    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
