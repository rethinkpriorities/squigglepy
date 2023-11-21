import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given, settings
from pytest import approx
from scipy import stats

from ..squigglepy.distributions import LognormalDistribution
from ..squigglepy.pdh import ProbabilityMassHistogram
from ..squigglepy import samplers


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=5),
)
def test_pmh_mean(norm_mean, norm_sd):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    pmh = ProbabilityMassHistogram.from_distribution(dist)
    assert pmh.mean() == approx(stats.lognorm.mean(dist.norm_sd, scale=np.exp(dist.norm_mean)))


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=1, max_value=999),
)
def test_pmh_fraction_of_ev(norm_mean, norm_sd, bin_num):
    fraction = bin_num / 1000
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    pmh = ProbabilityMassHistogram.from_distribution(dist)
    assert pmh.fraction_of_ev(dist.inv_fraction_of_ev(fraction)) == approx(fraction)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=2, max_value=998),
)
def test_pmh_inv_fraction_of_ev(norm_mean, norm_sd, bin_num):
    # The nth value stored in the PMH represents a value between the nth and n+1th edges
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    pmh = ProbabilityMassHistogram.from_distribution(dist)
    fraction = bin_num / pmh.num_bins
    prev_fraction = fraction - 1 / pmh.num_bins
    next_fraction = fraction
    assert pmh.inv_fraction_of_ev(fraction) > dist.inv_fraction_of_ev(prev_fraction)
    assert pmh.inv_fraction_of_ev(fraction) < dist.inv_fraction_of_ev(next_fraction)


# @given(
#     norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
#     norm_mean2=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
#     norm_sd1=st.floats(min_value=0.1, max_value=3),
#     norm_sd2=st.floats(min_value=0.1, max_value=3),
# )
# @settings(max_examples=1)
# def test_lognorm_product_summary_stats(norm_mean1, norm_sd1, norm_mean2, norm_sd2):
def test_lognorm_product_summary_stats():
    norm_mean1 = 0
    norm_sd1 = 1
    norm_mean2 = 1
    norm_sd2 = 0.7
    dist1 = LognormalDistribution(norm_mean=norm_mean1, norm_sd=norm_sd1)
    dist2 = LognormalDistribution(norm_mean=norm_mean2, norm_sd=norm_sd2)
    dist_prod = LognormalDistribution(
        norm_mean=norm_mean1 + norm_mean2, norm_sd=np.sqrt(norm_sd1**2 + norm_sd2**2)
    )
    pmh1 = ProbabilityMassHistogram.from_distribution(dist1)
    pmh2 = ProbabilityMassHistogram.from_distribution(dist2)
    pmh_prod = pmh1 * pmh2
    print("Ratio:", pmh_prod.std() / dist_prod.lognorm_sd - 1)
    assert pmh_prod.histogram_mean() == approx(dist_prod.lognorm_mean)
    assert pmh_prod.std() == approx(dist_prod.lognorm_sd)

def test_lognorm_sample():
    dist1 = LognormalDistribution(norm_mean=0, norm_sd=1)
    dist2 = LognormalDistribution(norm_mean=1, norm_sd=0.7)
    dist_prod = LognormalDistribution(
        norm_mean=1, norm_sd=np.sqrt(1 + 0.7**2)
    )
    num_samples = 1e6
    samples1 = samplers.sample(dist1, num_samples)
    samples2 = samplers.sample(dist2, num_samples)
    samples = samples1 * samples2
    print("Ratio:", np.std(samples) / dist_prod.lognorm_sd - 1)
    assert np.std(samples) == approx(dist_prod.lognorm_sd)
