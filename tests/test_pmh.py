import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given
from pytest import approx
from scipy import stats

from ..squigglepy.distributions import LognormalDistribution
from ..squigglepy.pdh import ProbabilityMassHistogram

@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=5),
)
def test_pmh_mean_equals_analytic(norm_mean, norm_sd):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    pmh = ProbabilityMassHistogram.from_distribution(dist)
    assert pmh.mean() == approx(stats.lognorm.mean(dist.norm_sd, scale=np.exp(dist.norm_mean)))


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=1, max_value=999),
)
def test_pmh_fraction_of_ev_equals_analytic(norm_mean, norm_sd, bin_num):
    fraction = bin_num / 1000
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    pmh = ProbabilityMassHistogram.from_distribution(dist)
    assert pmh.fraction_of_ev(dist.inv_fraction_of_ev(fraction)) == approx(fraction)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=2, max_value=998),
)
def test_pmh_inv_fraction_of_ev_approximates_analytic(norm_mean, norm_sd, bin_num):
    # The nth value stored in the PMH represents a value between the nth and n+1th edges
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    pmh = ProbabilityMassHistogram.from_distribution(dist)
    fraction = bin_num / pmh.num_bins
    prev_fraction = fraction - 1 / pmh.num_bins
    next_fraction = fraction
    assert pmh.inv_fraction_of_ev(fraction) > dist.inv_fraction_of_ev(prev_fraction)
    assert pmh.inv_fraction_of_ev(fraction) < dist.inv_fraction_of_ev(next_fraction)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=1, max_value=999),
)
def test_pmh_inv_fraction_of_ev_inverts_fraction_of_ev(norm_mean, norm_sd, bin_num):
    fraction = bin_num / 1000
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    pmh = ProbabilityMassHistogram.from_distribution(dist)
    # assert pmh.fraction_of_ev(pmh.inv_fraction_of_ev(fraction)) == approx(fraction)
