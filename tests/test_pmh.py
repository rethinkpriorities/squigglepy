import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from scipy import stats

from ..squigglepy.distributions import LognormalDistribution
from ..squigglepy.pdh import ProbabilityMassHistogram

@given(
    norm_mean=st.floats(min_value=np.log(0.001), max_value=np.log(1000)),
    norm_sd=st.floats(min_value=0.01, max_value=np.log(10)),
)
def test_pmh_mean_equals_analytic_mean(norm_mean, norm_sd):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    assert ProbabilityMassHistogram.from_distribution(dist).mean() == pytest.approx(stats.lognorm.mean(dist.norm_sd, scale=np.exp(dist.norm_mean)))


def test_basic():
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    pmh = ProbabilityMassHistogram.from_distribution(dist)
    assert pmh.mean() == pytest.approx(stats.lognorm.mean(dist.norm_sd, scale=np.exp(dist.norm_mean)))
