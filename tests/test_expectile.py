import hypothesis.strategies as st
import numpy as np
import pytest
import warnings
from hypothesis import assume, given, settings

from ..squigglepy.distributions import LognormalDistribution
from ..squigglepy.utils import ConvergenceWarning


@given(
    norm_mean=st.floats(min_value=np.log(0.01), max_value=np.log(1e6)),
    norm_sd=st.floats(min_value=0.1, max_value=2.5),
    ev_quantile=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=1000)
def test_inv_fraction_of_ev_inverts_fraction_of_ev(
    norm_mean, norm_sd, ev_quantile
):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    assert dist.fraction_of_ev(
        dist.inv_fraction_of_ev(ev_quantile)
    ) == pytest.approx(ev_quantile, 2e-5 / ev_quantile)


def test_basic():
    dist = LognormalDistribution(lognorm_mean=2, lognorm_sd=1.0)
    ev_quantile = 0.25
    assert dist.fraction_of_ev(
        dist.inv_fraction_of_ev(ev_quantile)
    ) == pytest.approx(ev_quantile)
