import hypothesis.strategies as st
import numpy as np
import pytest
from pytest import approx
from scipy import stats
import warnings
from hypothesis import assume, given, settings

from ..squigglepy.distributions import LognormalDistribution, NormalDistribution
from ..squigglepy.utils import ConvergenceWarning


@given(
    norm_mean=st.floats(min_value=np.log(0.01), max_value=np.log(1e6)),
    norm_sd=st.floats(min_value=0.1, max_value=2.5),
    ev_fraction=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=1000)
def test_lognorm_inv_contribution_to_ev_inverts_contribution_to_ev(
    norm_mean, norm_sd, ev_fraction
):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    assert dist.contribution_to_ev(dist.inv_contribution_to_ev(ev_fraction)) == approx(
        ev_fraction, 2e-5 / ev_fraction
    )


def test_lognorm_basic():
    dist = LognormalDistribution(lognorm_mean=2, lognorm_sd=1.0)
    ev_fraction = 0.25
    assert dist.contribution_to_ev(dist.inv_contribution_to_ev(ev_fraction)) == approx(ev_fraction)


@given(
    mu=st.floats(min_value=-10, max_value=10),
    sigma=st.floats(min_value=0.01, max_value=100),
)
def test_norm_contribution_to_ev(mu, sigma):
    dist = NormalDistribution(mean=mu, sd=sigma)

    assert dist.contribution_to_ev(mu + 99 * sigma) == approx(1)
    assert dist.contribution_to_ev(mu - 99 * sigma) == approx(0)

    # midpoint represents less than half the EV if mu > 0 b/c the larger
    # values are weighted heavier, and vice versa if mu < 0
    if mu > 1e-6:
        assert dist.contribution_to_ev(mu) < 0.5
    elif mu < -1e-6:
        assert dist.contribution_to_ev(mu) > 0.5
    elif mu == 0:
        assert dist.contribution_to_ev(mu) == approx(0.5)

    # contribution_to_ev should be monotonic
    assert dist.contribution_to_ev(mu - 2 * sigma) < dist.contribution_to_ev(mu - 1 * sigma)
    assert dist.contribution_to_ev(mu -  sigma) < dist.contribution_to_ev(mu)
    assert dist.contribution_to_ev(mu) < dist.contribution_to_ev(mu + sigma)
    assert dist.contribution_to_ev(mu + sigma) < dist.contribution_to_ev(mu + 2 * sigma)


@given(
    mu=st.floats(min_value=-10, max_value=10),
    sigma=st.floats(min_value=0.01, max_value=10),
)
def test_norm_inv_contribution_to_ev(mu, sigma):
    dist = NormalDistribution(mean=mu, sd=sigma)

    assert dist.inv_contribution_to_ev(1 - 1e-9) > mu + 3 * sigma
    assert dist.inv_contribution_to_ev(1e-9) < mu - 3 * sigma

    # midpoint represents less than half the EV if mu > 0 b/c the larger
    # values are weighted heavier, and vice versa if mu < 0
    if mu > 1e-6:
        assert dist.inv_contribution_to_ev(0.5) > mu
    elif mu < -1e-6:
        assert dist.inv_contribution_to_ev(0.5) < mu
    elif mu == 0:
        assert dist.inv_contribution_to_ev(0.5) == approx(mu)

    # inv_contribution_to_ev should be monotonic
    assert dist.inv_contribution_to_ev(0.05) < dist.inv_contribution_to_ev(0.25)
    assert dist.inv_contribution_to_ev(0.25) < dist.inv_contribution_to_ev(0.5)
    assert dist.inv_contribution_to_ev(0.5) < dist.inv_contribution_to_ev(0.75)
    assert dist.inv_contribution_to_ev(0.75) < dist.inv_contribution_to_ev(0.95)


@given(
    mu=st.floats(min_value=-10, max_value=10),
    sigma=st.floats(min_value=0.01, max_value=10),
    ev_fraction=st.floats(min_value=0.0001, max_value=0.9999),
)
def test_norm_inv_contribution_to_ev_inverts_contribution_to_ev(mu, sigma, ev_fraction):
    dist = NormalDistribution(mean=mu, sd=sigma)
    assert dist.contribution_to_ev(dist.inv_contribution_to_ev(ev_fraction)) == approx(ev_fraction, abs=1e-8)
