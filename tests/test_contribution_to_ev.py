import hypothesis.strategies as st
import numpy as np
import pytest
from pytest import approx
from scipy import stats
import warnings
from hypothesis import assume, example, given, settings

from ..squigglepy.distributions import (
    BetaDistribution,
    GammaDistribution,
    LognormalDistribution,
    NormalDistribution,
    UniformDistribution,
)
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
    assert dist.contribution_to_ev(mu - sigma) < dist.contribution_to_ev(mu)
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
    assert dist.contribution_to_ev(dist.inv_contribution_to_ev(ev_fraction)) == approx(
        ev_fraction, abs=1e-8
    )


def test_uniform_contribution_to_ev_basic():
    dist = UniformDistribution(-1, 1)
    assert dist.contribution_to_ev(-1) == approx(0)
    assert dist.contribution_to_ev(1) == approx(1)
    assert dist.contribution_to_ev(0) == approx(0.5)


@given(prop=st.floats(min_value=0, max_value=1))
def test_standard_uniform_contribution_to_ev(prop):
    dist = UniformDistribution(0, 1)
    assert dist.contribution_to_ev(prop) == approx(prop)


@given(
    a=st.floats(min_value=-10, max_value=10),
    b=st.floats(min_value=-10, max_value=10),
)
def test_uniform_contribution_to_ev(a, b):
    if a > b:
        a, b = b, a
    if abs(a - b) < 1e-20:
        return None
    dist = UniformDistribution(x=a, y=b)
    assert dist.contribution_to_ev(a) == approx(0)
    assert dist.contribution_to_ev(b) == approx(1)
    assert dist.contribution_to_ev(a - 1) == approx(0)
    assert dist.contribution_to_ev(b + 1) == approx(1)

    assert dist.contribution_to_ev(a, normalized=False) == approx(0)
    if not (a < 0 and b > 0):
        assert dist.contribution_to_ev(b, normalized=False) == approx(abs(a + b) / 2)
    else:
        total_contribution = (a**2 + b**2) / 2 / (b - a)
        assert dist.contribution_to_ev(b, normalized=False) == approx(total_contribution)


@given(
    a=st.floats(min_value=-10, max_value=10),
    b=st.floats(min_value=-10, max_value=10),
)
def test_uniform_inv_contribution_to_ev(a, b):
    if a > b:
        a, b = b, a
    if abs(a - b) < 1e-20:
        return None
    dist = UniformDistribution(x=a, y=b)
    assert dist.inv_contribution_to_ev(0) == approx(a)
    assert dist.inv_contribution_to_ev(1) == approx(b)
    assert dist.inv_contribution_to_ev(0.25) == approx((a + b) / 2)


@given(
    a=st.floats(min_value=-10, max_value=10),
    b=st.floats(min_value=-10, max_value=10),
    prop=st.floats(min_value=0, max_value=1),
)
def test_uniform_inv_contribution_to_ev_inverts_contribution_to_ev(a, b, prop):
    if a > b:
        a, b = b, a
    if abs(a - b) < 1e-20:
        return None
    dist = UniformDistribution(x=a, y=b)
    assert dist.contribution_to_ev(dist.inv_contribution_to_ev(prop)) == approx(prop)


@given(
    ab=st.floats(min_value=0.01, max_value=10),
)
@example(ab=1)
def test_beta_contribution_to_ev_basic(ab):
    dist = BetaDistribution(ab, ab)
    assert dist.contribution_to_ev(0) == approx(0)
    assert dist.contribution_to_ev(1) == approx(1)
    assert dist.contribution_to_ev(1, normalized=False) == approx(0.5)
    assert dist.contribution_to_ev(0.5) > 0
    assert dist.contribution_to_ev(0.5) <= 0.5


@given(
    a=st.floats(min_value=0.5, max_value=10),
    b=st.floats(min_value=0.5, max_value=10),
    fraction=st.floats(min_value=0, max_value=1),
)
def test_beta_inv_contribution_ev_inverts_contribution_to_ev(a, b, fraction):
    # Note: The answers do become a bit off for small fractional values of a, b
    dist = BetaDistribution(a, b)
    tolerance = 1e-6 if a < 1 or b < 1 else 1e-8
    assert dist.contribution_to_ev(dist.inv_contribution_to_ev(fraction)) == approx(
        fraction, rel=tolerance
    )


@given(
    shape=st.floats(min_value=0.1, max_value=100),
    scale=st.floats(min_value=0.1, max_value=100),
    x=st.floats(min_value=0, max_value=100),
)
def test_gamma_contribution_to_ev_basic(shape, scale, x):
    dist = GammaDistribution(shape, scale)
    assert dist.contribution_to_ev(0) == approx(0)
    assert dist.contribution_to_ev(1) < dist.contribution_to_ev(2) or (
        dist.contribution_to_ev(1) == 0 and dist.contribution_to_ev(2) == 0
    )
    if shape * scale <= 1:
        assert dist.contribution_to_ev(shape * scale, normalized=False) < stats.gamma.cdf(
            shape * scale, shape, scale=scale
        )
    assert dist.contribution_to_ev(100 * shape * scale, normalized=False) == approx(
        shape * scale, rel=1e-3
    )


@given(
    shape=st.floats(min_value=0.1, max_value=100),
    scale=st.floats(min_value=0.1, max_value=100),
    fraction=st.floats(min_value=0, max_value=1 - 1e-6),
)
@example(shape=1, scale=2, fraction=0.5)
def test_gamma_inv_contribution_ev_inverts_contribution_to_ev(shape, scale, fraction):
    dist = GammaDistribution(shape, scale)
    tolerance = 1e-6 if shape < 1 or scale < 1 else 1e-8
    assert dist.contribution_to_ev(dist.inv_contribution_to_ev(fraction)) == approx(
        fraction, rel=tolerance
    )
