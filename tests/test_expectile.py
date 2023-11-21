import hypothesis.strategies as st
import numpy as np
import pytest
import warnings
from hypothesis import assume, given, settings

from ..squigglepy.distributions import LognormalDistribution
from ..squigglepy.utils import ConvergenceWarning


@given(
    norm_mean=st.floats(min_value=-10, max_value=10),
    norm_sd=st.floats(min_value=0.01, max_value=10),
)
def test_inv_expectile_gives_correct_mean(norm_mean, norm_sd):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    assert dist.inv_expectile(
        np.exp(norm_mean + norm_sd**2 / 2)
    ) == pytest.approx(0.5)


@given(
    norm_mean=st.floats(min_value=-100, max_value=100),
    norm_sd=st.floats(min_value=0.01, max_value=5),
)
def test_expectile_gives_correct_mean(norm_mean, norm_sd):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    assert dist.expectile(0.5) == pytest.approx(
        np.exp(norm_mean + norm_sd**2 / 2)
    )


@given(
    lognorm_mean=st.floats(min_value=0.01, max_value=100),
    lognorm_sd=st.floats(min_value=0.001, max_value=100),
    expectile=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=1000)
def test_inv_expectile_inverts_expectile(lognorm_mean, lognorm_sd, expectile):
    dist = LognormalDistribution(lognorm_mean=lognorm_mean, lognorm_sd=lognorm_sd)
    with warnings.catch_warnings(record=True) as w:
        found = dist.inv_expectile(dist.expectile(expectile))
        if len(w) > 0 and isinstance(w, ConvergenceWarning):
            diff = float(str(w[0].message).rsplit(' ', 1)[1])
            assert found == pytest.approx(expectile, diff)
        elif expectile < 0.95:
            assert found == pytest.approx(expectile, 1e-5)
        else:
            assert found == pytest.approx(expectile, 1e-3)

def test_expectile_specific_values():
    # I got these specific values from the R package expectreg. Set
    # max_iter=1000 because that's what expectreg uses by default, so this
    # should guarantee that both values are the same: if they're inaccurate,
    # they'll both be inaccurate by the same amount (assuming R and Python
    # represent floats the same way, etc.).
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    assert dist.expectile(0.0001, max_iter=1000) == pytest.approx(0.08657134)
    assert dist.expectile(0.001, max_iter=1000) == pytest.approx(0.158501)
    assert dist.expectile(0.01, max_iter=1000) == pytest.approx(0.3166604)
    assert dist.expectile(0.1, max_iter=1000) == pytest.approx(0.7209488)
    assert dist.expectile(0.5, max_iter=1000) == pytest.approx(1.648721)
    assert dist.expectile(0.9, max_iter=1000) == pytest.approx(3.770423)
    assert dist.expectile(0.99, max_iter=1000) == pytest.approx(8.584217)
    assert dist.expectile(0.999, max_iter=1000) == pytest.approx(17.14993)
    assert dist.expectile(0.9999, max_iter=1000) == pytest.approx(19.71332)

    dist = LognormalDistribution(norm_mean=5, norm_sd=2)
    assert dist.expectile(0.0001, max_iter=1000) == pytest.approx(5.100107)
    assert dist.expectile(0.001, max_iter=1000) == pytest.approx(15.79393)
    assert dist.expectile(0.01, max_iter=1000) == pytest.approx(56.45435)
    assert dist.expectile(0.99, max_iter=1000) == pytest.approx(21302.24)
    assert dist.expectile(0.999, max_iter=1000) == pytest.approx(38450.37)
    assert dist.expectile(0.9999, max_iter=1000) == pytest.approx(38450.37)

    dist = LognormalDistribution(norm_mean=-5, norm_sd=2)
    assert dist.expectile(0.001, max_iter=1000) == pytest.approx(0.0007170433)
    assert dist.expectile(0.1, max_iter=1000) == pytest.approx(0.01135446)
    assert dist.expectile(0.9, max_iter=1000) == pytest.approx(0.2183066)
    assert dist.expectile(0.999, max_iter=1000) == pytest.approx(1.745644)

    dist = LognormalDistribution(norm_mean=2, norm_sd=0.1)
    assert dist.expectile(0.001, max_iter=1000) == pytest.approx(5.821271)
    assert dist.expectile(0.1, max_iter=1000) == pytest.approx(6.813301)
    assert dist.expectile(0.9, max_iter=1000) == pytest.approx(8.094002)
    assert dist.expectile(0.999, max_iter=1000) == pytest.approx(9.473339)

def test_contribution_to_ev():
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    print(dist.contribution_to_ev(0.5))
    print(dist.contribution_to_ev(0.001))
    print(dist.contribution_to_ev(0.999))
