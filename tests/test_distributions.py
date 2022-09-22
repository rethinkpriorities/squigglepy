import pytest
from ..squigglepy.distributions import (to, const, uniform, norm, lognorm,
                                        binomial, beta, bernoulli, discrete,
                                        tdist, log_tdist, triangular,
                                        exponential, gamma, mixture)


def test_to_is_log_when_all_positive():
    assert to(1, 2) == lognorm(1, 2)


def test_to_is_norm_when_not_all_positive():
    assert to(-1, 2) == norm(-1, 2)


def test_to_is_norm_when_zero():
    assert to(0, 10) == norm(0, 10)


def to_passes_lclip_rclip():
    assert to(3, 5, lclip=0, rclip=10) == lognorm(3, 5, lclip=0, rclip=10)
    assert to(-3, 3, lclip=-4, rclip=4) == norm(-3, 3, lclip=-4, rclip=4)


def test_const():
    assert const(1) == [1, None, 'const', None, None]


def test_uniform():
    assert uniform(0, 1) == [0, 1, 'uniform', None, None]


# TODO: test norm, lognorm


def test_binomial():
    assert binomial(10, 0.1) == [10, 0.1, 'binomial', None, None]


def test_beta():
    assert beta(10, 1) == [10, 1, 'beta', None, None]


def test_bernoulli():
    assert bernoulli(0.1) == [0.1, None, 'bernoulli', None, None]


def test_discrete():
    assert discrete({'a': 0.9, 'b': 0.1}) == [{'a': 0.9, 'b': 0.1},
                                              None, 'discrete', None, None]
    assert discrete([0, 1]) == [[0, 1], None, 'discrete', None, None]



def test_discrete_raises_on_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        discrete(2)
    assert 'inputs to discrete must be a dict or list' in str(excinfo.value)


# TODO: test tdist, log_tdist


def test_triangular():
    assert triangular(1, 3, 5) == [1, 3, 'triangular', 5, None, None]


def test_triangular_lclip_rclip():
    assert triangular(2, 4, 6,
                      lclip=3,
                      rclip=5) == [2, 4, 'triangular', 6, 3, 5]


def test_exponential():
    assert exponential(10) == [10, None, 'exponential', None, None]


def test_exponential_rclip_lclip():
    assert exponential(10, lclip=10, rclip=15) == [10, None, 'exponential', 10, 15]


def test_gamma():
    assert gamma(10, 2) == [10, 2, 'gamma', None, None]


def test_gamma_default_scale():
    assert gamma(10) == [10, 1, 'gamma', None, None]


def test_gamma_rclip_lclip():
    assert gamma(10, 2, lclip=10, rclip=15) == [10, 2, 'gamma', 10, 15]


# TODO: test mixture
