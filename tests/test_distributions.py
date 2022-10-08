import pytest
from ..squigglepy.distributions import (to, const, uniform, norm, lognorm,
                                        binomial, beta, bernoulli, discrete,
                                        tdist, log_tdist, triangular,
                                        poisson, exponential, gamma, mixture)


def test_to_is_log_when_all_positive():
    assert to(1, 2) == lognorm(1, 2)


def test_to_is_norm_when_not_all_positive():
    assert to(-1, 2) == norm(-1, 2)


def test_to_is_norm_when_zero():
    assert to(0, 10) == norm(0, 10)


def to_passes_lclip_rclip():
    assert to(3, 5, lclip=0, rclip=10) == lognorm(3, 5, lclip=0, rclip=10)
    assert to(-3, 3, lclip=-4, rclip=4) == norm(-3, 3, lclip=-4, rclip=4)


def to_passes_credibility():
    assert to(3, 5, credibility=0.8) == lognorm(3, 5, credibility=0.8)
    assert to(-3, 3, credibility=0.8) == norm(-3, 3, credibility=0.8)


def test_const():
    assert const(1) == [1, None, 'const', None, None]


def test_norm():
    assert norm(1, 2) == [1, 2, 'norm', 0.9, None, None]


def test_norm_with_mean_sd():
    assert norm(mean=1, sd=2) == [1, 2, 'norm-mean', None, None]


def test_norm_with_just_sd_infers_zero_mean():
    assert norm(sd=2) == [0, 2, 'norm-mean', None, None]


def test_norm_raises_value_error():
    with pytest.raises(ValueError):
        norm()
    with pytest.raises(ValueError):
        norm(x=1, y=2, mean=3, sd=4)


def test_norm_passes_lclip_rclip():
    assert norm(1, 2, lclip=0, rclip=3) == [1, 2, 'norm', 0.9, 0, 3]
    assert norm(mean=1, sd=2, lclip=0, rclip=3) == [1, 2, 'norm-mean', 0, 3]
    assert norm(sd=2, lclip=0, rclip=3) == [0, 2, 'norm-mean', 0, 3]


def test_norm_passes_credibility():
    assert norm(1, 2, credibility=0.8) == [1, 2, 'norm', 0.8, None, None]


def test_lognorm():
    assert lognorm(1, 2) == [1, 2, 'log', 0.9, None, None]


def test_lognorm_with_mean_sd():
    assert lognorm(mean=1, sd=2) == [1, 2, 'log-mean', None, None]


def test_lognorm_with_just_sd_infers_zero_mean():
    assert lognorm(sd=2) == [0, 2, 'log-mean', None, None]


def test_lognorm_raises_value_error():
    with pytest.raises(ValueError):
        lognorm()
    with pytest.raises(ValueError):
        lognorm(x=1, y=2, mean=3, sd=4)


def test_lognorm_passes_lclip_rclip():
    assert lognorm(1, 2, lclip=0, rclip=3) == [1, 2, 'log', 0.9, 0, 3]
    assert lognorm(mean=1, sd=2, lclip=0, rclip=3) == [1, 2, 'log-mean', 0, 3]
    assert lognorm(sd=2, lclip=0, rclip=3) == [0, 2, 'log-mean', 0, 3]


def test_lognorm_passes_credibility():
    assert lognorm(1, 2, credibility=0.8) == [1, 2, 'log', 0.8, None, None]


def test_uniform():
    assert uniform(0, 1) == [0, 1, 'uniform', None, None]


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


def test_tdist():
    assert tdist(1, 3, 5) == [1, 3, 'tdist', 5, 0.9, None, None]


def test_tdist_passes_lclip_rclip():
    assert tdist(2, 4, t=6, lclip=3, rclip=5) == [2, 4, 'tdist', 6, 0.9, 3, 5]


def test_tdist_passes_credibility():
    assert (tdist(2, 4, t=5, credibility=0.8) ==
            [2, 4, 'tdist', 5, 0.8, None, None])


def test_log_tdist():
    assert log_tdist(1, 3, 5) == [1, 3, 'log-tdist', 5, 0.9, None, None]


def test_log_tdist_passes_lclip_rclip():
    assert log_tdist(2, 4, t=6, lclip=3, rclip=5) == [2, 4, 'log-tdist', 6, 0.9, 3, 5]


def test_log_tdist_passes_credibility():
    assert (log_tdist(2, 4, t=5, credibility=0.8) ==
            [2, 4, 'log-tdist', 5, 0.8, None, None])


def test_triangular():
    assert triangular(1, 3, 5) == [1, 3, 'triangular', 5, None, None]


def test_triangular_lclip_rclip():
    assert triangular(2, 4, 6,
                      lclip=3,
                      rclip=5) == [2, 4, 'triangular', 6, 3, 5]


def test_exponential():
    assert exponential(10) == [10, None, 'exponential', None, None]


def test_exponential_rclip_lclip():
    assert (exponential(10, lclip=10, rclip=15) ==
            [10, None, 'exponential', 10, 15])


def test_poisson():
    assert poisson(10) == [10, None, 'poisson', None, None]


def test_poisson_rclip_lclip():
    assert poisson(10, lclip=10, rclip=15) == [10, None, 'poisson', 10, 15]


def test_gamma():
    assert gamma(10, 2) == [10, 2, 'gamma', None, None]


def test_gamma_default_scale():
    assert gamma(10) == [10, 1, 'gamma', None, None]


def test_gamma_rclip_lclip():
    assert gamma(10, 2, lclip=10, rclip=15) == [10, 2, 'gamma', 10, 15]


def test_mixture():
    test = mixture([norm(1, 2), norm(3, 4)], [0.4, 0.6])
    expected = [[norm(1, 2), norm(3, 4)], [0.4, 0.6], 'mixture', None, None]
    assert test == expected


def test_mixture_different_distributions():
    test = mixture([lognorm(1, 10), gamma(3)], [0.1, 0.9])
    expected = [[lognorm(1, 10), gamma(3)], [0.1, 0.9], 'mixture', None, None]
    assert test == expected


def test_mixture_no_weights():
    test = mixture([lognorm(1, 10), gamma(3)])
    expected = [[lognorm(1, 10), gamma(3)], None, 'mixture', None, None]
    assert test == expected


def test_mixture_lclip_rclip():
    test = mixture([norm(1, 2), norm(3, 4)], [0.4, 0.6], lclip=1, rclip=4)
    expected = [[norm(1, 2), norm(3, 4)], [0.4, 0.6], 'mixture', 1, 4]
    assert test == expected


def test_mixture_different_format():
    test = mixture([[0.4, norm(1, 2)], [0.6, norm(3, 4)]])
    expected = [[[0.4, norm(1, 2)], [0.6, norm(3, 4)]], None, 'mixture', None, None]
    assert test == expected
