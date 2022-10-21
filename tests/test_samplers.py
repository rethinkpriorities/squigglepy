import pytest
import numpy as np
from unittest.mock import patch, Mock

from ..squigglepy.distributions import (const, uniform, norm, lognorm,
                                        binomial, beta, bernoulli, discrete,
                                        tdist, log_tdist, triangular, chisquare,
                                        poisson, exponential, gamma, mixture)
from ..squigglepy import samplers
from ..squigglepy.samplers import (normal_sample, lognormal_sample, mixture_sample,
                                   discrete_sample, log_t_sample, t_sample, sample)


class FakeRNG:
    def normal(self, mu, sigma):
        return round(mu, 2), round(sigma, 2)

    def lognormal(self, mu, sigma):
        return round(mu, 2), round(sigma, 2)

    def uniform(self, low, high):
        return low, high

    def binomial(self, n, p):
        return n, p

    def beta(self, a, b):
        return a, b

    def bernoulli(self, p):
        return p

    def gamma(self, shape, scale):
        return shape, scale

    def poisson(self, lam):
        return lam

    def exponential(self, scale):
        return scale

    def triangular(self, left, mode, right):
        return left, mode, right

    def standard_t(self, t):
        return t

    def chisquare(self, df):
        return df


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_norm(mocker):
    assert normal_sample(1, 2) == (1, 2)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_norm(mocker):
    assert sample(norm(1, 2)) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_norm_shorthand(mocker):
    assert ~norm(1, 2) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_norm_with_credibility(mocker):
    assert sample(norm(1, 2, credibility=70)) == (1.5, 0.48)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_norm_with_just_sd_infers_zero_mean():
    assert sample(norm(sd=2)) == (0, 2)


@patch.object(samplers, 'normal_sample', Mock(return_value=100))
def test_sample_norm_passes_lclip_rclip():
    assert sample(norm(1, 2)) == 100
    assert sample(norm(1, 2, lclip=1, rclip=3)) == 3
    assert ~norm(1, 2, lclip=1, rclip=3) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_lognorm(mocker):
    assert lognormal_sample(1, 2) == (1, 2)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_lognorm(mocker):
    assert sample(lognorm(1, 2)) == (0.35, 0.21)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_lognorm_with_credibility(mocker):
    assert sample(lognorm(1, 2, credibility=70)) == (0.35, 0.33)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_shorthand_lognorm(mocker):
    assert ~lognorm(1, 2) == (0.35, 0.21)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_shorthand_lognorm_with_credibility(mocker):
    assert ~lognorm(1, 2, credibility=70) == (0.35, 0.33)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_lognorm_with_just_sd_infers_zero_mean():
    assert sample(lognorm(sd=2)) == (0, 2)


@patch.object(samplers, 'lognormal_sample', Mock(return_value=100))
def test_sample_lognorm_passes_lclip_rclip():
    assert sample(lognorm(1, 2)) == 100
    assert sample(lognorm(1, 2, lclip=1, rclip=3)) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_uniform():
    assert sample(uniform(1, 2)) == (1, 2)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_binomial():
    assert sample(binomial(10, 0.1)) == (10, 0.1)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_beta():
    assert sample(beta(10, 1)) == (10, 1)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_bernoulli():
    assert sample(bernoulli(0.1)) == 0


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_discrete():
    assert discrete_sample([0, 1, 2]) == 0


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_discrete_alt_format():
    assert discrete_sample([[0.9, 'a'], [0.1, 'b']]) == 'a'


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_discrete_alt2_format():
    assert discrete_sample({'a': 0.9, 'b': 0.1}) == 'a'


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_discrete():
    assert sample(discrete([0, 1, 2])) == 0


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_discrete_alt_format():
    assert sample(discrete([[0.9, 'a'], [0.1, 'b']])) == 'a'


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_discrete_alt2_format():
    assert sample(discrete({'a': 0.9, 'b': 0.1})) == 'a'

@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_discrete_shorthand():
    assert ~discrete([0, 1, 2]) == 0
    assert ~discrete([[0.9, 'a'], [0.1, 'b']]) == 'a'
    assert ~discrete({'a': 0.9, 'b': 0.1}) == 'a'


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'normal_sample', Mock(return_value=1))
def test_tdist(mocker):
    assert round(t_sample(1, 2, 3), 2) == 1


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'normal_sample', Mock(return_value=1))
def test_tdist_with_credibility(mocker):
    assert round(t_sample(1, 2, 3, credibility=70), 2) == 1


def test_tdist_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        t_sample(10, 5, 3)
    assert '`high value` cannot be lower than `low value`' in str(execinfo.value)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'normal_sample', Mock(return_value=1))
def test_sample_tdist(mocker):
    assert round(sample(tdist(1, 2, 3)), 2) == 1


@patch.object(samplers, 't_sample', Mock(return_value=100))
def test_sample_tdist_passes_lclip_rclip():
    assert sample(tdist(1, 2, 3)) == 100
    assert sample(tdist(1, 2, 3, lclip=1, rclip=3)) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'normal_sample', Mock(return_value=1))
def test_log_tdist(mocker):
    assert round(log_t_sample(1, 2, 3), 2) == 2.72


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'normal_sample', Mock(return_value=1))
def test_log_tdist_with_credibility(mocker):
    assert round(log_t_sample(1, 2, 3, credibility=70), 2) == 2.72


def test_log_tdist_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        log_t_sample(10, 5, 3)
    assert '`high value` cannot be lower than `low value`' in str(execinfo.value)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'normal_sample', Mock(return_value=1))
def teslog_t_sample_log_tdist(mocker):
    assert round(sample(log_tdist(1, 2, 3)), 2) == 1 / 3


@patch.object(samplers, 'log_t_sample', Mock(return_value=100))
def teslog_t_sample_log_tdist_passes_lclip_rclip():
    assert sample(log_tdist(1, 2, 3)) == 100
    assert sample(log_tdist(1, 2, 3, lclip=1, rclip=3)) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_triangular():
    assert sample(triangular(10, 20, 30)) == (10, 20, 30)


@patch.object(samplers, 'triangular_sample', Mock(return_value=100))
def test_sample_triangular_passes_lclip_rclip():
    assert sample(triangular(1, 2, 3)) == 100
    assert sample(triangular(1, 2, 3, lclip=1, rclip=3)) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_exponential():
    assert sample(exponential(10)) == 10


@patch.object(samplers, 'exponential_sample', Mock(return_value=100))
def test_sample_exponential_passes_lclip_rclip():
    assert sample(exponential(1)) == 100
    assert sample(exponential(1, lclip=1, rclip=3)) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_chisquare():
    assert sample(chisquare(9)) == 9


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_poisson():
    assert sample(poisson(10)) == 10


@patch.object(samplers, 'poisson_sample', Mock(return_value=100))
def test_sample_poisson_passes_lclip_rclip():
    assert sample(poisson(1)) == 100
    assert sample(poisson(1, lclip=1, rclip=3)) == 3


def test_sample_const():
    assert sample(const(11)) == 11


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_gamma_default():
    assert sample(gamma(10)) == (10, 1)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_gamma():
    assert sample(gamma(10, 2)) == (10, 2)


@patch.object(samplers, 'gamma_sample', Mock(return_value=100))
def test_sample_gamma_passes_lclip_rclip():
    assert sample(gamma(1, 2)) == 100
    assert sample(gamma(1, 2, lclip=1, rclip=3)) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_sample(mocker):
    assert mixture_sample([norm(1, 2), norm(3, 4)], [0.2, 0.8]) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_sample_alt_format(mocker):
    assert mixture_sample([[0.2, norm(1, 2)], [0.8, norm(3, 4)]]) == (1.5, 0.3)


@patch.object(samplers, 'normal_sample', Mock(return_value=100))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_sample_rclip_lclip(mocker):
    assert mixture_sample([norm(1, 2), norm(3, 4)], [0.2, 0.8]) == 100
    assert mixture_sample([norm(1, 2, rclip=3), norm(3, 4)], [0.2, 0.8]) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_sample_no_weights(mocker):
    assert mixture_sample([norm(1, 2), norm(3, 4)]) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_sample_different_distributions(mocker):
    assert mixture_sample([lognorm(1, 2), norm(3, 4)]) == (0.35, 0.21)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_mixture(mocker):
    assert sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8])) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_mixture_alt_format(mocker):
    assert sample(mixture([[0.2, norm(1, 2)], [0.8, norm(3, 4)]])) == (1.5, 0.3)


@patch.object(samplers, 'normal_sample', Mock(return_value=100))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_mixture_rclip_lclip(mocker):
    assert sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8])) == 100
    assert sample(mixture([norm(1, 2, rclip=3), norm(3, 4)], [0.2, 0.8])) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_mixture_no_weights(mocker):
    assert sample(mixture([norm(1, 2), norm(3, 4)])) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_mixture_different_distributions(mocker):
    assert sample(mixture([lognorm(1, 2), norm(3, 4)])) == (0.35, 0.21)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_n_gt_1(mocker):
    assert np.array_equal(sample(norm(1, 2), n=5), np.array([(1.5, 0.3)] * 5))


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_shorthand_n_gt_1(mocker):
    assert np.array_equal(norm(1, 2) @ 5, np.array([(1.5, 0.3)] * 5))


def test_sample_n_is_0_is_error():
    with pytest.raises(ValueError) as execinfo:
        sample(norm(1, 5), n=0)
    assert 'n must be >= 1' in str(execinfo.value)


def test_sample_n_is_0_is_error_shorthand():
    with pytest.raises(ValueError) as execinfo:
        norm(1, 5) @ 0
    assert 'n must be >= 1' in str(execinfo.value)


def test_sample_callable():
    def sample_fn():
        return 1
    assert sample(sample_fn) == 1


def test_sample_invalid_input():
    with pytest.raises(ValueError) as execinfo:
        sample([1, 5])
    assert 'must be a distribution' in str(execinfo.value)


@patch.object(samplers, 'normal_sample', Mock(return_value=100))
def test_sample_math():
    assert ~(norm(0, 1) + norm(1, 2)) == 200


@patch.object(samplers, 'normal_sample', Mock(return_value=10))
@patch.object(samplers, 'lognormal_sample', Mock(return_value=100))
def test_sample_complex_math():
    obj = (2 ** norm(0, 1)) - (8 * 6) + 2 + (lognorm(10, 100) / 11) + 8
    expected = (2 ** 10) - (8 * 6) + 2 + (100 / 11) + 8
    assert ~obj == expected
