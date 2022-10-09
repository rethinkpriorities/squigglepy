import pytest
from unittest.mock import patch, Mock

from ..squigglepy.distributions import (const, uniform, norm, lognorm,
                                        binomial, beta, bernoulli, discrete,
                                        tdist, log_tdist, triangular,
                                        poisson, exponential, gamma, mixture)
from ..squigglepy.rng import set_seed
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


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_norm(mocker):
    assert normal_sample(1, 2) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_norm_with_mean_sd(mocker):
    assert normal_sample(mean=1, sd=2) == (1, 2)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_norm_with_credibility(mocker):
    assert normal_sample(1, 2, credibility=0.7) == (1.5, 0.48)


def test_norm_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        normal_sample(10, 5)
    assert '`high value` cannot be lower than `low value`' in str(execinfo.value)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_norm(mocker):
    assert sample(norm(1, 2)) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_norm_with_just_sd_infers_zero_mean():
    assert sample(norm(sd=2)) == (0, 2)


@patch.object(samplers, 'normal_sample', Mock(return_value=100))
def test_sample_norm_passes_lclip_rclip():
    assert sample(norm(1, 2)) == 100
    assert sample(norm(1, 2, lclip=1, rclip=3)) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_lognorm(mocker):
    assert lognormal_sample(1, 2) == (0.35, 0.21)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_lognorm_with_mean_sd(mocker):
    assert lognormal_sample(mean=1, sd=2) == (1, 2)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_lognorm_with_credibility(mocker):
    assert lognormal_sample(1, 2, credibility=0.7) == (0.35, 0.33)


def test_lognorm_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        lognormal_sample(10, 5)
    assert '`high value` cannot be lower than `low value`' in str(execinfo.value)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_lognorm(mocker):
    assert sample(lognorm(1, 2)) == (0.35, 0.21)


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
    set_seed(42)
    assert sample(bernoulli(0.1)) == 0


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_sample_discrete():
    assert sample(discrete({'a': 0.9, 'b': 0.1})) == 'a'
    assert sample(discrete([0, 1, 2])) == 0


def test_sample_discrete_error():
    with pytest.raises(ValueError) as execinfo:
        discrete_sample('error')
    assert 'inputs to discrete_sample must be a dict or list' in str(execinfo.value)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_tdist(mocker):
    assert t_sample(1, 2, 3) == 2.5


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_tdist_with_credibility(mocker):
    assert round(t_sample(1, 2, 3, credibility=0.7), 2) == 2.79


def test_tdist_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        t_sample(10, 5, 3)
    assert '`high value` cannot be lower than `low value`' in str(execinfo.value)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_sample_tdist(mocker):
    assert sample(tdist(1, 2, 3)) == 2.5


@patch.object(samplers, 't_sample', Mock(return_value=100))
def test_sample_tdist_passes_lclip_rclip():
    assert sample(tdist(1, 2, 3)) == 100
    assert sample(tdist(1, 2, 3, lclip=1, rclip=3)) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_log_tdist(mocker):
    assert round(log_t_sample(1, 2, 3), 2) == 2.83


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def test_log_tdist_with_credibility(mocker):
    assert round(log_t_sample(1, 2, 3, credibility=0.7), 2) == 3.45


def test_log_tdist_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        log_t_sample(10, 5, 3)
    assert '`high value` cannot be lower than `low value`' in str(execinfo.value)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
def teslog_t_sample_log_tdist(mocker):
    assert round(sample(log_tdist(1, 2, 3)), 2) == 2.82


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
def test_sample_poisson():
    assert sample(poisson(10)) == 10


@patch.object(samplers, 'poisson_sample', Mock(return_value=100))
def test_sample_poisson_passes_lclip_rclip():
    assert sample(poisson(1)) == 100
    assert sample(poisson(1, lclip=1, rclip=3)) == 3


def test_sample_const():
    assert sample(const(10)) == 10


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
def test_mixture(mocker):
    assert mixture_sample([norm(1, 2), norm(3, 4)], [0.2, 0.8]) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_different_format(mocker):
    assert mixture_sample([[0.2, norm(1, 2)], [0.8, norm(3, 4)]]) == (1.5, 0.3)


@patch.object(samplers, 'normal_sample', Mock(return_value=100))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_rclip_lclip(mocker):
    assert mixture_sample([norm(1, 2), norm(3, 4)], [0.2, 0.8]) == 100
    assert mixture_sample([norm(1, 2, rclip=3), norm(3, 4)], [0.2, 0.8]) == 3


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_no_weights(mocker):
    assert mixture_sample([norm(1, 2), norm(3, 4)]) == (1.5, 0.3)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_different_distributions(mocker):
    assert mixture_sample([lognorm(1, 2), norm(3, 4)]) == (0.35, 0.21)


@patch.object(samplers, '_get_rng', Mock(return_value=FakeRNG()))
@patch.object(samplers, 'uniform_sample', Mock(return_value=0))
def test_mixture_sample(mocker):
    assert sample(mixture([lognorm(1, 2), norm(3, 4)])) == (0.35, 0.21)
