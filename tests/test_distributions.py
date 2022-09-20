from ..squigglepy.distributions import (to, const, uniform, norm, lognorm, binomial,
                                        beta, bernoulli, discrete, tdist, log_tdist,
                                        triangular, exponential, mixture)

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

# TODO: test norm, lognorm, binomial, beta, bernoulli, discrete, tdist, log_tdist,
#       triangular, exponential, mixture

