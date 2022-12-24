import pytest
import numpy as np
from ..squigglepy.distributions import (to, const, uniform, norm, lognorm,
                                        binomial, beta, bernoulli, discrete,
                                        tdist, log_tdist, triangular, chisquare,
                                        poisson, exponential, gamma, mixture,
                                        lclip, rclip, clip, dist_round, dist_fn,
                                        dist_max, dist_ceil, dist_floor, zero_inflated,
                                        inf0)


def test_to_is_log_when_all_positive():
    assert to(1, 2).type == 'lognorm'


def test_to_is_norm_when_not_all_positive():
    assert to(-1, 2).type == 'norm'


def test_to_is_norm_when_zero():
    assert to(0, 10).type == 'norm'


def to_passes_lclip_rclip():
    assert to(3, 5, lclip=0, rclip=10) == lognorm(3, 5, lclip=0, rclip=10)
    assert to(-3, 3, lclip=-4, rclip=4) == norm(-3, 3, lclip=-4, rclip=4)


def to_passes_credibility():
    assert to(3, 5, credibility=80) == lognorm(3, 5, credibility=80)
    assert to(-3, 3, credibility=80) == norm(-3, 3, credibility=80)


def test_const():
    assert const(1).type == 'const'
    assert const(1).x == 1
    assert str(const(1)) == '<Distribution> const(1)'


def test_const_wraps_float():
    assert const(0.1).type == 'const'
    assert const(0.1).x == 0.1
    assert str(const(0.1)) == '<Distribution> const(0.1)'


def test_const_wraps_str():
    assert const('a').type == 'const'
    assert const('a').x == 'a'
    assert str(const('a')) == '<Distribution> const(a)'


def test_const_wraps_list():
    assert const([1, 2, 3]).type == 'const'
    assert const([1, 2, 3]).x == [1, 2, 3]
    assert str(const([1, 2, 3])) == '<Distribution> const([1, 2, 3])'


def test_norm():
    assert norm(1, 2).type == 'norm'
    assert norm(1, 2).x == 1
    assert norm(1, 2).y == 2
    assert norm(1, 2).mean == 1.5
    assert round(norm(1, 2).sd, 2) == 0.3
    assert norm(1, 2).credibility == 90
    assert norm(1, 2).lclip is None
    assert norm(1, 2).rclip is None
    assert str(norm(1, 2)) == '<Distribution> norm(mean=1.5, sd=0.3)'


def test_norm_with_mean_sd():
    assert norm(mean=1, sd=2).type == 'norm'
    assert norm(mean=1, sd=2).x is None
    assert norm(mean=1, sd=2).y is None
    assert norm(mean=1, sd=2).mean == 1
    assert norm(mean=1, sd=2).sd == 2
    assert norm(mean=1, sd=2).credibility == 90
    assert norm(mean=1, sd=2).lclip is None
    assert norm(mean=1, sd=2).rclip is None


def test_norm_with_just_sd_infers_zero_mean():
    assert norm(sd=2).type == 'norm'
    assert norm(sd=2).x is None
    assert norm(sd=2).y is None
    assert norm(sd=2).mean == 0
    assert norm(sd=2).sd == 2
    assert norm(sd=2).credibility == 90
    assert norm(sd=2).lclip is None
    assert norm(sd=2).rclip is None


def test_norm_blank_raises_value_error():
    with pytest.raises(ValueError) as execinfo:
        norm()
    assert 'must define either x/y or mean/sd' in str(execinfo.value)


def test_norm_overdefinition_value_error():
    with pytest.raises(ValueError) as execinfo:
        norm(x=1, y=2, mean=3, sd=4)
    assert 'cannot define both' in str(execinfo.value)


def test_norm_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        norm(10, 5)
    assert '`high value` cannot be lower than `low value`' in str(execinfo.value)


def test_norm_passes_lclip_rclip():
    obj = norm(1, 2, lclip=1)
    assert obj.type == 'norm'
    assert obj.lclip == 1
    assert obj.rclip is None
    assert str(obj) == '<Distribution> norm(mean=1.5, sd=0.3, lclip=1)'
    obj = norm(1, 2, rclip=1)
    assert obj.type == 'norm'
    assert obj.lclip is None
    assert obj.rclip == 1
    assert str(obj) == '<Distribution> norm(mean=1.5, sd=0.3, rclip=1)'
    obj = norm(1, 2, lclip=0, rclip=3)
    assert obj.type == 'norm'
    assert obj.lclip == 0
    assert obj.rclip == 3
    assert str(obj) == '<Distribution> norm(mean=1.5, sd=0.3, lclip=0, rclip=3)'
    obj = norm(mean=1, sd=2, lclip=0, rclip=3)
    assert obj.type == 'norm'
    assert obj.lclip == 0
    assert obj.rclip == 3
    assert str(obj) == '<Distribution> norm(mean=1, sd=2, lclip=0, rclip=3)'
    obj = norm(sd=2, lclip=0, rclip=3)
    assert obj.type == 'norm'
    assert obj.lclip == 0
    assert obj.rclip == 3
    assert str(obj) == '<Distribution> norm(mean=0, sd=2, lclip=0, rclip=3)'


def test_norm_passes_credibility():
    obj = norm(1, 2, credibility=80)
    assert obj.type == 'norm'
    assert obj.credibility == 80


def test_lognorm():
    assert lognorm(1, 2).type == 'lognorm'
    assert lognorm(1, 2).x == 1
    assert lognorm(1, 2).y == 2
    assert round(lognorm(1, 2).mean, 2) == 0.35
    assert round(lognorm(1, 2).sd, 2) == 0.21
    assert lognorm(1, 2).credibility == 90
    assert lognorm(1, 2).lclip is None
    assert lognorm(1, 2).rclip is None
    assert str(lognorm(1, 2)) == '<Distribution> lognorm(mean=0.35, sd=0.21)'


def test_lognorm_with_mean_sd():
    assert lognorm(mean=1, sd=2).type == 'lognorm'
    assert lognorm(mean=1, sd=2).x is None
    assert lognorm(mean=1, sd=2).y is None
    assert lognorm(mean=1, sd=2).mean == 1
    assert lognorm(mean=1, sd=2).sd == 2
    assert lognorm(mean=1, sd=2).credibility == 90
    assert lognorm(mean=1, sd=2).lclip is None
    assert lognorm(mean=1, sd=2).rclip is None
    assert str(lognorm(mean=1, sd=2)) == '<Distribution> lognorm(mean=1, sd=2)'


def test_lognorm_with_just_sd_infers_zero_mean():
    assert lognorm(sd=2).type == 'lognorm'
    assert lognorm(sd=2).x is None
    assert lognorm(sd=2).y is None
    assert lognorm(sd=2).mean == 0
    assert lognorm(sd=2).sd == 2
    assert lognorm(sd=2).credibility == 90
    assert lognorm(sd=2).lclip is None
    assert lognorm(sd=2).rclip is None


def test_lognorm_blank_raises_value_error():
    with pytest.raises(ValueError) as execinfo:
        lognorm()
    assert 'must define either x/y or mean/sd' in str(execinfo.value)


def test_lognorm_overdefinition_value_error():
    with pytest.raises(ValueError) as execinfo:
        lognorm(x=1, y=2, mean=3, sd=4)
    assert 'cannot define both' in str(execinfo.value)


def test_lognorm_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        lognorm(10, 5)
    assert '`high value` cannot be lower than `low value`' in str(execinfo.value)


def test_lognorm_must_be_gt_0():
    with pytest.raises(ValueError) as execinfo:
        lognorm(0, 5)
    assert 'lognormal distribution must have values >= 0' in str(execinfo.value)
    with pytest.raises(ValueError) as execinfo:
        lognorm(-5, 5)
    assert 'lognormal distribution must have values >= 0' in str(execinfo.value)


def test_lognorm_passes_lclip_rclip():
    obj = lognorm(1, 2, lclip=1)
    assert obj.type == 'lognorm'
    assert obj.lclip == 1
    assert obj.rclip is None
    assert str(obj) == '<Distribution> lognorm(mean=0.35, sd=0.21, lclip=1)'
    obj = lognorm(1, 2, rclip=1)
    assert obj.type == 'lognorm'
    assert obj.lclip is None
    assert obj.rclip == 1
    assert str(obj) == '<Distribution> lognorm(mean=0.35, sd=0.21, rclip=1)'
    obj = lognorm(1, 2, lclip=0, rclip=3)
    assert obj.type == 'lognorm'
    assert obj.lclip == 0
    assert obj.rclip == 3
    assert str(obj) == '<Distribution> lognorm(mean=0.35, sd=0.21, lclip=0, rclip=3)'
    obj = lognorm(mean=1, sd=2, lclip=0, rclip=3)
    assert obj.type == 'lognorm'
    assert obj.lclip == 0
    assert obj.rclip == 3
    assert str(obj) == '<Distribution> lognorm(mean=1, sd=2, lclip=0, rclip=3)'
    obj = lognorm(sd=2, lclip=0, rclip=3)
    assert obj.type == 'lognorm'
    assert obj.lclip == 0
    assert obj.rclip == 3
    assert str(obj) == '<Distribution> lognorm(mean=0, sd=2, lclip=0, rclip=3)'


def test_lognorm_passes_credibility():
    obj = lognorm(1, 2, credibility=80)
    assert obj.type == 'lognorm'
    assert obj.credibility == 80


def test_uniform():
    assert uniform(0, 1).type == 'uniform'
    assert uniform(0, 1).x == 0
    assert uniform(0, 1).y == 1
    assert str(uniform(0, 1)) == '<Distribution> uniform(0, 1)'


def test_binomial():
    assert binomial(10, 0.1).type == 'binomial'
    assert binomial(10, 0.1).n == 10
    assert binomial(10, 0.1).p == 0.1
    assert str(binomial(10, 0.1)) == '<Distribution> binomial(n=10, p=0.1)'


def test_binomial_must_be_between_0_and_1():
    with pytest.raises(ValueError) as execinfo:
        binomial(10, -1)
    assert 'must be between 0 and 1' in str(execinfo.value)
    with pytest.raises(ValueError) as execinfo:
        binomial(10, 1.1)
    assert 'must be between 0 and 1' in str(execinfo.value)


def test_beta():
    assert beta(10, 1).type == 'beta'
    assert beta(10, 1).a == 10
    assert beta(10, 1).b == 1
    assert str(beta(10, 1)) == '<Distribution> beta(a=10, b=1)'


def test_bernoulli():
    assert bernoulli(0.1).type == 'bernoulli'
    assert bernoulli(0.1).p == 0.1
    assert str(bernoulli(0.1)) == '<Distribution> bernoulli(p=0.1)'


def test_tdist():
    assert tdist(1, 3, 5).type == 'tdist'
    assert tdist(1, 3, 5).x == 1
    assert tdist(1, 3, 5).y == 3
    assert tdist(1, 3, 5).t == 5
    assert tdist(1, 3, 5).credibility == 90
    assert tdist(1, 3, 5).lclip is None
    assert tdist(1, 3, 5).rclip is None
    assert str(tdist(1, 3, 5)) == '<Distribution> tdist(x=1, y=3, t=5)'


def test_pure_tdist():
    assert tdist(t=5).type == 'tdist'
    assert tdist(t=5).x is None
    assert tdist(t=5).y is None
    assert tdist(t=5).t == 5
    assert tdist(t=5).credibility is None
    assert tdist(t=5).lclip is None
    assert tdist(t=5).rclip is None
    assert str(tdist(t=5)) == '<Distribution> tdist(t=5)'


def test_default_pure_tdist():
    assert tdist().type == 'tdist'
    assert tdist().x is None
    assert tdist().y is None
    assert tdist().t == 20
    assert tdist().credibility is None
    assert tdist().lclip is None
    assert tdist().rclip is None
    assert str(tdist()) == '<Distribution> tdist(t=20)'


def test_tdist_passes_lclip_rclip():
    obj = tdist(1, 3, t=5, lclip=3)
    assert obj.type == 'tdist'
    assert obj.lclip == 3
    assert obj.rclip is None
    assert obj.credibility == 90
    assert str(obj) == '<Distribution> tdist(x=1, y=3, t=5, lclip=3)'
    obj = tdist(1, 3, t=5, rclip=3)
    assert obj.type == 'tdist'
    assert obj.lclip is None
    assert obj.rclip == 3
    assert obj.credibility == 90
    assert str(obj) == '<Distribution> tdist(x=1, y=3, t=5, rclip=3)'
    obj = tdist(1, 3, t=5, lclip=3, rclip=5)
    assert obj.type == 'tdist'
    assert obj.lclip == 3
    assert obj.rclip == 5
    assert obj.credibility == 90
    assert str(obj) == '<Distribution> tdist(x=1, y=3, t=5, lclip=3, rclip=5)'


def test_tdist_passes_credibility():
    obj = tdist(1, 3, t=5, credibility=80)
    assert obj.type == 'tdist'
    assert obj.credibility == 80
    assert str(obj) == '<Distribution> tdist(x=1, y=3, t=5, credibility=80)'


def test_log_tdist():
    assert log_tdist(1, 3, 5).type == 'log_tdist'
    assert log_tdist(1, 3, 5).x == 1
    assert log_tdist(1, 3, 5).y == 3
    assert log_tdist(1, 3, 5).t == 5
    assert log_tdist(1, 3, 5).credibility == 90
    assert log_tdist(1, 3, 5).lclip is None
    assert log_tdist(1, 3, 5).rclip is None
    assert str(log_tdist(1, 3, 5)) == '<Distribution> log_tdist(x=1, y=3, t=5)'


def test_pure_log_tdist():
    assert log_tdist(t=5).type == 'log_tdist'
    assert log_tdist(t=5).x is None
    assert log_tdist(t=5).y is None
    assert log_tdist(t=5).t == 5
    assert log_tdist(t=5).credibility is None
    assert log_tdist(t=5).lclip is None
    assert log_tdist(t=5).rclip is None
    assert str(log_tdist(t=5)) == '<Distribution> log_tdist(t=5)'


def test_default_pure_log_tdist():
    assert log_tdist().type == 'log_tdist'
    assert log_tdist().x is None
    assert log_tdist().y is None
    assert log_tdist().t == 1
    assert log_tdist().credibility is None
    assert log_tdist().lclip is None
    assert log_tdist().rclip is None
    assert str(log_tdist()) == '<Distribution> log_tdist(t=1)'


def test_log_tdist_passes_lclip_rclip():
    obj = log_tdist(1, 3, t=5, lclip=3)
    assert obj.type == 'log_tdist'
    assert obj.lclip == 3
    assert obj.rclip is None
    assert obj.credibility == 90
    assert str(obj) == '<Distribution> log_tdist(x=1, y=3, t=5, lclip=3)'
    obj = log_tdist(1, 3, t=5, rclip=3)
    assert obj.type == 'log_tdist'
    assert obj.lclip is None
    assert obj.rclip == 3
    assert obj.credibility == 90
    assert str(obj) == '<Distribution> log_tdist(x=1, y=3, t=5, rclip=3)'
    obj = log_tdist(1, 3, t=5, lclip=3, rclip=5)
    assert obj.type == 'log_tdist'
    assert obj.lclip == 3
    assert obj.rclip == 5
    assert obj.credibility == 90
    assert str(obj) == '<Distribution> log_tdist(x=1, y=3, t=5, lclip=3, rclip=5)'


def test_log_tdist_passes_credibility():
    obj = log_tdist(1, 3, t=5, credibility=80)
    assert obj.type == 'log_tdist'
    assert obj.credibility == 80


def test_triangular():
    assert triangular(1, 3, 5).type == 'triangular'
    assert triangular(1, 3, 5).left == 1
    assert triangular(1, 3, 5).mode == 3
    assert triangular(1, 3, 5).right == 5
    assert triangular(1, 3, 5).lclip is None
    assert triangular(1, 3, 5).rclip is None
    assert str(triangular(1, 3, 5)) == '<Distribution> triangular(1, 3, 5)'


def test_triangular_lclip_rclip():
    obj = triangular(2, 4, 6, lclip=3)
    assert obj.type == 'triangular'
    assert obj.lclip == 3
    assert obj.rclip is None
    assert str(obj) == '<Distribution> triangular(2, 4, 6, lclip=3)'
    obj = triangular(2, 4, 6, rclip=3)
    assert obj.type == 'triangular'
    assert obj.lclip is None
    assert obj.rclip == 3
    assert str(obj) == '<Distribution> triangular(2, 4, 6, rclip=3)'
    obj = triangular(2, 4, 6, lclip=3, rclip=5)
    assert obj.type == 'triangular'
    assert obj.lclip == 3
    assert obj.rclip == 5
    assert str(obj) == '<Distribution> triangular(2, 4, 6, lclip=3, rclip=5)'


def test_exponential():
    assert exponential(10).type == 'exponential'
    assert exponential(10).scale == 10
    assert str(exponential(10)) == '<Distribution> exponential(10)'


def test_exponential_lclip_rclip():
    obj = exponential(10, lclip=10)
    assert obj.type == 'exponential'
    assert obj.lclip == 10
    assert obj.rclip is None
    assert str(obj) == '<Distribution> exponential(10, lclip=10)'
    obj = exponential(10, rclip=10)
    assert obj.type == 'exponential'
    assert obj.lclip is None
    assert obj.rclip == 10
    assert str(obj) == '<Distribution> exponential(10, rclip=10)'
    obj = exponential(10, lclip=10, rclip=15)
    assert obj.type == 'exponential'
    assert obj.lclip == 10
    assert obj.rclip == 15
    assert str(obj) == '<Distribution> exponential(10, lclip=10, rclip=15)'


def test_poisson():
    assert poisson(10).type == 'poisson'
    assert poisson(10).lam == 10
    assert str(poisson(10)) == '<Distribution> poisson(10)'


def test_poisson_lclip_rclip():
    obj = poisson(10, lclip=10)
    assert obj.type == 'poisson'
    assert obj.lclip == 10
    assert obj.rclip is None
    assert str(obj) == '<Distribution> poisson(10, lclip=10)'
    obj = poisson(10, rclip=10)
    assert obj.type == 'poisson'
    assert obj.lclip is None
    assert obj.rclip == 10
    assert str(obj) == '<Distribution> poisson(10, rclip=10)'
    obj = poisson(10, lclip=10, rclip=15)
    assert obj.type == 'poisson'
    assert obj.lclip == 10
    assert obj.rclip == 15
    assert str(obj) == '<Distribution> poisson(10, lclip=10, rclip=15)'


def test_chisquare():
    assert chisquare(10).type == 'chisquare'
    assert chisquare(10).df == 10
    assert str(chisquare(10)) == '<Distribution> chisquare(10)'


def test_gamma():
    assert gamma(10, 2).type == 'gamma'
    assert gamma(10, 2).shape == 10
    assert gamma(10, 2).scale == 2
    assert str(gamma(10, 2)) == '<Distribution> gamma(shape=10, scale=2)'


def test_gamma_default_scale():
    assert gamma(10).type == 'gamma'
    assert gamma(10).shape == 10
    assert gamma(10).scale == 1
    assert str(gamma(10)) == '<Distribution> gamma(shape=10, scale=1)'


def test_gamma_lclip_rclip():
    obj = gamma(10, 2, lclip=10)
    assert obj.type == 'gamma'
    assert obj.lclip == 10
    assert obj.rclip is None
    assert str(obj) == '<Distribution> gamma(shape=10, scale=2, lclip=10)'
    obj = gamma(10, 2, rclip=10)
    assert obj.type == 'gamma'
    assert obj.lclip is None
    assert obj.rclip == 10
    assert str(obj) == '<Distribution> gamma(shape=10, scale=2, rclip=10)'
    obj = gamma(10, 2, lclip=10, rclip=15)
    assert obj.type == 'gamma'
    assert obj.lclip == 10
    assert obj.rclip == 15
    assert str(obj) == '<Distribution> gamma(shape=10, scale=2, lclip=10, rclip=15)'


def test_discrete():
    obj = discrete({'a': 0.9, 'b': 0.1})
    assert obj.type == 'discrete'
    assert obj.items == {'a': 0.9, 'b': 0.1}
    obj = discrete([0, 1])
    assert obj.type == 'discrete'
    assert obj.items == [0, 1]
    assert str(obj) == '<Distribution> discrete'


def test_discrete_list():
    obj = discrete([1, 2, 3])
    assert obj.type == 'discrete'
    assert obj.items == [1, 2, 3]


def test_discrete_works_on_numpy():
    obj = discrete(np.array([1, 2, 3]))
    assert obj.type == 'discrete'
    assert obj.items == [1, 2, 3]


def test_discrete_raises_on_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        discrete(2)
    assert 'inputs to discrete must be a dict or list' in str(excinfo.value)


def test_mixture():
    obj = mixture([norm(1, 2), norm(3, 4)], [0.4, 0.6])
    assert obj.type == 'mixture'
    assert obj.dists[0].type == 'norm'
    assert obj.dists[0].x == 1
    assert obj.dists[0].y == 2
    assert obj.dists[1].type == 'norm'
    assert obj.dists[1].x == 3
    assert obj.dists[1].y == 4
    assert obj.weights == [0.4, 0.6]
    assert str(obj) == '<Distribution> mixture'


def test_mixture_different_distributions():
    obj = mixture([lognorm(1, 10), gamma(3)], [0.4, 0.6])
    assert obj.type == 'mixture'
    assert obj.dists[0].type == 'lognorm'
    assert obj.dists[0].x == 1
    assert obj.dists[0].y == 10
    assert obj.dists[1].type == 'gamma'
    assert obj.dists[1].shape == 3
    assert obj.weights == [0.4, 0.6]


def test_mixture_with_numbers():
    obj = mixture([10, gamma(3)], [0.4, 0.6])
    assert obj.type == 'mixture'
    assert obj.dists[0] == 10
    assert obj.dists[1].type == 'gamma'
    assert obj.dists[1].shape == 3
    assert obj.weights == [0.4, 0.6]


def test_mixture_no_weights():
    obj = mixture([lognorm(1, 10), gamma(3)])
    assert obj.type == 'mixture'
    assert obj.weights == [0.5, 0.5]


def test_mixture_lclip_rclip():
    obj = mixture([norm(1, 2), norm(3, 4)], [0.4, 0.6], lclip=1, rclip=4)
    assert obj.type == 'mixture'
    assert obj.lclip == 1
    assert obj.rclip == 4


def test_mixture_different_format():
    obj = mixture([[0.4, norm(1, 2)], [0.6, norm(3, 4)]])
    assert obj.type == 'mixture'
    assert obj.dists[0].type == 'norm'
    assert obj.dists[0].x == 1
    assert obj.dists[0].y == 2
    assert obj.dists[1].type == 'norm'
    assert obj.dists[1].x == 3
    assert obj.dists[1].y == 4
    assert obj.weights == [0.4, 0.6]


def test_mixture_can_be_discrete():
    obj = mixture({'a': 0.9, 'b': 0.1})
    assert obj.type == 'mixture'
    assert obj.dists == ['a', 'b']
    assert obj.weights == [0.9, 0.1]
    obj = mixture([0, 1])
    assert obj.type == 'mixture'
    assert obj.dists == [0, 1]
    assert obj.weights == [0.5, 0.5]
    assert str(obj) == '<Distribution> mixture'


def test_zero_inflated():
    obj = zero_inflated(0.6, norm(1, 2))
    assert obj.type == 'mixture'
    assert obj.dists == [0, norm(1, 2)]
    assert obj.weights == [0.6, 0.4]


def test_inf0():
    obj = inf0(0.6, norm(1, 2))
    assert obj.type == 'mixture'
    assert obj.dists == [0, norm(1, 2)]
    assert obj.weights == [0.6, 0.4]


def test_zero_inflated_raises_error():
    with pytest.raises(ValueError) as execinfo:
        zero_inflated(1.1, norm(1, 2))
    assert 'must be between 0 and 1' in str(execinfo.value)


def test_lt_distribution():
    obj = norm(0, 1) < norm(1, 2)
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right.type == 'norm'
    assert obj.right.x == 1
    assert obj.right.y == 2
    assert obj.fn_str == '<'
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) < norm(mean=1.5, sd=0.3)'


def test_lte_distribution():
    obj = norm(0, 1) <= norm(1, 2)
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right.type == 'norm'
    assert obj.right.x == 1
    assert obj.right.y == 2
    assert obj.fn_str == '<='
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) <= norm(mean=1.5, sd=0.3)'


def test_gt_distribution():
    obj = norm(0, 1) > norm(1, 2)
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right.type == 'norm'
    assert obj.right.x == 1
    assert obj.right.y == 2
    assert obj.fn_str == '>'
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) > norm(mean=1.5, sd=0.3)'


def test_gte_distribution():
    obj = norm(0, 1) >= norm(1, 2)
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right.type == 'norm'
    assert obj.right.x == 1
    assert obj.right.y == 2
    assert obj.fn_str == '>='
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) >= norm(mean=1.5, sd=0.3)'


def test_eq_distribution():
    obj = norm(0, 1) == norm(1, 2)
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right.type == 'norm'
    assert obj.right.x == 1
    assert obj.right.y == 2
    assert obj.fn_str == '=='
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) == norm(mean=1.5, sd=0.3)'


def test_ne_distribution():
    obj = norm(0, 1) != norm(1, 2)
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right.type == 'norm'
    assert obj.right.x == 1
    assert obj.right.y == 2
    assert obj.fn_str == '!='
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) != norm(mean=1.5, sd=0.3)'


def test_add_distribution():
    obj = norm(0, 1) + 12
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right == 12
    assert obj.fn_str == '+'
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) + 12'


def test_radd_distribution():
    obj = 12 + norm(0, 1)
    assert obj.type == 'complex'
    assert obj.left == 12
    assert obj.right.type == 'norm'
    assert obj.right.x == 0
    assert obj.right.y == 1
    assert obj.fn_str == '+'
    assert str(obj) == '<Distribution> 12 + norm(mean=0.5, sd=0.3)'


def test_sub_distribution():
    obj = norm(0, 1) - 12
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right == 12
    assert obj.fn_str == '-'
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) - 12'


def test_rsub_distribution():
    obj = 12 - norm(0, 1)
    assert obj.type == 'complex'
    assert obj.left == 12
    assert obj.right.type == 'norm'
    assert obj.right.x == 0
    assert obj.right.y == 1
    assert obj.fn_str == '-'
    assert str(obj) == '<Distribution> 12 - norm(mean=0.5, sd=0.3)'


def test_mul_distribution():
    obj = norm(0, 1) * 12
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right == 12
    assert obj.fn_str == '*'
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) * 12'


def test_rmul_distribution():
    obj = 12 * norm(0, 1)
    assert obj.type == 'complex'
    assert obj.left == 12
    assert obj.right.type == 'norm'
    assert obj.right.x == 0
    assert obj.right.y == 1
    assert obj.fn_str == '*'
    assert str(obj) == '<Distribution> 12 * norm(mean=0.5, sd=0.3)'


def test_truediv_distribution():
    obj = norm(0, 1) / 12
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right == 12
    assert obj.fn_str == '/'
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) / 12'


def test_rtruediv_distribution():
    obj = 12 / norm(0, 1)
    assert obj.type == 'complex'
    assert obj.left == 12
    assert obj.right.type == 'norm'
    assert obj.right.x == 0
    assert obj.right.y == 1
    assert obj.fn_str == '/'
    assert str(obj) == '<Distribution> 12 / norm(mean=0.5, sd=0.3)'


def test_floordiv_distribution():
    obj = norm(0, 1) // 12
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right == 12
    assert obj.fn_str == '//'
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) // 12'


def test_rfloordiv_distribution():
    obj = 12 // norm(0, 1)
    assert obj.type == 'complex'
    assert obj.left == 12
    assert obj.right.type == 'norm'
    assert obj.right.x == 0
    assert obj.right.y == 1
    assert obj.fn_str == '//'
    assert str(obj) == '<Distribution> 12 // norm(mean=0.5, sd=0.3)'


def test_pow_distribution():
    obj = norm(0, 1) ** 2
    assert obj.type == 'complex'
    assert obj.left.type == 'norm'
    assert obj.left.x == 0
    assert obj.left.y == 1
    assert obj.right == 2
    assert obj.fn_str == '**'
    assert str(obj) == '<Distribution> norm(mean=0.5, sd=0.3) ** 2'


def test_rpow_distribution():
    obj = 2 ** norm(0, 1)
    assert obj.type == 'complex'
    assert obj.left == 2
    assert obj.right.type == 'norm'
    assert obj.right.x == 0
    assert obj.right.y == 1
    assert obj.fn_str == '**'
    assert str(obj) == '<Distribution> 2 ** norm(mean=0.5, sd=0.3)'


def test_complex_math():
    obj = (2 ** norm(0, 1)) - (8 * 6) + 2 + (lognorm(10, 100) / 11) + 8
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> 2 ** norm(mean=0.5, sd=0.3) - 48 + 2 + lognorm(mean=3.45, sd=0.7) / 11 + 8'


def test_pipe():
    obj = norm(0, 1) >> lclip(2)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> lclip(norm(mean=0.5, sd=0.3), 2)'


def test_two_pipes():
    obj = norm(0, 1) >> lclip(2) >> dist_round
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> round(lclip(norm(mean=0.5, sd=0.3), 2), 0)'


def test_dist_fn():
    def mirror(x):
        return 1 - x if x > 0.5 else x

    obj = dist_fn(norm(0, 1), mirror)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> mirror(norm(mean=0.5, sd=0.3))'
    obj = norm(0, 1) >> dist_fn(mirror)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> mirror(norm(mean=0.5, sd=0.3))'


def test_dist_fn_vectorize():
    @np.vectorize
    def mirror(x):
        return 1 - x if x > 0.5 else x

    obj = dist_fn(norm(0, 1), mirror)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> mirror(norm(mean=0.5, sd=0.3))'
    obj = norm(0, 1) >> dist_fn(mirror)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> mirror(norm(mean=0.5, sd=0.3))'


def test_dist_fn2():
    def mirror(x, y):
        return 1 - x if x > y else x

    obj = dist_fn(norm(0, 10), norm(1, 2), mirror)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> mirror(norm(mean=5.0, sd=3.04), norm(mean=1.5, sd=0.3))'
    obj = norm(0, 10) >> dist_fn(norm(1, 2), mirror)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> mirror(norm(mean=5.0, sd=3.04), norm(mean=1.5, sd=0.3))'


def test_dist_fn_list():
    def mirror(x):
        return 1 - x if x > 0.5 else x

    def mirror2(x):
        return 1 + x if x > 0.5 else x

    obj = dist_fn(norm(0, 1), [mirror, mirror2])
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> mirror2(mirror(norm(mean=0.5, sd=0.3)))'
    obj = norm(0, 1) >> dist_fn([mirror, mirror2])
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> mirror2(mirror(norm(mean=0.5, sd=0.3)))'


def test_max():
    obj = dist_max(norm(0, 1), lognorm(0.1, 1))
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> max(norm(mean=0.5, sd=0.3), lognorm(mean=-1.15, sd=0.7))'


def test_min():
    obj = dist_max(norm(0, 1), lognorm(0.1, 1))
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> max(norm(mean=0.5, sd=0.3), lognorm(mean=-1.15, sd=0.7))'


def test_round():
    obj = dist_round(norm(0, 1))
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> round(norm(mean=0.5, sd=0.3), 0)'
    obj = norm(0, 1) >> dist_round
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> round(norm(mean=0.5, sd=0.3), 0)'


def test_round_two_digits():
    obj = dist_round(norm(0, 1), digits=2)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> round(norm(mean=0.5, sd=0.3), 2)'
    obj = norm(0, 1) >> dist_round(2)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> round(norm(mean=0.5, sd=0.3), 2)'


def test_ceil():
    obj = dist_ceil(norm(0, 1))
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> ceil(norm(mean=0.5, sd=0.3))'
    obj = norm(0, 1) >> dist_ceil
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> ceil(norm(mean=0.5, sd=0.3))'


def test_floor():
    obj = dist_floor(norm(0, 1))
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> floor(norm(mean=0.5, sd=0.3))'
    obj = norm(0, 1) >> dist_floor
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> floor(norm(mean=0.5, sd=0.3))'


def test_lclip():
    obj = lclip(norm(0, 1), 0.5)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> lclip(norm(mean=0.5, sd=0.3), 0.5)'
    obj = norm(0, 1) >> lclip(0.5)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> lclip(norm(mean=0.5, sd=0.3), 0.5)'


def test_rclip():
    obj = rclip(norm(0, 1), 0.5)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> rclip(norm(mean=0.5, sd=0.3), 0.5)'
    obj = norm(0, 1) >> rclip(0.5)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> rclip(norm(mean=0.5, sd=0.3), 0.5)'


def test_clip():
    obj = clip(norm(0, 1), 0.5, 0.9)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> rclip(lclip(norm(mean=0.5, sd=0.3), 0.5), 0.9)'
    obj = norm(0, 1) >> clip(0.5, 0.9)
    assert obj.type == 'complex'
    assert str(obj) == '<Distribution> rclip(lclip(norm(mean=0.5, sd=0.3), 0.5), 0.9)'
