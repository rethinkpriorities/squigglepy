import pytest
from ..squigglepy.distributions import (to, const, uniform, norm, lognorm,
                                        binomial, beta, bernoulli, discrete,
                                        tdist, log_tdist, triangular,
                                        poisson, exponential, gamma, mixture)


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


def test_beta():
    assert beta(10, 1).type == 'beta'
    assert beta(10, 1).a == 10
    assert beta(10, 1).b == 1
    assert str(beta(10, 1)) == '<Distribution> beta(a=10, b=1)'


def test_bernoulli():
    assert bernoulli(0.1).type == 'bernoulli'
    assert bernoulli(0.1).p == 0.1
    assert str(bernoulli(0.1)) == '<Distribution> bernoulli(p=0.1)'


def test_discrete():
    obj = discrete({'a': 0.9, 'b': 0.1})
    assert obj.type == 'discrete'
    assert obj.items == {'a': 0.9, 'b': 0.1}
    obj = discrete([0, 1])
    assert obj.type == 'discrete'
    assert obj.items == [0, 1]
    assert str(obj) == '<Distribution> discrete'


def test_discrete_raises_on_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        discrete(2)
    assert 'inputs to discrete must be a dict or list' in str(excinfo.value)


def test_tdist():
    assert tdist(1, 3, 5).type == 'tdist'
    assert tdist(1, 3, 5).x == 1
    assert tdist(1, 3, 5).y == 3
    assert tdist(1, 3, 5).t == 5
    assert tdist(1, 3, 5).credibility == 90
    assert tdist(1, 3, 5).lclip is None
    assert tdist(1, 3, 5).rclip is None
    assert str(tdist(1, 3, 5)) == '<Distribution> tdist(x=1, y=3, t=5)'


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
