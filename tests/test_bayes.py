import pytest

from ..squigglepy.bayes import simple_bayes, bayesnet, update, average
from ..squigglepy.samplers import sample
from ..squigglepy.distributions import discrete, norm, beta, mixture
from ..squigglepy.rng import set_seed


def test_simple_bayes():
    out = simple_bayes(prior=0.01,
                       likelihood_h=0.8,
                       likelihood_not_h=0.096)
    assert round(out, 2) == 0.08


def test_bayesnet():
    set_seed(42)
    out = bayesnet(lambda: {'a': 1, 'b': 2},
                   find=lambda e: e['a'],
                   conditional_on=lambda e: e['b'],
                   n=100)
    assert out == 1


def test_bayesnet_conditional():
    def define_event():
        a = sample(discrete([1, 2]))
        b = 1 if a == 1 else 2
        return {'a': a, 'b': b}

    set_seed(42)
    out = bayesnet(define_event,
                   find=lambda e: e['a'] == 1,
                   n=100)
    assert round(out, 1) == 0.5

    out = bayesnet(define_event,
                   find=lambda e: e['a'] == 1,
                   conditional_on=lambda e: e['b'] == 1,
                   n=100)
    assert round(out, 1) == 1

    out = bayesnet(define_event,
                   find=lambda e: e['a'] == 2,
                   conditional_on=lambda e: e['b'] == 1,
                   n=100)
    assert round(out, 1) == 0

    out = bayesnet(define_event,
                   find=lambda e: e['a'] == 1,
                   conditional_on=lambda e: e['b'] == 2,
                   n=100)
    assert round(out, 1) == 0


def test_bayesnet_reduce_fn():
    out = bayesnet(lambda: {'a': 1, 'b': 2},
                   find=lambda e: e['a'],
                   reduce_fn=sum,
                   n=100)
    assert out == 100


def test_bayesnet_raw():
    out = bayesnet(lambda: {'a': 1, 'b': 2},
                   find=lambda e: e['a'],
                   raw=True,
                   n=100)
    assert out == [1] * 100


def test_bayesnet_cache():
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches = len(_squigglepy_internal_bayesnet_caches)

    def define_event():
        return {'a': 1, 'b': 2}
    bayesnet(define_event,
             find=lambda e: e['a'],
             n=100)
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches2 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches < n_caches2

    bayesnet(define_event,
             find=lambda e: e['a'],
             n=100)
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches3 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches2 == n_caches3

    bayesnet(define_event,
             find=lambda e: e['b'],
             n=100)
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches4 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches2 == n_caches4
    assert _squigglepy_internal_bayesnet_caches.get(define_event)['metadata']['n'] == 100


def test_bayesnet_cache_n_error():
    def define_event():
        return {'a': 1, 'b': 2}
    bayesnet(define_event,
             find=lambda e: e['a'],
             n=100)
    with pytest.raises(ValueError) as excinfo:
        bayesnet(define_event,
                 find=lambda e: e['a'],
                 n=1000)
    assert '100 results cached but requested 1000' in str(excinfo.value)


def test_bayesnet_insufficent_samples_error():
    with pytest.raises(ValueError) as excinfo:
        bayesnet(lambda: {'a': 1, 'b': 2},
                 find=lambda e: e['a'],
                 conditional_on=lambda e: e['b'] == 3,
                 n=100)
    assert 'insufficient samples' in str(excinfo.value)


def test_update_normal():
    out = update(list(range(10)), list(range(5, 15)))
    out[1] = round(out[1], 2)
    expected = [7.0, 2.03, 'norm-mean', None, None]
    assert out == expected


def test_update_normal_evidence_weight():
    out = update(list(range(10)), list(range(5, 15)), evidence_weight=3)
    out[1] = round(out[1], 2)
    # TODO: This seems wrong?
    expected = [16.5, 1.44, 'norm-mean', None, None]
    assert out == expected


def test_update_beta():
    out = update(beta(1, 1), beta(2, 2), type='beta')
    expected = beta(3, 3)
    assert out == expected


def test_update_not_implemented():
    with pytest.raises(ValueError) as excinfo:
        update(1, 2, type='error')
    assert 'type `error` not supported' in str(excinfo.value)


def test_average():
    out = average(norm(1, 2), norm(3, 4))
    expected = mixture([norm(1, 2), norm(3, 4)], [0.5, 0.5])
    assert out == expected
