import os
import pytest

from ..squigglepy.bayes import simple_bayes, bayesnet, update, average
from ..squigglepy.samplers import sample
from ..squigglepy.distributions import discrete, norm, beta, gamma
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


def test_bayesnet_noop():
    out = bayesnet()
    assert out is None


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


def test_bayesnet_cache_multiple():
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

    def define_event2():
        return {'a': 4, 'b': 6}
    bayesnet(define_event2,
             find=lambda e: e['b'],
             n=1000)
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches4 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches2 < n_caches4
    assert _squigglepy_internal_bayesnet_caches.get(define_event)['metadata']['n'] == 100
    assert _squigglepy_internal_bayesnet_caches.get(define_event2)['metadata']['n'] == 1000

    bayesnet(define_event2,
             find=lambda e: e['a'],
             n=100)
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches5 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches4 == n_caches5

    bayesnet(define_event,
             find=lambda e: e['a'],
             n=100)
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches6 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches4 == n_caches6


def test_bayesnet_reload_cache():
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
             n=100,
             reload_cache=True)
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches4 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches3 == n_caches4
    assert _squigglepy_internal_bayesnet_caches.get(define_event)['metadata']['n'] == 100


def test_bayesnet_dont_use_cache():
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches = len(_squigglepy_internal_bayesnet_caches)

    def define_event():
        return {'a': 1, 'b': 2}
    bayesnet(define_event,
             find=lambda e: e['a'],
             memcache=False,
             n=100)
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches2 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches == n_caches2


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


@pytest.fixture
def cachefile():
    cachefile = 'testcache'
    yield cachefile
    os.remove(cachefile + '.sqcache')


def test_bayesnet_cachefile(cachefile):
    assert not os.path.exists(cachefile + '.sqcache')

    def define_event():
        return {'a': 1, 'b': 2}

    bayesnet(define_event,
             find=lambda e: e['a'],
             dump_cache_file=cachefile,
             n=100)

    out = bayesnet(define_event,
                   find=lambda e: e['a'],
                   raw=True,
                   n=100)
    assert os.path.exists(cachefile + '.sqcache')
    assert set(out) == set([1])

    out = bayesnet(define_event,
                   find=lambda e: e['a'],
                   conditional_on=lambda e: e['b'] == 2,
                   raw=True,
                   n=100)
    assert os.path.exists(cachefile + '.sqcache')
    assert set(out) == set([1])

    def define_event():
        return {'a': 2, 'b': 3}

    out = bayesnet(define_event,
                   find=lambda e: e['a'],
                   load_cache_file=cachefile,
                   memcache=False,
                   raw=True,
                   n=100)
    assert set(out) == set([1])

    out = bayesnet(define_event,
                   find=lambda e: e['a'],
                   memcache=False,
                   raw=True,
                   n=100)
    assert set(out) == set([2])


def test_bayesnet_cachefile_primary(cachefile):
    assert not os.path.exists(cachefile + '.sqcache')

    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches = len(_squigglepy_internal_bayesnet_caches)

    def define_event():
        return {'a': 1, 'b': 2}

    bayesnet(define_event, find=lambda e: e['a'], n=100)

    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches2 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches2 == n_caches + 1
    assert not os.path.exists(cachefile + '.sqcache')

    def define_event2():
        return {'a': 2, 'b': 3}

    bayesnet(define_event2,
             find=lambda e: e['a'],
             dump_cache_file=cachefile,
             memcache=False,
             n=100)

    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches3 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches3 == n_caches2
    assert os.path.exists(cachefile + '.sqcache')

    out = bayesnet(define_event,
                   find=lambda e: e['a'],
                   raw=True,
                   n=100)
    assert set(out) == set([1])

    out = bayesnet(define_event,
                   load_cache_file=cachefile,
                   cache_file_primary=False,
                   find=lambda e: e['a'],
                   raw=True,
                   n=100)
    assert set(out) == set([1])

    out = bayesnet(define_event,
                   load_cache_file=cachefile,
                   cache_file_primary=True,
                   find=lambda e: e['a'],
                   raw=True,
                   n=100)
    assert set(out) == set([2])

    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches4 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches4 == n_caches2
    assert os.path.exists(cachefile + '.sqcache')


def test_bayesnet_cachefile_will_also_memcache(cachefile):
    assert not os.path.exists(cachefile + '.sqcache')
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches = len(_squigglepy_internal_bayesnet_caches)

    def define_event():
        return {'a': 1, 'b': 2}

    out = bayesnet(define_event,
                   find=lambda e: e['a'],
                   dump_cache_file=cachefile,
                   memcache=False,
                   raw=True,
                   n=100)

    assert os.path.exists(cachefile + '.sqcache')
    assert set(out) == set([1])
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches2 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches2 == n_caches

    out = bayesnet(define_event,
                   find=lambda e: e['a'],
                   dump_cache_file=cachefile,
                   memcache=True,
                   raw=True,
                   n=100)

    assert os.path.exists(cachefile + '.sqcache')
    assert set(out) == set([1])
    from ..squigglepy.bayes import _squigglepy_internal_bayesnet_caches
    n_caches3 = len(_squigglepy_internal_bayesnet_caches)
    assert n_caches3 == n_caches + 1


def test_bayesnet_cachefile_insufficent_samples_error(cachefile):
    assert not os.path.exists(cachefile + '.sqcache')

    def define_event():
        return {'a': 1, 'b': 2}

    bayesnet(define_event,
             find=lambda e: e['a'],
             dump_cache_file=cachefile,
             n=100)
    assert os.path.exists(cachefile + '.sqcache')

    with pytest.raises(ValueError) as excinfo:
        bayesnet(define_event,
                 load_cache_file=cachefile,
                 find=lambda e: e['a'],
                 n=1000)
    assert 'insufficient samples' in str(excinfo.value)


def test_bayesnet_multicore():
    def define_event():
        return {'a': 1, 'b': 2}

    out = bayesnet(define_event,
                   find=lambda e: e['a'],
                   cores=2,
                   n=100)
    assert out == 1
    assert not os.path.exists('test-core-0.sqcache')


def test_update_normal():
    out = update(norm(1, 10), norm(5, 15))
    assert out.type == 'norm'
    assert round(out.mean, 2) == 7.51
    assert round(out.sd, 2) == 2.03


def test_update_normal_evidence_weight():
    out = update(norm(1, 10), norm(5, 15), evidence_weight=3)
    assert out.type == 'norm'
    assert round(out.mean, 2) == 8.69
    assert round(out.sd, 2) == 1.48


def test_update_beta():
    out = update(beta(1, 1), beta(2, 2))
    assert out.type == 'beta'
    assert out.a == 3
    assert out.b == 3


def test_update_not_implemented():
    with pytest.raises(ValueError) as excinfo:
        update(gamma(1), gamma(2))
    assert 'type `gamma` not supported' in str(excinfo.value)


def test_update_not_matching():
    with pytest.raises(ValueError) as excinfo:
        update(norm(1, 2), beta(1, 2))
    assert 'can only update distributions of the same type' in str(excinfo.value)


def test_average():
    out = average(norm(1, 2), norm(3, 4))
    assert out.type == 'mixture'
    assert out.dists[0].type == 'norm'
    assert out.dists[0].x == 1
    assert out.dists[0].y == 2
    assert out.dists[1].type == 'norm'
    assert out.dists[1].x == 3
    assert out.dists[1].y == 4
    assert out.weights == [0.5, 0.5]
