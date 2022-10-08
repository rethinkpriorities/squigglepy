import math

import numpy as np

from tqdm import tqdm
from datetime import datetime

from .distributions import norm, beta, mixture


_squigglepy_internal_bayesnet_caches = {}


def simple_bayes(likelihood_h, likelihood_not_h, prior):
    """
    p(h|e) = (p(e|h)*p(h)) / (p(e|h)*p(h) + p(e|~h)*(1-p(h)))

    p(h|e) is called posterior
    p(e|h) is called likelihood
    p(h) is called prior
    """
    return ((likelihood_h * prior) /
            (likelihood_h * prior +
             likelihood_not_h * (1 - prior)))


def bayesnet(event_fn, n=1, find=None, conditional_on=None,
             reduce_fn=None, raw=False, cache=True,
             reload_cache=False, verbose=False):
    events = None
    if not reload_cache:
        if verbose:
            print('Checking cache...')
        events = _squigglepy_internal_bayesnet_caches.get(event_fn)
        if events:
            if events['metadata']['n'] < n:
                raise ValueError(('{} results cached but ' +
                                  'requested {}').format(events['metadata']['n'], n))
            else:
                if verbose:
                    print('...Cached data found. Using it.')
                events = events['events']
    elif verbose:
        print('Reloading cache...')

    if events is None:
        if verbose:
            print('Generating Bayes net...')
            events = [event_fn() for _ in tqdm(range(n))]
        else:
            events = [event_fn() for _ in range(n)]
        if verbose:
            print('...Generated')
        if cache:
            if verbose:
                print('Caching...')
            metadata = {'n': n, 'last_generated': datetime.now()}
            _squigglepy_internal_bayesnet_caches[event_fn] = {'events': events,
                                                              'metadata': metadata}
            if verbose:
                print('...Cached')

    if conditional_on is not None:
        if verbose:
            print('Filtering conditional...')
        events = [e for e in events if conditional_on(e)]

    if len(events) < 1:
        raise ValueError('insufficient samples for condition')

    if conditional_on and verbose:
        print('...Done')

    if find is None:
        if verbose:
            print('...Reducing')
        return events if reduce_fn is None else reduce_fn(events)
    else:
        events = [find(e) for e in events]
        if raw:
            return events
        else:
            if verbose:
                print('...Reducing')
            reduce_fn = np.mean if reduce_fn is None else reduce_fn
            return reduce_fn(events)


def update(prior, evidence, evidence_weight=1, type='normal'):
    if type == 'normal':  # TODO: Infer
        prior_mean = np.mean(prior)  # TODO: Get from class, not samples
        prior_var = np.std(prior) ** 2
        evidence_mean = np.mean(evidence)
        evidence_var = np.std(evidence) ** 2
        return norm(mean=((evidence_var * prior_mean +
                           evidence_weight * (prior_var * evidence_mean)) /
                          (evidence_var + prior_var)),
                    sd=math.sqrt((evidence_var * prior_var) /
                                 (evidence_weight * evidence_var + prior_var)))
    elif type == 'beta':
        prior_a = prior[0]
        prior_b = prior[1]
        evidence_a = evidence[0]
        evidence_b = evidence[1]
        return beta(prior_a + evidence_a, prior_b + evidence_b)
    else:
        raise ValueError('type `{}` not supported.'.format(type))


def average(prior, evidence, weights=[0.5, 0.5]):
    return mixture([prior, evidence], weights)
