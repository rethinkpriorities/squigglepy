import math
import numpy as np

from .distributions import norm, beta, mixture


def simple_bayes(likelihood_h, likelihood_not_h, prior):
    """
    p(h|e) = (p(e|h)*p(h)) / (p(e|h)*p(h) + p(e|~h)*(1-p(h)))

    p(h|e) is called posterior
    p(e|h) is called likelihood
    p(h) is called prior
    """
    return (likelihood_h * prior) / (likelihood_h * prior + likelihood_not_h * (1 - prior))


def bayesnet(event_fn, n=1, find=None, conditional_on=None):
    events = [event_fn() for _ in range(n)]
    if conditional_on is not None:
        events = [e for e in events if conditional_on(e)]
    if len(events) < 1:
        raise ValueError('insufficient samples for condition')
    if find is not None:
        return sum([find(e) for e in events]) / len(events)
    else:
        return events


def update(prior, evidence, evidence_weight=1, type='normal'):
    if type == 'normal': #TODO: Infer
        prior_mean = np.mean(prior) # TODO: Get from class, not samples
        prior_var = np.std(prior) ** 2
        evidence_mean = np.mean(evidence)
        evidence_var = np.std(evidence) ** 2
        return norm(mean=(evidence_var * prior_mean + evidence_weight * (prior_var * evidence_mean)) / (evidence_var + prior_var),
                    sd=math.sqrt((evidence_var * prior_var) / (evidence_weight * evidence_var + prior_var)))
    elif type == 'beta':
        prior_a = prior[0]
        prior_b = prior[1]
        evidence_a = evidence[0]
        evidence_b = evidence[1]
        return beta(prior_a + evidence_a, prior_b + evidence_b)
    else:
        raise ValueError('type `{}` not supported.'.format(type))


def average(prior, evidence, weights=[0.5,0.5]):
    return mixture([prior, evidence], weights)

