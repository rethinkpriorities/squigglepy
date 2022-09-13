import math
import numpy as np

from .distributions import norm, beta, mixture


def bayes(likelihood_h, likelihood_not_h, prior):
    """
    p(h|e) = (p(e|h)*p(h)) / (p(e|h)*p(h) + p(e|~h)*(1-p(h)))

    p(h|e) is called posterior
    p(e|h) is called likelihood
    p(h) is called prior
    """
    return (likelihood_h * prior) / (likelihood_h * prior + likelihood_not_h * (1 - prior))


def update(prior, evidence, type='normal'):
    if type == 'normal': #TODO: Infer
        prior_mean = np.mean(prior) # TODO: Get from class, not samples
        prior_var = np.std(prior) ** 2
        evidence_mean = np.mean(evidence)
        evidence_var = np.std(evidence) ** 2
        return norm(mean=(evidence_var * prior_mean + prior_var * evidence_mean) / (evidence_var + prior_var),
                    sd=math.sqrt((evidence_var * prior_var) / (evidence_var + prior_var)))
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
