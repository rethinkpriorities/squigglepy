import numpy as np

from tqdm import tqdm

from .distributions import const, BaseDistribution
from .utils import event_occurs, _process_weights_values


def _get_rng():
    from .rng import _squigglepy_internal_rng
    return _squigglepy_internal_rng


def normal_sample(mean, sd):
    return _get_rng().normal(mean, sd)


def lognormal_sample(mean, sd):
    return _get_rng().lognormal(mean, sd)


def t_sample(low, high, t, credibility=90):
    credibility /= 100
    if low > high:
        raise ValueError('`high value` cannot be lower than `low value`')
    elif low == high:
        return low
    else:
        mu = (high + low) / 2
        rangex = (high - low) / 2
        return _get_rng().standard_t(t) * rangex * 0.6/credibility + mu


def log_t_sample(low, high, t, credibility=90):
    credibility /= 100
    if low > high:
        raise ValueError('`high value` cannot be lower than `low value`')
    elif low < 0:
        raise ValueError('log_t_sample cannot handle negative values')
    elif low == high:
        return low
    else:
        log_low = np.log(low)
        log_high = np.log(high)
        mu = (log_high + log_low) / 2
        rangex = (log_high - log_low) / 2
        return np.exp(_get_rng().standard_t(t) * rangex * 0.6/credibility + mu)


def binomial_sample(n, p):
    return _get_rng().binomial(n, p)


def beta_sample(a, b):
    return _get_rng().beta(a, b)


def bernoulli_sample(p):
    return int(event_occurs(p))


def triangular_sample(left, mode, right):
    return _get_rng().triangular(left, mode, right)


def poisson_sample(lam):
    return _get_rng().poisson(lam)


def exponential_sample(scale):
    return _get_rng().exponential(scale)


def gamma_sample(shape, scale):
    return _get_rng().gamma(shape, scale)


def uniform_sample(low, high):
    return _get_rng().uniform(low, high)


def discrete_sample(items):
    weights, values = _process_weights_values(None, items)
    values = [const(v) for v in values]
    return mixture_sample(values, weights)


def mixture_sample(values, weights=None):
    weights, values = _process_weights_values(weights, values)

    if len(values) == 1:
        return sample(values[0])

    r_ = uniform_sample(0, 1)
    weights = np.cumsum(weights)

    for i, dist in enumerate(values):
        weight = weights[i]
        if r_ <= weight:
            return sample(dist)

    return sample(dist)


def sample(var, n=1, lclip=None, rclip=None, verbose=False):
    n = int(n)
    if n > 1:
        if verbose:
            return np.array([sample(var,
                                    n=1,
                                    lclip=lclip,
                                    rclip=rclip) for _ in tqdm(range(n))])
        else:
            return np.array([sample(var,
                                    n=1,
                                    lclip=lclip,
                                    rclip=rclip) for _ in range(n)])
    elif n <= 0:
        raise ValueError('n must be >= 1')

    if callable(var):
        out = var()

    elif not isinstance(var, BaseDistribution):
        raise ValueError('input to sample is malformed - must be a distribution')

    elif var.type == 'const':
        out = var.x

    elif var.type == 'uniform':
        out = uniform_sample(var.x, var.y)

    elif var.type == 'discrete':
        out = discrete_sample(var.items)

    elif var.type == 'norm':
        out = normal_sample(mean=var.mean, sd=var.sd)

    elif var.type == 'lognorm':
        out = lognormal_sample(mean=var.mean, sd=var.sd)

    elif var.type == 'binomial':
        out = binomial_sample(n=var.n, p=var.p)

    elif var.type == 'beta':
        out = beta_sample(a=var.a, b=var.b)

    elif var.type == 'bernoulli':
        out = bernoulli_sample(p=var.p)

    elif var.type == 'poisson':
        out = poisson_sample(lam=var.lam)

    elif var.type == 'exponential':
        out = exponential_sample(scale=var.scale)

    elif var.type == 'gamma':
        out = gamma_sample(shape=var.shape, scale=var.scale)

    elif var.type == 'triangular':
        out = triangular_sample(var.left, var.mode, var.right)

    elif var.type == 'tdist':
        out = t_sample(var.x, var.y, var.t, credibility=var.credibility)

    elif var.type == 'log-tdist':
        out = log_t_sample(var.x, var.y, var.t, credibility=var.credibility)

    elif var.type == 'mixture':
        out = mixture_sample(var.dists, var.weights)

    else:
        raise ValueError('{} sampler not found'.format(var.type))

    lclip_ = None
    rclip_ = None
    if not callable(var):
        lclip_ = var.lclip
        rclip_ = var.rclip

    if lclip is None and lclip_ is not None:
        lclip = lclip_
    if rclip is None and rclip_ is not None:
        rclip = rclip_
    if lclip is not None and lclip_ is not None:
        lclip = max(lclip, lclip_)
    if rclip is not None and rclip_ is not None:
        rclip = min(rclip, rclip_)

    if lclip is not None and out < lclip:
        return lclip
    elif rclip is not None and out > rclip:
        return rclip
    else:
        return out
