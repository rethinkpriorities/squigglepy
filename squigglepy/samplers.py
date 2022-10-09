import numpy as np

from scipy import stats
from tqdm import tqdm

from .distributions import const
from .utils import event_occurs, _process_weights_values


def _get_rng():
    from .rng import _squigglepy_internal_rng
    return _squigglepy_internal_rng


def normal_sample(low=None, high=None, mean=None, sd=None, credibility=0.9):
    if mean is None:
        if low > high:
            raise ValueError('`high value` cannot be lower than `low value`')
        elif low == high:
            return low
        mu = (high + low) / 2
        cdf_value = 0.5 + 0.5 * credibility
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (high - mu) / normed_sigma
    else:
        mu = mean
        sigma = sd
    return _get_rng().normal(mu, sigma)


def lognormal_sample(low=None, high=None, mean=None, sd=None,
                     credibility=0.9):
    if (low is not None and low < 0) or (mean is not None and mean < 0):
        raise ValueError('lognormal_sample cannot handle negative values')
    if mean is None:
        if low > high:
            raise ValueError('`high value` cannot be lower than `low value`')
        elif low == high:
            return low
        log_low = np.log(low)
        log_high = np.log(high)
        mu = (log_high + log_low) / 2
        cdf_value = 0.5 + 0.5 * credibility
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (log_high - mu) / normed_sigma
    else:
        mu = mean
        sigma = sd
    return _get_rng().lognormal(mu, sigma)


def t_sample(low, high, t, credibility=0.9):
    if low > high:
        raise ValueError('`high value` cannot be lower than `low value`')
    elif low == high:
        return low
    else:
        mu = (high + low) / 2
        rangex = (high - low) / 2
        return _get_rng().standard_t(t) * rangex * 0.6/credibility + mu


def log_t_sample(low, high, t, credibility=0.9):
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
    if isinstance(items, dict):
        values = [const(k) for k in items.keys()]
        weights = list(items.values())
    elif isinstance(items, list):
        if isinstance(items[0], list):
            weights = [i[0] for i in items]
            values = [const(i[1]) for i in items]
        else:
            weights = None
            values = [const(i) for i in items]
    else:
        raise ValueError('inputs to discrete_sample must be a dict or list')

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

    elif not isinstance(var, list) and not (len(var) == 5 or len(var) == 6):
        raise ValueError('input to sample is malformed - must be sample data')

    elif var[2] == 'const':
        out = var[0]

    elif var[2] == 'uniform':
        out = uniform_sample(var[0], var[1])

    elif var[2] == 'discrete':
        out = discrete_sample(var[0])

    elif var[2] == 'norm':
        out = normal_sample(var[0], var[1], credibility=var[3])

    elif var[2] == 'norm-mean':
        out = normal_sample(mean=var[0], sd=var[1], credibility=var[3])

    elif var[2] == 'log':
        out = lognormal_sample(var[0], var[1], credibility=var[3])

    elif var[2] == 'log-mean':
        out = lognormal_sample(mean=var[0], sd=var[1])

    elif var[2] == 'binomial':
        out = binomial_sample(n=var[0], p=var[1])

    elif var[2] == 'beta':
        out = beta_sample(a=var[0], b=var[1])

    elif var[2] == 'bernoulli':
        out = bernoulli_sample(p=var[0])

    elif var[2] == 'poisson':
        out = poisson_sample(lam=var[0])

    elif var[2] == 'exponential':
        out = exponential_sample(scale=var[0])

    elif var[2] == 'gamma':
        out = gamma_sample(shape=var[0], scale=var[1])

    elif var[2] == 'triangular':
        out = triangular_sample(var[0], var[1], var[3])

    elif var[2] == 'tdist':
        out = t_sample(var[0], var[1], var[3], credibility=var[4])

    elif var[2] == 'log-tdist':
        out = log_t_sample(var[0], var[1], var[3], credibility=var[4])

    elif var[2] == 'mixture':
        out = mixture_sample(var[0], var[1])

    else:
        raise ValueError('{} sampler not found'.format(var[2]))

    lclip_ = None
    rclip_ = None
    if not callable(var):
        if var[2] == 'tdist' or var[2] == 'log-tdist':
            lclip_ = var[5]
            rclip_ = var[6]
        elif var[2] == 'norm' or var[2] == 'log' or var[2] == 'triangular':
            lclip_ = var[4]
            rclip_ = var[5]
        else:
            lclip_ = var[3]
            rclip_ = var[4]

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
