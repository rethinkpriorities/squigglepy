def to(x, y, credibility=0.9, lclip=None, rclip=None):
    return [x, y, 'log' if x > 0 else 'norm', credibility, lclip, rclip]


def const(x):
    return [x, None, 'const', None, None]


def uniform(x, y):
    return [x, y, 'uniform', None, None]


def norm(x=None, y=None, credibility=0.9, mean=None, sd=None,
         lclip=None, rclip=None):
    if mean is None and sd is None and x is not None and y is not None:
        return [x, y, 'norm', credibility, lclip, rclip]
    elif mean is None and sd is not None and x is None and y is None:
        return [0, sd, 'norm-mean', lclip, rclip]
    elif mean is not None and sd is not None and x is None and y is None:
        return [mean, sd, 'norm-mean', lclip, rclip]
    else:
        raise ValueError


def lognorm(x=None, y=None, credibility=0.9, mean=None, sd=None,
            lclip=None, rclip=None):
    if mean is None and sd is None and x is not None and y is not None:
        return [x, y, 'log', credibility, lclip, rclip]
    elif mean is None and sd is not None and x is None and y is None:
        return [0, sd, 'log-mean', lclip, rclip]
    elif mean is not None and sd is not None and x is None and y is None:
        return [mean, sd, 'log-mean', lclip, rclip]
    else:
        raise ValueError


def binomial(n, p):
    return [n, p, 'binomial', None, None]


def beta(a, b):
    return [a, b, 'beta', None, None]


def bernoulli(p):
    if not isinstance(p, float) or isinstance(p, int):
        raise ValueError('bernoulli p must be a float or int')
    if p < 0 or p > 1:
        raise ValueError('bernoulli p must be 0-1')
    return [p, None, 'bernoulli', None, None]


def discrete(items):
    if not isinstance(items, dict) and not isinstance(items, list):
        raise ValueError('inputs to discrete must be a dict or list')
    return [items, None, 'discrete', None, None]


def tdist(x, y, t, credibility=0.9, lclip=None, rclip=None):
    return [x, y, 'tdist', t, credibility, lclip, rclip]


def log_tdist(x, y, t, credibility=0.9, lclip=None, rclip=None):
    return [x, y, 'log-tdist', t, credibility, lclip, rclip]


def triangular(left, mode, right, lclip=None, rclip=None):
    return [left, mode, 'triangular', right, lclip, rclip]


def poisson(lam, lclip=None, rclip=None):
    return [lam, None, 'poisson', lclip, rclip]


def exponential(scale, lclip=None, rclip=None):
    return [scale, None, 'exponential', lclip, rclip]


def gamma(shape, scale=1, lclip=None, rclip=None):
    return [shape, scale, 'gamma', lclip, rclip]


def mixture(dists, weights=None, lclip=None, rclip=None):
    return [dists, weights, 'mixture', lclip, rclip]
