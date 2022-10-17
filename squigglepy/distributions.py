import numpy as np
from scipy import stats

from .utils import _process_weights_values


class BaseDistribution:
    def __init__(self):
        self.x = None
        self.y = None
        self.n = None
        self.p = None
        self.t = None
        self.a = None
        self.b = None
        self.shape = None
        self.scale = None
        self.credibility = None
        self.mean = None
        self.sd = None
        self.left = None
        self.mode = None
        self.right = None
        self.lclip = None
        self.rclip = None
        self.lam = None
        self.items = None
        self.dists = None
        self.weights = None
        self.type = 'BaseDistribution'

    def __str__(self):
        return '<Distribution> {}'.format(self.type)

    def __repr__(self):
        return str(self)


class ConstantDistribution(BaseDistribution):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.type = 'const'

    def __str__(self):
        return '<Distribution> {}({})'.format(self.type, self.x)


def const(x):
    return ConstantDistribution(x)


class UniformDistribution(BaseDistribution):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.type = 'uniform'

    def __str__(self):
        return '<Distribution> {}({}, {})'.format(self.type, self.x, self.y)


def uniform(x, y):
    return UniformDistribution(x=x, y=y)


class NormalDistribution(BaseDistribution):
    def __init__(self, x=None, y=None, mean=None, sd=None,
                 credibility=90, lclip=None, rclip=None):
        super().__init__()
        self.x = x
        self.y = y
        self.credibility = credibility
        self.mean = mean
        self.sd = sd
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'norm'

        if self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError('`high value` cannot be lower than `low value`')

        if (self.x is None or self.y is None) and self.sd is None:
            raise ValueError('must define either x/y or mean/sd')
        elif (self.x is not None or self.y is not None) and self.sd is not None:
            raise ValueError('must define either x/y or mean/sd -- cannot define both')
        elif self.sd is not None and self.mean is None:
            self.mean = 0

        if self.mean is None and self.sd is None:
            self.mean = (self.x + self.y) / 2
            cdf_value = 0.5 + 0.5 * (self.credibility / 100)
            normed_sigma = stats.norm.ppf(cdf_value)
            self.sd = (self.y - self.mean) / normed_sigma

    def __str__(self):
        out = '<Distribution> {}(mean={}, sd={}'.format(self.type,
                                                        round(self.mean, 2),
                                                        round(self.sd, 2))
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def norm(x=None, y=None, credibility=90, mean=None, sd=None,
         lclip=None, rclip=None):
    return NormalDistribution(x=x, y=y, credibility=credibility, mean=mean, sd=sd,
                              lclip=lclip, rclip=rclip)


class LognormalDistribution(BaseDistribution):
    def __init__(self, x=None, y=None, mean=None, sd=None,
                 credibility=90, lclip=None, rclip=None):
        super().__init__()
        self.x = x
        self.y = y
        self.credibility = credibility
        self.mean = mean
        self.sd = sd
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'lognorm'

        if self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError('`high value` cannot be lower than `low value`')

        if (self.x is None or self.y is None) and self.sd is None:
            raise ValueError('must define either x/y or mean/sd')
        elif (self.x is not None or self.y is not None) and self.sd is not None:
            raise ValueError('must define either x/y or mean/sd -- cannot define both')
        elif self.sd is not None and self.mean is None:
            self.mean = 0

        if self.mean is None and self.sd is None:
            self.mean = (np.log(self.x) + np.log(self.y)) / 2
            cdf_value = 0.5 + 0.5 * (self.credibility / 100)
            normed_sigma = stats.norm.ppf(cdf_value)
            self.sd = (np.log(self.y) - self.mean) / normed_sigma

    def __str__(self):
        out = '<Distribution> {}(mean={}, sd={}'.format(self.type,
                                                        round(self.mean, 2),
                                                        round(self.sd, 2))
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def lognorm(x=None, y=None, credibility=90, mean=None, sd=None,
            lclip=None, rclip=None):
    return LognormalDistribution(x=x, y=y, credibility=credibility, mean=mean, sd=sd,
                                 lclip=lclip, rclip=rclip)


def to(x, y, credibility=90, lclip=None, rclip=None):
    if x > 0:
        return lognorm(x=x, y=y, credibility=credibility, lclip=lclip, rclip=rclip)
    else:
        return norm(x=x, y=y, credibility=credibility, lclip=lclip, rclip=rclip)


class BinomialDistribution(BaseDistribution):
    def __init__(self, n, p):
        super().__init__()
        self.n = n
        self.p = p
        self.type = 'binomial'

    def __str__(self):
        return '<Distribution> {}(n={}, p={})'.format(self.type, self.n, self.p)


def binomial(n, p):
    return BinomialDistribution(n=n, p=p)


class BetaDistribution(BaseDistribution):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.type = 'beta'

    def __str__(self):
        return '<Distribution> {}(a={}, b={})'.format(self.type, self.a, self.b)


def beta(a, b):
    return BetaDistribution(a, b)


class BernoulliDistribution(BaseDistribution):
    def __init__(self, p):
        super().__init__()
        if not isinstance(p, float) or isinstance(p, int):
            raise ValueError('bernoulli p must be a float or int')
        if p < 0 or p > 1:
            raise ValueError('bernoulli p must be 0-1')
        self.p = p
        self.type = 'bernoulli'

    def __str__(self):
        return '<Distribution> {}(p={})'.format(self.type, self.p)


def bernoulli(p):
    return BernoulliDistribution(p)


class DiscreteDistribution(BaseDistribution):
    def __init__(self, items):
        super().__init__()
        if not isinstance(items, dict) and not isinstance(items, list):
            raise ValueError('inputs to discrete must be a dict or list')
        self.items = items
        self.type = 'discrete'


def discrete(items):
    return DiscreteDistribution(items)


class TDistribution(BaseDistribution):
    def __init__(self, x, y, t, credibility=90, lclip=None, rclip=None):
        super().__init__()
        self.x = x
        self.y = y
        self.t = t
        self.credibility = credibility
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'tdist'

    def __str__(self):
        out = '<Distribution> {}(x={}, y={}, t={}'.format(self.type, self.x, self.y, self.t)
        if self.credibility != 90:
            out += ', credibility={}'.format(self.credibility)
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def tdist(x, y, t, credibility=90, lclip=None, rclip=None):
    return TDistribution(x=x, y=y, t=t, credibility=credibility, lclip=lclip, rclip=rclip)


class LogTDistribution(BaseDistribution):
    def __init__(self, x, y, t, credibility=90, lclip=None, rclip=None):
        super().__init__()
        self.x = x
        self.y = y
        self.t = t
        self.credibility = credibility
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'log_tdist'

    def __str__(self):
        out = '<Distribution> {}(x={}, y={}, t={}'.format(self.type, self.x, self.y, self.t)
        if self.credibility != 90:
            out += ', credibility={}'.format(self.credibility)
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def log_tdist(x, y, t, credibility=90, lclip=None, rclip=None):
    return LogTDistribution(x=x, y=y, t=t, credibility=credibility, lclip=lclip, rclip=rclip)


class TriangularDistribution(BaseDistribution):
    def __init__(self, left, mode, right, lclip=None, rclip=None):
        super().__init__()
        self.left = left
        self.mode = mode
        self.right = right
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'triangular'

    def __str__(self):
        out = '<Distribution> {}({}, {}, {}'.format(self.type, self.left, self.mode, self.right)
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def triangular(left, mode, right, lclip=None, rclip=None):
    return TriangularDistribution(left=left, mode=mode, right=right, lclip=lclip, rclip=rclip)


class PoissonDistribution(BaseDistribution):
    def __init__(self, lam, lclip=None, rclip=None):
        super().__init__()
        self.lam = lam
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'poisson'

    def __str__(self):
        out = '<Distribution> {}({}'.format(self.type, self.lam)
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def poisson(lam, lclip=None, rclip=None):
    return PoissonDistribution(lam=lam, lclip=lclip, rclip=rclip)


class ExponentialDistribution(BaseDistribution):
    def __init__(self, scale, lclip=None, rclip=None):
        super().__init__()
        self.scale = scale
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'exponential'

    def __str__(self):
        out = '<Distribution> {}({}'.format(self.type, self.scale)
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def exponential(scale, lclip=None, rclip=None):
    return ExponentialDistribution(scale=scale, lclip=lclip, rclip=rclip)


class GammaDistribution(BaseDistribution):
    def __init__(self, shape, scale=1, lclip=None, rclip=None):
        super().__init__()
        self.shape = shape
        self.scale = scale
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'gamma'

    def __str__(self):
        out = '<Distribution> {}(shape={}, scale={}'.format(self.type, self.shape, self.scale)
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def gamma(shape, scale=1, lclip=None, rclip=None):
    return GammaDistribution(shape=shape, scale=scale, lclip=lclip, rclip=rclip)


class MixtureDistribution(BaseDistribution):
    def __init__(self, dists, weights=None, lclip=None, rclip=None):
        super().__init__()
        weights, dists = _process_weights_values(weights, dists)
        self.dists = dists
        self.weights = weights
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'mixture'


def mixture(dists, weights=None, lclip=None, rclip=None):
    return MixtureDistribution(dists=dists, weights=weights, lclip=lclip, rclip=rclip)
