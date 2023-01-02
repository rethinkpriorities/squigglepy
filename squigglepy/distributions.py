import operator
import numpy as np
from scipy import stats

from .utils import _process_weights_values, _is_numpy, _round


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
        self.fn = None
        self.fn_str = None
        self.lclip = None
        self.rclip = None
        self.lam = None
        self.df = None
        self.items = None
        self.dists = None
        self.weights = None
        self.type = 'base'

    def __str__(self):
        return '<Distribution> {}'.format(self.type)

    def __repr__(self):
        return str(self)


class OperableDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self.type = 'base'

    def __invert__(self):
        from .samplers import sample
        return sample(self)

    def __matmul__(self, n):
        if isinstance(n, int):
            from .samplers import sample
            return sample(self, n=n)
        else:
            raise ValueError

    def __rshift__(self, fn):
        if callable(fn):
            return fn(self)
        elif isinstance(fn, ComplexDistribution):
            return ComplexDistribution(self, fn.left, fn.fn, fn.fn_str, infix=False)
        else:
            raise ValueError

    def __rmatmul__(self, n):
        return self.__matmul__(n)

    def __gt__(self, dist):
        return ComplexDistribution(self, dist, operator.gt, '>')

    def __ge__(self, dist):
        return ComplexDistribution(self, dist, operator.ge, '>=')

    def __lt__(self, dist):
        return ComplexDistribution(self, dist, operator.lt, '<')

    def __le__(self, dist):
        return ComplexDistribution(self, dist, operator.le, '<=')

    def __eq__(self, dist):
        return ComplexDistribution(self, dist, operator.le, '==')

    def __ne__(self, dist):
        return ComplexDistribution(self, dist, operator.le, '!=')

    def __add__(self, dist):
        return ComplexDistribution(self, dist, operator.add, '+')

    def __radd__(self, dist):
        return ComplexDistribution(dist, self, operator.add, '+')

    def __sub__(self, dist):
        return ComplexDistribution(self, dist, operator.sub, '-')

    def __rsub__(self, dist):
        return ComplexDistribution(dist, self, operator.sub, '-')

    def __mul__(self, dist):
        return ComplexDistribution(self, dist, operator.mul, '*')

    def __rmul__(self, dist):
        return ComplexDistribution(dist, self, operator.mul, '*')

    def __truediv__(self, dist):
        return ComplexDistribution(self, dist, operator.truediv, '/')

    def __rtruediv__(self, dist):
        return ComplexDistribution(dist, self, operator.truediv, '/')

    def __floordiv__(self, dist):
        return ComplexDistribution(self, dist, operator.floordiv, '//')

    def __rfloordiv__(self, dist):
        return ComplexDistribution(dist, self, operator.floordiv, '//')

    def __pow__(self, dist):
        return ComplexDistribution(self, dist, operator.pow, '**')

    def __rpow__(self, dist):
        return ComplexDistribution(dist, self, operator.pow, '**')


class ComplexDistribution(OperableDistribution):
    def __init__(self, left, right=None, fn=operator.add, fn_str='+', infix=True):
        super().__init__()
        self.left = left
        self.right = right
        self.fn = fn
        self.fn_str = fn_str
        self.infix = infix
        self.type = 'complex'

    def __str__(self):
        if self.right is None and self.infix:
            out = '<Distribution> {} {}'.format(self.fn_str,
                                                str(self.left).replace('<Distribution> ', ''))
        elif self.right is None and not self.infix:
            out = '<Distribution> {}({})'.format(self.fn_str,
                                                 str(self.left).replace('<Distribution> ', ''))
        elif self.right is not None and self.infix:
            out = '<Distribution> {} {} {}'.format(str(self.left).replace('<Distribution> ', ''),
                                                   self.fn_str,
                                                   str(self.right).replace('<Distribution> ', ''))
        elif self.right is not None and not self.infix:
            out = '<Distribution> {}({}, {})'
            out = out.format(self.fn_str,
                             str(self.left).replace('<Distribution> ', ''),
                             str(self.right).replace('<Distribution> ', ''))
        else:
            raise ValueError
        return out


def _get_fname(f, name):
    if name is None:
        if isinstance(f, np.vectorize):
            name = f.pyfunc.__name__
        else:
            name = f.__name__
    return name


def dist_fn(dist1, dist2=None, fn=None, name=None):
    """
    Initialize a distribution that has a custom function applied to the result.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function or list
        Typically, the distribution to apply the function to. Could also be a function
        or list of functions if ``dist_fn`` is being used in a pipe.
    dist2 : Distribution or function or list or None
        Typically, the second distribution to apply the function to if the function takes
        two arguments. Could also be a function or list of functions if ``dist_fn`` is
        being used in a pipe.
    fn : function or None
        The function to apply to the distribution(s).
    name : str or None
        By default, ``fn.__name__`` will be used to name the function. But you can pass
        a custom name.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated
        when it is sampled.

    Examples
    --------
    >>> def double(x):
    >>>     return x * 2
    >>> dist_fn(norm(0, 1), double)
    <Distribution> double(norm(mean=0.5, sd=0.3))
    >>> norm(0, 1) >> dist_fn(double)
    <Distribution> double(norm(mean=0.5, sd=0.3))
    """
    if isinstance(dist1, list) and callable(dist1[0]) and dist2 is None and fn is None:
        fn = dist1

        def out_fn(d):
            out = d
            for f in fn:
                out = ComplexDistribution(out,
                                          None,
                                          fn=f,
                                          fn_str=_get_fname(f, name),
                                          infix=False)
            return out

        return out_fn

    if callable(dist1) and dist2 is None and fn is None:
        return lambda d: dist_fn(d, fn=dist1)

    if isinstance(dist2, list) and callable(dist2[0]) and fn is None:
        fn = dist2
        dist2 = None

    if callable(dist2) and fn is None:
        fn = dist2
        dist2 = None

    if not isinstance(fn, list):
        fn = [fn]

    out = dist1
    for f in fn:
        out = ComplexDistribution(out,
                                  dist2,
                                  fn=f,
                                  fn_str=_get_fname(f, name),
                                  infix=False)

    return out


def dist_max(dist1, dist2):
    """
    Initialize the calculation of the maximum value of two distributions.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and determine the max of.
    dist2 : Distribution
        The second distribution to sample and determine the max of.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated
        when it is sampled.

    Examples
    --------
    >>> dist_max(norm(0, 1), norm(1, 2))
    <Distribution> max(norm(mean=0.5, sd=0.3), norm(mean=1.5, sd=0.3))
    """
    return dist_fn(dist1, dist2, np.maximum, name='max')


def dist_min(dist1, dist2):
    """
    Initialize the calculation of the minimum value of two distributions.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and determine the min of.
    dist2 : Distribution
        The second distribution to sample and determine the min of.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_min(norm(0, 1), norm(1, 2))
    <Distribution> min(norm(mean=0.5, sd=0.3), norm(mean=1.5, sd=0.3))
    """
    return dist_fn(dist1, dist2, np.minimum, name='min')


def dist_round(dist1, digits=0):
    """
    Initialize the rounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then round.
    digits : int
        The number of digits to round to.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_round(norm(0, 1))
    <Distribution> round(norm(mean=0.5, sd=0.3), 0)
    """
    if isinstance(dist1, int) and digits == 0:
        return lambda d: dist_round(d, digits=dist1)
    else:
        return dist_fn(dist1, digits, _round, name='round')


def dist_ceil(dist1):
    """
    Initialize the ceiling rounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then ceiling round.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_ceil(norm(0, 1))
    <Distribution> ceil(norm(mean=0.5, sd=0.3))
    """
    return dist_fn(dist1, None, np.ceil)


def dist_floor(dist1):
    """
    Initialize the floor rounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then floor round.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_floor(norm(0, 1))
    <Distribution> floor(norm(mean=0.5, sd=0.3))
    """
    return dist_fn(dist1, None, np.floor)


@np.vectorize
def _lclip(n, val=None):
    if val is None:
        return n
    else:
        return val if n < val else n


def lclip(dist1, val=None):
    """
    Initialize the clipping/bounding of the output of the distribution by the lower value.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function
        The distribution to clip. If this is a funciton, it will return a partial that will
        be suitable for use in piping.
    val : int or float or None
        The value to use as the lower bound for clipping.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> lclip(norm(0, 1), 0.5)
    <Distribution> lclip(norm(mean=0.5, sd=0.3), 0.5)
    """
    if (isinstance(dist1, int) or isinstance(dist1, float)) and val is None:
        return lambda d: lclip(d, dist1)
    else:
        return dist_fn(dist1, val, _lclip, name='lclip')


@np.vectorize
def _rclip(n, val=None):
    if val is None:
        return n
    else:
        return val if n > val else n


def rclip(dist1, val=None):
    """
    Initialize the clipping/bounding of the output of the distribution by the upper value.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function
        The distribution to clip. If this is a funciton, it will return a partial that will
        be suitable for use in piping.
    val : int or float or None
        The value to use as the upper bound for clipping.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> rclip(norm(0, 1), 0.5)
    <Distribution> rclip(norm(mean=0.5, sd=0.3), 0.5)
    """
    if (isinstance(dist1, int) or isinstance(dist1, float)) and val is None:
        return lambda d: rclip(d, dist1)
    else:
        return dist_fn(dist1, val, _rclip, name='rclip')


def clip(dist1, left, right=None):
    """
    Initialize the clipping/bounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function
        The distribution to clip. If this is a funciton, it will return a partial that will
        be suitable for use in piping.
    left : int or float or None
        The value to use as the lower bound for clipping.
    right : int or float or None
        The value to use as the upper bound for clipping.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> clip(norm(0, 1), 0.5, 0.9)
    <Distribution> rclip(lclip(norm(mean=0.5, sd=0.3), 0.5), 0.9)
    """
    if ((isinstance(dist1, int) or isinstance(dist1, float)) and
       (isinstance(left, int) or isinstance(left, float)) and right is None):
        return lambda d: rclip(lclip(d, dist1), left)
    else:
        return rclip(lclip(dist1, left), right)


class ConstantDistribution(OperableDistribution):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.type = 'const'

    def __str__(self):
        return '<Distribution> {}({})'.format(self.type, self.x)


def const(x):
    """
    Initialize a constant distribution.

    Constant distributions always return the same value no matter what.

    Parameters
    ----------
    x : anything
        The value the constant distribution should always return.

    Returns
    -------
    ConstantDistribution

    Examples
    --------
    >>> const(1)
    <Distribution> const(1)
    """
    return ConstantDistribution(x)


class UniformDistribution(OperableDistribution):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.type = 'uniform'

    def __str__(self):
        return '<Distribution> {}({}, {})'.format(self.type, self.x, self.y)


def uniform(x, y):
    """
    Initialize a uniform random distribution.

    Parameters
    ----------
    x : float
        The smallest value the uniform distribution will return.
    y : float
        The largest value the uniform distribution will return.

    Returns
    -------
    UniformDistribution

    Examples
    --------
    >>> uniform(0, 1)
    <Distribution> uniform(0, 1)
    """
    return UniformDistribution(x=x, y=y)


class NormalDistribution(OperableDistribution):
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
    """
    Initialize a normal distribution.

    Can be defined either via a credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI) or defined via ``mean`` and ``sd``.

    Parameters
    ----------
    x : float
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    credibility : float
        The range of the credibility interval. Defaults to 90. Ignored if the distribution is
        defined instead by ``mean`` and ``sd``.
    mean : float or None
        The mean of the normal distribution. If not defined, defaults to 0.
    sd : float
        The standard deviation of the normal distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    NormalDistribution

    Examples
    --------
    >>> norm(0, 1)
    <Distribution> norm(mean=0.5, sd=0.3)
    >>> norm(mean=1, sd=2)
    <Distribution> norm(mean=1, sd=2)
    """
    return NormalDistribution(x=x, y=y, credibility=credibility, mean=mean, sd=sd,
                              lclip=lclip, rclip=rclip)


class LognormalDistribution(OperableDistribution):
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
        if self.x is not None and self.x <= 0:
            raise ValueError('lognormal distribution must have values >= 0.')

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
    """
    Initialize a lognormal distribution.

    Can be defined either via a credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI) or defined via ``mean`` and ``sd``.

    Parameters
    ----------
    x : float
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
        Must be a value greater than 0.
    y : float
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
        Must be a value greater than 0.
    credibility : float
        The range of the credibility interval. Defaults to 90. Ignored if the distribution is
        defined instead by ``mean`` and ``sd``.
    mean : float or None
        The mean of the lognormal distribution. If not defined, defaults to 0.
    sd : float
        The standard deviation of the lognormal distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    LognormalDistribution

    Examples
    --------
    >>> lognorm(1, 10)
    <Distribution> lognorm(mean=1.15, sd=0.7)
    >>> lognorm(mean=1, sd=2)
    <Distribution> lognorm(mean=1, sd=2)
    """
    return LognormalDistribution(x=x, y=y, credibility=credibility, mean=mean, sd=sd,
                                 lclip=lclip, rclip=rclip)


def to(x, y, credibility=90, lclip=None, rclip=None):
    """
    Initialize a distribution from ``x`` to ``y``.

    The distribution will be lognormal by default, unless ``x`` is less than or equal to 0,
    in which case it will become a normal distribution.

    The distribution will default to be a 90% credible interval between ``x`` and ``y`` unless
    ``credibility`` is passed.

    Parameters
    ----------
    x : float
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    credibility : float
        The range of the credibility interval. Defaults to 90.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    ``LognormalDistribution`` if ``x`` > 0, otherwise a ``NormalDistribution``

    Examples
    --------
    >>> to(1, 10)
    <Distribution> lognorm(mean=1.15, sd=0.7)
    >>> to(-10, 10)
    <Distribution> norm(mean=0.0, sd=6.08)
    """
    if x > 0:
        return lognorm(x=x, y=y, credibility=credibility, lclip=lclip, rclip=rclip)
    else:
        return norm(x=x, y=y, credibility=credibility, lclip=lclip, rclip=rclip)


class BinomialDistribution(OperableDistribution):
    def __init__(self, n, p):
        super().__init__()
        self.n = n
        self.p = p
        self.type = 'binomial'
        if self.p < 0 or self.p > 1:
            raise ValueError('p must be between 0 and 1')

    def __str__(self):
        return '<Distribution> {}(n={}, p={})'.format(self.type, self.n, self.p)


def binomial(n, p):
    """
    Initialize a binomial distribution.

    Parameters
    ----------
    n : int
        The number of trials.
    p : float
        The probability of success for each trial. Must be between 0 and 1.

    Returns
    -------
    BinomialDistribution

    Examples
    --------
    >>> binomial(1, 0.1)
    <Distribution> binomial(1, 0.1)
    """
    return BinomialDistribution(n=n, p=p)


class BetaDistribution(OperableDistribution):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.type = 'beta'

    def __str__(self):
        return '<Distribution> {}(a={}, b={})'.format(self.type, self.a, self.b)


def beta(a, b):
    """
    Initialize a beta distribution.

    Parameters
    ----------
    a : float
        The alpha shape value of the distribution. Typically takes the value of the
        number of trials that resulted in a success.
    b : float
        The beta shape value of the distribution. Typically takes the value of the
        number of trials that resulted in a failure.

    Returns
    -------
    BetaDistribution

    Examples
    --------
    >>> beta(1, 2)
    <Distribution> beta(1, 2)
    """
    return BetaDistribution(a, b)


class BernoulliDistribution(OperableDistribution):
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
    """
    Initialize a Bernoulli distribution.

    Parameters
    ----------
    p : float
        The probability of the binary event. Must be between 0 and 1.

    Returns
    -------
    BernoulliDistribution

    Examples
    --------
    >>> bernoulli(0.1)
    <Distribution> bernoulli(p=0.1)
    """
    return BernoulliDistribution(p)


class DiscreteDistribution(OperableDistribution):
    def __init__(self, items):
        super().__init__()
        if (not isinstance(items, dict) and
           not isinstance(items, list) and
           not _is_numpy(items)):
            raise ValueError('inputs to discrete must be a dict or list')
        self.items = list(items) if _is_numpy(items) else items
        self.type = 'discrete'


def discrete(items):
    """
    Initialize a discrete distribution (aka categorical distribution).

    Parameters
    ----------
    items : list or dict
        The values that the discrete distribution will return and their associated
        weights (or likelihoods of being returned when sampled).

    Returns
    -------
    DiscreteDistribution

    Examples
    --------
    >>> discrete({0: 0.1, 1: 0.9})  # 10% chance of returning 0, 90% chance of returning 1
    <Distribution> discrete
    >>> discrete([[0.1, 0], [0.9, 1]])  # Different notation for the same thing.
    <Distribution> discrete
    >>> discrete([0, 1, 2])  # When no weights are given, all have equal chance of happening.
    <Distribution> discrete
    >>> discrete({'a': 0.1, 'b': 0.9})  # Values do not have to be numbers.
    <Distribution> discrete
    """
    return DiscreteDistribution(items)


class TDistribution(OperableDistribution):
    def __init__(self, x=None, y=None, t=20, credibility=90, lclip=None, rclip=None):
        super().__init__()
        self.x = x
        self.y = y
        self.t = t
        self.df = t
        self.credibility = credibility
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'tdist'

        if (self.x is None or self.y is None) and not (self.x is None and self.y is None):
            raise ValueError('must define either both `x` and `y` or neither.')
        elif self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError('`high value` cannot be lower than `low value`')

        if self.x is None:
            self.credibility = None

    def __str__(self):
        if self.x is not None:
            out = '<Distribution> {}(x={}, y={}, t={}'.format(self.type, self.x, self.y, self.t)
        else:
            out = '<Distribution> {}(t={}'.format(self.type, self.t)
        if self.credibility != 90 and self.credibility is not None:
            out += ', credibility={}'.format(self.credibility)
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def tdist(x=None, y=None, t=20, credibility=90, lclip=None, rclip=None):
    """
    Initialize a t-distribution.

    Is defined either via a loose credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI). Unlike the normal and lognormal distributions, this
    credible interval is an approximation and is not precisely defined.

    If ``x`` and ``y`` are not defined, can just return a classic t-distribution defined via
    ``t`` as the number of degrees of freedom.

    Parameters
    ----------
    x : float or None
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float or None
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    t : float
        The number of degrees of freedom of the t-distribution. Defaults to 20.
    credibility : float
        The range of the credibility interval. Defaults to 90.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    TDistribution

    Examples
    --------
    >>> tdist(0, 1, 2)
    <Distribution> tdist(x=0, y=1, t=2)
    >>> tdist()
    <Distribution> tdist(t=1)
    """
    return TDistribution(x=x, y=y, t=t, credibility=credibility, lclip=lclip, rclip=rclip)


class LogTDistribution(OperableDistribution):
    def __init__(self, x=None, y=None, t=1, credibility=90, lclip=None, rclip=None):
        super().__init__()
        self.x = x
        self.y = y
        self.t = t
        self.df = t
        self.credibility = credibility
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'log_tdist'

        if (self.x is None or self.y is None) and not (self.x is None and self.y is None):
            raise ValueError('must define either both `x` and `y` or neither.')
        elif self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError('`high value` cannot be lower than `low value`')

        if self.x is None:
            self.credibility = None

    def __str__(self):
        if self.x is not None:
            out = '<Distribution> {}(x={}, y={}, t={}'.format(self.type, self.x, self.y, self.t)
        else:
            out = '<Distribution> {}(t={}'.format(self.type, self.t)
        if self.credibility != 90 and self.credibility is not None:
            out += ', credibility={}'.format(self.credibility)
        if self.lclip is not None:
            out += ', lclip={}'.format(self.lclip)
        if self.rclip is not None:
            out += ', rclip={}'.format(self.rclip)
        out += ')'
        return out


def log_tdist(x=None, y=None, t=1, credibility=90, lclip=None, rclip=None):
    """
    Initialize a log t-distribution, which is a t-distribution in log-space.

    Is defined either via a loose credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI). Unlike the normal and lognormal distributions, this
    credible interval is an approximation and is not precisely defined.

    If ``x`` and ``y`` are not defined, can just return a classic t-distribution defined via
    ``t`` as the number of degrees of freedom, but in log-space.

    Parameters
    ----------
    x : float or None
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float or None
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    t : float
        The number of degrees of freedom of the t-distribution. Defaults to 1.
    credibility : float
        The range of the credibility interval. Defaults to 90.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    LogTDistribution

    Examples
    --------
    >>> log_tdist(0, 1, 2)
    <Distribution> log_tdist(x=0, y=1, t=2)
    >>> log_tdist()
    <Distribution> log_tdist(t=1)
    """
    return LogTDistribution(x=x, y=y, t=t, credibility=credibility, lclip=lclip, rclip=rclip)


class TriangularDistribution(OperableDistribution):
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
    """
    Initialize a triangular distribution.

    Parameters
    ----------
    left : float
        The smallest value of the triangular distribution.
    mode : float
        The most common value of the triangular distribution.
    right : float
        The largest value of the triangular distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    TriangularDistribution

    Examples
    --------
    >>> triangular(1, 2, 3)
    <Distribution> triangular(1, 2, 3)
    """
    return TriangularDistribution(left=left, mode=mode, right=right, lclip=lclip, rclip=rclip)


class PoissonDistribution(OperableDistribution):
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
    """
    Initialize a poisson distribution.

    Parameters
    ----------
    lam : float
        The lambda value of the poisson distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    PoissonDistribution

    Examples
    --------
    >>> poisson(1)
    <Distribution> poisson(1)
    """
    return PoissonDistribution(lam=lam, lclip=lclip, rclip=rclip)


class ChiSquareDistribution(OperableDistribution):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.type = 'chisquare'
        if self.df <= 0:
            raise ValueError('df must be positive')

    def __str__(self):
        return '<Distribution> {}({})'.format(self.type, self.df)


def chisquare(df):
    """
    Initialize a chi-square distribution.

    Parameters
    ----------
    df : float
        The degrees of freedom. Must be positive.

    Returns
    -------
    ChiSquareDistribution

    Examples
    --------
    >>> chisquare(2)
    <Distribution> chiaquare(2)
    """
    return ChiSquareDistribution(df=df)


class ExponentialDistribution(OperableDistribution):
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
    """
    Initialize an exponential distribution.

    Parameters
    ----------
    scale : float
        The scale value of the exponential distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    ExponentialDistribution

    Examples
    --------
    >>> exponential(1)
    <Distribution> exponential(1)
    """
    return ExponentialDistribution(scale=scale, lclip=lclip, rclip=rclip)


class GammaDistribution(OperableDistribution):
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
    """
    Initialize a gamma distribution.

    Parameters
    ----------
    shape : float
        The shape value of the exponential distribution.
    scale : float
        The scale value of the exponential distribution. Defaults to 1.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    GammaDistribution

    Examples
    --------
    >>> gamma(10, 1)
    <Distribution> gamma(shape=10, scale=1)
    """
    return GammaDistribution(shape=shape, scale=scale, lclip=lclip, rclip=rclip)


class MixtureDistribution(OperableDistribution):
    def __init__(self, dists, weights=None, relative_weights=None, lclip=None, rclip=None):
        super().__init__()
        weights, dists = _process_weights_values(weights, relative_weights, dists)
        self.dists = dists
        self.weights = weights
        self.lclip = lclip
        self.rclip = rclip
        self.type = 'mixture'


def mixture(dists, weights=None, relative_weights=None, lclip=None, rclip=None):
    """
    Initialize a mixture distribution, which is a combination of different distributions.

    Parameters
    ----------
    dists : list or dict
        The distributions to mix. Can also be defined as a list of weights and distributions.
    weights : list or None
        The weights for each distribution.
    relative_weights : list or None
        Relative weights, which if given will be weights that are normalized
        to sum to 1.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    MixtureDistribution

    Examples
    --------
    >>> mixture([norm(1, 2), norm(3, 4)], weights=[0.1, 0.9])
    <Distribution> mixture
    >>> mixture([[0.1, norm(1, 2)], [0.9, norm(3, 4)]])  # Different notation for the same thing.
    <Distribution> mixture
    >>> mixture([norm(1, 2), norm(3, 4)])  # When no weights are given, all have equal chance
    >>>                                    # of happening.
    <Distribution> mixture
    """
    return MixtureDistribution(dists=dists,
                               weights=weights,
                               relative_weights=relative_weights,
                               lclip=lclip,
                               rclip=rclip)


def zero_inflated(p_zero, dist):
    """
    Initialize an arbitrary zero-inflated distribution.

    Parameters
    ----------
    p_zero : float
        The chance of the distribution returning zero
    dist : Distribution
        The distribution to sample from when not zero

    Returns
    -------
    MixtureDistribution

    Examples
    --------
    >>> zero_inflated(0.6, norm(1, 2))
    <Distribution> mixture
    """
    if p_zero > 1 or p_zero < 0 or not isinstance(p_zero, float):
        raise ValueError('`p_zero` must be between 0 and 1')
    return MixtureDistribution(dists=[0, dist], weights=p_zero)


def inf0(p_zero, dist):
    """
    Initialize an arbitrary zero-inflated distribution.

    Alias for ``zero_inflated``.

    Parameters
    ----------
    p_zero : float
        The chance of the distribution returning zero
    dist : Distribution
        The distribution to sample from when not zero

    Returns
    -------
    MixtureDistribution

    Examples
    --------
    >>> inf0(0.6, norm(1, 2))
    <Distribution> mixture
    """
    return zero_inflated(p_zero=p_zero, dist=dist)
