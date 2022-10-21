import numpy as np

from tqdm import tqdm
from scipy import stats

from .utils import event_occurs, _process_weights_values


def _get_rng():
    from .rng import _squigglepy_internal_rng
    return _squigglepy_internal_rng


def normal_sample(mean, sd):
    """
    Sample a random number according to a normal distribution.

    Parameters
    ----------
    mean : float
        The mean of the normal distribution that is being sampled.
    sd : float
        The standard deviation of the normal distribution that is being sampled.

    Returns
    -------
    float
        A random number sampled from a normal distribution defined by
        ``mean`` and ``sd``.

    Examples
    --------
    >>> set_seed(42)
    >>> normal_sample(0, 1)
    0.30471707975443135
    """
    return _get_rng().normal(mean, sd)


def lognormal_sample(mean, sd):
    """
    Sample a random number according to a lognormal distribution.

    Parameters
    ----------
    mean : float
        The mean of the lognormal distribution that is being sampled.
    sd : float
        The standard deviation of the lognormal distribution that is being sampled.

    Returns
    -------
    float
        A random number sampled from a lognormal distribution defined by
        ``mean`` and ``sd``.

    Examples
    --------
    >>> set_seed(42)
    >>> lognormal_sample(0, 1)
    1.3562412406168636
    """
    return _get_rng().lognormal(mean, sd)


def t_sample(low=None, high=None, t=1, credibility=90):
    """
    Sample a random number according to a t-distribution.

    The t-distribution is defined with degrees of freedom via the ``t``
    parameter. Additionally, a loose credibility interval can be defined
    via the t-distribution using the ``low`` and ``high`` values. This will be a
    90% CI by default unless you change ``credibility.`` Unlike the normal and
    lognormal samplers, this credible interval is an approximation and is
    not precisely defined.

    Parameters
    ----------
    low : float or None
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    high : float or None
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    t : float
        The number of degrees of freedom of the t-distribution. Defaults to 1.
    credibility : float
        The range of the credibility interval. Defaults to 90.

    Returns
    -------
    float
        A random number sampled from a lognormal distribution defined by
        ``mean`` and ``sd``.

    Examples
    --------
    >>> set_seed(42)
    >>> t_sample(1, 2, t=4)
    2.7887113716855985
    """
    if low is None and high is None:
        return _get_rng().standard_t(t)
    elif low is None or high is None:
        raise ValueError('must define either both `x` and `y` or neither.')
    elif low > high:
        raise ValueError('`high value` cannot be lower than `low value`')
    elif low == high:
        return low
    else:
        mu = (high + low) / 2
        cdf_value = 0.5 + 0.5 * (credibility / 100)
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (high - mu) / normed_sigma
        return normal_sample(mu, sigma) / ((chi_square_sample(t) / t) ** 0.5)


def log_t_sample(low=None, high=None, t=1, credibility=90):
    """
    Sample a random number according to a log-t-distribution.

    The log-t-distribution is a t-distribution in log-space. It is defined with
    degrees of freedom via the ``t`` parameter. Additionally, a loose credibility
    interval can be defined via the t-distribution using the ``low`` and ``high``
    values. This will be a 90% CI by default unless you change ``credibility.``
    Unlike the normal and lognormal samplers, this credible interval is an
    approximation and is not precisely defined.

    Parameters
    ----------
    low : float or None
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    high : float or None
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    t : float
        The number of degrees of freedom of the t-distribution. Defaults to 1.
    credibility : float
        The range of the credibility interval. Defaults to 90.

    Returns
    -------
    float
        A random number sampled from a lognormal distribution defined by
        ``mean`` and ``sd``.

    Examples
    --------
    >>> set_seed(42)
    >>> log_t_sample(1, 2, t=4)
    2.052949773846356
    """
    if low is None and high is None:
        return np.exp(_get_rng().standard_t(t))
    elif low > high:
        raise ValueError('`high value` cannot be lower than `low value`')
    elif low < 0:
        raise ValueError('log_t_sample cannot handle negative values')
    elif low == high:
        return low
    else:
        log_low = np.log(low)
        log_high = np.log(high)
        mu = (log_high + log_low) / 2
        cdf_value = 0.5 + 0.5 * (credibility / 100)
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (log_high - mu) / normed_sigma
        return np.exp(normal_sample(mu, sigma) / ((chi_square_sample(t) / t) ** 0.5))


def binomial_sample(n, p):
    """
    Sample a random number according to a binomial distribution.

    Parameters
    ----------
    n : int
        The number of trials.
    p : float
        The probability of success for each trial. Must be between 0 and 1.

    Returns
    -------
    int
        A random number sampled from a binomial distribution defined by
        ``n`` and ``p``. The random number should be between 0 and ``n``.

    Examples
    --------
    >>> set_seed(42)
    >>> binomial_sample(10, 0.1)
    2
    """
    return _get_rng().binomial(n, p)


def beta_sample(a, b):
    """
    Sample a random number according to a beta distribution.

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
    float
        A random number sampled from a beta distribution defined by
        ``a`` and ``b``.

    Examples
    --------
    >>> set_seed(42)
    >>> beta_sample(1, 1)
    0.22145847498048798
    """
    return _get_rng().beta(a, b)


def bernoulli_sample(p):
    """
    Sample 1 with probability ``p`` and 0 otherwise.

    Parameters
    ----------
    p : float
        The probability of success. Must be between 0 and 1.

    Returns
    -------
    int
        Either 0 or 1

    Examples
    --------
    >>> set_seed(42)
    >>> bernoulli_sample(0.5)
    0
    """
    return int(event_occurs(p))


def triangular_sample(left, mode, right):
    """
    Sample a random number according to a triangular distribution.

    Parameters
    ----------
    left : float
        The smallest value of the triangular distribution.
    mode : float
        The most common value of the triangular distribution.
    right : float
        The largest value of the triangular distribution.

    Returns
    -------
    float
        A random number sampled from a triangular distribution.

    Examples
    --------
    >>> set_seed(42)
    >>> triangular_sample(1, 2, 3)
    2.327625176788963
    """
    return _get_rng().triangular(left, mode, right)


def poisson_sample(lam):
    """
    Sample a random number according to a poisson distribution.

    Parameters
    ----------
    lam : float
        The lambda value of the poisson distribution.

    Returns
    -------
    int
        A random number sampled from a poisson distribution.

    Examples
    --------
    >>> set_seed(42)
    >>> poisson_sample(10)
    13
    """
    return _get_rng().poisson(lam)


def exponential_sample(scale):
    """
    Sample a random number according to an exponential distribution.

    Parameters
    ----------
    scale : float
        The scale value of the exponential distribution.

    Returns
    -------
    int
        A random number sampled from an exponential distribution.

    Examples
    --------
    >>> set_seed(42)
    >>> exponential_sample(10)
    24.042086039659946
    """
    return _get_rng().exponential(scale)


def gamma_sample(shape, scale):
    """
    Sample a random number according to a gamma distribution.

    Parameters
    ----------
    shape : float
        The shape value of the exponential distribution.
    scale : float
        The scale value of the exponential distribution. Defaults to 1.

    Returns
    -------
    int
        A random number sampled from an gamma distribution.

    Examples
    --------
    >>> set_seed(42)
    >>> gamma_sample(10, 2)
    21.290716894247602
    """
    return _get_rng().gamma(shape, scale)


def uniform_sample(low, high):
    """
    Sample a random number according to a uniform distribution.

    Parameters
    ----------
    low : float
        The smallest value the uniform distribution will return.
    high : float
        The largest value the uniform distribution will return.

    Returns
    -------
    float
        A random number sampled from a uniform distribution between
        ```low``` and ```high```.

    Examples
    --------
    >>> set_seed(42)
    >>> uniform_sample(0, 1)
    0.7739560485559633
    """
    return _get_rng().uniform(low, high)


def chi_square_sample(df):
    """
    Sample a random number according to a chi-square distribution.

    Parameters
    ----------
    df : float
        The number of degrees of freedom

    Returns
    -------
    float
        A random number sampled from a chi-square distribution.

    Examples
    --------
    >>> set_seed(42)
    >>> chi_square_sample(2)
    4.808417207931989
    """
    return _get_rng().chisquare(df)


def discrete_sample(items):
    """
    Sample a random value from a discrete distribution (aka categorical distribution).

    Parameters
    ----------
    items : list or dict
        The values that the discrete distribution will return and their associated
        weights (or likelihoods of being returned when sampled).

    Returns
    -------
    Various, based on items in ``items``

    Examples
    --------
    >>> set_seed(42)
    >>> # 10% chance of returning 0, 90% chance of returning 1
    >>> discrete_sample({0: 0.1, 1: 0.9})
    1
    >>> discrete_sample([[0.1, 0], [0.9, 1]])  # Different notation for the same thing.
    1
    >>> # When no weights are given, all have equal chance of happening.
    >>> discrete_sample([0, 1, 2])
    2
    >>> discrete_sample({'a': 0.1, 'b': 0.9})  # Values do not have to be numbers.
    'b'
    """
    weights, values = _process_weights_values(None, items)
    from .distributions import const
    values = [const(v) for v in values]
    return mixture_sample(values, weights)


def mixture_sample(values, weights=None):
    """
    Sample a ranom number from a mixture distribution.

    Parameters
    ----------
    values : list or dict
        The distributions to mix. Can also be defined as a list of weights and distributions.
    weights : list or None
        The weights for each distribution.

    Returns
    -------
    Various, based on items in ``values``

    Examples
    --------
    >>> set_seed(42)
    >>> mixture_sample([norm(1, 2), norm(3, 4)], weights=[0.1, 0.9])
    3.183867278765718
    >>> # Different notation for the same thing.
    >>> mixture_sample([[0.1, norm(1, 2)], [0.9, norm(3, 4)]])
    3.7859113725925972
    >>> # When no weights are given, all have equal chance of happening.
    >>> mixture_sample([norm(1, 2), norm(3, 4)])
    1.1041655362137777
    """
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


def sample(dist, n=1, lclip=None, rclip=None, verbose=False):
    """
    Sample random numbers from a given distribution.

    Parameters
    ----------
    dist : Distribution
        The distribution to sample random number from.
    n : int
        The number of random numbers to sample from the distribution. Default to 1.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.
    verbose : bool
        If True, will print out statements on computational progress.

    Returns
    -------
    Various, based on ``dist``.

    Examples
    --------
    >>> set_seed(42)
    >>> sample(norm(1, 2))
    1.592627415218455
    >>> sample(mixture([norm(1, 2), norm(3, 4)]))
    1.7281209657534462
    >>> sample(lognorm(1, 10), n=5, lclip=3)
    array([6.10817361, 3.        , 3.        , 3.45828454, 3.        ])
    """
    n = int(n)
    if n > 1:
        if verbose:
            return np.array([sample(dist,
                                    n=1,
                                    lclip=lclip,
                                    rclip=rclip) for _ in tqdm(range(n))])
        else:
            return np.array([sample(dist,
                                    n=1,
                                    lclip=lclip,
                                    rclip=rclip) for _ in range(n)])
    elif n <= 0:
        raise ValueError('n must be >= 1')

    from .distributions import BaseDistribution

    if callable(dist):
        out = dist()

    elif isinstance(dist, float) or isinstance(dist, int):
        return dist

    elif not isinstance(dist, BaseDistribution):
        raise ValueError('input to sample is malformed - must be a distribution')

    elif dist.type == 'const':
        out = dist.x

    elif dist.type == 'uniform':
        out = uniform_sample(dist.x, dist.y)

    elif dist.type == 'discrete':
        out = discrete_sample(dist.items)

    elif dist.type == 'norm':
        out = normal_sample(mean=dist.mean, sd=dist.sd)

    elif dist.type == 'lognorm':
        out = lognormal_sample(mean=dist.mean, sd=dist.sd)

    elif dist.type == 'binomial':
        out = binomial_sample(n=dist.n, p=dist.p)

    elif dist.type == 'beta':
        out = beta_sample(a=dist.a, b=dist.b)

    elif dist.type == 'bernoulli':
        out = bernoulli_sample(p=dist.p)

    elif dist.type == 'poisson':
        out = poisson_sample(lam=dist.lam)

    elif dist.type == 'chisquare':
        out = chi_square_sample(df=dist.df)

    elif dist.type == 'exponential':
        out = exponential_sample(scale=dist.scale)

    elif dist.type == 'gamma':
        out = gamma_sample(shape=dist.shape, scale=dist.scale)

    elif dist.type == 'triangular':
        out = triangular_sample(dist.left, dist.mode, dist.right)

    elif dist.type == 'tdist':
        out = t_sample(dist.x, dist.y, dist.t, credibility=dist.credibility)

    elif dist.type == 'log_tdist':
        out = log_t_sample(dist.x, dist.y, dist.t, credibility=dist.credibility)

    elif dist.type == 'mixture':
        out = mixture_sample(dist.dists, dist.weights)

    elif dist.type == 'complex':
        out = dist.fn(sample(dist.left), sample(dist.right))

    else:
        raise ValueError('{} sampler not found'.format(dist.type))

    if isinstance(out, BaseDistribution):
        return sample(out)

    lclip_ = None
    rclip_ = None
    if not callable(dist):
        lclip_ = dist.lclip
        rclip_ = dist.rclip

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
