import numpy as np

from tqdm import tqdm
from scipy import stats

from .distributions import rclip as fn_rclip, lclip as fn_lclip
from .utils import _process_weights_values, _is_dist, _simplify, _safe_len


def _get_rng():
    from .rng import _squigglepy_internal_rng
    return _squigglepy_internal_rng


def normal_sample(mean, sd, samples=1):
    """
    Sample a random number according to a normal distribution.

    Parameters
    ----------
    mean : float
        The mean of the normal distribution that is being sampled.
    sd : float
        The standard deviation of the normal distribution that is being sampled.
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().normal(mean, sd, samples))


def lognormal_sample(mean, sd, samples=1):
    """
    Sample a random number according to a lognormal distribution.

    Parameters
    ----------
    mean : float
        The mean of the lognormal distribution that is being sampled.
    sd : float
        The standard deviation of the lognormal distribution that is being sampled.
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().lognormal(mean, sd, samples))


def t_sample(low=None, high=None, t=1, samples=1, credibility=90):
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
    samples : int
        The number of samples to return.
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
        return _simplify(normal_sample(mu, sigma, samples) /
                         ((chi_square_sample(t, samples) / t) ** 0.5))


def log_t_sample(low=None, high=None, t=1, samples=1, credibility=90):
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
    samples : int
        The number of samples to return.
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
        return _simplify(np.exp(normal_sample(mu, sigma, samples) /
                         ((chi_square_sample(t, samples) / t) ** 0.5)))


def binomial_sample(n, p, samples=1):
    """
    Sample a random number according to a binomial distribution.

    Parameters
    ----------
    n : int
        The number of trials.
    p : float
        The probability of success for each trial. Must be between 0 and 1.
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().binomial(n, p, samples))


def beta_sample(a, b, samples=1):
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
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().beta(a, b, samples))


def bernoulli_sample(p, samples=1):
    """
    Sample 1 with probability ``p`` and 0 otherwise.

    Parameters
    ----------
    p : float
        The probability of success. Must be between 0 and 1.
    samples : int
        The number of samples to return.

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
    a = uniform_sample(0, 1, samples)
    if _safe_len(a) == 1:
        return int(a < p)
    else:
        return (a < p).astype(int)


def triangular_sample(left, mode, right, samples=1):
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
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().triangular(left, mode, right, samples))


def poisson_sample(lam, samples=1):
    """
    Sample a random number according to a poisson distribution.

    Parameters
    ----------
    lam : float
        The lambda value of the poisson distribution.
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().poisson(lam, samples))


def exponential_sample(scale, samples=1):
    """
    Sample a random number according to an exponential distribution.

    Parameters
    ----------
    scale : float
        The scale value of the exponential distribution.
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().exponential(scale, samples))


def gamma_sample(shape, scale, samples=1):
    """
    Sample a random number according to a gamma distribution.

    Parameters
    ----------
    shape : float
        The shape value of the exponential distribution.
    scale : float
        The scale value of the exponential distribution. Defaults to 1.
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().gamma(shape, scale, samples))


def uniform_sample(low, high, samples=1):
    """
    Sample a random number according to a uniform distribution.

    Parameters
    ----------
    low : float
        The smallest value the uniform distribution will return.
    high : float
        The largest value the uniform distribution will return.
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().uniform(low, high, samples))


def chi_square_sample(df, samples=1):
    """
    Sample a random number according to a chi-square distribution.

    Parameters
    ----------
    df : float
        The number of degrees of freedom
    samples : int
        The number of samples to return.

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
    return _simplify(_get_rng().chisquare(df, samples))


def discrete_sample(items, samples=1, verbose=False):
    """
    Sample a random value from a discrete distribution (aka categorical distribution).

    Parameters
    ----------
    items : list or dict
        The values that the discrete distribution will return and their associated
        weights (or likelihoods of being returned when sampled).
    samples : int
        The number of samples to return.
    verbose : bool
        If True, will print out statements on computational progress.

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
    return mixture_sample(values=values,
                          weights=weights,
                          samples=samples,
                          verbose=verbose)


def mixture_sample(values, weights=None, samples=1, verbose=False):
    """
    Sample a ranom number from a mixture distribution.

    Parameters
    ----------
    values : list or dict
        The distributions to mix. Can also be defined as a list of weights and distributions.
    weights : list or None
        The weights for each distribution.
    samples : int
        The number of samples to return.
    verbose : bool
        If True, will print out statements on computational progress.

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
        return sample(values[0], n=samples)

    def _run_mixture(values, weights):
        r_ = uniform_sample(0, 1)
        for i, dist in enumerate(values):
            weight = weights[i]
            if r_ <= weight:
                return sample(dist)
        return sample(dist)

    weights = np.cumsum(weights)
    r_ = tqdm(range(samples)) if verbose else range(samples)
    return _simplify(np.array([_run_mixture(values, weights) for _ in r_]))


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
    if n <= 0:
        raise ValueError('n must be >= 1')

    lclip_ = None
    rclip_ = None
    if _is_dist(dist):
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

    if lclip is not None or rclip is not None:
        dist.lclip = None
        dist.rclip = None
        if rclip:
            dist = fn_rclip(dist, rclip)
        if lclip:
            dist = fn_lclip(dist, lclip)

    if callable(dist):
        if n > 1:
            out = np.array([dist() for _ in (tqdm(range(n)) if verbose else range(n))])
        else:
            out = [dist()]

        r_ = tqdm(out) if verbose else out
        return _simplify(np.array([sample(o) if _is_dist(o) or callable(o) else o for o in r_]))

    elif (isinstance(dist, float) or
          isinstance(dist, int) or
          isinstance(dist, str) or
          dist is None):
        return _simplify(np.array([dist for _ in (tqdm(range(n)) if verbose else range(n))]))

    elif not _is_dist(dist):
        raise ValueError('input to sample is malformed - must ' +
                         'be a distribution but got {}'.format(type(dist)))

    elif dist.type == 'const':
        return _simplify(np.array([dist.x for _ in range(n)]))

    elif dist.type == 'uniform':
        return uniform_sample(dist.x, dist.y, samples=n)

    elif dist.type == 'discrete':
        return discrete_sample(dist.items, samples=n)

    elif dist.type == 'norm':
        return normal_sample(mean=dist.mean, sd=dist.sd, samples=n)

    elif dist.type == 'lognorm':
        return lognormal_sample(mean=dist.mean, sd=dist.sd, samples=n)

    elif dist.type == 'binomial':
        return binomial_sample(n=dist.n, p=dist.p, samples=n)

    elif dist.type == 'beta':
        return beta_sample(a=dist.a, b=dist.b, samples=n)

    elif dist.type == 'bernoulli':
        return bernoulli_sample(p=dist.p, samples=n)

    elif dist.type == 'poisson':
        return poisson_sample(lam=dist.lam, samples=n)

    elif dist.type == 'chisquare':
        return chi_square_sample(df=dist.df, samples=n)

    elif dist.type == 'exponential':
        return exponential_sample(scale=dist.scale, samples=n)

    elif dist.type == 'gamma':
        return gamma_sample(shape=dist.shape, scale=dist.scale, samples=n)

    elif dist.type == 'triangular':
        return triangular_sample(dist.left, dist.mode, dist.right, samples=n)

    elif dist.type == 'tdist':
        return t_sample(dist.x, dist.y, dist.t, credibility=dist.credibility, samples=n)

    elif dist.type == 'log_tdist':
        return log_t_sample(dist.x, dist.y, dist.t, credibility=dist.credibility, samples=n)

    elif dist.type == 'mixture':
        return mixture_sample(dist.dists, dist.weights, samples=n, verbose=verbose)

    elif dist.type == 'complex':
        if dist.right is None:
            out = dist.fn(sample(dist.left, n=n, verbose=verbose))
        else:
            out = dist.fn(sample(dist.left, n=n, verbose=verbose),
                          sample(dist.right, n=n, verbose=verbose))

        if _is_dist(out) or callable(out):
            return sample(out, n=n)
        else:
            return out

    else:
        raise ValueError('{} sampler not found'.format(dist.type))
