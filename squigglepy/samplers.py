import bisect
import os
import time

import numpy as np
import pathos.multiprocessing as mp

from numpy.typing import NDArray

from scipy import stats

from .utils import (
    _process_weights_values,
    _process_discrete_weights_values,
    is_dist,
    is_sampleable,
    _simplify,
    _enlist,
    _safe_len,
    _core_cuts,
    _init_tqdm,
    _tick_tqdm,
    _flush_tqdm,
)

from .distributions import (
    BaseDistribution,
    BernoulliDistribution,
    BetaDistribution,
    BinomialDistribution,
    ChiSquareDistribution,
    ComplexDistribution,
    ConstantDistribution,
    CategoricalDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GeometricDistribution,
    LogTDistribution,
    LognormalDistribution,
    InverseLognormalDistribution,
    MixtureDistribution,
    NormalDistribution,
    OperableDistribution,
    ParetoDistribution,
    PoissonDistribution,
    TDistribution,
    TriangularDistribution,
    PERTDistribution,
    UniformDistribution,
    const,
)

_squigglepy_internal_sample_caches = {}


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


def inverse_lognormal_sample(mean, sd, samples=1):
    """
    Sample a random number according to an inverse lognormal distribution.

    This is equivalent to taking the reciprocal of samples from a lognormal distribution,
    producing a left-skewed distribution (rather than the right-skewed lognormal).

    Parameters
    ----------
    mean : float
        The mean of the underlying normal distribution in log space.
    sd : float
        The standard deviation of the underlying normal distribution in log space.
    samples : int
        The number of samples to return.

    Returns
    -------
    float
        A random number sampled from an inverse lognormal distribution.

    Examples
    --------
    >>> set_seed(42)
    >>> inverse_lognormal_sample(0, 1)
    0.7373871753169729
    """
    # Generate lognormal samples and take their reciprocals
    lognorm_samples = _get_rng().lognormal(mean, sd, samples)
    return _simplify(1.0 / lognorm_samples)


def t_sample(low=None, high=None, t=20, samples=1, credibility=90):
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
        The number of degrees of freedom of the t-distribution. Defaults to 20.
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
        return _get_rng().standard_t(t, samples)
    elif low is None or high is None:
        raise ValueError("must define either both `x` and `y` or neither.")
    elif low > high:
        raise ValueError("`high value` cannot be lower than `low value`")
    elif low == high:
        return low
    else:
        mu = (high + low) / 2
        cdf_value = 0.5 + 0.5 * (credibility / 100)
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (high - mu) / normed_sigma
        return _simplify(
            normal_sample(mu, sigma, samples) / ((chi_square_sample(t, samples) / t) ** 0.5)
        )


def log_t_sample(low=None, high=None, t=20, samples=1, credibility=90):
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
        The low value of a credible interval defined by ``credibility``.
        Must be greater than 0. Defaults to a 90% CI.
    high : float or None
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    t : float
        The number of degrees of freedom of the t-distribution. Defaults to 20.
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
        raise ValueError("`high value` cannot be lower than `low value`")
    elif low < 0:
        raise ValueError("log_t_sample cannot handle negative values")
    elif low == high:
        return low
    else:
        log_low = np.log(low)
        log_high = np.log(high)
        mu = (log_high + log_low) / 2
        cdf_value = 0.5 + 0.5 * (credibility / 100)
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (log_high - mu) / normed_sigma
        return _simplify(
            np.exp(
                normal_sample(mu, sigma, samples) / ((chi_square_sample(t, samples) / t) ** 0.5)
            )
        )


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


def pert_sample(left, mode, right, lam, samples=1):
    """
    Sample a random number according to a PERT distribution.

    Parameters
    ----------
    left : float
        The smallest value of the PERT distribution.
    mode : float
        The most common value of the PERT distribution.
    right : float
        The largest value of the PERT distribution.
    lam : float
        The lambda of the PERT distribution.
    samples : int
        The number of samples to return.

    Returns
    -------
    float
        A random number sampled from a PERT distribution.

    Examples
    --------
    >>> set_seed(42)
    >>> pert_sample(1, 2, 3, 4)
    2.327625176788963
    """
    r = right - left
    alpha = 1 + lam * (mode - left) / r
    beta = 1 + lam * (right - mode) / r
    return left + beta_sample(a=alpha, b=beta, samples=samples) * r


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
        The shape value of the gamma distribution.
    scale : float
        The scale value of the gamma distribution. Defaults to 1.
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


def pareto_sample(shape, samples=1):
    """
    Sample a random number according to a pareto distribution.

    Parameters
    ----------
    shape : float
        The shape value of the pareto distribution.

    Returns
    -------
    int
        A random number sampled from an pareto distribution.

    Examples
    --------
    >>> set_seed(42)
    >>> pareto_sample(1)
    10.069666324736094
    """
    # Add 1 because Numpy's "pareto" sampler actually samples from a Lomax
    # distribution
    return _simplify(_get_rng().pareto(shape, samples) + 1)


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


def discrete_sample(items, samples=1, verbose=False, _multicore_tqdm_n=1, _multicore_tqdm_cores=1):
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
    _multicore_tqdm_n : int
        The total number of samples to use for printing tqdm's interface. This is meant to only
        be used internally by squigglepy to make the progress bar printing work well for
        multicore. This parameter can be safely ignored by the user.
    _multicore_tqdm_cores : int
        The total number of cores to use for printing tqdm's interface. This is meant to only
        be used internally by squigglepy to make the progress bar printing work well for
        multicore. This parameter can be safely ignored by the user.

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
    weights, values = _process_discrete_weights_values(items)

    values = [const(v) for v in values]

    return mixture_sample(
        values=values,
        weights=weights,
        samples=samples,
        verbose=verbose,
        _multicore_tqdm_n=_multicore_tqdm_n,
        _multicore_tqdm_cores=_multicore_tqdm_cores,
    )


def geometric_sample(p, samples=1):
    """
    Sample a random number according to a geometric distribution.

    Parameters
    ----------
    p : float
        The probability of success of an individual trial. Must be between 0 and 1.
    samples : int
        The number of samples to return.

    Returns
    -------
    int
        A random number sampled from a geometric distribution.

    Examples
    --------
    >>> set_seed(42)
    >>> geometric_sample(0.1)
    2
    """
    return _simplify(_get_rng().geometric(p, samples))


def _mixture_sample_for_large_n(
    values,
    weights=None,
    relative_weights=None,
    samples=1,
    verbose=False,
    _multicore_tqdm_n=1,
    _multicore_tqdm_cores=1,
):
    def _run_presample(dist, pbar):
        _tick_tqdm(pbar)
        return _enlist(sample(dist, n=samples))

    pbar = _init_tqdm(verbose=verbose, total=len(values))
    values = [_run_presample(v, pbar) for v in values]
    _flush_tqdm(pbar)

    def _run_mixture(picker, i, pbar):
        _tick_tqdm(pbar, _multicore_tqdm_cores)
        index = bisect.bisect_left(weights, picker)
        return values[index][i]

    weights = np.cumsum(weights)
    picker = uniform_sample(0, 1, samples=samples)

    tqdm_samples = samples if _multicore_tqdm_cores == 1 else _multicore_tqdm_n
    pbar = _init_tqdm(verbose=verbose, total=tqdm_samples)
    out = _simplify([_run_mixture(p, i, pbar) for i, p in enumerate(_enlist(picker))])
    _flush_tqdm(pbar)

    return out


def _mixture_sample_for_small_n(
    values,
    weights=None,
    relative_weights=None,
    samples=1,
    verbose=False,
    _multicore_tqdm_n=1,
    _multicore_tqdm_cores=1,
):
    def _run_mixture(values, weights, pbar=None, tick=1):
        r_ = uniform_sample(0, 1)
        _tick_tqdm(pbar, tick)
        for i, dist in enumerate(values):
            weight = weights[i]
            if r_ <= weight:
                return sample(dist)
        return sample(dist)

    weights = np.cumsum(weights)
    tqdm_samples = samples if _multicore_tqdm_cores == 1 else _multicore_tqdm_n
    pbar = _init_tqdm(verbose=verbose, total=tqdm_samples)
    out = _simplify(
        [
            _run_mixture(values=values, weights=weights, pbar=pbar, tick=_multicore_tqdm_cores)
            for _ in range(samples)
        ]
    )
    _flush_tqdm(pbar)
    return out


def mixture_sample(
    values,
    weights=None,
    relative_weights=None,
    samples=1,
    verbose=False,
    _multicore_tqdm_n=1,
    _multicore_tqdm_cores=1,
):
    """
    Sample a ranom number from a mixture distribution.

    Parameters
    ----------
    values : list or dict
        The distributions to mix. Can also be defined as a list of weights and distributions.
    weights : list or None
        The weights for each distribution.
    relative_weights : list or None
        Relative weights, which if given will be weights that are normalized
        to sum to 1.
    samples : int
        The number of samples to return.
    verbose : bool
        If True, will print out statements on computational progress.
    _multicore_tqdm_n : int
        The total number of samples to use for printing tqdm's interface. This is meant to only
        be used internally by squigglepy to make the progress bar printing work well for
        multicore. This parameter can be safely ignored by the user.
    _multicore_tqdm_cores : int
        The total number of cores to use for printing tqdm's interface. This is meant to only
        be used internally by squigglepy to make the progress bar printing work well for
        multicore. This parameter can be safely ignored by the user.

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
    weights, values = _process_weights_values(weights, relative_weights, values)

    if len(values) == 1:
        return sample(values[0], n=samples)

    if samples > 100:
        return _mixture_sample_for_large_n(
            values=values,
            weights=weights,
            samples=samples,
            verbose=verbose,
            _multicore_tqdm_n=_multicore_tqdm_n,
            _multicore_tqdm_cores=_multicore_tqdm_cores,
        )
    else:
        return _mixture_sample_for_small_n(
            values=values,
            weights=weights,
            samples=samples,
            verbose=verbose,
            _multicore_tqdm_n=_multicore_tqdm_n,
            _multicore_tqdm_cores=_multicore_tqdm_cores,
        )


def sample_correlated_group(
    requested_dist: BaseDistribution, n: int, verbose=False
) -> NDArray[np.float64]:
    """
    Samples a correlated distribution, alongside
    all other correlated distributions in the same group.

    The samples for other variables are stored in the distributions themselves
    (in `_correlated_samples`).

    This is necessary, because the sampling needs to happen all at once, regardless
    of where the distributions are used in the binary tree of operations.
    """
    group = requested_dist.correlation_group
    assert group is not None

    samples = np.column_stack(
        [
            # Skip correlation to prevent infinite recursion
            # TODO: Check that this does not interfere
            # with other correlated distributions downstream
            sample(dist, n, verbose=verbose, _correlate_if_needed=False)
            for dist in group.correlated_dists
        ]
    )
    # Induce correlation
    samples = group.induce_correlation(samples)

    # Store the samples in each distribution
    # except the one we are sampling from
    # so it requires resampling next time
    requested_samples = None
    for i, target_distribution in enumerate(group.correlated_dists):
        if requested_dist is not target_distribution:
            # Store the samples in the distribution
            target_distribution._correlated_samples = samples[:, i]
        else:
            # Store the samples we requested
            requested_samples = samples[:, i]

    assert requested_samples is not None

    return requested_samples


def sample(
    dist=None,
    n=1,
    lclip=None,
    rclip=None,
    memcache=False,
    reload_cache=False,
    dump_cache_file=None,
    load_cache_file=None,
    cache_file_primary=False,
    verbose=None,
    cores=1,
    _multicore_tqdm_n=1,
    _multicore_tqdm_cores=1,
    _correlate_if_needed=True,
):
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
    memcache : bool
        If True, will attempt to load the results in-memory for future calculations if
        a cache is present. Otherwise will save the results to an in-memory cache. Each cache
        will be matched based on ``dist``. Default ``False``.
    reload_cache : bool
        If True, any existing cache will be ignored and recalculated. Default ``False``.
    dump_cache_file : str or None
        If present, will write out the cache to a numpy file with this path with
        ``.sqlcache.npy`` appended to the file name.
    load_cache_file : str or None
        If present, will first attempt to load and use a cache from a file with this
        path with ``.sqlcache.npy`` appended to the file name.
    cache_file_primary : bool
        If both an in-memory cache and file cache are present, the file
        cache will be used for the cache if this is True, and the in-memory cache
        will be used otherwise. Defaults to False.
    verbose : bool
        If True, will print out statements on computational progress. If False, will not.
        If None (default), will be True when ``n`` is greater than or equal to 1M.
    cores : int
        If 1, runs on a single core / process. If greater than 1, will run on a multiprocessing
        pool with that many cores / processes.
    _multicore_tqdm_n : int
        The total number of samples to use for printing tqdm's interface. This is meant to only
        be used internally by squigglepy to make the progress bar printing work well for
        multicore. This parameter can be safely ignored by the user.
    _multicore_tqdm_cores : int
        The total number of cores to use for printing tqdm's interface. This is meant to only
        be used internally by squigglepy to make the progress bar printing work well for
        multicore. This parameter can be safely ignored by the user.

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
        raise ValueError("n must be >= 1")

    if not is_sampleable(dist):
        error = "input to sample is malformed - {} is not a sampleable type.".format(type(dist))
        raise ValueError(error)

    if verbose is None:
        verbose = n >= 1000000

    # Simplify distribution analytically before sampling
    if isinstance(dist, OperableDistribution):
        dist = dist.simplify()

    # Handle loading from cache
    samples = None
    has_in_mem_cache = str(dist) in _squigglepy_internal_sample_caches
    if load_cache_file:
        cache_path = load_cache_file + ".sqcache.npy"
        has_file_cache = os.path.exists(cache_path) if load_cache_file else False

    if load_cache_file and not has_file_cache and verbose:
        print("Warning: cache file `{}.sqcache.npy` not found.".format(load_cache_file))

    if (load_cache_file or memcache) and not reload_cache:
        if load_cache_file and has_file_cache and (not has_in_mem_cache or cache_file_primary):
            if verbose:
                print("Loading from cache file (`{}`)...".format(cache_path))
            with open(cache_path, "rb") as f:
                samples = np.load(f)

        elif memcache and has_in_mem_cache:
            if verbose:
                print("Loading from in-memory cache...")
            samples = _squigglepy_internal_sample_caches.get(str(dist))

    # Handle multicore
    if samples is None and cores > 1:
        if verbose:
            print("Generating samples with {} cores...".format(cores))
        with mp.ProcessingPool(cores) as pool:
            cuts = _core_cuts(n, cores)

            def multicore_sample(core, total_n=n, total_cores=cores, verbose=False):
                batch = sample(
                    dist=dist,
                    n=cuts[core],
                    _multicore_tqdm_n=total_n,
                    _multicore_tqdm_cores=total_cores,
                    lclip=lclip,
                    rclip=rclip,
                    memcache=False,
                    verbose=verbose,
                    cores=1,
                )
                if verbose:
                    print("Shuffling data...")
                with open("test-core-{}.npy".format(core), "wb") as f:
                    np.save(f, batch)
                return None

            pool_results = pool.amap(multicore_sample, range(cores - 1))
            multicore_sample(cores - 1, verbose=verbose)
            if verbose:
                print("Waiting for other cores...")
            while not pool_results.ready():
                if verbose:
                    print(".", end="", flush=True)
                time.sleep(1)

        if verbose:
            print("Collecting data...")
        samples = np.array([])
        pbar = _init_tqdm(verbose=verbose, total=cores)
        for core in range(cores):
            with open("test-core-{}.npy".format(core), "rb") as f:
                samples = np.concatenate((samples, np.load(f, allow_pickle=True)), axis=None)
            os.remove("test-core-{}.npy".format(core))
            _tick_tqdm(pbar, 1)
        _flush_tqdm(pbar)
        if verbose:
            print("...Collected!")

    # Handle lclip/rclip
    if samples is None:
        lclip_ = None
        rclip_ = None
        if is_dist(dist):
            lclip_ = dist.lclip
            rclip_ = dist.rclip

        if lclip is None and lclip_ is not None:
            lclip = lclip_
        elif lclip is not None and lclip_ is not None:
            lclip = max(lclip, lclip_)

        if rclip is None and rclip_ is not None:
            rclip = rclip_
        elif rclip is not None and rclip_ is not None:
            rclip = min(rclip, rclip_)

    # Start sampling
    if samples is None:
        if callable(dist):
            if n > 1:

                def run_dist(dist, pbar=None, tick=1):
                    dist = dist()
                    _tick_tqdm(pbar, tick)
                    return dist

                tqdm_samples = n if _multicore_tqdm_cores == 1 else _multicore_tqdm_n
                pbar = _init_tqdm(verbose=verbose, total=tqdm_samples)
                out = np.array(
                    [run_dist(dist=dist, pbar=pbar, tick=_multicore_tqdm_cores) for _ in range(n)]
                )
                _flush_tqdm(pbar)
            else:
                out = [dist()]

            def run_dist(dist, pbar=None, tick=1):
                samp = sample(dist) if is_dist(dist) or callable(dist) else dist
                _tick_tqdm(pbar, tick)
                return samp

            pbar = _init_tqdm(verbose=verbose, total=len(out) * _multicore_tqdm_cores)
            samples = _simplify(
                np.array([run_dist(dist=o, pbar=pbar, tick=_multicore_tqdm_cores) for o in out])
            )
            _flush_tqdm(pbar)

        elif (
            isinstance(dist, float)
            or isinstance(dist, int)
            or isinstance(dist, str)
            or dist is None
        ):
            samples = _simplify(np.array([dist for _ in range(n)]))

        # Distribution is part of a correlation group
        # and has not been sampled yet
        elif (
            isinstance(dist, BaseDistribution)
            and _correlate_if_needed
            and dist.correlation_group is not None
            and dist._correlated_samples is None
        ):
            # Samples the entire correlated group at once
            samples = sample_correlated_group(dist, n=n, verbose=verbose)

        # Distribution has already been sampled
        # as part of a correlation group
        elif (
            isinstance(dist, BaseDistribution)
            and _correlate_if_needed
            and dist.correlation_group is not None
            and dist._correlated_samples is not None
        ):
            samples = dist._correlated_samples
            # This forces the distribution to be resampled
            # if the user attempts to sample from it again
            dist._correlated_samples = None

        elif isinstance(dist, ConstantDistribution):
            samples = _simplify(np.array([dist.x for _ in range(n)]))

        elif isinstance(dist, UniformDistribution):
            samples = uniform_sample(dist.x, dist.y, samples=n)

        elif isinstance(dist, CategoricalDistribution):
            samples = discrete_sample(
                dist.items,
                samples=n,
                _multicore_tqdm_n=_multicore_tqdm_n,
                _multicore_tqdm_cores=_multicore_tqdm_cores,
            )

        elif isinstance(dist, NormalDistribution):
            samples = normal_sample(mean=dist.mean, sd=dist.sd, samples=n)

        elif isinstance(dist, LognormalDistribution):
            samples = lognormal_sample(mean=dist.norm_mean, sd=dist.norm_sd, samples=n)

        elif isinstance(dist, InverseLognormalDistribution):
            samples = inverse_lognormal_sample(mean=dist.norm_mean, sd=dist.norm_sd, samples=n)

        elif isinstance(dist, BinomialDistribution):
            samples = binomial_sample(n=dist.n, p=dist.p, samples=n)

        elif isinstance(dist, BetaDistribution):
            samples = beta_sample(a=dist.a, b=dist.b, samples=n)

        elif isinstance(dist, BernoulliDistribution):
            samples = bernoulli_sample(p=dist.p, samples=n)

        elif isinstance(dist, PoissonDistribution):
            samples = poisson_sample(lam=dist.lam, samples=n)

        elif isinstance(dist, ChiSquareDistribution):
            samples = chi_square_sample(df=dist.df, samples=n)

        elif isinstance(dist, ExponentialDistribution):
            samples = exponential_sample(scale=dist.scale, samples=n)

        elif isinstance(dist, GammaDistribution):
            samples = gamma_sample(shape=dist.shape, scale=dist.scale, samples=n)

        elif isinstance(dist, ParetoDistribution):
            samples = pareto_sample(shape=dist.shape, samples=n)

        elif isinstance(dist, TriangularDistribution):
            samples = triangular_sample(dist.left, dist.mode, dist.right, samples=n)

        elif isinstance(dist, PERTDistribution):
            samples = pert_sample(dist.left, dist.mode, dist.right, dist.lam, samples=n)

        elif isinstance(dist, TDistribution):
            samples = t_sample(dist.x, dist.y, dist.t, credibility=dist.credibility, samples=n)

        elif isinstance(dist, LogTDistribution):
            samples = log_t_sample(dist.x, dist.y, dist.t, credibility=dist.credibility, samples=n)

        elif isinstance(dist, MixtureDistribution):
            samples = mixture_sample(
                dist.dists,
                dist.weights,
                samples=n,
                verbose=verbose,
                _multicore_tqdm_n=_multicore_tqdm_n,
                _multicore_tqdm_cores=_multicore_tqdm_cores,
            )

        elif isinstance(dist, GeometricDistribution):
            samples = geometric_sample(p=dist.p, samples=n)

        elif isinstance(dist, ComplexDistribution):
            if dist.right is None:
                samples = dist.fn(sample(dist.left, n=n, verbose=verbose))
            else:
                samples = dist.fn(
                    sample(dist.left, n=n, verbose=verbose),
                    sample(dist.right, n=n, verbose=verbose),
                )

            if is_dist(samples) or callable(samples):
                samples = sample(samples, n=n)

        else:
            raise ValueError("{} sampler not found".format(type(dist)))

    # Use lclip / rclip
    if _safe_len(samples) > 1:
        if lclip is not None:
            samples = np.maximum(samples, lclip)
        if rclip is not None:
            samples = np.minimum(samples, rclip)
    else:
        if lclip is not None:
            samples = lclip if samples < lclip else samples
        if rclip is not None:
            samples = rclip if samples > rclip else samples

    # Save to cache
    if memcache and (not has_in_mem_cache or reload_cache):
        if verbose:
            print("Caching in-memory...")
        _squigglepy_internal_sample_caches[str(dist)] = samples
        if verbose:
            print("...Cached")

    if dump_cache_file:
        cache_path = dump_cache_file + ".sqcache.npy"
        if verbose:
            print("Writing cache to file `{}`...".format(cache_path))
        with open(cache_path, "wb") as f:
            np.save(f, samples)
        if verbose:
            print("...Cached")

    # Return
    return np.array(samples) if isinstance(samples, list) else samples
