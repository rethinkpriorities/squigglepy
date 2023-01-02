import math
import numpy as np

from tqdm import tqdm
from scipy import stats
from datetime import datetime


def _process_weights_values(weights=None, relative_weights=None, values=None):
    if weights is not None and relative_weights is not None:
        raise ValueError('can only pass either `weights` or `relative_weights`, not both.')
    if values is None:
        raise ValueError('must pass `values`')

    relative = False
    if relative_weights is not None:
        weights = relative_weights
        relative = True

    if isinstance(weights, float):
        weights = [weights]
    elif isinstance(weights, np.ndarray):
        weights = list(weights)
    elif not isinstance(weights, list) and weights is not None:
        raise ValueError('passed weights must be a list or array')

    if isinstance(values, np.ndarray):
        values = list(values)
    elif isinstance(values, dict):
        if weights is None:
            weights = list(values.values())
            values = list(values.keys())
        else:
            raise ValueError('cannot pass dict and weights separately')
    elif not isinstance(values, list) and not isinstance(values, dict):
        raise ValueError('passed values must be a list, dict, or array')

    if weights is None:
        if isinstance(values[0], list) and len(values[0]) == 2:
            weights = [v[0] for v in values]
            values = [v[1] for v in values]
        else:
            len_ = len(values)
            weights = [1 / len_ for _ in range(len_)]

    sum_weights = sum(weights)

    if relative:
        weights = [w / sum_weights for w in weights]
    else:
        if len(weights) == len(values) - 1 and sum_weights < 1:
            weights.append(1 - sum_weights)
        elif sum_weights <= 0.99 or sum_weights >= 1.01:
            raise ValueError('weights don\'t sum to 1 -' +
                             ' they sum to {}'.format(sum_weights))

    if len(weights) != len(values):
        raise ValueError('weights and values not same length')

    new_weights = []
    new_values = []
    for i, w in enumerate(weights):
        if w < 0:
            raise ValueError('weight cannot be negative')
        if w > 0:
            new_weights.append(w)
            new_values.append(values[i])

    return new_weights, new_values


def _is_numpy(a):
    return type(a).__module__ == np.__name__


def _is_dist(dist):
    from .distributions import BaseDistribution
    return isinstance(dist, BaseDistribution)


def _round(x, digits=0):
    if digits is None:
        return x

    x = np.round(x, digits)

    if _safe_len(x) > 1:
        return np.array([int(y) if digits == 0 else y for y in x])
    else:
        return int(x) if digits == 0 else x


def _simplify(a):
    if _is_numpy(a):
        a = a.tolist() if a.size == 1 else a
    if isinstance(a, list):
        a = a[0] if len(a) == 1 else a
    return a


def _safe_len(a):
    if _is_numpy(a):
        return a.size
    elif isinstance(a, list):
        return len(a)
    elif a is None:
        return 0
    else:
        return 1


def _core_cuts(n, cores):
    cuts = [math.floor(n / cores) for _ in range(cores)]
    delta = n - sum(cuts)
    cuts[-1] += delta
    return cuts


def _init_tqdm(verbose=True, total=None):
    if verbose:
        return tqdm(total=total)
    else:
        return None


def _tick_tqdm(pbar, tick_size=1):
    if pbar:
        pbar.update(tick_size)
    return pbar


def _flush_tqdm(pbar):
    if pbar is not None:
        pbar.close()
    return pbar


def event_occurs(p):
    """
    Return True with probability ``p`` and False with probability ``1 - p``.

    Parameters
    ----------
    p : float
        The probability of returning True. Must be between 0 and 1.

    Examples
    --------
    >>> set_seed(42)
    >>> event_occurs(p=0.5)
    False
    """
    if _is_dist(p) or callable(p):
        from .samplers import sample
        p = sample(p)
    from .rng import _squigglepy_internal_rng
    return _squigglepy_internal_rng.uniform(0, 1) < p


def event_happens(p):
    """
    Return True with probability ``p`` and False with probability ``1 - p``.

    Alias for ``event_occurs``.

    Parameters
    ----------
    p : float
        The probability of returning True. Must be between 0 and 1.

    Examples
    --------
    >>> set_seed(42)
    >>> event_happens(p=0.5)
    False
    """
    return event_occurs(p)


def event(p):
    """
    Return True with probability ``p`` and False with probability ``1 - p``.

    Alias for ``event_occurs``.

    Parameters
    ----------
    p : float
        The probability of returning True. Must be between 0 and 1.

    Returns
    -------
    bool

    Examples
    --------
    >>> set_seed(42)
    >>> event(p=0.5)
    False
    """
    return event_occurs(p)


def one_in(p, digits=0, verbose=True):
    """
    Convert a probability into "1 in X" notation.

    Parameters
    ----------
    p : float
        The probability to convert.
    digits : int
        The number of digits to round the result to. Defaults to 0. If ``digits``
        is 0, the result will be converted to int instead of float.
    verbose : logical
        If True, will return a string with "1 in X". If False, will just return X.

    Returns
    -------
    str if ``verbose`` is True. Otherwise, int if ``digits`` is 0 or float if ``digits`` > 0.

    Examples
    --------
    >>> one_in(0.1)
    "1 in 10"
    """
    p = _round(1 / p, digits)
    return '1 in {:,}'.format(p) if verbose else p


def get_percentiles(data,
                    percentiles=[1, 5, 10, 20, 30, 40, 50,
                                 60, 70, 80, 90, 95, 99],
                    reverse=False,
                    digits=None):
    """
    Print the percentiles of the data.

    Parameters
    ----------
    data : list or np.array
        The data to calculate percentiles for.
    percentiles : list
        A list of percentiles to calculate. Must be values between 0 and 100.
    reverse : bool
        If `True`, the percentile values are reversed (e.g., 95th and 5th percentile
        swap values.)
    digits : int or None
        The number of digits to display (using rounding).

    Returns
    -------
    dict
        A dictionary of the given percentiles.

    Examples
    --------
    >>> get_percentiles(range(100), percentiles=[25, 50, 75])
    {25: 24.75, 50: 49.5, 75: 74.25}
    """
    percentiles = percentiles if isinstance(percentiles, list) else [percentiles]
    percentile_labels = list(reversed(percentiles)) if reverse else percentiles
    percentiles = np.percentile(data, percentiles)
    percentiles = [_round(p, digits) for p in percentiles]
    if len(percentile_labels) == 1:
        return percentiles[0]
    else:
        return dict(list(zip(percentile_labels, percentiles)))


def get_log_percentiles(data,
                        percentiles=[1, 5, 10, 20, 30, 40, 50,
                                     60, 70, 80, 90, 95, 99],
                        reverse=False, display=True, digits=1):
    """
    Print the log (base 10) of the percentiles of the data.

    Parameters
    ----------
    data : list or np.array
        The data to calculate percentiles for.
    percentiles : list
        A list of percentiles to calculate. Must be values between 0 and 100.
    reverse : bool
        If True, the percentile values are reversed (e.g., 95th and 5th percentile
        swap values.)
    display : bool
        If True, the function returns an easy to read display.
    digits : int or None
        The number of digits to display (using rounding).

    Returns
    -------
    dict
        A dictionary of the given percentiles. If ``display`` is true, will be str values.
        Otherwise will be float values. 10 to the power of the value gives the true percentile.

    Examples
    --------
    >>> get_percentiles(range(100), percentiles=[25, 50, 75])
    {25: 24.75, 50: 49.5, 75: 74.25}
    """
    percentiles = get_percentiles(data,
                                  percentiles=percentiles,
                                  reverse=reverse,
                                  digits=digits)
    if isinstance(percentiles, dict):
        if display:
            return dict([(k, '10^{}'.format(_round(np.log10(v), digits))) for
                         k, v in percentiles.items()])
        else:
            return dict([(k, _round(np.log10(v), digits)) for
                        k, v in percentiles.items()])
    else:
        if display:
            return '10^{}'.format(_round(np.log10(percentiles), digits))
        else:
            return _round(np.log10(percentiles), digits)


def geomean(a, weights=None, relative_weights=None):
    """
    Calculate the geometric mean.

    Parameters
    ----------
    a : list or np.array
        The values to calculate the geometric mean of.
    weights : list or None
        The weights, if a weighted geometric mean is desired.
    relative_weights : list or None
        Relative weights, which if given will be weights that are normalized
        to sum to 1.

    Returns
    -------
    float

    Examples
    --------
    >>> geomean([1, 3, 10])
    3.1072325059538595
    """
    weights, a = _process_weights_values(weights, relative_weights, a)
    return stats.mstats.gmean(a, weights=weights)


def p_to_odds(p):
    """
    Calculate the decimal odds from a given probability.

    Parameters
    ----------
    p : float
        The probability to calculate decimal odds for. Must be between 0 and 1.

    Returns
    -------
    float
        Decimal odds

    Examples
    --------
    >>> p_to_odds(0.1)
    0.1111111111111111
    """
    return p / (1 - p)


def odds_to_p(odds):
    """
    Calculate the probability from given decimal odds.

    Parameters
    ----------
    odds : float
        The decimal odds to calculate the probability for.

    Returns
    -------
    float
        Probability

    Examples
    --------
    >>> odds_to_p(0.1)
    0.09090909090909091
    """
    return odds / (1 + odds)


def geomean_odds(a, weights=None, relative_weights=None):
    """
    Calculate the geometric mean of odds.

    Parameters
    ----------
    a : list or np.array
        The probabilities to calculate the geometric mean of. These are converted to odds
        before the geometric mean is taken..
    weights : list or None
        The weights, if a weighted geometric mean is desired.
    relative_weights : list or None
        Relative weights, which if given will be weights that are normalized
        to sum to 1.

    Returns
    -------
    float

    Examples
    --------
    >>> geomean_odds([0.1, 0.3, 0.9])
    0.42985748800076845
    """
    weights, a = _process_weights_values(weights, relative_weights, a)
    a = p_to_odds(np.array(a))
    return odds_to_p(geomean(a, weights=weights))


def laplace(s, n=None, time_passed=None,
            time_remaining=None, time_fixed=False):
    """
    Return probability of success on next trial given Laplace's law of succession.

    Also can be used to calculate a time-invariant version defined in
    https://www.lesswrong.com/posts/wE7SK8w8AixqknArs/a-time-invariant-version-of-laplace-s-rule

    Parameters
    ----------
    s : int
        The number of successes among ``n`` past trials or among ``time_passed`` amount of time.
    n : int or None
        The number of trials that contain the successes (and/or failures). Leave as None if
        time-invariant mode is desired.
    time_passed : float or None
        The amount of time that has passed when the successes (and/or failures) occured for
        calculating a time-invariant Laplace.
    time_remaining : float or None
        We are calculating the likelihood of observing at least one success over this time
        period.
    time_fixed : bool
        This should be False if the time period is variable - that is, if the time period
        was chosen specifically to include the most recent success. Otherwise the time period
        is fixed and this should be True. Defaults to False.

    Returns
    -------
    float
        The probability of at least one success in the next trial or ``time_remaining`` amount
        of time.

    Examples
    --------
    >>> # The sun has risen the past 100,000 days. What are the odds it rises again tomorrow?
    >>> laplace(s=100*K, n=100*K)
    0.999990000199996
    >>> # The last time a nuke was used in war was 77 years ago. What are the odds a nuke
    >>> # is used in the next year, not considering any information other than this naive prior?
    >>> laplace(s=1, time_passed=77, time_remaining=1, time_fixed=False)
    0.012820512820512664
    """
    if n is not None and s > n:
        raise ValueError('`s` cannot be greater than `n`')
    elif time_passed is None and time_remaining is None and n is not None:
        return (s + 1) / (n + 2)
    elif time_passed is not None and time_remaining is not None and s == 0:
        return 1 - ((1 + time_remaining/time_passed) ** -1)
    elif (time_passed is not None and time_remaining is not None
          and s > 0 and not time_fixed):
        return 1 - ((1 + time_remaining/time_passed) ** -s)
    elif (time_passed is not None and time_remaining is not None
          and s > 0 and time_fixed):
        return 1 - ((1 + time_remaining/time_passed) ** -(s + 1))
    elif time_passed is not None and time_remaining is None and s == 0:
        return 1 - ((1 + 1/time_passed) ** -1)
    elif (time_passed is not None and time_remaining is None
          and s > 0 and not time_fixed):
        return 1 - ((1 + 1/time_passed) ** -s)
    elif (time_passed is not None and time_remaining is None
          and s > 0 and time_fixed):
        return 1 - ((1 + 1/time_passed) ** -(s + 1))
    elif time_passed is None and n is None:
        raise ValueError('Must define `time_passed` or `n`')
    elif time_passed is None and time_remaining is not None:
        raise ValueError('Must define `time_passed`')
    else:
        raise ValueError('Fatal logic error - programmer made mistake!')


def roll_die(sides, n=1):
    """
    Roll a die.

    Parameters
    ----------
    sides : int
        The number of sides of the die that is rolled.
    n : int
        The number of dice to be rolled.

    Returns
    -------
    int or list
        Returns the value of each die roll.

    Examples
    --------
    >>> set_seed(42)
    >>> roll_die(6)
    5
    """
    if _is_dist(sides) or callable(sides):
        from .samplers import sample
        sides = sample(sides)
    if not isinstance(n, int):
        raise ValueError('can only roll an integer number of times')
    elif sides < 2:
        raise ValueError('cannot roll less than a 2-sided die.')
    elif not isinstance(sides, int):
        raise ValueError('can only roll an integer number of sides')
    else:
        from .samplers import sample
        from .distributions import discrete
        return sample(discrete(list(range(1, sides + 1))), n=n) if sides > 0 else None


def flip_coin(n=1):
    """
    Flip a coin.

    Parameters
    ----------
    n : int
        The number of coins to be flipped.

    Returns
    -------
    str or list
        Returns the value of each coin flip, as either "heads" or "tails"

    Examples
    --------
    >>> set_seed(42)
    >>> flip_coin()
    'heads'
    """
    rolls = roll_die(2, n=n)
    if isinstance(rolls, int):
        rolls = [rolls]
    flips = ['heads' if d == 2 else 'tails' for d in rolls]
    return flips[0] if len(flips) == 1 else flips


def kelly(my_price, market_price, deference=0, bankroll=1, resolve_date=None, current=0):
    """
    Calculate the Kelly criterion.

    Parameters
    ----------
    my_price : float
        The price (or probability) you give for the given event.
    market_price : float
        The price the market is giving for that event.
    deference : float
        How much deference (or weight) do you give the market price? Use 0.5 for half Kelly
        and 0.75 for quarter Kelly. Defaults to 0, which is full Kelly.
    bankroll : float
        How much money do you have to bet? Defaults to 1.
    resolve_date : str or None
        When will the event happen, the market resolve, and you get your money back? Used for
        calculating expected ARR. Give in YYYY-MM-DD format. Defaults to None, which means
        ARR is not calculated.
    current : float
        How much do you already have invested in this event? Used for calculating the
        additional amount you should invest. Defaults to 0.

    Returns
    -------
    dict
        A dict of values specifying:
        * ``my_price``
        * ``market_price``
        * ``deference``
        * ``adj_price`` : an adjustment to ``my_price`` once ``deference`` is taken
          into account.
        * ``delta_price`` : the absolute difference between ``my_price`` and ``market_price``.
        * ``adj_delta_price`` : the absolute difference between ``adj_price`` and
          ``market_price``.
        * ``kelly`` : the kelly criterion indicating the percentage of ``bankroll``
          you should bet.
        * ``target`` : the target amount of money you should have invested
        * ``current``
        * ``delta`` : the amount of money you should invest given what you already
          have invested
        * ``max_gain`` : the amount of money you would gain if you win
        * ``modeled_gain`` : the expected value you would win given ``adj_price``
        * ``expected_roi`` : the expected return on investment
        * ``expected_arr`` : the expected ARR given ``resolve_date``
        * ``resolve_date``

    Examples
    --------
    >>> kelly(my_price=0.7, market_price=0.4, deference=0.5, bankroll=100)
    {'my_price': 0.7, 'market_price': 0.4, 'deference': 0.5, 'adj_price': 0.55,
     'delta_price': 0.3, 'adj_delta_price': 0.15, 'kelly': 0.25, 'target': 25.0,
     'current': 0, 'delta': 25.0, 'max_gain': 62.5, 'modeled_gain': 23.13,
     'expected_roi': 0.375, 'expected_arr': None, 'resolve_date': None}
    """
    if market_price >= 1 or market_price <= 0:
        raise ValueError('market_price must be >0 and <1')
    if my_price >= 1 or my_price <= 0:
        raise ValueError('my_price must be >0 and <1')
    if deference > 1 or deference < 0:
        raise ValueError('deference must be >=0 and <=1')
    adj_price = my_price * (1 - deference) + market_price * deference
    kelly = np.abs(adj_price - ((1 - adj_price) * (market_price / (1 - market_price))))
    target = bankroll * kelly
    expected_roi = np.abs((adj_price / market_price) - 1)
    if resolve_date is None:
        expected_arr = None
    else:
        resolve_date = datetime.strptime(resolve_date, '%Y-%m-%d')
        expected_arr = ((expected_roi + 1) ** (365 / (resolve_date - datetime.now()).days)) - 1
    return {'my_price': round(my_price, 2),
            'market_price': round(market_price, 2),
            'deference': round(deference, 3),
            'adj_price': round(adj_price, 2),
            'delta_price': round(np.abs(market_price - my_price), 2),
            'adj_delta_price': round(np.abs(market_price - adj_price), 2),
            'kelly': round(kelly, 3),
            'target': round(target, 2),
            'current': round(current, 2),
            'delta': round(target - current, 2),
            'max_gain': round(target / market_price, 2),
            'modeled_gain': round((adj_price * (target / market_price) +
                                  (1 - adj_price) * -target), 2),
            'expected_roi': round(expected_roi, 3),
            'expected_arr': round(expected_arr, 3) if expected_arr is not None else None,
            'resolve_date': resolve_date}


def full_kelly(my_price, market_price, bankroll=1, resolve_date=None, current=0):
    """
    Alias for ``kelly`` where ``deference`` is 0.

    Parameters
    ----------
    my_price : float
        The price (or probability) you give for the given event.
    market_price : float
        The price the market is giving for that event.
    bankroll : float
        How much money do you have to bet? Defaults to 1.
    resolve_date : str or None
        When will the event happen, the market resolve, and you get your money back? Used for
        calculating expected ARR. Give in YYYY-MM-DD format. Defaults to None, which means
        ARR is not calculated.
    current : float
        How much do you already have invested in this event? Used for calculating the
        additional amount you should invest. Defaults to 0.

    Returns
    -------
    dict
        A dict of values specifying:
        * ``my_price``
        * ``market_price``
        * ``deference``
        * ``adj_price`` : an adjustment to ``my_price`` once ``deference`` is taken
          into account.
        * ``delta_price`` : the absolute difference between ``my_price`` and ``market_price``.
        * ``adj_delta_price`` : the absolute difference between ``adj_price`` and
          ``market_price``.
        * ``kelly`` : the kelly criterion indicating the percentage of ``bankroll``
          you should bet.
        * ``target`` : the target amount of money you should have invested
        * ``current``
        * ``delta`` : the amount of money you should invest given what you already
          have invested
        * ``max_gain`` : the amount of money you would gain if you win
        * ``modeled_gain`` : the expected value you would win given ``adj_price``
        * ``expected_roi`` : the expected return on investment
        * ``expected_arr`` : the expected ARR given ``resolve_date``
        * ``resolve_date``

    Examples
    --------
    >>> full_kelly(my_price=0.7, market_price=0.4, bankroll=100)
    {'my_price': 0.7, 'market_price': 0.4, 'deference': 0, 'adj_price': 0.7,
     'delta_price': 0.3, 'adj_delta_price': 0.3, 'kelly': 0.5, 'target': 50.0,
     'current': 0, 'delta': 50.0, 'max_gain': 125.0, 'modeled_gain': 72.5,
     'expected_roi': 0.75, 'expected_arr': None, 'resolve_date': None}
    """
    return kelly(my_price=my_price,
                 market_price=market_price,
                 bankroll=bankroll,
                 resolve_date=resolve_date,
                 current=current,
                 deference=0)


def half_kelly(my_price, market_price, bankroll=1, resolve_date=None, current=0):
    """
    Alias for ``kelly`` where ``deference`` is 0.5.

    Parameters
    ----------
    my_price : float
        The price (or probability) you give for the given event.
    market_price : float
        The price the market is giving for that event.
    bankroll : float
        How much money do you have to bet? Defaults to 1.
    resolve_date : str or None
        When will the event happen, the market resolve, and you get your money back? Used for
        calculating expected ARR. Give in YYYY-MM-DD format. Defaults to None, which means
        ARR is not calculated.
    current : float
        How much do you already have invested in this event? Used for calculating the
        additional amount you should invest. Defaults to 0.

    Returns
    -------
    dict
        A dict of values specifying:
        * ``my_price``
        * ``market_price``
        * ``deference``
        * ``adj_price`` : an adjustment to ``my_price`` once ``deference`` is taken
          into account.
        * ``delta_price`` : the absolute difference between ``my_price`` and ``market_price``.
        * ``adj_delta_price`` : the absolute difference between ``adj_price`` and
          ``market_price``.
        * ``kelly`` : the kelly criterion indicating the percentage of ``bankroll``
          you should bet.
        * ``target`` : the target amount of money you should have invested
        * ``current``
        * ``delta`` : the amount of money you should invest given what you already
          have invested
        * ``max_gain`` : the amount of money you would gain if you win
        * ``modeled_gain`` : the expected value you would win given ``adj_price``
        * ``expected_roi`` : the expected return on investment
        * ``expected_arr`` : the expected ARR given ``resolve_date``
        * ``resolve_date``

    Examples
    --------
    >>> half_kelly(my_price=0.7, market_price=0.4, bankroll=100)
    {'my_price': 0.7, 'market_price': 0.4, 'deference': 0.5, 'adj_price': 0.55,
     'delta_price': 0.3, 'adj_delta_price': 0.15, 'kelly': 0.25, 'target': 25.0,
     'current': 0, 'delta': 25.0, 'max_gain': 62.5, 'modeled_gain': 23.13,
     'expected_roi': 0.375, 'expected_arr': None, 'resolve_date': None}
    """
    return kelly(my_price=my_price,
                 market_price=market_price,
                 bankroll=bankroll,
                 resolve_date=resolve_date,
                 current=current,
                 deference=0.5)


def quarter_kelly(my_price, market_price, bankroll=1, resolve_date=None, current=0):
    """
    Alias for ``kelly`` where ``deference`` is 0.75.

    Parameters
    ----------
    my_price : float
        The price (or probability) you give for the given event.
    market_price : float
        The price the market is giving for that event.
    bankroll : float
        How much money do you have to bet? Defaults to 1.
    resolve_date : str or None
        When will the event happen, the market resolve, and you get your money back? Used for
        calculating expected ARR. Give in YYYY-MM-DD format. Defaults to None, which means
        ARR is not calculated.
    current : float
        How much do you already have invested in this event? Used for calculating the
        additional amount you should invest. Defaults to 0.

    Returns
    -------
    dict
        A dict of values specifying:
        * ``my_price``
        * ``market_price``
        * ``deference``
        * ``adj_price`` : an adjustment to ``my_price`` once ``deference`` is taken
          into account.
        * ``delta_price`` : the absolute difference between ``my_price`` and ``market_price``.
        * ``adj_delta_price`` : the absolute difference between ``adj_price`` and
          ``market_price``.
        * ``kelly`` : the kelly criterion indicating the percentage of ``bankroll``
          you should bet.
        * ``target`` : the target amount of money you should have invested
        * ``current``
        * ``delta`` : the amount of money you should invest given what you already
          have invested
        * ``max_gain`` : the amount of money you would gain if you win
        * ``modeled_gain`` : the expected value you would win given ``adj_price``
        * ``expected_roi`` : the expected return on investment
        * ``expected_arr`` : the expected ARR given ``resolve_date``
        * ``resolve_date``

    Examples
    --------
    >>> quarter_kelly(my_price=0.7, market_price=0.4, bankroll=100)
    {'my_price': 0.7, 'market_price': 0.4, 'deference': 0.75, 'adj_price': 0.48,
     'delta_price': 0.3, 'adj_delta_price': 0.08, 'kelly': 0.125, 'target': 12.5,
     'current': 0, 'delta': 12.5, 'max_gain': 31.25, 'modeled_gain': 8.28,
     'expected_roi': 0.188, 'expected_arr': None, 'resolve_date': None}
    """
    return kelly(my_price=my_price,
                 market_price=market_price,
                 bankroll=bankroll,
                 resolve_date=resolve_date,
                 current=current,
                 deference=0.75)
