import numpy as np
from scipy import stats


def _process_weights_values(weights, values):
    if isinstance(weights, float):
        weights = [weights]
    elif isinstance(weights, np.ndarray):
        weights = list(weights)
    elif not isinstance(weights, list) and weights is not None:
        raise ValueError('passed weights must be a list')

    if isinstance(values, np.ndarray):
        values = list(values)
    elif not isinstance(values, list):
        raise ValueError('passed values must be a list')

    if weights is None:
        if isinstance(values[0], list) and len(values[0]) == 2:
            weights = [v[0] for v in values]
            values = [v[1] for v in values]
        else:
            len_ = len(values)
            weights = [1 / len_ for _ in range(len_)]

    sum_weights = sum(weights)

    if len(weights) == len(values) - 1 and sum_weights < 1:
        weights.append(1 - sum_weights)
    elif sum_weights <= 0.99 or sum_weights >= 1.01:
        raise ValueError('weights don\'t sum to 1 -' +
                         ' they sum to {}'.format(sum_weights))

    if len(weights) != len(values):
        raise ValueError('weights and values not same length')

    return weights, values


def event_occurs(p):
    from .rng import _squigglepy_internal_rng
    return _squigglepy_internal_rng.uniform(0, 1) < p


def event_happens(p):
    return event_occurs(p)


def event(p):
    return event_occurs(p)


def get_percentiles(data,
                    percentiles=[1, 5, 10, 20, 30, 40, 50,
                                 60, 70, 80, 90, 95, 99],
                    reverse=False,
                    digits=None):
    percentile_labels = list(reversed(percentiles)) if reverse else percentiles
    percentiles = np.percentile(data, percentiles)
    if digits is not None:
        if digits == 0:
            percentiles = [int(p) for p in percentiles]
        else:
            percentiles = np.round(percentiles, digits)
    return dict(list(zip(percentile_labels, percentiles)))


def get_log_percentiles(data,
                        percentiles=[1, 5, 10, 20, 30, 40, 50,
                                     60, 70, 80, 90, 95, 99],
                        reverse=False, display=True, digits=1):
    percentiles = get_percentiles(data,
                                  percentiles=percentiles,
                                  reverse=reverse,
                                  digits=digits)
    if display:
        return dict([(k, '10^{}'.format(np.round(np.log10(v), digits))) for
                     k, v in percentiles.items()])
    else:
        return dict([(k, np.round(np.log10(v), digits)) for
                    k, v in percentiles.items()])


def geomean(a, weights=None):
    if weights is not None:
        weights, a = _process_weights_values(weights, a)
    return stats.mstats.gmean(a, weights=weights)


def p_to_odds(p):
    return p / (1 - p)


def odds_to_p(odds):
    return odds / (1 + odds)


def geomean_odds(a, weights=None):
    a = p_to_odds(np.array(a))
    return odds_to_p(geomean(a, weights=weights))


def laplace(s, n=None, time_passed=None,
            time_remaining=None, time_fixed=False):
    # Returns probability of success on next trial
    if n is not None and s > n:
        raise ValueError('`s` cannot be greater than `n`')
    elif time_passed is None and time_remaining is None and n is not None:
        return (s + 1) / (n + 2)
    elif time_passed is not None and time_remaining is not None and s == 0:
        # https://www.lesswrong.com/posts/wE7SK8w8AixqknArs/a-time-invariant-version-of-laplace-s-rule
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
    from .samplers import sample
    from .distributions import discrete
    if sides < 2:
        raise ValueError('cannot roll less than a 2-sided die.')
    elif not isinstance(sides, int):
        raise ValueError('can only roll an integer number of sides')
    else:
        return sample(discrete(list(range(1, sides + 1))), n=n) if sides > 0 else None


def flip_coin(n=1):
    rolls = roll_die(2, n=n)
    if isinstance(rolls, int):
        rolls = [rolls]
    flips = ['heads' if d == 2 else 'tails' for d in rolls]
    return flips[0] if len(flips) == 1 else flips
