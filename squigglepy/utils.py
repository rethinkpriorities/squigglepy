import math
import random
import numpy as np
from scipy import stats


def event_occurs(p):
    return random.random() < p


def numerize(oom_num):
    oom_num = int(oom_num)
    ooms = ['thousand', 'million', 'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion', 'septillion', 'octillion', 'nonillion', 'decillion']

    if oom_num == 0:
        return 'one'
    elif oom_num == 1:
        return 'ten'
    elif oom_num == 2:
        return 'hundred'
    elif oom_num > 35:
        return numerize(oom_num - 33) + ' decillion'
    elif oom_num < 0:
        return numerize(-oom_num) + 'th'
    elif oom_num % 3 == 0:
        return 'one ' + ooms[(oom_num // 3) - 1]
    else:
        return str(10 ** (oom_num % 3)) + ' ' + ooms[(oom_num // 3) - 1]


def format_gb(gb):
    if gb >= 1000:
        tb = np.round(gb / 1000)
    else:
        return str(gb) + ' GB'
    
    if tb >= 1000:
        pb = np.round(tb / 1000)
    else:
        return str(tb) + ' TB'

    if pb >= 10000:
        return numerize(math.log10(pb)) + ' PB'
    else:
        return str(pb) + ' PB'
    

def get_percentiles(data, percentiles=[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99], reverse=False, digits=None):
    percentile_labels = list(reversed(percentiles)) if reverse else percentiles
    percentiles = np.percentile(data, percentiles)
    if digits is not None:
        percentiles = np.round(percentiles, digits)
    return dict(list(zip(percentile_labels, percentiles)))


def get_log_percentiles(data, percentiles, reverse=False, display=True, digits=1):
    percentiles = get_percentiles(data, percentiles=percentiles, reverse=reverse, digits=digits)
    if display:
        return dict([(k, '10^{} (~{})'.format(np.round(np.log10(v), digits), numerize(np.log10(v)))) for k, v in percentiles.items()])
    else:
        return dict([(k, np.round(np.log10(v), digits)) for k, v in percentiles.items()])


def geomean(a, weights=None):
    return stats.mstats.gmean(a, weights=weights)

def p_to_odds(p):
    return p / (1 - p)

def odds_to_p(odds):
    return odds / (1 + odds)

def geomean_odds(a, weights=None):
    a = p_to_odds(np.array(a))
    return odds_to_p(stats.mstats.gmean(a, weights=weights))


def laplace(s, n=None, time_passed=None, time_remaining=None, time_fixed=False):
	# Returns probability of success on next trial
	if time_passed is None and time_remaining is None and n is not None:
		return (s + 1) / (n + 2)
	elif time_passed is not None and time_remaining is not None and s == 0:
		# https://www.lesswrong.com/posts/wE7SK8w8AixqknArs/a-time-invariant-version-of-laplace-s-rule
		return 1 - ((1 + time_remaining/time_passed) ** -1)
	elif time_passed is not None and time_remaining is not None and s > 0 and not time_fixed:
		return 1 - ((1 + time_remaining/time_passed) ** -s)
	elif time_passed is not None and time_remaining is not None and s > 0 and time_fixed:
		return 1 - ((1 + time_remaining/time_passed) ** -(s + 1))
	else:
		raise ValueError

