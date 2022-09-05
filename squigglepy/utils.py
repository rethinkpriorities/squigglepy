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

