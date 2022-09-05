import math
import random
import numpy as np
from scipy import stats


def normal_sample(low, high, interval):
    if low > high:
        raise ValueError('`high value` cannot be lower than `low value`')
    if low == high:
        return low
    else:
        mu = (high + low) / 2
        cdf_value = 0.5 + 0.5 * interval
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (high - mu) / normed_sigma
        return np.random.normal(mu, sigma)

    
def lognormal_sample(low, high, interval):
    if low > high:
        raise ValueError('`high value` cannot be lower than `low value`')
    if low < 0:
        raise ValueError('lognormal_sample cannot handle negative values')
    if low == high:
        return low
    else:
        log_low = np.log(low)
        log_high = np.log(high)
        mu = (log_high + log_low) / 2
        cdf_value = 0.5 + 0.5 * interval
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (log_high - mu) / normed_sigma
        return np.random.lognormal(mu, sigma)


def t_sample(low, high, t, interval):
    if low > high:
        raise ValueError('`high value` cannot be lower than `low value`')
    if low == high:
        return low
    else:
        mu = (high + low) / 2
        rangex = (high - low) / 2
        return np.random.standard_t(t) * rangex * 0.6/interval + mu


def log_t_sample(low, high, t, interval):
    if low > high:
        raise ValueError('`high value` cannot be lower than `low value`')
    if low < 0:
        raise ValueError('log_t_sample cannot handle negative values')
    if low == high:
        return low
    else:
        log_low = np.log(low)
        log_high = np.log(high)
        mu = (log_high + log_low) / 2
        rangex = (log_high - log_low) / 2
        return np.exp(np.random.standard_t(t) * rangex * 0.6/interval + mu)
    

def to(x, y, lclip=None, rclip=None):
    return [x, y, 'log' if x > 0 else 'norm', lclip, rclip]

def norm(x, y, lclip=None, rclip=None):
    return [x, y, 'norm', lclip, rclip]

def const(x):
    return [x, None, 'const', None, None]

def lognorm(x, y, lclip=None, rclip=None):
    return [x, y, 'log', lclip, rclip]

def weighted_lognorm(logs, weights, lclip=None, rclip=None):
    return [logs, weights, 'weighted_log', lclip, rclip]

def distributed_lognorm(logs, weights, lclip=None, rclip=None):
    return [logs, weights, 'distributed_log', lclip, rclip]

def tdist(x, y, t, lclip=None, rclip=None):
    return [x, y, 'tdist', t, lclip, rclip]

def log_tdist(x, y, t, lclip=None, rclip=None):
    return [x, y, 'log-tdist', t, lclip, rclip]


def sample(var, credibility=0.9, n=1):
    n = int(n)
    if n > 1:
        return np.array([sample(var) for _ in range(n)])
    elif n <= 0:
        return ValueError('n must be >= 1')

    if callable(var):
        return var()

    if not isinstance(var, list) and not (len(var) == 5 or len(var) == 6):
        raise ValueError('input to sample is malformed - must be sample data')

    if var[2] == 'const':
        out = var[0]

    elif var[2] == 'norm':
        out = normal_sample(var[0], var[1], credibility)

    elif var[2] == 'log':
        out = lognormal_sample(var[0], var[1], credibility)

    elif var[2] == 'weighted_log':
        weights = var[1]
        sum_weights = sum(weights)
        if sum_weights <= 0.99 or sum_weights >= 1.01:
            raise ValueError('weighted_log weights don\'t sum to 1 - they sum to {}'.format(sum_weights))
        if len(weights) != len(var[0]):
            raise ValueError('weighted_log weights and distributions not same length')
        log_result = 0
        for i, log_data in enumerate(var[0]):
            log_value = lognormal_sample(log_data[0], log_data[1], credibility)
            weight = weights[i]
            log_result += log_value * weight
        out = log_result

    elif var[2] == 'distributed_log':
        weights = var[1]
        sum_weights = sum(weights)
        if sum_weights <= 0.99 or sum_weights >= 1.01:
            raise ValueError('distributed_log weights don\'t sum to 1 - they sum to {}'.format(sum_weights))
        if len(weights) != len(var[0]):
            raise ValueError('distributed_log weights and distributions not same length')
        r_ = random.random()
        weights = np.cumsum(weights)
        done = False
        for i, log_data in enumerate(var[0]):
            if not done:
                weight = weights[i]
                if r_ <= weight:
                    out = lognormal_sample(log_data[0], log_data[1], credibility)
                    done = True

    elif var[2] == 'tdist':
        out = t_sample(var[0], var[1], var[3], credibility)

    elif var[2] == 'log-tdist':
        out = log_t_sample(var[0], var[1], var[3], credibility)

    else:
        raise ValueError('{} sampler not found'.format(var[2]))

    if var[2] == 'tdist' or var[2] == 'log-tdist':
        lclip = var[4]
        rclip = var[5]
    else:
        lclip = var[3]
        rclip = var[4]

    if lclip and out < lclip:
        out = lclip
    if rclip and out > rclip:
        out = rclip

    return out

