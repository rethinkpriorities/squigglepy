def to(x, y, lclip=None, rclip=None):
    return [x, y, 'log' if x > 0 else 'norm', lclip, rclip]

def const(x):
    return [x, None, 'const', None, None]

def uniform(x, y):
    return [x, y, 'uniform', None, None]

def norm(x=None, y=None, mean=None, sd=None, lclip=None, rclip=None):
    if mean is None and sd is None and x is not None and y is not None:
        return [x, y, 'norm', lclip, rclip]
    elif mean is None and sd is not None and x is None and y is None:
        return [0, sd, 'norm-mean', lclip, rclip]
    elif mean is not None and sd is not None and x is None and y is None:
        return [mean, sd, 'norm-mean', lclip, rclip]
    else:
        raise ValueError

def lognorm(x=None, y=None, mean=None, sd=None, lclip=None, rclip=None):
    if mean is None and sd is None and x is not None and y is not None:
        return [x, y, 'log', lclip, rclip]
    elif mean is None and sd is not None and x is None and y is None:
        return [0, sd, 'log-mean', lclip, rclip]
    elif mean is not None and sd is not None and x is None and y is None:
        return [mean, sd, 'log-mean', lclip, rclip]
    else:
        raise ValueError

def binomial(n, p):
    return [n, p, 'binomial', None, None]

def beta(a, b):
    return [a, b, 'beta', None, None]

def bernoulli(p):
    return [p, None, 'beta', None, None]

def tdist(x, y, t, lclip=None, rclip=None):
    return [x, y, 'tdist', t, lclip, rclip]

def log_tdist(x, y, t, lclip=None, rclip=None):
    return [x, y, 'log-tdist', t, lclip, rclip]

def triangular(left, mode, right, lclip=None, rclip=None):
    return [left, mode, 'triangular', right, lclip, rclip]

def exponential(scale):
    return [scale, None, 'exponential', None, None]

def mixture(dists, weights, lclip=None, rclip=None):
    return [dists, weights, 'mixture', lclip, rclip]

