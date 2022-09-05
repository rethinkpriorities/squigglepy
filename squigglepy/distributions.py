def to(x, y, lclip=None, rclip=None):
    return [x, y, 'log' if x > 0 else 'norm', lclip, rclip]

def const(x):
    return [x, None, 'const', None, None]

def norm(x=None, y=None, mean=None, sd=None, lclip=None, rclip=None):
    if mean is None:
        return [x, y, 'norm', lclip, rclip]
    else:
        return [mean, sd, 'norm-mean', lclip, rclip]

def lognorm(x=None, y=None, mean=None, sd=None, lclip=None, rclip=None):
    if mean is None:
        return [x, y, 'log', lclip, rclip]
    else:
        return [mean, sd, 'log-mean', lclip, rclip]

def tdist(x, y, t, lclip=None, rclip=None):
    if mean is None:
        return [x, y, 'tdist', t, lclip, rclip]
    else:
        return [mean, sd, 'tdist-mean', lclip, rclip]

def log_tdist(x, y, t, lclip=None, rclip=None):
    if mean is None:
        return [x, y, 'log-tdist', t, lclip, rclip]
    else:
        return [mean, sd, 'log-tdist-mean', lclip, rclip]

def mixture(dists, weights, lclip=None, rclip=None):
    return [dists, weights, 'mixture', lclip, rclip]

