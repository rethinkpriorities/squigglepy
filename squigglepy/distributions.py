def to(x, y, lclip=None, rclip=None):
    return [x, y, 'log' if x > 0 else 'norm', lclip, rclip]

def norm(x, y, lclip=None, rclip=None):
    return [x, y, 'norm', lclip, rclip]

def const(x):
    return [x, None, 'const', None, None]

def lognorm(x, y, lclip=None, rclip=None):
    return [x, y, 'log', lclip, rclip]

def tdist(x, y, t, lclip=None, rclip=None):
    return [x, y, 'tdist', t, lclip, rclip]

def log_tdist(x, y, t, lclip=None, rclip=None):
    return [x, y, 'log-tdist', t, lclip, rclip]

def mixture(dists, weights, lclip=None, rclip=None):
    return [dists, weights, 'mixture', lclip, rclip]

