import numpy as np

_squigglepy_internal_rng = np.random.default_rng()


def set_seed(seed):
    global _squigglepy_internal_rng
    _squigglepy_internal_rng = np.random.default_rng(seed)
    return _squigglepy_internal_rng
