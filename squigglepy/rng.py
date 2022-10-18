import numpy as np

_squigglepy_internal_rng = np.random.default_rng()


def set_seed(seed):
    """
    Set the seed of the random number generator used by Squigglepy.

    The RNG is a ``np.random.default_rng`` under the hood.

    Parameters
    ----------
    seed : float
        The seed to use for the RNG.

    Returns
    -------
    np.random.default_rng
        The RNG used internally.

    Examples
    --------
    >>> set_seed(42)
    Generator(PCG64) at 0x127EDE9E0
    """
    global _squigglepy_internal_rng
    _squigglepy_internal_rng = np.random.default_rng(seed)
    return _squigglepy_internal_rng
