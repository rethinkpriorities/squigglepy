from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures
import numpy as np


class MultithreadedRNG:
    def __init__(self, seed=None, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s) for s in seq.spawn(threads)]
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)

    def fill(self, n, method, var):
        self.step = np.ceil(n / self.threads).astype(np.int_)

        def _fill(random_state, var):
            fn = getattr(random_state, method)
            var['size'] = self.n
            return fn(**var)

        futures = {}
        for i in range(self.threads):
            args = (_fill,
                    self._random_generators[i],
                    self.step,
                    var)
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)
        return futures

    def __del__(self):
        self.executor.shutdown(False)


_squigglepy_internal_rng = MultithreadedRNG()


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
    _squigglepy_internal_rng = MultithreadedRNG(seed)
    return _squigglepy_internal_rng
