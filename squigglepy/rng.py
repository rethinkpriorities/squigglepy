from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures
import numpy as np

from .utils import flatten


class MultithreadedRNG:
    def __init__(self, seed=None, max_threads=None):
        self.seed = seed
        self.max_threads = max_threads
        self.calls = 0

    def fill(self, n, method, var):
        self.calls += 1
        if n < 1000:
            random_state = default_rng(self.seed + self.calls)
            fn = getattr(random_state, method)
            var['size'] = n
            return fn(**var)
        else:
            threads = multiprocessing.cpu_count()
            if self.max_threads is not None and threads > self.max_threads:
                threads = self.max_threads

            seq = SeedSequence(self.seed + self.calls)
            self._random_generators = [default_rng(s) for s in seq.spawn(threads)]
            self.executor = concurrent.futures.ThreadPoolExecutor(threads)

            if n < threads:
                threads = n

            self.step = np.ceil(n / threads).astype(np.int_)

            def _fill(random_state, step, var, i):
                fn = getattr(random_state, method)
                var['size'] = step
                return (i, fn(**var))

            futures = []
            for i in range(threads):
                args = (_fill,
                        self._random_generators[i],
                        self.step,
                        var,
                        i)
                futures.append(self.executor.submit(*args))
            out = [f.result() for f in concurrent.futures.as_completed(futures)]
            return flatten([o[1] for o in sorted(out)])


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
