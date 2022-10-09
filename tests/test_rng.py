from ..squigglepy.rng import set_seed
from ..squigglepy.samplers import sample
from ..squigglepy.distributions import norm


def test_seed():
    set_seed(42)
    test = sample(norm(1, 10000))
    set_seed(42)
    expected = sample(norm(1, 10000))
    assert test == expected
