from .utils import is_dist
from .distributions import OperableDistribution


class Die(OperableDistribution):
    """
    A distribution representing a die roll.

    Supports exploding dice mechanics where additional dice are rolled
    when certain values are rolled.
    """

    def __init__(self, sides=None, explode_on=None):
        super().__init__()

        if is_dist(sides) or callable(sides):
            from .samplers import sample

            sides = sample(sides)
        if sides is None:
            raise ValueError("sides must be specified")
        if not isinstance(sides, int):
            raise ValueError("can only roll an integer number of sides")
        if sides < 2:
            raise ValueError("cannot roll less than a 2-sided die.")
        if explode_on is not None:
            if not isinstance(explode_on, list):
                explode_on = [explode_on]
            if len(explode_on) >= sides:
                raise ValueError("cannot explode on every value")
            for val in explode_on:
                if not isinstance(val, int) or val < 1 or val > sides:
                    raise ValueError(f"explode_on values must be integers between 1 and {sides}")
        self.sides = sides
        self.explode_on = explode_on

    def __str__(self):
        explode_out = (
            "" if self.explode_on is None else ", explodes on {}".format(str(self.explode_on))
        )
        out = "<Distribution> Die({}{})".format(self.sides, explode_out)
        return out


def die(sides, explode_on=None):
    """
    Create a distribution for a die.

    Parameters
    ----------
    sides : int
        The number of sides of the die that is rolled.
    explode_on : list or int or None
        An additional die will be rolled if the initial die rolls any of these values.
        Implements "exploding dice" mechanics. The exploding continues recursively
        until a non-exploding value is rolled.

    Returns
    -------
    Die
        A distribution that models a die roll, returning values from 1 to sides.

    Examples
    --------
    >>> die(6)
    <Distribution> Die(6)
    >>> die(6, explode_on=6)  # D6 that explodes on 6
    <Distribution> Die(6, explodes on [6])
    """
    return Die(sides=sides, explode_on=explode_on)


class Coin(OperableDistribution):
    """
    A distribution representing a coin flip.

    Returns either "heads" or "tails" with equal probability.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        out = "<Distribution> Coin"
        return out


def coin():
    """
    Create a distribution for a coin flip.

    Returns
    -------
    Coin
        A distribution that models a coin flip, returning either "heads" or "tails"
        with equal probability.

    Examples
    --------
    >>> coin()
    <Distribution> Coin
    >>> ~coin()  # Sample a coin flip
    'heads'  # or 'tails'
    """
    return Coin()
