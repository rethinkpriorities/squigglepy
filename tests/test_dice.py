import pytest

from ..squigglepy.rng import set_seed
from ..squigglepy.dice import die, coin, Die, Coin
from ..squigglepy.distributions import OperableDistribution
from ..squigglepy.samplers import sample


def test_die_basic():
    """Test basic die roll."""
    set_seed(42)
    result = ~die(6)
    assert result == 1


def test_die_multiple_samples():
    """Test rolling multiple dice."""
    set_seed(42)
    result = die(6) @ 10
    assert len(result) == 10
    for r in result:
        assert 1 <= r <= 6


def test_die_different_sides():
    """Test dice with different numbers of sides."""
    set_seed(42)
    for sides in [4, 8, 10, 12, 20, 100]:
        result = die(sides) @ 100
        assert min(result) >= 1
        assert max(result) <= sides


def test_die_exploding():
    """Test exploding dice mechanics."""
    set_seed(42)
    # With exploding dice on 6, we can get values > 6
    result = die(6, explode_on=6) @ 1000
    # At least some results should be > 6 (from explosions)
    assert max(result) > 6
    # All results should be >= 1
    assert min(result) >= 1


def test_die_exploding_multiple_values():
    """Test exploding on multiple values."""
    set_seed(42)
    result = die(6, explode_on=[5, 6]) @ 1000
    # Should have higher chance of explosions
    assert max(result) > 6


def test_die_str():
    """Test die string representation."""
    d = die(6)
    assert "<Distribution> Die(6)" == str(d)

    d_exploding = die(6, explode_on=6)
    assert "<Distribution> Die(6, explodes on [6])" == str(d_exploding)

    d_multi_explode = die(6, explode_on=[5, 6])
    assert "<Distribution> Die(6, explodes on [5, 6])" == str(d_multi_explode)


def test_die_invalid_sides():
    """Test that invalid sides raise errors."""
    with pytest.raises(ValueError) as excinfo:
        die(1)
    assert "cannot roll less than a 2-sided die" in str(excinfo.value)


def test_die_invalid_sides_type():
    """Test that non-integer sides raise errors."""
    with pytest.raises(ValueError) as excinfo:
        die(2.5)
    assert "can only roll an integer number of sides" in str(excinfo.value)


def test_die_explode_on_all_values():
    """Test that exploding on all values raises error."""
    with pytest.raises(ValueError) as excinfo:
        die(6, explode_on=[1, 2, 3, 4, 5, 6])
    assert "cannot explode on every value" in str(excinfo.value)


def test_die_explode_invalid_value():
    """Test that invalid explode_on values raise errors."""
    with pytest.raises(ValueError) as excinfo:
        die(6, explode_on=7)
    assert "explode_on values must be integers between 1 and 6" in str(excinfo.value)


def test_coin_basic():
    """Test basic coin flip."""
    set_seed(42)
    result = ~coin()
    assert result == "tails"


def test_coin_multiple_samples():
    """Test flipping multiple coins."""
    set_seed(42)
    result = coin() @ 10
    assert len(result) == 10
    for r in result:
        assert r in ["heads", "tails"]


def test_coin_distribution():
    """Test that coin flips are approximately 50/50."""
    set_seed(42)
    result = coin() @ 10000
    heads_count = sum(1 for r in result if r == "heads")
    # Should be close to 50%
    assert 4500 <= heads_count <= 5500


def test_coin_str():
    """Test coin string representation."""
    c = coin()
    assert "<Distribution> Coin" == str(c)


def test_die_is_operable():
    """Test that die can be used in operations."""
    set_seed(42)
    d = die(6)
    # Test that it's an OperableDistribution
    assert isinstance(d, OperableDistribution)
    assert isinstance(d, Die)
    # Test basic operations
    double_die = d * 2
    result = ~double_die
    assert result == 2


def test_coin_is_operable():
    """Test that coin is an OperableDistribution."""
    c = coin()
    assert isinstance(c, OperableDistribution)
    assert isinstance(c, Coin)


def test_die_sample_function():
    """Test using die with sample function."""
    set_seed(42)
    result = sample(die(6), n=5)
    assert len(result) == 5
    for r in result:
        assert 1 <= r <= 6


def test_coin_sample_function():
    """Test using coin with sample function."""
    set_seed(42)
    result = sample(coin(), n=5)
    assert len(result) == 5
    for r in result:
        assert r in ["heads", "tails"]
