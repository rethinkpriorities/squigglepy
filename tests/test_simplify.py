from hypothesis import assume, example, given
import hypothesis.strategies as st
import numpy as np
import pytest
from pytest import approx

from ..squigglepy.distributions import *

def test_simplify_add_normal():
    x = NormalDistribution(mean=1, sd=1)
    y = NormalDistribution(mean=2, sd=2)
    sum2 = x + y
    simplified2 = sum2.simplify()
    assert isinstance(simplified2, NormalDistribution)
    assert simplified2.mean == 3
    assert simplified2.sd == approx(np.sqrt(5))

def test_simplify_add_3_normals():
    x = NormalDistribution(mean=1, sd=1)
    y = NormalDistribution(mean=2, sd=2)
    z = NormalDistribution(mean=-3, sd=2)
    sum3_left = (x + y) + z
    sum3_right = x + (y + z)
    simplified3_left = sum3_left.simplify()
    simplified3_right = sum3_right.simplify()
    assert isinstance(simplified3_left, NormalDistribution)
    assert simplified3_left.mean == 0
    assert simplified3_left.sd == approx(np.sqrt(1 + 4 + 4))
    assert isinstance(simplified3_right, NormalDistribution)
    assert simplified3_right.mean == 0
    assert simplified3_right.sd == approx(np.sqrt(1 + 4 + 4))

def test_simplify_normal_plus_const():
    x = NormalDistribution(mean=0, sd=1)
    y = 2
    sum2 = x + y
    simplified = sum2.simplify()
    assert isinstance(simplified, NormalDistribution)
    assert simplified.mean == 2
    assert simplified.sd == approx(1)

def simplify_scale_normal():
    x = NormalDistribution(mean=2, sd=4)
    y = 1.5
    product2 = x * y
    simplified = product2.simplify()
    assert isinstance(simplified, NormalDistribution)
    assert simplified.mean == approx(3)
    assert simplified.sd == approx(6)

def test_simplify_mul_3_normals():
    x = LognormalDistribution(norm_mean=1, norm_sd=1)
    y = LognormalDistribution(norm_mean=2, norm_sd=2)
    z = LognormalDistribution(norm_mean=-2, norm_sd=3)
    product3_left = (x * y) * z
    product3_right = x * (y * z)
    simplified3_left = product3_left.simplify()
    simplified3_right = product3_right.simplify()
    assert isinstance(simplified3_left, LognormalDistribution)
    assert simplified3_left.norm_mean == 1
    assert simplified3_left.norm_sd == approx(np.sqrt(1 + 4 + 9))
    assert isinstance(simplified3_right, LognormalDistribution)
    assert simplified3_right.norm_mean == 1
    assert simplified3_right.norm_sd == approx(np.sqrt(1 + 4 + 9))

def test_simplify_sub():
    x = NormalDistribution(mean=1, sd=1)
    y = NormalDistribution(mean=2, sd=2)
    difference = x - y
    simplified = difference.simplify()
    assert isinstance(simplified, NormalDistribution)
    assert simplified.mean == -1
    assert simplified.sd == approx(np.sqrt(5))
