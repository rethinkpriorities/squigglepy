from ..squigglepy.numbers import K, M, B, T


def test_thousand():
    assert K == 1000


def test_million():
    assert M == 10 ** 6


def test_billion():
    assert B == 10 ** 9


def test_trillion():
    assert T == 10 ** 12
