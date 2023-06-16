import scipy.stats as stats
import pytest
from .. import squigglepy as sq
from hypothesis import given
import hypothesis.strategies as st

def check_correlation(a, b, corr):
    a_samples = a @ 1000
    b_samples = b @ 1000
    assert stats.pearsonr(a_samples, b_samples).statistic == pytest.approx(corr, abs=0.1)

@given(st.floats(-1, 1))
def test_basic_correlate(corr):
    a, b = sq.uniform(-1, 1), sq.to(0, 3)
    a, b = sq.correlate((a, b), [[1, corr], [corr, 1]])
    check_correlation(a, b, corr)
