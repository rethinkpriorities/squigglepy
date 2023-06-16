import scipy.stats as stats
from .. import squigglepy as sq
from hypothesis import given, assume, note, example
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import warnings


def check_correlation_from_matrix(dists, corr):
    samples = np.column_stack([dist @ 1000 for dist in dists])
    estimated_corr = stats.spearmanr(samples).statistic
    if len(dists) == 2:
        assert np.all(np.isclose(estimated_corr, corr[0, 1], atol=0.08)), estimated_corr
    else:
        assert np.all(np.isclose(estimated_corr, corr, atol=0.08)), estimated_corr


def check_correlation_from_parameter(a, b, corr):
    samples = np.column_stack([a @ 3_000, b @ 3_000])
    estimated_corr = stats.spearmanr(samples).statistic
    assert np.all(np.isclose(estimated_corr, corr, atol=0.05)), estimated_corr


@st.composite
def correlation_matrices(draw, min_size=2, max_size=20):
    # Generate a random list of correlations
    n_variables = draw(st.integers(min_size, max_size))
    correlation_matrix = draw(
        arrays(np.float64, (n_variables, n_variables), elements=st.floats(-0.99, 0.99))
    )
    # Reflect the matrix
    correlation_matrix = np.tril(correlation_matrix) + np.tril(correlation_matrix, -1).T
    # Fill the diagonal with 1s
    np.fill_diagonal(correlation_matrix, 1)

    # Reject if not positive semi-definite
    assume(np.all(np.linalg.eigvals(correlation_matrix) >= 0))

    return correlation_matrix


@st.composite
def random_distributions(draw):
    i = draw(st.integers(0, 6))
    if i == 0:
        # Uniform
        a = draw(st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False))
        b = draw(
            st.floats(
                min_value=a,
                allow_infinity=False,
                allow_nan=False,
                exclude_min=True,
                allow_subnormal=False,
            )
        )
        return sq.uniform(a, b)

    elif i == 2:
        # Normal / Log-Normal
        # Susceptible to subnormal values
        a = draw(st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False))
        b = draw(
            st.floats(
                min_value=a,
                allow_infinity=False,
                allow_nan=False,
                exclude_min=True,
                allow_subnormal=False,
            )
        )
        return sq.to(a, b)
    elif i == 3:
        # Binomial
        # Susceptible to overflows/underflows
        n = draw(st.integers(1, 500))
        p = draw(st.floats(0.01, 0.999, exclude_min=True, exclude_max=True))
        return sq.binomial(n, p)
    elif i == 4:
        # Bernoulli
        # Quite susceptible to overflows/underflows
        p = draw(st.floats(0.01, 0.999, exclude_min=True, exclude_max=True, allow_subnormal=False))
        return sq.bernoulli(p)
    elif i == 5:
        # Discrete
        items = draw(
            st.dictionaries(
                st.floats(0, 1), st.floats(allow_infinity=False, allow_nan=False), min_size=1
            )
        )
        return sq.discrete(items)
    else:
        # Exponential
        # This distribution is VERY finicky
        a = draw(
            st.floats(
                min_value=0,
                max_value=1e20,  # Prevents overflow
                exclude_min=True,
                exclude_max=True,
                allow_infinity=False,
                allow_nan=False,
                allow_subnormal=False,  # Prevents overflow (again)
            )
        )
        return sq.exponential(a)


@st.composite
def distributions_with_correlation(draw, min_size=2, max_size=20):
    dist = tuple(draw(st.lists(random_distributions(), min_size=min_size, max_size=max_size)))
    corr = draw(correlation_matrices(min_size=len(dist), max_size=len(dist)))
    note(f"Distributions: {dist}")
    note(f"Correlation matrix: {corr}")
    return dist, corr


@given(st.floats(-0.999, 0.999))
@example(corr=0.5).via("discovered failure")
def test_basic_correlate(corr):
    """
    Test a basic example of correlation between two distributions.
    This ensures that the resulting distributions are correlated as expected.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        a, b = sq.uniform(-1, 1), sq.to(0, 3)
        a, b = sq.correlate((a, b), [[1, corr], [corr, 1]])
        check_correlation_from_parameter(a, b, corr)


@given(distributions_with_correlation())
def test_arbitrary_correlates(dist_corrs):
    """
    Test multi-variable correlation with an arbitrarily generated correlation matrix.
    Ensures the resulting corr. matrix is as expected.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        dists, corrs = dist_corrs
        dists = sq.correlate(dists, corrs)
        check_correlation_from_matrix(dists, corrs)

    # Check marginal distributions
