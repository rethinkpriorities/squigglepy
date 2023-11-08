from typing import Callable

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, note
from hypothesis.extra.numpy import arrays

from .. import squigglepy as sq

CONTINUOUS_DISTRIBUTIONS = [
    sq.uniform,
    sq.norm,
    sq.lognorm,
    # sq.to, # Disabled to help isolate errors to either normal or lognormal
    sq.beta,
    sq.tdist,
    # sq.log_tdist,  # TODO: Re-enable when overflows are fixed
    sq.triangular,
    sq.chisquare,
    sq.exponential,
    sq.gamma,
    sq.pareto,
    sq.pert,
]

DISCRETE_DISTRIBUTIONS = [
    sq.binomial,
    sq.bernoulli,
    sq.discrete,
    sq.poisson,
]

ALL_DISTRIBUTIONS = CONTINUOUS_DISTRIBUTIONS + DISCRETE_DISTRIBUTIONS


@st.composite
def distributions_with_correlation(draw, min_size=2, max_size=20, continuous_only=False):
    dists = tuple(
        draw(
            st.lists(
                random_distributions(continuous_only),
                min_size=min_size,
                max_size=max_size,
            )
        )
    )
    corr = draw(correlation_matrices(min_size=len(dists), max_size=len(dists)))
    note(f"Distributions: {dists}")
    note(f"Correlation matrix: {corr}")
    return dists, corr


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
def random_distributions(
    draw, continuous_only: bool = False, discrete_only: bool = False
) -> sq.OperableDistribution:
    assert not (continuous_only and discrete_only), "Cannot be both continuous and discrete"

    if continuous_only:
        dist = instantiate_with_parameters(draw, draw(st.sampled_from(CONTINUOUS_DISTRIBUTIONS)))
        assert isinstance(dist, sq.ContinuousDistribution), f"{dist} is not continuous"
    elif discrete_only:
        dist = instantiate_with_parameters(draw, draw(st.sampled_from(DISCRETE_DISTRIBUTIONS)))
        assert isinstance(dist, sq.DiscreteDistribution), f"{dist} is not discrete"
    else:
        dist = instantiate_with_parameters(draw, draw(st.sampled_from(ALL_DISTRIBUTIONS)))
        assert isinstance(dist, sq.OperableDistribution), f"{dist} is not an operable distribution"

    return dist


def instantiate_with_parameters(draw, dist_fn: Callable) -> sq.OperableDistribution:
    if dist_fn == sq.uniform:
        a = draw(
            st.floats(-1e30, 1e30, allow_nan=False, allow_infinity=False, allow_subnormal=False)
        )
        b = draw(
            st.floats(
                min_value=a + 2,
                max_value=a + 1e30,
                allow_nan=False,
                allow_subnormal=False,
                exclude_min=True,
            )
        )
        return dist_fn(a, b)
    elif dist_fn in (sq.norm, sq.tdist):
        # Distributions that receive confidence intervals
        a = draw(
            st.floats(-1e30, 1e30, allow_infinity=False, allow_nan=False, allow_subnormal=False)
        )
        b = draw(
            st.floats(
                min_value=a + 0.01,
                max_value=a + 1e30,
                allow_infinity=False,
                allow_nan=False,
                exclude_min=True,
                allow_subnormal=False,
            )
        )
        return dist_fn(a, b)
    elif dist_fn in (sq.lognorm, sq.log_tdist):
        # Distributions that receive confidence intervals starting from 0
        a = draw(
            st.floats(
                0.005,
                1e20,
                allow_infinity=False,
                allow_nan=False,
                allow_subnormal=False,
                exclude_min=True,
            )
        )
        b = draw(
            st.floats(
                min_value=a + 0.05,
                max_value=a + 1e20,
                allow_infinity=False,
                allow_nan=False,
                exclude_min=True,
                allow_subnormal=False,
            )
        )
        return dist_fn(a, b)
    elif dist_fn == sq.binomial:
        n = draw(st.integers(1, 500))
        p = draw(st.floats(0.01, 0.999, exclude_min=True, exclude_max=True))
        return dist_fn(n, p)
    elif dist_fn == sq.bernoulli:
        p = draw(st.floats(0.01, 0.999, exclude_min=True, exclude_max=True, allow_subnormal=False))
        return dist_fn(p)
    elif dist_fn == sq.discrete:
        items = draw(
            st.dictionaries(
                st.floats(allow_infinity=False, allow_nan=False),
                st.floats(0, 1, exclude_min=True),
                min_size=1,
            )
        )
        # Normalize the probabilities
        normalized_items = dict()
        value_sum = sum(items.values())
        for k, v in items.items():
            normalized_items[k] = v / value_sum

        return dist_fn(normalized_items)
    elif dist_fn == sq.exponential:
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
        return dist_fn(a)
    elif dist_fn == sq.beta:
        a = draw(
            st.floats(
                min_value=0.01,
                max_value=100,
                exclude_min=True,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        )
        b = draw(
            st.floats(
                min_value=0.01,
                max_value=100,
                exclude_min=True,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        )
        return dist_fn(a, b)

    elif dist_fn == sq.triangular:
        a = draw(
            st.floats(-1e30, 1e30, allow_infinity=False, allow_nan=False, allow_subnormal=False)
        )
        b = draw(
            st.floats(
                min_value=a + 0.05,
                max_value=a + 1e30,
                allow_infinity=False,
                allow_nan=False,
                allow_subnormal=False,
                exclude_min=True,
            )
        )
        c = draw(
            st.floats(
                min_value=a,
                max_value=b,
                allow_infinity=False,
                allow_nan=False,
                allow_subnormal=False,
            )
        )
        return dist_fn(a, c, b)

    elif dist_fn == sq.pert:
        low = draw(
            st.floats(-1e30, 1e30, allow_infinity=False, allow_nan=False, allow_subnormal=False)
        )
        mode_offset = draw(
            st.floats(
                min_value=0.05,
                max_value=1e30,
                allow_infinity=False,
                allow_nan=False,
            )
        )
        high_offset = draw(
            st.floats(
                min_value=0.05,
                max_value=1e30,
                allow_infinity=False,
                allow_nan=False,
            )
        )
        shape = draw(
            st.floats(
                min_value=0.05,
                max_value=1e30,
                allow_infinity=False,
                allow_nan=False,
            )
        )
        return dist_fn(low, low + mode_offset, low + mode_offset + high_offset, shape)

    elif dist_fn == sq.poisson:
        lambda_ = draw(st.integers(1, 1000))
        return dist_fn(lambda_)

    elif dist_fn == sq.chisquare:
        df = draw(st.integers(1, 1000))
        return dist_fn(df)

    elif dist_fn == sq.gamma:
        shape = draw(
            st.floats(
                min_value=0.01,
                max_value=100,
                exclude_min=True,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        )
        scale = draw(
            st.floats(
                min_value=0.01,
                max_value=100,
                exclude_min=True,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        )
        return dist_fn(shape, scale)

    elif dist_fn == sq.pareto:
        b = draw(
            st.floats(
                min_value=1.01,
                max_value=10.0,
                exclude_min=True,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        )
        return dist_fn(b)
    else:
        raise NotImplementedError(f"Unknown distribution {dist_fn}")
