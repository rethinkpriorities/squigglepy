import scipy.stats as stats

from .. import squigglepy as sq
from .strategies import distributions_with_correlation
from hypothesis import given, assume, note, example
import hypothesis.strategies as st
import numpy as np
import warnings
from copy import deepcopy


def check_correlation_from_matrix(dists, corr, atol=0.08):
    samples = np.column_stack([dist @ 3_000 for dist in dists])
    estimated_corr = stats.spearmanr(samples).statistic
    note(f"Estimated correlation: {estimated_corr}")
    if len(dists) == 2:
        note(f"Desired correlation: {corr[0, 1]}")
        assert np.all(np.isclose(estimated_corr, corr[0, 1], atol=atol))
    else:
        note(f"Desired correlation: {corr}")
        assert np.all(np.isclose(estimated_corr, corr, atol=atol))
    return samples


def check_correlation_from_parameter(dists_or_samples, corr, atol=0.08):
    assert len(dists_or_samples) == 2
    if isinstance(dists_or_samples[0], sq.OperableDistribution):
        # Sample
        samples = np.column_stack([dists_or_samples[0] @ 5_000, dists_or_samples[1] @ 5_000])
    else:
        assert isinstance(dists_or_samples[0], np.ndarray)
        samples = np.column_stack(dists_or_samples)

    note(f"Desired correlation: {corr}")
    estimated_corr = stats.spearmanr(samples).statistic
    note(f"Estimated correlation: {estimated_corr}")
    assert np.all(np.isclose(estimated_corr, corr, atol=atol))


@given(st.floats(-0.999, 0.999))
@example(corr=0.5).via("discovered failure")
def test_basic_correlates(corr):
    """
    Test a basic example of correlation between two distributions.
    This ensures that the resulting distributions are correlated as expected.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        a_params = (-1, 1)
        b_params = (0, 1)

        a, b = sq.UniformDistribution(*a_params), sq.NormalDistribution(
            mean=b_params[0], sd=b_params[1]
        )
        a, b = sq.correlate((a, b), corr, tolerance=None)

        # Sample
        a_samples = a @ 3_000
        b_samples = b @ 3_000
        check_correlation_from_parameter((a, b), corr)

        # Check marginal distributions
        # a (uniform)
        assert np.isclose(
            np.mean(a_samples), np.mean(a_params), atol=0.08
        ), f"Mean: {np.mean(a_samples)} != {np.mean(a_params)}"
        expected_sd = np.sqrt((a_params[1] - a_params[0]) ** 2 / 12)
        assert np.isclose(
            np.std(a_samples), expected_sd, atol=0.08
        ), f"SD: {np.std(a_samples)} != {expected_sd}"
        # b (normal)
        assert np.isclose(np.mean(b_samples), b_params[0], atol=0.08), np.mean(b_samples)
        assert np.isclose(np.std(b_samples), b_params[1], atol=0.08), np.std(b_samples)


@given(distributions_with_correlation())
def test_arbitrary_correlates(dist_corrs):
    """
    Test multi-variable correlation with a series of arbitrary random variables,
    and an arbitrarily generated correlation matrix.

    Ensures the resulting corr. matrix is as expected, and that the marginal
    distributionsremain intact.
    """

    uncorrelated_dists, corrs = dist_corrs
    correlated_dists = sq.correlate(
        uncorrelated_dists, corrs, tolerance=None, _min_unique_samples=1_000
    )
    try:
        # The tolerance is quite high here, given that we're only
        # interested in very signifcant errors. The user would be warned
        # if the correlation was too far off anyway (with less tolerance).
        check_correlation_from_matrix(correlated_dists, corrs, atol=0.1)
    except ValueError as e:
        assume("repeated values" not in str(e))
        raise e

    # Check that marginal distributions are preserved
    group = correlated_dists[0].correlation_group
    assert group is not None
    uncorr_samples = np.column_stack(
        [
            sq.sample(dist, 3_000, _correlate_if_needed=False)
            for dist in group.correlated_dists
        ]
    )
    corr_samples = group.induce_correlation(uncorr_samples)

    assert np.isclose(
        np.mean(uncorr_samples), np.mean(corr_samples), rtol=0.01
    ), "Means are not equal, violating integrity of marginal distributions"
    assert np.isclose(
        np.std(uncorr_samples), np.std(corr_samples), rtol=0.01
    ), "SDs are not equal, violating integrity of marginal distributions"
    assert np.isclose(
        np.median(uncorr_samples), np.median(corr_samples), rtol=0.01
    )
    assert np.isclose(np.max(uncorr_samples), np.max(corr_samples), rtol=0.01)
    assert np.isclose(np.min(uncorr_samples), np.min(corr_samples), rtol=0.01)


def test_correlated_resampling():
    """
    Tests that correlated distributions can be resampled
    without stale samples being used (self._correlated_samples)
    """
    uncorrelated_dists = sq.to(2, 30), sq.uniform(-3, 6), sq.beta(50, 100)
    correlated_dists = sq.correlate(
        uncorrelated_dists, 0.8, tolerance=None
    )

    first_samples = np.column_stack([d @ 1_000 for d in correlated_dists])
    second_samples = np.column_stack([d @ 1_000 for d in correlated_dists])
    
    assert not np.allclose(
        first_samples, second_samples
    ), "Resampling correlated distributions produces the same samples"
