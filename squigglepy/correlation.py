"""
This module implements the Iman-Conover method for inducing correlations between distributions.

Some of the code has been adapted from Abraham Lee's mcerp package (https://github.com/tisimst/mcerp/).
"""

# Parts of `induce_correlation` are licensed as follows:

# BSD 3-Clause License

# Copyright (c) 2018, Abraham Lee
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import rankdata, spearmanr
from scipy.stats.distributions import norm as _scipy_norm
from numpy.typing import NDArray
from copy import deepcopy

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .distributions import OperableDistribution


def correlate(
    variables: tuple[OperableDistribution, ...],
    correlation: Union[NDArray[np.float64], list[list[float]], np.float64, float],
    tolerance: Union[float, np.float64, None] = 0.05,
    _min_unique_samples: int = 100,
):
    """
    Correlate a set of variables according to a rank correlation matrix.

    This employs the Iman-Conover method to induce the correlation while
    preserving the original marginal distributions.

    This method works on a best-effort basis, and may fail to induce the desired
    correlation depending on the distributions provided. An exception will be raised
    if that's the case.

    Parameters
    ----------
    variables : tuple of distributions
        The variables to correlate as a tuple of distributions.

        The distributions must be able to produce enough unique samples for the method
        to be able to induce the desired correlation by shuffling the samples.

        Discrete distributions are notably hard to correlate this way,
        as it's common for them to result in very few unique samples.

    correlation : 2d-array or float
        An n-by-n array that defines the desired Spearman rank correlation coefficients.
        This matrix must be symmetric and positives emi-definite; and must not be confused with
        a covariance matrix.

        Correlation parameters can only be between -1 and 1, exclusive
        (including extremely close approximations).

        If a float is provided, all variables will be correlated with the same coefficient.

    tolerance : float, optional
        If provided, overrides the absolute tolerance used to check if the resulting
        correlation matrix matches the desired correlation matrix. Defaults to 0.05.

        Checking can also be disabled by passing None.

    Returns
    -------
    correlated_variables : tuple of distributions
        The correlated variables as a tuple of distributions in the same order as
        the input variables.

    Examples
    --------
    Suppose we want to correlate two variables with a correlation coefficient of 0.65:
    >>> solar_radiation, temperature = sq.gamma(300, 100), sq.to(22, 28)
    >>> solar_radiation, temperature = sq.correlate((solar_radiation, temperature), 0.7)
    >>> print(np.corrcoef(solar_radiation @ 1000, temperature @ 1000)[0, 1])
        0.6975960649767123

    Or you could pass a correlation matrix:
    >>> funding_gap, cost_per_delivery, effect_size = (
            sq.to(20_000, 80_000), sq.to(30, 80), sq.beta(2, 5)
        )
    >>> funding_gap, cost_per_delivery, effect_size = sq.correlate(
            (funding_gap, cost_per_delivery, effect_size),
            [[1, 0.6, -0.5], [0.6, 1, -0.2], [-0.5, -0.2, 1]]
        )
    >>> print(np.corrcoef(funding_gap @ 1000, cost_per_delivery @ 1000, effect_size @ 1000))
        array([[ 1.      ,  0.580520  , -0.480149],
               [ 0.580962,  1.        , -0.187831],
               [-0.480149, -0.187831  ,  1.        ]])

    """
    if not isinstance(variables, tuple):
        variables = tuple(variables)

    if len(variables) < 2:
        raise ValueError("You must provide at least two variables to correlate.")

    assert all(v.correlation_group is None for v in variables)

    # Convert a float to a correlation matrix
    if (
        isinstance(correlation, float)
        or isinstance(correlation, np.floating)
        or isinstance(correlation, int)
    ):
        correlation_parameter = np.float64(correlation)

        assert (
            -1 < correlation_parameter < 1
        ), "Correlation parameter must be between -1 and 1, exclusive."
        # Generate a correlation matrix with
        # pairwise correlations equal to the correlation parameter
        correlation_matrix: NDArray[np.float64] = np.full(
            (len(variables), len(variables)), correlation_parameter
        )
        # Set the diagonal to 1
        np.fill_diagonal(correlation_matrix, 1)
    else:
        # Coerce the correlation matrix into a numpy array
        correlation_matrix: NDArray[np.float64] = np.array(correlation, dtype=np.float64)

    tolerance = float(tolerance) if tolerance is not None else None

    # Deepcopy the variables to avoid modifying the originals
    variables = deepcopy(variables)

    # Create the correlation group
    CorrelationGroup(variables, correlation_matrix, tolerance, _min_unique_samples)

    return variables


@dataclass
class CorrelationGroup:
    """
    An object that holds metadata for a group of correlated distributions.
    This object is not intended to be used directly by the user, but
    rather during sampling to induce correlations between distributions.
    """

    correlated_dists: tuple[OperableDistribution]
    correlation_matrix: NDArray[np.float64]
    correlation_tolerance: Union[float, None] = 0.05
    min_unique_samples: int = 100

    def __post_init__(self):
        # Check that the correlation matrix is square of the expected size
        assert (
            self.correlation_matrix.shape[0]
            == self.correlation_matrix.shape[1]
            == len(self.correlated_dists)
        ), "Correlation matrix must be square, and of the length of the number of dists. provided."

        # Check that the diagonal of the correlation matrix is all ones
        assert np.all(np.diag(self.correlation_matrix) == 1), "Diagonal must be all ones."

        # Check that values are between -1 and 1
        assert (
            -1 <= np.min(self.correlation_matrix) and np.max(self.correlation_matrix) <= 1
        ), "Correlation matrix values must be between -1 and 1."

        # Check that the correlation matrix is positive semi-definite
        assert np.all(
            np.linalg.eigvals(self.correlation_matrix) >= 0
        ), "Matrix must be positive semi-definite."

        # Check that the correlation matrix is symmetric
        assert np.all(
            self.correlation_matrix == self.correlation_matrix.T
        ), "Matrix must be symmetric."

        # Link the correlation group to each distribution
        for dist in self.correlated_dists:
            dist.correlation_group = self

    def induce_correlation(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Induce a set of correlations on a column-wise dataset

        Parameters
        ----------
        data : 2d-array
            An m-by-n array where m is the number of samples and n is the
            number of independent variables, each column of the array corresponding
            to each variable
        corrmat : 2d-array
            An n-by-n array that defines the desired correlation coefficients
            (between -1 and 1). Note: the matrix must be symmetric and
            positive-definite in order to induce.

        Returns
        -------
        new_data : 2d-array
            An m-by-n array that has the desired correlations.

        """
        # Check that each column doesn't have too little unique values
        for column in data.T:
            if not self.has_sufficient_sample_diversity(column):
                raise ValueError(
                    "The data has too many repeated values to induce a correlation. "
                    "This might be because of too few samples, or too many repeated samples."
                )

        # If the correlation matrix is the identity matrix, just return the data
        if np.all(self.correlation_matrix == np.eye(self.correlation_matrix.shape[0])):
            return data

        # Create a rank-matrix
        data_rank = np.vstack([rankdata(datai, method="min") for datai in data.T]).T

        # Generate van der Waerden scores
        data_rank_score = data_rank / (data_rank.shape[0] + 1.0)
        data_rank_score = _scipy_norm(0, 1).ppf(data_rank_score)

        # Calculate the lower triangular matrix of the Cholesky decomposition
        # of the desired correlation matrix
        p = cholesky(self.correlation_matrix, lower=True)

        # Calculate the current correlations
        t = np.corrcoef(data_rank_score, rowvar=False)

        # Calculate the lower triangular matrix of the Cholesky decomposition
        # of the current correlation matrix
        q = cholesky(t, lower=True)

        # Calculate the re-correlation matrix
        s = np.dot(p, np.linalg.inv(q))

        # Calculate the re-sampled matrix
        new_data = np.dot(data_rank_score, s.T)

        # Create the new rank matrix
        new_data_rank = np.vstack([rankdata(datai, method="min") for datai in new_data.T]).T

        # Sort the original data according to the new rank matrix
        self._sort_data_according_to_rank(data, data_rank, new_data_rank)

        # # Check correlation
        if self.correlation_tolerance:
            self._check_empirical_correlation(data)

        return data

    def _sort_data_according_to_rank(
        self,
        data: NDArray[np.float64],
        data_rank: NDArray[np.float64],
        new_data_rank: NDArray[np.float64],
    ):
        """Sorts the original data according to new_data_rank, in place."""
        assert (
            data.shape == data_rank.shape == new_data_rank.shape
        ), "All input arrays must have the same shape"
        for i in range(data.shape[1]):
            _, order = np.unique(
                np.hstack((data_rank[:, i], new_data_rank[:, i])), return_inverse=True
            )
            old_order = order[: new_data_rank.shape[0]]
            new_order = order[-new_data_rank.shape[0] :]
            tmp = data[np.argsort(old_order), i][new_order]
            data[:, i] = tmp[:]

    def _check_empirical_correlation(self, samples: NDArray[np.float64]):
        """
        Ensures that the empirical correlation matrix is
        the same as the desired correlation matrix.
        """
        assert self.correlation_tolerance is not None

        # Compute the empirical correlation matrix
        empirical_correlation = spearmanr(samples).statistic
        if len(self.correlated_dists) == 2:
            # empirical_correlation is a scalar
            properly_correlated = np.isclose(
                empirical_correlation,
                self.correlation_matrix[0, 1],
                atol=self.correlation_tolerance,
                rtol=0,
            )
        else:
            # empirical_correlation is a matrix
            properly_correlated = np.allclose(
                empirical_correlation,
                self.correlation_matrix,
                atol=self.correlation_tolerance,
                rtol=0,
            )
        if not properly_correlated:
            raise RuntimeError(
                "Failed to induce the desired correlation between samples. "
                "This might be because of too little diversity in the samples. "
                "You can relax the tolerance by passing `tolerance` to correlate()."
            )

    def has_sufficient_sample_diversity(
        self,
        samples: NDArray[np.float64],
        relative_threshold: float = 0.7,
        absolute_threshold=None,
    ) -> bool:
        """
        Check if there is there are sufficient unique samples to work with in the data.
        """

        if absolute_threshold is None:
            absolute_threshold = self.min_unique_samples

        unique_samples = len(np.unique(samples, axis=0))
        n_samples = len(samples)

        diversity = unique_samples / n_samples

        return (diversity >= relative_threshold) and (unique_samples >= absolute_threshold)
