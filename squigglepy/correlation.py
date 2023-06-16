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
from scipy.stats import rankdata
from scipy.stats.distributions import norm
from numpy.typing import NDArray

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .distributions import OperableDistribution


def correlate(
    variables: tuple[OperableDistribution, ...],
    correlation: NDArray[np.float64] | list[list[float]] | np.float64 | float,
):
    """
    Correlate a set of variables according to a rank correlation matrix.

    This employs the Iman-Conover method to induce the correlation while
    preserving the original marginal distributions.

    Parameters
    ----------
    variables : tuple of distributions
        The variables to correlate as a tuple of distributions.

    correlation : 2d-array or float
        An n-by-n array that defines the desired Spearman rank correlation coefficients.
        This matrix must be symmetric and positives emi-definite; and must not be confused with
        a covariance matrix.

        Correlation parameters can only be between -1 and 1, exclusive
        (including extremely close approximations).

        If a float is provided, all variables will be correlated with the same coefficient.

    Returns
    -------
    correlated_variables : tuple of distributions
        The correlated variables as a tuple of distributions in the same order as
        the input variables.

    Examples
    --------
    >>> a, b = sq.uniform(-1, 1), sq.to(0, 3)
    >>> a, b = sq.correlate((a, b), [[1, 0.9], [0.9, 1]])
    >>> a_samples, b_samples = a @ 1000, b @ 1000
    >>> print(np.corrcoef(a_samples, b_samples).statistic)
        0.8923975890079759
    """
    if not isinstance(variables, tuple):
        variables = tuple(variables)

    assert len(variables) >= 2, "Must provide at least two variables to correlate."

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

    # Coerce the correlation matrix into a numpy array
    correlation_matrix: NDArray[np.float64] = np.array(correlation, dtype=np.float64)

    # Create the correlation group
    CorrelationGroup(variables, correlation_matrix)

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
        # Create a rank-matrix
        data_rank = np.vstack([rankdata(datai) for datai in data.T]).T

        # Generate van der Waerden scores
        data_rank_score = data_rank / (data_rank.shape[0] + 1.0)
        data_rank_score = norm(0, 1).ppf(data_rank_score)

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
        new_data_rank = np.vstack([rankdata(datai) for datai in new_data.T]).T

        # Sort the original data according to the new rank matrix
        self._sort_data_according_to_rank(data, data_rank, new_data_rank)

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
            # Get the sorted indices of data_rank and new_data_rank
            old_order = np.argsort(data_rank[:, i])
            new_order = np.argsort(new_data_rank[:, i])

            # Sort data according to old_order
            sorted_data = data[old_order, i]

            # Re-order the sorted data according to new_order
            data[:, i] = sorted_data[new_order]
