"""
Core MFPCA implementation with numerical stability and performance optimization.

多変量関数的主成分分析のコア実装
数値安定性とパフォーマンスの最適化を重視
"""

import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline, splrep, splev
from typing import Optional, Tuple, List, Union
import warnings


class MFPCA:
    """
    Multivariate Functional Principal Component Analysis

    多変量時系列データの関数的主成分分析を実行するクラス。
    数値安定性、高速性、正確性を重視した実装。

    Parameters
    ----------
    n_components : int, optional
        Number of principal components to compute (default: None, compute all)
    smoothing : bool, optional
        Whether to smooth the data (default: True)
    smoothing_param : float, optional
        Smoothing parameter for B-spline (default: None, auto-select)
    n_basis : int, optional
        Number of basis functions (default: 20)
    basis_type : str, optional
        Type of basis functions: 'bspline', 'fourier' (default: 'bspline')
    center : bool, optional
        Whether to center the data (default: True)
    regularization : float, optional
        Regularization parameter for numerical stability (default: 1e-10)

    Attributes
    ----------
    eigenfunctions_ : ndarray of shape (n_components, n_timepoints, n_variables)
        The principal eigenfunctions
    eigenvalues_ : ndarray of shape (n_components,)
        The eigenvalues (variance explained by each component)
    scores_ : ndarray of shape (n_samples, n_components)
        Principal component scores
    mean_function_ : ndarray of shape (n_timepoints, n_variables)
        Mean function across all samples
    variance_explained_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each component
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        smoothing: bool = True,
        smoothing_param: Optional[float] = None,
        n_basis: int = 20,
        basis_type: str = 'bspline',
        center: bool = True,
        regularization: float = 1e-10
    ):
        self.n_components = n_components
        self.smoothing = smoothing
        self.smoothing_param = smoothing_param
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.center = center
        self.regularization = regularization

        # Will be set during fitting
        self.eigenfunctions_ = None
        self.eigenvalues_ = None
        self.scores_ = None
        self.mean_function_ = None
        self.variance_explained_ratio_ = None
        self.time_grid_ = None
        self.n_variables_ = None
        self.covariance_operator_ = None

    def fit(
        self,
        X: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> 'MFPCA':
        """
        Fit the MFPCA model to multivariate functional data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Multivariate functional data
        time_grid : ndarray of shape (n_timepoints,), optional
            Time grid for the observations (default: equally spaced [0, 1])

        Returns
        -------
        self : MFPCA
            Fitted MFPCA model
        """
        # Input validation
        X = self._validate_input(X)
        n_samples, n_timepoints, n_variables = X.shape
        self.n_variables_ = n_variables

        # Set time grid
        if time_grid is None:
            self.time_grid_ = np.linspace(0, 1, n_timepoints)
        else:
            self.time_grid_ = np.asarray(time_grid)
            if len(self.time_grid_) != n_timepoints:
                raise ValueError("time_grid length must match X.shape[1]")

        # Smooth data if requested
        if self.smoothing:
            X_smooth = self._smooth_data(X)
        else:
            X_smooth = X.copy()

        # Center data
        if self.center:
            self.mean_function_ = np.mean(X_smooth, axis=0)
            X_centered = X_smooth - self.mean_function_
        else:
            self.mean_function_ = np.zeros((n_timepoints, n_variables))
            X_centered = X_smooth

        # Compute covariance operator
        self.covariance_operator_ = self._compute_covariance_operator(X_centered)

        # Eigen-decomposition with numerical stability
        eigenvalues, eigenfunctions = self._eigen_decomposition(
            self.covariance_operator_,
            n_timepoints,
            n_variables
        )

        # Select number of components
        if self.n_components is None:
            n_comp = len(eigenvalues)
        else:
            n_comp = min(self.n_components, len(eigenvalues))

        self.eigenvalues_ = eigenvalues[:n_comp]
        self.eigenfunctions_ = eigenfunctions[:n_comp]

        # Compute scores (projection onto eigenfunctions)
        self.scores_ = self._compute_scores(X_centered)

        # CRITICAL: Recompute eigenvalues as score variances
        # This ensures eigenvalues match the functional PCA definition: λ_k = Var(ξ_k)
        # The eigenvalues from discrete eigen-decomposition may not match exactly
        # due to numerical integration and normalization effects
        self.eigenvalues_ = np.var(self.scores_, axis=0, ddof=0)

        # Compute variance explained
        total_variance = np.sum(self.eigenvalues_)
        if total_variance > 0:
            self.variance_explained_ratio_ = self.eigenvalues_ / total_variance
        else:
            self.variance_explained_ratio_ = np.zeros_like(self.eigenvalues_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Multivariate functional data to transform

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            Principal component scores
        """
        self._check_fitted()
        X = self._validate_input(X)

        # Smooth if necessary
        if self.smoothing:
            X_smooth = self._smooth_data(X)
        else:
            X_smooth = X.copy()

        # Center
        X_centered = X_smooth - self.mean_function_

        # Compute scores
        return self._compute_scores(X_centered)

    def fit_transform(
        self,
        X: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit the model and transform the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Multivariate functional data
        time_grid : ndarray of shape (n_timepoints,), optional
            Time grid for the observations

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            Principal component scores
        """
        return self.fit(X, time_grid).transform(X)

    def inverse_transform(
        self,
        scores: np.ndarray,
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Reconstruct data from principal component scores.

        Parameters
        ----------
        scores : ndarray of shape (n_samples, n_components)
            Principal component scores
        n_components : int, optional
            Number of components to use for reconstruction
            (default: all available components)

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_timepoints, n_variables)
            Reconstructed multivariate functional data
        """
        self._check_fitted()

        if n_components is None:
            n_components = scores.shape[1]

        n_samples = scores.shape[0]
        n_timepoints = len(self.time_grid_)
        n_variables = self.n_variables_

        # Reconstruct: X = mean + sum(score_i * eigenfunction_i)
        X_reconstructed = np.tile(
            self.mean_function_[np.newaxis, :, :],
            (n_samples, 1, 1)
        )

        for i in range(n_components):
            X_reconstructed += (
                scores[:, i, np.newaxis, np.newaxis] *
                self.eigenfunctions_[i, :, :]
            )

        return X_reconstructed

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate input data."""
        X = np.asarray(X)

        if X.ndim == 2:
            # Univariate case: (n_samples, n_timepoints) -> (n_samples, n_timepoints, 1)
            X = X[:, :, np.newaxis]
        elif X.ndim != 3:
            raise ValueError(
                f"X must be 2D or 3D array, got {X.ndim}D"
            )

        # Check for NaN or Inf
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values")

        return X

    def _smooth_data(self, X: np.ndarray) -> np.ndarray:
        """
        Smooth multivariate functional data using B-splines.

        高速で安定したB-splineスムージング
        """
        n_samples, n_timepoints, n_variables = X.shape
        X_smooth = np.zeros_like(X)

        # Auto-select smoothing parameter if not provided
        if self.smoothing_param is None:
            s = n_timepoints * 0.1  # Heuristic
        else:
            s = self.smoothing_param

        for i in range(n_samples):
            for j in range(n_variables):
                try:
                    # Use scipy's spline fitting with smoothing
                    tck = splrep(self.time_grid_, X[i, :, j], s=s, k=3)
                    X_smooth[i, :, j] = splev(self.time_grid_, tck)
                except Exception as e:
                    # Fallback: use original data if smoothing fails
                    warnings.warn(
                        f"Smoothing failed for sample {i}, variable {j}: {e}. "
                        "Using original data."
                    )
                    X_smooth[i, :, j] = X[i, :, j]

        return X_smooth

    def _compute_covariance_operator(
        self,
        X_centered: np.ndarray
    ) -> np.ndarray:
        """
        Compute the covariance operator efficiently.

        共分散作用素を高速に計算
        数値安定性のため、正則化を適用
        """
        n_samples, n_timepoints, n_variables = X_centered.shape

        # Reshape to (n_samples, n_timepoints * n_variables)
        X_flat = X_centered.reshape(n_samples, -1)

        # Compute covariance matrix: C = (1/n) * X^T * X
        # Using einsum for efficiency
        cov = np.einsum('ij,ik->jk', X_flat, X_flat) / n_samples

        # Add regularization for numerical stability
        cov += self.regularization * np.eye(cov.shape[0])

        return cov

    def _eigen_decomposition(
        self,
        covariance_operator: np.ndarray,
        n_timepoints: int,
        n_variables: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigen-decomposition with numerical stability.

        数値安定性を重視した固有値分解
        SVDを使用して対称行列の固有値分解を実行
        """
        # Ensure symmetry (numerical errors can break symmetry)
        cov_sym = (covariance_operator + covariance_operator.T) / 2

        # Use SVD for better numerical stability
        # For symmetric matrices, SVD gives eigendecomposition
        try:
            # Use eigh for symmetric matrices (faster and more stable)
            eigenvalues, eigenvectors = linalg.eigh(cov_sym)

            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Filter out negative eigenvalues (numerical noise)
            positive_idx = eigenvalues > max(1e-10, self.regularization)
            eigenvalues = eigenvalues[positive_idx]
            eigenvectors = eigenvectors[:, positive_idx]

        except linalg.LinAlgError:
            # Fallback to SVD if eigh fails
            warnings.warn("eigh failed, falling back to SVD")
            U, s, Vt = linalg.svd(cov_sym, full_matrices=False)
            eigenvalues = s
            eigenvectors = U

            # Filter positive eigenvalues
            positive_idx = eigenvalues > max(1e-10, self.regularization)
            eigenvalues = eigenvalues[positive_idx]
            eigenvectors = eigenvectors[:, positive_idx]

        # Reshape eigenvectors to (n_components, n_timepoints, n_variables)
        n_components = len(eigenvalues)
        eigenfunctions = eigenvectors.T.reshape(
            n_components, n_timepoints, n_variables
        )

        # Normalize eigenfunctions
        # Note: eigenvalues will be recomputed from score variances after projection
        for i in range(n_components):
            norm = np.sqrt(np.trapz(
                np.sum(eigenfunctions[i]**2, axis=1),
                self.time_grid_
            ))
            if norm > 1e-10:
                eigenfunctions[i] /= norm

        return eigenvalues, eigenfunctions

    def _compute_scores(self, X_centered: np.ndarray) -> np.ndarray:
        """
        Compute principal component scores via numerical integration.

        数値積分を使用してスコアを計算
        """
        n_samples = X_centered.shape[0]
        n_components = len(self.eigenvalues_)
        scores = np.zeros((n_samples, n_components))

        # Compute inner product: score_ij = <X_i, psi_j>
        for i in range(n_samples):
            for j in range(n_components):
                # Inner product via trapezoidal integration
                integrand = np.sum(
                    X_centered[i] * self.eigenfunctions_[j],
                    axis=1
                )
                scores[i, j] = np.trapz(integrand, self.time_grid_)

        return scores

    def _check_fitted(self):
        """Check if the model has been fitted."""
        if self.eigenfunctions_ is None:
            raise ValueError(
                "This MFPCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

    def explained_variance_cumsum(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio.

        Returns
        -------
        cumsum : ndarray
            Cumulative sum of explained variance ratios
        """
        self._check_fitted()
        return np.cumsum(self.variance_explained_ratio_)

    def reconstruct_partial(
        self,
        X: np.ndarray,
        n_components: int
    ) -> np.ndarray:
        """
        Reconstruct data using only first n_components.

        最初のn個の主成分のみを使用してデータを再構成

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Original data
        n_components : int
            Number of components to use

        Returns
        -------
        X_reconstructed : ndarray
            Reconstructed data
        """
        scores = self.transform(X)
        return self.inverse_transform(scores[:, :n_components], n_components)

    def get_reconstruction_error(
        self,
        X: np.ndarray,
        n_components: Optional[int] = None
    ) -> float:
        """
        Compute reconstruction error (RMSE).

        再構成誤差を計算

        Parameters
        ----------
        X : ndarray
            Original data
        n_components : int, optional
            Number of components to use (default: all)

        Returns
        -------
        rmse : float
            Root mean squared error
        """
        if n_components is None:
            n_components = len(self.eigenvalues_)

        X_reconstructed = self.reconstruct_partial(X, n_components)
        mse = np.mean((X - X_reconstructed) ** 2)
        return np.sqrt(mse)
