"""
Theoretically rigorous MFPCA implementation based on functional data analysis literature.

論文に基づいた理論的に厳密なMFPCA実装

References:
-----------
1. Ramsay, J. O., & Silverman, B. W. (2005). Functional Data Analysis. Springer.
2. Happ, C., & Greven, S. (2018). Multivariate Functional Principal Component Analysis
   for Data Observed on Different (Dimensional) Domains. JASA, 113(522), 649-659.
3. Jacques, J., & Preda, C. (2014). Functional data clustering: a survey.
   Advances in Data Analysis and Classification, 8(3), 231-255.
"""

import numpy as np
from scipy import linalg
from typing import Optional, Tuple, Union
import warnings

from .basis import BasisSystem, create_basis


class TheoreticalMFPCA:
    """
    Multivariate Functional Principal Component Analysis with theoretical rigor.

    理論的に厳密な多変量関数的主成分分析

    Mathematical Formulation:
    -------------------------
    For multivariate functional data X_i(t) ∈ R^p, i=1,...,n, t ∈ [0,1]:

    1. Mean function: μ(t) = E[X(t)]

    2. Covariance operator: C(s,t) = E[(X(s) - μ(s))(X(t) - μ(t))^T]

    3. Eigenvalue problem: ∫ C(s,t) φ_k(t) dt = λ_k φ_k(s)

    4. Karhunen-Loève expansion: X_i(t) = μ(t) + Σ_k ξ_{ik} φ_k(t)

    5. Scores: ξ_{ik} = ∫ [X_i(t) - μ(t)]^T φ_k(t) dt

    Implementation via Basis Expansion:
    -----------------------------------
    - Represent data: X_i(t) ≈ Σ_j c_{ij} B_j(t) where {B_j} is a basis
    - Coefficient matrix: C = (c_{ij}) ∈ R^{n × (p·K)} where K is n_basis
    - Covariance in basis space: Σ = (1/n) C^T W C where W is Gram matrix
    - Eigendecomposition: Σ V = V Λ
    - Eigenfunctions: φ_k(t) = Σ_j v_{kj} B_j(t)

    Parameters
    ----------
    n_components : int, optional
        Number of principal components (default: None, all)
    n_basis : int or 'auto'
        Number of basis functions per variable (default: 'auto')
        If 'auto', automatically selects based on n_timepoints (recommended)
        Rule of thumb: n_basis ≈ 0.4 * n_timepoints (capped at 50)
    basis_type : str
        Type of basis: 'bspline' or 'fourier' (default: 'bspline')
    basis_degree : int
        Degree for B-spline basis (default: 3)
    smoothing : bool
        Whether to use smoothing penalty (default: True)
    smoothing_penalty : float, optional
        Smoothing parameter λ in penalized estimation (default: None, auto)
        If None, automatically selects based on estimated noise level
    penalty_order : int
        Order of derivative in penalty (default: 2)
    center : bool
        Whether to center data (default: True)
    regularization : float
        Regularization for numerical stability (default: 1e-8)

    Attributes
    ----------
    basis_ : BasisSystem
        Basis function system
    coefficients_ : ndarray of shape (n_samples, n_basis * n_variables)
        Basis expansion coefficients for data
    mean_coefficients_ : ndarray of shape (n_basis * n_variables,)
        Coefficients for mean function
    eigenvalues_ : ndarray of shape (n_components,)
        Eigenvalues λ_k
    eigenvector_coefficients_ : ndarray of shape (n_basis * n_variables, n_components)
        Basis coefficients for eigenfunctions
    scores_ : ndarray of shape (n_samples, n_components)
        Principal component scores ξ_{ik}
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        n_basis: Union[int, str] = 'auto',
        basis_type: str = 'bspline',
        basis_degree: int = 3,
        smoothing: bool = True,
        smoothing_penalty: Optional[float] = None,
        penalty_order: int = 2,
        center: bool = True,
        regularization: float = 1e-8
    ):
        self.n_components = n_components
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.basis_degree = basis_degree
        self.smoothing = smoothing
        self.smoothing_penalty = smoothing_penalty
        self.penalty_order = penalty_order
        self.center = center
        self.regularization = regularization

        # Fitted attributes
        self.basis_ = None
        self.coefficients_ = None
        self.mean_coefficients_ = None
        self.eigenvalues_ = None
        self.eigenvector_coefficients_ = None
        self.scores_ = None
        self.time_grid_ = None
        self.n_variables_ = None
        self.gram_matrix_ = None
        self.penalty_matrix_ = None

    def fit(
        self,
        X: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> 'TheoreticalMFPCA':
        """
        Fit MFPCA model using basis expansion method.

        基底展開法によるMFPCAモデルの学習

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Multivariate functional data
        time_grid : ndarray of shape (n_timepoints,), optional
            Time points (default: equally spaced [0, 1])

        Returns
        -------
        self : TheoreticalMFPCA
            Fitted model
        """
        # Validate input
        X = self._validate_input(X)
        n_samples, n_timepoints, n_variables = X.shape
        self.n_variables_ = n_variables

        # Set time grid
        if time_grid is None:
            self.time_grid_ = np.linspace(0, 1, n_timepoints)
        else:
            self.time_grid_ = np.asarray(time_grid)
            if len(self.time_grid_) != n_timepoints:
                raise ValueError(
                    f"time_grid length ({len(self.time_grid_)}) must match "
                    f"X.shape[1] ({n_timepoints})"
                )

        # Auto-select n_basis if needed
        if self.n_basis == 'auto' or self.n_basis is None:
            n_basis = self._auto_select_n_basis(n_timepoints)
            if self.n_basis == 'auto':
                print(f"Auto-selected n_basis={n_basis} for n_timepoints={n_timepoints}")
        else:
            n_basis = int(self.n_basis)
            # Validate user-provided n_basis
            if n_basis < self.basis_degree + 1:
                warnings.warn(
                    f"n_basis={n_basis} is too small for degree={self.basis_degree}. "
                    f"Minimum is {self.basis_degree + 1}. Auto-selecting instead."
                )
                n_basis = self._auto_select_n_basis(n_timepoints)
            elif n_basis < n_timepoints // 4:
                warnings.warn(
                    f"n_basis={n_basis} may be too small for n_timepoints={n_timepoints}. "
                    f"Consider using at least {n_timepoints // 4} basis functions for better accuracy."
                )
            elif n_basis > n_timepoints:
                warnings.warn(
                    f"n_basis={n_basis} exceeds n_timepoints={n_timepoints}. "
                    f"This may cause overfitting. Reducing to n_timepoints."
                )
                n_basis = n_timepoints

        # Store the selected n_basis
        self.n_basis_used_ = n_basis

        # Create basis system
        domain = (self.time_grid_[0], self.time_grid_[-1])
        self.basis_ = create_basis(
            self.basis_type,
            n_basis,
            domain=domain,
            degree=self.basis_degree
        )

        # Evaluate basis on time grid
        B = self.basis_.evaluate(self.time_grid_)  # (n_timepoints, n_basis)

        # Compute Gram matrix: G = ∫ B(t) B(t)^T dt
        self.gram_matrix_ = self.basis_.gram_matrix(self.time_grid_)

        # Compute penalty matrix for smoothing
        if self.smoothing:
            self.penalty_matrix_ = self.basis_.penalty_matrix(
                self.time_grid_,
                order=self.penalty_order
            )
        else:
            self.penalty_matrix_ = np.zeros((self.basis_.n_basis, self.basis_.n_basis))

        # Step 1: Compute basis expansion coefficients for each variable
        # X_ij(t) ≈ Σ_k c_{ijk} B_k(t) for variable j
        self.coefficients_ = self._compute_basis_coefficients(X, B)

        # Step 2: Compute mean function coefficients
        if self.center:
            self.mean_coefficients_ = np.mean(self.coefficients_, axis=0)
            C_centered = self.coefficients_ - self.mean_coefficients_
        else:
            self.mean_coefficients_ = np.zeros(self.basis_.n_basis * n_variables)
            C_centered = self.coefficients_

        # Step 3: Compute covariance matrix in basis space
        # Σ = (1/n) C^T C (standard covariance)
        # W (Gram matrix) will be used later for L2 inner products
        Sigma = (C_centered.T @ C_centered) / n_samples

        # Add regularization for numerical stability
        Sigma += self.regularization * np.eye(Sigma.shape[0])

        # Step 4: Eigendecomposition
        eigenvalues, eigenvectors = self._eigen_decomposition(Sigma)

        # Step 5: Select components
        if self.n_components is None:
            n_comp = len(eigenvalues)
        else:
            n_comp = min(self.n_components, len(eigenvalues))

        self.eigenvalues_ = eigenvalues[:n_comp]
        self.eigenvector_coefficients_ = eigenvectors[:, :n_comp]

        # Step 6: Normalize eigenfunctions FIRST to have unit L2 norm
        self._normalize_eigenfunctions()

        # Step 7: Compute scores via L2 projection
        # ξ_{ik} = ∫ [X_i(t) - μ(t)]^T φ_k(t) dt
        # This must be done AFTER normalization for correct projections
        self.scores_ = self._compute_scores_l2(X)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component scores.

        データを主成分スコアに変換

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Data to transform

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            PC scores
        """
        self._check_fitted()

        X = self._validate_input(X)

        # Compute scores via L2 projection (same as in fit)
        scores = self._compute_scores_l2(X)

        return scores

    def fit_transform(
        self,
        X: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit model and transform data."""
        return self.fit(X, time_grid).transform(X)

    def inverse_transform(
        self,
        scores: np.ndarray,
        n_components: Optional[int] = None,
        time_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reconstruct data from scores.

        スコアからデータを再構成

        X_reconstructed(t) = μ(t) + Σ_k ξ_k φ_k(t)

        Parameters
        ----------
        scores : ndarray of shape (n_samples, n_components)
            PC scores
        n_components : int, optional
            Number of components to use (default: all in scores)
        time_grid : ndarray, optional
            Time points for reconstruction (default: training grid)

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_timepoints, n_variables)
            Reconstructed data
        """
        self._check_fitted()

        if n_components is None:
            n_components = scores.shape[1]

        if time_grid is None:
            time_grid = self.time_grid_

        # Evaluate basis on reconstruction grid
        B = self.basis_.evaluate(time_grid)
        n_timepoints = len(time_grid)

        # Reconstruct coefficients: c = c_mean + Σ_k ξ_k v_k
        C_recon = self.mean_coefficients_ + scores[:, :n_components] @ \
                  self.eigenvector_coefficients_[:, :n_components].T

        # Reconstruct functions: X(t) = Σ_j c_j B_j(t)
        n_samples = scores.shape[0]
        X_reconstructed = np.zeros((n_samples, n_timepoints, self.n_variables_))

        for i in range(n_samples):
            for j in range(self.n_variables_):
                # Coefficients for variable j
                c_ij = C_recon[i, j * self.basis_.n_basis:(j + 1) * self.basis_.n_basis]
                # Reconstruct: X_ij(t) = Σ_k c_ijk B_k(t)
                X_reconstructed[i, :, j] = B @ c_ij

        return X_reconstructed

    def get_eigenfunctions(
        self,
        time_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Evaluate eigenfunctions on time grid.

        固有関数を時間グリッド上で評価

        Parameters
        ----------
        time_grid : ndarray, optional
            Time points (default: training grid)

        Returns
        -------
        eigenfunctions : ndarray of shape (n_components, n_timepoints, n_variables)
            Evaluated eigenfunctions
        """
        self._check_fitted()

        if time_grid is None:
            time_grid = self.time_grid_

        B = self.basis_.evaluate(time_grid)
        n_timepoints = len(time_grid)
        n_components = self.eigenvector_coefficients_.shape[1]

        eigenfunctions = np.zeros((n_components, n_timepoints, self.n_variables_))

        for k in range(n_components):
            for j in range(self.n_variables_):
                # Coefficients for k-th eigenfunction, j-th variable
                v_kj = self.eigenvector_coefficients_[
                    j * self.basis_.n_basis:(j + 1) * self.basis_.n_basis, k
                ]
                # Evaluate: φ_k^(j)(t) = Σ_m v_{kjm} B_m(t)
                eigenfunctions[k, :, j] = B @ v_kj

        return eigenfunctions

    def get_mean_function(
        self,
        time_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Evaluate mean function on time grid.

        平均関数を評価

        Returns
        -------
        mean_function : ndarray of shape (n_timepoints, n_variables)
        """
        self._check_fitted()

        if time_grid is None:
            time_grid = self.time_grid_

        B = self.basis_.evaluate(time_grid)
        n_timepoints = len(time_grid)

        mean_function = np.zeros((n_timepoints, self.n_variables_))

        for j in range(self.n_variables_):
            c_j = self.mean_coefficients_[j * self.basis_.n_basis:(j + 1) * self.basis_.n_basis]
            mean_function[:, j] = B @ c_j

        return mean_function

    def explained_variance_ratio(self) -> np.ndarray:
        """
        Compute variance explained ratio.

        Returns
        -------
        variance_ratio : ndarray of shape (n_components,)
        """
        self._check_fitted()
        total_var = np.sum(self.eigenvalues_)
        if total_var > 0:
            return self.eigenvalues_ / total_var
        else:
            return np.zeros_like(self.eigenvalues_)

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and format input data."""
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 2:
            # Univariate: (n_samples, n_timepoints) -> (n_samples, n_timepoints, 1)
            X = X[:, :, np.newaxis]
        elif X.ndim != 3:
            raise ValueError(f"X must be 2D or 3D, got {X.ndim}D")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values")

        return X

    def _compute_basis_coefficients(
        self,
        X: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Compute basis expansion coefficients.

        基底展開係数の計算

        Solves: min_c ||X - B c||^2 + λ c^T P c

        where P is penalty matrix for smoothing.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Data
        B : ndarray of shape (n_timepoints, n_basis)
            Basis matrix

        Returns
        -------
        C : ndarray of shape (n_samples, n_basis * n_variables)
            Coefficients
        """
        n_samples, n_timepoints, n_variables = X.shape

        # Auto-select smoothing parameter if not provided
        if self.smoothing and self.smoothing_penalty is None:
            # Improved heuristic based on noise level estimation
            sigma_noise = self._estimate_noise_level(X)

            # Base ratio of penalty matrices
            trace_BTB = np.trace(B.T @ B)
            trace_P = np.trace(self.penalty_matrix_) + 1e-10

            # Adjust based on noise level
            # Low noise (< 0.1): light smoothing
            # Medium noise (0.1-0.5): moderate smoothing
            # High noise (> 0.5): strong smoothing
            if sigma_noise < 0.1:
                lambda_smooth = 0.001 * trace_BTB / trace_P
            elif sigma_noise < 0.5:
                lambda_smooth = 0.01 * trace_BTB / trace_P
            else:
                lambda_smooth = 0.05 * trace_BTB / trace_P

            # Store for reference
            self.smoothing_penalty_used_ = lambda_smooth
        else:
            lambda_smooth = self.smoothing_penalty if self.smoothing else 0.0
            self.smoothing_penalty_used_ = lambda_smooth

        # Solve for each variable separately
        # (B^T B + λ P) c = B^T x
        BTB = B.T @ B
        if self.smoothing:
            A = BTB + lambda_smooth * self.penalty_matrix_
        else:
            A = BTB

        # Add regularization for stability
        A += self.regularization * np.eye(self.basis_.n_basis)

        C = np.zeros((n_samples, n_variables * self.basis_.n_basis))

        for i in range(n_samples):
            for j in range(n_variables):
                x_ij = X[i, :, j]
                b_ij = B.T @ x_ij

                # Solve linear system
                try:
                    c_ij = linalg.solve(A, b_ij, assume_a='pos')
                except linalg.LinAlgError:
                    # Fallback to least squares
                    c_ij = linalg.lstsq(A, b_ij)[0]

                C[i, j * self.basis_.n_basis:(j + 1) * self.basis_.n_basis] = c_ij

        return C

    def _construct_weight_matrix(self) -> np.ndarray:
        """
        Construct block-diagonal weight matrix with Gram matrices.

        W = diag(G, G, ..., G) where G is Gram matrix

        Returns
        -------
        W : ndarray of shape (n_basis * n_variables, n_basis * n_variables)
        """
        W = np.kron(np.eye(self.n_variables_), self.gram_matrix_)
        return W

    def _eigen_decomposition(self, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigendecomposition with numerical stability.

        固有値分解（数値安定性を考慮）

        Returns
        -------
        eigenvalues : ndarray
            Sorted eigenvalues (descending)
        eigenvectors : ndarray
            Corresponding eigenvectors (columns)
        """
        # Ensure symmetry
        Sigma_sym = (Sigma + Sigma.T) / 2

        # Use eigh for symmetric matrices
        try:
            eigenvalues, eigenvectors = linalg.eigh(Sigma_sym)

            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Filter negative eigenvalues (numerical noise)
            positive_mask = eigenvalues > max(1e-10, self.regularization)
            eigenvalues = eigenvalues[positive_mask]
            eigenvectors = eigenvectors[:, positive_mask]

        except linalg.LinAlgError as e:
            warnings.warn(f"Eigendecomposition failed: {e}. Using SVD.")
            U, s, Vt = linalg.svd(Sigma_sym, full_matrices=False)
            eigenvalues = s
            eigenvectors = U

            positive_mask = eigenvalues > max(1e-10, self.regularization)
            eigenvalues = eigenvalues[positive_mask]
            eigenvectors = eigenvectors[:, positive_mask]

        return eigenvalues, eigenvectors

    def _normalize_eigenfunctions(self):
        """
        Normalize eigenfunctions to unit L2 norm via numerical integration.

        固有関数のL2ノルムを1に正規化

        ||φ_k||^2 = ∫ φ_k(t)^T φ_k(t) dt

        Uses numerical integration on the actual eigenfunction evaluations
        rather than Gram matrix to ensure consistency with reconstruction.
        """
        # Evaluate eigenfunctions on time grid
        eigenfuncs = self.get_eigenfunctions(self.time_grid_)

        # Integration weights (trapezoidal rule)
        dt = (self.time_grid_[-1] - self.time_grid_[0]) / (len(self.time_grid_) - 1)
        weights = np.ones(len(self.time_grid_))
        weights[0] = weights[-1] = 0.5
        weights *= dt

        for k in range(self.eigenvector_coefficients_.shape[1]):
            # Compute L2 norm: ||φ_k||^2 = ∫ ||φ_k(t)||^2 dt
            # where ||φ_k(t)||^2 = sum over variables of φ_k^(j)(t)^2
            integrand = np.sum(eigenfuncs[k]**2, axis=1)  # Sum over variables
            norm_squared = np.sum(integrand * weights)

            if norm_squared > 1e-10:
                norm = np.sqrt(norm_squared)
                # Normalize eigenfunction coefficients
                self.eigenvector_coefficients_[:, k] /= norm
                # Note: Scores will be computed separately via L2 projection after this

    def _compute_scores_l2(self, X: np.ndarray) -> np.ndarray:
        """
        Compute scores via L2 projection.

        L2射影によるスコアの計算

        ξ_{ik} = ∫ [X_i(t) - μ(t)]^T φ_k(t) dt

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Original data

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            PC scores via L2 projection
        """
        n_samples = X.shape[0]
        n_components = self.eigenvector_coefficients_.shape[1]

        # Get mean function and eigenfunctions
        mean_func = self.get_mean_function(self.time_grid_)
        eigenfuncs = self.get_eigenfunctions(self.time_grid_)

        # Integration weights (trapezoidal rule)
        dt = (self.time_grid_[-1] - self.time_grid_[0]) / (len(self.time_grid_) - 1)
        weights = np.ones(len(self.time_grid_))
        weights[0] = weights[-1] = 0.5
        weights *= dt

        # Compute scores
        scores = np.zeros((n_samples, n_components))

        for i in range(n_samples):
            for k in range(n_components):
                # ξ_{ik} = ∫ [X_i(t) - μ(t)]^T φ_k(t) dt
                # = sum over t and j: [X_i(t,j) - μ(t,j)] * φ_k(t,j) * weight(t)
                X_centered = X[i] - mean_func  # (n_timepoints, n_variables)
                integrand = np.sum(X_centered * eigenfuncs[k], axis=1)  # Sum over variables
                scores[i, k] = np.sum(integrand * weights)

        return scores

    def _auto_select_n_basis(self, n_timepoints: int) -> int:
        """
        Auto-select number of basis functions based on n_timepoints.

        自動的に基底関数の数を選択

        Rule of thumb from FDA literature:
        - n_basis should be roughly n_timepoints/2 to n_timepoints/3
        - But not too small (min 10-15) or too large (max 50)

        Parameters
        ----------
        n_timepoints : int
            Number of time points in data

        Returns
        -------
        n_basis : int
            Recommended number of basis functions
        """
        if n_timepoints < 20:
            # For very short series, use half the timepoints
            n_basis = max(self.basis_degree + 1, n_timepoints // 2)
        elif n_timepoints < 50:
            # For medium series, use 40-50% of timepoints
            n_basis = max(15, int(n_timepoints * 0.45))
        else:
            # For long series, use 30-40% but cap at reasonable maximum
            n_basis = max(20, min(int(n_timepoints * 0.4), 50))

        # Ensure minimum for basis degree
        n_basis = max(n_basis, self.basis_degree + 2)

        return n_basis

    def _estimate_noise_level(self, X: np.ndarray) -> float:
        """
        Estimate noise level in data for smoothing parameter selection.

        データのノイズレベルを推定

        Uses median absolute deviation of second differences.

        Returns
        -------
        sigma_noise : float
            Estimated noise standard deviation
        """
        # Compute second differences along time axis
        d2 = np.diff(X, n=2, axis=1)

        # Robust estimate: MAD / 0.6745
        mad = np.median(np.abs(d2))
        sigma_noise = mad / 0.6745

        return sigma_noise

    def _check_fitted(self):
        """Check if model is fitted."""
        if self.basis_ is None:
            raise ValueError(
                "Model not fitted. Call fit() before using this method."
            )

    def get_reconstruction_error(
        self,
        X: np.ndarray,
        n_components: Optional[int] = None,
        metric: str = 'rmse'
    ) -> float:
        """
        Compute reconstruction error.

        再構成誤差の計算

        Parameters
        ----------
        X : ndarray
            Original data
        n_components : int, optional
            Number of components to use
        metric : str
            Error metric: 'rmse', 'mse', 'mae'

        Returns
        -------
        error : float
        """
        if n_components is None:
            n_components = self.eigenvector_coefficients_.shape[1]

        scores = self.transform(X)
        X_recon = self.inverse_transform(scores[:, :n_components], n_components)

        diff = X - X_recon

        if metric == 'rmse':
            return np.sqrt(np.mean(diff ** 2))
        elif metric == 'mse':
            return np.mean(diff ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(diff))
        else:
            raise ValueError(f"Unknown metric: {metric}")
