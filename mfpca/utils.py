"""
Utility functions for MFPCA.

MFPCA用のユーティリティ関数
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.stats import multivariate_normal


def create_time_grid(
    start: float = 0.0,
    end: float = 1.0,
    n_points: int = 100,
    irregular: bool = False,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create time grid for functional data.

    関数型データ用の時間グリッドを作成

    Parameters
    ----------
    start : float, optional
        Start time (default: 0.0)
    end : float, optional
        End time (default: 1.0)
    n_points : int, optional
        Number of time points (default: 100)
    irregular : bool, optional
        Whether to create irregular grid (default: False)
    seed : int, optional
        Random seed for irregular grid

    Returns
    -------
    time_grid : ndarray
        Time grid
    """
    if irregular:
        if seed is not None:
            np.random.seed(seed)
        # Generate irregular grid
        time_grid = np.sort(np.random.uniform(start, end, n_points))
    else:
        # Regular grid
        time_grid = np.linspace(start, end, n_points)

    return time_grid


def compute_covariance_surface(
    X: np.ndarray,
    time_grid: Optional[np.ndarray] = None,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute covariance surface for functional data.

    関数型データの共分散曲面を計算

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timepoints, n_variables)
        Functional data
    time_grid : ndarray, optional
        Time grid
    center : bool, optional
        Whether to center the data (default: True)

    Returns
    -------
    cov_surface : ndarray of shape (n_timepoints, n_timepoints, n_variables, n_variables)
        Covariance surface
    mean_function : ndarray of shape (n_timepoints, n_variables)
        Mean function
    time_grid : ndarray
        Time grid used
    """
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, :, np.newaxis]

    n_samples, n_timepoints, n_variables = X.shape

    if time_grid is None:
        time_grid = np.linspace(0, 1, n_timepoints)

    # Compute mean function
    mean_function = np.mean(X, axis=0)

    # Center data
    if center:
        X_centered = X - mean_function
    else:
        X_centered = X

    # Compute covariance surface
    cov_surface = np.zeros((n_timepoints, n_timepoints, n_variables, n_variables))

    for t1 in range(n_timepoints):
        for t2 in range(n_timepoints):
            # Cov(X(t1), X(t2)) = E[(X(t1) - mu(t1)) * (X(t2) - mu(t2))^T]
            cov_t1_t2 = np.zeros((n_variables, n_variables))
            for i in range(n_samples):
                cov_t1_t2 += np.outer(
                    X_centered[i, t1, :],
                    X_centered[i, t2, :]
                )
            cov_surface[t1, t2, :, :] = cov_t1_t2 / n_samples

    return cov_surface, mean_function, time_grid


def generate_synthetic_mfpca_data(
    n_samples: int = 100,
    n_timepoints: int = 50,
    n_variables: int = 3,
    n_components: int = 3,
    eigenvalues: Optional[np.ndarray] = None,
    noise_std: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic multivariate functional data for testing.

    テスト用の合成多変量関数型データを生成

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (default: 100)
    n_timepoints : int, optional
        Number of time points (default: 50)
    n_variables : int, optional
        Number of variables (default: 3)
    n_components : int, optional
        Number of true components (default: 3)
    eigenvalues : ndarray, optional
        True eigenvalues (default: exponentially decreasing)
    noise_std : float, optional
        Standard deviation of noise (default: 0.1)
    seed : int, optional
        Random seed

    Returns
    -------
    X : ndarray of shape (n_samples, n_timepoints, n_variables)
        Generated functional data
    true_scores : ndarray of shape (n_samples, n_components)
        True principal component scores
    true_eigenfunctions : ndarray of shape (n_components, n_timepoints, n_variables)
        True eigenfunctions
    time_grid : ndarray
        Time grid
    """
    if seed is not None:
        np.random.seed(seed)

    time_grid = np.linspace(0, 1, n_timepoints)

    # Generate true eigenvalues
    if eigenvalues is None:
        eigenvalues = np.exp(-np.arange(n_components))
    else:
        eigenvalues = np.asarray(eigenvalues)

    # Generate true eigenfunctions using Fourier basis
    true_eigenfunctions = np.zeros((n_components, n_timepoints, n_variables))

    for k in range(n_components):
        for j in range(n_variables):
            # Mix of sine and cosine with different frequencies
            freq = k + 1
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.5, 1.5)

            if k % 2 == 0:
                true_eigenfunctions[k, :, j] = amplitude * np.sin(
                    2 * np.pi * freq * time_grid + phase
                )
            else:
                true_eigenfunctions[k, :, j] = amplitude * np.cos(
                    2 * np.pi * freq * time_grid + phase
                )

    # Normalize eigenfunctions
    for k in range(n_components):
        norm = np.sqrt(np.trapz(
            np.sum(true_eigenfunctions[k]**2, axis=1),
            time_grid
        ))
        true_eigenfunctions[k] /= norm

    # Generate scores
    true_scores = np.random.randn(n_samples, n_components) * np.sqrt(eigenvalues)

    # Generate mean function
    mean_function = np.zeros((n_timepoints, n_variables))
    for j in range(n_variables):
        mean_function[:, j] = np.sin(2 * np.pi * time_grid) * (j + 1) * 0.5

    # Reconstruct data: X = mean + sum(score * eigenfunction)
    X = np.tile(mean_function[np.newaxis, :, :], (n_samples, 1, 1))

    for i in range(n_samples):
        for k in range(n_components):
            X[i] += true_scores[i, k] * true_eigenfunctions[k]

    # Add noise
    X += np.random.randn(n_samples, n_timepoints, n_variables) * noise_std

    return X, true_scores, true_eigenfunctions, time_grid


def align_eigenfunctions(
    eigenfunctions1: np.ndarray,
    eigenfunctions2: np.ndarray,
    time_grid: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two sets of eigenfunctions (resolve sign ambiguity).

    2つの固有関数セットを整列（符号の不定性を解決）

    Parameters
    ----------
    eigenfunctions1 : ndarray
        First set of eigenfunctions
    eigenfunctions2 : ndarray
        Second set of eigenfunctions
    time_grid : ndarray, optional
        Time grid for integration

    Returns
    -------
    aligned_eigenfunctions2 : ndarray
        Aligned second set
    correlations : ndarray
        Correlation coefficients
    """
    n_components = min(eigenfunctions1.shape[0], eigenfunctions2.shape[0])
    n_timepoints = eigenfunctions1.shape[1]

    if time_grid is None:
        time_grid = np.linspace(0, 1, n_timepoints)

    aligned_eigenfunctions2 = eigenfunctions2.copy()
    correlations = np.zeros(n_components)

    for k in range(n_components):
        # Compute correlation
        integrand = np.sum(
            eigenfunctions1[k] * eigenfunctions2[k],
            axis=1
        )
        correlation = np.trapz(integrand, time_grid)
        correlations[k] = correlation

        # Flip sign if negative correlation
        if correlation < 0:
            aligned_eigenfunctions2[k] *= -1
            correlations[k] *= -1

    return aligned_eigenfunctions2, correlations


def select_n_components(
    variance_explained: np.ndarray,
    threshold: float = 0.95
) -> int:
    """
    Select number of components based on variance explained.

    説明分散に基づいて成分数を選択

    Parameters
    ----------
    variance_explained : ndarray
        Variance explained by each component
    threshold : float, optional
        Cumulative variance threshold (default: 0.95)

    Returns
    -------
    n_components : int
        Number of components to retain
    """
    cumsum = np.cumsum(variance_explained)
    n_components = np.searchsorted(cumsum, threshold) + 1
    return min(n_components, len(variance_explained))


def compute_functional_correlation(
    X1: np.ndarray,
    X2: np.ndarray,
    time_grid: Optional[np.ndarray] = None
) -> float:
    """
    Compute correlation between two functional data sets.

    2つの関数型データセット間の相関を計算

    Parameters
    ----------
    X1 : ndarray of shape (n_timepoints, n_variables)
        First functional data
    X2 : ndarray of shape (n_timepoints, n_variables)
        Second functional data
    time_grid : ndarray, optional
        Time grid

    Returns
    -------
    correlation : float
        Functional correlation
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.shape != X2.shape:
        raise ValueError("X1 and X2 must have the same shape")

    n_timepoints = X1.shape[0]

    if time_grid is None:
        time_grid = np.linspace(0, 1, n_timepoints)

    # Compute L2 inner products
    inner_product = np.trapz(np.sum(X1 * X2, axis=1), time_grid)
    norm1 = np.sqrt(np.trapz(np.sum(X1**2, axis=1), time_grid))
    norm2 = np.sqrt(np.trapz(np.sum(X2**2, axis=1), time_grid))

    if norm1 > 1e-10 and norm2 > 1e-10:
        correlation = inner_product / (norm1 * norm2)
    else:
        correlation = 0.0

    return correlation


def bootstrap_mfpca(
    X: np.ndarray,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
    **mfpca_kwargs
) -> dict:
    """
    Perform bootstrap analysis for MFPCA.

    MFPCAのブートストラップ解析を実行
    信頼区間を推定

    Parameters
    ----------
    X : ndarray
        Functional data
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 100)
    confidence_level : float, optional
        Confidence level (default: 0.95)
    seed : int, optional
        Random seed
    **mfpca_kwargs
        Additional arguments for MFPCA

    Returns
    -------
    results : dict
        Dictionary containing bootstrap results
    """
    from .core import MFPCA

    if seed is not None:
        np.random.seed(seed)

    n_samples = X.shape[0]

    # Fit original model
    mfpca = MFPCA(**mfpca_kwargs)
    mfpca.fit(X)

    n_components = len(mfpca.eigenvalues_)

    # Storage for bootstrap results
    bootstrap_eigenvalues = []
    bootstrap_varexp = []

    for b in range(n_bootstrap):
        # Bootstrap sample (with replacement)
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]

        # Fit MFPCA
        mfpca_boot = MFPCA(**mfpca_kwargs)
        try:
            mfpca_boot.fit(X_boot)
            bootstrap_eigenvalues.append(mfpca_boot.eigenvalues_[:n_components])
            bootstrap_varexp.append(
                mfpca_boot.variance_explained_ratio_[:n_components]
            )
        except Exception:
            # Skip failed bootstrap samples
            continue

    bootstrap_eigenvalues = np.array(bootstrap_eigenvalues)
    bootstrap_varexp = np.array(bootstrap_varexp)

    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    results = {
        'original_eigenvalues': mfpca.eigenvalues_,
        'eigenvalues_mean': np.mean(bootstrap_eigenvalues, axis=0),
        'eigenvalues_std': np.std(bootstrap_eigenvalues, axis=0),
        'eigenvalues_ci_lower': np.percentile(
            bootstrap_eigenvalues, lower_percentile, axis=0
        ),
        'eigenvalues_ci_upper': np.percentile(
            bootstrap_eigenvalues, upper_percentile, axis=0
        ),
        'variance_explained_mean': np.mean(bootstrap_varexp, axis=0),
        'variance_explained_std': np.std(bootstrap_varexp, axis=0),
        'variance_explained_ci_lower': np.percentile(
            bootstrap_varexp, lower_percentile, axis=0
        ),
        'variance_explained_ci_upper': np.percentile(
            bootstrap_varexp, upper_percentile, axis=0
        ),
        'n_bootstrap': len(bootstrap_eigenvalues),
        'confidence_level': confidence_level
    }

    return results


def cross_validate_mfpca(
    X: np.ndarray,
    n_folds: int = 5,
    n_components_range: Optional[List[int]] = None,
    seed: Optional[int] = None,
    **mfpca_kwargs
) -> dict:
    """
    Cross-validation for selecting number of components.

    成分数を選択するための交差検証

    Parameters
    ----------
    X : ndarray
        Functional data
    n_folds : int, optional
        Number of folds (default: 5)
    n_components_range : list of int, optional
        Range of n_components to try
    seed : int, optional
        Random seed
    **mfpca_kwargs
        Additional arguments for MFPCA

    Returns
    -------
    results : dict
        Cross-validation results
    """
    from .core import MFPCA

    if seed is not None:
        np.random.seed(seed)

    n_samples = X.shape[0]

    # Create fold indices
    fold_size = n_samples // n_folds
    indices = np.random.permutation(n_samples)

    if n_components_range is None:
        # Try up to 10 components
        n_components_range = list(range(1, min(11, n_samples)))

    # Storage for CV errors
    cv_errors = {n_comp: [] for n_comp in n_components_range}

    for fold in range(n_folds):
        # Split data
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        X_train = X[train_idx]
        X_test = X[test_idx]

        # Fit MFPCA with max components
        mfpca = MFPCA(n_components=max(n_components_range), **mfpca_kwargs)
        mfpca.fit(X_train)

        # Compute reconstruction error for each n_components
        for n_comp in n_components_range:
            if n_comp <= len(mfpca.eigenvalues_):
                rmse = mfpca.get_reconstruction_error(X_test, n_comp)
                cv_errors[n_comp].append(rmse)

    # Average over folds
    mean_errors = {n_comp: np.mean(cv_errors[n_comp])
                   for n_comp in n_components_range
                   if len(cv_errors[n_comp]) > 0}

    best_n_components = min(mean_errors, key=mean_errors.get)

    results = {
        'cv_errors': cv_errors,
        'mean_errors': mean_errors,
        'best_n_components': best_n_components,
        'n_folds': n_folds
    }

    return results
