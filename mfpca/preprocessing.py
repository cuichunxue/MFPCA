"""
Preprocessing and smoothing utilities for functional data.

関数型データの前処理とスムージングユーティリティ
"""

import numpy as np
from scipy.interpolate import BSpline, splrep, splev, UnivariateSpline
from scipy.signal import savgol_filter
from typing import Optional, Tuple, Union
import warnings


class FunctionalDataPreprocessor:
    """
    Preprocessor for multivariate functional data.

    多変量関数型データの前処理クラス
    欠損値処理、外れ値検出、正規化などを実行

    Parameters
    ----------
    handle_missing : str, optional
        How to handle missing values: 'interpolate', 'remove', 'zero' (default: 'interpolate')
    outlier_detection : bool, optional
        Whether to detect and handle outliers (default: False)
    outlier_threshold : float, optional
        Z-score threshold for outlier detection (default: 3.0)
    normalize : bool, optional
        Whether to normalize each variable (default: False)
    normalize_method : str, optional
        Normalization method: 'zscore', 'minmax', 'robust' (default: 'zscore')
    """

    def __init__(
        self,
        handle_missing: str = 'interpolate',
        outlier_detection: bool = False,
        outlier_threshold: float = 3.0,
        normalize: bool = False,
        normalize_method: str = 'zscore'
    ):
        self.handle_missing = handle_missing
        self.outlier_detection = outlier_detection
        self.outlier_threshold = outlier_threshold
        self.normalize = normalize
        self.normalize_method = normalize_method

        # Statistics for normalization
        self.means_ = None
        self.stds_ = None
        self.mins_ = None
        self.maxs_ = None
        self.medians_ = None
        self.mads_ = None  # Median absolute deviation

    def fit(self, X: np.ndarray, time_grid: Optional[np.ndarray] = None) -> 'FunctionalDataPreprocessor':
        """
        Fit the preprocessor to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Input data
        time_grid : ndarray, optional
            Time grid

        Returns
        -------
        self
        """
        X = self._validate_input(X)
        n_samples, n_timepoints, n_variables = X.shape

        if self.normalize:
            if self.normalize_method == 'zscore':
                # Compute mean and std for each variable across all samples and time
                X_flat = X.reshape(-1, n_variables)
                self.means_ = np.nanmean(X_flat, axis=0)
                self.stds_ = np.nanstd(X_flat, axis=0)
                self.stds_[self.stds_ == 0] = 1.0  # Avoid division by zero

            elif self.normalize_method == 'minmax':
                X_flat = X.reshape(-1, n_variables)
                self.mins_ = np.nanmin(X_flat, axis=0)
                self.maxs_ = np.nanmax(X_flat, axis=0)
                # Avoid division by zero
                range_ = self.maxs_ - self.mins_
                range_[range_ == 0] = 1.0
                self.maxs_ = self.mins_ + range_

            elif self.normalize_method == 'robust':
                X_flat = X.reshape(-1, n_variables)
                self.medians_ = np.nanmedian(X_flat, axis=0)
                self.mads_ = np.nanmedian(
                    np.abs(X_flat - self.medians_), axis=0
                )
                self.mads_[self.mads_ == 0] = 1.0  # Avoid division by zero

        return self

    def transform(
        self,
        X: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transform the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timepoints, n_variables)
            Input data
        time_grid : ndarray, optional
            Time grid

        Returns
        -------
        X_transformed : ndarray
            Transformed data
        """
        X = self._validate_input(X)
        X_transformed = X.copy()

        # Handle missing values
        if self.handle_missing == 'interpolate':
            X_transformed = self._interpolate_missing(X_transformed, time_grid)
        elif self.handle_missing == 'zero':
            X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        elif self.handle_missing == 'remove':
            # Remove samples with any missing values
            mask = np.any(np.isnan(X_transformed), axis=(1, 2))
            X_transformed = X_transformed[~mask]

        # Outlier detection
        if self.outlier_detection:
            X_transformed = self._handle_outliers(X_transformed)

        # Normalization
        if self.normalize:
            X_transformed = self._normalize(X_transformed)

        return X_transformed

    def fit_transform(
        self,
        X: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform the data."""
        return self.fit(X, time_grid).transform(X, time_grid)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse normalization.

        Parameters
        ----------
        X : ndarray
            Normalized data

        Returns
        -------
        X_original : ndarray
            Data in original scale
        """
        if not self.normalize:
            return X

        X_inv = X.copy()

        if self.normalize_method == 'zscore':
            X_inv = X_inv * self.stds_ + self.means_
        elif self.normalize_method == 'minmax':
            X_inv = X_inv * (self.maxs_ - self.mins_) + self.mins_
        elif self.normalize_method == 'robust':
            X_inv = X_inv * self.mads_ + self.medians_

        return X_inv

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate input."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        elif X.ndim != 3:
            raise ValueError(f"X must be 2D or 3D, got {X.ndim}D")
        return X

    def _interpolate_missing(
        self,
        X: np.ndarray,
        time_grid: Optional[np.ndarray]
    ) -> np.ndarray:
        """Interpolate missing values."""
        n_samples, n_timepoints, n_variables = X.shape

        if time_grid is None:
            time_grid = np.arange(n_timepoints)

        for i in range(n_samples):
            for j in range(n_variables):
                series = X[i, :, j]
                if np.any(np.isnan(series)):
                    # Find non-NaN indices
                    valid_idx = ~np.isnan(series)
                    if np.sum(valid_idx) > 1:  # Need at least 2 points
                        # Linear interpolation
                        X[i, :, j] = np.interp(
                            time_grid,
                            time_grid[valid_idx],
                            series[valid_idx]
                        )
                    else:
                        # Fill with 0 if too few points
                        X[i, :, j] = 0.0

        return X

    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        """Detect and handle outliers using z-score method."""
        n_samples, n_timepoints, n_variables = X.shape

        for j in range(n_variables):
            # Compute z-scores across all samples and time points
            X_var = X[:, :, j].flatten()
            mean = np.mean(X_var)
            std = np.std(X_var)

            if std > 0:
                z_scores = np.abs((X[:, :, j] - mean) / std)
                outlier_mask = z_scores > self.outlier_threshold

                # Replace outliers with median
                if np.any(outlier_mask):
                    median = np.median(X_var)
                    X[:, :, j][outlier_mask] = median

        return X

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize data."""
        if self.normalize_method == 'zscore':
            return (X - self.means_) / self.stds_
        elif self.normalize_method == 'minmax':
            return (X - self.mins_) / (self.maxs_ - self.mins_)
        elif self.normalize_method == 'robust':
            return (X - self.medians_) / self.mads_
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")


class BSplineSmoother:
    """
    B-spline smoother for functional data.

    B-splineによる関数型データのスムージング
    高速で安定した実装

    Parameters
    ----------
    n_basis : int, optional
        Number of basis functions (default: 10)
    degree : int, optional
        Degree of B-spline (default: 3, cubic)
    smoothing : float, optional
        Smoothing parameter (default: None, auto-select)
    penalty_order : int, optional
        Order of derivative penalty (default: 2)
    """

    def __init__(
        self,
        n_basis: int = 10,
        degree: int = 3,
        smoothing: Optional[float] = None,
        penalty_order: int = 2
    ):
        self.n_basis = n_basis
        self.degree = degree
        self.smoothing = smoothing
        self.penalty_order = penalty_order

        self.knots_ = None
        self.coefficients_ = None

    def fit(
        self,
        y: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> 'BSplineSmoother':
        """
        Fit B-spline to data.

        Parameters
        ----------
        y : ndarray of shape (n_timepoints,) or (n_timepoints, n_variables)
            Data to smooth
        time_grid : ndarray, optional
            Time grid

        Returns
        -------
        self
        """
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_timepoints, n_variables = y.shape

        if time_grid is None:
            time_grid = np.linspace(0, 1, n_timepoints)

        # Auto-select smoothing parameter
        if self.smoothing is None:
            s = n_timepoints * 0.1
        else:
            s = self.smoothing

        # Fit B-spline for each variable
        self.coefficients_ = []
        self.knots_ = []

        for j in range(n_variables):
            try:
                # Use UnivariateSpline for automatic knot selection
                spl = UnivariateSpline(time_grid, y[:, j], s=s, k=self.degree)
                self.knots_.append(spl.get_knots())
                self.coefficients_.append(spl.get_coeffs())
            except Exception as e:
                warnings.warn(f"Smoothing failed for variable {j}: {e}")
                # Fallback: use simple spline fit
                tck = splrep(time_grid, y[:, j], k=min(self.degree, n_timepoints - 1))
                self.knots_.append(tck[0])
                self.coefficients_.append(tck[1])

        return self

    def transform(
        self,
        time_grid: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate smoothed function on new time grid.

        Parameters
        ----------
        time_grid : ndarray
            Time points for evaluation

        Returns
        -------
        y_smooth : ndarray of shape (len(time_grid), n_variables)
            Smoothed values
        """
        if self.coefficients_ is None:
            raise ValueError("Smoother not fitted. Call fit() first.")

        n_variables = len(self.coefficients_)
        y_smooth = np.zeros((len(time_grid), n_variables))

        for j in range(n_variables):
            spl = UnivariateSpline._from_tck((
                self.knots_[j],
                self.coefficients_[j],
                self.degree
            ))
            y_smooth[:, j] = spl(time_grid)

        return y_smooth

    def fit_transform(
        self,
        y: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform."""
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_timepoints = y.shape[0]

        if time_grid is None:
            time_grid = np.linspace(0, 1, n_timepoints)

        self.fit(y, time_grid)
        return self.transform(time_grid)


def savitzky_golay_smooth(
    X: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for smoothing.

    Savitzky-Golayフィルタによるスムージング
    高速で端点の処理が適切

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timepoints, n_variables)
        Data to smooth
    window_length : int, optional
        Length of filter window (must be odd, default: 11)
    polyorder : int, optional
        Order of polynomial (default: 3)

    Returns
    -------
    X_smooth : ndarray
        Smoothed data
    """
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, :, np.newaxis]

    n_samples, n_timepoints, n_variables = X.shape

    # Ensure window_length is odd and valid
    window_length = min(window_length, n_timepoints)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, polyorder + 2)

    X_smooth = np.zeros_like(X)

    for i in range(n_samples):
        for j in range(n_variables):
            X_smooth[i, :, j] = savgol_filter(
                X[i, :, j],
                window_length=window_length,
                polyorder=polyorder,
                mode='interp'
            )

    return X_smooth
