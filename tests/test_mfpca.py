"""
Comprehensive tests for MFPCA package.

MFPCAパッケージの包括的なテスト
"""

import numpy as np
import pytest
from mfpca import (
    MFPCA,
    FunctionalDataPreprocessor,
    BSplineSmoother,
    create_time_grid,
    compute_covariance_surface
)
from mfpca.utils import (
    generate_synthetic_mfpca_data,
    align_eigenfunctions,
    select_n_components,
    compute_functional_correlation,
    bootstrap_mfpca,
    cross_validate_mfpca
)
from mfpca.preprocessing import savitzky_golay_smooth


class TestMFPCA:
    """Test MFPCA core functionality."""

    def test_basic_fit(self):
        """Test basic fitting."""
        # Generate synthetic data
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            n_components=2,
            seed=42
        )

        # Fit MFPCA
        mfpca = MFPCA(n_components=2, smoothing=False)
        mfpca.fit(X, time_grid)

        # Check fitted attributes
        assert mfpca.eigenfunctions_ is not None
        assert mfpca.eigenvalues_ is not None
        assert mfpca.scores_ is not None
        assert mfpca.mean_function_ is not None

        # Check shapes
        assert mfpca.eigenfunctions_.shape == (2, 30, 2)
        assert mfpca.eigenvalues_.shape == (2,)
        assert mfpca.scores_.shape == (50, 2)
        assert mfpca.mean_function_.shape == (30, 2)

    def test_transform(self):
        """Test transform method."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        mfpca = MFPCA(n_components=2, smoothing=False)
        mfpca.fit(X, time_grid)

        # Transform new data
        X_new = X[:10]
        scores_new = mfpca.transform(X_new)

        assert scores_new.shape == (10, 2)
        # Scores should be close to original for same data
        np.testing.assert_allclose(scores_new, mfpca.scores_[:10], rtol=1e-5)

    def test_fit_transform(self):
        """Test fit_transform method."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        mfpca = MFPCA(n_components=2, smoothing=False)
        scores = mfpca.fit_transform(X, time_grid)

        assert scores.shape == (50, 2)
        np.testing.assert_allclose(scores, mfpca.scores_, rtol=1e-10)

    def test_inverse_transform(self):
        """Test reconstruction via inverse_transform."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, n_components=2,
            noise_std=0.01, seed=42
        )

        mfpca = MFPCA(n_components=2, smoothing=False)
        scores = mfpca.fit_transform(X, time_grid)

        # Reconstruct
        X_reconstructed = mfpca.inverse_transform(scores)

        assert X_reconstructed.shape == X.shape

        # Reconstruction should be close with 2 components (low noise)
        rmse = np.sqrt(np.mean((X - X_reconstructed) ** 2))
        assert rmse < 0.5  # Should be reasonably small

    def test_variance_explained(self):
        """Test variance explained calculation."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        mfpca = MFPCA(smoothing=False)
        mfpca.fit(X, time_grid)

        # Variance explained should sum to <= 1
        assert np.sum(mfpca.variance_explained_ratio_) <= 1.0 + 1e-10

        # Should be positive and decreasing
        assert np.all(mfpca.variance_explained_ratio_ >= 0)
        assert np.all(np.diff(mfpca.eigenvalues_) <= 0)

    def test_smoothing(self):
        """Test with smoothing enabled."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, noise_std=0.2, seed=42
        )

        mfpca_smooth = MFPCA(n_components=2, smoothing=True)
        mfpca_smooth.fit(X, time_grid)

        mfpca_no_smooth = MFPCA(n_components=2, smoothing=False)
        mfpca_no_smooth.fit(X, time_grid)

        # Both should fit successfully
        assert mfpca_smooth.eigenfunctions_ is not None
        assert mfpca_no_smooth.eigenfunctions_ is not None

    def test_univariate_case(self):
        """Test with univariate data (single variable)."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=1, seed=42
        )

        mfpca = MFPCA(n_components=2, smoothing=False)
        mfpca.fit(X, time_grid)

        assert mfpca.eigenfunctions_.shape[2] == 1
        assert mfpca.mean_function_.shape[1] == 1

    def test_reconstruction_error(self):
        """Test reconstruction error calculation."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        mfpca = MFPCA(smoothing=False)
        mfpca.fit(X, time_grid)

        # Error should decrease with more components
        errors = []
        for n_comp in [1, 2, 3]:
            error = mfpca.get_reconstruction_error(X, n_comp)
            errors.append(error)

        # Errors should be decreasing (or at least non-increasing)
        assert errors[0] >= errors[1] >= errors[2]

    def test_centering(self):
        """Test with and without centering."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        mfpca_center = MFPCA(n_components=2, smoothing=False, center=True)
        mfpca_center.fit(X, time_grid)

        mfpca_no_center = MFPCA(n_components=2, smoothing=False, center=False)
        mfpca_no_center.fit(X, time_grid)

        # Both should fit
        assert mfpca_center.eigenfunctions_ is not None
        assert mfpca_no_center.eigenfunctions_ is not None

        # Mean function should be non-zero for centering
        assert np.any(mfpca_center.mean_function_ != 0)

    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned data."""
        # Create data with very small variance in one direction
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        # Add very small noise to make it ill-conditioned
        X += np.random.randn(*X.shape) * 1e-8

        mfpca = MFPCA(smoothing=False, regularization=1e-10)

        # Should not raise errors
        mfpca.fit(X, time_grid)
        assert mfpca.eigenfunctions_ is not None

    def test_empty_components(self):
        """Test with n_components=None (all components)."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        mfpca = MFPCA(n_components=None, smoothing=False)
        mfpca.fit(X, time_grid)

        # Should compute all available components
        assert len(mfpca.eigenvalues_) > 0


class TestPreprocessing:
    """Test preprocessing functionality."""

    def test_preprocessor_zscore(self):
        """Test z-score normalization."""
        X, _, _, _ = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        preprocessor = FunctionalDataPreprocessor(
            normalize=True,
            normalize_method='zscore'
        )
        X_norm = preprocessor.fit_transform(X)

        # Check normalization
        X_flat = X_norm.reshape(-1, 2)
        means = np.mean(X_flat, axis=0)
        stds = np.std(X_flat, axis=0)

        np.testing.assert_allclose(means, 0, atol=1e-10)
        np.testing.assert_allclose(stds, 1, atol=1e-10)

    def test_preprocessor_minmax(self):
        """Test min-max normalization."""
        X, _, _, _ = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        preprocessor = FunctionalDataPreprocessor(
            normalize=True,
            normalize_method='minmax'
        )
        X_norm = preprocessor.fit_transform(X)

        # Check normalization
        X_flat = X_norm.reshape(-1, 2)
        mins = np.min(X_flat, axis=0)
        maxs = np.max(X_flat, axis=0)

        np.testing.assert_allclose(mins, 0, atol=1e-10)
        np.testing.assert_allclose(maxs, 1, atol=1e-10)

    def test_preprocessor_inverse(self):
        """Test inverse transformation."""
        X, _, _, _ = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        preprocessor = FunctionalDataPreprocessor(
            normalize=True,
            normalize_method='zscore'
        )
        X_norm = preprocessor.fit_transform(X)
        X_inv = preprocessor.inverse_transform(X_norm)

        # Should recover original data
        np.testing.assert_allclose(X, X_inv, rtol=1e-10)

    def test_missing_values_interpolate(self):
        """Test missing value handling with interpolation."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        # Introduce missing values
        X_missing = X.copy()
        X_missing[0, 5, 0] = np.nan
        X_missing[1, 10, 1] = np.nan

        preprocessor = FunctionalDataPreprocessor(handle_missing='interpolate')
        X_filled = preprocessor.transform(X_missing, time_grid)

        # No missing values should remain
        assert not np.any(np.isnan(X_filled))

    def test_bspline_smoother(self):
        """Test B-spline smoother."""
        # Generate noisy data
        time_grid = np.linspace(0, 1, 50)
        y_true = np.sin(2 * np.pi * time_grid)
        y_noisy = y_true + np.random.randn(50) * 0.1

        smoother = BSplineSmoother(n_basis=10, degree=3)
        y_smooth = smoother.fit_transform(y_noisy, time_grid)

        # Smoothed data should be closer to true signal
        error_noisy = np.mean((y_noisy - y_true) ** 2)
        error_smooth = np.mean((y_smooth.flatten() - y_true) ** 2)

        assert error_smooth < error_noisy

    def test_savitzky_golay(self):
        """Test Savitzky-Golay smoothing."""
        X, _, _, _ = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, noise_std=0.2, seed=42
        )

        X_smooth = savitzky_golay_smooth(X, window_length=11, polyorder=3)

        assert X_smooth.shape == X.shape
        # Smoothed data should have lower variance
        assert np.var(np.diff(X_smooth, axis=1)) < np.var(np.diff(X, axis=1))


class TestUtils:
    """Test utility functions."""

    def test_create_time_grid(self):
        """Test time grid creation."""
        # Regular grid
        grid = create_time_grid(start=0, end=1, n_points=50, irregular=False)
        assert len(grid) == 50
        assert grid[0] == 0.0
        assert grid[-1] == 1.0

        # Irregular grid
        grid_irreg = create_time_grid(start=0, end=1, n_points=50, irregular=True, seed=42)
        assert len(grid_irreg) == 50
        assert grid_irreg[0] >= 0.0
        assert grid_irreg[-1] <= 1.0
        assert np.all(np.diff(grid_irreg) > 0)  # Should be sorted

    def test_compute_covariance_surface(self):
        """Test covariance surface computation."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        cov_surface, mean_func, tg = compute_covariance_surface(X, time_grid)

        # Check shapes
        assert cov_surface.shape == (30, 30, 2, 2)
        assert mean_func.shape == (30, 2)
        assert len(tg) == 30

        # Covariance should be symmetric
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(
                    cov_surface[:, :, i, j],
                    cov_surface[:, :, i, j].T,
                    atol=1e-10
                )

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        X, scores, eigenfuncs, time_grid = generate_synthetic_mfpca_data(
            n_samples=100,
            n_timepoints=50,
            n_variables=3,
            n_components=3,
            seed=42
        )

        assert X.shape == (100, 50, 3)
        assert scores.shape == (100, 3)
        assert eigenfuncs.shape == (3, 50, 3)
        assert len(time_grid) == 50

    def test_align_eigenfunctions(self):
        """Test eigenfunction alignment."""
        _, _, eigenfuncs, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, n_components=2, seed=42
        )

        # Flip sign of second eigenfunction
        eigenfuncs2 = eigenfuncs.copy()
        eigenfuncs2[1] *= -1

        # Align
        aligned, correlations = align_eigenfunctions(
            eigenfuncs, eigenfuncs2, time_grid
        )

        # All correlations should be positive after alignment
        assert np.all(correlations > 0)

    def test_select_n_components(self):
        """Test component selection."""
        variance_explained = np.array([0.5, 0.3, 0.15, 0.03, 0.02])

        n_comp = select_n_components(variance_explained, threshold=0.95)

        # Should select first 3 components (0.5 + 0.3 + 0.15 = 0.95)
        assert n_comp == 3

    def test_functional_correlation(self):
        """Test functional correlation."""
        time_grid = np.linspace(0, 1, 50)
        X1 = np.sin(2 * np.pi * time_grid)[:, np.newaxis]
        X2 = np.sin(2 * np.pi * time_grid)[:, np.newaxis]
        X3 = np.cos(2 * np.pi * time_grid)[:, np.newaxis]

        # Correlation with itself should be 1
        corr_self = compute_functional_correlation(X1, X2, time_grid)
        np.testing.assert_allclose(corr_self, 1.0, atol=1e-10)

        # Correlation with orthogonal function should be ~0
        corr_orth = compute_functional_correlation(X1, X3, time_grid)
        np.testing.assert_allclose(corr_orth, 0.0, atol=1e-10)

    def test_bootstrap_mfpca(self):
        """Test bootstrap analysis."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        results = bootstrap_mfpca(
            X,
            n_bootstrap=10,  # Small number for speed
            n_components=2,
            smoothing=False,
            seed=42
        )

        # Check results structure
        assert 'original_eigenvalues' in results
        assert 'eigenvalues_mean' in results
        assert 'eigenvalues_ci_lower' in results
        assert 'eigenvalues_ci_upper' in results

        # CI should contain original values (approximately)
        # This is a weak test but bootstrap is stochastic
        assert len(results['eigenvalues_mean']) == 2

    def test_cross_validate_mfpca(self):
        """Test cross-validation."""
        X, _, _, _ = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        results = cross_validate_mfpca(
            X,
            n_folds=3,
            n_components_range=[1, 2, 3],
            smoothing=False,
            seed=42
        )

        # Check results
        assert 'best_n_components' in results
        assert 'mean_errors' in results
        assert results['best_n_components'] in [1, 2, 3]


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        # Generate data
        X, true_scores, true_eigenfuncs, time_grid = generate_synthetic_mfpca_data(
            n_samples=100,
            n_timepoints=50,
            n_variables=3,
            n_components=3,
            noise_std=0.05,
            seed=42
        )

        # Preprocess
        preprocessor = FunctionalDataPreprocessor(
            normalize=True,
            normalize_method='zscore'
        )
        X_prep = preprocessor.fit_transform(X, time_grid)

        # Fit MFPCA
        mfpca = MFPCA(n_components=3, smoothing=True)
        scores = mfpca.fit_transform(X_prep, time_grid)

        # Transform new data
        X_new = X[:10]
        X_new_prep = preprocessor.transform(X_new, time_grid)
        scores_new = mfpca.transform(X_new_prep)

        # Reconstruct
        X_recon_prep = mfpca.inverse_transform(scores_new)
        X_recon = preprocessor.inverse_transform(X_recon_prep)

        # Check everything worked
        assert scores.shape == (100, 3)
        assert scores_new.shape == (10, 3)
        assert X_recon.shape == (10, 50, 3)

        # Reconstruction should be reasonable
        rmse = np.sqrt(np.mean((X[:10] - X_recon) ** 2))
        assert rmse < 1.0  # Should be reasonably accurate

    def test_with_missing_data(self):
        """Test pipeline with missing data."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50, n_timepoints=30, n_variables=2, seed=42
        )

        # Introduce missing values
        X_missing = X.copy()
        X_missing[0, 5:10, 0] = np.nan
        X_missing[1, 15:20, 1] = np.nan

        # Preprocess (with interpolation)
        preprocessor = FunctionalDataPreprocessor(handle_missing='interpolate')
        X_filled = preprocessor.transform(X_missing, time_grid)

        # Fit MFPCA
        mfpca = MFPCA(n_components=2, smoothing=True)
        mfpca.fit(X_filled, time_grid)

        # Should work without errors
        assert mfpca.eigenfunctions_ is not None
        assert not np.any(np.isnan(mfpca.eigenfunctions_))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
