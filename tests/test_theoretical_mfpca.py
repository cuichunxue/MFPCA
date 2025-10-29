"""
Tests for theoretically rigorous MFPCA implementation.

理論的に厳密なMFPCA実装のテスト
"""

import numpy as np
import pytest
from mfpca import TheoreticalMFPCA, BSplineBasis, FourierBasis, create_basis
from mfpca.utils import generate_synthetic_mfpca_data


class TestBasisSystems:
    """Test basis function systems."""

    def test_bspline_basis_evaluation(self):
        """Test B-spline basis evaluation."""
        n_basis = 10
        basis = BSplineBasis(n_basis=n_basis, degree=3, domain=(0, 1))

        t = np.linspace(0, 1, 50)
        B = basis.evaluate(t)

        # Check shape
        assert B.shape == (50, n_basis)

        # Check partition of unity (for certain degree/knots combinations)
        # sum_j B_j(t) should be close to constant
        row_sums = np.sum(B, axis=1)
        assert np.std(row_sums) < 0.5  # Should be approximately constant

        # Check non-negativity
        assert np.all(B >= -1e-10)  # Allow small numerical errors

    def test_bspline_gram_matrix(self):
        """Test B-spline Gram matrix computation."""
        n_basis = 8
        basis = BSplineBasis(n_basis=n_basis, degree=3, domain=(0, 1))

        t = np.linspace(0, 1, 100)
        G = basis.gram_matrix(t)

        # Check shape
        assert G.shape == (n_basis, n_basis)

        # Check symmetry
        np.testing.assert_allclose(G, G.T, atol=1e-10)

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(G)
        assert np.all(eigenvalues > -1e-10)

    def test_fourier_basis_evaluation(self):
        """Test Fourier basis evaluation."""
        n_basis = 11  # 1 constant + 5 pairs
        basis = FourierBasis(n_basis=n_basis, domain=(0, 1))

        t = np.linspace(0, 1, 50)
        B = basis.evaluate(t)

        # Check shape
        assert B.shape == (50, n_basis)

        # Check orthonormality approximately
        # ∫ B_i(t) B_j(t) dt should be δ_ij
        G = basis.gram_matrix(t)
        I = np.eye(n_basis)
        np.testing.assert_allclose(G, I, atol=0.05)  # Numerical integration error

    def test_fourier_orthonormality(self):
        """Test Fourier basis orthonormality."""
        n_basis = 7
        basis = FourierBasis(n_basis=n_basis, domain=(0, 1), period=1.0)

        t = np.linspace(0, 1, 200)
        B = basis.evaluate(t)

        # Compute Gram matrix
        G = basis.gram_matrix(t)

        # Should be approximately identity
        I = np.eye(n_basis)
        np.testing.assert_allclose(G, I, atol=0.01)

    def test_create_basis_factory(self):
        """Test basis factory function."""
        # B-spline
        basis_bsp = create_basis('bspline', n_basis=10, degree=3)
        assert isinstance(basis_bsp, BSplineBasis)

        # Fourier
        basis_four = create_basis('fourier', n_basis=9)
        assert isinstance(basis_four, FourierBasis)

        # Invalid
        with pytest.raises(ValueError):
            create_basis('invalid', n_basis=10)


class TestTheoreticalMFPCA:
    """Test TheoreticalMFPCA class."""

    def test_basic_fit_bspline(self):
        """Test basic fitting with B-spline basis."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            n_components=2,
            seed=42
        )

        mfpca = TheoreticalMFPCA(
            n_components=2,
            n_basis=10,
            basis_type='bspline',
            smoothing=False
        )
        mfpca.fit(X, time_grid)

        # Check fitted attributes
        assert mfpca.coefficients_ is not None
        assert mfpca.eigenvalues_ is not None
        assert mfpca.scores_ is not None
        assert mfpca.eigenvector_coefficients_ is not None

        # Check shapes
        assert mfpca.coefficients_.shape == (50, 20)  # 2 variables * 10 basis
        assert mfpca.eigenvalues_.shape == (2,)
        assert mfpca.scores_.shape == (50, 2)
        assert mfpca.eigenvector_coefficients_.shape == (20, 2)

    def test_basic_fit_fourier(self):
        """Test basic fitting with Fourier basis."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            seed=42
        )

        mfpca = TheoreticalMFPCA(
            n_components=2,
            n_basis=11,
            basis_type='fourier',
            smoothing=False
        )
        mfpca.fit(X, time_grid)

        assert mfpca.eigenvalues_ is not None
        assert mfpca.scores_.shape == (50, 2)

    def test_transform(self):
        """Test transform method."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            seed=42
        )

        mfpca = TheoreticalMFPCA(n_components=2, n_basis=10, smoothing=False)
        mfpca.fit(X, time_grid)

        # Transform new data
        X_new = X[:10]
        scores_new = mfpca.transform(X_new)

        assert scores_new.shape == (10, 2)
        # Should be close to original scores for same data
        np.testing.assert_allclose(scores_new, mfpca.scores_[:10], rtol=1e-4)

    def test_inverse_transform(self):
        """Test reconstruction."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            n_components=2,
            noise_std=0.01,
            seed=42
        )

        mfpca = TheoreticalMFPCA(n_components=2, n_basis=15, smoothing=True)
        scores = mfpca.fit_transform(X, time_grid)

        # Reconstruct
        X_recon = mfpca.inverse_transform(scores, n_components=2)

        assert X_recon.shape == X.shape

        # Should be reasonable reconstruction
        rmse = np.sqrt(np.mean((X - X_recon) ** 2))
        assert rmse < 1.0  # With smoothing and low noise, should be good

    def test_eigenfunction_evaluation(self):
        """Test eigenfunction evaluation."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            seed=42
        )

        mfpca = TheoreticalMFPCA(n_components=3, n_basis=10)
        mfpca.fit(X, time_grid)

        # Evaluate eigenfunctions on original grid
        eigenfuncs = mfpca.get_eigenfunctions()

        assert eigenfuncs.shape == (3, 30, 2)

        # Evaluate on different grid
        time_grid_fine = np.linspace(0, 1, 100)
        eigenfuncs_fine = mfpca.get_eigenfunctions(time_grid_fine)

        assert eigenfuncs_fine.shape == (3, 100, 2)

    def test_mean_function(self):
        """Test mean function evaluation."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            seed=42
        )

        mfpca = TheoreticalMFPCA(n_basis=10, center=True)
        mfpca.fit(X, time_grid)

        mean_func = mfpca.get_mean_function()

        assert mean_func.shape == (30, 2)

        # Should be close to empirical mean
        empirical_mean = np.mean(X, axis=0)
        # Allow some difference due to smoothing
        np.testing.assert_allclose(mean_func, empirical_mean, rtol=0.5)

    def test_eigenfunction_orthogonality(self):
        """Test orthogonality of eigenfunctions (approximately)."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=100,
            n_timepoints=50,
            n_variables=2,
            seed=42
        )

        mfpca = TheoreticalMFPCA(n_components=3, n_basis=15, smoothing=False)
        mfpca.fit(X, time_grid)

        eigenfuncs = mfpca.get_eigenfunctions()

        # Compute inner products
        # <φ_i, φ_j> = ∫ φ_i(t)^T φ_j(t) dt
        n_comp = eigenfuncs.shape[0]
        inner_products = np.zeros((n_comp, n_comp))

        for i in range(n_comp):
            for j in range(n_comp):
                # Inner product via trapezoidal integration
                integrand = np.sum(eigenfuncs[i] * eigenfuncs[j], axis=1)
                inner_products[i, j] = np.trapz(integrand, time_grid)

        # Should be approximately identity (orthonormal)
        I = np.eye(n_comp)
        np.testing.assert_allclose(inner_products, I, atol=0.1)

    def test_variance_explained(self):
        """Test variance explained calculation."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            seed=42
        )

        mfpca = TheoreticalMFPCA(smoothing=False)
        mfpca.fit(X, time_grid)

        var_exp = mfpca.explained_variance_ratio()

        # Should sum to <= 1
        assert np.sum(var_exp) <= 1.0 + 1e-10

        # Should be positive and decreasing
        assert np.all(var_exp >= 0)
        assert np.all(np.diff(mfpca.eigenvalues_) <= 1e-10)

    def test_reconstruction_error_decreases(self):
        """Test that reconstruction error decreases with more components."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            seed=42
        )

        mfpca = TheoreticalMFPCA(n_basis=15, smoothing=False)
        mfpca.fit(X, time_grid)

        errors = []
        for n_comp in [1, 2, 3, 5]:
            error = mfpca.get_reconstruction_error(X, n_comp)
            errors.append(error)

        # Errors should be non-increasing
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-6

    def test_smoothing_penalty(self):
        """Test smoothing with penalty."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            noise_std=0.3,
            seed=42
        )

        # Without smoothing
        mfpca_no_smooth = TheoreticalMFPCA(
            n_components=3,
            n_basis=15,
            smoothing=False
        )
        mfpca_no_smooth.fit(X, time_grid)

        # With smoothing
        mfpca_smooth = TheoreticalMFPCA(
            n_components=3,
            n_basis=15,
            smoothing=True,
            smoothing_penalty=0.1
        )
        mfpca_smooth.fit(X, time_grid)

        # Both should fit
        assert mfpca_no_smooth.eigenvalues_ is not None
        assert mfpca_smooth.eigenvalues_ is not None

        # Smoothed eigenfunctions should be smoother (less high-frequency variation)
        eigenfuncs_no_smooth = mfpca_no_smooth.get_eigenfunctions()
        eigenfuncs_smooth = mfpca_smooth.get_eigenfunctions()

        # Measure smoothness via second derivative
        def compute_roughness(funcs):
            d2 = np.gradient(np.gradient(funcs, axis=1), axis=1)
            return np.mean(d2 ** 2)

        roughness_no_smooth = compute_roughness(eigenfuncs_no_smooth)
        roughness_smooth = compute_roughness(eigenfuncs_smooth)

        # Smoothed should have lower roughness
        assert roughness_smooth < roughness_no_smooth

    def test_consistency_with_true_components(self):
        """Test that recovered components are similar to true components."""
        # Generate data with known structure
        X, true_scores, true_eigenfuncs, time_grid = generate_synthetic_mfpca_data(
            n_samples=200,
            n_timepoints=50,
            n_variables=2,
            n_components=2,
            noise_std=0.05,
            seed=42
        )

        mfpca = TheoreticalMFPCA(
            n_components=2,
            n_basis=20,
            smoothing=True
        )
        mfpca.fit(X, time_grid)

        # Recovered scores should be correlated with true scores
        recovered_scores = mfpca.scores_

        for k in range(2):
            # Compute correlation (allowing for sign flip)
            corr = np.corrcoef(true_scores[:, k], recovered_scores[:, k])[0, 1]
            assert abs(corr) > 0.7  # Should have strong correlation

    def test_univariate_case(self):
        """Test with univariate data."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=1,
            seed=42
        )

        mfpca = TheoreticalMFPCA(n_components=2, n_basis=10)
        mfpca.fit(X, time_grid)

        assert mfpca.eigenvector_coefficients_.shape == (10, 2)
        assert mfpca.scores_.shape == (50, 2)

        eigenfuncs = mfpca.get_eigenfunctions()
        assert eigenfuncs.shape == (2, 30, 1)

    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned data."""
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=50,
            n_timepoints=30,
            n_variables=2,
            seed=42
        )

        # Add tiny noise to make ill-conditioned
        X += np.random.randn(*X.shape) * 1e-8

        mfpca = TheoreticalMFPCA(
            n_components=2,
            n_basis=10,
            regularization=1e-10
        )

        # Should not raise errors
        mfpca.fit(X, time_grid)
        assert mfpca.eigenvalues_ is not None


class TestIntegrationTheoretical:
    """Integration tests for TheoreticalMFPCA."""

    def test_full_pipeline(self):
        """Test complete pipeline."""
        # Generate data
        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=100,
            n_timepoints=50,
            n_variables=3,
            seed=42
        )

        # Fit
        mfpca = TheoreticalMFPCA(
            n_components=3,
            n_basis=15,
            basis_type='bspline',
            smoothing=True
        )
        scores = mfpca.fit_transform(X, time_grid)

        # Transform new data
        X_new = X[:10]
        scores_new = mfpca.transform(X_new)

        # Reconstruct
        X_recon = mfpca.inverse_transform(scores_new)

        # Get eigenfunctions and mean
        eigenfuncs = mfpca.get_eigenfunctions()
        mean_func = mfpca.get_mean_function()

        # Get variance explained
        var_exp = mfpca.explained_variance_ratio()

        # All should work
        assert scores.shape == (100, 3)
        assert scores_new.shape == (10, 3)
        assert X_recon.shape == (10, 50, 3)
        assert eigenfuncs.shape == (3, 50, 3)
        assert mean_func.shape == (50, 3)
        assert len(var_exp) == 3

    def test_comparison_with_fast_implementation(self):
        """Compare TheoreticalMFPCA with fast MFPCA."""
        from mfpca import MFPCA

        X, _, _, time_grid = generate_synthetic_mfpca_data(
            n_samples=100,
            n_timepoints=50,
            n_variables=2,
            seed=42
        )

        # Theoretical implementation
        mfpca_theory = TheoreticalMFPCA(
            n_components=3,
            n_basis=20,
            smoothing=False,
            center=True
        )
        mfpca_theory.fit(X, time_grid)

        # Fast implementation
        mfpca_fast = MFPCA(
            n_components=3,
            smoothing=False,
            center=True
        )
        mfpca_fast.fit(X, time_grid)

        # Both should give similar results
        # Eigenvalues should be similar
        eigenval_theory = mfpca_theory.eigenvalues_
        eigenval_fast = mfpca_fast.eigenvalues_

        # Allow some difference due to different approaches
        for i in range(3):
            ratio = eigenval_theory[i] / eigenval_fast[i]
            assert 0.5 < ratio < 2.0  # Should be in same ballpark

        # Variance explained should be similar
        var_theory = mfpca_theory.explained_variance_ratio()
        var_fast = mfpca_fast.variance_explained_ratio_

        np.testing.assert_allclose(var_theory, var_fast, rtol=0.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
