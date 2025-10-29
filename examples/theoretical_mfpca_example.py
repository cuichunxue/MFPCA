"""
Example demonstrating the theoretically rigorous MFPCA implementation.

理論的に厳密なMFPCA実装のデモンストレーション

This example shows:
1. Basis expansion methods (B-spline and Fourier)
2. Theoretical properties (orthonormality, variance decomposition)
3. Comparison with fast implementation
4. Mathematical rigor validation
"""

import numpy as np
import matplotlib.pyplot as plt
from mfpca import TheoreticalMFPCA, MFPCA, MFPCAVisualizer
from mfpca.basis import BSplineBasis, FourierBasis
from mfpca.utils import generate_synthetic_mfpca_data


def example_1_basis_functions():
    """
    Example 1: Visualizing different basis function systems.

    例1: 異なる基底関数系の可視化
    """
    print("\n" + "=" * 80)
    print("Example 1: Basis Function Systems")
    print("=" * 80)

    time_grid = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # B-spline basis
    print("\n1. B-spline Basis (degree 3, 10 basis functions)")
    bspline_basis = BSplineBasis(n_basis=10, degree=3, domain=(0, 1))
    B_bsp = bspline_basis.evaluate(time_grid)

    ax = axes[0, 0]
    for i in range(B_bsp.shape[1]):
        ax.plot(time_grid, B_bsp[:, i], label=f'B_{i+1}' if i < 3 else '', alpha=0.7)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Basis Value', fontsize=11)
    ax.set_title('B-spline Basis Functions (degree=3)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Partition of unity
    ax = axes[0, 1]
    row_sum = np.sum(B_bsp, axis=1)
    ax.plot(time_grid, row_sum, 'b-', linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Sum of Basis Functions', fontsize=11)
    ax.set_title('B-spline: Partition of Unity', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    print(f"   Partition of unity: mean={np.mean(row_sum):.4f}, std={np.std(row_sum):.4f}")

    # Fourier basis
    print("\n2. Fourier Basis (11 basis functions)")
    fourier_basis = FourierBasis(n_basis=11, domain=(0, 1))
    B_four = fourier_basis.evaluate(time_grid)

    ax = axes[1, 0]
    for i in range(min(7, B_four.shape[1])):
        label = ['Constant', 'sin(2π)', 'cos(2π)', 'sin(4π)', 'cos(4π)', 'sin(6π)', 'cos(6π)'][i]
        ax.plot(time_grid, B_four[:, i], label=label, alpha=0.7)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Basis Value', fontsize=11)
    ax.set_title('Fourier Basis Functions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Gram matrix for Fourier (should be identity)
    ax = axes[1, 1]
    G_four = fourier_basis.gram_matrix(time_grid)
    im = ax.imshow(G_four, cmap='RdBu_r', vmin=-0.1, vmax=1.1)
    ax.set_xlabel('Basis Index', fontsize=11)
    ax.set_ylabel('Basis Index', fontsize=11)
    ax.set_title('Fourier Gram Matrix (≈ Identity)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Check orthonormality
    I = np.eye(G_four.shape[0])
    error = np.linalg.norm(G_four - I, 'fro')
    print(f"   Orthonormality error: ||G - I||_F = {error:.6f}")

    plt.tight_layout()
    plt.savefig('basis_functions_comparison.png', dpi=150, bbox_inches='tight')
    print("\n   Plot saved to 'basis_functions_comparison.png'")


def example_2_theoretical_properties():
    """
    Example 2: Verifying theoretical properties of MFPCA.

    例2: MFPCAの理論的性質の検証
    """
    print("\n" + "=" * 80)
    print("Example 2: Theoretical Properties Validation")
    print("=" * 80)

    # Generate data
    print("\nGenerating synthetic data...")
    X, true_scores, true_eigenfuncs, time_grid = generate_synthetic_mfpca_data(
        n_samples=150,
        n_timepoints=80,
        n_variables=2,
        n_components=3,
        noise_std=0.1,
        seed=42
    )

    # Fit TheoreticalMFPCA
    print("\nFitting TheoreticalMFPCA (B-spline basis)...")
    mfpca = TheoreticalMFPCA(
        n_components=5,
        n_basis=20,
        basis_type='bspline',
        basis_degree=3,
        smoothing=True,
        center=True
    )
    mfpca.fit(X, time_grid)

    # Property 1: Orthonormality of eigenfunctions
    print("\n1. Verifying orthonormality of eigenfunctions:")
    eigenfuncs = mfpca.get_eigenfunctions()
    n_comp = eigenfuncs.shape[0]

    inner_products = np.zeros((n_comp, n_comp))
    for i in range(n_comp):
        for j in range(n_comp):
            integrand = np.sum(eigenfuncs[i] * eigenfuncs[j], axis=1)
            inner_products[i, j] = np.trapz(integrand, time_grid)

    print(f"   Inner product matrix <φ_i, φ_j>:")
    print(f"{inner_products}")
    print(f"   ||Inner products - I||_F = {np.linalg.norm(inner_products - np.eye(n_comp), 'fro'):.6f}")

    # Property 2: Variance decomposition
    print("\n2. Variance decomposition:")
    var_exp = mfpca.explained_variance_ratio()
    cumvar = np.cumsum(var_exp)
    for i in range(n_comp):
        print(f"   PC{i+1}: λ={mfpca.eigenvalues_[i]:.4f}, "
              f"var_exp={var_exp[i]:.4f}, cumvar={cumvar[i]:.4f}")

    # Property 3: Uncorrelated scores
    print("\n3. Uncorrelatedness of scores:")
    scores = mfpca.scores_
    score_corr = np.corrcoef(scores.T)
    np.fill_diagonal(score_corr, 0)  # Ignore diagonal
    max_off_diag_corr = np.max(np.abs(score_corr))
    print(f"   Maximum off-diagonal correlation: {max_off_diag_corr:.6f}")
    print(f"   (Should be close to 0 for large samples)")

    # Property 4: Karhunen-Loève expansion
    print("\n4. Karhunen-Loève expansion: X(t) = μ(t) + Σ ξ_k φ_k(t)")
    mean_func = mfpca.get_mean_function()
    sample_idx = 0

    # Manual reconstruction
    X_manual = mean_func.copy()
    for k in range(n_comp):
        X_manual += scores[sample_idx, k] * eigenfuncs[k]

    # Using inverse_transform
    X_recon = mfpca.inverse_transform(scores[sample_idx:sample_idx+1], n_components=n_comp)

    diff = np.linalg.norm(X_manual - X_recon[0])
    print(f"   ||X_manual - X_recon|| = {diff:.6f}")
    print(f"   (Should be ≈ 0)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Eigenvalues
    ax = axes[0, 0]
    ax.bar(range(1, n_comp + 1), mfpca.eigenvalues_, alpha=0.6)
    ax.plot(range(1, n_comp + 1), np.cumsum(var_exp), 'ro-', linewidth=2, label='Cumulative')
    ax.set_xlabel('Component', fontsize=11)
    ax.set_ylabel('Eigenvalue / Cumulative Variance', fontsize=11)
    ax.set_title('Eigenvalue Decomposition', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Eigenfunctions
    ax = axes[0, 1]
    for k in range(min(3, n_comp)):
        for j in range(2):
            ax.plot(time_grid, eigenfuncs[k, :, j],
                   label=f'PC{k+1}, Var{j+1}', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Eigenfunction Value', fontsize=11)
    ax.set_title('First 3 Eigenfunctions', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 3: Inner product matrix
    ax = axes[1, 0]
    im = ax.imshow(inner_products, cmap='RdBu_r', vmin=-0.2, vmax=1.2)
    ax.set_xlabel('Component j', fontsize=11)
    ax.set_ylabel('Component i', fontsize=11)
    ax.set_title('Inner Products <φ_i, φ_j> (≈ I)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Plot 4: Score correlation matrix
    ax = axes[1, 1]
    im = ax.imshow(np.abs(score_corr), cmap='Reds', vmin=0, vmax=0.5)
    ax.set_xlabel('Component j', fontsize=11)
    ax.set_ylabel('Component i', fontsize=11)
    ax.set_title('|Correlation(ξ_i, ξ_j)| (≈ 0)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('theoretical_properties_validation.png', dpi=150, bbox_inches='tight')
    print("\n   Plot saved to 'theoretical_properties_validation.png'")


def example_3_comparison_implementations():
    """
    Example 3: Comparing theoretical and fast implementations.

    例3: 理論的実装と高速実装の比較
    """
    print("\n" + "=" * 80)
    print("Example 3: Comparison of Implementations")
    print("=" * 80)

    # Generate data
    X, _, _, time_grid = generate_synthetic_mfpca_data(
        n_samples=100,
        n_timepoints=60,
        n_variables=3,
        n_components=3,
        seed=42
    )

    print("\n1. Fitting TheoreticalMFPCA (basis expansion)...")
    import time
    t0 = time.time()
    mfpca_theory = TheoreticalMFPCA(
        n_components=5,
        n_basis=15,
        basis_type='bspline',
        smoothing=True
    )
    mfpca_theory.fit(X, time_grid)
    t_theory = time.time() - t0
    print(f"   Time: {t_theory:.3f}s")

    print("\n2. Fitting MFPCA (fast implementation)...")
    t0 = time.time()
    mfpca_fast = MFPCA(
        n_components=5,
        smoothing=True
    )
    mfpca_fast.fit(X, time_grid)
    t_fast = time.time() - t0
    print(f"   Time: {t_fast:.3f}s")
    print(f"   Speed ratio: {t_theory / t_fast:.2f}x")

    # Compare results
    print("\n3. Comparing results:")
    print("\n   Eigenvalues:")
    print(f"   {'Component':<12} {'Theoretical':<15} {'Fast':<15} {'Ratio':<10}")
    print("   " + "-" * 52)
    for i in range(5):
        ratio = mfpca_theory.eigenvalues_[i] / mfpca_fast.eigenvalues_[i]
        print(f"   {i+1:<12} {mfpca_theory.eigenvalues_[i]:<15.6f} "
              f"{mfpca_fast.eigenvalues_[i]:<15.6f} {ratio:<10.4f}")

    print("\n   Variance explained:")
    var_theory = mfpca_theory.explained_variance_ratio()
    var_fast = mfpca_fast.variance_explained_ratio_
    for i in range(5):
        print(f"   PC{i+1}: Theory={var_theory[i]:.4f}, Fast={var_fast[i]:.4f}")

    # Compare reconstruction errors
    print("\n4. Reconstruction errors:")
    for n_comp in [1, 2, 3, 5]:
        error_theory = mfpca_theory.get_reconstruction_error(X, n_comp)
        error_fast = mfpca_fast.get_reconstruction_error(X, n_comp)
        print(f"   {n_comp} components: Theory={error_theory:.6f}, Fast={error_fast:.6f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Scree plots
    ax = axes[0, 0]
    ax.plot(range(1, 6), mfpca_theory.eigenvalues_, 'bo-', label='Theoretical', linewidth=2)
    ax.plot(range(1, 6), mfpca_fast.eigenvalues_, 'rs--', label='Fast', linewidth=2)
    ax.set_xlabel('Component', fontsize=11)
    ax.set_ylabel('Eigenvalue', fontsize=11)
    ax.set_title('Eigenvalues Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Variance explained
    ax = axes[0, 1]
    cumvar_theory = np.cumsum(var_theory)
    cumvar_fast = np.cumsum(var_fast)
    ax.plot(range(1, 6), cumvar_theory, 'bo-', label='Theoretical', linewidth=2)
    ax.plot(range(1, 6), cumvar_fast, 'rs--', label='Fast', linewidth=2)
    ax.axhline(0.95, color='k', linestyle='--', alpha=0.3, label='95%')
    ax.set_xlabel('Component', fontsize=11)
    ax.set_ylabel('Cumulative Variance Explained', fontsize=11)
    ax.set_title('Cumulative Variance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Score comparison
    ax = axes[0, 2]
    scores_theory = mfpca_theory.scores_
    scores_fast = mfpca_fast.scores_
    # Compare first two PCs
    corr01_theory = np.corrcoef(scores_theory[:, 0], scores_fast[:, 0])[0, 1]
    corr01_fast = np.corrcoef(scores_theory[:, 1], scores_fast[:, 1])[0, 1]
    ax.scatter(scores_theory[:, 0], scores_fast[:, 0], alpha=0.5, label=f'PC1 (r={abs(corr01_theory):.3f})')
    ax.scatter(scores_theory[:, 1], scores_fast[:, 1], alpha=0.5, label=f'PC2 (r={abs(corr01_fast):.3f})')
    ax.plot([-3, 3], [-3, 3], 'k--', alpha=0.3)
    ax.set_xlabel('Theoretical Scores', fontsize=11)
    ax.set_ylabel('Fast Scores', fontsize=11)
    ax.set_title('Score Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Eigenfunction comparison (first PC, first variable)
    eigenfuncs_theory = mfpca_theory.get_eigenfunctions()
    eigenfuncs_fast = mfpca_fast.eigenfunctions_

    for k in range(3):
        ax = axes[1, k]
        for j in range(min(2, X.shape[2])):
            # Align signs
            sign = np.sign(np.corrcoef(
                eigenfuncs_theory[k, :, j],
                eigenfuncs_fast[k, :, j]
            )[0, 1])
            ax.plot(time_grid, eigenfuncs_theory[k, :, j],
                   label=f'Theory Var{j+1}', linewidth=2, alpha=0.7)
            ax.plot(time_grid, sign * eigenfuncs_fast[k, :, j], '--',
                   label=f'Fast Var{j+1}', linewidth=2, alpha=0.7)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Eigenfunction Value', fontsize=11)
        ax.set_title(f'Eigenfunction {k+1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('implementation_comparison.png', dpi=150, bbox_inches='tight')
    print("\n   Plot saved to 'implementation_comparison.png'")


def example_4_smoothing_penalty():
    """
    Example 4: Effect of smoothing penalty.

    例4: スムージングペナルティの効果
    """
    print("\n" + "=" * 80)
    print("Example 4: Effect of Smoothing Penalty")
    print("=" * 80)

    # Generate noisy data
    X, _, _, time_grid = generate_synthetic_mfpca_data(
        n_samples=80,
        n_timepoints=50,
        n_variables=2,
        noise_std=0.3,
        seed=42
    )

    penalties = [0, 0.001, 0.01, 0.1]
    results = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, penalty in enumerate(penalties):
        print(f"\nFitting with smoothing penalty λ = {penalty}...")

        mfpca = TheoreticalMFPCA(
            n_components=3,
            n_basis=20,
            basis_type='bspline',
            smoothing=(penalty > 0),
            smoothing_penalty=penalty if penalty > 0 else None
        )
        mfpca.fit(X, time_grid)

        eigenfuncs = mfpca.get_eigenfunctions()

        # Compute roughness (integrated squared second derivative)
        d2 = np.gradient(np.gradient(eigenfuncs, axis=1), axis=1)
        roughness = np.mean(np.trapz(d2 ** 2, time_grid, axis=1))

        print(f"   Roughness: {roughness:.6f}")
        print(f"   Top 3 eigenvalues: {mfpca.eigenvalues_[:3]}")

        results.append({
            'penalty': penalty,
            'eigenfuncs': eigenfuncs,
            'roughness': roughness,
            'eigenvalues': mfpca.eigenvalues_
        })

        # Plot first eigenfunction
        ax = axes[idx // 2, idx % 2]
        for j in range(2):
            ax.plot(time_grid, eigenfuncs[0, :, j], label=f'Var {j+1}', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Eigenfunction Value', fontsize=11)
        ax.set_title(f'λ = {penalty}, Roughness = {roughness:.4f}',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('smoothing_penalty_effect.png', dpi=150, bbox_inches='tight')
    print("\n   Plot saved to 'smoothing_penalty_effect.png'")

    # Summary
    print("\n" + "-" * 80)
    print("Summary: Effect of increasing smoothing penalty")
    print("-" * 80)
    print(f"{'Penalty':<12} {'Roughness':<15} {'1st Eigenval':<15}")
    print("-" * 42)
    for r in results:
        print(f"{r['penalty']:<12.3f} {r['roughness']:<15.6f} {r['eigenvalues'][0]:<15.6f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("THEORETICAL MFPCA - Complete Demonstration")
    print("論文に基づいた理論的に厳密なMFPCA実装")
    print("=" * 80)

    example_1_basis_functions()
    example_2_theoretical_properties()
    example_3_comparison_implementations()
    example_4_smoothing_penalty()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
