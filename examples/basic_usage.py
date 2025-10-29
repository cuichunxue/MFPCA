"""
Basic usage example of MFPCA.

MFPCA の基本的な使用例
"""

import numpy as np
import matplotlib.pyplot as plt
from mfpca import MFPCA, FunctionalDataPreprocessor, MFPCAVisualizer
from mfpca.utils import generate_synthetic_mfpca_data


def main():
    """
    Basic MFPCA usage example.

    基本的なMFPCAの使用例
    """
    print("=" * 80)
    print("MFPCA - Basic Usage Example")
    print("=" * 80)

    # ==========================================================================
    # 1. Generate synthetic multivariate functional data
    # ==========================================================================
    print("\n1. Generating synthetic data...")

    n_samples = 100
    n_timepoints = 100
    n_variables = 3
    n_true_components = 3

    X, true_scores, true_eigenfuncs, time_grid = generate_synthetic_mfpca_data(
        n_samples=n_samples,
        n_timepoints=n_timepoints,
        n_variables=n_variables,
        n_components=n_true_components,
        noise_std=0.1,
        seed=42
    )

    print(f"   Data shape: {X.shape}")
    print(f"   Time points: {n_timepoints}")
    print(f"   Number of variables: {n_variables}")
    print(f"   Number of samples: {n_samples}")

    # ==========================================================================
    # 2. Preprocess the data (optional but recommended)
    # ==========================================================================
    print("\n2. Preprocessing data...")

    preprocessor = FunctionalDataPreprocessor(
        handle_missing='interpolate',
        normalize=True,
        normalize_method='zscore'
    )

    X_preprocessed = preprocessor.fit_transform(X, time_grid)
    print("   Data normalized using z-score normalization")

    # ==========================================================================
    # 3. Fit MFPCA model
    # ==========================================================================
    print("\n3. Fitting MFPCA model...")

    mfpca = MFPCA(
        n_components=5,  # Extract first 5 components
        smoothing=True,  # Apply smoothing
        center=True      # Center the data
    )

    mfpca.fit(X_preprocessed, time_grid)

    print(f"   Number of components extracted: {len(mfpca.eigenvalues_)}")
    print(f"   Eigenvalues: {mfpca.eigenvalues_}")
    print(f"   Variance explained: {mfpca.variance_explained_ratio_}")
    print(f"   Cumulative variance: {mfpca.explained_variance_cumsum()}")

    # ==========================================================================
    # 4. Transform data to principal component scores
    # ==========================================================================
    print("\n4. Computing principal component scores...")

    scores = mfpca.transform(X_preprocessed)
    print(f"   Scores shape: {scores.shape}")
    print(f"   First sample scores: {scores[0]}")

    # ==========================================================================
    # 5. Reconstruct data from principal components
    # ==========================================================================
    print("\n5. Reconstructing data...")

    # Reconstruct using different numbers of components
    for n_comp in [1, 2, 3, 5]:
        X_recon_prep = mfpca.inverse_transform(scores[:, :n_comp], n_comp)
        X_recon = preprocessor.inverse_transform(X_recon_prep)

        rmse = np.sqrt(np.mean((X - X_recon) ** 2))
        print(f"   RMSE with {n_comp} components: {rmse:.4f}")

    # ==========================================================================
    # 6. Visualize results
    # ==========================================================================
    print("\n6. Creating visualizations...")

    visualizer = MFPCAVisualizer(mfpca, figsize=(12, 8))

    # Create summary plot
    fig = visualizer.plot_summary(
        X_preprocessed,
        n_components=3,
        variable_names=['Temperature', 'Pressure', 'Humidity']
    )
    plt.savefig('mfpca_summary.png', dpi=150, bbox_inches='tight')
    print("   Summary plot saved to 'mfpca_summary.png'")

    # Plot scree plot
    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer.plot_scree(n_components=5, ax=ax)
    plt.savefig('scree_plot.png', dpi=150, bbox_inches='tight')
    print("   Scree plot saved to 'scree_plot.png'")

    # Plot eigenfunctions
    fig = visualizer.plot_eigenfunctions(
        n_components=3,
        variable_names=['Temperature', 'Pressure', 'Humidity'],
        separate_variables=True
    )
    plt.savefig('eigenfunctions.png', dpi=150, bbox_inches='tight')
    print("   Eigenfunctions plot saved to 'eigenfunctions.png'")

    # Plot scores
    fig, ax = plt.subplots(figsize=(10, 8))
    visualizer.plot_scores(components=(0, 1), ax=ax)
    plt.savefig('scores_plot.png', dpi=150, bbox_inches='tight')
    print("   Scores plot saved to 'scores_plot.png'")

    # Plot reconstruction
    fig = visualizer.plot_reconstruction(
        X_preprocessed,
        sample_idx=0,
        n_components=3,
        variable_names=['Temperature', 'Pressure', 'Humidity']
    )
    plt.savefig('reconstruction.png', dpi=150, bbox_inches='tight')
    print("   Reconstruction plot saved to 'reconstruction.png'")

    # ==========================================================================
    # 7. Analyze results
    # ==========================================================================
    print("\n7. Analysis summary:")
    print("   " + "=" * 76)

    cumvar = mfpca.explained_variance_cumsum()
    for i in range(min(5, len(mfpca.eigenvalues_))):
        print(f"   Component {i+1}:")
        print(f"      Eigenvalue: {mfpca.eigenvalues_[i]:.4f}")
        print(f"      Variance explained: {mfpca.variance_explained_ratio_[i]:.4f}")
        print(f"      Cumulative variance: {cumvar[i]:.4f}")

    # Find number of components for 95% variance
    n_comp_95 = np.searchsorted(cumvar, 0.95) + 1
    print(f"\n   Components needed for 95% variance: {n_comp_95}")

    print("\n" + "=" * 80)
    print("Basic usage example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
