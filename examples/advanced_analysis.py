"""
Advanced MFPCA analysis example.

高度なMFPCA解析の例
- Cross-validation for component selection
- Bootstrap confidence intervals
- Time series forecasting with MFPCA
- Clustering based on functional PCA scores
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mfpca import MFPCA, FunctionalDataPreprocessor, MFPCAVisualizer
from mfpca.utils import (
    generate_synthetic_mfpca_data,
    bootstrap_mfpca,
    cross_validate_mfpca,
    select_n_components
)


def example_cross_validation():
    """
    Example: Cross-validation for selecting optimal number of components.

    交差検証による最適な成分数の選択例
    """
    print("\n" + "=" * 80)
    print("Example 1: Cross-Validation for Component Selection")
    print("=" * 80)

    # Generate data
    X, _, _, time_grid = generate_synthetic_mfpca_data(
        n_samples=100,
        n_timepoints=50,
        n_variables=3,
        n_components=3,
        noise_std=0.15,
        seed=42
    )

    print("\nPerforming 5-fold cross-validation...")

    # Cross-validation
    cv_results = cross_validate_mfpca(
        X,
        n_folds=5,
        n_components_range=list(range(1, 11)),
        smoothing=True,
        seed=42
    )

    print(f"\nBest number of components: {cv_results['best_n_components']}")
    print("\nMean reconstruction errors:")
    for n_comp, error in cv_results['mean_errors'].items():
        print(f"   {n_comp} components: {error:.4f}")

    # Plot CV results
    fig, ax = plt.subplots(figsize=(10, 6))
    n_comps = sorted(cv_results['mean_errors'].keys())
    errors = [cv_results['mean_errors'][n] for n in n_comps]

    ax.plot(n_comps, errors, 'bo-', linewidth=2, markersize=8)
    ax.axvline(
        cv_results['best_n_components'],
        color='r',
        linestyle='--',
        label=f"Best: {cv_results['best_n_components']} components"
    )
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Mean Reconstruction Error (RMSE)', fontsize=12)
    ax.set_title('Cross-Validation: Component Selection', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('cv_component_selection.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'cv_component_selection.png'")


def example_bootstrap_confidence_intervals():
    """
    Example: Bootstrap confidence intervals for eigenvalues.

    ブートストラップによる固有値の信頼区間
    """
    print("\n" + "=" * 80)
    print("Example 2: Bootstrap Confidence Intervals")
    print("=" * 80)

    # Generate data
    X, _, _, time_grid = generate_synthetic_mfpca_data(
        n_samples=80,
        n_timepoints=50,
        n_variables=2,
        n_components=3,
        noise_std=0.1,
        seed=42
    )

    print("\nPerforming bootstrap analysis (100 iterations)...")

    # Bootstrap analysis
    bootstrap_results = bootstrap_mfpca(
        X,
        n_bootstrap=100,
        confidence_level=0.95,
        n_components=3,
        smoothing=True,
        seed=42
    )

    print(f"\nBootstrap completed: {bootstrap_results['n_bootstrap']} successful iterations")
    print(f"Confidence level: {bootstrap_results['confidence_level']}")

    print("\nEigenvalues with 95% confidence intervals:")
    for i in range(len(bootstrap_results['original_eigenvalues'])):
        orig = bootstrap_results['original_eigenvalues'][i]
        mean = bootstrap_results['eigenvalues_mean'][i]
        lower = bootstrap_results['eigenvalues_ci_lower'][i]
        upper = bootstrap_results['eigenvalues_ci_upper'][i]

        print(f"   Component {i+1}:")
        print(f"      Original: {orig:.4f}")
        print(f"      Mean: {mean:.4f}")
        print(f"      95% CI: [{lower:.4f}, {upper:.4f}]")

    # Plot bootstrap results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Eigenvalues with CI
    n_comp = len(bootstrap_results['original_eigenvalues'])
    components = np.arange(1, n_comp + 1)

    ax = axes[0]
    ax.errorbar(
        components,
        bootstrap_results['eigenvalues_mean'],
        yerr=[
            bootstrap_results['eigenvalues_mean'] - bootstrap_results['eigenvalues_ci_lower'],
            bootstrap_results['eigenvalues_ci_upper'] - bootstrap_results['eigenvalues_mean']
        ],
        fmt='o-',
        linewidth=2,
        markersize=8,
        capsize=5,
        label='Bootstrap mean ± 95% CI'
    )
    ax.plot(
        components,
        bootstrap_results['original_eigenvalues'],
        'rs--',
        linewidth=2,
        markersize=8,
        label='Original'
    )
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Eigenvalues with Bootstrap CI', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Variance explained with CI
    ax = axes[1]
    ax.errorbar(
        components,
        bootstrap_results['variance_explained_mean'],
        yerr=[
            bootstrap_results['variance_explained_mean'] - bootstrap_results['variance_explained_ci_lower'],
            bootstrap_results['variance_explained_ci_upper'] - bootstrap_results['variance_explained_mean']
        ],
        fmt='o-',
        linewidth=2,
        markersize=8,
        capsize=5,
        label='Bootstrap mean ± 95% CI'
    )
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title('Variance Explained with Bootstrap CI', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('bootstrap_confidence_intervals.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'bootstrap_confidence_intervals.png'")


def example_clustering_with_mfpca():
    """
    Example: Clustering multivariate functional data using MFPCA scores.

    MFPCAスコアを使用した多変量関数型データのクラスタリング
    """
    print("\n" + "=" * 80)
    print("Example 3: Clustering with MFPCA Scores")
    print("=" * 80)

    # Generate data with three distinct groups
    np.random.seed(42)

    n_samples_per_group = 40
    n_timepoints = 60
    n_variables = 2
    time_grid = np.linspace(0, 1, n_timepoints)

    X_list = []
    true_labels = []

    # Group 1: High frequency oscillations
    for i in range(n_samples_per_group):
        x = np.zeros((n_timepoints, n_variables))
        for j in range(n_variables):
            freq = 5 + np.random.randn() * 0.5
            phase = np.random.uniform(0, 2 * np.pi)
            x[:, j] = np.sin(2 * np.pi * freq * time_grid + phase)
        X_list.append(x)
        true_labels.append(0)

    # Group 2: Low frequency oscillations
    for i in range(n_samples_per_group):
        x = np.zeros((n_timepoints, n_variables))
        for j in range(n_variables):
            freq = 1 + np.random.randn() * 0.2
            phase = np.random.uniform(0, 2 * np.pi)
            x[:, j] = np.sin(2 * np.pi * freq * time_grid + phase)
        X_list.append(x)
        true_labels.append(1)

    # Group 3: Linear trends
    for i in range(n_samples_per_group):
        x = np.zeros((n_timepoints, n_variables))
        for j in range(n_variables):
            slope = np.random.uniform(1, 3)
            intercept = np.random.uniform(-1, 1)
            x[:, j] = slope * time_grid + intercept
        X_list.append(x)
        true_labels.append(2)

    X = np.array(X_list)
    true_labels = np.array(true_labels)

    # Add noise
    X += np.random.randn(*X.shape) * 0.1

    print(f"\nData shape: {X.shape}")
    print(f"Number of groups: 3")
    print(f"Samples per group: {n_samples_per_group}")

    # Preprocess
    preprocessor = FunctionalDataPreprocessor(normalize=True, normalize_method='zscore')
    X_preprocessed = preprocessor.fit_transform(X, time_grid)

    # Fit MFPCA
    print("\nFitting MFPCA...")
    mfpca = MFPCA(n_components=5, smoothing=True)
    scores = mfpca.fit_transform(X_preprocessed, time_grid)

    print(f"Variance explained by first 3 components: {np.sum(mfpca.variance_explained_ratio_[:3]):.2%}")

    # Clustering on MFPCA scores
    print("\nPerforming K-means clustering (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(scores[:, :3])

    # Compute clustering accuracy (with label matching)
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    accuracy = cm[row_ind, col_ind].sum() / len(true_labels)

    print(f"\nClustering accuracy: {accuracy:.2%}")

    # Visualization
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Sample curves from each group
    ax1 = plt.subplot(2, 3, 1)
    for group in range(3):
        idx = np.where(true_labels == group)[0][0]
        for j in range(n_variables):
            ax1.plot(
                time_grid,
                X[idx, :, j],
                label=f'Group {group+1}, Var {j+1}' if j == 0 else None,
                alpha=0.7
            )
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Sample Curves', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Plot 2: Scree plot
    ax2 = plt.subplot(2, 3, 2)
    visualizer = MFPCAVisualizer(mfpca)
    visualizer.plot_scree(n_components=5, ax=ax2)

    # Plot 3: Scores colored by true labels
    ax3 = plt.subplot(2, 3, 3)
    colors = ['red', 'blue', 'green']
    for group in range(3):
        mask = true_labels == group
        ax3.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=colors[group],
            label=f'True Group {group+1}',
            alpha=0.6,
            s=50
        )
    ax3.set_xlabel(f'PC1 ({mfpca.variance_explained_ratio_[0]:.1%})', fontsize=11)
    ax3.set_ylabel(f'PC2 ({mfpca.variance_explained_ratio_[1]:.1%})', fontsize=11)
    ax3.set_title('Scores (True Labels)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Plot 4: Scores colored by predicted labels
    ax4 = plt.subplot(2, 3, 4)
    for group in range(3):
        mask = predicted_labels == group
        ax4.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=colors[group],
            label=f'Cluster {group+1}',
            alpha=0.6,
            s=50
        )
    ax4.set_xlabel(f'PC1 ({mfpca.variance_explained_ratio_[0]:.1%})', fontsize=11)
    ax4.set_ylabel(f'PC2 ({mfpca.variance_explained_ratio_[1]:.1%})', fontsize=11)
    ax4.set_title(f'Scores (Predicted, Acc={accuracy:.1%})', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # Plot 5: First two eigenfunctions
    ax5 = plt.subplot(2, 3, 5)
    for j in range(n_variables):
        ax5.plot(time_grid, mfpca.eigenfunctions_[0, :, j], label=f'Var {j+1}')
    ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Time', fontsize=11)
    ax5.set_ylabel('PC1', fontsize=11)
    ax5.set_title('First Eigenfunction', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    for j in range(n_variables):
        ax6.plot(time_grid, mfpca.eigenfunctions_[1, :, j], label=f'Var {j+1}')
    ax6.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax6.set_xlabel('Time', fontsize=11)
    ax6.set_ylabel('PC2', fontsize=11)
    ax6.set_title('Second Eigenfunction', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('clustering_with_mfpca.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'clustering_with_mfpca.png'")


def example_anomaly_detection():
    """
    Example: Anomaly detection using reconstruction error.

    再構成誤差を使用した異常検知
    """
    print("\n" + "=" * 80)
    print("Example 4: Anomaly Detection with MFPCA")
    print("=" * 80)

    # Generate normal data
    X_normal, _, _, time_grid = generate_synthetic_mfpca_data(
        n_samples=100,
        n_timepoints=50,
        n_variables=2,
        n_components=3,
        noise_std=0.1,
        seed=42
    )

    # Generate anomalous data (different pattern)
    np.random.seed(123)
    n_anomalies = 10
    X_anomalies = np.random.randn(n_anomalies, 50, 2) * 2.0

    # Combine data
    X = np.vstack([X_normal, X_anomalies])
    is_anomaly = np.array([False] * 100 + [True] * 10)

    print(f"\nNormal samples: {np.sum(~is_anomaly)}")
    print(f"Anomalous samples: {np.sum(is_anomaly)}")

    # Fit MFPCA on all data (in practice, fit only on normal data)
    print("\nFitting MFPCA...")
    mfpca = MFPCA(n_components=3, smoothing=True)
    mfpca.fit(X[:100], time_grid)  # Fit only on normal data

    # Compute reconstruction errors
    print("\nComputing reconstruction errors...")
    reconstruction_errors = np.zeros(len(X))

    for i in range(len(X)):
        error = mfpca.get_reconstruction_error(X[i:i+1], n_components=3)
        reconstruction_errors[i] = error

    # Set threshold (e.g., 95th percentile of normal data)
    threshold = np.percentile(reconstruction_errors[:100], 95)

    # Detect anomalies
    detected_anomalies = reconstruction_errors > threshold

    # Compute metrics
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(is_anomaly, detected_anomalies)
    recall = recall_score(is_anomaly, detected_anomalies)
    f1 = f1_score(is_anomaly, detected_anomalies)

    print(f"\nAnomaly detection results:")
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1 Score: {f1:.2%}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Reconstruction errors
    ax = axes[0, 0]
    ax.scatter(
        np.arange(len(X))[~is_anomaly],
        reconstruction_errors[~is_anomaly],
        c='blue',
        label='Normal',
        alpha=0.6,
        s=50
    )
    ax.scatter(
        np.arange(len(X))[is_anomaly],
        reconstruction_errors[is_anomaly],
        c='red',
        label='Anomaly',
        alpha=0.6,
        s=50
    )
    ax.axhline(threshold, color='k', linestyle='--', label=f'Threshold ({threshold:.3f})')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Reconstruction Error (RMSE)', fontsize=11)
    ax.set_title('Reconstruction Errors', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Histogram of errors
    ax = axes[0, 1]
    ax.hist(
        reconstruction_errors[~is_anomaly],
        bins=20,
        alpha=0.6,
        label='Normal',
        color='blue'
    )
    ax.hist(
        reconstruction_errors[is_anomaly],
        bins=20,
        alpha=0.6,
        label='Anomaly',
        color='red'
    )
    ax.axvline(threshold, color='k', linestyle='--', label='Threshold')
    ax.set_xlabel('Reconstruction Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Example normal sample
    ax = axes[1, 0]
    sample_idx = 0
    X_recon = mfpca.inverse_transform(mfpca.transform(X[sample_idx:sample_idx+1]), n_components=3)
    for j in range(2):
        ax.plot(time_grid, X[sample_idx, :, j], label=f'Original Var{j+1}', alpha=0.7)
        ax.plot(time_grid, X_recon[0, :, j], '--', label=f'Reconstructed Var{j+1}', alpha=0.7)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Normal Sample (Error={reconstruction_errors[sample_idx]:.3f})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 4: Example anomalous sample
    ax = axes[1, 1]
    sample_idx = 105
    X_recon = mfpca.inverse_transform(mfpca.transform(X[sample_idx:sample_idx+1]), n_components=3)
    for j in range(2):
        ax.plot(time_grid, X[sample_idx, :, j], label=f'Original Var{j+1}', alpha=0.7)
        ax.plot(time_grid, X_recon[0, :, j], '--', label=f'Reconstructed Var{j+1}', alpha=0.7)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Anomalous Sample (Error={reconstruction_errors[sample_idx]:.3f})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('anomaly_detection.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'anomaly_detection.png'")


def main():
    """Run all advanced examples."""
    print("\n" + "=" * 80)
    print("MFPCA - Advanced Analysis Examples")
    print("=" * 80)

    # Run examples
    example_cross_validation()
    example_bootstrap_confidence_intervals()
    example_clustering_with_mfpca()
    example_anomaly_detection()

    print("\n" + "=" * 80)
    print("All advanced examples completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
