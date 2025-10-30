"""
SOM (Self-Organizing Maps) example with MFPCA integration.

このスクリプトは以下を実演します：
1. MFPCAによる多変量時系列データの次元削減
2. SOMによるMFPCAスコアのクラスタリング
3. R言語somと同等の可視化（特にcomponent planes）
4. MFPCAとSOMの統合ワークフロー

R言語のkohonen::somと同等以上の機能を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt

# MFPCAとSOMのインポート
from mfpca import (
    MFPCA,
    TheoreticalMFPCA,
    SOM,
    SOMVisualizer,
    FunctionalDataPreprocessor,
    MFPCAVisualizer
)
from mfpca.utils import generate_synthetic_mfpca_data


def example1_basic_som():
    """
    例1: 基本的なSOMの使用

    シンプルな多次元データでSOMの基本機能をデモンストレーション
    """
    print("=" * 70)
    print("Example 1: Basic SOM Usage")
    print("=" * 70)

    # データ生成（3つのクラスター）
    np.random.seed(42)
    cluster1 = np.random.randn(50, 4) + np.array([3, 3, 0, 0])
    cluster2 = np.random.randn(50, 4) + np.array([-3, -3, 0, 0])
    cluster3 = np.random.randn(50, 4) + np.array([0, 0, 3, -3])
    X = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0] * 50 + [1] * 50 + [2] * 50)

    print(f"Data shape: {X.shape}")
    print(f"Number of clusters: 3")

    # SOM学習
    print("\nTraining SOM...")
    som = SOM(
        grid_shape=(10, 10),
        topology='rectangular',
        n_iterations=1000,
        random_state=42
    )
    som.fit(X, verbose=True)

    # 可視化
    print("\nCreating visualizations...")
    visualizer = SOMVisualizer(som)

    # Component planes（R言語somの重要な機能）
    fig1 = visualizer.plot_component_planes(
        X,
        variable_names=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
    )
    plt.savefig('som_component_planes.png', dpi=150, bbox_inches='tight')
    print("Saved: som_component_planes.png")

    # 総合サマリー
    fig2 = visualizer.plot_summary(X, labels=labels)
    plt.savefig('som_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: som_summary.png")

    plt.show()


def example2_mfpca_with_som():
    """
    例2: MFPCAとSOMの統合

    多変量時系列データにMFPCAで次元削減を行い、
    得られたスコアをSOMでクラスタリング・可視化
    """
    print("\n" + "=" * 70)
    print("Example 2: MFPCA + SOM Integration")
    print("=" * 70)

    # 多変量時系列データの生成
    print("Generating multivariate time series data...")
    X, true_scores, true_eigenfuncs, time_grid = generate_synthetic_mfpca_data(
        n_samples=150,
        n_timepoints=50,
        n_variables=5,
        n_components=3,
        noise_std=0.1,
        seed=42
    )

    print(f"Data shape: {X.shape}")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Time points: {X.shape[1]}")
    print(f"  - Variables: {X.shape[2]}")

    # ステップ1: MFPCAで次元削減
    print("\nStep 1: Performing MFPCA...")
    mfpca = MFPCA(
        n_components=5,
        smoothing=True,
        center=True
    )
    mfpca_scores = mfpca.fit_transform(X, time_grid)

    print(f"MFPCA scores shape: {mfpca_scores.shape}")
    print(f"Variance explained: {mfpca.variance_explained_ratio_[:5]}")
    print(f"Cumulative variance: {mfpca.explained_variance_cumsum()[:5]}")

    # ステップ2: MFPCAスコアをSOMでクラスタリング
    print("\nStep 2: Clustering MFPCA scores with SOM...")
    som = SOM(
        grid_shape=(8, 8),
        topology='rectangular',
        neighborhood_function='gaussian',
        n_iterations=1000,
        random_state=42
    )
    som.fit(mfpca_scores, verbose=True)

    # BMUを取得
    bmu_indices = som.predict(mfpca_scores)
    print(f"\nBMU indices shape: {bmu_indices.shape}")
    print(f"Number of unique BMUs: {len(np.unique(bmu_indices))}")

    # ステップ3: 可視化
    print("\nStep 3: Visualizing results...")

    # 3-1: MFPCAの結果
    mfpca_vis = MFPCAVisualizer(mfpca)
    fig1 = mfpca_vis.plot_summary(X, n_components=5)
    plt.savefig('mfpca_som_mfpca_results.png', dpi=150, bbox_inches='tight')
    print("Saved: mfpca_som_mfpca_results.png")

    # 3-2: SOMのcomponent planes（どのMFPCA成分が重要か）
    som_vis = SOMVisualizer(som)
    fig2 = som_vis.plot_component_planes(
        mfpca_scores,
        variable_names=[f'MFPCA PC{i+1}' for i in range(5)]
    )
    plt.savefig('mfpca_som_component_planes.png', dpi=150, bbox_inches='tight')
    print("Saved: mfpca_som_component_planes.png")

    # 3-3: SOMのサマリー
    fig3 = som_vis.plot_summary(mfpca_scores)
    plt.savefig('mfpca_som_clustering.png', dpi=150, bbox_inches='tight')
    print("Saved: mfpca_som_clustering.png")

    # 3-4: BMUごとのサンプル分析
    fig4, axes = plt.subplots(2, 2, figsize=(14, 12))

    # U-matrix
    u_matrix = som.get_u_matrix()
    im1 = axes[0, 0].imshow(u_matrix, cmap='bone_r')
    axes[0, 0].set_title('U-Matrix\n(Cluster Boundaries)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])

    # Node counts
    counts = som.get_node_counts(mfpca_scores)
    im2 = axes[0, 1].imshow(counts, cmap='YlOrRd')
    axes[0, 1].set_title('Node Activation Counts', fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1])

    # 最も代表的なBMUのサンプルを表示
    axes[1, 0].set_title('Representative Samples from Top BMUs', fontweight='bold')
    top_bmus = np.argsort(np.bincount(bmu_indices))[-5:][::-1]
    for i, bmu in enumerate(top_bmus):
        samples_in_bmu = np.where(bmu_indices == bmu)[0]
        if len(samples_in_bmu) > 0:
            # 代表サンプル（BMU内の中央値）
            rep_sample = samples_in_bmu[len(samples_in_bmu) // 2]
            axes[1, 0].plot(
                time_grid,
                X[rep_sample, :, 0],  # 最初の変数のみプロット
                label=f'BMU {bmu} (n={len(samples_in_bmu)})',
                alpha=0.7
            )
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Variable 1 Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 分散説明率
    axes[1, 1].bar(range(1, 6), mfpca.variance_explained_ratio_[:5])
    axes[1, 1].set_title('MFPCA Variance Explained', fontweight='bold')
    axes[1, 1].set_xlabel('Principal Component')
    axes[1, 1].set_ylabel('Variance Ratio')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('mfpca_som_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: mfpca_som_analysis.png")

    plt.show()

    # 結果のサマリー
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"MFPCA reduced {X.shape[1] * X.shape[2]} dimensions to {mfpca_scores.shape[1]}")
    print(f"SOM organized {X.shape[0]} samples into {8*8} grid nodes")
    print(f"Active nodes: {len(np.unique(bmu_indices))} / 64")
    print(f"Top 5 variance explained: {np.sum(mfpca.variance_explained_ratio_[:5]):.2%}")


def example3_theoretical_mfpca_som():
    """
    例3: TheoreticalMFPCAとSOMの統合

    理論的に厳密なMFPCA実装とSOMの組み合わせ
    """
    print("\n" + "=" * 70)
    print("Example 3: TheoreticalMFPCA + SOM")
    print("=" * 70)

    # データ生成
    print("Generating data...")
    X, _, _, time_grid = generate_synthetic_mfpca_data(
        n_samples=100,
        n_timepoints=40,
        n_variables=3,
        n_components=3,
        noise_std=0.1,
        seed=42
    )

    print(f"Data shape: {X.shape}")

    # TheoreticalMFPCA
    print("\nPerforming TheoreticalMFPCA...")
    mfpca = TheoreticalMFPCA(
        n_components=4,
        n_basis=15,
        basis_type='bspline',
        smoothing=True,
        smoothing_penalty=0.01
    )
    scores = mfpca.fit_transform(X, time_grid)

    print(f"Scores shape: {scores.shape}")

    # SOM
    print("\nTraining SOM...")
    som = SOM(
        grid_shape=(6, 6),
        topology='hexagonal',  # 六角形グリッド（R言語と同じ）
        n_iterations=800,
        random_state=42
    )
    som.fit(scores, verbose=True)

    # 可視化
    print("\nVisualizing...")
    visualizer = SOMVisualizer(som)

    # Component planes
    fig = visualizer.plot_component_planes(
        scores,
        variable_names=[f'Theoretical PC{i+1}' for i in range(4)]
    )
    plt.savefig('theoretical_mfpca_som.png', dpi=150, bbox_inches='tight')
    print("Saved: theoretical_mfpca_som.png")

    plt.show()


def example4_compare_with_r_som():
    """
    例4: R言語のsomとの比較デモ

    R言語のkohonen::somで得られるような可視化を実演
    """
    print("\n" + "=" * 70)
    print("Example 4: R-style SOM Visualization")
    print("=" * 70)

    # Irisライクなデータ（3クラス、4変数）
    np.random.seed(42)
    n_per_class = 50

    # クラス1: Setosa
    setosa = np.random.randn(n_per_class, 4) * 0.3 + np.array([5.0, 3.4, 1.5, 0.2])

    # クラス2: Versicolor
    versicolor = np.random.randn(n_per_class, 4) * 0.5 + np.array([6.0, 2.8, 4.5, 1.4])

    # クラス3: Virginica
    virginica = np.random.randn(n_per_class, 4) * 0.6 + np.array([6.5, 3.0, 5.5, 2.0])

    X = np.vstack([setosa, versicolor, virginica])
    labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    print(f"Data shape: {X.shape}")
    print("Classes: Setosa (0), Versicolor (1), Virginica (2)")

    # SOM学習
    print("\nTraining SOM (R kohonen style)...")
    som = SOM(
        grid_shape=(7, 7),
        topology='hexagonal',  # R言語somのデフォルト
        neighborhood_function='gaussian',
        learning_rate_init=0.05,
        learning_rate_final=0.01,
        n_iterations=1000,
        random_state=42
    )
    som.fit(X, verbose=True)

    # R言語somスタイルの可視化
    visualizer = SOMVisualizer(som)

    # 1. Component planes（R言語somの主要機能）
    print("\nCreating R-style component planes...")
    fig1 = visualizer.plot_component_planes(
        X,
        variable_names=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'],
        cmap='RdYlBu_r'  # R言語と似たカラースキーム
    )
    fig1.suptitle('SOM Component Planes (R kohonen::som style)',
                  fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('r_style_component_planes.png', dpi=150, bbox_inches='tight')
    print("Saved: r_style_component_planes.png")

    # 2. 総合ビュー
    fig2 = visualizer.plot_summary(X, labels=labels)
    plt.savefig('r_style_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: r_style_summary.png")

    # 3. データマッピング（クラスごとに色分け）
    fig3 = visualizer.plot_data_mapping(X, labels=labels)
    plt.savefig('r_style_mapping.png', dpi=150, bbox_inches='tight')
    print("Saved: r_style_mapping.png")

    plt.show()

    # 各クラスの分布を分析
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS")
    print("=" * 70)

    bmu_indices = som.predict(X)

    for cls in range(3):
        class_name = ['Setosa', 'Versicolor', 'Virginica'][cls]
        class_bmus = bmu_indices[labels == cls]
        unique_bmus = np.unique(class_bmus)
        print(f"\n{class_name}:")
        print(f"  - Uses {len(unique_bmus)} unique nodes")
        print(f"  - Most common BMUs: {np.bincount(class_bmus).argsort()[-3:][::-1]}")


if __name__ == '__main__':
    # すべての例を実行
    print("\n" + "=" * 70)
    print("SOM EXAMPLES - R kohonen::som equivalent in Python")
    print("=" * 70)

    # 例1: 基本的なSOM
    example1_basic_som()

    # 例2: MFPCAとSOMの統合（推奨！）
    example2_mfpca_with_som()

    # 例3: TheoreticalMFPCAとSOM
    example3_theoretical_mfpca_som()

    # 例4: R言語スタイルの可視化
    example4_compare_with_r_som()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nKey Features:")
    print("  ✓ Component planes visualization (R kohonen::som equivalent)")
    print("  ✓ U-matrix for cluster boundary detection")
    print("  ✓ Hexagonal and rectangular topologies")
    print("  ✓ Integration with MFPCA for time series analysis")
    print("  ✓ Multiple neighborhood functions")
    print("  ✓ Training progress monitoring")
    print("\nThis Python implementation provides equal or better functionality")
    print("compared to R's kohonen::som package!")
