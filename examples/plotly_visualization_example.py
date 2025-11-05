"""
Plotlyを使用したMFPCAのインタラクティブ可視化の例

Interactive MFPCA Visualization with Plotly
"""

import numpy as np
from mfpca import MFPCA, MFPCAPlotlyVisualizer
from mfpca.utils import generate_synthetic_mfpca_data

# Plotlyのインストール確認
try:
    import plotly.graph_objects as go
    print("✓ Plotly is installed")
except ImportError:
    print("✗ Plotly is not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'plotly', 'kaleido'])
    import plotly.graph_objects as go


def main():
    """
    Plotlyを使用したインタラクティブ可視化のデモンストレーション
    """
    print("=" * 80)
    print("MFPCA Interactive Visualization with Plotly")
    print("Plotlyを使用したMFPCAのインタラクティブ可視化")
    print("=" * 80)

    # ========================================================================
    # 1. データ生成
    # ========================================================================
    print("\n1. Generating synthetic data...")
    print("   合成データを生成中...")

    # 3つの異なるグループを持つデータを生成
    np.random.seed(42)
    n_samples_per_group = 30
    n_timepoints = 60
    n_variables = 3

    time_grid = np.linspace(0, 1, n_timepoints)

    # グループ1: 高周波数
    X1, _, _, _ = generate_synthetic_mfpca_data(
        n_samples=n_samples_per_group,
        n_timepoints=n_timepoints,
        n_variables=n_variables,
        n_components=2,
        seed=42
    )

    # グループ2: 低周波数
    X2 = np.zeros((n_samples_per_group, n_timepoints, n_variables))
    for i in range(n_samples_per_group):
        for j in range(n_variables):
            freq = 1 + np.random.randn() * 0.1
            phase = np.random.uniform(0, 2*np.pi)
            X2[i, :, j] = 2 * np.sin(2*np.pi*freq*time_grid + phase)
    X2 += np.random.randn(*X2.shape) * 0.1

    # グループ3: トレンド
    X3 = np.zeros((n_samples_per_group, n_timepoints, n_variables))
    for i in range(n_samples_per_group):
        for j in range(n_variables):
            slope = np.random.uniform(1, 3)
            intercept = np.random.uniform(-1, 1)
            X3[i, :, j] = slope * time_grid + intercept
    X3 += np.random.randn(*X3.shape) * 0.2

    # 結合
    X = np.vstack([X1, X2, X3])
    labels = np.array(['High Freq'] * n_samples_per_group +
                     ['Low Freq'] * n_samples_per_group +
                     ['Trend'] * n_samples_per_group)

    print(f"   Data shape: {X.shape}")
    print(f"   Groups: {np.unique(labels)}")

    # ========================================================================
    # 2. MFPCA実行
    # ========================================================================
    print("\n2. Running MFPCA...")
    print("   MFPCA実行中...")

    mfpca = MFPCA(
        n_components=5,
        smoothing=True,
        center=True
    )

    mfpca.fit(X, time_grid)

    print(f"   ✓ Eigenvalues: {mfpca.eigenvalues_[:3]}")
    print(f"   ✓ Variance explained: {mfpca.variance_explained_ratio_[:3]}")

    # ========================================================================
    # 3. Plotly可視化
    # ========================================================================
    print("\n3. Creating interactive visualizations...")
    print("   インタラクティブな可視化を作成中...")

    visualizer = MFPCAPlotlyVisualizer(mfpca)

    # ------------------------------------------------------------------------
    # 3.1 2Dスコアプロット
    # ------------------------------------------------------------------------
    print("\n   3.1 2D Score Plot (PC1 vs PC2)")

    fig_2d = visualizer.plot_scores_2d(
        components=(0, 1),
        labels=labels,
        sample_names=[f'{labels[i]}-{i%n_samples_per_group}' for i in range(len(X))],
        title='MFPCA Scores: 2D View',
        point_size=10
    )

    # 保存
    fig_2d.write_html('mfpca_scores_2d.html')
    print("       ✓ Saved to: mfpca_scores_2d.html")

    # 静的画像としても保存（オプション）
    try:
        fig_2d.write_image('mfpca_scores_2d.png', width=800, height=600)
        print("       ✓ Saved to: mfpca_scores_2d.png")
    except Exception as e:
        print(f"       ⚠ Could not save PNG (install kaleido): {e}")

    # ------------------------------------------------------------------------
    # 3.2 3Dスコアプロット
    # ------------------------------------------------------------------------
    print("\n   3.2 3D Score Plot (PC1, PC2, PC3)")

    fig_3d = visualizer.plot_scores_3d(
        components=(0, 1, 2),
        labels=labels,
        sample_names=[f'{labels[i]}-{i%n_samples_per_group}' for i in range(len(X))],
        title='MFPCA Scores: 3D View',
        point_size=6
    )

    # 保存
    fig_3d.write_html('mfpca_scores_3d.html')
    print("       ✓ Saved to: mfpca_scores_3d.html")

    try:
        fig_3d.write_image('mfpca_scores_3d.png', width=900, height=700)
        print("       ✓ Saved to: mfpca_scores_3d.png")
    except Exception as e:
        print(f"       ⚠ Could not save PNG: {e}")

    # ------------------------------------------------------------------------
    # 3.3 スクリープロット
    # ------------------------------------------------------------------------
    print("\n   3.3 Scree Plot")

    fig_scree = visualizer.plot_scree(
        n_components=5,
        show_cumulative=True,
        title='Scree Plot - Variance Explained'
    )

    fig_scree.write_html('mfpca_scree.html')
    print("       ✓ Saved to: mfpca_scree.html")

    # ------------------------------------------------------------------------
    # 3.4 固有関数プロット
    # ------------------------------------------------------------------------
    print("\n   3.4 Eigenfunctions")

    fig_eigenfuncs = visualizer.plot_eigenfunctions(
        n_components=3,
        variable_names=['Temperature', 'Pressure', 'Humidity'],
        show_separately=False,
        title='MFPCA Eigenfunctions'
    )

    fig_eigenfuncs.write_html('mfpca_eigenfunctions.html')
    print("       ✓ Saved to: mfpca_eigenfunctions.html")

    # ------------------------------------------------------------------------
    # 3.5 データ再構成の比較
    # ------------------------------------------------------------------------
    print("\n   3.5 Reconstruction Comparison")

    # 各グループから1サンプルずつ
    sample_indices = [0, n_samples_per_group, 2*n_samples_per_group]

    for idx, sample_idx in enumerate(sample_indices):
        fig_recon = visualizer.plot_reconstruction(
            X,
            sample_idx=sample_idx,
            n_components=3,
            variable_names=['Temperature', 'Pressure', 'Humidity'],
            title=f'Reconstruction: {labels[sample_idx]} Sample'
        )

        filename = f'mfpca_reconstruction_{labels[sample_idx].replace(" ", "_")}.html'
        fig_recon.write_html(filename)
        print(f"       ✓ Saved to: {filename}")

    # ========================================================================
    # 4. 便利関数の使用例
    # ========================================================================
    print("\n4. Using convenience functions...")
    print("   便利関数の使用例...")

    from mfpca import plot_mfpca_scores_2d, plot_mfpca_scores_3d

    # 簡単な使い方
    fig = plot_mfpca_scores_2d(
        mfpca,
        components=(1, 2),  # PC2 vs PC3
        labels=labels,
        title='Quick 2D Plot: PC2 vs PC3'
    )
    fig.write_html('mfpca_quick_2d.html')
    print("   ✓ Saved quick 2D plot to: mfpca_quick_2d.html")

    # ========================================================================
    # 5. インタラクティブダッシュボード的な使用
    # ========================================================================
    print("\n5. Creating combined dashboard...")
    print("   複合ダッシュボードを作成中...")

    from plotly.subplots import make_subplots

    # 2x2のサブプロット
    fig_dashboard = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'scatter'}]
        ],
        subplot_titles=(
            'PC1 vs PC2',
            'PC2 vs PC3',
            'Variance Explained',
            'First Eigenfunction'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # PC1 vs PC2
    for label in np.unique(labels):
        mask = labels == label
        scores = mfpca.scores_
        fig_dashboard.add_trace(
            go.Scatter(
                x=scores[mask, 0],
                y=scores[mask, 1],
                mode='markers',
                name=label,
                marker=dict(size=8),
                showlegend=True
            ),
            row=1, col=1
        )

    # PC2 vs PC3
    for label in np.unique(labels):
        mask = labels == label
        fig_dashboard.add_trace(
            go.Scatter(
                x=scores[mask, 1],
                y=scores[mask, 2],
                mode='markers',
                name=label,
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=2
        )

    # Variance explained
    fig_dashboard.add_trace(
        go.Bar(
            x=np.arange(1, 6),
            y=mfpca.variance_explained_ratio_[:5],
            name='Variance',
            showlegend=False,
            marker_color='steelblue'
        ),
        row=2, col=1
    )

    # First eigenfunction
    eigenfuncs = mfpca.eigenfunctions_
    for j, var_name in enumerate(['Temp', 'Press', 'Humid']):
        fig_dashboard.add_trace(
            go.Scatter(
                x=time_grid,
                y=eigenfuncs[0, :, j],
                mode='lines',
                name=var_name,
                showlegend=(j < 3)
            ),
            row=2, col=2
        )

    # レイアウト更新
    fig_dashboard.update_xaxes(title_text="PC1", row=1, col=1)
    fig_dashboard.update_yaxes(title_text="PC2", row=1, col=1)
    fig_dashboard.update_xaxes(title_text="PC2", row=1, col=2)
    fig_dashboard.update_yaxes(title_text="PC3", row=1, col=2)
    fig_dashboard.update_xaxes(title_text="Component", row=2, col=1)
    fig_dashboard.update_yaxes(title_text="Variance", row=2, col=1)
    fig_dashboard.update_xaxes(title_text="Time", row=2, col=2)
    fig_dashboard.update_yaxes(title_text="Value", row=2, col=2)

    fig_dashboard.update_layout(
        title_text="MFPCA Interactive Dashboard",
        height=900,
        width=1400,
        template='plotly_white'
    )

    fig_dashboard.write_html('mfpca_dashboard.html')
    print("   ✓ Saved to: mfpca_dashboard.html")

    # ========================================================================
    # 完了
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ All visualizations created successfully!")
    print("✓ すべての可視化が正常に作成されました！")
    print("=" * 80)

    print("\n生成されたファイル:")
    print("  1. mfpca_scores_2d.html         - 2Dスコアプロット")
    print("  2. mfpca_scores_3d.html         - 3Dスコアプロット（回転可能）")
    print("  3. mfpca_scree.html             - スクリープロット")
    print("  4. mfpca_eigenfunctions.html    - 固有関数")
    print("  5. mfpca_reconstruction_*.html  - データ再構成比較")
    print("  6. mfpca_quick_2d.html          - クイック2Dプロット")
    print("  7. mfpca_dashboard.html         - 総合ダッシュボード")

    print("\n使い方:")
    print("  ブラウザでHTMLファイルを開くと、インタラクティブな可視化が表示されます。")
    print("  - マウスでズーム、パン、回転（3D）が可能")
    print("  - データポイントにホバーすると詳細情報を表示")
    print("  - 凡例をクリックしてグループの表示/非表示を切り替え")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
