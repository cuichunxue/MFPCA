"""
Tests for SOM implementation.

Self-Organizing Maps (SOM) のテスト
"""

import numpy as np
import pytest
from mfpca import SOM, SOMVisualizer


class TestSOM:
    """Test SOM class."""

    def test_basic_fit(self):
        """Test basic SOM fitting."""
        # データ生成
        np.random.seed(42)
        X = np.random.randn(100, 4)

        # SOM学習
        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X)

        # 学習後の属性をチェック
        assert som.weights_ is not None
        assert som.weights_.shape == (5, 5, 4)
        assert som.grid_positions_ is not None
        assert som._is_fitted

    def test_predict(self):
        """Test BMU prediction."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(4, 4), n_iterations=100, random_state=42)
        som.fit(X)

        # 予測
        bmu_indices = som.predict(X)

        # 形状チェック
        assert bmu_indices.shape == (50,)
        # 範囲チェック（0 ~ 15）
        assert np.all(bmu_indices >= 0)
        assert np.all(bmu_indices < 16)

    def test_transform(self):
        """Test coordinate transformation."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(4, 4), n_iterations=100, random_state=42)
        som.fit(X)

        # 変換
        coords = som.transform(X)

        # 形状チェック
        assert coords.shape == (50, 2)

    def test_hexagonal_topology(self):
        """Test hexagonal grid topology."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(
            grid_shape=(5, 5),
            topology='hexagonal',
            n_iterations=100,
            random_state=42
        )
        som.fit(X)

        assert som.weights_.shape == (5, 5, 3)

        # グリッド位置の確認
        positions = som.grid_positions_
        assert positions.shape == (25, 2)

    def test_pca_initialization(self):
        """Test PCA-based initialization."""
        np.random.seed(42)
        X = np.random.randn(100, 4)

        som = SOM(
            grid_shape=(5, 5),
            initialization='pca',
            n_iterations=100,
            random_state=42
        )
        som.fit(X)

        assert som.weights_ is not None

    def test_u_matrix(self):
        """Test U-matrix computation."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(4, 4), n_iterations=100, random_state=42)
        som.fit(X)

        u_matrix = som.get_u_matrix()

        # 形状チェック
        assert u_matrix.shape == (4, 4)
        # 非負性チェック
        assert np.all(u_matrix >= 0)

    def test_component_planes(self):
        """Test component planes extraction."""
        np.random.seed(42)
        X = np.random.randn(50, 4)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X)

        component_planes = som.get_component_planes()

        # 形状チェック
        assert component_planes.shape == (5, 5, 4)

    def test_node_counts(self):
        """Test node count computation."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(4, 4), n_iterations=100, random_state=42)
        som.fit(X)

        counts = som.get_node_counts(X)

        # 形状チェック
        assert counts.shape == (4, 4)
        # 総和チェック
        assert np.sum(counts) == 50

    def test_neighborhood_functions(self):
        """Test different neighborhood functions."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        for func in ['gaussian', 'bubble', 'mexican_hat']:
            som = SOM(
                grid_shape=(4, 4),
                neighborhood_function=func,
                n_iterations=100,
                random_state=42
            )
            som.fit(X)

            assert som.weights_ is not None

    def test_quantization_error(self):
        """Test quantization error calculation."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X, verbose=True)

        # 誤差が記録されているか確認
        assert len(som.quantization_errors_) > 0

        # 最終的な誤差は正の値
        assert som.quantization_errors_[-1] > 0

    def test_topographic_error(self):
        """Test topographic error calculation."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X, verbose=True)

        # 誤差が記録されているか確認
        assert len(som.topographic_errors_) > 0

        # トポグラフィック誤差は0~1の範囲
        assert 0 <= som.topographic_errors_[-1] <= 1

    def test_learning_rate_decay(self):
        """Test learning rate decay."""
        som = SOM(
            grid_shape=(5, 5),
            learning_rate_init=0.5,
            learning_rate_final=0.01,
            n_iterations=1000
        )

        # 開始時
        lr_start = som._get_learning_rate(0)
        assert abs(lr_start - 0.5) < 1e-6

        # 終了時
        lr_end = som._get_learning_rate(1000)
        assert abs(lr_end - 0.01) < 1e-6

        # 中間
        lr_mid = som._get_learning_rate(500)
        assert 0.01 < lr_mid < 0.5

    def test_sigma_decay(self):
        """Test neighborhood radius decay."""
        som = SOM(
            grid_shape=(10, 10),
            sigma_init=5.0,
            sigma_final=0.5,
            n_iterations=1000
        )

        # 開始時
        sigma_start = som._get_sigma(0)
        assert sigma_start > 4.0

        # 終了時
        sigma_end = som._get_sigma(1000)
        assert abs(sigma_end - 0.5) < 0.1

    def test_clustering_structure(self):
        """Test that SOM captures clustering structure."""
        np.random.seed(42)

        # 3つのクラスターを持つデータを生成
        cluster1 = np.random.randn(30, 3) + np.array([5, 5, 5])
        cluster2 = np.random.randn(30, 3) + np.array([-5, -5, -5])
        cluster3 = np.random.randn(30, 3) + np.array([5, -5, 0])
        X = np.vstack([cluster1, cluster2, cluster3])

        som = SOM(grid_shape=(10, 10), n_iterations=500, random_state=42)
        som.fit(X)

        # 各クラスターのBMUを取得
        bmu1 = som.predict(cluster1)
        bmu2 = som.predict(cluster2)
        bmu3 = som.predict(cluster3)

        # 同じクラスター内のサンプルは近いBMUを持つべき
        # （標準偏差が小さい）
        std1 = np.std(bmu1)
        std2 = np.std(bmu2)
        std3 = np.std(bmu3)

        # 異なるクラスター間のBMU平均値は異なるべき
        mean1 = np.mean(bmu1)
        mean2 = np.mean(bmu2)
        mean3 = np.mean(bmu3)

        # 少なくとも2つのクラスターは明確に分離されているべき
        diffs = [abs(mean1 - mean2), abs(mean2 - mean3), abs(mean1 - mean3)]
        assert max(diffs) > 5  # 異なるBMU領域に配置されている

    def test_not_fitted_error(self):
        """Test error when using unfitted SOM."""
        som = SOM(grid_shape=(5, 5))

        X = np.random.randn(10, 3)

        with pytest.raises(RuntimeError):
            som.predict(X)

        with pytest.raises(RuntimeError):
            som.get_u_matrix()

        with pytest.raises(RuntimeError):
            som.get_component_planes()


class TestSOMVisualizer:
    """Test SOMVisualizer class."""

    def test_visualizer_creation(self):
        """Test visualizer creation."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X)

        visualizer = SOMVisualizer(som)
        assert visualizer.som is som

    def test_plot_component_planes(self):
        """Test component planes visualization."""
        np.random.seed(42)
        X = np.random.randn(50, 4)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X)

        visualizer = SOMVisualizer(som)

        # デフォルトの変数名
        fig1 = visualizer.plot_component_planes()
        assert fig1 is not None

        # カスタム変数名
        fig2 = visualizer.plot_component_planes(
            X,
            variable_names=['Var1', 'Var2', 'Var3', 'Var4']
        )
        assert fig2 is not None

    def test_plot_u_matrix(self):
        """Test U-matrix visualization."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X)

        visualizer = SOMVisualizer(som)
        fig = visualizer.plot_u_matrix()

        assert fig is not None

    def test_plot_node_counts(self):
        """Test node counts visualization."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X)

        visualizer = SOMVisualizer(som)
        fig = visualizer.plot_node_counts(X)

        assert fig is not None

    def test_plot_training_progress(self):
        """Test training progress visualization."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X, verbose=True)

        visualizer = SOMVisualizer(som)
        fig = visualizer.plot_training_progress()

        assert fig is not None

    def test_plot_data_mapping(self):
        """Test data mapping visualization."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X)

        visualizer = SOMVisualizer(som)

        # ラベルなし
        fig1 = visualizer.plot_data_mapping(X)
        assert fig1 is not None

        # ラベルあり
        labels = np.random.randint(0, 3, size=50)
        fig2 = visualizer.plot_data_mapping(X, labels=labels)
        assert fig2 is not None

    def test_plot_summary(self):
        """Test summary visualization."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        som = SOM(grid_shape=(5, 5), n_iterations=100, random_state=42)
        som.fit(X, verbose=True)

        visualizer = SOMVisualizer(som)
        fig = visualizer.plot_summary(X)

        assert fig is not None


class TestSOMIntegration:
    """Integration tests for SOM."""

    def test_full_pipeline(self):
        """Test complete SOM pipeline."""
        np.random.seed(42)

        # データ生成
        X = np.random.randn(100, 5)

        # SOM学習
        som = SOM(
            grid_shape=(8, 8),
            topology='rectangular',
            n_iterations=500,
            random_state=42
        )
        som.fit(X, verbose=True)

        # 予測
        bmu_indices = som.predict(X)

        # 変換
        coords = som.transform(X)

        # U-matrix
        u_matrix = som.get_u_matrix()

        # Component planes
        component_planes = som.get_component_planes()

        # Node counts
        counts = som.get_node_counts(X)

        # すべて成功
        assert bmu_indices.shape == (100,)
        assert coords.shape == (100, 2)
        assert u_matrix.shape == (8, 8)
        assert component_planes.shape == (8, 8, 5)
        assert counts.shape == (8, 8)

    def test_small_data(self):
        """Test with very small dataset."""
        np.random.seed(42)
        X = np.random.randn(10, 2)

        som = SOM(grid_shape=(3, 3), n_iterations=50, random_state=42)
        som.fit(X)

        bmu_indices = som.predict(X)
        assert len(bmu_indices) == 10

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(50, 20)

        som = SOM(grid_shape=(5, 5), n_iterations=200, random_state=42)
        som.fit(X)

        assert som.weights_.shape == (5, 5, 20)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
