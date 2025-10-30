"""
Self-Organizing Maps (SOM) implementation.

R言語のkohonen::somと同等以上の機能を持つPython実装

References:
    Kohonen, T. (1982). Self-organized formation of topologically correct
    feature maps. Biological Cybernetics, 43(1), 59-69.
"""

import numpy as np
from typing import Optional, Tuple, Literal, Callable
import warnings


class SOM:
    """
    Self-Organizing Map (SOM) implementation.

    R言語のkohonen::somと同等以上の機能を提供するSOM実装

    Parameters
    ----------
    grid_shape : tuple of int
        SOMグリッドの形状 (rows, cols)
        例: (10, 10) で 10x10 のグリッド

    input_dim : int, optional
        入力データの次元数。fit時に自動決定される

    topology : {'rectangular', 'hexagonal'}, default='rectangular'
        グリッドのトポロジー
        - 'rectangular': 矩形グリッド
        - 'hexagonal': 六角形グリッド（R言語somと同様）

    neighborhood_function : {'gaussian', 'bubble', 'mexican_hat'}, default='gaussian'
        近傍関数のタイプ

    learning_rate_init : float, default=0.5
        初期学習率

    learning_rate_final : float, default=0.01
        最終学習率（線形減衰）

    sigma_init : float, optional
        初期近傍半径。Noneの場合は自動設定（グリッドサイズの半分）

    sigma_final : float, default=0.5
        最終近傍半径

    n_iterations : int, default=1000
        学習イテレーション数

    random_state : int, optional
        乱数シード

    initialization : {'random', 'pca'}, default='random'
        重み初期化方法
        - 'random': ランダム初期化
        - 'pca': PCAベースの初期化（より高速な収束）

    Attributes
    ----------
    weights_ : ndarray of shape (n_rows, n_cols, input_dim)
        学習後のSOM重みベクトル

    grid_positions_ : ndarray of shape (n_rows * n_cols, 2)
        グリッド上の各ノードの座標

    quantization_errors_ : list
        各イテレーションでの量子化誤差

    topographic_errors_ : list
        各イテレーションでのトポグラフィック誤差

    Examples
    --------
    >>> from mfpca import SOM
    >>> import numpy as np
    >>>
    >>> # データ生成
    >>> X = np.random.randn(100, 4)
    >>>
    >>> # SOM学習
    >>> som = SOM(grid_shape=(10, 10), n_iterations=1000)
    >>> som.fit(X)
    >>>
    >>> # 最も近いノードを取得
    >>> bmu_indices = som.predict(X)
    >>>
    >>> # U-matrixの取得
    >>> u_matrix = som.get_u_matrix()
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        input_dim: Optional[int] = None,
        topology: Literal['rectangular', 'hexagonal'] = 'rectangular',
        neighborhood_function: Literal['gaussian', 'bubble', 'mexican_hat'] = 'gaussian',
        learning_rate_init: float = 0.5,
        learning_rate_final: float = 0.01,
        sigma_init: Optional[float] = None,
        sigma_final: float = 0.5,
        n_iterations: int = 1000,
        random_state: Optional[int] = None,
        initialization: Literal['random', 'pca'] = 'random'
    ):
        self.grid_shape = grid_shape
        self.input_dim = input_dim
        self.topology = topology
        self.neighborhood_function = neighborhood_function
        self.learning_rate_init = learning_rate_init
        self.learning_rate_final = learning_rate_final
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.initialization = initialization

        # 学習後に設定される属性
        self.weights_ = None
        self.grid_positions_ = None
        self.quantization_errors_ = []
        self.topographic_errors_ = []
        self._is_fitted = False

        # 乱数生成器
        self.rng_ = np.random.RandomState(random_state)

    def _initialize_weights(self, X: np.ndarray) -> np.ndarray:
        """
        重みベクトルの初期化

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            訓練データ

        Returns
        -------
        weights : ndarray of shape (n_rows, n_cols, n_features)
            初期化された重みベクトル
        """
        n_rows, n_cols = self.grid_shape
        n_features = X.shape[1]

        if self.initialization == 'random':
            # データの範囲内でランダム初期化
            weights = np.zeros((n_rows, n_cols, n_features))
            for i in range(n_features):
                min_val = X[:, i].min()
                max_val = X[:, i].max()
                weights[:, :, i] = self.rng_.uniform(
                    min_val, max_val, size=(n_rows, n_cols)
                )

        elif self.initialization == 'pca':
            # PCAベースの初期化
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)

            # データの主成分を取得
            pca.fit(X)

            # グリッド上で線形補間
            weights = np.zeros((n_rows, n_cols, n_features))
            for i in range(n_rows):
                for j in range(n_cols):
                    # グリッド座標を[-1, 1]に正規化
                    coord_x = 2 * i / (n_rows - 1) - 1 if n_rows > 1 else 0
                    coord_y = 2 * j / (n_cols - 1) - 1 if n_cols > 1 else 0

                    # PCA空間での座標
                    pca_coord = np.array([coord_x, coord_y])

                    # 元の特徴空間に変換
                    weights[i, j] = pca.mean_ + pca_coord @ pca.components_[:2]

        return weights

    def _initialize_grid_positions(self) -> np.ndarray:
        """
        グリッド位置の初期化

        Returns
        -------
        positions : ndarray of shape (n_rows * n_cols, 2)
            各ノードのグリッド座標
        """
        n_rows, n_cols = self.grid_shape
        positions = []

        for i in range(n_rows):
            for j in range(n_cols):
                if self.topology == 'hexagonal':
                    # 六角形グリッド（R言語somと同様）
                    x = j + 0.5 * (i % 2)
                    y = i * np.sqrt(3) / 2
                else:
                    # 矩形グリッド
                    x = j
                    y = i

                positions.append([x, y])

        return np.array(positions)

    def _get_learning_rate(self, iteration: int) -> float:
        """学習率の取得（線形減衰）"""
        progress = iteration / self.n_iterations
        return self.learning_rate_init * (1 - progress) + \
               self.learning_rate_final * progress

    def _get_sigma(self, iteration: int) -> float:
        """近傍半径の取得（指数減衰）"""
        if self.sigma_init is None:
            # 自動設定：グリッドサイズの半分
            sigma_init = max(self.grid_shape) / 2.0
        else:
            sigma_init = self.sigma_init

        progress = iteration / self.n_iterations
        # 指数減衰
        return sigma_init * np.exp(-5 * progress) + self.sigma_final

    def _neighborhood_kernel(
        self,
        distance: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """
        近傍関数の計算

        Parameters
        ----------
        distance : ndarray
            BMUからの距離
        sigma : float
            近傍半径

        Returns
        -------
        kernel : ndarray
            近傍重み
        """
        if self.neighborhood_function == 'gaussian':
            return np.exp(-distance**2 / (2 * sigma**2))

        elif self.neighborhood_function == 'bubble':
            return (distance <= sigma).astype(float)

        elif self.neighborhood_function == 'mexican_hat':
            # Mexican hat wavelet
            return (1 - distance**2 / sigma**2) * \
                   np.exp(-distance**2 / (2 * sigma**2))

        else:
            raise ValueError(f"Unknown neighborhood function: {self.neighborhood_function}")

    def _find_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Best Matching Unit (BMU) の探索

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            入力ベクトル

        Returns
        -------
        bmu_indices : tuple of int
            BMUのグリッド座標 (row, col)
        """
        # 全ノードとの距離を計算
        distances = np.sum((self.weights_ - x)**2, axis=2)

        # 最小距離のノードを取得
        bmu_idx = np.argmin(distances)
        bmu_row = bmu_idx // self.grid_shape[1]
        bmu_col = bmu_idx % self.grid_shape[1]

        return bmu_row, bmu_col

    def _calculate_quantization_error(self, X: np.ndarray) -> float:
        """
        量子化誤差の計算

        各サンプルとそのBMUとの平均距離
        """
        errors = []
        for x in X:
            bmu_row, bmu_col = self._find_bmu(x)
            error = np.linalg.norm(x - self.weights_[bmu_row, bmu_col])
            errors.append(error)

        return np.mean(errors)

    def _calculate_topographic_error(self, X: np.ndarray) -> float:
        """
        トポグラフィック誤差の計算

        1位と2位のBMUが隣接していない割合
        """
        n_errors = 0

        for x in X:
            # 全ノードとの距離を計算
            distances = np.sum((self.weights_ - x)**2, axis=2).flatten()

            # 1位と2位のBMUを取得
            sorted_indices = np.argsort(distances)
            bmu1_idx = sorted_indices[0]
            bmu2_idx = sorted_indices[1]

            # グリッド座標に変換
            bmu1_pos = self.grid_positions_[bmu1_idx]
            bmu2_pos = self.grid_positions_[bmu2_idx]

            # 距離を計算
            dist = np.linalg.norm(bmu1_pos - bmu2_pos)

            # 隣接していない場合（距離 > 1.5）
            if dist > 1.5:
                n_errors += 1

        return n_errors / len(X)

    def fit(self, X: np.ndarray, verbose: bool = False) -> 'SOM':
        """
        SOMの学習

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            訓練データ

        verbose : bool, default=False
            学習進捗を表示するか

        Returns
        -------
        self : SOM
            学習済みのSOMオブジェクト
        """
        n_samples, n_features = X.shape

        # 入力次元の設定
        if self.input_dim is None:
            self.input_dim = n_features
        elif self.input_dim != n_features:
            raise ValueError(
                f"input_dim={self.input_dim} but X has {n_features} features"
            )

        # 重みの初期化
        self.weights_ = self._initialize_weights(X)

        # グリッド位置の初期化
        self.grid_positions_ = self._initialize_grid_positions()

        # 学習ループ
        self.quantization_errors_ = []
        self.topographic_errors_ = []

        for iteration in range(self.n_iterations):
            # パラメータの更新
            learning_rate = self._get_learning_rate(iteration)
            sigma = self._get_sigma(iteration)

            # ランダムにサンプルを選択
            idx = self.rng_.randint(0, n_samples)
            x = X[idx]

            # BMUを探索
            bmu_row, bmu_col = self._find_bmu(x)
            bmu_pos = self.grid_positions_[bmu_row * self.grid_shape[1] + bmu_col]

            # 全ノードの重みを更新
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    # BMUからの距離を計算
                    node_pos = self.grid_positions_[i * self.grid_shape[1] + j]
                    distance = np.linalg.norm(node_pos - bmu_pos)

                    # 近傍関数を適用
                    influence = self._neighborhood_kernel(distance, sigma)

                    # 重みを更新
                    self.weights_[i, j] += learning_rate * influence * \
                                          (x - self.weights_[i, j])

            # 定期的に誤差を計算
            if verbose and (iteration + 1) % (self.n_iterations // 10) == 0:
                qe = self._calculate_quantization_error(X)
                te = self._calculate_topographic_error(X)
                self.quantization_errors_.append(qe)
                self.topographic_errors_.append(te)
                print(f"Iteration {iteration + 1}/{self.n_iterations}: "
                      f"QE={qe:.4f}, TE={te:.4f}")

        # 最終誤差の計算
        final_qe = self._calculate_quantization_error(X)
        final_te = self._calculate_topographic_error(X)
        self.quantization_errors_.append(final_qe)
        self.topographic_errors_.append(final_te)

        self._is_fitted = True

        if verbose:
            print(f"\nTraining completed!")
            print(f"Final quantization error: {final_qe:.4f}")
            print(f"Final topographic error: {final_te:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        各サンプルのBMUを予測

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            入力データ

        Returns
        -------
        bmu_indices : ndarray of shape (n_samples,)
            各サンプルのBMUインデックス（0からn_rows*n_cols-1）
        """
        if not self._is_fitted:
            raise RuntimeError("SOM is not fitted yet. Call fit() first.")

        bmu_indices = np.zeros(len(X), dtype=int)

        for i, x in enumerate(X):
            bmu_row, bmu_col = self._find_bmu(x)
            bmu_indices[i] = bmu_row * self.grid_shape[1] + bmu_col

        return bmu_indices

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        データをSOMグリッド座標に変換

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            入力データ

        Returns
        -------
        coordinates : ndarray of shape (n_samples, 2)
            各サンプルのSOMグリッド座標
        """
        bmu_indices = self.predict(X)
        return self.grid_positions_[bmu_indices]

    def get_u_matrix(self) -> np.ndarray:
        """
        U-matrix（統合距離行列）の計算

        各ノードの平均近傍距離を表す。高い値はクラスター境界を示す。

        Returns
        -------
        u_matrix : ndarray of shape (n_rows, n_cols)
            U-matrix
        """
        if not self._is_fitted:
            raise RuntimeError("SOM is not fitted yet. Call fit() first.")

        n_rows, n_cols = self.grid_shape
        u_matrix = np.zeros((n_rows, n_cols))

        for i in range(n_rows):
            for j in range(n_cols):
                # 近傍ノードのインデックスを取得
                neighbors = []

                if self.topology == 'hexagonal':
                    # 六角形グリッドの近傍
                    offsets = [
                        (-1, 0), (1, 0),  # 上下
                        (0, -1), (0, 1),  # 左右
                        (-1, -1 if i % 2 == 0 else 1),  # 対角
                        (1, -1 if i % 2 == 0 else 1)
                    ]
                else:
                    # 矩形グリッドの近傍
                    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_rows and 0 <= nj < n_cols:
                        neighbors.append((ni, nj))

                # 近傍との距離の平均を計算
                if neighbors:
                    distances = [
                        np.linalg.norm(self.weights_[i, j] - self.weights_[ni, nj])
                        for ni, nj in neighbors
                    ]
                    u_matrix[i, j] = np.mean(distances)

        return u_matrix

    def get_component_planes(self) -> np.ndarray:
        """
        Component planes（成分面）の取得

        各変数ごとのSOMマップを返す（R言語somの重要な可視化）

        Returns
        -------
        component_planes : ndarray of shape (n_rows, n_cols, n_features)
            各変数のコンポーネントプレーン
        """
        if not self._is_fitted:
            raise RuntimeError("SOM is not fitted yet. Call fit() first.")

        return self.weights_.copy()

    def get_node_counts(self, X: np.ndarray) -> np.ndarray:
        """
        各ノードに割り当てられたサンプル数を計算

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            入力データ

        Returns
        -------
        counts : ndarray of shape (n_rows, n_cols)
            各ノードのサンプル数
        """
        bmu_indices = self.predict(X)
        counts = np.zeros(self.grid_shape, dtype=int)

        for bmu_idx in bmu_indices:
            row = bmu_idx // self.grid_shape[1]
            col = bmu_idx % self.grid_shape[1]
            counts[row, col] += 1

        return counts

    def get_node_vectors(self) -> np.ndarray:
        """
        全ノードの重みベクトルを1次元配列として取得

        Returns
        -------
        vectors : ndarray of shape (n_rows * n_cols, n_features)
            各ノードの重みベクトル
        """
        if not self._is_fitted:
            raise RuntimeError("SOM is not fitted yet. Call fit() first.")

        return self.weights_.reshape(-1, self.input_dim)
