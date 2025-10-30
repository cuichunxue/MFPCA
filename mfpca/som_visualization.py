"""
SOM visualization module.

R言語のkohonen::somと同等の可視化機能
特にcomponent planesの実装に重点を置いています
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from typing import Optional, List, Tuple
import warnings


class SOMVisualizer:
    """
    SOM visualization class.

    R言語のkohonen::plotと同等以上の可視化機能

    Parameters
    ----------
    som : SOM
        学習済みのSOMオブジェクト

    Examples
    --------
    >>> from mfpca import SOM, SOMVisualizer
    >>> som = SOM(grid_shape=(10, 10))
    >>> som.fit(X)
    >>> visualizer = SOMVisualizer(som)
    >>> visualizer.plot_component_planes(X, variable_names=['Var1', 'Var2', 'Var3'])
    """

    def __init__(self, som):
        self.som = som

    def plot_component_planes(
        self,
        X: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = 'RdYlBu_r',
        show_colorbar: bool = True
    ):
        """
        Component planes（成分面）のプロット

        R言語somの最も有用な可視化機能
        各変数がSOMのどの領域で高い/低い値を示すかを可視化

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), optional
            データ（サンプル数を表示するため）

        variable_names : list of str, optional
            変数名のリスト

        figsize : tuple of float, optional
            図のサイズ

        cmap : str, default='RdYlBu_r'
            カラーマップ

        show_colorbar : bool, default=True
            カラーバーを表示するか

        Returns
        -------
        fig : matplotlib.figure.Figure
            生成された図
        """
        component_planes = self.som.get_component_planes()
        n_features = component_planes.shape[2]

        # 変数名の設定
        if variable_names is None:
            variable_names = [f'Variable {i+1}' for i in range(n_features)]
        elif len(variable_names) != n_features:
            raise ValueError(f"variable_names length {len(variable_names)} "
                           f"does not match n_features {n_features}")

        # グリッドレイアウトの計算
        n_cols = int(np.ceil(np.sqrt(n_features)))
        n_rows = int(np.ceil(n_features / n_cols))

        # 図のサイズ設定
        if figsize is None:
            figsize = (n_cols * 4, n_rows * 3.5)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten() if n_features > 1 else [axes]

        # 各変数のcomponent planeをプロット
        for i in range(n_features):
            ax = axes[i]
            data = component_planes[:, :, i]

            # ヒートマップをプロット
            im = ax.imshow(
                data,
                cmap=cmap,
                interpolation='nearest',
                aspect='auto'
            )

            ax.set_title(variable_names[i], fontsize=12, fontweight='bold')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')

            # カラーバー
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Weight value', fontsize=9)

            # グリッド線
            ax.set_xticks(np.arange(self.som.grid_shape[1]))
            ax.set_yticks(np.arange(self.som.grid_shape[0]))
            ax.grid(True, alpha=0.3)

        # 余分なサブプロットを非表示
        for i in range(n_features, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        # タイトル
        fig.suptitle('SOM Component Planes - Variable Contributions',
                    fontsize=14, fontweight='bold', y=1.02)

        return fig

    def plot_u_matrix(
        self,
        figsize: Tuple[float, float] = (10, 8),
        cmap: str = 'bone_r',
        show_colorbar: bool = True
    ):
        """
        U-matrix（統合距離行列）のプロット

        クラスター境界を可視化（高い値＝境界）

        Parameters
        ----------
        figsize : tuple of float, default=(10, 8)
            図のサイズ

        cmap : str, default='bone_r'
            カラーマップ

        show_colorbar : bool, default=True
            カラーバーを表示するか

        Returns
        -------
        fig : matplotlib.figure.Figure
            生成された図
        """
        u_matrix = self.som.get_u_matrix()

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            u_matrix,
            cmap=cmap,
            interpolation='nearest',
            aspect='auto'
        )

        ax.set_title('U-Matrix (Unified Distance Matrix)\nHigher values indicate cluster boundaries',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Average distance to neighbors', fontsize=11)

        # グリッド線
        ax.set_xticks(np.arange(self.som.grid_shape[1]))
        ax.set_yticks(np.arange(self.som.grid_shape[0]))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def plot_node_counts(
        self,
        X: np.ndarray,
        figsize: Tuple[float, float] = (10, 8),
        cmap: str = 'YlOrRd',
        show_colorbar: bool = True,
        annotate: bool = True
    ):
        """
        各ノードに割り当てられたサンプル数のプロット

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            データ

        figsize : tuple of float, default=(10, 8)
            図のサイズ

        cmap : str, default='YlOrRd'
            カラーマップ

        show_colorbar : bool, default=True
            カラーバーを表示するか

        annotate : bool, default=True
            各ノードに数値を表示するか

        Returns
        -------
        fig : matplotlib.figure.Figure
            生成された図
        """
        counts = self.som.get_node_counts(X)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            counts,
            cmap=cmap,
            interpolation='nearest',
            aspect='auto'
        )

        ax.set_title('SOM Node Activation Counts',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Number of samples', fontsize=11)

        # 各ノードに数値を表示
        if annotate:
            for i in range(self.som.grid_shape[0]):
                for j in range(self.som.grid_shape[1]):
                    text = ax.text(
                        j, i, str(counts[i, j]),
                        ha="center", va="center",
                        color="white" if counts[i, j] > counts.max()/2 else "black",
                        fontsize=10,
                        fontweight='bold'
                    )

        # グリッド線
        ax.set_xticks(np.arange(self.som.grid_shape[1]))
        ax.set_yticks(np.arange(self.som.grid_shape[0]))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def plot_training_progress(
        self,
        figsize: Tuple[float, float] = (12, 5)
    ):
        """
        学習進捗のプロット

        Parameters
        ----------
        figsize : tuple of float, default=(12, 5)
            図のサイズ

        Returns
        -------
        fig : matplotlib.figure.Figure
            生成された図
        """
        if not self.som.quantization_errors_:
            warnings.warn("No training errors recorded. "
                        "Run fit() with verbose=True to record errors.")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 量子化誤差
        ax1.plot(self.som.quantization_errors_, 'o-', linewidth=2, markersize=6)
        ax1.set_title('Quantization Error', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Checkpoint', fontsize=11)
        ax1.set_ylabel('Error', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # トポグラフィック誤差
        ax2.plot(self.som.topographic_errors_, 'o-', linewidth=2, markersize=6, color='orange')
        ax2.set_title('Topographic Error', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Checkpoint', fontsize=11)
        ax2.set_ylabel('Error', fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def plot_data_mapping(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (10, 8),
        alpha: float = 0.6,
        s: float = 100
    ):
        """
        データのSOMマッピングを可視化

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            データ

        labels : ndarray of shape (n_samples,), optional
            各サンプルのラベル（色分け用）

        figsize : tuple of float, default=(10, 8)
            図のサイズ

        alpha : float, default=0.6
            透明度

        s : float, default=100
            マーカーサイズ

        Returns
        -------
        fig : matplotlib.figure.Figure
            生成された図
        """
        # データをSOM座標に変換
        som_coords = self.som.transform(X)

        fig, ax = plt.subplots(figsize=figsize)

        # U-matrixを背景として表示
        u_matrix = self.som.get_u_matrix()
        im = ax.imshow(
            u_matrix,
            cmap='gray_r',
            alpha=0.3,
            interpolation='nearest',
            aspect='auto'
        )

        # データ点をプロット
        if labels is None:
            scatter = ax.scatter(
                som_coords[:, 0],
                som_coords[:, 1],
                alpha=alpha,
                s=s,
                edgecolors='black',
                linewidth=1
            )
        else:
            # ラベルごとに色分け
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for idx, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    som_coords[mask, 0],
                    som_coords[mask, 1],
                    alpha=alpha,
                    s=s,
                    label=f'Class {label}',
                    color=colors[idx],
                    edgecolors='black',
                    linewidth=1
                )

            ax.legend(loc='best', fontsize=10)

        ax.set_title('SOM Data Mapping', fontsize=14, fontweight='bold')
        ax.set_xlabel('SOM X coordinate', fontsize=12)
        ax.set_ylabel('SOM Y coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def plot_hexagonal_map(
        self,
        values: np.ndarray,
        title: str = 'SOM Hexagonal Map',
        figsize: Tuple[float, float] = (10, 8),
        cmap: str = 'viridis',
        show_colorbar: bool = True
    ):
        """
        六角形グリッドでのSOMマップ表示

        Parameters
        ----------
        values : ndarray of shape (n_rows, n_cols)
            各ノードの値

        title : str, default='SOM Hexagonal Map'
            タイトル

        figsize : tuple of float, default=(10, 8)
            図のサイズ

        cmap : str, default='viridis'
            カラーマップ

        show_colorbar : bool, default=True
            カラーバーを表示するか

        Returns
        -------
        fig : matplotlib.figure.Figure
            生成された図
        """
        if self.som.topology != 'hexagonal':
            warnings.warn("SOM topology is not hexagonal. "
                        "Use plot_component_planes() for rectangular grids.")

        fig, ax = plt.subplots(figsize=figsize)

        # 六角形のパッチを作成
        hexagons = []
        colors = []

        for i in range(self.som.grid_shape[0]):
            for j in range(self.som.grid_shape[1]):
                # 六角形の中心座標
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3) / 2

                # 六角形を作成
                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=0.58,
                    orientation=0,
                    edgecolor='black',
                    linewidth=1
                )
                hexagons.append(hexagon)
                colors.append(values[i, j])

        # コレクションを作成
        collection = PatchCollection(hexagons, cmap=cmap, edgecolors='black')
        collection.set_array(np.array(colors))
        ax.add_collection(collection)

        # 軸の設定
        ax.set_xlim(-1, self.som.grid_shape[1] + 1)
        ax.set_ylim(-1, self.som.grid_shape[0] * np.sqrt(3) / 2 + 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')

        if show_colorbar:
            cbar = plt.colorbar(collection, ax=ax)
            cbar.set_label('Value', fontsize=11)

        plt.tight_layout()

        return fig

    def plot_summary(
        self,
        X: np.ndarray,
        variable_names: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (16, 12)
    ):
        """
        SOMの総合サマリー（4つのプロットを一度に表示）

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            データ

        variable_names : list of str, optional
            変数名のリスト

        labels : ndarray of shape (n_samples,), optional
            各サンプルのラベル

        figsize : tuple of float, default=(16, 12)
            図のサイズ

        Returns
        -------
        fig : matplotlib.figure.Figure
            生成された図
        """
        fig = plt.figure(figsize=figsize)

        # 2x2レイアウト
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. U-matrix
        ax1 = fig.add_subplot(gs[0, 0])
        u_matrix = self.som.get_u_matrix()
        im1 = ax1.imshow(u_matrix, cmap='bone_r', interpolation='nearest', aspect='auto')
        ax1.set_title('U-Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1, label='Distance')
        ax1.grid(True, alpha=0.3)

        # 2. Node counts
        ax2 = fig.add_subplot(gs[0, 1])
        counts = self.som.get_node_counts(X)
        im2 = ax2.imshow(counts, cmap='YlOrRd', interpolation='nearest', aspect='auto')
        ax2.set_title('Node Counts', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        plt.colorbar(im2, ax=ax2, label='# Samples')
        ax2.grid(True, alpha=0.3)

        # 各ノードに数値を表示
        for i in range(self.som.grid_shape[0]):
            for j in range(self.som.grid_shape[1]):
                ax2.text(
                    j, i, str(counts[i, j]),
                    ha="center", va="center",
                    color="white" if counts[i, j] > counts.max()/2 else "black",
                    fontsize=8
                )

        # 3. Data mapping
        ax3 = fig.add_subplot(gs[1, 0])
        som_coords = self.som.transform(X)
        ax3.imshow(u_matrix, cmap='gray_r', alpha=0.3, interpolation='nearest', aspect='auto')

        if labels is None:
            ax3.scatter(som_coords[:, 0], som_coords[:, 1], alpha=0.6, s=50, edgecolors='black')
        else:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for idx, label in enumerate(unique_labels):
                mask = labels == label
                ax3.scatter(
                    som_coords[mask, 0], som_coords[mask, 1],
                    alpha=0.6, s=50, label=f'Class {label}',
                    color=colors[idx], edgecolors='black'
                )
            ax3.legend(loc='best', fontsize=8)

        ax3.set_title('Data Mapping', fontsize=12, fontweight='bold')
        ax3.set_xlabel('SOM X')
        ax3.set_ylabel('SOM Y')
        ax3.grid(True, alpha=0.3)

        # 4. Training progress
        ax4 = fig.add_subplot(gs[1, 1])
        if self.som.quantization_errors_:
            ax4_twin = ax4.twinx()
            line1 = ax4.plot(
                self.som.quantization_errors_, 'o-',
                color='blue', label='Quantization Error'
            )
            line2 = ax4_twin.plot(
                self.som.topographic_errors_, 's-',
                color='orange', label='Topographic Error'
            )
            ax4.set_xlabel('Checkpoint')
            ax4.set_ylabel('Quantization Error', color='blue')
            ax4_twin.set_ylabel('Topographic Error', color='orange')
            ax4.tick_params(axis='y', labelcolor='blue')
            ax4_twin.tick_params(axis='y', labelcolor='orange')

            # 凡例を統合
            lines = line1 + line2
            labels_legend = [l.get_label() for l in lines]
            ax4.legend(lines, labels_legend, loc='best', fontsize=8)
        else:
            ax4.text(
                0.5, 0.5, 'No training progress data\n(Run fit with verbose=True)',
                ha='center', va='center', transform=ax4.transAxes
            )

        ax4.set_title('Training Progress', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        fig.suptitle('SOM Summary Visualization', fontsize=16, fontweight='bold', y=0.995)

        return fig
