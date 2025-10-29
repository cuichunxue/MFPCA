"""
Visualization tools for MFPCA results.

MFPCA結果の可視化ツール
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, List
import warnings


class MFPCAVisualizer:
    """
    Visualization tools for MFPCA analysis.

    MFPCA解析の可視化ツール

    Parameters
    ----------
    mfpca : MFPCA
        Fitted MFPCA model
    figsize : tuple, optional
        Default figure size (default: (12, 8))
    """

    def __init__(self, mfpca, figsize: Tuple[int, int] = (12, 8)):
        self.mfpca = mfpca
        self.figsize = figsize

    def plot_scree(
        self,
        n_components: Optional[int] = None,
        cumulative: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot scree plot (variance explained).

        スクリープロット（説明分散）を描画

        Parameters
        ----------
        n_components : int, optional
            Number of components to plot
        cumulative : bool, optional
            Whether to plot cumulative variance (default: True)
        ax : plt.Axes, optional
            Axes to plot on

        Returns
        -------
        ax : plt.Axes
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        if n_components is None:
            n_components = len(self.mfpca.eigenvalues_)

        components = np.arange(1, n_components + 1)
        var_exp = self.mfpca.variance_explained_ratio_[:n_components]

        ax.bar(components, var_exp, alpha=0.6, label='Individual')

        if cumulative:
            cumsum = np.cumsum(var_exp)
            ax.plot(components, cumsum, 'ro-', label='Cumulative')
            ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.3, label='95%')

        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Variance Explained', fontsize=12)
        ax.set_title('Scree Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        return ax

    def plot_eigenfunctions(
        self,
        n_components: Optional[int] = None,
        variable_names: Optional[List[str]] = None,
        separate_variables: bool = True,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot eigenfunctions.

        固有関数を描画

        Parameters
        ----------
        n_components : int, optional
            Number of components to plot
        variable_names : list of str, optional
            Names of variables
        separate_variables : bool, optional
            Whether to plot each variable separately (default: True)
        figsize : tuple, optional
            Figure size

        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        if n_components is None:
            n_components = min(4, len(self.mfpca.eigenvalues_))

        n_variables = self.mfpca.n_variables_

        if variable_names is None:
            variable_names = [f'Var {j+1}' for j in range(n_variables)]

        if figsize is None:
            if separate_variables:
                figsize = (14, 3 * n_components)
            else:
                figsize = self.figsize

        if separate_variables:
            fig, axes = plt.subplots(
                n_components, n_variables,
                figsize=figsize,
                sharex=True
            )
            if n_components == 1:
                axes = axes[np.newaxis, :]
            if n_variables == 1:
                axes = axes[:, np.newaxis]

            for k in range(n_components):
                for j in range(n_variables):
                    ax = axes[k, j]
                    ax.plot(
                        self.mfpca.time_grid_,
                        self.mfpca.eigenfunctions_[k, :, j],
                        'b-',
                        linewidth=2
                    )
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                    ax.set_ylabel(f'PC{k+1}', fontsize=10)
                    ax.grid(alpha=0.3)

                    if k == 0:
                        ax.set_title(variable_names[j], fontsize=11, fontweight='bold')
                    if k == n_components - 1:
                        ax.set_xlabel('Time', fontsize=10)

            fig.suptitle('Eigenfunctions', fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()

        else:
            fig, axes = plt.subplots(n_components, 1, figsize=figsize, sharex=True)
            if n_components == 1:
                axes = [axes]

            colors = plt.cm.tab10(np.linspace(0, 1, n_variables))

            for k in range(n_components):
                ax = axes[k]
                for j in range(n_variables):
                    ax.plot(
                        self.mfpca.time_grid_,
                        self.mfpca.eigenfunctions_[k, :, j],
                        color=colors[j],
                        label=variable_names[j],
                        linewidth=2
                    )
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.set_ylabel(f'PC{k+1}', fontsize=11)
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(alpha=0.3)

                if k == n_components - 1:
                    ax.set_xlabel('Time', fontsize=11)

            fig.suptitle('Eigenfunctions', fontsize=14, fontweight='bold')
            plt.tight_layout()

        return fig

    def plot_scores(
        self,
        components: Tuple[int, int] = (0, 1),
        labels: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot scatter of principal component scores.

        主成分スコアの散布図を描画

        Parameters
        ----------
        components : tuple of int, optional
            Which components to plot (default: (0, 1))
        labels : ndarray, optional
            Labels for coloring points
        ax : plt.Axes, optional
            Axes to plot on

        Returns
        -------
        ax : plt.Axes
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        comp1, comp2 = components
        scores = self.mfpca.scores_

        if labels is None:
            ax.scatter(scores[:, comp1], scores[:, comp2], alpha=0.6, s=50)
        else:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    scores[mask, comp1],
                    scores[mask, comp2],
                    alpha=0.6,
                    s=50,
                    color=colors[i],
                    label=f'{label}'
                )
            ax.legend()

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        var1 = self.mfpca.variance_explained_ratio_[comp1] * 100
        var2 = self.mfpca.variance_explained_ratio_[comp2] * 100

        ax.set_xlabel(f'PC{comp1+1} ({var1:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC{comp2+1} ({var2:.1f}%)', fontsize=12)
        ax.set_title('Principal Component Scores', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        return ax

    def plot_reconstruction(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        n_components: Optional[int] = None,
        variable_names: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot original vs reconstructed data.

        元データと再構成データを比較描画

        Parameters
        ----------
        X : ndarray
            Original data
        sample_idx : int, optional
            Which sample to plot (default: 0)
        n_components : int, optional
            Number of components for reconstruction
        variable_names : list of str, optional
            Names of variables
        figsize : tuple, optional
            Figure size

        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        n_variables = self.mfpca.n_variables_

        if variable_names is None:
            variable_names = [f'Variable {j+1}' for j in range(n_variables)]

        if n_components is None:
            n_components = len(self.mfpca.eigenvalues_)

        if figsize is None:
            figsize = (12, 3 * n_variables)

        # Reconstruct
        X_reconstructed = self.mfpca.reconstruct_partial(X, n_components)

        fig, axes = plt.subplots(n_variables, 1, figsize=figsize, sharex=True)
        if n_variables == 1:
            axes = [axes]

        for j in range(n_variables):
            ax = axes[j]
            ax.plot(
                self.mfpca.time_grid_,
                X[sample_idx, :, j],
                'b-',
                linewidth=2,
                label='Original',
                alpha=0.7
            )
            ax.plot(
                self.mfpca.time_grid_,
                X_reconstructed[sample_idx, :, j],
                'r--',
                linewidth=2,
                label=f'Reconstructed ({n_components} PCs)'
            )
            ax.set_ylabel(variable_names[j], fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.3)

            if j == n_variables - 1:
                ax.set_xlabel('Time', fontsize=11)

        rmse = self.mfpca.get_reconstruction_error(X[sample_idx:sample_idx+1], n_components)
        fig.suptitle(
            f'Original vs Reconstructed (Sample {sample_idx}, RMSE={rmse:.4f})',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()

        return fig

    def plot_mean_function(
        self,
        variable_names: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot mean function.

        平均関数を描画

        Parameters
        ----------
        variable_names : list of str, optional
            Names of variables
        ax : plt.Axes, optional
            Axes to plot on

        Returns
        -------
        ax : plt.Axes
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        n_variables = self.mfpca.n_variables_

        if variable_names is None:
            variable_names = [f'Variable {j+1}' for j in range(n_variables)]

        colors = plt.cm.tab10(np.linspace(0, 1, n_variables))

        for j in range(n_variables):
            ax.plot(
                self.mfpca.time_grid_,
                self.mfpca.mean_function_[:, j],
                color=colors[j],
                label=variable_names[j],
                linewidth=2
            )

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.set_title('Mean Function', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        return ax

    def plot_summary(
        self,
        X: np.ndarray,
        n_components: int = 3,
        variable_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Create comprehensive summary plot.

        包括的なサマリープロットを作成

        Parameters
        ----------
        X : ndarray
            Original data
        n_components : int, optional
            Number of components to show (default: 3)
        variable_names : list of str, optional
            Names of variables

        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Scree plot
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_scree(n_components=min(10, len(self.mfpca.eigenvalues_)), ax=ax1)

        # Score plot
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_scores(components=(0, 1), ax=ax2)

        # Mean function
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_mean_function(variable_names=variable_names, ax=ax3)

        # Eigenfunctions (first 3 components)
        n_comp_plot = min(n_components, len(self.mfpca.eigenvalues_))
        for k in range(n_comp_plot):
            ax = fig.add_subplot(gs[k+1 if k < 2 else k, :])

            n_variables = self.mfpca.n_variables_
            colors = plt.cm.tab10(np.linspace(0, 1, n_variables))

            if variable_names is None:
                var_names = [f'Var {j+1}' for j in range(n_variables)]
            else:
                var_names = variable_names

            for j in range(n_variables):
                ax.plot(
                    self.mfpca.time_grid_,
                    self.mfpca.eigenfunctions_[k, :, j],
                    color=colors[j],
                    label=var_names[j],
                    linewidth=2
                )

            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            var_exp = self.mfpca.variance_explained_ratio_[k] * 100
            ax.set_title(f'Eigenfunction {k+1} ({var_exp:.1f}%)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel(f'PC{k+1}', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle('MFPCA Summary', fontsize=16, fontweight='bold')

        return fig


def plot_covariance_surface(
    cov_surface: np.ndarray,
    time_grid: np.ndarray,
    variable_pair: Tuple[int, int] = (0, 0),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot covariance surface for a pair of variables.

    変数ペアの共分散曲面を描画

    Parameters
    ----------
    cov_surface : ndarray of shape (n_timepoints, n_timepoints, n_variables, n_variables)
        Covariance surface
    time_grid : ndarray
        Time grid
    variable_pair : tuple of int, optional
        Which variable pair to plot (default: (0, 0))
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    ax : plt.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    var1, var2 = variable_pair
    cov_ij = cov_surface[:, :, var1, var2]

    im = ax.contourf(time_grid, time_grid, cov_ij, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Covariance')

    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title(
        f'Covariance Surface: Var{var1+1} vs Var{var2+1}',
        fontsize=14,
        fontweight='bold'
    )

    return ax
