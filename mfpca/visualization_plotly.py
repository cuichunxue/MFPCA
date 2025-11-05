"""
Interactive visualization tools for MFPCA using Plotly.

Plotlyを使用したMFPCAのインタラクティブ可視化ツール
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Union, Tuple
import warnings


class MFPCAPlotlyVisualizer:
    """
    Interactive visualization tools for MFPCA analysis using Plotly.

    Plotlyを使用したMFPCA解析のインタラクティブ可視化ツール

    Parameters
    ----------
    mfpca : MFPCA or TheoreticalMFPCA
        Fitted MFPCA model
    """

    def __init__(self, mfpca):
        self.mfpca = mfpca
        self._check_fitted()

    def _check_fitted(self):
        """Check if MFPCA model is fitted."""
        if not hasattr(self.mfpca, 'eigenvalues_') or self.mfpca.eigenvalues_ is None:
            raise ValueError("MFPCA model must be fitted before visualization")

    def plot_scores_2d(
        self,
        components: Tuple[int, int] = (0, 1),
        labels: Optional[np.ndarray] = None,
        sample_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        color_continuous: bool = False,
        point_size: int = 8,
        show_origin_lines: bool = True
    ) -> go.Figure:
        """
        Create interactive 2D scatter plot of principal component scores.

        主成分スコアの2Dインタラクティブ散布図

        Parameters
        ----------
        components : tuple of int
            Which components to plot (default: (0, 1))
        labels : ndarray, optional
            Labels for coloring points
        sample_names : list of str, optional
            Names of samples for hover text
        title : str, optional
            Plot title
        color_continuous : bool
            Whether to use continuous color scale (default: False)
        point_size : int
            Size of scatter points (default: 8)
        show_origin_lines : bool
            Show lines at x=0 and y=0 (default: True)

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive figure
        """
        comp1, comp2 = components
        scores = self.mfpca.scores_

        if comp1 >= scores.shape[1] or comp2 >= scores.shape[1]:
            raise ValueError(f"Component indices must be < {scores.shape[1]}")

        # Get variance explained
        var_exp = self.mfpca.variance_explained_ratio_ if hasattr(self.mfpca, 'variance_explained_ratio_') \
                  else self.mfpca.explained_variance_ratio()

        # Create hover text
        if sample_names is None:
            sample_names = [f'Sample {i}' for i in range(len(scores))]

        hover_text = [
            f"{name}<br>PC{comp1+1}: {scores[i, comp1]:.3f}<br>PC{comp2+1}: {scores[i, comp2]:.3f}"
            for i, name in enumerate(sample_names)
        ]

        # Create figure
        if labels is not None:
            if color_continuous:
                # Continuous color scale
                fig = px.scatter(
                    x=scores[:, comp1],
                    y=scores[:, comp2],
                    color=labels,
                    hover_name=sample_names,
                    labels={
                        'x': f'PC{comp1+1} ({var_exp[comp1]:.1%})',
                        'y': f'PC{comp2+1} ({var_exp[comp2]:.1%})',
                        'color': 'Value'
                    },
                    color_continuous_scale='Viridis'
                )
            else:
                # Discrete color scale
                fig = px.scatter(
                    x=scores[:, comp1],
                    y=scores[:, comp2],
                    color=labels.astype(str) if isinstance(labels, np.ndarray) else labels,
                    hover_name=sample_names,
                    labels={
                        'x': f'PC{comp1+1} ({var_exp[comp1]:.1%})',
                        'y': f'PC{comp2+1} ({var_exp[comp2]:.1%})',
                        'color': 'Group'
                    }
                )
        else:
            # No labels
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scores[:, comp1],
                y=scores[:, comp2],
                mode='markers',
                marker=dict(size=point_size, color='steelblue'),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Scores'
            ))

        # Add origin lines
        if show_origin_lines:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        # Update layout
        if title is None:
            title = f'MFPCA Scores: PC{comp1+1} vs PC{comp2+1}'

        fig.update_layout(
            title=title,
            xaxis_title=f'PC{comp1+1} ({var_exp[comp1]:.1%})',
            yaxis_title=f'PC{comp2+1} ({var_exp[comp2]:.1%})',
            hovermode='closest',
            template='plotly_white',
            width=800,
            height=600
        )

        return fig

    def plot_scores_3d(
        self,
        components: Tuple[int, int, int] = (0, 1, 2),
        labels: Optional[np.ndarray] = None,
        sample_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        point_size: int = 5,
        show_origin_lines: bool = True
    ) -> go.Figure:
        """
        Create interactive 3D scatter plot of principal component scores.

        主成分スコアの3Dインタラクティブ散布図

        Parameters
        ----------
        components : tuple of int
            Which components to plot (default: (0, 1, 2))
        labels : ndarray, optional
            Labels for coloring points
        sample_names : list of str, optional
            Names of samples for hover text
        title : str, optional
            Plot title
        point_size : int
            Size of scatter points (default: 5)
        show_origin_lines : bool
            Show axis lines at origin (default: True)

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive 3D figure
        """
        comp1, comp2, comp3 = components
        scores = self.mfpca.scores_

        if max(components) >= scores.shape[1]:
            raise ValueError(f"Component indices must be < {scores.shape[1]}")

        # Get variance explained
        var_exp = self.mfpca.variance_explained_ratio_ if hasattr(self.mfpca, 'variance_explained_ratio_') \
                  else self.mfpca.explained_variance_ratio()

        # Create hover text
        if sample_names is None:
            sample_names = [f'Sample {i}' for i in range(len(scores))]

        hover_text = [
            f"{name}<br>PC{comp1+1}: {scores[i, comp1]:.3f}<br>"
            f"PC{comp2+1}: {scores[i, comp2]:.3f}<br>PC{comp3+1}: {scores[i, comp3]:.3f}"
            for i, name in enumerate(sample_names)
        ]

        # Create figure
        fig = go.Figure()

        if labels is not None:
            # Color by labels
            unique_labels = np.unique(labels)
            colors = px.colors.qualitative.Plotly

            for idx, label in enumerate(unique_labels):
                mask = labels == label
                fig.add_trace(go.Scatter3d(
                    x=scores[mask, comp1],
                    y=scores[mask, comp2],
                    z=scores[mask, comp3],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=colors[idx % len(colors)],
                        line=dict(width=0.5, color='white')
                    ),
                    text=[hover_text[i] for i in np.where(mask)[0]],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Group {label}'
                ))
        else:
            # Single color
            fig.add_trace(go.Scatter3d(
                x=scores[:, comp1],
                y=scores[:, comp2],
                z=scores[:, comp3],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color='steelblue',
                    line=dict(width=0.5, color='white')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Scores'
            ))

        # Add origin lines
        if show_origin_lines:
            # Get axis ranges
            x_range = [scores[:, comp1].min(), scores[:, comp1].max()]
            y_range = [scores[:, comp2].min(), scores[:, comp2].max()]
            z_range = [scores[:, comp3].min(), scores[:, comp3].max()]

            # X axis
            fig.add_trace(go.Scatter3d(
                x=x_range, y=[0, 0], z=[0, 0],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Y axis
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=y_range, z=[0, 0],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Z axis
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[0, 0], z=z_range,
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Update layout
        if title is None:
            title = f'MFPCA Scores: 3D View (PC{comp1+1}, PC{comp2+1}, PC{comp3+1})'

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f'PC{comp1+1} ({var_exp[comp1]:.1%})',
                yaxis_title=f'PC{comp2+1} ({var_exp[comp2]:.1%})',
                zaxis_title=f'PC{comp3+1} ({var_exp[comp3]:.1%})',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            hovermode='closest',
            template='plotly_white',
            width=900,
            height=700
        )

        return fig

    def plot_eigenfunctions(
        self,
        n_components: int = 3,
        variable_names: Optional[List[str]] = None,
        show_separately: bool = False,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot eigenfunctions interactively.

        固有関数のインタラクティブプロット

        Parameters
        ----------
        n_components : int
            Number of components to plot (default: 3)
        variable_names : list of str, optional
            Names of variables
        show_separately : bool
            Show each variable in separate subplot (default: False)
        title : str, optional
            Plot title

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive figure
        """
        # Get eigenfunctions
        if hasattr(self.mfpca, 'get_eigenfunctions'):
            eigenfuncs = self.mfpca.get_eigenfunctions()
        else:
            eigenfuncs = self.mfpca.eigenfunctions_

        n_comp = min(n_components, eigenfuncs.shape[0])
        time_grid = self.mfpca.time_grid_
        n_variables = eigenfuncs.shape[2]

        if variable_names is None:
            variable_names = [f'Variable {j+1}' for j in range(n_variables)]

        var_exp = self.mfpca.variance_explained_ratio_ if hasattr(self.mfpca, 'variance_explained_ratio_') \
                  else self.mfpca.explained_variance_ratio()

        if show_separately and n_variables > 1:
            # Separate subplots for each variable
            fig = make_subplots(
                rows=n_variables,
                cols=n_comp,
                subplot_titles=[
                    f'PC{k+1} ({var_exp[k]:.1%}) - {var_names}'
                    for var_names in variable_names
                    for k in range(n_comp)
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )

            for j in range(n_variables):
                for k in range(n_comp):
                    fig.add_trace(
                        go.Scatter(
                            x=time_grid,
                            y=eigenfuncs[k, :, j],
                            mode='lines',
                            name=f'PC{k+1} - {variable_names[j]}',
                            line=dict(width=2),
                            showlegend=(j == 0 and k == 0)
                        ),
                        row=j+1,
                        col=k+1
                    )

                    # Add zero line
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.3,
                        row=j+1,
                        col=k+1
                    )

            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Value")

        else:
            # All in one plot
            fig = go.Figure()

            colors = px.colors.qualitative.Plotly

            for k in range(n_comp):
                for j in range(n_variables):
                    color_idx = k * n_variables + j
                    fig.add_trace(go.Scatter(
                        x=time_grid,
                        y=eigenfuncs[k, :, j],
                        mode='lines',
                        name=f'PC{k+1} - {variable_names[j]}',
                        line=dict(width=2, color=colors[color_idx % len(colors)]),
                        legendgroup=f'PC{k+1}'
                    ))

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)

            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Eigenfunction Value")

        # Update layout
        if title is None:
            title = 'MFPCA Eigenfunctions'

        fig.update_layout(
            title=title,
            hovermode='x unified',
            template='plotly_white',
            width=1200 if show_separately else 900,
            height=800 if show_separately else 600
        )

        return fig

    def plot_scree(
        self,
        n_components: Optional[int] = None,
        show_cumulative: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive scree plot.

        インタラクティブなスクリープロット

        Parameters
        ----------
        n_components : int, optional
            Number of components to show
        show_cumulative : bool
            Show cumulative variance line (default: True)
        title : str, optional
            Plot title

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive figure
        """
        eigenvalues = self.mfpca.eigenvalues_
        var_exp = self.mfpca.variance_explained_ratio_ if hasattr(self.mfpca, 'variance_explained_ratio_') \
                  else self.mfpca.explained_variance_ratio()

        if n_components is None:
            n_components = len(eigenvalues)
        else:
            n_components = min(n_components, len(eigenvalues))

        components = np.arange(1, n_components + 1)

        fig = go.Figure()

        # Bar plot for individual variance
        fig.add_trace(go.Bar(
            x=components,
            y=var_exp[:n_components],
            name='Individual',
            marker_color='steelblue',
            hovertemplate='PC%{x}<br>Variance: %{y:.2%}<extra></extra>'
        ))

        # Line plot for cumulative variance
        if show_cumulative:
            cumvar = np.cumsum(var_exp[:n_components])
            fig.add_trace(go.Scatter(
                x=components,
                y=cumvar,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                yaxis='y2',
                hovertemplate='PC%{x}<br>Cumulative: %{y:.2%}<extra></extra>'
            ))

            # Add 95% line
            fig.add_hline(
                y=0.95,
                line_dash="dash",
                line_color="green",
                opacity=0.5,
                annotation_text="95%",
                yref='y2'
            )

        # Update layout
        if title is None:
            title = 'Scree Plot - Variance Explained'

        layout_dict = dict(
            title=title,
            xaxis=dict(title='Principal Component', dtick=1),
            yaxis=dict(title='Variance Explained', tickformat='.0%'),
            hovermode='x unified',
            template='plotly_white',
            width=900,
            height=600
        )

        if show_cumulative:
            layout_dict['yaxis2'] = dict(
                title='Cumulative Variance',
                overlaying='y',
                side='right',
                tickformat='.0%',
                range=[0, 1.05]
            )

        fig.update_layout(**layout_dict)

        return fig

    def plot_reconstruction(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        n_components: Optional[int] = None,
        variable_names: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot original vs reconstructed data.

        元データと再構成データの比較プロット

        Parameters
        ----------
        X : ndarray
            Original data
        sample_idx : int
            Which sample to plot (default: 0)
        n_components : int, optional
            Number of components for reconstruction
        variable_names : list of str, optional
            Names of variables
        title : str, optional
            Plot title

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive figure
        """
        if n_components is None:
            n_components = len(self.mfpca.eigenvalues_)

        # Reconstruct
        if hasattr(self.mfpca, 'reconstruct_partial'):
            X_recon = self.mfpca.reconstruct_partial(X, n_components)
        else:
            scores = self.mfpca.transform(X)
            X_recon = self.mfpca.inverse_transform(scores[:, :n_components], n_components)

        time_grid = self.mfpca.time_grid_
        n_variables = X.shape[2]

        if variable_names is None:
            variable_names = [f'Variable {j+1}' for j in range(n_variables)]

        # Create subplots
        fig = make_subplots(
            rows=n_variables,
            cols=1,
            subplot_titles=variable_names,
            vertical_spacing=0.1
        )

        colors_original = px.colors.qualitative.Plotly
        colors_recon = px.colors.qualitative.Set2

        for j in range(n_variables):
            # Original
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=X[sample_idx, :, j],
                    mode='lines',
                    name=f'{variable_names[j]} - Original',
                    line=dict(width=2, color=colors_original[j % len(colors_original)]),
                    legendgroup=variable_names[j],
                    showlegend=(j == 0)
                ),
                row=j+1,
                col=1
            )

            # Reconstructed
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=X_recon[sample_idx, :, j],
                    mode='lines',
                    name=f'{variable_names[j]} - Reconstructed',
                    line=dict(width=2, dash='dash', color=colors_recon[j % len(colors_recon)]),
                    legendgroup=variable_names[j],
                    showlegend=(j == 0)
                ),
                row=j+1,
                col=1
            )

        # Compute RMSE
        rmse = np.sqrt(np.mean((X[sample_idx] - X_recon[sample_idx])**2))

        # Update layout
        if title is None:
            title = f'Original vs Reconstructed (Sample {sample_idx}, {n_components} PCs, RMSE={rmse:.4f})'

        fig.update_layout(
            title=title,
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=300 * n_variables
        )

        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Value")

        return fig


# Convenience functions
def plot_mfpca_scores_2d(mfpca, **kwargs) -> go.Figure:
    """Convenience function for 2D score plot."""
    visualizer = MFPCAPlotlyVisualizer(mfpca)
    return visualizer.plot_scores_2d(**kwargs)


def plot_mfpca_scores_3d(mfpca, **kwargs) -> go.Figure:
    """Convenience function for 3D score plot."""
    visualizer = MFPCAPlotlyVisualizer(mfpca)
    return visualizer.plot_scores_3d(**kwargs)


def plot_mfpca_eigenfunctions(mfpca, **kwargs) -> go.Figure:
    """Convenience function for eigenfunction plot."""
    visualizer = MFPCAPlotlyVisualizer(mfpca)
    return visualizer.plot_eigenfunctions(**kwargs)


def plot_mfpca_scree(mfpca, **kwargs) -> go.Figure:
    """Convenience function for scree plot."""
    visualizer = MFPCAPlotlyVisualizer(mfpca)
    return visualizer.plot_scree(**kwargs)
