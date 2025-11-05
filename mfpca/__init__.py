"""
MFPCA - Multivariate Functional Principal Component Analysis

A professional implementation for analyzing multivariate time series data
using functional principal component analysis.

高速・安定・正確な多変量関数的主成分分析の実装

Two implementations are provided:
1. MFPCA: Fast implementation optimized for performance
2. TheoreticalMFPCA: Rigorous implementation following FDA literature
"""

from .core import MFPCA
from .core_theoretical import TheoreticalMFPCA
from .preprocessing import FunctionalDataPreprocessor, BSplineSmoother
from .utils import create_time_grid, compute_covariance_surface
from .visualization import MFPCAVisualizer
from .basis import BasisSystem, BSplineBasis, FourierBasis, create_basis

# Try to import Plotly visualizer (optional dependency)
try:
    from .visualization_plotly import (
        MFPCAPlotlyVisualizer,
        plot_mfpca_scores_2d,
        plot_mfpca_scores_3d,
        plot_mfpca_eigenfunctions,
        plot_mfpca_scree
    )
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    MFPCAPlotlyVisualizer = None
    plot_mfpca_scores_2d = None
    plot_mfpca_scores_3d = None
    plot_mfpca_eigenfunctions = None
    plot_mfpca_scree = None

__version__ = "1.2.0"
__author__ = "Data Science Professional"

__all__ = [
    "MFPCA",
    "TheoreticalMFPCA",
    "FunctionalDataPreprocessor",
    "BSplineSmoother",
    "MFPCAVisualizer",
    "MFPCAPlotlyVisualizer",
    "BasisSystem",
    "BSplineBasis",
    "FourierBasis",
    "create_basis",
    "create_time_grid",
    "compute_covariance_surface",
    "plot_mfpca_scores_2d",
    "plot_mfpca_scores_3d",
    "plot_mfpca_eigenfunctions",
    "plot_mfpca_scree",
]
