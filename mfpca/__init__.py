"""
MFPCA - Multivariate Functional Principal Component Analysis

A professional implementation for analyzing multivariate time series data
using functional principal component analysis.

高速・安定・正確な多変量関数的主成分分析の実装
"""

from .core import MFPCA
from .preprocessing import FunctionalDataPreprocessor, BSplineSmoother
from .utils import create_time_grid, compute_covariance_surface
from .visualization import MFPCAVisualizer

__version__ = "1.0.0"
__author__ = "Data Science Professional"

__all__ = [
    "MFPCA",
    "FunctionalDataPreprocessor",
    "BSplineSmoother",
    "MFPCAVisualizer",
    "create_time_grid",
    "compute_covariance_surface",
]
