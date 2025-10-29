"""
Basis function systems for functional data analysis.

関数データ解析のための基底関数系
Based on Ramsay & Silverman (2005) "Functional Data Analysis"
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import eval_legendre
from typing import Optional, Tuple
import warnings


class BasisSystem:
    """
    Base class for basis function systems.

    基底関数系の基底クラス
    """

    def __init__(self, n_basis: int, domain: Tuple[float, float] = (0, 1)):
        """
        Parameters
        ----------
        n_basis : int
            Number of basis functions
        domain : tuple
            Domain of the basis functions (default: (0, 1))
        """
        self.n_basis = n_basis
        self.domain = domain

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate basis functions at time points.

        Parameters
        ----------
        t : ndarray of shape (n_points,)
            Time points

        Returns
        -------
        basis_matrix : ndarray of shape (n_points, n_basis)
            Evaluated basis functions
        """
        raise NotImplementedError

    def gram_matrix(self, t: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Gram matrix: G_ij = ∫ B_i(t) B_j(t) dt

        グラム行列の計算（数値積分）

        Parameters
        ----------
        t : ndarray
            Time points for numerical integration
        weights : ndarray, optional
            Integration weights (default: trapezoidal rule)

        Returns
        -------
        G : ndarray of shape (n_basis, n_basis)
            Gram matrix
        """
        B = self.evaluate(t)

        if weights is None:
            # Trapezoidal rule weights
            weights = np.ones(len(t))
            weights[0] = weights[-1] = 0.5
            dt = (t[-1] - t[0]) / (len(t) - 1)
            weights *= dt

        # G_ij = sum_t B_i(t) B_j(t) w_t
        G = B.T @ np.diag(weights) @ B
        return G

    def penalty_matrix(self, t: np.ndarray, order: int = 2) -> np.ndarray:
        """
        Compute penalty matrix for derivative: P_ij = ∫ D^m B_i(t) D^m B_j(t) dt

        微分ペナルティ行列の計算

        Parameters
        ----------
        t : ndarray
            Time points
        order : int
            Order of derivative (default: 2, second derivative)

        Returns
        -------
        P : ndarray of shape (n_basis, n_basis)
            Penalty matrix
        """
        raise NotImplementedError


class BSplineBasis(BasisSystem):
    """
    B-spline basis system.

    B-spline基底系
    Ramsay & Silverman (2005), Chapter 3
    """

    def __init__(
        self,
        n_basis: int,
        degree: int = 3,
        domain: Tuple[float, float] = (0, 1),
        knots: Optional[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        n_basis : int
            Number of basis functions
        degree : int
            Degree of B-splines (default: 3, cubic)
        domain : tuple
            Domain
        knots : ndarray, optional
            Knot sequence (default: equally spaced interior knots)
        """
        super().__init__(n_basis, domain)
        self.degree = degree

        if knots is None:
            # Create equally spaced interior knots
            n_interior = n_basis - degree - 1
            if n_interior < 0:
                raise ValueError(f"n_basis must be >= degree + 1 = {degree + 1}")

            # Knot sequence with multiplicity at boundaries
            interior_knots = np.linspace(domain[0], domain[1], n_interior + 2)
            self.knots = np.concatenate([
                np.repeat(domain[0], degree + 1),
                interior_knots[1:-1],
                np.repeat(domain[1], degree + 1)
            ])
        else:
            self.knots = knots

        # Validate
        if len(self.knots) != n_basis + degree + 1:
            raise ValueError(
                f"Knot sequence length must be n_basis + degree + 1 = {n_basis + degree + 1}"
            )

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate B-spline basis functions.

        B-spline基底関数の評価

        Returns
        -------
        B : ndarray of shape (len(t), n_basis)
            Basis matrix
        """
        t = np.asarray(t)
        B = np.zeros((len(t), self.n_basis))

        for i in range(self.n_basis):
            # Extract knots for i-th basis function
            knots_i = self.knots[i:i + self.degree + 2]

            # Create B-spline object (coefficients: 1 at i-th position)
            c = np.zeros(self.n_basis)
            c[i] = 1.0

            try:
                # Evaluate using scipy's BSpline
                bspl = BSpline(self.knots, c, self.degree, extrapolate=False)
                B[:, i] = bspl(t)
                # Replace NaN (outside domain) with 0
                B[:, i] = np.nan_to_num(B[:, i], nan=0.0)
            except:
                # Fallback: manual evaluation using de Boor's algorithm
                B[:, i] = self._evaluate_bspline_manual(t, i)

        return B

    def _evaluate_bspline_manual(self, t: np.ndarray, basis_idx: int) -> np.ndarray:
        """Manual B-spline evaluation using Cox-de Boor recursion."""
        t = np.asarray(t)
        result = np.zeros_like(t)

        # Cox-de Boor recursion
        # B_{i,0}(t) = 1 if t_i <= t < t_{i+1}, else 0
        # B_{i,k}(t) = (t - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(t)
        #            + (t_{i+k+1} - t)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(t)

        for j, t_val in enumerate(t):
            result[j] = self._cox_de_boor(t_val, basis_idx, self.degree)

        return result

    def _cox_de_boor(self, t: float, i: int, k: int) -> float:
        """Cox-de Boor recursion for B-spline evaluation."""
        if k == 0:
            return 1.0 if self.knots[i] <= t < self.knots[i + 1] else 0.0

        # Left term
        denom1 = self.knots[i + k] - self.knots[i]
        left = 0.0
        if denom1 > 1e-10:
            left = (t - self.knots[i]) / denom1 * self._cox_de_boor(t, i, k - 1)

        # Right term
        denom2 = self.knots[i + k + 1] - self.knots[i + 1]
        right = 0.0
        if denom2 > 1e-10:
            right = (self.knots[i + k + 1] - t) / denom2 * self._cox_de_boor(t, i + 1, k - 1)

        return left + right

    def penalty_matrix(self, t: np.ndarray, order: int = 2) -> np.ndarray:
        """
        Compute roughness penalty matrix.

        P_ij = ∫ [D^m B_i(t)] [D^m B_j(t)] dt

        where D^m is the m-th derivative operator.
        """
        # For simplicity, use numerical differentiation and integration
        B = self.evaluate(t)

        # Compute derivatives numerically
        if order == 0:
            D_B = B
        else:
            D_B = B.copy()
            for _ in range(order):
                D_B = np.gradient(D_B, t, axis=0)

        # Integrate: P = ∫ D^m B^T D^m B dt
        weights = np.ones(len(t))
        weights[0] = weights[-1] = 0.5
        dt = (t[-1] - t[0]) / (len(t) - 1)
        weights *= dt

        P = D_B.T @ np.diag(weights) @ D_B

        return P


class FourierBasis(BasisSystem):
    """
    Fourier basis system.

    フーリエ基底系
    Ramsay & Silverman (2005), Chapter 3
    """

    def __init__(
        self,
        n_basis: int,
        period: Optional[float] = None,
        domain: Tuple[float, float] = (0, 1)
    ):
        """
        Parameters
        ----------
        n_basis : int
            Number of basis functions (should be odd for symmetry)
        period : float, optional
            Period of the basis (default: domain length)
        domain : tuple
            Domain
        """
        super().__init__(n_basis, domain)

        if period is None:
            self.period = domain[1] - domain[0]
        else:
            self.period = period

        # n_basis = 1 (constant) + 2k (k pairs of sin/cos)
        # For odd n_basis: k = (n_basis - 1) // 2
        # For even n_basis: add one more sine
        self.n_pairs = (n_basis - 1) // 2

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate Fourier basis functions.

        B_0(t) = 1/sqrt(T)  (constant)
        B_{2k-1}(t) = sqrt(2/T) * sin(2πkt/T)
        B_{2k}(t) = sqrt(2/T) * cos(2πkt/T)

        Returns
        -------
        B : ndarray of shape (len(t), n_basis)
            Basis matrix
        """
        t = np.asarray(t)
        B = np.zeros((len(t), self.n_basis))

        # Constant term
        B[:, 0] = 1.0 / np.sqrt(self.period)

        # Sine and cosine terms
        idx = 1
        for k in range(1, self.n_pairs + 1):
            omega = 2 * np.pi * k / self.period

            # Sine term
            if idx < self.n_basis:
                B[:, idx] = np.sqrt(2.0 / self.period) * np.sin(omega * t)
                idx += 1

            # Cosine term
            if idx < self.n_basis:
                B[:, idx] = np.sqrt(2.0 / self.period) * np.cos(omega * t)
                idx += 1

        return B

    def penalty_matrix(self, t: np.ndarray, order: int = 2) -> np.ndarray:
        """
        Compute roughness penalty matrix for Fourier basis.

        For Fourier basis, derivatives are analytical:
        D^m sin(ωt) = ω^m sin(ωt + mπ/2)
        D^m cos(ωt) = ω^m cos(ωt + mπ/2)
        """
        P = np.zeros((self.n_basis, self.n_basis))

        # Constant term: derivative is 0
        P[0, 0] = 0.0

        # For sine/cosine pairs
        idx = 1
        for k in range(1, self.n_pairs + 1):
            omega = 2 * np.pi * k / self.period
            penalty_value = (omega ** order) ** 2 * self.period / 2.0

            if idx < self.n_basis:
                P[idx, idx] = penalty_value  # Sine
                idx += 1
            if idx < self.n_basis:
                P[idx, idx] = penalty_value  # Cosine
                idx += 1

        return P


def create_basis(
    basis_type: str,
    n_basis: int,
    domain: Tuple[float, float] = (0, 1),
    **kwargs
) -> BasisSystem:
    """
    Factory function for creating basis systems.

    基底系のファクトリー関数

    Parameters
    ----------
    basis_type : str
        Type of basis: 'bspline', 'fourier'
    n_basis : int
        Number of basis functions
    domain : tuple
        Domain
    **kwargs
        Additional arguments for specific basis types

    Returns
    -------
    basis : BasisSystem
        Basis system object
    """
    if basis_type.lower() == 'bspline':
        degree = kwargs.get('degree', 3)
        return BSplineBasis(n_basis, degree=degree, domain=domain)

    elif basis_type.lower() == 'fourier':
        period = kwargs.get('period', None)
        return FourierBasis(n_basis, period=period, domain=domain)

    else:
        raise ValueError(f"Unknown basis type: {basis_type}")
