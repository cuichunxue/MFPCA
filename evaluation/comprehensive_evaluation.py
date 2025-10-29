"""
Comprehensive third-party evaluation of MFPCA implementations.

第三者による包括的なMFPCA実装の評価

This script objectively evaluates both MFPCA implementations on various
data patterns and theoretical properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mfpca import MFPCA, TheoreticalMFPCA
from mfpca.utils import generate_synthetic_mfpca_data

warnings.filterwarnings('ignore')


class MFPCAEvaluator:
    """
    Independent evaluator for MFPCA implementations.

    第三者評価クラス
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.results = {}

    def generate_data_pattern(
        self,
        pattern: str,
        n_samples: int = 100,
        n_timepoints: int = 50,
        n_variables: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate different data patterns for testing.

        Parameters
        ----------
        pattern : str
            One of: 'smooth', 'periodic', 'trend', 'noisy', 'sparse', 'mixed'
        """
        time_grid = np.linspace(0, 1, n_timepoints)
        X = np.zeros((n_samples, n_timepoints, n_variables))

        if pattern == 'smooth':
            # Smooth sinusoidal patterns
            for i in range(n_samples):
                for j in range(n_variables):
                    freq = 2 + np.random.randn() * 0.2
                    phase = np.random.uniform(0, 2*np.pi)
                    amplitude = 1 + np.random.randn() * 0.2
                    X[i, :, j] = amplitude * np.sin(2*np.pi*freq*time_grid + phase)
            # Add small noise
            X += np.random.randn(*X.shape) * 0.05

        elif pattern == 'periodic':
            # Clear periodic structure
            for i in range(n_samples):
                for j in range(n_variables):
                    # Mixture of frequencies
                    X[i, :, j] = (
                        np.sin(2*np.pi*2*time_grid) * (1 + 0.2*np.random.randn()) +
                        0.5*np.cos(2*np.pi*4*time_grid) * (1 + 0.2*np.random.randn()) +
                        0.3*np.sin(2*np.pi*6*time_grid) * (1 + 0.2*np.random.randn())
                    )
            X += np.random.randn(*X.shape) * 0.1

        elif pattern == 'trend':
            # Linear and nonlinear trends
            for i in range(n_samples):
                for j in range(n_variables):
                    trend_type = np.random.choice(['linear', 'quadratic', 'cubic'])
                    if trend_type == 'linear':
                        slope = np.random.uniform(-2, 2)
                        intercept = np.random.uniform(-1, 1)
                        X[i, :, j] = slope * time_grid + intercept
                    elif trend_type == 'quadratic':
                        a = np.random.uniform(-2, 2)
                        b = np.random.uniform(-1, 1)
                        c = np.random.uniform(-1, 1)
                        X[i, :, j] = a*time_grid**2 + b*time_grid + c
                    else:  # cubic
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(-1, 1)
                        c = np.random.uniform(-1, 1)
                        d = np.random.uniform(-1, 1)
                        X[i, :, j] = a*time_grid**3 + b*time_grid**2 + c*time_grid + d
            X += np.random.randn(*X.shape) * 0.2

        elif pattern == 'noisy':
            # High noise level
            for i in range(n_samples):
                for j in range(n_variables):
                    freq = 3 + np.random.randn() * 0.5
                    X[i, :, j] = np.sin(2*np.pi*freq*time_grid)
            # Heavy noise
            X += np.random.randn(*X.shape) * 0.5

        elif pattern == 'sparse':
            # Sparse with localized features
            for i in range(n_samples):
                for j in range(n_variables):
                    # Random bumps
                    n_bumps = np.random.randint(2, 5)
                    for _ in range(n_bumps):
                        center = np.random.uniform(0.2, 0.8)
                        width = np.random.uniform(0.05, 0.15)
                        amplitude = np.random.uniform(0.5, 2.0)
                        X[i, :, j] += amplitude * np.exp(-((time_grid - center)/width)**2)
            X += np.random.randn(*X.shape) * 0.1

        elif pattern == 'mixed':
            # Mixed patterns
            for i in range(n_samples):
                pattern_type = i % 3
                for j in range(n_variables):
                    if pattern_type == 0:  # Periodic
                        X[i, :, j] = np.sin(2*np.pi*3*time_grid + np.random.uniform(0, 2*np.pi))
                    elif pattern_type == 1:  # Trend
                        X[i, :, j] = 2*time_grid + np.random.uniform(-1, 1)
                    else:  # Localized
                        center = np.random.uniform(0.3, 0.7)
                        X[i, :, j] = 2*np.exp(-((time_grid - center)/0.2)**2)
            X += np.random.randn(*X.shape) * 0.15

        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return X, time_grid

    def evaluate_theoretical_properties(
        self,
        mfpca,
        X: np.ndarray,
        time_grid: np.ndarray,
        implementation: str
    ) -> Dict:
        """
        Evaluate theoretical properties of MFPCA.

        理論的性質の評価
        """
        results = {}

        # Get eigenfunctions
        if implementation == 'theoretical':
            eigenfuncs = mfpca.get_eigenfunctions(time_grid)
        else:
            eigenfuncs = mfpca.eigenfunctions_

        n_comp = eigenfuncs.shape[0]

        # 1. Orthonormality of eigenfunctions: <φ_i, φ_j> = δ_ij
        inner_products = np.zeros((n_comp, n_comp))
        for i in range(n_comp):
            for j in range(n_comp):
                integrand = np.sum(eigenfuncs[i] * eigenfuncs[j], axis=1)
                inner_products[i, j] = np.trapz(integrand, time_grid)

        identity = np.eye(n_comp)
        orthonormality_error = np.linalg.norm(inner_products - identity, 'fro')
        results['orthonormality_error'] = orthonormality_error
        results['inner_products'] = inner_products

        # 2. Uncorrelatedness of scores: Corr(ξ_i, ξ_j) ≈ 0 for i ≠ j
        scores = mfpca.scores_
        score_corr = np.corrcoef(scores.T)
        np.fill_diagonal(score_corr, 0)
        max_off_diag_corr = np.max(np.abs(score_corr))
        results['max_score_correlation'] = max_off_diag_corr
        results['score_correlation'] = score_corr

        # 3. Variance decomposition: sum(λ_k) / total_variance
        if implementation == 'theoretical':
            var_ratio = mfpca.explained_variance_ratio()
        else:
            var_ratio = mfpca.variance_explained_ratio_
        results['variance_explained_ratio'] = var_ratio
        results['cumulative_variance'] = np.cumsum(var_ratio)
        results['total_variance_explained'] = np.sum(var_ratio)

        # 4. Eigenvalues in descending order
        eigenvalues = mfpca.eigenvalues_
        is_descending = np.all(np.diff(eigenvalues) <= 1e-10)
        results['eigenvalues_descending'] = is_descending
        results['eigenvalues'] = eigenvalues

        return results

    def evaluate_reconstruction(
        self,
        mfpca,
        X: np.ndarray,
        n_components_list: List[int]
    ) -> Dict:
        """
        Evaluate reconstruction quality.

        再構成品質の評価
        """
        results = {}

        for n_comp in n_components_list:
            if n_comp > len(mfpca.eigenvalues_):
                continue

            # Reconstruction error
            if hasattr(mfpca, 'get_reconstruction_error'):
                rmse = mfpca.get_reconstruction_error(X, n_comp)
            else:
                X_recon = mfpca.reconstruct_partial(X, n_comp)
                rmse = np.sqrt(np.mean((X - X_recon)**2))

            results[f'{n_comp}_components'] = {
                'rmse': rmse,
                'n_components': n_comp
            }

        return results

    def evaluate_implementation(
        self,
        X: np.ndarray,
        time_grid: np.ndarray,
        implementation: str,
        n_components: int = 5,
        **kwargs
    ) -> Dict:
        """
        Evaluate one implementation.

        1つの実装を評価
        """
        results = {
            'implementation': implementation,
            'data_shape': X.shape
        }

        # Fit model and measure time
        start_time = time.time()

        try:
            if implementation == 'fast':
                mfpca = MFPCA(
                    n_components=n_components,
                    smoothing=kwargs.get('smoothing', True),
                    center=True
                )
            else:  # theoretical
                mfpca = TheoreticalMFPCA(
                    n_components=n_components,
                    n_basis=kwargs.get('n_basis', 15),
                    basis_type=kwargs.get('basis_type', 'bspline'),
                    smoothing=kwargs.get('smoothing', True),
                    center=True
                )

            mfpca.fit(X, time_grid)

            fit_time = time.time() - start_time
            results['fit_time'] = fit_time
            results['success'] = True

            # Theoretical properties
            theory_results = self.evaluate_theoretical_properties(
                mfpca, X, time_grid, implementation
            )
            results['theoretical_properties'] = theory_results

            # Reconstruction quality
            recon_results = self.evaluate_reconstruction(
                mfpca, X, [1, 2, 3, n_components]
            )
            results['reconstruction'] = recon_results

            # Memory footprint (approximate)
            results['n_parameters'] = (
                np.prod(mfpca.eigenfunctions_.shape if hasattr(mfpca, 'eigenfunctions_')
                       else mfpca.get_eigenfunctions(time_grid).shape) +
                np.prod(mfpca.eigenvalues_.shape) +
                np.prod(mfpca.scores_.shape)
            )

        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            print(f"Error in {implementation}: {e}")

        return results

    def run_evaluation(self, verbose: bool = True) -> Dict:
        """
        Run comprehensive evaluation.

        包括的な評価を実行
        """
        if verbose:
            print("=" * 80)
            print("COMPREHENSIVE MFPCA EVALUATION")
            print("第三者による包括的評価")
            print("=" * 80)

        patterns = ['smooth', 'periodic', 'trend', 'noisy', 'sparse', 'mixed']

        all_results = {}

        for pattern in patterns:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Testing Pattern: {pattern.upper()}")
                print(f"{'='*80}")

            # Generate data
            X, time_grid = self.generate_data_pattern(pattern, n_samples=100, n_timepoints=60)

            pattern_results = {
                'data_pattern': pattern,
                'data_shape': X.shape,
                'implementations': {}
            }

            # Test both implementations
            for impl in ['fast', 'theoretical']:
                if verbose:
                    print(f"\n{impl.upper()} Implementation:")
                    print("-" * 40)

                impl_results = self.evaluate_implementation(
                    X, time_grid, impl, n_components=5,
                    smoothing=True, n_basis=15
                )

                if impl_results['success']:
                    if verbose:
                        print(f"✓ Fit time: {impl_results['fit_time']:.3f}s")

                        theory = impl_results['theoretical_properties']
                        print(f"✓ Orthonormality error: {theory['orthonormality_error']:.6f}")
                        print(f"✓ Max score correlation: {theory['max_score_correlation']:.6f}")
                        print(f"✓ Total variance explained: {theory['total_variance_explained']:.4f}")
                        print(f"✓ Eigenvalues descending: {theory['eigenvalues_descending']}")

                        recon = impl_results['reconstruction']
                        if '3_components' in recon:
                            print(f"✓ Reconstruction RMSE (3 comp): {recon['3_components']['rmse']:.6f}")
                else:
                    if verbose:
                        print(f"✗ Failed: {impl_results.get('error', 'Unknown error')}")

                pattern_results['implementations'][impl] = impl_results

            all_results[pattern] = pattern_results

        self.results = all_results
        return all_results

    def generate_report(self) -> str:
        """
        Generate evaluation report.

        評価レポートを生成
        """
        if not self.results:
            return "No results available. Run evaluation first."

        report = []
        report.append("=" * 80)
        report.append("MFPCA IMPLEMENTATION EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary table
        report.append("SUMMARY TABLE")
        report.append("-" * 80)
        report.append(f"{'Pattern':<12} {'Impl':<12} {'Time(s)':<10} {'Orthog':<10} "
                     f"{'ScoreCorr':<12} {'VarExp':<10} {'RMSE(3)':<10}")
        report.append("-" * 80)

        for pattern, pattern_results in self.results.items():
            for impl, impl_results in pattern_results['implementations'].items():
                if impl_results['success']:
                    theory = impl_results['theoretical_properties']
                    recon = impl_results['reconstruction']
                    rmse_3 = recon.get('3_components', {}).get('rmse', float('nan'))

                    report.append(
                        f"{pattern:<12} {impl:<12} "
                        f"{impl_results['fit_time']:<10.3f} "
                        f"{theory['orthonormality_error']:<10.6f} "
                        f"{theory['max_score_correlation']:<12.6f} "
                        f"{theory['total_variance_explained']:<10.4f} "
                        f"{rmse_3:<10.6f}"
                    )

        report.append("")
        report.append("")

        # Detailed analysis
        report.append("DETAILED ANALYSIS BY PATTERN")
        report.append("=" * 80)

        for pattern, pattern_results in self.results.items():
            report.append("")
            report.append(f"Pattern: {pattern.upper()}")
            report.append("-" * 40)

            for impl, impl_results in pattern_results['implementations'].items():
                if not impl_results['success']:
                    report.append(f"{impl}: FAILED - {impl_results.get('error', 'Unknown')}")
                    continue

                report.append(f"\n{impl.upper()}:")
                theory = impl_results['theoretical_properties']

                report.append(f"  Fit time: {impl_results['fit_time']:.3f}s")
                report.append(f"  Orthonormality: {theory['orthonormality_error']:.6f}")
                report.append(f"  Score correlation: {theory['max_score_correlation']:.6f}")
                report.append(f"  Variance explained: {theory['cumulative_variance'][:3]}")
                report.append(f"  Top 3 eigenvalues: {theory['eigenvalues'][:3]}")

                recon = impl_results['reconstruction']
                report.append(f"  Reconstruction errors:")
                for key, val in recon.items():
                    report.append(f"    {key}: {val['rmse']:.6f}")

        report.append("")
        report.append("=" * 80)
        report.append("EVALUATION COMPLETE")
        report.append("=" * 80)

        return "\n".join(report)

    def plot_comparison(self, save_path: str = 'evaluation_comparison.png'):
        """
        Create comparison plots.

        比較プロットを作成
        """
        if not self.results:
            print("No results to plot. Run evaluation first.")
            return

        patterns = list(self.results.keys())
        n_patterns = len(patterns)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Collect data
        fast_times = []
        theory_times = []
        fast_orthog = []
        theory_orthog = []
        fast_rmse = []
        theory_rmse = []
        fast_var_exp = []
        theory_var_exp = []

        for pattern in patterns:
            impl_results = self.results[pattern]['implementations']

            if 'fast' in impl_results and impl_results['fast']['success']:
                fast_times.append(impl_results['fast']['fit_time'])
                fast_orthog.append(impl_results['fast']['theoretical_properties']['orthonormality_error'])
                fast_rmse.append(impl_results['fast']['reconstruction'].get('3_components', {}).get('rmse', 0))
                fast_var_exp.append(impl_results['fast']['theoretical_properties']['total_variance_explained'])
            else:
                fast_times.append(np.nan)
                fast_orthog.append(np.nan)
                fast_rmse.append(np.nan)
                fast_var_exp.append(np.nan)

            if 'theoretical' in impl_results and impl_results['theoretical']['success']:
                theory_times.append(impl_results['theoretical']['fit_time'])
                theory_orthog.append(impl_results['theoretical']['theoretical_properties']['orthonormality_error'])
                theory_rmse.append(impl_results['theoretical']['reconstruction'].get('3_components', {}).get('rmse', 0))
                theory_var_exp.append(impl_results['theoretical']['theoretical_properties']['total_variance_explained'])
            else:
                theory_times.append(np.nan)
                theory_orthog.append(np.nan)
                theory_rmse.append(np.nan)
                theory_var_exp.append(np.nan)

        x = np.arange(n_patterns)
        width = 0.35

        # Plot 1: Computation time
        ax = axes[0]
        ax.bar(x - width/2, fast_times, width, label='Fast', alpha=0.8)
        ax.bar(x + width/2, theory_times, width, label='Theoretical', alpha=0.8)
        ax.set_xlabel('Data Pattern', fontsize=11)
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_title('Computation Time Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Plot 2: Orthonormality error
        ax = axes[1]
        ax.bar(x - width/2, fast_orthog, width, label='Fast', alpha=0.8)
        ax.bar(x + width/2, theory_orthog, width, label='Theoretical', alpha=0.8)
        ax.set_xlabel('Data Pattern', fontsize=11)
        ax.set_ylabel('Frobenius Norm Error', fontsize=11)
        ax.set_title('Orthonormality Error ||<φ_i,φ_j> - I||', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.set_yscale('log')

        # Plot 3: Reconstruction error
        ax = axes[2]
        ax.bar(x - width/2, fast_rmse, width, label='Fast', alpha=0.8)
        ax.bar(x + width/2, theory_rmse, width, label='Theoretical', alpha=0.8)
        ax.set_xlabel('Data Pattern', fontsize=11)
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title('Reconstruction Error (3 components)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Plot 4: Variance explained
        ax = axes[3]
        ax.bar(x - width/2, fast_var_exp, width, label='Fast', alpha=0.8)
        ax.bar(x + width/2, theory_var_exp, width, label='Theoretical', alpha=0.8)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, label='Perfect (1.0)')
        ax.set_xlabel('Data Pattern', fontsize=11)
        ax.set_ylabel('Total Variance Explained', fontsize=11)
        ax.set_title('Variance Decomposition', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim([0.9, 1.05])

        # Plot 5: Speed ratio
        ax = axes[4]
        speed_ratio = np.array(theory_times) / np.array(fast_times)
        colors = ['green' if r < 2 else 'orange' if r < 5 else 'red' for r in speed_ratio]
        ax.bar(x, speed_ratio, color=colors, alpha=0.8)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Same speed')
        ax.set_xlabel('Data Pattern', fontsize=11)
        ax.set_ylabel('Theoretical / Fast', fontsize=11)
        ax.set_title('Speed Ratio (lower is better for Theoretical)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Plot 6: Overall score (lower is better)
        ax = axes[5]
        # Normalized scores: orthog_error + rmse
        fast_score = np.array(fast_orthog) * 100 + np.array(fast_rmse)
        theory_score = np.array(theory_orthog) * 100 + np.array(theory_rmse)

        ax.bar(x - width/2, fast_score, width, label='Fast', alpha=0.8)
        ax.bar(x + width/2, theory_score, width, label='Theoretical', alpha=0.8)
        ax.set_xlabel('Data Pattern', fontsize=11)
        ax.set_ylabel('Combined Score (lower=better)', fontsize=11)
        ax.set_title('Overall Quality Score', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")

        return fig


def main():
    """Run comprehensive evaluation."""
    print("\n" + "=" * 80)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("Independent third-party assessment of MFPCA implementations")
    print("=" * 80)

    # Create evaluator
    evaluator = MFPCAEvaluator(random_seed=42)

    # Run evaluation
    results = evaluator.run_evaluation(verbose=True)

    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING FINAL REPORT")
    print("=" * 80)
    report = evaluator.generate_report()
    print("\n" + report)

    # Save report to file
    with open('evaluation_report.txt', 'w') as f:
        f.write(report)
    print("\nReport saved to: evaluation_report.txt")

    # Create comparison plots
    print("\nCreating comparison plots...")
    evaluator.plot_comparison('evaluation_comparison.png')

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - evaluation_report.txt")
    print("  - evaluation_comparison.png")
    print("\n")


if __name__ == "__main__":
    main()
