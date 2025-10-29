"""Quick test to identify the bug."""

import numpy as np
import sys
import traceback

from mfpca import TheoreticalMFPCA

# Generate simple data
np.random.seed(42)
n_samples = 20
n_timepoints = 60
n_variables = 2

time_grid = np.linspace(0, 1, n_timepoints)
X = np.random.randn(n_samples, n_timepoints, n_variables)

print(f"Data shape: {X.shape}")
print(f"Time grid shape: {time_grid.shape}")

try:
    mfpca = TheoreticalMFPCA(
        n_components=3,
        n_basis=15,
        basis_type='bspline',
        smoothing=True,
        center=True
    )

    print("\nFitting...")
    mfpca.fit(X, time_grid)
    print(f"✓ Fit successful")
    print(f"  Eigenvalues: {mfpca.eigenvalues_}")

    print("\nEvaluating eigenfunctions...")
    eigenfuncs = mfpca.get_eigenfunctions(time_grid)
    print(f"✓ Eigenfunction evaluation successful")
    print(f"  Shape: {eigenfuncs.shape}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
