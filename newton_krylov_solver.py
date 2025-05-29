import numpy as np
from scipy.optimize import newton_krylov
from scipy.optimize import OptimizeResult
import logging
from typing import Tuple, Optional
from pathlib import Path
import json

def compute_residual(F_diag: np.ndarray, simulation, target_stress: np.ndarray) -> np.ndarray:
    """
    Compute residual between current and target principal stresses.

    Args:
        F_diag: Current deformation gradient diagonal
        simulation: DAMASKSimulation instance
        target_stress: Target principal stresses

    Returns:
        Residual vector
    """
    # Run simulation with current F_diag
    simulation.run_trial_step(F_diag)

    # Get current principal stresses
    current_stress = simulation.get_principal_stresses()

    # Compute residual
    residual = current_stress - target_stress

    return residual

def solve(simulation,
          target_stress: np.ndarray,
          F_diag_guess: Optional[np.ndarray] = None,
          maxiter: int = 50,
          f_tol: float = 1e-6,
          f_rtol: float = 1e-6,
          x_tol: float = 1e-6,
          x_rtol: float = 1e-6,
          verbose: bool = True) -> Tuple[np.ndarray, OptimizeResult]:
    """
    Solve for deformation gradient diagonal that achieves target principal stresses
    using Newton-Krylov method.

    Args:
        simulation: DAMASKSimulation instance
        target_stress: Target principal stresses
        F_diag_guess: Initial guess for deformation gradient diagonal
        maxiter: Maximum number of iterations
        f_tol: Absolute tolerance for function value
        f_rtol: Relative tolerance for function value
        x_tol: Absolute tolerance for solution
        x_rtol: Relative tolerance for solution
        verbose: Whether to print progress

    Returns:
        Tuple of (optimal F_diag, optimization result)
    """
    if F_diag_guess is None:
        F_diag_guess = np.ones(3)  # Start with identity deformation

    def residual_func(F_diag):
        return compute_residual(F_diag, simulation, target_stress)

    # Run Newton-Krylov optimization
    result = newton_krylov(
        residual_func,
        F_diag_guess,
        method='gmres',
        maxiter=maxiter,
        f_tol=f_tol,
        f_rtol=f_rtol,
        x_tol=x_tol,
        x_rtol=x_rtol,
        verbose=verbose
    )

    return result, OptimizeResult(
        x=result,
        success=True,
        message="Optimization completed successfully",
        fun=residual_func(result),
        nfev=simulation.n_trial_steps,
        nit=result.nit if hasattr(result, 'nit') else None
    )

def save_results(result: OptimizeResult, output_dir: Path):
    """Save optimization results to file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save optimization results
    results = {
        'success': result.success,
        'message': result.message,
        'nfev': result.nfev,
        'nit': result.nit,
        'fun': result.fun.tolist() if isinstance(result.fun, np.ndarray) else result.fun,
        'x': result.x.tolist() if isinstance(result.x, np.ndarray) else result.x
    }

    with open(output_dir / 'optimization_results.json', 'w') as f:
        json.dump(results, f, indent=4)
