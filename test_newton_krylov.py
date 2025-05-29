import numpy as np
from pathlib import Path
import logging
from newton_krylov_solver import solve as newton_krylov_solve
from ls_solver import solve as ls_solve
from damask_simulation import DAMASKSimulation

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('newton_krylov_test.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Initialize simulation
    simulation = DAMASKSimulation(
        workdir=Path('workdir'),
        jobname='newton_krylov_test',
        load_file=Path('load.yaml')
    )

    # Test cases with different stress levels
    test_cases = [
        {
            'name': 'low_stress',
            'target_stress': np.array([100.0, 50.0, 0.0])  # Low stress level
        },
        {
            'name': 'medium_stress',
            'target_stress': np.array([500.0, 250.0, 0.0])  # Medium stress level
        },
        {
            'name': 'high_stress',
            'target_stress': np.array([1000.0, 500.0, 0.0])  # High stress level
        }
    ]

    # Run tests
    for case in test_cases:
        logger.info(f"\nTesting case: {case['name']}")
        logger.info(f"Target stress: {case['target_stress']}")

        # Run Newton-Krylov solver
        logger.info("\nRunning Newton-Krylov solver...")
        newton_result, newton_opt_result = newton_krylov_solve(
            simulation=simulation,
            target_stress=case['target_stress'],
            maxiter=50,
            f_tol=1e-6,
            f_rtol=1e-6,
            x_tol=1e-6,
            x_rtol=1e-6,
            verbose=True
        )
        logger.info(f"Newton-Krylov result: {newton_result}")
        logger.info(f"Number of function evaluations: {newton_opt_result.nfev}")
        logger.info(f"Final residual: {newton_opt_result.fun}")

        # Run Least Squares solver
        logger.info("\nRunning Least Squares solver...")
        ls_result, ls_opt_result = ls_solve(
            simulation=simulation,
            target_stress=case['target_stress'],
            maxiter=50,
            ftol=1e-6,
            xtol=1e-6,
            verbose=True
        )
        logger.info(f"Least Squares result: {ls_result}")
        logger.info(f"Number of function evaluations: {ls_opt_result.nfev}")
        logger.info(f"Final residual: {ls_opt_result.fun}")

        # Compare results
        logger.info("\nComparison:")
        logger.info(f"Residual difference: {np.linalg.norm(newton_opt_result.fun - ls_opt_result.fun)}")
        logger.info(f"Solution difference: {np.linalg.norm(newton_result - ls_result)}")
        logger.info(f"Function evaluations ratio (NK/LS): {newton_opt_result.nfev / ls_opt_result.nfev}")

if __name__ == '__main__':
    main()
