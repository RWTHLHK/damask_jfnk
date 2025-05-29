import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from typing import Tuple, Optional
import logging
from .damask_io import DAMASKSimulation
import sys

def compute_equiv_strain_increment(F_prev: np.ndarray, F_new: np.ndarray) -> float:
    """
    Compute equivalent strain increment between two deformation gradients.

    Args:
        F_prev: Previous deformation gradient
        F_new: New deformation gradient

    Returns:
        Equivalent strain increment
    """
    # Compute incremental deformation gradient
    F_inc = F_new @ np.linalg.inv(F_prev)

    # Compute principal logarithmic strains
    C = F_inc.T @ F_inc  # Right Cauchy-Green tensor
    eigenvals = np.linalg.eigvalsh(C)
    log_strains = 0.5 * np.log(eigenvals)

    # Compute equivalent strain increment using J2 norm
    dev_strains = log_strains - np.mean(log_strains)
    equiv_strain = np.sqrt(2/3 * np.sum(dev_strains**2))

    return equiv_strain

class GMRESTRIAL:
    """
    Helper class to generate unique jobnames for GMRES trial runs.
    """
    def __init__(self, prefix="trialGMRES"):
        self.prefix = prefix
        self.counter = 0
    def next(self):
        self.counter += 1
        return f"{self.prefix}{self.counter}"

def residual(F: np.ndarray,
            F_prev: np.ndarray,
            damask_sim: DAMASKSimulation,
            target_stresses: np.ndarray,
            is_initial: bool = False,
            restart_increment: Optional[int] = None,
            dot_eps_eq: float = 1e-3,
            target_delta_eps: float = 1e-6,
            max_trial_increments: int = 100,
            verbose: bool = True,
            jobname: Optional[str] = None) -> np.ndarray:
    """
    Compute residual between current and target principal stresses.

    Args:
        F: Current deformation gradient (3x3 array)
        F_prev: Previous deformation gradient (3x3 array)
        damask_sim: DAMASKSimulation instance
        target_stresses: Target principal stresses [σ1, σ2, σ3] in descending order
        is_initial: Whether this is the initial step
        restart_increment: Increment to restart from (if not initial)
        dot_eps_eq: Target equivalent strain rate
        target_delta_eps: Target equivalent strain increment per increment
        verbose: Whether to print progress
        jobname: Optional jobname for the trial run

    Returns:
        Residual vector [σ1 - σ1_target, σ2 - σ2_target, σ3 - σ3_target]
    """
    if verbose:
        logger = logging.getLogger(__name__)
        logger.info(f"Computing residual for F: {F.diagonal()}")

    # Compute equivalent strain increment
    delta_eps_eq = compute_equiv_strain_increment(F_prev, F)

    # Compute time step and number of increments
    t = delta_eps_eq / dot_eps_eq
    trial_increments = int(delta_eps_eq / target_delta_eps)
    N = max(2, min(trial_increments, max_trial_increments))
    if verbose:
        logger.info(f"delta_eps_eq: {delta_eps_eq:.6f}")
        logger.info(f"t: {t:.6f}, N: {N}")

    if jobname is None:
        jobname = "trialGMRES"

    # Run DAMASK with current F
    result_file = damask_sim.run_step(
        F=F,
        t=t,
        N=N,
        is_initial=is_initial,
        is_trial=True,
        restart_increment=restart_increment,
        jobname=jobname
    )

    # Get current principal stresses
    current_stresses = damask_sim.postprocess(jobname)

    # Compute residual
    residual_vec = current_stresses - target_stresses

    if verbose:
        logger.info(f"Current stresses: {current_stresses}")
        logger.info(f"Target stresses: {target_stresses}")
        logger.info(f"Residual: {residual_vec}")

    return residual_vec

class JFNKOperator(LinearOperator):
    """
    Jacobian-Free Newton-Krylov operator for computing matrix-vector products
    using finite difference approximation.
    """
    def __init__(self,
                 F0: np.ndarray,
                 F_prev: np.ndarray,
                 damask_sim: DAMASKSimulation,
                 target_stresses: np.ndarray,
                 is_initial: bool = False,
                 restart_increment: Optional[int] = None,
                 F_bounds: Tuple[float, float] = (1e-6, 1.05),
                 dot_eps_eq: float = 1e-3,
                 target_delta_eps: float = 1e-6,
                 eps: float = 1e-6,
                 verbose: bool = True,
                 max_trial_increments: int = 100,
                 trial_counter: Optional[GMRESTRIAL] = None):
        """
        Initialize the JFNK operator.

        Args:
            F0: Current deformation gradient
            F_prev: Previous deformation gradient
            damask_sim: DAMASKSimulation instance
            target_stresses: Target principal stresses
            is_initial: Whether this is the initial step
            restart_increment: Increment to restart from
            F_bounds: Tuple of (min, max) values for F components
            dot_eps_eq: Target equivalent strain rate
            target_delta_eps: Target equivalent strain increment per increment
            eps: Finite difference step size
            verbose: Whether to print progress
            max_trial_increments: Maximum number of trial increments
            trial_counter: Optional GMRESTRIAL instance for generating unique jobnames
        """
        super().__init__(shape=(3, 3), dtype=np.float64)
        self.F0 = F0
        self.F_prev = F_prev
        self.damask_sim = damask_sim
        self.target_stresses = target_stresses
        self.is_initial = is_initial
        self.restart_increment = restart_increment
        self.F_bounds = F_bounds
        self.dot_eps_eq = dot_eps_eq
        self.target_delta_eps = target_delta_eps
        self.eps = eps
        self.verbose = verbose
        self.max_trial_increments = max_trial_increments
        self.trial_counter = trial_counter if trial_counter is not None else GMRESTRIAL()
        if verbose:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Compute matrix-vector product using finite difference approximation.

        Args:
            v: Input vector (reshaped to 3x3 matrix)

        Returns:
            Matrix-vector product
        """
        # v is a vector of length 3 (for the diagonal of F)
        F_plus = self.F0 + self.eps * np.diag(v)
        F_minus = self.F0 - self.eps * np.diag(v)
        F_plus = np.clip(F_plus, self.F_bounds[0], self.F_bounds[1])
        F_minus = np.clip(F_minus, self.F_bounds[0], self.F_bounds[1])
        jobname_plus = self.trial_counter.next()
        jobname_minus = self.trial_counter.next()
        r_plus = residual(
            F_plus, self.F_prev, self.damask_sim, self.target_stresses,
            is_initial=self.is_initial,
            restart_increment=self.restart_increment,
            dot_eps_eq=self.dot_eps_eq,
            target_delta_eps=self.target_delta_eps,
            max_trial_increments=self.max_trial_increments,
            verbose=self.verbose,
            jobname=jobname_plus
        )
        r_minus = residual(
            F_minus, self.F_prev, self.damask_sim, self.target_stresses,
            is_initial=self.is_initial,
            restart_increment=self.restart_increment,
            dot_eps_eq=self.dot_eps_eq,
            target_delta_eps=self.target_delta_eps,
            max_trial_increments=self.max_trial_increments,
            verbose=self.verbose,
            jobname=jobname_minus
        )
        Jv = (r_plus - r_minus) / (2 * self.eps)
        return Jv.flatten()

def gmres_solver(F_diag_init: np.ndarray,
                F_prev: np.ndarray,
                damask_sim: DAMASKSimulation,
                target_stresses: np.ndarray,
                is_initial: bool = False,
                restart_increment: Optional[int] = None,
                F_bounds: Tuple[float, float] = (1e-6, 1.05),
                maxiter: int = 100,
                tol: float = 1e-6,
                dot_eps_eq: float = 1e-3,
                target_delta_eps: float = 1e-6,
                verbose: bool = True,
                max_trial_increments: int = 100) -> Tuple[np.ndarray, bool]:
    """
    Solve for F that gives target principal stresses using GMRES.

    Args:
        F_diag_init: Initial guess for diagonal of F
        F_prev: Previous deformation gradient
        damask_sim: DAMASKSimulation instance
        target_stresses: Target principal stresses [σ1, σ2, σ3] in descending order
        is_initial: Whether this is the initial step
        restart_increment: Increment to restart from (if not initial)
        F_bounds: Tuple of (min, max) values for F components
        maxiter: Maximum number of GMRES iterations
        tol: Tolerance for GMRES convergence
        dot_eps_eq: Target equivalent strain rate
        target_delta_eps: Target equivalent strain increment per increment
        verbose: Whether to print progress
        max_trial_increments: Maximum number of trial increments

    Returns:
        Tuple of (optimal F diagonal, success flag)
    """
    if verbose:
        logger = logging.getLogger(__name__)
        logger.info("Starting GMRES solve...")
        logger.info(f"Initial F diagonal: {F_diag_init}")
        logger.info(f"Target stresses: {target_stresses}")

    # Initial F
    F0 = np.diag(F_diag_init)

    trial_counter = GMRESTRIAL()
    # Initial residual
    r0 = residual(
        F0, F_prev, damask_sim, target_stresses,
        is_initial=is_initial,
        restart_increment=restart_increment,
        dot_eps_eq=dot_eps_eq,
        target_delta_eps=target_delta_eps,
        max_trial_increments=max_trial_increments,
        verbose=verbose,
        jobname=trial_counter.next()
    )
    if np.all(np.abs(r0) < tol):
        if verbose:
            logger.info("Initial guess already satisfies tolerance!")
        return F_diag_init, True

    # Create JFNK operator
    A = JFNKOperator(
        F0=F0,
        F_prev=F_prev,
        damask_sim=damask_sim,
        target_stresses=target_stresses,
        is_initial=is_initial,
        restart_increment=restart_increment,
        F_bounds=F_bounds,
        dot_eps_eq=dot_eps_eq,
        target_delta_eps=target_delta_eps,
        eps=1e-6,
        verbose=verbose,
        max_trial_increments=max_trial_increments,
        trial_counter=trial_counter
    )

    # Solve using GMRES
    try:
        x, info = gmres(
            A,
            -r0,
            maxiter=maxiter,
            rtol=tol
        )
        if info == 0:
            dF_diag = x  # shape (3,)
            F_new = F0 + np.diag(dF_diag)
            F_new = np.clip(F_new, F_bounds[0], F_bounds[1])
            if verbose:
                logger.info("GMRES converged!")
                logger.info(f"Final F diagonal: {F_new.diagonal()}")
            return F_new.diagonal(), True
        else:
            if verbose:
                logger.warning(f"GMRES did not converge. Info: {info}")
            return F_diag_init, False

    except Exception as e:
        if verbose:
            logger.error(f"Error in GMRES solve: {str(e)}")
        return F_diag_init, False
