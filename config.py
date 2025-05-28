from dataclasses import dataclass
import numpy as np

@dataclass
class DamaskSimConfig:
    """
    Configuration for a DAMASK simulation run.
    Contains all file paths and initial deformation state.
    run_increments: Number of increments for production (正式run) steps, fixed value.
    Per-step parameters (t, N) should be passed directly to run_step, not stored here.
    """
    workdir: str           # Working directory for DAMASK input/output
    logsdir: str           # Directory for log files
    load_yaml: str         # Path to the initial load.yaml
    grid_file: str         # Path to the grid.vti file
    material_file: str     # Path to the material.yaml file
    base_jobname: str      # Base job name for DAMASK output files
    F_init: np.ndarray     # 3x3 initial total deformation gradient
    run_increments: int    # Number of increments for production (正式run) steps, fixed

@dataclass
class JFNKConfig:
    """
    Configuration for the JFNK optimization process.
    Contains all target values and optimization settings.
    F_diag is always constrained to [-1.05, 1.05] for each component.
    min_trial_increments/max_trial_increments bound the number of increments per trial step.
    """
    target_triax: float        # Target stress triaxiality
    target_lode: float         # Target Lode angle
    min_trial_increments: int  # Minimum increments for trial steps
    max_trial_increments: int  # Maximum increments for trial steps
    tol: float                 # Tolerance for convergence
    max_iter: int              # Maximum number of JFNK iterations

@dataclass
class SimulationState:
    """
    State of the current simulation, including the current deformation gradient and step index.
    """
    F_current: np.ndarray      # 3x3 current total deformation gradient
    step: int = 0              # Current macro step index
