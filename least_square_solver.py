import numpy as np
from scipy.optimize import least_squares

# --- Residual and helper function (self-contained) ---
def compute_equiv_strain_increment(F_old, F_new):
    """
    Compute the equivalent strain increment between two deformation gradients.
    Uses principal logarithmic strain difference and J2 norm.
    """
    F_inc = F_new @ np.linalg.inv(F_old)
    U, s, Vh = np.linalg.svd(F_inc)
    logV = np.log(s)
    eqv_strain = np.sqrt(2/3 * np.sum(logV**2))
    return eqv_strain


def residual(
    F_diag,                      # Diagonal of new F
    F_prev,                      # Previous F (3x3)
    damask_sim,                  # DAMASKSimulation instance
    target_stresses,             # Target principal stresses (descending order)
    is_initial,                  # bool, whether this is the first step
    restart_increment,           # int, for --restart (None if is_initial)
    jobname,                     # str, unique jobname for this trial
    dot_eps_eq,                  # float, target equivalent strain rate
    target_delta_eps,            # float, target equivalent strain increment per increment (for trial runs)
    min_trial_increments=2,
    max_trial_increments=100
):
    """
    Compute the residual vector for principal stresses: [sigma1-sigma1_target, sigma2-sigma2_target, sigma3-sigma3_target].
    """
    F_new = np.diag(F_diag)
    # 1. Compute equivalent strain increment
    delta_eps_eq = compute_equiv_strain_increment(F_prev, F_new)
    # 2. Compute loading time t for this step
    N = max(min_trial_increments, min(max_trial_increments, int(np.ceil(delta_eps_eq / target_delta_eps))))
    t = delta_eps_eq / dot_eps_eq
    # 3. Run DAMASK trial step
    damask_sim.run_step(
        F=F_new, t=t, N=N,
        is_initial=is_initial, is_trial=True,
        restart_increment=restart_increment,
        jobname=jobname
    )
    # 4. Postprocess: get principal stresses (descending order)
    principal_stresses = damask_sim.postprocess(jobname)
    # 5. Return residual
    return principal_stresses - target_stresses


def least_squares_solver(
    F_diag_init,
    F_prev,
    damask_sim,
    target_stresses,
    is_initial,
    restart_increment,
    dot_eps_eq,
    target_delta_eps=1e-6,
    min_trial_increments=2,
    max_trial_increments=100,
    **least_squares_kwargs
):
    """
    Solve for F_diag that matches target principal stresses using least squares.
    The lower bound is always [1e-6, 1e-6, 1e-6] to allow for both tension and compression directions.
    The upper bound is always [1.5, 1.5, 1.5].
    target_delta_eps: Target equivalent strain increment per increment (for trial runs)
    verbose=2: Show optimization process (can be overridden by least_squares_kwargs)
    Returns: optimized F_diag, result object
    """
    trial_counter = [0]
    def fun(F_diag):
        trial_counter[0] += 1
        jobname = f"trialLS{trial_counter[0]}"
        res = residual(
            F_diag=F_diag,
            F_prev=F_prev,
            damask_sim=damask_sim,
            target_stresses=target_stresses,
            is_initial=is_initial,
            restart_increment=restart_increment,
            jobname=jobname,
            dot_eps_eq=dot_eps_eq,
            target_delta_eps=target_delta_eps,
            min_trial_increments=min_trial_increments,
            max_trial_increments=max_trial_increments
        )
        return res
    # Lower bound is always [1e-6, 1e-6, 1e-6] to allow for compression
    lower = [1e-6, 1e-6, 1e-6]
    upper = [2.0, 2.0, 2.0]
    bounds = (lower, upper)
    if 'verbose' not in least_squares_kwargs:
        least_squares_kwargs['verbose'] = 2
    result = least_squares(fun, F_diag_init, bounds=bounds, **least_squares_kwargs)
    return result.x, result

# --- Test block for residual ---
if __name__ == '__main__':
    from damask_jfnk.config import DamaskSimConfig, JFNKConfig
    from damask_jfnk.damask_io import DAMASKSimulation

    # 1. Prepare simulation config
    sim_cfg = DamaskSimConfig(
        workdir='/home/doelz-admin/projects/damask_jfnk/workdir',
        logsdir='/home/doelz-admin/projects/damask_jfnk/logs',
        load_yaml='/home/doelz-admin/projects/damask_jfnk/workdir/load.yaml',
        grid_file='/home/doelz-admin/projects/damask_jfnk/workdir/grid.vti',
        material_file='/home/doelz-admin/projects/damask_jfnk/workdir/material.yaml',
        base_jobname='grid_load_material',
        F_init=np.eye(3),
        run_increments=10
    )
    jfnk_cfg = JFNKConfig(
        target_triax=1.0,
        target_lode=0.5,
        trial_increments=2,
        tol=1e-6,
        max_iter=10
    )
    sim = DAMASKSimulation(sim_cfg)

    # 2. Set up test parameters
    F_prev = np.eye(3)
    F_diag = np.array([1.01, 1.0, 1.0])
    is_initial = True
    restart_increment = None
    jobname = "test_residual"
    target_stresses = np.array([1.0, 1.0, 1.0])
    dot_eps_eq = 1.0  # Example value, adjust as needed
    target_delta_eps = 0.01  # Example value, adjust as needed

    print("\n--- Residual Function Test (with DAMASKSimulation) ---")
    print(f"F_diag: {F_diag}")

    res = residual(
        F_diag=F_diag,
        F_prev=F_prev,
        damask_sim=sim,
        target_stresses=target_stresses,
        is_initial=is_initial,
        restart_increment=restart_increment,
        jobname=jobname,
        dot_eps_eq=dot_eps_eq,
        target_delta_eps=target_delta_eps
    )
    print(f"Residual: {res}")

def target_principal_stresses(mean_stress, von_mises, lode_angle):
    """
    Compute target principal stresses from mean_stress, von_mises, and lode_angle (in radians).
    Returns: principal_stresses (shape [3,], descending order)
    """
    J2 = (von_mises ** 2) / 1.5
    sqrtJ2 = np.sqrt(J2)
    s1 = mean_stress + (4 * sqrtJ2 / 3) * np.cos(lode_angle)
    s2 = mean_stress + (4 * sqrtJ2 / 3) * np.cos(lode_angle - 2 * np.pi / 3)
    s3 = mean_stress + (4 * sqrtJ2 / 3) * np.cos(lode_angle + 2 * np.pi / 3)
    return np.sort([s1, s2, s3])[::-1]

# New residual for JFNK/GMRES: compare homogenized principal stresses to target
from damask_jfnk.postprocessing import compute_homogenized_principal_stresses

def residual_principal(F_diag, F_prev, damask_sim, target_principal_stresses, *args, **kwargs):
    """
    Compute the residual as the difference between homogenized principal stresses and target principal stresses.
    Additional args/kwargs can be passed to damask_sim.run_step as needed.
    """
    F_new = np.diag(F_diag)
    # Run DAMASK, get result file
    result_file = damask_sim.run_step(F=F_new, *args, **kwargs)
    principal_stresses = compute_homogenized_principal_stresses(result_file)
    return principal_stresses - target_principal_stresses
