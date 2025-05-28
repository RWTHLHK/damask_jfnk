import numpy as np
from scipy.optimize import least_squares

# --- Residual and helper function (self-contained) ---
def compute_equiv_strain_increment(F_old, F_new):
    """
    Compute the equivalent strain increment between two deformation gradients.
    Uses principal logarithmic strain difference and J2 norm.
    """
    delta_F = np.dot(np.linalg.inv(F_old), F_new)
    U, s, Vh = np.linalg.svd(delta_F)
    logV = np.log(s)
    eqv_strain = np.sqrt(2/3 * np.sum(logV**2))
    return eqv_strain


def residual(
    F_diag,                      # Diagonal of new F
    F_prev,                      # Previous F (3x3)
    damask_sim,                  # DAMASKSimulation instance
    jfnk_cfg,                    # JFNKConfig instance
    is_initial,                  # bool, whether this is the first step
    restart_increment,           # int, for --restart (None if is_initial)
    jobname,                     # str, unique jobname for this trial
    target_triax,                # float, target triaxiality
    target_lode,                 # float, target lode angle
    dot_eps_eq,                  # float, target equivalent strain rate
    target_delta_eps            # float, target equivalent strain increment per increment (for trial runs)
):
    """
    Compute the residual vector for JFNK: [triaxiality - target_triax, lode_angle - target_lode].
    target_delta_eps: Target equivalent strain increment per increment (for trial runs)
    """
    F_new = np.diag(F_diag)
    # 1. Compute equivalent strain increment
    delta_eps_eq = compute_equiv_strain_increment(F_prev, F_new)
    # 2. Compute loading time t for this step
    N = max(jfnk_cfg.min_trial_increments, min(jfnk_cfg.max_trial_increments, int(np.ceil(delta_eps_eq / target_delta_eps))))
    t = delta_eps_eq / dot_eps_eq
    # 3. Run DAMASK trial step
    damask_sim.run_step(
        F=F_new, t=t, N=N,
        is_initial=is_initial, is_trial=True,
        restart_increment=restart_increment,
        jobname=jobname
    )
    # 4. Postprocess
    _, triax, lode = damask_sim.postprocess(
        is_initial=is_initial, is_trial=True, jobname=jobname
    )
    # 5. Return residual
    return np.array([triax - target_triax, lode - target_lode])


def least_squares_solver(
    F_diag_init,
    F_prev,
    damask_sim,
    jfnk_cfg,
    is_initial,
    restart_increment,
    target_triax,
    target_lode,
    dot_eps_eq,
    target_delta_eps=1e-6,
    **least_squares_kwargs
):
    """
    Solve for F_diag that matches target triaxiality and lode angle using least squares.
    The lower bound is always [1e-6, 1e-6, 1e-6] to allow for both tension and compression directions.
    The upper bound is always [1.05, 1.05, 1.05].
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
            jfnk_cfg=jfnk_cfg,
            is_initial=is_initial,
            restart_increment=restart_increment,
            jobname=jobname,
            target_triax=target_triax,
            target_lode=target_lode,
            dot_eps_eq=dot_eps_eq,
            target_delta_eps=target_delta_eps
        )
        # print(f"LS trial {trial_counter[0]}: F_diag={F_diag}, residual={res}")
        return res
    # Lower bound is always [1e-6, 1e-6, 1e-6] to allow for compression
    lower = [1e-6, 1e-6, 1e-6]
    upper = [1.5, 1.5, 1.5]
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
    target_triax = jfnk_cfg.target_triax
    target_lode = jfnk_cfg.target_lode
    dot_eps_eq = 1.0  # Example value, adjust as needed
    target_delta_eps = 0.01  # Example value, adjust as needed

    print("\n--- Residual Function Test (with DAMASKSimulation) ---")
    print(f"F_diag: {F_diag}")

    res = residual(
        F_diag=F_diag,
        F_prev=F_prev,
        damask_sim=sim,
        jfnk_cfg=jfnk_cfg,
        is_initial=is_initial,
        restart_increment=restart_increment,
        jobname=jobname,
        target_triax=target_triax,
        target_lode=target_lode,
        dot_eps_eq=dot_eps_eq,
        target_delta_eps=target_delta_eps
    )
    print(f"Residual: {res}")
