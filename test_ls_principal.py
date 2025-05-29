import numpy as np
from damask_jfnk.config import DamaskSimConfig
from damask_jfnk.damask_io import DAMASKSimulation
from damask_jfnk.least_square_solver import least_squares_solver, compute_equiv_strain_increment

if __name__ == '__main__':
    # 1. Prepare simulation config
    sim_cfg = DamaskSimConfig(
        workdir='/home/doelz-admin/projects/damask_jfnk/workdir',
        logsdir='/home/doelz-admin/projects/damask_jfnk/logs',
        load_yaml='/home/doelz-admin/projects/damask_jfnk/workdir/load.yaml',
        grid_file='/home/doelz-admin/projects/damask_jfnk/workdir/grid.vti',
        material_file='/home/doelz-admin/projects/damask_jfnk/workdir/material.yaml',
        base_jobname='grid_load_material',
        F_init=np.eye(3),
        run_increments=4
    )
    sim = DAMASKSimulation(sim_cfg)

    # 2. Set up test parameters
    F_prev = np.eye(3)
    F_diag_init = np.array([1.01, 1.001, 1.0001])
    target_stresses = np.array([100.0, 50.0, 25.0])  # Example target principal stresses (MPa)
    is_initial = True
    restart_increment = None
    dot_eps_eq = 1e-3
    target_delta_eps = 1e-6
    min_trial_increments = 2
    max_trial_increments = 10

    print("\n--- Least Squares Solver Test (Principal Stresses Target) ---")
    print(f"Initial F_diag: {F_diag_init}")
    print(f"Target principal stresses: {target_stresses}")

    F_diag_opt, result = least_squares_solver(
        F_diag_init=F_diag_init,
        F_prev=F_prev,
        damask_sim=sim,
        target_stresses=target_stresses,
        is_initial=is_initial,
        restart_increment=restart_increment,
        dot_eps_eq=dot_eps_eq,
        target_delta_eps=target_delta_eps,
        min_trial_increments=min_trial_increments,
        max_trial_increments=max_trial_increments,
        verbose=2
    )

    print(f"\nOptimized F_diag: {F_diag_opt}")
    F_new = np.diag(F_diag_opt)
    # Run a production step with the optimized F_diag
    delta_eps_eq = compute_equiv_strain_increment(F_prev, F_new)
    t = delta_eps_eq / dot_eps_eq
    result_file = sim.run_step(
        F=F_new,
        t=t,
        N=4,
        is_initial=is_initial,
        is_trial=False,
        restart_increment=restart_increment,
    )
    final_stresses = target_stresses + result.fun
    print(f"Final principal stresses: {final_stresses}")
    print(f"Target principal stresses: {target_stresses}")
    print(f"Absolute error: {np.abs(final_stresses - target_stresses)}")
