import numpy as np
from damask_jfnk.config import DamaskSimConfig, JFNKConfig
from damask_jfnk.damask_io import DAMASKSimulation
from damask_jfnk.least_square_solver import least_squares_solver, compute_equiv_strain_increment
import math

def run_workflow(
    damask_sim: DAMASKSimulation,
    jfnk_cfg: JFNKConfig,
    F_init: np.ndarray,
    dot_eps_eq: float,
    max_steps: int = 10,
    target_delta_eps: float = 1e-6
):
    """
    Main workflow for controlling stress state in DAMASK simulations.

    Args:
        damask_sim: DAMASKSimulation instance
        jfnk_cfg: JFNKConfig instance with target triaxiality and lode angle
        F_init: Initial deformation gradient (3x3), should be slightly larger than identity
        dot_eps_eq: Target equivalent strain rate
        max_steps: Maximum number of steps to run
        target_delta_eps: Target equivalent strain increment per increment (for trial runs, default 1e-6)

    Returns:
        List of (F, triax, lode) tuples for each successful step
    """
    results = []
    F_prev = np.eye(3)  # Start with identity matrix
    step = 0
    num_increments = 0
    # ratio = np.exp(target_delta_eps/np.sqrt(2))
    ratio = 1.01
    while step < max_steps:
        print(f"\n--- Step {step + 1}/{max_steps} ---")

        # 1. Initial run or trial run
        is_initial = (step == 0)
        restart_increment = None if is_initial else num_increments

        # Generate unique jobnames
        trial_jobname = f"trial_step{step + 1}"

        # 2. Run least squares solver to find optimal F
        F_diag_init = np.diag(F_init)
        F_opt, result = least_squares_solver(
            F_diag_init=F_diag_init,
            F_prev=F_prev,
            damask_sim=damask_sim,
            jfnk_cfg=jfnk_cfg,
            is_initial=is_initial,
            restart_increment=restart_increment,
            target_triax=jfnk_cfg.target_triax,
            target_lode=jfnk_cfg.target_lode,
            dot_eps_eq=dot_eps_eq,
            target_delta_eps=target_delta_eps,
            xtol=1e-4,
            ftol=1e-4,
            gtol=1e-4
        )

        if not result.success:
            print(f"Step {step + 1} failed to converge")
            break

        # 3. Production run with optimal F
        F_new = np.diag(F_opt)
        delta_eps_eq = compute_equiv_strain_increment(F_prev, F_new)
        t = delta_eps_eq / dot_eps_eq
        N = damask_sim.sim_cfg.run_increments  # Use fixed increments for production run
        print(f"Running production step with F:\n{F_new}, N={N}, t={t:.4f}")
        damask_sim.run_step(
            F=F_new,
            t=t,
            N=N,
            is_initial=is_initial,
            is_trial=False,
            restart_increment=restart_increment
        )

        # 4. Use optimized triax and lode from least_squares_solver result
        # result.fun = [triax - target_triax, lode - target_lode] at optimum
        triax = result.fun[0] + jfnk_cfg.target_triax
        lode = result.fun[1] + jfnk_cfg.target_lode
        print(f"Step {step + 1} results:")
        print(f"Triaxiality: {triax:.4f} (target: {jfnk_cfg.target_triax:.4f})")
        print(f"Lode angle: {lode:.4f} (target: {jfnk_cfg.target_lode:.4f})")

        # Store results
        results.append((F_new, triax, lode))

        # Update F_prev for next step
        F_prev = F_new
        F_init = F_new @ F_prev
        step += 1

        # Update number of increments for next restart
        num_increments += N
        print(f"num_increments: {num_increments}")
    return results

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
        run_increments=6  # Fixed increments for production run
    )

    # 2. Prepare JFNK config (trial step increments are adaptive)
    jfnk_cfg = JFNKConfig(
        target_triax=0.33,
        target_lode=0.5,
        min_trial_increments=2,   # Minimum increments per trial step
        max_trial_increments=50,  # Maximum increments per trial step
        tol=1e-3,
        max_iter=10
    )

    # 3. Create simulation instance
    sim = DAMASKSimulation(sim_cfg)

    # 4. Run workflow with F_init slightly larger than identity
    F_init = np.array([
        [1.0001, 0, 0],
        [0, 1.001, 0],
        [0, 0, 1.01]
    ])

    # User can set target_delta_eps here
    target_delta_eps = 1e-3
    results = run_workflow(
        damask_sim=sim,
        jfnk_cfg=jfnk_cfg,
        F_init=F_init,
        dot_eps_eq=1e-3,
        max_steps=10,
        target_delta_eps=target_delta_eps
    )

    # 5. Print final results
    print("\nFinal results:")
    for i, (F, triax, lode) in enumerate(results):
        print(f"\nStep {i + 1}:")
        print(f"F:\n{F}")
        print(f"Triaxiality: {triax:.4f}")
        print(f"Lode angle: {lode:.4f}")
