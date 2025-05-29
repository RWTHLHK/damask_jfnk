import numpy as np
from .config import DamaskSimConfig
from .damask_io import DAMASKSimulation
from .least_square_solver import least_squares_solver, compute_equiv_strain_increment
import logging
import os
from typing import Tuple

def target_principal_stresses(mean_stress, triax, lode_angle):
    """
    Compute target principal stresses from mean_stress, triaxiality, and lode_angle (in radians).
    Returns: principal_stresses (shape [3,], descending order)
    """
    # triax = mean_stress / (von_mises + 1e-12)
    von_mises = mean_stress / (triax + 1e-12)
    J2 = (von_mises ** 2) / 1.5
    sqrtJ2 = np.sqrt(J2)
    s1 = mean_stress + (4 * sqrtJ2 / 3) * np.cos(lode_angle)
    s2 = mean_stress + (4 * sqrtJ2 / 3) * np.cos(lode_angle - 2 * np.pi / 3)
    s3 = mean_stress + (4 * sqrtJ2 / 3) * np.cos(lode_angle + 2 * np.pi / 3)
    return np.sort([s1, s2, s3])[::-1]

def generate_mean_stress_sequence(start, mid, stop, n_fast, n_slow):
    """
    Generate a sequence of mean stresses: first n_fast steps with larger increments, then n_slow steps with smaller increments.
    """
    fast = np.linspace(start, mid, n_fast, endpoint=False)
    slow = np.linspace(mid, stop, n_slow)
    return np.concatenate([fast, slow])

def run_workflow(
    damask_sim: DAMASKSimulation,
    F_init: np.ndarray,
    triax: float,
    lode: float,
    mean_stress_seq: np.ndarray,
    dot_eps_eq: float = 1e-3,
    target_delta_eps: float = 1e-6,
    min_trial_increments: int = 2,
    max_trial_increments: int = 10,
    min_time_step: float = 1e-4,
    tol: float = 1e-5,
    verbose: bool = True
):
    results = []
    F_prev = np.eye(3)
    F_diag_init = np.diag(F_init)
    is_initial = True
    restart_increment = None
    total_increment = 0  # Track total number of production increments
    N_prod = 10  # Number of increments per production run
    for i, mean_stress in enumerate(mean_stress_seq):
        if verbose:
            print(f"\n--- Step {i+1}/{len(mean_stress_seq)} ---")
            print(f"Target mean stress: {mean_stress:.3f}")
        # 1. 计算目标主应力
        target_stresses = target_principal_stresses(mean_stress, triax, lode)
        # 2. 用LS solver求解F_diag
        F_diag_opt, result = least_squares_solver(
            F_diag_init=F_diag_init,
            F_prev=F_prev,
            damask_sim=damask_sim,
            target_stresses=target_stresses,
            is_initial=is_initial,
            restart_increment=restart_increment,
            dot_eps_eq=dot_eps_eq,
            target_delta_eps=target_delta_eps,
            min_trial_increments=min_trial_increments,
            max_trial_increments=max_trial_increments,
            ftol=tol, xtol=tol, gtol=tol, verbose=2
        )
        F_new = np.diag(F_diag_opt)
        # 3. 生产步
        delta_eps_eq = compute_equiv_strain_increment(F_prev, F_new)
        t = max(delta_eps_eq / dot_eps_eq, min_time_step)  # Use max to ensure minimum time step
        if verbose:
            print(f"Time step: {t:.2e} (min: {min_time_step:.2e})")
        result_file = damask_sim.run_step(
            F=F_new,
            t=t,
            N=N_prod,
            is_initial=is_initial,
            is_trial=False,
            restart_increment=restart_increment
        )
        final_stresses = target_stresses + result.fun
        if verbose:
            print(f"Step {i+1} results:")
            print(f"Final principal stresses: {final_stresses}")
            print(f"Target principal stresses: {target_stresses}")
            print(f"Absolute error: {np.abs(final_stresses - target_stresses)}")
        results.append({
            'step': i+1,
            'mean_stress': mean_stress,
            'target_stresses': target_stresses,
            'F_diag_opt': F_diag_opt,
            'final_stresses': final_stresses,
            'abs_error': np.abs(final_stresses - target_stresses),
            'time_step': t  # Add time step to results
        })
        # 更新F_prev和F_diag_init
        F_prev = F_new
        # Add small random perturbation to F_diag_init
        F_diag_init = F_diag_opt + np.random.uniform(-1e-4, 1e-4, size=3)
        if verbose:
            print(f"Next step initial F_diag: {F_diag_init}")
        is_initial = False
        total_increment += N_prod
        restart_increment = total_increment
    return results

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration
    config = DamaskSimConfig(
        workdir='/home/doelz-admin/projects/damask_jfnk/workdir',
        logsdir='/home/doelz-admin/projects/damask_jfnk/logs',
        load_yaml='/home/doelz-admin/projects/damask_jfnk/workdir/load.yaml',
        grid_file='/home/doelz-admin/projects/damask_jfnk/workdir/grid.vti',
        material_file='/home/doelz-admin/projects/damask_jfnk/workdir/material.yaml',
        base_jobname='grid_load_material',
        F_init=np.eye(3),
        run_increments=10
    )
    sim = DAMASKSimulation(config)

    # Initial F
    F0 = np.array([
        [1.001, 0, 0],
        [0, 0.9997, 0],
        [0, 0, 0.9996]
    ])

    # 固定 triax/lode
    triax = 0.33
    lode = 0.1
    # 生成 mean stress 序列
    mean_stress_seq = generate_mean_stress_sequence(
        start=30, mid=50, stop=150, n_fast=8, n_slow=80
    )

    # Run workflow
    results = run_workflow(
        damask_sim=sim,
        F_init=F0,
        triax=triax,
        lode=lode,
        mean_stress_seq=mean_stress_seq,
        dot_eps_eq=1e-3,
        target_delta_eps=1e-6,
        min_trial_increments=2,
        max_trial_increments=10,
        min_time_step=1e-4,
        tol=1e-5,
        verbose=True
    )

    # Print summary
    logger.info("\nSummary of all steps:")
    for r in results:
        logger.info(f"Step {r['step']}: mean_stress={r['mean_stress']:.3f}, F_diag={r['F_diag_opt']}, final_stresses={r['final_stresses']}, abs_error={r['abs_error']}, time_step={r['time_step']:.2e}")

if __name__ == "__main__":
    main()
