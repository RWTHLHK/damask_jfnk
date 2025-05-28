import numpy as np
from damask_jfnk.config import DamaskSimConfig
from damask_jfnk.damask_io import DAMASKSimulation

def test_damask_simulation():
    # 1. Prepare simulation config (adjust paths as needed)
    sim_cfg = DamaskSimConfig(
        workdir='./workdir',
        logsdir='./logs',
        load_yaml='./workdir/load.yaml',
        grid_file='./workdir/grid.vti',
        material_file='./workdir/material.yaml',
        base_jobname='grid_load_material',
        F_init=np.eye(3),
        run_increments=10
    )
    sim = DAMASKSimulation(sim_cfg)

    # 2. Test modify_load_step
    F_initial = np.array([[1.01, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = 5.0
    N = 4
    f_out = 1
    f_restart = 1
    sim.modify_load_step(step_index=0, F=F_initial, t=t, N=N, f_out=f_out, f_restart=f_restart)
    print("Modified load step 0.")

    # 3. Test add_trial_load_step
    F_trial = np.array([[1.02, 0, 0], [0, 1, 0], [0, 0, 1]])
    sim.add_trial_load_step(F=F_trial, t=t, N=N, f_out=f_out, f_restart=f_restart)
    print("Added trial load step.")

    # 4. Test run_step (initial run)
    result_file = sim.run_step(
        F=F_initial, t=t, N=N,
        is_initial=True, is_trial=False,
        jobname="test_initial"
    )
    print(f"Initial run result file: {result_file}")

    # 5. Test run_step (trial run)
    result_file = sim.run_step(
        F=F_trial, t=t, N=N,
        is_initial=False, is_trial=True,
        restart_increment=0,
        jobname="test_trial"
    )
    print(f"Trial run result file: {result_file}")

    # 6. Test postprocess
    stress_mean, triax, lode = sim.postprocess(
        is_initial=False, is_trial=True, jobname="test_trial"
    )
    print(f"Postprocess results: stress_mean={stress_mean}, triax={triax}, lode={lode}")

if __name__ == '__main__':
    test_damask_simulation()
