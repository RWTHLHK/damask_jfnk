import damask
import shutil
import os
import subprocess
import logging
from .config import DamaskSimConfig
import numpy as np
import shutil
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class JobNameManager:
    """
    Manages unique job names and file cleanup for serial JFNK workflow.
    Generates jobnames f1, f2, ... and cleans up files after use.
    """
    def __init__(self, base_jobname='grid_load_material', workdir='./workdir', logsdir='./logs'):
        self.counter = 0
        self.base_jobname = base_jobname
        self.workdir = workdir
        self.logsdir = logsdir
        self.create_dirs()
    def create_dirs(self):
        """Create work and logs directories if they don't exist."""
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(self.logsdir, exist_ok=True)
    def next_jobname(self):
        self.counter += 1
        # Return just the base identifier for the job
        return f"f{self.counter}"
    def prepare_restart_files(self, jobname_id):
        # Copy restart, result, and .sta files to the workdir
        base_path = f"{self.workdir}/{self.base_jobname}"
        new_path = f"{self.workdir}/{jobname_id}"
        shutil.copyfile(f"{base_path}_restart.hdf5", f"{new_path}_restart.hdf5")
        shutil.copyfile(f"{base_path}.hdf5", f"{new_path}.hdf5")
        if os.path.exists(f"{base_path}.sta"):
            shutil.copyfile(f"{base_path}.sta", f"{new_path}.sta")
    def cleanup(self, jobname_id):
        # Remove files related to this jobname identifier from workdir
        workdir_path = f"{self.workdir}/{jobname_id}"
        for ext in [".hdf5", "_restart.hdf5", ".sta"]:
            fname = f"{workdir_path}{ext}"
            if os.path.exists(fname):
                os.remove(fname)
    def job_log_path(self, jobname_id):
        """Generate path for job-specific log file."""
        return os.path.join(self.logsdir, f"{jobname_id}.log")

class DAMASKSimulation:
    """
    Encapsulates all DAMASK simulation operations, including load step management,
    running DAMASK, postprocessing, and cleanup.
    """
    def __init__(self, sim_cfg: DamaskSimConfig):
        """
        Initialize the DAMASKSimulation with a simulation configuration.
        """
        self.sim_cfg = sim_cfg
        self.load_case = None
        self.load_case_path = sim_cfg.load_yaml
        self._load_loadcase()
        os.makedirs(sim_cfg.workdir, exist_ok=True)
        os.makedirs(sim_cfg.logsdir, exist_ok=True)

    def _inversion(self, l, fill=0):
        """
        Convert 'x' to fill value and vice versa in a nested list (for boundary conditions).
        """
        return [self._inversion(i, fill) if isinstance(i, list) else fill if i == 'x' else 'x' for i in l]

    def _load_loadcase(self):
        """Load the load.yaml file as a DAMASK LoadcaseGrid object."""
        self.load_case = damask.LoadcaseGrid.load(self.load_case_path)
        if self.load_case is None:
            raise RuntimeError(f"Failed to load {self.load_case_path}.")

    def save_loadcase(self, path=None):
        """Save the current load_case to a YAML file."""
        if path is None:
            path = self.load_case_path
        self.load_case.save(path)

    def modify_load_step(self, step_index: int, F: np.ndarray, t: float, N: int, f_out: int = 1, f_restart: int = 1):
        """
        Modify a specific load step using the same interface and format as add_trial_load_step.
        """
        if step_index >= len(self.load_case['loadstep']):
            raise IndexError(f"Load step {step_index} does not exist")
        new_step_dict = {
            'boundary_conditions': {
                'mechanical': {
                    'F': F.tolist(),
                    'P': self._inversion(F.tolist())
                }
            },
            'discretization': {'t': t, 'N': N},
            'f_out': f_out,
            'f_restart': f_restart
        }
        self.load_case['loadstep'][step_index].update(new_step_dict)

    def add_trial_load_step(self, F: np.ndarray, t: float, N: int, f_out: int = 1, f_restart: int = 1):
        """
        Add a new load step with the specified F, t, N, and optional output/restart frequency.
        """
        loadstep = {
            'boundary_conditions': {
                'mechanical': {
                    'F': F.tolist(),
                    'P': self._inversion(F.tolist())
                }
            },
            'discretization': {'t': t, 'N': N},
            'f_out': f_out,
            'f_restart': f_restart
        }
        self.load_case['loadstep'].append(loadstep)

    def run_step(self, F: np.ndarray, t: float, N: int, is_initial: bool = False, is_trial: bool = False, f_out: int = 1, f_restart: int = 1, restart_increment: int = None, jobname: str = None) -> str:
        """
        Run a DAMASK simulation step.
        - For trial runs (is_trial=True, is_initial=False):
            * Copy the current load.yaml to load_trial.yaml
            * Add the trial step to load_trial.yaml
            * Run DAMASK with --load load_trial.yaml
        - For production/initial runs: operate on the main load.yaml as before.
        - Only add --restart for non-initial runs.
        - If is_trial: cleanup result after run.
        - jobname: unique jobname for this run (trial/production). If None, use sim_cfg.base_jobname.
        Returns: result file path.
        """
        if jobname is None:
            jobname = self.sim_cfg.base_jobname
        workdir = self.sim_cfg.workdir
        # --- Trial run logic: operate on a copy of load.yaml ---
        if (not is_initial) and is_trial:
            # 1. Copy the current load.yaml to load_trial.yaml
            orig_load = self.sim_cfg.load_yaml
            trial_load = os.path.join(workdir, 'load_trial.yaml')
            shutil.copyfile(orig_load, trial_load)
            # 2. Set load_case_path to load_trial.yaml and reload
            self.load_case_path = trial_load
            self._load_loadcase()
            # 3. Add the trial step to load_trial.yaml
            self.add_trial_load_step(F, t, N, f_out=f_out, f_restart=f_restart)
            self.save_loadcase(trial_load)
            load_file_for_cmd = 'load_trial.yaml'
        else:
            # --- Production/initial run logic: operate on main load.yaml ---
            self.load_case_path = self.sim_cfg.load_yaml
            self._load_loadcase()
            if is_initial:
                self.modify_load_step(step_index=0, F=F, t=t, N=N, f_out=f_out, f_restart=f_restart)
            else:
                self.add_trial_load_step(F, t, N, f_out=f_out, f_restart=f_restart)
            self.save_loadcase()
            load_file_for_cmd = os.path.basename(self.sim_cfg.load_yaml)
        # --- Auto copy restart files for trial runs ---
        if (not is_initial) and is_trial:
            prod_jobname = self.sim_cfg.base_jobname
            files = [
                (f"{prod_jobname}.hdf5", f"{jobname}.hdf5"),
                (f"{prod_jobname}_restart.hdf5", f"{jobname}_restart.hdf5"),
                (f"{prod_jobname}.sta", f"{jobname}.sta"),
            ]
            for src_name, dst_name in files:
                src = os.path.join(workdir, src_name)
                dst = os.path.join(workdir, dst_name)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
        # --- Build DAMASK_grid command ---
        cmd = [
            'DAMASK_grid',
            '--load', load_file_for_cmd,
            '--geom', os.path.basename(self.sim_cfg.grid_file),
            '--material', os.path.basename(self.sim_cfg.material_file),
            '--jobname', jobname
        ]
        # Only add --restart for non-initial runs
        if not is_initial:
            # print(f"Adding --restart {restart_increment}")
            if restart_increment is None:
                raise ValueError("restart_increment must be provided for non-initial runs.")
            cmd += ['--restart', str(restart_increment)]
        log_path = os.path.join(self.sim_cfg.logsdir, f'{jobname}.log')
        with open(log_path, 'a') as log_file:
            print(f"Running command: {' '.join(cmd)}", file=log_file)
            subprocess.run(cmd, check=True, cwd=self.sim_cfg.workdir, stdout=log_file, stderr=log_file)
        result_file = os.path.join(self.sim_cfg.workdir, f'{jobname}.hdf5')
        return result_file

    def cleanup(self, jobname: str):
        """
        Remove result files for the given jobname from the workdir and the corresponding log file from logsdir.
        """
        for ext in [".hdf5", "_restart.hdf5", ".sta"]:
            fname = os.path.join(self.sim_cfg.workdir, f"{jobname}{ext}")
            if os.path.exists(fname):
                os.remove(fname)
        # Also remove the log file in logsdir
        log_file = os.path.join(self.sim_cfg.logsdir, f"{jobname}.log")
        if os.path.exists(log_file):
            os.remove(log_file)

    def postprocess(self, is_initial: bool, is_trial: bool, jobname: str=None):
        """
        Postprocess the DAMASK result file for the given jobname and return relevant metrics.
        If is_trial is True, also cleanup the result and log files.
        """
        if jobname is not None:
            result_file = os.path.join(self.sim_cfg.workdir, f"{jobname}.hdf5")
        else:
            result_file = os.path.join(self.sim_cfg.workdir, f"{self.sim_cfg.base_jobname}.hdf5")
        result = damask.Result(result_file)
        if is_trial:
            result.add_stress_Cauchy(F='F')
        stress_eq = result.place('sigma')
        if stress_eq is None:
            raise ValueError("No 'sigma' field found in the result file. Check if the simulation output is correct.")
        stress = list(stress_eq.values())[-1]
        stress_mean_tensor = np.mean(stress, axis=0)
        stress_mean = np.array([
            stress_mean_tensor[0, 0],
            stress_mean_tensor[1, 1],
            stress_mean_tensor[2, 2],
            stress_mean_tensor[1, 2],
            stress_mean_tensor[0, 2],
            stress_mean_tensor[0, 1],
        ]) / 1e6
        hydrostatic = np.trace(stress_mean_tensor) / 3.0
        dev = stress_mean_tensor - np.eye(3) * hydrostatic
        von_mises = np.sqrt(1.5 * np.sum(dev**2))
        triaxiality = hydrostatic / (von_mises + 1e-12)

        # Compute lode angle with numerical stability checks
        J2 = 0.5 * np.sum(dev**2)
        if J2 < 1e-12:  # If J2 is too small, lode angle is undefined
            lode_angle = 0.0
        else:
            try:
                J3 = np.linalg.det(dev)
                lode_arg = (3 * np.sqrt(3) / 2) * J3 / (J2 ** 1.5)
                lode_arg = np.clip(lode_arg, -1, 1)
                lode_angle = (1.0 / 3.0) * np.arcsin(lode_arg)
            except (RuntimeWarning, OverflowError):
                # If overflow occurs, use a more stable calculation
                # For very small deviatoric stress, lode angle is approximately 0
                lode_angle = 0.0

        if is_trial:
            self.cleanup(jobname)
        return stress_mean, triaxiality, lode_angle

