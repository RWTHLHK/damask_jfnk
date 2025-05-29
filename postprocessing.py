import damask
import numpy as np
import matplotlib.pyplot as plt

def compute_all_increments_triax_lode_eqstrain(result_file='grid_load_material.hdf5'):
    """
    Compute triaxiality, Lode angle, and equivalent strain for all increments in a DAMASK result file.
    Returns arrays: eq_strain, triaxiality, lode_angle (all shape: [n_increments]), skipping the first increment (which may be inaccurate).
    """
    result = damask.Result(result_file)
    result.add_strain(F='F')
    result.add_stress_Cauchy(F='F')
    result.add_equivalent_Mises('epsilon_V^0.0(F)')
    stress_eq = result.place('sigma')
    strain_eq = result.place('epsilon_V^0.0(F)')
    if stress_eq is None or strain_eq is None:
        raise ValueError("Missing 'sigma' or 'epsilon_V^0.0(F)' in result file.")
    eq_strain = []
    triaxiality = []
    lode_angle = []
    for key in stress_eq.keys():
        stress = stress_eq[key]
        strain = strain_eq[key]
        eq_strain_val = np.mean(strain)
        eq_strain.append(eq_strain_val)
        stress_mean_tensor = np.mean(stress, axis=0)
        hydrostatic = np.trace(stress_mean_tensor) / 3.0
        dev = stress_mean_tensor - np.eye(3) * hydrostatic
        von_mises = np.sqrt(1.5 * np.sum(dev**2))
        triax = hydrostatic / (von_mises + 1e-12)
        J2 = 0.5 * np.sum(dev**2)
        J3 = np.linalg.det(dev)
        sqrtJ2 = np.sqrt(J2)
        if sqrtJ2 > 1e-12:
            lode_arg = (3 * np.sqrt(3) / 2) * J3 / (J2 ** 1.5)
            lode_arg = np.clip(lode_arg, -1, 1)
            lode = (1.0 / 3.0) * np.arcsin(lode_arg)
        else:
            lode = 0.0
        triaxiality.append(triax)
        lode_angle.append(lode)
    # Skip the first increment, which may be inaccurate
    return np.array(eq_strain[1:]), np.array(triaxiality[1:]), np.array(lode_angle[1:])

def plot_triax_lode_vs_eqstrain(eq_strain, triaxiality, lode_angle):
    """
    Plot triaxiality and Lode angle vs. equivalent strain.
    Triaxiality y-axis is set to [-1, 1].
    Lode angle y-axis is set to [-pi/6, pi/6] (physical range).
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(eq_strain, triaxiality, marker='o')
    plt.xlabel('Equivalent Strain')
    plt.ylabel('Triaxiality')
    plt.title('Triaxiality vs. Equivalent Strain')
    plt.ylim(-1, 1)  # Physical range for triaxiality
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(eq_strain, lode_angle, marker='o')
    plt.xlabel('Equivalent Strain')
    plt.ylabel('Lode Angle (rad)')
    plt.title('Lode Angle vs. Equivalent Strain')
    plt.ylim(-np.pi/6, np.pi/6)  # Physical range for Lode angle
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('result.png')

def compute_homogenized_principal_stresses(result_file='grid_load_material.hdf5'):
    """
    Compute the principal stresses (sorted) of the homogenized Cauchy stress tensor from the last increment.
    Returns: principal_stresses (sorted, shape [3,])
    """
    result = damask.Result(result_file)
    result.add_stress_Cauchy(F='F')
    stress_eq = result.place('sigma')
    if stress_eq is None:
        raise ValueError("No 'sigma' field found in the result file. Check if the simulation output is correct.")
    # Get last increment
    stress = list(stress_eq.values())[-1]  # shape: (n, 3, 3)
    stress_mean_tensor = np.mean(stress, axis=0)  # shape: (3, 3)
    principal_stresses = np.linalg.eigvalsh(stress_mean_tensor)
    principal_stresses = np.sort(principal_stresses)[::-1]  # descending order
    return principal_stresses

if __name__ == "__main__":
    result_file = '/home/doelz-admin/projects/damask_jfnk/workdir/grid_load_material.hdf5'
    eq_strain, triaxiality, lode_angle = compute_all_increments_triax_lode_eqstrain(result_file)
    plot_triax_lode_vs_eqstrain(eq_strain, triaxiality, lode_angle)
