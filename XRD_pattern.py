import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ase.io import read, write
from ase.visualize import view
import structure_models, generate_structures, strain, calc_intensity
# https://wiki.fysik.dtu.dk/ase/ase/xrdebye.html

def compute_reference_intensities(structure, bulk_dims,
                                 var1_range, var2_range,
                                 fixed_idx, fixed_vals,
                                 twin_angles, twin_pops,
                                 cell_params=None,
                                 strain_tensor=None):
    """
    Computes normalized reference intensities for an un-twinned bulk structure.

    Parameters:
    - structure: ASE Atoms of one unit cell
    - bulk_dims: tuple of repetitions (nx, ny, nz)
    - var1_range, var2_range: arrays for the two varying Miller indices
    - fixed_idx: 'h','k', or 'l' indicating which index is held constant
    - fixed_vals: list of values for the fixed index
    - twin_angles, twin_pops: arrays of angles/fractions (for rotation precompute)
    - cell_params, strain_tensor: optional; if provided, apply strain first

    Returns:
    - dict mapping fixed_val -> 2D intensity array over var1×var2 grid
    """
    rot_mats = calc_intensity.precompute_rotation_matrices(twin_angles)
    # Reciprocal lattice
    recip = np.round(np.linalg.inv(structure.get_cell().T) * (2*np.pi), 10)
    # Mesh for var1 and var2
    v1_grid, v2_grid = np.meshgrid(var1_range, var2_range, indexing='ij')
    grid_pts = v1_grid.size
    refs = {}

    for val in fixed_vals:
        # Build H, K, L arrays
        if fixed_idx == 'h':
            arr = np.hstack((
                np.full((grid_pts,1), val),
                v1_grid.ravel()[:,None],
                v2_grid.ravel()[:,None]
            ))
        elif fixed_idx == 'k':
            arr = np.hstack((
                v1_grid.ravel()[:,None],
                np.full((grid_pts,1), val),
                v2_grid.ravel()[:,None]
            ))
        else:  # 'l'
            arr = np.hstack((
                v1_grid.ravel()[:,None],
                v2_grid.ravel()[:,None],
                np.full((grid_pts,1), val)
            ))

        I = calc_intensity.I_hkl([structure], arr, recip,
                                 twin_angles=[twin_angles[0]]*len(twin_angles),
                                 twin_fractions=[twin_pops[0]]*len(twin_pops),
                                 rotation_matrices=rot_mats)
        I = I.reshape(len(var1_range), len(var2_range))
        # Define the width w (e.g., w = 5)
        w = 10

        # Find the indices closest to zero in var1_range and var2_range
        hi, ki = np.argmin(np.abs(var1_range)), np.argmin(np.abs(var2_range))

        # Calculate half the width for centering the square
        half_w = w // 2

        # Zero out a square of width w centered at (hi, ki)
        I[max(0, hi - half_w):hi + half_w + 1, max(0, ki - half_w):ki + half_w + 1] = 0
        I = gaussian_filter(I, sigma=1.0)
        I /= np.max(I)
        refs[val] = I

    return refs


def compute_model_intensities(models, cell_params, strain_tensor,
                              bulk_dims, var1_range, var2_range,
                              fixed_idx, fixed_vals,
                              twin_angles, twin_pops):
    """As before: applies strain, twins, and returns dict of model->{val: intensity}
    """
    rot_mats = calc_intensity.precompute_rotation_matrices(twin_angles)
    strained_cells = strain.apply_strain(cell_params, strain_tensor,
                                        twin_angles, rot_mats)
    results = {}
    v1_grid, v2_grid = np.meshgrid(var1_range, var2_range, indexing='ij')
    grid_pts = v1_grid.size

    for model in models:
        name = structure_models.model_names[models.index(model)]
        twin_bulks = generate_structures.compute_twin_bulks_parallel(
            strained_cells, structure_models.get_formula(model), bulk_dims,
            model, model)
        recip = np.round(np.linalg.inv(twin_bulks[0].get_cell().T)*(2*np.pi), 10)
        model_dict = {}

        for val in fixed_vals:
            # H/K/L assignment based on fixed_idx
            if fixed_idx == 'h':
                arr = np.hstack((
                    np.full((grid_pts,1), val),
                    v1_grid.ravel()[:,None],
                    v2_grid.ravel()[:,None]
                ))
            elif fixed_idx == 'k':
                arr = np.hstack((
                    v1_grid.ravel()[:,None],
                    np.full((grid_pts,1), val),
                    v2_grid.ravel()[:,None]
                ))
            else:
                arr = np.hstack((
                    v1_grid.ravel()[:,None],
                    v2_grid.ravel()[:,None],
                    np.full((grid_pts,1), val)
                ))

            I = calc_intensity.I_hkl(twin_bulks, arr, recip,
                                     twin_angles, twin_pops, rot_mats)
            I = I.reshape(len(var1_range), len(var2_range))
            # Define the width w (e.g., w = 5)
            w = 10

            # Find the indices closest to zero in var1_range and var2_range
            hi, ki = np.argmin(np.abs(var1_range)), np.argmin(np.abs(var2_range))

            # Calculate half the width for centering the square
            half_w = w // 2

            # Zero out a square of width w centered at (hi, ki)
            I[max(0, hi - half_w):hi + half_w + 1, max(0, ki - half_w):ki + half_w + 1] = 0
            I = gaussian_filter(I, sigma=1.0)
            I /= np.max(I)
            model_dict[val] = I

        results[name] = model_dict
    return results


def plot_intensity_maps(intensities, var1_range, var2_range,
                        fixed_idx, fixed_vals,
                        ref_diff_model=None, ref_original_structure=None):
    """
    First plots absolute maps (viridis) for each model & cut,
    then plots difference maps (RdBu) against ref_model.
    """
    v1_grid, v2_grid = np.meshgrid(var1_range, var2_range, indexing='ij')
    # Absolute maps
    for val in fixed_vals:
        for model, data in intensities.items():
            if ref_original_structure:
                #I = data[val] - ref_original_structure[val]    # with negative values
                #I = np.maximum(data[val] - ref_original_structure[val], 0)  # without negative values
                I = ref_original_structure[val] # plot one structure only
                I = gaussian_filter(I, sigma=1.0)
                I /= np.max(I)
                data[val] = I
            else:
                I = data[val]
            fig, ax = plt.subplots(figsize=(6,6))
            c = ax.pcolormesh(v1_grid, v2_grid, I, shading='auto', cmap='viridis')
            fig.colorbar(c, ax=ax, label='Intensity')
            ax.set_xlabel({'h':'k','k':'h','l':'h'}[fixed_idx])
            ax.set_ylabel({'h':'l','k':'l','l':'k'}[fixed_idx])
            ax.set_title(f"{model}: {fixed_idx}={val}")
            plt.show()
    # Difference maps
    if ref_diff_model:
        for val in fixed_vals:
            for model, data in intensities.items():
                I_diff = data[val] - intensities[ref_diff_model][val]
                fig, ax = plt.subplots(figsize=(6,6))
                norm = plt.Normalize(vmin=I_diff.min(), vmax=I_diff.max())
                c = ax.pcolormesh(v1_grid, v2_grid, I_diff, shading='auto', cmap='RdBu', norm=norm)
                fig.colorbar(c, ax=ax, label='Intensity Difference')
                ax.set_xlabel({'h':'k','k':'h','l':'h'}[fixed_idx])
                ax.set_ylabel({'h':'l','k':'l','l':'k'}[fixed_idx])
                ax.set_title(f"{model} - {ref_diff_model}: {fixed_idx}={val}")
                plt.show()

# Lattice parameters for input
# Orthorombic (distorted) lattice constants (Å)
a, b, c = 10.971314, 18.9833, 18.51410
alpha, beta, gamma = 90, 90, 90
cell_params = (a, b, c, alpha, beta, gamma)

# Input paremeters for computation
bulk_dimensions = (5, 5, 5)
# For now taking out 90 degrees to line up correctly; easy and lazy solution
twin_angles, twin_pops = [-90, -30, 30], [np.float64(.33), np.float64(.33), np.float64(.33)]

strain_tensor = np.array([
    [-0.0, 0.0, 0.0],  # ε_xx, ε_xy, ε_xz
    [0.0, 0.0, 0.0], # ε_yx, ε_yy, ε_yz
    [0.0, 0.0, 0.0]       # ε_xz, ε_yz, ε_zz
])

structure = read('structures/CsV3Sb5.cif')
bulk_original = structure.repeat(bulk_dimensions)
#view(bulk_original)
# Precompute reference intensities for original structure for each l-cut

models = structure_models.models

from ase.build import bulk

# Create a silicon bulk crystal (diamond structure)
si_bulk = bulk('Si', crystalstructure='diamond', a=5.431, cubic=True)

x_vals = np.arange(-1,1,0.05)
y_vals = np.arange(-1,1,0.05)
fixed = 'l'
fixed_vals = [0.0, 0.5]

reference_structure = compute_reference_intensities(si_bulk, (5,5,5),
                                     x_vals, y_vals,
                                     fixed, fixed_vals,
                                     [-90,-90,-90], [1.0,0.0,0.0],
                                     cell_params, strain_tensor)

ints = compute_model_intensities(models, cell_params, strain_tensor,
                                 bulk_dimensions, x_vals, y_vals,
                                 fixed, fixed_vals,
                                 twin_angles, twin_pops)

# Plot absolute + difference maps in one call
plot_intensity_maps(
    ints, x_vals, y_vals,
    fixed, fixed_vals,
    ref_diff_model=list(ints.keys())[0],  # e.g. 'Tri-H'
    ref_original_structure=None
)






