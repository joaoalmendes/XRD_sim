import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ase.io import read
from ase.visualize import view
import structure_models, generate_structures, strain, calc_intensity
from ase import Atoms

# Lattice parameters for input
# Orthorombic (distorted) lattice constants (Å)
a, b, c = 10.971314, 18.9833, 18.51410
alpha, beta, gamma = 90, 90, 90
cell_params = (a, b, c, alpha, beta, gamma)
# Input paremeters for computation
bulk_dimensions = (5, 5, 5)
# For now taking out 90 degrees to line up correctly; easy and lazy solution
twin_angles, twin_populations = [-90, -30, 30], [np.float64(.33), np.float64(.33), np.float64(.33)]

h_range = np.arange(-1, 1, 0.05)
k_range = np.arange(-1, 1, 0.05)

l_cuts = [np.float64(0), np.float64(0.50)]

# Plotting and precomputing functions and values
rotation_matrices = calc_intensity.precompute_rotation_matrices(twin_angles)

# Compute strained unit cells (vectorized)
# Define a small strain tensor
strain_tensor = np.array([
    [-0.0, 0.0, 0.0],  # ε_xx, ε_xy, ε_xz
    [0.0, 0.0, 0.0], # ε_yx, ε_yy, ε_yz
    [0.0, 0.0, 0.0]       # ε_xz, ε_yz, ε_zz
])
strained_unit_cells = strain.apply_strain(cell_params, strain_tensor, twin_angles, rotation_matrices)

original_structure = read('structures/CsV3Sb5.cif')
bulk_original = original_structure.repeat(bulk_dimensions)
#view(bulk_original)
# Precompute reference intensities for original structure for each l-cut
h_grid, k_grid = np.meshgrid(h_range, k_range, indexing='ij')
hk_array = np.column_stack((h_grid.ravel(), k_grid.ravel()))
reference_intensities = []
recip_lattice_original = np.round(np.linalg.inv(bulk_original.get_cell().T) * (2 * np.pi), decimals=10)
for l in l_cuts:
    hkl_values = np.hstack((hk_array, np.full((hk_array.shape[0], 1), l)))
    # Compute intensity for original bulk (no twinning, so use single structure)
    intensity = calc_intensity.I_hkl([bulk_original], hkl_values, recip_lattice_original, twin_angles=[-90]*3, twin_fractions=[0.33]*3, rotation_matrices=rotation_matrices)
    intensity = intensity.reshape(len(h_range), len(k_range))
    # Zero out a small square around (0,0)
    #hw = 2
    #hi, ki = np.argmin(np.abs(h_range)), np.argmin(np.abs(k_range))
    #intensity[max(0, hi-hw):hi+hw+1, max(0, ki-hw):ki+hw+1] = 0
    intensity = gaussian_filter(intensity, sigma=1.0)
    intensity /= np.max(intensity)  # Normalize
    reference_intensities.append(intensity)
    """fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Plot the intensity as a heatmap
    c = ax.pcolormesh(h_grid, k_grid, intensity, shading='auto', cmap='viridis')
    fig.colorbar(c, label='Intensity')

    # Labels and title
    ax.set_xlabel('h')
    ax.set_ylabel('k')
    ax.set_title(f"Reciprocal Space Intensity (l={l}) for the original structure")
    #plt.show()
    #plt.close()"""

reference_intensities = np.array(reference_intensities)  # Shape: [num_l_cuts, len(h_range), len(k_range)]

# Compute twin bulks in parallel
intensities = []  # List to store intensity arrays for each model
models = structure_models.models
model_names = ['Tri-H', 'SoD', 'Stripe']
for model in models:
    model_intensities = []  # Store intensities for each l-cut for this model
    m1, m2 = model, model
    formula = structure_models.get_formula(m1)
    twin_bulks = generate_structures.compute_twin_bulks_parallel(strained_unit_cells, formula, bulk_dimensions, m1, m2)
    # Compute reciprocal lattice (using the first bulk, as they share the basis)
    recip_lattice = np.round(np.linalg.inv(twin_bulks[0].get_cell().T) * (2 * np.pi), decimals=10)

    h_grid, k_grid = np.meshgrid(h_range, k_range, indexing='ij')  # Ensure correct indexing
    hk_array = np.column_stack((h_grid.ravel(), k_grid.ravel()))  # Flatten h and k values

    for l_idx, l in enumerate(l_cuts):
        hkl_values = np.hstack((hk_array, np.full((hk_array.shape[0], 1), l)))  # Add l column
        intensity = calc_intensity.I_hkl(twin_bulks, hkl_values, recip_lattice, twin_angles, twin_populations, rotation_matrices)
        intensity = intensity.reshape(len(h_range), len(k_range))  # Reshape back to mesh
        # Zero out a small square (e.g., 5x5) around (0,0)
        hw = 2  # Half-width of square
        hi, ki = np.argmin(np.abs(h_range)), np.argmin(np.abs(k_range))
        intensity[max(0, hi-hw):hi+hw+1, max(0, ki-hw):ki+hw+1] = 0
        intensity = gaussian_filter(intensity, sigma=1.0)  # Smooth
        #intensity = np.log1p(intensity)  # Log scaling
        intensity /= np.max(intensity)  # Normalize
        # Subtract reference intensity
        intensity = intensity - reference_intensities[l_idx]
        intensity = np.clip(intensity, 0, None)  # Optional: Remove negative values
        model_intensities.append(intensity)
        # Plot the intensity map
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Plot the intensity as a heatmap
        c = ax.pcolormesh(h_grid, k_grid, intensity, shading='auto', cmap='viridis')
        fig.colorbar(c, label='Intensity')

        # Labels and title
        ax.set_xlabel('h')
        ax.set_ylabel('k')
        ax.set_title(f"Reciprocal Space Intensity (l={l}) for {model_names[models.index(model)]}")
        plt.show()
        #plt.savefig(f"sim_{model_names[models.index(model)]}_hk{l}.png", dpi = 300)
        plt.close()
    
    intensities.append(model_intensities)  # Append model_intensities after all l-cuts

# Convert to NumPy array for easier manipulation (shape: [num_models, num_l_cuts, len(h_range), len(k_range)])
intensities = np.array(intensities)

# Plot intensity differences between models
for l_idx, l in enumerate(l_cuts):
    # Example: Compute difference between Model 1 and Model 2
    ref_model_idx = 0  # Reference model (e.g., Model 1)
    for model_idx in range(1, len(models)):  # Compare other models to reference
        diff_intensity = intensities[model_idx, l_idx] - intensities[ref_model_idx, l_idx]
        
        # Plot the difference map
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        # Use a diverging colormap to show positive and negative differences
        vmin, vmax = np.min(diff_intensity), np.max(diff_intensity)
        vcenter = 0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        c = ax.pcolormesh(h_grid, k_grid, diff_intensity, shading='auto', cmap='RdBu', norm=norm)
        fig.colorbar(c, label='Intensity Difference')
        
        # Labels and title
        ax.set_xlabel('h')
        ax.set_ylabel('k')
        ax.set_title(f"Intensity Difference: {model_names[model_idx]} - {model_names[ref_model_idx]} (l={l})")
        #plt.show()
        #plt.savefig(f"diff_hk{l}_{model_names[model_idx]}_vs_{model_names[ref_model_idx]}.png", dpi=300)
        #plt.close()