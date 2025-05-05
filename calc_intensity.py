import numpy as np
from multiprocessing import Pool

### Diffraction pattern calculation functions

def atomic_form_factor(atom, q, wavelength):
# G_norm = |G| in Å⁻¹
    params = {
        'Cs': {
            'a': [20.3892, 19.1062, 10.6620, 1.4953],
            'b': [3.5690, 0.3107, 24.3879, 87.4372],
            'c': 4.3183
        },
        'V': {
            'a': [15.4512, 8.2176, 5.2340, 1.8504],
            'b': [0.2772, 1.0958, 3.2812, 14.4280],
            'c': 1.1001
        },
        'Sb': {
            'a': [20.8932, 19.1235, 11.2826, 2.02876],
            'b': [3.92819, 0.349002, 26.3553, 86.1015],
            'c': 4.91375
        },
        'Si': {
            'a': [21.0, 11.0, 5.0, 0.5],
            'b': [3.0, 1.5, 0.6, 12.0],
            'c': 1.0
        }
    }

    if atom not in params:
        raise ValueError(f"Unsupported atom type: {atom}")

    p = params[atom]
    a = np.array(p['a'])
    b = np.array(p['b'])
    c = p['c']

    G_norm = np.atleast_1d(q)
    s = G_norm / (4 * np.pi)

    f = np.sum(a[:, None] * np.exp(-b[:, None] * s**2), axis=0) + c

    return f

def F_hkl(atoms, G_vectors, q_values, wavelength, batch_size_atoms=500, batch_size_hkl=5000):    
    """Computes the structure factor F_hkl in a memory-efficient way using batch processing."""

    atom_types = np.unique(atoms.get_chemical_symbols())
    form_factors = {atom: atomic_form_factor(atom, q_values, wavelength) for atom in atom_types}

    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())

    num_hkl = G_vectors.shape[0]
    num_atoms = positions.shape[0]
    
    F_hkl_values = np.zeros(num_hkl, dtype=np.complex128)

    # Process hkl in batches
    for i in range(0, num_hkl, batch_size_hkl):
        G_batch = G_vectors[i:i + batch_size_hkl]  # Select batch of G vectors
        q_batch = q_values[i:i + batch_size_hkl]

        F_batch = np.zeros(G_batch.shape[0], dtype=np.complex128)

        # Process atoms in smaller batches
        for j in range(0, num_atoms, batch_size_atoms):
            pos_batch = positions[j:j + batch_size_atoms]  # Atom positions batch
            sym_batch = symbols[j:j + batch_size_atoms]  # Atom types batch
            
            # Compute form factors for this batch of atoms
            f_atoms = np.array([form_factors[s][i:i + batch_size_hkl] for s in sym_batch])  # Shape (batch_size_atoms, batch_size_hkl)

            # Compute phase terms for this subset of atoms
            phase_terms = np.exp(-1j * 2 * np.pi * np.dot(pos_batch, G_batch.T))  # Shape (batch_size_atoms, batch_size_hkl)

            # Correct broadcasting for multiplication
            F_batch += np.sum(f_atoms * phase_terms, axis=0)  # Sum over atoms in batch

        # Store the computed F_hkl values for this batch of hkl
        F_hkl_values[i:i + batch_size_hkl] = F_batch  

    return F_hkl_values

def compute_F_hkl_batch(args):
    """Wrapper function to compute F_hkl for a batch."""
    twin_bulk, G_batch, q_batch, twin_fraction, wavelength = args
    return (
        F_hkl(twin_bulk, G_batch, q_batch, wavelength) * twin_fraction
    )

def rotate_reciprocal_lattice(hkl_array, angle_deg, rotation_matrices):
    h_k_rot = np.dot(hkl_array[:, :2], rotation_matrices[angle_deg][:2, :2].T)  # Vectorized rotation
    return np.hstack((h_k_rot, hkl_array[:, 2][:, np.newaxis]))  # Preserve shape

def I_hkl(twin_bulks, hkl_array, r_lattice, twin_angles, twin_fractions, rotation_matrices, wavelength=(0.709300+0.713590)/2, batch_size_hkl=20000):
    """Computes XRD intensities efficiently using batch processing."""
    
    num_hkl = hkl_array.shape[0]
    F_total = np.zeros(num_hkl, dtype=np.complex128)  # Accumulate structure factors

    for twin_idx, (twin_bulk, twin_angle, twin_fraction) in enumerate(zip(twin_bulks, twin_angles, twin_fractions)):
        # Rotate reciprocal lattice (uses precomputed matrices)
        hkl_twin = rotate_reciprocal_lattice(hkl_array, twin_angle, rotation_matrices)
        G_vectors = np.dot(hkl_twin, r_lattice.T)  # Compute scattering vectors
        q_values = np.linalg.norm(G_vectors, axis=1)  # Compute |G|

        # Prepare batch arguments
        batch_args = [
            (twin_bulk, G_vectors[i:i + batch_size_hkl], q_values[i:i + batch_size_hkl], twin_fraction, wavelength)
            for i in range(0, num_hkl, batch_size_hkl)
        ]

        # Compute batches in parallel
        with Pool() as pool:
            F_batches = pool.map(compute_F_hkl_batch, batch_args)

        # Accumulate results
        for i, F_batch in enumerate(F_batches):
            start_idx = i * batch_size_hkl
            end_idx = min(start_idx + batch_size_hkl, num_hkl)
            np.add.at(F_total, np.arange(start_idx, end_idx), F_batch)

    return np.abs(F_total)**2  # Return intensity (|F|^2)

def precompute_rotation_matrices(angles_deg):
    """Precompute full 3D rotation matrices for each twin angle."""
    angles_rad = np.radians(angles_deg)
    cos_vals = np.cos(angles_rad)
    sin_vals = np.sin(angles_rad)

    # Create full 3D rotation matrices (assuming rotation about z-axis)
    return {
        angle: np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]  # No rotation in the z-direction
        ])
        for angle, c, s in zip(angles_deg, cos_vals, sin_vals)
    }