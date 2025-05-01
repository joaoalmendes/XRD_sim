import numpy as np

def degrees_to_radians(angle_deg):
    return np.radians(angle_deg)

def radians_to_degrees(angle_rad):
    return np.degrees(angle_rad)

def unit_cell_to_metric_tensor(cell_params):
    """Convert unit cell parameters to metric tensor G."""
    a, b, c, alpha, beta, gamma = cell_params
    alpha, beta, gamma = map(degrees_to_radians, [alpha, beta, gamma])

    G = np.array([
        [a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
        [a*b*np.cos(gamma), b**2, b*c*np.cos(alpha)],
        [a*c*np.cos(beta), b*c*np.cos(alpha), c**2]
    ])
    return G

def metric_tensor_to_unit_cell(G):
    """Convert metric tensor G back to unit cell parameters."""
    a_new = np.sqrt(G[0, 0])
    b_new = np.sqrt(G[1, 1])
    c_new = np.sqrt(G[2, 2])

    alpha_new = radians_to_degrees(np.arccos(G[1, 2] / (b_new * c_new)))
    beta_new = radians_to_degrees(np.arccos(G[0, 2] / (a_new * c_new)))
    gamma_new = radians_to_degrees(np.arccos(G[0, 1] / (a_new * b_new)))

    return a_new, b_new, c_new, alpha_new, beta_new, gamma_new

def apply_strain(cell_params, strain_tensors, twin_angles, rotation_matrices):
    """Applies different rotated strain matrices to the unit cell for each twin variant efficiently."""

    # Ensure strain_tensors is (N_twin, 3, 3)
    strain_tensors = np.array(strain_tensors)
    if strain_tensors.ndim == 2:  # If shape is (3, 3), expand to (N_twin, 3, 3)
        strain_tensors = np.tile(strain_tensors, (len(twin_angles), 1, 1))
    
    # Convert input unit cell parameters to metric tensor
    G = unit_cell_to_metric_tensor(cell_params)
    
    # Stack rotation matrices for all twin angles (N_twin, 3, 3)
    R_matrices = np.array([rotation_matrices[angle] for angle in twin_angles])

    # Rotate the strain tensors: ε' = R ε R^T
    rotated_strains = np.einsum('nij,njk,nlk->nil', R_matrices, strain_tensors, R_matrices)

    # Apply the strain to G: G' = (I + ε)^T G (I + ε)
    I_plus_strains = np.identity(3) + rotated_strains
    G_new = np.einsum('nij,jk,nlk->nil', I_plus_strains, G, I_plus_strains)

    # Convert back to unit cell parameters for each twin variant
    strain_unit_cell_params = np.array([metric_tensor_to_unit_cell(G_new[i]) for i in range(len(twin_angles))])

    return strain_unit_cell_params  # Returns a NumPy array of shape (N_twin, 6)
