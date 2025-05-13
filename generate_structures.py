import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from multiprocessing import Pool
import structure_models
from ase.visualize import view
from ase.io import read, write
from ase.build import stack


### Unit cell generation and modification 

def model_positions_to_ASE_positions(model_positions):
    input_pos_list = []
    for i in model_positions.values():
        for j in i:
            input_pos_list.append(j)
    return input_pos_list

def layer(model, cell_params, formula):
    """Create a single in-plane bulk (one layer) by repeating a model."""
    a, b, c, alpha, beta, gamma = cell_params
    cell_matrix = cellpar_to_cell([a, b, c, alpha, beta, gamma])
    positions = model_positions_to_ASE_positions(model)
    cartesian_positions = np.dot(positions, cell_matrix.T)  # Correct axis order
    unit_cell = Atoms(formula, positions=cartesian_positions, cell=cell_matrix, pbc=[True, True, False])
    
    return unit_cell

def build_multi_layer_unit(layer_1, layer_2, N_layers, cell_params, shift_vectors):    
    """
    Builds a multilayer ASE Atoms object by alternating two layers and applying shifts per 2-layer block.

    Parameters:
    - layer_1: ASE Atoms object (first type of layer)
    - layer_2: ASE Atoms object (second type of layer)
    - N_layers: Total number of layers (must be even)
    - cell_params: Tuple of (a, b, c) cell dimensions
    - shift_vectors: Optional list of (dx, dy) shift tuples per 2-layer block, relative to (a, b)

    Returns:
    - ASE Atoms object containing the full stacked structure.
    """
    a, b, c, alpha, beta, gamma = cell_params

    interlayer_spacing = c * 1  # in Angs
    supercell_vacuum = c * 0  # in Angs

    # 3) Loop to build each layer and collect them
    layers = []
    for i, (dx, dy) in enumerate(shift_vectors):
        layer_A = layer_1.copy()
        layer_A.translate((0.0, 0.0, 2*i * interlayer_spacing))
        layers.append(layer_A)

        layer_B = layer_2.copy()
        # compute real‐space in-plane shift = fx * a_vec + fy * b_vec
        shift_xy = dx * np.array([a, 0.0, 0.0]) + dy * np.array([0.0, b, 0.0])
        # add z-offset = i * interlayer
        shift = shift_xy + np.array((0.0, 0.0, (2*i + 1) * interlayer_spacing))
        layer_B.translate(shift)
        layers.append(layer_B)

    # Combine into one Atoms object
    stacked = sum(layers[1:], layers[0])

    # 5) Enlarge the cell in z to add vacuum
    cell = stacked.get_cell()
    cell[2,2] = N_layers * interlayer_spacing + supercell_vacuum
    stacked.set_cell(cell)
    stacked.set_pbc([True, True, False])

    return stacked

def build_full_bulk(unit_cell, num_repeats):
    """Creates the full bulk structure by repeating the two-layer unit cell in the z-direction."""
    return unit_cell.repeat(num_repeats)

def generate_bulk(cell_params, formula, bulk_dimensions, model1, model2=structure_models.model_2, shift = 0.5, N_layers=8):
    """Helper function to generate a bulk structure for a given strained unit cell."""
    layer_1 = layer(model1, cell_params, formula)
    layer_2 = layer(model2, cell_params, formula)

    shift_vectors = [(shift , 0.0), 
                    (-1 * shift , 0.0), 
                     (0.0 , shift),
                     (0.0 , -1 * shift),]

    """if model1 == structure_models.model_3:
        N_layers = 4

        shift_vectors = [(shift , 0.0), 
                        (-1 * shift , 0.0)]"""
    
    layered_cell = build_multi_layer_unit(layer_1, layer_2, N_layers=N_layers, cell_params=cell_params, shift_vectors=shift_vectors)
    names = structure_models.model_names
    models = structure_models.models
    write(f"./structures/CsV3Sb5_CDW_{names[models.index(model1)]}.cif", layered_cell)

    full_bulk = build_full_bulk(layered_cell, bulk_dimensions)

    return full_bulk

def compute_twin_bulks_parallel(strained_unit_cells, formula, bulk_dimensions, model1, model2):
    """Computes twin bulk structures in parallel, ensuring output remains a list of ASE Atoms objects."""
    
    # Create argument tuples for each bulk
    args = [(cell_params, formula, bulk_dimensions, model1, model2) for cell_params in strained_unit_cells]

    with Pool() as pool:
        twin_bulks = pool.starmap(generate_bulk, args)  # Use starmap to unpack arguments

    return twin_bulks  # Return as a list, NOT a NumPy array







