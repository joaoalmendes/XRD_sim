import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from multiprocessing import Pool
import structure_models

### Unit cell generation and modification 

def model_positions_to_ASE_positions(model_positions):
    input_pos_list = []
    for i in model_positions.values():
        for j in i:
            input_pos_list.append(j)
    return input_pos_list

def unit_cell(cell_params, formula, input_positions):
    a, b, c, alpha, beta, gamma = cell_params
    cell_matrix = cellpar_to_cell([a, b, c, alpha, beta, gamma])
    atoms = Atoms(
        formula,
        positions=np.dot(input_positions, cell_matrix),
        cell=cell_matrix,
        pbc=True,
    )

    return atoms

def in_plane_bulk(model, cell_params, formula, in_plane_repeats, N_layers):
    """Create a single in-plane bulk (one layer) by repeating a model."""
    a, b, c, alpha, beta, gamma = cell_params
    #if N_layers == 4:
    #    c *= 2
    if model == structure_models.model_3:
        b = a/2
    cell_matrix = cellpar_to_cell([a, b, c, alpha, beta, gamma])
    positions = model_positions_to_ASE_positions(model)
    unit_cell = Atoms(formula, positions=np.dot(positions, cell_matrix), cell=cell_matrix, pbc=True)

    bulk_layer = unit_cell.repeat((in_plane_repeats[0], in_plane_repeats[1], 1))
    
    return bulk_layer

def build_multi_layer_unit(layer_1, layer_2, N_layers, cell_params, shift_vectors=None):    
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
    if N_layers % 2 != 0:
        raise ValueError("N_layers must be even.")

    a, b, c, alpha, beta, gamma = cell_params
    dz = layer_1.cell[2, 2]  # assumes all layers have same thickness

    num_blocks = N_layers // 2
    if shift_vectors is None:
        shift_vectors = [(0.0, 0.0)] * num_blocks

    if len(shift_vectors) != num_blocks:
        raise ValueError("shift_vectors must have length N_layers // 2")

    full_layers = []
    for i in range(N_layers):
        is_odd = (i % 2 == 1)
        base_layer = layer_1.copy() if not is_odd else layer_2.copy()

        block_index = i // 2
        dx, dy = shift_vectors[block_index] if is_odd else (0.0, 0.0)

        base_layer.positions[:, 0] += dx * a
        base_layer.positions[:, 1] += dy * b
        base_layer.positions[:, 2] += i * dz

        full_layers.append(base_layer)

    full_structure = sum(full_layers, Atoms())

    new_cell = layer_1.cell.copy()
    new_cell[2, 2] = dz * N_layers
    full_structure.set_cell(new_cell)

    return full_structure

def build_full_bulk(unit_cell, num_repeats):
    """Creates the full bulk structure by repeating the two-layer unit cell in the z-direction."""
    return unit_cell.repeat((1, 1, num_repeats[2]))

def generate_bulk(cell_params, formula, bulk_dimensions, model1, model2, shift = 0.5, N_layers=8):
    """Helper function to generate a bulk structure for a given strained unit cell."""
    layer_1 = in_plane_bulk(model1, cell_params, formula, bulk_dimensions, N_layers)
    layer_2 = in_plane_bulk(model2, cell_params, formula, bulk_dimensions, N_layers)

    shift_vectors = [(shift , 0.0), 
                    (-1 * shift , 0.0), 
                     (0.0 , shift),
                     (0.0 , -1 * shift),]

    if model1 == structure_models.model_3:
        N_layers = 4

        shift_vectors = [(shift , 0.0), 
                        (-1 * shift , 0.0)]
    
    layered_cell = build_multi_layer_unit(layer_1, layer_2, N_layers=N_layers, cell_params=cell_params, shift_vectors=shift_vectors)
    full_bulk = build_full_bulk(layered_cell, bulk_dimensions)

    return full_bulk

def compute_twin_bulks_parallel(strained_unit_cells, formula, bulk_dimensions, model1, model2):
    """Computes twin bulk structures in parallel, ensuring output remains a list of ASE Atoms objects."""
    
    # Create argument tuples for each bulk
    args = [(cell_params, formula, bulk_dimensions, model1, model2) for cell_params in strained_unit_cells]

    with Pool() as pool:
        twin_bulks = pool.starmap(generate_bulk, args)  # Use starmap to unpack arguments

    return twin_bulks  # Return as a list, NOT a NumPy array







