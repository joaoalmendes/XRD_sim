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

def DoubleLayered_unit_cell(layer_1, layer_2, interlayer_shift=None):
    """Creates a new unit cell with two stacked layers, applying an interlayer shift."""
    
    # Make copies to avoid modifying original layers
    layer_1 = layer_1.copy()
    layer_2 = layer_2.copy()

    # Apply interlayer shift to layer_2 if specified
    if interlayer_shift:
        dx, dy = interlayer_shift
        layer_2.positions[:, 0] += dx
        layer_2.positions[:, 1] += dy

    # Stack layer_2 on top of layer_1
    layer_2.positions[:, 2] += layer_1.cell[2, 2]  # Shift by c-axis value
    
    # Combine both layers into a single Atoms object
    two_layer_unit = layer_1 + layer_2  # ASE allows adding Atoms objects
    
    # Update the unit cell height to fit both layers
    new_cell = layer_1.cell.copy()
    new_cell[2, 2] *= 2  # Double the c-axis length
    two_layer_unit.set_cell(new_cell)
    
    return two_layer_unit

def QuadrupledLayered_unit_cell(layer_1, layer_2, layer_3, layer_4, interlayer_shifts=None):
    """Creates a new unit cell with four stacked layers, applying interlayer shifts if specified.
    
    Parameters:
    - layer_1, layer_2, layer_3, layer_4: ASE Atoms objects for the four layers
    - interlayer_shifts: List of (dx, dy) shifts for layers 2, 3, and 4 relative to the previous one
    """

    # Make copies to avoid modifying the original layers
    layers = [layer_1.copy(), layer_2.copy(), layer_3.copy(), layer_4.copy()]

    # Default shifts if not provided
    if interlayer_shifts is None:
        interlayer_shifts = [(0, 0), (0, 0), (0, 0)]  # No shift by default

    if len(interlayer_shifts) != 3:
        raise ValueError("interlayer_shifts must be a list of 3 (dx, dy) tuples.")

    # Apply interlayer shifts to layers 2, 3, and 4
    for i, shift in enumerate(interlayer_shifts):
        dx, dy = shift
        layers[i + 1].positions[:, 0] += dx  # Shift in x
        layers[i + 1].positions[:, 1] += dy  # Shift in y

    # Correctly stack layers along the c-axis
    for i in range(1, 4):  # Start from layer 2, shifting based on the previous one
        layers[i].positions[:, 2] += i * layers[0].cell[2, 2]  # Ensure proper stacking

    # Combine all four layers into a single Atoms object
    four_layer_unit = sum(layers, Atoms())

    # Update the unit cell height to fit four layers
    new_cell = layers[0].cell.copy()
    new_cell[2, 2] *= 4  # Correctly increase the c-axis length
    four_layer_unit.set_cell(new_cell)

    return four_layer_unit

def build_full_bulk(unit_cell, num_repeats):
    """Creates the full bulk structure by repeating the two-layer unit cell in the z-direction."""
    return unit_cell.repeat((1, 1, num_repeats[2]))

def generate_bulk(cell_params, formula, bulk_dimensions, model1, model2, shift = -0.5, N_layers=4):
    """Helper function to generate a bulk structure for a given strained unit cell."""
    layer_1 = in_plane_bulk(model1, cell_params, formula, bulk_dimensions, N_layers)
    layer_2 = in_plane_bulk(model2, cell_params, formula, bulk_dimensions, N_layers)

    if model1 == structure_models.model_3:
        N_layers = 2

    if N_layers == 2:
        interlayer_shift = (shift * cell_params[0], 0.0 * cell_params[1])  # Adjust shift based on a
        two_layer_cell = DoubleLayered_unit_cell(layer_1, layer_2, interlayer_shift)
        full_bulk = build_full_bulk(two_layer_cell, bulk_dimensions)
    if N_layers == 4:
        interlayer_shift = [(0.0 * cell_params[0], shift * cell_params[1]), (0.0 * cell_params[0], 0.0 * cell_params[1]), (0.0 * cell_params[0], shift * cell_params[1])]
        four_layer_cell = QuadrupledLayered_unit_cell(layer_1, layer_2, layer_2, layer_2, interlayer_shift)
        full_bulk = build_full_bulk(four_layer_cell, bulk_dimensions)

    return full_bulk

def compute_twin_bulks_parallel(strained_unit_cells, formula, bulk_dimensions, model1, model2):
    """Computes twin bulk structures in parallel, ensuring output remains a list of ASE Atoms objects."""
    
    # Create argument tuples for each bulk
    args = [(cell_params, formula, bulk_dimensions, model1, model2) for cell_params in strained_unit_cells]

    with Pool() as pool:
        twin_bulks = pool.starmap(generate_bulk, args)  # Use starmap to unpack arguments

    return twin_bulks  # Return as a list, NOT a NumPy array







