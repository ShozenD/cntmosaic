import numpy as np

def ravel_matrix(matrix: np.ndarray) -> tuple:
		"""
		Ravel a matrix in column-major order (Fortran order).
		"""

		x_idx, y_idx = np.indices(matrix.shape)
  
		x_indices = x_idx.ravel(order='F')
		y_indices = y_idx.ravel(order='F')
		values = matrix.ravel(order='F')
  
		return x_indices, y_indices, values

def generate_vega_expression(tick_pos, tick_labels):
    # Build a mapping of the integer tick position (as string) to the corresponding label.
    mapping_entries = []
    for pos, label in zip(tick_pos, tick_labels):
        # Convert pos to integer to eliminate decimals if applicable.
        mapping_entries.append(f"'{int(pos)}':'{label}'")
    mapping_str = "{" + ", ".join(mapping_entries) + "}"
    # Append the selector "[datum.value]" to form the full expression.
    return mapping_str + "[datum.value]"