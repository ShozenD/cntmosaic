import numpy as np


def _default_style(label_angle_x: int = 0, legend_position: str = "right") -> dict:
    """Return the base Altair style config shared by all mosaic plot functions."""
    return {
        "x_axis": {
            "labelFontSize": 10,
            "titleFontSize": 10,
            "titleFontWeight": "normal",
            "labelFontWeight": "normal",
            "labelAngle": label_angle_x,
            "grid": False,
        },
        "y_axis": {
            "labelFontSize": 10,
            "titleFontSize": 10,
            "titleFontWeight": "normal",
            "labelFontWeight": "normal",
            "grid": False,
        },
        "title": {"fontSize": 10, "fontWeight": "normal", "anchor": "middle"},
        "legend": {
            "labelFontSize": 10,
            "labelFontWeight": "normal",
            "titleFontSize": 10,
            "titleFontWeight": "normal",
            "orient": legend_position,
        },
    }


def _merge_style(default: dict, override: dict | None) -> dict:
    """Merge *override* into *default*, updating nested dicts in-place."""
    if not override:
        return default
    for key, val in override.items():
        if key in default:
            default[key].update(val)
        else:
            default[key] = val
    return default


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