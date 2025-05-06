import numpy as np
from numpy.typing import NDArray
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()
import matplotlib.pyplot as plt

from ..utils import AgeBins, depixilate
from ._utils import ravel_matrix, generate_vega_expression
from ..preprocess._utils import check_required_columns, expand_age_interval

def plot_mosaic(
	matrix: np.ndarray,
	title: str='Contact pattern',
	xlabel: str='Age of contacting individual',
	ylabel: str='Age of contacted individual',
	zlabel: str=None,
	width: int | float=250,
	height: int | float=250,
 	style_config: dict = None
) -> alt.Chart:
	"""
	Plot a mosaic visualization of a contact matrix.

	Parameters
	----------
	matrix : np.ndarray
			A 2D array representing the contact intensity or rate matrix.
	title : str, optional
			The title of the chart. Default is 'Contact pattern'.
	xlabel : str, optional
			Label for the x-axis. Default is 'Age of contacting individual'.
	ylabel : str, optional
			Label for the y-axis. Default is 'Age of contacted individual'.
	zlabel : str, optional
			Label for the color scale legend. If None, the legend is not displayed.
	width : int, optional
			The width of the chart in pixels. Default is 250.
	height : int, optional
			The height of the chart in pixels. Default is 250.
	style_config : dict, optional
			A dictionary to override default style settings for axes, title, and legend.
			The keys can include 'x_axis', 'y_axis', 'title', and 'legend'.

	Returns
	-------
	alt.Chart
			An Altair Chart object representing the mosaic visualisation of a contact matrix.

	Notes
	-----
	The function flattens the input matrix using `ravel_matrix` to extract the x and y indices along with corresponding values.
	It then constructs a DataFrame and configures the chart properties using default style settings,
	which can be further customized via the `style_config` parameter.
	"""
	# Default configurations for axis, title, and legend
	default_config = {
			'x_axis': {
					'labelFontSize': 10,
					'titleFontSize': 10,
					'titleFontWeight': 'normal',
					'labelFontWeight': 'normal',
					'labelAngle': 0,
					'grid': False
			},
			'y_axis': {
					'labelFontSize': 10,
					'titleFontSize': 10,
					'titleFontWeight': 'normal',
					'labelFontWeight': 'normal',
					'grid': False
			},
			'title': {
					'fontSize': 10,
					'fontWeight': 'normal',
					'anchor': 'middle'
			},
			'legend': {
					'labelFontSize': 10,
					'labelFontWeight': 'normal',
					'titleFontSize': 10,
					'titleFontWeight': 'normal',
					'orient': 'right'
			}
	}
	if style_config:
		for key in style_config:
			if key in default_config:
				default_config[key].update(style_config[key])
			else:
				default_config[key] = style_config[key]
	
	x_indices, y_indices, values = ravel_matrix(matrix)
	source = pd.DataFrame({'x': x_indices,
                        'y': y_indices,
                        'z': values})

	tick_values = list(range(0, matrix.shape[0], 10))
 
	chart = alt.Chart(source).mark_rect().encode(
		x = alt.X(
			'x:O',
			axis = alt.Axis(
		 		values=tick_values,
				**default_config['x_axis'],
			),
			title = xlabel,
		),
		y = alt.Y(
			'y:O',
			scale = alt.Scale(reverse=True),
			axis = alt.Axis(
		 		values=tick_values,
				**default_config['y_axis'],
			),
			title = ylabel,
		),
		color = alt.Color(
			'z:Q',
			scale = alt.Scale(scheme='spectral', reverse=True),
			title = zlabel,
			legend=alt.Legend(**default_config['legend']) if zlabel else None
		)
	).properties(
		width=width,
		height=height,
		title=alt.TitleParams(
			text=title,
			**default_config['title']
		),
	)
 
	return chart

def plot_mosaic_pixilated(
  matrix: np.ndarray,
  age_bins: AgeBins,
  title: str='Contact pattern',
	xlabel: str='Age of contacting individual',
	ylabel: str='Age of contacted individual',
	zlabel: str=None,
	width: int | float=250,
	height: int | float=250,
  style_config: dict = None
) -> alt.Chart:  
  # Default configurations for axis, title, and legend
	default_config = {
			'x_axis': {
					'labelFontSize': 10,
					'titleFontSize': 10,
					'titleFontWeight': 'normal',
					'labelFontWeight': 'normal',
					'labelAngle': -45,
					'grid': False
			},
			'y_axis': {
					'labelFontSize': 10,
					'titleFontSize': 10,
					'titleFontWeight': 'normal',
					'labelFontWeight': 'normal',
					'grid': False
			},
			'title': {
					'fontSize': 10,
					'fontWeight': 'normal',
					'anchor': 'middle'
			},
			'legend': {
					'labelFontSize': 10,
					'labelFontWeight': 'normal',
					'titleFontSize': 10,
					'titleFontWeight': 'normal',
					'orient': 'right'
			}
	}
	if style_config:
		for key in style_config:
			if key in default_config:
				default_config[key].update(style_config[key])
			else:
				default_config[key] = style_config[key]
    
	expanded_matrix = depixilate(matrix, age_bins)
	
	x_indices, y_indices, values = ravel_matrix(expanded_matrix)
	source = pd.DataFrame({'x': x_indices,
                        'y': y_indices,
                        'z': values})

	tick_pos = [np.floor(np.mean([age_bins.left[i], age_bins.right[i] + 1])) for i in range(len(age_bins.left))]
	tick_labels = [f'[{age_bins.left[i]},{age_bins.right[i] + 1})' for i in range(len(age_bins.left))]
	expression = generate_vega_expression(tick_pos, tick_labels)
 
	chart = alt.Chart(source).mark_rect().encode(
		x = alt.X(
			'x:O',
			axis = alt.Axis(
		 		values=tick_pos,
				labelExpr=expression,
				**default_config['x_axis'],
			),
			title = xlabel,
		),
		y = alt.Y(
			'y:O',
			scale = alt.Scale(reverse=True),
			axis = alt.Axis(
		 		values=tick_pos,
				labelExpr=expression,
				**default_config['y_axis'],
			),
			title = ylabel,
		),
		color = alt.Color(
			'z:Q',
			scale = alt.Scale(scheme='spectral', reverse=True),
			title = zlabel,
			legend=alt.Legend(**default_config['legend']) if zlabel else None
		)
	).properties(
		width=width,
		height=height,
		title=alt.TitleParams(
			text=title,
			**default_config['title']
		),
	)
 
	return chart

def plot_mosaic_marginal(
  mcint: np.ndarray,
  mcint_lb: np.ndarray = None,
  mcint_ub: np.ndarray = None,
  width: int = 250,
  height: int = 250,
  title: str = 'Contact intensity',
  style_config: dict = None
) -> alt.Chart:
  """
	Plot the marginal contact intensity with optional uncertainty bands.
	
 Parameters
	----------
	mcint : np.ndarray
		Array representing the main contact intensity values.
	mcint_lb : np.ndarray, optional
		Array representing the lower bound of the uncertainty band. If provided, both mcint_lb and mcint_ub are used to
		display an error band around the line plot. Default is None.
	mcint_ub : np.ndarray, optional
		Array representing the upper bound of the uncertainty band. If provided, both mcint_lb and mcint_ub are used to
		display an error band around the line plot. Default is None.
	width : int, optional
		The width of the resulting chart in pixels. Default is 250.
	height : int, optional
		The height of the resulting chart in pixels. Default is 250.
	title : str, optional
		The title for the chart. Default is 'Contact intensity'.
	style_config : dict, optional
		A dictionary for overriding default style configurations for the axes and title. The keys should correspond to the
		configuration parts ('x_axis', 'y_axis', or 'title') and the values should be dictionaries of style parameters.
		Default is None.
  
	Returns
	-------
	alt.Chart
		An Altair Chart object that visualizes the marginal contact intensity. When error bounds are provided, the chart includes
		an error band alongside the main line plot.
	"""
  
  config = {
    'x_axis': {
      'values': list(range(0, 100, 10)),
      'labelFontSize': 10,
      'titleFontSize': 10,
      'titleFontWeight': 'normal',
      'labelFontWeight': 'normal',
      'labelAngle': 0,
      'grid': True
    },
    'y_axis': {
      'values': list(range(0, 100, 5)),
      'labelFontSize': 10,
      'titleFontSize': 10,
      'titleFontWeight': 'normal',
      'labelFontWeight': 'normal',
      'grid': True
    },
    'title': {
      'fontSize': 10,
      'fontWeight': 'normal',
      'anchor': 'middle'
    }
  }
  
  if style_config:
    for key in style_config:
      if key in config:
        config[key].update(style_config[key])
      else:
        config[key] = style_config[key]
  
  df = pd.DataFrame({'x': np.arange(mcint.size), 'y': mcint})
  has_band = mcint_lb is not None and mcint_ub is not None
  if has_band:
    df['l'] = mcint_lb
    df['u'] = mcint_ub
  
  x_axis = alt.Axis(title='Age of contacting individuals', **config['x_axis'])
  y_axis = alt.Axis(title='Contact intensity', **config['y_axis'])
  
  base = alt.Chart(df).encode(x=alt.X('x:O', axis=x_axis), y=alt.Y('y:Q', axis=y_axis))
  line = base.mark_line()
  
  if has_band:
    band = alt.Chart(df).mark_errorband().encode(
      x=alt.X('x:O', axis=x_axis),
      y=alt.Y('l:Q', axis=y_axis),
      y2='u:Q'
    )
    chart = band + line
  else:
    chart = line
    
  return chart.properties(width=width, height=height,
              title=alt.TitleParams(text=title, **config['title']))

		
def plot_mosaic_empirical(
		data: pd.DataFrame,
	ax,
	title: str = 'Empirical contact intensity',
	xlabel: str | None = 'Age of contacting individual',
	ylabel: str | None = 'Age of contacted individual',
	vmin: float = None,
	vmax: float = None,
	cbar_ax=None,
	cbar_label: str | None = 'Contact intensity',
	cmap: str = 'Spectral_r'
):
	# Check inputs
	check_required_columns(data)
	
	# Check if the data is coarse-grained
	is_coarse = 'age_grp_cnt' in data.columns
	if is_coarse: data = expand_age_interval(data, 'age_grp_cnt')
	
	data['cint'] = data['y'] / data['N']
	A = data['age_part'].max() - data['age_part'].min() + 1
	cint = data['cint'].values.reshape(A, A).T
	
	im = ax.imshow(cint, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

	if is_coarse:
		# Set y tick labels at the center of the intervals
		ytick_locs = data['age_grp_cnt'].apply(lambda x: x.mid).unique()
		ytick_labels = data['age_grp_cnt'].unique()
		ytick_labels = [str(x) for x in ytick_labels]
	
		ax.set_yticks(ytick_locs)
		ax.set_yticklabels(ytick_labels)

		# Set major grid lines at the lower bounds of the intervals
		ytick_locs_minor = data['age_grp_cnt'].apply(lambda x: x.left).unique()
		ax.set_yticks(ytick_locs_minor, minor=True)  # Set these as minor ticks
	
	ax.tick_params(axis='both', which='major', labelsize=8)
	ax.grid(which='major', axis='both', visible=False)
	ax.set_title(title, fontsize=9, loc='left')
	ax.set_xlabel(xlabel, fontsize=8)
	ax.set_ylabel(ylabel, fontsize=8)
 
	# Add color bar if a cbar_ax is provided
	if cbar_ax:
		cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
		cbar.set_label(cbar_label, fontsize=8)
		cbar.ax.tick_params(labelsize=8)
 
	return im