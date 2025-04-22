import numpy as np
from numpy.typing import NDArray
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()
import matplotlib.pyplot as plt

from ._utils import ravel_matrix
from ..preprocess._utils import check_required_columns, expand_age_interval

def plot_mosaic(
  matrix: np.ndarray,
  title='Contact pattern',
  xlabel='Age of contacting individual',
  ylabel='Age of contacted individual',
  zlabel=None,
  axisLabelFontSize=10,
  axisTitleFontSize=10,
  axisTitleFontWeight='normal',
  axisLabelFontWeight='normal',
  titleFontSize=10,
  titleFontWeight='normal',
  legend=None,
  legendTitle=None,
  legendLabelFontSize=10,
  legendLabelFontWeight='normal',
  legendTitleFontSize=10,
  legendTitleFontWeight='normal',
	legendOrient='right',
  width=250,
  height=250
) -> alt.Chart: 
  
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
        labelFontSize=axisLabelFontSize,
        titleFontSize=axisTitleFontSize,
        titleFontWeight=axisTitleFontWeight,
        labelFontWeight=axisLabelFontWeight,
       	labelAngle=0,
        grid=False
      ),
			title = xlabel,
		),
		y = alt.Y(
			'y:O',
			scale = alt.Scale(reverse=True),
			axis = alt.Axis(
     		values=tick_values,
        labelFontSize=axisLabelFontSize,
        titleFontSize=axisTitleFontSize,
        titleFontWeight=axisTitleFontWeight,
        labelFontWeight=axisLabelFontWeight,
       	grid=False
      ),
			title = ylabel,
		),
		color = alt.Color(
			'z:Q',
			scale = alt.Scale(scheme='spectral', reverse=True),
			title = zlabel,
			legend = alt.Legend(
     		title=legendTitle,
        labelFontSize=legendLabelFontSize,
				labelFontWeight=legendLabelFontWeight,
				titleFontSize=legendTitleFontSize,
				titleFontWeight=legendTitleFontWeight,
				orient=legendOrient,
      ) if legend else None
		)
	).properties(
		width=width,
		height=height,
		title=alt.TitleParams(
    	text=title,
     	fontSize=titleFontSize,
      fontWeight=titleFontWeight,
      anchor='middle'
    ),
	)
 
	return chart

def plot_mosaic_marginal(
    mcint,
    ax,
    mcint_lb: NDArray | None = None,
    mcint_ub: NDArray | None = None,
    color: str = '#de425b',
    title: str = None,
    xlabel: str | None = 'Age of contacting individual',
    ylabel: str | None = 'Intensity',
    **kwargs
):
    """Plot the marginal contact intensity on a given axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the marginal contact intensity.
    mcint : ndarray
        The marginal contact intensity to plot.
    mcint_lb : ndarray or None, optional
        Lower bound of the marginal contact intensity. Default is None.
    mcint_ub : ndarray or None, optional
        Upper bound of the marginal contact intensity. Default is None.
    color : str, optional
        Color of the plot. Default is '#de425b'.
    title : str or None, optional
        Title of the plot. Default is None.
    **kwargs
        Additional keyword arguments to pass to the plot function.
    
    Returns
    -------
    None
    """
    ax.plot(mcint, c=color, **kwargs)
        
    if mcint_lb is not None and mcint_ub is not None:
        ax.fill_between(
			np.arange(len(mcint)),
			mcint_lb,
			mcint_ub,
			color=color,
			alpha=0.2
		)
    
    ax.set_title(title, fontsize=9, loc='left') if title else None
    ax.set_ylabel(ylabel, fontsize=8) if ylabel else None
    ax.set_xlabel(xlabel, fontsize=8) if xlabel else None
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    
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