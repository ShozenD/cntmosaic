import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

from ..preprocess._utils import check_required_columns, expand_age_interval

def plot_contact_matrix(
    ax,
    cint: NDArray,
    title: str = 'True Contact Intensity',
    xlabel: str | None = 'Age of contacting individual',
    ylabel: str | None = 'Age of contacted individual',
    vmin: float = None,
    vmax: float = None,
    cbar_ax=None,
    cbar_label: str = 'Contact Intensity',
    cmap: str = 'Spectral_r'
):
    """
    Plot a social contact intensity matrix on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the intensity matrix.
    cint : ndarray
        The contact intensity matrix to plot.
    title : str, optional
        Title of the plot. Default is 'True Contact Intensity'.
    xlabel : str or None, optional
        Label for the x-axis. If None, the x-axis labels are removed.
        Default is 'Age of contacting individual'.
    ylabel : str or None, optional
        Label for the y-axis. If None, the y-axis labels are removed.
        Default is 'Age of contacted individual'.
    vmin : float, optional
        Minimum value for the color scale. Defaults to the minimum of `cint`.
    vmax : float, optional
        Maximum value for the color scale. Defaults to the maximum of `cint`.
    cbar_ax : matplotlib.axes.Axes or None, optional
        Axis for the color bar. If None, no color bar is added.
    cbar_label : str, optional
        Label for the color bar. Default is 'Contact Intensity'.
    cmap : str, optional
        Colormap for the plot. Default is 'Spectral_r'.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The resulting image plot.
    """
    # Set vmin and vmax if not provided
    vmin = cint.min() if vmin is None else vmin
    vmax = cint.max() if vmax is None else vmax

    # Plot the matrix
    im = ax.imshow(cint, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9, loc='left')

    # Set x-axis label
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    else:
        ax.set_xticklabels([])

    # Set y-axis label
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    else:
        ax.set_yticklabels([])

    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(which='major', axis='both', visible=False)

    # Add color bar if a cbar_ax is provided
    if cbar_ax:
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    return im

def plot_contact_marginal(
    ax,
    mcint,
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
        
    if title:
        ax.set_title(title, fontsize=9, loc='left')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if xlabel:  
        ax.set_xlabel(xlabel, fontsize=8)

    ax.tick_params(axis='both', which='major', labelsize=8)
    
def plot_empirical_contact_matrix(
	ax,
	data: pd.DataFrame,
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
 
	if is_coarse:
		data = expand_age_interval(data, 'age_grp_cnt')
  
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