import matplotlib.pyplot as plt
from ..vis._visuals import plot_mosaic, plot_mosaic_marginal

def count_leaf_elements(d):
  count = 0
  for value in d.values():
    if isinstance(value, dict):
      count += count_leaf_elements(value)
    else:
      count += 1
  return count

class ModelVisualiser:
  def __init__(self, summariser):
    self.summariser = summariser
    
  def plot_rate(self):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = plot_mosaic(self.summariser.summarise_rate()[1], ax=ax, title='Posterior contact rate')
    
    # Add a colorbar
    fig.subplots_adjust(right=0.8)  # Make space for the colorbar
    cbar_ax = fig.add_axes([0.82, 0.15, 0.05, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Contact rate')
    
    # Set colorbar title fontsize and tick size
    cbar_ax.yaxis.label.set_fontsize(9)  # Set label fontsize
    cbar_ax.tick_params(labelsize=8)  # Set tick font size
    
    fig.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust layout to account for colorbar
    
    return fig, ax
  
  def plot_cint(self):
    sum_cint = self.summariser.summarise_cint()
    num_elements = count_leaf_elements(sum_cint)
    
    fig, ax = plt.subplots(num_elements // 3, 3, figsize=(15, 5 * (num_elements // 3)))
    axes = ax.flatten()
    
    # Create a list to store all the plot objects for the colorbar
    plot_objects = []
    
    for i, (key, value) in enumerate(sum_cint.items()):
      if isinstance(value, dict):
        for j, (subkey, subvalue) in enumerate(value.items()):
          im = plot_mosaic(subvalue[1], ax=axes[i * 3 + j], title=f'Posterior contact intensity {key} {subkey}')
          plot_objects.append(im)
      else:
        im = plot_mosaic(value[1], ax=axes[i], title=f'Posterior contact intensity {key}')
        plot_objects.append(im)
    
    # Add a colorbar that applies to all subplots
    fig.subplots_adjust(right=0.9)  # Make space for the colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(plot_objects[0], cax=cbar_ax, label='Contact intensity')
    cbar_ax.yaxis.label.set_fontsize(9)  # Set label fontsize
    cbar_ax.tick_params(labelsize=8)  # Set tick font size
    
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to account for colorbar

    return fig, ax
  
  def plot_mcint(self, evaluator=None):
    sum_mcint = self.summariser.summarise_mcint()
    num_elements = count_leaf_elements(sum_mcint)
    
    fig, ax = plt.subplots(num_elements // 3, 3, figsize=(15, 5 * (num_elements // 3)))
    axes = ax.flatten()
    for i, (key, value) in enumerate(sum_mcint.items()):
      if isinstance(value, dict):
        for j, (subkey, subvalue) in enumerate(value.items()):
          _ = plot_mosaic_marginal(
            subvalue[1],
            axes[i * 3 + j],
            subvalue[0],
            subvalue[2],
            color='#004c6d',
            title=f'Posterior marginal contact intensity {key} {subkey}',
            label='Posterior estimate'
          )
          if evaluator is not None:
            _ = plot_mosaic_marginal(
              evaluator.mcint_true[key][subkey],
              axes[i * 3 + j],
              color='#de425b',
              label='True value'
            )
      else:
        axes[i] = plot_mosaic_marginal(
          value[1],
          axes[i],
          value[0],
          value[2],
          title=f'Posterior marginal contact intensity {key}'
        )
    fig.tight_layout()

    return fig, ax