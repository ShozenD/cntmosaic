import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
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
    rate_array = self.summariser.summarise_rate()  # assuming rate_array is a 2D numpy array
    
    return plot_mosaic(
                rate_array[1],
                title='Posterior contact rate',
                zlabel='Contact rate'
            )
  
  def plot_cint(self):
    sum_cint = self.summariser.summarise_cint()
    num_elements = count_leaf_elements(sum_cint)

    charts = []  # Store all individual charts here

    for key, value in sum_cint.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                chart = plot_mosaic(
                    subvalue[1],
                    title=f'Posterior contact intensity {key} {subkey}',
                    zlabel='Contact intensity'
                )
                charts.append(chart)
        else:
            chart = plot_mosaic(
                value[1],
                title=f'Posterior contact intensity {key}',
                zlabel='Contact intensity'
            )
            charts.append(chart)

    # Layout charts in a grid
    # Determine number of rows and columns based on number of elements
    cols = 3
    rows = (len(charts) + cols - 1) // cols

    # Build grid using Altair's concat operators
    chart_grid = None
    for r in range(rows):
        row_charts = charts[r * cols:(r + 1) * cols]
        row = row_charts[0]
        for c in row_charts[1:]:
            row |= c
        chart_grid = row if chart_grid is None else chart_grid & row

    return chart_grid

  
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