import numpy as np
import pandas as pd
import altair as alt
from ..vis._visuals import plot_mosaic

def count_leaf_elements(d):
  count = 0
  for value in d.values():
    if isinstance(value, dict):
      count += count_leaf_elements(value)
    else:
      count += 1
  return count

def df_from_dict(d):
  dfs = []
  # case with no sub-category
  if not isinstance(d, dict):
    if len(d.shape) == 3:
      _, m, n = d.shape
      i_idx, j_idx = np.indices((m, n))
      df_matrix = pd.DataFrame({
        'x': i_idx.flatten(order='F'),
        'y': j_idx.flatten(order='F'),
        'z': d[1].flatten(order='F')
      })
      df_matrix['label'] = 'general'
      dfs.append(df_matrix)
    elif len(d.shape) == 2: # Vector
      _, n = d.shape
      i_idx = np.arange(n)
      df_vector = pd.DataFrame({
        'x': i_idx,
        'y': d[1],
        'l': d[0],
        'u': d[2]
      })
      df_vector['label'] = 'general'
      dfs.append(df_vector)
    return pd.concat(dfs, ignore_index=True)
  
  for key, values in d.items():
    if values.ndim == 3: # Matrix
      _, m, n = values.shape
      i_idx, j_idx = np.indices((m, n))
      df_matrix = pd.DataFrame({
        'x': i_idx.flatten(order='F'),
        'y': j_idx.flatten(order='F'),
        'z': values[1].flatten(order='F')
      })
      df_matrix['label'] = key
      dfs.append(df_matrix)
    elif values.ndim == 2: # Vector
      _, n = values.shape
      i_idx = np.arange(n)
      df_vector = pd.DataFrame({
        'x': i_idx,
        'y': values[1],
        'l': values[0],
        'u': values[2]
      })
      df_vector['label'] = key
      dfs.append(df_vector)
  
  return pd.concat(dfs, ignore_index=True)

class ModelVisualiser:
  default_config = {
      'x_axis': {
          'labelFontSize': 10,
          'titleFontSize': 10,
          'titleFontWeight': 'normal',
          'labelFontWeight': 'normal',
          'labelAngle': 0,
      },
      'y_axis': {
          'labelFontSize': 10,
          'titleFontSize': 10,
          'titleFontWeight': 'normal',
          'labelFontWeight': 'normal',
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
  
  def __init__(self, summariser):
    self.summariser = summariser
    
  def plot_rate(self, width=250, height=250, style_config=None):
    chart = plot_mosaic(self.summariser.summarise_rate()[1],
                        title='Estimated contact rate',
                        zlabel='rate',
                        width=width,
                        height=height,
                        style_config=style_config)    
    return chart
  
  def plot_cint(self, width=250, height=250, facet_columns=3, style_config=None):
    if style_config:
      for key, conf in style_config.items():
        self.default_config.setdefault(key, {}).update(conf)

    tick_values = np.arange(0, 100, 10)
    sum_cint = self.summariser.summarise_cint()

    return {
      k: alt.Chart(df_from_dict(v)).mark_rect().encode(
        x=alt.X(
          'x:O',
          axis=alt.Axis(values=tick_values, grid=False, **self.default_config['x_axis']),
          title='Age of contacting individuals'
        ),
        y=alt.Y(
          'y:O',
          scale=alt.Scale(reverse=True),
          axis=alt.Axis(values=tick_values, grid=False, **self.default_config['y_axis']),
          title='Age of contacted individuals'
        ),
        color=alt.Color(
          'z:Q',
          scale=alt.Scale(scheme='spectral', reverse=True),
          title='intensity',
          legend=alt.Legend(**self.default_config['legend'])
        ),
        facet=alt.Facet('label:N', title=None, columns=facet_columns),
      ).properties(
        width=width,
        height=height,
        title={'text': [f'Estimated contact intensity: {k}'], **self.default_config['title']}
      )
      for k, v in sum_cint.items()
    }
    
  def plot_mcint(self, evaluator=None, width=250, height=250, style_config=None):
    if style_config:
      for key, conf in style_config.items():
        self.default_config.setdefault(key, {}).update(conf)
        
    sum_mcint = self.summariser.summarise_mcint()

    x_axis = alt.Axis(
      title='Age of contacting individuals',
      grid=True,
      **self.default_config['x_axis']
    )
    y_axis = alt.Axis(
      title='Contact intensity',
      grid=True,
      **self.default_config['y_axis']
    )

    charts = {}
    for key, val in sum_mcint.items():
      source = df_from_dict(val)

      if evaluator is not None:
        mcint_true = evaluator.mcint_true[key]
        # Create a dataframe from all arrays in mcint_true
        df_mcint_true = pd.concat([
          pd.DataFrame({
            'label': k,
            'x': np.arange(v.size),
            'y_true': v
          }) for k, v in mcint_true.items()
        ])
        # Merge with source dataframe on the common keys
        source = pd.merge(source, df_mcint_true, on=['label', 'x'], how='left')
        line = alt.Chart(source).mark_line(color='red').encode(
          x=alt.X('x:Q', axis=x_axis),
          y=alt.Y('y_true:Q', title='Contact intensity')
        )

      base = alt.Chart(source).mark_line().encode(
        x=alt.X('x:Q', axis=x_axis),
        y=alt.Y('y:Q', axis=y_axis)
      )

      band = alt.Chart(source).mark_errorband().encode(
        x=alt.X('x:Q', axis=x_axis),
        y=alt.Y('l:Q', title='Contact intensity'),
        y2='u:Q'
      )

      chart = base + band + line if evaluator is not None else base + band
      charts[key] = chart.properties(width=width, height=height).facet(
        column=alt.Column('label:N', title=None),
        columns=3,
        title=alt.TitleParams(
          text=[f'Estimated contact intensity: {key}'],
          **self.default_config['title']
        )
      )
      
      
    # Layout charts in a grid
    # Determine number of rows and columns based on number of elements
    # cols = 3
    # rows = (len(charts) + cols - 1) // cols

    # Build grid using Altair's concat operators
    # chart_grid = None
    #for r in range(rows):
    #    row_charts = charts[r * cols:(r + 1) * cols]
    #    row = row_charts[0]
    #    for c in row_charts[1:]:
    #        row |= c
    #    chart_grid = row if chart_grid is None else chart_grid & row
    #return chart_grid
      
    return charts