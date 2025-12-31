"""Visualization utilities for Bayesian contact matrix models.

This module provides visualization tools for analyzing and presenting
contact matrix estimation results from BRC model family.
"""

import altair as alt
import numpy as np
import pandas as pd

from ..models import BRCfine, BRCrefine, HiBRCfine, HiBRCrefine
from ..vis._visuals import plot_mosaic
from .evaluator._ModelEvaluatorBRC import ModelEvaluatorBRC
from .summariser._ModelSummariserBRC import ModelSummariserBRC


def count_leaf_elements(d):
    """Count the number of leaf (non-dictionary) elements in a nested dictionary.

    Recursively traverses a nested dictionary structure and counts all terminal
    values that are not themselves dictionaries.

    Parameters
    ----------
    d : dict
        A possibly nested dictionary structure.

    Returns
    -------
    int
        Total count of leaf elements in the dictionary.

    Examples
    --------
    >>> count_leaf_elements({'a': 1, 'b': {'c': 2, 'd': 3}})
    3
    >>> count_leaf_elements({'x': np.array([1, 2, 3])})
    1
    """
    count = 0
    for value in d.values():
        if isinstance(value, dict):
            count += count_leaf_elements(value)
        else:
            count += 1
    return count


def df_from_dict(d):
    """Convert model summary arrays to pandas DataFrame for plotting.

    Transforms model output dictionaries containing contact matrices or vectors
    into a long-form DataFrame suitable for Altair visualization. Handles both
    3D arrays (matrices with median/quantiles) and 2D arrays (vectors with
    lower/median/upper bounds).

    Parameters
    ----------
    d : dict or numpy.ndarray
        Model summary output. Can be:
        - numpy.ndarray of shape (3, m, n): Matrix with [lower, median, upper]
        - numpy.ndarray of shape (3, n): Vector with [lower, median, upper]
        - dict: Nested structure mapping labels to arrays (for stratified models)

    Returns
    -------
    pandas.DataFrame
        Long-form DataFrame with columns:
        - For matrices: 'x' (row index), 'y' (column index), 'z' (median value),
          'label' (stratification category)
        - For vectors: 'x' (index), 'y' (median), 'l' (lower bound),
          'u' (upper bound), 'label' (stratification category)

    Notes
    -----
    Arrays are flattened in Fortran order (column-major) to match contact matrix
    indexing conventions where age groups vary fastest along columns.

    Examples
    --------
    >>> matrix = np.random.rand(3, 10, 10)  # [lower, median, upper]
    >>> df = df_from_dict(matrix)
    >>> df.columns
    Index(['x', 'y', 'z', 'label'], dtype='object')

    >>> stratified = {'male': matrix, 'female': matrix}
    >>> df = df_from_dict(stratified)
    >>> df['label'].unique()
    array(['male', 'female'], dtype=object)
    """
    dfs = []
    # case with no sub-category
    if not isinstance(d, dict):
        if len(d.shape) == 3:
            _, m, n = d.shape
            i_idx, j_idx = np.indices((m, n))
            df_matrix = pd.DataFrame(
                {
                    "x": i_idx.flatten(order="F"),
                    "y": j_idx.flatten(order="F"),
                    "z": d[1].flatten(order="F"),
                }
            )
            df_matrix["label"] = "general"
            dfs.append(df_matrix)
        elif len(d.shape) == 2:  # Vector
            _, n = d.shape
            i_idx = np.arange(n)
            df_vector = pd.DataFrame({"x": i_idx, "y": d[1], "l": d[0], "u": d[2]})
            df_vector["label"] = "general"
            dfs.append(df_vector)
        return pd.concat(dfs, ignore_index=True)

    for key, values in d.items():
        if values.ndim == 3:  # Matrix
            _, m, n = values.shape
            i_idx, j_idx = np.indices((m, n))
            df_matrix = pd.DataFrame(
                {
                    "x": i_idx.flatten(order="F"),
                    "y": j_idx.flatten(order="F"),
                    "z": values[1].flatten(order="F"),
                }
            )
            df_matrix["label"] = key
            dfs.append(df_matrix)
        elif values.ndim == 2:  # Vector
            _, n = values.shape
            i_idx = np.arange(n)
            df_vector = pd.DataFrame(
                {"x": i_idx, "y": values[1], "l": values[0], "u": values[2]}
            )
            df_vector["label"] = key
            dfs.append(df_vector)

    return pd.concat(dfs, ignore_index=True)


class ModelVisualiser:
    """Visualization interface for BRC family contact matrix models.

    Provides publication-ready plots for model outputs including contact rates,
    contact intensities, and marginal contact intensities with credible intervals.
    Uses Altair for declarative visualization with consistent styling.

    Parameters
    ----------
    summariser : ModelSummariserBRC
        Model summariser instance containing posterior summaries of the fitted model.

    Attributes
    ----------
    summariser : ModelSummariserBRC
        The model summariser used to extract posterior statistics.
    default_config : dict
        Default styling configuration for plot axes, titles, and legends.
        Contains nested dicts for 'x_axis', 'y_axis', 'title', and 'legend'.

    Examples
    --------
    >>> from cntmosaic.analysis import ModelSummariserBRC, ModelVisualiser
    >>> from cntmosaic.models import BRCfine
    >>>
    >>> # After fitting a model
    >>> summariser = ModelSummariserBRC(model)
    >>> visualiser = ModelVisualiser(summariser)
    >>>
    >>> # Generate plots
    >>> rate_chart = visualiser.plot_rate()
    >>> cint_chart = visualiser.plot_cint()

    Notes
    -----
    All plotting methods return Altair Chart objects which can be:
    - Displayed in Jupyter notebooks
    - Saved to file: `chart.save('output.html')` or `chart.save('output.png')`
    - Composed with other charts using Altair operators (+, |, &)
    """

    default_config = {
        "x_axis": {
            "labelFontSize": 10,
            "titleFontSize": 10,
            "titleFontWeight": "normal",
            "labelFontWeight": "normal",
            "labelAngle": 0,
        },
        "y_axis": {
            "labelFontSize": 10,
            "titleFontSize": 10,
            "titleFontWeight": "normal",
            "labelFontWeight": "normal",
        },
        "title": {"fontSize": 10, "fontWeight": "normal", "anchor": "middle"},
        "legend": {
            "labelFontSize": 10,
            "labelFontWeight": "normal",
            "titleFontSize": 10,
            "titleFontWeight": "normal",
            "orient": "right",
        },
    }

    def __init__(self, summariser: ModelSummariserBRC):
        self.summariser = summariser

    def plot_rate(self, width=250, height=250, style_config=None):
        """Plot posterior median contact rate matrix as a heatmap.

        Creates a heatmap visualization of the estimated contact rate matrix,
        showing the median posterior rate of contacts between age groups.

        Parameters
        ----------
        width : int, optional
            Width of the plot in pixels, by default 250.
        height : int, optional
            Height of the plot in pixels, by default 250.
        style_config : dict, optional
            Custom style configuration to override defaults. Should contain
            keys like 'x_axis', 'y_axis', 'title', 'legend' with nested style
            parameters, by default None.

        Returns
        -------
        altair.Chart
            Altair chart object displaying the contact rate heatmap.

        Examples
        --------
        >>> visualiser = ModelVisualiser(summariser)
        >>> chart = visualiser.plot_rate(width=400, height=400)
        >>> chart.save('contact_rate.html')

        Notes
        -----
        The rate matrix represents the expected number of contacts per person
        per day between age groups, estimated from the model's posterior distribution.
        """
        chart = plot_mosaic(
            self.summariser.summarise_rate()[1],
            title="Estimated contact rate",
            zlabel="rate",
            width=width,
            height=height,
            style_config=style_config,
        )
        return chart

    def plot_cint(self, width=250, height=250, facet_columns=3, style_config=None):
        """Plot posterior median contact intensity matrix as a heatmap.

        Visualizes the estimated contact intensity matrix (rate adjusted for
        population structure). For hierarchical models (HiBRCfine, HiBRCrefine),
        returns separate plots for each stratification level.

        Parameters
        ----------
        width : int, optional
            Width of each plot in pixels, by default 250.
        height : int, optional
            Height of each plot in pixels, by default 250.
        facet_columns : int, optional
            Number of columns for faceted plots in hierarchical models,
            by default 3.
        style_config : dict, optional
            Custom style configuration to override defaults. Should contain
            keys like 'x_axis', 'y_axis', 'title', 'legend' with nested style
            parameters, by default None.

        Returns
        -------
        altair.Chart or dict of altair.Chart
            For BRCfine/BRCrefine: Single Altair chart object.
            For HiBRCfine/HiBRCrefine: Dictionary mapping stratification
            levels to their respective chart objects.

        Examples
        --------
        >>> # For non-hierarchical models
        >>> chart = visualiser.plot_cint(width=400, height=400)
        >>> chart.save('contact_intensity.html')

        >>> # For hierarchical models
        >>> charts = visualiser.plot_cint(facet_columns=2)
        >>> for level, chart in charts.items():
        ...     chart.save(f'cint_{level}.html')

        Notes
        -----
        Contact intensity differs from contact rate by accounting for the
        population age distribution, making it more suitable for comparing
        contact patterns across different populations or subgroups.
        """
        if style_config:
            for key, conf in style_config.items():
                self.default_config.setdefault(key, {}).update(conf)

        tick_values = np.arange(0, 100, 10)
        sum_cint = self.summariser.summarise_cint()

        if type(self.summariser.model) in (BRCfine, BRCrefine):
            return (
                alt.Chart(df_from_dict(sum_cint))
                .mark_rect()
                .encode(
                    x=alt.X(
                        "x:O",
                        axis=alt.Axis(
                            values=tick_values,
                            grid=False,
                            **self.default_config["x_axis"],
                        ),
                        title="Age of contacting individuals",
                    ),
                    y=alt.Y(
                        "y:O",
                        scale=alt.Scale(reverse=True),
                        axis=alt.Axis(
                            values=tick_values,
                            grid=False,
                            **self.default_config["y_axis"],
                        ),
                        title="Age of contacted individuals",
                    ),
                    color=alt.Color(
                        "z:Q",
                        scale=alt.Scale(scheme="spectral", reverse=True),
                        title="intensity",
                        legend=alt.Legend(**self.default_config["legend"]),
                    ),
                )
                .properties(
                    width=width,
                    height=height,
                    title={
                        "text": ["Estimated contact intensity"],
                        **self.default_config["title"],
                    },
                )
            )

        elif type(self.model) in (HiBRCfine, HiBRCrefine):
            return {
                k: alt.Chart(df_from_dict(v))
                .mark_rect()
                .encode(
                    x=alt.X(
                        "x:O",
                        axis=alt.Axis(
                            values=tick_values,
                            grid=False,
                            **self.default_config["x_axis"],
                        ),
                        title="Age of contacting individuals",
                    ),
                    y=alt.Y(
                        "y:O",
                        scale=alt.Scale(reverse=True),
                        axis=alt.Axis(
                            values=tick_values,
                            grid=False,
                            **self.default_config["y_axis"],
                        ),
                        title="Age of contacted individuals",
                    ),
                    color=alt.Color(
                        "z:Q",
                        scale=alt.Scale(scheme="spectral", reverse=True),
                        title="intensity",
                        legend=alt.Legend(**self.default_config["legend"]),
                    ),
                    facet=alt.Facet("label:N", title=None, columns=facet_columns),
                )
                .properties(
                    width=width,
                    height=height,
                    title={
                        "text": [f"Estimated contact intensity: {k}"],
                        **self.default_config["title"],
                    },
                )
                for k, v in sum_cint.items()
            }

    def plot_mcint(
        self,
        evaluator: ModelEvaluatorBRC | None = None,
        width=250,
        height=250,
        style_config=None,
    ):
        """Plot marginal contact intensity by age with credible intervals.

        Generates line plots showing the age-specific marginal contact intensity
        (total contacts per age group) with 95% credible intervals. Optionally
        overlays ground truth values if an evaluator is provided.

        Parameters
        ----------
        evaluator : ModelEvaluatorBRC or None, optional
            Model evaluator containing ground truth contact patterns for
            comparison. If provided, true values are overlaid in red,
            by default None.
        width : int, optional
            Width of each plot in pixels, by default 250.
        height : int, optional
            Height of each plot in pixels, by default 250.
        style_config : dict, optional
            Custom style configuration to override defaults. Should contain
            keys like 'x_axis', 'y_axis', 'title', 'legend' with nested style
            parameters, by default None.

        Returns
        -------
        altair.Chart or dict of altair.Chart
            For BRCfine/BRCrefine: Single Altair chart with line + error band.
            For HiBRCfine/HiBRCrefine: Dictionary mapping stratification
            levels to faceted chart objects.

        Examples
        --------
        >>> # Plot without ground truth
        >>> chart = visualiser.plot_mcint(width=500, height=300)
        >>> chart.save('marginal_intensity.html')

        >>> # Plot with ground truth comparison
        >>> from cntmosaic.analysis import ModelEvaluatorBRC
        >>> evaluator = ModelEvaluatorBRC(model, true_data)
        >>> chart = visualiser.plot_mcint(evaluator=evaluator)
        >>> chart.save('marginal_intensity_comparison.html')

        Notes
        -----
        Marginal contact intensity represents the total expected number of
        contacts for individuals of each age, summed across all contact ages.
        This provides a summary view of age-specific contact levels.

        The error band represents the 95% credible interval from the posterior
        distribution, quantifying uncertainty in the estimates.
        """

        if style_config:
            for key, conf in style_config.items():
                self.default_config.setdefault(key, {}).update(conf)

        sum_mcint = self.summariser.summarise_mcint()

        x_axis = alt.Axis(
            title="Age of contacting individuals",
            grid=True,
            **self.default_config["x_axis"],
        )
        y_axis = alt.Axis(
            title="Contact intensity", grid=True, **self.default_config["y_axis"]
        )

        if type(self.summariser.model) in (BRCfine, BRCrefine):
            # For BRC models, we assume mcint is a single matrix
            source = df_from_dict(sum_mcint)
            if evaluator is not None:
                mcint_true = evaluator.mcint_true
                # Create a dataframe from all arrays in mcint_true
                df_mcint_true = pd.DataFrame(
                    {"x": np.arange(mcint_true.size), "y_true": mcint_true}
                )
                # Merge with source dataframe on the common keys
                source = pd.merge(source, df_mcint_true, on="x", how="left")
                line = (
                    alt.Chart(source)
                    .mark_line(color="red")
                    .encode(
                        x=alt.X("x:Q", axis=x_axis),
                        y=alt.Y("y_true:Q", title="Contact intensity"),
                    )
                )

            base = (
                alt.Chart(source)
                .mark_line()
                .encode(x=alt.X("x:Q", axis=x_axis), y=alt.Y("y:Q", axis=y_axis))
            )

            band = (
                alt.Chart(source)
                .mark_errorband()
                .encode(
                    x=alt.X("x:Q", axis=x_axis),
                    y=alt.Y("l:Q", title="Contact intensity"),
                    y2="u:Q",
                )
            )

            chart = base + band + line if evaluator is not None else base + band
            return chart.properties(width=width, height=height)

        elif type(self.summariser.model) in (HiBRCfine, HiBRCrefine):
            charts = {}
            for key, val in sum_mcint.items():
                source = df_from_dict(val)

                if evaluator is not None:
                    mcint_true = evaluator.mcint_true[key]
                    # Create a dataframe from all arrays in mcint_true
                    df_mcint_true = pd.concat(
                        [
                            pd.DataFrame(
                                {"label": k, "x": np.arange(v.size), "y_true": v}
                            )
                            for k, v in mcint_true.items()
                        ]
                    )
                    # Merge with source dataframe on the common keys
                    source = pd.merge(
                        source, df_mcint_true, on=["label", "x"], how="left"
                    )
                    line = (
                        alt.Chart(source)
                        .mark_line(color="red")
                        .encode(
                            x=alt.X("x:Q", axis=x_axis),
                            y=alt.Y("y_true:Q", title="Contact intensity"),
                        )
                    )

                base = (
                    alt.Chart(source)
                    .mark_line()
                    .encode(x=alt.X("x:Q", axis=x_axis), y=alt.Y("y:Q", axis=y_axis))
                )

                band = (
                    alt.Chart(source)
                    .mark_errorband()
                    .encode(
                        x=alt.X("x:Q", axis=x_axis),
                        y=alt.Y("l:Q", title="Contact intensity"),
                        y2="u:Q",
                    )
                )

                chart = base + band + line if evaluator is not None else base + band
                charts[key] = chart.properties(width=width, height=height).facet(
                    column=alt.Column("label:N", title=None),
                    columns=3,
                    title=alt.TitleParams(
                        text=[f"Estimated contact intensity: {key}"],
                        **self.default_config["title"],
                    ),
                )

            return charts
