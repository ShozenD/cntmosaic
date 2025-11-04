"""Utilities for managing tutorial notebooks and figures."""

from pathlib import Path
from typing import Optional, Union, Any


# Directory for storing tutorial figures
FIGURE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "tutorials" / "figures"
)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_tutorial_figure(
    chart: Any, name: str, format: str = "png", scale_factor: float = 2.0, **kwargs
) -> Path:
    """
    Save an Altair chart to the tutorial figures directory.

    This function provides a consistent way to save figures generated in tutorial
    notebooks. The figures are saved to `docs/tutorials/figures/` and are version
    controlled (not stripped by nbstripout) to allow displaying outputs in
    documentation without committing heavy notebook files.

    Parameters
    ----------
    chart : altair.Chart or altair.LayerChart or altair.ConcatChart
        The Altair chart to save. Can be any Altair chart object.
    name : str
        Base name for the figure file (without extension).
    format : str, default='png'
        File format. Supported formats: 'png', 'svg', 'pdf', 'html'.
        - 'png': Raster image, good for embedding (requires altair_saver or selenium)
        - 'svg': Vector format, scalable, good for web and editing
        - 'pdf': Vector format, good for publications
        - 'html': Interactive HTML file with Vega-Lite
    scale_factor : float, default=2.0
        Resolution scaling for PNG output. Higher values produce larger, clearer images.
        - 1.0: Standard resolution
        - 2.0: High resolution (recommended, good for retina displays)
        - 3.0: Very high resolution
        Only applies to PNG format.
    **kwargs
        Additional keyword arguments passed to `chart.save()`.
        For PNG/SVG/PDF: Can include 'embed_options', 'vega_version', etc.
        For HTML: Can include 'embed_options', 'inline', etc.

    Returns
    -------
    Path
        Path to the saved figure file.

    Examples
    --------
    >>> import altair as alt
    >>> import pandas as pd
    >>> from cntmosaic.utils import save_tutorial_figure
    >>>
    >>> # Create a chart
    >>> data = pd.DataFrame({
    ...     'x': [1, 2, 3, 4, 5],
    ...     'y': [1, 4, 9, 16, 25]
    ... })
    >>> chart = alt.Chart(data).mark_line().encode(
    ...     x='x',
    ...     y='y'
    ... )
    >>>
    >>> # Save for tutorial documentation (PNG)
    >>> save_tutorial_figure(chart, "example_plot")
    >>> chart  # Display in notebook
    >>>
    >>> # Save as SVG for scalable vector graphics
    >>> save_tutorial_figure(chart, "example_plot_vector", format="svg")
    >>>
    >>> # Save as interactive HTML
    >>> save_tutorial_figure(chart, "example_plot_interactive", format="html")
    >>>
    >>> # Save with custom scale factor
    >>> save_tutorial_figure(
    ...     chart,
    ...     "example_plot_hires",
    ...     scale_factor=3.0
    ... )

    Notes
    -----
    - Figures are saved to `docs/tutorials/figures/`
    - The directory is created automatically if it doesn't exist
    - These figures are version controlled (allowed in .gitignore)
    - Use this instead of committing notebook cell outputs
    - For PNG output, you may need altair_saver or selenium/chromedriver installed
    - SVG and HTML formats work without additional dependencies

    See Also
    --------
    altair.Chart.save : Underlying save function
    """
    # Ensure name doesn't have extension
    name = Path(name).stem

    # Construct full path
    filepath = FIGURE_DIR / f"{name}.{format}"

    # Save chart based on format
    if format.lower() in ["png", "svg", "pdf"]:
        # For static formats, use scale_factor if PNG
        if format.lower() == "png":
            chart.save(str(filepath), scale_factor=scale_factor, **kwargs)
        else:
            chart.save(str(filepath), **kwargs)
    elif format.lower() == "html":
        # For HTML, save interactive version
        chart.save(str(filepath), **kwargs)
    else:
        raise ValueError(
            f"Unsupported format: {format}. "
            "Supported formats: 'png', 'svg', 'pdf', 'html'"
        )

    return filepath


def list_tutorial_figures() -> list[Path]:
    """
    List all saved tutorial figures.

    Returns
    -------
    list of Path
        List of paths to all figure files in the tutorial figures directory.

    Examples
    --------
    >>> from cntmosaic.utils import list_tutorial_figures
    >>> figures = list_tutorial_figures()
    >>> for fig in figures:
    ...     print(fig.name)
    """
    if not FIGURE_DIR.exists():
        return []

    # Get all image and HTML files
    patterns = ["*.png", "*.pdf", "*.svg", "*.html", "*.jpg", "*.jpeg"]
    figures = []
    for pattern in patterns:
        figures.extend(FIGURE_DIR.glob(pattern))

    return sorted(figures)


def convert_notebook_to_html(
    notebook_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    execute: bool = False,
) -> Path:
    """
    Convert a Jupyter notebook to HTML for documentation.

    Requires nbconvert to be installed: `pip install nbconvert`

    Parameters
    ----------
    notebook_path : str or Path
        Path to the notebook file (.ipynb).
    output_dir : str or Path, optional
        Directory to save the HTML file. If None, saves to
        `docs/tutorials/` by default.
    execute : bool, default=False
        Whether to execute the notebook before converting. If False,
        uses the outputs already stored in the notebook.

    Returns
    -------
    Path
        Path to the generated HTML file.

    Examples
    --------
    >>> from cntmosaic.utils import convert_notebook_to_html
    >>>
    >>> # Convert notebook with existing outputs
    >>> html_path = convert_notebook_to_html("tutorials/tutorial1.ipynb")
    >>>
    >>> # Execute and convert
    >>> html_path = convert_notebook_to_html(
    ...     "tutorials/tutorial2.ipynb",
    ...     execute=True
    ... )

    Notes
    -----
    - Requires `nbconvert` package
    - If execute=True, the notebook must run without errors
    - HTML files include all outputs (plots, tables, etc.)

    Raises
    ------
    ImportError
        If nbconvert is not installed.
    FileNotFoundError
        If notebook_path does not exist.
    """
    try:
        import nbconvert
        from nbconvert import HTMLExporter
    except ImportError:
        raise ImportError(
            "nbconvert is required for this function. "
            "Install it with: pip install nbconvert"
        )

    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Set output directory
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent / "docs" / "tutorials"
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert notebook
    exporter = HTMLExporter()
    if execute:
        from nbconvert.preprocessors import ExecutePreprocessor

        exporter.register_preprocessor(ExecutePreprocessor, enabled=True)

    (body, resources) = exporter.from_filename(notebook_path)

    # Save HTML
    output_path = output_dir / f"{notebook_path.stem}.html"
    output_path.write_text(body)

    return output_path
