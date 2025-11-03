# Tutorial Notebooks

This directory contains Jupyter notebooks that demonstrate how to use the `cntmosaic` package for contact matrix estimation and analysis.

## 📚 Available Tutorials

- **Tutorial_Prem.ipynb** - Using the Prem model for contact matrix estimation
- **Tutorial_SocialMix.ipynb** - Social mixing patterns analysis
- **Tutorial_Generate_Contact_Data.ipynb** - Generating synthetic contact data

## 🔧 Setup

### First-time Setup

1. **Install dependencies** (including `nbstripout`):
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure git to strip notebook outputs** (one-time setup):
   ```bash
   nbstripout --install --attributes .gitattributes
   ```
   
   This ensures that notebook outputs (which can be very large) are automatically removed before committing to git.

### Running Tutorials

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

2. **Navigate to** `tutorials/` and open any notebook

3. **Run the cells** - outputs will display in your local notebook

## 💾 Saving Figures for Documentation

When you want to display plots in documentation (without committing heavy notebook outputs), use the provided utility function:

```python
from cntmosaic.utils import save_tutorial_figure
import altair as alt
import pandas as pd

# Create your chart
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [1, 4, 9, 16, 25]
})

chart = alt.Chart(data).mark_line().encode(
    x='x',
    y='y'
).properties(
    width=400,
    height=300,
    title='Example Plot'
)

# Save for documentation (this WILL be committed to git)
save_tutorial_figure(chart, "my_plot")

# Display in notebook
chart
```

### Format Options

```python
# Save as PNG (default, good for embedding)
save_tutorial_figure(chart, "my_plot", format="png", scale_factor=2.0)

# Save as SVG (vector graphics, scalable)
save_tutorial_figure(chart, "my_plot_vector", format="svg")

# Save as interactive HTML
save_tutorial_figure(chart, "my_plot_interactive", format="html")
```

Figures are saved to `docs/tutorials/figures/` and are version controlled, allowing outputs to be displayed in documentation without committing the full notebook outputs.

## 🔄 Git Workflow

### What Gets Committed

✅ **Committed to git:**
- Notebook files (`.ipynb`) with code and markdown cells
- Saved figures in `docs/tutorials/figures/`

❌ **NOT committed (automatically stripped):**
- Cell outputs (plots, tables, print statements)
- Execution counts
- Widget state

### How It Works

The `nbstripout` tool is configured via `.gitattributes` to automatically strip outputs when you commit:

```bash
# Normal git workflow - outputs stripped automatically
git add tutorials/Tutorial_Example.ipynb
git commit -m "Add example tutorial"
```

Your local notebook keeps all outputs for viewing, but git only stores the clean version.

### Checking Stripped Output

To see what will be committed (without outputs):

```bash
git show HEAD:tutorials/Tutorial_Example.ipynb | jupyter nbconvert --stdin --to notebook --stdout
```

## 📊 Converting Notebooks to HTML

To share tutorials with outputs as HTML documentation:

```python
from cntmosaic.utils import convert_notebook_to_html

# Convert with existing outputs
html_path = convert_notebook_to_html("tutorials/Tutorial_Prem.ipynb")

# Or execute and convert
html_path = convert_notebook_to_html(
    "tutorials/Tutorial_Prem.ipynb",
    execute=True
)
```

Requires `nbconvert`:
```bash
pip install nbconvert
```

## 🤝 Contributing Tutorials

When adding new tutorials:

1. **Create your notebook** with rich explanations and examples
2. **Save key figures** using `save_tutorial_figure()` for documentation
3. **Commit everything** - `nbstripout` will handle the stripping
4. **No need to** manually clear outputs or worry about file sizes

## 🐛 Troubleshooting

### Outputs still being committed?

Check that `nbstripout` is properly installed:

```bash
nbstripout --status
```

Should show:
```
nbstripout is installed in repository
```

If not, run:
```bash
nbstripout --install --attributes .gitattributes
```

### Want to see stripped version locally?

```bash
nbstripout tutorials/Tutorial_Example.ipynb
```

(This modifies the file - you may want to restore it with `git checkout`)

### Figures not showing up in git?

Check that your figures are in the right location:
```bash
ls docs/tutorials/figures/
```

And that `.gitignore` has the exception:
```
!docs/tutorials/figures/*.png
!docs/tutorials/figures/*.pdf
!docs/tutorials/figures/*.svg
!docs/tutorials/figures/*.html
```

### PNG export not working?

For PNG export of Altair charts, you may need additional dependencies:

```bash
# Option 1: Use altair_saver with selenium
pip install altair_saver selenium
# Download chromedriver from https://chromedriver.chromium.org/

# Option 2: Use vl-convert (recommended, no browser needed)
pip install vl-convert-python
```

SVG and HTML formats work without additional dependencies.

## 📖 Additional Resources

- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [nbstripout Documentation](https://github.com/kynan/nbstripout)
- [Altair Documentation](https://altair-viz.github.io/)
- [cntmosaic Documentation](../docs/)

## 💡 Tips

- **Prefer SVG format**: For Altair charts, SVG is often the best choice - it's scalable, loads fast, and doesn't require extra dependencies
- **Interactive HTML**: Save interactive charts as HTML for exploratory documentation
- **Clear outputs before debugging**: If you're having git issues, manually clear outputs with `Cell -> All Output -> Clear` in Jupyter
- **Large datasets**: Don't load large datasets in tutorials - use sample data or provide download links
- **Execution time**: Keep tutorials reasonably fast (<5 minutes per notebook)
- **Dependencies**: If tutorials need extra packages, add them to `requirements.txt`
