# Notebook rules

Each notebook is a jupytext pair: `name.py` (percent format) plus `name.ipynb`.
The pairing is configured in `pyproject.toml` (`[tool.jupytext]`).

## Editing

- **The `.py` file is the source of truth.** Edit only the `.py`, never the `.ipynb` JSON.
- After editing, sync the pair: `uv run jupytext --sync notebooks/<name>.py`
- Code cells start with `# %%` and markdown cells with `# %% [markdown]`.
  Each markdown line is commented with `# `.

## Creating a notebook

Name it with a numeric prefix, e.g. `notebooks/01_explore_data.py`:

```python
# %% [markdown]
# # Title
# One or two sentences on what this notebook shows.

# %%
import numpy as np

import <package>
```

Then create the `.ipynb` with `uv run jupytext --sync notebooks/01_explore_data.py`.

## Content

- Notebooks demonstrate, explore and visualize. They **use** the package and don't
  extend it. Anything reusable (more than a few lines, or used twice) goes into the
  package and is imported from there.
- They must run top to bottom in a fresh kernel, with no reliance on cell execution order.
- Put imports in the first code cell. Load data with relative paths such as `../data/...`.
- Keep markdown short: say what each section shows and what to notice.
