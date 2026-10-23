# Agent instructions for camovox

Cable morphologies from voxels

This file is the single source of rules for every AI agent working in this repo
(Cursor, Claude, Codex, ChatGPT, etc.). The `CLAUDE.md` files only import it.
Folder-specific rules live in `tests/AGENTS.md` and `notebooks/AGENTS.md`.

## Workflow

Work moves through files in `docs/`, so any agent can pick up any stage:

1. **Design**: `docs/design/` is the high-level design, split into chapters. Start at
   `docs/design/README.md` (goals, terminology, principles, scope, pipeline and a chapter index).
2. **Plan**: each feature gets a plan in `docs/plans/NNN-short-name.md`, copied from
   `docs/plans/_template.md`. A plan is a checklist of small, verifiable steps.
3. **Implement**: read `docs/design/README.md`, the chapters the plan links, and the active plan first. Tick off steps as
   you finish them. If the work forces a design change, update the design or plan
   and record why in `docs/decisions.md`. Don't diverge from the design without recording it.
4. **Document**: write docstrings, notebooks and the README.

If you're asked to implement something with no plan and it is more than a small
change, propose a short plan first.

## Environment

- Use `uv` for everything. Never call `pip`, `python` or `pytest` directly.
  - Run code: `uv run python ...`, `uv run pytest`, `uv run jupytext ...`
  - Add a dependency: `uv add <pkg>`. For dev-only: `uv add --dev <pkg>`
- `uv sync` installs `camovox` in editable mode (hatchling), so
  `import camovox` always reflects the current source.
- Never edit `.venv/` or `uv.lock` by hand.

## Layout

```
camovox/   package source
tests/          pytest tests (kept small, see tests/AGENTS.md)
notebooks/      jupytext-paired notebooks (see notebooks/AGENTS.md)
data/           local data, gitignored except data/README.md
docs/           design/, plans/, decisions.md
```

## Reuse before you write

This project often tackles new problems, so it's tempting to hand-roll everything.
Before implementing any nontrivial algorithm or data structure, check whether a
well-maintained package already does it, and prefer that package. Examples:

- graphs/networks: `networkx` (or `igraph`/`rustworkx` if performance matters)
- numerics, linear algebra, optimization, interpolation, sparse, spatial: `numpy`, `scipy`
- tabular data: `pandas` (or `polars`)
- images/voxels/morphology: `scikit-image`, `scipy.ndimage`
- meshes/geometry: `trimesh`, `shapely`
- ML/stats: `scikit-learn`, `statsmodels`
- symbolic math: `sympy`

When you find a candidate, say so before writing custom code: which package, what it
covers, and what it doesn't. When adding a dependency, note why in `docs/decisions.md`.
Write from scratch only when no package fits, and say why. Keep custom code thin
around library calls.

## Code style

- Type hints on all function signatures.
- NumPy-style docstrings on every public module, class and function: `Parameters`,
  `Returns`, `Raises` where relevant, and an `Examples` section for main entry points.
  State array shapes and physical units, e.g. `points : (N, 3) ndarray of float, in µm`.
- Format and lint with ruff: `uv run ruff format .` and `uv run ruff check --fix .`.
- Prefer small, pure functions that take and return arrays. Avoid hidden global
  state. Pass randomness in as `rng: np.random.Generator` and never seed globally.
- Vectorize with numpy where it's natural. Don't contort code for speed until
  profiling shows it matters.
- No hard-coded absolute paths. Take paths as arguments (`pathlib.Path`).
- Private helpers start with `_`. Expose the public API in `camovox/__init__.py`.

## Testing

Testing is deliberately light during development. You decide when a test is
worth adding, guided by `tests/AGENTS.md`. Run tests with `uv run pytest`.

## Before you say a task is done

1. `uv run ruff format .` and `uv run ruff check .` are clean.
2. `uv run pytest` passes.
3. Any edited notebook `.py` files are synced: `uv run jupytext --sync notebooks/<name>.py`.
4. The active plan's checklist is updated.
