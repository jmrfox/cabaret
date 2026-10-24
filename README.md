# cabaret

**CAB**le **A**pproximation by **RE**sistor **T**rees: reduced cable models of
neurons from 3D reconstructions.
Formerly `camovox`. Distribution name: `cabaret-neuro` (import as `cabaret`).

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # create .venv and install cabaret in editable mode
uv run pytest            # run tests
uv run ruff format .     # format
uv run ruff check .      # lint
```

Notebooks live in `notebooks/` as jupytext pairs (`.py` + `.ipynb`). Edit the `.py` file, then:

```bash
uv run jupytext --sync notebooks/*.py
```

Project docs: [design](docs/design/README.md) · [plans](docs/plans/) · [decisions](docs/decisions.md).
Instructions for AI agents: [AGENTS.md](AGENTS.md).
