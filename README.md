# camovox

Cable morphologies from voxels

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # create .venv and install camovox in editable mode
uv run pytest            # run tests
uv run ruff format .     # format
uv run ruff check .      # lint
```

Notebooks live in `notebooks/` as jupytext pairs (`.py` + `.ipynb`). Edit the `.py` file, then:

```bash
uv run jupytext --sync notebooks/*.py
```

Project docs: [design](docs/design.md) · [plans](docs/plans/) · [decisions](docs/decisions.md).
Instructions for AI agents: [AGENTS.md](AGENTS.md).
