"""Loading morphology data."""

from pathlib import Path

import trimesh


def load_mesh(path: Path, scale: float = 1.0) -> trimesh.Trimesh:
    """Load a surface mesh.

    Parameters
    ----------
    path : pathlib.Path
        Any format trimesh reads (``.obj``, ``.ply``, ``.stl``, ``.off``, ...).
    scale : float
        Factor that converts file units to µm (e.g. 0.005 for 5 nm pixels).

    Returns
    -------
    trimesh.Trimesh
        Vertices in µm. Multi-part files are concatenated.

    Examples
    --------
    >>> mesh = load_mesh(Path("spine.obj"), scale=1.0)  # doctest: +SKIP
    """
    loaded = trimesh.load(Path(path), force="mesh", process=True)
    if scale != 1.0:
        loaded.apply_scale(scale)
    return loaded
