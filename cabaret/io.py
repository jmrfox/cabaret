"""Loading morphology data."""

from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray


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
    >>> mesh = load_mesh(Path("data/TS1.obj"), scale=0.005)  # doctest: +SKIP
    """
    loaded = trimesh.load(Path(path), force="mesh", process=True)
    if scale != 1.0:
        loaded.apply_scale(scale)
    return loaded


def load_points(path: Path, scale: float = 1.0) -> NDArray[np.float64]:
    """Load whitespace-separated ``x y z`` points (AZ or neck-point files).

    Parameters
    ----------
    path : pathlib.Path
    scale : float
        Factor that converts file units to µm (0.005 for the TS data).

    Returns
    -------
    (K, 3) ndarray of float, in µm
    """
    return np.loadtxt(Path(path), ndmin=2) * scale


def load_axon_groups(path: Path) -> list[list[int]]:
    """Load synapse-to-axon groupings, one axon per line.

    Each line lists 1-based synapse indices (row numbers in the matching AZ
    file), separated by commas, e.g. ``TS1_syn_axon_dist.txt``.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    list of list of int
        0-based synapse indices per axon.
    """
    groups = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            groups.append([int(tok) - 1 for tok in line.split(",")])
    return groups
