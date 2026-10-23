"""Relating separately traced spines to the whole-cell mesh.

The TS spine meshes were traced separately from the whole cell, so each
spine exists twice: as a detailed TS mesh, and as a coarser part of the
segmented cell (neuromesh-cgal parts). This module finds, for a TS spine:

- which cell parts are the cell's own version of the spine (the **spine
  parts**). These are parts mostly covered by the TS mesh;
- where the TS spine meets the rest of the cell (the **parent**). This is the
  TS volume that lies inside the parent, or within a small tolerance of its
  surface. Those FEM nodes form the spine's **neck port**: tied into one
  equipotential electrode, with their membrane removed, because that region
  is really parent cytoplasm.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from camovox.tet import TetMesh


def load_parts(directory: Path, scale: float = 1.0) -> list[trimesh.Trimesh]:
    """Load neuromesh-cgal partition parts (``segment_NNN.obj``) in order.

    Parameters
    ----------
    directory : pathlib.Path
        Folder written by ``mesh_partition``.
    scale : float
        Factor converting file units to µm (0.005 for the TS data).

    Returns
    -------
    list of trimesh.Trimesh
        ``parts[i]`` is segment ``i``. Vertices in µm.
    """
    files = sorted(Path(directory).glob("segment_*.obj"))
    parts = []
    for f in files:
        m = trimesh.load(f, force="mesh", process=False)
        if scale != 1.0:
            m.apply_scale(scale)
        parts.append(m)
    return parts


def spine_parts(
    spine: trimesh.Trimesh,
    parts: Sequence[trimesh.Trimesh],
    tolerance: float = 0.3,
    min_coverage: float = 0.5,
) -> NDArray[np.int64]:
    """Cell parts that are the cell's own tracing of a spine.

    A part counts if at least ``min_coverage`` of its vertices lie within
    ``tolerance`` of the spine mesh. A large parent dendrite or soma touched
    by the spine is only covered near the contact, so it fails the test.

    Parameters
    ----------
    spine : trimesh.Trimesh
        Separately traced spine surface, in µm.
    parts : sequence of trimesh.Trimesh
        Cell parts (see :func:`load_parts`), in µm.
    tolerance : float, in µm
    min_coverage : float

    Returns
    -------
    (K,) ndarray of int
        Part indices.
    """
    tree = cKDTree(spine.vertices)
    lo, hi = spine.bounds
    out = []
    for i, p in enumerate(parts):
        plo, phi = p.bounds
        if np.any(plo > hi + tolerance) or np.any(phi < lo - tolerance):
            continue
        d, _ = tree.query(p.vertices, distance_upper_bound=tolerance)
        if np.mean(np.isfinite(d)) >= min_coverage:
            out.append(i)
    return np.array(out, dtype=np.int64)


def neck_contacts(
    tets: TetMesh,
    parts: Sequence[trimesh.Trimesh],
    exclude: ArrayLike,
    neck_points: ArrayLike,
    distance: float = 0.2,
    radius: float = 3.0,
) -> list[NDArray[np.int64]]:
    """FEM nodes of a spine that lie in or against its parent, per neck.

    A node is a contact if it is inside a parent part, or within
    ``distance`` of a parent-part vertex. Only nodes within ``radius`` of a
    neck point are considered, so contacts elsewhere (a spine lying against
    another process) are ignored. Each contact is assigned to its nearest
    neck point.

    Parameters
    ----------
    tets : TetMesh
        Tetrahedral mesh of the TS spine, in µm.
    parts : sequence of trimesh.Trimesh
        Cell parts, in µm.
    exclude : (K,) array_like of int
        Parts that are the spine itself (see :func:`spine_parts`). All
        others are parent.
    neck_points : (M, 3) array_like of float, in µm
    distance : float, in µm
        Surface tolerance for the mismatch between the two tracings.
    radius : float, in µm
        Search radius around each neck point.

    Returns
    -------
    list of M (C,) ndarray of int
        Node indices for each neck. A neck with no contact gets an empty
        array.
    """
    necks = np.atleast_2d(np.asarray(neck_points, dtype=float))
    d_neck = np.linalg.norm(tets.points[:, None, :] - necks[None], axis=2)
    cand = np.flatnonzero(d_neck.min(axis=1) < radius)
    pts = tets.points[cand]
    excl = set(np.asarray(exclude, dtype=int).tolist())
    lo, hi = pts.min(axis=0) - radius, pts.max(axis=0) + radius
    contact = np.zeros(len(cand), dtype=bool)
    for i, p in enumerate(parts):
        if i in excl:
            continue
        plo, phi = p.bounds
        if np.any(plo > hi) or np.any(phi < lo):
            continue
        d, _ = cKDTree(p.vertices).query(pts)
        contact |= d < distance
        close = np.flatnonzero(d < 1.0)
        if p.is_watertight and close.size:
            # Chunked: trimesh's ray test allocates per point x candidate
            # triangle and can exhaust memory on large meshes.
            for start in range(0, close.size, 2000):
                chunk = close[start : start + 2000]
                contact[chunk] |= p.contains(pts[chunk])
    nodes = cand[contact]
    owner = d_neck[nodes].argmin(axis=1)
    return [nodes[owner == k] for k in range(len(necks))]
