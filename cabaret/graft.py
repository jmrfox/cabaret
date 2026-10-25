"""Relating a separately traced region to a segmented whole-cell mesh.

A region of interest (a spine, a stretch of dendrite) is often traced in
more detail than the whole cell, so it exists twice: as its own detailed
surface, and as a coarser part of the segmented cell (e.g. neuromesh-cgal
parts). This module finds, for such a region:

- which cell parts are the cell's own version of the region (the **region
  parts**). These are parts mostly covered by the region's surface;
- where the region meets the rest of the cell (the **parent**). This is the
  region's volume that lies inside the parent, or within a small tolerance
  of its surface. Those FEM nodes form an **attachment**: tied into one
  equipotential electrode, with their membrane removed, because that volume
  is really parent cytoplasm (see :func:`cabaret.build_reference`).
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from cabaret.tet import TetMesh


def load_parts(directory: Path, scale: float = 1.0) -> list[trimesh.Trimesh]:
    """Load neuromesh-cgal partition parts (``segment_NNN.obj``) in order.

    Parameters
    ----------
    directory : pathlib.Path
        Folder written by ``mesh_partition``.
    scale : float
        Factor converting file units to µm.

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


def region_parts(
    region: trimesh.Trimesh,
    parts: Sequence[trimesh.Trimesh],
    tolerance: float = 0.3,
    min_coverage: float = 0.5,
) -> NDArray[np.int64]:
    """Cell parts that are the cell's own tracing of a region.

    A part counts if at least ``min_coverage`` of its vertices lie within
    ``tolerance`` of the region's surface. A large parent dendrite or soma
    touched by the region is only covered near the contact, so it fails the
    test.

    Parameters
    ----------
    region : trimesh.Trimesh
        Separately traced region surface, in µm.
    parts : sequence of trimesh.Trimesh
        Cell parts (see :func:`load_parts`), in µm.
    tolerance : float, in µm
    min_coverage : float

    Returns
    -------
    (K,) ndarray of int
        Part indices.
    """
    tree = cKDTree(region.vertices)
    lo, hi = region.bounds
    out = []
    for i, p in enumerate(parts):
        plo, phi = p.bounds
        if np.any(plo > hi + tolerance) or np.any(phi < lo - tolerance):
            continue
        d, _ = tree.query(p.vertices, distance_upper_bound=tolerance)
        if np.mean(np.isfinite(d)) >= min_coverage:
            out.append(i)
    return np.array(out, dtype=np.int64)


def attachment_contacts(
    tets: TetMesh,
    parts: Sequence[trimesh.Trimesh],
    exclude: ArrayLike,
    attachment_points: ArrayLike,
    distance: float = 0.2,
    radius: float = 3.0,
) -> list[NDArray[np.int64]]:
    """FEM nodes of a region that lie in or against its parent, per attachment.

    A node is a contact if it is inside a parent part, or within
    ``distance`` of a parent-part vertex. Only nodes within ``radius`` of an
    attachment point are considered, so contacts elsewhere (a region lying
    against another process) are ignored. Each contact is assigned to its
    nearest attachment point.

    Parameters
    ----------
    tets : TetMesh
        Tetrahedral mesh of the region, in µm.
    parts : sequence of trimesh.Trimesh
        Cell parts, in µm.
    exclude : (K,) array_like of int
        Parts that are the region itself (see :func:`region_parts`). All
        others are parent.
    attachment_points : (M, 3) array_like of float, in µm
        One point per attachment, near where the region joins the parent
        (e.g. a spine's neck point).
    distance : float, in µm
        Surface tolerance for the mismatch between the two tracings.
    radius : float, in µm
        Search radius around each attachment point.

    Returns
    -------
    list of M (C,) ndarray of int
        Node indices for each attachment. An attachment with no contact
        gets an empty array.
    """
    anchors = np.atleast_2d(np.asarray(attachment_points, dtype=float))
    d_anchor = np.linalg.norm(tets.points[:, None, :] - anchors[None], axis=2)
    cand = np.flatnonzero(d_anchor.min(axis=1) < radius)
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
    owner = d_anchor[nodes].argmin(axis=1)
    return [nodes[owner == k] for k in range(len(anchors))]
