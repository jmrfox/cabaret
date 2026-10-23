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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from camovox.tet import TetMesh

if TYPE_CHECKING:
    from camovox.circuit import CompartmentGraph
    from camovox.sites import MembraneSurface, Sites
    from camovox.tet import FidelityReport


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


@dataclass
class TSSpine:
    """A TS spine prepared for analysis (see :func:`load_spine`).

    Parameters
    ----------
    name : str
        Spine name, e.g. ``"TS1"``.
    mesh : trimesh.Trimesh
        TS surface, in µm.
    tets : TetMesh
        Its tetrahedral mesh.
    fidelity : FidelityReport
        Area and volume change from meshing.
    model : CompartmentGraph
        FEM network of the TS mesh, without neck ports.
    spine : CompartmentGraph
        ``model`` with one electrode per neck (membrane removed there).
    neck_nodes : list of int
        Electrode node of each neck in ``spine``.
    neck_ports : list of (N,) ndarray of float
        Port weights of each neck (one-hot on its electrode).
    contacts : list of ndarray of int
        FEM nodes tied into each neck electrode.
    own_parts : ndarray of int
        Cell parts that are the cell's own tracing of this spine.
    membrane : MembraneSurface
        Real membrane of ``spine``, for placing patches.
    az : Sites or None
        Active-zone sites, if the spine has an AZ file.
    """

    name: str
    mesh: trimesh.Trimesh
    tets: TetMesh
    fidelity: "FidelityReport"
    model: "CompartmentGraph"
    spine: "CompartmentGraph"
    neck_nodes: list[int]
    neck_ports: list[NDArray[np.float64]]
    contacts: list[NDArray[np.int64]]
    own_parts: NDArray[np.int64]
    membrane: "MembraneSurface"
    az: "Sites | None"


def load_spine(
    name: str,
    data_dir: Path,
    parts: Sequence[trimesh.Trimesh],
    scale: float = 0.005,
    edge_length: float = 0.2,
    envelope: float = 0.005,
    psd_area: float = 0.05,
    contact_distance: float = 0.2,
    contact_radius: float = 3.0,
) -> TSSpine:
    """Load a TS spine with its FEM model, neck port(s) and active zones.

    These are the steps of notebook 02 in one call:
    1. load and tetrahedralize ``<name>.obj`` (cached in ``data_dir/cache``);
    2. build the FEM model;
    3. find the neck contacts with the cell parts, and tie each neck into
       an electrode;
    4. snap the AZs (``<name>_AZ.txt``, if present) onto the remaining
       membrane, with the axon grouping if ``<name>_syn_axon_dist.txt``
       exists.

    Parameters
    ----------
    name : str
        Spine name, e.g. ``"TS1"``.
    data_dir : pathlib.Path
        Folder with the TS files (``../data`` from the notebooks).
    parts : sequence of trimesh.Trimesh
        Cell parts in µm (see :func:`load_parts`).
    scale : float
        µm per pixel of the data files.
    edge_length, envelope : float, in µm
        Meshing settings (see :func:`camovox.tetrahedralize`).
    psd_area : float, in µm²
        Placeholder AZ patch area.
    contact_distance, contact_radius : float, in µm
        See :func:`neck_contacts`.

    Returns
    -------
    TSSpine

    Raises
    ------
    ValueError
        If a neck has no contact with the cell.
    """
    from camovox.circuit import add_electrode, point_port
    from camovox.fem import fem_model
    from camovox.io import load_axon_groups, load_mesh, load_points
    from camovox.sites import MembraneSurface, sites_from_points
    from camovox.tet import tetrahedralize_cached

    data_dir = Path(data_dir)
    mesh = load_mesh(data_dir / f"{name}.obj", scale=scale)
    cache = data_dir / "cache" / f"{name}_e{edge_length:g}_env{envelope:g}.npz"
    tets, fidelity = tetrahedralize_cached(
        mesh, cache, edge_length=edge_length, envelope=envelope
    )
    model, _ = fem_model(tets)
    own = spine_parts(mesh, parts)
    necks = load_points(data_dir / f"{name}_neckpoint.txt", scale)
    contacts = neck_contacts(tets, parts, own, necks, contact_distance, contact_radius)
    spine, neck_nodes = model, []
    for k, c in enumerate(contacts):
        if c.size == 0:
            raise ValueError(f"{name}: neck {k} has no contact with the cell")
        spine, node = add_electrode(spine, c)
        neck_nodes.append(node)
    neck_ports = [point_port(spine, n) for n in neck_nodes]
    membrane = MembraneSurface.from_model(tets, spine)
    az = None
    az_file = data_dir / f"{name}_AZ.txt"
    if az_file.exists():
        axon_file = data_dir / f"{name}_syn_axon_dist.txt"
        groups = load_axon_groups(axon_file) if axon_file.exists() else None
        az = sites_from_points(
            membrane,
            load_points(az_file, scale),
            "synapse",
            area=psd_area,
            axon_groups=groups,
        )
    return TSSpine(
        name,
        mesh,
        tets,
        fidelity,
        model,
        spine,
        neck_nodes,
        neck_ports,
        contacts,
        own,
        membrane,
        az,
    )
