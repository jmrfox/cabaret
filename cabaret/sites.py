"""Membrane sites: patches where current enters or is read out.

A site is a **patch** of membrane, never a point, because a point source in
3D has divergent input resistance. A patch is a geodesic disk on the
membrane, grown from the snapped site position until its membrane area
reaches the site's area (the PSD area).

PLACEHOLDERS:
- the default PSD area (0.05 µm²) is not data;
- `sample_uniform` enforces spacing with Euclidean, not geodesic, distance.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from cabaret.circuit import CompartmentGraph
from cabaret.tet import TetMesh

# µm². Literature-based guess: mammalian spine synapses average ~0.04 µm²
# (Schikorski & Stevens 1997); no owl data. See docs/design/sites-and-scenarios.md.
PLACEHOLDER_PSD_AREA = 0.05


def ball_patch(
    graph: CompartmentGraph, center: ArrayLike, radius: float
) -> NDArray[np.float64]:
    """Membrane weights of the nodes inside a ball.

    Parameters
    ----------
    graph : CompartmentGraph
    center : (3,) array_like of float, in µm
    radius : float, in µm

    Returns
    -------
    (N,) ndarray of float, in µm²
        Membrane area of each node inside the ball, zero elsewhere.

    Raises
    ------
    ValueError
        If the ball contains no membrane.
    """
    d = np.linalg.norm(graph.position - np.asarray(center, dtype=float), axis=1)
    w = np.where(d <= radius, graph.area, 0.0)
    if w.sum() <= 0:
        raise ValueError("ball contains no membrane nodes")
    return w


@dataclass
class MembraneSurface:
    """The membrane (boundary vertices) of a FEM model, for placing patches.

    Build it with :meth:`from_model`.

    Parameters
    ----------
    vertices : (S,) ndarray of int
        Model node index of each membrane vertex.
    area : (S,) ndarray of float, in µm²
        Lumped membrane area of each membrane vertex.
    position : (S, 3) ndarray of float, in µm
    graph : (S, S) scipy.sparse.csr_matrix
        Surface-edge graph weighted by edge length (µm), for geodesics.
    n_nodes : int
        Number of nodes in the model, for building full-length weights.
    """

    vertices: NDArray[np.int64]
    area: NDArray[np.float64]
    position: NDArray[np.float64]
    graph: sp.csr_matrix
    n_nodes: int
    _tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the KD-tree used by :meth:`snap`."""
        self._tree = cKDTree(self.position)

    @classmethod
    def from_model(cls, tets: TetMesh, ref: CompartmentGraph) -> "MembraneSurface":
        """Extract the membrane surface from a mesh and its FEM model.

        Parameters
        ----------
        tets : TetMesh
        ref : CompartmentGraph
            FEM reference model built from ``tets``.

        Returns
        -------
        MembraneSurface
        """
        faces = tets.boundary_faces()
        verts, local = np.unique(faces, return_inverse=True)
        local = local.reshape(faces.shape)
        e = np.concatenate([local[:, [0, 1]], local[:, [1, 2]], local[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        pos = tets.points[verts]
        length = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)
        g = sp.coo_matrix((length, (e[:, 0], e[:, 1])), shape=(len(verts),) * 2)
        return cls(
            vertices=verts.astype(np.int64),
            area=ref.area[verts],
            position=pos,
            graph=(g + g.T).tocsr(),
            n_nodes=ref.n_nodes,
        )

    def snap(self, points: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Nearest membrane vertex of each point.

        Parameters
        ----------
        points : (K, 3) array_like of float, in µm

        Returns
        -------
        (K,) ndarray of int
            Local (surface) vertex index.
        (K,) ndarray of float, in µm
            Distance from each point to its vertex.
        """
        d, idx = self._tree.query(np.atleast_2d(points))
        return idx.astype(np.int64), d

    def patch(self, local_vertex: int, area: float) -> NDArray[np.float64]:
        """Geodesic disk of membrane around a surface vertex.

        Vertices are added in order of geodesic distance until their summed
        membrane area reaches ``area``.

        Parameters
        ----------
        local_vertex : int
            Surface vertex index (from :meth:`snap`).
        area : float, in µm²
            Target patch area.

        Returns
        -------
        (N,) ndarray of float, in µm²
            Membrane area of each model node inside the patch, zero elsewhere.
        """
        radius = 4.0 * np.sqrt(area / np.pi) + 1e-9
        dist = dijkstra(self.graph, indices=local_vertex, limit=radius)
        order = np.argsort(dist)
        order = order[np.isfinite(dist[order])]
        cum = np.cumsum(self.area[order])
        k = int(np.searchsorted(cum, area)) + 1
        w = np.zeros(self.n_nodes)
        sel = order[: min(k, len(order))]
        w[self.vertices[sel]] = self.area[sel]
        return w


@dataclass
class Sites:
    """A set of membrane sites with their patches.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per site: ``site_id``, ``x, y, z`` (µm, as given), ``area``
        (µm²), ``kind`` (``synapse``, ``neck`` or ``probe``), ``axon_id``
        (nullable), ``source``, ``snap_distance`` (µm), ``patch_area`` (µm²,
        as realized).
    patches : list of (N,) ndarray of float
        Patch weights per site, in the same order as ``table``.
    """

    table: pd.DataFrame
    patches: list[NDArray[np.float64]]

    def __len__(self) -> int:
        """Return the number of sites."""
        return len(self.table)

    def subset(self, mask: ArrayLike) -> "Sites":
        """Sites where ``mask`` is true.

        Parameters
        ----------
        mask : (len(self),) array_like of bool

        Returns
        -------
        Sites
        """
        m = np.asarray(mask, dtype=bool)
        idx = np.flatnonzero(m)
        return Sites(
            self.table.iloc[idx].reset_index(drop=True), [self.patches[i] for i in idx]
        )

    def concat(self, other: "Sites") -> "Sites":
        """Append another site set.

        Parameters
        ----------
        other : Sites

        Returns
        -------
        Sites
        """
        table = pd.concat([self.table, other.table], ignore_index=True)
        return Sites(table, self.patches + other.patches)


def sites_from_points(
    surface: MembraneSurface,
    points: ArrayLike,
    kind: str,
    area: float = PLACEHOLDER_PSD_AREA,
    snap_tol: float = 0.25,
    axon_groups: list[list[int]] | None = None,
    source: str = "data",
) -> Sites:
    """Make sites from given positions (e.g. AZ or neck points).

    Parameters
    ----------
    surface : MembraneSurface
    points : (K, 3) array_like of float, in µm
    kind : {"synapse", "neck", "probe"}
    area : float, in µm²
        Patch area per site. The default is a placeholder, not data.
    snap_tol : float, in µm
        Maximum distance from a point to the membrane. AZ points lie up to
        about 0.16 µm off the TS meshes (presynaptic side).
    axon_groups : list of list of int, optional
        0-based point indices per axon (see :func:`cabaret.io.load_axon_groups`).
    source : str
        Provenance label.

    Returns
    -------
    Sites

    Raises
    ------
    ValueError
        If any point is farther than ``snap_tol`` from the membrane.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    local, dist = surface.snap(pts)
    if np.any(dist > snap_tol):
        bad = np.flatnonzero(dist > snap_tol)
        raise ValueError(
            f"points {bad.tolist()} lie more than {snap_tol} µm from the membrane"
        )
    axon = np.full(len(pts), pd.NA, dtype=object)
    for a, members in enumerate(axon_groups or []):
        for i in members:
            axon[i] = a
    patches = [surface.patch(int(v), area) for v in local]
    table = pd.DataFrame(
        {
            "site_id": [f"{kind}{i}" for i in range(len(pts))],
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": pts[:, 2],
            "area": area,
            "kind": kind,
            "axon_id": pd.array(axon, dtype="Int64"),
            "source": source,
            "snap_distance": dist,
            "patch_area": [p.sum() for p in patches],
        }
    )
    return Sites(table, patches)


def sample_uniform(
    surface: MembraneSurface,
    n: int,
    rng: np.random.Generator,
    area: float = PLACEHOLDER_PSD_AREA,
    min_spacing: float = 0.0,
    exclude: ArrayLike | None = None,
    exclude_radius: float = 0.0,
    max_tries: int = 100_000,
) -> Sites:
    """Area-weighted uniform sites on the membrane.

    Parameters
    ----------
    surface : MembraneSurface
    n : int
        Number of sites.
    rng : numpy.random.Generator
    area : float, in µm²
        Patch area per site (placeholder default).
    min_spacing : float, in µm
        Minimum Euclidean distance between sites (PLACEHOLDER for geodesic
        Poisson-disk spacing).
    exclude : (K, 3) array_like of float, optional, in µm
        Positions to keep away from (e.g. neck points).
    exclude_radius : float, in µm
    max_tries : int

    Returns
    -------
    Sites

    Raises
    ------
    RuntimeError
        If ``n`` sites can't be placed within ``max_tries`` draws.
    """
    p = surface.area / surface.area.sum()
    excl = None if exclude is None else np.atleast_2d(exclude)
    chosen: list[int] = []
    tries = 0
    while len(chosen) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"placed only {len(chosen)} of {n} sites")
        v = int(rng.choice(len(p), p=p))
        x = surface.position[v]
        if (
            chosen
            and np.min(np.linalg.norm(surface.position[chosen] - x, axis=1))
            < min_spacing
        ):
            continue
        if (
            excl is not None
            and np.min(np.linalg.norm(excl - x, axis=1)) < exclude_radius
        ):
            continue
        chosen.append(v)
    pts = surface.position[chosen]
    return sites_from_points(surface, pts, "synapse", area=area, source="uniform")


def jitter_sites(
    surface: MembraneSurface,
    sites: Sites,
    radius: float,
    rng: np.random.Generator,
) -> Sites:
    """Move each site to a random membrane point within a geodesic radius.

    Each site moves to a vertex drawn area-weighted from the geodesic disk of
    ``radius`` around its current position. Area, kind and axon labels are
    kept. This tests how much results depend on exact synapse positions.

    Parameters
    ----------
    surface : MembraneSurface
    sites : Sites
    radius : float, in µm
        Geodesic jitter radius.
    rng : numpy.random.Generator

    Returns
    -------
    Sites
    """
    local, _ = surface.snap(sites.table[["x", "y", "z"]].to_numpy())
    new_pts = np.empty((len(local), 3))
    for i, v in enumerate(local):
        dist = dijkstra(surface.graph, indices=int(v), limit=radius)
        cand = np.flatnonzero(np.isfinite(dist))
        p = surface.area[cand] / surface.area[cand].sum()
        new_pts[i] = surface.position[rng.choice(cand, p=p)]
    out = sites_from_points(
        surface,
        new_pts,
        kind=str(sites.table["kind"].iloc[0]),
        area=float(sites.table["area"].iloc[0]),
        source=f"jitter{radius:g}",
    )
    out.table["axon_id"] = sites.table["axon_id"].to_numpy()
    return out
