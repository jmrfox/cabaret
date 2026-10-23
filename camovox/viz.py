"""Visualization of meshes and FEM solutions.

Thin wrappers over PyVista (3D) and matplotlib (profiles). Every 3D function
returns a ``pyvista.Plotter``: call ``.show()`` to display it, or
``.screenshot(path)`` with ``off_screen=True``. For anything these helpers
don't cover, build the grid with :func:`to_pyvista` and use PyVista directly,
or export it with :func:`write_vtu` and open it in ParaView.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camovox.circuit import CompartmentGraph
from camovox.tet import TetMesh

if TYPE_CHECKING:
    import matplotlib.axes
    import pyvista as pv

    from camovox.reduce import PortTree


def to_pyvista(
    tets: TetMesh,
    point_data: Mapping[str, ArrayLike] | None = None,
    cell_data: Mapping[str, ArrayLike] | None = None,
) -> "pv.UnstructuredGrid":
    """Convert a tetrahedral mesh and its fields to a PyVista grid.

    Parameters
    ----------
    tets : TetMesh
    point_data : mapping of str to (N,) or (N, k) array_like, optional
        Fields per vertex (e.g. voltage).
    cell_data : mapping of str to (M,) or (M, k) array_like, optional
        Fields per tetrahedron (e.g. current density).

    Returns
    -------
    pyvista.UnstructuredGrid
    """
    import pyvista as pv

    m = len(tets.tets)
    cells = np.hstack([np.full((m, 1), 4), tets.tets]).ravel()
    celltypes = np.full(m, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, tets.points)
    for name, val in (point_data or {}).items():
        grid.point_data[name] = np.asarray(val)
    for name, val in (cell_data or {}).items():
        grid.cell_data[name] = np.asarray(val)
    return grid


def write_vtu(
    path: Path,
    tets: TetMesh,
    point_data: Mapping[str, ArrayLike] | None = None,
    cell_data: Mapping[str, ArrayLike] | None = None,
) -> None:
    """Write a mesh and its fields to a ``.vtu`` file (opens in ParaView).

    Parameters
    ----------
    path : pathlib.Path
    tets : TetMesh
    point_data, cell_data : mapping of str to array_like, optional
        See :func:`to_pyvista`.
    """
    to_pyvista(tets, point_data, cell_data).save(Path(path))


def _plotter(plotter: "pv.Plotter | None", off_screen: bool | None) -> "pv.Plotter":
    import pyvista as pv

    return plotter if plotter is not None else pv.Plotter(off_screen=off_screen)


def _bar(values: ArrayLike, title: str, log_scale: bool = False) -> dict:
    """Colour-bar arguments with tick precision matched to the data range."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    fmt = "%.3g"
    if log_scale:
        fmt = "%.1e"
    elif v.size:
        lo, hi = float(v.min()), float(v.max())
        span, mag = hi - lo, max(abs(lo), abs(hi))
        if mag >= 1e4 or 0 < mag < 1e-2:
            fmt = "%.2e"
        elif span > 0:
            # Enough decimals that 5 evenly spaced ticks are distinct.
            fmt = f"%.{max(0, int(np.ceil(-np.log10(span / 4))) + 1)}f"
    # Narrower than the default so the end labels stay inside the image.
    return {"title": title, "fmt": fmt, "n_labels": 5, "width": 0.6, "position_x": 0.2}


def _surface(grid: "pv.UnstructuredGrid") -> "pv.PolyData":
    return grid.extract_surface(algorithm="dataset_surface")


def _ghost(pl: "pv.Plotter", tets: TetMesh, opacity: float = 0.15) -> None:
    pl.add_mesh(_surface(to_pyvista(tets)), color="lightgray", opacity=opacity)


def plot_surface(
    tets: TetMesh,
    scalars: ArrayLike | None = None,
    name: str = "value",
    cmap: str = "viridis",
    log_scale: bool = False,
    clim: Sequence[float] | None = None,
    show_edges: bool = False,
    opacity: float = 1.0,
    categories: bool = False,
    plotter: "pv.Plotter | None" = None,
    off_screen: bool | None = None,
) -> "pv.Plotter":
    """Show the membrane surface, optionally coloured by a vertex field.

    Parameters
    ----------
    tets : TetMesh
    scalars : (N,) array_like of float, optional
        Vertex field, e.g. voltage in mV. Units belong in ``name``.
    name : str
        Colour bar title, e.g. ``"V (mV)"``.
    cmap : str
        Matplotlib colormap name. Use a diverging map (``"RdBu_r"``) for
        signed errors.
    log_scale : bool
        Logarithmic colour scale (for attenuation over orders of magnitude).
    clim : (2,) sequence of float, optional
        Fixed colour limits (keep them fixed across frames of an animation).
    show_edges : bool
    opacity : float
    categories : bool
        Treat ``scalars`` as integer labels with one discrete colour each.
    plotter : pyvista.Plotter, optional
        Add to an existing plotter instead of creating one.
    off_screen : bool, optional
        Render without a window. ``None`` follows ``pyvista.OFF_SCREEN``.

    Returns
    -------
    pyvista.Plotter
    """
    pl = _plotter(plotter, off_screen)
    if scalars is None:
        surf = _surface(to_pyvista(tets))
        pl.add_mesh(surf, color="lightgray", show_edges=show_edges, opacity=opacity)
        return pl
    surf = _surface(to_pyvista(tets, point_data={name: scalars}))
    bar = _bar(surf.point_data[name], name, log_scale)
    if categories:
        bar["fmt"] = "%.0f"
    pl.add_mesh(
        surf,
        scalars=name,
        cmap=cmap,
        log_scale=log_scale,
        clim=clim,
        show_edges=show_edges,
        opacity=opacity,
        categories=categories,
        scalar_bar_args=bar,
    )
    return pl


def plot_slice(
    tets: TetMesh,
    scalars: ArrayLike,
    name: str = "value",
    normal: str | Sequence[float] = "z",
    origin: Sequence[float] | None = None,
    cmap: str = "viridis",
    clim: Sequence[float] | None = None,
    plotter: "pv.Plotter | None" = None,
    off_screen: bool | None = None,
) -> "pv.Plotter":
    """Show a vertex field on a planar cut through the volume.

    Parameters
    ----------
    tets : TetMesh
    scalars : (N,) array_like of float
    name : str
        Colour bar title with units.
    normal : {"x", "y", "z"} or (3,) sequence of float
        Plane normal.
    origin : (3,) sequence of float, optional, in µm
        A point on the plane. Defaults to the mesh centre.
    cmap : str
    clim : (2,) sequence of float, optional
    plotter : pyvista.Plotter, optional
    off_screen : bool, optional
        Render without a window. ``None`` follows ``pyvista.OFF_SCREEN``.

    Returns
    -------
    pyvista.Plotter
    """
    grid = to_pyvista(tets, point_data={name: scalars})
    cut = grid.slice(normal=normal, origin=origin)
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(
        cut,
        scalars=name,
        cmap=cmap,
        clim=clim,
        scalar_bar_args=_bar(cut.point_data[name], name),
    )
    _ghost(pl, tets)
    return pl


def plot_isosurfaces(
    tets: TetMesh,
    scalars: ArrayLike,
    n: int = 10,
    name: str = "value",
    cmap: str = "viridis",
    plotter: "pv.Plotter | None" = None,
    off_screen: bool | None = None,
) -> "pv.Plotter":
    """Show surfaces of constant value inside the volume.

    For a potential field these are equipotentials, which are the planned
    compartment boundaries of the partition. Levels are spaced evenly
    between the field's minimum and maximum, excluding the extremes.

    Parameters
    ----------
    tets : TetMesh
    scalars : (N,) array_like of float
        Subtract the minimum first if the field has a large offset.
    n : int
        Number of levels.
    name : str
        Colour bar title with units.
    cmap : str
    plotter : pyvista.Plotter, optional
    off_screen : bool, optional
        Render without a window. ``None`` follows ``pyvista.OFF_SCREEN``.

    Returns
    -------
    pyvista.Plotter
    """
    s = np.asarray(scalars, dtype=float)
    grid = to_pyvista(tets, point_data={name: s})
    levels = np.linspace(s.min(), s.max(), n + 2)[1:-1]
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(
        grid.contour(isosurfaces=levels, scalars=name),
        scalars=name,
        cmap=cmap,
        clim=(s.min(), s.max()),
        scalar_bar_args=_bar(s, name),
    )
    _ghost(pl, tets)
    return pl


def plot_streamlines(
    tets: TetMesh,
    vectors: ArrayLike,
    source_center: Sequence[float],
    source_radius: float,
    n_points: int = 200,
    name: str = "current density (A/µm²)",
    max_steps: int = 20000,
    plotter: "pv.Plotter | None" = None,
    off_screen: bool | None = None,
) -> "pv.Plotter":
    """Show current paths as streamlines of a per-tetrahedron vector field.

    Lines follow the field's direction (integrated on the unit vector field,
    so tiny magnitudes don't stall them) and are coloured by its magnitude on
    a log scale.

    Parameters
    ----------
    tets : TetMesh
    vectors : (M, 3) array_like of float
        Vector per tetrahedron, e.g. from :func:`camovox.fem.current_density`.
    source_center : (3,) sequence of float, in µm
        Centre of the seed sphere (e.g. the injection site).
    source_radius : float, in µm
    n_points : int
        Number of seed points.
    name : str
        Colour bar title for the magnitude, with units.
    max_steps : int
        Maximum integration steps per line.
    plotter : pyvista.Plotter, optional
    off_screen : bool, optional
        Render without a window. ``None`` follows ``pyvista.OFF_SCREEN``.

    Returns
    -------
    pyvista.Plotter
    """
    vec = np.asarray(vectors, dtype=float)
    mag = np.linalg.norm(vec, axis=1)
    unit = vec / np.where(mag > 0, mag, 1.0)[:, None]
    grid = to_pyvista(tets, cell_data={"dir": unit, name: mag})
    grid = grid.cell_data_to_point_data()
    size = float(np.linalg.norm(np.ptp(tets.points, axis=0)))
    lines = grid.streamlines(
        vectors="dir",
        source_center=source_center,
        source_radius=source_radius,
        n_points=n_points,
        integration_direction="both",
        max_steps=max_steps,
        max_length=10.0 * size,
        initial_step_length=0.2,
        step_unit="cl",
    )
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(
        lines.tube(radius=0.004 * size),
        scalars=name,
        log_scale=True,
        scalar_bar_args=_bar(lines.point_data[name], name, log_scale=True),
    )
    _ghost(pl, tets)
    return pl


def _patch_centers(tets: TetMesh, patches: Sequence[ArrayLike]) -> NDArray[np.float64]:
    out = np.empty((len(patches), 3))
    n = len(tets.points)
    for k, w in enumerate(patches):
        # Weights may be longer than the mesh (extra nodes such as electrodes
        # are appended after the mesh vertices); only mesh vertices have
        # positions.
        w = np.asarray(w, dtype=float)[:n]
        out[k] = (w[:, None] * tets.points).sum(axis=0) / w.sum()
    return out


def plot_sites(
    tets: TetMesh,
    patches: Sequence[ArrayLike],
    labels: Sequence[float] | None = None,
    name: str = "site",
    cmap: str = "tab20",
    marker_radius: float | None = None,
    plotter: "pv.Plotter | None" = None,
    off_screen: bool | None = None,
) -> "pv.Plotter":
    """Show sites as coloured spheres on a translucent membrane.

    Each patch is drawn as a sphere at its area-weighted centre, because real
    patches (about 0.05 µm²) are too small to see at whole-spine scale. Colours
    are categorical, one per distinct label.

    Parameters
    ----------
    tets : TetMesh
    patches : sequence of (N,) array_like of float
        Patch weights per vertex (nonzero inside the patch).
    labels : sequence of float, optional
        One value per patch (e.g. axon ID). Defaults to the patch index.
    name : str
        Colour bar title.
    cmap : str
    marker_radius : float, optional, in µm
        Sphere radius. Defaults to 1.5% of the mesh diagonal.
    plotter : pyvista.Plotter, optional
    off_screen : bool, optional
        Render without a window. ``None`` follows ``pyvista.OFF_SCREEN``.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    lab = np.arange(len(patches)) if labels is None else np.asarray(labels)
    size = float(np.linalg.norm(np.ptp(tets.points, axis=0)))
    r = 0.015 * size if marker_radius is None else marker_radius
    pts = pv.PolyData(_patch_centers(tets, patches))
    pts.point_data[name] = lab
    spheres = pts.glyph(geom=pv.Sphere(radius=r), scale=False, orient=False)
    pl = _plotter(plotter, off_screen)
    _ghost(pl, tets, opacity=0.3)
    pl.add_mesh(
        spheres,
        scalars=name,
        cmap=cmap,
        categories=True,
        scalar_bar_args={
            "title": name,
            "fmt": "%.0f",
            "n_labels": len(set(lab)),
            "width": 0.6,
            "position_x": 0.2,
        },
    )
    return pl


def plot_mesh_quality(
    graph: CompartmentGraph,
    tets: TetMesh,
    fraction: float = 0.2,
    plotter: "pv.Plotter | None" = None,
    off_screen: bool | None = None,
) -> "pv.Plotter":
    """Show the most negative couplings (κ < 0) of a FEM network as lines.

    Parameters
    ----------
    graph : CompartmentGraph
        FEM reference network.
    tets : TetMesh
        The mesh it was built from.
    fraction : float
        Share of negative couplings to draw, most negative first (1 = all).
    plotter : pyvista.Plotter, optional
    off_screen : bool, optional
        Render without a window. ``None`` follows ``pyvista.OFF_SCREEN``.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    neg = np.flatnonzero(graph.kappa < 0)
    neg = neg[np.argsort(graph.kappa[neg])][: max(1, int(fraction * len(neg)))]
    e = graph.edges[neg]
    lines = pv.PolyData(
        graph.position, lines=np.hstack([np.full((len(e), 1), 2), e]).ravel()
    )
    title = "kappa (µm)"
    lines.cell_data[title] = graph.kappa[neg]
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(
        lines,
        scalars=title,
        cmap="magma",
        line_width=2,
        scalar_bar_args=_bar(graph.kappa[neg], title),
    )
    _ghost(pl, tets)
    return pl


def plot_port_tree(
    tree: "PortTree",
    labels: Sequence[str],
    colors: Sequence[float] | None = None,
    root: int = 0,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Draw a resistor tree radially around a root port.

    The root (e.g. the neck) sits at the centre. Each node's distance from the
    centre is its **resistance from the root** (MΩ, the axis scale), so
    electrically remote synapses sit far out. Subtrees get angular sectors in
    proportion to their number of ports, so branches never cross.

    Parameters
    ----------
    tree : camovox.reduce.PortTree
    labels : sequence of str
        One label per port, in port order.
    colors : sequence of float, optional
        One value per port for the dot colour (e.g. lobe), drawn with tab10.
    root : int
        Port index placed at the centre.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    g = tree.graph()
    r0 = int(tree.port_node[root])
    dist = nx.single_source_dijkstra_path_length(g, r0, weight="resistance")
    children = nx.bfs_tree(g, r0)
    is_port = np.zeros(tree.n_nodes, dtype=bool)
    is_port[tree.port_node] = True
    leaves = {
        v: max(1, sum(is_port[u] for u in nx.descendants(children, v) | {v})) for v in g
    }
    angle: dict[int, float] = {r0: 0.0}

    def place(v: int, lo: float, hi: float) -> None:
        angle[v] = 0.5 * (lo + hi)
        kids = list(children.successors(v))
        total = sum(leaves[k] for k in kids) or 1
        start = lo
        for k in kids:
            span = (hi - lo) * leaves[k] / total
            place(k, start, start + span)
            start += span

    place(r0, 0.0, 2 * np.pi)
    angle[r0] = 0.0
    pos = {
        v: (dist[v] / 1e6 * np.cos(angle[v]), dist[v] / 1e6 * np.sin(angle[v]))
        for v in g
    }
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    for a, b in g.edges:
        ax.plot(*zip(pos[a], pos[b], strict=True), color="gray", lw=1, zorder=1)
    internal = [v for v in g if not is_port[v]]
    ax.scatter(*np.array([pos[v] for v in internal]).T, s=8, color="gray", zorder=2)
    port_nodes = [int(v) for v in tree.port_node]
    xy = np.array([pos[v] for v in port_nodes])
    c = (
        None
        if colors is None
        else [f"C{int(x) % 10}" if x >= 0 else "k" for x in colors]
    )
    ax.scatter(xy[:, 0], xy[:, 1], s=60, c=c, zorder=3)
    for (x, y), text in zip(xy, labels, strict=True):
        ax.annotate(text, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlabel("MΩ from the root")
    ax.set_ylabel("MΩ from the root")
    return ax


def plot_profile(
    x: ArrayLike,
    series: Mapping[str, ArrayLike],
    xlabel: str,
    ylabel: str,
    reference: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Plot quantities against a 1D coordinate, with optional references.

    Parameters
    ----------
    x : (K,) array_like of float
        Shared coordinate for ``series``, e.g. distance along a cable in µm.
    series : mapping of str to (K,) array_like of float
        Model results, drawn as markers.
    xlabel, ylabel : str
        Axis labels with units.
    reference : mapping of str to (x_ref, y_ref), optional
        Reference curves (e.g. analytic solutions), drawn as lines.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    for label, (xr, yr) in (reference or {}).items():
        ax.plot(xr, yr, "-", lw=1.5, label=label)
    for label, y in series.items():
        ax.plot(x, y, ".", ms=3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    return ax


def vertex_distance_along(
    tets: TetMesh, axis: Sequence[float], origin: Sequence[float]
) -> NDArray[np.float64]:
    """Signed distance of each vertex along an axis, in µm.

    Parameters
    ----------
    tets : TetMesh
    axis : (3,) sequence of float
        Direction (normalized internally).
    origin : (3,) sequence of float, in µm

    Returns
    -------
    (N,) ndarray of float, in µm
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return (tets.points - np.asarray(origin, dtype=float)) @ a
