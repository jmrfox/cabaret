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


def _plotter(plotter: "pv.Plotter | None", off_screen: bool) -> "pv.Plotter":
    import pyvista as pv

    return plotter if plotter is not None else pv.Plotter(off_screen=off_screen)


def plot_surface(
    tets: TetMesh,
    scalars: ArrayLike | None = None,
    name: str = "value",
    cmap: str = "viridis",
    log_scale: bool = False,
    clim: Sequence[float] | None = None,
    show_edges: bool = False,
    opacity: float = 1.0,
    plotter: "pv.Plotter | None" = None,
    off_screen: bool = False,
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
    plotter : pyvista.Plotter, optional
        Add to an existing plotter instead of creating one.
    off_screen : bool

    Returns
    -------
    pyvista.Plotter
    """
    data = {} if scalars is None else {name: scalars}
    surf = to_pyvista(tets, point_data=data).extract_surface(
        algorithm="dataset_surface"
    )
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(
        surf,
        scalars=name if scalars is not None else None,
        cmap=cmap,
        log_scale=log_scale,
        clim=clim,
        show_edges=show_edges,
        opacity=opacity,
        color=None if scalars is not None else "lightgray",
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
    off_screen: bool = False,
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
    off_screen : bool

    Returns
    -------
    pyvista.Plotter
    """
    grid = to_pyvista(tets, point_data={name: scalars})
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(
        grid.slice(normal=normal, origin=origin), scalars=name, cmap=cmap, clim=clim
    )
    pl.add_mesh(
        grid.extract_surface(algorithm="dataset_surface"),
        color="lightgray",
        opacity=0.15,
    )
    return pl


def plot_isosurfaces(
    tets: TetMesh,
    scalars: ArrayLike,
    n: int = 10,
    name: str = "value",
    cmap: str = "viridis",
    plotter: "pv.Plotter | None" = None,
    off_screen: bool = False,
) -> "pv.Plotter":
    """Show surfaces of constant value inside the volume.

    For a potential field these are equipotentials, which are the planned
    compartment boundaries of the partition.

    Parameters
    ----------
    tets : TetMesh
    scalars : (N,) array_like of float
    n : int
        Number of evenly spaced levels.
    name : str
    cmap : str
    plotter : pyvista.Plotter, optional
    off_screen : bool

    Returns
    -------
    pyvista.Plotter
    """
    grid = to_pyvista(tets, point_data={name: scalars})
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(grid.contour(isosurfaces=n, scalars=name), scalars=name, cmap=cmap)
    pl.add_mesh(
        grid.extract_surface(algorithm="dataset_surface"),
        color="lightgray",
        opacity=0.15,
    )
    return pl


def plot_streamlines(
    tets: TetMesh,
    vectors: ArrayLike,
    source_center: Sequence[float],
    source_radius: float,
    n_points: int = 200,
    name: str = "J",
    plotter: "pv.Plotter | None" = None,
    off_screen: bool = False,
) -> "pv.Plotter":
    """Show current paths as streamlines of a per-tetrahedron vector field.

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
    plotter : pyvista.Plotter, optional
    off_screen : bool

    Returns
    -------
    pyvista.Plotter
    """
    grid = to_pyvista(tets, cell_data={name: vectors}).cell_data_to_point_data()
    lines = grid.streamlines(
        vectors=name,
        source_center=source_center,
        source_radius=source_radius,
        n_points=n_points,
        integration_direction="both",
    )
    lines.point_data["|J|"] = np.linalg.norm(lines.point_data[name], axis=1)
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(lines.tube(radius=0.01 * source_radius), scalars="|J|", log_scale=True)
    pl.add_mesh(
        grid.extract_surface(algorithm="dataset_surface"),
        color="lightgray",
        opacity=0.15,
    )
    return pl


def plot_sites(
    tets: TetMesh,
    patches: Sequence[ArrayLike],
    plotter: "pv.Plotter | None" = None,
    off_screen: bool = False,
) -> "pv.Plotter":
    """Show membrane patches on the surface, one colour per patch.

    Parameters
    ----------
    tets : TetMesh
    patches : sequence of (N,) array_like of float
        Patch weights per vertex (nonzero inside the patch), e.g. from
        :func:`camovox.sites.ball_patch`.
    plotter : pyvista.Plotter, optional
    off_screen : bool

    Returns
    -------
    pyvista.Plotter
    """
    label = np.full(len(tets.points), np.nan)
    for k, w in enumerate(patches):
        label[np.asarray(w) > 0] = k
    surf = to_pyvista(tets, point_data={"site": label}).extract_surface(
        algorithm="dataset_surface"
    )
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(surf, color="lightgray", opacity=0.4)
    pl.add_mesh(surf.threshold(-0.5, scalars="site"), scalars="site", cmap="tab20")
    return pl


def plot_mesh_quality(
    graph: CompartmentGraph,
    tets: TetMesh,
    plotter: "pv.Plotter | None" = None,
    off_screen: bool = False,
) -> "pv.Plotter":
    """Show negative couplings (κ < 0) of a FEM network as coloured lines.

    Parameters
    ----------
    graph : CompartmentGraph
        FEM reference network.
    tets : TetMesh
        The mesh it was built from.
    plotter : pyvista.Plotter, optional
    off_screen : bool

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    neg = graph.kappa < 0
    e = graph.edges[neg]
    lines = pv.PolyData(
        graph.position, lines=np.hstack([np.full((len(e), 1), 2), e]).ravel()
    )
    lines.cell_data["kappa (µm)"] = graph.kappa[neg]
    pl = _plotter(plotter, off_screen)
    pl.add_mesh(lines, scalars="kappa (µm)", cmap="magma", line_width=2)
    pl.add_mesh(
        to_pyvista(tets).extract_surface(algorithm="dataset_surface"),
        color="lightgray",
        opacity=0.15,
    )
    return pl


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
