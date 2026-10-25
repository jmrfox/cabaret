"""Building the reference model from a closed surface.

The reference model is the FEM network of a region's interior, ready for
port analysis:

1. the surface is filled with tetrahedra (or a given volume mesh is used);
2. the FEM network is built on the mesh vertices;
3. each **attachment**, a set of FEM nodes where the region joins the rest
   of the cell (a spine neck, the cut ends of a dendrite segment), is tied
   into one equipotential electrode, and its membrane is removed;
4. the remaining membrane is extracted, and optional site points (synapses,
   probes) are snapped onto it as membrane patches.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray

from cabaret.circuit import CompartmentGraph, add_electrode, point_port
from cabaret.fem import fem_model
from cabaret.sites import (
    PLACEHOLDER_PSD_AREA,
    MembraneSurface,
    Sites,
    sites_from_points,
)
from cabaret.tet import (
    FidelityReport,
    TetMesh,
    _fidelity,
    tetrahedralize,
    tetrahedralize_cached,
)

Attachments = Sequence[ArrayLike] | Callable[[TetMesh], Sequence[ArrayLike]]


@dataclass
class ReferenceModel:
    """A region's FEM reference model with its attachment ports and sites.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The region's closed surface, in µm.
    tets : TetMesh
        Its tetrahedral mesh.
    fidelity : FidelityReport
        Area and volume change from meshing.
    fem : CompartmentGraph
        FEM network, one node per mesh vertex, without electrodes.
    network : CompartmentGraph
        ``fem`` with one electrode node per attachment. Use this one for
        port analysis.
    attachment_nodes : list of int
        Electrode node of each attachment in ``network``.
    attachment_ports : list of (N,) ndarray of float
        Port weights of each attachment (one-hot on its electrode).
    attachment_contacts : list of ndarray of int
        FEM nodes tied into each attachment electrode.
    membrane : MembraneSurface
        The real membrane of ``network``, for placing patches.
    sites : Sites or None
        Sites snapped onto the membrane, if site points were given.
    """

    mesh: trimesh.Trimesh
    tets: TetMesh
    fidelity: FidelityReport
    fem: CompartmentGraph
    network: CompartmentGraph
    attachment_nodes: list[int]
    attachment_ports: list[NDArray[np.float64]]
    attachment_contacts: list[NDArray[np.int64]]
    membrane: MembraneSurface
    sites: Sites | None


def build_reference(
    mesh: trimesh.Trimesh,
    attachments: Attachments = (),
    site_points: ArrayLike | None = None,
    site_kind: str = "synapse",
    site_area: float = PLACEHOLDER_PSD_AREA,
    site_groups: list[list[int]] | None = None,
    edge_length: float | None = None,
    envelope: float | None = None,
    coarsen: bool = False,
    cache: Path | None = None,
    tets: TetMesh | None = None,
) -> ReferenceModel:
    """Build the reference model of a region from its surface.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Closed surface, in µm.
    attachments : sequence of array_like of int, or callable
        FEM node sets where the region joins the rest of the cell, one per
        attachment. A callable receives the tetrahedral mesh and returns
        them, for attachments that are found geometrically (e.g.
        :func:`cabaret.graft.attachment_contacts`).
    site_points : (K, 3) array_like of float, optional, in µm
        Site positions, snapped onto the membrane left after attachments.
    site_kind : str
        Kind label of the sites (see :func:`cabaret.sites_from_points`).
    site_area : float, in µm²
        Patch area per site. The default is a placeholder, not data.
    site_groups : list of list of int, optional
        0-based site indices per group (e.g. per presynaptic axon).
    edge_length, envelope, coarsen
        Meshing settings (see :func:`cabaret.tetrahedralize`).
        ``edge_length`` is required unless ``tets`` is given.
    cache : pathlib.Path, optional
        ``.npz`` file for the tetrahedral mesh (see
        :func:`cabaret.tetrahedralize_cached`).
    tets : TetMesh, optional
        Use this volume mesh instead of meshing (e.g. a refined mesh for a
        convergence study). The meshing settings are then ignored.

    Returns
    -------
    ReferenceModel

    Raises
    ------
    ValueError
        If an attachment has no nodes, or neither ``tets`` nor
        ``edge_length`` is given.

    Examples
    --------
    >>> surface = trimesh.creation.capsule(height=4.0, radius=0.5)  # doctest: +SKIP
    >>> base = lambda tets: [np.flatnonzero(tets.points[:, 2] < -2.2)]  # doctest: +SKIP
    >>> ref = build_reference(surface, base, edge_length=0.2)  # doctest: +SKIP
    >>> ports = ref.attachment_ports  # doctest: +SKIP
    >>> z = port_impedance(ref.network, Passive(), ports)  # doctest: +SKIP
    """
    if tets is not None:
        fidelity = _fidelity(mesh, tets)
    elif edge_length is None:
        raise ValueError("give edge_length (to mesh) or tets")
    elif cache is not None:
        tets, fidelity = tetrahedralize_cached(
            mesh, cache, edge_length, envelope, coarsen=coarsen
        )
    else:
        tets, fidelity = tetrahedralize(mesh, edge_length, envelope, coarsen=coarsen)
    fem, _ = fem_model(tets)
    contacts = [
        np.asarray(c, dtype=np.int64)
        for c in (attachments(tets) if callable(attachments) else attachments)
    ]
    network, nodes = fem, []
    for k, c in enumerate(contacts):
        if c.size == 0:
            raise ValueError(f"attachment {k} has no nodes")
        network, node = add_electrode(network, c)
        nodes.append(node)
    membrane = MembraneSurface.from_model(tets, network)
    sites = None
    if site_points is not None:
        sites = sites_from_points(
            membrane, site_points, site_kind, area=site_area, axon_groups=site_groups
        )
    return ReferenceModel(
        mesh=mesh,
        tets=tets,
        fidelity=fidelity,
        fem=fem,
        network=network,
        attachment_nodes=nodes,
        attachment_ports=[point_port(network, n) for n in nodes],
        attachment_contacts=contacts,
        membrane=membrane,
        sites=sites,
    )
