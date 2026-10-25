"""Toric spines of barn owl space-specific neurons (research module).

The TS dataset (``data/``) has, per spine ``<name>`` (e.g. ``TS1``):

- ``<name>.obj``: the spine surface, traced separately from the cell, in
  0.005 µm pixels;
- ``<name>_neckpoint.txt``: one point per neck, where the spine joins the
  cell;
- ``<name>_AZ.txt``: active-zone (synapse) positions, if any;
- ``<name>_syn_axon_dist.txt``: which AZs share a presynaptic axon, if any.

Each neck is an attachment, found against the segmented whole cell
(neuromesh-cgal parts, :func:`cabaret.graft.load_parts`), and the AZs are
the sites.
"""

from collections.abc import Sequence
from pathlib import Path

import trimesh

from cabaret.graft import attachment_contacts, region_parts
from cabaret.io import load_axon_groups, load_mesh, load_points
from cabaret.reference import ReferenceModel, build_reference
from cabaret.sites import PLACEHOLDER_PSD_AREA
from cabaret.tet import TetMesh

SCALE = 0.005  # µm per pixel of the TS data files


def load_toric_spine(
    name: str,
    data_dir: Path,
    parts: Sequence[trimesh.Trimesh],
    scale: float = SCALE,
    edge_length: float = 0.2,
    envelope: float = 0.005,
    coarsen: bool = False,
    tets: TetMesh | None = None,
    psd_area: float = PLACEHOLDER_PSD_AREA,
    contact_distance: float = 0.2,
    contact_radius: float = 3.0,
) -> ReferenceModel:
    """Load a toric spine as a reference model: necks attached, AZs as sites.

    Parameters
    ----------
    name : str
        Spine name, e.g. ``"TS1"``.
    data_dir : pathlib.Path
        Folder with the TS files (``../data`` from the notebooks).
    parts : sequence of trimesh.Trimesh
        Cell parts in µm (see :func:`cabaret.graft.load_parts`).
    scale : float
        µm per pixel of the data files.
    edge_length, envelope, coarsen
        Meshing settings (see :func:`cabaret.tetrahedralize`). The mesh is
        cached in ``data_dir/cache``, one file per setting.
    tets : TetMesh, optional
        Use this volume mesh instead of meshing.
    psd_area : float, in µm²
        AZ patch area. The default is a placeholder, not data.
    contact_distance, contact_radius : float, in µm
        See :func:`cabaret.graft.attachment_contacts`.

    Returns
    -------
    ReferenceModel
        One attachment per neck, in the order of the neck-point file; AZs
        as ``"synapse"`` sites, grouped by axon when the file exists.

    Raises
    ------
    ValueError
        If a neck has no contact with the cell.

    Examples
    --------
    >>> parts = graft.load_parts("../data/neuromesh/parts", SCALE)  # doctest: +SKIP
    >>> ts1 = load_toric_spine("TS1", "../data", parts)  # doctest: +SKIP
    >>> neck = ts1.attachment_ports[0]  # doctest: +SKIP
    """
    data_dir = Path(data_dir)
    mesh = load_mesh(data_dir / f"{name}.obj", scale=scale)
    own = region_parts(mesh, parts)
    necks = load_points(data_dir / f"{name}_neckpoint.txt", scale)
    suffix = "_coarse" if coarsen else ""
    cache = data_dir / "cache" / f"{name}_e{edge_length:g}_env{envelope:g}{suffix}.npz"

    def find_necks(t: TetMesh) -> list:
        contacts = attachment_contacts(
            t, parts, own, necks, contact_distance, contact_radius
        )
        for k, c in enumerate(contacts):
            if c.size == 0:
                raise ValueError(f"{name}: neck {k} has no contact with the cell")
        return contacts

    az_file = data_dir / f"{name}_AZ.txt"
    axon_file = data_dir / f"{name}_syn_axon_dist.txt"
    return build_reference(
        mesh,
        attachments=find_necks,
        site_points=load_points(az_file, scale) if az_file.exists() else None,
        site_kind="synapse",
        site_area=psd_area,
        site_groups=load_axon_groups(axon_file) if axon_file.exists() else None,
        edge_length=edge_length,
        envelope=envelope,
        coarsen=coarsen,
        cache=cache,
        tets=tets,
    )
