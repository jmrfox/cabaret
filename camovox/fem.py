"""P1 finite-element reference model.

The cell interior obeys current conservation, and the membrane is a Robin
boundary condition. With linear (P1) elements and lumped boundary mass, the
discrete system is an electrical network on the mesh vertices:

- the stiffness matrix gives the couplings, ``kappa_ij = -K_ij`` (µm), where
  ``K`` is assembled with unit conductivity;
- each boundary vertex carries a third of the area of its boundary
  triangles as membrane.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from camovox.circuit import CompartmentGraph
from camovox.tet import TetMesh


@dataclass(frozen=True)
class FemReport:
    """Quality of the FEM network.

    Parameters
    ----------
    n_negative : int
        Number of couplings with κ < 0 (from obtuse dihedral angles). These
        act as negative conductances.
    negative_fraction : float
        Sum of |κ| over negative couplings, divided by the sum of |κ|.
    """

    n_negative: int
    negative_fraction: float


def _lumped_boundary_area(tets: TetMesh) -> NDArray[np.float64]:
    faces = tets.boundary_faces()
    p = tets.points[faces]
    tri_area = 0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
    )
    area = np.zeros(len(tets.points))
    np.add.at(area, faces.ravel(), np.repeat(tri_area / 3.0, 3))
    return area


def fem_model(tets: TetMesh) -> tuple[CompartmentGraph, FemReport]:
    """Build the FEM reference network from a tetrahedral mesh.

    Parameters
    ----------
    tets : TetMesh
        Volume mesh, in µm.

    Returns
    -------
    CompartmentGraph
        One node per mesh vertex.
    FemReport
        Count and weight of negative couplings.

    Examples
    --------
    >>> ref, report = fem_model(tets)  # doctest: +SKIP
    """
    from skfem import Basis, ElementTetP1, MeshTet
    from skfem.models.poisson import laplace

    mesh = MeshTet(tets.points.T, tets.tets.T)
    k = laplace.assemble(Basis(mesh, ElementTetP1()))
    upper = sp.triu(k, k=1).tocoo()
    kappa = -upper.data
    keep = kappa != 0.0
    edges = np.column_stack([upper.row, upper.col])[keep].astype(np.int64)
    kappa = kappa[keep]

    neg = kappa < 0
    report = FemReport(
        n_negative=int(neg.sum()),
        negative_fraction=float(np.abs(kappa[neg]).sum() / np.abs(kappa).sum()),
    )
    graph = CompartmentGraph(
        area=_lumped_boundary_area(tets),
        position=tets.points.copy(),
        edges=edges,
        kappa=kappa,
    )
    return graph, report
