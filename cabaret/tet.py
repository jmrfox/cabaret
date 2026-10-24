"""Tetrahedral meshing of a closed surface."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray


@dataclass(frozen=True)
class TetMesh:
    """A tetrahedral volume mesh.

    Parameters
    ----------
    points : (N, 3) ndarray of float, in µm
    tets : (M, 4) ndarray of int
        Vertex indices of each tetrahedron.
    """

    points: NDArray[np.float64]
    tets: NDArray[np.int64]

    def save(self, path: Path) -> None:
        """Save to a compressed ``.npz`` file.

        Parameters
        ----------
        path : pathlib.Path
        """
        np.savez_compressed(Path(path), points=self.points, tets=self.tets)

    @classmethod
    def load(cls, path: Path) -> "TetMesh":
        """Load a mesh written by :meth:`save`.

        Parameters
        ----------
        path : pathlib.Path

        Returns
        -------
        TetMesh
        """
        with np.load(Path(path)) as f:
            return cls(points=f["points"], tets=f["tets"].astype(np.int64))

    def boundary_faces(self) -> NDArray[np.int64]:
        """Triangles that belong to exactly one tetrahedron.

        Returns
        -------
        (F, 3) ndarray of int
            Vertex indices (orientation not guaranteed).
        """
        t = self.tets
        faces = np.concatenate(
            [t[:, [0, 1, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 3]], t[:, [1, 2, 3]]]
        )
        key = np.sort(faces, axis=1)
        _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
        return faces[idx[counts == 1]]

    def volumes(self) -> NDArray[np.float64]:
        """Volume of each tetrahedron, in µm³."""
        p = self.points[self.tets]
        d = np.einsum(
            "ij,ij->i",
            p[:, 1] - p[:, 0],
            np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]),
        )
        return np.abs(d) / 6.0

    def surface_area(self) -> float:
        """Total boundary area, in µm²."""
        p = self.points[self.boundary_faces()]
        return float(
            0.5
            * np.linalg.norm(
                np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
            ).sum()
        )


@dataclass(frozen=True)
class FidelityReport:
    """How much tetrahedralization changed the surface.

    Parameters
    ----------
    area_in, area_out : float, in µm²
        Surface area before and after.
    volume_in, volume_out : float, in µm³
        Enclosed volume before and after.
    n_points, n_tets : int
        Size of the volume mesh.
    """

    area_in: float
    area_out: float
    volume_in: float
    volume_out: float
    n_points: int
    n_tets: int

    @property
    def area_error(self) -> float:
        """Relative change in surface area."""
        return self.area_out / self.area_in - 1.0

    @property
    def volume_error(self) -> float:
        """Relative change in volume."""
        return self.volume_out / self.volume_in - 1.0


def tetrahedralize(
    mesh: trimesh.Trimesh,
    edge_length: float,
    envelope: float | None = None,
    optimize: bool = True,
) -> tuple[TetMesh, FidelityReport]:
    """Fill a closed surface with tetrahedra using fTetWild.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Closed surface, in µm.
    edge_length : float, in µm
        Target tetrahedron edge length.
    envelope : float, optional, in µm
        Maximum allowed distance between the output and input surfaces.
        Defaults to ``edge_length / 50``. Must be well below the thinnest
        neck radius. PLACEHOLDER: no automatic check against neck size yet.
    optimize : bool
        Run fTetWild's element-quality optimization.

    Returns
    -------
    TetMesh
    FidelityReport

    Examples
    --------
    >>> sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    >>> tets, report = tetrahedralize(sphere, edge_length=1.0)  # doctest: +SKIP
    """
    import pytetwild  # imported lazily: it pulls in pyvista/vtk

    if envelope is None:
        envelope = edge_length / 50.0
    diag = float(np.linalg.norm(mesh.extents))
    points, tets = pytetwild.tetrahedralize(
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int32),
        edge_length_abs=edge_length,
        epsilon=envelope / diag,
        optimize=optimize,
        quiet=True,
    )
    used, inverse = np.unique(tets, return_inverse=True)
    tm = TetMesh(points=points[used], tets=inverse.reshape(tets.shape).astype(np.int64))
    return tm, _fidelity(mesh, tm)


def _fidelity(mesh: trimesh.Trimesh, tm: TetMesh) -> FidelityReport:
    return FidelityReport(
        area_in=float(mesh.area),
        area_out=tm.surface_area(),
        volume_in=float(abs(mesh.volume)),
        volume_out=float(tm.volumes().sum()),
        n_points=len(tm.points),
        n_tets=len(tm.tets),
    )


def tetrahedralize_cached(
    mesh: trimesh.Trimesh,
    cache: Path,
    edge_length: float,
    envelope: float | None = None,
    optimize: bool = True,
) -> tuple[TetMesh, FidelityReport]:
    """Like :func:`tetrahedralize`, but reuse a mesh saved at ``cache``.

    The cache is not keyed on the parameters: delete the file after changing
    ``edge_length`` or ``envelope``.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Closed surface, in µm.
    cache : pathlib.Path
        ``.npz`` file to read if it exists, otherwise to write.
    edge_length, envelope, optimize
        See :func:`tetrahedralize`.

    Returns
    -------
    TetMesh
    FidelityReport
    """
    cache = Path(cache)
    if cache.exists():
        tm = TetMesh.load(cache)
        return tm, _fidelity(mesh, tm)
    tm, report = tetrahedralize(mesh, edge_length, envelope, optimize)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tm.save(cache)
    return tm, report
