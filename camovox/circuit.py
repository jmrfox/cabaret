"""Passive electrical networks and their solvers.

A :class:`CompartmentGraph` is any network of nodes carrying membrane area,
coupled by edges carrying a geometric conductance factor κ (µm). The FEM
reference model and effective cable models are both compartment graphs, so
they are solved by the same functions here.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import ArrayLike, NDArray

from camovox.params import Passive


@dataclass(frozen=True)
class CompartmentGraph:
    """A passive electrical network.

    Parameters
    ----------
    area : (N,) ndarray of float, in µm²
        Membrane area carried by each node. Interior nodes have zero area.
    position : (N, 3) ndarray of float, in µm
        Node positions (used for site placement and diagnostics only).
    edges : (E, 2) ndarray of int
        Node index pairs, each undirected edge listed once.
    kappa : (E,) ndarray of float, in µm
        Geometric conductance factor of each edge, so that the axial
        conductance is ``kappa / r_L``.
    """

    area: NDArray[np.float64]
    position: NDArray[np.float64]
    edges: NDArray[np.int64]
    kappa: NDArray[np.float64]

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.area)

    def laplacian(self, params: Passive) -> sp.csc_matrix:
        """Axial conductance Laplacian.

        Parameters
        ----------
        params : Passive

        Returns
        -------
        (N, N) scipy.sparse.csc_matrix, in S
        """
        g = params.axial_conductance(self.kappa)
        i, j = self.edges[:, 0], self.edges[:, 1]
        n = self.n_nodes
        off = sp.coo_matrix((-g, (i, j)), shape=(n, n))
        off = off + off.T
        diag = -np.asarray(off.sum(axis=1)).ravel()
        return (off + sp.diags(diag)).tocsc()

    def system_matrix(self, params: Passive, omega: float = 0.0) -> sp.csc_matrix:
        """Admittance matrix ``L + G_m + i ω C``.

        Parameters
        ----------
        params : Passive
        omega : float
            Angular frequency, in rad/s. Zero gives the real DC matrix.

        Returns
        -------
        (N, N) scipy.sparse.csc_matrix, in S
        """
        a = self.laplacian(params) + sp.diags(params.membrane_conductance(self.area))
        if omega != 0.0:
            a = a + 1j * omega * sp.diags(params.capacitance(self.area))
        return a.tocsc()


def _normalize(weights: ArrayLike, n: int) -> NDArray[np.float64]:
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    return w / total


def transfer_impedance(
    graph: CompartmentGraph,
    params: Passive,
    w_in: ArrayLike,
    w_out: ArrayLike | None = None,
    freqs: ArrayLike = 0.0,
) -> NDArray[np.complex128]:
    """Transfer impedance between two membrane patches.

    Unit current is injected, distributed over nodes in proportion to
    ``w_in``, and the voltage is read as the ``w_out``-weighted average.

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
    w_in : (N,) array_like of float
        Injection weights (e.g. patch membrane areas). Normalized to sum 1.
    w_out : (N,) array_like of float, optional
        Readout weights. Defaults to ``w_in`` (input impedance).
    freqs : float or array_like of float, in Hz
        Frequencies. 0 gives the DC resistance.

    Returns
    -------
    ndarray of complex, in Ω
        One value per frequency (shape of ``freqs``).
    """
    n = graph.n_nodes
    win = _normalize(w_in, n)
    wout = win if w_out is None else _normalize(w_out, n)
    f = np.atleast_1d(np.asarray(freqs, dtype=float))
    out = np.empty(f.shape, dtype=complex)
    for k, fk in enumerate(f):
        a = graph.system_matrix(params, omega=2 * np.pi * fk)
        v = spla.splu(a).solve(win.astype(a.dtype))
        out[k] = wout @ v
    return out.reshape(np.shape(freqs))


def input_resistance(graph: CompartmentGraph, params: Passive, w: ArrayLike) -> float:
    """DC input resistance of a membrane patch.

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
    w : (N,) array_like of float
        Patch weights. Normalized to sum 1.

    Returns
    -------
    float, in Ω
    """
    return float(transfer_impedance(graph, params, w).real)


def time_constants(graph: CompartmentGraph, params: Passive, k: int = 3) -> NDArray:
    """Slowest passive time constants.

    Solves the generalized eigenproblem ``(L + G_m) v = μ C v`` for the
    smallest μ, using shift-invert at zero. Interior nodes carry no
    capacitance, so ``C`` is singular. Their modes have infinite μ and are
    never returned.

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
    k : int
        Number of time constants.

    Returns
    -------
    (k,) ndarray of float, in s
        Sorted from slowest to fastest.
    """
    a = graph.system_matrix(params)
    c = params.capacitance(graph.area)
    lu = spla.splu(a)
    op = spla.LinearOperator(a.shape, matvec=lambda x: lu.solve(c * x), dtype=float)
    vals = spla.eigs(op, k=k, which="LM", return_eigenvectors=False)
    tau = np.sort(np.abs(vals.real))[::-1]
    return tau
