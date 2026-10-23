"""Passive electrical networks and their solvers.

A :class:`CompartmentGraph` is any network of nodes carrying membrane area,
coupled by edges carrying a geometric conductance factor κ (µm). The FEM
reference model and effective cable models are both compartment graphs, so
they are solved by the same functions here.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

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
    g_ground : (N,) ndarray of float, in S, optional
        Extra conductance from each node to ground (rest), independent of
        the membrane parameters. Used for boundary conditions such as the
        parent a spine neck attaches to.
    c_ground : (N,) ndarray of float, in F, optional
        Extra capacitance from each node to ground.
    """

    area: NDArray[np.float64]
    position: NDArray[np.float64]
    edges: NDArray[np.int64]
    kappa: NDArray[np.float64]
    g_ground: NDArray[np.float64] | None = None
    c_ground: NDArray[np.float64] | None = None

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
        g = params.membrane_conductance(self.area)
        if self.g_ground is not None:
            g = g + self.g_ground
        a = self.laplacian(params) + sp.diags(g)
        if omega != 0.0:
            a = a + 1j * omega * sp.diags(self.capacitance(params))
        return a.tocsc()

    def capacitance(self, params: Passive) -> NDArray[np.float64]:
        """Total capacitance to ground of each node (membrane + extra).

        Parameters
        ----------
        params : Passive

        Returns
        -------
        (N,) ndarray of float, in F
        """
        c = params.capacitance(self.area)
        return c if self.c_ground is None else c + self.c_ground


def attach_sink(
    graph: CompartmentGraph,
    weights: ArrayLike,
    resistance: float,
    capacitance: float = 0.0,
) -> CompartmentGraph:
    """Connect a patch to a lumped parent compartment held at rest.

    The patch is joined to ground through ``resistance`` in parallel with
    ``capacitance``, distributed over the patch nodes in proportion to
    ``weights``. This models a spine neck joining a large parent (soma or
    dendrite) whose own input impedance is given.

    Parameters
    ----------
    graph : CompartmentGraph
    weights : (N,) array_like of float
        Patch weights (e.g. from a neck-point patch). Normalized to sum 1.
    resistance : float, in Ω
        Input resistance of the parent seen from the neck.
    capacitance : float, in F
        Capacitance of the parent (0 for a purely resistive sink).

    Returns
    -------
    CompartmentGraph
        A copy with ``g_ground`` and ``c_ground`` increased.
    """
    w = _normalize(weights, graph.n_nodes)
    g0 = np.zeros(graph.n_nodes) if graph.g_ground is None else graph.g_ground
    c0 = np.zeros(graph.n_nodes) if graph.c_ground is None else graph.c_ground
    return replace(graph, g_ground=g0 + w / resistance, c_ground=c0 + w * capacitance)


def _normalize(weights: ArrayLike, n: int) -> NDArray[np.float64]:
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    return w / total


def solve_dc(
    graph: CompartmentGraph, params: Passive, injection: ArrayLike
) -> NDArray[np.float64]:
    """Steady-state voltage field for a current injection.

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
    injection : (N,) array_like of float, in A
        Current injected at each node (positive = into the cell).

    Returns
    -------
    (N,) ndarray of float, in V
        Voltage relative to rest at every node.

    Examples
    --------
    >>> w = ball_patch(ref, center, 0.5)  # doctest: +SKIP
    >>> v = solve_dc(ref, Passive(), 1e-11 * w / w.sum())  # 10 pA  # doctest: +SKIP
    """
    i = np.asarray(injection, dtype=float)
    if i.shape != (graph.n_nodes,):
        raise ValueError(f"injection must have shape ({graph.n_nodes},)")
    return spla.splu(graph.system_matrix(params)).solve(i)


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
    c = graph.capacitance(params)
    lu = spla.splu(a)
    op = spla.LinearOperator(a.shape, matvec=lambda x: lu.solve(c * x), dtype=float)
    vals = spla.eigs(op, k=k, which="LM", return_eigenvectors=False)
    tau = np.sort(np.abs(vals.real))[::-1]
    return tau


def port_impedance(
    graph: CompartmentGraph,
    params: Passive,
    patches: Sequence[ArrayLike],
    freqs: ArrayLike = 0.0,
) -> NDArray[np.complex128]:
    """Impedance matrix between membrane patches (ports).

    ``Z[k, i, j]`` is the patch-averaged voltage at port ``i`` per unit
    current injected into port ``j`` at frequency ``freqs[k]``. Each
    frequency costs one sparse factorization, and all ports share it.

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
    patches : sequence of P (N,) array_like of float
        Port weights. Each is normalized to sum 1.
    freqs : float or array_like of float, in Hz

    Returns
    -------
    (F, P, P) ndarray of complex, in Ω
        Symmetric in the last two axes.
    """
    f = np.atleast_1d(np.asarray(freqs, dtype=float))
    z = [PortSolver(graph, params, fk).impedance(patches) for fk in f]
    return np.stack(z).astype(complex)


class PortSolver:
    """Factorize a network once, then compute port matrices for many patch sets.

    Use this when the same network is probed with many site sets (null
    models, jitter tests): each new set costs only back-substitutions.

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
    freq : float, in Hz
        Frequency of the factorized admittance matrix (0 for DC).
    """

    def __init__(self, graph: CompartmentGraph, params: Passive, freq: float = 0.0):
        self.graph = graph
        a = graph.system_matrix(params, omega=2 * np.pi * freq)
        self._dtype = a.dtype
        self._lu = spla.splu(a)

    def impedance(self, patches: Sequence[ArrayLike]) -> NDArray:
        """Port matrix for a set of patches.

        Parameters
        ----------
        patches : sequence of P (N,) array_like of float
            Port weights, each normalized to sum 1.

        Returns
        -------
        (P, P) ndarray, in Ω
            Real at DC, complex otherwise.
        """
        w = np.column_stack([_normalize(p, self.graph.n_nodes) for p in patches])
        v = self._lu.solve(w.astype(self._dtype))
        z = w.T @ v
        return z.real if self._dtype.kind == "f" else z


def add_electrode(
    graph: CompartmentGraph,
    nodes: ArrayLike,
    kappa: float = 1e4,
    remove_membrane: bool = True,
) -> tuple[CompartmentGraph, int]:
    """Tie a set of nodes into one equipotential port.

    A new node is added and joined to every listed node by a very large
    coupling, so the set behaves as a perfect conductor (an electrode).
    Use this for a spine's attachment to its parent: the tied region stands
    for parent cytoplasm, which is nearly isopotential at spine scale.

    Parameters
    ----------
    graph : CompartmentGraph
    nodes : (K,) array_like of int
        Node indices to tie together.
    kappa : float, in µm
        Coupling of each node to the electrode. It must be much larger than
        the network's own couplings (typically below 1 µm).
    remove_membrane : bool
        Set the tied nodes' membrane area to zero (and drop their ground
        shunts). Their membrane is not real: it lies inside or against the
        parent.

    Returns
    -------
    CompartmentGraph
        The network with one extra node (index ``graph.n_nodes``).
    int
        Index of the electrode node. Use a one-hot weight vector on it as a
        port.
    """
    idx = np.unique(np.asarray(nodes, dtype=np.int64))
    if idx.size == 0:
        raise ValueError("electrode needs at least one node")
    n = graph.n_nodes
    area = np.append(graph.area, 0.0)
    g0 = None if graph.g_ground is None else np.append(graph.g_ground, 0.0)
    c0 = None if graph.c_ground is None else np.append(graph.c_ground, 0.0)
    if remove_membrane:
        area[idx] = 0.0
        if g0 is not None:
            g0[idx] = 0.0
        if c0 is not None:
            c0[idx] = 0.0
    new_edges = np.column_stack([idx, np.full(idx.size, n)])
    out = CompartmentGraph(
        area=area,
        position=np.vstack([graph.position, graph.position[idx].mean(axis=0)]),
        edges=np.vstack([graph.edges, new_edges]),
        kappa=np.concatenate([graph.kappa, np.full(idx.size, float(kappa))]),
        g_ground=g0,
        c_ground=c0,
    )
    return out, n


def point_port(graph: CompartmentGraph, node: int) -> NDArray[np.float64]:
    """Port weights for a single node (e.g. an electrode from :func:`add_electrode`).

    Parameters
    ----------
    graph : CompartmentGraph
    node : int

    Returns
    -------
    (N,) ndarray of float
        1 at ``node``, 0 elsewhere.
    """
    w = np.zeros(graph.n_nodes)
    w[node] = 1.0
    return w
