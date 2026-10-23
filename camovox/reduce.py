"""Reduced models by tree realization.

A spine's ports (synapses + neck) are related by resistance distances (the
resistance between two ports with membrane removed). If those distances are
close to a **tree metric**, a tree of resistors with internal branch points
reproduces them (see "Loops without loops" in ``docs/design.md``). This
module:

1. fits such a tree: neighbor-joining topology, then non-negative branch
   resistances by least squares;
2. simplifies it by merging short branches, which gives smaller models;
3. puts the FEM membrane onto the tree's nodes;
4. turns the result into a :class:`~camovox.circuit.CompartmentGraph` that
   the same solvers can evaluate against the FEM.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import nnls

from camovox.circuit import CompartmentGraph, PortSolver
from camovox.params import UM_TO_CM, Passive


@dataclass(frozen=True)
class PortTree:
    """A tree of resistors whose leaves (or nodes) carry the ports.

    Parameters
    ----------
    n_nodes : int
        Number of tree nodes.
    edges : (E, 2) ndarray of int
        Node pairs.
    resistance : (E,) ndarray of float, in Ω
        Branch resistances (axial, membrane excluded).
    port_node : (P,) ndarray of int
        Tree node that carries each port, in port order.
    fit_error : float
        Relative error of the tree's port distances against the target
        distances (Frobenius norm over all port pairs).
    """

    n_nodes: int
    edges: NDArray[np.int64]
    resistance: NDArray[np.float64]
    port_node: NDArray[np.int64]
    fit_error: float

    def graph(self) -> nx.Graph:
        """Return the tree as a networkx graph with ``resistance`` edge attributes."""
        g = nx.Graph()
        g.add_nodes_from(range(self.n_nodes))
        for (a, b), r in zip(self.edges, self.resistance, strict=True):
            g.add_edge(int(a), int(b), resistance=float(r))
        return g

    def distances(self) -> NDArray[np.float64]:
        """Port-to-port path resistances implied by the tree, in Ω."""
        lengths = dict(
            nx.all_pairs_dijkstra_path_length(self.graph(), weight="resistance")
        )
        p = len(self.port_node)
        d = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                d[i, j] = lengths[int(self.port_node[i])][int(self.port_node[j])]
        return d


def _nj_topology(d: NDArray[np.float64]) -> tuple[int, list[tuple[int, int]]]:
    """Neighbor-joining topology. Ports are nodes 0…P−1, internal nodes after."""
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

    p = len(d)
    names = [f"p{i}" for i in range(p)]
    lower = [[float(d[i, j]) for j in range(i + 1)] for i in range(p)]
    tree = DistanceTreeConstructor().nj(DistanceMatrix(names, lower))
    ids: dict[int, int] = {}
    next_id = p

    def node_id(clade) -> int:
        nonlocal next_id
        if clade.name in names:
            return names.index(clade.name)
        if id(clade) not in ids:
            ids[id(clade)] = next_id
            next_id += 1
        return ids[id(clade)]

    edges = []
    for parent in tree.find_clades(order="level"):
        for child in parent.clades:
            edges.append((node_id(parent), node_id(child)))
    return next_id, edges


def fit_tree(d: ArrayLike) -> PortTree:
    """Fit a tree of resistors to port resistance distances.

    The topology comes from neighbor-joining (Saitou & Nei 1987). Branch
    resistances are then refitted by non-negative least squares on all port
    pairs, so no branch has a negative resistance.

    Parameters
    ----------
    d : (P, P) array_like of float, in Ω
        Symmetric port distance matrix (e.g. from
        :class:`camovox.ports.ResistanceMetric`, neck = port 0).

    Returns
    -------
    PortTree
        Ports are leaves 0…P−1, in the order of ``d``.

    Examples
    --------
    >>> tree = fit_tree(metric.distances(az.patches))  # doctest: +SKIP
    >>> tree.fit_error  # doctest: +SKIP
    """
    d = np.asarray(d, dtype=float)
    d = 0.5 * (d + d.T)
    p = len(d)
    n_nodes, edges = _nj_topology(d)
    g = nx.Graph(edges)
    edge_index = {frozenset(e): k for k, e in enumerate(edges)}
    rows, target = [], []
    for i in range(p):
        for j in range(i + 1, p):
            path = nx.shortest_path(g, i, j)
            row = np.zeros(len(edges))
            for a, b in zip(path[:-1], path[1:], strict=True):
                row[edge_index[frozenset((a, b))]] = 1.0
            rows.append(row)
            target.append(d[i, j])
    a_mat, target = np.array(rows), np.array(target)
    resistance, _ = nnls(a_mat, target)
    fit_err = np.linalg.norm(a_mat @ resistance - target) / np.linalg.norm(target)
    return PortTree(
        n_nodes=n_nodes,
        edges=np.array(edges, dtype=np.int64),
        resistance=resistance,
        port_node=np.arange(p, dtype=np.int64),
        fit_error=float(fit_err),
    )


def contract_tree(tree: PortTree, min_resistance: float) -> PortTree:
    """Merge every branch shorter than ``min_resistance`` into one node.

    This gives a smaller model: ports on merged nodes share a compartment.
    Distances change by at most the merged resistances.

    Parameters
    ----------
    tree : PortTree
    min_resistance : float, in Ω

    Returns
    -------
    PortTree
        Nodes renumbered 0…n−1. ``fit_error`` is carried over from the
        input; evaluate the contracted tree directly to see its accuracy.
    """
    g = tree.graph()
    short = nx.Graph()
    short.add_nodes_from(g.nodes)
    short.add_edges_from(
        (a, b) for a, b, r in g.edges(data="resistance") if r < min_resistance
    )
    group = {}
    for k, comp in enumerate(nx.connected_components(short)):
        for v in comp:
            group[v] = k
    new_edges, new_r = {}, {}
    for a, b, r in g.edges(data="resistance"):
        ga, gb = group[a], group[b]
        if ga == gb:
            continue
        key = (min(ga, gb), max(ga, gb))
        new_edges[key] = key
        new_r[key] = r
    keys = sorted(new_edges)
    return PortTree(
        n_nodes=len(set(group.values())),
        edges=np.array(keys, dtype=np.int64).reshape(-1, 2),
        resistance=np.array([new_r[k] for k in keys]),
        port_node=np.array([group[int(v)] for v in tree.port_node], dtype=np.int64),
        fit_error=tree.fit_error,
    )


def assign_membrane(
    graph: CompartmentGraph,
    params: Passive,
    patches: Sequence[ArrayLike],
    ground: ArrayLike,
) -> NDArray[np.int64]:
    """Assign each membrane vertex to the port it behaves most like.

    With the ``ground`` port held at rest, current is injected at each other
    port in turn. A vertex's "fingerprint" is its voltages across these
    injections. Each port's own fingerprint is its row of the port matrix,
    and the ground port's fingerprint is all zeros. Each vertex goes to the
    port with the nearest fingerprint, so membrane lands on the compartment
    that will see the same voltages.

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
    patches : sequence of P (N,) array_like of float
        Non-ground port weights.
    ground : (N,) array_like of float
        Weights of the ground port (e.g. the neck electrode).

    Returns
    -------
    (N,) ndarray of int
        For every node: 0 = ground port, k = patch k−1, −1 = no membrane.
    """
    from camovox.circuit import _normalize, attach_sink

    clamped = attach_sink(graph, ground, resistance=1.0)
    solver = PortSolver(clamped, params)
    w = np.column_stack([_normalize(p, graph.n_nodes) for p in patches])
    v = solver._lu.solve(w)  # (N, P): voltage at every node per injection
    z = w.T @ v  # (P, P) port fingerprints
    finger = np.vstack([np.zeros(len(patches)), z])  # ground port first
    has_membrane = graph.area > 0
    out = np.full(graph.n_nodes, -1, dtype=np.int64)
    vm = v[has_membrane]
    d2 = (vm**2).sum(1)[:, None] - 2 * vm @ finger.T + (finger**2).sum(1)[None, :]
    out[has_membrane] = np.argmin(d2, axis=1)
    return out


def tree_to_graph(
    tree: PortTree,
    params: Passive,
    node_area: ArrayLike,
    port_position: ArrayLike | None = None,
) -> CompartmentGraph:
    """Build a compartment network from a resistor tree.

    Parameters
    ----------
    tree : PortTree
    params : Passive
        Only ``r_L`` is used. It converts each branch resistance into a
        geometric conductance factor κ, so the graph keeps the right
        resistances when solved with the same ``r_L``.
    node_area : (n_nodes,) array_like of float, in µm²
        Membrane on each tree node.
    port_position : (P, 3) array_like of float, optional, in µm
        Positions for port nodes, for plotting. Other nodes are placed at the
        mean of their neighbours.

    Returns
    -------
    CompartmentGraph
    """
    r = np.maximum(tree.resistance, 1e-3)  # a zero branch would be a short circuit
    kappa = params.r_L / (r * UM_TO_CM)  # µm, so that kappa * 1e-4 / r_L = 1 / r
    pos = np.zeros((tree.n_nodes, 3))
    if port_position is not None:
        pos[tree.port_node] = np.asarray(port_position, dtype=float)
        known = np.zeros(tree.n_nodes, dtype=bool)
        known[tree.port_node] = True
        g = tree.graph()
        for _ in range(tree.n_nodes):
            todo = [v for v in range(tree.n_nodes) if not known[v]]
            if not todo:
                break
            for v in todo:
                nb = [u for u in g.neighbors(v) if known[u]]
                if nb:
                    pos[v] = pos[nb].mean(axis=0)
                    known[v] = True
    return CompartmentGraph(
        area=np.asarray(node_area, dtype=float),
        position=pos,
        edges=tree.edges.astype(np.int64),
        kappa=kappa,
    )


def port_matrix_error(z_model: ArrayLike, z_ref: ArrayLike) -> float:
    """Relative Frobenius error of a port matrix against a reference.

    Parameters
    ----------
    z_model, z_ref : (P, P) array_like

    Returns
    -------
    float
    """
    z_model, z_ref = np.asarray(z_model), np.asarray(z_ref)
    return float(np.linalg.norm(z_model - z_ref) / np.linalg.norm(z_ref))
