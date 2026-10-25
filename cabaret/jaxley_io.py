"""Export tree models to jaxley.

Each tree node becomes a one-compartment jaxley branch, rooted at a chosen
port (the neck). The mapping is exact for a resistor tree:

- a branch's ``resistive_load_in`` carries the resistance of its edge to
  its parent. jaxley turns a resistive load into a resistance as
  load × r_L, so load = R / (r_L · 10⁴) with r_L in Ω·cm and load in 1/µm;
- ``resistive_load_out`` is set to a negligible load (default 1e-6, about
  1 Ω at r_L = 100 Ω·cm), so the branch point after each compartment
  coincides electrically with that compartment. Do **not** go much lower:
  at 1e-12 jaxley's solvers lose precision (+2.5% with ``jaxley.dhs``, and
  ``jax.sparse`` fails);
- membrane is set directly through ``area``. A node without membrane gets a
  tiny area, because jaxley divides axial terms by area.

Geometry (radius, length) is set to arbitrary values first, because setting
it recomputes area and loads. Never make radius or length trainable on the
exported cell.

Loops (jaxley 0.14, private API):

- jaxley has no public way to close a loop. :func:`close_loop` appends
  compartment-to-compartment rows to the private ``cell._comp_edges``.
- Such an edge has no conductance of its own: its resistance is the sum of
  a ``resistive_load_out`` and a ``resistive_load_in`` of its two ends. Each
  loop a–b therefore gets a dummy compartment D without membrane, a tree
  child of a with ``resistive_load_in`` = R/2, and the closing edge D–b is
  oriented to use ``load_in(D) + load_out(b)``. The path a → D → b is then R
  plus two branch-point loads (about 2 Ω).
- ``jaxley.dhs``, jaxley's default solver, assumes a tree and returns wrong
  voltages for a looped cell without warning. Simulate through
  :func:`integrate`, which uses ``jax.sparse`` for looped models.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from cabaret.params import UM_TO_CM, Passive
from cabaret.reduce import PortTree


@dataclass
class JaxleyModel:
    """A jaxley cell built from a tree, with the node ↔ branch mapping.

    Parameters
    ----------
    cell : jaxley.Cell
        One branch (one compartment) per tree node, with a passive ``Leak``
        channel.
    branch_of_node : (n,) ndarray of int
        Branch index of each tree node.
    port_branch : (P,) ndarray of int
        Branch index of each port.
    n_loops : int
        Number of loops closed with :func:`close_loop`. A cell with loops
        must be simulated with ``jax.sparse`` (see :func:`integrate`).
    """

    cell: object
    branch_of_node: NDArray[np.int64]
    port_branch: NDArray[np.int64]
    n_loops: int = 0


def close_loop(cell: object, dummy: int, other: int) -> None:
    """Join two compartments of a jaxley cell by an extra axial edge.

    The edge's resistance is ``resistive_load_in`` of ``dummy`` plus
    ``resistive_load_out`` of ``other`` (both times r_L). This uses the
    private ``cell._comp_edges`` table of jaxley 0.14, so call it after
    every other change to the cell's morphology.

    Parameters
    ----------
    cell : jaxley.Cell
        Modified in place.
    dummy : int
        Compartment index whose ``resistive_load_in`` carries the edge.
    other : int
        Compartment index at the other end; only its ``resistive_load_out``
        (normally a negligible branch-point load) enters the edge.
    """
    import pandas as pd

    # Both rows give load_out(other) + load_in(dummy): jaxley takes the
    # source's load_out when `ordered` is 1, its load_in when 0, and the
    # sink's other load.
    rows = pd.DataFrame(
        {
            "source": [other, dummy],
            "sink": [dummy, other],
            "ordered": [1, 0],
            "type": [0, 0],
        }
    )
    cell._comp_edges = pd.concat([cell._comp_edges, rows], ignore_index=True)
    cell._init_solvers()


def integrate(model: JaxleyModel, **kwargs: object) -> NDArray[np.float64]:
    """Run ``jaxley.integrate`` with a voltage solver that suits the model.

    Parameters
    ----------
    model : JaxleyModel
    **kwargs
        Passed to ``jaxley.integrate`` (e.g. ``delta_t``).

    Returns
    -------
    ndarray of float
        Recordings, as returned by jaxley.

    Raises
    ------
    ValueError
        If the model has loops and a voltage solver other than
        ``jax.sparse`` is requested.
    """
    import jaxley as jx

    if model.n_loops:
        solver = kwargs.setdefault("voltage_solver", "jax.sparse")
        if solver != "jax.sparse":
            raise ValueError(
                f"model has {model.n_loops} loop(s); voltage_solver must be "
                f"'jax.sparse', not {solver!r} (which assumes a tree)"
            )
    return np.asarray(jx.integrate(model.cell, **kwargs))


def to_jaxley(
    tree: PortTree,
    node_area: ArrayLike,
    params: Passive,
    root_port: int = 0,
    min_area: float = 1e-4,
    branchpoint_load: float = 1e-6,
    root_shunt: float | None = None,
    loops: Sequence[tuple[int, int, float]] = (),
) -> JaxleyModel:
    """Build a passive jaxley cell from a tree model.

    Parameters
    ----------
    tree : PortTree
    node_area : (n,) array_like of float, in µm²
    params : Passive
        Membrane and cytoplasm parameters (``Leak`` reversal at 0 mV, so
        voltages are deviations from rest).
    root_port : int
        Port placed at the root branch (the neck).
    min_area : float, in µm²
        Area given to nodes without membrane.
    branchpoint_load : float, in 1/µm
        ``resistive_load_out`` of every compartment (see module notes).
    root_shunt : float, optional, in Ω
        Extra resistance from the root to rest, added to its leak. A small
        value (e.g. 1 kΩ) holds the neck near rest, matching
        ``attach_sink(..., resistance=root_shunt)`` on the cabaret side.
    loops : sequence of (int, int, float)
        Extra edges ``(node_a, node_b, resistance in Ω)`` that close loops
        on the tree. Each adds a dummy branch without membrane (see module
        notes). Simulate the result with :func:`integrate`.

    Returns
    -------
    JaxleyModel

    Examples
    --------
    >>> jm = to_jaxley(fit.tree, fit.node_area, MEMBRANE)  # doctest: +SKIP
    >>> jm.cell.branch(int(jm.port_branch[3])).comp(0).record("v")  # doctest: +SKIP
    >>> v = integrate(jm, delta_t=0.025)  # doctest: +SKIP
    """
    import jaxley as jx
    from jaxley.channels import Leak

    g = tree.graph()
    area = np.asarray(node_area, dtype=float)
    dummies = []
    for k, (a, b, r) in enumerate(loops):
        d = tree.n_nodes + k  # dummy leaf under a; half of R on each side
        g.add_edge(int(a), d, resistance=r / 2)
        dummies.append((d, int(b)))
    area = np.concatenate([area, np.zeros(len(dummies))])
    root = int(tree.port_node[root_port])
    bfs = nx.bfs_tree(g, root)
    order = list(nx.bfs_tree(g, root))
    branch_of = {v: k for k, v in enumerate(order)}
    parents = [
        -1 if v == root else branch_of[next(iter(bfs.predecessors(v)))] for v in order
    ]
    cell = jx.Cell([jx.Branch(jx.Compartment(), ncomp=1)] * len(order), parents=parents)
    cell.insert(Leak())
    cell.set("Leak_gLeak", params.g_pas)
    cell.set("Leak_eLeak", 0.0)
    cell.set("v", 0.0)
    cell.set("capacitance", params.c_m)
    cell.set("axial_resistivity", params.r_L)
    cell.set("radius", 1.0)  # arbitrary; area and loads are overwritten below
    cell.set("length", 1.0)
    for v in order:
        b = cell.branch(branch_of[v])
        b.set("area", max(area[v], min_area))
        r_parent = (
            0.0
            if v == root
            else g.edges[v, next(iter(bfs.predecessors(v)))]["resistance"]
        )
        b.set("resistive_load_in", max(r_parent, 1.0) * UM_TO_CM / params.r_L)
        b.set("resistive_load_out", branchpoint_load)
    if root_shunt is not None:
        a_root = max(area[root], min_area)
        g_extra = 1.0 / (root_shunt * a_root * 1e-8)  # S/cm² over the root's area
        cell.branch(branch_of[root]).set("Leak_gLeak", params.g_pas + g_extra)
    # One compartment per branch, so compartment index == branch index.
    for d, b in dummies:
        close_loop(cell, dummy=branch_of[d], other=branch_of[b])
    branch_of_node = np.array(
        [branch_of[v] for v in range(tree.n_nodes)], dtype=np.int64
    )
    return JaxleyModel(
        cell, branch_of_node, branch_of_node[tree.port_node], n_loops=len(dummies)
    )
