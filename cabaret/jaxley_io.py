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
"""

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
    """

    cell: object
    branch_of_node: NDArray[np.int64]
    port_branch: NDArray[np.int64]


def to_jaxley(
    tree: PortTree,
    node_area: ArrayLike,
    params: Passive,
    root_port: int = 0,
    min_area: float = 1e-4,
    branchpoint_load: float = 1e-6,
    root_shunt: float | None = None,
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

    Returns
    -------
    JaxleyModel

    Examples
    --------
    >>> jm = to_jaxley(fit.tree, fit.node_area, MEMBRANE)  # doctest: +SKIP
    >>> jm.cell.branch(int(jm.port_branch[3])).comp(0).record("v")  # doctest: +SKIP
    """
    import jaxley as jx
    from jaxley.channels import Leak

    area = np.asarray(node_area, dtype=float)
    g = tree.graph()
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
    branch_of_node = np.array(
        [branch_of[v] for v in range(tree.n_nodes)], dtype=np.int64
    )
    return JaxleyModel(cell, branch_of_node, branch_of_node[tree.port_node])
