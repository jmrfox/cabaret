"""Port-level analysis: resistance distances, loopiness, rank bounds.

Ports are membrane patches (synapses, necks). These functions work on the
FEM reference model and answer two design questions:
- how far are the ports from a tree metric (could a tree reproduce them)?
- how many compartments are needed at least, for a given error?
"""

from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camovox.circuit import CompartmentGraph, PortSolver, attach_sink
from camovox.params import Passive


class ResistanceMetric:
    """Axial resistance distances between ports, with membrane removed.

    The network keeps only its axial couplings and is grounded at the
    ``ground`` patch (e.g. the neck) through a negligible resistance. It is
    factorized once, so :meth:`distances` is cheap for many site sets.

    Resistance distance $d_{ij} = R_{ii} + R_{jj} - 2 R_{ij}$, where R is the
    grounded resistance matrix, is a metric on the ports. The ground patch
    is included as port 0, with $d_{0i} = R_{ii}$.

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
        Only ``r_L`` is used.
    ground : (N,) array_like of float
        Weights of the grounded patch.
    ground_resistance : float, in Ω
        Grounding resistance. It must be tiny compared with port distances
        (MΩ in spines), and it is subtracted from $d_{0i}$.
    """

    def __init__(
        self,
        graph: CompartmentGraph,
        params: Passive,
        ground: ArrayLike,
        ground_resistance: float = 1.0,
    ):
        axial = replace(
            graph, area=np.zeros_like(graph.area), g_ground=None, c_ground=None
        )
        self._solver = PortSolver(attach_sink(axial, ground, ground_resistance), params)
        self._rg = ground_resistance

    def distances(self, patches: Sequence[ArrayLike]) -> NDArray[np.float64]:
        """Distance matrix for a set of port patches.

        Parameters
        ----------
        patches : sequence of P (N,) array_like of float

        Returns
        -------
        (P + 1, P + 1) ndarray of float, in Ω
            Port 0 is the ground patch.
        """
        r = self._solver.impedance(list(patches))
        r = 0.5 * (r + r.T)
        diag = np.diag(r)
        out = np.zeros((len(r) + 1, len(r) + 1))
        out[1:, 1:] = diag[:, None] + diag[None, :] - 2 * r
        out[0, 1:] = out[1:, 0] = diag - self._rg
        return np.clip(out, 0.0, None)


def resistance_distances(
    graph: CompartmentGraph,
    params: Passive,
    patches: Sequence[ArrayLike],
    ground: ArrayLike,
    ground_resistance: float = 1.0,
) -> NDArray[np.float64]:
    """One-off :class:`ResistanceMetric` distances (see there).

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
    patches : sequence of P (N,) array_like of float
    ground : (N,) array_like of float
    ground_resistance : float, in Ω

    Returns
    -------
    (P + 1, P + 1) ndarray of float, in Ω
    """
    metric = ResistanceMetric(graph, params, ground, ground_resistance)
    return metric.distances(patches)


def four_point_violation(d: ArrayLike, normalize: str = "max") -> NDArray[np.float64]:
    """Four-point violation of every port quadruple.

    For ports i, j, k, l form the three pair sums ``d_ij + d_kl``,
    ``d_ik + d_jl`` and ``d_il + d_jk`` and sort them, ``a >= b >= c``. A
    metric is a tree metric (realizable by a tree with internal nodes) iff
    ``a == b`` for every quadruple (Buneman 1974).

    Parameters
    ----------
    d : (P, P) array_like of float
        Symmetric distance matrix.
    normalize : {"max", "gap", "none"}
        - ``"max"``: ``(a - b) / a``. This is the relative error scale a tree
          must accept on these distances. It is 0 for trees and 0.25 for
          four points evenly spaced on an ideal ring. Port self-terms (patch
          spreading resistance) add to every distance and dilute it.
        - ``"gap"``: ``(a - b) / (a - c)``. This is invariant to adding a
          leaf resistance to any port, so it measures intrinsic topology:
          0 for trees, 1 for an ideal 4-port ring. It is unstable when all
          three sums are nearly equal.
        - ``"none"``: ``a - b``, in the units of ``d``. This is also
          leaf-invariant.

    Returns
    -------
    (C(P, 4),) ndarray of float
        One value per quadruple.
    """
    d = np.asarray(d, dtype=float)
    q = np.array(list(combinations(range(len(d)), 4)))
    if len(q) == 0:
        return np.zeros(0)
    i, j, k, m = q.T
    sums = np.stack([d[i, j] + d[k, m], d[i, k] + d[j, m], d[i, m] + d[j, k]], axis=1)
    sums.sort(axis=1)
    a, b, c = sums[:, 2], sums[:, 1], sums[:, 0]
    if normalize == "none":
        return a - b
    den = {"max": a, "gap": a - c}[normalize]
    return np.where(den > 0, (a - b) / np.where(den > 0, den, 1.0), 0.0)


def worst_quadruples(
    d: ArrayLike, k: int = 1, normalize: str = "max"
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Find the port quadruples with the largest four-point violation.

    Parameters
    ----------
    d : (P, P) array_like of float
    k : int
        Number of quadruples to return.
    normalize : {"max", "gap", "none"}
        See :func:`four_point_violation`.

    Returns
    -------
    (k, 4) ndarray of int
        Port indices (rows of ``d``).
    (k,) ndarray of float
        Their violations, largest first.
    """
    v = four_point_violation(d, normalize)
    q = np.array(list(combinations(range(len(np.asarray(d))), 4)))
    top = np.argsort(v)[::-1][:k]
    return q[top], v[top]


def loopiness_index(d: ArrayLike, normalize: str = "max") -> dict[str, float]:
    """Summary of the four-point violation over all quadruples.

    Parameters
    ----------
    d : (P, P) array_like of float
    normalize : {"max", "gap", "none"}
        See :func:`four_point_violation`.

    Returns
    -------
    dict
        ``median``, ``p90``, ``max`` and ``mean`` of the violation.
    """
    v = four_point_violation(d, normalize)
    if len(v) == 0:
        return {"median": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "median": float(np.median(v)),
        "p90": float(np.quantile(v, 0.9)),
        "max": float(v.max()),
        "mean": float(v.mean()),
    }


def clamp_port(z: ArrayLike, k: int) -> NDArray:
    """Port matrix with port ``k`` voltage-clamped to rest (Schur complement).

    ``Z_c = Z_AA - Z_Ak Z_kA / Z_kk`` over the other ports A. With ``k`` the
    neck, this removes the parent's shared contribution and leaves the
    spine's own internal port structure.

    Parameters
    ----------
    z : (P, P) array_like of float or complex
    k : int
        Index of the port to clamp.

    Returns
    -------
    (P - 1, P - 1) ndarray
    """
    z = np.asarray(z)
    keep = np.delete(np.arange(len(z)), k)
    return z[np.ix_(keep, keep)] - np.outer(z[keep, k], z[k, keep]) / z[k, k]


def rank_bound(z: ArrayLike) -> NDArray[np.float64]:
    """Relative error of the best rank-m approximation of a port matrix.

    If every port of a reduced model sits on one of m compartments, its
    port matrix has rank at most m. By Eckart–Young, ``err[m]`` is then a
    lower bound on its relative (Frobenius) error against ``z``, whatever
    the model's configuration.

    Parameters
    ----------
    z : (P, P) array_like of float or complex
        Port impedance matrix (e.g. DC).

    Returns
    -------
    (P + 1,) ndarray of float
        ``err[m]`` for m = 0 … P (``err[0] = 1``, ``err[P] = 0``).
    """
    s = np.linalg.svd(np.asarray(z), compute_uv=False)
    tail = np.sqrt(np.cumsum((s**2)[::-1])[::-1])  # tail[m] = ||s[m:]||
    return np.append(tail / tail[0], 0.0)


def coupling_coefficients(z: ArrayLike) -> NDArray[np.float64]:
    """Compute normalized port coupling, ``Z_ij / sqrt(Z_ii Z_jj)``.

    For a passive network this lies in [0, 1]. It is 1 for two ports that
    always see the same voltage (one compartment), and 0 for ports that don't
    influence each other at all.

    Parameters
    ----------
    z : (P, P) array_like of float
        Real, symmetric port matrix (e.g. DC, neck clamped).

    Returns
    -------
    (P, P) ndarray of float
    """
    z = np.asarray(z, dtype=float)
    d = np.sqrt(np.diag(z))
    return z / np.outer(d, d)


def electrical_groups(z: ArrayLike, n_groups: int) -> NDArray[np.int64]:
    """Split ports into groups that are strongly coupled within, weakly across.

    Average-linkage hierarchical clustering on the distance
    ``1 - coupling_coefficients(z)``.

    Parameters
    ----------
    z : (P, P) array_like of float
    n_groups : int

    Returns
    -------
    (P,) ndarray of int
        Group label per port, 0 … n_groups - 1, ordered by first port.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - coupling_coefficients(z)
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    lab = fcluster(
        linkage(squareform(dist, checks=False), "average"), n_groups, "maxclust"
    )
    _, first = np.unique(lab, return_index=True)
    order = {old: new for new, old in enumerate(lab[np.sort(first)])}
    return np.array([order[x] for x in lab], dtype=np.int64)
