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

from camovox.circuit import CompartmentGraph, attach_sink, port_impedance
from camovox.params import Passive


def resistance_distances(
    graph: CompartmentGraph,
    params: Passive,
    patches: Sequence[ArrayLike],
    ground: ArrayLike,
    ground_resistance: float = 1.0,
) -> NDArray[np.float64]:
    """Axial resistance distances between ports, with membrane removed.

    The network is grounded at the ``ground`` patch (e.g. the neck) through
    a negligible resistance. The ground patch is returned as port 0.
    Resistance distance $d_{ij} = R_{ii} + R_{jj} - 2 R_{ij}$ is a metric, and
    $d_{0i} = R_{ii}$.

    Parameters
    ----------
    graph : CompartmentGraph
    params : Passive
        Only ``r_L`` is used.
    patches : sequence of P (N,) array_like of float
        Port weights.
    ground : (N,) array_like of float
        Weights of the grounded patch.
    ground_resistance : float, in Ω
        Grounding resistance. It must be tiny compared with port distances
        (MΩ in spines).

    Returns
    -------
    (P + 1, P + 1) ndarray of float, in Ω
    """
    axial = replace(graph, area=np.zeros_like(graph.area), g_ground=None, c_ground=None)
    grounded = attach_sink(axial, ground, ground_resistance)
    r = port_impedance(grounded, params, list(patches), 0.0)[0].real
    r = 0.5 * (r + r.T)
    diag = np.diag(r)
    d = diag[:, None] + diag[None, :] - 2 * r
    out = np.zeros((len(r) + 1, len(r) + 1))
    out[1:, 1:] = d
    out[0, 1:] = out[1:, 0] = diag - ground_resistance
    return np.clip(out, 0.0, None)


def four_point_violation(d: ArrayLike) -> NDArray[np.float64]:
    """Relative four-point violation of every port quadruple.

    For ports i, j, k, l form the three pair sums ``d_ij + d_kl``,
    ``d_ik + d_jl`` and ``d_il + d_jk`` and sort them, ``a >= b >= c``. A
    metric is a tree metric (realizable by a tree with internal nodes) iff
    ``a == b`` for every quadruple (Buneman 1974). The violation is
    ``(a - b) / a``, which is 0 for trees and 0.25 for a ring of four equally
    spaced ports.

    Parameters
    ----------
    d : (P, P) array_like of float
        Symmetric distance matrix.

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
    a, b = sums[:, 2], sums[:, 1]
    return np.where(a > 0, (a - b) / np.where(a > 0, a, 1.0), 0.0)


def loopiness_index(d: ArrayLike) -> dict[str, float]:
    """Summary of the four-point violation over all quadruples.

    Parameters
    ----------
    d : (P, P) array_like of float

    Returns
    -------
    dict
        ``median``, ``p90``, ``max`` and ``mean`` of the violation.
    """
    v = four_point_violation(d)
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
