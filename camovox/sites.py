"""Membrane sites (input/output patches).

PLACEHOLDER: this module only offers Euclidean ball patches. The design
calls for geodesic patches grown to a PSD area, a ``SiteSet`` table and
samplers. Those come in a later plan.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camovox.circuit import CompartmentGraph


def ball_patch(
    graph: CompartmentGraph, center: ArrayLike, radius: float
) -> NDArray[np.float64]:
    """Membrane weights of the nodes inside a ball.

    Parameters
    ----------
    graph : CompartmentGraph
    center : (3,) array_like of float, in µm
    radius : float, in µm

    Returns
    -------
    (N,) ndarray of float, in µm²
        Membrane area of each node inside the ball, zero elsewhere.

    Raises
    ------
    ValueError
        If the ball contains no membrane.
    """
    d = np.linalg.norm(graph.position - np.asarray(center, dtype=float), axis=1)
    w = np.where(d <= radius, graph.area, 0.0)
    if w.sum() <= 0:
        raise ValueError("ball contains no membrane nodes")
    return w
