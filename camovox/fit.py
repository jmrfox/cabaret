"""Gradient-based fitting of tree models to reference port matrices (JAX).

A tree model (see :mod:`camovox.reduce`) has a resistance per branch and a
membrane area per node. This module tunes both so the model's port
impedance matrices match the FEM reference at a few frequencies.

The solver is a small dense one written in JAX, so gradients are exact.
Reduced models have tens of nodes, so dense solves are cheap. Membrane
areas are parameterized as total × softmax, so the model's **total
membrane stays exactly equal** to the reference's. That matters because
channel densities will later be specified per membrane area.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camovox.params import UM2_TO_CM2, Passive
from camovox.reduce import PortTree


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    return jax, jnp


def port_impedance_jax(
    edges: NDArray[np.int64],
    resistance,
    area,
    port_nodes: NDArray[np.int64],
    params: Passive,
    freqs: Sequence[float],
):
    """Port impedance matrices of a small network, differentiably (JAX).

    Parameters
    ----------
    edges : (E, 2) ndarray of int
    resistance : (E,) array of float, in Ω
    area : (n,) array of float, in µm²
    port_nodes : (P,) ndarray of int
    params : Passive
    freqs : sequence of float, in Hz

    Returns
    -------
    (F, P, P) jax array of complex, in Ω
    """
    _, jnp = _jax()
    n = area.shape[0]
    g = 1.0 / resistance
    i, j = edges[:, 0], edges[:, 1]
    lap = jnp.zeros((n, n))
    lap = lap.at[i, j].add(-g).at[j, i].add(-g).at[i, i].add(g).at[j, j].add(g)
    gm = params.g_pas * area * UM2_TO_CM2
    cm = params.c_m * 1e-6 * area * UM2_TO_CM2
    rhs = (
        jnp.zeros((n, len(port_nodes)))
        .at[port_nodes, jnp.arange(len(port_nodes))]
        .set(1.0)
    )
    out = []
    for f in freqs:
        a = lap + jnp.diag(gm + 1j * 2 * np.pi * f * cm)
        v = jnp.linalg.solve(a, rhs.astype(a.dtype))
        out.append(v[port_nodes, :])
    return jnp.stack(out)


def _clamp(z, k: int = 0):
    """Schur complement clamping port k (JAX-compatible)."""
    _, jnp = _jax()
    keep = np.delete(np.arange(z.shape[-1]), k)
    return z[..., keep[:, None], keep[None, :]] - (
        z[..., keep, k][..., :, None]
        * z[..., k, keep][..., None, :]
        / z[..., k, k][..., None, None]
    )


@dataclass
class FitResult:
    """Outcome of :func:`fit_tree_model`.

    Parameters
    ----------
    tree : PortTree
        The tree with fitted branch resistances.
    node_area : (n,) ndarray of float, in µm²
        Fitted membrane per node (same total as before).
    loss : (steps,) ndarray of float
        Loss after each optimization step.
    """

    tree: PortTree
    node_area: NDArray[np.float64]
    loss: NDArray[np.float64]


def fit_tree_model(
    tree: PortTree,
    node_area: ArrayLike,
    z_target: ArrayLike,
    params: Passive,
    freqs: Sequence[float],
    clamp_port: int = 0,
    isolated_weight: float = 0.1,
    steps: int = 800,
    learning_rate: float = 0.05,
) -> FitResult:
    """Fit branch resistances and membrane areas to reference port matrices.

    Loss = mean over frequencies of the relative (Frobenius) error of the
    port matrix with port ``clamp_port`` clamped (the spine's internal
    structure), plus ``isolated_weight`` × the same for the isolated matrix.

    Parameters
    ----------
    tree : PortTree
        Starting tree (e.g. from :func:`camovox.reduce.fit_tree`).
    node_area : (n,) array_like of float, in µm²
        Starting membrane per node. Its total is conserved.
    z_target : (F, P, P) array_like of complex, in Ω
        Reference port matrices at ``freqs``, ports in ``tree.port_node``
        order.
    params : Passive
    freqs : sequence of float, in Hz
    clamp_port : int
        Port held at rest for the main loss term (the neck).
    isolated_weight : float
    steps : int
        Adam steps.
    learning_rate : float
        Adam step size, in log units of resistance and area.

    Returns
    -------
    FitResult
    """
    jax, jnp = _jax()
    import optax

    target = jnp.asarray(np.asarray(z_target), dtype=complex)
    t_clamped = _clamp(target, clamp_port)
    area0 = np.asarray(node_area, dtype=float)
    total = area0.sum()
    theta0 = {
        "log_r": jnp.log(jnp.asarray(np.maximum(tree.resistance, 1.0))),
        "logit_a": jnp.log(jnp.asarray(np.maximum(area0, 1e-6 * total) / total)),
    }
    edges, ports = tree.edges, tree.port_node

    def model(theta):
        area = total * jax.nn.softmax(theta["logit_a"])
        return port_impedance_jax(
            edges, jnp.exp(theta["log_r"]), area, ports, params, freqs
        )

    def rel(a, b):
        return jnp.mean(
            jnp.linalg.norm(a - b, axis=(-2, -1)) / jnp.linalg.norm(b, axis=(-2, -1))
        )

    def loss_fn(theta):
        z = model(theta)
        return rel(_clamp(z, clamp_port), t_clamped) + isolated_weight * rel(z, target)

    opt = optax.adam(learning_rate)
    state = opt.init(theta0)

    @jax.jit
    def step(theta, state):
        value, grad = jax.value_and_grad(loss_fn)(theta)
        updates, state = opt.update(grad, state)
        return optax.apply_updates(theta, updates), state, value

    theta, history = theta0, []
    for _ in range(steps):
        theta, state, value = step(theta, state)
        history.append(float(value))
    fitted = replace(tree, resistance=np.asarray(jnp.exp(theta["log_r"])))
    return FitResult(
        tree=fitted,
        node_area=np.asarray(total * jax.nn.softmax(theta["logit_a"])),
        loss=np.array(history),
    )
