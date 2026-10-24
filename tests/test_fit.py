import numpy as np
import pytest

import cabaret as cb
from cabaret import fit, jaxley_io, reduce

P = cb.Passive(r_L=100.0, c_m=1.0, g_pas=2.63e-4)


def _toy_tree():
    # Root 0 (neck), internal 1, leaves 2, 3, 4 (ports: 0, 2, 3, 4).
    return reduce.PortTree(
        n_nodes=5,
        edges=np.array([[0, 1], [1, 2], [1, 3], [0, 4]]),
        resistance=np.array([5e6, 10e6, 20e6, 8e6]),
        port_node=np.array([0, 2, 3, 4]),
        fit_error=0.0,
    )


def test_jax_impedance_matches_cabaret_solver():
    tree, area = _toy_tree(), np.array([5.0, 0.5, 2.0, 3.0, 1.0])
    g = reduce.tree_to_graph(tree, P, area)
    ref = cb.port_impedance(
        g, P, [cb.point_port(g, int(n)) for n in tree.port_node], [0, 500]
    )
    z = fit.port_impedance_jax(
        tree.edges, tree.resistance, area, tree.port_node, P, [0, 500]
    )
    np.testing.assert_allclose(np.asarray(z), ref, rtol=1e-8)


def test_fit_recovers_perturbed_parameters():
    tree, area = _toy_tree(), np.array([5.0, 0.5, 2.0, 3.0, 1.0])
    freqs = [0.0, 100.0]
    target = fit.port_impedance_jax(
        tree.edges, tree.resistance, area, tree.port_node, P, freqs
    )
    start = reduce.PortTree(
        5, tree.edges, tree.resistance * [1.5, 0.7, 1.3, 0.8], tree.port_node, 0.0
    )
    area0 = area * [1, 2, 0.5, 1, 1]
    res = fit.fit_tree_model(
        start,
        area0,
        target,
        P,
        freqs,
        steps=1500,
        learning_rate=0.05,
    )
    assert res.loss[-1] < 0.05 * res.loss[0]
    np.testing.assert_allclose(res.node_area.sum(), area0.sum())  # total conserved


@pytest.mark.slow
def test_jaxley_export_matches_cabaret_transient():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jaxley as jx

    tree, area = _toy_tree(), np.array([5.0, 0.0, 2.0, 3.0, 1.0])
    g = reduce.tree_to_graph(tree, P, area)
    dt = 0.025  # ms
    t = np.arange(0, 10, dt)
    i_pa = 50 * (t / 0.5) * np.exp(1 - t / 0.5)
    w = cb.point_port(g, 2)
    v_ref = cb.simulate_current(g, P, w, i_pa * 1e-12, dt * 1e-3, w[None])[:, 0] * 1e3
    jm = jaxley_io.to_jaxley(tree, area, P)
    branch = jm.cell.branch(int(jm.branch_of_node[2])).comp(0)
    branch.stimulate(i_pa * 1e-3, verbose=False)
    branch.record("v", verbose=False)
    v_jx = np.asarray(jx.integrate(jm.cell, delta_t=dt))[0]
    np.testing.assert_allclose(v_jx.max(), v_ref.max(), rtol=1e-3)
