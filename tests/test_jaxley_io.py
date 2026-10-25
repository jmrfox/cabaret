import numpy as np
import pytest
import trimesh

import cabaret as cb
from cabaret import jaxley_io, reduce


@pytest.fixture(autouse=True)
def _x64():
    import jax

    jax.config.update("jax_enable_x64", True)


def _record(jm, nodes):
    for n in nodes:
        jm.cell.branch(int(jm.branch_of_node[n])).comp(0).record("v", verbose=False)


@pytest.mark.slow
def test_closed_loop_matches_cabaret_ring():
    p = cb.Passive(r_L=100.0, c_m=1.0, g_pas=2.63e-4)
    # Ring 0-1-2-3-4-0: the tree is the path, the loop closes 4-0.
    tree = reduce.PortTree(
        n_nodes=5,
        edges=np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
        resistance=np.array([5e7, 1e8, 2e8, 8e7]),
        port_node=np.array([0, 2, 4]),
        fit_error=0.0,
    )
    area = np.array([1000.0, 200.0, 400.0, 600.0, 300.0])
    r_loop = 1.2e8
    g = reduce.tree_to_graph(tree, p, area)
    ring = cb.CompartmentGraph(
        area=g.area,
        position=g.position,
        edges=np.vstack([g.edges, [[4, 0]]]),
        kappa=np.append(g.kappa, p.r_L / (r_loop * 1e-4)),
    )
    dt = 0.025  # ms
    t = np.arange(0, 10, dt)
    i_pa = 50 * (t / 0.5) * np.exp(1 - t / 0.5)
    rec = [0, 2, 4]
    readout = np.stack([cb.point_port(ring, n) for n in rec])
    v_ref = cb.simulate_current(
        ring, p, cb.point_port(ring, 2), i_pa * 1e-12, dt * 1e-3, readout
    )
    jm = jaxley_io.to_jaxley(tree, area, p, loops=[(4, 0, r_loop)])
    assert jm.n_loops == 1
    stim = jm.cell.branch(int(jm.branch_of_node[2])).comp(0)
    stim.stimulate(i_pa * 1e-3, verbose=False)
    _record(jm, rec)
    v_jx = jaxley_io.integrate(jm, delta_t=dt)[:, 1:]  # drop t = 0
    np.testing.assert_allclose(v_jx.T, v_ref * 1e3, atol=1e-5 * np.abs(v_ref).max())
    with pytest.raises(ValueError, match="jax.sparse"):
        jaxley_io.integrate(jm, delta_t=dt, voltage_solver="jaxley.dhs")


@pytest.mark.slow
def test_looped_export_matches_fem_torus():
    # FEM ring with four electrode ports, Kron-reduced at DC to a 4-node
    # network: ring edges, two diagonals and shunts. The jaxley export (path
    # tree + 3 loops) must reproduce the FEM port impedances; the tree alone
    # must not. High g_pas makes the ring electrically long.
    p = cb.Passive(r_L=100.0, c_m=1.0, g_pas=0.05)
    torus = trimesh.creation.torus(major_radius=5.0, minor_radius=1.0)
    tets, _ = cb.tetrahedralize(torus, edge_length=0.8, envelope=0.04)
    ref, _ = cb.fem_model(tets)
    ports = []
    for ang in np.deg2rad([0, 90, 180, 270]):
        rim = [6.0 * np.cos(ang), 6.0 * np.sin(ang), 0.0]
        d = np.linalg.norm(ref.position - rim, axis=1)
        ref, k = cb.add_electrode(ref, np.flatnonzero((d < 0.6) & (ref.area > 0)))
        ports.append(k)
    z = cb.port_impedance(ref, p, [cb.point_port(ref, k) for k in ports])[0].real
    y = np.linalg.inv(z)
    assert np.all(y[~np.eye(4, dtype=bool)] < 0)  # positive couplings only

    def r(i, j):
        return -1.0 / y[i, j]

    tree = reduce.PortTree(
        n_nodes=4,
        edges=np.array([[0, 1], [1, 2], [2, 3]]),
        resistance=np.array([r(0, 1), r(1, 2), r(2, 3)]),
        port_node=np.arange(4),
        fit_error=0.0,
    )
    area = y.sum(axis=1) / (p.g_pas * 1e-8)  # shunt as membrane, in µm²
    loops = [(3, 0, r(3, 0)), (0, 2, r(0, 2)), (1, 3, r(1, 3))]
    i_na = 0.01
    v = {}
    for name, lp in [("looped", loops), ("tree", [])]:
        jm = jaxley_io.to_jaxley(tree, area, p, loops=lp)
        stim = jm.cell.branch(int(jm.branch_of_node[0])).comp(0)
        stim.stimulate(np.full(80, i_na), verbose=False)  # 2 ms ≫ τ = 0.02 ms
        _record(jm, range(4))
        v[name] = jaxley_io.integrate(jm, delta_t=0.025)[:, -1]
    v_fem = z[:, 0] * i_na * 1e-9 * 1e3  # mV
    np.testing.assert_allclose(v["looped"], v_fem, rtol=1e-4)
    assert np.abs(v["tree"] / v_fem - 1).max() > 0.05
