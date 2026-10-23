import numpy as np

import camovox as cv
from camovox import ports


def _graph(edges, kappa, n):
    return cv.CompartmentGraph(
        area=np.zeros(n),
        position=np.zeros((n, 3)),
        edges=np.asarray(edges),
        kappa=np.asarray(kappa, dtype=float),
    )


def _onehot(i, n):
    w = np.zeros(n)
    w[i] = 1.0
    return w


def test_resistance_distance_on_a_tree_is_path_sum():
    # Star with the ground at the centre (node 0) and ports on the spokes.
    p = cv.Passive(r_L=100.0)
    kappa = [1.0, 2.0, 4.0]
    g = _graph([[0, 1], [0, 2], [0, 3]], kappa, 4)
    d = ports.resistance_distances(
        g, p, [_onehot(i, 4) for i in (1, 2, 3)], _onehot(0, 4)
    )
    r = 1.0 / p.axial_conductance(kappa)  # spoke resistances (Ω)
    np.testing.assert_allclose(d[0, 1:], r, rtol=1e-6)
    np.testing.assert_allclose(d[1, 2], r[0] + r[1], rtol=1e-6)
    assert ports.loopiness_index(d)["max"] < 1e-9


def test_four_point_violation_of_a_ring():
    # Ring of 4 equal resistors R: adjacent 3R/4, opposite R.
    d = np.array(
        [
            [0, 0.75, 1.0, 0.75],
            [0.75, 0, 0.75, 1.0],
            [1.0, 0.75, 0, 0.75],
            [0.75, 1.0, 0.75, 0],
        ]
    )
    np.testing.assert_allclose(ports.four_point_violation(d), [0.25])


def test_rank_bound_endpoints(rng):
    a = rng.normal(size=(6, 6))
    err = ports.rank_bound(a @ a.T)
    np.testing.assert_allclose([err[0], err[-1]], [1.0, 0.0])
    assert np.all(np.diff(err) <= 1e-12)


def test_gap_normalized_violation_ignores_leaf_resistance(rng):
    # A ring of 4 is maximally non-tree in the leaf-invariant measure.
    ring = np.array(
        [
            [0, 0.75, 1.0, 0.75],
            [0.75, 0, 0.75, 1.0],
            [1.0, 0.75, 0, 0.75],
            [0.75, 1.0, 0.75, 0],
        ]
    )
    np.testing.assert_allclose(ports.four_point_violation(ring, "gap"), [1.0])
    # Adding a leaf resistance to one port shifts all its distances.
    leafy = ring.copy()
    leafy[0, 1:] += 2.0
    leafy[1:, 0] += 2.0
    for norm in ("gap", "none"):
        np.testing.assert_allclose(
            ports.four_point_violation(leafy, norm),
            ports.four_point_violation(ring, norm),
        )
    assert ports.four_point_violation(leafy, "max")[0] < 0.25
