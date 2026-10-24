import numpy as np

import cabaret as cb


def test_p1_gradient_is_exact_for_linear_field(rng):
    points = rng.uniform(-1.0, 1.0, size=(40, 3))
    # Random tetrahedra, dropping near-degenerate ones.
    tets = np.array([rng.choice(40, 4, replace=False) for _ in range(30)])
    p = points[tets]
    vol = np.abs(
        np.einsum(
            "ij,ij->i",
            p[:, 1] - p[:, 0],
            np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]),
        )
    )
    tm = cb.TetMesh(points=points, tets=tets[vol > 1e-3])
    g = np.array([0.3, -1.2, 2.5])
    grad = cb.tet_gradient(tm, points @ g + 0.7)
    np.testing.assert_allclose(grad, np.broadcast_to(g, grad.shape), rtol=1e-10)
