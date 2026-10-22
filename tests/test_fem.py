import numpy as np
import pytest
import trimesh

import camovox as cv
from camovox import validate as va


@pytest.fixture(scope="module")
def sphere_ref():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=8.0)
    tets, report = cv.tetrahedralize(mesh, edge_length=1.5)
    ref, _ = cv.fem_model(tets)
    return ref, report


@pytest.fixture(scope="module")
def cylinder_ref():
    # Short and thick so meshing stays fast. High g_pas keeps L/λ ≈ 0.3.
    mesh = trimesh.creation.cylinder(radius=2.0, height=100.0, sections=48)
    tets, _ = cv.tetrahedralize(mesh, edge_length=1.0, envelope=0.01)
    ref, _ = cv.fem_model(tets)
    return ref


@pytest.mark.slow
def test_sphere_matches_isopotential_solution(sphere_ref):
    ref, report = sphere_ref
    assert abs(report.area_error) < 0.01
    p = cv.Passive(r_L=100.0, c_m=1.0, g_pas=2.5e-5)
    w = cv.ball_patch(ref, [8.0, 0.0, 0.0], 1.0)
    rin = cv.input_resistance(ref, p, w)
    np.testing.assert_allclose(rin, va.sphere_input_resistance(8.0, p), rtol=0.01)
    tau = cv.time_constants(ref, p, k=1)
    np.testing.assert_allclose(tau[0], p.tau_m, rtol=1e-3)


@pytest.mark.slow
def test_sealed_cylinder_matches_cable_theory(cylinder_ref):
    ref = cylinder_ref
    p = cv.Passive(r_L=100.0, c_m=1.0, g_pas=1e-3)
    w0 = cv.ball_patch(ref, [0.0, 0.0, -50.0], 2.0)
    w1 = cv.ball_patch(ref, [0.0, 0.0, 50.0], 2.0)
    rin = cv.input_resistance(ref, p, w0)
    # End caps carry membrane that the analytic sealed cable lacks (~2% area).
    np.testing.assert_allclose(
        rin, va.sealed_cylinder_input_resistance(2.0, 100.0, p), rtol=0.03
    )
    atten = cv.transfer_impedance(ref, p, w0, w1).real / rin
    np.testing.assert_allclose(
        atten, va.sealed_cylinder_attenuation(100.0, 2.0, 100.0, p), rtol=0.01
    )
