import numpy as np
import pytest
import trimesh

import cabaret as cb
from cabaret import ports


@pytest.mark.slow
def test_build_reference_cylinder_between_two_end_electrodes():
    # A cylinder with both flat ends as attachments. With the membrane
    # removed, the resistance between the two end electrodes is exactly
    # r_L · L / A (uniform axial field), which tests the electrodes.
    length, radius = 4.0, 0.5  # µm
    surface = trimesh.creation.cylinder(radius=radius, height=length, sections=32)

    def ends(tets: cb.TetMesh) -> list[np.ndarray]:
        z = tets.points[:, 2]
        return [
            np.flatnonzero(z < -length / 2 + 1e-6),
            np.flatnonzero(z > length / 2 - 1e-6),
        ]

    ref = cb.build_reference(
        surface, ends, site_points=[[radius, 0.0, 0.0]], edge_length=0.25
    )
    assert ref.network.n_nodes == ref.fem.n_nodes + 2
    assert len(ref.attachment_ports) == 2
    for c in ref.attachment_contacts:
        assert c.size > 0
        np.testing.assert_allclose(ref.network.area[c], 0.0)  # membrane removed
    removed = sum(ref.fem.area[c].sum() for c in ref.attachment_contacts)
    np.testing.assert_allclose(
        ref.network.area.sum(), ref.fem.area.sum() - removed, rtol=1e-12
    )

    # The site patch lies on the remaining membrane, at about its target area.
    (patch,) = ref.sites.patches
    assert patch[np.concatenate(ref.attachment_contacts)].sum() == 0.0
    assert 0.05 <= patch.sum() < 0.1

    p = cb.Passive(r_L=100.0)
    metric = ports.ResistanceMetric(ref.network, p, ground=ref.attachment_ports[0])
    d = metric.distances([ref.attachment_ports[1]])
    area = ref.fidelity.volume_out / length  # cross-section of the polygonal cylinder
    expected = p.r_L * 1e4 * length / area  # Ω·cm → Ω·µm
    np.testing.assert_allclose(d[0, 1], expected, rtol=1e-3)


def test_build_reference_rejects_empty_attachment():
    tets = cb.TetMesh(
        points=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        tets=np.array([[0, 1, 2, 3]]),
    )
    surface = trimesh.Trimesh(tets.points, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    with pytest.raises(ValueError, match="attachment 0 has no nodes"):
        cb.build_reference(surface, [np.array([], dtype=int)], tets=tets)
