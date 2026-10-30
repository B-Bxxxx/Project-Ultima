import numpy as np
import trimesh

from src.common.placement import contact_center, place_mesh
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from tests.test_centerline import _tube
from tests.test_centerline_strategy import _elbow_up


def _foot_part():
    """A 10 x 10 foot on the bed and a 40 mm long arm sticking out in +X: the bounding box centre is far from the foot."""
    foot = trimesh.creation.box(extents=(10, 10, 10))
    foot.apply_translation([0, 0, 5])
    arm = trimesh.creation.box(extents=(40, 4, 4))
    arm.apply_translation([22, 0, 12])
    return trimesh.util.concatenate([foot, arm])


def test_contact_center_is_the_foot_not_the_bounding_box():
    m = _foot_part()
    assert np.allclose(contact_center(m), [0, 0], atol=1e-6)
    assert m.bounds.mean(axis=0)[0] > 10                                   # the box centre really is elsewhere
    out = place_mesh(m, center_xy=True, center_ref="contact")
    assert np.allclose(contact_center(out), [0, 0], atol=1e-6)
    assert abs(out.bounds.mean(axis=0)[0]) > 5                              # and the box is not centred
    ref = place_mesh(m, center_xy=True, center_ref="bbox")
    assert np.allclose(ref.bounds.mean(axis=0)[:2], [0, 0], atol=1e-6)


def test_contact_center_follows_rotation_offset_and_ignores_a_curved_bottom():
    m = _foot_part()
    out = place_mesh(m, rotation_deg=(0, 0, 90), center_ref="contact", offset_xy=(3, -2))
    assert np.allclose(contact_center(out), [3, -2], atol=1e-6)            # the offset is measured from the contact centre
    ball = trimesh.creation.icosphere(radius=5)
    assert np.all(np.isfinite(contact_center(ball)))                        # no flat face: lowest vertices


def test_default_placement_is_unchanged():
    m = _foot_part()
    assert np.allclose(place_mesh(m).vertices, place_mesh(m, center_ref="bbox").vertices)


def test_part_offset_moves_the_centerline_slicing_with_it():
    base = _tube(_elbow_up(), 4.0)
    moved = place_mesh(base, center_xy=True, center_ref="contact", offset_xy=(7.0, -4.0))
    a = place_mesh(base, center_xy=True, center_ref="contact")
    p = {"strategy": "centerline", "layer_height": 0.5}
    ma = UniversalFieldSlicerPlugin().slice(a, p)
    mb = UniversalFieldSlicerPlugin().slice(moved, p)
    assert len(ma.layers) == len(mb.layers)
    for la, lb in list(zip(ma.layers, mb.layers))[::10]:
        Pa = np.array(la.contours[0].points)
        Pb = np.array(lb.contours[0].points)
        assert np.allclose(Pb - Pa, [7.0, -4.0, 0.0], atol=0.02)
