import numpy as np

from retarget.core import RigidTransform


def test_rigid_transform_from_translation() -> None:
    translation = np.array([1.0, 2.0, 3.0])
    transform = RigidTransform.from_translation(translation)
    np.testing.assert_allclose(transform.translation, translation)
    np.testing.assert_allclose(transform.rotation, np.eye(3))
