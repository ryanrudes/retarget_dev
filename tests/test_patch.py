import numpy as np
from retarget.utils.geometry import fit_patch_frame


def test_fit_patch_frame() -> None:
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    transform = fit_patch_frame(points)

    np.testing.assert_allclose(transform.translation, np.array([1/3, 1/3, 0.0]))
    np.testing.assert_allclose(abs(transform.rotation[:, 2][2]), 1.0)