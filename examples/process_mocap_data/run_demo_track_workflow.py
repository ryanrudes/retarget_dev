"""This workflow uses backend/manual Vicon scene support.

It loads real VSK-derived marker calibration and bag data. For public scene
authoring, see new_api_example.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend_specs.ground_estimation_loader import load_ground_estimation_demo
from backend_specs.vicon_scene import LEFT_SHOE_SEGMENT
from backend_specs.vicon_vocab import (
    LeftShoePatchId,
    ViconSubjectId,
)
from demo_vocab import GroundEstimationTrackId


UNBAGGED_DIR = (
    Path(__file__).resolve().parents[2]
    / "bags"
    / "ground_estimation"
    / "unbagged"
)


def main() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    mocap = demo[GroundEstimationTrackId.MOCAP]
    clip = demo.slice_time(0.0, 1.0)
    mocap_clip = clip[GroundEstimationTrackId.MOCAP]
    left_shoe = (
        mocap_clip._subject(ViconSubjectId.LEFT_SHOE)
        ._segment(LEFT_SHOE_SEGMENT)
    )
    sole_target = left_shoe.segment_view.patch_target(LeftShoePatchId.SOLE)
    translations = left_shoe.translations()
    rotations = left_shoe.rotations()
    sole_points = left_shoe._patch_points(LeftShoePatchId.SOLE)
    sole_normals = left_shoe._patch_normals(LeftShoePatchId.SOLE)

    print("Demo tracks:", tuple(demo._tracks))
    print("Mocap timestamps:", np.asarray(mocap.timestamps[:5]))
    print("Clip timestamps:", np.asarray(mocap_clip.timestamps[:5]))
    print("Left shoe translations shape:", translations.shape)
    print("Left shoe rotations shape:", rotations.shape)
    print("First left shoe translation:", translations[0])
    print("First sole point:", sole_points[0])
    print("First sole normal:", sole_normals[0])
    print("Sole target:", sole_target)

    if GroundEstimationTrackId.CONTACTS in demo._tracks:
        contacts = clip[GroundEstimationTrackId.CONTACTS]
        print("Contact timestamps:", np.asarray(contacts.timestamps[:5]))
        print("Sole contact:", contacts.state(sole_target)[:5])
        print("Sole confidence:", contacts.confidence(sole_target)[:5])


if __name__ == "__main__":
    main()
