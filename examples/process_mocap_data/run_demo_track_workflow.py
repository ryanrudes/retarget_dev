from __future__ import annotations

from pathlib import Path

import numpy as np

from demo_specs import load_ground_estimation_demo
from demo_vocab import GroundEstimationTrackId
from mocap_specs import LEFT_SHOE_SEGMENT
from mocap_vocab import (
    LeftShoePatchId,
    LeftShoeSegmentId,
    ViconSubjectId,
)

from retarget.core import PatchHandle
from retarget.core.targets import PatchTarget


UNBAGGED_DIR = (
    Path(__file__).resolve().parents[2]
    / "bags"
    / "ground_estimation"
    / "unbagged"
)


def main() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    mocap = demo.get_track(GroundEstimationTrackId.MOCAP)
    clip = demo.slice_time(0.0, 1.0)
    mocap_clip = clip.get_track(GroundEstimationTrackId.MOCAP)
    left_shoe = mocap_clip.subject(ViconSubjectId.LEFT_SHOE).segment(LEFT_SHOE_SEGMENT)
    sole_target = PatchTarget(
        subject=ViconSubjectId.LEFT_SHOE,
        handle=PatchHandle(
            segment=LeftShoeSegmentId.LEFT_SHOE,
            patch=LeftShoePatchId.SOLE,
        ),
    )
    translations = left_shoe.translations()
    rotations = left_shoe.rotations()
    sole_points = left_shoe.patch_points(LeftShoePatchId.SOLE)
    sole_normals = left_shoe.patch_normals(LeftShoePatchId.SOLE)

    print("Demo tracks:", tuple(demo.tracks))
    print("Mocap timestamps:", np.asarray(mocap.timestamps[:5]))
    print("Clip timestamps:", np.asarray(mocap_clip.timestamps[:5]))
    print("Left shoe translations shape:", translations.shape)
    print("Left shoe rotations shape:", rotations.shape)
    print("First left shoe translation:", translations[0])
    print("First sole point:", sole_points[0])
    print("First sole normal:", sole_normals[0])
    print("Sole target:", sole_target)

    if GroundEstimationTrackId.CONTACTS in demo.tracks:
        contacts = clip.get_track(GroundEstimationTrackId.CONTACTS)
        print("Contact timestamps:", np.asarray(contacts.timestamps[:5]))
        print("Sole contact:", contacts.state(sole_target)[:5])
        print("Sole confidence:", contacts.confidence(sole_target)[:5])


if __name__ == "__main__":
    main()
