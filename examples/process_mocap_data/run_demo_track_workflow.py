"""Typed demo-track workflow over real VSK-derived bag data.

This loads real data through the backend ``load_ground_estimation_demo`` helper,
then uses the public typed API: ``demo.tracks["mocap"]`` and the
``mocap.subjects[...].segments[...]`` deep chain. Slicing the track (rather than
the demo) preserves the typed schema, so the chain stays statically typed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend_specs.ground_estimation_loader import load_ground_estimation_demo

UNBAGGED_DIR = (
    Path(__file__).resolve().parents[2]
    / "bags"
    / "ground_estimation"
    / "unbagged"
)


def main() -> None:
    if not UNBAGGED_DIR.is_dir():
        print(f"Skipping; no local bag found at {UNBAGGED_DIR}")
        return

    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    mocap = demo.tracks["mocap"]
    clip = mocap.slice_time(0.0, 1.0)

    shoe = clip.subjects["left_shoe"].segments["shoe"]
    translations = shoe.translations()
    rotations = shoe.rotations()
    sole = shoe.patches["sole"]
    sole_points = sole.points()
    sole_normals = sole.normals()

    print("Track names:", demo.track_ids())
    print("Mocap timestamps:", np.asarray(mocap.timestamps[:5]))
    print("Clip timestamps:", np.asarray(clip.timestamps[:5]))
    print("Left shoe translations shape:", translations.shape)
    print("Left shoe rotations shape:", rotations.shape)
    print("First left shoe translation:", translations[0])
    print("First sole point:", sole_points[0])
    print("First sole normal:", sole_normals[0])
    print("Sole target:", sole.target)


if __name__ == "__main__":
    main()
