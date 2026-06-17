"""Typed geometry-query example over real VSK-derived bag data.

Loads real data via the backend ``load_ground_estimation_demo`` helper and uses
the public typed deep chain for marker/patch geometry queries. Because the
backend authors segment-frame marker positions from the VSK file, modeled marker
positions are available alongside observed ones.
"""

from __future__ import annotations

from pathlib import Path

from backend_specs.ground_estimation_loader import load_ground_estimation_demo

from retarget.core import RotationFormat

REPO_ROOT = Path(__file__).resolve().parents[2]
UNBAGGED_DIR = REPO_ROOT / "bags" / "ground_estimation" / "unbagged"


def main() -> None:
    if not UNBAGGED_DIR.is_dir():
        print(f"Skipping; no local bag found at {UNBAGGED_DIR}")
        return

    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    clip = demo.tracks["mocap"].slice_time(0.0, 1.0)
    shoe = clip.subjects["left_shoe"].segments["shoe"]

    heel = shoe.markers["heel"]
    heel_observed = heel.positions()
    heel_modeled = heel.positions(modeled=True)
    heel_velocity = heel.velocities()
    shoe_quat = shoe.rotations(format=RotationFormat.QUATERNION_XYZW)

    sole = shoe.patches["sole"]
    sole_point = sole.points()
    sole_normal = sole.normals()

    print("Clip timesteps:", len(clip.timestamps))
    print("Heel observed:", heel_observed[:1])
    print("Heel modeled:", heel_modeled[:1])
    print("Heel velocity:", heel_velocity[:1])
    print("Shoe quaternion:", shoe_quat[:1])
    print("Sole target:", sole.target)
    print("Sole point:", sole_point[:1])
    print("Sole normal:", sole_normal[:1])


if __name__ == "__main__":
    main()
