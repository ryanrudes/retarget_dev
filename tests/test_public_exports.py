from __future__ import annotations

import retarget.core as core
import retarget.demo as demo
import retarget.io as io
import retarget.utils as utils


def _assert_all_exports(module: object) -> None:
    exports = tuple(getattr(module, "__all__"))
    assert len(exports) == len(set(exports)), f"{module.__name__}.__all__ contains duplicates"
    for name in exports:
        assert hasattr(module, name), f"{module.__name__}.{name} missing from module"


def test_public_exports_are_importable() -> None:
    for module in (core, demo, io, utils):
        _assert_all_exports(module)


def test_public_exports_include_critical_symbols() -> None:
    expected = {
        core: {
            "SignedAxis",
            "QuaternionOrder",
            "SceneSpec",
            "SegmentSpec",
            "PatchTarget",
            "build_scene",
        },
        demo: {"Demonstration", "DemonstrationView", "SyncPlan"},
        io: {"MarkerObservation", "marker_position", "read_marker_positions_from_vsk"},
        utils: {"point_in_polygon", "fit_patch_frame", "estimate_nominal_hz"},
    }
    for module, names in expected.items():
        exports = set(module.__all__)
        assert names <= exports
