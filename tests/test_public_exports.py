from __future__ import annotations

import retarget.contacts as contacts
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
    for module in (core, demo, io, utils, contacts):
        _assert_all_exports(module)


def test_public_exports_include_critical_symbols() -> None:
    expected = {
        core: {
            "SignedAxis",
            "QuaternionOrder",
            "Marker",
            "Patch",
            "Segment",
            "Subject",
            "Markers",
            "Subjects",
            "PatchTarget",
            "bind_scene",
        },
        demo: {
            "Demonstration",
            "DemonstrationView",
            "Tracks",
            "MocapTrack",
            "SyncPlan",
            "fill_pose_gaps",
        },
        io: {"MarkerObservation", "marker_position", "read_marker_positions_from_vsk"},
        utils: {
            "point_in_polygon",
            "fit_patch_frame",
            "estimate_nominal_hz",
            "intervals_from_mask",
            "clean_mask_by_time",
        },
        contacts: {
            "ContactDetector",
            "detect_contacts",
            "classify_contacts",
            "ContactDetectionConfig",
            "ContactPlan",
            "ContactQuery",
            "apply_contact_plan",
            "infer_ground",
            "SupportStateTrack",
            "SupportResolver",
            "SupportModel",
            "ground_plane",
            "priority",
            "highest_score",
            "multi_label",
            "speed_limit",
            "robust_jumps",
            "JumpDetector",
        },
    }
    for module, names in expected.items():
        exports = set(module.__all__)
        assert names <= exports
