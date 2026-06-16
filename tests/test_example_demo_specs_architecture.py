from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

from conftest import make_mocap_track
from retarget.demo.demo import Demonstration
from retarget.demo.mocap import MocapTrack

EXAMPLE_DEMO_SPECS = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "process_mocap_data"
    / "demo_specs.py"
)
EXAMPLE_DEMO_VOCAB = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "process_mocap_data"
    / "demo_vocab.py"
)
NEW_API_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "process_mocap_data"
    / "new_api_example.py"
)
EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "process_mocap_data"
)
RUN_DEMO_TRACK_WORKFLOW = EXAMPLE_DIR / "run_demo_track_workflow.py"

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import demo_specs  # noqa: E402
import demo_vocab  # noqa: E402


def _demo_specs_ast() -> ast.Module:
    return ast.parse(EXAMPLE_DEMO_SPECS.read_text())


def _demo_vocab_ast() -> ast.Module:
    return ast.parse(EXAMPLE_DEMO_VOCAB.read_text())


def test_demo_vocab_defines_only_track_id_class() -> None:
    tree = _demo_vocab_ast()
    class_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    assert class_names == ["GroundEstimationTrackId"]


def test_demo_specs_defines_no_classes() -> None:
    tree = _demo_specs_ast()
    class_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    assert class_names == []


def test_demo_specs_does_not_define_custom_demo_facade() -> None:
    tree = _demo_specs_ast()
    forbidden = {
        "GroundEstimationDemo",
        "GroundEstimationDemoView",
        "DemoFacade",
        "GroundEstimationDemoFacade",
    }
    class_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert class_names.isdisjoint(forbidden)


def test_demo_specs_loader_returns_generic_demonstration_annotation() -> None:
    tree = _demo_specs_ast()
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    loader = functions["load_ground_estimation_demo"]
    assert loader.returns is not None
    annotation = ast.unparse(loader.returns)
    assert annotation == "Demonstration[GroundEstimationTrackId]"


def test_demo_specs_constructs_generic_demonstration() -> None:
    tree = _demo_specs_ast()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    constructed_names = set()
    for call in calls:
        if isinstance(call.func, ast.Name):
            constructed_names.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            constructed_names.add(call.func.attr)
    assert "Demonstration" in constructed_names
    assert "GroundEstimationDemo" not in constructed_names
    assert "GroundEstimationDemoView" not in constructed_names


def test_load_ground_estimation_demo_returns_generic_demonstration(monkeypatch) -> None:
    track = make_mocap_track()

    def fake_load_mocap_track(root, scene):
        return track

    monkeypatch.setattr(demo_specs, "load_mocap_track", fake_load_mocap_track)
    demo = demo_specs.load_ground_estimation_demo(Path("dummy"))
    assert isinstance(demo, Demonstration)
    assert not hasattr(demo_specs, "GroundEstimationDemo")
    assert not hasattr(demo_specs, "GroundEstimationDemoView")


def test_load_ground_estimation_demo_rebases_time(monkeypatch) -> None:
    track = make_mocap_track()
    shifted = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps + 100.0,
        marker_frames=track.marker_frames,
    )

    def fake_load_mocap_track(root, scene):
        return shifted

    monkeypatch.setattr(demo_specs, "load_mocap_track", fake_load_mocap_track)
    demo = demo_specs.load_ground_estimation_demo(Path("dummy"))
    mocap = demo.get_track(demo_vocab.GroundEstimationTrackId.MOCAP)
    assert isinstance(mocap, MocapTrack)
    np.testing.assert_allclose(mocap.timestamps, track.timestamps)


def test_run_demo_track_workflow_uses_generic_get_track_pattern() -> None:
    source = RUN_DEMO_TRACK_WORKFLOW.read_text()
    assert ".get_track(" in source
    assert ".mocap" not in source
    assert "GroundEstimationDemo" not in source
    assert "GroundEstimationDemoView" not in source


def test_process_mocap_example_defines_no_demo_facade_classes() -> None:
    forbidden = {
        "GroundEstimationDemo",
        "GroundEstimationDemoView",
        "GroundEstimationDemoFacade",
        "DemoFacade",
    }
    offenders: dict[Path, list[str]] = {}
    for path in EXAMPLE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        class_names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        ]
        found = sorted(set(class_names) & forbidden)
        if found:
            offenders[path] = found
    assert offenders == {}


def test_process_mocap_example_does_not_reimplement_demo_container_methods() -> None:
    forbidden_method_names = {
        "track",
        "typed_track",
        "slice_time",
        "with_track",
        "with_alignment",
        "align",
        "resample_to",
    }
    offenders: dict[str, list[str]] = {}
    for path in EXAMPLE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
            if not any(
                token in cls.name for token in ("Demo", "Demonstration", "Facade")
            ):
                continue
            methods = {
                item.name
                for item in cls.body
                if isinstance(item, ast.FunctionDef)
            }
            found = sorted(methods & forbidden_method_names)
            if found:
                offenders[f"{path.name}:{cls.name}"] = found
    assert offenders == {}


def test_new_api_example_shows_typed_authoring_and_declaration_only_patches() -> None:
    source = NEW_API_EXAMPLE.read_text()
    assert "build_scene(subjects)" in source
    assert "Patch.rectangular(" in source
    assert 'toe_contact=Patch(' in source
    assert 'label="toe_contact_display"' in source
    assert 'shoe.patch_target("toe_contact")' in source
