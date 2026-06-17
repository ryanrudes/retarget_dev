from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

from conftest import make_mocap_track
from retarget.demo.demo import Demonstration
from retarget.demo.mocap import MocapTrack

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "process_mocap_data"
)
BACKEND_SPECS_DIR = EXAMPLE_ROOT / "backend_specs"
BACKEND_INIT = BACKEND_SPECS_DIR / "__init__.py"
BACKEND_VOCAB = BACKEND_SPECS_DIR / "vicon_vocab.py"
BACKEND_SCENE = BACKEND_SPECS_DIR / "vicon_scene.py"
BACKEND_LOADER = BACKEND_SPECS_DIR / "ground_estimation_loader.py"
DEMO_VOCAB = EXAMPLE_ROOT / "demo_vocab.py"
NEW_API_EXAMPLE = EXAMPLE_ROOT / "new_api_example.py"
RUN_DEMO_TRACK_WORKFLOW = EXAMPLE_ROOT / "run_demo_track_workflow.py"
RUN_CORE_GEOMETRY_BASICS = EXAMPLE_ROOT / "run_core_geometry_basics.py"

if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from backend_specs import ground_estimation_loader  # noqa: E402
import demo_vocab  # noqa: E402


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def test_backend_specs_package_and_modules_exist() -> None:
    assert BACKEND_SPECS_DIR.is_dir()
    assert BACKEND_INIT.is_file()
    assert BACKEND_VOCAB.is_file()
    assert BACKEND_SCENE.is_file()
    assert BACKEND_LOADER.is_file()


def test_demo_vocab_defines_only_track_id_class() -> None:
    tree = _module_ast(DEMO_VOCAB)
    class_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    assert class_names == ["GroundEstimationTrackId"]


def test_backend_vocab_defines_expected_classes() -> None:
    tree = _module_ast(BACKEND_VOCAB)
    class_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    assert class_names == [
        "ViconSubjectId",
        "LeftShoeSegmentId",
        "LeftShoePatchId",
        "LeftShoeMarkerId",
    ]


def test_backend_loader_defines_no_classes() -> None:
    tree = _module_ast(BACKEND_LOADER)
    class_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    assert class_names == []


def test_backend_support_files_are_clearly_backend_manual() -> None:
    for path in (
        BACKEND_VOCAB,
        BACKEND_SCENE,
        BACKEND_LOADER,
        RUN_DEMO_TRACK_WORKFLOW,
        RUN_CORE_GEOMETRY_BASICS,
    ):
        source = path.read_text().lower()
        assert "backend" in source
        assert "manual" in source


def test_ground_estimation_loader_returns_generic_demonstration_annotation() -> None:
    tree = _module_ast(BACKEND_LOADER)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    loader = functions["load_ground_estimation_demo"]
    assert loader.returns is not None
    annotation = ast.unparse(loader.returns)
    assert annotation == "Demonstration[GroundEstimationTrackId]"


def test_ground_estimation_loader_constructs_generic_demonstration() -> None:
    tree = _module_ast(BACKEND_LOADER)
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

    monkeypatch.setattr(
        ground_estimation_loader,
        "load_mocap_track",
        fake_load_mocap_track,
    )
    demo = ground_estimation_loader.load_ground_estimation_demo(Path("dummy"))
    assert isinstance(demo, Demonstration)
    assert not hasattr(ground_estimation_loader, "GroundEstimationDemo")
    assert not hasattr(ground_estimation_loader, "GroundEstimationDemoView")


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

    monkeypatch.setattr(
        ground_estimation_loader,
        "load_mocap_track",
        fake_load_mocap_track,
    )
    demo = ground_estimation_loader.load_ground_estimation_demo(Path("dummy"))
    mocap = demo[demo_vocab.GroundEstimationTrackId.MOCAP]
    assert isinstance(mocap, MocapTrack)
    np.testing.assert_allclose(mocap.timestamps, track.timestamps)


def test_run_demo_track_workflow_uses_enum_keyed_track_lookup() -> None:
    source = RUN_DEMO_TRACK_WORKFLOW.read_text()
    assert "GroundEstimationTrackId.MOCAP" in source
    assert "backend/manual" in source.lower()
    assert "internal" in source.lower()
    assert ".track_ids()" in source
    assert ".get_track(" not in source
    assert "._tracks" not in source
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
    for path in EXAMPLE_ROOT.rglob("*.py"):
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
    for path in EXAMPLE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for cls in [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]:
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
    assert source.count('vicon_name="Left_Shoe_Improved"') >= 2
    assert "left_shoe=Subject(" in source
    assert "shoe=Segment(" in source
    assert "class GroundEstimationSubjects(Subjects):" in source
    assert "class GroundEstimationTracks(Tracks):" in source
    assert "build_demonstration(GroundEstimationTracks(mocap=mocap))" in source
    assert 'mocap = demo.tracks["mocap"]' in source
    assert "GroundEstimationTrackId" not in source
    assert 'left_shoe = mocap.subjects["left_shoe"]' in source
    assert 'shoe = left_shoe.segments["shoe"]' in source
    assert 'heel = shoe.markers["heel"]' in source
    assert 'sole = shoe.patches["sole"]' in source
    assert 'heel_positions = heel.positions()' in source
    assert 'sole_points = sole.points()' in source
    assert "load_mocap_track(" in source
    assert "UnbaggedDirectory(" in source
    assert "UNBAGGED_DIR.is_dir()" in source
    assert "Skipping bag-backed demo" in source
    assert 'sole_handle = shoe_spec.patch("sole")' in source
    assert 'sole_label = shoe_spec.patch_label("sole")' in source
    assert 'shoe_spec.patch_spec("toe_contact")' in source
    assert "load_ground_estimation_demo(" not in source
    assert "shoe_spec.marker_type.heel" not in source
    assert "shoe_spec.patch_type.sole" not in source
    assert "typed_tracks" not in source
    assert "get_track(" not in source
    assert "mocap.subject(" not in source
    assert "mocap.segment(" not in source
    assert "marker_positions(" not in source
    assert "patch_points(" not in source
    assert 'left_shoe_track = ' not in source
