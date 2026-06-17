from __future__ import annotations

import ast
import sys
from pathlib import Path

from conftest import make_mocap_track
from retarget.demo import Demonstration

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "process_mocap_data"
BACKEND_SPECS_DIR = EXAMPLE_ROOT / "backend_specs"
BACKEND_SCENE = BACKEND_SPECS_DIR / "vicon_scene.py"
BACKEND_LOADER = BACKEND_SPECS_DIR / "ground_estimation_loader.py"
NEW_API_EXAMPLE = EXAMPLE_ROOT / "new_api_example.py"
RUN_DEMO_TRACK_WORKFLOW = EXAMPLE_ROOT / "run_demo_track_workflow.py"
RUN_CORE_GEOMETRY_BASICS = EXAMPLE_ROOT / "run_core_geometry_basics.py"

if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from backend_specs import ground_estimation_loader  # noqa: E402

_FORBIDDEN_TRANSITIONAL_NAMES = (
    "GroundEstimationTrackId",
    "TypedDemonstration",
    "typed_tracks",
    "get_track(",
    "._tracks",
    "._subject(",
    "._segment(",
    "._marker_positions(",
    "._patch_points(",
    "._patch_normals(",
    "._patch_contacts(",
)


def test_backend_modules_exist_and_vocab_modules_removed() -> None:
    assert BACKEND_SCENE.is_file()
    assert BACKEND_LOADER.is_file()
    assert not (BACKEND_SPECS_DIR / "vicon_vocab.py").exists()
    assert not (EXAMPLE_ROOT / "demo_vocab.py").exists()


def test_loader_returns_typed_demonstration_annotation() -> None:
    tree = ast.parse(BACKEND_LOADER.read_text())
    loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_ground_estimation_demo"
    )
    assert loader.returns is not None
    assert ast.unparse(loader.returns) == "Demonstration[GroundEstimationTracks]"


def test_load_ground_estimation_demo_builds_typed_demonstration(monkeypatch) -> None:
    track = make_mocap_track()

    def fake_from_unbagged(cls, root, subjects, **kwargs):
        return track

    monkeypatch.setattr(
        ground_estimation_loader.MocapTrack,
        "from_unbagged",
        classmethod(fake_from_unbagged),
    )
    demo = ground_estimation_loader.load_ground_estimation_demo(Path("dummy"))
    assert isinstance(demo, Demonstration)
    assert isinstance(demo.tracks["mocap"], type(track))


def test_examples_are_free_of_transitional_names() -> None:
    offenders: dict[str, list[str]] = {}
    for path in EXAMPLE_ROOT.rglob("*.py"):
        source = path.read_text()
        found = [name for name in _FORBIDDEN_TRANSITIONAL_NAMES if name in source]
        if found:
            offenders[path.name] = found
    assert offenders == {}


def test_new_api_example_is_typed_first() -> None:
    source = NEW_API_EXAMPLE.read_text()
    required = (
        "bind_scene(subjects)",
        "region=RectangularRegion(",
        "toe_contact=Patch(",
        'label="toe_contact_display"',
        "class GroundEstimationSubjects(Subjects):",
        "class GroundEstimationTracks(Tracks):",
        "Demonstration(GroundEstimationTracks(mocap=",
        'demo.tracks["mocap"]',
        'mocap.subjects["left_shoe"]',
        'left_shoe.segments["shoe"]',
        'shoe.markers["heel"]',
        'shoe.patches["sole"]',
        ".positions()",
        ".points()",
        "MocapTrack.from_unbagged(",
        "UnbaggedDirectory(",
        "UNBAGGED_DIR.is_dir()",
        "Skipping bag-backed demo",
    )
    for fragment in required:
        assert fragment in source, fragment


def test_backend_examples_use_typed_track_access() -> None:
    for path in (RUN_DEMO_TRACK_WORKFLOW, RUN_CORE_GEOMETRY_BASICS):
        source = path.read_text()
        assert 'demo.tracks["mocap"]' in source
        assert 'subjects["left_shoe"]' in source
        assert 'segments["shoe"]' in source
