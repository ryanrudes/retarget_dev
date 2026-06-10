from __future__ import annotations

import pytest

from demo_facade import GroundEstimationDemo, GroundEstimationDemoView
from demo_specs import GroundEstimationTrackId
from retarget.demo.demo import Demonstration
from conftest import make_mocap_track


def test_facade_wraps_demonstration() -> None:
    mocap = make_mocap_track()
    inner = Demonstration(tracks={GroundEstimationTrackId.MOCAP: mocap})
    demo = GroundEstimationDemo.wrap(inner)
    assert demo.mocap is mocap
    assert demo.track(GroundEstimationTrackId.MOCAP) is mocap


def test_facade_slice_returns_view() -> None:
    demo = GroundEstimationDemo.wrap(
        Demonstration(tracks={GroundEstimationTrackId.MOCAP: make_mocap_track()})
    )
    clip = demo.slice_time(0.0, 0.2)
    assert isinstance(clip, GroundEstimationDemoView)
    assert len(clip.mocap.timestamps) == 2


def test_facade_optional_track_properties_raise_clear_errors() -> None:
    demo = GroundEstimationDemo.wrap(
        Demonstration(tracks={GroundEstimationTrackId.MOCAP: make_mocap_track()})
    )
    with pytest.raises(KeyError, match="No video track"):
        _ = demo.video
    with pytest.raises(KeyError, match="No SMPL track"):
        _ = demo.smpl
    with pytest.raises(KeyError, match="No contact track"):
        _ = demo.contacts
