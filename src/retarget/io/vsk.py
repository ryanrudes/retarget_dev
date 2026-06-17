"""Read segment-frame marker positions from Vicon ``.vsk`` skeleton files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from retarget.core.types import Vec3

MM_PER_METER = 1000


def read_marker_positions_from_vsk(vsk_file: Path | str) -> dict[str, Vec3]:
    """Read marker positions from a VSK file, keyed by the marker's Vicon name.

    Positions are converted from millimeters to meters. Callers map these
    external Vicon names onto authored markers via ``Marker.vicon_name``.
    """
    tree = ET.parse(vsk_file)
    root = tree.getroot()

    marker_positions: dict[str, Vec3] = {}
    markers = root.find("MarkerSet/Markers")
    if markers is None:
        raise ValueError(f"No MarkerSet/Markers found in {vsk_file}")

    for marker in markers:
        attrib = marker.attrib
        name = attrib["NAME"]
        position = (
            np.array(list(map(float, attrib["POSITION"].split(" ")))) / MM_PER_METER
        )
        marker_positions[name] = position

    return marker_positions
