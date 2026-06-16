import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

from retarget.core import MarkerId, Vec3

MM_PER_METER = 1000


def read_marker_positions_from_vsk[T: MarkerId](
    vsk_file: Path | str,
    marker_type: type[T]
) -> dict[T, Vec3]:
    """
    Read the marker positions from a VSK file and return a dictionary of marker positions.

    Args:
        vsk_file: The path to the VSK file.
        marker_type: The vocabulary of marker IDs.

    Returns:
        A dictionary of marker positions keyed by marker ID.
    """
    tree = ET.parse(vsk_file)
    root = tree.getroot()

    marker_positions: dict[T, np.ndarray] = {}

    for marker in root.find("MarkerSet/Markers"):
        attrib = marker.attrib

        name = attrib["NAME"]
        position = np.array(list(map(float, attrib["POSITION"].split(" ")))) / MM_PER_METER
        # radius = float(attrib["RADIUS"]) / MM_PER_METER
        # segment = attrib["SEGMENT"]

        if name not in marker_type:
            raise ValueError(f"Marker {name} from {vsk_file} not found in vocabulary {marker_type.__name__}")

        marker_positions[marker_type(name)] = position
    
    # Ensure that all markers in the vocabulary are present
    missing_markers = set(marker_type) - set(marker_positions.keys())
    if missing_markers:
        raise ValueError(f"Missing markers in {vsk_file}: {missing_markers}")

    return marker_positions