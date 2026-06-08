from ros2_unbag.core.processors.base import Processor


def _get_nested_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _set_nested_attr(obj, path: str, value):
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


@Processor(["vicon_bridge/msg/Markers"], ["drop_empty_name_markers"])
def drop_empty_name_markers(
    msg,
    markers_field: str = "markers",
    marker_name_field: str = "marker_name",
    subject_name_field: str = "subject_name",
    segment_name_field: str = "segment_name",
):
    """
    Remove markers whose marker_name, subject_name, and segment_name are all empty.

    Args:
        markers_field:
            Attribute path containing the marker list. Usually "markers".

        marker_name_field:
            Marker name attribute. Usually "marker_name".

        subject_name_field:
            Subject name attribute. Usually "subject_name".

        segment_name_field:
            Segment name attribute. Usually "segment_name".
    """
    markers = _get_nested_attr(msg, markers_field)
    kept = [
        marker
        for marker in markers
        if not (
            getattr(marker, marker_name_field) == ""
            and getattr(marker, subject_name_field) == ""
            and getattr(marker, segment_name_field) == ""
        )
    ]
    _set_nested_attr(msg, markers_field, kept)
    return msg
