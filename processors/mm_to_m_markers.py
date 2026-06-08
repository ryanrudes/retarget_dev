from ros2_unbag.core.processors.base import Processor


def _get_nested_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


@Processor(["vicon_bridge/msg/Markers"], ["mm_to_m_translations"])
def mm_to_m_translations(
    msg,
    markers_field: str = "markers",
    translation_field: str = "translation",
    scale: str = "1000",
):
    """
    Convert marker translations from millimeters to meters.

    Args:
        markers_field:
            Attribute path containing the marker list. Usually "markers".

        translation_field:
            Attribute path on each marker with x/y/z fields. Usually "translation".

        scale:
            Divisor applied to x, y, and z. Default "1000" converts mm to m.
    """
    divisor = float(scale)
    markers = _get_nested_attr(msg, markers_field)

    for marker in markers:
        translation = _get_nested_attr(marker, translation_field)
        translation.x /= divisor
        translation.y /= divisor
        translation.z /= divisor

    return msg
