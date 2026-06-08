"""Load all vicon marker processors for use with a single --use-processor flag."""

import importlib.util
from pathlib import Path

for _name in (
    "drop_empty_name_markers",
    "mm_to_m_markers",
    "filter_non_occluded_markers",
):
    _path = Path(__file__).with_name(f"{_name}.py")
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
