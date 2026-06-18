#!/usr/bin/env bash
#
# Unbag a Vicon ROS 2 bag to JSON (tf + markers), running the full export
# pipeline so you don't have to paste it every time.
#
# Usage (with the ros_env active -- see README):
#   scripts/unbag.sh <bag_name>
#
# <bag_name> is a directory under bags/ that holds <bag_name>_*.db3. Output goes
# to bags/<bag_name>/unbagged/. The built vicon_bridge install is sourced here for
# the vicon_bridge/Markers message type, so you only need ros2 (ros_env) on PATH.
set -euo pipefail

name="${1:?usage: scripts/unbag.sh <bag_name>   (a directory under bags/)}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# Source the built bridge so /vicon/markers (vicon_bridge/Markers) deserializes.
# colcon's setup.bash references unset vars (e.g. COLCON_TRACE), so relax nounset
# just for the source.
if [ -f rosws/install/setup.bash ]; then
  set +u
  # shellcheck disable=SC1091
  source rosws/install/setup.bash
  set -u
fi

shopt -s nullglob
db3=(bags/"$name"/*.db3)
shopt -u nullglob
if [ ${#db3[@]} -eq 0 ]; then
  echo "no .db3 found under bags/$name/" >&2
  exit 1
fi

ros2 unbag "${db3[0]}" \
  --output-dir "bags/$name/unbagged" \
  --export /tf:text/json@single_file \
  --export /vicon/markers:text/json@single_file \
  --use-processor ./processors/markers_preprocessors.py \
  --processing /vicon/markers:keep_non_occluded_markers:markers_field=markers,occluded_field=occluded,discard_eps=0.02 \
  --processing /vicon/markers:drop_empty_name_markers \
  --processing /vicon/markers:mm_to_m_translations \
  --resample /tf:nearest,0.02
