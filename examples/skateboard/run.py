from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scene import get_demo

demo = get_demo()

mocap = demo.tracks["mocap"]

left_foot = mocap.subjects["left_foot"]
left_shoe = left_foot.segments["shoe"]

sole = left_shoe.patches["sole"]
sole_points = sole.points()
sole_unit_normals = sole.normals()

# `sole.region` is statically a `RectangularRegion` (the schema declares
# `sole: Patch[RectangularRegion]`), so width/height read with no isinstance check.
width = sole.region.width
height = sole.region.height

# The oriented rectangle comes straight from the library: `boundary_points()`
# transports the contact-region polygon into the world frame at every timestep.
corners_world = np.asarray(sole.boundary_points(), dtype=np.float64)  # (T, 4, 3)

origins = np.asarray(sole_points, dtype=np.float64)  # (T, 3)
normals = np.asarray(sole_unit_normals, dtype=np.float64)  # (T, 3)
num_frames = corners_world.shape[0]

# Square axes: equal aspect with shared limits spanning the full trajectory.
all_points = corners_world.reshape(-1, 3)
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
center = (mins + maxs) / 2.0
half_span = float((maxs - mins).max()) / 2.0
half_span = max(half_span, 1e-3) * 1.1
lo = center - half_span
hi = center + half_span

normal_length = 0.5 * min(width, height)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

rect = Poly3DCollection([corners_world[0]], facecolor="tab:blue", alpha=0.4, edgecolor="k")
ax.add_collection3d(rect)
(origin_dot,) = ax.plot([], [], [], "o", color="tab:red")
(normal_line,) = ax.plot([], [], [], "-", color="tab:green", linewidth=2)

ax.set_xlim(lo[0], hi[0])
ax.set_ylim(lo[1], hi[1])
ax.set_zlim(lo[2], hi[2])
ax.set_box_aspect((1, 1, 1))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


def update(frame: int) -> tuple[object, object, object]:
    rect.set_verts([corners_world[frame]])
    o = origins[frame]
    n = o + normal_length * normals[frame]
    origin_dot.set_data([o[0]], [o[1]])
    origin_dot.set_3d_properties([o[2]])
    normal_line.set_data([o[0], n[0]], [o[1], n[1]])
    normal_line.set_3d_properties([o[2], n[2]])
    ax.set_title(f"sole frame {frame} / {num_frames - 1}")
    return rect, origin_dot, normal_line


anim = FuncAnimation(fig, update, frames=num_frames, interval=33, blit=False)

plt.show()
