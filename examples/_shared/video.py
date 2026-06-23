"""Real-time video of the contact timeline(s) + 3D geometry inspection(s).

Each experiment becomes one row: an animated 3D geometry view (foot/board
markers, the fitted sole & board patches with their outward normals, and the
z=0 ground plane) next to its contact timeline. The timeline is drawn once in
full with a vertical cursor that sweeps across in sync with the geometry --
exactly how a scrubber moves in video-editing software.

Playback is real time: the global clock runs from 0 to the longest capture, and
every panel samples the data frame nearest the current time (holding its last
frame once its own capture ends). The 3D axes get fixed, cubic limits computed
over the whole trajectory, so nothing rescales mid-playback.

One experiment -> an individual video; several -> a stacked, directly-comparable
merged video.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import patheffects as path_effects
from matplotlib.animation import FuncAnimation, PillowWriter, writers
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from .console import console
from .contact_timeline import TimelineSpec, _draw_timeline

NORMAL_SCALE = 0.04  # metres; length of the drawn outward-normal stubs
_LIMIT_PAD = 1.05  # inflate the cubic 3D limits slightly so nothing touches the edge

# One demo panel (geometry column + timeline column) in inches, and the screen-friendly
# aspect we tile panels toward so many demos read as a compact grid, not a long column.
_PANEL_W = 22.0
_PANEL_H = 6.2
_TARGET_ASPECT = 16.0 / 9.0
# The geometry block is a 3x4 grid of unit-square cells: the isometric view spans the top
# 3x3 (a 3-unit square) and the three orthographic views are 1-unit squares in the bottom
# row. Sizing the column to 3/4 of the panel height makes every cell exactly square.
_GEOM_W = _PANEL_H * 3.0 / 4.0
_ISO_ZOOM = 0.95  # keep the projected cube inside its axes rect (3D draws unclipped; >1 spills)

# The three orthographic side projections drawn under each isometric view, as
# (horizontal axis, vertical axis, label) with axes indexed x=0, y=1, z=2.
_ORTHO_VIEWS = ((0, 1, "top xy"), (0, 2, "front xz"), (1, 2, "side yz"))


@dataclass
class _Geom3DSeries:
    """Per-frame world geometry plus fixed cubic axis limits for one experiment."""

    foot: npt.NDArray[np.float64]  # (T, Nf, 3) foot markers
    deck: npt.NDArray[np.float64]  # (T, Nd, 3) board markers
    sole_corners: npt.NDArray[np.float64]  # (T, 4, 3)
    surf_corners: npt.NDArray[np.float64]  # (T, 4, 3)
    sole_origin: npt.NDArray[np.float64]  # (T, 3)
    sole_normal: npt.NDArray[np.float64]  # (T, 3)
    surf_origin: npt.NDArray[np.float64]  # (T, 3)
    surf_normal: npt.NDArray[np.float64]  # (T, 3)
    ground: npt.NDArray[np.float64]  # (4, 3) fixed z=0 rectangle
    lo: npt.NDArray[np.float64]  # (3,) cubic lower limit
    hi: npt.NDArray[np.float64]  # (3,) cubic upper limit


def _marker_series(segment: Any) -> npt.NDArray[np.float64]:
    # modeled=True -> clean rigid-body world positions over every frame.
    names = list(segment.markers.keys())
    return np.stack(
        [np.asarray(segment.markers[name].positions(modeled=True), dtype=np.float64) for name in names],
        axis=1,
    )


def _build_geom3d_series(spec: TimelineSpec) -> _Geom3DSeries:
    mocap = spec.mocap
    shoe = mocap.subjects["left_foot"].segments["shoe"]
    board = mocap.subjects["balance_board"].segments["board"]
    sole = shoe.patches["sole"]
    surface = board.patches["surface"]

    foot = _marker_series(shoe)
    deck = _marker_series(board)
    sole_corners = np.asarray(sole.boundary_points(), dtype=np.float64)
    surf_corners = np.asarray(surface.boundary_points(), dtype=np.float64)
    sole_origin = np.asarray(sole.points(), dtype=np.float64)
    sole_normal = np.asarray(sole.normals(), dtype=np.float64)
    surf_origin = np.asarray(surface.points(), dtype=np.float64)
    surf_normal = np.asarray(surface.normals(), dtype=np.float64)

    # Fixed cubic limits over the WHOLE trajectory, so the view never rescales.
    moving = np.concatenate(
        [foot.reshape(-1, 3), deck.reshape(-1, 3), sole_corners.reshape(-1, 3), surf_corners.reshape(-1, 3)],
        axis=0,
    )
    mins = np.nanmin(moving, axis=0)
    maxs = np.nanmax(moving, axis=0)
    ground = np.array(
        [[mins[0], mins[1], 0.0], [maxs[0], mins[1], 0.0], [maxs[0], maxs[1], 0.0], [mins[0], maxs[1], 0.0]]
    )
    allpts = np.concatenate([moving, ground], axis=0)
    mins = np.nanmin(allpts, axis=0)
    maxs = np.nanmax(allpts, axis=0)
    center = (mins + maxs) / 2.0
    half = max(float((maxs - mins).max()) / 2.0, 1e-3) * _LIMIT_PAD
    return _Geom3DSeries(
        foot=foot,
        deck=deck,
        sole_corners=sole_corners,
        surf_corners=surf_corners,
        sole_origin=sole_origin,
        sole_normal=sole_normal,
        surf_origin=surf_origin,
        surf_normal=surf_normal,
        ground=ground,
        lo=center - half,
        hi=center + half,
    )


def _normal_segment(origin: npt.NDArray[np.float64], normal: npt.NDArray[np.float64]) -> tuple[list, list, list]:
    unit = normal / (np.linalg.norm(normal) or 1.0)
    tip = origin + NORMAL_SCALE * unit
    return [origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]]


def _init_geom3d(ax: Any, s: _Geom3DSeries) -> dict[str, Any]:
    foot0, deck0 = s.foot[0], s.deck[0]
    foot_sc = ax.scatter(foot0[:, 0], foot0[:, 1], foot0[:, 2], c="tab:blue", s=30, label="foot markers")
    deck_sc = ax.scatter(deck0[:, 0], deck0[:, 1], deck0[:, 2], c="tab:purple", s=30, marker="s", label="board markers")

    sole_rect = Poly3DCollection([s.sole_corners[0]], facecolor="tab:blue", alpha=0.35, edgecolor="tab:blue", linewidths=2)
    surf_rect = Poly3DCollection([s.surf_corners[0]], facecolor="tab:purple", alpha=0.35, edgecolor="tab:purple", linewidths=2)
    ground_rect = Poly3DCollection([s.ground], facecolor="tab:green", alpha=0.15, edgecolor="tab:green", linewidths=1.5, linestyle="--")
    ax.add_collection3d(sole_rect)
    ax.add_collection3d(surf_rect)
    ax.add_collection3d(ground_rect)

    (sole_n,) = ax.plot(*_normal_segment(s.sole_origin[0], s.sole_normal[0]), color="tab:blue", lw=2)
    (surf_n,) = ax.plot(*_normal_segment(s.surf_origin[0], s.surf_normal[0]), color="tab:purple", lw=2)

    ax.set_xlim(s.lo[0], s.hi[0])
    ax.set_ylim(s.lo[1], s.hi[1])
    ax.set_zlim(s.lo[2], s.hi[2])
    ax.set_box_aspect((1, 1, 1), zoom=_ISO_ZOOM)
    # A 3D axis draws its tick labels and axis labels *outside* its rectangle -- the z-axis
    # numbers in particular jut out to the right. That overflow is what made the isometric
    # view look wider than the three orthographic squares below it and bleed into the
    # timeline. Strip the tick labels, axis labels, and tick marks (keep the gridded panes
    # for depth), so the iso's drawn extent stays within its cells and lines up with them.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.legend(loc="upper left", fontsize=5, framealpha=0.5, handletextpad=0.3, borderpad=0.25, labelspacing=0.2)
    return {"foot": foot_sc, "deck": deck_sc, "sole": sole_rect, "surf": surf_rect, "sole_n": sole_n, "surf_n": surf_n}


def _update_geom3d(art: dict[str, Any], s: _Geom3DSeries, i: int) -> None:
    art["foot"]._offsets3d = (s.foot[i, :, 0], s.foot[i, :, 1], s.foot[i, :, 2])
    art["deck"]._offsets3d = (s.deck[i, :, 0], s.deck[i, :, 1], s.deck[i, :, 2])
    art["sole"].set_verts([s.sole_corners[i]])
    art["surf"].set_verts([s.surf_corners[i]])
    xs, ys, zs = _normal_segment(s.sole_origin[i], s.sole_normal[i])
    art["sole_n"].set_data(xs, ys)
    art["sole_n"].set_3d_properties(zs)
    xs, ys, zs = _normal_segment(s.surf_origin[i], s.surf_normal[i])
    art["surf_n"].set_data(xs, ys)
    art["surf_n"].set_3d_properties(zs)


def _normal_2d(origin: npt.NDArray[np.float64], normal: npt.NDArray[np.float64], a: int, b: int) -> tuple[list, list]:
    unit = normal / (np.linalg.norm(normal) or 1.0)
    tip = origin + NORMAL_SCALE * unit
    return [origin[a], tip[a]], [origin[b], tip[b]]


def _init_ortho(ax: Any, s: _Geom3DSeries, a: int, b: int, label: str) -> dict[str, Any]:
    """A flat (a, b)-plane projection of the scene, with fixed square limits."""
    foot0, deck0 = s.foot[0][:, [a, b]], s.deck[0][:, [a, b]]
    foot_sc = ax.scatter(foot0[:, 0], foot0[:, 1], c="tab:blue", s=8)
    deck_sc = ax.scatter(deck0[:, 0], deck0[:, 1], c="tab:purple", s=8, marker="s")

    sole_poly = MplPolygon(s.sole_corners[0][:, [a, b]], closed=True, facecolor="tab:blue", alpha=0.35, edgecolor="tab:blue", lw=1.0)
    surf_poly = MplPolygon(s.surf_corners[0][:, [a, b]], closed=True, facecolor="tab:purple", alpha=0.35, edgecolor="tab:purple", lw=1.0)
    ax.add_patch(sole_poly)
    ax.add_patch(surf_poly)

    # The ground is the static z=0 rectangle; projected it is a rectangle (top
    # view) or a flat line (front/side views). Drawn once, never updated.
    ground = np.vstack([s.ground[:, [a, b]], s.ground[:1, [a, b]]])
    ax.plot(ground[:, 0], ground[:, 1], color="tab:green", ls="--", lw=1.0, alpha=0.7)

    (sole_n,) = ax.plot(*_normal_2d(s.sole_origin[0], s.sole_normal[0], a, b), color="tab:blue", lw=1.3)
    (surf_n,) = ax.plot(*_normal_2d(s.surf_origin[0], s.surf_normal[0], a, b), color="tab:purple", lw=1.3)

    # Fixed limits taken from the same cube as the isometric view, equal aspect so
    # the projection stays undistorted and never rescales mid-playback.
    # The cube limits are equal on every axis, so equal aspect fills the square cell exactly.
    # Drop ticks/axis labels entirely and label the view with a tiny in-axes tag, so the three
    # squares sit flush against each other and the iso with no decoration eating into them.
    ax.set_xlim(s.lo[a], s.hi[a])
    ax.set_ylim(s.lo[b], s.hi[b])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.035, 0.965, label, transform=ax.transAxes, fontsize=6, va="top", ha="left")
    return {"foot": foot_sc, "deck": deck_sc, "sole": sole_poly, "surf": surf_poly, "sole_n": sole_n, "surf_n": surf_n}


def _update_ortho(art: dict[str, Any], s: _Geom3DSeries, a: int, b: int, i: int) -> None:
    art["foot"].set_offsets(s.foot[i][:, [a, b]])
    art["deck"].set_offsets(s.deck[i][:, [a, b]])
    art["sole"].set_xy(s.sole_corners[i][:, [a, b]])
    art["surf"].set_xy(s.surf_corners[i][:, [a, b]])
    art["sole_n"].set_data(*_normal_2d(s.sole_origin[i], s.sole_normal[i], a, b))
    art["surf_n"].set_data(*_normal_2d(s.surf_origin[i], s.surf_normal[i], a, b))


@dataclass
class _Panel:
    name: str
    series: _Geom3DSeries
    iso_artists: dict[str, Any]
    ortho_artists: list[tuple[dict[str, Any], int, int]]
    t: npt.NDArray[np.float64]
    cursors: list[Any]

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def dynamic_artists(self) -> list[Any]:
        """Every artist that moves per frame -- the only thing re-drawn while blitting.

        Static content (axes, labels, legend, ground plane, the whole timeline plot)
        is captured once into the background, so each frame redraws just these.
        """
        arts: list[Any] = [
            self.iso_artists[k] for k in ("foot", "deck", "sole", "surf", "sole_n", "surf_n")
        ]
        for art, _, _ in self.ortho_artists:
            arts += [art[k] for k in ("foot", "deck", "sole", "surf", "sole_n", "surf_n")]
        arts += self.cursors
        return arts

    def update(self, global_t: float) -> None:
        clock = min(global_t, self.duration)
        i = int(np.abs(self.t - clock).argmin())
        _update_geom3d(self.iso_artists, self.series, i)
        for art, a, b in self.ortho_artists:
            _update_ortho(art, self.series, a, b, i)
        for cursor in self.cursors:
            cursor.set_xdata([clock, clock])


def _add_panel(container: Any, name: str, spec: TimelineSpec, *, announce: bool = True) -> _Panel:
    # The geometry column is kept narrow: its content (a square isometric view above a row
    # of three small orthographic squares) is height-bound, so a wide column would just
    # frame it in whitespace. Narrowing it also hands the extra width to the timeline.
    # Geometry block: a 3x3 isometric square above a row of three 1x1 orthographic squares.
    # The axes are placed by hand (in column-relative coords) rather than via a gridspec, so
    # their rectangles line up *exactly*: the iso spans the full width and the three orthos
    # tile the bottom quarter at width 1/3 each -- so the orthos together are precisely as
    # wide as the iso. Manually-placed axes are ignored by constrained layout, so they keep
    # these positions. The column is sized to 3/4 of the panel height, making every cell square.
    geom_col, timeline_col = container.subfigures(1, 2, width_ratios=[_GEOM_W, _PANEL_W - _GEOM_W])
    ax_iso = geom_col.add_axes((0.0, 0.25, 1.0, 0.75), projection="3d")
    ortho_axes = [geom_col.add_axes((j / 3.0, 0.0, 1.0 / 3.0, 0.25)) for j in range(3)]

    series = _build_geom3d_series(spec)
    iso_artists = _init_geom3d(ax_iso, series)
    ortho_artists = [
        (_init_ortho(ax, series, a, b, label), a, b)
        for ax, (a, b, label) in zip(ortho_axes, _ORTHO_VIEWS)
    ]

    panel = _draw_timeline(timeline_col, spec, f"{name} -- contact timeline")
    # A white halo keeps the cursor visible across the green/red heat strips it sweeps.
    cursor_halo = [path_effects.withStroke(linewidth=2.6, foreground="white")]
    cursors = [
        ax.axvline(panel.t[0], color="k", lw=1.2, zorder=5, path_effects=cursor_halo)
        for ax in panel.axes
    ]
    if announce:  # parallel workers rebuild the same panels silently -- only announce once
        console.print(f"[bold]{name}[/]: {panel.summary}")
    return _Panel(
        name=name,
        series=series,
        iso_artists=iso_artists,
        ortho_artists=ortho_artists,
        t=panel.t,
        cursors=cursors,
    )


def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


# Each worker rebuilds the figure once (~0.3s) and runs its own ffmpeg encoder, so beyond a
# handful of workers the per-worker overhead and encoder contention outweigh the gain --
# measured throughput plateaus around 6 and regresses past 8. Keep the fan-out modest.
_MIN_FRAMES_PER_WORKER = 48  # below this a worker's fixed setup dominates its render time
_MAX_WORKERS = 8


def _grid_shape(n: int, panel_aspect: float, *, target: float = _TARGET_ASPECT, empty_penalty: float = 0.15) -> tuple[int, int]:
    """Rows x cols tiling whose overall aspect sits closest to ``target`` (a 16:9 screen).

    Each panel is very wide, so a single column of many demos comes out absurdly tall.
    We search every column count, score it by how far the resulting figure aspect is from
    ``target`` (log-ratio, so 2x and 0.5x are penalised equally) plus a small charge per
    empty cell, and break ties toward more columns (landscape reads better than portrait).
    """
    best_key: tuple[float, int] | None = None
    best = (n, 1)
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        aspect = panel_aspect * cols / rows
        score = abs(math.log(aspect / target)) + empty_penalty * (rows * cols - n)
        key = (score, -cols)
        if best_key is None or key < best_key:
            best_key, best = key, (rows, cols)
    return best


def _build_figure(
    named_specs: Sequence[tuple[str, TimelineSpec]], dpi: int, *, announce: bool = True
) -> tuple[Any, list[_Panel]]:
    """Build the demo grid and freeze its layout, ready for blitted playback."""
    n = len(named_specs)
    rows, cols = _grid_shape(n, _PANEL_W / _PANEL_H)
    fig = plt.figure(figsize=(_PANEL_W * cols, _PANEL_H * rows), layout="constrained")
    fig.set_dpi(dpi)
    engine = fig.get_layout_engine()
    if engine is not None:  # tighten the gaps so panels are not adrift in whitespace
        engine.set(w_pad=0.03, h_pad=0.03, wspace=0.015, hspace=0.015)
    cells = fig.subfigures(rows, cols, squeeze=False).ravel()
    panels = [_add_panel(cell, name, spec, announce=announce) for cell, (name, spec) in zip(cells, named_specs)]
    # Let constrained layout place everything once, then freeze it: re-solving the
    # layout every frame is slow and makes panels jitter during playback.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, panels


def _ffmpeg_encode_command(width: int, height: int, fps: int, out_path: Path) -> list[str]:
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
        str(out_path),
    ]


def _blit_frames_to_ffmpeg(
    fig: Any,
    panels: list[_Panel],
    out_path: Path,
    *,
    fps: int,
    frames: range,
    on_frame: Any = None,
) -> None:
    """Blit dynamic artists over a frozen background and pipe ``frames`` raw to ffmpeg.

    The expensive part of an animation is re-rendering static content (axes, legends,
    the entire timeline plot) every frame. Here the static scene is drawn *once* into a
    cached background; each frame only restores it and re-draws the handful of moving
    artists, then the finished RGBA buffer is streamed to ffmpeg. This skips both
    matplotlib's per-frame full redraw and ``savefig`` re-rendering inside the writer.
    """
    dynamic = [art for panel in panels for art in panel.dynamic_artists()]
    for art in dynamic:
        art.set_animated(True)  # excluded from the background draw; blitted on top per frame

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    height, width = np.asarray(fig.canvas.buffer_rgba()).shape[:2]

    proc = subprocess.Popen(_ffmpeg_encode_command(width, height, fps, out_path), stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        for panel in panels:
            panel.update(frame / fps)
        fig.canvas.restore_region(background)
        for art in dynamic:
            # draw_artist bypasses Axes3D.draw, which is what normally re-runs the 3D->2D
            # projection, so a 3D artist would otherwise stay frozen at its frame-0 screen
            # position. Re-project it here (the fixed camera makes this cheap and correct).
            project = getattr(art, "do_3d_projection", None)
            if project is not None:
                project()
            art.axes.draw_artist(art)
        fig.canvas.blit(fig.bbox)
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if on_frame is not None:
            on_frame()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg exited with status {proc.returncode}")


@dataclass
class _SegmentJob:
    """Render inputs shared with fork()ed workers via inherited memory (never pickled).

    ``progress`` is a shared ``mp.Value`` counter every worker bumps once per encoded
    frame, so the parent can show a true frames-completed bar across all workers rather
    than a coarse one-tick-per-worker bar.
    """

    named_specs: Sequence[tuple[str, TimelineSpec]]
    fps: int
    dpi: int
    progress: Any  # mp.Value("i"); created with the fork context so workers inherit it


# Set in the parent immediately before the fork pool starts; each worker inherits it as a
# copy-on-write global. Only tiny (start, end, path) tasks cross the process boundary, so the
# unpicklable bound specs never have to be pickled -- fork's memory inheritance carries them.
_ACTIVE_JOB: _SegmentJob | None = None


def _render_segment(task: tuple[int, int, Path]) -> Path:
    """Worker: build a private figure from the inherited job and render one frame range."""
    start, end, seg_path = task
    job = _ACTIVE_JOB
    assert job is not None, "worker started without an active render job"
    fig, panels = _build_figure(job.named_specs, job.dpi, announce=False)
    counter = job.progress

    def bump() -> None:
        with counter.get_lock():
            counter.value += 1

    _blit_frames_to_ffmpeg(fig, panels, seg_path, fps=job.fps, frames=range(start, end), on_frame=bump)
    plt.close(fig)
    return seg_path


def _chunk_ranges(num_frames: int, workers: int) -> list[tuple[int, int]]:
    """Split ``[0, num_frames)`` into ``workers`` contiguous, near-equal, non-empty ranges."""
    workers = max(1, min(workers, num_frames))
    bounds = np.linspace(0, num_frames, workers + 1).round().astype(int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(workers) if bounds[i + 1] > bounds[i]]


def _concat_segments(seg_paths: list[Path], out_path: Path) -> None:
    """Losslessly join per-worker segments (identical encoder settings) into one mp4."""
    listing = "\n".join(f"file '{p.resolve().as_posix()}'" for p in seg_paths)
    list_file = out_path.parent / f".{out_path.stem}_concat.txt"
    list_file.write_text(listing)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(list_file), "-c", "copy", str(out_path)],
            check=True,
        )
    finally:
        list_file.unlink(missing_ok=True)


def _render_parallel(
    named_specs: Sequence[tuple[str, TimelineSpec]],
    out_path: Path,
    *,
    fps: int,
    num_frames: int,
    dpi: int,
    workers: int,
) -> None:
    """Render contiguous frame ranges across forked workers, then concatenate the segments."""
    global _ACTIVE_JOB
    chunks = _chunk_ranges(num_frames, workers)
    seg_dir = out_path.parent / f".{out_path.stem}_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(start, end, seg_dir / f"seg_{i:04d}.mp4") for i, (start, end) in enumerate(chunks)]

    ctx = mp.get_context("fork")
    counter = ctx.Value("i", 0)  # shared frame tally; workers inherit it via fork
    _ACTIVE_JOB = _SegmentJob(named_specs=named_specs, fps=fps, dpi=dpi, progress=counter)
    try:
        with ctx.Pool(len(tasks)) as pool, _progress() as progress:
            bar = progress.add_task(f"rendering ({len(tasks)} workers)", total=num_frames)
            result = pool.map_async(_render_segment, tasks)
            while not result.ready():
                progress.update(bar, completed=counter.value)
                result.wait(0.2)
            result.get()  # surface any worker exception
            progress.update(bar, completed=num_frames)
    finally:
        _ACTIVE_JOB = None

    _concat_segments([path for _, _, path in tasks], out_path)
    shutil.rmtree(seg_dir, ignore_errors=True)


def _render_pillow_fallback(
    fig: Any, panels: list[_Panel], out_path: Path, *, fps: int, num_frames: int
) -> None:
    """Slow gif fallback for when ffmpeg is unavailable (still a full redraw per frame)."""
    def update(frame: int) -> list[Any]:
        for panel in panels:
            panel.update(frame / fps)
        return []

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000.0 / fps, blit=False)
    with _progress() as progress:
        task = progress.add_task("rendering", total=num_frames)
        anim.save(
            out_path,
            writer=PillowWriter(fps=fps),
            dpi=120,
            progress_callback=lambda frame, total: progress.update(task, completed=frame + 1),
        )


def _worker_count(num_frames: int) -> int:
    """How many fork workers are worth using: bounded by cores, frames, and a cap of one core."""
    if not hasattr(os, "fork"):
        return 1
    cores = os.cpu_count() or 1
    return max(1, min(cores, _MAX_WORKERS, num_frames // _MIN_FRAMES_PER_WORKER))


def render_video(
    named_specs: Sequence[tuple[str, TimelineSpec]],
    output_path: str | Path,
    *,
    fps: int = 30,
    dpi: int = 120,
) -> Path:
    """Render the (individual or merged) real-time video and return the saved path.

    With ffmpeg available the render is blitted (static scene drawn once, only moving
    artists redrawn) and fanned out across CPU cores: each worker renders a contiguous
    range of frames to its own segment, and the segments are joined losslessly.
    """
    # Render to file off-screen with Agg. Critically on macOS this must happen *before*
    # any figure is built: the interactive Cocoa backend initialises AppKit, and forking
    # a process once AppKit is live deadlocks the workers (they hang at frame 0). Agg never
    # touches Cocoa, so the parallel fork pool is safe. Saving never needs a GUI anyway.
    plt.switch_backend("Agg")
    fig, panels = _build_figure(named_specs, dpi, announce=True)
    duration = max(panel.duration for panel in panels)
    num_frames = int(round(duration * fps)) + 1

    use_ffmpeg = writers.is_available("ffmpeg")
    out_path = Path(output_path).with_suffix(".mp4" if use_ffmpeg else ".gif")
    workers = _worker_count(num_frames) if use_ffmpeg else 1
    detail = f", {workers} workers" if workers > 1 else ""
    console.print(f"rendering [bold]{num_frames}[/] frames ({duration:.1f}s @ {fps} fps{detail}) -> {out_path}")

    if use_ffmpeg and workers > 1:
        # Workers build their own figures; the parent's is only needed for the frame count.
        plt.close(fig)
        _render_parallel(named_specs, out_path, fps=fps, num_frames=num_frames, dpi=dpi, workers=workers)
    elif use_ffmpeg:
        with _progress() as progress:
            task = progress.add_task("rendering", total=num_frames)
            _blit_frames_to_ffmpeg(
                fig, panels, out_path, fps=fps, frames=range(num_frames),
                on_frame=lambda: progress.advance(task),
            )
        plt.close(fig)
    else:
        _render_pillow_fallback(fig, panels, out_path, fps=fps, num_frames=num_frames)
        plt.close(fig)

    console.print(f"[green]saved[/] {out_path}")
    return out_path
