"""
Interactive viser tool for authoring a 3D pose trajectory for any object mesh.

Drag/rotate the object with the 3D gizmo, scrub the shared "Time (s)" slider to
where you want a waypoint, and click "Add Waypoint" to record it. Load an
existing trajectory (demo or previously exported) to view it alongside your
edit, scrub to a frame and click "Capture as Waypoint" to stage its pose onto
the gizmo as a starting point for a new waypoint. Export densifies the sparse
waypoints (PCHIP position / RotationSpline rotation) to FPS and saves the same
{"pos": (T,3), "rot": (T,4) xyzw, "fps": FPS} format consumed by
`phc/utils/traj_generator_3d.py`'s `real_traj` mode.

Note: exported rotation is used as an absolute world orientation after the initial
lift phase (not re-anchored like position), so author the first waypoint's
orientation close to the object's natural resting orientation (roughly identity).
"""

from __future__ import annotations

import sys
import os

sys.path.append(os.getcwd())

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import trimesh
import tyro
import viser

from phc.utils.object_assets import OBJECT_RECIPES, ObjectName
from phc.utils.object_traj_waypoints import (
    Waypoint,
    interpolate,
    load_trajectory,
    load_waypoints,
    save_waypoints,
    xyzw_to_wxyz,
)

FPS: float = 30.0
"""Fixed sample rate for preview, capture, and export. Embedded in exports so
loads can verify a file was authored at the same rate."""


@dataclass
class Config:
    object_name: ObjectName | None = None
    """Known object name (e.g. "hockey_stick") to look up mesh_path/mesh_scale defaults from
    phc.utils.object_assets.OBJECT_RECIPES. Required unless mesh_path and mesh_scale are both given."""
    output_path: Path | None = None
    """Where to save the densified {pos, rot, fps} trajectory. Defaults to
    data/custom_{object_name}_traj.pkl if object_name is set, else data/custom_object_traj.pkl."""
    mesh_path: Path | None = None
    """Mesh to display (trimesh-loadable .stl/.obj). Overrides the object_name preset if both
    given; required if object_name is None."""
    mesh_scale: float | None = None
    """Scale applied to the mesh. Overrides the object_name preset if both given; required if
    object_name is None."""
    port: int = 8080
    """Viser server port."""
    load_path: Path | None = None
    """Existing trajectory (dict {pos,rot[,fps]} or bare (T,3) pos-only) to load at startup."""


@dataclass
class EditState:
    waypoint: Waypoint | None = None


@dataclass
class PlayState:
    playing: bool = False


def time_gradient_colors(n: int) -> np.ndarray:
    """(N,3) uint8 colors fading blue (start) -> red (end), to show trajectory direction."""
    frac = np.linspace(0.0, 1.0, n)[:, None]
    start = np.array([60.0, 90.0, 255.0])
    end = np.array([255.0, 60.0, 60.0])
    return (start * (1 - frac) + end * frac).astype(np.uint8)


def main(cfg: Config) -> None:
    if cfg.object_name is not None:
        asset = OBJECT_RECIPES[cfg.object_name].asset
        mesh_path = cfg.mesh_path if cfg.mesh_path is not None else asset.mesh_path
        mesh_scale = cfg.mesh_scale if cfg.mesh_scale is not None else asset.mesh_scale
        output_path = (
            cfg.output_path
            if cfg.output_path is not None
            else Path(f"data/custom_{cfg.object_name}_traj.pkl")
        )
    else:
        assert cfg.mesh_path is not None and cfg.mesh_scale is not None, (
            "Must provide --object-name, or both --mesh-path and --mesh-scale"
        )
        mesh_path = cfg.mesh_path
        mesh_scale = cfg.mesh_scale
        output_path = cfg.output_path if cfg.output_path is not None else Path("data/custom_object_traj.pkl")

    sidecar_path = output_path.with_suffix(".waypoints.json")
    waypoints = load_waypoints(sidecar_path)

    mesh = trimesh.load_mesh(str(mesh_path))
    mesh.apply_scale(mesh_scale)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    server = viser.ViserServer(port=cfg.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=6, height=6, cell_size=0.5)

    gizmo = server.scene.add_transform_controls("/gizmo", scale=0.3)
    server.scene.add_mesh_simple(
        "/gizmo/mesh", vertices=vertices, faces=faces, color=(90, 200, 255)
    )
    # Marks the exact (0,0,0)-of-mesh point that "position" refers to, relative to the stick geometry.
    server.scene.add_icosphere("/gizmo/origin_marker", radius=0.02, color=(255, 0, 0))

    preview_mesh = server.scene.add_mesh_simple(
        "/preview/mesh",
        vertices=vertices,
        faces=faces,
        color=(255, 140, 0),
        opacity=0.4,
    )
    preview_marker = server.scene.add_icosphere(
        "/preview/origin_marker", radius=0.02, color=(255, 0, 0), opacity=0.4
    )

    if waypoints:
        gizmo.position = waypoints[0].position
        gizmo.wxyz = waypoints[0].wxyz

    edit_state = EditState()
    play_state = PlayState()
    loaded: dict = {"T": 0, "duration": 0.0}
    row_handles: list[viser.GuiInputHandle] = []

    with server.gui.add_folder("Waypoints"):
        commit_button = server.gui.add_button("Add Waypoint")
        cancel_button = server.gui.add_button("Cancel Edit")
        cancel_button.visible = False
        waypoint_list_folder = server.gui.add_folder("List")

    with server.gui.add_folder("Timeline"):
        time_slider = server.gui.add_slider(
            "Time (s)", min=0.0, max=0.01, step=1.0 / FPS, initial_value=0.0
        )
        play_button = server.gui.add_button("Play / Pause")
        duration_field = server.gui.add_number(
            "Duration (s)", initial_value=0.01, min=0.0, step=1.0 / FPS
        )
        loaded_duration_display = server.gui.add_markdown("Loaded duration: —")

    with server.gui.add_folder("Export"):
        save_button = server.gui.add_button("Save Trajectory")
        export_status = server.gui.add_markdown("")

    load_folder = server.gui.add_folder("Load & Playback")
    with load_folder:
        load_path_input = server.gui.add_text(
            "Trajectory Path",
            initial_value=str(cfg.load_path) if cfg.load_path is not None else "",
        )
        load_button = server.gui.add_button("Load")
        load_status = server.gui.add_markdown("")

    def set_editing(wp: Waypoint | None) -> None:
        edit_state.waypoint = wp
        commit_button.label = "Update Waypoint" if wp is not None else "Add Waypoint"
        cancel_button.visible = wp is not None

    def update_bounds() -> None:
        if waypoints and duration_field.value < waypoints[-1].time:
            duration_field.value = waypoints[-1].time
        time_slider.max = max(duration_field.value, loaded["duration"], 0.01)

    def refresh_scene_from_time() -> None:
        t = time_slider.value
        if len(waypoints) >= 2:
            pos, xyzw = interpolate(waypoints, np.array([t]))
            preview_mesh.position = tuple(pos[0].tolist())
            preview_mesh.wxyz = tuple(xyzw_to_wxyz(xyzw[0]).tolist())
            preview_marker.position = tuple(pos[0].tolist())
        if loaded["T"] > 0:
            frame = int(round(min(t, loaded["duration"]) * FPS))
            frame = max(0, min(frame, loaded["T"] - 1))
            pos = loaded["pos"][frame]
            xyzw = loaded["rot"][frame]
            loaded["mesh"].position = tuple(pos.tolist())
            loaded["mesh"].wxyz = tuple(xyzw_to_wxyz(xyzw).tolist())
            loaded["marker"].position = tuple(pos.tolist())

    def clear_loaded_scene() -> None:
        for key in ("point_cloud", "mesh", "marker", "capture_button"):
            handle = loaded.get(key)
            if handle is not None:
                handle.remove()
        loaded["T"] = 0
        loaded["duration"] = 0.0
        loaded_duration_display.content = "Loaded duration: —"

    def do_capture(_) -> None:
        set_editing(None)
        frame = int(round(min(time_slider.value, loaded["duration"]) * FPS))
        frame = max(0, min(frame, loaded["T"] - 1))
        pos = loaded["pos"][frame]
        xyzw = loaded["rot"][frame]
        gizmo.position = tuple(pos.tolist())
        gizmo.wxyz = tuple(xyzw_to_wxyz(xyzw).tolist())
        time_slider.value = frame / FPS
        refresh_scene_from_time()

    def do_load(_) -> None:
        path = Path(load_path_input.value)
        if not path.exists():
            load_status.content = f"**File not found:** `{path}`"
            return
        pos, rot, source_fps = load_trajectory(path)
        clear_loaded_scene()

        T = pos.shape[0]
        loaded["pos"] = pos
        loaded["rot"] = rot
        loaded["T"] = T
        loaded["duration"] = (T - 1) / FPS
        loaded["point_cloud"] = server.scene.add_point_cloud(
            "/loaded/path_points",
            points=pos,
            colors=time_gradient_colors(T),
            point_size=0.015,
        )
        loaded["mesh"] = server.scene.add_mesh_simple(
            "/loaded/mesh", vertices=vertices, faces=faces, color=(80, 200, 120)
        )
        loaded["marker"] = server.scene.add_icosphere(
            "/loaded/current_point", radius=0.025, color=(255, 0, 0)
        )

        with load_folder:
            capture_button = server.gui.add_button("Capture as Waypoint")
            capture_button.on_click(do_capture)
            loaded["capture_button"] = capture_button

        loaded_duration_display.content = (
            f"Loaded duration: {loaded['duration']:.2f}s ({T} frames)"
        )
        update_bounds()
        refresh_scene_from_time()

        status = f"Loaded {T} frames from `{path}`"
        if source_fps is None:
            status += f" — note: no fps metadata, assuming {FPS} fps"
        elif abs(source_fps - FPS) > 1e-6:
            status += f" — recorded at {source_fps} fps, tool uses {FPS} fps"
        load_status.content = status

    load_button.on_click(do_load)

    def rebuild_waypoint_rows() -> None:
        for h in row_handles:
            h.remove()
        row_handles.clear()

        with waypoint_list_folder:
            for wp in waypoints:
                row = server.gui.add_button_group(
                    f"t={wp.time:.2f}", ("Edit", "Delete")
                )
                row_handles.append(row)

                def make_on_click(wp=wp):
                    def _(event: viser.GuiEvent) -> None:
                        if event.target.value == "Edit":
                            time_slider.value = wp.time
                            gizmo.position = wp.position
                            gizmo.wxyz = wp.wxyz
                            set_editing(wp)
                            refresh_scene_from_time()
                        else:  # "Delete"
                            waypoints.remove(wp)
                            if edit_state.waypoint is wp:
                                set_editing(None)
                            save_waypoints(sidecar_path, waypoints)
                            rebuild_waypoint_rows()
                            update_bounds()
                            refresh_scene_from_time()

                    return _

                row.on_click(make_on_click())

    def commit_waypoint(_) -> None:
        t = time_slider.value
        other_times = [wp.time for wp in waypoints if wp is not edit_state.waypoint]
        if any(abs(t - ot) < 1e-6 for ot in other_times):
            export_status.content = "A waypoint already exists at this time — Load it to edit, or move the scrubber."
            return

        was_editing = edit_state.waypoint is not None
        if edit_state.waypoint is not None:
            edit_state.waypoint.time = t
            edit_state.waypoint.position = tuple(gizmo.position)
            edit_state.waypoint.wxyz = tuple(gizmo.wxyz)
        else:
            waypoints.append(
                Waypoint(time=t, position=tuple(gizmo.position), wxyz=tuple(gizmo.wxyz))
            )
        waypoints.sort(key=lambda wp: wp.time)

        save_waypoints(sidecar_path, waypoints)
        set_editing(None)
        rebuild_waypoint_rows()
        update_bounds()
        if not was_editing:
            time_slider.value = min(t + 1.0, time_slider.max)
        export_status.content = ""
        refresh_scene_from_time()

    commit_button.on_click(commit_waypoint)

    def do_cancel_edit(_) -> None:
        set_editing(None)

    cancel_button.on_click(do_cancel_edit)

    def on_duration_update(_) -> None:
        if waypoints and duration_field.value < waypoints[-1].time:
            duration_field.value = waypoints[-1].time
        time_slider.max = max(duration_field.value, loaded["duration"], 0.01)

    duration_field.on_update(on_duration_update)

    time_slider.on_update(lambda _: refresh_scene_from_time())

    def toggle_playing(_) -> None:
        play_state.playing = not play_state.playing

    play_button.on_click(toggle_playing)

    def do_save(_) -> None:
        if len(waypoints) < 2:
            export_status.content = "**Need at least 2 waypoints to export.**"
            return
        if duration_field.value <= waypoints[0].time:
            export_status.content = "**Duration must be after the first waypoint's time.**"
            return
        start = waypoints[0].time
        end = duration_field.value
        n_frames = max(2, round((end - start) * FPS) + 1)
        t = np.linspace(start, end, n_frames)
        pos, rot = interpolate(waypoints, t)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pos": pos, "rot": rot, "fps": FPS}, output_path, compress=3)
        export_status.content = f"Saved {n_frames} frames to `{output_path}`"
        print(f"Saved {n_frames} frames to {output_path}")

    save_button.on_click(do_save)

    rebuild_waypoint_rows()
    update_bounds()

    if cfg.load_path is not None:
        do_load(None)

    print(f"Open http://localhost:{cfg.port} in your browser")
    frame_dt = 1.0 / FPS
    last_update = time.time()
    while True:
        now = time.time()
        if play_state.playing and (now - last_update) >= frame_dt:
            next_t = time_slider.value + frame_dt
            time_slider.value = next_t if next_t <= time_slider.max else 0.0
            last_update = now
        refresh_scene_from_time()
        time.sleep(0.01)


if __name__ == "__main__":
    main(tyro.cli(Config))
