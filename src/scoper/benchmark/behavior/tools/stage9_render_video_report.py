"""Stage 9 real-render/video closed-loop report.

Stage 9 upgrades the already-validated symbolic/top-down closed loop into RGB video
evidence. The important execution contract is unchanged:

task generation -> symbolic scene reset -> oracle/teacher public tools ->
scoper.benchmark.behavior.execution_core action trace -> optional OmniGibson RGB rendering.

In other words, the trajectory is driven by ``scripts/scoper.benchmark.behavior.execution_core`` actions;
OmniGibson is used as a renderer that follows those symbolic ego poses and object state
updates. This keeps Stage 9 aligned with Stage 8 gold traces while making video evidence
available for human review and later VLM work.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import traceback
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scoper.benchmark.behavior.execution_core.navigation import NavigationTarget, sample_interaction_candidates
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject
from scoper.benchmark.behavior.execution_core.occupancy_utils import (
    SceneTraversability,
    draw_occupancy_path_overlay,
    indoor_mask_as_traversability,
    mask_traversability_with_reference,
    scene_wall_slice_occupancy_as_traversability,
    write_scene_mesh_occupancy_debug,
    write_scene_traversability_debug,
    write_scene_wall_slice_occupancy_debug,
)

from scoper.benchmark.behavior.bddl_adapter import BDDLAdapter
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.benchmark.behavior.brains.teachers.dispatcher import default_dispatcher
from scoper.benchmark.behavior.episode_runner import EpisodeRunner
from scoper.benchmark.behavior.hooks import HookBus
from scoper.benchmark.behavior.json_utils import json_safe
from scoper.benchmark.behavior.render.ego_arm_visual import ego_arm_visual_config_from_env
from scoper.benchmark.behavior.render.gripper_visual import gripper_visual_config_from_env
from scoper.benchmark.behavior.render.particle_segmentation_replay import particle_segmentation_policy
from scoper.benchmark.behavior.render.video import SymbolicVideoRecorder
from scoper.benchmark.behavior.scheduler import build_scheduler_from_task_spec
from scoper.benchmark.behavior.scene.doors import door_not_load_categories, is_door_like, non_door_categories
from scoper.benchmark.behavior.scene.navigation_settings import (
    STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS,
    STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN,
)
from scoper.benchmark.behavior.scene.profiler import profile_scenes
from scoper.benchmark.behavior.task_generation.profile_task_generator import generate_stage4_tasks
from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates, match_scene_to_templates
from scoper.benchmark.behavior.task_generation.stage5_validator import validate_stage5_reset_mutation
from scoper.benchmark.behavior.tools.visual_report import DEFAULT_PROCTHOR_TEMPLATE_DIR, html_footer, html_header, write_json
from scoper.benchmark.behavior.trace import read_trace


def _env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


STAGE9_SAMPLE_CATEGORIES = ("static_simple", "prepare_then_place")
DISABLED_DOOR_TRAVERSABILITY_CLEARANCE_M = 0.02
RENDER_SCENE_LOAD_CATEGORIES = (
    "floors",
    "floor",
    "walls",
    "wall",
    "ceilings",
    "ceiling",
    "door",
    "room_light",
    "downlight",
    "square_light",
    "fixed_window",
    "window_blind",
)


class Stage9RenderError(RuntimeError):
    pass


class Stage9NavigationSelectionError(Stage9RenderError):
    def __init__(self, selection: Mapping[str, Any]) -> None:
        self.selection = dict(selection)
        super().__init__("No navigation traversability candidate keeps every primary target reachable.")


class _FrozenMarker:
    """Small stand-in for AABBObject so overlay labels stay at reset positions."""

    def __init__(self, name: str, center: Sequence[float]) -> None:
        self.name = str(name)
        self._center = tuple(float(v) for v in center)

    @property
    def xy_center(self) -> tuple[float, float]:
        return (float(self._center[0]), float(self._center[1]))


def _free_disabled_door_footprints_in_traversability(
    traversability: SceneTraversability,
    task_spec: Mapping[str, Any],
    *,
    scene_layout_mode: str,
    load_doors: bool,
) -> SceneTraversability:
    if load_doors or str(scene_layout_mode or "") != "source_json_full":
        return traversability
    doors = _source_json_door_aabb_objects(task_spec)
    return _free_aabb_footprints_in_traversability(
        traversability,
        doors,
        label="disabled_door_footprints",
        margin_m=DISABLED_DOOR_TRAVERSABILITY_CLEARANCE_M,
    )


def _attach_disabled_doorway_openings_to_traversability(
    traversability: SceneTraversability,
    task_spec: Mapping[str, Any],
    *,
    scene_layout_mode: str,
    load_doors: bool,
) -> SceneTraversability:
    if load_doors or str(scene_layout_mode or "") != "source_json_full":
        return traversability
    doors = [_doorway_opening_object_from_source_json(obj) for obj in _source_json_door_aabb_objects(task_spec)]
    doors = [door for door in doors if door is not None]
    if not doors:
        return traversability

    doorway_objects = _merge_doorway_opening_objects(
        tuple(getattr(traversability, "doorway_opening_objects", ()) or ()),
        tuple(doors),
    )
    metadata = dict(getattr(traversability, "metadata", None) or {})
    metadata["disabled_doorway_openings"] = {
        "object_count": int(len(doors)),
        "margin_m": float(DISABLED_DOOR_TRAVERSABILITY_CLEARANCE_M),
        "objects": aabb_objects_for_log(doors)[:200],
    }
    allowed_map = getattr(traversability, "doorway_opening_allowed_map", None)
    allowed_resolution = getattr(traversability, "doorway_opening_allowed_map_resolution", None)
    allowed_size = getattr(traversability, "doorway_opening_allowed_map_size", None)
    if allowed_map is None and getattr(traversability, "indoor_mask", None) is not None:
        allowed_map = traversability.indoor_mask
        allowed_resolution = traversability.map_resolution
        allowed_size = traversability.map_size

    return SceneTraversability(
        floor_map=traversability.floor_map,
        map_resolution=traversability.map_resolution,
        map_size=traversability.map_size,
        floor=traversability.floor,
        source=traversability.source,
        indoor_mask=traversability.indoor_mask,
        doorway_opening_objects=doorway_objects,
        doorway_opening_allowed_map=allowed_map,
        doorway_opening_allowed_map_resolution=allowed_resolution,
        doorway_opening_allowed_map_size=allowed_size,
        metadata=metadata,
    )


def _doorway_opening_object_from_source_json(obj: Any) -> AABBObject | None:
    try:
        center = tuple(float(v) for v in getattr(obj, "center")[:3])
        extent = tuple(float(v) for v in getattr(obj, "extent")[:3])
    except Exception:
        return None
    states = dict(getattr(obj, "states", {}) or {})
    states["doorway_opening"] = True
    states["doorway_opening_source"] = "disabled_door_loading"
    states["navigation_opening_margin"] = float(DISABLED_DOOR_TRAVERSABILITY_CLEARANCE_M)
    return AABBObject(
        name=str(getattr(obj, "name", "")),
        category=str(getattr(obj, "category", "door") or "door"),
        center=center,
        extent=extent,
        visible=False,
        abilities=tuple(getattr(obj, "abilities", ()) or ()),
        states=states,
        root_position=getattr(obj, "root_position", None),
        root_orientation_xyzw=getattr(obj, "root_orientation_xyzw", None),
        root_yaw=getattr(obj, "root_yaw", None),
        pose_source=getattr(obj, "pose_source", None),
    )


def _merge_doorway_opening_objects(
    existing: Sequence[AABBObject],
    added: Sequence[AABBObject],
) -> tuple[AABBObject, ...]:
    merged: list[AABBObject] = []
    seen: set[tuple[str, str]] = set()
    for obj in tuple(existing) + tuple(added):
        key = (str(getattr(obj, "name", "")), str(getattr(obj, "category", "")))
        if key in seen:
            continue
        seen.add(key)
        merged.append(obj)
    return tuple(merged)


def _source_json_door_aabb_objects(task_spec: Mapping[str, Any]) -> list[Any]:
    source = task_spec.get("source") or {}
    if not source.get("source_json"):
        return []
    try:
        from scoper.benchmark.behavior.scene.variant_layout import merge_source_json_layout

        merged = merge_source_json_layout(task_spec, [], {}, include_pickupable_background=True, load_doors=True)
    except Exception:
        return []
    return [
        obj
        for obj in list(getattr(merged, "objects", []) or [])
        if is_door_like(getattr(obj, "category", ""), getattr(obj, "name", ""))
    ]


def _free_aabb_footprints_in_traversability(
    traversability: SceneTraversability,
    objects: Sequence[Any],
    *,
    label: str,
    margin_m: float = 0.0,
) -> SceneTraversability:
    floor_map = getattr(traversability, "floor_map", None)
    if floor_map is None or not objects:
        return traversability
    source_map = np.asarray(floor_map)
    if source_map.ndim != 2 or source_map.size == 0:
        return traversability
    try:
        resolution = float(traversability.map_resolution or 0.0)
        map_size = int(traversability.map_size or source_map.shape[0])
    except Exception:
        return traversability
    if resolution <= 0.0 or map_size <= 0:
        return traversability

    updated = source_map.copy()
    indoor_mask = getattr(traversability, "indoor_mask", None)
    indoor = None
    if indoor_mask is not None:
        indoor_array = np.asarray(indoor_mask)
        if indoor_array.shape == updated.shape:
            indoor = indoor_array > 0
    clear_records: list[dict[str, Any]] = []
    total_changed = 0
    margin = max(0.0, float(margin_m))
    for obj in objects:
        try:
            cx, cy = float(obj.center[0]), float(obj.center[1])
            ex, ey = max(0.0, float(obj.extent[0])), max(0.0, float(obj.extent[1]))
        except Exception:
            continue
        min_x = cx - 0.5 * ex - margin
        max_x = cx + 0.5 * ex + margin
        min_y = cy - 0.5 * ey - margin
        max_y = cy + 0.5 * ey + margin
        c0, c1 = _world_interval_to_grid_span(
            min_x,
            max_x,
            resolution=resolution,
            map_size=map_size,
            limit=updated.shape[1],
        )
        r0, r1 = _world_interval_to_grid_span(
            min_y,
            max_y,
            resolution=resolution,
            map_size=map_size,
            limit=updated.shape[0],
        )
        if c0 is None or r0 is None:
            continue
        region = updated[r0 : r1 + 1, c0 : c1 + 1]
        allowed = np.ones(region.shape, dtype=bool)
        if indoor is not None:
            allowed = indoor[r0 : r1 + 1, c0 : c1 + 1]
        changed = int(np.count_nonzero((region <= 0) & allowed))
        if changed > 0:
            region = region.copy()
            region[allowed] = 255
            updated[r0 : r1 + 1, c0 : c1 + 1] = region
        total_changed += changed
        clear_records.append(
            {
                "name": str(getattr(obj, "name", "")),
                "category": str(getattr(obj, "category", "")),
                "cell_bounds": [int(r0), int(c0), int(r1), int(c1)],
                "changed_cells": int(changed),
            }
        )

    if not clear_records:
        return traversability
    metadata = dict(getattr(traversability, "metadata", None) or {})
    metadata[str(label)] = {
        "object_count": int(len(clear_records)),
        "changed_cells": int(total_changed),
        "objects": clear_records[:200],
        "margin_m": float(margin),
    }
    source = "{}&{}_free".format(getattr(traversability, "source", None) or "traversability", label)
    return SceneTraversability(
        floor_map=updated,
        map_resolution=resolution,
        map_size=map_size,
        floor=int(getattr(traversability, "floor", 0) or 0),
        source=source,
        indoor_mask=indoor_mask,
        metadata=metadata,
    )


def _world_interval_to_grid_span(
    min_world: float,
    max_world: float,
    *,
    resolution: float,
    map_size: int,
    limit: int,
) -> tuple[int | None, int | None]:
    lo = min(float(min_world), float(max_world))
    hi = max(float(min_world), float(max_world))
    raw_start = int(np.floor(lo / float(resolution) + float(map_size) / 2.0))
    raw_end = int(np.floor(hi / float(resolution) + float(map_size) / 2.0))
    if raw_end < 0 or raw_start >= int(limit):
        return None, None
    start = raw_start
    end = raw_end
    start = max(0, min(int(limit) - 1, start))
    end = max(0, min(int(limit) - 1, end))
    if end < start:
        return None, None
    return start, end


def build_stage9_render_video_report(
    *,
    out_root: Path,
    run_id: str | None = None,
    procthor_template_dir: Path = DEFAULT_PROCTHOR_TEMPLATE_DIR,
    scene_models: list[str] | None = None,
    sample_categories: Sequence[str] = STAGE9_SAMPLE_CATEGORIES,
    seed: int = 0,
    max_high_level_steps: int = 120,
    render: bool = False,
    require_render: bool = False,
    video_fps: int = 10,
    renders_per_frame: int = 2,
    render_nav_max_frames: int = 0,
    navigation_speed_mps: float | None = 0.4,
    navigation_turn_speed_radps: float | None = 0.8,
    all_templates: bool = False,
    template_ids: Sequence[str] | None = None,
    scene_layout_mode: str = "auto",
    load_doors: bool = False,
) -> dict[str, Any]:
    """Generate Stage 9 task/execution/render artifacts and return a manifest."""

    scene_models = scene_models or ["Rs_int"]
    run_id = run_id or "{}_stage9_render_video".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage9_render_video" / run_id
    logs_dir = output_dir / "logs"
    episodes_dir = output_dir / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    episodes_dir.mkdir(exist_ok=True)

    render_env = probe_render_environment(render_requested=render)
    selected_template_ids = [str(item).strip() for item in (template_ids or []) if str(item).strip()]
    generation = build_stage9_task_bank(
        procthor_template_dir=procthor_template_dir,
        scene_models=scene_models,
        seed=seed,
        template_ids=selected_template_ids or None,
    )
    if selected_template_ids:
        selected_tasks = select_stage9_template_id_tasks(generation["tasks"], selected_template_ids)
    elif all_templates:
        selected_tasks = select_stage9_all_template_tasks(generation["tasks"])
    else:
        selected_tasks = select_stage9_tasks(generation["tasks"], sample_categories=sample_categories)

    write_json(logs_dir / "task_generation.json", generation["summary"])
    write_json(logs_dir / "task_bank.json", generation["tasks"])
    write_json(logs_dir / "rejected_tasks.json", generation["rejected"])
    write_json(logs_dir / "selected_tasks.json", selected_tasks)
    write_json(logs_dir / "render_environment.json", render_env)

    episode_results = []
    for task in selected_tasks:
        episode_dir = episodes_dir / _safe_filename(str(task.get("id") or "task"))
        try:
            result = run_stage9_episode(
                task,
                output_dir=episode_dir,
                rng_seed=seed,
                max_high_level_steps=max_high_level_steps,
                render_requested=render,
                render_env=render_env,
                require_render=require_render,
                video_fps=video_fps,
                renders_per_frame=renders_per_frame,
                render_nav_max_frames=render_nav_max_frames,
                navigation_speed_mps=navigation_speed_mps,
                navigation_turn_speed_radps=navigation_turn_speed_radps,
                scene_layout_mode=scene_layout_mode,
                load_doors=load_doors,
            )
        except Exception as exc:
            result = build_stage9_fatal_episode_result(
                task,
                output_dir=episode_dir,
                exc=exc,
                render_requested=render,
                render_env=render_env,
            )
        episode_results.append(result)

    summary = summarize_stage9_results(
        generation=generation,
        selected_tasks=selected_tasks,
        episode_results=episode_results,
        render_requested=render,
        require_render=require_render,
        render_env=render_env,
    )
    render_failure_matrix = build_render_failure_matrix(episode_results)
    write_json(logs_dir / "stage9_summary.json", summary)
    write_json(logs_dir / "stage9_episodes.json", episode_results)
    write_json(logs_dir / "video_quality.json", [item.get("video_quality") for item in episode_results])
    write_json(logs_dir / "render_failure_matrix.json", render_failure_matrix)

    # Keep canonical root-level links for the first rendered / fallback sample.
    _copy_first_existing(episode_results, "path_overlay", output_dir / "path_overlay.png")
    _copy_first_existing(episode_results, "execution_video", output_dir / "execution.mp4")
    _copy_first_existing(episode_results, "execution_srt", output_dir / "execution.srt")

    write_stage9_index(output_dir / "index.html", summary, episode_results)
    write_task_generation_html(output_dir / "task_generation.html", generation, selected_tasks)
    write_scene_reset_html(output_dir / "scene_reset.html", episode_results)
    write_oracle_plan_execution_html(output_dir / "oracle_plan_execution.html", episode_results)
    write_topdown_replay_html(output_dir / "topdown_replay.html", episode_results)
    write_video_quality_html(output_dir / "video_quality_check.html", summary, episode_results)
    write_render_failure_matrix_html(output_dir / "render_failure_matrix.html", summary, render_failure_matrix)

    artifacts = {
        "index": "index.html",
        "task_generation": "task_generation.html",
        "scene_reset": "scene_reset.html",
        "oracle_plan_execution": "oracle_plan_execution.html",
        "topdown_replay": "topdown_replay.html",
        "video_quality_check": "video_quality_check.html",
        "render_failure_matrix": "render_failure_matrix.html",
        "path_overlay": "path_overlay.png",
        "task_bank_json": "logs/task_bank.json",
        "selected_tasks_json": "logs/selected_tasks.json",
        "stage9_episodes_json": "logs/stage9_episodes.json",
        "summary_json": "logs/stage9_summary.json",
        "video_quality_json": "logs/video_quality.json",
        "render_failure_matrix_json": "logs/render_failure_matrix.json",
        "render_environment_json": "logs/render_environment.json",
    }
    if (output_dir / "execution.mp4").exists():
        artifacts["video"] = "execution.mp4"
    if (output_dir / "execution.srt").exists():
        artifacts["srt"] = "execution.srt"

    manifest = {
        "run_id": run_id,
        "stage": "stage9_render_video",
        "status": summary["status"],
        "visual_gate": summary["visual_gate"],
        "scene_models": list(scene_models),
        "seed": int(seed),
        "generated_total": int(generation["summary"].get("task_count") or len(generation["tasks"])),
        "selected_task_count": len(selected_tasks),
        "selected_categories": [task.get("category") for task in selected_tasks],
        "selected_template_ids_filter": selected_template_ids,
        "all_templates": bool(all_templates),
        "symbolic_execution_passed": summary["symbolic_execution_passed"],
        "render_requested": bool(render),
        "render_attempted": summary["render_attempted"],
        "rendered_task_count": summary["rendered_task_count"],
        "video_quality_passed": summary["video_quality_passed"],
        "require_render": bool(require_render),
        "video_fps": int(video_fps),
        "renders_per_frame": int(renders_per_frame),
        "render_nav_max_frames": int(render_nav_max_frames),
        "navigation_speed_mps": navigation_speed_mps,
        "navigation_turn_speed_radps": navigation_turn_speed_radps,
        "scene_layout_mode": scene_layout_mode,
        "load_doors": bool(load_doors),
        "render_environment": render_env,
        "failure_kind_counts": summary["failure_kind_counts"],
        "render_failure_counts": summary["render_failure_counts"],
        "failure_codes": summary["failure_codes"],
        "blocked_reasons": summary["blocked_reasons"],
        "artifacts": artifacts,
        "notes": [
            "Stage 9 trajectories are driven by scoper.benchmark.behavior.execution_core actions through public tools.",
            "OmniGibson is used as an RGB renderer that follows symbolic ego poses and synchronized object centers/states.",
            "If render is unavailable, symbolic top-down replay is retained and real RGB is marked blocked.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {key: str(output_dir / value) for key, value in artifacts.items()}
    return manifest


def build_stage9_task_bank(
    *,
    procthor_template_dir: Path,
    scene_models: Sequence[str],
    seed: int,
    template_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    profiles = profile_scenes(tuple(scene_models))
    templates = load_procthor_templates(procthor_template_dir)
    match_records = []
    for profile in profiles:
        match_records.extend(match_scene_to_templates(profile, templates, max_bindings_per_template=2))
    generation = generate_stage4_tasks(
        profiles=profiles,
        templates=templates,
        match_records=match_records,
        template_ids=template_ids,
        max_tasks_per_template_scene=1,
        seed=seed,
    )
    summary = dict(generation.get("summary") or {})
    summary.setdefault("task_count", len(generation.get("tasks") or []))
    summary.setdefault("rejected_count", len(generation.get("rejected") or []))
    summary["scene_models"] = list(scene_models)
    summary["categories"] = dict(sorted(Counter(str(task.get("category")) for task in generation.get("tasks") or []).items()))
    generation["summary"] = summary
    return generation


def select_stage9_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    sample_categories: Sequence[str] = STAGE9_SAMPLE_CATEGORIES,
) -> list[dict[str, Any]]:
    """Pick one deterministic task per required Stage 9 sample category."""

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for category in sample_categories:
        candidates = [task for task in tasks if str(task.get("category")) == str(category)]
        candidates.sort(key=lambda item: (str((item.get("template") or {}).get("template_id")), str(item.get("id"))))
        if not candidates:
            continue
        task = copy.deepcopy(dict(candidates[0]))
        if str(task.get("id")) in seen_ids:
            continue
        selected.append(task)
        seen_ids.add(str(task.get("id")))
    return selected


def select_stage9_all_template_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pick one deterministic generated task for every template id."""

    selected: list[dict[str, Any]] = []
    seen_templates: set[str] = set()
    candidates = sorted(
        (copy.deepcopy(dict(task)) for task in tasks),
        key=lambda item: (
            str(_task_template_id(item)),
            str((item.get("source") or {}).get("scene_model") or ""),
            str(item.get("category") or ""),
            str(item.get("id") or ""),
        ),
    )
    for task in candidates:
        template_id = str(_task_template_id(task) or task.get("id") or "")
        if not template_id or template_id in seen_templates:
            continue
        selected.append(task)
        seen_templates.add(template_id)
    return selected


def select_stage9_template_id_tasks(
    tasks: Sequence[Mapping[str, Any]],
    template_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Pick one deterministic generated task for each requested template id."""

    by_template: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        template_id = str(_task_template_id(task) or "")
        if not template_id:
            continue
        by_template.setdefault(template_id, []).append(copy.deepcopy(dict(task)))
    for candidates in by_template.values():
        candidates.sort(
            key=lambda item: (
                str((item.get("source") or {}).get("scene_model") or ""),
                str(item.get("category") or ""),
                str(item.get("id") or ""),
            )
        )

    selected: list[dict[str, Any]] = []
    seen_templates: set[str] = set()
    for template_id in template_ids:
        template_id = str(template_id).strip()
        if not template_id or template_id in seen_templates:
            continue
        candidates = by_template.get(template_id) or []
        if not candidates:
            continue
        selected.append(copy.deepcopy(dict(candidates[0])))
        seen_templates.add(template_id)
    return selected


def _task_template_id(task: Mapping[str, Any]) -> Any:
    return (task.get("template") or {}).get("template_id") or (task.get("source") or {}).get("template_id")


def probe_render_environment(*, render_requested: bool) -> dict[str, Any]:
    """Return a diagnostic render gate record without importing OmniGibson eagerly."""

    og_available = importlib.util.find_spec("omnigibson") is not None
    gpu_env = os.environ.get("EB_BEHAVIOR_GPU") == "1"
    nvidia_smi = shutil.which("nvidia-smi") is not None
    can_attempt = bool(render_requested and gpu_env and og_available)
    blocked_reasons = []
    if render_requested and not gpu_env:
        blocked_reasons.append("EB_BEHAVIOR_GPU is not set to 1")
    if render_requested and not og_available:
        blocked_reasons.append("omnigibson is not importable in this Python environment")
    if render_requested and not nvidia_smi:
        blocked_reasons.append("nvidia-smi is not available")
    if not render_requested:
        blocked_reasons.append("render was not requested")
    return {
        "render_requested": bool(render_requested),
        "eb_behavior_gpu": gpu_env,
        "omnigibson_importable": og_available,
        "nvidia_smi_available": nvidia_smi,
        "can_attempt_render": can_attempt,
        "blocked_reasons": blocked_reasons,
    }


def run_stage9_episode(
    task_spec: Mapping[str, Any],
    *,
    output_dir: Path,
    rng_seed: int,
    max_high_level_steps: int,
    render_requested: bool,
    render_env: Mapping[str, Any],
    require_render: bool,
    video_fps: int,
    renders_per_frame: int,
    render_nav_max_frames: int,
    navigation_speed_mps: float | None,
    navigation_turn_speed_radps: float | None,
    scene_layout_mode: str = "auto",
    load_doors: bool = False,
    simulator_ready_callback: Callable[[], None] | None = None,
    trace_quality_gate: Any = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    task = copy.deepcopy(dict(task_spec))
    requested_scene_layout_mode = str(scene_layout_mode or "auto")
    episode_scene_layout_mode = effective_stage9_scene_layout_mode(requested_scene_layout_mode, task)
    write_json(logs_dir / "selected_task.json", task)
    write_json(logs_dir / "task_spec.json", task)
    scene_layout_artifacts = write_source_scene_layout_artifacts(
        task,
        output_dir=output_dir,
        scene_layout_mode=episode_scene_layout_mode,
    )
    if scene_layout_artifacts:
        write_json(logs_dir / "scene_layout_artifacts.json", scene_layout_artifacts)

    adapter = BDDLAdapter()
    stage5_result = validate_stage5_reset_mutation(
        task,
        rng_seed=rng_seed,
        run_reachability=False,
        scene_layout_mode=episode_scene_layout_mode,
        load_doors=load_doors,
    )
    write_json(logs_dir / "stage5_validation.json", stage5_result)

    from scoper.benchmark.behavior.task_generation.static_scene_admission import (
        build_symbolic_navigation_context,
    )

    symbolic_navigation_context = build_symbolic_navigation_context(task)
    probe_env = BehaviorClosedLoopToolEnv(
        adapter=adapter,
        max_high_level_steps=max_high_level_steps,
        scene_layout_mode=episode_scene_layout_mode,
        load_doors=load_doors,
        scene_traversability=symbolic_navigation_context["scene_traversability"],
        occupancy_agent_radius=STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS,
        occupancy_safety_margin=STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN,
        allow_navigation_candidate_fallbacks=False,
    )
    reset_obs = probe_env.reset_to_task(task, rng_seed=rng_seed)
    initial_goal = probe_env.check_goal(model_stopped=False)
    start_markers = capture_role_markers(probe_env)
    expected_plan, teacher_info = expected_oracle_plan(probe_env, task)
    write_json(logs_dir / "scene_reset.json", scene_reset_report(probe_env, reset_obs, initial_goal))
    write_json(logs_dir / "oracle_plan.json", {"teacher": teacher_info, "plan": expected_plan})
    draw_episode_path_overlay(output_dir / "path_overlay.png", probe_env, start_markers, [reset_obs.get("agent_pose") or {}])

    render_manifest: dict[str, Any] | None = None
    render_error: dict[str, Any] | None = None
    episode_result = None
    run_env = None
    trace_path = logs_dir / "trace.jsonl"
    symbolic_fallback_trace_path: Path | None = None

    if render_requested and bool(render_env.get("can_attempt_render")):
        try:
            episode_result, run_env, render_manifest = run_episode_with_render(
                task,
                adapter=adapter,
                output_dir=output_dir,
                trace_path=trace_path,
                rng_seed=rng_seed,
                max_high_level_steps=max_high_level_steps,
                video_fps=video_fps,
                renders_per_frame=renders_per_frame,
                render_nav_max_frames=render_nav_max_frames,
                navigation_speed_mps=navigation_speed_mps,
                navigation_turn_speed_radps=navigation_turn_speed_radps,
                scene_layout_mode=episode_scene_layout_mode,
                load_doors=load_doors,
                simulator_ready_callback=simulator_ready_callback,
                trace_quality_gate=trace_quality_gate,
            )
            write_json(logs_dir / "render_manifest.json", render_manifest)
        except Exception as exc:
            from scoper.benchmark.behavior.render.evaluation_session import FreshProcessRequired
            if isinstance(exc, FreshProcessRequired):
                raise  # The session owner must retry the exact task in a fresh process.
            render_error = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(logs_dir / "render_error.json", render_error)
            if require_render:
                # Still run symbolic fallback below so the report contains an execution trace.
                pass

    if episode_result is None:
        symbolic_fallback_trace_path = logs_dir / "symbolic_fallback_trace.jsonl"
        episode_result, run_env = run_episode_symbolic(
            task,
            adapter=adapter,
            trace_path=symbolic_fallback_trace_path,
            rng_seed=rng_seed,
            max_high_level_steps=max_high_level_steps,
            scene_layout_mode=episode_scene_layout_mode,
            load_doors=load_doors,
        )

    records = read_trace(trace_path) if trace_path.exists() else []
    step_records = [record for record in records if record.get("record_type") == "step"]
    transitions = [dict(record.get("transition") or {}) for record in step_records]
    poses = core_frame_pose_records(run_env or probe_env, reset_obs)
    if not poses:
        poses = [reset_obs.get("agent_pose") or {}]
        poses.extend(
            transition.get("agent_pose") or {}
            for transition in transitions
            if isinstance(transition.get("agent_pose"), Mapping)
        )
    draw_episode_path_overlay(output_dir / "path_overlay.png", run_env or probe_env, start_markers, poses)
    occupancy_debug = write_episode_occupancy_debug(logs_dir, run_env or probe_env, prefix="navigation_occupancy")

    tool_trace = build_tool_trace(step_records)
    predicate_progress = build_predicate_progress(initial_goal, transitions)
    occupancy_path = build_occupancy_path_report(run_env or probe_env, transitions, reset_obs)
    if occupancy_debug:
        occupancy_path["debug_artifacts"] = occupancy_debug
    write_json(logs_dir / "tool_trace.json", tool_trace)
    write_json(logs_dir / "predicate_progress.json", predicate_progress)
    write_json(logs_dir / "metrics.json", episode_result.metrics)
    write_json(logs_dir / "occupancy_path_report.json", occupancy_path)

    video_quality = analyze_video_quality(render_manifest, output_dir=output_dir)
    video_quality = with_visual_mutation_quality_issues(
        video_quality,
        visual_mutation_issues_from_tool_trace(tool_trace),
    )
    write_json(logs_dir / "video_quality.json", video_quality)

    execution_ok = episode_result.status == "passed"
    render_status = render_status_from_episode(render_requested, render_env, render_manifest, render_error, video_quality)
    failure_code = classify_episode_failure(episode_result)
    if render_requested and render_manifest is None and render_error is not None:
        failure_code = failure_code or "RENDER_FAILED"
    failed_tool_calls = failed_tool_calls_from_trace(tool_trace)
    failure_summary = summarize_episode_failure(
        stage5_result=stage5_result,
        execution_ok=execution_ok,
        episode_status=episode_result.status,
        failure_code=failure_code,
        failure_reason=episode_result.failure_reason,
        render_status=render_status,
        render_error=render_error,
        video_quality=video_quality,
        failed_tool_calls=failed_tool_calls,
    )

    summary = {
        "task_id": task.get("id"),
        "template_id": _task_template_id(task),
        "category": task.get("category"),
        "scene_model": (task.get("source") or {}).get("scene_model"),
        "instruction": task.get("instruction"),
        "stage5_admitted": bool(stage5_result.get("stage5_reset_mutation_ok")),
        "teacher": teacher_info,
        "execution_ok": execution_ok,
        "episode_status": episode_result.status,
        "task_success": float((episode_result.final_goal or {}).get("task_success", 0.0) or 0.0),
        "transition_count": len(transitions),
        "trace_record_count": len(records),
        "render_requested": bool(render_requested),
        "render_status": render_status,
        "render_error": render_error,
        "video_quality_status": video_quality.get("status"),
        "failure_code": failure_code,
        "failure_reason": episode_result.failure_reason,
        "failed_tool_calls": failed_tool_calls,
        "failure_summary": failure_summary,
        "requested_scene_layout_mode": requested_scene_layout_mode,
        "scene_layout_mode": episode_scene_layout_mode,
        "load_doors": bool(load_doors),
        "scene_layout_artifacts": scene_layout_artifacts,
    }
    write_json(logs_dir / "episode_summary.json", summary)
    write_episode_index(output_dir / "index.html", task, summary, expected_plan, tool_trace, video_quality)
    artifacts = episode_artifacts(output_dir, render_manifest)
    if render_error is not None:
        artifacts["render_error"] = "logs/render_error.json"
    if symbolic_fallback_trace_path is not None and symbolic_fallback_trace_path.exists():
        artifacts["symbolic_fallback_trace"] = "logs/symbolic_fallback_trace.jsonl"
    return {
        **summary,
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "final_goal": episode_result.final_goal,
        "metrics": episode_result.metrics,
        "stage5_result": stage5_result,
        "initial_goal": initial_goal,
        "expected_plan": expected_plan,
        "tool_trace": tool_trace,
        "predicate_progress": predicate_progress,
        "occupancy_path": occupancy_path,
        "video_quality": video_quality,
    }


def write_source_scene_layout_artifacts(
    task_spec: Mapping[str, Any],
    *,
    output_dir: Path,
    scene_layout_mode: str,
) -> dict[str, Any]:
    if scene_layout_mode != "source_json_full":
        return {}
    source = task_spec.get("source") or {}
    source_json = source.get("source_json")
    scene_model = source.get("scene_model")
    if not source_json or not scene_model:
        return {"status": "skipped", "reason": "missing_source_json_or_scene_model"}
    from scoper.resources import resolve_resource, behavior_assets_root
    scene_file = resolve_resource(str(source_json))
    if not scene_file.exists():
        return {"status": "skipped", "reason": "source_json_missing", "scene_file": str(scene_file)}
    artifacts_dir = output_dir / "scene_layout"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    assets_root = behavior_assets_root()
    try:
        from scoper.benchmark.behavior.scene_geometry.asset_catalog import AssetCatalog
        from scoper.benchmark.behavior.scene_geometry.bev import render_bev
        from scoper.benchmark.behavior.scene_geometry.category_classifier import CategoryClassifier
        from scoper.benchmark.behavior.scene_geometry.occupancy_debug import write_scene_occupancy_artifacts
        from scoper.benchmark.behavior.scene_geometry.rules import load_rules
        from scoper.benchmark.behavior.scene_geometry.taxonomy_index import TaxonomyIndex

        catalog = AssetCatalog.build(assets_root)
        classifier = CategoryClassifier(catalog, TaxonomyIndex.build(), load_rules())
        bev_manifest = render_bev(
            scene_model=str(scene_model),
            scene_file=scene_file,
            output_png=artifacts_dir / "scene_variant_bev.png",
            labels_json=artifacts_dir / "scene_variant_labels.json",
            catalog=catalog,
            classifier=classifier,
            assets_root=assets_root,
        )
        occupancy = write_scene_occupancy_artifacts(
            scene_model=str(scene_model),
            scene_file=scene_file,
            output_dir=artifacts_dir,
            prefix="scene_variant_occupancy",
            catalog=catalog,
            classifier=classifier,
            assets_root=assets_root,
            agent_radius=STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS,
            safety_margin=STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN,
        )
        return {
            "status": "passed",
            "scene_model": str(scene_model),
            "scene_file": str(scene_file),
            "bev_png": str(artifacts_dir / "scene_variant_bev.png"),
            "labels_json": str(artifacts_dir / "scene_variant_labels.json"),
            "label_count": bev_manifest.get("label_count"),
            "occupancy": occupancy,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "scene_model": str(scene_model),
            "scene_file": str(scene_file),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def effective_stage9_scene_layout_mode(requested_mode: str, task_spec: Mapping[str, Any]) -> str:
    mode = str(requested_mode or "auto").strip().lower()
    if mode == "auto":
        source = task_spec.get("source") if isinstance(task_spec, Mapping) else {}
        return "source_json_full" if isinstance(source, Mapping) and source.get("source_json") else "task_roles"
    if mode in {"task_roles", "source_json_full"}:
        return mode
    raise ValueError("scene_layout_mode must be 'auto', 'task_roles', or 'source_json_full'.")


def build_stage9_fatal_episode_result(
    task_spec: Mapping[str, Any],
    *,
    output_dir: Path,
    exc: Exception,
    render_requested: bool,
    render_env: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    task = copy.deepcopy(dict(task_spec))
    error = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    video_quality = {"status": "not_available", "passed": False, "issues": ["episode failed before render/video analysis"]}
    render_status = "failed:fatal_error" if render_requested and bool(render_env.get("can_attempt_render")) else render_status_from_episode(
        render_requested, render_env, None, None, video_quality
    )
    failure_summary = {
        "failure_kind": "stage9_runtime_error",
        "failure_code": "STAGE9_EPISODE_RUNTIME_ERROR",
        "failure_reason": "{}: {}".format(error["error_type"], error["error_message"]),
        "render_error_type": error["error_type"] if render_requested else None,
        "render_error_message": error["error_message"] if render_requested else None,
        "video_quality_issues": list(video_quality["issues"]),
        "failed_tool_call_count": 0,
        "has_recovered_tool_failures": False,
    }
    summary = {
        "task_id": task.get("id"),
        "template_id": _task_template_id(task),
        "category": task.get("category"),
        "scene_model": (task.get("source") or {}).get("scene_model"),
        "instruction": task.get("instruction"),
        "stage5_admitted": False,
        "teacher": {"ready": False},
        "execution_ok": False,
        "episode_status": "failed",
        "task_success": 0.0,
        "transition_count": 0,
        "trace_record_count": 0,
        "render_requested": bool(render_requested),
        "render_status": render_status,
        "render_error": error if render_requested else None,
        "video_quality_status": video_quality["status"],
        "failure_code": "STAGE9_EPISODE_RUNTIME_ERROR",
        "failure_reason": failure_summary["failure_reason"],
        "failed_tool_calls": [],
        "failure_summary": failure_summary,
    }
    write_json(logs_dir / "selected_task.json", task)
    write_json(logs_dir / "task_spec.json", task)
    write_json(logs_dir / "fatal_error.json", error)
    if render_requested:
        write_json(logs_dir / "render_error.json", error)
    write_json(logs_dir / "video_quality.json", video_quality)
    write_json(logs_dir / "episode_summary.json", summary)
    write_fatal_episode_index(output_dir / "index.html", task, summary, error, video_quality)
    artifacts = {
        "index": "index.html",
        "selected_task": "logs/selected_task.json",
        "task_spec": "logs/task_spec.json",
        "fatal_error": "logs/fatal_error.json",
        "video_quality": "logs/video_quality.json",
    }
    if render_requested:
        artifacts["render_error"] = "logs/render_error.json"
    return {
        **summary,
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "final_goal": {},
        "metrics": {},
        "stage5_result": {},
        "initial_goal": {},
        "expected_plan": [],
        "tool_trace": [],
        "predicate_progress": [],
        "occupancy_path": {"available": False},
        "video_quality": video_quality,
    }


def run_episode_symbolic(
    task_spec: Mapping[str, Any],
    *,
    adapter: BDDLAdapter,
    trace_path: Path,
    rng_seed: int,
    max_high_level_steps: int,
    scene_layout_mode: str = "auto",
    load_doors: bool = False,
) -> tuple[Any, BehaviorClosedLoopToolEnv]:
    episode_scene_layout_mode = effective_stage9_scene_layout_mode(scene_layout_mode, task_spec)
    brain, _source = default_dispatcher().select(task_spec)
    from scoper.benchmark.behavior.task_generation.static_scene_admission import (
        build_symbolic_navigation_context,
    )

    navigation_context = build_symbolic_navigation_context(task_spec)
    env = BehaviorClosedLoopToolEnv(
        adapter=adapter,
        max_high_level_steps=max_high_level_steps,
        scene_layout_mode=episode_scene_layout_mode,
        load_doors=load_doors,
        scene_traversability=navigation_context["scene_traversability"],
        occupancy_agent_radius=STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS,
        occupancy_safety_margin=STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN,
        allow_navigation_candidate_fallbacks=False,
    )
    scheduler = build_scheduler_from_task_spec(task_spec)
    result = EpisodeRunner(env, max_high_level_steps=max_high_level_steps).run(
        task_spec,
        brain=brain,
        scheduler=scheduler,
        trace_path=trace_path,
        rng_seed=rng_seed,
    )
    return result, env


def run_episode_with_render(
    task_spec: Mapping[str, Any],
    *,
    adapter: BDDLAdapter,
    output_dir: Path,
    trace_path: Path,
    rng_seed: int,
    max_high_level_steps: int,
    video_fps: int,
    renders_per_frame: int,
    render_nav_max_frames: int,
    navigation_speed_mps: float | None,
    navigation_turn_speed_radps: float | None,
    scene_layout_mode: str = "auto",
    load_doors: bool = False,
    simulator_ready_callback: Callable[[], None] | None = None,
    trace_quality_gate: Any = None,
) -> tuple[Any, BehaviorClosedLoopToolEnv, dict[str, Any]]:
    from scoper.benchmark.behavior.hooks.video_recorder import VideoRecorderHook
    from scoper.benchmark.behavior.hooks.base import Hook
    from scoper.benchmark.behavior.render.ego_renderer import EgoRenderer
    from scoper.benchmark.behavior.scene.placement_render import build as build_render_scene

    episode_scene_layout_mode = effective_stage9_scene_layout_mode(scene_layout_mode, task_spec)
    render_result = build_render_scene(
        task_spec,
        adapter=adapter,
        rng_seed=rng_seed,
        scene_layout_mode=episode_scene_layout_mode,
        load_doors=load_doors,
    )
    write_json(output_dir / "logs" / "render_scene_report.json", render_result.render_report)
    rendered_visibility_query_enabled = not _env_flag_enabled("EB_BEHAVIOR_DISABLE_RENDERED_VISIBILITY_QUERY")
    env_config = make_stage9_render_env_config(
        task_spec,
        scene_layout_mode=episode_scene_layout_mode,
        include_visibility_modalities=rendered_visibility_query_enabled,
        load_doors=load_doors,
    )
    gripper_visual_config = gripper_visual_config_from_env()
    ego_arm_visual_config = ego_arm_visual_config_from_env()
    write_json(
        output_dir / "logs" / "render_env_config.json",
        render_env_config_for_log(
            env_config,
            render_result.og_objects,
            render_result.render_aliases,
            render_result.render_alias_metadata,
            gripper_visual_config=gripper_visual_config,
            ego_arm_visual_config=ego_arm_visual_config,
        ),
    )

    particle_policy = particle_segmentation_policy(task_spec)
    renderer = EgoRenderer(
        env_config,
        render_result.og_objects,
        renders_per_frame=renders_per_frame,
        render_aliases=render_result.render_aliases,
        render_alias_metadata=render_result.render_alias_metadata,
        reset_diagnostics_path=output_dir / "logs" / "renderer_reset_timing.jsonl",
        gripper_visual_config=gripper_visual_config,
        ego_arm_visual_config=ego_arm_visual_config,
        particle_systems_enabled=bool(particle_policy["source_particle_systems_enabled"]),
        native_particle_source_visuals_enabled=bool(particle_policy["source_native_particle_source_visuals_enabled"]),
        particle_source_stream_visuals_enabled=bool(particle_policy["source_particle_source_stream_visuals_enabled"]),
    )
    try:
        renderer.reset()
        write_json(output_dir / "logs" / "render_runtime_scene_report.json", renderer.scene_summary())
        if simulator_ready_callback is not None:
            simulator_ready_callback()
        if hasattr(renderer, "render_identity_map"):
            write_json(output_dir / "logs" / "render_identity_map.json", renderer.render_identity_map())
        scene_traversability = renderer.scene_traversability(occupancy_resolution=0.10)
        scene_traversability = _attach_disabled_doorway_openings_to_traversability(
            scene_traversability,
            task_spec,
            scene_layout_mode=episode_scene_layout_mode,
            load_doors=load_doors,
        )
        scene_navigation_objects = renderer.scene_navigation_objects(load_doors=load_doors)
        interaction_faces_by_name = renderer.interaction_faces_by_name(load_doors=load_doors)
        traversability_debug = write_scene_traversability_debug(scene_traversability, output_dir / "logs")
        scene_indoor_traversability = indoor_mask_as_traversability(scene_traversability)
        raw_scene_mesh_occupancy = renderer.scene_mesh_occupancy(
            occupancy_resolution=0.10,
            map_size=scene_traversability.map_size,
        )
        scene_mesh_occupancy = mask_traversability_with_reference(
            raw_scene_mesh_occupancy,
            scene_indoor_traversability,
            label="floor_insseg",
        )
        line_of_sight_traversability = (
            mask_traversability_with_reference(
                raw_scene_mesh_occupancy,
                scene_indoor_traversability,
                label="floor_insseg",
            )
            if bool(getattr(scene_indoor_traversability, "enabled", False))
            else scene_mesh_occupancy
        )
        mesh_occupancy_debug = write_scene_mesh_occupancy_debug(
            scene_mesh_occupancy,
            output_dir / "logs",
            navigation_objects=scene_navigation_objects,
            reference_traversability=scene_traversability,
        )
        wall_slice_resolution = 0.05
        wall_slice_map_size = None
        if scene_traversability.map_size and scene_traversability.map_resolution:
            wall_slice_map_size = int(round(float(scene_traversability.map_size) * float(scene_traversability.map_resolution) / wall_slice_resolution))
        scene_wall_slice_occupancy = renderer.scene_wall_slice_occupancy(
            occupancy_resolution=wall_slice_resolution,
            map_size=wall_slice_map_size,
            slice_heights=(0.2, 0.8, 1.2),
            visual_fallback=False,
            exterior_barriers=scene_navigation_objects,
            interior_mask=getattr(scene_traversability, "indoor_mask", None),
            interior_mask_resolution=getattr(scene_traversability, "map_resolution", None),
            interior_mask_size=getattr(scene_traversability, "map_size", None),
        )
        wall_slice_occupancy_debug = write_scene_wall_slice_occupancy_debug(scene_wall_slice_occupancy, output_dir / "logs")
        wall_slice_traversability = scene_wall_slice_occupancy_as_traversability(scene_wall_slice_occupancy)
        wall_slice_traversability = mask_traversability_with_reference(
            wall_slice_traversability,
            scene_indoor_traversability,
            label="floor_insseg",
        )
        navigation_traversability, navigation_selection = select_stage9_navigation_traversability(
            task_spec,
            (
                ("wall_slice", wall_slice_traversability),
                ("scene_traversability", scene_traversability),
                ("mesh_occupancy", scene_mesh_occupancy),
            ),
            adapter=adapter,
            rng_seed=rng_seed,
            interaction_faces_by_name=interaction_faces_by_name,
            scene_navigation_objects=scene_navigation_objects,
            scene_layout_mode=episode_scene_layout_mode,
            load_doors=load_doors,
        )
        mesh_alignment_debug = {"enabled": False, "reason": "disabled_by_default"}
        if _env_flag_enabled("EB_BEHAVIOR_ENABLE_MESH_ALIGNMENT_DEBUG"):
            mesh_alignment_debug = renderer.render_mesh_occupancy_alignment(
                scene_mesh_occupancy,
                output_dir / "visual_inspection",
                navigation_objects=scene_navigation_objects,
                reference_traversability=scene_traversability,
            )
        write_json(
            output_dir / "logs" / "navigation_context.json",
            {
                "scene_traversability": dict(scene_traversability.metadata or {}),
                "scene_traversability_debug": traversability_debug,
                "scene_indoor_traversability": dict(scene_indoor_traversability.metadata or {}),
                "scene_mesh_occupancy": dict(scene_mesh_occupancy.metadata or {}),
                "scene_mesh_occupancy_debug": mesh_occupancy_debug,
                "line_of_sight_traversability": dict(line_of_sight_traversability.metadata or {}),
                "scene_wall_slice_occupancy": dict(scene_wall_slice_occupancy.get("metadata") or {}),
                "scene_wall_slice_occupancy_debug": wall_slice_occupancy_debug,
                "navigation_traversability": dict(navigation_traversability.metadata or {}),
                "navigation_traversability_selection": navigation_selection,
                "stage9_occupancy_config": {
                    "agent_radius": STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS,
                    "safety_margin": STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN,
                    "erosion_radius_m": STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS
                    + STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN,
                },
                "scene_mesh_occupancy_alignment_debug": mesh_alignment_debug,
                "scene_navigation_objects": aabb_objects_for_log(scene_navigation_objects),
                "interaction_faces_by_name": interaction_faces_by_name,
            },
        )
        recorder = SymbolicVideoRecorder(
            output_dir / "execution.mp4",
            output_dir / "frames",
            fps=video_fps,
            save_frames=True,
            burn_in_subtitles=True,
        )
        hook = VideoRecorderHook(
            renderer,
            recorder,
            nav_max_frames=render_nav_max_frames,
            navigation_speed_mps=navigation_speed_mps,
            navigation_turn_speed_radps=navigation_turn_speed_radps,
            action_hold_seconds=0.3,
            final_hold_seconds=0.5,
        )
        brain, _source = default_dispatcher().select(task_spec)
        quality_events = []
        class QualityEventsHook(Hook):
            def on_scheduler_events(self, env, events, phase, high_level_step):
                quality_events.extend(events)

        if trace_quality_gate is not None:
            brain = _TraceQualityBrain(brain, trace_quality_gate, quality_events)
        action_visibility_mode = "rendered_required" if rendered_visibility_query_enabled else "line_of_sight_required"
        env = BehaviorClosedLoopToolEnv(
            adapter=adapter,
            max_high_level_steps=max_high_level_steps,
            renderer=renderer,
            use_rendered_visibility=rendered_visibility_query_enabled,
            action_visibility_mode=action_visibility_mode,
            require_rendered_held_pose=rendered_visibility_query_enabled,
            scene_traversability=navigation_traversability,
            line_of_sight_traversability=line_of_sight_traversability,
            interaction_faces_by_name=interaction_faces_by_name,
            scene_navigation_objects=scene_navigation_objects,
            occupancy_agent_radius=STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS,
            occupancy_safety_margin=STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN,
            scene_layout_mode=episode_scene_layout_mode,
            load_doors=load_doors,
        )
        scheduler = build_scheduler_from_task_spec(task_spec)
        result = EpisodeRunner(
            env,
            max_high_level_steps=max_high_level_steps,
            raise_infrastructure_errors=True,
        ).run(
            task_spec,
            brain=brain,
            hooks=HookBus([hook, QualityEventsHook()]),
            scheduler=scheduler,
            trace_path=trace_path,
            rng_seed=rng_seed,
        )
        if hook.manifest is None:
            raise Stage9RenderError("VideoRecorderHook did not produce a manifest.")
        return result, env, dict(hook.manifest)
    finally:
        try:
            renderer.close()
        except Exception:
            pass


class _TraceQualityBrain:
    def __init__(self, brain: Any, gate: Any, scheduler_events: list | None = None) -> None:
        self._brain = brain
        self._gate = gate
        self._scheduler_events = scheduler_events

    def __getattr__(self, name: str) -> Any:
        return getattr(self._brain, name)

    def act(self, brain_input: Any) -> Any:
        from scoper.benchmark.common import OracleTraceQualityError

        if brain_input.last_transition is not None:
            transition_step = int(brain_input.last_transition.get("high_level_step") or 0)
            events = None if self._scheduler_events is None else [
                event for event in self._scheduler_events
                if int((event.get("payload") or {}).get("step", transition_step)) <= transition_step
            ]
            report = self._gate.observe_transition(brain_input.last_transition, scheduler_events=events)
            if report["status"] == "failed":
                raise OracleTraceQualityError(report)
        return self._brain.act(brain_input)


def make_stage9_render_env_config(
    task_spec: Mapping[str, Any],
    *,
    scene_layout_mode: str = "auto",
    include_visibility_modalities: bool = False,
    visibility_modality: str = "seg_instance",
    load_doors: bool = False,
) -> dict[str, Any]:
    from scoper.benchmark.behavior.behavior_online_env import make_env_config

    episode_scene_layout_mode = effective_stage9_scene_layout_mode(scene_layout_mode, task_spec)
    env_config = make_env_config(task_spec, scene_layout_mode=episode_scene_layout_mode, load_doors=load_doors)
    sensors = env_config.get("env", {}).get("external_sensors") or []
    for sensor in sensors:
        if sensor.get("name") == "ego_camera":
            if include_visibility_modalities:
                if visibility_modality not in {"seg_instance", "seg_instance_id"}:
                    raise ValueError("visibility_modality must be seg_instance or seg_instance_id")
            # EgoRenderer owns Isaac 5.1's fast instance-ID annotator. Adding either
            # legacy modality here activates the unstable OgnSdInstanceMapping graph.
            sensor["modalities"] = ["rgb"]
    scene = env_config.setdefault("scene", {})
    if episode_scene_layout_mode == "source_json_full":
        scene.pop("load_object_categories", None)
        if load_doors:
            scene.pop("not_load_object_categories", None)
    else:
        scene["load_object_categories"] = (
            list(RENDER_SCENE_LOAD_CATEGORIES) if load_doors else non_door_categories(RENDER_SCENE_LOAD_CATEGORIES)
        )
        if not load_doors:
            scene["not_load_object_categories"] = door_not_load_categories()
    env_config["occupancy_agent_radius"] = STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS
    env_config["occupancy_safety_margin"] = STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN
    return env_config


def aabb_objects_for_log(objects: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for obj in objects:
        rows.append(
            {
                "name": getattr(obj, "name", None),
                "category": getattr(obj, "category", None),
                "center": list(getattr(obj, "center", []) or []),
                "extent": list(getattr(obj, "extent", []) or []),
                "abilities": list(getattr(obj, "abilities", []) or []),
                "states": dict(getattr(obj, "states", {}) or {}),
                "visible": bool(getattr(obj, "visible", True)),
            }
        )
    return rows


def select_stage9_navigation_traversability(
    task_spec: Mapping[str, Any],
    candidates: Sequence[tuple[str, Any]],
    *,
    adapter: BDDLAdapter,
    rng_seed: int,
    interaction_faces_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    scene_navigation_objects: Sequence[Any] = (),
    scene_layout_mode: str = "auto",
    load_doors: bool = False,
    require_reachable: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Pick the first render navigation map that keeps primary targets reachable.

    Wall-slice occupancy is more faithful to structural geometry, so it remains first
    choice. Some large BEHAVIOR scenes have collision-wall slices that flood-fill away
    valid indoor regions; in that case Stage 9 should fall back to mesh or scene
    traversability instead of producing a zero-motion RGB failure.
    """

    reports = []
    fallback = None
    for name, traversability in candidates:
        enabled = bool(getattr(traversability, "enabled", False))
        report = {
            "name": str(name),
            "enabled": enabled,
            "source": getattr(traversability, "source", None),
        }
        if not enabled:
            report["status"] = "disabled"
            reports.append(report)
            continue
        if _stage9_navigation_candidate_is_unmasked_mesh(traversability):
            report["status"] = "rejected"
            report["reason"] = "mesh_occupancy_missing_scene_traversability_reference_mask"
            report["primary_targets_reachable"] = False
            reports.append(report)
            continue
        if fallback is None:
            fallback = (name, traversability, report)
        report.update(
            _stage9_navigation_candidate_report(
                task_spec,
                traversability,
                adapter=adapter,
                rng_seed=rng_seed,
                interaction_faces_by_name=interaction_faces_by_name,
                scene_navigation_objects=scene_navigation_objects,
                scene_layout_mode=scene_layout_mode,
                load_doors=load_doors,
            )
        )
        reports.append(report)
        if report.get("primary_targets_reachable"):
            return traversability, {
                "selected": str(name),
                "reason": "primary_targets_reachable",
                "candidates": reports,
            }

    if fallback is not None:
        name, traversability, _report = fallback
        selection = {
            "selected": str(name),
            "reason": "fallback_first_enabled_no_candidate_passed",
            "candidates": reports,
        }
        if require_reachable:
            raise Stage9NavigationSelectionError(selection)
        return traversability, selection

    name, traversability = candidates[0]
    selection = {
        "selected": str(name),
        "reason": "fallback_first_candidate_none_enabled",
        "candidates": reports,
    }
    if require_reachable:
        raise Stage9NavigationSelectionError(selection)
    return traversability, selection


def _stage9_navigation_candidate_is_unmasked_mesh(traversability: Any) -> bool:
    metadata = getattr(traversability, "metadata", {}) or {}
    source = str(getattr(traversability, "source", None) or metadata.get("source") or "")
    if source != "scene.mesh_projection":
        return False
    return not bool(metadata.get("reference_mask"))


def _stage9_navigation_candidate_report(
    task_spec: Mapping[str, Any],
    traversability: Any,
    *,
    adapter: BDDLAdapter,
    rng_seed: int,
    interaction_faces_by_name: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    scene_navigation_objects: Sequence[Any],
    scene_layout_mode: str = "auto",
    load_doors: bool = False,
) -> dict[str, Any]:
    try:
        episode_scene_layout_mode = effective_stage9_scene_layout_mode(scene_layout_mode, task_spec)
        env = BehaviorClosedLoopToolEnv(
            adapter=adapter,
            max_high_level_steps=1,
            scene_traversability=traversability,
            interaction_faces_by_name=interaction_faces_by_name,
            scene_navigation_objects=scene_navigation_objects,
            occupancy_agent_radius=STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS,
            occupancy_safety_margin=STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN,
            scene_layout_mode=episode_scene_layout_mode,
            load_doors=load_doors,
        )
        env.reset_to_task(task_spec, rng_seed=rng_seed)
        occupancy = env.core.occupancy if env.core is not None else None
        if occupancy is None:
            return {"status": "failed", "reason": "occupancy_unavailable", "primary_targets_reachable": False}
        role_reports = _stage9_primary_role_reachability_reports(env, occupancy)
        reachable = bool(role_reports) and all(bool(item.get("same_component_candidates")) for item in role_reports)
        return {
            "status": "passed" if reachable else "unreachable_targets",
            "primary_targets_reachable": reachable,
            "start_pose_repair": dict(occupancy.metadata.get("start_pose_repair") or {}),
            "component_count": int(occupancy.metadata.get("component_count") or 0),
            "roles": role_reports,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": "{}: {}".format(type(exc).__name__, exc),
            "primary_targets_reachable": False,
        }


def _stage9_primary_role_reachability_reports(env: BehaviorClosedLoopToolEnv, occupancy: Any) -> list[dict[str, Any]]:
    role_names = _stage9_primary_navigation_roles(env.current_task_spec)
    reports = []
    start_xy = env.core.ego_state.pose.xy
    start_component = int(occupancy.component_at(start_xy))
    for role_name in role_names:
        object_name = env.role_map.get(role_name)
        obj = env.core.objects.get(object_name) if object_name else None
        if obj is None:
            reports.append({"role": role_name, "object": object_name, "status": "object_missing"})
            continue
        target = NavigationTarget.from_aabb_object(obj)
        candidates = sample_interaction_candidates(
            target,
            env.core.config.navigation.candidate_count,
            env.core.config.navigation.interaction_distance,
        )
        free_candidates = 0
        same_component_candidates = 0
        nearest_component = -1
        nearest_distance = None
        component_counts: dict[int, int] = {}
        for xy in candidates:
            if not occupancy.is_free(xy):
                continue
            free_candidates += 1
            component = int(occupancy.component_at(xy))
            component_counts[component] = component_counts.get(component, 0) + 1
            distance = float(np.linalg.norm(np.asarray(start_xy, dtype=float) - np.asarray(xy, dtype=float)))
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_component = component
            if start_component != -1 and component == start_component:
                same_component_candidates += 1
        reports.append(
            {
                "role": role_name,
                "object": object_name,
                "status": "checked",
                "start_component": start_component,
                "candidate_count": len(candidates),
                "free_candidates": free_candidates,
                "same_component_candidates": same_component_candidates,
                "component_counts": {str(key): int(value) for key, value in sorted(component_counts.items())},
                "nearest_candidate_component": nearest_component,
                "nearest_candidate_distance": nearest_distance,
            }
        )
    return reports


def _stage9_primary_navigation_roles(task_spec: Mapping[str, Any]) -> list[str]:
    roles: list[str] = []
    task_roles = set(str(role) for role in (task_spec.get("roles") or {}))
    family = task_spec.get("family_spec") if isinstance(task_spec, Mapping) else {}
    human = family.get("human") if isinstance(family, Mapping) else {}
    primary = human.get("primary_task") if isinstance(human, Mapping) else {}
    if isinstance(primary, Mapping):
        for key in ("object_role", "receptacle_role"):
            role = primary.get(key)
            if role:
                roles.append(str(role))
    for predicate in (task_spec.get("goal") or {}).get("predicates") or []:
        if not isinstance(predicate, Mapping):
            continue
        args = predicate.get("args")
        if isinstance(args, Mapping):
            for value in args.values():
                if isinstance(value, str) and value in task_roles:
                    roles.append(value)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    roles.extend(str(item) for item in value if isinstance(item, str) and item in task_roles)
        for key, value in predicate.items():
            if key == "args":
                continue
            if (str(key).endswith("_role") or str(key) in {"object", "receptacle", "container", "source", "target"}):
                if isinstance(value, str) and value in task_roles:
                    roles.append(value)
    if not roles:
        for role in ("target_object", "primary_object", "target_receptacle", "primary_receptacle"):
            if role in (task_spec.get("roles") or {}):
                roles.append(role)
    if not roles:
        roles.extend(sorted(task_roles))
    unique = []
    for role in roles:
        if role not in unique:
            unique.append(role)
    return unique


def render_env_config_for_log(
    env_config: Mapping[str, Any],
    og_objects: list[dict],
    render_aliases: Mapping[str, str] | None = None,
    render_alias_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    gripper_visual_config: Mapping[str, Any] | None = None,
    ego_arm_visual_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    logged = copy.deepcopy(dict(env_config))
    logged["objects"] = [
        {key: value for key, value in dict(obj).items() if not str(key).startswith("_")}
        for obj in og_objects
    ]
    logged["render_aliases"] = dict(sorted(dict(render_aliases or {}).items()))
    logged["render_alias_metadata"] = dict(sorted(dict(render_alias_metadata or {}).items()))
    if gripper_visual_config is not None:
        logged["gripper_visual_config"] = json_safe(dict(gripper_visual_config))
    if ego_arm_visual_config is not None:
        logged["ego_arm_visual_config"] = json_safe(dict(ego_arm_visual_config))
    return logged


def expected_oracle_plan(env: BehaviorClosedLoopToolEnv, task_spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        brain, source = default_dispatcher().select(task_spec)
        brain.on_episode_start(env, task_spec)
        calls = list(getattr(brain, "_queue", []) or [])
        plan = [{"tool_name": call.tool_name, "arguments": dict(call.arguments), "reasoning": call.reasoning} for call in calls]
        plan.append({"tool_name": "stop", "arguments": {}, "reasoning": "oracle queue exhausted"})
        return plan, {"teacher_source": source, "teacher_type": type(brain).__name__, "ready": True}
    except Exception as exc:
        return [], {"teacher_source": None, "teacher_type": None, "ready": False, "failure": str(exc)}


def analyze_video_quality(render_manifest: Mapping[str, Any] | None, *, output_dir: Path) -> dict[str, Any]:
    if not render_manifest:
        return {
            "status": "not_available",
            "passed": False,
            "issues": ["render manifest is unavailable"],
        }

    issues: list[str] = []
    video_path = Path(str(render_manifest.get("video") or ""))
    srt_path = Path(str(render_manifest.get("subtitle_path") or ""))
    frames_dir = Path(str(render_manifest.get("frames_dir") or ""))
    frame_paths = sorted(frames_dir.glob("frame_*.png")) if frames_dir.exists() else []
    manifest_count = int(render_manifest.get("frame_count") or 0)

    if not video_path.exists():
        issues.append("execution.mp4 does not exist")
    if not srt_path.exists():
        issues.append("execution.srt does not exist")
    if not frames_dir.exists():
        issues.append("frames directory does not exist")
    if manifest_count <= 0:
        issues.append("render manifest frame_count is zero")
    if frame_paths and manifest_count and len(frame_paths) != manifest_count:
        issues.append("frame dump count ({}) does not match manifest frame_count ({})".format(len(frame_paths), manifest_count))

    video_frame_count = None
    if video_path.exists():
        try:
            import imageio.v2 as imageio

            reader = imageio.get_reader(str(video_path))
            video_frame_count = sum(1 for _ in reader)
            reader.close()
            if manifest_count and video_frame_count != manifest_count:
                issues.append("mp4 frame count ({}) does not match manifest frame_count ({})".format(video_frame_count, manifest_count))
        except Exception as exc:
            issues.append("could not read mp4: {}: {}".format(type(exc).__name__, exc))

    sample = sample_frame_arrays(frame_paths, max_samples=24)
    frame_shape = list(sample[0].shape) if sample else None
    resolution_ok = bool(frame_shape and frame_shape[0] >= 64 and frame_shape[1] >= 64 and frame_shape[-1] >= 3)
    if frame_shape and not resolution_ok:
        issues.append("frame resolution is too small: {}".format(frame_shape))

    black_frames = 0
    white_frames = 0
    nonblank_frames = 0
    diffs: list[float] = []
    previous = None
    for frame in sample:
        rgb = frame[:, :, :3].astype(np.float32)
        mean = float(rgb.mean())
        std = float(rgb.std())
        if mean < 2.0:
            black_frames += 1
        if mean > 253.0:
            white_frames += 1
        if std > 1.0 and 2.0 <= mean <= 253.0:
            nonblank_frames += 1
        if previous is not None:
            diffs.append(float(np.mean(np.abs(rgb - previous))))
        previous = rgb
    if sample and nonblank_frames == 0:
        issues.append("sampled frames appear blank, all black, or all white")
    static_pair_ratio = float(sum(1 for diff in diffs if diff < 0.5) / len(diffs)) if diffs else 0.0
    if len(diffs) >= 2 and static_pair_ratio >= 0.99:
        issues.append("sampled frames appear static")

    passed = not issues
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "video": _relative_or_str(video_path, output_dir),
        "srt": _relative_or_str(srt_path, output_dir),
        "frames_dir": _relative_or_str(frames_dir, output_dir),
        "manifest_frame_count": manifest_count,
        "frame_dump_count": len(frame_paths),
        "mp4_frame_count": video_frame_count,
        "duration_seconds": render_manifest.get("duration_seconds"),
        "fps": render_manifest.get("fps"),
        "frame_shape": frame_shape,
        "resolution_ok": resolution_ok,
        "sample_count": len(sample),
        "black_frame_count": black_frames,
        "white_frame_count": white_frames,
        "nonblank_frame_count": nonblank_frames,
        "static_pair_ratio": static_pair_ratio,
        "issues": issues,
    }


def visual_mutation_issues_from_tool_trace(tool_trace: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for item in tool_trace:
        state_delta = item.get("state_delta") if isinstance(item, Mapping) else None
        for mutation in _visual_mutations_from_state_delta(state_delta):
            if _is_unsupported_visual_mutation(mutation):
                continue
            if mutation.get("success") is not False:
                continue
            issues.append(
                {
                    "step": item.get("step"),
                    "tool_name": item.get("tool_name"),
                    "object": mutation.get("object"),
                    "state": mutation.get("state"),
                    "reason": mutation.get("reason"),
                    "error": mutation.get("error"),
                    "mutation": dict(mutation),
                }
            )
    return issues


def visual_mutation_issues_from_transitions(transitions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tool_trace = []
    for transition in transitions:
        tool_trace.append(
            {
                "step": transition.get("high_level_step"),
                "tool_name": transition.get("tool_name"),
                "state_delta": transition.get("state_delta"),
            }
        )
    return visual_mutation_issues_from_tool_trace(tool_trace)


def with_visual_mutation_quality_issues(video_quality: Mapping[str, Any], issues: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not issues:
        return dict(video_quality)
    out = dict(video_quality)
    existing = list(out.get("issues") or [])
    issue_text = [_visual_mutation_issue_text(issue) for issue in issues]
    out["issues"] = existing + issue_text
    out["visual_mutation_issues"] = [dict(issue) for issue in issues]
    out["passed"] = False
    out["status"] = "failed"
    return out


def _visual_mutations_from_state_delta(state_delta: Any) -> list[Mapping[str, Any]]:
    if not isinstance(state_delta, Mapping):
        return []
    mutations = state_delta.get("visual_mutations") or []
    if not isinstance(mutations, Sequence) or isinstance(mutations, (str, bytes)):
        return []
    return [mutation for mutation in mutations if isinstance(mutation, Mapping)]


def _is_unsupported_visual_mutation(mutation: Mapping[str, Any]) -> bool:
    if mutation.get("unsupported_visual_state") is True:
        return True
    if mutation.get("status") == "unsupported_visual_state":
        return True
    return mutation.get("state") == "saturated" and mutation.get("error") == "missing_saturated_state"


def _visual_mutation_issue_text(issue: Mapping[str, Any]) -> str:
    location = "step {}".format(issue.get("step")) if issue.get("step") is not None else "unknown step"
    tool = str(issue.get("tool_name") or "unknown_tool")
    obj = str(issue.get("object") or "unknown_object")
    state = str(issue.get("state") or "unknown_state")
    reason = str(issue.get("error") or issue.get("reason") or "visual_mutation_failed")
    return "visual mutation failed at {} tool {} for {}.{}: {}".format(location, tool, obj, state, reason)


def sample_frame_arrays(frame_paths: Sequence[Path], *, max_samples: int) -> list[np.ndarray]:
    if not frame_paths:
        return []
    if len(frame_paths) <= max_samples:
        chosen = list(frame_paths)
    else:
        idx = np.linspace(0, len(frame_paths) - 1, max_samples).round().astype(int)
        chosen = [frame_paths[int(i)] for i in idx]
    out = []
    for path in chosen:
        try:
            from PIL import Image

            out.append(np.asarray(Image.open(path).convert("RGB")))
        except Exception:
            continue
    return out


def render_status_from_episode(
    render_requested: bool,
    render_env: Mapping[str, Any],
    render_manifest: Mapping[str, Any] | None,
    render_error: Mapping[str, Any] | None,
    video_quality: Mapping[str, Any],
) -> str:
    if not render_requested:
        return "blocked:not_requested"
    if not bool(render_env.get("can_attempt_render")):
        return "blocked:environment"
    if render_error is not None:
        return "failed:render_error"
    if render_manifest is None:
        return "failed:no_manifest"
    if video_quality.get("passed"):
        return "passed"
    return "failed:video_quality"


def failed_tool_calls_from_trace(tool_trace: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in tool_trace:
        if item.get("success") is not False:
            continue
        rows.append(
            {
                "step": item.get("step"),
                "tool_name": item.get("tool_name"),
                "arguments": item.get("arguments") or {},
                "env_feedback": item.get("env_feedback") or {},
                "task_progress": item.get("task_progress"),
            }
        )
    return rows


def summarize_episode_failure(
    *,
    stage5_result: Mapping[str, Any],
    execution_ok: bool,
    episode_status: str,
    failure_code: str | None,
    failure_reason: str | None,
    render_status: str,
    render_error: Mapping[str, Any] | None,
    video_quality: Mapping[str, Any],
    failed_tool_calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage5_ok = bool(stage5_result.get("stage5_reset_mutation_ok"))
    render_status_text = str(render_status or "")
    render_error_type = (render_error or {}).get("error_type")
    render_error_message = (render_error or {}).get("error_message")
    video_issues = list(video_quality.get("issues") or [])
    render_failure_codes = {"RENDER_FAILED", "VIDEO_QUALITY_FAILED"}
    episode_failed = (not bool(execution_ok)) or str(episode_status) != "passed" or (
        bool(failure_code) and str(failure_code) not in render_failure_codes
    )

    if not stage5_ok:
        failure_kind = "stage5_not_admitted"
    elif episode_failed:
        failure_kind = "symbolic_execution"
    elif render_status_text == "passed" and bool(video_quality.get("passed")):
        failure_kind = "none"
    elif render_error is not None or render_status_text == "failed:render_error":
        failure_kind = "render_error"
    elif render_status_text.startswith("blocked"):
        failure_kind = "render_blocked"
    elif render_status_text.startswith("failed") or video_quality.get("status") == "failed":
        failure_kind = "video_quality"
    else:
        failure_kind = "none"

    return {
        "failure_kind": failure_kind,
        "failure_code": failure_code,
        "failure_reason": failure_reason,
        "render_error_type": render_error_type,
        "render_error_message": render_error_message,
        "video_quality_issues": video_issues,
        "failed_tool_call_count": len(failed_tool_calls),
        "failed_tool_calls": list(failed_tool_calls),
        "has_recovered_tool_failures": bool(failed_tool_calls) and failure_kind == "none",
    }


def build_render_failure_matrix(episode_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in episode_results:
        failure = dict(item.get("failure_summary") or {})
        artifacts = item.get("artifacts") or {}
        render_error = item.get("render_error") or {}
        quality = item.get("video_quality") or {}
        row = {
            "task_id": item.get("task_id"),
            "template_id": item.get("template_id"),
            "category": item.get("category"),
            "scene_model": item.get("scene_model"),
            "instruction": item.get("instruction"),
            "stage5_admitted": item.get("stage5_admitted"),
            "execution_ok": item.get("execution_ok"),
            "episode_status": item.get("episode_status"),
            "task_success": item.get("task_success"),
            "render_status": item.get("render_status"),
            "video_quality_status": quality.get("status") or item.get("video_quality_status"),
            "failure_kind": failure.get("failure_kind") or "unknown",
            "failure_code": failure.get("failure_code") or item.get("failure_code"),
            "failure_reason": failure.get("failure_reason") or item.get("failure_reason"),
            "render_error_type": failure.get("render_error_type") or render_error.get("error_type"),
            "render_error_message": failure.get("render_error_message") or render_error.get("error_message"),
            "video_quality_issues": failure.get("video_quality_issues") or quality.get("issues") or [],
            "failed_tool_call_count": failure.get("failed_tool_call_count", len(item.get("failed_tool_calls") or [])),
            "has_recovered_tool_failures": failure.get("has_recovered_tool_failures", False),
            "episode_link": _episode_rel(item, "index"),
            "video_link": _episode_rel(item, "execution_video") if artifacts.get("execution_video") else "",
            "render_error_link": _episode_rel(item, "render_error") if artifacts.get("render_error") else "",
        }
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("template_id") or ""), str(row.get("task_id") or "")))
    return rows


def summarize_stage9_results(
    *,
    generation: Mapping[str, Any],
    selected_tasks: Sequence[Mapping[str, Any]],
    episode_results: Sequence[Mapping[str, Any]],
    render_requested: bool,
    require_render: bool,
    render_env: Mapping[str, Any],
) -> dict[str, Any]:
    symbolic_passed = sum(1 for item in episode_results if item.get("execution_ok"))
    rendered = sum(1 for item in episode_results if item.get("render_status") == "passed")
    quality_passed = sum(1 for item in episode_results if (item.get("video_quality") or {}).get("passed"))
    render_attempted = sum(1 for item in episode_results if str(item.get("render_status") or "").startswith(("passed", "failed")))
    failure_codes = sorted({str(item.get("failure_code")) for item in episode_results if item.get("failure_code")})
    failure_kind_counts = Counter(str((item.get("failure_summary") or {}).get("failure_kind") or "unknown") for item in episode_results)
    render_failure_counts = Counter(
        str(item.get("render_status") or "unknown")
        for item in episode_results
        if str(item.get("render_status") or "") != "passed"
    )
    blocked_reasons = list(render_env.get("blocked_reasons") or [])
    blocked_reasons.extend(
        str(item.get("render_status"))
        for item in episode_results
        if str(item.get("render_status") or "").startswith("blocked")
    )
    blocked_reasons = sorted(set(blocked_reasons))
    if not selected_tasks:
        return {
            "status": "failed",
            "visual_gate": "stage9_sample_selection_failed",
            "generated_total": int((generation.get("summary") or {}).get("task_count") or len(generation.get("tasks") or [])),
            "selected_task_count": 0,
            "selected_categories": [],
            "symbolic_execution_passed": 0,
            "symbolic_execution_failed": 0,
            "render_requested": bool(render_requested),
            "render_attempted": 0,
            "rendered_task_count": 0,
            "video_quality_passed": 0,
            "require_render": bool(require_render),
            "render_environment": dict(render_env),
            "render_statuses": {},
            "failure_kind_counts": {},
            "render_failure_counts": {},
            "failure_codes": ["STAGE9_SAMPLE_SELECTION_FAILED"],
            "blocked_reasons": blocked_reasons,
        }
    execution_failed = symbolic_passed < len(selected_tasks)
    render_complete = rendered >= len(selected_tasks) and quality_passed >= len(selected_tasks)
    if execution_failed:
        status = "failed"
        visual_gate = "symbolic_execution_failed"
    elif render_complete:
        status = "passed"
        visual_gate = "real_rgb_video_validated_pending_human_review"
    elif require_render:
        status = "failed"
        visual_gate = "required_real_rgb_render_failed"
    else:
        status = "blocked"
        visual_gate = "real_rgb_render_blocked_symbolic_fallback_passed"
    return {
        "status": status,
        "visual_gate": visual_gate,
        "generated_total": int((generation.get("summary") or {}).get("task_count") or len(generation.get("tasks") or [])),
        "selected_task_count": len(selected_tasks),
        "selected_categories": [task.get("category") for task in selected_tasks],
        "symbolic_execution_passed": symbolic_passed,
        "symbolic_execution_failed": len(selected_tasks) - symbolic_passed,
        "render_requested": bool(render_requested),
        "render_attempted": render_attempted,
        "rendered_task_count": rendered,
        "video_quality_passed": quality_passed,
        "require_render": bool(require_render),
        "render_environment": dict(render_env),
        "render_statuses": dict(sorted(Counter(str(item.get("render_status")) for item in episode_results).items())),
        "failure_kind_counts": dict(sorted(failure_kind_counts.items())),
        "render_failure_counts": dict(sorted(render_failure_counts.items())),
        "failure_codes": failure_codes,
        "blocked_reasons": blocked_reasons,
    }


def capture_role_markers(env: BehaviorClosedLoopToolEnv) -> dict[str, _FrozenMarker]:
    markers: dict[str, _FrozenMarker] = {}
    for role_name, instance_name in env.role_map.items():
        if role_name == "floor":
            continue
        obj = env.core.objects.get(instance_name)
        if obj is not None:
            markers[role_name] = _FrozenMarker(obj.name, tuple(obj.center))
    return markers


def scene_reset_report(env: BehaviorClosedLoopToolEnv, reset_obs: Mapping[str, Any], initial_goal: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics_fn = getattr(env, "reset_pose_diagnostics", None)
    pose_diagnostics = diagnostics_fn() if callable(diagnostics_fn) else {}
    return {
        "agent_pose": reset_obs.get("agent_pose"),
        "pose_diagnostics": json_safe(pose_diagnostics),
        "held_objects": reset_obs.get("held_objects"),
        "role_map": dict(env.role_map),
        "placement_report": json_safe(getattr(env.scene_result, "placement_report", {})),
        "initial_goal": initial_goal,
        "object_count": len(getattr(env.core, "objects", {}) or {}),
        "object_catalog": reset_obs.get("object_catalog"),
    }


def core_frame_pose_records(env: BehaviorClosedLoopToolEnv, reset_obs: Mapping[str, Any]) -> list[dict[str, float]]:
    core = getattr(env, "core", None)
    trace = getattr(core, "trace", None)
    events = list(getattr(trace, "events", []) or [])
    poses = [_pose_record(reset_obs.get("agent_pose") or {})]
    for event in events:
        if getattr(event, "event", None) != "frame_pose":
            continue
        payload = getattr(event, "payload", {}) or {}
        record = _pose_record(payload.get("pose"))
        if record:
            poses.append(record)
    return [pose for pose in poses if pose]


def _pose_record(pose: Any) -> dict[str, float]:
    if isinstance(pose, Mapping):
        if "x" not in pose or "y" not in pose:
            return {}
        return {"x": float(pose["x"]), "y": float(pose["y"])}
    if hasattr(pose, "x") and hasattr(pose, "y"):
        return {"x": float(pose.x), "y": float(pose.y)}
    return {}


def draw_episode_path_overlay(
    path: Path,
    env: BehaviorClosedLoopToolEnv,
    start_markers: Mapping[str, _FrozenMarker],
    poses: Sequence[Mapping[str, Any]],
) -> None:
    core = getattr(env, "core", None)
    occupancy = getattr(core, "occupancy", None)
    if occupancy is None:
        return
    path_poses = [
        {"x": float(pose["x"]), "y": float(pose["y"])}
        for pose in poses
        if isinstance(pose, Mapping) and "x" in pose and "y" in pose
    ]
    draw_occupancy_path_overlay(occupancy, path_poses, path, role_markers=dict(start_markers))


def write_episode_occupancy_debug(logs_dir: Path, env: BehaviorClosedLoopToolEnv, prefix: str = "occupancy") -> dict[str, str]:
    core = getattr(env, "core", None)
    occupancy = getattr(core, "occupancy", None)
    if occupancy is None:
        return {}
    return occupancy.write_debug_artifacts(logs_dir, prefix=prefix)


def build_tool_trace(step_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in step_records:
        transition = dict(record.get("transition") or {})
        brain = dict(record.get("brain") or {})
        goal = transition.get("goal_result") or {}
        scheduler_events = list(record.get("scheduler_events") or [])
        rows.append(
            {
                "step": transition.get("high_level_step") or record.get("high_level_step"),
                "brain": brain,
                "tool_name": transition.get("tool_name"),
                "arguments": transition.get("arguments") or {},
                "success": transition.get("success"),
                "frames": transition.get("frames"),
                "env_feedback": transition.get("env_feedback") or {},
                "state_delta": transition.get("state_delta"),
                "scheduler_events": scheduler_events,
                "task_success": goal.get("task_success"),
                "task_progress": goal.get("task_progress"),
                "agent_pose": transition.get("agent_pose") or {},
            }
        )
    return rows


def build_predicate_progress(initial_goal: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [{"step": 0, "tool": "reset", "goal_result": dict(initial_goal)}]
    for transition in transitions:
        rows.append(
            {
                "step": transition.get("high_level_step"),
                "tool": transition.get("tool_name"),
                "goal_result": dict(transition.get("goal_result") or {}),
            }
        )
    return rows


def build_occupancy_path_report(
    env: BehaviorClosedLoopToolEnv,
    transitions: Sequence[Mapping[str, Any]],
    reset_obs: Mapping[str, Any],
) -> dict[str, Any]:
    core = env.core
    occupancy = getattr(core, "occupancy", None)
    if core is None or occupancy is None:
        return {"available": False}
    path_rows = []
    for transition in transitions:
        pose = transition.get("agent_pose") or {}
        if "x" not in pose or "y" not in pose:
            continue
        xy = (float(pose["x"]), float(pose["y"]))
        path_rows.append(
            {
                "step": transition.get("high_level_step"),
                "tool": transition.get("tool_name"),
                "success": transition.get("success"),
                "agent_xy": [xy[0], xy[1]],
                "component": occupancy.component_at(xy),
                "is_free": occupancy.is_free(xy),
                "frames": transition.get("frames"),
            }
        )
    reset_pose = reset_obs.get("agent_pose") or {}
    return {
        "available": True,
        "source": occupancy.source,
        "metadata": occupancy.metadata,
        "start_xy": [float(reset_pose.get("x", core.ego_state.pose.x)), float(reset_pose.get("y", core.ego_state.pose.y))],
        "transition_path": path_rows,
    }


def classify_episode_failure(result: Any) -> str | None:
    if str(getattr(result, "status", "")) == "passed" and float(result.final_goal.get("task_success", 0.0) or 0.0) == 1.0:
        return None
    if getattr(result, "failure_code", None):
        return str(result.failure_code)
    if not result.transitions:
        return "ORACLE_PLAN_FAILED"
    if any(not transition.get("success", False) for transition in result.transitions):
        return "TOOL_EXECUTION_FAILED"
    if float(result.final_goal.get("task_success", 0.0) or 0.0) != 1.0:
        return "GOAL_CHECK_FAILED"
    return None


def episode_artifacts(output_dir: Path, render_manifest: Mapping[str, Any] | None) -> dict[str, str]:
    artifacts = {
        "index": "index.html",
        "trace": "logs/trace.jsonl",
        "selected_task": "logs/selected_task.json",
        "task_spec": "logs/task_spec.json",
        "scene_reset": "logs/scene_reset.json",
        "render_identity_map": "logs/render_identity_map.json",
        "oracle_plan": "logs/oracle_plan.json",
        "tool_trace": "logs/tool_trace.json",
        "predicate_progress": "logs/predicate_progress.json",
        "metrics": "logs/metrics.json",
        "path_overlay": "path_overlay.png",
        "video_quality": "logs/video_quality.json",
    }
    if render_manifest:
        for key, value in (
            ("execution_video", render_manifest.get("video")),
            ("execution_srt", render_manifest.get("subtitle_path")),
            ("frames_dir", render_manifest.get("frames_dir")),
            ("render_manifest", render_manifest.get("manifest_path")),
        ):
            if value:
                artifacts[key] = _relative_or_str(Path(str(value)), output_dir)
    return artifacts


def write_stage9_index(path: Path, summary: Mapping[str, Any], episode_results: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 9 Render Video"),
        "<main>",
        "<h1>Stage 9 Render Video</h1>",
        "<p class='muted'>Task generation, symbolic oracle/tool execution, and GPU-gated RGB video evidence.</p>",
        "<section class='cards'>",
        _card(summary.get("status"), "Status"),
        _card(summary.get("symbolic_execution_passed"), "Symbolic Passed"),
        _card(summary.get("rendered_task_count"), "Rendered Tasks"),
        _card(summary.get("video_quality_passed"), "Quality Passed"),
        "</section>",
        "<h2>Artifacts</h2>",
        "<ul>",
        "<li><a href='task_generation.html'>task_generation.html</a></li>",
        "<li><a href='scene_reset.html'>scene_reset.html</a></li>",
        "<li><a href='oracle_plan_execution.html'>oracle_plan_execution.html</a></li>",
        "<li><a href='topdown_replay.html'>topdown_replay.html</a></li>",
        "<li><a href='video_quality_check.html'>video_quality_check.html</a></li>",
        "<li><a href='render_failure_matrix.html'>render_failure_matrix.html</a></li>",
        "<li><a href='logs/stage9_summary.json'>logs/stage9_summary.json</a></li>",
        "<li><a href='logs/render_failure_matrix.json'>logs/render_failure_matrix.json</a></li>",
        "</ul>",
        "<h2>Episode Samples</h2>",
        "<table><tr><th>Task</th><th>Category</th><th>Execution</th><th>Render</th><th>Quality</th><th>Failure Kind</th><th>Links</th></tr>",
    ]
    for item in episode_results:
        artifacts = item.get("artifacts") or {}
        links = ["<a href='{}'>episode</a>".format(escape(_episode_rel(item, "index")))]
        if artifacts.get("execution_video"):
            links.append("<a href='{}'>mp4</a>".format(escape(_episode_rel(item, "execution_video"))))
        if artifacts.get("path_overlay"):
            links.append("<a href='{}'>path</a>".format(escape(_episode_rel(item, "path_overlay"))))
        if artifacts.get("render_error"):
            links.append("<a href='{}'>render_error</a>".format(escape(_episode_rel(item, "render_error"))))
        failure = item.get("failure_summary") or {}
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(item.get("task_id"))),
                escape(str(item.get("category"))),
                yes_no(item.get("execution_ok")),
                escape(str(item.get("render_status"))),
                escape(str((item.get("video_quality") or {}).get("status"))),
                escape(str(failure.get("failure_kind") or "-")),
                " ".join(links),
            )
        )
    body.extend(["</table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_task_generation_html(path: Path, generation: Mapping[str, Any], selected_tasks: Sequence[Mapping[str, Any]]) -> None:
    summary = generation.get("summary") or {}
    body = [
        html_header("Stage 9 Task Generation"),
        "<main>",
        "<h1>Task Generation</h1>",
        "<section class='cards'>",
        _card(summary.get("task_count"), "Generated"),
        _card(summary.get("rejected_count"), "Rejected"),
        _card(len(selected_tasks), "Selected"),
        "</section>",
        "<h2>Selected Render Samples</h2>",
        "<table><tr><th>Task</th><th>Template</th><th>Category</th><th>Instruction</th></tr>",
    ]
    for task in selected_tasks:
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(task.get("id"))),
                escape(str((task.get("template") or {}).get("template_id"))),
                escape(str(task.get("category"))),
                escape(str(task.get("instruction"))),
            )
        )
    body.extend(["</table>", "<h2>Generated Categories</h2>", _dict_table(summary.get("categories") or {}), "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_scene_reset_html(path: Path, episode_results: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 9 Scene Reset"),
        "<main>",
        "<h1>Scene Reset</h1>",
        "<table><tr><th>Task</th><th>Stage 5</th><th>Initial Progress</th><th>Placement Log</th></tr>",
    ]
    for item in episode_results:
        initial_goal = item.get("initial_goal") or {}
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td><a href='{}'>scene_reset.json</a></td></tr>".format(
                escape(str(item.get("task_id"))),
                yes_no((item.get("stage5_result") or {}).get("stage5_reset_mutation_ok")),
                escape(str(initial_goal.get("task_progress"))),
                escape(_episode_rel(item, "scene_reset")),
            )
        )
    body.extend(["</table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_oracle_plan_execution_html(path: Path, episode_results: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 9 Oracle Plan Execution"),
        "<main>",
        "<h1>Oracle Plan And Execution</h1>",
    ]
    for item in episode_results:
        body.extend(
            [
                "<h2>{}</h2>".format(escape(str(item.get("task_id")))),
                "<p><strong>Teacher:</strong> <code>{}</code> <code>{}</code></p>".format(
                    escape(str((item.get("teacher") or {}).get("teacher_source"))),
                    escape(str((item.get("teacher") or {}).get("teacher_type"))),
                ),
                "<p><strong>Execution:</strong> {} task_success={} steps={}</p>".format(
                    yes_no(item.get("execution_ok")),
                    escape(str(item.get("task_success"))),
                    escape(str(item.get("transition_count"))),
                ),
                "<table><tr><th>Step</th><th>Planned Tool</th><th>Executed Tool</th><th>Success</th><th>Progress</th></tr>",
            ]
        )
        plan = list(item.get("expected_plan") or [])
        trace = list(item.get("tool_trace") or [])
        max_len = max(len(plan), len(trace))
        for idx in range(max_len):
            p = plan[idx] if idx < len(plan) else {}
            t = trace[idx] if idx < len(trace) else {}
            body.append(
                "<tr><td>{}</td><td><code>{}</code> {}</td><td><code>{}</code> {}</td><td>{}</td><td>{}</td></tr>".format(
                    idx + 1,
                    escape(str(p.get("tool_name") or "")),
                    escape(json.dumps(p.get("arguments") or {}, sort_keys=True)),
                    escape(str(t.get("tool_name") or "")),
                    escape(json.dumps(t.get("arguments") or {}, sort_keys=True)),
                    yes_no(t.get("success")),
                    escape(str(t.get("task_progress"))),
                )
            )
        body.append("</table>")
    body.extend(["</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_topdown_replay_html(path: Path, episode_results: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 9 Topdown Replay"),
        "<main>",
        "<h1>Top-down Replay Fallback</h1>",
        "<p class='muted'>This fallback is always produced, even when RGB rendering is blocked.</p>",
        "<div class='template-grid'>",
    ]
    for item in episode_results:
        overlay = _episode_rel(item, "path_overlay")
        media = (
            "<img src='{}' style='max-width:100%;border:1px solid var(--line)' alt='path overlay'>".format(escape(overlay))
            if overlay
            else "<p class='muted'>No top-down overlay was produced for this episode.</p>"
        )
        body.append(
            "<article class='template-card'><h2>{}</h2><p>{}</p>{}</article>".format(
                escape(str(item.get("category"))),
                escape(str(item.get("instruction"))),
                media,
            )
        )
    body.extend(["</div>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_video_quality_html(path: Path, summary: Mapping[str, Any], episode_results: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 9 Video Quality Check"),
        "<main>",
        "<h1>Video Quality Check</h1>",
        "<section class='cards'>",
        _card(summary.get("render_requested"), "Render Requested"),
        _card(summary.get("render_attempted"), "Render Attempted"),
        _card(summary.get("rendered_task_count"), "Rendered"),
        _card(summary.get("video_quality_passed"), "Quality Passed"),
        "</section>",
        "<table><tr><th>Task</th><th>Render</th><th>Status</th><th>Frames</th><th>Resolution</th><th>Static Ratio</th><th>Issues</th><th>Video</th></tr>",
    ]
    for item in episode_results:
        quality = item.get("video_quality") or {}
        artifacts = item.get("artifacts") or {}
        video_link = "<a href='{}'>mp4</a>".format(escape(_episode_rel(item, "execution_video"))) if artifacts.get("execution_video") else "-"
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}/{}</td><td>{}</td><td>{:.3f}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(item.get("task_id"))),
                escape(str(item.get("render_status"))),
                escape(str(quality.get("status"))),
                escape(str(quality.get("frame_dump_count"))),
                escape(str(quality.get("manifest_frame_count"))),
                escape(str(quality.get("frame_shape"))),
                float(quality.get("static_pair_ratio") or 0.0),
                escape("; ".join(quality.get("issues") or [])),
                video_link,
            )
        )
    body.extend(["</table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_render_failure_matrix_html(path: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 9 Render Failure Matrix"),
        "<main>",
        "<h1>Render Failure Matrix</h1>",
        "<p class='muted'>One render attempt per selected template task; failures are recorded without retrying.</p>",
        "<section class='cards'>",
        _card(summary.get("selected_task_count"), "Selected"),
        _card(summary.get("render_attempted"), "Render Attempted"),
        _card(summary.get("rendered_task_count"), "Rendered"),
        _card(summary.get("video_quality_passed"), "Quality Passed"),
        "</section>",
        "<h2>Failure Kinds</h2>",
        _dict_table(summary.get("failure_kind_counts") or {}),
        "<h2>Render Status Failures</h2>",
        _dict_table(summary.get("render_failure_counts") or {}),
        "<h2>Templates</h2>",
        "<table><tr><th>Template</th><th>Task</th><th>Category</th><th>Execution</th><th>Render</th><th>Quality</th><th>Failure Kind</th><th>Failure Code</th><th>Render Error</th><th>Video Issues</th><th>Recovered Tool Fails</th><th>Links</th></tr>",
    ]
    for row in rows:
        links = []
        if row.get("episode_link"):
            links.append("<a href='{}'>episode</a>".format(escape(str(row.get("episode_link")))))
        if row.get("video_link"):
            links.append("<a href='{}'>mp4</a>".format(escape(str(row.get("video_link")))))
        if row.get("render_error_link"):
            links.append("<a href='{}'>render_error</a>".format(escape(str(row.get("render_error_link")))))
        render_error = "{}{}".format(
            row.get("render_error_type") or "",
            ": " + str(row.get("render_error_message")) if row.get("render_error_message") else "",
        )
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(row.get("template_id") or "-")),
                escape(str(row.get("task_id") or "-")),
                escape(str(row.get("category") or "-")),
                yes_no(row.get("execution_ok")),
                escape(str(row.get("render_status") or "-")),
                escape(str(row.get("video_quality_status") or "-")),
                escape(str(row.get("failure_kind") or "-")),
                escape(str(row.get("failure_code") or "-")),
                escape(render_error or "-"),
                escape("; ".join(str(item) for item in row.get("video_quality_issues") or [])),
                escape(str(row.get("failed_tool_call_count") or 0)),
                " ".join(links) or "-",
            )
        )
    body.extend(["</table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_fatal_episode_index(
    path: Path,
    task_spec: Mapping[str, Any],
    summary: Mapping[str, Any],
    error: Mapping[str, Any],
    video_quality: Mapping[str, Any],
) -> None:
    body = [
        html_header("Stage 9 Fatal Episode"),
        "<main>",
        "<h1>{}</h1>".format(escape(str(task_spec.get("id")))),
        "<p>{}</p>".format(escape(str(task_spec.get("instruction")))),
        "<section class='cards'>",
        _card(summary.get("episode_status"), "Episode"),
        _card(summary.get("render_status"), "Render"),
        _card((summary.get("failure_summary") or {}).get("failure_kind"), "Failure Kind"),
        _card(video_quality.get("status"), "Video Quality"),
        "</section>",
        "<h2>Artifacts</h2><ul>",
        "<li><a href='logs/task_spec.json'>logs/task_spec.json</a></li>",
        "<li><a href='logs/fatal_error.json'>logs/fatal_error.json</a></li>",
        "<li><a href='logs/episode_summary.json'>logs/episode_summary.json</a></li>",
        "<li><a href='logs/video_quality.json'>logs/video_quality.json</a></li>",
        "</ul>",
        "<h2>Error</h2>",
        "<table><tr><th>Type</th><th>Message</th></tr><tr><td><code>{}</code></td><td>{}</td></tr></table>".format(
            escape(str(error.get("error_type") or "")),
            escape(str(error.get("error_message") or "")),
        ),
        "</main>",
        html_footer(),
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def write_episode_index(
    path: Path,
    task_spec: Mapping[str, Any],
    summary: Mapping[str, Any],
    expected_plan: Sequence[Mapping[str, Any]],
    tool_trace: Sequence[Mapping[str, Any]],
    video_quality: Mapping[str, Any],
) -> None:
    body = [
        html_header("Stage 9 Episode"),
        "<main>",
        "<h1>{}</h1>".format(escape(str(task_spec.get("id")))),
        "<p>{}</p>".format(escape(str(task_spec.get("instruction")))),
        "<section class='cards'>",
        _card(summary.get("episode_status"), "Episode"),
        _card(summary.get("task_success"), "Task Success"),
        _card(summary.get("render_status"), "Render"),
        _card(video_quality.get("status"), "Video Quality"),
        "</section>",
        "<h2>Artifacts</h2><ul>",
        "<li><a href='path_overlay.png'>path_overlay.png</a></li>",
        "<li><a href='logs/trace.jsonl'>logs/trace.jsonl</a></li>",
        "<li><a href='logs/tool_trace.json'>logs/tool_trace.json</a></li>",
        "<li><a href='logs/video_quality.json'>logs/video_quality.json</a></li>",
    ]
    if (path.parent / "execution.mp4").exists():
        body.append("<li><a href='execution.mp4'>execution.mp4</a></li>")
    if (path.parent / "execution.srt").exists():
        body.append("<li><a href='execution.srt'>execution.srt</a></li>")
    body.extend(["</ul>", "<h2>Plan vs Execution</h2>", "<table><tr><th>Step</th><th>Plan</th><th>Execution</th><th>Success</th></tr>"])
    for idx in range(max(len(expected_plan), len(tool_trace))):
        plan = expected_plan[idx] if idx < len(expected_plan) else {}
        trace = tool_trace[idx] if idx < len(tool_trace) else {}
        body.append(
            "<tr><td>{}</td><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>".format(
                idx + 1,
                escape(str(plan.get("tool_name") or "")),
                escape(str(trace.get("tool_name") or "")),
                yes_no(trace.get("success")),
            )
        )
    body.extend(["</table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def _copy_first_existing(episode_results: Sequence[Mapping[str, Any]], artifact_key: str, target: Path) -> None:
    for item in episode_results:
        artifacts = item.get("artifacts") or {}
        rel = artifacts.get(artifact_key)
        if not rel:
            continue
        source = Path(str(item.get("output_dir"))) / rel
        if source.exists() and source.is_file():
            shutil.copyfile(source, target)
            return


def _episode_rel(item: Mapping[str, Any], artifact_key: str) -> str:
    artifacts = item.get("artifacts") or {}
    rel = str(artifacts.get(artifact_key) or "")
    if not rel:
        return ""
    base = Path(str(item.get("output_dir"))).name
    return "episodes/{}/{}".format(base, rel)


def _relative_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _card(value: Any, label: str) -> str:
    return "<article><div class='num'>{}</div><div>{}</div></article>".format(escape(str(value)), escape(label))


def _dict_table(data: Mapping[str, Any]) -> str:
    rows = ["<table><tr><th>Key</th><th>Value</th></tr>"]
    for key, value in sorted(data.items()):
        rows.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(key)), escape(str(value))))
    rows.append("</table>")
    return "\n".join(rows)


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


__all__ = [
    "STAGE9_SAMPLE_CATEGORIES",
    "analyze_video_quality",
    "build_stage9_render_video_report",
    "probe_render_environment",
    "select_stage9_tasks",
    "visual_mutation_issues_from_tool_trace",
    "visual_mutation_issues_from_transitions",
    "with_visual_mutation_quality_issues",
]
