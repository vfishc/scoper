"""Record oracle closed-loop EB-ProcTHOR tool executions as videos.

This runner treats the VLM brain as perfect: it derives the correct tool calls
from the task predicates, then executes those calls through the public
``ProcTHORClosedLoopToolEnv`` interface. Videos are sampled from real simulator
substeps rather than from high-level tool-call keyframes.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scoper.benchmark.procthor.closed_loop_tool_env import ProcTHORClosedLoopToolEnv
from scoper.benchmark.procthor.manipulathor_config import manipulathor_env_kwargs
from scoper.benchmark.procthor.online_task_loader import load_task_bank
from scoper.benchmark.procthor.online_task_spec import validate_online_task_spec
from scoper.benchmark.procthor.paths import resolve_repo_path
from scoper.benchmark.procthor.scheduler import (
    CompositeScheduler,
    DisturbanceConfig,
    DisturbanceScheduler,
    FailureConfig,
    FailureScheduler,
    NoOpScheduler,
    Scheduler,
)


class OracleToolVideoError(ValueError):
    """Raised when oracle video recording cannot proceed."""


AI2THOR_VIDEO_FPS = 30
AI2THOR_VIDEO_MOVE_STEP = 1.0 / 60.0
AI2THOR_VIDEO_ROTATE_DEGREES = 90.0 / 60.0
AI2THOR_VIDEO_LOOK_DEGREES = 15.0 / 10.0
INSPECTION_VIDEO_MOVE_STEP = 0.05
INSPECTION_VIDEO_ROTATE_DEGREES = 5.0
INSPECTION_VIDEO_LOOK_DEGREES = 5.0
INSPECTION_VIDEO_LOW_LEVEL_FRAME_REPEAT = 1
INSPECTION_VIDEO_EVENT_HOLD_SECONDS = 1.0
INSPECTION_VIDEO_POST_TOOL_HOLD_SECONDS = INSPECTION_VIDEO_EVENT_HOLD_SECONDS
INSPECTION_VIDEO_FINAL_HOLD_SECONDS = 1.0


def run_oracle_tool_video_recording(
    task_bank_path: str,
    output_dir: str,
    task_ids: Optional[Sequence[str]] = None,
    first_per_category: bool = False,
    max_tasks: Optional[int] = None,
    width: int = 640,
    height: int = 480,
    cloud_rendering: bool = True,
    gpu_device: Optional[int] = 0,
    fps: int = AI2THOR_VIDEO_FPS,
    move_execution_step: Optional[float] = INSPECTION_VIDEO_MOVE_STEP,
    rotate_execution_degrees: Optional[float] = INSPECTION_VIDEO_ROTATE_DEGREES,
    look_execution_degrees: Optional[float] = INSPECTION_VIDEO_LOOK_DEGREES,
    snap_to_grid: Optional[bool] = None,
    low_level_frame_repeat: int = INSPECTION_VIDEO_LOW_LEVEL_FRAME_REPEAT,
    post_tool_hold_seconds: float = INSPECTION_VIDEO_POST_TOOL_HOLD_SECONDS,
    pre_tool_hold_seconds: Optional[float] = None,
    scheduler_event_hold_seconds: Optional[float] = None,
    final_hold_seconds: float = INSPECTION_VIDEO_FINAL_HOLD_SECONDS,
    save_frames: bool = True,
    scheduler_config: Optional[Mapping[str, Any]] = None,
    brain_kind: str = "oracle",
    burn_in_subtitles: bool = True,
    manipulathor_arm: bool = False,
    manipulathor_visual_attach: bool = True,
    agent_radius: Optional[float] = None,
    env_factory: Any = ProcTHORClosedLoopToolEnv,
) -> Tuple[List[dict], dict]:
    """Record selected tasks as smooth oracle tool-execution videos."""

    task_bank = load_task_bank(str(resolve_repo_path(task_bank_path)))
    tasks = _select_tasks(task_bank, task_ids=task_ids, first_per_category=first_per_category)
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]
    if not tasks:
        raise OracleToolVideoError("No tasks selected for oracle video recording.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "width": width,
        "height": height,
        "cloud_rendering": cloud_rendering,
        "gpu_device": gpu_device,
        "max_nav_steps_per_tool": 1000,
        "max_total_low_level_steps": 100000,
        "move_execution_step": move_execution_step,
        "rotate_execution_degrees": rotate_execution_degrees,
        "look_execution_degrees": look_execution_degrees,
        "snap_to_grid": snap_to_grid,
    }
    env_kwargs.update(
        manipulathor_env_kwargs(
            manipulathor_arm=manipulathor_arm,
            manipulathor_visual_attach=manipulathor_visual_attach,
            agent_radius=agent_radius,
        )
    )
    env = env_factory(**env_kwargs)
    records: List[dict] = []
    try:
        for episode_index, task in enumerate(tasks, start=1):
            records.append(
                record_oracle_tool_episode(
                    task,
                    env=env,
                    output_dir=output,
                    episode_index=episode_index,
                    fps=fps,
                    pre_tool_hold_seconds=pre_tool_hold_seconds,
                    post_tool_hold_seconds=post_tool_hold_seconds,
                    scheduler_event_hold_seconds=scheduler_event_hold_seconds,
                    final_hold_seconds=final_hold_seconds,
                    low_level_frame_repeat=low_level_frame_repeat,
                    save_frames=save_frames,
                    scheduler_config=scheduler_config,
                    brain_kind=brain_kind,
                    burn_in_subtitles=burn_in_subtitles,
                )
            )
    finally:
        env.close()

    summary = summarize_video_records(records, task_bank_size=len(task_bank))
    _write_jsonl(records, output / "episode_results.jsonl")
    _write_json(summary, output / "summary.json")
    return records, summary


def record_oracle_tool_episode(
    task_spec: Mapping[str, Any],
    env: ProcTHORClosedLoopToolEnv,
    output_dir: Path,
    episode_index: int,
    fps: int,
    pre_tool_hold_seconds: Optional[float],
    post_tool_hold_seconds: float,
    scheduler_event_hold_seconds: Optional[float],
    final_hold_seconds: float,
    low_level_frame_repeat: int,
    save_frames: bool,
    scheduler_config: Optional[Mapping[str, Any]] = None,
    brain_kind: str = "oracle",
    burn_in_subtitles: bool = True,
    simulator_ready_callback: Optional[Callable[[], None]] = None,
    trace_quality_gate: Any = None,
) -> dict:
    """Run one perfect-brain episode and record a smooth video.

    Implemented on top of :class:`EpisodeRunner` + a brain (``OracleBrain`` by
    default, or ``RecoveryOracleBrain`` if ``brain_kind=='recovery'``) +
    :class:`VideoRecorderHook`. The returned record schema is preserved so
    existing summary aggregation and CLI consumers keep working.
    """

    from scoper.benchmark.procthor.brains.recovery_oracle import RecoveryOracleBrain
    from scoper.benchmark.procthor.brains.teachers import get_dispatcher
    from scoper.benchmark.procthor.episode_runner import EpisodeRunner
    from scoper.benchmark.procthor.hooks import ExecutionTimelineHook, Hook, HookBus
    from scoper.benchmark.procthor.hooks.video_recorder import VideoRecorderHook

    validate_online_task_spec(task_spec)
    episode_name = "{:03d}_{}".format(episode_index, _safe_label(task_spec["id"]))
    video_path = output_dir / "videos" / "{}.mp4".format(episode_name)
    frames_dir = output_dir / "frames" / episode_name
    recorder = SmoothVideoRecorder(
        video_path=video_path,
        frames_dir=frames_dir,
        fps=fps,
        save_frames=save_frames,
        burn_in_subtitles=burn_in_subtitles,
    )

    video_hook = VideoRecorderHook(
        recorder,
        low_level_frame_repeat=low_level_frame_repeat,
        pre_tool_hold_seconds=pre_tool_hold_seconds,
        post_tool_hold_seconds=post_tool_hold_seconds,
        scheduler_event_hold_seconds=scheduler_event_hold_seconds,
        final_hold_seconds=final_hold_seconds,
        record_only_on_passed_final_hold=True,
    )
    timeline_hook = ExecutionTimelineHook(output_dir=output_dir, episode_name=episode_name)

    class _SimulatorReadyHook(Hook):
        name = "simulator_ready"

        def on_episode_start(self, env, task_spec, reset_observation):
            if simulator_ready_callback is not None:
                simulator_ready_callback()
            return None

    # Build scheduler from explicit config if provided; otherwise let the
    # runner auto-derive one from task_spec['disturbance_spec'].
    scheduler = _build_scheduler(scheduler_config) if scheduler_config else None

    brain = RecoveryOracleBrain() if brain_kind == "recovery" else get_dispatcher().build(task_spec)
    if trace_quality_gate is not None:
        brain = _TraceQualityBrain(brain, trace_quality_gate)
    runner = EpisodeRunner(env, require_passed_status=True)
    trace_path = output_dir / "logs" / "{}.trace.jsonl".format(episode_name)
    result = runner.run(
        task_spec,
        brain=brain,
        hooks=HookBus([video_hook, timeline_hook, _SimulatorReadyHook()]),
        scheduler=scheduler,
        trace_path=trace_path,
    )

    manifest = video_hook.manifest or {"frame_count": 0, "duration_seconds": 0.0}
    timeline_manifest = timeline_hook.manifest or {}

    record = {
        "episode_index": episode_index,
        "task_id": task_spec["id"],
        "category": task_spec["category"],
        "house_id": task_spec["source"]["house_id"],
        "instruction": task_spec["instruction"],
        "status": result.status,
        "failure_reason": result.failure_reason,
        "failure_type": result.failure_type,
        "failure_code": result.failure_code,
        "failure_details": _json_safe(result.failure_details),
        "tool_calls": [_json_safe(t) for t in result.transitions],
        "scheduler_events": _json_safe(result.scheduler_events),
        "init_mutations": _json_safe(result.init_mutations),
        "scene_mutations": _json_safe(result.scene_mutations),
        "hook_artifacts": _json_safe(result.hook_artifacts),
        "video": str(video_path),
        "frames_dir": str(frames_dir) if save_frames else None,
        "timeline": timeline_manifest.get("timeline"),
        "timeline_json": timeline_manifest.get("timeline_json"),
        "trace": str(trace_path),
        "fps": fps,
        "video_timing": {
            "move_execution_step": env.move_execution_step,
            "rotate_execution_degrees": env.rotate_execution_degrees,
            "look_execution_degrees": env.look_execution_degrees,
            "snap_to_grid": env.snap_to_grid,
            "allow_force_navigation_fallback": getattr(
                env, "allow_force_navigation_fallback", None
            ),
            "max_force_navigation_fallbacks_per_tool": getattr(
                env, "max_force_navigation_fallbacks_per_tool", None
            ),
            "allow_force_open_container_fallback": getattr(
                env, "allow_force_open_container_fallback", None
            ),
            "max_force_open_container_fallbacks_per_tool": getattr(
                env, "max_force_open_container_fallbacks_per_tool", None
            ),
            "allow_force_arm_open_close_collision_fallback": getattr(
                env, "allow_force_arm_open_close_collision_fallback", None
            ),
            "max_force_arm_open_close_collision_fallbacks_per_tool": getattr(
                env,
                "max_force_arm_open_close_collision_fallbacks_per_tool",
                None,
            ),
            "allow_public_placed_open_container_navigation": getattr(
                env,
                "allow_public_placed_open_container_navigation",
                None,
            ),
            "low_level_frame_repeat": low_level_frame_repeat,
            "pre_tool_hold_seconds": (
                post_tool_hold_seconds
                if pre_tool_hold_seconds is None
                else pre_tool_hold_seconds
            ),
            "post_tool_hold_seconds": post_tool_hold_seconds,
            "scheduler_event_hold_seconds": (
                post_tool_hold_seconds
                if scheduler_event_hold_seconds is None
                else scheduler_event_hold_seconds
            ),
            "final_hold_seconds": final_hold_seconds,
            "manipulathor_arm": getattr(env, "manipulathor_arm", None),
            "manipulathor_visual_attach_pickup": getattr(
                env,
                "manipulathor_visual_attach_pickup",
                None,
            ),
            "agent_radius": getattr(env, "agent_radius", None),
        },
        "final_goal": result.final_goal,
        "model_stopped": bool(getattr(env, "model_stopped", False)),
        "video_manifest": manifest,
        "subtitle_path": manifest.get("subtitle_path"),
        "video_frame_count": manifest.get("frame_count", 0),
        "video_duration_seconds": manifest.get("duration_seconds", 0.0),
        "episode_elapsed_seconds": result.elapsed_seconds,
    }
    return record


class _TraceQualityBrain:
    def __init__(self, brain: Any, gate: Any) -> None:
        self._brain = brain
        self._gate = gate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._brain, name)

    def act(self, brain_input: Any) -> Any:
        from scoper.benchmark.common import OracleTraceQualityError

        if brain_input.last_transition is not None:
            report = self._gate.observe_transition(brain_input.last_transition)
            if report["status"] == "failed":
                raise OracleTraceQualityError(report)
        return self._brain.act(brain_input)


class SmoothVideoRecorder:
    """Write simulator observations to MP4 at a fixed real-time sampling rate."""

    def __init__(
        self,
        video_path: Path,
        frames_dir: Path,
        fps: int,
        save_frames: bool,
        burn_in_subtitles: bool = True,
    ):
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.fps = int(fps)
        self.save_frames = bool(save_frames)
        self.burn_in_subtitles = bool(burn_in_subtitles)
        self.frame_records: List[dict] = []
        self._writer = None
        self._frame_count = 0
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    def capture_event(
        self,
        event: Any,
        action_record: Mapping[str, Any],
        repeat: int = 1,
    ) -> dict:
        frame = getattr(event, "frame", None)
        if frame is None:
            return {}
        frame = _as_rgb_array(frame)
        subtitle = str(action_record.get("subtitle") or "")
        srt_text = str(action_record.get("srt_text") or subtitle)
        output_frame = (
            _draw_subtitle(frame, subtitle) if self.burn_in_subtitles and subtitle else frame
        )
        latest_record = {}
        for _ in range(max(1, int(repeat))):
            self._ensure_writer()
            self._writer.append_data(output_frame)
            image_path = None
            if self.save_frames:
                image_path = self.frames_dir / "frame_{:06d}.png".format(self._frame_count)
                Image.fromarray(output_frame).save(image_path)
            latest_record = {
                "frame_index": self._frame_count,
                "video_timestamp_seconds": self._frame_count / float(self.fps),
                "image_path": str(image_path) if image_path is not None else None,
                "action": action_record.get("action"),
                "label": action_record.get("label"),
                "subtitle": subtitle,
                "srt_text": srt_text,
                "agent_pose": action_record.get("agent_pose"),
            }
            self.frame_records.append(latest_record)
            self._frame_count += 1
        return latest_record

    def hold_current(
        self,
        env: ProcTHORClosedLoopToolEnv,
        seconds: float,
        label: str,
        subtitle: str = "",
        srt_text: str = "",
    ) -> None:
        frames = int(round(max(0.0, float(seconds)) * self.fps))
        if frames <= 0:
            return
        self.capture_event(
            env.controller.last_event,
            {
                "action": "Hold",
                "label": label,
                "success": True,
                "agent_pose": _agent_pose(env.controller.last_event.metadata),
                "subtitle": subtitle,
                "srt_text": srt_text or subtitle,
            },
            repeat=frames,
        )

    def current_time_seconds(self) -> float:
        return self._frame_count / float(self.fps) if self.fps else 0.0

    def close(self) -> dict:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        manifest = {
            "video": str(self.video_path),
            "frames_dir": str(self.frames_dir) if self.save_frames else None,
            "fps": self.fps,
            "frame_count": self._frame_count,
            "duration_seconds": self._frame_count / float(self.fps) if self.fps else 0.0,
            "burn_in_subtitles": self.burn_in_subtitles,
            "frame_records": self.frame_records,
        }
        manifest_path = self.video_path.with_suffix(".manifest.json")
        srt_path = self.video_path.with_suffix(".srt")
        _write_srt(self.frame_records, srt_path, fps=self.fps)
        _write_json(manifest, manifest_path)
        manifest["manifest_path"] = str(manifest_path)
        manifest["subtitle_path"] = str(srt_path)
        return manifest

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        import imageio.v2 as imageio

        self._writer = imageio.get_writer(
            str(self.video_path),
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
        )


def summarize_video_records(
    records: Sequence[Mapping[str, Any]],
    task_bank_size: Optional[int] = None,
) -> dict:
    category_counts = Counter(record["category"] for record in records)
    passed_category_counts = Counter(
        record["category"] for record in records if record["status"] == "passed"
    )
    failed_category_counts = Counter(
        record["category"] for record in records if record["status"] == "failed"
    )
    failure_reasons = Counter(
        record.get("failure_reason") or "unknown"
        for record in records
        if record["status"] == "failed"
    )
    failure_code_counts = Counter(
        record.get("failure_code") or "unknown"
        for record in records
        if record["status"] == "failed"
    )
    tool_counts = Counter()
    low_level_counts = Counter()
    for record in records:
        for call in record.get("tool_calls", []):
            tool_counts[call.get("tool_name") or "unknown"] += 1
            for low_level in call.get("low_level_actions", []):
                low_level_counts[low_level.get("action") or "unknown"] += 1

    return {
        "task_bank_size": task_bank_size,
        "num_tasks": len(records),
        "num_passed": sum(1 for record in records if record["status"] == "passed"),
        "num_failed": sum(1 for record in records if record["status"] == "failed"),
        "category_counts": dict(sorted(category_counts.items())),
        "passed_category_counts": dict(sorted(passed_category_counts.items())),
        "failed_category_counts": dict(sorted(failed_category_counts.items())),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "failure_code_counts": dict(sorted(failure_code_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "low_level_action_counts": dict(sorted(low_level_counts.items())),
        "videos": [record.get("video") for record in records],
        "total_video_frames": sum(int(record.get("video_frame_count", 0)) for record in records),
        "total_video_seconds": sum(float(record.get("video_duration_seconds", 0.0)) for record in records),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Record perfect-brain EB-ProcTHOR tool executions as videos."
    )
    parser.add_argument("--task-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids", default=None)
    parser.add_argument("--first-per-category", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-cloud-rendering", action="store_true")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--fps", type=int, default=AI2THOR_VIDEO_FPS)
    parser.add_argument(
        "--smooth-video-preset",
        choices=("inspection_fast", "ai2thor_video", "none"),
        default="inspection_fast",
        help=(
            "inspection_fast uses 0.05m moves and 5deg rotations at 30fps; "
            "ai2thor_video uses 1m/60 frames and 90deg/60 frames at 30fps."
        ),
    )
    parser.add_argument("--move-execution-step", type=float, default=None)
    parser.add_argument("--rotate-execution-degrees", type=float, default=None)
    parser.add_argument("--look-execution-degrees", type=float, default=None)
    parser.add_argument(
        "--snap-to-grid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="AI2-THOR snapToGrid override. Defaults off when MoveAhead substeps are enabled.",
    )
    parser.add_argument(
        "--low-level-frame-repeat",
        type=int,
        default=INSPECTION_VIDEO_LOW_LEVEL_FRAME_REPEAT,
        help="Repeat each low-level simulator frame this many times in the output video.",
    )
    parser.add_argument("--pre-tool-hold-seconds", type=float, default=None)
    parser.add_argument("--post-tool-hold-seconds", type=float, default=None)
    parser.add_argument("--scheduler-event-hold-seconds", type=float, default=None)
    parser.add_argument("--final-hold-seconds", type=float, default=None)
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--no-burn-in-subtitles", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument(
        "--manipulathor-arm",
        action="store_true",
        help="Use AI2-THOR's official ManipulaTHOR arm agent mode.",
    )
    parser.add_argument(
        "--manipulathor-visual-attach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the ManipulaTHOR arm and simulate pickup/put-down visually without physical arm pickup.",
    )
    parser.add_argument(
        "--agent-radius",
        type=float,
        default=None,
        help="Optional AI2-THOR SetAgentRadius override in meters. Visual-attach mode defaults to 0.3.",
    )
    parser.add_argument(
        "--brain",
        choices=("oracle", "recovery"),
        default="oracle",
        help="Brain type: 'oracle' (blind plan) or 'recovery' (feedback-driven re-planning).",
    )
    parser.add_argument(
        "--scheduler-config",
        type=str,
        default=None,
        help="Path to a JSON file with scheduler configuration (disturbance/failure configs).",
    )
    args = parser.parse_args(argv)
    _apply_video_preset(args)

    scheduler_config = None
    if args.scheduler_config:
        scheduler_config = json.loads(Path(args.scheduler_config).read_text(encoding="utf-8"))

    _, summary = run_oracle_tool_video_recording(
        task_bank_path=args.task_bank,
        output_dir=args.output_dir,
        task_ids=args.task_ids,
        first_per_category=args.first_per_category,
        max_tasks=args.max_tasks,
        width=args.width,
        height=args.height,
        cloud_rendering=not args.no_cloud_rendering,
        gpu_device=args.gpu_device,
        fps=args.fps,
        move_execution_step=args.move_execution_step,
        rotate_execution_degrees=args.rotate_execution_degrees,
        look_execution_degrees=args.look_execution_degrees,
        snap_to_grid=args.snap_to_grid,
        low_level_frame_repeat=args.low_level_frame_repeat,
        pre_tool_hold_seconds=args.pre_tool_hold_seconds,
        post_tool_hold_seconds=args.post_tool_hold_seconds,
        scheduler_event_hold_seconds=args.scheduler_event_hold_seconds,
        final_hold_seconds=args.final_hold_seconds,
        save_frames=not args.no_save_frames,
        scheduler_config=scheduler_config,
        brain_kind=args.brain,
        burn_in_subtitles=not args.no_burn_in_subtitles,
        manipulathor_arm=args.manipulathor_arm,
        manipulathor_visual_attach=args.manipulathor_visual_attach,
        agent_radius=args.agent_radius,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if summary["num_failed"] and not args.allow_failures:
        raise SystemExit(1)


def _apply_video_preset(args: argparse.Namespace) -> None:
    if args.smooth_video_preset == "none":
        if args.pre_tool_hold_seconds is None:
            args.pre_tool_hold_seconds = INSPECTION_VIDEO_EVENT_HOLD_SECONDS
        if args.post_tool_hold_seconds is None:
            args.post_tool_hold_seconds = INSPECTION_VIDEO_POST_TOOL_HOLD_SECONDS
        if args.scheduler_event_hold_seconds is None:
            args.scheduler_event_hold_seconds = INSPECTION_VIDEO_EVENT_HOLD_SECONDS
        if args.final_hold_seconds is None:
            args.final_hold_seconds = INSPECTION_VIDEO_FINAL_HOLD_SECONDS
        return
    if args.smooth_video_preset == "ai2thor_video":
        if args.fps == AI2THOR_VIDEO_FPS:
            args.fps = AI2THOR_VIDEO_FPS
        if args.move_execution_step is None:
            args.move_execution_step = AI2THOR_VIDEO_MOVE_STEP
        if args.rotate_execution_degrees is None:
            args.rotate_execution_degrees = AI2THOR_VIDEO_ROTATE_DEGREES
        if args.look_execution_degrees is None:
            args.look_execution_degrees = AI2THOR_VIDEO_LOOK_DEGREES
        if args.pre_tool_hold_seconds is None:
            args.pre_tool_hold_seconds = INSPECTION_VIDEO_EVENT_HOLD_SECONDS
        if args.post_tool_hold_seconds is None:
            args.post_tool_hold_seconds = INSPECTION_VIDEO_EVENT_HOLD_SECONDS
        if args.scheduler_event_hold_seconds is None:
            args.scheduler_event_hold_seconds = INSPECTION_VIDEO_EVENT_HOLD_SECONDS
        if args.final_hold_seconds is None:
            args.final_hold_seconds = 1.0
        return
    if args.move_execution_step is None:
        args.move_execution_step = INSPECTION_VIDEO_MOVE_STEP
    if args.rotate_execution_degrees is None:
        args.rotate_execution_degrees = INSPECTION_VIDEO_ROTATE_DEGREES
    if args.look_execution_degrees is None:
        args.look_execution_degrees = INSPECTION_VIDEO_LOOK_DEGREES
    if args.pre_tool_hold_seconds is None:
        args.pre_tool_hold_seconds = INSPECTION_VIDEO_EVENT_HOLD_SECONDS
    if args.post_tool_hold_seconds is None:
        args.post_tool_hold_seconds = INSPECTION_VIDEO_POST_TOOL_HOLD_SECONDS
    if args.scheduler_event_hold_seconds is None:
        args.scheduler_event_hold_seconds = INSPECTION_VIDEO_EVENT_HOLD_SECONDS
    if args.final_hold_seconds is None:
        args.final_hold_seconds = INSPECTION_VIDEO_FINAL_HOLD_SECONDS


def _select_tasks(
    task_bank: Mapping[str, Mapping[str, Any]],
    task_ids: Optional[Sequence[str]],
    first_per_category: bool,
) -> List[Mapping[str, Any]]:
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in task_bank]
        if missing:
            raise OracleToolVideoError("Unknown task ids: {}".format(", ".join(missing)))
        tasks = [task_bank[task_id] for task_id in task_ids]
    else:
        tasks = list(task_bank.values())
    if first_per_category:
        selected = []
        seen = set()
        for task in sorted(tasks, key=lambda item: (item["category"], item["id"])):
            if task["category"] in seen:
                continue
            seen.add(task["category"])
            selected.append(task)
        tasks = selected
    return tasks


def _as_rgb_array(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return np.ascontiguousarray(array)


def _draw_subtitle(frame: np.ndarray, subtitle: str) -> np.ndarray:
    text = " ".join(str(subtitle or "").split())
    if not text:
        return frame
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    font = _subtitle_font(width)
    margin = max(8, width // 50)
    max_text_width = max(1, width - 2 * margin)
    lines = _wrap_subtitle(text, draw, font, max_text_width, max_lines=3)
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [box[3] - box[1] for box in line_boxes] or [16]
    line_height = max(line_heights) + 4
    band_height = min(height // 3, line_height * len(lines) + 2 * margin)
    top = height - band_height
    draw.rectangle((0, top, width, height), fill=(0, 0, 0, 170))
    y = top + margin
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = max(margin, (width - (bbox[2] - bbox[0])) // 2)
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height
    return np.ascontiguousarray(Image.alpha_composite(image, overlay).convert("RGB"))


def _subtitle_font(width: int):
    size = max(14, min(28, width // 36))
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_subtitle(text: str, draw: ImageDraw.ImageDraw, font: Any, max_width: int, max_lines: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = "{} {}".format(current, word).strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if words and len(lines) == max_lines:
        consumed = " ".join(lines).split()
        if len(consumed) < len(words):
            lines[-1] = lines[-1].rstrip(".") + "..."
    return lines


def _write_srt(frame_records: Sequence[Mapping[str, Any]], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments = _subtitle_segments(frame_records, fps=fps)
    lines: List[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(
            "{} --> {}".format(
                _srt_timestamp(segment["start"]),
                _srt_timestamp(segment["end"]),
            )
        )
        lines.append(segment["text"])
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _subtitle_segments(frame_records: Sequence[Mapping[str, Any]], fps: int) -> List[dict]:
    segments: List[dict] = []
    current_text = None
    current_start = 0.0
    last_time = 0.0
    frame_duration = 1.0 / float(fps or 30)
    for record in frame_records:
        text = str(record.get("srt_text") or record.get("subtitle") or "").strip()
        timestamp = float(record.get("video_timestamp_seconds", 0.0) or 0.0)
        if not text:
            if current_text:
                segments.append(
                    {"start": current_start, "end": max(timestamp, current_start + frame_duration), "text": current_text}
                )
                current_text = None
            last_time = timestamp
            continue
        if text != current_text:
            if current_text:
                segments.append(
                    {"start": current_start, "end": max(timestamp, current_start + frame_duration), "text": current_text}
                )
            current_text = text
            current_start = timestamp
        last_time = timestamp
    if current_text:
        segments.append(
            {
                "start": current_start,
                "end": max(last_time + frame_duration, current_start + frame_duration),
                "text": current_text,
            }
        )
    return segments


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)


def _agent_pose(metadata: Mapping[str, Any]) -> dict:
    agent = metadata.get("agent") or {}
    return {
        "position": dict(agent.get("position") or {}),
        "rotation": dict(agent.get("rotation") or {}),
        "cameraHorizon": agent.get("cameraHorizon", 0.0),
        "isStanding": agent.get("isStanding"),
    }


def _build_scheduler(config: Optional[Mapping[str, Any]]) -> Scheduler:
    """Build a Scheduler from a JSON config dict.

    Expected format:
    {
        "seed": 42,
        "disturbances": [ {...DisturbanceConfig fields...}, ... ],
        "failures": [ {...FailureConfig fields...}, ... ]
    }
    """
    if not config:
        return NoOpScheduler()
    seed = int(config.get("seed", 0))
    children: list = []
    disturbances = config.get("disturbances")
    if disturbances:
        children.append(DisturbanceScheduler(configs=disturbances, seed=seed))
    failures = config.get("failures")
    if failures:
        children.append(FailureScheduler(configs=failures, seed=seed))
    if not children:
        return NoOpScheduler()
    if len(children) == 1:
        return children[0]
    return CompositeScheduler(children)


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return label.strip("_") or "item"


def _write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")


def _write_json(record: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(record), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "head_rgb"}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()


__all__ = [
    "OracleToolVideoError",
    "record_oracle_tool_episode",
    "run_oracle_tool_video_recording",
    "summarize_video_records",
]
