"""Video recorder hook for scoper.benchmark.behavior EpisodeRunner (P5).

CRITICAL design note: the scoper.benchmark.behavior runner only fires ``on_episode_start`` /
``before_tool`` / ``after_tool`` / ``on_episode_end`` (NOT ``on_low_level`` — confirmed by
the stage-7 API map). So this hook captures frames at STEP BOUNDARIES, not from a low-level
frame stream. For smooth navigation it harvests the per-action ``frame_pose`` events the
EgoSymbolicCore emits to ``env.core.trace`` between before_tool and after_tool (via a trace
cursor), rendering one ego frame per harvested pose; 0-frame actions (pick/place/toggle)
get a single held frame at the resulting pose. Navigation can either keep every symbolic
pose or downsample to a fixed cap, and can repeat frames according to a requested
meters-per-second speed so video motion is not artificially jumpy.

The hook is renderer-agnostic: ``renderer`` must expose ``render_at_pose(ego_pose) ->
np.ndarray (HxWx3 uint8)``. On GPU that's ``render.ego_renderer.EgoRenderer``; in CPU tests
a fake renderer returning numpy frames is injected. Read-only w.r.t. env/transition state.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Mapping, Optional

from scoper.benchmark.behavior.hooks.base import Hook, HookEvent
from scoper.benchmark.behavior.render import subtitles
from scoper.benchmark.behavior.render.video import SymbolicVideoRecorder


class VideoRecorderHook(Hook):
    name = "video_recorder"

    def __init__(
        self,
        renderer: Any,
        recorder: SymbolicVideoRecorder,
        *,
        nav_max_frames: int = 0,
        navigation_speed_mps: float | None = 0.4,
        navigation_turn_speed_radps: float | None = 0.8,
        action_hold_seconds: float = 0.6,
        final_hold_seconds: float = 1.0,
        record_reset_frame: bool = True,
    ):
        self.renderer = renderer
        self.recorder = recorder
        self.nav_max_frames = int(nav_max_frames)
        self.navigation_speed_mps = None if navigation_speed_mps is None else float(navigation_speed_mps)
        self.navigation_turn_speed_radps = (
            None if navigation_turn_speed_radps is None else float(navigation_turn_speed_radps)
        )
        self.action_hold_seconds = float(action_hold_seconds)
        self.final_hold_seconds = float(final_hold_seconds)
        self.record_reset_frame = bool(record_reset_frame)
        self._trace_cursor = 0
        self._pending_start_subtitle = ("", "")
        self._manifest: Optional[dict] = None

    # ---- lifecycle ----
    def on_episode_start(self, env, task_spec, reset_observation) -> Optional[List[HookEvent]]:
        self._trace_cursor = self._trace_len(env)
        if self.record_reset_frame:
            short, srt = subtitles.episode_start_subtitle(task_spec)
            self._sync_renderer_from_core(env)
            pose = env.core.ego_state.pose
            frame = self._render_stable_pose(pose)
            self.recorder.hold(
                frame,
                seconds=self.action_hold_seconds,
                subtitle=short,
                srt_text=srt,
                label="reset",
                agent_pose=_pose_as_dict(pose),
            )
        return [HookEvent("video_started", "episode_start", {"video": str(self.recorder.video_path)})]

    def before_tool(self, env, tool_call, high_level_step) -> Optional[List[HookEvent]]:
        # Remember where the trace was so after_tool can harvest just this action's frames.
        self._trace_cursor = self._trace_len(env)
        self._pending_start_subtitle = subtitles.tool_start_subtitle(
            tool_call.tool_name, dict(tool_call.arguments or {}), high_level_step, tool_call.reasoning
        )
        return None

    def after_tool(self, env, tool_call, transition, high_level_step) -> Optional[List[HookEvent]]:
        start_short, start_srt = self._pending_start_subtitle
        # 1) Smooth motion frames: render each pose this action emitted to the trace.
        speed_resampled = self.navigation_speed_mps is not None and self.navigation_speed_mps > 0.0
        nav_payloads = self._harvest_frame_payloads(env, downsample=not speed_resampled)
        rendered_payloads = self._speed_resampled_payloads(nav_payloads) if speed_resampled else nav_payloads
        rendered_nav_frames = 0
        if rendered_payloads:
            # Drop the first render after an action segment starts. Isaac/RTX camera output can
            # briefly reflect a default / previous pose before it stabilizes at the requested pose.
            self._render_payload_frame(env, rendered_payloads[0])
        for payload in rendered_payloads:
            pose, frame = self._render_payload_frame(env, payload)
            self.recorder.add_frame(
                frame,
                subtitle=start_short,
                srt_text=start_srt,
                label=tool_call.tool_name,
                agent_pose=_pose_as_dict(pose),
            )
            rendered_nav_frames += 1
        # 2) Result frame held with the outcome subtitle (always at least one frame per step).
        result_short, result_srt = subtitles.tool_result_subtitle(tool_call.tool_name, transition)
        self._sync_renderer_from_core(env)
        result_pose = env.core.ego_state.pose
        final_frame = self._render_stable_pose(result_pose)
        self.recorder.hold(
            final_frame, seconds=self.action_hold_seconds, subtitle=result_short, srt_text=result_srt,
            label="{}_result".format(tool_call.tool_name),
            agent_pose=_pose_as_dict(result_pose),
        )
        return [
            HookEvent(
                "video_step",
                "after_tool",
                {
                    "high_level_step": high_level_step,
                    "nav_pose_count": len(nav_payloads),
                    "rendered_nav_pose_count": len(rendered_payloads),
                    "nav_frames": rendered_nav_frames,
                    "navigation_speed_mps": self.navigation_speed_mps,
                    "navigation_turn_speed_radps": self.navigation_turn_speed_radps,
                    "speed_resampled": speed_resampled,
                    "nav_max_frames": self.nav_max_frames,
                },
            )
        ]

    def on_scheduler_events(self, env, events, phase, high_level_step) -> Optional[List[HookEvent]]:
        emitted: list[HookEvent] = []
        for event in events or []:
            short, srt = subtitles.scheduler_event_subtitle(event)
            if not srt:
                continue
            self._sync_renderer_from_core(env)
            pose = env.core.ego_state.pose
            frame = self._render_stable_pose(pose)
            self.recorder.hold(
                frame,
                seconds=self.action_hold_seconds,
                subtitle=short,
                srt_text=srt,
                label=str(event.get("name") or "scheduler_event"),
                agent_pose=_pose_as_dict(pose),
            )
            emitted.append(
                HookEvent(
                    "video_scheduler_event",
                    "scheduler",
                    {
                        "high_level_step": high_level_step,
                        "phase": phase,
                        "scheduler_event": event.get("name"),
                    },
                )
            )
        return emitted or None

    def on_episode_end(self, env, final_goal, status) -> Optional[List[HookEvent]]:
        short, srt = subtitles.episode_end_subtitle(final_goal, status)
        if self.final_hold_seconds > 0.0:
            self._sync_renderer_from_core(env)
            pose = env.core.ego_state.pose
            frame = self._render_stable_pose(pose)
            self.recorder.hold(
                frame,
                seconds=self.final_hold_seconds,
                subtitle=short,
                srt_text=srt,
                label="final",
                agent_pose=_pose_as_dict(pose),
            )
        self._manifest = self.recorder.close()
        return [HookEvent("video_finalized", "episode_end", dict(self._manifest))]

    @property
    def manifest(self) -> Optional[dict]:
        return self._manifest

    # ---- helpers ----
    @staticmethod
    def _trace_len(env) -> int:
        core = getattr(env, "core", None)
        trace = getattr(core, "trace", None)
        events = getattr(trace, "events", None)
        return len(events) if events is not None else 0

    def _harvest_frame_payloads(self, env, *, downsample: bool = True) -> List[Mapping[str, Any]]:
        core = getattr(env, "core", None)
        trace = getattr(core, "trace", None)
        events = list(getattr(trace, "events", []) or [])
        new_events = events[self._trace_cursor:]
        payloads = [dict(e.payload) for e in new_events if e.event == "frame_pose" and "pose" in e.payload]
        self._trace_cursor = len(events)
        return _downsample(payloads, self.nav_max_frames) if downsample else payloads

    def _sync_renderer_from_core(self, env) -> None:
        if hasattr(self.renderer, "sync_from_core"):
            self.renderer.sync_from_core(getattr(env, "core", None))

    def _render_stable_pose(self, pose: Any) -> Any:
        self.renderer.render_at_pose(pose)
        return self.renderer.render_at_pose(pose)

    def _render_payload_frame(self, env, payload: Mapping[str, Any]) -> tuple[Any, Any]:
        pose = payload["pose"]
        if payload.get("held_pose") is not None and hasattr(self.renderer, "set_held_pose"):
            try:
                self.renderer.set_held_pose(payload["held_pose"], verify_readback=False, ego_pose=pose)
            except TypeError:
                self.renderer.set_held_pose(payload["held_pose"])
        else:
            self._sync_renderer_from_core(env)
        # Navigation can change camera yaw/pitch on every payload. Flush the
        # camera read at this exact pose, just as for the action-result frame;
        # otherwise RGB can still belong to the preceding navigation pose.
        return pose, self._render_stable_pose(pose)

    def _speed_resampled_payloads(self, payloads: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        if len(payloads) <= 1 or not self.navigation_speed_mps or self.navigation_speed_mps <= 0.0:
            return list(payloads)
        fps = max(1, int(self.recorder.fps or 30))
        meters_per_frame = float(self.navigation_speed_mps) / float(fps)
        radians_per_frame = None
        if self.navigation_turn_speed_radps is not None and self.navigation_turn_speed_radps > 0.0:
            radians_per_frame = float(self.navigation_turn_speed_radps) / float(fps)
        if meters_per_frame <= 0.0:
            return list(payloads)

        out: List[Mapping[str, Any]] = []
        previous = dict(payloads[0])
        for current_raw in payloads[1:]:
            current = dict(current_raw)
            previous_pose = previous["pose"]
            current_pose = current["pose"]
            distance = _pose_xy_distance(previous_pose, current_pose)
            yaw_delta = _pose_yaw_distance(previous_pose, current_pose)
            if distance <= 1e-9 and yaw_delta <= 1e-9:
                out.append(current)
                previous = current
                continue
            linear_steps = int(math.ceil(distance / meters_per_frame)) if distance > 1e-9 else 1
            angular_steps = 1
            if radians_per_frame is not None and yaw_delta > 1e-9:
                angular_steps = int(math.ceil(yaw_delta / radians_per_frame))
            steps = max(1, linear_steps, angular_steps)
            for step in range(1, steps + 1):
                t = step / float(steps)
                out.append(_interpolate_payload(previous, current, t))
            previous = current
        return out


def build_recorder(output_dir, episode_name: str, fps: int = 30, save_frames: bool = True) -> SymbolicVideoRecorder:
    output = Path(output_dir)
    return SymbolicVideoRecorder(
        video_path=output / "videos" / "{}.mp4".format(episode_name),
        frames_dir=output / "frames" / episode_name,
        fps=fps,
        save_frames=save_frames,
    )


def _downsample(items: List[Any], max_count: int) -> List[Any]:
    if max_count <= 0 or len(items) <= max_count:
        return list(items)
    import numpy as np

    idx = np.linspace(0, len(items) - 1, max_count).round().astype(int)
    seen: set = set()
    out: List[Any] = []
    for i in idx:
        i = int(i)
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def _pose_xy_distance(a: Any, b: Any) -> float:
    try:
        ax, ay = float(getattr(a, "x")), float(getattr(a, "y"))
        bx, by = float(getattr(b, "x")), float(getattr(b, "y"))
    except Exception:
        try:
            ax, ay = float(a["x"]), float(a["y"])
            bx, by = float(b["x"]), float(b["y"])
        except Exception:
            return 0.0
    return float(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)


def _pose_yaw_distance(a: Any, b: Any) -> float:
    return abs(_angle_delta(_pose_field(a, "yaw", 0.0), _pose_field(b, "yaw", 0.0)))


def _interpolate_payload(start: Mapping[str, Any], end: Mapping[str, Any], t: float) -> Mapping[str, Any]:
    t = max(0.0, min(1.0, float(t)))
    start_pose = start["pose"]
    end_pose = end["pose"]
    pose = _interpolate_pose(start_pose, end_pose, t)
    payload = dict(end)
    payload["pose"] = pose
    held_name = payload.get("held_object_name")
    if held_name:
        payload["held_pose"] = _held_pose_for_payload(pose, payload)
    return payload


def _interpolate_pose(start: Any, end: Any, t: float) -> Any:
    values = {
        "x": _lerp(_pose_field(start, "x", 0.0), _pose_field(end, "x", 0.0), t),
        "y": _lerp(_pose_field(start, "y", 0.0), _pose_field(end, "y", 0.0), t),
        "z": _lerp(_pose_field(start, "z", 1.55), _pose_field(end, "z", 1.55), t),
        "yaw": _lerp_angle(_pose_field(start, "yaw", 0.0), _pose_field(end, "yaw", 0.0), t),
        "pitch": _lerp(_pose_field(start, "pitch", 0.0), _pose_field(end, "pitch", 0.0), t),
    }
    pose_cls = type(end)
    try:
        return pose_cls(**values)
    except Exception:
        from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose

        return EgoPose(**values)


def _held_pose_for_payload(pose: Any, payload: Mapping[str, Any]) -> Any:
    try:
        from scoper.benchmark.behavior.execution_core.held_object import held_pose_from_ego

        return held_pose_from_ego(
            pose,
            object_name=str(payload.get("held_object_name") or ""),
            visual_name=payload.get("held_visual_name"),
        )
    except Exception:
        return payload.get("held_pose")


def _pose_as_dict(pose: Any) -> dict[str, float]:
    if hasattr(pose, "as_dict"):
        try:
            return dict(pose.as_dict())
        except Exception:
            pass
    return {
        "x": _pose_field(pose, "x", 0.0),
        "y": _pose_field(pose, "y", 0.0),
        "z": _pose_field(pose, "z", 0.0),
        "yaw": _pose_field(pose, "yaw", 0.0),
        "pitch": _pose_field(pose, "pitch", 0.0),
    }


def _pose_field(pose: Any, field: str, default: float) -> float:
    if isinstance(pose, Mapping):
        return float(pose.get(field, default))
    return float(getattr(pose, field, default))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _lerp_angle(a: float, b: float, t: float) -> float:
    delta = _angle_delta(a, b)
    return float(a + delta * t)


def _angle_delta(a: float, b: float) -> float:
    return (float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["VideoRecorderHook", "build_recorder"]
