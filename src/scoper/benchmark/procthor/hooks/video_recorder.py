"""Video recorder hook for EB-ProcTHOR EpisodeRunner.

Wraps :class:`SmoothVideoRecorder` so that running an episode through
``EpisodeRunner`` produces an MP4 trajectory equivalent to the previous
``oracle_tool_video_runner._install_recorder_callback`` path.

The hook is read-only with respect to env state and tool transitions. It
captures simulator events at three points:

- ``on_episode_start`` records the reset frame.
- ``on_low_level`` records each simulator substep frame.
- ``before_tool`` and ``after_tool`` hold the current view so action/result
  subtitles remain readable.
- ``on_episode_end`` holds for ``final_hold_seconds`` if the episode passed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

from scoper.benchmark.procthor.brains.base import ToolCall
from scoper.benchmark.procthor.closed_loop_tool_env import _agent_pose
from scoper.benchmark.procthor.hooks.base import Hook, HookEvent


class VideoRecorderHook(Hook):
    name = "video_recorder"

    def __init__(
        self,
        recorder: Any,
        *,
        low_level_frame_repeat: int = 1,
        pre_tool_hold_seconds: Optional[float] = None,
        post_tool_hold_seconds: float = 1.0,
        scheduler_event_hold_seconds: Optional[float] = None,
        final_hold_seconds: float = 0.0,
        record_reset_frame: bool = True,
        record_only_on_passed_final_hold: bool = False,
        enable_subtitles: bool = True,
    ):
        self.recorder = recorder
        self.low_level_frame_repeat = max(1, int(low_level_frame_repeat))
        self.post_tool_hold_seconds = float(post_tool_hold_seconds)
        self.pre_tool_hold_seconds = float(
            post_tool_hold_seconds
            if pre_tool_hold_seconds is None
            else pre_tool_hold_seconds
        )
        self.scheduler_event_hold_seconds = float(
            post_tool_hold_seconds
            if scheduler_event_hold_seconds is None
            else scheduler_event_hold_seconds
        )
        self.final_hold_seconds = float(final_hold_seconds)
        self.record_reset_frame = bool(record_reset_frame)
        self.record_only_on_passed_final_hold = bool(record_only_on_passed_final_hold)
        self.enable_subtitles = bool(enable_subtitles)
        self._current_subtitle = ""
        self._current_srt_text = ""
        self._closed = False
        self._manifest: Optional[dict] = None

    def on_episode_start(
        self,
        env: Any,
        task_spec: Mapping[str, Any],
        reset_observation: Mapping[str, Any],
    ) -> Optional[List[HookEvent]]:
        self._set_subtitles(*_episode_start_subtitle(task_spec, reset_observation))
        setattr(env, "_artifact_video_event_bounds", [])
        if self.record_reset_frame:
            self.recorder.capture_event(
                env.controller.last_event,
                {
                    "action": "reset",
                    "label": "reset",
                    "success": True,
                    "agent_pose": _agent_pose(env.controller.last_event.metadata),
                    "subtitle": self._subtitle(),
                    "srt_text": self._srt_text(),
                },
            )
        return [HookEvent("video_started", "episode_start", {"video": str(self.recorder.video_path)})]

    def before_tool(
        self,
        env: Any,
        tool_call: ToolCall,
        high_level_step: int,
    ) -> Optional[List[HookEvent]]:
        self._set_subtitles(*_tool_start_subtitle(env, tool_call, high_level_step))
        if self.pre_tool_hold_seconds > 0.0:
            self._hold_current(
                env,
                seconds=self.pre_tool_hold_seconds,
                label="before {}".format(tool_call.tool_name),
                phase="before_tool",
                high_level_step=high_level_step,
                tool_name=tool_call.tool_name,
                subtitle=self._subtitle(),
                srt_text=self._srt_text(),
            )
        return [
            HookEvent(
                "video_subtitle",
                "before_tool",
                {
                    "text": self._current_subtitle,
                    "srt_text": self._current_srt_text,
                    "high_level_step": high_level_step,
                },
            )
        ]

    def on_low_level(
        self,
        env: Any,
        action_record: Mapping[str, Any],
        event: Any,
    ) -> Optional[List[HookEvent]]:
        record = dict(action_record)
        record.setdefault("subtitle", self._subtitle())
        record.setdefault("srt_text", self._srt_text())
        self.recorder.capture_event(
            event,
            record,
            repeat=self.low_level_frame_repeat,
        )
        return None

    def on_scheduler_events(
        self,
        env: Any,
        events: Iterable[Mapping[str, Any]],
        phase: str,
        high_level_step: int,
    ) -> Optional[List[HookEvent]]:
        emitted: List[HookEvent] = []
        for event in events or []:
            subtitle, srt_text = _scheduler_subtitle(event)
            if not subtitle and not srt_text:
                continue
            self._set_subtitles(subtitle, srt_text)
            if self.scheduler_event_hold_seconds > 0.0:
                self._hold_current(
                    env,
                    seconds=self.scheduler_event_hold_seconds,
                    label="scheduler {}".format(event.get("name") or "event"),
                    phase="scheduler",
                    high_level_step=high_level_step,
                    event_name=str(event.get("name") or ""),
                    subtitle=self._subtitle(),
                    srt_text=self._srt_text(),
                )
            else:
                _publish_video_bounds(
                    env,
                    {
                        "phase": "scheduler",
                        "high_level_step": high_level_step,
                        "event_name": str(event.get("name") or ""),
                        "label": "scheduler {}".format(event.get("name") or "event"),
                        "start_seconds": _recorder_seconds(self.recorder),
                        "end_seconds": _recorder_seconds(self.recorder),
                    },
                )
            emitted.append(
                HookEvent(
                    "video_subtitle",
                    "scheduler",
                    {
                        "text": self._current_subtitle,
                        "srt_text": self._current_srt_text,
                        "phase": phase,
                        "high_level_step": high_level_step,
                    },
                )
            )
        return emitted or None

    def after_tool(
        self,
        env: Any,
        tool_call: ToolCall,
        transition: Mapping[str, Any],
        high_level_step: int,
    ) -> Optional[List[HookEvent]]:
        self._set_subtitles(*_tool_result_subtitle(tool_call, transition))
        if self.post_tool_hold_seconds > 0.0:
            self._hold_current(
                env,
                seconds=self.post_tool_hold_seconds,
                label="after {}".format(tool_call.tool_name),
                phase="after_tool",
                high_level_step=high_level_step,
                tool_name=tool_call.tool_name,
                subtitle=self._subtitle(),
                srt_text=self._srt_text(),
            )
        else:
            _publish_video_bounds(
                env,
                {
                    "phase": "after_tool",
                    "high_level_step": high_level_step,
                    "tool_name": tool_call.tool_name,
                    "label": "after {}".format(tool_call.tool_name),
                    "start_seconds": _recorder_seconds(self.recorder),
                    "end_seconds": _recorder_seconds(self.recorder),
                },
            )
        return [
            HookEvent(
                "video_subtitle",
                "after_tool",
                {
                    "text": self._current_subtitle,
                    "srt_text": self._current_srt_text,
                    "high_level_step": high_level_step,
                },
            )
        ]

    def on_episode_end(
        self,
        env: Any,
        final_goal: Mapping[str, Any],
        status: str,
    ) -> Optional[List[HookEvent]]:
        self._set_subtitles(
            "Episode {}. Final task_success={}.".format(
                status,
                final_goal.get("task_success"),
            ),
            "\n".join(
                [
                    "Episode: {}".format(status),
                    "Final task_success: {}".format(final_goal.get("task_success")),
                    "Final task_progress: {}".format(final_goal.get("task_progress")),
                ]
            ),
        )
        if self.final_hold_seconds > 0.0:
            if not self.record_only_on_passed_final_hold or status == "passed":
                self._hold_current(
                    env,
                    seconds=self.final_hold_seconds,
                    label="final",
                    phase="episode_end",
                    subtitle=self._subtitle(),
                    srt_text=self._srt_text(),
                )
        else:
            _publish_video_bounds(
                env,
                {
                    "phase": "episode_end",
                    "label": "final",
                    "start_seconds": _recorder_seconds(self.recorder),
                    "end_seconds": _recorder_seconds(self.recorder),
                },
            )
        self._manifest = self.recorder.close()
        self._closed = True
        return [HookEvent("video_finalized", "episode_end", dict(self._manifest))]

    @property
    def manifest(self) -> Optional[dict]:
        return self._manifest

    def _subtitle(self) -> str:
        if not self.enable_subtitles:
            return ""
        return _shorten(self._current_subtitle, 220)

    def _srt_text(self) -> str:
        if not self.enable_subtitles:
            return ""
        return _shorten_multiline(self._current_srt_text or self._current_subtitle, 900)

    def _set_subtitles(self, subtitle: str, srt_text: Optional[str] = None) -> None:
        self._current_subtitle = str(subtitle or "")
        self._current_srt_text = str(srt_text if srt_text is not None else subtitle or "")

    def _hold_current(
        self,
        env: Any,
        *,
        seconds: float,
        label: str,
        phase: str,
        subtitle: str,
        srt_text: str,
        high_level_step: Optional[int] = None,
        tool_name: Optional[str] = None,
        event_name: Optional[str] = None,
    ) -> None:
        start = _recorder_seconds(self.recorder)
        self.recorder.hold_current(
            env,
            seconds=seconds,
            label=label,
            subtitle=subtitle,
            srt_text=srt_text,
        )
        end = _recorder_seconds(self.recorder)
        _publish_video_bounds(
            env,
            {
                "phase": phase,
                "high_level_step": high_level_step,
                "tool_name": tool_name,
                "event_name": event_name,
                "label": label,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": max(0.0, end - start),
            },
        )


def build_video_recorder(
    *,
    output_dir: Path,
    episode_name: str,
    fps: int,
    save_frames: bool,
    burn_in_subtitles: bool = True,
):
    """Construct a :class:`SmoothVideoRecorder` with the conventional layout."""

    from scoper.benchmark.procthor.oracle_tool_video_runner import SmoothVideoRecorder

    output = Path(output_dir)
    video_path = output / "videos" / "{}.mp4".format(episode_name)
    frames_dir = output / "frames" / episode_name
    return SmoothVideoRecorder(
        video_path=video_path,
        frames_dir=frames_dir,
        fps=fps,
        save_frames=save_frames,
        burn_in_subtitles=burn_in_subtitles,
    )


def _episode_start_subtitle(
    task_spec: Mapping[str, Any],
    reset_observation: Mapping[str, Any],
) -> tuple[str, str]:
    text = "Task: {}".format(task_spec.get("instruction", ""))
    srt_lines = [
        "Task: {}".format(task_spec.get("instruction", "")),
        "Category: {}".format(task_spec.get("category", "unknown")),
    ]
    context = _task_context_lines(task_spec)
    if context:
        srt_lines.append("Must consider: {}".format("; ".join(context[:5])))
    init_mutations = list(reset_observation.get("init_mutations") or [])
    if init_mutations:
        init_text = "; ".join(_mutation_text(m) for m in init_mutations[:3])
        text += " | Init adapted: {}".format(
            "; ".join(_mutation_text(m) for m in init_mutations[:3])
        )
        srt_lines.append("Scene initialization: {}".format(init_text))
    else:
        srt_lines.append("Scene initialization: no adaptation recorded.")
    return text, "\n".join(srt_lines)


def _tool_start_subtitle(
    env: Any,
    tool_call: ToolCall,
    high_level_step: int,
) -> tuple[str, str]:
    if tool_call.tool_name == "answer":
        question_id = tool_call.arguments.get("question_id")
        qa = (
            getattr(env, "_human_qa_state", {})
            .get("pending", {})
            .get(question_id, {})
        )
        question = qa.get("question") if isinstance(qa, Mapping) else None
        answer_set = qa.get("answer_set") if isinstance(qa, Mapping) else None
        value = tool_call.arguments.get("value")
        short = "Step {}: answer QA '{}' -> {}".format(high_level_step, question_id, value)
        lines = [
            "Action: answer QA '{}'".format(question_id),
            "Why: respond to the pending human question before continuing.",
        ]
        if question:
            lines.append("Question: {}".format(question))
        if answer_set:
            lines.append("Allowed answers: {}".format(", ".join(map(str, answer_set))))
        lines.append("Answer: {}".format(value))
        return short, "\n".join(lines)
    short = "Step {}: execute {}{}".format(
        high_level_step,
        tool_call.tool_name,
        _format_arguments(tool_call.arguments),
    )
    lines = [
        "Action: {}{}".format(
            tool_call.tool_name,
            _format_arguments(tool_call.arguments),
        )
    ]
    if tool_call.reasoning:
        short += " | {}".format(tool_call.reasoning)
        lines.append("Why: {}".format(tool_call.reasoning))
    else:
        lines.append("Why: {}".format(_default_tool_reason(tool_call)))
    return short, "\n".join(lines)


def _tool_result_subtitle(
    tool_call: ToolCall,
    transition: Mapping[str, Any],
) -> tuple[str, str]:
    status = "succeeded" if transition.get("success") else "failed"
    text = "{} {}.".format(tool_call.tool_name, status)
    lines = [
        "Result: {} {}".format(tool_call.tool_name, status),
    ]
    feedback = str(transition.get("env_feedback") or "").strip()
    if feedback:
        text += " {}".format(feedback)
        lines.append("Feedback: {}".format(feedback))
    if transition.get("done"):
        text += " Done: {}.".format(transition.get("done_reason"))
        lines.append("Done reason: {}".format(transition.get("done_reason")))
    goal = transition.get("goal_result") if isinstance(transition.get("goal_result"), Mapping) else {}
    if goal:
        lines.append(
            "Progress: task_success={}, task_progress={}".format(
                goal.get("task_success"),
                goal.get("task_progress"),
            )
        )
    raw = transition.get("raw_execution") if isinstance(transition.get("raw_execution"), Mapping) else {}
    low_level = raw.get("action") or _last_low_level_action(transition)
    if low_level:
        lines.append("Low-level action/result: {}".format(low_level))
    return text, "\n".join(lines)


def _scheduler_subtitle(event: Mapping[str, Any]) -> tuple[str, str]:
    name = str(event.get("name") or "")
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    if name == "human_qa_posted":
        question = payload.get("question")
        answer = payload.get("evaluator_answer")
        answer_set = payload.get("answer_set") or []
        lines = [
            "Human QA posted: {}".format(payload.get("id")),
            "Question: {}".format(question or ""),
        ]
        if answer_set:
            lines.append("Allowed answers: {}".format(", ".join(map(str, answer_set))))
        if answer is not None:
            lines.append("Expected/evaluator answer: {}".format(answer))
        lines.append("Effect: agent should answer this QA before continuing if it is relevant.")
        return "Human QA posted: {}. {}".format(payload.get("id"), question or ""), "\n".join(lines)
    if name == "human_insert_posted":
        instruction = payload.get("instruction")
        predicates = payload.get("goal_predicates") or []
        lines = [
            "Human inserted extra instruction: {}".format(payload.get("id")),
            "Instruction: {}".format(instruction or ""),
        ]
        if predicates:
            lines.append(
                "New obligations: {}".format(
                    "; ".join(_predicate_text(p) for p in predicates if isinstance(p, Mapping))
                )
            )
        lines.append("Effect: agent should incorporate this instruction into the remaining plan.")
        return (
            "Human inserted extra instruction: {}. {}".format(payload.get("id"), instruction or ""),
            "\n".join(lines),
        )
    if name == "human_qa_unanswered":
        ids = ", ".join(payload.get("ids") or [])
        return "Unanswered human QA: {}.".format(ids), "Unanswered human QA: {}".format(ids)
    if name.startswith("failure_"):
        short = "Injected failure: {} on {}.".format(
            name,
            payload.get("tool") or payload.get("config_tool") or "tool",
        )
        return short, _failure_event_text(name, payload)
    if name in {
        "close_object",
        "open_object",
        "toggle_on",
        "toggle_off",
        "dirty_object",
        "relocate_object",
        "restore_initial_state",
    }:
        short = "Scene disturbance: {} on {}.".format(
            name,
            payload.get("target_role") or payload.get("object_id") or "object",
        )
        return short, _scene_change_text(name, payload)
    return "", ""


def _format_arguments(arguments: Mapping[str, Any]) -> str:
    if not arguments:
        return "()"
    parts = ["{}={}".format(key, value) for key, value in sorted(arguments.items())]
    return "({})".format(_shorten(", ".join(parts), 90))


def _mutation_text(mutation: Mapping[str, Any]) -> str:
    typ = mutation.get("type") or "mutation"
    role = mutation.get("object_role") or "object"
    if typ == "relocate_object":
        return "relocated {} to {}".format(
            role,
            mutation.get("to_receptacle_role") or mutation.get("to_receptacle_object_id"),
        )
    return "{} on {}".format(typ, role)


def _task_context_lines(task_spec: Mapping[str, Any]) -> List[str]:
    lines: List[str] = []
    goal = task_spec.get("goal") if isinstance(task_spec.get("goal"), Mapping) else {}
    predicates = list(goal.get("predicates") or [])
    if predicates:
        lines.append("goals: {}".format("; ".join(_predicate_text(p) for p in predicates[:4])))
    if goal.get("require_model_stop"):
        lines.append("must call stop")
    req = task_spec.get("online_requirements")
    if isinstance(req, Mapping):
        expected_events = list(req.get("expected_events") or [])
        if expected_events:
            lines.append("expected events: {}".format(", ".join(map(str, expected_events[:4]))))
    disturbance = task_spec.get("disturbance_spec")
    if isinstance(disturbance, Mapping):
        disturbances = [
            str(d.get("type"))
            for d in disturbance.get("disturbances") or []
            if isinstance(d, Mapping)
        ]
        if disturbances:
            lines.append("possible runtime disturbances: {}".format(", ".join(disturbances)))
    return lines


def _predicate_text(predicate: Mapping[str, Any]) -> str:
    typ = predicate.get("type") or "predicate"
    obj = predicate.get("object_role")
    rec = predicate.get("receptacle_role")
    if obj and rec:
        return "{} {} -> {}".format(typ, obj, rec)
    if obj:
        return "{} {}".format(typ, obj)
    return str(typ)


def _default_tool_reason(tool_call: ToolCall) -> str:
    target = tool_call.arguments.get("target")
    if tool_call.tool_name == "navigate_to":
        if target:
            return "move near {} so the next operation can see or reach it.".format(target)
        return "move to a useful viewpoint for the next operation."
    if tool_call.tool_name == "scan":
        return "inspect nearby views to find or re-localize relevant objects."
    if tool_call.tool_name == "look_for":
        if target:
            return "bring {} into view before deciding the next action.".format(target)
        return "bring the requested object into view."
    if tool_call.tool_name == "pick_up":
        if target:
            return "hold {} so it can be moved to the required receptacle or device.".format(target)
        return "hold the target object for the next placement or preparation step."
    if tool_call.tool_name == "put_down":
        if target:
            return "place the held object at {} to satisfy the current goal predicate.".format(target)
        return "place the held object to satisfy the current goal predicate."
    if tool_call.tool_name == "open":
        if target:
            return "make {} accessible before placing or retrieving an object.".format(target)
        return "make the container accessible."
    if tool_call.tool_name == "close":
        if target:
            return "restore {} to the required closed state.".format(target)
        return "restore the container to the required closed state."
    if tool_call.tool_name == "toggle_on":
        if target:
            return "turn on {} because this device state is needed for the task.".format(target)
        return "turn on the device needed for the task."
    if tool_call.tool_name == "toggle_off":
        if target:
            return "turn off {} because the task requires or cleanup expects it.".format(target)
        return "turn off the device required by the task."
    if tool_call.tool_name == "slice":
        if target:
            return "slice {} to satisfy the preparation state goal.".format(target)
        return "slice the target object to satisfy the preparation state goal."
    if tool_call.tool_name == "stop":
        return "declare completion after the goal conditions have been satisfied."
    return "follow the teacher policy for the current task goal."


def _scene_change_text(name: str, payload: Mapping[str, Any]) -> str:
    target = payload.get("target_role") or payload.get("object_id") or "object"
    lines = [
        "Scene change: {} on {}".format(name, target),
        "Why it matters: the agent may need to detect the changed state and adapt the remaining actions.",
    ]
    before = payload.get("before") if isinstance(payload.get("before"), Mapping) else {}
    after = payload.get("after") if isinstance(payload.get("after"), Mapping) else {}
    before_text = _object_state_summary(before)
    after_text = _object_state_summary(after)
    activation_text = _activation_summary(payload.get("activation"))
    if activation_text:
        lines.append("Trigger: {}".format(activation_text))
    if before_text:
        lines.append("Before: {}".format(before_text))
    if after_text:
        lines.append("After: {}".format(after_text))
    if payload.get("old_position") and payload.get("new_position"):
        lines.append(
            "Position changed: {} -> {}".format(
                _position_text(payload.get("old_position")),
                _position_text(payload.get("new_position")),
            )
        )
    if payload.get("new_receptacle_object_id"):
        lines.append("Moved to receptacle: {}".format(payload.get("new_receptacle_object_id")))
    if payload.get("error_message"):
        lines.append("Simulator note: {}".format(payload.get("error_message")))
    return "\n".join(lines)


def _activation_summary(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return ""
    activation_type = value.get("type")
    if activation_type == "after_tool_result":
        bits = []
        if value.get("tool"):
            bits.append("after {} succeeded".format(value.get("tool")))
        if value.get("target_role"):
            bits.append("on {}".format(value.get("target_role")))
        if value.get("required_parent_role"):
            bits.append("while object was on {}".format(value.get("required_parent_role")))
        if value.get("distinct_tool_count") is not None:
            bits.append("{} other completed action(s) after arming".format(
                value.get("distinct_tool_count")
            ))
        armed_by = value.get("armed_by") if isinstance(value.get("armed_by"), Mapping) else {}
        if armed_by:
            bits.append(
                "armed by step {} {}({})".format(
                    armed_by.get("step"),
                    armed_by.get("tool"),
                    armed_by.get("target"),
                )
            )
        return "; ".join(str(bit) for bit in bits if bit)
    if activation_type:
        return str(activation_type)
    return ""


def _failure_event_text(name: str, payload: Mapping[str, Any]) -> str:
    lines = [
        "Injected failure: {}".format(name),
        "Affected tool: {}".format(payload.get("tool") or payload.get("config_tool") or "tool"),
    ]
    if payload.get("reason"):
        lines.append("Reason: {}".format(payload.get("reason")))
    if payload.get("message"):
        lines.append("Message: {}".format(payload.get("message")))
    lines.append("Effect: the agent should recover or the failure reason should be recorded.")
    return "\n".join(lines)


def _object_state_summary(obj: Mapping[str, Any]) -> str:
    if not obj:
        return ""
    parts = []
    for key in ("objectId", "objectType", "isOpen", "isToggled", "isDirty", "visible", "openness"):
        if key in obj:
            parts.append("{}={}".format(key, obj.get(key)))
    if obj.get("parentReceptacles"):
        parts.append("parentReceptacles={}".format(obj.get("parentReceptacles")))
    if isinstance(obj.get("position"), Mapping):
        parts.append("position={}".format(_position_text(obj.get("position"))))
    return ", ".join(parts)


def _position_text(position: Any) -> str:
    if not isinstance(position, Mapping):
        return str(position)
    return "({:.2f}, {:.2f}, {:.2f})".format(
        float(position.get("x", 0.0) or 0.0),
        float(position.get("y", 0.0) or 0.0),
        float(position.get("z", 0.0) or 0.0),
    )


def _last_low_level_action(transition: Mapping[str, Any]) -> str:
    actions = transition.get("low_level_actions")
    if not isinstance(actions, list) or not actions:
        return ""
    last = actions[-1]
    if not isinstance(last, Mapping):
        return ""
    action = last.get("action") or last.get("label")
    success = last.get("success")
    if success is None:
        return str(action or "")
    return "{} success={}".format(action, success)


def _recorder_seconds(recorder: Any) -> float:
    getter = getattr(recorder, "current_time_seconds", None)
    if callable(getter):
        return float(getter())
    frame_count = getattr(recorder, "_frame_count", 0)
    fps = float(getattr(recorder, "fps", 30) or 30)
    return float(frame_count) / fps


def _publish_video_bounds(env: Any, bounds: Mapping[str, Any]) -> None:
    queue = getattr(env, "_artifact_video_event_bounds", None)
    if not isinstance(queue, list):
        queue = []
        setattr(env, "_artifact_video_event_bounds", queue)
    queue.append(dict(bounds))


def _shorten(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _shorten_multiline(text: str, max_chars: int) -> str:
    lines = [" ".join(line.split()) for line in str(text or "").splitlines()]
    text = "\n".join(line for line in lines if line)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


__all__ = ["VideoRecorderHook", "build_video_recorder"]
