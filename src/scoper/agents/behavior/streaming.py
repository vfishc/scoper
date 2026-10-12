from __future__ import annotations
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence
from scoper.benchmark.common import LogicalEvidenceSampler
from scoper.benchmark.behavior.brains.base import ToolCall
from scoper.agents.behavior.config import StreamConfig
from scoper.agents.behavior.response_parser import MODEL_VISIBLE_TOOLS
from scoper.benchmark.behavior.json_utils import _json_safe
STREAM_SOURCES = {'fps_sample', 'tool_result', 'reset', 'interrupt', 'scheduler_event', 'episode_end'}

@dataclass
class StreamFrameObservation:
    stream_id: str
    task_id: str
    high_level_step: int
    active_tool_name: str
    active_tool_args: dict[str, Any]
    frame_index: int
    timestamp_s: float
    source: str
    is_keyframe: bool
    keyframe_reason: str | None
    agent_pose: dict[str, Any]
    head_rgb: Any | None
    visible_object_catalog: list[dict[str, Any]]
    prompt_metadata: dict[str, Any]
    privileged_metadata: dict[str, Any]

@dataclass
class StreamDecision:
    stream_decision: str
    tool_call: ToolCall | None = None
    reasoning: str = ''
    raw_response: str = ''
    parse_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def interrupted(self) -> bool:
        return self.stream_decision == 'interrupt' and self.tool_call is not None

@dataclass(frozen=True)
class TimestampedPose:
    pose: dict[str, Any]
    timestamp_s: float

class StreamObserver(Protocol):

    def before_stream_decision(self, frame: StreamFrameObservation) -> Mapping[str, Any]:
        ...

    def after_stream_decision(self, frame: StreamFrameObservation, decision: StreamDecision, *, vlm_called: bool, latency_seconds: float, correlation: Mapping[str, Any]) -> None:
        ...

class StreamResponseParser:

    def __init__(self, allowed_tools: Optional[Iterable[str]]=None) -> None:
        self.allowed_tools = set(allowed_tools or MODEL_VISIBLE_TOOLS)

    def set_allowed_tools(self, allowed_tools: Optional[Iterable[str]]) -> None:
        if allowed_tools:
            self.allowed_tools = set((str(tool) for tool in allowed_tools))

    def parse(self, raw_response: Any) -> StreamDecision:
        text = raw_response if isinstance(raw_response, str) else json.dumps(raw_response, ensure_ascii=False, default=str)
        candidates = _json_candidates(text)
        best_error = 'no JSON object found'
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as exc:
                best_error = str(exc)
                continue
            if not isinstance(data, Mapping):
                best_error = 'JSON object is not a mapping'
                continue
            decision = str(data.get('stream_decision') or '').strip().lower()
            reasoning = str(data.get('reasoning') or data.get('rationale') or '')
            if decision == 'continue':
                return StreamDecision('continue', reasoning=reasoning, raw_response=text)
            if decision != 'interrupt':
                best_error = "stream_decision must be 'continue' or 'interrupt'"
                continue
            name = data.get('tool_name') or data.get('name') or data.get('tool')
            args = data.get('arguments') or data.get('args') or {}
            if not isinstance(name, str) or name not in self.allowed_tools:
                return StreamDecision('continue', reasoning=reasoning, raw_response=text, parse_error='stream_invalid_interrupt_tool', metadata={'invalid_tool_name': name})
            if not isinstance(args, Mapping):
                args = {}
            return StreamDecision('interrupt', tool_call=ToolCall(name, dict(args), reasoning=reasoning), reasoning=reasoning, raw_response=text)
        return StreamDecision('continue', reasoning='stream_parse_error: {}'.format(best_error), raw_response=text, parse_error='stream_parse_error')

class StreamController:

    def __init__(self, brain: Any | None=None, *, config: StreamConfig | None=None, decision_fn: Callable[[StreamFrameObservation], StreamDecision | ToolCall | Mapping[str, Any] | str] | None=None, log_path: str | Path | None=None, mode: str='model_backed', pre_render_sample_rate_hz: float | None=None, render_candidate_fps: float | None=None, skipped_frame_settle_renders: int=0) -> None:
        if mode not in {'model_backed', 'passive_expert_capture'}:
            raise ValueError('stream controller mode is invalid')
        if mode == 'passive_expert_capture' and (brain is not None or decision_fn is not None):
            raise ValueError('passive expert capture cannot have a brain or decision function')
        self.brain = brain
        self.mode = mode
        self.config = config or StreamConfig()
        self.decision_fn = decision_fn
        self.log_path = Path(log_path) if log_path is not None else None
        self.pre_render_sampler = LogicalEvidenceSampler(rate_hz=pre_render_sample_rate_hz) if pre_render_sample_rate_hz is not None else None
        self.render_candidate_fps = float(render_candidate_fps) if render_candidate_fps is not None else None
        if self.render_candidate_fps is not None and self.render_candidate_fps <= 0.0:
            raise ValueError('render_candidate_fps must be positive')
        self.skipped_frame_settle_renders = int(skipped_frame_settle_renders)
        if self.skipped_frame_settle_renders < 0:
            raise ValueError('skipped_frame_settle_renders cannot be negative')
        self.records: list[dict[str, Any]] = []
        self._calls_for_tool = 0
        self._active_tool_key: tuple[int, str] | None = None
        self._observers: list[StreamObserver] = []
        self._fh = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.log_path.open('w', encoding='utf-8')

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def add_observer(self, observer: StreamObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: StreamObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def on_stream_start(self, context: Mapping[str, Any]) -> None:
        if self.pre_render_sampler is not None:
            self.pre_render_sampler.reset()
        if self.brain is not None and hasattr(self.brain, 'on_stream_start'):
            self.brain.on_stream_start(context)

    def should_render_frame(self, timestamp_s: float, *, is_keyframe: bool=False) -> bool:
        if self.pre_render_sampler is None:
            return True
        return self.pre_render_sampler.consider(timestamp_s, force=is_keyframe).emit

    def on_stream_end(self, summary: Mapping[str, Any]) -> None:
        if self.brain is not None and hasattr(self.brain, 'on_stream_end'):
            self.brain.on_stream_end(summary)
        self.close()

    def on_frame(self, frame: StreamFrameObservation) -> StreamDecision:
        correlations = [dict(observer.before_stream_decision(frame)) for observer in self._observers]
        for correlation in correlations:
            required = {'stream_id', 'stream_obs_id', 'frame_id'}
            if set(correlation) < required:
                raise RuntimeError('stream observer correlation is incomplete')
        correlation = correlations[0] if correlations else {}
        self._reset_call_budget_if_needed(frame)
        should_call = self._should_call_vlm(frame)
        started = time.time()
        if should_call:
            context_setter = getattr(self.brain, 'set_model_call_context', None)
            context = None
            if callable(context_setter) and correlation.get('model_call_id'):
                from scoper.agents.behavior.vlm_brain import ModelCallContext
                context = ModelCallContext(episode_id=str(correlation.get('episode_id') or ''), model_call_id=str(correlation['model_call_id']), call_type='stream_decision', decision_id=None, action_id=correlation.get('action_id'), stream_obs_id=correlation.get('stream_obs_id'), verification_id=None)
                context_setter(context)
            try:
                decision = self._call_decision(frame)
            finally:
                if context is not None:
                    context_setter(None)
            latency = time.time() - started
            self._calls_for_tool += 1
            vlm_called = True
        else:
            if self.mode == 'passive_expert_capture':
                decision = StreamDecision('no_call', reasoning='passive_expert_capture', metadata={'decision_source': 'passive_capture'})
            else:
                decision = StreamDecision('no_call', reasoning='stream_call_budget_exhausted', metadata={'decision_source': 'call_budget'})
            latency = 0.0
            vlm_called = False
        if correlation:
            decision.metadata['trajectory_correlation'] = dict(correlation)
        for observer, observer_correlation in zip(self._observers, correlations):
            observer.after_stream_decision(frame, decision, vlm_called=vlm_called, latency_seconds=latency, correlation=observer_correlation)
        record = stream_frame_log_record(frame, decision, vlm_called=vlm_called, latency_seconds=latency)
        self.records.append(record)
        if self._fh is not None:
            self._fh.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True))
            self._fh.write('\n')
            self._fh.flush()
        return decision

    def _reset_call_budget_if_needed(self, frame: StreamFrameObservation) -> None:
        key = (int(frame.high_level_step), str(frame.active_tool_name))
        if key != self._active_tool_key:
            self._active_tool_key = key
            self._calls_for_tool = 0

    def _should_call_vlm(self, frame: StreamFrameObservation) -> bool:
        if self.decision_fn is None and (not callable(getattr(self.brain, 'act_stream_frame', None))):
            return False
        max_calls = int(self.config.max_calls_per_tool)
        if max_calls <= 0:
            return True
        if frame.is_keyframe:
            return True
        return self._calls_for_tool < max_calls

    def _call_decision(self, frame: StreamFrameObservation) -> StreamDecision:
        try:
            if self.decision_fn is not None:
                raw = self.decision_fn(frame)
            elif self.brain is not None and hasattr(self.brain, 'act_stream_frame'):
                raw = self.brain.act_stream_frame(frame)
            else:
                return StreamDecision('continue', reasoning='stream_controller_no_brain')
        except Exception:
            raise
        return coerce_stream_decision(raw)

def coerce_stream_decision(value: Any) -> StreamDecision:
    if isinstance(value, StreamDecision):
        return value
    if isinstance(value, ToolCall):
        return StreamDecision('interrupt', tool_call=value, reasoning=value.reasoning)
    if isinstance(value, Mapping):
        return StreamResponseParser().parse(json.dumps(dict(value), ensure_ascii=False, default=str))
    if isinstance(value, str):
        return StreamResponseParser().parse(value)
    return StreamDecision('continue', reasoning='stream_decision_unrecognized_type')

def stream_frame_log_record(frame: StreamFrameObservation, decision: StreamDecision, *, vlm_called: bool, latency_seconds: float) -> dict[str, Any]:
    record = {'stream_id': frame.stream_id, 'task_id': frame.task_id, 'high_level_step': frame.high_level_step, 'active_tool_name': frame.active_tool_name, 'active_tool_args': dict(frame.active_tool_args), 'frame_index': frame.frame_index, 'timestamp_s': frame.timestamp_s, 'source': frame.source, 'is_keyframe': frame.is_keyframe, 'keyframe_reason': frame.keyframe_reason, 'agent_pose': dict(frame.agent_pose), 'visible_object_catalog': list(frame.visible_object_catalog), 'prompt_metadata': dict(frame.prompt_metadata), 'privileged_metadata': dict(frame.privileged_metadata), 'vlm_called': bool(vlm_called), 'stream_decision': decision.stream_decision, 'reasoning': decision.reasoning, 'raw_response': decision.raw_response, 'latency_seconds': float(latency_seconds)}
    if decision.parse_error:
        record['stream_parse_error'] = decision.parse_error
    if decision.tool_call is not None:
        record['interrupt_tool_call'] = {'tool_name': decision.tool_call.tool_name, 'arguments': dict(decision.tool_call.arguments), 'reasoning': decision.tool_call.reasoning}
    if decision.metadata:
        record['decision_metadata'] = dict(decision.metadata)
    return _json_safe(record)

def build_logical_timeline_from_poses(poses: Sequence[Any], *, navigation_speed_mps: float=0.4, turn_speed_radps: float=0.8, scan_view_seconds: float | None=None) -> list[TimestampedPose]:
    pose_dicts = [_pose_to_dict(pose) for pose in poses]
    if not pose_dicts:
        return []
    timeline = [TimestampedPose(pose_dicts[0], 0.0)]
    timestamp = 0.0
    for prev, current in zip(pose_dicts[:-1], pose_dicts[1:]):
        distance = math.sqrt((float(current['x']) - float(prev['x'])) ** 2 + (float(current['y']) - float(prev['y'])) ** 2 + (float(current['z']) - float(prev['z'])) ** 2)
        yaw_dt = abs(_angle_delta(float(current['yaw']), float(prev['yaw']))) / float(turn_speed_radps)
        move_dt = distance / float(navigation_speed_mps)
        dt = max(move_dt, yaw_dt)
        if scan_view_seconds is not None:
            dt = max(dt, float(scan_view_seconds))
        timestamp += max(dt, 0.0)
        timeline.append(TimestampedPose(current, timestamp))
    return timeline

def resample_timeline_by_fps(timeline: Sequence[TimestampedPose], input_fps: float) -> list[TimestampedPose]:
    if not timeline:
        return []
    fps = float(input_fps)
    if fps <= 0.0:
        raise ValueError('input_fps must be positive.')
    end_time = float(timeline[-1].timestamp_s)
    if end_time <= 1e-09:
        return [TimestampedPose(dict(timeline[-1].pose), 0.0)]
    interval = 1.0 / fps
    samples: list[TimestampedPose] = []
    sample_time = 0.0
    index = 0
    while sample_time <= end_time + 1e-09:
        while index < len(timeline) - 2 and timeline[index + 1].timestamp_s < sample_time:
            index += 1
        left = timeline[index]
        right = timeline[min(index + 1, len(timeline) - 1)]
        samples.append(TimestampedPose(_interpolate_pose(left, right, sample_time), round(sample_time, 6)))
        sample_time += interval
    if samples[-1].timestamp_s < end_time - 1e-06:
        samples.append(TimestampedPose(dict(timeline[-1].pose), round(end_time, 6)))
    return samples

def _interpolate_pose(left: TimestampedPose, right: TimestampedPose, timestamp_s: float) -> dict[str, Any]:
    if right.timestamp_s <= left.timestamp_s + 1e-09:
        return dict(right.pose)
    ratio = (float(timestamp_s) - float(left.timestamp_s)) / (float(right.timestamp_s) - float(left.timestamp_s))
    ratio = min(max(ratio, 0.0), 1.0)
    pose = {}
    for key in ('x', 'y', 'z', 'pitch'):
        pose[key] = float(left.pose.get(key, 0.0)) + (float(right.pose.get(key, 0.0)) - float(left.pose.get(key, 0.0))) * ratio
    pose['yaw'] = _normalize_angle(float(left.pose.get('yaw', 0.0)) + _angle_delta(float(right.pose.get('yaw', 0.0)), float(left.pose.get('yaw', 0.0))) * ratio)
    return pose

def _pose_to_dict(pose: Any) -> dict[str, Any]:
    if isinstance(pose, Mapping):
        return {'x': float(pose.get('x', 0.0)), 'y': float(pose.get('y', 0.0)), 'z': float(pose.get('z', 1.55)), 'yaw': float(pose.get('yaw', 0.0)), 'pitch': float(pose.get('pitch', 0.0))}
    return {'x': float(getattr(pose, 'x', 0.0)), 'y': float(getattr(pose, 'y', 0.0)), 'z': float(getattr(pose, 'z', 1.55)), 'yaw': float(getattr(pose, 'yaw', 0.0)), 'pitch': float(getattr(pose, 'pitch', 0.0))}

def _normalize_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi

def _angle_delta(target: float, source: float) -> float:
    return _normalize_angle(float(target) - float(source))

def _strip_think_blocks(text: str) -> str:
    return re.sub('<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer('```(?:json)?\\s*(.*?)```', text, re.DOTALL | re.IGNORECASE):
        candidates.extend(_balanced_json_objects(match.group(1)))
    candidates.extend(_balanced_json_objects(_strip_think_blocks(text)))
    return sorted(dict.fromkeys(candidates), key=len, reverse=True)

def _balanced_json_objects(text: str) -> list[str]:
    out: list[str] = []
    starts: list[int] = []
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == '{':
            starts.append(index)
        elif char == '}' and starts:
            start = starts.pop()
            if not starts:
                out.append(text[start:index + 1])
    return out
__all__ = ['StreamController', 'StreamDecision', 'StreamFrameObservation', 'StreamObserver', 'StreamResponseParser', 'TimestampedPose', 'build_logical_timeline_from_poses', 'coerce_stream_decision', 'resample_timeline_by_fps', 'stream_frame_log_record']
