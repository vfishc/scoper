from __future__ import annotations
import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from scoper.harness.support.confirmation import LogicalEvidenceSampler, TransitionEvidence

@dataclass(frozen=True)
class EvidenceFrame:
    frame_id: str
    kind: str
    rgb: Any
    public_observation: Mapping[str, Any]
    high_level_step: int
    logical_time_s: float | None = None
    action_id: str | None = None

    def public_record(self) -> dict[str, Any]:
        return {'frame_id': self.frame_id, 'kind': self.kind, 'public_observation': copy.deepcopy(dict(self.public_observation)), 'high_level_step': self.high_level_step, 'logical_time_s': self.logical_time_s, 'action_id': self.action_id}

class EvidenceObserver(Protocol):

    def on_evidence_frame(self, frame: EvidenceFrame) -> None:
        ...

    def on_evidence_reset(self, episode_id: str) -> None:
        ...

class RuntimeEvidenceBridge:

    def __init__(self, *, stream_window_size: int=8) -> None:
        if isinstance(stream_window_size, bool) or stream_window_size < 1:
            raise ValueError('stream_window_size must be a positive integer')
        self.stream_window_size = stream_window_size
        self._observers: list[EvidenceObserver] = []
        self._stream: deque[EvidenceFrame] = deque(maxlen=stream_window_size)
        self._by_step: dict[int, dict[str, EvidenceFrame]] = {}
        self._episode_id = ''
        self._ordinal = 0
        self._model_call_count = 0

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    def add_observer(self, observer: EvidenceObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: EvidenceObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def reset(self, episode_id: str) -> None:
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('episode_id must be non-empty text')
        self._episode_id = episode_id
        self._ordinal = 0
        self._model_call_count = 0
        self._stream.clear()
        self._by_step.clear()
        for observer in tuple(self._observers):
            callback = getattr(observer, 'on_evidence_reset', None)
            if callable(callback):
                callback(episode_id)

    def note_model_call(self) -> None:
        self._model_call_count += 1

    def record(self, *, kind: str, rgb: Any, public_observation: Mapping[str, Any], high_level_step: int, frame_id: str | None=None, logical_time_s: float | None=None, action_id: str | None=None) -> EvidenceFrame:
        if not self._episode_id:
            raise RuntimeError('RuntimeEvidenceBridge.reset must be called first')
        if kind not in {'current', 'pre_action', 'post_action', 'immediate_post_action', 'confirmation', 'stream'}:
            raise ValueError(f'unsupported evidence kind: {kind}')
        if isinstance(high_level_step, bool) or high_level_step < 0:
            raise ValueError('high_level_step must be a non-negative integer')
        self._ordinal += 1
        resolved_id = frame_id or f'runtime-frame-{self._ordinal:06d}'
        frame = EvidenceFrame(frame_id=resolved_id, kind=kind, rgb=rgb, public_observation=copy.deepcopy(dict(public_observation)), high_level_step=high_level_step, logical_time_s=logical_time_s, action_id=action_id)
        if kind == 'stream':
            self._stream.append(frame)
        else:
            self._by_step.setdefault(high_level_step, {})[kind] = frame
        for observer in tuple(self._observers):
            callback = getattr(observer, 'on_evidence_frame', None)
            if callable(callback):
                callback(frame)
        return frame

    def step_frames(self, high_level_step: int) -> Mapping[str, EvidenceFrame]:
        return dict(self._by_step.get(high_level_step, {}))

    def stream_frames(self) -> tuple[EvidenceFrame, ...]:
        return tuple(self._stream)

    def public_snapshot(self, high_level_step: int) -> dict[str, Any]:
        return {'schema_version': 'runtime-evidence-snapshot.v1', 'episode_id': self._episode_id, 'high_level_step': high_level_step, 'frames': {kind: frame.public_record() for kind, frame in sorted(self._by_step.get(high_level_step, {}).items())}, 'stream_window': [frame.public_record() for frame in self._stream], 'model_call_count': self._model_call_count}

    def deliver_stream_to(self, brain: Any) -> None:
        callback = getattr(brain, 'on_stream_evidence', None)
        if not callable(callback):
            return
        for frame in self._stream:
            value = frame.public_record()
            try:
                callback(value, rgb=frame.rgb)
            except TypeError:
                callback(value, frame.rgb)

class RuntimeEvidenceCoordinator:

    def __init__(self, brain: Any, *, bridge: RuntimeEvidenceBridge | None=None, stream_parent_style: str='prefixed', stream_rate_hz: float=1.0) -> None:
        if stream_parent_style not in {'action', 'prefixed'}:
            raise ValueError('stream_parent_style must be action or prefixed')
        self.brain = brain
        self.bridge = bridge or RuntimeEvidenceBridge(stream_window_size=16)
        self.stream_parent_style = stream_parent_style
        self.logical_sampler = LogicalEvidenceSampler(rate_hz=stream_rate_hz)
        self._last_logical_time_s: float | None = None
        self._current_by_step: dict[int, EvidenceFrame] = {}
        self._stream_ordinal = 0

    @property
    def enabled(self) -> bool:
        return callable(getattr(self.brain, 'configure_method_event_sink', None))

    @property
    def visual_only(self) -> bool:
        definition = getattr(self.brain, 'harness_definition', None)
        return bool(getattr(definition, 'visual_only_state_enabled', False))

    def _project_public_observation(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return {} if self.visual_only else value

    def reset(self, episode_id: str) -> None:
        self.bridge.reset(episode_id)
        self.logical_sampler.reset()
        self._last_logical_time_s = None
        self._current_by_step.clear()
        self._stream_ordinal = 0

    def _register_parent(self, parent_id: str, frame_ids: tuple[str, ...]) -> None:
        sink = getattr(self.brain, 'event_sink', None)
        register = getattr(sink, 'register_parent', None)
        if callable(register) and frame_ids:
            register(parent_id, frame_ids)

    def prepare_decision(self, *, step: int, decision_id: str, observation: Mapping[str, Any]) -> EvidenceFrame:
        frame = self.bridge.record(kind='current', rgb=observation.get('head_rgb'), public_observation=self._project_public_observation(observation), high_level_step=step)
        self._current_by_step[step] = frame
        self._register_parent(decision_id, (frame.frame_id,))
        return frame

    def note_policy_call(self) -> None:
        self.bridge.note_model_call()

    def record_stream(self, *, step: int, rgb: Any, public_observation: Mapping[str, Any], logical_time_s: float | None=None, is_keyframe: bool=False, event_name: str | None=None, action_id: str | None=None, force: bool=False) -> EvidenceFrame | None:
        stream_buffer = getattr(self.brain, 'stream_buffer', None)
        if stream_buffer is None:
            stream_buffer = getattr(self.brain, '_stream_buffer', None)
        if not bool(getattr(stream_buffer, 'active', False)):
            return None
        if logical_time_s is not None:
            timestamp = float(logical_time_s)
            if self._last_logical_time_s is not None:
                timestamp = max(timestamp, self._last_logical_time_s)
            sampling = self.logical_sampler.consider(timestamp, force=force or is_keyframe)
            self._last_logical_time_s = timestamp
            if not sampling.emit:
                return None
        self._stream_ordinal += 1
        ordinal = self._stream_ordinal
        frame = self.bridge.record(kind='stream', rgb=rgb, public_observation=self._project_public_observation(public_observation), high_level_step=step, logical_time_s=timestamp if logical_time_s is not None else None, action_id=action_id)
        callback = getattr(self.brain, 'on_stream_evidence', None)
        if callable(callback):
            callback({'frame_id': frame.frame_id, 'ordinal': ordinal, 'frame_kind': 'event_keyframe' if is_keyframe and (not self.visual_only) else 'stream_sample', 'event_name_or_null': event_name if is_keyframe and (not self.visual_only) else None, 'logical_time_s': timestamp if logical_time_s is not None else 0.0}, rgb=rgb)
        return frame

    def record_final_stream(self, *, step: int, rgb: Any, public_observation: Mapping[str, Any], action_id: str) -> EvidenceFrame | None:
        latest = self.bridge.stream_frames()
        logical_time = next((frame.logical_time_s for frame in reversed(latest) if frame.high_level_step == step and frame.logical_time_s is not None), 0.0)
        return self.record_stream(step=step, rgb=rgb, public_observation=public_observation, logical_time_s=float(logical_time), is_keyframe=True, event_name='action_end', action_id=action_id, force=True)

    def prepare_action_result(self, *, step: int, action_id: str, post_observation: Mapping[str, Any] | None=None, immediate_post_observation: Mapping[str, Any] | None=None, confirmation_observation: Mapping[str, Any] | None=None) -> tuple[EvidenceFrame, ...]:
        pre = self._current_by_step.get(step)
        if pre is None:
            pre = self.bridge.record(kind='pre_action', rgb=None, public_observation={}, high_level_step=step, action_id=action_id)
        else:
            pre = self.bridge.record(kind='pre_action', rgb=pre.rgb, public_observation=self._project_public_observation(pre.public_observation), high_level_step=step, logical_time_s=pre.logical_time_s, action_id=action_id)
        if immediate_post_observation is not None or confirmation_observation is not None:
            if immediate_post_observation is None or confirmation_observation is None:
                raise ValueError('v12 action evidence requires immediate and confirmation observations')
            immediate = self.bridge.record(kind='immediate_post_action', rgb=immediate_post_observation.get('head_rgb'), public_observation=self._project_public_observation(immediate_post_observation), high_level_step=step, action_id=action_id)
            confirmation = self.bridge.record(kind='confirmation', rgb=confirmation_observation.get('head_rgb'), public_observation=self._project_public_observation(confirmation_observation), high_level_step=step, action_id=action_id)
            transition = TransitionEvidence(pre.frame_id, immediate.frame_id, confirmation.frame_id, pre.rgb, immediate.rgb, confirmation.rgb)
            action_frames: tuple[EvidenceFrame, ...] = (pre, immediate, confirmation)
            action_frame_ids = transition.frame_ids
        else:
            if post_observation is None:
                raise ValueError('post_observation is required for legacy action evidence')
            post = self.bridge.record(kind='post_action', rgb=post_observation.get('head_rgb'), public_observation=self._project_public_observation(post_observation), high_level_step=step, action_id=action_id)
            action_frames = (pre, post)
            action_frame_ids = (pre.frame_id, post.frame_id)
        active_ids = tuple((str(item) for item in getattr(self.brain, 'active_stream_evidence_frame_ids', ())))
        if not active_ids:
            stream_buffer = getattr(self.brain, 'stream_buffer', None)
            if stream_buffer is None:
                stream_buffer = getattr(self.brain, '_stream_buffer', None)
            active_ids = tuple((str(getattr(item, 'frame_id')) for item in getattr(stream_buffer, 'refs', ())))
        if len(action_frame_ids) == 3 or getattr(getattr(self.brain, 'harness_definition', None), 'version', None) == '20':
            self._register_parent(action_id, action_frame_ids)
            if active_ids:
                self._register_parent(f'stream:{action_id}', active_ids)
        elif active_ids and self.stream_parent_style == 'action':
            self._register_parent(action_id, active_ids)
        else:
            self._register_parent(action_id, action_frame_ids)
            if active_ids:
                self._register_parent(f'stream:{action_id}', active_ids)
        return action_frames

    def artifacts(self) -> Mapping[str, Any]:
        stream_buffer = getattr(self.brain, 'stream_buffer', None)
        if stream_buffer is None:
            stream_buffer = getattr(self.brain, '_stream_buffer', None)
        artifacts = {'schema_version': 'runtime-evidence-bridge-report.v1', 'episode_id': self.bridge.episode_id, 'model_call_count': self.bridge.model_call_count, 'stream_frame_count': len(self.bridge.stream_frames()), 'capture_persistence': False, 'model_facing_window': dict(getattr(stream_buffer, 'last_window_metrics', {}) or {})}
        observe_metrics = getattr(self.brain, 'passive_observe_metrics', None)
        if isinstance(observe_metrics, Mapping):
            artifacts['passive_observe'] = copy.deepcopy(dict(observe_metrics))
        return artifacts
__all__ = ['EvidenceFrame', 'EvidenceObserver', 'RuntimeEvidenceBridge', 'RuntimeEvidenceCoordinator']
