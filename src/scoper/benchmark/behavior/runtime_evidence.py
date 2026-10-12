from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.common import RuntimeEvidenceCoordinator
from scoper.benchmark.behavior.export.ids import action_id

class BehaviorStreamEvidenceObserver:

    def __init__(self, coordinator: RuntimeEvidenceCoordinator) -> None:
        self.coordinator = coordinator
        self._ordinal = 0

    def before_stream_decision(self, frame: Any) -> Mapping[str, Any]:
        self._ordinal += 1
        step = int(getattr(frame, 'high_level_step', 0) or 0)
        recorded = self.coordinator.record_stream(step=step, rgb=getattr(frame, 'head_rgb', None), public_observation={} if self.coordinator.visual_only else {'visible_object_catalog': list(getattr(frame, 'visible_object_catalog', ()) or ()), 'prompt_metadata': dict(getattr(frame, 'prompt_metadata', {}) or {})}, logical_time_s=float(getattr(frame, 'timestamp_s', 0.0) or 0.0), is_keyframe=bool(getattr(frame, 'is_keyframe', False)), event_name=str(getattr(frame, 'keyframe_reason', '') or getattr(frame, 'source', '')) if bool(getattr(frame, 'is_keyframe', False)) else None, action_id=action_id(step) if step > 0 else None)
        frame_id = recorded.frame_id if recorded is not None else f'runtime-ignored-{self._ordinal:06d}'
        return {'stream_id': str(getattr(frame, 'stream_id', 'runtime-stream')), 'stream_obs_id': f'runtime-sobs-{self._ordinal:06d}', 'frame_id': frame_id, 'episode_id': self.coordinator.bridge.episode_id, 'action_id': action_id(step) if step > 0 else None, 'model_call_id': None}

    def after_stream_decision(self, frame: Any, decision: Any, *, vlm_called: bool, latency_seconds: float, correlation: Mapping[str, Any]) -> None:
        del frame, decision, vlm_called, latency_seconds, correlation
__all__ = ['BehaviorStreamEvidenceObserver', 'RuntimeEvidenceCoordinator']
