from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from scoper.harness.confirmation import LogicalEvidenceSampler
from scoper.harness.contracts import MethodContractError
from scoper.harness.validation import exact_mapping, finite_number, integer, non_empty_string, optional_string, sha256, string_sequence
STREAM_REF_FIELDS = frozenset({'frame_id', 'ordinal', 'frame_kind', 'event_name_or_null'})
STREAM_REF_V2_FIELDS = STREAM_REF_FIELDS | {'logical_time_s', 'media_sha256'}

@dataclass(frozen=True)
class StreamFrameRef:
    frame_id: str
    ordinal: int
    frame_kind: str
    event_name_or_null: str | None
    logical_time_s: float | None = None
    media_sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'StreamFrameRef':
        fields = STREAM_REF_V2_FIELDS if set(value) == STREAM_REF_V2_FIELDS else STREAM_REF_FIELDS
        item = exact_mapping(value, fields, pointer='/stream_frame_ref')
        frame_id = non_empty_string(item['frame_id'], pointer='/stream_frame_ref/frame_id')
        ordinal = integer(item['ordinal'], pointer='/stream_frame_ref/ordinal', minimum=1)
        frame_kind = item['frame_kind']
        if frame_kind not in {'stream_sample', 'event_keyframe'}:
            raise MethodContractError('method_schema_invalid', 'stream frame kind is invalid')
        event_name = optional_string(item['event_name_or_null'], pointer='/stream_frame_ref/event_name_or_null')
        if (frame_kind == 'event_keyframe') != (event_name is not None):
            raise MethodContractError('method_schema_invalid', 'event keyframe and event_name must agree')
        logical_time = finite_number(item['logical_time_s'], pointer='/stream_frame_ref/logical_time_s', minimum=0) if fields == STREAM_REF_V2_FIELDS else None
        media_hash = sha256(item['media_sha256'], pointer='/stream_frame_ref/media_sha256') if fields == STREAM_REF_V2_FIELDS else None
        return cls(frame_id, ordinal, frame_kind, event_name, logical_time, media_hash)

    def to_mapping(self) -> dict[str, Any]:
        result = {'frame_id': self.frame_id, 'ordinal': self.ordinal, 'frame_kind': self.frame_kind, 'event_name_or_null': self.event_name_or_null}
        if self.logical_time_s is not None and self.media_sha256 is not None:
            result.update({'logical_time_s': self.logical_time_s, 'media_sha256': self.media_sha256})
        return result

def select_model_facing_stream_window(refs: Sequence[StreamFrameRef], *, max_refs: int=16) -> tuple[StreamFrameRef, ...]:
    limit = integer(max_refs, pointer='/stream_window/max_refs', minimum=1)
    ordered = tuple(refs)
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return (ordered[-1],)
    indices = tuple((index * (len(ordered) - 1) // (limit - 1) for index in range(limit)))
    return tuple((ordered[index] for index in indices))

@dataclass(frozen=True)
class ActionEndReconcileRequest:
    episode_id: str
    action_id: str
    reconcile_proposal_id: str
    active_tool: str
    base_revision: int
    ordered_public_frame_ids: tuple[str, ...]
    event_keyframe_ids: tuple[str, ...]
    proposal_mode: str = 'reconcile'

    def to_mapping(self) -> dict[str, Any]:
        result = {'schema_version': 'method-passive-observe-request.v1' if self.proposal_mode == 'observe' else 'method-passive-reconcile-request.v1', 'episode_id': self.episode_id, 'action_id': self.action_id, 'reconcile_proposal_id': self.reconcile_proposal_id, 'active_tool': self.active_tool, 'base_revision': self.base_revision, 'ordered_public_frame_ids': list(self.ordered_public_frame_ids), 'event_keyframe_ids': list(self.event_keyframe_ids)}
        if self.proposal_mode == 'observe':
            result['proposal_mode'] = 'observe'
        return result

class StreamEvidenceBuffer:

    def __init__(self, *, stream_enabled_tools: Sequence[str]=('look_for', 'navigate_to', 'navigate_to_room', 'scan'), max_refs: int=16, selection_mode: str='legacy_tail') -> None:
        self._enabled_tools = string_sequence(list(stream_enabled_tools), pointer='/stream_enabled_tools', non_empty=True, sorted_utf8=True)
        self._max_refs = integer(max_refs, pointer='/stream_buffer/max_refs', minimum=1)
        if selection_mode not in {'legacy_tail', 'uniform_v2'}:
            raise MethodContractError('method_schema_invalid', 'stream selection mode is invalid')
        self._selection_mode = selection_mode
        self._episode_id: str | None = None
        self._action_id: str | None = None
        self._active_tool: str | None = None
        self._base_revision: int | None = None
        self._refs: list[StreamFrameRef] = []
        self._pending_request: ActionEndReconcileRequest | None = None
        self._pending_claimed = False
        self._last_window_metrics: dict[str, Any] = {'candidate_frame_count': 0, 'selected_frame_count': 0, 'logical_time_span_seconds': 0.0, 'consecutive_duplicate_count': 0}
        self._consecutive_duplicate_count = 0
        self._candidate_frame_count = 0
        self._last_candidate_media_sha256: str | None = None
        self._logical_sampler = LogicalEvidenceSampler(rate_hz=1.0)

    @property
    def active(self) -> bool:
        return self._action_id is not None

    @property
    def refs(self) -> tuple[StreamFrameRef, ...]:
        if self._selection_mode == 'uniform_v2':
            return select_model_facing_stream_window(self._refs, max_refs=self._max_refs)
        return tuple(self._refs)

    @property
    def last_window_metrics(self) -> Mapping[str, Any]:
        return dict(self._last_window_metrics)

    def reset_episode(self) -> None:
        self._episode_id = None
        self._action_id = None
        self._active_tool = None
        self._base_revision = None
        self._refs = []
        self._pending_request = None
        self._pending_claimed = False
        self._consecutive_duplicate_count = 0
        self._candidate_frame_count = 0
        self._last_candidate_media_sha256 = None
        self._logical_sampler.reset()

    def begin_action(self, *, episode_id: str, action_id: str, active_tool: str, base_revision: int) -> None:
        if self.active or self._pending_request is not None:
            raise MethodContractError('method_stream_lifecycle_invalid', 'previous stream action is not reconciled')
        tool = non_empty_string(active_tool, pointer='/active_tool')
        if tool not in self._enabled_tools:
            raise MethodContractError('method_stream_tool_forbidden', 'tool is not enabled for passive stream evidence')
        self._episode_id = non_empty_string(episode_id, pointer='/episode_id')
        self._action_id = non_empty_string(action_id, pointer='/action_id')
        self._active_tool = tool
        self._base_revision = integer(base_revision, pointer='/base_revision')
        self._refs = []
        self._consecutive_duplicate_count = 0
        self._candidate_frame_count = 0
        self._last_candidate_media_sha256 = None
        self._logical_sampler.reset()

    def append(self, value: Mapping[str, Any]) -> None:
        if not self.active:
            raise MethodContractError('method_stream_lifecycle_invalid', 'no active stream action')
        ref = StreamFrameRef.from_mapping(value)
        if self._refs and ref.ordinal <= self._refs[-1].ordinal:
            raise MethodContractError('method_stream_order_invalid', 'stream evidence ordinals must strictly increase')
        if any((item.frame_id == ref.frame_id for item in self._refs)):
            raise MethodContractError('method_stream_order_invalid', 'stream frame ID is duplicated')
        if self._selection_mode == 'uniform_v2':
            self._candidate_frame_count += 1
            if ref.logical_time_s is None or ref.media_sha256 is None:
                raise MethodContractError('method_schema_invalid', 'uniform stream evidence requires logical time and media hash')
            duplicate = ref.media_sha256 == self._last_candidate_media_sha256 or any((item.media_sha256 == ref.media_sha256 for item in self._refs))
            self._last_candidate_media_sha256 = ref.media_sha256
            if duplicate:
                self._consecutive_duplicate_count += 1
                return
            sampling = self._logical_sampler.consider(ref.logical_time_s, force=ref.frame_kind == 'event_keyframe' and ref.event_name_or_null == 'action_end')
            if not sampling.emit:
                return
        if self._selection_mode == 'legacy_tail' and len(self._refs) >= self._max_refs:
            evict_index = next((index for index, item in enumerate(self._refs) if item.frame_kind == 'stream_sample'), None)
            if evict_index is None:
                raise MethodContractError('method_stream_buffer_capacity_exceeded', 'event keyframes exceed bounded buffer')
            del self._refs[evict_index]
        self._refs.append(ref)

    def finish_action(self, *, action_id: str, reconcile_proposal_id: str, proposal_mode: str='reconcile') -> ActionEndReconcileRequest:
        if not self.active or action_id != self._action_id:
            raise MethodContractError('method_stream_lifecycle_invalid', 'stream action identity differs')
        if proposal_mode not in {'reconcile', 'observe'}:
            raise MethodContractError('method_schema_invalid', 'stream proposal mode is invalid')
        selected = self.refs
        request = ActionEndReconcileRequest(self._episode_id, self._action_id, non_empty_string(reconcile_proposal_id, pointer='/reconcile_proposal_id'), self._active_tool, self._base_revision, tuple((item.frame_id for item in selected)), tuple((item.frame_id for item in selected if item.frame_kind == 'event_keyframe')), proposal_mode)
        times = [item.logical_time_s for item in selected if item.logical_time_s is not None]
        self._last_window_metrics = {'candidate_frame_count': self._candidate_frame_count if self._selection_mode == 'uniform_v2' else len(self._refs), 'selected_frame_count': len(selected), 'logical_time_span_seconds': max(times) - min(times) if len(times) >= 2 else 0.0, 'consecutive_duplicate_count': self._consecutive_duplicate_count}
        self._pending_request = request
        self._pending_claimed = False
        self._episode_id = None
        self._action_id = None
        self._active_tool = None
        self._base_revision = None
        self._refs = []
        return request

    def take_action_end_reconcile(self) -> ActionEndReconcileRequest:
        if self._pending_request is None:
            raise MethodContractError('method_stream_lifecycle_invalid', 'no action-end reconcile request is pending')
        if self._pending_claimed:
            raise MethodContractError('method_stream_lifecycle_invalid', 'action-end reconcile request is already claimed')
        self._pending_claimed = True
        return self._pending_request

    def complete_action_end_reconcile(self, request: ActionEndReconcileRequest, decision: Any) -> None:
        from scoper.harness.controller import ControllerDecision
        if self._pending_request is None or not self._pending_claimed or request != self._pending_request:
            raise MethodContractError('method_stream_lifecycle_invalid', 'action-end reconcile acknowledgement differs')
        if not isinstance(decision, ControllerDecision):
            raise MethodContractError('method_stream_lifecycle_invalid', 'reconcile acknowledgement is not a Controller decision')
        expected_revision = request.base_revision + int(decision.accepted_by_harness)
        if decision.memory.episode_id != request.episode_id or decision.proposal_id != request.reconcile_proposal_id or decision.proposal_mode_or_null != request.proposal_mode or (decision.parent_action_id_or_null != request.action_id) or (decision.base_revision_or_null != request.base_revision) or (decision.new_revision != expected_revision) or (decision.reason == 'route_valid_no_commit'):
            raise MethodContractError('method_stream_lifecycle_invalid', 'reconcile decision revision/mode differs')
        self._pending_request = None
        self._pending_claimed = False

    def assert_ready_for_policy_decision(self) -> None:
        if self.active or self._pending_request is not None:
            raise MethodContractError('method_stream_reconcile_required', 'action-end reconcile must complete before the next policy decision')
__all__ = ['ActionEndReconcileRequest', 'StreamEvidenceBuffer', 'StreamFrameRef', 'select_model_facing_stream_window']
