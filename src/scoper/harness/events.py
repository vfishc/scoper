from __future__ import annotations
from dataclasses import dataclass
import copy
from typing import Any, Mapping, Protocol, Sequence
from scoper.harness.contracts import MethodContractError
from scoper.harness.validation import non_empty_string, public_json, string_sequence

@dataclass(frozen=True)
class VerificationCallContext:
    verification_id: str
    model_call_id: str
    parent_action_id: str
    evidence_frame_ids: tuple[str, ...]

@dataclass(frozen=True)
class MemoryProposalCallContext:
    proposal_id: str
    model_call_id: str
    mode: str
    parent_record_id: str
    evidence_frame_ids: tuple[str, ...]

class MethodEventSink(Protocol):

    def evidence_frame_ids(self, parent_record_id: str) -> tuple[str, ...]:
        ...

    def begin_verification(self, parent_action_id: str) -> VerificationCallContext:
        ...

    def begin_reverification(self, parent_action_id: str, evidence_frame_ids: Sequence[str]) -> VerificationCallContext:
        ...

    def finish_verification(self, context: VerificationCallContext, output_or_error: Mapping[str, Any]) -> str:
        ...

    def emit_memory_update(self, parent_action_id: str, payload: Mapping[str, Any]) -> str:
        ...

    def begin_memory_proposal(self, mode: str, parent_record_id: str) -> MemoryProposalCallContext:
        ...

    def finish_memory_proposal(self, context: MemoryProposalCallContext, proposal_or_error: Mapping[str, Any], accepted_mutation: Mapping[str, Any] | None) -> tuple[str, ...]:
        ...

class InMemoryMethodEventSink:

    def __init__(self, parent_evidence: Mapping[str, Sequence[str]] | None=None, *, model_output_provider: Any | None=None, capture_raw_output: bool=False) -> None:
        self._parent_evidence: dict[str, tuple[str, ...]] = {}
        self._model_call_ordinal = 0
        self._verification_ordinal = 0
        self._proposal_ordinal = 0
        self._event_ordinal = 0
        self._active_verifications: dict[str, VerificationCallContext] = {}
        self._active_proposals: dict[str, MemoryProposalCallContext] = {}
        self._events: list[dict[str, Any]] = []
        self._model_output_provider = model_output_provider
        self.capture_raw_output = bool(capture_raw_output)
        self.runtime_mode = 'structured_heads_v1'
        self.model_identity: Mapping[str, Any] = {}
        for parent_id, frames in (parent_evidence or {}).items():
            self.register_parent(parent_id, frames)

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._events)

    def configure_runtime_identity(self, *, runtime_mode: str, model_identity: Mapping[str, Any]) -> None:
        if self._active_verifications or self._active_proposals:
            raise MethodContractError('method_lifecycle_invalid', 'cannot change Method runtime identity during a model call')
        self.runtime_mode = non_empty_string(runtime_mode, pointer='/method_runtime_mode')
        self.model_identity = copy.deepcopy(dict(model_identity))

    def register_parent(self, parent_record_id: str, evidence_frame_ids: Sequence[str]) -> None:
        parent = non_empty_string(parent_record_id, pointer='/parent_record_id')
        frames = string_sequence(list(evidence_frame_ids), pointer='/evidence_frame_ids', non_empty=True)
        existing = self._parent_evidence.get(parent)
        if existing is not None and existing != frames:
            raise MethodContractError('method_event_correlation_invalid', 'parent evidence was registered inconsistently')
        self._parent_evidence[parent] = frames

    def _evidence(self, parent_record_id: str) -> tuple[str, ...]:
        parent = non_empty_string(parent_record_id, pointer='/parent_record_id')
        try:
            return self._parent_evidence[parent]
        except KeyError as exc:
            raise MethodContractError('method_event_correlation_invalid', 'parent record has no registered evidence') from exc

    def evidence_frame_ids(self, parent_record_id: str) -> tuple[str, ...]:
        return self._evidence(parent_record_id)

    def _model_call_id(self) -> str:
        self._model_call_ordinal += 1
        return 'mcall-{:08d}'.format(self._model_call_ordinal)

    def _event_id(self) -> str:
        self._event_ordinal += 1
        return 'mevt-{:08d}'.format(self._event_ordinal)

    def _append_event(self, *, event_type: str, parent_record_id: str, correlation: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
        event_id = self._event_id()
        event = {'schema_version': 'method-event.v1', 'event_id': event_id, 'event_type': event_type, 'parent_record_id': parent_record_id, 'correlation': dict(correlation), 'payload': dict(payload)}
        if self.runtime_mode != 'structured_heads_v1':
            event['payload'] = {**event['payload'], 'method_runtime_mode': self.runtime_mode, 'hardqa_model_identity': copy.deepcopy(dict(self.model_identity))}
        public_json(event, pointer='/method_event', include_future_action=True)
        self._store_event(event)
        return event_id

    def _store_event(self, event: Mapping[str, Any]) -> None:
        self._events.append(dict(event))

    def record_component_model_call(self, context: Any, *, component: str, request: Mapping[str, Any], response_or_error: Mapping[str, Any], image_payloads: Mapping[str, bytes], latency_seconds: float, error_code: str | None) -> None:
        del image_payloads
        if not self.capture_raw_output:
            return
        raw = None
        if callable(self._model_output_provider):
            value = self._model_output_provider(component)
            if isinstance(value, Mapping):
                raw = dict(value)
        self._events.append({'schema_version': 'method-model-call-diagnostic.v1', 'event_type': 'model_call', 'parent_record_id': getattr(context, 'parent_record_id', None), 'correlation': {'model_call_id': getattr(context, 'model_call_id', None), 'component': component, 'evidence_frame_ids': list(getattr(context, 'evidence_frame_ids', ()) or ())}, 'payload': {'request': copy.deepcopy(dict(request)), 'response': copy.deepcopy(dict(response_or_error)), 'error_code': error_code, 'latency_seconds': float(latency_seconds), 'raw_output': raw}})

    def begin_verification(self, parent_action_id: str) -> VerificationCallContext:
        frames = self._evidence(parent_action_id)
        if len(frames) not in {2, 3}:
            raise MethodContractError('required_frame_missing', 'verification correlation requires legacy pre/post or v12 three-frame evidence')
        self._verification_ordinal += 1
        context = VerificationCallContext('verify-{:08d}'.format(self._verification_ordinal), self._model_call_id(), parent_action_id, frames)
        self._active_verifications[context.verification_id] = context
        return context

    def begin_reverification(self, parent_action_id: str, evidence_frame_ids: Sequence[str]) -> VerificationCallContext:
        parent = non_empty_string(parent_action_id, pointer='/parent_action_id')
        frames = string_sequence(list(evidence_frame_ids), pointer='/evidence_frame_ids', non_empty=True)
        if len(frames) != 3 or len(set(frames)) != 3:
            raise MethodContractError('required_frame_missing', 'reverification requires three distinct ordered evidence frames')
        self._verification_ordinal += 1
        context = VerificationCallContext('verify-{:08d}'.format(self._verification_ordinal), self._model_call_id(), parent, frames)
        self._active_verifications[context.verification_id] = context
        return context

    def finish_verification(self, context: VerificationCallContext, output_or_error: Mapping[str, Any]) -> str:
        active = self._active_verifications.get(context.verification_id)
        if active != context:
            raise MethodContractError('method_event_correlation_invalid', 'verification context is not active')
        if not isinstance(output_or_error, Mapping):
            raise MethodContractError('method_schema_invalid', 'verification result must be an object')
        response_call_id = output_or_error.get('verifier_call_id')
        if response_call_id is not None and response_call_id != context.model_call_id:
            raise MethodContractError('method_event_correlation_invalid', 'verifier response call ID differs')
        if 'error_code' not in output_or_error and response_call_id is None:
            raise MethodContractError('method_event_correlation_invalid', 'verifier response call ID is missing')
        response_frames = output_or_error.get('evidence_frame_ids')
        if response_frames is not None:
            checked_frames = string_sequence(response_frames, pointer='/verification_output/evidence_frame_ids', non_empty=True)
            if checked_frames != context.evidence_frame_ids:
                raise MethodContractError('method_event_correlation_invalid', 'verifier response evidence differs')
        if 'error_code' not in output_or_error and response_frames is None:
            raise MethodContractError('method_event_correlation_invalid', 'verifier response evidence is missing')
        public_json(output_or_error, pointer='/verification_output', include_future_action=True)
        del self._active_verifications[context.verification_id]
        return self._append_event(event_type='verification', parent_record_id=context.parent_action_id, correlation={'verification_id': context.verification_id, 'model_call_id': context.model_call_id, 'evidence_frame_ids': list(context.evidence_frame_ids)}, payload=output_or_error)

    def emit_memory_update(self, parent_action_id: str, payload: Mapping[str, Any]) -> str:
        self._evidence(parent_action_id)
        return self._append_event(event_type='memory_update', parent_record_id=parent_action_id, correlation={'evidence_frame_ids': list(self._evidence(parent_action_id))}, payload=payload)

    def begin_memory_proposal(self, mode: str, parent_record_id: str) -> MemoryProposalCallContext:
        if mode not in {'update', 'reconcile', 'route', 'observe'}:
            raise MethodContractError('method_schema_invalid', 'memory proposal mode is invalid')
        frames = self._evidence(parent_record_id)
        self._proposal_ordinal += 1
        context = MemoryProposalCallContext('mprop-{:08d}'.format(self._proposal_ordinal), self._model_call_id(), mode, parent_record_id, frames)
        self._active_proposals[context.proposal_id] = context
        return context

    def finish_memory_proposal(self, context: MemoryProposalCallContext, proposal_or_error: Mapping[str, Any], accepted_mutation: Mapping[str, Any] | None) -> tuple[str, ...]:
        active = self._active_proposals.get(context.proposal_id)
        if active != context:
            raise MethodContractError('method_event_correlation_invalid', 'memory proposal context is not active')
        if not isinstance(proposal_or_error, Mapping):
            raise MethodContractError('method_schema_invalid', 'memory proposal result must be an object')
        response_proposal_id = proposal_or_error.get('proposal_id')
        if response_proposal_id is not None and response_proposal_id != context.proposal_id:
            raise MethodContractError('method_event_correlation_invalid', 'memory proposal response ID differs')
        if 'error_code' not in proposal_or_error and response_proposal_id is None:
            raise MethodContractError('method_event_correlation_invalid', 'memory proposal response ID is missing')
        response_mode = proposal_or_error.get('mode')
        if response_mode is not None and response_mode != context.mode:
            raise MethodContractError('method_event_correlation_invalid', 'memory proposal mode differs')
        if 'error_code' not in proposal_or_error and response_mode is None:
            raise MethodContractError('method_event_correlation_invalid', 'memory proposal mode is missing')
        response_frames = proposal_or_error.get('evidence_frame_ids')
        if response_frames is not None:
            checked_frames = string_sequence(response_frames, pointer='/memory_proposal/evidence_frame_ids', non_empty=context.mode != 'route')
            if checked_frames != context.evidence_frame_ids:
                raise MethodContractError('method_event_correlation_invalid', 'memory proposal evidence differs')
        if 'error_code' not in proposal_or_error and response_frames is None:
            raise MethodContractError('method_event_correlation_invalid', 'memory proposal evidence is missing')
        if 'error_code' in proposal_or_error and accepted_mutation is not None:
            raise MethodContractError('method_event_correlation_invalid', 'an errored proposal cannot commit a mutation')
        public_json(proposal_or_error, pointer='/memory_proposal', include_future_action=True)
        if accepted_mutation is not None:
            public_json(accepted_mutation, pointer='/accepted_mutation', include_future_action=True)
        del self._active_proposals[context.proposal_id]
        proposal_event_id = self._append_event(event_type='memory_proposal', parent_record_id=context.parent_record_id, correlation={'proposal_id': context.proposal_id, 'model_call_id': context.model_call_id, 'mode': context.mode, 'evidence_frame_ids': list(context.evidence_frame_ids)}, payload=proposal_or_error)
        event_ids = [proposal_event_id]
        if accepted_mutation is not None:
            event_ids.append(self._append_event(event_type='memory_update', parent_record_id=context.parent_record_id, correlation={'proposal_event_id': proposal_event_id, 'proposal_id': context.proposal_id, 'evidence_frame_ids': list(context.evidence_frame_ids)}, payload=accepted_mutation))
        return tuple(event_ids)

    def assert_closed(self) -> None:
        if self._active_verifications or self._active_proposals:
            raise MethodContractError('method_event_correlation_invalid', 'one or more Method model calls are unclosed')

    def reset_episode(self) -> None:
        self._parent_evidence.clear()
        self._active_verifications.clear()
        self._active_proposals.clear()

class NullMethodEventSink(InMemoryMethodEventSink):

    def _store_event(self, event: Mapping[str, Any]) -> None:
        return None
__all__ = ['InMemoryMethodEventSink', 'MemoryProposalCallContext', 'MethodEventSink', 'NullMethodEventSink', 'VerificationCallContext']
