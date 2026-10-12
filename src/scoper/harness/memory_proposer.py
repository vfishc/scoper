from __future__ import annotations
from typing import Any, Mapping, Protocol, Sequence
from scoper.harness.validation import exact_mapping, non_empty_string, ordered_image_roles, public_json, sha256, string_sequence
from scoper.harness.contracts import MethodContractError
from scoper.harness.evidence_memory import EvidenceMemory
from scoper.harness.vocabulary import OBSERVE_MAX_RETAINED_FACTS, OBSERVE_REQUEST_SCHEMA_VERSION
MEMORY_PROPOSER_REQUEST_FIELDS = frozenset({'schema_version', 'proposal_id', 'mode', 'base_revision', 'evidence_frame_ids', 'images', 'previous_evidence_memory', 'public_observation_or_null', 'observed_action_or_null', 'public_transition_or_null', 'public_entity_candidates', 'candidate_memory_ids'})
MEMORY_PROPOSER_REQUEST_V3_FIELDS = frozenset({'schema_version', 'proposal_id', 'mode', 'base_revision', 'evidence_frame_ids', 'images', 'previous_evidence_memory', 'public_instruction', 'public_task_context', 'observation_tool_call_or_null'})
IMAGE_METADATA_FIELDS = frozenset({'media_sha256', 'mime_type', 'width', 'height'})

class MemoryProposerClient(Protocol):
    checkpoint_sha256: str

    def propose(self, request: Mapping[str, Any], *, image_payloads: Mapping[str, bytes]) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...

def _image_metadata(value: Any, *, pointer: str) -> dict[str, Any]:
    item = exact_mapping(value, IMAGE_METADATA_FIELDS, pointer=pointer)
    mime_type = item['mime_type']
    if mime_type not in {'image/png', 'image/jpeg'}:
        raise MethodContractError('method_media_invalid', 'memory image MIME type is invalid', pointer=pointer)
    output = {'media_sha256': sha256(item['media_sha256'], pointer=f'{pointer}/media_sha256'), 'mime_type': mime_type, 'width': item['width'], 'height': item['height']}
    for field in ('width', 'height'):
        if isinstance(output[field], bool) or not isinstance(output[field], int) or output[field] < 1:
            raise MethodContractError('method_media_invalid', 'memory image shape is invalid', pointer=f'{pointer}/{field}')
    return output

def build_memory_proposer_request(*, proposal_id: str, mode: str, base_revision: int, evidence_frame_ids: Sequence[str], images: Mapping[str, Mapping[str, Any]], previous_evidence_memory: EvidenceMemory, public_observation_or_null: Mapping[str, Any] | None, observed_action_or_null: Mapping[str, Any] | None, public_transition_or_null: Mapping[str, Any] | None, public_entity_refs: Sequence[str], candidate_memory_ids: Sequence[str], public_instruction: str | None=None, public_task_context: Mapping[str, Any] | None=None, observation_tool_call_or_null: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if mode not in {'update', 'reconcile', 'route', 'observe'}:
        raise MethodContractError('method_schema_invalid', 'memory proposal mode is invalid')
    if isinstance(base_revision, bool) or not isinstance(base_revision, int) or base_revision < 0:
        raise MethodContractError('method_schema_invalid', 'base revision is invalid')
    frames = string_sequence(list(evidence_frame_ids), pointer='/memory_proposer_request/evidence_frame_ids', non_empty=mode != 'route')
    image_rows = {non_empty_string(name, pointer='/memory_proposer_request/images'): _image_metadata(images[name], pointer=f'/memory_proposer_request/images/{name}') for name in ordered_image_roles(images)}
    if mode != 'route' and (not image_rows):
        raise MethodContractError('required_frame_missing', 'memory proposal images are missing')
    transition_roles = {'pre_rgb', 'immediate_post_rgb', 'confirmation_rgb'}
    is_v12_update = mode == 'update' and set(image_rows) == transition_roles
    if is_v12_update and len(frames) != 3:
        raise MethodContractError('required_frame_missing', 'v12 memory update requires three ordered evidence frames')
    if mode == 'update' and (not is_v12_update) and (set(image_rows) != {'pre_rgb', 'post_rgb'}):
        raise MethodContractError('method_schema_invalid', 'memory update image roles differ from a supported contract')
    if mode == 'observe' and len(image_rows) > OBSERVE_MAX_RETAINED_FACTS:
        raise MethodContractError('method_schema_invalid', 'observe accepts at most sixteen ordered images')
    refs = string_sequence(list(public_entity_refs), pointer='/memory_proposer_request/public_entity_refs', sorted_utf8=True)
    candidates = string_sequence(list(candidate_memory_ids), pointer='/memory_proposer_request/candidate_memory_ids', sorted_utf8=True)
    request = {'schema_version': OBSERVE_REQUEST_SCHEMA_VERSION if mode == 'observe' else 'memory-proposer-request.v2' if is_v12_update else 'memory-proposer-request.v1', 'proposal_id': non_empty_string(proposal_id, pointer='/memory_proposer_request/proposal_id'), 'mode': mode, 'base_revision': base_revision, 'evidence_frame_ids': list(frames), 'images': image_rows, 'previous_evidence_memory': previous_evidence_memory.validated().to_mapping(), 'public_observation_or_null': None if public_observation_or_null is None else dict(public_observation_or_null), 'observed_action_or_null': None if observed_action_or_null is None else dict(observed_action_or_null), 'public_transition_or_null': None if public_transition_or_null is None else dict(public_transition_or_null), 'public_entity_candidates': [{'public_entity_ref': item} for item in refs], 'candidate_memory_ids': list(candidates)}
    expected_fields = MEMORY_PROPOSER_REQUEST_FIELDS
    if mode == 'observe':
        if public_observation_or_null is not None or observed_action_or_null is not None or public_transition_or_null is not None or refs or candidates:
            raise MethodContractError('method_schema_invalid', 'observe request received context outside its public whitelist')
        raw_task_context = dict(public_task_context or {})
        allowed_context_fields = frozenset({'active_subtask_id', 'active_subtask_id_or_null', 'human_messages'})
        if set(raw_task_context) - allowed_context_fields:
            raise MethodContractError('method_schema_invalid', 'observe task context contains non-public fields', pointer='/memory_proposer_request/public_task_context')
        if 'active_subtask_id' in raw_task_context and 'active_subtask_id_or_null' in raw_task_context:
            raise MethodContractError('method_schema_invalid', 'observe task context contains duplicate active-subtask fields', pointer='/memory_proposer_request/public_task_context')
        task_context = {'active_subtask_id_or_null': raw_task_context.get('active_subtask_id', raw_task_context.get('active_subtask_id_or_null')), 'human_messages': raw_task_context.get('human_messages', [])}
        request = {key: request[key] for key in ('schema_version', 'proposal_id', 'mode', 'base_revision', 'evidence_frame_ids', 'images', 'previous_evidence_memory')}
        request.update({'public_instruction': non_empty_string(public_instruction, pointer='/memory_proposer_request/public_instruction'), 'public_task_context': dict(task_context), 'observation_tool_call_or_null': None if observation_tool_call_or_null is None else dict(observation_tool_call_or_null)})
        expected_fields = MEMORY_PROPOSER_REQUEST_V3_FIELDS
    elif any((value is not None for value in (public_instruction, public_task_context, observation_tool_call_or_null))):
        raise MethodContractError('method_schema_invalid', 'observe-only public context was supplied to another mode')
    exact_mapping(request, expected_fields, pointer='/memory_proposer_request')
    public_json(request, pointer='/memory_proposer_request', include_future_action=True)
    return request
__all__ = ['IMAGE_METADATA_FIELDS', 'MEMORY_PROPOSER_REQUEST_FIELDS', 'MEMORY_PROPOSER_REQUEST_V3_FIELDS', 'MemoryProposerClient', 'build_memory_proposer_request']
