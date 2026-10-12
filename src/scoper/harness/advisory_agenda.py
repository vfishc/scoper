from __future__ import annotations
import copy
import os
import re
import time
from typing import Any, Callable, Mapping, Sequence
from scoper.harness.canonical import canonical_json_bytes, sha256_bytes, sha256_json, strict_json_loads, validate_json_value
ADVISORY_AGENDA_COMPONENT_ID = 'instruction-advisory-agenda.v1'
ADVISORY_AGENDA_QWEN_EXPERIMENT_ENV = 'EMBODIED_METHOD_QWEN_ADVISORY_AGENDA'
ADVISORY_AGENDA_V2_COMPONENT_ID = 'instruction-advisory-agenda.v2'
ADVISORY_AGENDA_V2_QWEN_EXPERIMENT_ENV = 'EMBODIED_METHOD_QWEN_ADVISORY_AGENDA_V2'
ADVISORY_AGENDA_DIAGNOSTICS_JSONL_ENV = 'EMBODIED_METHOD_ADVISORY_AGENDA_DIAGNOSTICS_JSONL'
ADVISORY_AGENDA_MAX_NEW_TOKENS = 512
ADVISORY_AGENDA_SCHEMA_VERSION = 'instruction-advisory-agenda.v1'
ADVISORY_AGENDA_REQUEST_SCHEMA_VERSION = 'instruction-advisory-agenda-request.v1'
ADVISORY_AGENDA_POLICY_FOOTER_SCHEMA_VERSION = 'instruction-advisory-agenda-policy-footer.v1'
ADVISORY_AGENDA_POLICY_FOOTER_PREFIX_BYTES = b'ADVISORY_AGENDA_REFERENCE='
ADVISORY_AGENDA_V2_POLICY_FOOTER_PREFIX_BYTES = b'ADVISORY_AGENDA_V2_REFERENCE='
ADVISORY_AGENDA_V2_REQUEST_SCHEMA_VERSION = 'instruction-advisory-agenda-request.v2'
ADVISORY_AGENDA_V2_POLICY_FOOTER_SCHEMA_VERSION = 'instruction-advisory-agenda-policy-footer.v2'
ADVISORY_AGENDA_MAX_ITEMS = 8
ADVISORY_AGENDA_MAX_CONSECUTIVE_FAILURES = 2
ADVISORY_AGENDA_SERIALIZER_IDENTITY_BYTES = b'canonical-json-instruction-advisory-agenda.v1;items=id,requirement,mode,status,evidence-refs,next-intent-or-null;modes=required,ongoing,conditional;statuses=pending,active,watching,supported,blocked;max-items=8'
ADVISORY_AGENDA_SYSTEM_BYTES = b'You maintain a small advisory agenda for an embodied policy. The agenda is a fallible reference, not an action command and not proof of task success.\nDerive items only from the instruction and PUBLIC_CONTEXT. Never infer hidden simulator state, private goal progress, object IDs, or Binding state.\nRepresent required outcomes or ordered stages as required, constraints that must remain true as ongoing, and if/may/event clauses as conditional. Keep alternatives such as \'on or in\', quantifiers such as \'all\', and prohibitions in requirement text.\nA condition without public evidence remains watching. Not seeing an object does not prove absence or completion. Use supported only when public evidence or public action history supports the requirement; revise it if later evidence contradicts it.\nPreserve stable item IDs when updating. Keep requirement and next_intent_or_null short. next_intent_or_null describes an intent, never a mandatory ToolCall.\nReturn one compact JSON object only. Exact top-level fields are active_item_id_or_null, items, schema_version. schema_version must be instruction-advisory-agenda.v1. Each item has exactly evidence_refs, id, mode, next_intent_or_null, requirement, status. mode is required, ongoing, or conditional. status is pending, active, watching, supported, or blocked. Use at most eight items and at most one active item. Do not use code fences.\nExample shape: {"active_item_id_or_null":"a1","items":[{"evidence_refs":[],"id":"a1","mode":"required","next_intent_or_null":"locate the mug","requirement":"wash the mug","status":"active"}],"schema_version":"instruction-advisory-agenda.v1"}\n'
ADVISORY_AGENDA_POLICY_GUIDANCE = 'This agenda is fallible advisory context generated from public information. Use it to remember instruction clauses, but follow the current image, public evidence, allowed tools, and recovery mode when they disagree. Agenda status never requires or forbids a ToolCall and never proves final task success.'
ADVISORY_AGENDA_COMPLETION_GUIDANCE = 'The model-maintained agenda currently marks every required item supported. Reconsider whether stop is appropriate using the current image and public evidence. The agenda is fallible and is not proof of task success.'
ADVISORY_AGENDA_V2_SYSTEM_BYTES = ADVISORY_AGENDA_SYSTEM_BYTES.replace(b'Use supported only when public evidence or public action history supports the requirement; revise it if later evidence contradicts it.\n', b'Use supported only when evidence_refs cites current certainty=verified effective_facts. Tool requests, action history, scene observations, and assumed effects never prove completion. Revise supported if later verified evidence contradicts it.\n')
ADVISORY_AGENDA_V2_STOP_REASK_BYTES = b'Soft review: required agenda items remain unsupported by verified public evidence. Review the full instruction and current public context, then return the best next ToolCall. You may still return stop if you judge it appropriate.'
_ITEM_MODES = frozenset({'required', 'ongoing', 'conditional'})
_ITEM_STATUSES = frozenset({'pending', 'active', 'watching', 'supported', 'blocked'})
_ITEM_ID_RE = re.compile('[A-Za-z][A-Za-z0-9_-]{0,31}\\Z')
_PHYSICAL_TOOLS = frozenset({'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'slice', 'clean', 'saturate', 'fill', 'heat', 'cool'})

def _text(value: Any, *, field: str, maximum: int, nullable: bool=False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f'advisory agenda {field} must be 1..{maximum} characters')
    return value

def validate_advisory_agenda(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {'active_item_id_or_null', 'items', 'schema_version'}:
        raise ValueError('advisory agenda has unexpected top-level fields')
    if value.get('schema_version') != ADVISORY_AGENDA_SCHEMA_VERSION:
        raise ValueError('advisory agenda schema version differs')
    raw_items = value.get('items')
    if not isinstance(raw_items, list) or not raw_items or len(raw_items) > ADVISORY_AGENDA_MAX_ITEMS:
        raise ValueError('advisory agenda item count is invalid')
    items: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, Mapping) or set(raw) != {'evidence_refs', 'id', 'mode', 'next_intent_or_null', 'requirement', 'status'}:
            raise ValueError('advisory agenda item fields differ')
        identifier = _text(raw.get('id'), field='item id', maximum=32)
        assert isinstance(identifier, str)
        if not _ITEM_ID_RE.fullmatch(identifier) or identifier in identifiers:
            raise ValueError('advisory agenda item id is invalid or duplicated')
        identifiers.add(identifier)
        mode = raw.get('mode')
        status = raw.get('status')
        if mode not in _ITEM_MODES or status not in _ITEM_STATUSES:
            raise ValueError('advisory agenda item mode or status is invalid')
        refs = raw.get('evidence_refs')
        if not isinstance(refs, list) or len(refs) > 8 or any((not isinstance(item, str) or not item or len(item) > 128 for item in refs)) or (len(set(refs)) != len(refs)):
            raise ValueError('advisory agenda evidence refs are invalid')
        items.append({'evidence_refs': list(refs), 'id': identifier, 'mode': mode, 'next_intent_or_null': _text(raw.get('next_intent_or_null'), field='next intent', maximum=160, nullable=True), 'requirement': _text(raw.get('requirement'), field='requirement', maximum=240), 'status': status})
    active = value.get('active_item_id_or_null')
    if active is not None and active not in identifiers:
        raise ValueError('advisory agenda active item is unknown')
    result = {'active_item_id_or_null': active, 'items': items, 'schema_version': ADVISORY_AGENDA_SCHEMA_VERSION}
    validate_json_value(result, json_pointer='/advisory_agenda')
    return result

def parse_advisory_agenda_text(text: str) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text:
        raise ValueError('advisory agenda output must be non-empty text')
    stripped = text.strip(' \t\r\n')
    value = strict_json_loads(stripped, source='advisory agenda output')
    return validate_advisory_agenda(value)

def advisory_agenda_descriptor(*, version: int=1) -> Mapping[str, Any]:
    if version not in {1, 2}:
        raise ValueError('advisory agenda version is unsupported')
    return {'schema_version': f'instruction-advisory-agenda-adapter.v{version}', 'component_id': ADVISORY_AGENDA_V2_COMPONENT_ID if version == 2 else ADVISORY_AGENDA_COMPONENT_ID, 'agenda_schema_version': ADVISORY_AGENDA_SCHEMA_VERSION, 'request_schema_version': ADVISORY_AGENDA_V2_REQUEST_SCHEMA_VERSION if version == 2 else ADVISORY_AGENDA_REQUEST_SCHEMA_VERSION, 'policy_footer_schema_version': ADVISORY_AGENDA_V2_POLICY_FOOTER_SCHEMA_VERSION if version == 2 else ADVISORY_AGENDA_POLICY_FOOTER_SCHEMA_VERSION, 'system_prompt_sha256': sha256_bytes(ADVISORY_AGENDA_V2_SYSTEM_BYTES if version == 2 else ADVISORY_AGENDA_SYSTEM_BYTES), 'serializer_sha256': sha256_bytes(ADVISORY_AGENDA_SERIALIZER_IDENTITY_BYTES), 'max_items': ADVISORY_AGENDA_MAX_ITEMS, 'semantics': 'advisory_fail_soft_one_stop_review' if version == 2 else 'advisory_fail_soft_no_tool_or_stop_gate'}

def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def _tool_call(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    nested = value.get('tool_call')
    if isinstance(nested, Mapping):
        value = nested
    name = value.get('tool_name', value.get('name'))
    arguments = value.get('arguments', value.get('public_arguments', {}))
    if not isinstance(name, str) or not name or (not isinstance(arguments, Mapping)):
        return None
    result = {'arguments': copy.deepcopy(dict(arguments)), 'tool_name': name}
    validate_json_value(result, json_pointer='/advisory_agenda/tool_call')
    return result

def _history(policy_context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = policy_context.get('tool_history')
    if not isinstance(raw, list):
        raise ValueError('advisory agenda requires public tool history')
    result = []
    for item in raw[-16:]:
        call = _tool_call(item)
        if call is None:
            raise ValueError('advisory agenda public history contains an invalid ToolCall')
        result.append(call)
    return result

def _public_prompt(policy_context: Mapping[str, Any]) -> Mapping[str, Any]:
    prompt = _mapping(policy_context.get('public_observation_without_rgb'))
    fields = ('current_room_handle_or_null', 'held_object_handles', 'human_messages', 'public_notices', 'visited_room_handles', 'visible_issues')
    return {field: copy.deepcopy(prompt[field]) for field in fields if field in prompt}

def _last_transition(policy_context: Mapping[str, Any]) -> Any:
    return copy.deepcopy(policy_context.get('last_public_transition', policy_context.get('last_public_transition_or_null')))

def _public_state(policy_context: Mapping[str, Any]) -> Mapping[str, Any]:
    memory = policy_context.get('evidence_memory')
    if not isinstance(memory, Mapping):
        raise ValueError('advisory agenda requires public evidence memory')
    state = {'assumed_completed_actions': copy.deepcopy(policy_context.get('assumed_completed_actions', [])), 'effective_facts': copy.deepcopy(policy_context.get('effective_facts', [])), 'evidence_memory': copy.deepcopy(dict(memory)), 'last_public_transition_or_null': _last_transition(policy_context), 'public_prompt_context': _public_prompt(policy_context), 'recovery_or_null': copy.deepcopy(policy_context.get('recovery')), 'tool_history': _history(policy_context)}
    validate_json_value(state, json_pointer='/advisory_agenda/public_context')
    return state

def _public_state_v2(policy_context: Mapping[str, Any]) -> Mapping[str, Any]:
    state = dict(_public_state(policy_context))
    state.pop('assumed_completed_actions', None)
    state.pop('tool_history', None)
    validate_json_value(state, json_pointer='/advisory_agenda_v2/public_context')
    return state

def advisory_agenda_request(policy_context: Mapping[str, Any], *, previous_agenda_or_null: Mapping[str, Any] | None, trigger_reasons: Sequence[str]) -> Mapping[str, Any]:
    instruction = policy_context.get('instruction')
    if not isinstance(instruction, str) or not instruction:
        raise ValueError('advisory agenda requires the public instruction')
    if previous_agenda_or_null is not None:
        previous_agenda_or_null = validate_advisory_agenda(previous_agenda_or_null)
    reasons = list(trigger_reasons)
    if not reasons or any((not isinstance(item, str) or not item for item in reasons)):
        raise ValueError('advisory agenda trigger reasons are invalid')
    request = {'instruction': instruction, 'mode': 'initialize' if previous_agenda_or_null is None else 'update', 'previous_agenda_or_null': copy.deepcopy(previous_agenda_or_null), 'public_context': _public_state(policy_context), 'schema_version': ADVISORY_AGENDA_REQUEST_SCHEMA_VERSION, 'trigger_reasons': reasons}
    validate_json_value(request, json_pointer='/advisory_agenda_request')
    return request

def advisory_agenda_v2_request(policy_context: Mapping[str, Any], *, previous_agenda_or_null: Mapping[str, Any] | None, trigger_reasons: Sequence[str]) -> Mapping[str, Any]:
    request = dict(advisory_agenda_request(policy_context, previous_agenda_or_null=previous_agenda_or_null, trigger_reasons=trigger_reasons))
    request['public_context'] = _public_state_v2(policy_context)
    request['schema_version'] = ADVISORY_AGENDA_V2_REQUEST_SCHEMA_VERSION
    validate_json_value(request, json_pointer='/advisory_agenda_v2_request')
    return request

def advisory_agenda_model_messages(request: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(request, Mapping):
        raise ValueError('advisory agenda request must be a mapping')
    system_bytes = ADVISORY_AGENDA_V2_SYSTEM_BYTES if request.get('schema_version') == ADVISORY_AGENDA_V2_REQUEST_SCHEMA_VERSION else ADVISORY_AGENDA_SYSTEM_BYTES
    return ({'role': 'system', 'content': [{'type': 'text', 'text': system_bytes.decode('utf-8')}]}, {'role': 'user', 'content': [{'type': 'text', 'text': canonical_json_bytes(request).decode('utf-8')}]})

def advisory_agenda_policy_user_bytes(base_user_bytes: bytes, projection: Mapping[str, Any]) -> bytes:
    if not isinstance(base_user_bytes, bytes) or not base_user_bytes:
        raise ValueError('advisory agenda base policy bytes are invalid')
    validate_json_value(projection, json_pointer='/advisory_agenda_policy_footer')
    return base_user_bytes + b'\n' + ADVISORY_AGENDA_POLICY_FOOTER_PREFIX_BYTES + canonical_json_bytes(projection)

def advisory_agenda_v2_policy_user_bytes(base_user_bytes: bytes, projection: Mapping[str, Any]) -> bytes:
    if not isinstance(base_user_bytes, bytes) or not base_user_bytes:
        raise ValueError('advisory agenda base policy bytes are invalid')
    validate_json_value(projection, json_pointer='/advisory_agenda_v2_policy_footer')
    return base_user_bytes + b'\n' + ADVISORY_AGENDA_V2_POLICY_FOOTER_PREFIX_BYTES + canonical_json_bytes(projection)

class AdvisoryAgendaController:

    def __init__(self, generator: Callable[[Sequence[Mapping[str, Any]]], str], *, version: int=1) -> None:
        if not callable(generator):
            raise TypeError('advisory agenda generator must be callable')
        self._generator = generator
        if version not in {1, 2}:
            raise ValueError('advisory agenda version is unsupported')
        self.version = version
        self.start()

    def start(self) -> None:
        self.agenda: Mapping[str, Any] | None = None
        self.revision = 0
        self.consecutive_failures = 0
        self.updates_paused = False
        self._attempted = False
        self._last_evidence_sha: str | None = None
        self._last_human_sha: str | None = None
        self._last_recovery_sha: str | None = None
        self._last_history_sha: str | None = None
        self._completion_advisory = False
        self._stop_reask_used = False
        self.audit_events: list[Mapping[str, Any]] = []

    def finish(self) -> None:
        return None

    @staticmethod
    def _signatures(policy_context: Mapping[str, Any]) -> Mapping[str, str]:
        state = _public_state(policy_context)
        memory = _mapping(state['evidence_memory'])
        prompt = _mapping(state['public_prompt_context'])
        recovery = {'incidents': memory.get('incidents', []), 'pending_obligations': memory.get('pending_obligations', []), 'recovery': state['recovery_or_null']}
        return {'evidence': sha256_json({'facts': memory.get('facts', []), 'effective_facts': state['effective_facts'], 'assumed_completed_actions': state['assumed_completed_actions']}), 'human': sha256_json({'human_messages': prompt.get('human_messages', []), 'public_notices': prompt.get('public_notices', [])}), 'recovery': sha256_json(recovery), 'history': sha256_json(state['tool_history'])}

    def _state(self, policy_context: Mapping[str, Any]) -> Mapping[str, Any]:
        return _public_state_v2(policy_context) if self.version == 2 else _public_state(policy_context)

    def _controller_signatures(self, policy_context: Mapping[str, Any]) -> Mapping[str, str]:
        if self.version == 1:
            return self._signatures(policy_context)
        state = self._state(policy_context)
        memory = _mapping(state['evidence_memory'])
        prompt = _mapping(state['public_prompt_context'])
        return {'evidence': sha256_json({'facts': memory.get('facts', []), 'effective_facts': state['effective_facts']}), 'human': sha256_json({'human_messages': prompt.get('human_messages', []), 'public_notices': prompt.get('public_notices', [])}), 'recovery': sha256_json({'incidents': memory.get('incidents', []), 'pending_obligations': memory.get('pending_obligations', []), 'recovery': state['recovery_or_null']}), 'history': ''}

    def _trigger_reasons(self, policy_context: Mapping[str, Any], signatures: Mapping[str, str]) -> list[str]:
        reasons: list[str] = []
        if not self._attempted:
            reasons.append('episode_initialize')
        elif signatures['evidence'] != self._last_evidence_sha:
            reasons.append('public_evidence_changed')
        if self._attempted and signatures['human'] != self._last_human_sha:
            reasons.append('public_event_changed')
        history = _history(policy_context)
        if self.version == 1 and self._attempted and (signatures['history'] != self._last_history_sha) and history and (history[-1]['tool_name'] in _PHYSICAL_TOOLS):
            reasons.append('physical_action_recorded')
        if self.version == 2 and self._attempted and (signatures['recovery'] != self._last_recovery_sha):
            reasons.append('public_recovery_changed')
        return list(dict.fromkeys(reasons))

    def before_policy(self, policy_context: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            signatures = self._controller_signatures(policy_context)
            reasons = self._trigger_reasons(policy_context, signatures)
        except Exception as exc:
            self._record_failure(('public_projection_failed',), exc, None, 0.0)
            return self.policy_projection()
        if reasons and (not self.updates_paused):
            request: Mapping[str, Any] | None = None
            started = time.monotonic()
            try:
                builder = advisory_agenda_v2_request if self.version == 2 else advisory_agenda_request
                request = builder(policy_context, previous_agenda_or_null=self.agenda, trigger_reasons=reasons)
                raw = self._generator(advisory_agenda_model_messages(request))
                agenda = parse_advisory_agenda_text(raw)
                if self.version == 2:
                    verified_ids = {item.get('fact_id') for item in request['public_context']['effective_facts'] if isinstance(item, Mapping) and item.get('certainty') == 'verified' and isinstance(item.get('fact_id'), str)}
                    if any((item['status'] == 'supported' and (not item['evidence_refs'] or not set(item['evidence_refs']).issubset(verified_ids)) for item in agenda['items'])):
                        raise ValueError('agenda v2 supported status lacks verified effective-fact evidence')
            except Exception as exc:
                self._record_failure(reasons, exc, request, time.monotonic() - started)
            else:
                self.agenda = agenda
                self.revision += 1
                self.consecutive_failures = 0
                self.audit_events.append({'agenda_revision': self.revision, 'duration_seconds': time.monotonic() - started, 'request_sha256': sha256_json(request), 'response_sha256': sha256_json(agenda), 'status': 'updated', 'trigger_reasons': reasons})
        self._attempted = True
        self._last_evidence_sha = signatures['evidence']
        self._last_human_sha = signatures['human']
        self._last_recovery_sha = signatures['recovery']
        self._last_history_sha = signatures['history']
        memory = _mapping(policy_context.get('evidence_memory'))
        required = [item for item in (self.agenda or {}).get('items', []) if item.get('mode') == 'required']
        self._completion_advisory = bool(required and all((item.get('status') == 'supported' for item in required)) and (not any((isinstance(item, Mapping) and item.get('status') == 'open' for field in ('incidents', 'pending_obligations') for item in memory.get(field, [])))))
        if len(self.audit_events) > 64:
            self.audit_events = self.audit_events[-64:]
        return self.policy_projection()

    def _record_failure(self, reasons: Sequence[str], exc: Exception, request: Mapping[str, Any] | None, duration: float) -> None:
        self._attempted = True
        self.consecutive_failures += 1
        if self.consecutive_failures >= ADVISORY_AGENDA_MAX_CONSECUTIVE_FAILURES:
            self.updates_paused = True
        self.audit_events.append({'agenda_revision': self.revision, 'duration_seconds': duration, 'error_type': type(exc).__name__, 'request_sha256': None if request is None else sha256_json(request), 'status': 'failed_soft', 'trigger_reasons': list(reasons), 'updates_paused': self.updates_paused})

    def policy_projection(self) -> Mapping[str, Any]:
        projection = {'agenda_or_null': copy.deepcopy(self.agenda), 'agenda_revision': self.revision, 'advisory_only': True, 'completion_advisory_or_null': ADVISORY_AGENDA_COMPLETION_GUIDANCE if self._completion_advisory else None, 'guidance': ADVISORY_AGENDA_POLICY_GUIDANCE, 'schema_version': ADVISORY_AGENDA_V2_POLICY_FOOTER_SCHEMA_VERSION if self.version == 2 else ADVISORY_AGENDA_POLICY_FOOTER_SCHEMA_VERSION, 'updates_paused_after_failures': self.updates_paused}
        validate_json_value(projection, json_pointer='/advisory_agenda_policy_footer')
        return projection

    def stop_reask_message_bytes(self, tool_name: str) -> bytes | None:
        if self.version != 2 or tool_name != 'stop' or self._stop_reask_used or (self.agenda is None):
            return None
        required = [item for item in self.agenda['items'] if item['mode'] == 'required']
        if not required or all((item['status'] == 'supported' for item in required)):
            return None
        self._stop_reask_used = True
        return ADVISORY_AGENDA_V2_STOP_REASK_BYTES

    def diagnostics(self) -> Mapping[str, Any]:
        return {'adapter': advisory_agenda_descriptor(version=self.version), 'agenda_revision': self.revision, 'consecutive_failures': self.consecutive_failures, 'events': copy.deepcopy(self.audit_events), 'updates_paused': self.updates_paused}

def append_advisory_agenda_diagnostics_from_env(diagnostics: Mapping[str, Any]) -> bool:
    path = os.environ.get(ADVISORY_AGENDA_DIAGNOSTICS_JSONL_ENV)
    if not path:
        return False
    try:
        validate_json_value(diagnostics, json_pointer='/advisory_agenda_diagnostics')
        payload = canonical_json_bytes(diagnostics) + b'\n'
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 384)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
    except (OSError, TypeError, ValueError):
        return False
    return True
__all__ = ['ADVISORY_AGENDA_COMPONENT_ID', 'ADVISORY_AGENDA_DIAGNOSTICS_JSONL_ENV', 'ADVISORY_AGENDA_MAX_ITEMS', 'ADVISORY_AGENDA_MAX_NEW_TOKENS', 'ADVISORY_AGENDA_POLICY_FOOTER_PREFIX_BYTES', 'ADVISORY_AGENDA_POLICY_FOOTER_SCHEMA_VERSION', 'ADVISORY_AGENDA_QWEN_EXPERIMENT_ENV', 'ADVISORY_AGENDA_REQUEST_SCHEMA_VERSION', 'ADVISORY_AGENDA_SCHEMA_VERSION', 'ADVISORY_AGENDA_SERIALIZER_IDENTITY_BYTES', 'ADVISORY_AGENDA_SYSTEM_BYTES', 'ADVISORY_AGENDA_V2_COMPONENT_ID', 'ADVISORY_AGENDA_V2_POLICY_FOOTER_PREFIX_BYTES', 'ADVISORY_AGENDA_V2_POLICY_FOOTER_SCHEMA_VERSION', 'ADVISORY_AGENDA_V2_QWEN_EXPERIMENT_ENV', 'ADVISORY_AGENDA_V2_REQUEST_SCHEMA_VERSION', 'ADVISORY_AGENDA_V2_STOP_REASK_BYTES', 'ADVISORY_AGENDA_V2_SYSTEM_BYTES', 'AdvisoryAgendaController', 'append_advisory_agenda_diagnostics_from_env', 'advisory_agenda_descriptor', 'advisory_agenda_model_messages', 'advisory_agenda_policy_user_bytes', 'advisory_agenda_request', 'advisory_agenda_v2_policy_user_bytes', 'advisory_agenda_v2_request', 'parse_advisory_agenda_text', 'validate_advisory_agenda']
