from __future__ import annotations
import copy
import os
import re
import unicodedata
from typing import Any, Mapping
from scoper.harness.canonical import canonical_json_bytes, sha256_json, validate_json_value
QWEN_POLICY_LOOP_GUARD_ENV = 'EMBODIED_METHOD_QWEN_LOOP_GUARD'
QWEN_POLICY_LOOP_GUARD_DIAGNOSTICS_JSONL_ENV = 'EMBODIED_METHOD_QWEN_LOOP_GUARD_DIAGNOSTICS_JSONL'
QWEN_POLICY_LOOP_GUARD_ID = 'qwen3.5-policy-semantic-exploration-guard.d4c8-public-verified-progress-stop-option'
QWEN_POLICY_READINESS_GUARD_ID = 'qwen3.5-policy-readiness-retry.d1'
QWEN_POLICY_LOOP_GUARD_SCHEMA_VERSION = 'qwen-policy-semantic-exploration-rejection.v1'
QWEN_POLICY_LOOP_GUARD_RETRY_SCHEMA_VERSION = 'qwen-policy-semantic-exploration-retry.d4c8-public-verified-progress-stop-option'
QWEN_POLICY_LOOP_GUARD_MAX_REASKS = 2
QWEN_POLICY_LOOP_GUARD_SCAN_PER_ROOM_LIMIT = int(os.environ.get('EMBODIED_METHOD_QWEN_SCAN_PER_ROOM_LIMIT', '2'))
QWEN_POLICY_LOOP_GUARD_REQUIRE_ROOM_ENTRY = os.environ.get('EMBODIED_METHOD_QWEN_REQUIRE_ROOM_ENTRY', '0') == '1'
QWEN_POLICY_LOOP_GUARD_LOOK_PER_TARGET_LIMIT = 2
QWEN_POLICY_LOOP_GUARD_NAVIGATE_PER_TARGET_LIMIT = 2
QWEN_POLICY_LOOP_GUARD_OBSERVATION_LIMIT = 4
QWEN_POLICY_LOOP_GUARD_PREPARATION_LIMIT = 6
QWEN_POLICY_LOOP_GUARD_PHASE_NORMAL = 'normal'
QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_NAVIGATION = 'commit_navigation'
QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_PHYSICAL = 'commit_physical'
QWEN_POLICY_VERIFIED_HELD_GUARD_ID = 'qwen3.5-policy-verified-held-reask.d4c2'
QWEN_POLICY_VERIFIED_HELD_MAX_REASKS = 1
_OBSERVATION_TOOLS = frozenset({'look_for', 'scan'})
_PREPARATION_TOOLS = frozenset({'look_for', 'navigate_to', 'scan'})
_NON_PHYSICAL_TOOLS = frozenset({'answer', 'get_context', 'look_for', 'navigate_to', 'navigate_to_room', 'scan', 'stop'})
_VOLATILE_PUBLIC_KEYS = frozenset({'base_revision', 'budget_status', 'created_step', 'elapsed_seconds', 'evidence_frame_ids', 'frame_id', 'high_level_step', 'last_updated_step', 'media_sha256', 'recovery_attempts', 'revision', 'step', 'timestamp', 'updated_step'})
_TARGET_TOKEN_RE = re.compile('[\\W_]+', flags=re.UNICODE)
_LEADING_ARTICLES = frozenset({'a', 'an', 'the'})
_READINESS_STATUS = 'not_interaction_ready'
_READINESS_REASONS = frozenset({'not_interaction_ready', 'target_not_interaction_ready', 'recent_focus_not_interaction_ready', 'successful_navigation_focus_required', 'navigation_context_required'})

def _canonical_tool_call(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {'arguments', 'tool_name'}:
        raise ValueError('Qwen exploration guard requires one exact ToolCall')
    if not isinstance(value.get('tool_name'), str) or not value['tool_name']:
        raise ValueError('Qwen exploration guard tool name is invalid')
    if not isinstance(value.get('arguments'), Mapping):
        raise ValueError('Qwen exploration guard arguments must be an object')
    result = {'arguments': copy.deepcopy(dict(value['arguments'])), 'tool_name': value['tool_name']}
    validate_json_value(result, json_pointer='/exploration_guard/tool_call')
    return result

def _tool_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    nested = value.get('tool_call')
    if isinstance(nested, Mapping):
        value = nested
    name = value.get('tool_name', value.get('name'))
    arguments = value.get('arguments', value.get('public_arguments', {}))
    if not isinstance(name, str) or not name or (not isinstance(arguments, Mapping)):
        return None
    return _canonical_tool_call({'arguments': arguments, 'tool_name': name})

def _stable_public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_public_value(item) for key, item in value.items() if isinstance(key, str) and key not in _VOLATILE_PUBLIC_KEYS}
    if isinstance(value, (list, tuple)):
        return [_stable_public_value(item) for item in value]
    return copy.deepcopy(value)

def _memory_rows(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        rows = [{**copy.deepcopy(dict(item)), 'obligation_id': str(identifier)} if isinstance(item, Mapping) else item for identifier, item in value.items()]
    elif isinstance(value, list):
        rows = list(value)
    else:
        rows = []
    stable = [_stable_public_value(item) for item in rows if isinstance(item, Mapping)]
    return sorted(stable, key=canonical_json_bytes)

def _observation(policy_context: Mapping[str, Any], public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any]:
    value = policy_context.get('public_observation_without_rgb')
    if not isinstance(value, Mapping):
        value = policy_context.get('public_observation')
    output = copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    if isinstance(public_observation_or_null, Mapping):
        for field in ('current_room', 'current_room_handle_or_null', 'held_object_handles', 'held_object_types', 'human_messages', 'visited_room_handles', 'visited_rooms'):
            if field in public_observation_or_null:
                output[field] = copy.deepcopy(public_observation_or_null[field])
    return output

def _history(policy_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = policy_context.get('tool_history', [])
    if not isinstance(raw, list):
        raise ValueError('Qwen exploration guard requires public tool history')
    output = []
    for item in raw:
        call = _tool_call(item)
        if call is None:
            raise ValueError('Qwen exploration guard history row is invalid')
        output.append(call)
    return output

def _room_key(policy_context: Mapping[str, Any], public_observation_or_null: Mapping[str, Any] | None=None) -> str:
    observation = _observation(policy_context, public_observation_or_null)
    value = observation.get('current_room_handle_or_null', observation.get('current_room'))
    return value if isinstance(value, str) and value else '<unknown-room>'

def _normalize_target(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return '<no-target>'
    normalized = unicodedata.normalize('NFKC', value).casefold().strip()
    tokens = [item for item in _TARGET_TOKEN_RE.split(normalized) if item]
    if tokens and tokens[0] in _LEADING_ARTICLES:
        tokens = tokens[1:]
    return ' '.join(tokens) or '<no-target>'

def _semantic_key(policy_context: Mapping[str, Any], call: Mapping[str, Any], public_observation_or_null: Mapping[str, Any] | None=None) -> tuple[str, str] | None:
    name = call['tool_name']
    if name == 'scan':
        return (name, _room_key(policy_context, public_observation_or_null))
    if name in {'look_for', 'navigate_to'}:
        return (name, _normalize_target(call['arguments'].get('target')))
    return None

def _physical_fingerprint(call: Mapping[str, Any]) -> tuple[str, str, bytes] | None:
    name = call.get('tool_name')
    if not isinstance(name, str) or name in _NON_PHYSICAL_TOOLS:
        return None
    arguments = call.get('arguments')
    if not isinstance(arguments, Mapping):
        return None
    target = _normalize_target(arguments.get('target'))
    remaining = {str(key): copy.deepcopy(value) for key, value in arguments.items() if key != 'target'}
    return (name, target, canonical_json_bytes(remaining))

def qwen_policy_readiness_failure(public_transition_feedback_or_null: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(public_transition_feedback_or_null, Mapping):
        return None
    feedback = public_transition_feedback_or_null
    nested = feedback.get('qwen_public_transition_feedback')
    if isinstance(nested, Mapping):
        feedback = nested
    binding = feedback.get('binding_admission_or_null')
    if not isinstance(binding, Mapping):
        binding = feedback.get('binding')
    if not isinstance(binding, Mapping):
        return None
    status = binding.get('status')
    reason = binding.get('reason')
    if status != _READINESS_STATUS and reason not in _READINESS_REASONS:
        return None
    call = _tool_call(feedback.get('tool_call'))
    if call is None:
        call = _tool_call(feedback)
    if call is None or _physical_fingerprint(call) is None:
        return None
    result = {'binding_status': status if isinstance(status, str) else None, 'binding_reason': reason if isinstance(reason, str) else None, 'tool_call': call}
    validate_json_value(result, json_pointer='/qwen_policy_readiness_failure')
    return result

def _readiness_progress_material(policy_context: Mapping[str, Any], public_observation_or_null: Mapping[str, Any] | None) -> Mapping[str, Any]:
    observation = _observation(policy_context, public_observation_or_null)
    memory = policy_context.get('evidence_memory')
    facts = memory.get('facts', []) if isinstance(memory, Mapping) else []
    return {'room': _room_key(policy_context, public_observation_or_null), 'visited_rooms': _stable_public_value(observation.get('visited_room_handles', observation.get('visited_rooms', []))), 'held_object_handles': _stable_public_value(observation.get('held_object_handles', [])), 'held_object_types': _stable_public_value(observation.get('held_object_types', [])), 'meaningful_facts': sorted([_stable_public_value(item) for item in facts if isinstance(item, Mapping) and item.get('relation_or_property') not in {'seen', 'visible'}], key=canonical_json_bytes)}

def qwen_policy_loop_material(policy_context: Mapping[str, Any], public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any]:
    if not isinstance(policy_context, Mapping):
        raise ValueError('Qwen exploration guard policy context must be an object')
    memory = policy_context.get('evidence_memory')
    if not isinstance(memory, Mapping):
        raise ValueError('Qwen exploration guard requires public evidence memory')
    observation = _observation(policy_context, public_observation_or_null)
    raw_facts = memory.get('facts', [])
    meaningful_facts = [_stable_public_value(item) for item in raw_facts if isinstance(item, Mapping) and item.get('relation_or_property') not in {'seen', 'visible'}] if isinstance(raw_facts, list) else []
    material = {'held_object_handles': _stable_public_value(observation.get('held_object_handles', [])), 'held_object_types': _stable_public_value(observation.get('held_object_types', [])), 'human_messages': _stable_public_value(observation.get('human_messages', [])), 'incident_lifecycle': _memory_rows(memory.get('incidents', [])), 'obligation_lifecycle': _memory_rows(memory.get('pending_obligations', [])), 'meaningful_facts': sorted(meaningful_facts, key=canonical_json_bytes), 'room': _room_key(policy_context, public_observation_or_null), 'visited_rooms': _stable_public_value(observation.get('visited_room_handles', observation.get('visited_rooms', [])))}
    validate_json_value(material, json_pointer='/exploration_guard/material')
    return material

def _pending_qa(policy_context: Mapping[str, Any]) -> bool:
    return any((isinstance(item, Mapping) and item.get('type') == 'qa' for item in _observation(policy_context).get('human_messages', [])))

def qwen_policy_loop_guard_bypassed(policy_context: Mapping[str, Any]) -> bool:
    if not isinstance(policy_context, Mapping):
        raise ValueError('Qwen exploration guard policy context must be an object')
    return policy_context.get('mode') == 'observe' or _pending_qa(policy_context)

def _required_recovery_call(policy_context: Mapping[str, Any]) -> dict[str, Any] | None:
    if policy_context.get('mode') != 'recovery':
        return None
    recovery = policy_context.get('recovery')
    if not isinstance(recovery, Mapping):
        return None
    failed = _tool_call(recovery.get('failed_tool_call'))
    if failed is None:
        return None
    if recovery.get('placement_recovery') is True:
        required = _tool_call(recovery.get('required_tool_call'))
        if required is not None:
            return required
    phase = recovery.get('phase')
    target = failed['arguments'].get('target')
    if phase == 'look':
        if isinstance(target, str) and target:
            return {'arguments': {'target': target}, 'tool_name': 'look_for'}
        return {'arguments': {}, 'tool_name': 'scan'}
    if phase == 'navigate' and isinstance(target, str) and target:
        return {'arguments': {'target': target}, 'tool_name': 'navigate_to'}
    if phase in {'retry', 'final_retry'}:
        return failed
    return None

def _allowed_tool_names(policy_context: Mapping[str, Any]) -> list[str]:
    specs = policy_context.get('allowed_tool_specs')
    if not isinstance(specs, list):
        raise ValueError('Qwen exploration guard requires allowed tool specs')
    names = []
    for raw in specs:
        name = raw.get('name') if isinstance(raw, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValueError('Qwen exploration guard tool spec is invalid')
        names.append(name)
    return sorted(set(names), key=lambda item: item.encode('utf-8'))

def qwen_policy_verified_held_refs(policy_context: Mapping[str, Any]) -> list[str]:
    if not isinstance(policy_context, Mapping):
        raise ValueError('Qwen held guard policy context must be an object')
    facts = policy_context.get('effective_facts', [])
    if not isinstance(facts, list) or any((not isinstance(item, Mapping) for item in facts)):
        raise ValueError('Qwen held guard effective facts are invalid')
    states: dict[str, list[tuple[str, Any]]] = {}
    public_refs: dict[str, set[str]] = {}
    for fact in facts:
        if fact.get('relation_or_property') != 'held_by_agent':
            continue
        subject = fact.get('subject_public_role')
        if not isinstance(subject, str) or not subject:
            continue
        key = _normalize_target(subject)
        states.setdefault(key, []).append((str(fact.get('certainty', '')), fact.get('object_public_role_or_value')))
        public_refs.setdefault(key, set()).add(subject)
    verified: list[str] = []
    for key, rows in states.items():
        if rows and all((certainty == 'verified' and value is True for certainty, value in rows)):
            verified.append(sorted(public_refs[key], key=lambda item: item.encode('utf-8'))[0])
    return sorted(verified, key=lambda item: item.encode('utf-8'))

def _verified_physical_progress_material(policy_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    facts = policy_context.get('effective_facts', [])
    if not isinstance(facts, list) or any((not isinstance(item, Mapping) for item in facts)):
        raise ValueError('Qwen physical-progress effective facts are invalid')
    rows = [_stable_public_value(item) for item in facts if item.get('certainty') == 'verified' and item.get('status', 'current') == 'current' and (item.get('relation_or_property') not in {'seen', 'visible'})]
    return sorted(rows, key=canonical_json_bytes)

class ExplorationLoopGuard:

    def __init__(self) -> None:
        self.start()

    def start(self) -> None:
        self._material_sha256: str | None = None
        self._accepted_preparation_keys: list[tuple[str, str]] = []
        self._physical_call_accepted_in_episode = False
        self._pending_physical_call: bytes | None = None
        self._preserve_exploration_epoch_until_physical_admission = False
        self._readiness_block: tuple[str, str, bytes] | None = None
        self._readiness_progress_sha256: str | None = None
        self._rejection_trace: list[dict[str, Any]] = []
        self.rejection_count = 0
        self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_NORMAL
        self._verified_physical_progress_sha256: str | None = None

    def finish(self) -> None:
        self._accepted_preparation_keys = []
        self._material_sha256 = None
        self._physical_call_accepted_in_episode = False
        self._pending_physical_call = None
        self._preserve_exploration_epoch_until_physical_admission = False
        self._readiness_block = None
        self._readiness_progress_sha256 = None
        self._rejection_trace = []
        self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_NORMAL
        self._verified_physical_progress_sha256 = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def rejection_trace(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(copy.deepcopy(self._rejection_trace))

    def exhaustion_message(self) -> str:
        trace = canonical_json_bytes(self._rejection_trace).decode('utf-8')
        return f'Qwen continued a prohibited no-progress ToolCall loop; trace={trace}'

    def reask_context(self, policy_context: Mapping[str, Any], rejection: Mapping[str, Any]) -> dict[str, Any]:
        output = copy.deepcopy(dict(policy_context))
        specs = output.get('allowed_tool_specs')
        blocked = rejection.get('temporarily_blocked_tool_names', ())
        if isinstance(specs, list) and isinstance(blocked, list):
            output['allowed_tool_specs'] = [copy.deepcopy(spec) for spec in specs if not isinstance(spec, Mapping) or spec.get('name') not in set(blocked)]
        return output

    def _sync(self, policy_context: Mapping[str, Any], public_observation_or_null: Mapping[str, Any] | None) -> bool:
        verified_progress_sha256 = sha256_json(_verified_physical_progress_material(policy_context))
        if self._verified_physical_progress_sha256 is None:
            self._verified_physical_progress_sha256 = verified_progress_sha256
        elif verified_progress_sha256 != self._verified_physical_progress_sha256:
            self._verified_physical_progress_sha256 = verified_progress_sha256
            self._accepted_preparation_keys = []
            self._physical_call_accepted_in_episode = True
            self._pending_physical_call = None
            self._preserve_exploration_epoch_until_physical_admission = False
            self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_NORMAL
        readiness_progress_sha256 = sha256_json(_readiness_progress_material(policy_context, public_observation_or_null))
        if self._readiness_block is not None and self._readiness_progress_sha256 is not None and (readiness_progress_sha256 != self._readiness_progress_sha256):
            self._readiness_block = None
        self._readiness_progress_sha256 = readiness_progress_sha256
        material_sha256 = sha256_json(qwen_policy_loop_material(policy_context, public_observation_or_null=public_observation_or_null))
        if material_sha256 != self._material_sha256:
            if not self._preserve_exploration_epoch_until_physical_admission:
                self._accepted_preparation_keys = []
            self._material_sha256 = material_sha256
            if not self._preserve_exploration_epoch_until_physical_admission:
                self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_NORMAL
            return True
        return False

    def _sync_readiness(self, public_transition_feedback_or_null: Mapping[str, Any] | None) -> None:
        feedback = public_transition_feedback_or_null
        if isinstance(feedback, Mapping):
            nested = feedback.get('qwen_public_transition_feedback')
            if isinstance(nested, Mapping):
                feedback = nested
        call = _tool_call(feedback.get('tool_call')) or _tool_call(feedback) if isinstance(feedback, Mapping) else None
        if self._pending_physical_call is not None and call is not None and (canonical_json_bytes(call) == self._pending_physical_call):
            binding = feedback.get('binding_admission_or_null')
            status = binding.get('status') if isinstance(binding, Mapping) else None
            if status == 'success':
                self._accepted_preparation_keys = []
                self._physical_call_accepted_in_episode = True
                self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_NORMAL
                self._preserve_exploration_epoch_until_physical_admission = False
            else:
                self._preserve_exploration_epoch_until_physical_admission = True
            self._pending_physical_call = None
        elif self._pending_physical_call is not None and call is not None:
            self._pending_physical_call = None
        failure = qwen_policy_readiness_failure(public_transition_feedback_or_null)
        if failure is not None:
            call = _canonical_tool_call(failure['tool_call'])
            self._readiness_block = _physical_fingerprint(call)
            self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_NORMAL
            return
        if self._readiness_block is None or not isinstance(public_transition_feedback_or_null, Mapping):
            return
        feedback = public_transition_feedback_or_null.get('qwen_public_transition_feedback', public_transition_feedback_or_null)
        if not isinstance(feedback, Mapping):
            return
        call = _tool_call(feedback.get('tool_call')) or _tool_call(feedback)
        if call is None:
            return
        binding = feedback.get('binding_admission_or_null')
        status = binding.get('status') if isinstance(binding, Mapping) else None
        if call['tool_name'] in {'navigate_to', 'navigate_to_room'}:
            if status not in {'success', 'known'}:
                return
            target = _normalize_target(call['arguments'].get('target'))
            if target == self._readiness_block[1]:
                self._readiness_block = None
            return
        if _physical_fingerprint(call) is not None and status not in {_READINESS_STATUS, *tuple(_READINESS_REASONS)}:
            self._readiness_block = None

    def _reason(self, policy_context: Mapping[str, Any], call: Mapping[str, Any], public_observation_or_null: Mapping[str, Any] | None, public_transition_feedback_or_null: Mapping[str, Any] | None) -> tuple[str, list[str], Mapping[str, Any]] | None:
        fingerprint = _physical_fingerprint(call)
        if fingerprint is not None and fingerprint == self._readiness_block:
            return ('readiness_retry_requires_navigation', [call['tool_name']], {'readiness_guard_id': QWEN_POLICY_READINESS_GUARD_ID, 'blocked_tool_call': call, 'semantic_target': fingerprint[1]})
        key = _semantic_key(policy_context, call, public_observation_or_null=public_observation_or_null)
        if key is None:
            return None
        name, target = key
        if self._phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_PHYSICAL and name in _PREPARATION_TOOLS:
            return ('no_progress_commit_physical_requires_action', sorted(_PREPARATION_TOOLS), {'phase': self._phase, 'semantic_target': target})
        if name == 'scan' and QWEN_POLICY_LOOP_GUARD_REQUIRE_ROOM_ENTRY and (not any((item.get('tool_name') in {'navigate_to', 'navigate_to_room'} for item in _history(policy_context)))):
            return ('room_entry_required_before_scan', ['scan', 'stop'], {'semantic_target': target, 'required_tools': ['navigate_to']})
        if self._phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_NAVIGATION and name in _OBSERVATION_TOOLS:
            return ('no_progress_commit_navigation_requires_navigation_or_action', sorted(_OBSERVATION_TOOLS), {'phase': self._phase, 'semantic_target': target})
        counts = {'look_for_target': sum((1 for item in self._accepted_preparation_keys if item == ('look_for', target))), 'navigate_to_target': sum((1 for item in self._accepted_preparation_keys if item == ('navigate_to', target))), 'observation': sum((1 for item in self._accepted_preparation_keys if item[0] in _OBSERVATION_TOOLS)), 'preparation': len(self._accepted_preparation_keys), 'scan_room': sum((1 for item in self._accepted_preparation_keys if item == ('scan', target)))}
        limits = {'look_for_target': QWEN_POLICY_LOOP_GUARD_LOOK_PER_TARGET_LIMIT, 'navigate_to_target': QWEN_POLICY_LOOP_GUARD_NAVIGATE_PER_TARGET_LIMIT, 'observation': QWEN_POLICY_LOOP_GUARD_OBSERVATION_LIMIT, 'preparation': QWEN_POLICY_LOOP_GUARD_PREPARATION_LIMIT, 'scan_room': QWEN_POLICY_LOOP_GUARD_SCAN_PER_ROOM_LIMIT}
        if counts['preparation'] >= limits['preparation']:
            return ('no_progress_preparation_budget_exhausted', sorted(_PREPARATION_TOOLS), {'counts': counts, 'limits': limits, 'semantic_target': target})
        if name in _OBSERVATION_TOOLS and counts['observation'] >= limits['observation']:
            return ('no_progress_observation_budget_exhausted', sorted(_OBSERVATION_TOOLS), {'counts': counts, 'limits': limits, 'semantic_target': target})
        if name == 'scan' and sum((1 for item in self._accepted_preparation_keys if item[0] == 'scan')) >= limits['scan_room']:
            return ('no_progress_scan_room_budget_exhausted', ['scan'], {'counts': counts, 'limits': limits, 'semantic_target': target})
        if name == 'look_for' and counts['look_for_target'] >= limits['look_for_target']:
            return ('no_progress_look_for_target_budget_exhausted', [], {'counts': counts, 'limits': limits, 'semantic_target': target})
        if name == 'navigate_to' and counts['navigate_to_target'] >= limits['navigate_to_target']:
            return ('no_progress_navigate_target_budget_exhausted', [], {'counts': counts, 'limits': limits, 'semantic_target': target})
        return None

    def rejection(self, policy_context: Mapping[str, Any], candidate_tool_call: Mapping[str, Any], *, reask_index: int, public_observation_or_null: Mapping[str, Any] | None=None, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
        call = _canonical_tool_call(candidate_tool_call)
        self._sync_readiness(public_transition_feedback_or_null)
        self._sync(policy_context, public_observation_or_null)
        required = _required_recovery_call(policy_context)
        if required is not None and call == required:
            return None
        fingerprint = _physical_fingerprint(call)
        readiness_candidate = fingerprint is not None and fingerprint == self._readiness_block
        if _pending_qa(policy_context):
            return None
        if policy_context.get('mode') == 'observe' and (not readiness_candidate):
            return None
        exhausted = self._reason(policy_context, call, public_observation_or_null, public_transition_feedback_or_null)
        if exhausted is None:
            if self._phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_PHYSICAL and call['tool_name'] == 'stop' and (not self._physical_call_accepted_in_episode):
                exhausted = ('no_progress_commit_physical_requires_action', sorted(_PREPARATION_TOOLS | {'stop'}), {'phase': self._phase})
            elif self._phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_NAVIGATION and call['tool_name'] == 'stop' and (not self._physical_call_accepted_in_episode):
                exhausted = ('no_progress_commit_navigation_requires_navigation_or_action', sorted(_OBSERVATION_TOOLS | {'stop'}), {'phase': self._phase})
            else:
                return None
        reason, blocked_names, budget = exhausted
        allowed = _allowed_tool_names(policy_context)
        alternatives = [name for name in allowed if name not in blocked_names]
        if reason == 'room_entry_required_before_scan':
            self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_NAVIGATION
        if not alternatives:
            return None
        if reason.startswith('no_progress_') and self._phase == QWEN_POLICY_LOOP_GUARD_PHASE_NORMAL:
            if 'navigate_to' in alternatives:
                self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_NAVIGATION
            elif any((name not in _PREPARATION_TOOLS | {'stop'} for name in alternatives)):
                self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_PHYSICAL
            else:
                return None
        if self._phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_NAVIGATION:
            blocked_names = sorted(set(blocked_names) | _OBSERVATION_TOOLS | ({call['tool_name']} if self._physical_call_accepted_in_episode else {'stop', call['tool_name']}))
        elif self._phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_PHYSICAL:
            blocked_names = sorted(set(blocked_names) | _PREPARATION_TOOLS | (set() if self._physical_call_accepted_in_episode else {'stop'}))
        alternatives = [name for name in allowed if name not in blocked_names]
        if not alternatives:
            self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_NORMAL
            return None
        if reason == 'readiness_retry_requires_navigation' and (not any((name in _PREPARATION_TOOLS for name in alternatives))):
            return None
        payload = {'allowed_alternative_tool_names': alternatives, 'attempt': reask_index + 2, 'guidance': 'The last physical action was not interaction-ready and was not executed. Navigate to the same public target before retrying that physical action. An observation action is allowed, but do not repeat the blocked physical ToolCall.' if reason == 'readiness_retry_requires_navigation' else 'This exploration candidate exceeded the public no-progress budget and was not executed. Choose a different allowed canonical ToolCall that advances the task. Do not vary parameters to continue the same observation or navigation behavior.', 'max_reasks': QWEN_POLICY_LOOP_GUARD_MAX_REASKS, 'prohibited_candidate': call, 'public_budget': budget, 'rejected_reason': reason, 'schema_version': QWEN_POLICY_LOOP_GUARD_SCHEMA_VERSION, 'temporarily_blocked_tool_names': blocked_names}
        validate_json_value(payload, json_pointer='/exploration_guard/rejection')
        trace_row = {'attempt': payload['attempt'], 'prohibited_candidate': copy.deepcopy(payload['prohibited_candidate']), 'rejected_reason': payload['rejected_reason'], 'temporarily_blocked_tool_names': list(payload['temporarily_blocked_tool_names'])}
        validate_json_value(trace_row, json_pointer='/exploration_guard/rejection_trace')
        self._rejection_trace.append(trace_row)
        self._rejection_trace = self._rejection_trace[-(QWEN_POLICY_LOOP_GUARD_MAX_REASKS + 1):]
        self.rejection_count += 1
        return payload

    def accept(self, policy_context: Mapping[str, Any], candidate_tool_call: Mapping[str, Any], *, public_observation_or_null: Mapping[str, Any] | None=None, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> None:
        call = _canonical_tool_call(candidate_tool_call)
        self._sync_readiness(public_transition_feedback_or_null)
        self._sync(policy_context, public_observation_or_null)
        if qwen_policy_loop_guard_bypassed(policy_context):
            return
        required = _required_recovery_call(policy_context)
        if required is not None and call == required:
            if call['tool_name'] not in _NON_PHYSICAL_TOOLS:
                self._pending_physical_call = canonical_json_bytes(call)
            return
        key = _semantic_key(policy_context, call, public_observation_or_null=public_observation_or_null)
        if key is not None:
            self._accepted_preparation_keys.append(key)
            self._accepted_preparation_keys = self._accepted_preparation_keys[-128:]
            if self._phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_NAVIGATION and call['tool_name'] in {'navigate_to', 'navigate_to_room'}:
                self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_PHYSICAL
        elif self._phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_NAVIGATION and call['tool_name'] not in _NON_PHYSICAL_TOOLS:
            self._phase = QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_PHYSICAL
            self._pending_physical_call = canonical_json_bytes(call)
        elif call['tool_name'] not in _NON_PHYSICAL_TOOLS:
            self._pending_physical_call = canonical_json_bytes(call)
        if call['tool_name'] not in _NON_PHYSICAL_TOOLS:
            self._pending_physical_call = canonical_json_bytes(call)

def qwen_policy_loop_guard_message_bytes(rejection: Mapping[str, Any]) -> bytes:
    if not isinstance(rejection, Mapping) or set(rejection) != {'allowed_alternative_tool_names', 'attempt', 'guidance', 'max_reasks', 'prohibited_candidate', 'public_budget', 'rejected_reason', 'schema_version', 'temporarily_blocked_tool_names'}:
        raise ValueError('Qwen exploration guard rejection fields differ')
    if rejection.get('schema_version') != QWEN_POLICY_LOOP_GUARD_SCHEMA_VERSION:
        raise ValueError('Qwen exploration guard rejection schema differs')
    validate_json_value(rejection, json_pointer='/exploration_guard/rejection')
    return canonical_json_bytes(rejection)

def qwen_policy_loop_guard_retry_user_bytes(policy_context: Mapping[str, Any], rejection: Mapping[str, Any], *, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes:
    if not isinstance(policy_context, Mapping):
        raise ValueError('Qwen loop-guard retry context must be an object')
    allowed_names = rejection.get('allowed_alternative_tool_names')
    if not isinstance(allowed_names, list) or any((not isinstance(item, str) or not item for item in allowed_names)):
        raise ValueError('Qwen loop-guard retry alternatives are invalid')
    allowed = set(allowed_names)
    specs = policy_context.get('allowed_tool_specs')
    if not isinstance(specs, list):
        raise ValueError('Qwen loop-guard retry tool specs are missing')
    filtered_specs = [copy.deepcopy(dict(raw)) for raw in specs if isinstance(raw, Mapping) and raw.get('name') in allowed]
    filtered_specs.sort(key=lambda item: item['name'].encode('utf-8'))
    if {item.get('name') for item in filtered_specs} != allowed:
        raise ValueError('Qwen loop-guard retry tool specs are incomplete')
    instruction = policy_context.get('instruction')
    mode = policy_context.get('mode')
    memory = policy_context.get('evidence_memory')
    if not isinstance(instruction, str) or not instruction:
        raise ValueError('Qwen loop-guard retry instruction is invalid')
    if mode not in {'step', 'recovery', 'observe'}:
        raise ValueError('Qwen loop-guard retry mode is invalid')
    if not isinstance(memory, Mapping):
        raise ValueError('Qwen loop-guard retry public state is invalid')
    observation = _observation(policy_context, public_observation_or_null=public_observation_or_null)
    incidents = [item for item in _memory_rows(memory.get('incidents', [])) if item.get('status') == 'open']
    obligations = [item for item in _memory_rows(memory.get('pending_obligations', [])) if item.get('status', 'open') == 'open']
    compact = {'allowed_tool_specs': filtered_specs, 'attempt': rejection.get('attempt'), 'held_object_handles': _stable_public_value(observation.get('held_object_handles', [])), 'held_object_types': _stable_public_value(observation.get('held_object_types', [])), 'instruction': instruction, 'mode': mode, 'open_incidents': incidents, 'open_obligations': obligations, 'schema_version': QWEN_POLICY_LOOP_GUARD_RETRY_SCHEMA_VERSION}
    reason = rejection.get('rejected_reason')
    validate_json_value(compact, json_pointer='/exploration_guard/retry')
    phase = rejection.get('public_budget', {}).get('phase')
    max_reasks = rejection.get('max_reasks')
    if not isinstance(max_reasks, int) or isinstance(max_reasks, bool):
        raise ValueError('Qwen loop-guard retry limit is invalid')
    final_retry = rejection.get('attempt') == max_reasks + 1
    filtered_names = [item['name'] for item in filtered_specs]
    if reason == 'readiness_retry_requires_navigation':
        directive_allowed_names = [name for name in filtered_names if name in _PREPARATION_TOOLS]
    elif phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_PHYSICAL:
        directive_allowed_names = [name for name in filtered_names if name not in _NON_PHYSICAL_TOOLS or name == 'stop']
    else:
        directive_allowed_names = [name for name in filtered_names if name in {'navigate_to', 'navigate_to_room'} or name not in _NON_PHYSICAL_TOOLS or name == 'stop']
    if not directive_allowed_names:
        raise ValueError('Qwen loop-guard retry has no directive-compatible tool')
    allowed_names_bytes = canonical_json_bytes(directive_allowed_names)
    allowed_suffix = b' Allowed tool_name values are ' + allowed_names_bytes + b'.'
    if reason == 'readiness_retry_requires_navigation':
        blocked = rejection.get('prohibited_candidate', {})
        arguments = blocked.get('arguments', {}) if isinstance(blocked, Mapping) else {}
        target = arguments.get('target') if isinstance(arguments, Mapping) else None
        target_bytes = canonical_json_bytes(target if isinstance(target, str) and target else 'the same public target')
        directive = (b'Final retry: output navigate_to for target ' + target_bytes + b' now. Do not repeat the rejected physical ToolCall.' if final_retry else b'The last physical action was not interaction-ready. Output navigate_to for target ' + target_bytes + b' or one allowed observation ToolCall. Do not retry the blocked physical ToolCall yet.') + allowed_suffix
    elif phase == QWEN_POLICY_LOOP_GUARD_PHASE_COMMIT_PHYSICAL:
        directive = (b'Final retry: output exactly one allowed physical ToolCall now; do not repeat the rejected ToolCall.' if final_retry else b'Exploration is exhausted. Output one allowed physical ToolCall now.') + allowed_suffix
    else:
        directive = (b'Final retry: output exactly one allowed navigation or physical ToolCall now; do not repeat the rejected ToolCall.' if final_retry else b'Exploration is exhausted. Output one allowed navigation or physical ToolCall now.') + allowed_suffix
    return canonical_json_bytes(compact) + b'\nQWEN_COMMITMENT=' + directive

def append_qwen_policy_loop_guard_diagnostics_from_env(guard: ExplorationLoopGuard) -> bool:
    path = os.environ.get(QWEN_POLICY_LOOP_GUARD_DIAGNOSTICS_JSONL_ENV)
    if not path:
        return False
    try:
        payload = {'guard_id': QWEN_POLICY_LOOP_GUARD_ID, 'guard_phase': guard.phase, 'rejection_trace': [copy.deepcopy(dict(item)) for item in guard.rejection_trace], 'schema_version': 'qwen-policy-loop-guard-diagnostics.v1'}
        validate_json_value(payload, json_pointer='/exploration_guard_diagnostics')
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 384)
        try:
            os.write(descriptor, canonical_json_bytes(payload) + b'\n')
        finally:
            os.close(descriptor)
    except (OSError, TypeError, ValueError):
        return False
    return True

def qwen_policy_loop_guard_user_bytes(canonical_policy_context_bytes: bytes, policy_context: Mapping[str, Any], *, context_footer_enabled: bool, experiment_profile: str | None=None, public_transition_feedback_or_null: Mapping[str, Any] | None=None, policy_decision_check_enabled: bool=False) -> bytes:
    if not isinstance(canonical_policy_context_bytes, bytes) or canonical_policy_context_bytes != canonical_json_bytes(policy_context):
        raise ValueError('Qwen loop-guard context bytes are not exact canonical JSON')
    if experiment_profile is not None:
        from scoper.harness.prompt_contracts import qwen35_policy_experiment_user_bytes
        return qwen35_policy_experiment_user_bytes(canonical_policy_context_bytes, policy_context, profile=experiment_profile, public_transition_feedback_or_null=public_transition_feedback_or_null)
    if context_footer_enabled:
        from scoper.harness.prompt_contracts import qwen35_policy_user_bytes
        return qwen35_policy_user_bytes(canonical_policy_context_bytes, policy_context)
    if policy_decision_check_enabled:
        from scoper.harness.prompt_contracts import policy_decision_check_user_bytes
        return policy_decision_check_user_bytes(canonical_policy_context_bytes)
    return canonical_policy_context_bytes
__all__ = ['QWEN_POLICY_LOOP_GUARD_ENV', 'QWEN_POLICY_LOOP_GUARD_DIAGNOSTICS_JSONL_ENV', 'QWEN_POLICY_LOOP_GUARD_ID', 'QWEN_POLICY_LOOP_GUARD_LOOK_PER_TARGET_LIMIT', 'QWEN_POLICY_LOOP_GUARD_MAX_REASKS', 'QWEN_POLICY_LOOP_GUARD_NAVIGATE_PER_TARGET_LIMIT', 'QWEN_POLICY_LOOP_GUARD_OBSERVATION_LIMIT', 'QWEN_POLICY_LOOP_GUARD_PREPARATION_LIMIT', 'QWEN_POLICY_LOOP_GUARD_SCAN_PER_ROOM_LIMIT', 'QWEN_POLICY_LOOP_GUARD_SCHEMA_VERSION', 'QWEN_POLICY_LOOP_GUARD_RETRY_SCHEMA_VERSION', 'QWEN_POLICY_READINESS_GUARD_ID', 'QWEN_POLICY_VERIFIED_HELD_GUARD_ID', 'QWEN_POLICY_VERIFIED_HELD_MAX_REASKS', 'ExplorationLoopGuard', 'append_qwen_policy_loop_guard_diagnostics_from_env', 'qwen_policy_loop_guard_bypassed', 'qwen_policy_loop_material', 'qwen_policy_loop_guard_message_bytes', 'qwen_policy_loop_guard_retry_user_bytes', 'qwen_policy_loop_guard_user_bytes', 'qwen_policy_readiness_failure', 'qwen_policy_verified_held_refs']
