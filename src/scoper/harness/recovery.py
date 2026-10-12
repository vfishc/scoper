from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
from scoper.harness.canonical import sha256_json
from scoper.harness.contracts import MethodContractError
from scoper.harness.validation import exact_mapping, finite_number, integer, non_empty_string, optional_string, public_json, sha256, string_sequence
from scoper.harness.evidence_memory import EvidenceMemory, PendingObligation, RecoveryIncident, enforce_capacity, MemoryCapacity
TOOL_SPEC_FIELDS = frozenset({'name', 'description', 'parameters', 'arguments_schema'})
JSON_SCHEMA_FIELDS = frozenset({'type', 'properties', 'required', 'enum', 'items', 'description', 'additionalProperties', 'minimum', 'maximum'})
JSON_SCHEMA_TYPES = frozenset({'array', 'boolean', 'integer', 'number', 'object', 'string'})

@dataclass(frozen=True)
class RecoveryBudgets:
    max_recovery_attempts_per_incident: int = 2
    max_total_recovery_actions: int = 5
    max_observe_actions_per_ambiguous: int = 2

    def __post_init__(self) -> None:
        integer(self.max_recovery_attempts_per_incident, pointer='/recovery/max_recovery_attempts_per_incident', minimum=1)
        integer(self.max_total_recovery_actions, pointer='/recovery/max_total_recovery_actions', minimum=1)
        integer(self.max_observe_actions_per_ambiguous, pointer='/recovery/max_observe_actions_per_ambiguous', minimum=1)

@dataclass(frozen=True)
class RecoveryLedger:
    total_recovery_actions: int = 0
    incident_attempts: tuple[tuple[str, int], ...] = ()
    obligation_observe_actions: tuple[tuple[str, int], ...] = ()
    recorded_actions: tuple[tuple[str, str], ...] = ()
    incident_observation_actions: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        integer(self.total_recovery_actions, pointer='/recovery_ledger/total_recovery_actions')
        for pointer, values in (('/recovery_ledger/incident_attempts', self.incident_attempts), ('/recovery_ledger/obligation_observe_actions', self.obligation_observe_actions), ('/recovery_ledger/incident_observation_actions', self.incident_observation_actions)):
            identifiers = []
            for identifier, count in values:
                identifiers.append(non_empty_string(identifier, pointer=pointer))
                integer(count, pointer=pointer)
            if identifiers != sorted(identifiers, key=lambda item: item.encode('utf-8')) or len(set(identifiers)) != len(identifiers):
                raise MethodContractError('method_schema_invalid', 'recovery ledger is not canonical')
        action_ids = []
        for action_id, fingerprint in self.recorded_actions:
            action_ids.append(non_empty_string(action_id, pointer='/recovery_ledger/recorded_actions'))
            sha256(fingerprint, pointer='/recovery_ledger/recorded_actions/fingerprint')
        if action_ids != sorted(action_ids, key=lambda item: item.encode('utf-8')) or len(set(action_ids)) != len(action_ids):
            raise MethodContractError('method_schema_invalid', 'recorded recovery actions are not canonical')

    def incident_count(self, incident_id: str) -> int:
        return dict(self.incident_attempts).get(incident_id, 0)

    def observe_count(self, obligation_id: str) -> int:
        return dict(self.obligation_observe_actions).get(obligation_id, 0)

    def incident_observation_count(self, incident_id: str) -> int:
        return dict(self.incident_observation_actions).get(incident_id, 0)

@dataclass(frozen=True)
class RecoveryContext:
    incident_id: str
    failed_tool_call: Mapping[str, Any]
    failure_kind: str
    canonical_action: str
    required_effects: tuple[str, ...]
    retry_index: int = 0
    max_retries: int = 2
    phase: str = 'retry'
    evidence_frame_ids: tuple[str, ...] = ()
    preparation_attempts: int = 0
    recovery_round: int = 0
    dispatched_attempts: int = 0
    last_preparation_result_or_null: Mapping[str, Any] | None = None
    effect_bindings: tuple[Mapping[str, Any], ...] = ()
    binding_failure_reason_or_null: str | None = None
    uniquely_resolved_public_target: bool = False
    public_failure_reason_or_null: str | None = None
    open_prerequisite_attempted: bool = False

    def __post_init__(self) -> None:
        non_empty_string(self.incident_id, pointer='/recovery_context/incident_id')
        call = exact_mapping(self.failed_tool_call, frozenset({'tool_name', 'arguments'}), pointer='/recovery_context/failed_tool_call')
        non_empty_string(call['tool_name'], pointer='/recovery_context/failed_tool_call/tool_name')
        if not isinstance(call['arguments'], Mapping):
            raise MethodContractError('method_schema_invalid', 'recovery failed ToolCall arguments must be an object')
        public_json(call, pointer='/recovery_context/failed_tool_call', include_future_action=False)
        if self.failure_kind not in {'failure', 'ambiguous', 'admission'}:
            raise MethodContractError('method_schema_invalid', 'recovery failure kind is invalid')
        non_empty_string(self.canonical_action, pointer='/recovery_context/canonical_action')
        string_sequence(list(self.required_effects), pointer='/recovery_context/required_effects', sorted_utf8=True)
        integer(self.retry_index, pointer='/recovery_context/retry_index', minimum=0)
        integer(self.max_retries, pointer='/recovery_context/max_retries', minimum=1)
        if self.retry_index > self.max_retries:
            raise MethodContractError('method_schema_invalid', 'recovery retry index exceeds maximum')
        if self.phase not in {'look', 'navigate', 'open_prerequisite', 'retry', 'final_retry', 'exhausted'}:
            raise MethodContractError('method_schema_invalid', 'recovery phase is invalid')
        string_sequence(list(self.evidence_frame_ids), pointer='/recovery_context/evidence_frame_ids')
        integer(self.preparation_attempts, pointer='/recovery_context/preparation_attempts', minimum=0)
        integer(self.recovery_round, pointer='/recovery_context/recovery_round', minimum=0)
        integer(self.dispatched_attempts, pointer='/recovery_context/dispatched_attempts', minimum=0)
        if self.dispatched_attempts != self.retry_index:
            raise MethodContractError('method_schema_invalid', 'recovery dispatched attempts must equal retry index')
        if self.last_preparation_result_or_null is not None:
            _validate_preparation_result(self.last_preparation_result_or_null)
        for index, binding in enumerate(self.effect_bindings):
            if not isinstance(binding, Mapping):
                raise MethodContractError('method_schema_invalid', 'recovery effect binding must be an object')
            public_json(binding, pointer=f'/recovery_context/effect_bindings/{index}', include_future_action=False)
        optional_string(self.binding_failure_reason_or_null, pointer='/recovery_context/binding_failure_reason_or_null')
        if not isinstance(self.uniquely_resolved_public_target, bool):
            raise MethodContractError('method_schema_invalid', 'unique public target flag must be boolean')
        optional_string(self.public_failure_reason_or_null, pointer='/recovery_context/public_failure_reason_or_null')
        if not isinstance(self.open_prerequisite_attempted, bool):
            raise MethodContractError('method_schema_invalid', 'open prerequisite attempt flag must be boolean')

    def to_mapping(self) -> dict[str, Any]:
        output = {'incident_id': self.incident_id, 'failed_tool_call': {'tool_name': self.failed_tool_call['tool_name'], 'arguments': dict(self.failed_tool_call['arguments'])}, 'failure_kind': self.failure_kind, 'canonical_action': self.canonical_action, 'required_effects': list(self.required_effects), 'retry_index': self.retry_index, 'max_retries': self.max_retries, 'phase': self.phase, 'evidence_frame_ids': list(self.evidence_frame_ids), 'preparation_attempts': self.preparation_attempts, 'recovery_round': self.recovery_round, 'dispatched_attempts': self.dispatched_attempts, 'last_preparation_result_or_null': None if self.last_preparation_result_or_null is None else dict(self.last_preparation_result_or_null), 'binding_failure_reason_or_null': self.binding_failure_reason_or_null, 'uniquely_resolved_public_target': self.uniquely_resolved_public_target}
        if self.public_failure_reason_or_null is not None:
            output['public_failure_reason_or_null'] = self.public_failure_reason_or_null
        return output
PREPARATION_RESULT_STATUSES = ('found', 'not_found', 'motion_failed', 'binding_failed', 'navigation_ready', 'navigation_not_ready', 'navigation_failed', 'scan_completed')

def _validate_preparation_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = exact_mapping(value, frozenset({'tool_name', 'status', 'interaction_ready', 'frame_id_or_null'}), pointer='/preparation_result')
    tool_name = non_empty_string(result['tool_name'], pointer='/preparation_result/tool_name')
    if tool_name not in {'look_for', 'navigate_to', 'scan'}:
        raise MethodContractError('method_schema_invalid', 'preparation tool is invalid')
    status = non_empty_string(result['status'], pointer='/preparation_result/status')
    if status not in PREPARATION_RESULT_STATUSES:
        raise MethodContractError('method_schema_invalid', 'preparation status is invalid')
    if not isinstance(result['interaction_ready'], bool):
        raise MethodContractError('method_schema_invalid', 'preparation interaction_ready must be boolean')
    frame_id = optional_string(result['frame_id_or_null'], pointer='/preparation_result/frame_id_or_null')
    output = {'tool_name': tool_name, 'status': status, 'interaction_ready': result['interaction_ready'], 'frame_id_or_null': frame_id}
    public_json(output, pointer='/preparation_result', include_future_action=False)
    return output

@dataclass(frozen=True)
class RecoveryPlan:
    mode: str
    reason: str
    incident_id_or_null: str | None
    obligation_id_or_null: str | None
    allowed_tool_names: tuple[str, ...]
    required_tool_call_or_null: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {'step', 'recovery', 'observe', 'promote_to_incident', 'agent_failure'}:
            raise MethodContractError('method_schema_invalid', 'recovery plan mode is invalid')
        non_empty_string(self.reason, pointer='/recovery_plan/reason')
        incident_id = optional_string(self.incident_id_or_null, pointer='/recovery_plan/incident_id_or_null')
        obligation_id = optional_string(self.obligation_id_or_null, pointer='/recovery_plan/obligation_id_or_null')
        tools = string_sequence(list(self.allowed_tool_names), pointer='/recovery_plan/allowed_tool_names', sorted_utf8=True)
        if tools != self.allowed_tool_names:
            raise MethodContractError('method_schema_invalid', 'recovery plan tools must be a tuple')
        if self.mode == 'step' and (incident_id is not None or obligation_id is not None or (not tools)):
            raise MethodContractError('method_schema_invalid', 'step recovery plan is inconsistent')
        if self.mode == 'recovery' and (incident_id is None or obligation_id is not None or (not tools) or ('stop' in tools)):
            raise MethodContractError('method_schema_invalid', 'recovery plan is inconsistent')
        if self.mode == 'observe' and (incident_id is not None or obligation_id is None or (not tools) or ('stop' in tools)):
            raise MethodContractError('method_schema_invalid', 'observe recovery plan is inconsistent')
        if self.mode == 'promote_to_incident' and (incident_id is not None or obligation_id is None or tools):
            raise MethodContractError('method_schema_invalid', 'promotion recovery plan is inconsistent')
        if self.mode == 'agent_failure' and tools:
            raise MethodContractError('method_schema_invalid', 'agent-failure plan cannot allow tools')
        required_call = self.required_tool_call_or_null
        if required_call is not None:
            call = exact_mapping(required_call, frozenset({'tool_name', 'arguments'}), pointer='/recovery_plan/required_tool_call_or_null')
            required_tool = non_empty_string(call['tool_name'], pointer='/recovery_plan/required_tool_call_or_null/tool_name')
            if not isinstance(call['arguments'], Mapping):
                raise MethodContractError('method_schema_invalid', 'required recovery ToolCall arguments must be an object')
            public_json(call, pointer='/recovery_plan/required_tool_call_or_null', include_future_action=False)
            if self.mode != 'recovery' or required_tool not in tools:
                raise MethodContractError('method_schema_invalid', 'required recovery ToolCall differs from the recovery plan')

def plan_recovery(memory: EvidenceMemory, ledger: RecoveryLedger, public_tool_specs: Sequence[Mapping[str, Any]], *, budgets: RecoveryBudgets=RecoveryBudgets(), observation_tool_names: Sequence[str]=('look_for', 'scan'), suppress_repeated_incident_observation: bool=True, prioritize_oldest_incident: bool=True) -> RecoveryPlan:
    memory = memory.validated()
    tools = _validate_public_tool_specs(public_tool_specs)
    observation_tools = string_sequence(list(observation_tool_names), pointer='/observation_tool_names', sorted_utf8=True)
    if not isinstance(suppress_repeated_incident_observation, bool):
        raise MethodContractError('method_schema_invalid', 'incident observation suppression flag must be boolean')
    if not isinstance(prioritize_oldest_incident, bool):
        raise MethodContractError('method_schema_invalid', 'incident priority flag must be boolean')
    open_incidents = sorted((incident for incident in memory.incidents if incident.status == 'open'), key=lambda incident: (incident.created_step if prioritize_oldest_incident else -incident.created_step, incident.incident_id.encode('utf-8')))
    open_obligations = sorted((obligation for obligation in memory.pending_obligations if obligation.status == 'open'), key=lambda obligation: (obligation.created_step, obligation.obligation_id.encode('utf-8')))
    if not open_incidents and (not open_obligations):
        return RecoveryPlan('step', 'no_recovery_required', None, None, tools)
    if ledger.total_recovery_actions >= budgets.max_total_recovery_actions:
        return RecoveryPlan('agent_failure', 'recovery_budget_exceeded', None, None, ())
    if open_incidents:
        incident = open_incidents[0]
        attempts = max(incident.recovery_attempts, ledger.incident_count(incident.incident_id))
        if attempts >= budgets.max_recovery_attempts_per_incident:
            return RecoveryPlan('agent_failure', 'recovery_budget_exceeded', incident.incident_id, None, ())
        if incident.failure_type == 'public_interaction_not_ready':
            allowed = tuple((tool for tool in tools if tool == 'navigate_to'))
            if not allowed:
                return RecoveryPlan('agent_failure', 'recovery_navigation_tool_unavailable', incident.incident_id, None, ())
            return RecoveryPlan('recovery', 'interaction_readiness_requires_navigation', incident.incident_id, None, allowed)
        observation_already_used = suppress_repeated_incident_observation and ledger.incident_observation_count(incident.incident_id) > 0
        allowed = tuple((tool for tool in tools if tool != 'stop' and (not (observation_already_used and tool in observation_tools))))
        if not allowed:
            return RecoveryPlan('agent_failure', 'recovery_non_observation_tool_unavailable' if observation_already_used else 'recovery_public_tool_unavailable', incident.incident_id, None, ())
        return RecoveryPlan('recovery', 'unresolved_incident_after_observation' if observation_already_used else 'unresolved_incident', incident.incident_id, None, allowed)
    obligation = open_obligations[0]
    if obligation.type == 'observe':
        if ledger.observe_count(obligation.obligation_id) >= budgets.max_observe_actions_per_ambiguous:
            return RecoveryPlan('promote_to_incident', 'observe_budget_exhausted', None, obligation.obligation_id, ())
        allowed = tuple((tool for tool in observation_tools if tool in tools))
        if not allowed:
            return RecoveryPlan('agent_failure', 'recovery_observation_tool_unavailable', None, obligation.obligation_id, ())
        return RecoveryPlan('observe', 'pending_observe_obligation', None, obligation.obligation_id, allowed)
    return RecoveryPlan('agent_failure', 'act_obligation_without_open_incident', None, obligation.obligation_id, ())

def plan_recovery_v13(memory: EvidenceMemory, ledger: RecoveryLedger, public_tool_specs: Sequence[Mapping[str, Any]], *, contexts: Mapping[str, RecoveryContext], preparation_tool_names: Sequence[str]=('look_for', 'navigate_to', 'scan'), max_retries: int=2) -> RecoveryPlan:
    memory = memory.validated()
    tools = _validate_public_tool_specs(public_tool_specs)
    integer(max_retries, pointer='/recovery/max_retries', minimum=1)
    open_incidents = sorted((incident for incident in memory.incidents if incident.status == 'open'), key=lambda item: (item.created_step, item.incident_id.encode('utf-8')))
    if not open_incidents:
        return RecoveryPlan('step', 'no_recovery_required', None, None, tools)
    incident = open_incidents[0]
    context = contexts.get(incident.incident_id)
    if context is None:
        return RecoveryPlan('recovery', 'recovery_context_missing', incident.incident_id, None, tuple((tool for tool in tools if tool != 'stop')))
    if context.retry_index >= max_retries or context.phase == 'exhausted':
        return RecoveryPlan('step', 'recovery_assumed_continue', None, None, tools)
    failed = context.failed_tool_call
    tool_name = failed['tool_name']
    target = failed['arguments'].get('target')
    if context.phase == 'open_prerequisite' and context.public_failure_reason_or_null == 'container_closed' and (not context.open_prerequisite_attempted) and (tool_name == 'put_down') and isinstance(target, str) and target and ('open' in tools):
        return RecoveryPlan('recovery', 'open_closed_container_before_retry', incident.incident_id, None, ('open',), required_tool_call_or_null={'tool_name': 'open', 'arguments': {'target': target}})
    if context.phase == 'look':
        configured = set(preparation_tool_names)
        if isinstance(target, str) and target and ('look_for' in configured) and ('look_for' in tools):
            return RecoveryPlan('recovery', 'look_for_before_original_retry', incident.incident_id, None, ('look_for',), required_tool_call_or_null={'tool_name': 'look_for', 'arguments': {'target': target}})
        if 'scan' in configured and 'scan' in tools:
            return RecoveryPlan('recovery', 'scan_before_original_retry', incident.incident_id, None, ('scan',), required_tool_call_or_null={'tool_name': 'scan', 'arguments': {}})
        return RecoveryPlan('recovery', 'recovery_preparation_tool_unavailable', incident.incident_id, None, tuple((tool for tool in tools if tool != 'stop')))
    if context.phase == 'navigate':
        target = failed['arguments'].get('target')
        if isinstance(target, str) and target and ('navigate_to' in tools):
            return RecoveryPlan('recovery', 'navigate_before_original_retry', incident.incident_id, None, ('navigate_to',), required_tool_call_or_null={'tool_name': 'navigate_to', 'arguments': {'target': target}})
        return RecoveryPlan('recovery', 'recovery_navigation_tool_unavailable', incident.incident_id, None, tuple((tool for tool in tools if tool != 'stop')))
    if tool_name not in tools:
        return RecoveryPlan('recovery', 'original_retry_tool_unavailable', incident.incident_id, None, tuple((tool for tool in tools if tool != 'stop')))
    return RecoveryPlan('recovery', 'final_locked_original_attempt' if context.phase == 'final_retry' else 'retry_original_action', incident.incident_id, None, (tool_name,), required_tool_call_or_null=failed)

def record_recovery_action_v13(ledger: RecoveryLedger, plan: RecoveryPlan, tool_call: Mapping[str, Any], *, action_id: str, public_tool_specs: Sequence[Mapping[str, Any]], allowed_public_entity_refs: Sequence[str], public_precondition_satisfied: bool) -> RecoveryLedger:
    validate_recovery_tool_call(tool_call, plan, public_tool_specs=public_tool_specs, allowed_public_entity_refs=allowed_public_entity_refs, public_precondition_satisfied=public_precondition_satisfied)
    non_empty_string(action_id, pointer='/recovery_action_id')
    return ledger

def _validate_public_tool_specs(values: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise MethodContractError('method_schema_invalid', 'public tool specs must be a non-empty list')
    names = []
    for index, spec in enumerate(values):
        if not isinstance(spec, Mapping):
            raise MethodContractError('method_schema_invalid', 'public tool spec must be an object')
        if not set(spec).issubset(TOOL_SPEC_FIELDS):
            raise MethodContractError('method_schema_invalid', 'public tool spec has unknown fields')
        schema_fields = [field for field in ('parameters', 'arguments_schema') if field in spec]
        if len(schema_fields) != 1:
            raise MethodContractError('method_schema_invalid', 'public tool spec must have exactly one argument schema')
        name = non_empty_string(spec.get('name'), pointer='/public_tool_specs/{}/name'.format(index))
        if 'description' in spec:
            non_empty_string(spec['description'], pointer='/public_tool_specs/{}/description'.format(index))
        parameters = spec[schema_fields[0]]
        _validate_json_schema(parameters, pointer='/public_tool_specs/{}/{}'.format(index, schema_fields[0]), require_closed_object=True)
        public_json(spec, pointer='/public_tool_specs/{}'.format(index), include_future_action=True)
        names.append(name)
    if names != sorted(names, key=lambda item: item.encode('utf-8')) or len(set(names)) != len(names):
        raise MethodContractError('method_schema_invalid', 'public tool specs are not canonical')
    return tuple(names)

def _validate_json_schema(value: Any, *, pointer: str, require_closed_object: bool=False) -> None:
    if not isinstance(value, Mapping) or not set(value).issubset(JSON_SCHEMA_FIELDS):
        raise MethodContractError('method_schema_invalid', 'public tool argument schema is invalid')
    expected_type = value.get('type')
    if expected_type not in JSON_SCHEMA_TYPES:
        raise MethodContractError('method_schema_invalid', 'public tool argument type is invalid')
    if 'description' in value:
        non_empty_string(value['description'], pointer='{}/description'.format(pointer))
    if 'enum' in value:
        enum = value['enum']
        if not isinstance(enum, list) or not enum:
            raise MethodContractError('method_schema_invalid', 'public tool enum is invalid')
        public_json(enum, pointer='{}/enum'.format(pointer), include_future_action=True)
        encoded = [sha256_json(item) for item in enum]
        if len(set(encoded)) != len(encoded):
            raise MethodContractError('method_schema_invalid', 'public tool enum is duplicated')
    numeric_bounds = {}
    for field in ('minimum', 'maximum'):
        if field in value:
            if expected_type not in {'integer', 'number'}:
                raise MethodContractError('method_schema_invalid', 'numeric bound has non-numeric type')
            numeric_bounds[field] = finite_number(value[field], pointer='{}/{}'.format(pointer, field))
    if numeric_bounds.get('minimum', float('-inf')) > numeric_bounds.get('maximum', float('inf')):
        raise MethodContractError('method_schema_invalid', 'public tool numeric bounds are inverted')
    object_fields = {'properties', 'required', 'additionalProperties'}
    if expected_type == 'object':
        properties = value.get('properties')
        required = value.get('required')
        additional = value.get('additionalProperties')
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise MethodContractError('method_schema_invalid', 'object argument schema is incomplete')
        if not isinstance(additional, bool) or (require_closed_object and additional):
            raise MethodContractError('method_schema_invalid', 'object argument schema is not closed')
        if any((not isinstance(key, str) or not key for key in properties)):
            raise MethodContractError('method_schema_invalid', 'object property name is invalid')
        required_fields = string_sequence(required, pointer='{}/required'.format(pointer))
        if any((field not in properties for field in required_fields)):
            raise MethodContractError('method_schema_invalid', 'required argument lacks a property schema')
        for name, schema in properties.items():
            _validate_json_schema(schema, pointer='{}/properties/{}'.format(pointer, name))
    elif any((field in value for field in object_fields)):
        raise MethodContractError('method_schema_invalid', 'non-object schema has object fields')
    if expected_type == 'array':
        if 'items' not in value:
            raise MethodContractError('method_schema_invalid', 'array argument schema lacks items')
        _validate_json_schema(value['items'], pointer='{}/items'.format(pointer))
    elif 'items' in value:
        raise MethodContractError('method_schema_invalid', 'non-array schema has items')

def _argument_matches_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    expected_type = schema.get('type')
    if expected_type == 'string' and (not isinstance(value, str)):
        return False
    if expected_type == 'boolean' and (not isinstance(value, bool)):
        return False
    if expected_type == 'integer' and (isinstance(value, bool) or not isinstance(value, int)):
        return False
    if expected_type == 'number' and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return False
    if expected_type == 'number' and (not isinstance(value, bool)):
        try:
            finite_number(value, pointer='/tool_call/arguments')
        except MethodContractError:
            return False
    if expected_type == 'array':
        if not isinstance(value, list):
            return False
        item_schema = schema.get('items')
        if isinstance(item_schema, Mapping) and any((not _argument_matches_schema(item, item_schema) for item in value)):
            return False
    if expected_type == 'object':
        if not isinstance(value, Mapping):
            return False
        try:
            _validate_arguments(value, schema)
        except MethodContractError:
            return False
    if 'enum' in schema and value not in schema['enum']:
        return False
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if 'minimum' in schema and value < schema['minimum']:
            return False
        if 'maximum' in schema and value > schema['maximum']:
            return False
    return True

def _validate_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    if any((not isinstance(key, str) for key in arguments)):
        raise MethodContractError('method_schema_invalid', 'ToolCall argument names must be strings')
    properties = schema['properties']
    required = schema['required']
    if any((key not in arguments for key in required)):
        raise MethodContractError('method_recovery_tool_forbidden', 'required tool argument is missing')
    if schema.get('additionalProperties') is False and any((key not in properties for key in arguments)):
        raise MethodContractError('method_recovery_tool_forbidden', 'unknown tool argument is forbidden')
    for key, argument in arguments.items():
        argument_schema = properties.get(key)
        if isinstance(argument_schema, Mapping) and (not _argument_matches_schema(argument, argument_schema)):
            raise MethodContractError('method_recovery_tool_forbidden', 'tool argument differs from public schema')

def validate_recovery_tool_call(value: Mapping[str, Any], plan: RecoveryPlan, *, public_tool_specs: Sequence[Mapping[str, Any]], allowed_public_entity_refs: Sequence[str], public_precondition_satisfied: bool, public_entity_argument_names: Sequence[str]=('object', 'receptacle', 'source', 'target')) -> Mapping[str, Any]:
    tool_call = exact_mapping(value, frozenset({'tool_name', 'arguments'}), pointer='/tool_call')
    tool_name = non_empty_string(tool_call['tool_name'], pointer='/tool_call/tool_name')
    if tool_name not in plan.allowed_tool_names:
        raise MethodContractError('method_recovery_tool_forbidden', 'recovery ToolCall bypasses public schema/budget')
    if not isinstance(tool_call['arguments'], Mapping):
        raise MethodContractError('method_schema_invalid', 'ToolCall arguments must be an object')
    _validate_public_tool_specs(public_tool_specs)
    spec = next((item for item in public_tool_specs if item.get('name') == tool_name), None)
    if spec is None:
        raise MethodContractError('method_recovery_tool_forbidden', 'tool is absent from public schema')
    schema = spec.get('parameters', spec.get('arguments_schema'))
    _validate_arguments(tool_call['arguments'], schema)
    string_sequence(list(allowed_public_entity_refs), pointer='/allowed_public_entity_refs', sorted_utf8=True)
    string_sequence(list(public_entity_argument_names), pointer='/public_entity_argument_names', sorted_utf8=True)
    if not isinstance(public_precondition_satisfied, bool):
        raise MethodContractError('method_schema_invalid', 'public precondition result must be boolean')
    if not public_precondition_satisfied:
        raise MethodContractError('method_recovery_precondition_failed', 'public recovery precondition is not satisfied')
    if plan.required_tool_call_or_null is not None and sha256_json(tool_call) != sha256_json(plan.required_tool_call_or_null):
        raise MethodContractError('method_recovery_tool_forbidden', 'recovery ToolCall differs from the required public retry')
    public_json(tool_call, pointer='/tool_call', include_future_action=False)
    return tool_call

def record_recovery_action(ledger: RecoveryLedger, plan: RecoveryPlan, tool_call: Mapping[str, Any], *, action_id: str, public_tool_specs: Sequence[Mapping[str, Any]], allowed_public_entity_refs: Sequence[str], public_precondition_satisfied: bool, budgets: RecoveryBudgets=RecoveryBudgets(), observation_tool_names: Sequence[str]=('look_for', 'scan'), public_entity_argument_names: Sequence[str]=('object', 'receptacle', 'source', 'target')) -> RecoveryLedger:
    identifier = non_empty_string(action_id, pointer='/recovery_action_id')
    exact_mapping(tool_call, frozenset({'tool_name', 'arguments'}), pointer='/tool_call')
    public_json(tool_call, pointer='/tool_call', include_future_action=False)
    fingerprint = sha256_json({'mode': plan.mode, 'incident_id_or_null': plan.incident_id_or_null, 'obligation_id_or_null': plan.obligation_id_or_null, 'tool_call': tool_call})
    recorded = dict(ledger.recorded_actions)
    if identifier in recorded:
        if recorded[identifier] != fingerprint:
            raise MethodContractError('method_event_correlation_invalid', 'recovery action ID was replayed inconsistently')
        return ledger
    validate_recovery_tool_call(tool_call, plan, public_tool_specs=public_tool_specs, allowed_public_entity_refs=allowed_public_entity_refs, public_precondition_satisfied=public_precondition_satisfied, public_entity_argument_names=public_entity_argument_names)
    if ledger.total_recovery_actions >= budgets.max_total_recovery_actions:
        raise MethodContractError('recovery_budget_exceeded', 'total recovery budget is exhausted')
    incident_counts = dict(ledger.incident_attempts)
    observe_counts = dict(ledger.obligation_observe_actions)
    incident_observation_counts = dict(ledger.incident_observation_actions)
    observation_tools = string_sequence(list(observation_tool_names), pointer='/observation_tool_names', sorted_utf8=True)
    if plan.mode == 'recovery':
        incident_id = non_empty_string(plan.incident_id_or_null, pointer='/plan/incident_id')
        count = incident_counts.get(incident_id, 0) + 1
        if count > budgets.max_recovery_attempts_per_incident:
            raise MethodContractError('recovery_budget_exceeded', 'incident recovery budget is exhausted')
        incident_counts[incident_id] = count
        if tool_call['tool_name'] in observation_tools:
            incident_observation_counts[incident_id] = incident_observation_counts.get(incident_id, 0) + 1
    elif plan.mode == 'observe':
        obligation_id = non_empty_string(plan.obligation_id_or_null, pointer='/plan/obligation_id')
        count = observe_counts.get(obligation_id, 0) + 1
        if count > budgets.max_observe_actions_per_ambiguous:
            raise MethodContractError('recovery_budget_exceeded', 'observe budget is exhausted')
        observe_counts[obligation_id] = count
    else:
        raise MethodContractError('method_recovery_tool_forbidden', 'plan is not an executable recovery')
    return RecoveryLedger(total_recovery_actions=ledger.total_recovery_actions + 1, incident_attempts=tuple(sorted(incident_counts.items(), key=lambda item: item[0].encode('utf-8'))), obligation_observe_actions=tuple(sorted(observe_counts.items(), key=lambda item: item[0].encode('utf-8'))), recorded_actions=tuple(sorted((*ledger.recorded_actions, (identifier, fingerprint)))), incident_observation_actions=tuple(sorted(incident_observation_counts.items(), key=lambda item: item[0].encode('utf-8'))))

def promote_observe_obligation(memory: EvidenceMemory, obligation_id: str, *, action_id: str, evidence_frame_ids: Sequence[str], step: int, confidence: float, capacity: MemoryCapacity=MemoryCapacity()) -> EvidenceMemory:
    memory = memory.validated()
    obligation = next((item for item in memory.pending_obligations if item.obligation_id == obligation_id and item.status == 'open' and (item.type == 'observe')), None)
    if obligation is None:
        raise MethodContractError('method_public_reference_invalid', 'observe obligation is missing')
    incident = RecoveryIncident.create(episode_id=memory.episode_id, action_id=action_id, failure_type='unresolved_visual_effect', confidence=confidence, evidence_frame_ids=evidence_frame_ids, step=step)
    incidents = list(memory.incidents)
    existing_incident = next((item for item in incidents if item.incident_id == incident.incident_id), None)
    if existing_incident is None:
        incidents.append(incident)
    else:
        incidents[incidents.index(existing_incident)] = replace(existing_incident, confidence=incident.confidence, evidence_frame_ids=incident.evidence_frame_ids, last_updated_step=incident.last_updated_step, status='open')
    obligations = [replace(item, status='resolved') if item.obligation_id == obligation.obligation_id else item for item in memory.pending_obligations]
    if not any((item.status == 'open' and item.type == 'act' and (item.public_roles == obligation.public_roles) and (item.source_incident_id_or_null == incident.incident_id) for item in obligations)):
        obligations.append(PendingObligation.create(episode_id=memory.episode_id, obligation_type='act', public_roles=obligation.public_roles, created_step=step, source_incident_id_or_null=incident.incident_id))
    updated = EvidenceMemory(memory.episode_id, memory.revision + 1, memory.facts, tuple(sorted(incidents, key=lambda item: item.incident_id.encode('utf-8'))), tuple(sorted(obligations, key=lambda item: item.obligation_id.encode('utf-8'))))
    return enforce_capacity(updated, capacity)

def stop_allowed(memory: EvidenceMemory, *, policy_requested_stop: bool, public_tool_contract_allows_stop: bool, last_physical_outcome_or_null: str | None, recovery_assumed_continue: bool=False, has_assumed_completed_actions: bool=False, has_unresolved_held_placement: bool=False) -> bool:
    memory = memory.validated()
    if not isinstance(policy_requested_stop, bool) or not isinstance(public_tool_contract_allows_stop, bool) or (not isinstance(recovery_assumed_continue, bool)) or (not isinstance(has_assumed_completed_actions, bool)) or (not isinstance(has_unresolved_held_placement, bool)):
        raise MethodContractError('method_schema_invalid', 'stop gates must be boolean')
    if last_physical_outcome_or_null not in {None, 'success', 'failure', 'ambiguous'}:
        raise MethodContractError('method_schema_invalid', 'last physical outcome is invalid')
    return policy_requested_stop and public_tool_contract_allows_stop and (not any((incident.status == 'open' for incident in memory.incidents))) and (not any((obligation.status == 'open' for obligation in memory.pending_obligations))) and (not has_unresolved_held_placement) and (recovery_assumed_continue or has_assumed_completed_actions or last_physical_outcome_or_null not in {'failure', 'ambiguous'})
__all__ = ['PREPARATION_RESULT_STATUSES', 'RecoveryBudgets', 'RecoveryContext', 'RecoveryLedger', 'RecoveryPlan', 'plan_recovery', 'plan_recovery_v13', 'promote_observe_obligation', 'record_recovery_action', 'record_recovery_action_v13', 'stop_allowed', 'validate_recovery_tool_call']
