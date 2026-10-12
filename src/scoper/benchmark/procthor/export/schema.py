from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
CONTRACT_VERSION = 'trajectory-contract.v1'
COMMON_CONTRACT_VERSION = 'trajectory-common-contract.v2'
COMMON_CONTRACT_SHA256 = '6dc78b73dcad0bdbc60c0c53f2f9af83cc3afe49e00513d91dcd921c501d8431'
COMMON_CONTRACT_BYTES = 2011
BACKEND = 'eb_procthor'
TASK_SET_SCHEMA_VERSION = 'eb-procthor-task-set.v1'
TASK_BANK_HASH_SCHEMA_VERSION = 'task-bank-hash.v1'
TASK_SET_HASH_SCHEMA_VERSION = 'task-set-hash.v1'
EXECUTION_AUDIT_SCHEMA_VERSION = 'execution-audit.v1'
STATE_SCHEMA_VERSION = 'eb-procthor-state.v1'
RECORD_TYPES = ('episode_start', 'decision_input', 'action_start', 'low_level_action', 'stream_observation', 'stream_decision', 'intervention', 'action_end', 'verification', 'memory_update', 'teacher_correction', 'episode_end')
ROLLOUT_PURPOSES = ('admission', 'training', 'dev_eval', 'test_eval', 'dagger')
ROLLOUT_ARMS = ('clean', 'disturbance', 'failure', 'mixed', 'learner')
POLICY_MODES = ('teacher', 'oracle', 'vlm', 'method')
DERIVED_KINDS = ('step_decision', 'stream_decision', 'verification_recovery', 'preference', 'memory_update', 'memory_reconcile', 'memory_route')
SEED_NAMESPACES = ('environment', 'scheduler', 'model', 'data_sampling', 'renderer')
ERROR_PRECEDENCE = ('infra_error', 'evaluator_invalid', 'task_invalid', 'budget_exceeded', 'invalid_action', 'agent_failure', 'none')
EFFECT_STATUSES = ('achieved', 'not_achieved', 'partial', 'ambiguous', 'unknown', 'not_applicable')
OBSERVABILITY = ('visible', 'occluded', 'out_of_view', 'visually_ambiguous', 'frame_missing', 'unknown', 'not_applicable')
OPTIONAL_BRAIN_LIFECYCLE = ('prepare_input', 'on_action_result')
SCHEMA_VERSIONS = {'asset_cache_manifest': 'asset-cache-manifest.v1', 'benchmark_protocol': 'benchmark-protocol.v1', 'binding': 'binding-record.v1', 'common_contract': COMMON_CONTRACT_VERSION, 'contract': CONTRACT_VERSION, 'derived_index': 'derived-sample-index.v1', 'derived_manifest': 'derived-export-manifest.v1', 'episode_key': 'episode-key.v1', 'episode_manifest': 'episode-manifest.v1', 'event_index': 'trajectory-record.v1', 'frame': 'frame-record.v1', 'memory_delta': 'memory-delta.v1', 'memory_proposer_io': 'memory-proposer-io.v1', 'method_event': 'method-event.v1', 'method_runtime': 'method-runtime.v1', 'model_io': 'model-io.v1', 'payload': 'trajectory-payload.v1', 'provenance': 'provenance-document.v1', 'quality': 'trajectory-quality.v1', 'run_invocation': 'run-invocation.v1', 'run_manifest': 'run-manifest.v1', 'task_snapshot': 'task-snapshot.v1', 'verifier_io': 'transition-verifier-io.v1'}
SHA256_RE = re.compile('^[0-9a-f]{64}$')
RUN_ID_RE = re.compile('^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
RFC3339_UTC_RE = re.compile('^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}Z$')
EPISODE_KEY_RE = re.compile('^ebk1_[0-9a-f]{64}$')
EPISODE_ID_RE = re.compile('^ep-[0-9a-f]{24}$')
RECORD_ID_RE = re.compile('^rec-[0-9]{8}$')

class TrajectoryExportError(ValueError):

    def __init__(self, code: str, message: str, json_pointer: str | None=None):
        super().__init__('{}: {}'.format(code, message))
        self.code = code
        self.message = message
        self.json_pointer = json_pointer

class SchemaValidationError(TrajectoryExportError):
    pass

def compiled_common_contract() -> dict[str, Any]:
    return {'contract_version': CONTRACT_VERSION, 'derived_kinds': list(DERIVED_KINDS), 'effect_statuses': list(EFFECT_STATUSES), 'error_precedence': list(ERROR_PRECEDENCE), 'observability': list(OBSERVABILITY), 'optional_brain_lifecycle': list(OPTIONAL_BRAIN_LIFECYCLE), 'policy_modes': list(POLICY_MODES), 'record_types': list(RECORD_TYPES), 'rollout_arms': list(ROLLOUT_ARMS), 'rollout_purposes': list(ROLLOUT_PURPOSES), 'schema_version': COMMON_CONTRACT_VERSION, 'schema_versions': dict(SCHEMA_VERSIONS), 'seed_namespaces': list(SEED_NAMESPACES)}

def validate_json_value(value: Any, *, json_pointer: str='') -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError('json_non_finite', 'NaN and Infinity are forbidden', json_pointer or '/')
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_value(item, json_pointer='{}/{}'.format(json_pointer, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError('json_key_invalid', 'JSON object keys must be strings', json_pointer or '/')
            validate_json_value(item, json_pointer='{}/{}'.format(json_pointer, _escape_json_pointer(key)))
        return
    raise SchemaValidationError('json_type_invalid', "unsupported JSON value type '{}'".format(type(value).__name__), json_pointer or '/')

def strict_json_loads(text: str, *, source: str='<memory>') -> Any:

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SchemaValidationError('json_duplicate_key', "duplicate object key '{}' in {}".format(key, source))
            result[key] = value
        return result

    def invalid_constant(token: str) -> None:
        raise SchemaValidationError('json_non_finite', '{} is forbidden in {}'.format(token, source))
    try:
        value = json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    except SchemaValidationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SchemaValidationError('json_parse_error', 'invalid JSON in {}: {}'.format(source, exc)) from exc
    validate_json_value(value)
    return value

def load_json_strict(path: Path | str) -> Any:
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise SchemaValidationError('json_read_error', "cannot read '{}': {}".format(resolved, exc)) from exc
    return strict_json_loads(text, source=str(resolved))

def load_jsonl_strict(path: Path | str) -> list[Any]:
    resolved = Path(path)
    try:
        lines = resolved.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError) as exc:
        raise SchemaValidationError('json_read_error', "cannot read '{}': {}".format(resolved, exc)) from exc
    records: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        records.append(strict_json_loads(line, source='{} line {}'.format(resolved, line_number)))
    return records

def validate_common_contract_fixture(path: Path | str | None=None) -> Mapping[str, Any]:
    from scoper.benchmark.procthor.export.ids import canonical_json_bytes, sha256_bytes
    fixture_path = Path(path) if path is not None else _default_common_fixture_path()
    try:
        raw = fixture_path.read_bytes()
    except OSError as exc:
        raise SchemaValidationError('common_contract_read_error', "cannot read '{}': {}".format(fixture_path, exc)) from exc
    try:
        parsed = strict_json_loads(raw.decode('utf-8'), source=str(fixture_path))
    except UnicodeError as exc:
        raise SchemaValidationError('common_contract_utf8_invalid', 'fixture is not UTF-8') from exc
    expected = compiled_common_contract()
    expected_bytes = canonical_json_bytes(expected) + b'\n'
    if parsed != expected:
        raise SchemaValidationError('common_contract_compiled_mismatch', 'fixture content does not match compiled common constants')
    if raw != expected_bytes:
        raise SchemaValidationError('common_contract_not_canonical', 'fixture bytes must be canonical JSON followed by one newline')
    if len(raw) != COMMON_CONTRACT_BYTES or sha256_bytes(raw) != COMMON_CONTRACT_SHA256:
        raise SchemaValidationError('common_contract_golden_mismatch', 'fixture size or SHA256 differs from the frozen golden')
    return parsed

def validate_task_set_envelope(value: Mapping[str, Any]) -> None:
    required = {'schema_version', 'name', 'version', 'lineage_id', 'benchmark_split', 'task_ids'}
    _require_exact_keys(value, required, 'task_set_schema_invalid')
    _require_equal(value, 'schema_version', TASK_SET_SCHEMA_VERSION)
    for key in ('name', 'version', 'lineage_id'):
        _require_non_empty_string(value, key)
    _require_enum(value, 'benchmark_split', ('train', 'dev', 'test'))
    task_ids = value['task_ids']
    if not isinstance(task_ids, list) or not task_ids:
        raise SchemaValidationError('task_set_schema_invalid', 'task_ids must be a non-empty ordered list', '/task_ids')
    seen: set[str] = set()
    for index, task_id in enumerate(task_ids):
        if not isinstance(task_id, str) or not task_id:
            raise SchemaValidationError('task_set_schema_invalid', 'task_ids must contain non-empty strings', '/task_ids/{}'.format(index))
        if task_id in seen:
            raise SchemaValidationError('task_set_duplicate_id', "duplicate task id '{}'".format(task_id), '/task_ids/{}'.format(index))
        seen.add(task_id)
TRAJECTORY_RECORD_KEYS = {'schema_version', 'record_id', 'record_type', 'episode_id', 'sequence_index', 'step_id', 'decision_id', 'action_id', 'low_level_id', 'stream_id', 'stream_obs_id', 'intervention_id', 'model_call_id', 'logical_time_s', 'monotonic_time_ns', 'wall_time_utc', 'public_payload', 'privileged_payload'}

def validate_trajectory_record(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, TRAJECTORY_RECORD_KEYS, 'trajectory_record_schema_invalid')
    _require_equal(value, 'schema_version', SCHEMA_VERSIONS['event_index'])
    if not isinstance(value['record_id'], str) or RECORD_ID_RE.fullmatch(value['record_id']) is None:
        raise SchemaValidationError('trajectory_record_schema_invalid', 'invalid record_id', '/record_id')
    _require_enum(value, 'record_type', RECORD_TYPES)
    if not isinstance(value['episode_id'], str) or EPISODE_ID_RE.fullmatch(value['episode_id']) is None:
        raise SchemaValidationError('trajectory_record_schema_invalid', 'invalid episode_id', '/episode_id')
    _require_int(value, 'sequence_index', minimum=1)
    _require_int(value, 'step_id', minimum=0)
    for key in ('decision_id', 'action_id', 'low_level_id', 'stream_id', 'stream_obs_id', 'intervention_id', 'model_call_id'):
        if value[key] is not None and (not isinstance(value[key], str) or not value[key]):
            raise SchemaValidationError('trajectory_record_schema_invalid', '{} must be string or null'.format(key), '/' + key)
    _require_number(value, 'logical_time_s', minimum=0.0)
    _require_int(value, 'monotonic_time_ns', minimum=0)
    if not isinstance(value['wall_time_utc'], str) or RFC3339_UTC_RE.fullmatch(value['wall_time_utc']) is None:
        raise SchemaValidationError('trajectory_record_schema_invalid', 'wall_time_utc must be RFC3339 UTC with microseconds', '/wall_time_utc')
    for key in ('public_payload', 'privileged_payload'):
        if not isinstance(value[key], bool):
            raise SchemaValidationError('trajectory_record_schema_invalid', '{} must be boolean'.format(key), '/' + key)

def validate_payload_record(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, {'schema_version', 'record_id', 'record_type', 'payload'}, 'trajectory_payload_schema_invalid')
    _require_equal(value, 'schema_version', SCHEMA_VERSIONS['payload'])
    if not isinstance(value['record_id'], str) or RECORD_ID_RE.fullmatch(value['record_id']) is None:
        raise SchemaValidationError('trajectory_payload_schema_invalid', 'invalid record_id', '/record_id')
    _require_enum(value, 'record_type', RECORD_TYPES)
    if not isinstance(value['payload'], dict):
        raise SchemaValidationError('trajectory_payload_schema_invalid', 'payload must be an object', '/payload')
    validate_json_value(value['payload'], json_pointer='/payload')
PUBLIC_PAYLOAD_FIELDS = {'episode_start': {'task_id', 'instruction', 'public_task_metadata', 'reset_frame_id', 'initial_public_observation', 'tool_specs', 'tool_schema_version', 'budget_initial', 'exposure_profile'}, 'decision_input': {'decision_id', 'input_kind', 'input_frame_ids', 'observation', 'prompt_context', 'allowed_tool_specs', 'tool_history', 'last_public_transition', 'public_working_memory_before'}, 'action_start': {'decision_id', 'requested_tool_call', 'policy_source', 'public_target_ref', 'pre_action_frame_id', 'public_memory_before', 'budget_before'}, 'low_level_action': {'parent_action_id', 'order', 'exposed_to_policy', 'public_feedback', 'frame_id'}, 'stream_observation': {'stream_id', 'stream_obs_id', 'parent_action_id', 'frame_id', 'frame_index', 'timestamp_s', 'source', 'is_keyframe', 'keyframe_reason', 'active_tool', 'active_arguments', 'public_progress', 'public_visible_catalog', 'public_memory_delta', 'recent_frame_ids'}, 'stream_decision': {'stream_obs_id', 'vlm_called', 'decision', 'interrupt_tool_call', 'decision_source', 'model_call_id', 'parse_status', 'parse_error', 'latency_seconds'}, 'intervention': {'intervention_id', 'public_notice', 'notice_exposed', 'public_evidence_frame_ids'}, 'action_end': {'post_action_frame_id', 'immediate_post_action_frame_id', 'confirmation_frame_id', 'executor_status', 'public_feedback', 'last_public_transition', 'public_memory_after', 'public_memory_delta', 'budget_after', 'cost'}, 'verification': {'verification_id', 'parent_action_id', 'evidence_frame_ids', 'model_call_id', 'predicted_effect', 'predicted_state_delta', 'confidence', 'abstained', 'recommended_tool_call', 'parse_status', 'latency_seconds', 'verifier_io_schema_version', 'verifier_checkpoint_sha256', 'outcome_probabilities', 'effect_probabilities', 'observability_probabilities', 'selected_outcome', 'selected_effects', 'selected_observability', 'calibration_sha256'}, 'memory_update': {'proposal_id', 'proposal_mode', 'base_memory_revision', 'proposer_call_id', 'proposer_checkpoint_sha256', 'fact_id', 'operation', 'fact_type', 'public_entity_ref', 'previous_value', 'proposed_value', 'accepted_value', 'status', 'confidence', 'reason', 'evidence_record_ids', 'accepted_by_harness', 'exposed_to_policy', 'abstain_reason_or_null', 'candidate_pool_hash'}, 'teacher_correction': {'correction_id', 'decision_input_record_id', 'learner_tool_call', 'learner_action_end_record_id', 'input_frame_ids', 'public_context', 'public_memory'}, 'episode_end': {'final_frame_id', 'episode_status', 'done_reason', 'final_public_feedback', 'public_cost_summary'}}
PRIVILEGED_PAYLOAD_FIELDS = {'episode_start': {'private_task_ref', 'initial_snapshot_id', 'native_initial_goal', 'effective_initial_goal', 'role_bindings', 'init_mutations', 'scene_mutations', 'house_id', 'house_path_hash', 'original_procthor_split', 'reset_diagnostics', 'reachability_diagnostics', 'controller_diagnostics'}, 'decision_input': {'state_snapshot_id', 'policy_source', 'policy_id', 'policy_hash', 'teacher_private_context'}, 'action_start': {'resolved_target', 'state_before_snapshot_id', 'native_goal_before', 'effective_goal_before', 'agent_pose_before', 'native_inventory_before', 'effective_inventory_before', 'expected_effects', 'execution_profile'}, 'low_level_action': {'low_level_id', 'requested_action', 'executed_action', 'kwargs', 'last_action_success', 'error_message', 'pose_before', 'pose_after', 'native_inventory', 'effective_inventory', 'event_ordinal', 'simulator_time_s', 'visible_object_ids', 'source', 'fallback_entries'}, 'stream_observation': {'event_ordinal', 'agent_pose', 'target_object_ids', 'target_visible', 'target_distance_m', 'interaction_ready', 'instance_bbox_refs', 'segmentation_refs', 'first_visible', 'first_interactable', 'native_metadata_ref', 'effective_snapshot_id'}, 'stream_decision': {'teacher_reference_tool_call', 'offline_label_status'}, 'intervention': {'event_kind', 'configured_type', 'trigger', 'phase', 'target_roles', 'target_object_ids', 'state_before_snapshot_id', 'requested_mutation', 'state_after_snapshot_id', 'application_status', 'native_mutation_result', 'effective_mutation_result', 'pre_intervention_frame_id', 'post_intervention_frame_id', 'first_visually_observable_frame_id', 'first_visible_status', 'recovery_obligation', 'scheduler_truth', 'fallback_entries'}, 'action_end': {'state_after_snapshot_id', 'normalized_state_delta', 'native_goal_after', 'effective_goal_after', 'native_predicate_delta', 'effective_predicate_delta', 'intended_effects', 'native_achieved_effects', 'effective_achieved_effects', 'native_intended_effect_status', 'compatibility_intended_effect_status', 'intended_effect_status', 'effect_observability', 'effect_label_provenance', 'observability_provenance', 'error', 'fallback_audit', 'next_obligation'}, 'verification': {'ground_truth_action_end_record_id', 'native_ground_truth_effect', 'effective_ground_truth_effect', 'calibration_bucket'}, 'memory_update': {'private_entity_ids', 'native_truth', 'effective_truth', 'truth_source', 'router_selected_at_next_decision', 'gold_memory_delta_or_null', 'observability', 'label_provenance', 'truth_track'}, 'teacher_correction': {'teacher_tool_call', 'chosen_tool_call', 'rejected_tool_call', 'state_match', 'takeover_executed', 'reason', 'failure_taxonomy', 'confidence', 'teacher_id', 'teacher_hash'}, 'episode_end': {'final_snapshot_id', 'native_final_goal', 'effective_final_goal', 'native_task_success', 'compatibility_task_success', 'error', 'intervention_summary', 'recovery_summary', 'fallback_summary', 'capture_integrity', 'eligibility', 'artifact_paths'}}

def validate_event_payload(record_type: str, payload: Mapping[str, Any], *, privacy: str) -> None:
    if record_type not in RECORD_TYPES:
        raise SchemaValidationError('trajectory_payload_schema_invalid', 'unsupported record type')
    if privacy == 'public':
        required = PUBLIC_PAYLOAD_FIELDS[record_type]
    elif privacy == 'privileged':
        required = PRIVILEGED_PAYLOAD_FIELDS[record_type]
    else:
        raise SchemaValidationError('trajectory_payload_schema_invalid', 'privacy must be public or privileged')
    if not isinstance(payload, Mapping):
        raise SchemaValidationError('trajectory_payload_schema_invalid', 'payload must be an object', '/payload')
    keys = set(payload)
    missing = required - keys
    extra = keys - required - {'extensions'}
    if missing:
        raise SchemaValidationError('trajectory_payload_schema_invalid', '{} {} payload missing keys: {}'.format(record_type, privacy, ', '.join(sorted(missing))), '/payload')
    if extra:
        raise SchemaValidationError('trajectory_payload_schema_invalid', '{} {} payload has unknown keys: {}'.format(record_type, privacy, ', '.join(sorted(extra))), '/payload')
    if 'extensions' in payload and (not isinstance(payload['extensions'], Mapping)):
        raise SchemaValidationError('trajectory_payload_schema_invalid', 'extensions must be an object', '/payload/extensions')
    validate_json_value(dict(payload), json_pointer='/payload')
    _validate_event_payload_values(record_type, payload, privacy=privacy)

def validate_tool_call(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, {'tool_name', 'arguments', 'reasoning'}, 'tool_call_schema_invalid')
    _require_non_empty_string(value, 'tool_name')
    if not isinstance(value['arguments'], dict):
        raise SchemaValidationError('tool_call_schema_invalid', 'arguments must be an object', '/arguments')
    if value['reasoning'] is not None and (not isinstance(value['reasoning'], str)):
        raise SchemaValidationError('tool_call_schema_invalid', 'reasoning must be string or null', '/reasoning')
    validate_json_value(dict(value))

def validate_budget_state(value: Mapping[str, Any]) -> None:
    required = {'high_level_used', 'high_level_limit', 'low_level_used', 'low_level_limit', 'model_calls_used', 'model_calls_limit', 'prompt_tokens_used', 'completion_tokens_used'}
    _require_exact_keys(value, required, 'budget_state_schema_invalid')
    for key in ('high_level_used', 'high_level_limit', 'low_level_used', 'low_level_limit'):
        _require_int(value, key, minimum=0)
    for key in ('model_calls_used', 'model_calls_limit', 'prompt_tokens_used', 'completion_tokens_used'):
        if value[key] is not None:
            _require_int(value, key, minimum=0)

def validate_cost_state(value: Mapping[str, Any]) -> None:
    required = {'low_level_actions', 'stream_frames', 'logical_seconds', 'wall_seconds', 'model_calls', 'prompt_tokens', 'completion_tokens'}
    _require_exact_keys(value, required, 'cost_state_schema_invalid')
    for key in ('low_level_actions', 'stream_frames', 'model_calls'):
        _require_int(value, key, minimum=0)
    for key in ('logical_seconds', 'wall_seconds'):
        _require_number(value, key, minimum=0.0)
    for key in ('prompt_tokens', 'completion_tokens'):
        if value[key] is not None:
            _require_int(value, key, minimum=0)

def validate_error_info(value: Mapping[str, Any]) -> None:
    required = {'error_class', 'error_code', 'message', 'retryable', 'source'}
    _require_exact_keys(value, required, 'error_info_schema_invalid')
    _require_enum(value, 'error_class', ERROR_PRECEDENCE)
    for key in ('error_code', 'message'):
        if value[key] is not None and (not isinstance(value[key], str)):
            raise SchemaValidationError('error_info_schema_invalid', '{} must be string or null'.format(key), '/' + key)
    if value['retryable'] is not None and (not isinstance(value['retryable'], bool)):
        raise SchemaValidationError('error_info_schema_invalid', 'retryable must be boolean or null', '/retryable')
    if value['source'] not in ('runner', 'executor', 'controller', 'scheduler', 'model', 'evaluator', 'capture', None):
        raise SchemaValidationError('error_info_schema_invalid', 'unsupported error source', '/source')

def validate_label_provenance(value: Mapping[str, Any]) -> None:
    required = {'source', 'source_ref', 'confidence', 'review_status'}
    _require_exact_keys(value, required, 'label_provenance_schema_invalid')
    _require_enum(value, 'source', ('ai2thor_metadata', 'effective_overlay', 'renderer', 'teacher_rule', 'human_review', 'model_prediction', 'derived_rule', 'unavailable'))
    if value['source_ref'] is not None and (not isinstance(value['source_ref'], str)):
        raise SchemaValidationError('label_provenance_schema_invalid', 'source_ref must be string or null')
    confidence = value['confidence']
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or (not math.isfinite(float(confidence))) or (not 0.0 <= float(confidence) <= 1.0):
            raise SchemaValidationError('label_provenance_schema_invalid', 'confidence must be in [0,1]')
    _require_enum(value, 'review_status', ('unreviewed', 'accepted', 'rejected', 'not_applicable'))

def validate_state_value(value: Mapping[str, Any]) -> None:
    required = {'value', 'status', 'source', 'source_detail', 'error_code'}
    allowed = required | {'conflict_candidates'}
    _require_keys(value, required, 'state_value_schema_invalid')
    extra = set(value) - allowed
    if extra:
        raise SchemaValidationError('state_value_schema_invalid', 'unknown keys: {}'.format(', '.join(sorted(extra))))
    _require_enum(value, 'status', ('known', 'unknown', 'not_applicable', 'conflict', 'error'))
    _require_enum(value, 'source', ('ai2thor_metadata', 'sticky_overlay', 'visual_attachment_overlay', 'goal_evaluator', 'scheduler', 'derived_rule', 'unavailable'))
    if value['status'] == 'known' and value['value'] is None:
        raise SchemaValidationError('state_value_schema_invalid', 'known StateValue requires non-null value')
    if value['status'] != 'known' and value['status'] != 'conflict' and (value['value'] is not None):
        raise SchemaValidationError('state_value_schema_invalid', 'non-known StateValue must have null value')
    if value['status'] == 'conflict' and (not isinstance(value.get('conflict_candidates'), list)):
        raise SchemaValidationError('state_value_schema_invalid', 'conflict StateValue requires conflict_candidates')
    for key in ('source_detail', 'error_code'):
        if value[key] is not None and (not isinstance(value[key], str)):
            raise SchemaValidationError('state_value_schema_invalid', '{} must be string or null'.format(key))
    validate_json_value(dict(value))

def _validate_event_payload_values(record_type: str, payload: Mapping[str, Any], *, privacy: str) -> None:
    if privacy == 'public':
        if record_type == 'episode_start':
            validate_budget_state(payload['budget_initial'])
        elif record_type == 'decision_input':
            _require_enum(payload, 'input_kind', ('brain_act', 'teacher_act'))
        elif record_type == 'action_start':
            validate_tool_call(payload['requested_tool_call'])
            validate_budget_state(payload['budget_before'])
            _require_enum(payload, 'policy_source', ('teacher', 'oracle', 'recovery_oracle', 'learner'))
        elif record_type == 'stream_decision':
            if not isinstance(payload['vlm_called'], bool):
                raise SchemaValidationError('trajectory_payload_schema_invalid', 'vlm_called must be boolean')
            _require_enum(payload, 'decision', ('continue', 'interrupt', 'no_call'))
            _require_enum(payload, 'decision_source', ('online_model', 'monitor_disabled', 'call_budget', 'parse_fallback'))
            _require_enum(payload, 'parse_status', ('ok', 'invalid', 'error', 'not_called'))
        elif record_type == 'action_end':
            _require_enum(payload, 'executor_status', ('accepted', 'succeeded', 'failed', 'interrupted', 'blocked', 'unknown'))
            validate_budget_state(payload['budget_after'])
            validate_cost_state(payload['cost'])
        elif record_type == 'episode_end':
            _require_enum(payload, 'episode_status', ('passed', 'failed', 'invalid'))
    elif record_type == 'decision_input':
        _require_enum(payload, 'policy_source', ('teacher', 'oracle', 'recovery_oracle', 'learner'))
    elif record_type == 'action_start':
        _require_enum(payload, 'execution_profile', ('native_ai2thor', 'semantic_compatibility'))
    elif record_type == 'intervention':
        _require_enum(payload, 'event_kind', ('applied', 'evidence_update'))
        _require_enum(payload, 'application_status', ('applied', 'failed', 'no_effect'))
        _require_enum(payload, 'first_visible_status', ('pending', 'observed', 'never_observed', 'unknown'))
    elif record_type == 'action_end':
        for key in ('native_intended_effect_status', 'compatibility_intended_effect_status', 'intended_effect_status'):
            _require_enum(payload, key, EFFECT_STATUSES)
        _require_enum(payload, 'effect_observability', OBSERVABILITY)
        validate_error_info(payload['error'])
    elif record_type == 'memory_update':
        _require_enum(payload, 'truth_track', ('native', 'effective'))
    elif record_type == 'episode_end':
        validate_error_info(payload['error'])

def validate_provenance_document(value: Mapping[str, Any], document_type: str | None=None) -> None:
    from scoper.benchmark.procthor.export.ids import sha256_json
    _require_exact_keys(value, {'schema_version', 'document_type', 'content_hash', 'content'}, 'provenance_schema_invalid')
    _require_equal(value, 'schema_version', SCHEMA_VERSIONS['provenance'])
    _require_enum(value, 'document_type', ('protocol', 'policy', 'capture_config'))
    if document_type is not None and value['document_type'] != document_type:
        raise SchemaValidationError('provenance_schema_invalid', "expected document_type '{}'".format(document_type), '/document_type')
    _require_sha256(value, 'content_hash')
    validate_json_value(value['content'], json_pointer='/content')
    if sha256_json(value['content']) != value['content_hash']:
        raise SchemaValidationError('provenance_hash_mismatch', 'content_hash does not match content', '/content_hash')

def validate_quality_report(value: Mapping[str, Any]) -> None:
    required = {'schema_version', 'episode_id', 'structural_valid', 'quality_valid', 'capture_capabilities', 'checks', 'errors', 'warnings', 'exporter_eligibility', 'exclusion_reasons'}
    _require_keys(value, required, 'quality_schema_invalid')
    _require_equal(value, 'schema_version', SCHEMA_VERSIONS['quality'])
    for key in ('structural_valid', 'quality_valid'):
        if not isinstance(value[key], bool):
            raise SchemaValidationError('quality_schema_invalid', '{} must be boolean'.format(key), '/' + key)
    if not isinstance(value['checks'], dict):
        raise SchemaValidationError('quality_schema_invalid', 'checks must be an object', '/checks')
    required_checks = {'records', 'references', 'frames', 'leakage', 'effects', 'split', 'hashes', 'native_profile', 'audit_completeness'}
    missing_checks = required_checks - set(value['checks'])
    if missing_checks:
        raise SchemaValidationError('quality_schema_invalid', 'missing checks: {}'.format(', '.join(sorted(missing_checks))), '/checks')
    for key in ('errors', 'warnings', 'exclusion_reasons'):
        if not isinstance(value[key], list):
            raise SchemaValidationError('quality_schema_invalid', '{} must be a list'.format(key), '/' + key)
    if not isinstance(value['exporter_eligibility'], dict):
        raise SchemaValidationError('quality_schema_invalid', 'exporter_eligibility must be an object', '/exporter_eligibility')
    validate_json_value(dict(value))

def validate_episode_manifest(value: Mapping[str, Any], *, verify_content_hash: bool=True) -> None:
    from scoper.benchmark.procthor.export.ids import sha256_json
    required = {'schema_version', 'contract_version', 'bundle_status', 'backend', 'run_id', 'episode_id', 'episode_key', 'episode_key_payload', 'task', 'scene', 'rollout', 'eligibility', 'seeds', 'versions', 'timing', 'counts', 'terminal', 'fallback_summary', 'artifacts', 'bundle_content_sha256', 'quality_summary'}
    _require_exact_keys(value, required, 'episode_manifest_schema_invalid')
    _require_equal(value, 'schema_version', SCHEMA_VERSIONS['episode_manifest'])
    _require_equal(value, 'contract_version', CONTRACT_VERSION)
    _require_equal(value, 'bundle_status', 'complete')
    _require_equal(value, 'backend', BACKEND)
    if not isinstance(value['run_id'], str) or RUN_ID_RE.fullmatch(value['run_id']) is None:
        raise SchemaValidationError('episode_manifest_schema_invalid', 'invalid run_id', '/run_id')
    if not isinstance(value['episode_id'], str) or EPISODE_ID_RE.fullmatch(value['episode_id']) is None:
        raise SchemaValidationError('episode_manifest_schema_invalid', 'invalid episode_id', '/episode_id')
    if not isinstance(value['episode_key'], str) or EPISODE_KEY_RE.fullmatch(value['episode_key']) is None:
        raise SchemaValidationError('episode_manifest_schema_invalid', 'invalid episode_key', '/episode_key')
    for key in ('episode_key_payload', 'task', 'scene', 'rollout', 'eligibility', 'seeds', 'versions', 'timing', 'counts', 'terminal', 'fallback_summary', 'quality_summary'):
        if not isinstance(value[key], dict):
            raise SchemaValidationError('episode_manifest_schema_invalid', '{} must be an object'.format(key), '/' + key)
    _require_exact_keys(value['episode_key_payload'], {'schema_version', 'backend', 'task_id', 'task_hash', 'source_split', 'scene_id', 'layout_id', 'execution_profile', 'rollout_arm', 'intervention_hash', 'policy_id', 'policy_hash', 'protocol_hash', 'base_seed', 'repeat_id'}, 'episode_manifest_schema_invalid')
    nested_keys = {'task': {'id', 'hash', 'family', 'template_id', 'source_split', 'task_split', 'lineage_id', 'task_bank_hash', 'task_set_hash'}, 'scene': {'scene_id', 'layout_id', 'house_path_hash', 'house_content_sha256', 'dataset_version', 'asset_version'}, 'rollout': {'purpose', 'arm', 'execution_profile', 'repeat_id'}, 'eligibility': {'train_eligible', 'admission_eligible', 'exclusion_reasons'}, 'timing': {'started_at_utc', 'ended_at_utc', 'elapsed_seconds'}, 'counts': {'records', 'actions', 'low_level_actions', 'stream_observations', 'frames', 'interventions', 'model_calls', 'prompt_tokens', 'completion_tokens'}, 'terminal': {'episode_status', 'done_reason', 'error_class', 'error_code', 'native_task_success', 'compatibility_task_success'}, 'quality_summary': {'valid', 'errors', 'warnings'}}
    for key, expected in nested_keys.items():
        _require_exact_keys(value[key], expected, 'episode_manifest_schema_invalid')
    for key in ('id', 'family', 'template_id', 'lineage_id'):
        _require_non_empty_string(value['task'], key)
    for key in ('hash', 'task_bank_hash', 'task_set_hash'):
        _require_sha256(value['task'], key)
    _require_enum(value['task'], 'source_split', ('train', 'val', 'test'))
    _require_enum(value['task'], 'task_split', ('train', 'dev', 'test'))
    _require_non_empty_string(value['scene'], 'scene_id')
    if value['scene']['layout_id'] != '':
        raise SchemaValidationError('episode_manifest_schema_invalid', 'ProcTHOR v1 layout_id must be empty', '/scene/layout_id')
    for key in ('house_path_hash', 'house_content_sha256'):
        _require_sha256(value['scene'], key)
    for key in ('dataset_version', 'asset_version'):
        _require_non_empty_string(value['scene'], key)
    _require_enum(value['rollout'], 'purpose', ROLLOUT_PURPOSES)
    _require_enum(value['rollout'], 'arm', ROLLOUT_ARMS)
    _require_enum(value['rollout'], 'execution_profile', ('native_ai2thor', 'semantic_compatibility'))
    _require_int(value['rollout'], 'repeat_id', minimum=0)
    for key in ('train_eligible', 'admission_eligible'):
        if not isinstance(value['eligibility'][key], bool):
            raise SchemaValidationError('episode_manifest_schema_invalid', 'eligibility booleans are invalid', '/eligibility/' + key)
    if not isinstance(value['eligibility']['exclusion_reasons'], list):
        raise SchemaValidationError('episode_manifest_schema_invalid', 'exclusion_reasons must be a list')
    _require_exact_keys(value['seeds'], {'base', *SEED_NAMESPACES}, 'episode_manifest_schema_invalid')
    _require_int(value['seeds'], 'base', minimum=0, maximum=2 ** 31 - 1)
    for namespace in SEED_NAMESPACES:
        seed = value['seeds'][namespace]
        _require_exact_keys(seed, {'value', 'consumed', 'consumer', 'reason'}, 'episode_manifest_schema_invalid')
        _require_int(seed, 'value', minimum=0, maximum=2 ** 31 - 1)
        if type(seed['consumed']) is not bool:
            raise SchemaValidationError('episode_manifest_schema_invalid', 'seed consumed must be boolean')
        for key in ('consumer', 'reason'):
            if seed[key] is not None and (not isinstance(seed[key], str)):
                raise SchemaValidationError('episode_manifest_schema_invalid', 'seed consumer/reason must be string or null')
    required_versions = {'common_contract', 'protocol', 'tool_schema', 'policy', 'capture', 'code', 'ai2thor', 'procthor'}
    _require_exact_keys(value['versions'], required_versions, 'episode_manifest_schema_invalid')
    common_version = value['versions']['common_contract']
    _require_exact_keys(common_version, {'schema_version', 'sha256'}, 'episode_manifest_schema_invalid')
    _require_equal(common_version, 'schema_version', COMMON_CONTRACT_VERSION)
    _require_sha256(common_version, 'sha256')
    for key in ('protocol', 'policy', 'capture'):
        _require_sha256(value['versions'], key)
    for key in ('started_at_utc', 'ended_at_utc'):
        if not isinstance(value['timing'][key], str) or RFC3339_UTC_RE.fullmatch(value['timing'][key]) is None:
            raise SchemaValidationError('episode_manifest_schema_invalid', 'manifest timing is invalid', '/timing/' + key)
    _require_number(value['timing'], 'elapsed_seconds', minimum=0.0)
    for key in ('records', 'actions', 'low_level_actions', 'stream_observations', 'frames', 'interventions', 'model_calls'):
        _require_int(value['counts'], key, minimum=0)
    for key in ('prompt_tokens', 'completion_tokens'):
        if value['counts'][key] is not None:
            _require_int(value['counts'], key, minimum=0)
    _require_enum(value['terminal'], 'episode_status', ('passed', 'failed', 'invalid'))
    _require_enum(value['terminal'], 'error_class', ERROR_PRECEDENCE)
    invalid_terminal_field = _invalid_terminal_string_field(value['terminal'])
    if invalid_terminal_field is not None:
        raise SchemaValidationError('episode_manifest_schema_invalid', '{} must be string or null'.format(invalid_terminal_field), '/terminal/{}'.format(invalid_terminal_field))
    for key in ('native_task_success', 'compatibility_task_success'):
        if value['terminal'][key] is not None and (not isinstance(value['terminal'][key], bool)):
            raise SchemaValidationError('episode_manifest_schema_invalid', 'task success must be boolean or null')
    if not isinstance(value['quality_summary']['valid'], bool):
        raise SchemaValidationError('episode_manifest_schema_invalid', 'quality valid must be boolean')
    for key in ('errors', 'warnings'):
        if not isinstance(value['quality_summary'][key], list):
            raise SchemaValidationError('episode_manifest_schema_invalid', 'quality errors/warnings must be lists')
    if not isinstance(value['artifacts'], list):
        raise SchemaValidationError('episode_manifest_schema_invalid', 'artifacts must be a list', '/artifacts')
    paths: set[str] = set()
    previous_path: str | None = None
    for index, artifact in enumerate(value['artifacts']):
        if not isinstance(artifact, dict):
            raise SchemaValidationError('episode_manifest_schema_invalid', 'artifact must be an object', '/artifacts/{}'.format(index))
        _require_exact_keys(artifact, {'path', 'media_type', 'privacy', 'bytes', 'sha256', 'record_count'}, 'episode_manifest_schema_invalid')
        path = artifact['path']
        if not isinstance(path, str) or not path or _unsafe_relative_path(path):
            raise SchemaValidationError('episode_manifest_schema_invalid', 'invalid artifact path', '/artifacts/{}/path'.format(index))
        if path in paths or (previous_path is not None and path < previous_path):
            raise SchemaValidationError('episode_manifest_schema_invalid', 'artifact paths must be unique and sorted', '/artifacts/{}'.format(index))
        paths.add(path)
        previous_path = path
        _require_enum(artifact, 'privacy', ('index', 'public', 'privileged', 'provenance', 'quality'))
        _require_int(artifact, 'bytes', minimum=0)
        _require_sha256(artifact, 'sha256')
        if artifact['record_count'] is not None:
            _require_int(artifact, 'record_count', minimum=0)
    _require_sha256(value, 'bundle_content_sha256')
    if verify_content_hash and sha256_json(value['artifacts']) != value['bundle_content_sha256']:
        raise SchemaValidationError('bundle_content_hash_mismatch', 'bundle_content_sha256 does not match artifacts', '/bundle_content_sha256')
    validate_json_value(dict(value))

def validate_run_manifest_minimal(value: Mapping[str, Any]) -> None:
    required = {'schema_version', 'contract_version', 'run_id', 'backend', 'created_at_utc', 'run_identity_hash', 'common_contract_sha256', 'protocol_hash', 'policy_hash', 'capture_config_hash', 'provenance_documents', 'code_commit', 'dirty_worktree', 'dirty_diff_hash', 'task_bank_path', 'task_bank_file_sha256', 'task_bank_hash', 'task_set_path', 'task_set_file_sha256', 'task_set_hash', 'task_selection_hash', 'task_order_hash', 'task_set_version', 'task_lineage_id', 'source_split', 'task_split', 'rollout_purpose', 'rollout_arm', 'base_seed', 'repeat_ids', 'shard_index', 'shard_count', 'runtime_contract', 'asset_cache', 'asset_cache_manifest_path', 'asset_cache_manifest_sha256'}
    _require_exact_keys(value, required, 'run_manifest_schema_invalid')
    _require_equal(value, 'schema_version', SCHEMA_VERSIONS['run_manifest'])
    _require_equal(value, 'contract_version', CONTRACT_VERSION)
    _require_equal(value, 'backend', BACKEND)
    if not isinstance(value['run_id'], str) or RUN_ID_RE.fullmatch(value['run_id']) is None:
        raise SchemaValidationError('run_manifest_schema_invalid', 'invalid run_id', '/run_id')
    if not isinstance(value['created_at_utc'], str) or RFC3339_UTC_RE.fullmatch(value['created_at_utc']) is None:
        raise SchemaValidationError('run_manifest_schema_invalid', 'created_at_utc is invalid', '/created_at_utc')
    for key in ('run_identity_hash', 'common_contract_sha256', 'protocol_hash', 'policy_hash', 'capture_config_hash', 'dirty_diff_hash', 'task_bank_file_sha256', 'task_bank_hash', 'task_set_file_sha256', 'task_set_hash', 'task_selection_hash', 'task_order_hash', 'asset_cache_manifest_sha256'):
        _require_sha256(value, key)
    if value['common_contract_sha256'] != COMMON_CONTRACT_SHA256:
        raise SchemaValidationError('common_contract_hash_mismatch', 'run manifest common contract hash is not frozen golden')
    _require_non_empty_string(value, 'task_lineage_id')
    _require_non_empty_string(value, 'task_set_version')
    _require_enum(value, 'source_split', ('train', 'val', 'test'))
    _require_enum(value, 'task_split', ('train', 'dev', 'test'))
    _require_enum(value, 'rollout_purpose', ROLLOUT_PURPOSES)
    _require_enum(value, 'rollout_arm', ROLLOUT_ARMS)
    _require_int(value, 'base_seed', minimum=0, maximum=2 ** 31 - 1)
    repeat_ids = value['repeat_ids']
    if not isinstance(repeat_ids, list) or not repeat_ids or any((isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in repeat_ids)) or (repeat_ids != sorted(set(repeat_ids))):
        raise SchemaValidationError('run_manifest_schema_invalid', 'repeat_ids must be non-empty, sorted, and unique')
    _require_int(value, 'shard_index', minimum=0)
    _require_int(value, 'shard_count', minimum=1)
    if value['shard_index'] >= value['shard_count']:
        raise SchemaValidationError('run_manifest_schema_invalid', 'shard_index must be less than shard_count')
    if not isinstance(value['dirty_worktree'], bool):
        raise SchemaValidationError('run_manifest_schema_invalid', 'dirty_worktree must be boolean')
    for key in ('code_commit', 'task_bank_path', 'task_set_path', 'asset_cache_manifest_path'):
        _require_non_empty_string(value, key)
    provenance = value['provenance_documents']
    _require_exact_keys(provenance, {'protocol', 'policy', 'capture_config'}, 'run_manifest_schema_invalid')
    for name in ('protocol', 'policy', 'capture_config'):
        _require_exact_keys(provenance[name], {'path', 'sha256'}, 'run_manifest_schema_invalid')
        _require_non_empty_string(provenance[name], 'path')
        _require_sha256(provenance[name], 'sha256')
        if provenance[name]['path'] != 'provenance/{}.json'.format(name):
            raise SchemaValidationError('run_manifest_schema_invalid', 'provenance document path is not fixed')
    if not isinstance(value['runtime_contract'], dict):
        raise SchemaValidationError('run_manifest_schema_invalid', 'runtime_contract must be an object', '/runtime_contract')
    if not isinstance(value['asset_cache'], list):
        raise SchemaValidationError('run_manifest_schema_invalid', 'asset_cache must be a list', '/asset_cache')
    _validate_asset_cache_entries(value['asset_cache'])
    if value['asset_cache_manifest_path'] != '.asset_cache/manifest.json':
        raise SchemaValidationError('run_manifest_schema_invalid', 'asset cache manifest path is not fixed')
    expected_identity = compute_run_identity_hash(value)
    if value['run_identity_hash'] != expected_identity:
        raise SchemaValidationError('run_identity_hash_mismatch', 'run_identity_hash does not match portable manifest identity')
    validate_json_value(dict(value))

def compute_run_identity_hash(value: Mapping[str, Any]) -> str:
    from scoper.benchmark.procthor.export.ids import canonical_json_bytes, sha256_bytes
    payload = strict_json_loads(canonical_json_bytes(dict(value)).decode('utf-8'))
    payload.pop('created_at_utc', None)
    payload.pop('run_identity_hash', None)
    payload.pop('task_bank_path', None)
    payload.pop('task_set_path', None)
    payload.pop('asset_cache_manifest_path', None)
    for document in payload.get('provenance_documents', {}).values():
        document.pop('path', None)
    for entry in payload.get('asset_cache', []):
        entry.pop('relative_path', None)
        entry.pop('bytes', None)
    return sha256_bytes(canonical_json_bytes(payload))

def validate_asset_cache_manifest(value: Mapping[str, Any]) -> None:
    from scoper.benchmark.procthor.export.ids import sha256_json
    _require_exact_keys(value, {'schema_version', 'backend', 'entries', 'cache_content_hash'}, 'asset_cache_manifest_invalid')
    _require_equal(value, 'schema_version', SCHEMA_VERSIONS['asset_cache_manifest'])
    _require_equal(value, 'backend', BACKEND)
    if not isinstance(value['entries'], list):
        raise SchemaValidationError('asset_cache_manifest_invalid', 'entries must be a list', '/entries')
    _validate_asset_cache_entries(value['entries'])
    _require_sha256(value, 'cache_content_hash')
    if sha256_json(value['entries']) != value['cache_content_hash']:
        raise SchemaValidationError('asset_cache_manifest_invalid', 'cache_content_hash differs from entries')

def _validate_asset_cache_entries(entries: list[Any]) -> None:
    previous: tuple[str, str] | None = None
    for index, entry in enumerate(entries):
        _require_exact_keys(entry, {'json_pointer', 'asset_kind', 'content_sha256', 'relative_path', 'bytes', 'dataset_name', 'dataset_revision'}, 'asset_cache_manifest_invalid')
        _require_equal(entry, 'json_pointer', '/source/house_path')
        _require_equal(entry, 'asset_kind', 'house')
        _require_sha256(entry, 'content_sha256')
        _require_non_empty_string(entry, 'relative_path')
        relative = entry['relative_path']
        if _unsafe_relative_path(relative) or not relative.startswith('.asset_cache/houses/'):
            raise SchemaValidationError('asset_cache_manifest_invalid', 'asset path must be under .asset_cache/houses')
        _require_int(entry, 'bytes', minimum=0)
        for key in ('dataset_name', 'dataset_revision'):
            if entry[key] is not None and (not isinstance(entry[key], str) or not entry[key]):
                raise SchemaValidationError('asset_cache_manifest_invalid', 'dataset fields must be non-empty strings or null')
        order = (entry['json_pointer'], entry['content_sha256'])
        if previous is not None and order <= previous:
            raise SchemaValidationError('asset_cache_manifest_invalid', 'asset entries must be sorted and unique')
        previous = order

def _default_common_fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'docs/spec_assets' / 'cross_backend_common.json'

def _escape_json_pointer(value: str) -> str:
    return value.replace('~', '~0').replace('/', '~1')

def _invalid_terminal_string_field(value: Mapping[str, Any]) -> str | None:
    for key in ('done_reason', 'error_code'):
        if value[key] is not None and (not isinstance(value[key], str)):
            return key
    return None

def _require_keys(value: Mapping[str, Any], required: set[str], code: str) -> None:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(code, 'value must be an object')
    missing = required - set(value)
    if missing:
        raise SchemaValidationError(code, 'missing keys: {}'.format(', '.join(sorted(missing))))

def _require_exact_keys(value: Mapping[str, Any], required: set[str], code: str) -> None:
    _require_keys(value, required, code)
    extra = set(value) - required
    if extra:
        raise SchemaValidationError(code, 'unknown keys: {}'.format(', '.join(sorted(extra))))

def _require_equal(value: Mapping[str, Any], key: str, expected: Any) -> None:
    if value.get(key) != expected:
        raise SchemaValidationError('schema_value_invalid', '{} must equal {!r}'.format(key, expected), '/' + key)

def _require_non_empty_string(value: Mapping[str, Any], key: str) -> None:
    if not isinstance(value.get(key), str) or not value[key]:
        raise SchemaValidationError('schema_value_invalid', '{} must be a non-empty string'.format(key), '/' + key)

def _require_enum(value: Mapping[str, Any], key: str, allowed: Iterable[str]) -> None:
    if value.get(key) not in allowed:
        raise SchemaValidationError('schema_value_invalid', '{} has unsupported value {!r}'.format(key, value.get(key)), '/' + key)

def _require_int(value: Mapping[str, Any], key: str, *, minimum: int | None=None, maximum: int | None=None) -> None:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise SchemaValidationError('schema_value_invalid', '{} must be an integer'.format(key), '/' + key)
    if minimum is not None and item < minimum:
        raise SchemaValidationError('schema_value_invalid', '{} must be >= {}'.format(key, minimum), '/' + key)
    if maximum is not None and item > maximum:
        raise SchemaValidationError('schema_value_invalid', '{} must be <= {}'.format(key, maximum), '/' + key)

def _require_number(value: Mapping[str, Any], key: str, *, minimum: float | None=None) -> None:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)) or (not math.isfinite(float(item))):
        raise SchemaValidationError('schema_value_invalid', '{} must be a finite number'.format(key), '/' + key)
    if minimum is not None and float(item) < minimum:
        raise SchemaValidationError('schema_value_invalid', '{} must be >= {}'.format(key, minimum), '/' + key)

def _require_sha256(value: Mapping[str, Any], key: str) -> None:
    item = value.get(key)
    if not isinstance(item, str) or SHA256_RE.fullmatch(item) is None:
        raise SchemaValidationError('schema_value_invalid', '{} must be lowercase SHA256 hex'.format(key), '/' + key)

def _unsafe_relative_path(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_absolute() or '..' in candidate.parts or '\\' in path
__all__ = ['BACKEND', 'COMMON_CONTRACT_BYTES', 'COMMON_CONTRACT_SHA256', 'COMMON_CONTRACT_VERSION', 'CONTRACT_VERSION', 'DERIVED_KINDS', 'EFFECT_STATUSES', 'ERROR_PRECEDENCE', 'EXECUTION_AUDIT_SCHEMA_VERSION', 'OBSERVABILITY', 'OPTIONAL_BRAIN_LIFECYCLE', 'POLICY_MODES', 'RECORD_TYPES', 'ROLLOUT_ARMS', 'ROLLOUT_PURPOSES', 'SCHEMA_VERSIONS', 'SEED_NAMESPACES', 'STATE_SCHEMA_VERSION', 'SchemaValidationError', 'TASK_BANK_HASH_SCHEMA_VERSION', 'TASK_SET_HASH_SCHEMA_VERSION', 'TASK_SET_SCHEMA_VERSION', 'TrajectoryExportError', 'compiled_common_contract', 'compute_run_identity_hash', 'load_json_strict', 'load_jsonl_strict', 'strict_json_loads', 'validate_common_contract_fixture', 'validate_asset_cache_manifest', 'validate_episode_manifest', 'validate_event_payload', 'validate_budget_state', 'validate_cost_state', 'validate_error_info', 'validate_label_provenance', 'validate_json_value', 'validate_payload_record', 'validate_provenance_document', 'validate_quality_report', 'validate_run_manifest_minimal', 'validate_state_value', 'validate_task_set_envelope', 'validate_tool_call', 'validate_trajectory_record']
