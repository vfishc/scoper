from __future__ import annotations
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from scoper.benchmark.behavior.paths import repo_root as behavior_repo_root
from scoper.benchmark.behavior.task_spec import TaskSpecValidationError, validate_behavior_task_spec
from scoper.benchmark.behavior.export.ids import SEED_NAMESPACES, SHA256_RE, SOURCE_SPLIT_RE, canonical_json_bytes, derive_all_seeds, json_deep_copy, sha256_bytes, sha256_file, sha256_json, strict_json_load, strict_json_loads, validate_json_value, validate_run_id
from scoper.benchmark.behavior.export.task_identity import normalize_behavior_task_for_identity, validate_resolved_protocol
from scoper.benchmark.behavior.export.public_entity_binding import canonical_public_handles, validate_public_entity_binding_delta
from scoper.benchmark.behavior.export.types import BundleStructuralError, BundleValidationResult, TaskBankEntry, TaskBankValidationResult, TrajectoryConfigError
CONTRACT_VERSION = 'trajectory-contract.v1'
COMMON_CONTRACT_VERSION = 'trajectory-common-contract.v2'
TASK_BANK_VERSION = 'eb-behavior-task-bank.v1'
TASK_BANK_HASH_VERSION = 'task-bank-hash.v1'
EPISODE_MANIFEST_VERSION = 'episode-manifest.v1'
EVENT_INDEX_VERSION = 'trajectory-record.v1'
PAYLOAD_VERSION = 'trajectory-payload.v1'
QUALITY_VERSION = 'trajectory-quality.v1'
RECORD_TYPES = ('episode_start', 'decision_input', 'action_start', 'low_level_action', 'stream_observation', 'stream_decision', 'intervention', 'action_end', 'verification', 'memory_update', 'teacher_correction', 'episode_end')
ROLLOUT_PURPOSES = ('admission', 'training', 'dev_eval', 'test_eval', 'dagger')
ROLLOUT_ARMS = ('clean', 'disturbance', 'failure', 'mixed', 'learner')
POLICY_MODES = ('teacher', 'oracle', 'vlm', 'method')
POLICY_SOURCES = ('teacher', 'oracle', 'recovery_oracle', 'learner', 'pending_interrupt')
INPUT_KINDS = ('brain_act', 'teacher_act', 'pending_interrupt')
LEGACY_BRAIN_ACT_COMPAT_CODE_COMMITS = frozenset({'7b1fdf58b852ecfb698c002f5c885afc32c71a64'})
DERIVED_KINDS = ('step_decision', 'stream_decision', 'verification_recovery', 'preference', 'memory_update', 'memory_reconcile', 'memory_route')
ERROR_PRECEDENCE = ('infra_error', 'evaluator_invalid', 'task_invalid', 'budget_exceeded', 'invalid_action', 'agent_failure', 'none')
EFFECT_STATUSES = ('achieved', 'not_achieved', 'partial', 'ambiguous', 'unknown', 'not_applicable')
OBSERVABILITY = ('visible', 'occluded', 'out_of_view', 'visually_ambiguous', 'frame_missing', 'unknown', 'not_applicable')
OPTIONAL_BRAIN_LIFECYCLE = ('prepare_input', 'on_action_result')
SCHEMA_VERSIONS = {'asset_cache_manifest': 'asset-cache-manifest.v1', 'benchmark_protocol': 'benchmark-protocol.v1', 'binding': 'binding-record.v1', 'common_contract': COMMON_CONTRACT_VERSION, 'contract': CONTRACT_VERSION, 'derived_index': 'derived-sample-index.v1', 'derived_manifest': 'derived-export-manifest.v1', 'episode_key': 'episode-key.v1', 'episode_manifest': EPISODE_MANIFEST_VERSION, 'event_index': EVENT_INDEX_VERSION, 'frame': 'frame-record.v1', 'memory_delta': 'memory-delta.v1', 'memory_proposer_io': 'memory-proposer-io.v1', 'method_event': 'method-event.v1', 'method_runtime': 'method-runtime.v1', 'model_io': 'model-io.v1', 'payload': PAYLOAD_VERSION, 'provenance': 'provenance-document.v1', 'quality': QUALITY_VERSION, 'run_invocation': 'run-invocation.v1', 'run_manifest': 'run-manifest.v1', 'task_snapshot': 'task-snapshot.v1', 'verifier_io': 'transition-verifier-io.v1'}
COMPILED_COMMON_CONTRACT = {'contract_version': CONTRACT_VERSION, 'derived_kinds': list(DERIVED_KINDS), 'effect_statuses': list(EFFECT_STATUSES), 'error_precedence': list(ERROR_PRECEDENCE), 'observability': list(OBSERVABILITY), 'optional_brain_lifecycle': list(OPTIONAL_BRAIN_LIFECYCLE), 'policy_modes': list(POLICY_MODES), 'record_types': list(RECORD_TYPES), 'rollout_arms': list(ROLLOUT_ARMS), 'rollout_purposes': list(ROLLOUT_PURPOSES), 'schema_version': COMMON_CONTRACT_VERSION, 'schema_versions': SCHEMA_VERSIONS, 'seed_namespaces': list(SEED_NAMESPACES)}
COMMON_CONTRACT_SHA256 = '6dc78b73dcad0bdbc60c0c53f2f9af83cc3afe49e00513d91dcd921c501d8431'
COMMON_CONTRACT_BYTES = 2011
_TASK_BANK_FIELDS = {'schema_version', 'name', 'version', 'lineage_id', 'source_split', 'task_split', 'tasks'}
_EVENT_FIELDS = {'schema_version', 'record_id', 'record_type', 'episode_id', 'sequence_index', 'step_id', 'decision_id', 'action_id', 'low_level_id', 'stream_id', 'stream_obs_id', 'intervention_id', 'model_call_id', 'logical_time_s', 'monotonic_time_ns', 'wall_time_utc', 'public_payload', 'privileged_payload'}
_PAYLOAD_FIELDS = {'schema_version', 'record_id', 'record_type', 'payload'}
_RFC3339_UTC_RE = re.compile('^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}Z$')
_PUBLIC_PAYLOAD_REQUIRED = {'episode_start': {'task_id', 'instruction', 'public_task_metadata', 'reset_frame_id', 'initial_public_observation', 'tool_specs', 'tool_schema_version', 'budget_initial', 'exposure_profile'}, 'decision_input': {'decision_id', 'input_kind', 'input_frame_ids', 'observation', 'prompt_context', 'allowed_tool_specs', 'tool_history', 'last_public_transition', 'public_working_memory_before', 'source_stream_obs_id'}, 'action_start': {'decision_id', 'requested_tool_call', 'policy_source', 'public_target_ref', 'pre_action_frame_id', 'public_memory_before', 'budget_before'}, 'low_level_action': {'parent_action_id', 'order', 'exposed_to_policy', 'public_feedback', 'frame_id'}, 'stream_observation': {'stream_id', 'stream_obs_id', 'parent_action_id', 'frame_id', 'frame_index', 'timestamp_s', 'source', 'is_keyframe', 'keyframe_reason', 'active_tool', 'active_arguments', 'public_progress', 'public_visible_catalog', 'public_memory_delta', 'recent_frame_ids'}, 'stream_decision': {'stream_obs_id', 'vlm_called', 'decision', 'interrupt_tool_call', 'decision_source', 'model_call_id', 'parse_status', 'parse_error', 'latency_seconds'}, 'intervention': {'intervention_id', 'public_notice', 'notice_exposed', 'public_evidence_frame_ids'}, 'action_end': {'post_action_frame_id', 'immediate_post_action_frame_id', 'confirmation_frame_id', 'executor_status', 'public_feedback', 'last_public_transition', 'public_memory_after', 'public_memory_delta', 'budget_after', 'cost'}, 'verification': {'verification_id', 'parent_action_id', 'evidence_frame_ids', 'model_call_id', 'predicted_effect', 'predicted_state_delta', 'confidence', 'abstained', 'recommended_tool_call', 'parse_status', 'latency_seconds', 'verifier_io_schema_version', 'verifier_checkpoint_sha256', 'outcome_probabilities', 'effect_probabilities', 'observability_probabilities', 'selected_outcome', 'selected_effects', 'selected_observability', 'calibration_sha256'}, 'memory_update': {'proposal_id', 'proposal_mode', 'base_memory_revision', 'proposer_call_id', 'proposer_checkpoint_sha256', 'fact_id', 'operation', 'fact_type', 'public_entity_ref', 'previous_value', 'proposed_value', 'accepted_value', 'status', 'confidence', 'reason', 'evidence_record_ids', 'accepted_by_harness', 'exposed_to_policy', 'abstain_reason_or_null', 'candidate_pool_hash'}, 'teacher_correction': {'correction_id', 'decision_input_record_id', 'learner_tool_call', 'learner_action_end_record_id', 'input_frame_ids', 'public_context', 'public_memory'}, 'episode_end': {'final_frame_id', 'episode_status', 'done_reason', 'final_public_feedback', 'public_cost_summary'}}
_PRIVILEGED_PAYLOAD_REQUIRED = {'episode_start': {'private_task_ref', 'initial_snapshot_id', 'initial_goal', 'role_bindings', 'init_mutations', 'scene_mutations', 'placement_report', 'reset_pose_diagnostics', 'renderer_diagnostics'}, 'decision_input': {'state_snapshot_id', 'policy_source', 'policy_id', 'policy_hash', 'teacher_private_context'}, 'action_start': {'resolved_target', 'state_before_snapshot_id', 'goal_before', 'agent_pose_before', 'held_object_ids_before', 'expected_effects'}, 'low_level_action': {'low_level_id', 'requested_primitive', 'executed_primitive', 'pose_before', 'pose_after', 'held_pose', 'primitive_status', 'error', 'visible_object_ids', 'simulator_time_s', 'source'}, 'stream_observation': {'agent_pose', 'target_object_ids', 'target_visible', 'target_distance_m', 'interaction_ready', 'rendered_bbox_refs', 'segmentation_refs', 'first_visible', 'first_interactable'}, 'stream_decision': {'teacher_reference_tool_call', 'offline_label_status'}, 'intervention': {'event_kind', 'configured_type', 'trigger', 'phase', 'target_roles', 'target_object_ids', 'state_before_snapshot_id', 'requested_mutation', 'state_after_snapshot_id', 'application_status', 'pre_intervention_frame_id', 'post_intervention_frame_id', 'first_visually_observable_frame_id', 'first_visible_status', 'recovery_obligation', 'scheduler_truth'}, 'action_end': {'state_after_snapshot_id', 'normalized_state_delta', 'goal_after', 'predicate_delta', 'intended_effects', 'achieved_effects', 'intended_effect_status', 'effect_observability', 'effect_label_provenance', 'observability_provenance', 'error', 'fallback_audit', 'next_obligation'}, 'verification': {'ground_truth_action_end_record_id', 'ground_truth_effect', 'calibration_bucket'}, 'memory_update': {'private_entity_ids', 'evaluator_truth', 'truth_source', 'router_selected_at_next_decision', 'gold_memory_delta_or_null', 'observability', 'label_provenance'}, 'teacher_correction': {'teacher_tool_call', 'chosen_tool_call', 'rejected_tool_call', 'state_match', 'takeover_executed', 'reason', 'failure_taxonomy', 'confidence', 'teacher_id', 'teacher_hash'}, 'episode_end': {'final_snapshot_id', 'final_goal', 'predicate_results', 'error', 'disturbance_summary', 'recovery_summary', 'capture_integrity', 'eligibility', 'artifact_paths'}}
_PUBLIC_PAYLOAD_OPTIONAL = {'action_start': {'public_held_object_handles_before'}, 'action_end': {'public_held_object_handles_after', 'public_entity_binding_delta'}}

def common_contract_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'docs' / 'spec_assets' / 'cross_backend_common.json'

def validate_common_contract_fixture(path: Path | None=None) -> Mapping[str, Any]:
    fixture = path or common_contract_path()
    raw = fixture.read_bytes()
    if len(raw) != COMMON_CONTRACT_BYTES or sha256_bytes(raw) != COMMON_CONTRACT_SHA256:
        raise TrajectoryConfigError('common_contract_hash_mismatch', 'shared common fixture bytes/hash changed')
    parsed = strict_json_loads(raw, source=str(fixture))
    if raw != canonical_json_bytes(parsed) + b'\n':
        raise TrajectoryConfigError('common_contract_not_canonical', 'shared common fixture is not canonical JSON + newline')
    if parsed != COMPILED_COMMON_CONTRACT:
        raise TrajectoryConfigError('common_contract_compiled_mismatch', 'compiled constants differ from shared fixture')
    return parsed

def validate_task_bank_envelope(envelope: Mapping[str, Any], *, repo_root: Path, asserted_source_split: str | None=None, asserted_task_split: str | None=None) -> TaskBankValidationResult:
    if not isinstance(envelope, Mapping):
        raise TrajectoryConfigError('task_bank_schema_invalid', 'capture task bank must be a mapping')
    if set(envelope) != _TASK_BANK_FIELDS:
        missing = sorted(_TASK_BANK_FIELDS - set(envelope))
        extra = sorted(set(envelope) - _TASK_BANK_FIELDS)
        raise TrajectoryConfigError('task_bank_schema_invalid', f'task bank fields mismatch; missing={missing}, extra={extra}')
    if envelope['schema_version'] != TASK_BANK_VERSION:
        raise TrajectoryConfigError('task_bank_schema_invalid', f'schema_version must be {TASK_BANK_VERSION}')
    for field in ('name', 'version', 'lineage_id'):
        _require_non_empty_string(envelope[field], field, 'task_bank_schema_invalid')
    source_split = envelope['source_split']
    if not isinstance(source_split, str) or SOURCE_SPLIT_RE.fullmatch(source_split) is None:
        raise TrajectoryConfigError('task_bank_schema_invalid', 'source_split is not a portable split name')
    task_split = envelope['task_split']
    if task_split not in {'train', 'dev', 'test'}:
        raise TrajectoryConfigError('task_bank_schema_invalid', 'task_split must be train, dev, or test')
    if asserted_source_split is not None and asserted_source_split != source_split:
        raise TrajectoryConfigError('split_metadata_conflict', '--source-split conflicts with task bank')
    if asserted_task_split is not None and asserted_task_split != task_split:
        raise TrajectoryConfigError('split_metadata_conflict', '--task-split conflicts with task bank')
    tasks = envelope['tasks']
    if not isinstance(tasks, list) or not tasks:
        raise TrajectoryConfigError('task_bank_schema_invalid', 'tasks must be a non-empty list')
    entries: list[TaskBankEntry] = []
    seen_ids: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise TrajectoryConfigError('task_bank_schema_invalid', f'task {index} is not a mapping')
        try:
            validate_behavior_task_spec(task)
        except TaskSpecValidationError as exc:
            raise TrajectoryConfigError('task_bank_schema_invalid', f'task {index} failed validation: {exc}') from exc
        task_id = task.get('id')
        if task_id in seen_ids:
            raise TrajectoryConfigError('duplicate_task_id', f'duplicate task id {task_id!r}')
        seen_ids.add(task_id)
        source = task.get('source')
        if isinstance(source, Mapping):
            for field in ('source_split', 'split'):
                if field in source and source[field] != source_split:
                    raise TrajectoryConfigError('split_metadata_conflict', f'task {task_id!r} source.{field} conflicts with envelope')
        identity = normalize_behavior_task_for_identity(task, repo_root=repo_root)
        entries.append(TaskBankEntry(task_id=task_id, task_hash=identity.task_hash, identity=identity, task=task))
    frozen_envelope = json_deep_copy(envelope)
    portable_payload = task_bank_hash_payload(envelope, entries)
    return TaskBankValidationResult(envelope=frozen_envelope, entries=tuple(entries), task_bank_hash=sha256_json(portable_payload))

def task_bank_hash_payload(envelope: Mapping[str, Any], entries: Sequence[TaskBankEntry] | Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
    task_rows = []
    for entry in entries:
        if isinstance(entry, TaskBankEntry):
            task_rows.append({'task_id': entry.task_id, 'task_hash': entry.task_hash})
        else:
            task_rows.append({'task_id': entry['task_id'], 'task_hash': entry['task_hash']})
    task_rows.sort(key=lambda item: item['task_id'])
    return {'schema_version': TASK_BANK_HASH_VERSION, 'backend': 'scoper.benchmark.behavior', 'task_bank': {'schema_version': envelope['schema_version'], 'name': envelope['name'], 'version': envelope['version'], 'lineage_id': envelope['lineage_id'], 'source_split': envelope['source_split'], 'task_split': envelope['task_split']}, 'tasks': task_rows}

def select_task_bank_entries(entries: Sequence[TaskBankEntry], *, shard_index: int=0, shard_count: int=1, limit: int=0) -> tuple[tuple[TaskBankEntry, ...], str, str]:
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise TrajectoryConfigError('shard_invalid', 'shard_count must be at least 1')
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or (not 0 <= shard_index < shard_count):
        raise TrajectoryConfigError('shard_invalid', 'shard_index must be in [0, shard_count)')
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise TrajectoryConfigError('limit_invalid', 'limit must be a non-negative integer')
    selected = [entry for index, entry in enumerate(entries) if index % shard_count == shard_index]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise TrajectoryConfigError('empty_task_selection', 'task selection is empty')
    selection_rows = sorted(({'task_id': entry.task_id, 'task_hash': entry.task_hash} for entry in selected), key=lambda item: item['task_id'])
    return (tuple(selected), sha256_json(selection_rows), sha256_json([entry.task_id for entry in selected]))

def validate_provenance_document(value: Mapping[str, Any], *, document_type: str | None=None) -> None:
    required = {'schema_version', 'document_type', 'content_hash', 'content'}
    _require_exact_fields(value, required, 'provenance_invalid')
    if value['schema_version'] != 'provenance-document.v1':
        raise TrajectoryConfigError('provenance_invalid', 'unknown provenance schema')
    if not isinstance(value['document_type'], str) or value['document_type'] not in {'protocol', 'policy', 'capture_config'}:
        raise TrajectoryConfigError('provenance_invalid', 'invalid document_type')
    if document_type is not None and value['document_type'] != document_type:
        raise TrajectoryConfigError('provenance_invalid', 'provenance document_type mismatch')
    if not isinstance(value['content_hash'], str) or SHA256_RE.fullmatch(value['content_hash']) is None:
        raise TrajectoryConfigError('provenance_invalid', 'content_hash must be lowercase sha256')
    if not isinstance(value['content'], Mapping):
        raise TrajectoryConfigError('provenance_invalid', 'content must be a JSON object')
    if value['content_hash'] != sha256_json(value['content']):
        raise TrajectoryConfigError('provenance_hash_mismatch', 'content_hash does not match content')

def validate_run_invocation(value: Mapping[str, Any]) -> None:
    required = {'schema_version', 'invocation_id', 'started_at_utc', 'sanitized_command_argv', 'resume', 'host_runtime', 'run_identity_hash', 'recovery'}
    _require_exact_fields(value, required, 'run_invocation_invalid')
    if value['schema_version'] != 'run-invocation.v1':
        raise TrajectoryConfigError('run_invocation_invalid', 'unknown invocation schema')
    if not isinstance(value['started_at_utc'], str) or _RFC3339_UTC_RE.fullmatch(value['started_at_utc']) is None:
        raise TrajectoryConfigError('run_invocation_invalid', 'started_at_utc is not fixed RFC3339 UTC')
    if not isinstance(value['resume'], bool):
        raise TrajectoryConfigError('run_invocation_invalid', 'resume must be a boolean')
    argv = value['sanitized_command_argv']
    if not isinstance(argv, list) or any((not isinstance(item, str) for item in argv)):
        raise TrajectoryConfigError('run_invocation_invalid', 'sanitized_command_argv must be a string list')
    host = value['host_runtime']
    _require_exact_fields(host, {'hostname', 'python'}, 'run_invocation_invalid')
    for field in ('hostname', 'python'):
        if not isinstance(host[field], str) or not host[field]:
            raise TrajectoryConfigError('run_invocation_invalid', f'host_runtime.{field} must be non-empty')
    if not isinstance(value['run_identity_hash'], str) or SHA256_RE.fullmatch(value['run_identity_hash']) is None:
        raise TrajectoryConfigError('run_invocation_invalid', 'run_identity_hash must be lowercase sha256')
    recovery = value['recovery']
    if recovery is not None:
        _require_exact_fields(recovery, {'source_initializing_path', 'archived_relative_path', 'reason', 'recovered_at_utc'}, 'run_invocation_invalid')
        for field in ('source_initializing_path', 'archived_relative_path'):
            path = recovery[field]
            if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or ('..' in PurePosixPath(path).parts) or ('\\' in path):
                raise TrajectoryConfigError('run_invocation_invalid', f'recovery {field} is not a safe relative path')
        if recovery['reason'] != 'resume_initializing_recovery':
            raise TrajectoryConfigError('run_invocation_invalid', 'recovery reason is invalid')
        if not isinstance(recovery['recovered_at_utc'], str) or _RFC3339_UTC_RE.fullmatch(recovery['recovered_at_utc']) is None:
            raise TrajectoryConfigError('run_invocation_invalid', 'recovery timestamp is invalid')
    identifier_payload = {key: item for key, item in value.items() if key != 'invocation_id'}
    expected_identifier = f'inv-{sha256_json(identifier_payload)[:24]}'
    if value['invocation_id'] != expected_identifier:
        raise TrajectoryConfigError('run_invocation_invalid', 'invocation_id does not match canonical row')
    from scoper.benchmark.behavior.export.leakage import assert_safe_metadata
    assert_safe_metadata(value, repo_root=behavior_repo_root())
    validate_json_value(value)

def validate_terminal_input(value: Mapping[str, Any]) -> None:
    required = {'episode_status', 'done_reason', 'error_class', 'error_code'}
    optional = {'ended_at_utc', 'seed_usage'}
    if not isinstance(value, Mapping):
        raise TrajectoryConfigError('terminal_invalid', 'terminal must be a mapping')
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise TrajectoryConfigError('terminal_invalid', f'fields are not exact; missing={sorted(missing)}, extra={sorted(extra)}')
    _validate_terminal_scalar_fields(value, structural=False)
    if 'ended_at_utc' in value and (not isinstance(value['ended_at_utc'], str) or _RFC3339_UTC_RE.fullmatch(value['ended_at_utc']) is None):
        raise TrajectoryConfigError('terminal_invalid', 'ended_at_utc is not fixed RFC3339 UTC')
    if 'seed_usage' in value:
        validate_seed_usage_input(value['seed_usage'])

def validate_seed_usage_input(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TrajectoryConfigError('seed_usage_invalid', 'seed_usage must be a mapping')
    unknown = set(value) - set(SEED_NAMESPACES)
    if unknown:
        raise TrajectoryConfigError('seed_usage_invalid', f'unknown seed namespaces: {sorted(unknown)}')
    required = {'consumed', 'consumer', 'reason'}
    for namespace, entry in value.items():
        _require_exact_fields(entry, required, 'seed_usage_invalid')
        if not isinstance(entry['consumed'], bool):
            raise TrajectoryConfigError('seed_usage_invalid', f'{namespace}.consumed must be a boolean, not a truthy value')
        for field in ('consumer', 'reason'):
            if entry[field] is not None and (not isinstance(entry[field], str) or not entry[field]):
                raise TrajectoryConfigError('seed_usage_invalid', f'{namespace}.{field} must be null or a non-empty string')

def _validate_terminal_scalar_fields(value: Mapping[str, Any], *, structural: bool) -> None:
    error = BundleStructuralError if structural else TrajectoryConfigError
    code = 'manifest_schema_invalid' if structural else 'terminal_invalid'
    if not isinstance(value['episode_status'], str) or value['episode_status'] not in {'passed', 'failed', 'invalid'}:
        raise error(code, 'terminal episode_status is invalid')
    if not isinstance(value['error_class'], str) or value['error_class'] not in ERROR_PRECEDENCE:
        raise error(code, 'terminal error_class is invalid')
    for field in ('done_reason', 'error_code'):
        if value[field] is not None and (not isinstance(value[field], str) or not value[field]):
            raise error(code, f'terminal {field} must be null or a non-empty string')

def compute_run_identity_hash(manifest: Mapping[str, Any]) -> str:
    identity = json_deep_copy(manifest)
    identity.pop('created_at_utc', None)
    identity.pop('run_identity_hash', None)
    identity.pop('task_bank_path', None)
    identity.pop('asset_cache_manifest_path', None)
    documents = identity.get('provenance_documents')
    if isinstance(documents, Mapping):
        identity['provenance_documents'] = {key: {'sha256': item.get('sha256')} for key, item in documents.items() if isinstance(item, Mapping)}
    cache = identity.get('asset_cache')
    if isinstance(cache, list):
        identity['asset_cache'] = [{'json_pointer': item.get('json_pointer'), 'content_sha256': item.get('content_sha256')} for item in cache if isinstance(item, Mapping)]
    return sha256_json(identity)

def validate_run_manifest(manifest: Mapping[str, Any], *, run_root: Path | None=None) -> None:
    required = {'schema_version', 'contract_version', 'run_id', 'backend', 'created_at_utc', 'run_identity_hash', 'protocol_hash', 'policy_hash', 'capture_config_hash', 'common_contract_sha256', 'provenance_documents', 'code_commit', 'dirty_worktree', 'dirty_diff_hash', 'task_bank_path', 'task_bank_file_sha256', 'task_bank_hash', 'task_selection_hash', 'task_order_hash', 'task_bank_version', 'task_lineage_id', 'source_split', 'task_split', 'rollout_purpose', 'rollout_arm', 'base_seed', 'repeat_ids', 'shard_index', 'shard_count', 'runtime_contract', 'asset_cache', 'asset_cache_manifest_path', 'asset_cache_manifest_sha256'}
    _require_exact_fields(manifest, required, 'run_manifest_invalid')
    if manifest['schema_version'] != 'run-manifest.v1' or manifest['contract_version'] != CONTRACT_VERSION:
        raise TrajectoryConfigError('run_manifest_invalid', 'run manifest version is unknown')
    if manifest['backend'] != 'scoper.benchmark.behavior':
        raise TrajectoryConfigError('run_manifest_invalid', 'run manifest backend must be scoper.benchmark.behavior')
    validate_run_id(manifest['run_id'])
    if not isinstance(manifest['created_at_utc'], str) or _RFC3339_UTC_RE.fullmatch(manifest['created_at_utc']) is None:
        raise TrajectoryConfigError('run_manifest_invalid', 'created_at_utc is not fixed RFC3339 UTC')
    if manifest['common_contract_sha256'] != COMMON_CONTRACT_SHA256:
        raise TrajectoryConfigError('common_contract_hash_mismatch', 'run manifest common contract hash is invalid')
    for field in ('run_identity_hash', 'protocol_hash', 'policy_hash', 'capture_config_hash', 'dirty_diff_hash', 'task_bank_file_sha256', 'task_bank_hash', 'task_selection_hash', 'task_order_hash', 'asset_cache_manifest_sha256'):
        if not isinstance(manifest[field], str) or SHA256_RE.fullmatch(manifest[field]) is None:
            raise TrajectoryConfigError('run_manifest_invalid', f'{field} must be lowercase sha256')
    if manifest['task_bank_version'] != TASK_BANK_VERSION:
        raise TrajectoryConfigError('run_manifest_invalid', 'task bank version is invalid')
    for field in ('code_commit', 'task_lineage_id'):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise TrajectoryConfigError('run_manifest_invalid', f'{field} must be a non-empty string')
    if not isinstance(manifest['dirty_worktree'], bool):
        raise TrajectoryConfigError('run_manifest_invalid', 'dirty_worktree must be a boolean')
    if not isinstance(manifest['source_split'], str) or SOURCE_SPLIT_RE.fullmatch(manifest['source_split']) is None:
        raise TrajectoryConfigError('run_manifest_invalid', 'source_split is invalid')
    if not isinstance(manifest['rollout_purpose'], str) or manifest['rollout_purpose'] not in ROLLOUT_PURPOSES or (not isinstance(manifest['rollout_arm'], str)) or (manifest['rollout_arm'] not in ROLLOUT_ARMS):
        raise TrajectoryConfigError('run_manifest_invalid', 'rollout purpose/arm is invalid')
    if not isinstance(manifest['task_split'], str) or manifest['task_split'] not in {'train', 'dev', 'test'}:
        raise TrajectoryConfigError('run_manifest_invalid', 'task_split is invalid')
    if isinstance(manifest['base_seed'], bool) or not isinstance(manifest['base_seed'], int) or (not 0 <= manifest['base_seed'] < 2 ** 31):
        raise TrajectoryConfigError('run_manifest_invalid', 'base_seed must be in [0, 2**31)')
    repeat_ids = manifest['repeat_ids']
    if not isinstance(repeat_ids, list) or not repeat_ids or any((isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in repeat_ids)) or (repeat_ids != sorted(set(repeat_ids))):
        raise TrajectoryConfigError('run_manifest_invalid', 'repeat_ids must be sorted unique non-negative integers')
    shard_count = manifest['shard_count']
    shard_index = manifest['shard_index']
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1 or isinstance(shard_index, bool) or (not isinstance(shard_index, int)) or (not 0 <= shard_index < shard_count):
        raise TrajectoryConfigError('run_manifest_invalid', 'shard index/count are invalid')
    if not isinstance(manifest['runtime_contract'], Mapping) or not isinstance(manifest['asset_cache'], list):
        raise TrajectoryConfigError('run_manifest_invalid', 'runtime_contract/asset_cache types are invalid')
    task_bank_path = manifest['task_bank_path']
    if not isinstance(task_bank_path, str) or not task_bank_path or task_bank_path.startswith(('/', '~/', 'file:')) or re.match('^[A-Za-z]:[\\\\/]', task_bank_path):
        raise TrajectoryConfigError('run_manifest_invalid', 'task_bank_path is not sanitized')
    if manifest['run_identity_hash'] != compute_run_identity_hash(manifest):
        raise TrajectoryConfigError('run_manifest_conflict', 'run identity hash does not match manifest')
    documents = manifest['provenance_documents']
    if not isinstance(documents, Mapping) or set(documents) != {'protocol', 'policy', 'capture_config'}:
        raise TrajectoryConfigError('run_manifest_invalid', 'provenance_documents keys are invalid')
    expected_document_paths = {'protocol': 'provenance/protocol.json', 'policy': 'provenance/policy.json', 'capture_config': 'provenance/capture_config.json'}
    for name, item in documents.items():
        _require_exact_fields(item, {'path', 'sha256'}, 'run_manifest_invalid')
        if item['path'] != expected_document_paths[name]:
            raise TrajectoryConfigError('run_manifest_invalid', f'provenance path for {name} is invalid')
        if not isinstance(item['sha256'], str) or SHA256_RE.fullmatch(item['sha256']) is None:
            raise TrajectoryConfigError('run_manifest_invalid', f'provenance hash for {name} is invalid')
    if manifest['asset_cache_manifest_path'] != '.asset_cache/manifest.json':
        raise TrajectoryConfigError('run_manifest_invalid', 'asset cache manifest path is invalid')
    expected_entries = sorted(manifest['asset_cache'], key=lambda item: (item.get('json_pointer') if isinstance(item, Mapping) else '', item.get('content_sha256') if isinstance(item, Mapping) else ''))
    if expected_entries != manifest['asset_cache']:
        raise TrajectoryConfigError('run_manifest_invalid', 'asset_cache entries must be sorted')
    for entry in manifest['asset_cache']:
        _validate_asset_cache_entry(entry)
    if run_root is not None:
        root = Path(run_root)
        from scoper.benchmark.behavior.export.leakage import assert_safe_metadata
        assert_safe_metadata(manifest, repo_root=behavior_repo_root())
        for name, item in documents.items():
            path = _safe_artifact_path(root, item['path'])
            if not path.is_file() or sha256_file(path) != item['sha256']:
                raise TrajectoryConfigError('provenance_hash_mismatch', f'run provenance {name} differs from manifest')
            document = _read_canonical_json(path)
            validate_provenance_document(document, document_type=name)
            if name == 'protocol':
                try:
                    normalized_protocol = validate_resolved_protocol(document['content'])
                except TrajectoryConfigError as exc:
                    raise TrajectoryConfigError('protocol_schema_invalid', str(exc)) from exc
                if normalized_protocol != document['content']:
                    raise TrajectoryConfigError('protocol_schema_invalid', 'resolved protocol denylists are not canonically sorted')
            expected_hash_field = {'protocol': 'protocol_hash', 'policy': 'policy_hash', 'capture_config': 'capture_config_hash'}[name]
            if document['content_hash'] != manifest[expected_hash_field]:
                raise TrajectoryConfigError('provenance_hash_mismatch', f'run {name} content hash differs from manifest identity hash')
            assert_safe_metadata(document, repo_root=behavior_repo_root())
        cache_manifest = _safe_artifact_path(root, manifest['asset_cache_manifest_path'])
        if not cache_manifest.is_file() or sha256_file(cache_manifest) != manifest['asset_cache_manifest_sha256']:
            raise TrajectoryConfigError('asset_cache_hash_mismatch', 'asset cache manifest differs from run manifest')
        cache_document = _read_canonical_json(cache_manifest)
        expected_cache_fields = {'schema_version', 'backend', 'entries', 'cache_content_hash'}
        if not isinstance(cache_document, Mapping) or set(cache_document) != expected_cache_fields or cache_document['schema_version'] != 'asset-cache-manifest.v1' or (cache_document['backend'] != 'scoper.benchmark.behavior') or (not isinstance(cache_document['entries'], list)) or (cache_document['cache_content_hash'] != sha256_json(cache_document['entries'])) or (cache_document['entries'] != manifest['asset_cache']):
            raise TrajectoryConfigError('asset_cache_hash_mismatch', 'asset cache manifest content is invalid')
        disk_entries = sorted(cache_document['entries'], key=lambda item: (item.get('json_pointer'), item.get('content_sha256')))
        if disk_entries != cache_document['entries']:
            raise TrajectoryConfigError('asset_cache_hash_mismatch', 'asset cache entries are not sorted')
        for entry in cache_document['entries']:
            try:
                _validate_asset_cache_entry(entry)
            except TrajectoryConfigError as exc:
                raise TrajectoryConfigError('asset_cache_hash_mismatch', str(exc)) from exc
            asset_path = _safe_artifact_path(root, entry['relative_path'])
            if not asset_path.is_file() or asset_path.stat().st_size != entry['bytes'] or sha256_file(asset_path) != entry['content_sha256']:
                raise TrajectoryConfigError('asset_cache_hash_mismatch', 'asset cache bytes do not match entry')
            parsed_asset = strict_json_load(asset_path)
            if asset_path.read_bytes() != canonical_json_bytes(parsed_asset):
                raise TrajectoryConfigError('asset_cache_hash_mismatch', 'cached JSON is not canonical bytes')
    validate_json_value(manifest)

def validate_event_envelope(value: Mapping[str, Any], *, expected_sequence: int | None=None) -> None:
    if set(value) != _EVENT_FIELDS:
        raise BundleStructuralError('event_schema_invalid', 'event index fields are not exact')
    if value['schema_version'] != EVENT_INDEX_VERSION or value['record_type'] not in RECORD_TYPES:
        raise BundleStructuralError('event_schema_invalid', 'unknown event schema or record_type')
    sequence = value['sequence_index']
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise BundleStructuralError('event_schema_invalid', 'sequence_index must start at 1')
    if expected_sequence is not None and sequence != expected_sequence:
        raise BundleStructuralError('event_sequence_invalid', 'sequence_index is not contiguous')
    if value['record_id'] != f'rec-{sequence:08d}':
        raise BundleStructuralError('event_sequence_invalid', 'record_id does not match sequence ordinal')
    if not isinstance(value['episode_id'], str) or not value['episode_id']:
        raise BundleStructuralError('event_schema_invalid', 'episode_id is missing')
    step = value['step_id']
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise BundleStructuralError('event_schema_invalid', 'step_id must be a non-negative integer')
    for key in ('decision_id', 'action_id', 'low_level_id', 'stream_id', 'stream_obs_id', 'intervention_id', 'model_call_id'):
        if value[key] is not None and (not isinstance(value[key], str) or not value[key]):
            raise BundleStructuralError('event_schema_invalid', f'{key} must be string or null')
    for key in ('logical_time_s', 'monotonic_time_ns'):
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)) or value[key] < 0:
            raise BundleStructuralError('event_schema_invalid', f'{key} must be non-negative')
    if _RFC3339_UTC_RE.fullmatch(value['wall_time_utc'] or '') is None:
        raise BundleStructuralError('event_schema_invalid', 'wall_time_utc is not fixed RFC3339 UTC')
    if not isinstance(value['public_payload'], bool) or not isinstance(value['privileged_payload'], bool):
        raise BundleStructuralError('event_schema_invalid', 'payload presence flags must be booleans')
    id_patterns = {'decision_id': 'dec-\\d{6}', 'action_id': 'act-\\d{6}', 'low_level_id': 'll-\\d{6}-\\d{6}', 'stream_id': 'stream-\\d{6}', 'stream_obs_id': 'sobs-\\d{6}-\\d{6}', 'intervention_id': 'int-\\d{6}', 'model_call_id': 'mcall-\\d{8}'}
    for key, pattern in id_patterns.items():
        if value[key] is not None and re.fullmatch(pattern, value[key]) is None:
            raise BundleStructuralError('event_schema_invalid', f'{key} has invalid v1 format')
    record_type = value['record_type']
    if record_type in {'episode_start', 'episode_end'}:
        if step != 0 or any((value[key] is not None for key in id_patterns)):
            raise BundleStructuralError('event_schema_invalid', 'episode boundary ids/step are invalid')
    elif record_type == 'decision_input':
        if step < 1 or value['decision_id'] != f'dec-{step:06d}':
            raise BundleStructuralError('event_schema_invalid', 'decision_input id does not match step')
    elif record_type in {'action_start', 'action_end'}:
        if step < 1 or value['action_id'] != f'act-{step:06d}':
            raise BundleStructuralError('event_schema_invalid', f'{record_type} action id does not match step')
    elif record_type == 'low_level_action':
        if step < 1 or value['action_id'] != f'act-{step:06d}' or value['low_level_id'] is None:
            raise BundleStructuralError('event_schema_invalid', 'low_level_action ids do not match step')
    elif record_type in {'stream_observation', 'stream_decision'}:
        if step < 1 or value['action_id'] != f'act-{step:06d}' or value['stream_id'] != f'stream-{step:06d}' or (value['stream_obs_id'] is None):
            raise BundleStructuralError('event_schema_invalid', f'{record_type} ids do not match step')
    elif record_type == 'intervention' and value['intervention_id'] is None:
        raise BundleStructuralError('event_schema_invalid', 'intervention record requires intervention_id')

def validate_payload_envelope(value: Mapping[str, Any], *, privacy: str) -> None:
    if set(value) != _PAYLOAD_FIELDS or value['schema_version'] != PAYLOAD_VERSION:
        raise BundleStructuralError('payload_schema_invalid', 'payload envelope fields/schema are invalid')
    record_type = value['record_type']
    required_by_type = _PUBLIC_PAYLOAD_REQUIRED if privacy == 'public' else _PRIVILEGED_PAYLOAD_REQUIRED
    if record_type not in required_by_type or not isinstance(value['payload'], Mapping):
        raise BundleStructuralError('payload_schema_invalid', 'payload record_type or mapping is invalid')
    payload_keys = set(value['payload'])
    missing = required_by_type[record_type] - payload_keys
    optional = _PUBLIC_PAYLOAD_OPTIONAL.get(record_type, set()) if privacy == 'public' else set()
    extra = payload_keys - required_by_type[record_type] - optional - {'extensions'}
    if missing or extra:
        raise BundleStructuralError('payload_schema_invalid', f'{privacy} {record_type} payload missing={sorted(missing)} extra={sorted(extra)}')
    if 'extensions' in value['payload'] and (not isinstance(value['payload']['extensions'], Mapping)):
        raise BundleStructuralError('payload_schema_invalid', 'extensions must be a mapping')
    if 'policy_source' in value['payload'] and value['payload']['policy_source'] not in POLICY_SOURCES:
        raise BundleStructuralError('payload_schema_invalid', 'policy_source is not a trajectory v1 enum')
    if privacy == 'public' and record_type == 'action_start':
        payload = value['payload']
        if 'public_held_object_handles_before' in payload:
            handles = canonical_public_handles(payload['public_held_object_handles_before'])
            if handles is None or list(handles) != payload['public_held_object_handles_before']:
                raise BundleStructuralError('payload_schema_invalid', 'public held-before handles are not canonical')
    if privacy == 'public' and record_type == 'action_end':
        payload = value['payload']
        if 'public_held_object_handles_after' in payload:
            handles = canonical_public_handles(payload['public_held_object_handles_after'])
            if handles is None or list(handles) != payload['public_held_object_handles_after']:
                raise BundleStructuralError('payload_schema_invalid', 'public held-after handles are not canonical')
        if 'public_entity_binding_delta' in payload:
            if 'public_held_object_handles_after' not in payload:
                raise BundleStructuralError('payload_schema_invalid', 'public entity binding lacks held-after evidence')
            try:
                delta = validate_public_entity_binding_delta(payload['public_entity_binding_delta'])
            except TrajectoryConfigError as exc:
                raise BundleStructuralError('payload_schema_invalid', str(exc)) from exc
            if delta['source'] != 'action_end_unique_held_delta' or delta['public_alias'] not in payload['public_held_object_handles_after']:
                raise BundleStructuralError('payload_schema_invalid', 'public entity binding disagrees with action-end held evidence')
    validate_json_value(value)

def build_artifact_table(bundle_root: Path) -> list[Mapping[str, Any]]:
    root = Path(bundle_root)
    artifacts: list[Mapping[str, Any]] = []
    for path in sorted(root.rglob('*'), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise BundleStructuralError('artifact_symlink_forbidden', f'artifact is a symlink: {path}')
        if not path.is_file() or path.name in {'manifest.json', 'manifest.json.tmp', 'capture_state.json'}:
            continue
        relative = path.relative_to(root).as_posix()
        privacy = _artifact_privacy(relative)
        media_type = _media_type(path)
        record_count = None
        if path.suffix == '.jsonl':
            record_count = _count_jsonl_records(path)
        artifacts.append({'path': relative, 'media_type': media_type, 'privacy': privacy, 'bytes': path.stat().st_size, 'sha256': sha256_file(path), 'record_count': record_count})
    return artifacts

def validate_episode_bundle(bundle_root: Path, *, expected_episode_key: str | None=None, expected_episode_id: str | None=None, expected_run_id: str | None=None) -> BundleValidationResult:
    root = Path(bundle_root)
    manifest_path = root / 'manifest.json'
    if not root.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
        raise BundleStructuralError('manifest_missing', 'final episode manifest is missing')
    manifest = _read_canonical_json(manifest_path)
    _validate_episode_manifest_mapping(manifest)
    if expected_episode_key is not None and manifest['episode_key'] != expected_episode_key:
        raise BundleStructuralError('episode_conflict', 'episode key differs from requested key')
    if expected_episode_id is not None and manifest['episode_id'] != expected_episode_id:
        raise BundleStructuralError('episode_conflict', 'episode id differs from requested id')
    if expected_run_id is not None and manifest['run_id'] != expected_run_id:
        raise BundleStructuralError('episode_conflict', 'run id differs from requested run')
    actual_artifacts = build_artifact_table(root)
    if actual_artifacts != manifest['artifacts']:
        raise BundleStructuralError('artifact_table_mismatch', 'manifest artifact table differs from disk')
    content_hash = sha256_json(actual_artifacts)
    if manifest['bundle_content_sha256'] != content_hash:
        raise BundleStructuralError('bundle_hash_mismatch', 'bundle_content_sha256 does not match artifacts')
    _validate_raw_records(root, manifest)
    _validate_static_documents(root, manifest)
    _validate_quality_document(root, manifest)
    return BundleValidationResult(episode_id=manifest['episode_id'], episode_key=manifest['episode_key'], manifest_sha256=sha256_file(manifest_path), bundle_content_sha256=content_hash, structural_valid=True, quality_valid=bool(manifest['quality_summary']['valid']), errors=tuple(manifest['quality_summary']['errors']), warnings=tuple(manifest['quality_summary']['warnings']))

def validate_episode_bundle_candidate(bundle_root: Path, manifest: Mapping[str, Any]) -> None:
    root = Path(bundle_root)
    _validate_episode_manifest_mapping(manifest)
    artifacts = build_artifact_table(root)
    if artifacts != manifest['artifacts'] or sha256_json(artifacts) != manifest['bundle_content_sha256']:
        raise BundleStructuralError('artifact_table_mismatch', 'candidate artifact table/hash differs from disk')
    _validate_raw_records(root, manifest)
    _validate_static_documents(root, manifest)
    _validate_quality_document(root, manifest)

def validate_raw_episode_files(bundle_root: Path, *, episode_id: str, task_hash: str) -> None:
    root = Path(bundle_root)
    _validate_raw_records(root, {'episode_id': episode_id})
    _validate_static_documents(root, {'task': {'hash': task_hash}})

def _validate_episode_manifest_mapping(manifest: Mapping[str, Any]) -> None:
    required = {'schema_version', 'contract_version', 'bundle_status', 'backend', 'run_id', 'episode_id', 'episode_key', 'episode_key_payload', 'task', 'scene', 'rollout', 'eligibility', 'seeds', 'versions', 'timing', 'counts', 'terminal', 'artifacts', 'bundle_content_sha256', 'quality_summary'}
    if set(manifest) != required:
        raise BundleStructuralError('manifest_schema_invalid', 'episode manifest top-level fields are not exact')
    if manifest['schema_version'] != EPISODE_MANIFEST_VERSION or manifest['contract_version'] != CONTRACT_VERSION:
        raise BundleStructuralError('manifest_schema_invalid', 'episode manifest version is unknown')
    if manifest['bundle_status'] != 'complete' or manifest['backend'] != 'scoper.benchmark.behavior':
        raise BundleStructuralError('manifest_schema_invalid', 'bundle is not a complete BEHAVIOR bundle')
    if not isinstance(manifest['artifacts'], list):
        raise BundleStructuralError('manifest_schema_invalid', 'artifacts must be a list')
    paths = [item.get('path') for item in manifest['artifacts'] if isinstance(item, Mapping)]
    if len(paths) != len(manifest['artifacts']) or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise BundleStructuralError('manifest_schema_invalid', 'artifact paths must be unique and sorted')
    if not isinstance(manifest['bundle_content_sha256'], str) or SHA256_RE.fullmatch(manifest['bundle_content_sha256']) is None:
        raise BundleStructuralError('manifest_schema_invalid', 'bundle content hash is invalid')
    _require_fields(manifest['quality_summary'], {'valid', 'errors', 'warnings'}, 'manifest_schema_invalid', structural=True)
    _require_fields(manifest['eligibility'], {'train_eligible', 'admission_eligible', 'exclusion_reasons'}, 'manifest_schema_invalid', structural=True)
    _require_exact_fields(manifest['terminal'], {'episode_status', 'done_reason', 'error_class', 'error_code'}, 'manifest_schema_invalid', structural=True)
    _validate_terminal_scalar_fields(manifest['terminal'], structural=True)
    if manifest['episode_key_payload'].get('task_hash') != manifest['task'].get('hash'):
        raise BundleStructuralError('manifest_identity_mismatch', 'task hash differs from episode key payload')
    from scoper.benchmark.behavior.export.ids import EpisodeKeyInput, build_episode_identity
    try:
        key_input = EpisodeKeyInput(**manifest['episode_key_payload'])
        expected_key, expected_id = build_episode_identity(key_input)
    except (TypeError, TrajectoryConfigError) as exc:
        raise BundleStructuralError('manifest_identity_mismatch', f'EpisodeKey payload is invalid: {exc}') from exc
    if (manifest['episode_key'], manifest['episode_id']) != (expected_key, expected_id):
        raise BundleStructuralError('manifest_identity_mismatch', 'EpisodeKey/id do not match payload')
    nested_exact = {'task': {'id', 'hash', 'family', 'template_id', 'source_split', 'task_split', 'lineage_id', 'task_bank_hash'}, 'scene': {'scene_id', 'layout_id', 'source_json_content_sha256', 'asset_version'}, 'rollout': {'purpose', 'arm', 'execution_profile', 'repeat_id'}, 'eligibility': {'train_eligible', 'admission_eligible', 'exclusion_reasons'}, 'timing': {'started_at_utc', 'ended_at_utc', 'elapsed_seconds'}, 'counts': {'records', 'actions', 'low_level_actions', 'stream_observations', 'frames', 'interventions', 'model_calls', 'prompt_tokens', 'completion_tokens'}, 'terminal': {'episode_status', 'done_reason', 'error_class', 'error_code'}, 'quality_summary': {'valid', 'errors', 'warnings'}}
    for field, expected in nested_exact.items():
        if not isinstance(manifest[field], Mapping) or set(manifest[field]) != expected:
            raise BundleStructuralError('manifest_schema_invalid', f'manifest {field} fields are not exact')
    if manifest['task']['task_split'] not in {'train', 'dev', 'test'}:
        raise BundleStructuralError('manifest_schema_invalid', 'manifest task_split is invalid')
    if manifest['task']['id'] != manifest['episode_key_payload']['task_id'] or manifest['task']['source_split'] != manifest['episode_key_payload']['source_split'] or manifest['scene']['scene_id'] != manifest['episode_key_payload']['scene_id'] or (manifest['scene']['layout_id'] != manifest['episode_key_payload']['layout_id']) or (manifest['rollout']['arm'] != manifest['episode_key_payload']['rollout_arm']) or (manifest['rollout']['repeat_id'] != manifest['episode_key_payload']['repeat_id']):
        raise BundleStructuralError('manifest_identity_mismatch', 'manifest identity fields differ from EpisodeKey payload')
    if manifest['rollout']['purpose'] not in ROLLOUT_PURPOSES or manifest['rollout']['arm'] not in ROLLOUT_ARMS:
        raise BundleStructuralError('manifest_schema_invalid', 'manifest rollout purpose/arm is invalid')
    if manifest['rollout']['execution_profile'] != 'semantic_compatibility':
        raise BundleStructuralError('manifest_schema_invalid', 'manifest execution profile is invalid')
    if _RFC3339_UTC_RE.fullmatch(manifest['timing']['started_at_utc'] or '') is None or _RFC3339_UTC_RE.fullmatch(manifest['timing']['ended_at_utc'] or '') is None or isinstance(manifest['timing']['elapsed_seconds'], bool) or (not isinstance(manifest['timing']['elapsed_seconds'], (int, float))) or (manifest['timing']['elapsed_seconds'] < 0):
        raise BundleStructuralError('manifest_schema_invalid', 'manifest timing fields are invalid')
    if set(manifest['seeds']) != {'base', *SEED_NAMESPACES}:
        raise BundleStructuralError('manifest_schema_invalid', 'manifest seeds are incomplete')
    if isinstance(manifest['seeds']['base'], bool) or not isinstance(manifest['seeds']['base'], int):
        raise BundleStructuralError('manifest_schema_invalid', 'base seed must be integer')
    for namespace in SEED_NAMESPACES:
        seed = manifest['seeds'][namespace]
        _require_exact_fields(seed, {'value', 'consumed', 'consumer', 'reason'}, 'manifest_schema_invalid', structural=True)
    if manifest['seeds']['base'] != manifest['episode_key_payload']['base_seed']:
        raise BundleStructuralError('manifest_identity_mismatch', 'manifest base seed differs from EpisodeKey')
    expected_seeds = derive_all_seeds(manifest['episode_key'])
    for namespace, expected_seed in expected_seeds.items():
        seed = manifest['seeds'][namespace]
        if isinstance(seed['value'], bool) or not isinstance(seed['value'], int) or seed['value'] != expected_seed or (not isinstance(seed['consumed'], bool)) or (seed['consumer'] is not None and (not isinstance(seed['consumer'], str) or not seed['consumer'])) or (seed['reason'] is not None and (not isinstance(seed['reason'], str) or not seed['reason'])):
            raise BundleStructuralError('manifest_identity_mismatch', f'manifest seed {namespace} is invalid')
    if not isinstance(manifest['versions'], Mapping) or set(manifest['versions']) != {'common_contract', 'protocol', 'tool_schema', 'policy', 'capture', 'code', 'renderer', 'simulator'}:
        raise BundleStructuralError('manifest_schema_invalid', 'manifest versions fields are invalid')
    if manifest['versions']['common_contract'] != {'schema_version': COMMON_CONTRACT_VERSION, 'sha256': COMMON_CONTRACT_SHA256}:
        raise BundleStructuralError('manifest_schema_invalid', 'manifest common contract version/hash is invalid')
    for field in ('protocol', 'policy', 'capture'):
        if not isinstance(manifest['versions'][field], str) or SHA256_RE.fullmatch(manifest['versions'][field]) is None:
            raise BundleStructuralError('manifest_schema_invalid', f'manifest version hash {field} is invalid')
    validate_json_value(manifest)

def _validate_raw_records(root: Path, manifest: Mapping[str, Any]) -> None:
    index = _read_jsonl(root / 'trajectory.jsonl')
    public = _read_jsonl(root / 'public' / 'trajectory.jsonl')
    privileged = _read_jsonl(root / 'privileged' / 'trajectory.jsonl')
    legacy_policy_source = _legacy_policy_source_compatibility(root, manifest)
    if not index:
        raise BundleStructuralError('records_missing', 'trajectory index is empty')
    public_by_id: dict[str, Mapping[str, Any]] = {}
    private_by_id: dict[str, Mapping[str, Any]] = {}
    for row in public:
        validate_payload_envelope(_policy_source_validation_view(row, legacy_policy_source), privacy='public')
        if row['record_id'] in public_by_id:
            raise BundleStructuralError('duplicate_record_id', 'duplicate public payload record id')
        public_by_id[row['record_id']] = row
    for row in privileged:
        validate_payload_envelope(_policy_source_validation_view(row, legacy_policy_source), privacy='privileged')
        if row['record_id'] in private_by_id:
            raise BundleStructuralError('duplicate_record_id', 'duplicate privileged payload record id')
        private_by_id[row['record_id']] = row
    previous_logical = -1.0
    previous_monotonic = -1
    seen_record_ids: set[str] = set()
    action_types: dict[str, list[str]] = {}
    for sequence, row in enumerate(index, 1):
        validate_event_envelope(row, expected_sequence=sequence)
        if row['episode_id'] != manifest['episode_id']:
            raise BundleStructuralError('reference_episode_mismatch', 'event belongs to another episode')
        if row['record_id'] in seen_record_ids:
            raise BundleStructuralError('duplicate_record_id', 'duplicate event record id')
        seen_record_ids.add(row['record_id'])
        if row['logical_time_s'] < previous_logical or row['monotonic_time_ns'] < previous_monotonic:
            raise BundleStructuralError('event_time_invalid', 'event time is not non-decreasing')
        previous_logical = row['logical_time_s']
        previous_monotonic = row['monotonic_time_ns']
        if row['public_payload'] != (row['record_id'] in public_by_id):
            raise BundleStructuralError('payload_presence_mismatch', 'public payload flag does not match sidecar')
        if row['privileged_payload'] != (row['record_id'] in private_by_id):
            raise BundleStructuralError('payload_presence_mismatch', 'privileged payload flag does not match sidecar')
        for sidecar in (public_by_id.get(row['record_id']), private_by_id.get(row['record_id'])):
            if sidecar is not None and sidecar['record_type'] != row['record_type']:
                raise BundleStructuralError('payload_type_mismatch', 'sidecar record_type differs from index')
        if row['action_id'] is not None:
            action_types.setdefault(row['action_id'], []).append(row['record_type'])
    if index[0]['record_type'] != 'episode_start' or index[-1]['record_type'] != 'episode_end':
        raise BundleStructuralError('record_order_invalid', 'episode_start/end boundaries are invalid')
    counts = Counter((row['record_type'] for row in index))
    if counts['episode_start'] != 1 or counts['episode_end'] != 1:
        raise BundleStructuralError('record_cardinality_invalid', 'episode boundary cardinality is invalid')
    decision_counts = Counter((row['decision_id'] for row in index if row['record_type'] == 'decision_input'))
    if any((identifier is None or count != 1 for identifier, count in decision_counts.items())):
        raise BundleStructuralError('record_cardinality_invalid', 'decision_id cardinality is invalid')
    observation_counts = Counter((row['stream_obs_id'] for row in index if row['record_type'] == 'stream_observation'))
    if any((identifier is None or count != 1 for identifier, count in observation_counts.items())):
        raise BundleStructuralError('record_cardinality_invalid', 'stream_obs_id cardinality is invalid')
    for action, record_types in action_types.items():
        if record_types.count('action_start') != 1 or record_types.count('action_end') != 1:
            raise BundleStructuralError('record_cardinality_invalid', f'action {action} is not closed')
        if record_types.index('action_start') > record_types.index('action_end'):
            raise BundleStructuralError('record_order_invalid', f'action {action} ends before it starts')
    decision_positions = {row['decision_id']: position for position, row in enumerate(index) if row['record_type'] == 'decision_input'}
    public_by_record = {row['record_id']: row for row in public}
    private_by_record = {row['record_id']: row for row in privileged}
    decision_policy_sources: dict[str, str] = {}
    for row in index:
        if row['record_type'] != 'decision_input':
            continue
        public_row = public_by_record.get(row['record_id'])
        private_row = private_by_record.get(row['record_id'])
        if public_row is None or private_row is None:
            continue
        public_view = _policy_source_validation_view(public_row, legacy_policy_source)
        private_view = _policy_source_validation_view(private_row, legacy_policy_source)
        input_kind = public_view['payload'].get('input_kind')
        source = private_view['payload'].get('policy_source')
        if input_kind not in INPUT_KINDS:
            raise BundleStructuralError('payload_schema_invalid', 'decision input_kind is invalid')
        if (input_kind == 'pending_interrupt') != (source == 'pending_interrupt'):
            raise BundleStructuralError('payload_schema_invalid', 'pending interrupt input_kind/policy_source disagree')
        decision_policy_sources[row['decision_id']] = source
    for position, row in enumerate(index):
        if row['record_type'] != 'action_start':
            continue
        decision = row['decision_id']
        public_row = public_by_record.get(row['record_id'])
        public_view = _policy_source_validation_view(public_row, legacy_policy_source) if public_row is not None else None
        if decision not in decision_positions or decision_positions[decision] >= position or public_row is None or (public_row['payload']['decision_id'] != decision) or (public_view['payload']['policy_source'] != decision_policy_sources.get(decision)):
            raise BundleStructuralError('record_order_invalid', 'action_start lacks an earlier matching decision_input')
    stream_counts = Counter((row['stream_obs_id'] for row in index if row['record_type'] == 'stream_decision'))
    for stream_obs in {row['stream_obs_id'] for row in index if row['record_type'] == 'stream_observation'}:
        if stream_counts[stream_obs] != 1:
            raise BundleStructuralError('record_cardinality_invalid', 'stream observation lacks exactly one decision')
    frames = _read_optional_jsonl(root / 'public' / 'frames.jsonl')
    frame_ids = _validate_frame_records(root, manifest['episode_id'], frames)
    states = _read_optional_jsonl(root / 'privileged' / 'state_snapshots.jsonl')
    _validate_state_snapshots(manifest['episode_id'], states)
    state_ids = {row.get('snapshot_id') for row in states if isinstance(row, Mapping)}
    model_calls = _read_optional_jsonl(root / 'public' / 'model_io.jsonl')
    _validate_model_calls(manifest['episode_id'], model_calls)
    model_ids = {row.get('model_call_id') for row in model_calls if isinstance(row, Mapping)}
    annotations = _read_optional_jsonl(root / 'privileged' / 'frame_annotations.jsonl')
    _validate_frame_annotations(manifest['episode_id'], annotations, frame_ids)
    replay_states = _read_optional_jsonl(root / 'privileged' / 'render_replay_states.jsonl')
    _validate_render_replay_states(manifest['episode_id'], replay_states, frame_ids)
    bindings = _read_optional_jsonl(root / 'privileged' / 'binding_log.jsonl')
    _validate_binding_records(manifest['episode_id'], bindings)
    all_public_values = list(public_by_id.values()) + frames + model_calls
    from scoper.benchmark.behavior.export.leakage import assert_no_public_leakage
    episode_start_public = next((row for row in public_by_id.values() if row.get('record_type') == 'episode_start'), None)
    exposure_profile = episode_start_public['payload'].get('exposure_profile') if episode_start_public is not None else None
    if exposure_profile not in {'visual_minimal', 'structured_agent', 'privileged_oracle'}:
        raise BundleStructuralError('exposure_profile_invalid', 'episode exposure profile is invalid')
    for row in all_public_values:
        assert_no_public_leakage(row, exposure_profile=exposure_profile, allow_privileged_oracle=exposure_profile == 'privileged_oracle')
    _validate_references(index, [*public, *privileged, *frames, *model_calls, *states, *annotations, *replay_states, *bindings], frame_ids, state_ids, model_ids)
    _validate_manifest_counts(manifest, index, frames, model_calls)

def _legacy_policy_source_compatibility(root: Path, manifest: Mapping[str, Any]) -> str | None:
    versions = manifest.get('versions')
    code_commit = versions.get('code') if isinstance(versions, Mapping) else None
    if code_commit not in LEGACY_BRAIN_ACT_COMPAT_CODE_COMMITS:
        return None
    policy_path = root.parents[1] / 'provenance' / 'policy.json'
    if not policy_path.is_file():
        return None
    policy_document = strict_json_load(policy_path)
    policy = policy_document.get('content') if isinstance(policy_document, Mapping) else None
    if not isinstance(policy, Mapping):
        return None
    if sha256_json(policy) != versions.get('policy'):
        return None
    return {'teacher': 'teacher', 'oracle': 'oracle', 'vlm': 'learner', 'method': 'learner'}.get(policy.get('mode'))

def _policy_source_validation_view(row: Mapping[str, Any], legacy_policy_source: str | None) -> Mapping[str, Any]:
    payload = row.get('payload')
    if legacy_policy_source is None or not isinstance(payload, Mapping) or payload.get('policy_source') != 'brain_act':
        return row
    normalized = json_deep_copy(row)
    normalized['payload']['policy_source'] = legacy_policy_source
    return normalized

def _validate_frame_records(root: Path, episode_id: str, rows: list[Mapping[str, Any]]) -> set[str]:
    required = {'schema_version', 'frame_id', 'episode_id', 'sequence_index', 'step_id', 'action_id', 'stream_id', 'stream_obs_id', 'frame_kind', 'source', 'logical_time_s', 'monotonic_time_ns', 'wall_time_utc', 'path', 'sha256', 'bytes', 'width', 'height', 'channels', 'dtype', 'color_space', 'encoding', 'encoding_parameters', 'camera', 'capture', 'public_visibility_summary', 'quality'}
    frame_ids: set[str] = set()
    for sequence, row in enumerate(rows, 1):
        if set(row) != required or row['schema_version'] != 'frame-record.v1':
            raise BundleStructuralError('frame_schema_invalid', 'frame record fields/schema are invalid')
        if row['sequence_index'] != sequence or row['episode_id'] != episode_id:
            raise BundleStructuralError('frame_schema_invalid', 'frame sequence/episode is invalid')
        if row['frame_id'] in frame_ids:
            raise BundleStructuralError('frame_schema_invalid', 'duplicate frame_id')
        frame_ids.add(row['frame_id'])
        if row['channels'] != 3 or row['dtype'] != 'uint8' or row['color_space'] != 'sRGB' or (row['encoding'] not in {'png', 'jpeg'}) or (not isinstance(row['width'], int)) or (not isinstance(row['height'], int)) or (row['width'] < 1) or (row['height'] < 1):
            raise BundleStructuralError('frame_schema_invalid', 'frame shape/encoding metadata is invalid')
        _require_fields(row['camera'], {'pose', 'intrinsics', 'exposure_status'}, 'frame_schema_invalid', structural=True)
        _require_fields(row['capture'], {'render_sync_status', 'renders_per_frame', 'settle_policy', 'source_frame_id'}, 'frame_schema_invalid', structural=True)
        if row['camera']['exposure_status'] not in {'exposed', 'withheld', 'unavailable'}:
            raise BundleStructuralError('frame_schema_invalid', 'camera exposure status is invalid')
        if not isinstance(row['quality'], Mapping) or not isinstance(row['quality'].get('valid'), bool):
            raise BundleStructuralError('frame_schema_invalid', 'frame quality metadata is invalid')
        path = _safe_artifact_path(root, row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise BundleStructuralError('frame_hash_mismatch', f"frame bytes/hash mismatch for {row['frame_id']}")
        from scoper.benchmark.behavior.export.quality import compute_rgb_quality, decode_image_bytes
        try:
            decoded, image_metadata = decode_image_bytes(path.read_bytes())
        except TrajectoryConfigError as exc:
            raise BundleStructuralError('frame_decode_invalid', str(exc)) from exc
        expected_format = 'PNG' if row['encoding'] == 'png' else 'JPEG'
        if image_metadata['format'] != expected_format or decoded.shape != (row['height'], row['width'], 3) or (not _quality_values_equal(compute_rgb_quality(decoded), row['quality'])):
            raise BundleStructuralError('frame_quality_mismatch', 'frame decode/shape/quality differs from record')
    return frame_ids

def _quality_values_equal(expected: Any, recorded: Any) -> bool:
    if isinstance(expected, bool) or isinstance(recorded, bool):
        return type(expected) is type(recorded) and expected == recorded
    if isinstance(expected, float) or isinstance(recorded, float):
        if not isinstance(expected, float) or not isinstance(recorded, float):
            return False
        return math.isfinite(expected) and math.isfinite(recorded) and math.isclose(expected, recorded, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(expected, Mapping) or isinstance(recorded, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(recorded, Mapping):
            return False
        return set(expected) == set(recorded) and all((_quality_values_equal(expected[key], recorded[key]) for key in expected))
    if isinstance(expected, list) or isinstance(recorded, list):
        return isinstance(expected, list) and isinstance(recorded, list) and (len(expected) == len(recorded)) and all((_quality_values_equal(left, right) for left, right in zip(expected, recorded)))
    return type(expected) is type(recorded) and expected == recorded

def validate_state_value(value: Any) -> None:
    required = {'value', 'status', 'source', 'source_detail', 'error_code'}
    if not isinstance(value, Mapping) or set(value) != required:
        raise BundleStructuralError('state_value_invalid', 'StateValue fields are not exact')
    if value['status'] not in {'known', 'unknown', 'not_applicable', 'conflict', 'error'}:
        raise BundleStructuralError('state_value_invalid', 'StateValue status is invalid')
    if value['source'] not in {'symbolic_core', 'omnigibson_state', 'renderer', 'goal_evaluator', 'scheduler', 'derived_rule', 'unavailable'}:
        raise BundleStructuralError('state_value_invalid', 'StateValue source is invalid')
    if value['status'] == 'known' and value['value'] is None:
        raise BundleStructuralError('state_value_invalid', 'known StateValue cannot contain null')
    if value['status'] not in {'known', 'conflict'} and value['value'] is not None:
        raise BundleStructuralError('state_value_invalid', 'non-known StateValue must contain null')

def _validate_state_snapshots(episode_id: str, rows: list[Mapping[str, Any]]) -> None:
    required = {'schema_version', 'snapshot_id', 'episode_id', 'step_id', 'action_id', 'record_id', 'boundary', 'scope', 'capture_status', 'agent', 'objects', 'goal_predicates', 'snapshot_errors'}
    property_names = {'open', 'toggled_on', 'dirty', 'sliced', 'cooked', 'heated', 'cooled', 'filled', 'saturated'}
    for ordinal, row in enumerate(rows, 1):
        if set(row) != required or row['schema_version'] != 'eb-behavior-state.v1':
            raise BundleStructuralError('state_snapshot_invalid', 'state snapshot fields/schema are invalid')
        if row['snapshot_id'] != f'state-{ordinal:08d}' or row['episode_id'] != episode_id:
            raise BundleStructuralError('state_snapshot_invalid', 'state snapshot ordinal/episode is invalid')
        if row['boundary'] not in {'reset', 'pre_action', 'post_action', 'intervention_before', 'intervention_after', 'episode_end'} or row['scope'] not in {'task_relevant', 'full_world'} or row['capture_status'] not in {'complete', 'partial', 'failed'}:
            raise BundleStructuralError('state_snapshot_invalid', 'state snapshot enum is invalid')
        if not isinstance(row['agent'], Mapping) or set(row['agent']) != {'pose', 'room', 'held_object_ids'}:
            raise BundleStructuralError('state_snapshot_invalid', 'state snapshot agent fields are invalid')
        for value in row['agent'].values():
            validate_state_value(value)
        if not isinstance(row['objects'], list) or not isinstance(row['goal_predicates'], list):
            raise BundleStructuralError('state_snapshot_invalid', 'state snapshot objects/goals must be lists')
        object_ids = [item.get('object_id') for item in row['objects'] if isinstance(item, Mapping)]
        if object_ids != sorted(object_ids) or len(object_ids) != len(set(object_ids)):
            raise BundleStructuralError('state_snapshot_invalid', 'snapshot objects must be sorted and unique')
        for item in row['objects']:
            if set(item) != {'object_id', 'role_names', 'category', 'pose', 'room', 'visible', 'interaction_ready', 'properties', 'relations'}:
                raise BundleStructuralError('state_snapshot_invalid', 'snapshot object fields are invalid')
            if item['role_names'] != sorted(set(item['role_names'])) or set(item['properties']) != property_names:
                raise BundleStructuralError('state_snapshot_invalid', 'snapshot roles/properties are invalid')
            for field in ('pose', 'room', 'visible', 'interaction_ready'):
                validate_state_value(item[field])
            for value in item['properties'].values():
                validate_state_value(value)
            relation_keys = []
            for relation in item['relations']:
                if set(relation) != {'predicate', 'object_id', 'value'}:
                    raise BundleStructuralError('state_snapshot_invalid', 'snapshot relation fields are invalid')
                relation_keys.append((relation['predicate'], relation['object_id']))
                validate_state_value(relation['value'])
            if relation_keys != sorted(relation_keys):
                raise BundleStructuralError('state_snapshot_invalid', 'snapshot relations are not sorted')
        for predicate in row['goal_predicates']:
            if set(predicate) != {'predicate_id', 'expression', 'value'}:
                raise BundleStructuralError('state_snapshot_invalid', 'goal predicate fields are invalid')
            validate_state_value(predicate['value'])
        if not isinstance(row['snapshot_errors'], list):
            raise BundleStructuralError('state_snapshot_invalid', 'snapshot_errors must be a list')

def _validate_model_calls(episode_id: str, rows: list[Mapping[str, Any]]) -> None:
    required = {'schema_version', 'model_call_id', 'episode_id', 'call_type', 'decision_id', 'action_id', 'stream_obs_id', 'verification_id', 'provider', 'adapter_type', 'model', 'checkpoint_hash', 'adapter_version', 'prompt_template_id', 'prompt_template_hash', 'request', 'response', 'usage', 'attempts', 'latency_seconds', 'error'}
    request_fields = {'messages', 'image_frame_ids', 'generation_config', 'exposure_profile'}
    response_fields = {'raw_text', 'parsed_output', 'parse_status', 'parse_error', 'finish_reason'}
    usage_fields = {'prompt_tokens', 'completion_tokens', 'total_tokens', 'source'}
    attempt_fields = {'attempt_index', 'started_at_utc', 'latency_seconds', 'status', 'error_type', 'status_code', 'message', 'backoff_seconds'}
    for ordinal, row in enumerate(rows, 1):
        if set(row) != required or row['schema_version'] != 'model-io.v1':
            raise BundleStructuralError('model_io_schema_invalid', 'model I/O fields/schema are invalid')
        if row['model_call_id'] != f'mcall-{ordinal:08d}' or row['episode_id'] != episode_id:
            raise BundleStructuralError('model_io_schema_invalid', 'model I/O ordinal/episode is invalid')
        if row['call_type'] not in {'step_decision', 'stream_decision', 'verification', 'memory'}:
            raise BundleStructuralError('model_io_schema_invalid', 'model I/O call_type is invalid')
        if set(row['request']) != request_fields or set(row['response']) != response_fields or set(row['usage']) != usage_fields:
            raise BundleStructuralError('model_io_schema_invalid', 'model I/O nested fields are invalid')
        if row['response']['parse_status'] not in {'ok', 'invalid', 'error', 'not_returned'}:
            raise BundleStructuralError('model_io_schema_invalid', 'model I/O parse status is invalid')
        if row['usage']['source'] not in {'provider', 'local_tokenizer', 'unavailable'}:
            raise BundleStructuralError('model_io_schema_invalid', 'model I/O usage source is invalid')
        if not isinstance(row['attempts'], list):
            raise BundleStructuralError('model_io_schema_invalid', 'model attempts must be a list')
        for attempt_index, attempt in enumerate(row['attempts'], 1):
            if set(attempt) != attempt_fields or attempt['attempt_index'] != attempt_index:
                raise BundleStructuralError('model_io_schema_invalid', 'model attempt fields/ordinal are invalid')
            if attempt['status'] not in {'succeeded', 'retryable_error', 'fatal_error'}:
                raise BundleStructuralError('model_io_schema_invalid', 'model attempt status is invalid')

def _validate_frame_annotations(episode_id: str, rows: list[Mapping[str, Any]], frame_ids: set[str]) -> None:
    required = {'schema_version', 'frame_id', 'episode_id', 'camera_pose_truth', 'camera_intrinsics_truth', 'agent_pose_truth', 'visible_object_ids', 'objects', 'visibility_source', 'segmentation_status', 'symbolic_fov_fallback', 'renderer_object_resolution', 'annotation_errors'}
    for row in rows:
        if set(row) != required or row['episode_id'] != episode_id or row['frame_id'] not in frame_ids:
            raise BundleStructuralError('frame_annotation_invalid', 'frame annotation fields/references are invalid')

def _validate_render_replay_states(episode_id: str, rows: list[Mapping[str, Any]], frame_ids: set[str]) -> None:
    from scoper.benchmark.behavior.render.particle_segmentation_replay import validate_render_replay_state
    seen: set[str] = set()
    for row in rows:
        frame_id = row.get('frame_id') if isinstance(row, Mapping) else None
        if not isinstance(frame_id, str) or frame_id not in frame_ids or frame_id in seen:
            raise BundleStructuralError('render_replay_state_invalid', 'render replay frame reference is invalid')
        try:
            validate_render_replay_state(row, episode_id=episode_id, frame_id=frame_id)
        except ValueError as exc:
            raise BundleStructuralError('render_replay_state_invalid', str(exc)) from exc
        seen.add(frame_id)

def _validate_binding_records(episode_id: str, rows: list[Mapping[str, Any]]) -> None:
    required = {'schema_version', 'binding_id', 'episode_id', 'decision_id', 'action_id', 'public_reference', 'resolver_config_hash', 'status', 'candidates', 'selected_object_id', 'resolver_path', 'error'}
    for ordinal, row in enumerate(rows, 1):
        if set(row) != required or row['schema_version'] != 'binding-record.v1':
            raise BundleStructuralError('binding_schema_invalid', 'binding record fields/schema are invalid')
        if row['binding_id'] != f'bind-{ordinal:08d}' or row['episode_id'] != episode_id:
            raise BundleStructuralError('binding_schema_invalid', 'binding ordinal/episode is invalid')
        if row['status'] not in {'resolved', 'ambiguous', 'not_found', 'invalid'}:
            raise BundleStructuralError('binding_schema_invalid', 'binding status is invalid')

def _validate_references(index: list[Mapping[str, Any]], sidecars: list[Mapping[str, Any]], frame_ids: set[str], state_ids: set[Any], model_ids: set[Any]) -> None:
    record_ids = {row['record_id'] for row in index}
    action_ids = {row['action_id'] for row in index if row['record_type'] == 'action_start'}
    decision_ids = {row['decision_id'] for row in index if row['record_type'] == 'decision_input'}
    stream_obs_ids = {row['stream_obs_id'] for row in index if row['record_type'] == 'stream_observation'}
    intervention_ids = {row['intervention_id'] for row in index if row['record_type'] == 'intervention'}
    pools = {'record': record_ids, 'frame': frame_ids, 'state': state_ids, 'model': model_ids, 'action': action_ids, 'decision': decision_ids, 'stream_obs': stream_obs_ids, 'intervention': intervention_ids}
    for value in [*index, *sidecars]:
        for scalar in _iter_scalars(value):
            kind = _reference_kind(scalar)
            if kind is not None and scalar not in pools[kind]:
                raise BundleStructuralError('dangling_reference', f'dangling {kind} reference {scalar}')

def _validate_static_documents(root: Path, manifest: Mapping[str, Any]) -> None:
    public_task = _read_canonical_json(root / 'public' / 'task.json')
    private_task = _read_canonical_json(root / 'privileged' / 'task.json')
    _require_fields(public_task, {'schema_version', 'privacy', 'task_id', 'instruction', 'public_task_metadata', 'tool_schema_version'}, 'task_snapshot_invalid', structural=True)
    _require_fields(private_task, {'schema_version', 'privacy', 'task_id', 'task_hash', 'full_task_spec', 'task_path_resolutions'}, 'task_snapshot_invalid', structural=True)
    if public_task['schema_version'] != 'task-snapshot.v1' or public_task['privacy'] != 'public':
        raise BundleStructuralError('task_snapshot_invalid', 'public task snapshot version/privacy is invalid')
    if private_task['schema_version'] != 'task-snapshot.v1' or private_task['privacy'] != 'privileged':
        raise BundleStructuralError('task_snapshot_invalid', 'private task snapshot version/privacy is invalid')
    if sha256_json(private_task['full_task_spec']) != private_task['task_hash']:
        raise BundleStructuralError('task_hash_mismatch', 'private task snapshot hash is invalid')
    if private_task['task_hash'] != manifest['task']['hash']:
        raise BundleStructuralError('task_hash_mismatch', 'private task and manifest task hashes differ')
    if private_task['task_id'] != public_task['task_id']:
        raise BundleStructuralError('task_snapshot_invalid', 'public/private task snapshot ids differ')
    if 'id' in manifest['task'] and private_task['task_id'] != manifest['task']['id']:
        raise BundleStructuralError('task_snapshot_invalid', 'task snapshot ids differ from manifest')
    full_source = private_task['full_task_spec'].get('source')
    if not isinstance(full_source, Mapping):
        raise BundleStructuralError('task_snapshot_invalid', 'private task source is invalid')
    if 'scene' in manifest:
        if full_source.get('scene_model') != manifest['scene']['scene_id'] or (full_source.get('layout_variant_id') or '') != manifest['scene']['layout_id']:
            raise BundleStructuralError('task_snapshot_invalid', 'task scene identity differs from manifest')
        _validate_task_path_resolutions(private_task, manifest)
    else:
        _validate_task_path_resolutions(private_task, None)
    if set(public_task['public_task_metadata']) - {'family', 'category', 'template', 'difficulty', 'public_handles'}:
        raise BundleStructuralError('public_task_metadata_invalid', 'public task metadata contains non-allowlisted keys')
    from scoper.benchmark.behavior.export.leakage import assert_no_public_leakage
    assert_no_public_leakage(public_task)
    for document_type in ('protocol', 'policy', 'capture_config'):
        document = _read_canonical_json(root / 'provenance' / f'{document_type}.json')
        try:
            validate_provenance_document(document, document_type=document_type)
        except TrajectoryConfigError as exc:
            raise BundleStructuralError('provenance_invalid', str(exc)) from exc
        if 'versions' in manifest:
            expected_hash = {'protocol': manifest['versions']['protocol'], 'policy': manifest['versions']['policy'], 'capture_config': manifest['versions']['capture']}[document_type]
            if document['content_hash'] != expected_hash:
                raise BundleStructuralError('provenance_hash_mismatch', f'episode {document_type} hash differs from manifest')
        if document_type == 'protocol' and 'scene' in manifest:
            assets = document['content'].get('assets') if isinstance(document['content'], Mapping) else None
            if not isinstance(assets, Mapping) or manifest['scene']['asset_version'] != {'behavior_assets': assets.get('behavior_assets_version'), 'scene_profile': assets.get('scene_profile_version')}:
                raise BundleStructuralError('provenance_invalid', 'scene asset versions differ from protocol')

def _validate_task_path_resolutions(private_task: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> None:
    resolutions = private_task['task_path_resolutions']
    if not isinstance(resolutions, list):
        raise BundleStructuralError('task_snapshot_invalid', 'task path resolutions must be a list')
    pointers = []
    source_resolution = None
    for resolution in resolutions:
        if not isinstance(resolution, Mapping) or set(resolution) != {'json_pointer', 'locator_kind', 'normalized_locator', 'content_sha256'}:
            raise BundleStructuralError('task_snapshot_invalid', 'task path resolution fields are invalid')
        if resolution['locator_kind'] not in {'repo_json', 'external_json'}:
            raise BundleStructuralError('task_snapshot_invalid', 'task path locator kind is invalid')
        if not isinstance(resolution['content_sha256'], str) or SHA256_RE.fullmatch(resolution['content_sha256']) is None:
            raise BundleStructuralError('task_snapshot_invalid', 'task path content hash is invalid')
        expected_prefix = f"repo-json://sha256/{resolution['content_sha256']}/" if resolution['locator_kind'] == 'repo_json' else f"external-json://sha256/{resolution['content_sha256']}"
        locator = resolution['normalized_locator']
        if not isinstance(locator, str) or (resolution['locator_kind'] == 'repo_json' and (not locator.startswith(expected_prefix)) or (resolution['locator_kind'] == 'external_json' and locator != expected_prefix)):
            raise BundleStructuralError('task_snapshot_invalid', 'normalized task locator is invalid')
        pointers.append(resolution['json_pointer'])
        if resolution['json_pointer'] == '/source/source_json':
            source_resolution = resolution
    if pointers != sorted(pointers) or len(pointers) != len(set(pointers)):
        raise BundleStructuralError('task_snapshot_invalid', 'task path resolutions must be sorted and unique')
    full_source = private_task['full_task_spec']['source']
    source_locator = full_source.get('source_json')
    if source_locator is None:
        if source_resolution is not None or (manifest is not None and manifest['scene']['source_json_content_sha256'] is not None):
            raise BundleStructuralError('task_snapshot_invalid', 'unexpected source_json resolution/hash')
    elif source_resolution is None or source_locator != source_resolution['normalized_locator'] or (manifest is not None and manifest['scene']['source_json_content_sha256'] != source_resolution['content_sha256']):
        raise BundleStructuralError('task_snapshot_invalid', 'source_json resolution/hash differs from task/manifest')

def _validate_quality_document(root: Path, manifest: Mapping[str, Any]) -> None:
    quality = _read_canonical_json(root / 'quality_report.json')
    required = {'schema_version', 'episode_id', 'structural_valid', 'quality_valid', 'capture_capabilities', 'checks', 'errors', 'warnings', 'exporter_eligibility', 'exclusion_reasons'}
    if set(quality) != required or quality['schema_version'] != QUALITY_VERSION:
        raise BundleStructuralError('quality_schema_invalid', 'quality report fields/schema are invalid')
    if quality['episode_id'] != manifest['episode_id'] or quality['structural_valid'] is not True:
        raise BundleStructuralError('quality_schema_invalid', 'quality report episode/structural status is invalid')
    for check in ('records', 'references', 'frames', 'leakage', 'effects', 'split', 'hashes'):
        if check not in quality['checks']:
            raise BundleStructuralError('quality_schema_invalid', f'quality check {check!r} is missing')
    summary = manifest['quality_summary']
    if summary['valid'] != quality['quality_valid'] or summary['errors'] != quality['errors'] or summary['warnings'] != quality['warnings']:
        raise BundleStructuralError('quality_summary_mismatch', 'manifest quality summary differs from report')
    if quality['exclusion_reasons'] != manifest['eligibility']['exclusion_reasons']:
        raise BundleStructuralError('quality_summary_mismatch', 'quality exclusion reasons differ from manifest')
    capture_document = _read_canonical_json(root / 'provenance' / 'capture_config.json')
    capture_content = capture_document['content']
    expected_profile = capture_content.get('profile') if isinstance(capture_content, Mapping) else None
    capabilities = quality['capture_capabilities']
    if not isinstance(capabilities, Mapping) or capabilities.get('profile') != expected_profile:
        raise BundleStructuralError('quality_schema_invalid', 'capture capabilities profile differs from provenance')
    if expected_profile not in {'minimal', 'training', 'debug'}:
        raise BundleStructuralError('quality_schema_invalid', 'capture profile is invalid')
    if not quality['quality_valid'] and manifest['eligibility']['train_eligible']:
        raise BundleStructuralError('eligibility_invalid', 'quality-invalid bundle is marked train eligible')
    _validate_manifest_eligibility(root, manifest, quality)
    _validate_admission_preflight_document(root, manifest)

def _validate_admission_preflight_document(root: Path, manifest: Mapping[str, Any]) -> None:
    path = root / 'privileged' / 'admission_preflight.json'
    terminal = manifest['terminal']
    if not path.exists():
        initial_goal_evaluator_invalid = terminal['error_code'] == 'ADMISSION_EVALUATOR_DISAGREEMENT' and _has_initial_goal_invalid_terminal_evidence(root)
        if terminal['error_code'] in {'ADMISSION_UNREACHABLE', 'ADMISSION_MISSING_REQUIRED_ASSET', 'ADMISSION_EVALUATOR_DISAGREEMENT', 'ADMISSION_INFRA_RENDERER_ERROR', 'ADMISSION_INFRA_CONTROLLER_ERROR', 'ADMISSION_INFRA_RESET_ERROR', 'ADMISSION_INFRA_CAPTURE_ERROR'} and (not initial_goal_evaluator_invalid):
            raise BundleStructuralError('admission_preflight_missing', 'admission terminal lacks diagnostic evidence')
        return
    value = _read_canonical_json(path)
    required = {'schema_version', 'stage', 'status', 'error_class', 'failure_code', 'disposition', 'cell', 'navigation_map', 'start', 'required_roles', 'evaluators', 'alignment'}
    if set(value) != required or value['schema_version'] != 'behavior-admission-preflight.v1':
        raise BundleStructuralError('admission_preflight_invalid', 'admission preflight fields/schema are invalid')
    expected_combinations = {'pass': ('none', None, 'admitted'), 'fail': ('task_invalid', 'ADMISSION_UNREACHABLE', 'rejected')}
    if value['status'] == 'invalid':
        if value['error_class'] == 'infra_error':
            valid_code = value['failure_code'] in {'ADMISSION_INFRA_RESET_ERROR', 'ADMISSION_INFRA_RENDERER_ERROR', 'ADMISSION_INFRA_CONTROLLER_ERROR', 'ADMISSION_INFRA_CAPTURE_ERROR'}
        else:
            valid_code = value['error_class'] == 'evaluator_invalid' and value['failure_code'] == 'ADMISSION_EVALUATOR_DISAGREEMENT'
        if not valid_code or value['disposition'] != 'needs_rerun':
            raise BundleStructuralError('admission_preflight_invalid', 'invalid admission classification is inconsistent')
    elif value.get('status') not in expected_combinations or (value.get('error_class'), value.get('failure_code'), value.get('disposition')) != expected_combinations[value['status']]:
        raise BundleStructuralError('admission_preflight_invalid', 'admission classification is inconsistent')
    if value['stage'] != 'reachability':
        raise BundleStructuralError('admission_preflight_invalid', 'admission preflight stage is not reachability')
    from scoper.benchmark.behavior.benchmark.admission_schema import validate_admission_cell
    try:
        validate_admission_cell(value['cell'])
    except TrajectoryConfigError as exc:
        raise BundleStructuralError('admission_preflight_invalid', str(exc)) from exc
    _validate_admission_diagnostics(value)
    if value['cell']['task_hash'] != manifest['task']['hash'] or value['cell']['scene_id'] != manifest['scene']['scene_id'] or value['cell']['layout_id'] != manifest['scene']['layout_id'] or (value['cell']['repeat_id'] != manifest['rollout']['repeat_id']) or (value['cell']['environment_seed'] != manifest['seeds']['environment']['value']) or (value['cell']['execution_profile'] != manifest['rollout']['execution_profile']) or (value['cell']['rollout_arm'] != manifest['rollout']['arm']) or (value['cell']['source_json_content_sha256'] != manifest['scene']['source_json_content_sha256']):
        raise BundleStructuralError('admission_preflight_invalid', 'admission cell differs from episode identity')
    expected_terminal = {'pass': None, 'fail': value['failure_code'], 'invalid': value['failure_code']}
    if value['status'] not in expected_terminal:
        raise BundleStructuralError('admission_preflight_invalid', 'admission status is invalid')
    if value['status'] != 'pass':
        if terminal['episode_status'] != 'invalid' or terminal['error_class'] != value['error_class'] or terminal['error_code'] != expected_terminal[value['status']] or (manifest['counts']['actions'] != 0) or (manifest['counts']['stream_observations'] != 0) or (manifest['counts']['model_calls'] != 0) or (manifest['eligibility']['train_eligible'] is not False) or (manifest['eligibility']['admission_eligible'] is not False):
            raise BundleStructuralError('admission_terminal_mismatch', 'rejected/invalid admission terminal is inconsistent')

def _has_initial_goal_invalid_terminal_evidence(root: Path) -> bool:
    for row in _read_optional_jsonl(root / 'privileged' / 'trajectory.jsonl'):
        if row.get('record_type') != 'episode_end':
            continue
        payload = row.get('payload')
        eligibility = payload.get('eligibility') if isinstance(payload, Mapping) else None
        error = payload.get('error') if isinstance(payload, Mapping) else None
        return bool(isinstance(eligibility, Mapping) and eligibility.get('stage') == 'initial_goal_false' and (eligibility.get('status') == 'invalid') and (eligibility.get('disposition') == 'needs_rerun') and isinstance(error, Mapping) and (error.get('error_class') == 'evaluator_invalid') and (error.get('error_code') == 'ADMISSION_EVALUATOR_DISAGREEMENT'))
    return False

def _validate_admission_diagnostics(value: Mapping[str, Any]) -> None:
    start = value['start']
    if not isinstance(start, Mapping) or set(start) != {'pose', 'cell', 'component', 'free'}:
        raise BundleStructuralError('admission_preflight_invalid', 'admission start diagnostics are invalid')
    if not isinstance(value['navigation_map'], Mapping) or not isinstance(value['alignment'], Mapping):
        raise BundleStructuralError('admission_preflight_invalid', 'admission map/alignment diagnostics are invalid')
    if not isinstance(value['required_roles'], list) or not isinstance(value['evaluators'], Mapping):
        raise BundleStructuralError('admission_preflight_invalid', 'admission role/evaluator diagnostics are invalid')
    role_fields = ('interaction_candidate_count', 'free_candidate_count', 'same_component_candidate_count', 'path_planned_candidate_count')
    for role in value['required_roles']:
        if not isinstance(role, Mapping) or any((field not in role for field in role_fields)):
            raise BundleStructuralError('admission_preflight_invalid', 'required role diagnostics are incomplete')
        counts = [role[field] for field in role_fields]
        if any((isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts)):
            raise BundleStructuralError('admission_preflight_invalid', 'required role counts are invalid')
        interaction, free, same_component, path_planned = counts
        if not path_planned <= same_component <= free <= interaction:
            raise BundleStructuralError('admission_preflight_invalid', 'required role counts are not monotonic')
        histogram = role.get('rejection_reason_histogram')
        samples = role.get('rejected_candidate_samples')
        if not isinstance(histogram, Mapping) or any((not isinstance(key, str) or isinstance(count, bool) or (not isinstance(count, int)) or (count < 0) for key, count in histogram.items())) or (not isinstance(samples, list)) or (len(samples) > 24):
            raise BundleStructuralError('admission_preflight_invalid', 'rejected candidate diagnostics are invalid')

def _validate_manifest_counts(manifest: Mapping[str, Any], index: list[Mapping[str, Any]], frames: list[Mapping[str, Any]], model_calls: list[Mapping[str, Any]]) -> None:
    counts = manifest.get('counts')
    if not isinstance(counts, Mapping):
        return
    prompt_tokens = [call.get('usage', {}).get('prompt_tokens') for call in model_calls if isinstance(call.get('usage'), Mapping) and isinstance(call['usage'].get('prompt_tokens'), int)]
    completion_tokens = [call.get('usage', {}).get('completion_tokens') for call in model_calls if isinstance(call.get('usage'), Mapping) and isinstance(call['usage'].get('completion_tokens'), int)]
    expected = {'records': len(index), 'actions': sum((row['record_type'] == 'action_start' for row in index)), 'low_level_actions': sum((row['record_type'] == 'low_level_action' for row in index)), 'stream_observations': sum((row['record_type'] == 'stream_observation' for row in index)), 'frames': len(frames), 'interventions': len({row['intervention_id'] for row in index if row['record_type'] == 'intervention'}), 'model_calls': len(model_calls), 'prompt_tokens': sum(prompt_tokens) if prompt_tokens else None, 'completion_tokens': sum(completion_tokens) if completion_tokens else None}
    if counts != expected:
        raise BundleStructuralError('manifest_counts_mismatch', 'manifest counts differ from raw artifacts')

def _validate_manifest_eligibility(root: Path, manifest: Mapping[str, Any], quality: Mapping[str, Any]) -> None:
    protocol_document = _read_canonical_json(root / 'provenance' / 'protocol.json')
    protocol = protocol_document['content']
    data_policy = protocol.get('data_policy') if isinstance(protocol, Mapping) else {}
    reasons = []
    task = manifest['task']
    rollout = manifest['rollout']
    terminal = manifest['terminal']
    if task['source_split'] != 'train':
        reasons.append('source_split_nontrain')
    if task['task_split'] != 'train':
        reasons.append('task_split_nontrain')
    if rollout['purpose'] not in {'admission', 'training', 'dagger'}:
        reasons.append('rollout_purpose_nontrain')
    if task['lineage_id'] in (data_policy or {}).get('nontrain_lineage_ids', []):
        reasons.append('nontrain_lineage')
    if task['hash'] in (data_policy or {}).get('nontrain_task_hashes', []):
        reasons.append('nontrain_task_hash')
    if not quality['quality_valid']:
        reasons.append('quality_invalid')
    if terminal['error_class'] in {'infra_error', 'evaluator_invalid', 'task_invalid'}:
        reasons.append(f"terminal_{terminal['error_class']}")
    public_rows = _read_jsonl(root / 'public' / 'trajectory.jsonl')
    episode_start = next((row for row in public_rows if row['record_type'] == 'episode_start'), None)
    if episode_start is not None and episode_start['payload'].get('exposure_profile') == 'privileged_oracle':
        reasons.append('privileged_oracle_exposure')
    reasons = sorted(set(reasons))
    eligibility = manifest['eligibility']
    if eligibility['exclusion_reasons'] != reasons or eligibility['train_eligible'] != (not reasons):
        raise BundleStructuralError('eligibility_invalid', 'manifest train eligibility is not derivable from source data')
    expected_admission = _derive_admission_eligibility(root, manifest, train_eligible=not reasons)
    if eligibility['admission_eligible'] is not expected_admission:
        raise BundleStructuralError('eligibility_invalid', 'manifest admission eligibility is not derivable from source evidence')

def _derive_admission_eligibility(root: Path, manifest: Mapping[str, Any], *, train_eligible: bool) -> bool:
    rollout = manifest['rollout']
    terminal = manifest['terminal']
    if rollout['purpose'] != 'admission' or not train_eligible:
        return False
    if terminal['episode_status'] != 'passed' or terminal['error_class'] != 'none':
        return False
    preflight_path = root / 'privileged' / 'admission_preflight.json'
    if not preflight_path.is_file():
        return False
    preflight = _read_canonical_json(preflight_path)
    if preflight.get('status') != 'pass' or preflight.get('disposition') != 'admitted':
        return False
    policy_document = _read_canonical_json(root / 'provenance' / 'policy.json')
    policy = policy_document.get('content') if isinstance(policy_document, Mapping) else None
    if not isinstance(policy, Mapping) or policy.get('mode') not in {'teacher', 'oracle'}:
        return False
    private_rows = _read_jsonl(root / 'privileged' / 'trajectory.jsonl')
    episode_end = next((row for row in private_rows if row['record_type'] == 'episode_end'), None)
    final_goal = episode_end.get('payload', {}).get('final_goal') if episode_end is not None else None
    predicate_success = final_goal.get('predicate_success') if isinstance(final_goal, Mapping) else None
    return not isinstance(predicate_success, bool) and isinstance(predicate_success, (int, float)) and (float(predicate_success) == 1.0)

def _artifact_privacy(relative: str) -> str:
    if relative == 'trajectory.jsonl':
        return 'index'
    first = PurePosixPath(relative).parts[0]
    if first == 'public':
        return 'public'
    if first == 'privileged':
        return 'privileged'
    if first == 'provenance':
        return 'provenance'
    if relative == 'quality_report.json' or first == 'quality':
        return 'quality'
    raise BundleStructuralError('artifact_path_invalid', f'artifact has no allowed privacy class: {relative}')

def _media_type(path: Path) -> str:
    if path.suffix == '.json':
        return 'application/json'
    if path.suffix == '.jsonl':
        return 'application/x-ndjson'
    if path.suffix == '.png':
        return 'image/png'
    if path.suffix in {'.jpg', '.jpeg'}:
        return 'image/jpeg'
    if path.suffix == '.gz':
        return 'application/gzip'
    return 'application/octet-stream'

def _count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open('rb') as handle:
        for raw in handle:
            if not raw.endswith(b'\n'):
                raise BundleStructuralError('jsonl_not_terminated', f'JSONL line is not newline-terminated: {path}')
            strict_json_loads(raw, source=str(path))
            count += 1
    return count

def _read_canonical_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BundleStructuralError('artifact_missing', f'required JSON missing: {path}')
    raw = path.read_bytes()
    value = strict_json_loads(raw, source=str(path))
    if not isinstance(value, Mapping):
        raise BundleStructuralError('json_schema_invalid', f'JSON document is not a mapping: {path}')
    if raw != canonical_json_bytes(value) + b'\n':
        raise BundleStructuralError('json_not_canonical', f'JSON document is not canonical + newline: {path}')
    return value

def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise BundleStructuralError('artifact_missing', f'required JSONL missing: {path}')
    return _read_optional_jsonl(path)

def _read_optional_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows: list[Mapping[str, Any]] = []
    with path.open('rb') as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.endswith(b'\n'):
                raise BundleStructuralError('jsonl_not_terminated', f'line {number} in {path} lacks newline')
            value = strict_json_loads(raw, source=f'{path}:{number}')
            if not isinstance(value, Mapping):
                raise BundleStructuralError('jsonl_schema_invalid', f'line {number} in {path} is not an object')
            if raw != canonical_json_bytes(value) + b'\n':
                raise BundleStructuralError('jsonl_not_canonical', f'line {number} in {path} is not canonical')
            rows.append(value)
    return rows

def _safe_artifact_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BundleStructuralError('artifact_path_invalid', 'artifact path must be non-empty string')
    pure = PurePosixPath(relative)
    if pure.is_absolute() or '..' in pure.parts or '\\' in relative:
        raise BundleStructuralError('artifact_path_invalid', f'unsafe artifact path {relative!r}')
    path = root.joinpath(*pure.parts)
    if path.is_symlink():
        raise BundleStructuralError('artifact_symlink_forbidden', f'artifact path is symlink: {relative}')
    return path

def _iter_scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_scalars(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_scalars(item)
    else:
        yield value

def _reference_kind(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    patterns = (('rec-\\d{8}', 'record'), ('frm-\\d{8}', 'frame'), ('state-\\d{8}', 'state'), ('mcall-\\d{8}', 'model'), ('act-\\d{6}', 'action'), ('dec-\\d{6}', 'decision'), ('sobs-\\d{6}-\\d{6}', 'stream_obs'), ('int-\\d{6}', 'intervention'))
    for pattern, kind in patterns:
        if re.fullmatch(pattern, value):
            return kind
    return None

def _validate_asset_cache_entry(entry: Any) -> None:
    fields = {'json_pointer', 'asset_kind', 'content_sha256', 'relative_path', 'bytes'}
    _require_exact_fields(entry, fields, 'run_manifest_invalid')
    if not isinstance(entry['json_pointer'], str) or not entry['json_pointer'].startswith('/'):
        raise TrajectoryConfigError('run_manifest_invalid', 'asset cache json_pointer is invalid')
    if entry['asset_kind'] != 'json':
        raise TrajectoryConfigError('run_manifest_invalid', 'asset cache asset_kind must be json')
    if not isinstance(entry['content_sha256'], str) or SHA256_RE.fullmatch(entry['content_sha256']) is None:
        raise TrajectoryConfigError('run_manifest_invalid', 'asset cache content_sha256 is invalid')
    expected_path = f".asset_cache/json/{entry['content_sha256']}.json"
    if entry['relative_path'] != expected_path:
        raise TrajectoryConfigError('run_manifest_invalid', 'asset cache relative_path is invalid')
    if isinstance(entry['bytes'], bool) or not isinstance(entry['bytes'], int) or entry['bytes'] < 0:
        raise TrajectoryConfigError('run_manifest_invalid', 'asset cache bytes must be a non-negative integer')

def _require_non_empty_string(value: Any, field: str, code: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrajectoryConfigError(code, f'{field} must be a non-empty string')
    return value

def _require_exact_fields(value: Any, fields: set[str], code: str, *, structural: bool=False) -> None:
    error = BundleStructuralError if structural else TrajectoryConfigError
    if not isinstance(value, Mapping):
        raise error(code, 'value must be a mapping')
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        raise error(code, f'fields are not exact; missing={sorted(missing)}, extra={sorted(extra)}')

def _require_fields(value: Any, fields: set[str], code: str, *, structural: bool=False) -> None:
    if not isinstance(value, Mapping):
        error = BundleStructuralError if structural else TrajectoryConfigError
        raise error(code, 'value must be a mapping')
    missing = fields - set(value)
    if missing:
        error = BundleStructuralError if structural else TrajectoryConfigError
        raise error(code, f'missing fields: {sorted(missing)}')
__all__ = ['COMMON_CONTRACT_BYTES', 'COMMON_CONTRACT_SHA256', 'COMMON_CONTRACT_VERSION', 'COMPILED_COMMON_CONTRACT', 'CONTRACT_VERSION', 'DERIVED_KINDS', 'EFFECT_STATUSES', 'ERROR_PRECEDENCE', 'EVENT_INDEX_VERSION', 'OBSERVABILITY', 'PAYLOAD_VERSION', 'POLICY_MODES', 'QUALITY_VERSION', 'RECORD_TYPES', 'ROLLOUT_ARMS', 'ROLLOUT_PURPOSES', 'SCHEMA_VERSIONS', 'TASK_BANK_HASH_VERSION', 'TASK_BANK_VERSION', 'build_artifact_table', 'common_contract_path', 'select_task_bank_entries', 'task_bank_hash_payload', 'validate_common_contract_fixture', 'validate_episode_bundle', 'validate_episode_bundle_candidate', 'validate_event_envelope', 'validate_payload_envelope', 'validate_provenance_document', 'validate_run_invocation', 'validate_raw_episode_files', 'validate_run_manifest', 'compute_run_identity_hash', 'validate_task_bank_envelope']
