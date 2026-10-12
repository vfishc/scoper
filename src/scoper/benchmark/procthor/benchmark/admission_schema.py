from __future__ import annotations
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping
from scoper.benchmark.procthor.export.ids import sha256_json
from scoper.benchmark.procthor.export.schema import SHA256_RE, SchemaValidationError, validate_json_value
ADMISSION_STAGE_ORDER = ('schema_identity', 'semantic_constraints', 'deterministic_reset', 'initial_goal_false', 'reachability', 'rgb_quality', 'task_relevant_observability', 'visible_action_effect', 'teacher_success', 'mutation_verified', 'trajectory_quality')
ADMISSION_STATUSES = frozenset({'pass', 'fail', 'invalid', 'not_applicable'})
ADMISSION_ERROR_CLASSES = frozenset({'none', 'task_invalid', 'infra_error', 'evaluator_invalid'})
ADMISSION_DISPOSITIONS = frozenset({'admitted', 'rejected', 'needs_rerun'})
ADMISSION_FAILURE_CODES = frozenset({'ADMISSION_SCHEMA_IDENTITY_FAILED', 'ADMISSION_SEMANTIC_CONSTRAINT_FAILED', 'ADMISSION_RESET_NONDETERMINISTIC', 'ADMISSION_INITIAL_GOAL_TRUE', 'ADMISSION_UNREACHABLE', 'ADMISSION_RGB_INVALID', 'ADMISSION_NOT_VISUALLY_OBSERVABLE', 'ADMISSION_EFFECT_NOT_VISIBLE', 'ADMISSION_TEACHER_FAILED', 'ADMISSION_MUTATION_NOT_VERIFIED', 'ADMISSION_TRAJECTORY_INVALID', 'ADMISSION_INFRA_RESET_ERROR', 'ADMISSION_INFRA_RENDERER_ERROR', 'ADMISSION_INFRA_CONTROLLER_ERROR', 'ADMISSION_INFRA_CAPTURE_ERROR', 'ADMISSION_EVALUATOR_DISAGREEMENT', 'ADMISSION_PROTOCOL_MISMATCH', 'ADMISSION_MISSING_REQUIRED_CELL'})
ADMISSION_INFRA_FAILURE_CODES = frozenset({'ADMISSION_INFRA_RESET_ERROR', 'ADMISSION_INFRA_RENDERER_ERROR', 'ADMISSION_INFRA_CONTROLLER_ERROR', 'ADMISSION_INFRA_CAPTURE_ERROR'})
ADMISSION_EVALUATOR_FAILURE_CODES = frozenset({'ADMISSION_EVALUATOR_DISAGREEMENT', 'ADMISSION_PROTOCOL_MISMATCH', 'ADMISSION_MISSING_REQUIRED_CELL'})
ADMISSION_STAGE_FAILURE_CODES = {'schema_identity': 'ADMISSION_SCHEMA_IDENTITY_FAILED', 'semantic_constraints': 'ADMISSION_SEMANTIC_CONSTRAINT_FAILED', 'deterministic_reset': 'ADMISSION_RESET_NONDETERMINISTIC', 'initial_goal_false': 'ADMISSION_INITIAL_GOAL_TRUE', 'reachability': 'ADMISSION_UNREACHABLE', 'rgb_quality': 'ADMISSION_RGB_INVALID', 'task_relevant_observability': 'ADMISSION_NOT_VISUALLY_OBSERVABLE', 'visible_action_effect': 'ADMISSION_EFFECT_NOT_VISIBLE', 'teacher_success': 'ADMISSION_TEACHER_FAILED', 'mutation_verified': 'ADMISSION_MUTATION_NOT_VERIFIED', 'trajectory_quality': 'ADMISSION_TRAJECTORY_INVALID'}
STATIC_ADMISSION_STAGES = frozenset({'schema_identity', 'semantic_constraints'})
RUNTIME_ADMISSION_STAGES = tuple((stage for stage in ADMISSION_STAGE_ORDER if stage not in STATIC_ADMISSION_STAGES))
_CELL_ID_RE = re.compile('^adcell-[0-9a-f]{24}$')
_ROLLOUT_ARM_ORDER = {name: index for index, name in enumerate(('clean', 'disturbance', 'failure', 'mixed', 'learner'))}

@dataclass(frozen=True)
class AdmissionCell:
    schema_version: str
    backend: str
    task_hash: str
    house_id: str
    scene_id: str
    layout_id: str
    house_content_sha256: str
    repeat_id: int
    environment_seed: int
    execution_profile: str
    rollout_arm: str

    @property
    def cell_id(self) -> str:
        return 'adcell-' + sha256_json(asdict(self))[:24]

    def to_mapping(self) -> Mapping[str, Any]:
        return {'cell_id': self.cell_id, **asdict(self)}

def build_admission_cell(episode_context: Any) -> AdmissionCell:
    payload = getattr(episode_context, 'episode_key_payload', None)
    task = getattr(episode_context, 'task', None)
    identity = getattr(task, 'identity', None)
    seeds = getattr(episode_context, 'seeds', None)
    if not isinstance(payload, Mapping) or identity is None or (not isinstance(seeds, Mapping)):
        _fail('admission_cell_invalid', 'EpisodeContext identity is unavailable')
    source = getattr(identity, 'identity_task', {}).get('source', {})
    resolution = next((item for item in getattr(identity, 'path_resolutions', ()) if getattr(item, 'json_pointer', None) == '/source/house_path'), None)
    house_id = source.get('house_id')
    cell = AdmissionCell(schema_version='task-admission-cell.v1', backend='eb_procthor', task_hash=payload.get('task_hash'), house_id=house_id, scene_id=payload.get('scene_id'), layout_id=payload.get('layout_id'), house_content_sha256=getattr(resolution, 'content_sha256', None), repeat_id=payload.get('repeat_id'), environment_seed=seeds.get('environment'), execution_profile=payload.get('execution_profile'), rollout_arm=payload.get('rollout_arm'))
    validate_admission_cell(cell.to_mapping())
    return cell

def validate_admission_cell(value: Mapping[str, Any]) -> None:
    required = {'cell_id', 'schema_version', 'backend', 'task_hash', 'house_id', 'scene_id', 'layout_id', 'house_content_sha256', 'repeat_id', 'environment_seed', 'execution_profile', 'rollout_arm'}
    _exact_mapping(value, required, 'admission_cell_invalid')
    if value['schema_version'] != 'task-admission-cell.v1' or value['backend'] != 'eb_procthor':
        _fail('admission_cell_invalid', 'cell schema/backend is invalid')
    for field in ('task_hash', 'house_content_sha256'):
        if not isinstance(value[field], str) or SHA256_RE.fullmatch(value[field]) is None:
            _fail('admission_cell_invalid', '{} is invalid'.format(field))
    for field in ('house_id', 'scene_id', 'execution_profile', 'rollout_arm'):
        if not isinstance(value[field], str) or not value[field]:
            _fail('admission_cell_invalid', '{} must be non-empty'.format(field))
    if value['house_id'] != value['scene_id']:
        _fail('admission_cell_invalid', 'ProcTHOR house_id and scene_id must match')
    if not isinstance(value['layout_id'], str) or value['layout_id'] != '':
        _fail('admission_cell_invalid', 'ProcTHOR v1 layout_id must be empty')
    for field in ('repeat_id', 'environment_seed'):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
            _fail('admission_cell_invalid', '{} must be a non-negative integer'.format(field))
    if value['environment_seed'] >= 2 ** 31:
        _fail('admission_cell_invalid', 'environment_seed must be below 2**31')
    payload = {key: value[key] for key in required if key != 'cell_id'}
    if value['cell_id'] != 'adcell-' + sha256_json(payload)[:24] or _CELL_ID_RE.fullmatch(value['cell_id']) is None:
        _fail('admission_cell_invalid', 'cell_id does not match canonical identity')
    validate_json_value(dict(value))

def validate_reachability_preflight(value: Mapping[str, Any]) -> None:
    required = {'schema_version', 'stage', 'status', 'error_class', 'failure_code', 'disposition', 'cell', 'house', 'controller', 'renderer', 'start', 'required_roles', 'evaluators', 'alignment'}
    _exact_mapping(value, required, 'admission_preflight_invalid')
    if value['schema_version'] != 'procthor-admission-preflight.v1' or value['stage'] != 'reachability':
        _fail('admission_preflight_invalid', 'preflight schema/stage is invalid')
    combinations = {'pass': ('none', None, 'admitted'), 'fail': ('task_invalid', 'ADMISSION_UNREACHABLE', 'rejected')}
    if value['status'] == 'invalid':
        allowed = ADMISSION_INFRA_FAILURE_CODES if value['error_class'] == 'infra_error' else ADMISSION_EVALUATOR_FAILURE_CODES if value['error_class'] == 'evaluator_invalid' else frozenset()
        if value['failure_code'] not in allowed or value['disposition'] != 'needs_rerun':
            _fail('admission_preflight_invalid', 'invalid preflight classification is inconsistent')
    elif value['status'] not in combinations or (value['error_class'], value['failure_code'], value['disposition']) != combinations[value['status']]:
        _fail('admission_preflight_invalid', 'preflight classification is inconsistent')
    validate_admission_cell(value['cell'])
    for field in ('house', 'controller', 'renderer', 'start', 'alignment'):
        if not isinstance(value[field], Mapping):
            _fail('admission_preflight_invalid', '{} diagnostics must be an object'.format(field))
    if not isinstance(value['required_roles'], list):
        _fail('admission_preflight_invalid', 'required_roles must be a list')
    _exact_mapping(value['evaluators'], {'native_interaction_poses', 'independent_reachable_graph_audit'}, 'admission_preflight_invalid')
    conclusions = tuple(value['evaluators'].values())
    if value['status'] == 'invalid':
        if any((item is not None and (not isinstance(item, bool)) for item in conclusions)):
            _fail('admission_preflight_invalid', 'invalid evaluator values have wrong types')
    elif any((not isinstance(item, bool) for item in conclusions)):
        _fail('admission_preflight_invalid', 'evaluator conclusions must be booleans')
    start = value['start']
    if set(start) != {'pose', 'reachable_position_count', 'nearest_distance_m', 'on_reachable_graph'}:
        _fail('admission_preflight_invalid', 'start diagnostics are not exact')
    if start['on_reachable_graph'] is not None and (not isinstance(start['on_reachable_graph'], bool)):
        _fail('admission_preflight_invalid', 'start graph result must be boolean or null')
    for role in value['required_roles']:
        if not isinstance(role, Mapping):
            _fail('admission_preflight_invalid', 'role diagnostic must be an object')
        for field in ('role', 'object_id', 'object_exists', 'interaction_pose_count', 'reachable_interaction_pose_count', 'primary_reachable', 'audit_reachable', 'activation_chain', 'rejection_reasons'):
            if field not in role:
                _fail('admission_preflight_invalid', 'role diagnostic is incomplete')
        counts = (role['interaction_pose_count'], role['reachable_interaction_pose_count'])
        if any((isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts)):
            _fail('admission_preflight_invalid', 'role pose counts are invalid')
        if counts[1] > counts[0]:
            _fail('admission_preflight_invalid', 'reachable pose count exceeds total poses')
        if not isinstance(role['activation_chain'], list) or not isinstance(role['rejection_reasons'], list):
            _fail('admission_preflight_invalid', 'role activation/rejection evidence is invalid')
    validate_json_value(dict(value))

def validate_admission_stage(value: Mapping[str, Any]) -> None:
    required = {'schema_version', 'task_id', 'task_hash', 'backend', 'stage', 'status', 'failure_code_or_null', 'error_class', 'repeat_id_or_null', 'episode_id_or_null', 'rollout_arm_or_null', 'evidence_refs', 'metrics', 'details'}
    _exact_mapping(value, required, 'admission_stage_invalid')
    if value['schema_version'] != 'task-admission-stage.v1' or value['backend'] != 'eb_procthor':
        _fail('admission_stage_invalid', 'stage schema/backend is invalid')
    if value['stage'] not in ADMISSION_STAGE_ORDER or value['status'] not in ADMISSION_STATUSES:
        _fail('admission_stage_invalid', 'stage/status is invalid')
    if value['error_class'] not in ADMISSION_ERROR_CLASSES:
        _fail('admission_stage_invalid', 'error_class is invalid')
    code = value['failure_code_or_null']
    if code is not None and code not in ADMISSION_FAILURE_CODES:
        _fail('admission_stage_invalid', 'failure code is invalid')
    if value['status'] in {'pass', 'not_applicable'}:
        if value['error_class'] != 'none' or code is not None:
            _fail('admission_stage_invalid', 'successful/not-applicable stage has error')
    elif value['status'] == 'fail':
        if value['error_class'] != 'task_invalid' or code != ADMISSION_STAGE_FAILURE_CODES[value['stage']]:
            _fail('admission_stage_invalid', 'failed stage uses a non-normative classification')
    else:
        allowed = ADMISSION_INFRA_FAILURE_CODES if value['error_class'] == 'infra_error' else ADMISSION_EVALUATOR_FAILURE_CODES if value['error_class'] == 'evaluator_invalid' else frozenset()
        if code not in allowed:
            _fail('admission_stage_invalid', 'invalid stage uses wrong failure-code class')
    if not isinstance(value['task_id'], str) or not value['task_id']:
        _fail('admission_stage_invalid', 'task_id is invalid')
    if not isinstance(value['task_hash'], str) or SHA256_RE.fullmatch(value['task_hash']) is None:
        _fail('admission_stage_invalid', 'task_hash is invalid')
    repeat = value['repeat_id_or_null']
    if repeat is not None and (isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0):
        _fail('admission_stage_invalid', 'repeat id is invalid')
    for field in ('episode_id_or_null', 'rollout_arm_or_null'):
        if value[field] is not None and (not isinstance(value[field], str) or not value[field]):
            _fail('admission_stage_invalid', '{} is invalid'.format(field))
    if not isinstance(value['evidence_refs'], list) or not isinstance(value['metrics'], Mapping) or (not isinstance(value['details'], Mapping)):
        _fail('admission_stage_invalid', 'stage evidence/metrics/details shape is invalid')
    for reference in value['evidence_refs']:
        _exact_mapping(reference, {'artifact_sha256', 'relative_path_or_record_id', 'kind'}, 'admission_stage_invalid')
        if not isinstance(reference['artifact_sha256'], str) or SHA256_RE.fullmatch(reference['artifact_sha256']) is None:
            _fail('admission_stage_invalid', 'evidence artifact hash is invalid')
        if any((not isinstance(reference[field], str) or not reference[field] for field in ('relative_path_or_record_id', 'kind'))):
            _fail('admission_stage_invalid', 'evidence reference strings are invalid')
        locator = reference['relative_path_or_record_id']
        if _is_machine_evidence_locator(locator):
            _fail('admission_stage_invalid', 'evidence reference is not portable')
    cell = value['details'].get('cell')
    if cell is not None:
        validate_admission_cell(cell)
    if value['status'] == 'not_applicable':
        evidence = value['details'].get('evidence')
        if not isinstance(evidence, Mapping) or evidence.get('status') != 'not_applicable':
            _fail('admission_stage_invalid', 'not-applicable stage requires explicit evidence')
    validate_json_value(dict(value))

def _is_machine_evidence_locator(value: str) -> bool:
    if value.startswith(('/', '~/', 'file:', '\\\\')) or re.match('^[A-Za-z]:[\\\\/]', value):
        return True
    if ':' in value and (not value.startswith(('sha256:', 'record:', 'evidence:'))):
        return True
    return '..' in PurePosixPath(value.replace('\\', '/')).parts

def validate_admission_task(value: Mapping[str, Any]) -> None:
    required = {'schema_version', 'task_id', 'task_hash', 'lineage_id', 'source_split', 'family', 'template_id', 'scene_id', 'asset_lineage_ids', 'required_cells', 'stage_results', 'disposition', 'primary_failure_code_or_null', 'secondary_failure_codes', 'admitted_tracks', 'quality_summary'}
    _exact_mapping(value, required, 'admission_task_invalid')
    if value['schema_version'] != 'task-admission-task.v1' or value['disposition'] not in ADMISSION_DISPOSITIONS:
        _fail('admission_task_invalid', 'task schema/disposition is invalid')
    for field in ('task_id', 'lineage_id', 'source_split', 'scene_id'):
        if not isinstance(value[field], str) or not value[field]:
            _fail('admission_task_invalid', '{} must be non-empty'.format(field))
    if not isinstance(value['task_hash'], str) or SHA256_RE.fullmatch(value['task_hash']) is None:
        _fail('admission_task_invalid', 'task_hash is invalid')
    for field in ('family', 'template_id'):
        if value[field] is not None and (not isinstance(value[field], str) or not value[field]):
            _fail('admission_task_invalid', '{} must be null or non-empty'.format(field))
    if value['asset_lineage_ids'] != sorted(set(value['asset_lineage_ids'])) or any((not isinstance(item, str) or not item for item in value['asset_lineage_ids'])):
        _fail('admission_task_invalid', 'asset lineage ids must be sorted unique strings')
    if not isinstance(value['required_cells'], list) or not isinstance(value['stage_results'], list):
        _fail('admission_task_invalid', 'required cells/stage results must be lists')
    cell_ids = []
    for cell in value['required_cells']:
        validate_admission_cell(cell)
        if cell['task_hash'] != value['task_hash'] or cell['scene_id'] != value['scene_id']:
            _fail('admission_task_invalid', 'required cell identity differs from task')
        cell_ids.append(cell['cell_id'])
    if cell_ids != sorted(set(cell_ids)):
        _fail('admission_task_invalid', 'required cells must be sorted and unique')
    for stage in value['stage_results']:
        validate_admission_stage(stage)
        if stage['task_id'] != value['task_id'] or stage['task_hash'] != value['task_hash']:
            _fail('admission_task_invalid', 'stage task identity differs')
    _validate_complete_stage_matrix(value)
    primary = value['primary_failure_code_or_null']
    if primary is not None and primary not in ADMISSION_FAILURE_CODES:
        _fail('admission_task_invalid', 'primary failure code is invalid')
    if value['secondary_failure_codes'] != sorted(set(value['secondary_failure_codes'])) or any((code not in ADMISSION_FAILURE_CODES for code in value['secondary_failure_codes'])):
        _fail('admission_task_invalid', 'secondary failure codes are invalid')
    if value['admitted_tracks'] != sorted(set(value['admitted_tracks'])) or any((not isinstance(item, str) or not item for item in value['admitted_tracks'])):
        _fail('admission_task_invalid', 'admitted tracks are invalid')
    summary = value['quality_summary']
    _exact_mapping(summary, {'required_cell_count', 'pass_count', 'fail_count', 'invalid_count'}, 'admission_task_invalid')
    if any((isinstance(summary[field], bool) or not isinstance(summary[field], int) or summary[field] < 0 for field in summary)):
        _fail('admission_task_invalid', 'quality summary counts must be non-negative integers')
    expected_summary = {'required_cell_count': len(value['required_cells']), 'pass_count': sum((row['status'] == 'pass' for row in value['stage_results'])), 'fail_count': sum((row['status'] == 'fail' for row in value['stage_results'])), 'invalid_count': sum((row['status'] == 'invalid' for row in value['stage_results']))}
    if summary != expected_summary:
        _fail('admission_task_invalid', 'quality summary differs from required stage evidence')
    expected_disposition, expected_primary = task_disposition(value['stage_results'])
    if value['disposition'] != expected_disposition or primary != expected_primary:
        _fail('admission_task_invalid', 'task disposition does not follow stage evidence')
    validate_json_value(dict(value))

def validate_admission_report(value: Mapping[str, Any]) -> None:
    required = {'schema_version', 'backend', 'benchmark_id', 'benchmark_version', 'protocol_sha256', 'protocol_status', 'candidate_artifact_schema', 'candidate_artifact_hash', 'source_task_bank_hash', 'common_contract_sha256', 'code_commit', 'dirty_diff_sha256_or_null', 'stage_order', 'stage_implementation_hashes', 'required_repeat_ids', 'required_arm_rules', 'execution_profiles', 'tasks', 'counts_by_disposition', 'counts_by_failure_code', 'counts_by_family', 'counts_by_template', 'counts_by_scene', 'invalid_episode_count_by_error_class', 'input_run_manifest_hashes', 'complete_candidate_set', 'report_content_hash', 'created_at_utc'}
    _exact_mapping(value, required, 'admission_report_invalid')
    if value['schema_version'] != 'task-admission-report.v1' or value['backend'] != 'eb_procthor':
        _fail('admission_report_invalid', 'report schema/backend is invalid')
    for field in ('benchmark_id', 'benchmark_version', 'protocol_status', 'candidate_artifact_schema', 'code_commit', 'created_at_utc'):
        if not isinstance(value[field], str) or not value[field]:
            _fail('admission_report_invalid', '{} must be non-empty'.format(field))
    for field in ('protocol_sha256', 'candidate_artifact_hash', 'source_task_bank_hash', 'common_contract_sha256'):
        if not isinstance(value[field], str) or SHA256_RE.fullmatch(value[field]) is None:
            _fail('admission_report_invalid', '{} is invalid'.format(field))
    dirty = value['dirty_diff_sha256_or_null']
    if dirty is not None and (not isinstance(dirty, str) or SHA256_RE.fullmatch(dirty) is None):
        _fail('admission_report_invalid', 'dirty diff hash is invalid')
    if value['stage_order'] != list(ADMISSION_STAGE_ORDER):
        _fail('admission_report_invalid', 'stage order is not normative')
    hashes = value['stage_implementation_hashes']
    if not isinstance(hashes, Mapping) or set(hashes) != set(ADMISSION_STAGE_ORDER) or any((not isinstance(item, str) or SHA256_RE.fullmatch(item) is None for item in hashes.values())):
        _fail('admission_report_invalid', 'stage implementation hashes are invalid')
    task_ids = []
    for task in value['tasks']:
        validate_admission_task(task)
        task_ids.append(task['task_id'])
    if task_ids != sorted(set(task_ids)):
        _fail('admission_report_invalid', 'tasks must be sorted and unique')
    if value['counts_by_disposition'] != _counts(value['tasks'], 'disposition'):
        _fail('admission_report_invalid', 'disposition counts differ')
    if value['counts_by_failure_code'] != _failure_counts(value['tasks']):
        _fail('admission_report_invalid', 'failure counts differ')
    for report_field, task_field in (('counts_by_family', 'family'), ('counts_by_template', 'template_id'), ('counts_by_scene', 'scene_id')):
        if value[report_field] != _counts(value['tasks'], task_field):
            _fail('admission_report_invalid', '{} differs'.format(report_field))
    if value['invalid_episode_count_by_error_class'] != _invalid_counts(value['tasks']):
        _fail('admission_report_invalid', 'invalid error-class counts differ')
    repeats = value['required_repeat_ids']
    if not isinstance(repeats, list) or repeats != sorted(set(repeats)) or any((isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in repeats)):
        _fail('admission_report_invalid', 'required repeat ids are invalid')
    if not isinstance(value['required_arm_rules'], list):
        _fail('admission_report_invalid', 'required arm rules must be a list')
    profiles = value['execution_profiles']
    if not isinstance(profiles, list) or profiles != sorted(set(profiles)) or any((item not in {'native_ai2thor', 'semantic_compatibility'} for item in profiles)):
        _fail('admission_report_invalid', 'execution profiles are invalid')
    run_hashes = value['input_run_manifest_hashes']
    if not isinstance(run_hashes, list) or run_hashes != sorted(set(run_hashes)) or any((not isinstance(item, str) or SHA256_RE.fullmatch(item) is None for item in run_hashes)):
        _fail('admission_report_invalid', 'input run manifest hashes are invalid')
    if not isinstance(value['complete_candidate_set'], bool):
        _fail('admission_report_invalid', 'complete_candidate_set must be boolean')
    identity = {key: item for key, item in value.items() if key not in {'created_at_utc', 'report_content_hash'}}
    if value['report_content_hash'] != sha256_json(identity):
        _fail('admission_report_invalid', 'report content hash does not match')
    validate_json_value(dict(value))

def stage_sort_key(value: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (ADMISSION_STAGE_ORDER.index(value['stage']), -1 if value['repeat_id_or_null'] is None else int(value['repeat_id_or_null']), _ROLLOUT_ARM_ORDER.get(str(value['rollout_arm_or_null'] or ''), -1), str((value.get('details') or {}).get('execution_profile') or ''))

def task_disposition(stages: list[Mapping[str, Any]]) -> tuple[str, str | None]:
    failed = sorted((stage for stage in stages if stage['status'] == 'fail'), key=stage_sort_key)
    if failed:
        return ('rejected', failed[0]['failure_code_or_null'])
    if any((stage['status'] == 'invalid' for stage in stages)):
        return ('needs_rerun', None)
    return ('admitted', None)

def report_has_unresolved_evidence(value: Mapping[str, Any]) -> bool:
    validate_admission_report(value)
    return any((stage['status'] == 'invalid' or stage['failure_code_or_null'] == 'ADMISSION_MISSING_REQUIRED_CELL' for task in value['tasks'] for stage in task['stage_results']))

def _validate_complete_stage_matrix(value: Mapping[str, Any]) -> None:
    stages = value['stage_results']
    if stages != sorted(stages, key=stage_sort_key):
        _fail('admission_task_invalid', 'stage results are not in normative order')
    static_counts = {stage: 0 for stage in STATIC_ADMISSION_STAGES}
    required_ids = {cell['cell_id'] for cell in value['required_cells']}
    runtime_counts: dict[tuple[str, str], int] = {}
    for row in stages:
        name = row['stage']
        cell = row['details'].get('cell')
        if name in STATIC_ADMISSION_STAGES:
            if any((row[field] is not None for field in ('repeat_id_or_null', 'episode_id_or_null', 'rollout_arm_or_null'))):
                _fail('admission_task_invalid', 'static stage has runtime identity')
            static_counts[name] += 1
            continue
        if not isinstance(cell, Mapping) or cell.get('cell_id') not in required_ids:
            _fail('admission_task_invalid', 'runtime stage is not bound to a required cell')
        if row['repeat_id_or_null'] != cell['repeat_id'] or row['rollout_arm_or_null'] != cell['rollout_arm']:
            _fail('admission_task_invalid', 'runtime stage identity differs from cell')
        key = (name, cell['cell_id'])
        runtime_counts[key] = runtime_counts.get(key, 0) + 1
    if any((count != 1 for count in static_counts.values())):
        _fail('admission_task_invalid', 'each static stage must appear exactly once')
    expected = {(stage, cell_id) for stage in RUNTIME_ADMISSION_STAGES for cell_id in required_ids}
    if set(runtime_counts) != expected or any((count != 1 for count in runtime_counts.values())):
        _fail('admission_task_invalid', 'runtime stage matrix is incomplete or duplicated')

def _counts(rows: list[Mapping[str, Any]], field: str) -> Mapping[str, int]:
    return {key: sum((str(row.get(field)) == key for row in rows)) for key in sorted({str(row.get(field)) for row in rows})}

def _failure_counts(rows: list[Mapping[str, Any]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for code in (row.get('primary_failure_code_or_null'), *(row.get('secondary_failure_codes') or [])):
            if code is not None:
                counts[str(code)] = counts.get(str(code), 0) + 1
    return {key: counts[key] for key in sorted(counts)}

def _invalid_counts(rows: list[Mapping[str, Any]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for stage in row['stage_results']:
            if stage['status'] == 'invalid':
                key = str(stage['error_class'])
                counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}

def _exact_mapping(value: Any, required: set[str], code: str) -> None:
    if not isinstance(value, Mapping) or set(value) != required:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        _fail(code, 'fields are not exact: {}'.format(actual))

def _fail(code: str, message: str) -> None:
    raise SchemaValidationError(code, message)
__all__ = ['ADMISSION_DISPOSITIONS', 'ADMISSION_ERROR_CLASSES', 'ADMISSION_EVALUATOR_FAILURE_CODES', 'ADMISSION_FAILURE_CODES', 'ADMISSION_INFRA_FAILURE_CODES', 'ADMISSION_STAGE_FAILURE_CODES', 'ADMISSION_STAGE_ORDER', 'ADMISSION_STATUSES', 'RUNTIME_ADMISSION_STAGES', 'STATIC_ADMISSION_STAGES', 'AdmissionCell', 'build_admission_cell', 'stage_sort_key', 'task_disposition', 'report_has_unresolved_evidence', 'validate_admission_cell', 'validate_admission_report', 'validate_admission_stage', 'validate_admission_task', 'validate_reachability_preflight']
