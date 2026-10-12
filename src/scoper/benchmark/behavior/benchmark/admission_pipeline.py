from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.behavior.export.ids import canonical_json_line, json_deep_copy, sha256_file, sha256_json, strict_json_load
from scoper.benchmark.behavior.export.schema import COMMON_CONTRACT_SHA256
from scoper.benchmark.behavior.export.types import BundleConflictError, BundleStructuralError, TrajectoryConfigError
from scoper.benchmark.behavior.benchmark.admission_schema import ADMISSION_STAGE_ORDER, RUNTIME_ADMISSION_STAGES, STATIC_ADMISSION_STAGES, stage_sort_key, validate_admission_cell, validate_admission_report, validate_admission_stage, validate_admission_task

class TaskAdmissionPipeline:

    def __init__(self, *, protocol: Mapping[str, Any], protocol_sha256: str, candidate_artifact_schema: str, candidate_artifact_hash: str, code_commit: str, dirty_diff_sha256_or_null: str | None, execution_profiles: Sequence[str]) -> None:
        self.protocol = json_deep_copy(protocol)
        self.protocol_sha256 = protocol_sha256
        self.candidate_artifact_schema = candidate_artifact_schema
        self.candidate_artifact_hash = candidate_artifact_hash
        self.code_commit = code_commit
        self.dirty_diff_sha256_or_null = dirty_diff_sha256_or_null
        self.execution_profiles = tuple(execution_profiles)

    def stage_from_preflight(self, *, task_id: str, task_hash: str, episode_id: str, preflight_sha256: str, preflight_relative_path: str, preflight: Mapping[str, Any]) -> Mapping[str, Any]:
        cell = preflight.get('cell')
        validate_admission_cell(cell)
        status = preflight.get('status')
        error_class = preflight.get('error_class')
        code = preflight.get('failure_code')
        stage = {'schema_version': 'task-admission-stage.v1', 'task_id': task_id, 'task_hash': task_hash, 'backend': 'scoper.benchmark.behavior', 'stage': 'reachability', 'status': status, 'failure_code_or_null': code, 'error_class': error_class, 'repeat_id_or_null': cell['repeat_id'], 'episode_id_or_null': episode_id, 'rollout_arm_or_null': cell['rollout_arm'], 'evidence_refs': [{'artifact_sha256': preflight_sha256, 'relative_path_or_record_id': preflight_relative_path, 'kind': 'trajectory_bundle_admission_preflight'}], 'metrics': _preflight_metrics(preflight), 'details': {'cell': json_deep_copy(cell), 'execution_profile': cell['execution_profile'], 'disposition': preflight.get('disposition'), 'navigation_map_hash': (preflight.get('navigation_map') or {}).get('navigation_map_hash')}}
        validate_admission_stage(stage)
        return stage

    def task_result(self, *, task: Mapping[str, Any], task_hash: str, lineage_id: str, source_split: str, required_cells: Sequence[Mapping[str, Any]], stage_results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        cells = sorted((json_deep_copy(cell) for cell in required_cells), key=lambda item: item['cell_id'])
        stages = _complete_stage_matrix(task_id=task['id'], task_hash=task_hash, cells=cells, stages=stage_results)
        fail_rows = [row for row in stages if row['status'] == 'fail']
        invalid_rows = [row for row in stages if row['status'] == 'invalid']
        if fail_rows:
            disposition = 'rejected'
            primary = fail_rows[0]['failure_code_or_null']
        elif invalid_rows:
            disposition = 'needs_rerun'
            primary = None
        else:
            disposition = 'admitted'
            primary = None
        codes = sorted({row['failure_code_or_null'] for row in stages if row['failure_code_or_null'] is not None and row['failure_code_or_null'] != primary})
        source = task.get('source') if isinstance(task.get('source'), Mapping) else {}
        template = task.get('template') if isinstance(task.get('template'), Mapping) else {}
        result = {'schema_version': 'task-admission-task.v1', 'task_id': task['id'], 'task_hash': task_hash, 'lineage_id': lineage_id, 'source_split': source_split, 'family': task.get('family') or task.get('category'), 'template_id': source.get('template_id') or template.get('template_id') or task.get('template_id'), 'scene_id': source.get('scene_model'), 'asset_lineage_ids': sorted(set(task.get('asset_lineage_ids') or [])), 'required_cells': cells, 'stage_results': stages, 'disposition': disposition, 'primary_failure_code_or_null': primary, 'secondary_failure_codes': codes, 'admitted_tracks': ['structured_agent', 'visual_minimal'] if disposition == 'admitted' else [], 'quality_summary': {'required_cell_count': len(cells), 'pass_count': sum((row['status'] == 'pass' for row in stages)), 'fail_count': len(fail_rows), 'invalid_count': len(invalid_rows)}}
        validate_admission_task(result)
        return result

    def report(self, tasks: Sequence[Mapping[str, Any]], *, input_run_manifest_hashes: Sequence[str], complete_candidate_set: bool=True, created_at_utc: str | None=None) -> Mapping[str, Any]:
        rows = sorted((json_deep_copy(task) for task in tasks), key=lambda item: item['task_id'])
        for row in rows:
            validate_admission_task(row)
        benchmark = self.protocol.get('benchmark') if isinstance(self.protocol.get('benchmark'), Mapping) else {}
        lifecycle = self.protocol.get('lifecycle') if isinstance(self.protocol.get('lifecycle'), Mapping) else {}
        admission = self.protocol.get('admission') if isinstance(self.protocol.get('admission'), Mapping) else {}
        seed_policy = self.protocol.get('seed_policy') if isinstance(self.protocol.get('seed_policy'), Mapping) else {}
        report = {'schema_version': 'task-admission-report.v1', 'backend': 'scoper.benchmark.behavior', 'benchmark_id': benchmark.get('id'), 'benchmark_version': benchmark.get('version'), 'protocol_sha256': self.protocol_sha256, 'protocol_status': lifecycle.get('status'), 'candidate_artifact_schema': self.candidate_artifact_schema, 'candidate_artifact_hash': self.candidate_artifact_hash, 'source_task_bank_hash': self.candidate_artifact_hash, 'complete_candidate_set': complete_candidate_set, 'common_contract_sha256': COMMON_CONTRACT_SHA256, 'code_commit': self.code_commit, 'dirty_diff_sha256_or_null': self.dirty_diff_sha256_or_null, 'stage_order': list(ADMISSION_STAGE_ORDER), 'stage_implementation_hashes': _stage_implementation_hashes(), 'required_repeat_ids': list(seed_policy.get('admission_repeat_ids') or []), 'required_arm_rules': json_deep_copy(admission.get('required_arm_rules') or []), 'execution_profiles': sorted(set(self.execution_profiles)), 'tasks': rows, 'counts_by_disposition': _count(rows, 'disposition'), 'counts_by_failure_code': _failure_counts(rows), 'counts_by_family': _count(rows, 'family'), 'counts_by_template': _count(rows, 'template_id'), 'counts_by_scene': _count(rows, 'scene_id'), 'invalid_episode_count_by_error_class': _invalid_counts(rows), 'input_run_manifest_hashes': sorted(set(input_run_manifest_hashes)), 'report_content_hash': '', 'created_at_utc': created_at_utc or _utc_now()}
        identity = json_deep_copy(report)
        identity.pop('created_at_utc')
        identity.pop('report_content_hash')
        report['report_content_hash'] = sha256_json(identity)
        validate_admission_report(report)
        return report

def freeze_admitted_task_cells(*, candidate_task_bank: Mapping[str, Any], admission_report: Mapping[str, Any], protocol: Mapping[str, Any], report_artifact_root: Path, output_path: Path) -> Mapping[str, Any]:
    from scoper.benchmark.behavior.export.schema import validate_task_bank_envelope
    from scoper.benchmark.behavior.export.task_identity import validate_resolved_protocol
    from scoper.benchmark.behavior.paths import repo_root
    validate_admission_report(admission_report)
    validate_admission_evidence_files(admission_report, artifact_root=report_artifact_root)
    resolved_protocol = validate_resolved_protocol(protocol)
    if sha256_json(resolved_protocol) != admission_report['protocol_sha256']:
        raise TrajectoryConfigError('ADMISSION_PROTOCOL_MISMATCH', 'freeze protocol differs from report')
    if admission_report['common_contract_sha256'] != COMMON_CONTRACT_SHA256:
        raise BundleConflictError('admission_artifact_conflict', 'common contract differs from report')
    if admission_report['complete_candidate_set'] is not True:
        raise TrajectoryConfigError('admission_needs_rerun', 'limited admission report cannot be frozen')
    candidate = validate_task_bank_envelope(candidate_task_bank, repo_root=repo_root())
    if candidate.task_bank_hash not in {admission_report['candidate_artifact_hash'], admission_report['source_task_bank_hash']}:
        raise BundleConflictError('admission_artifact_conflict', 'candidate task bank hash differs from report')
    candidate_ids = {entry.task_id for entry in candidate.entries}
    report_ids = {task['task_id'] for task in admission_report['tasks']}
    if candidate_ids != report_ids:
        raise BundleConflictError('admission_artifact_conflict', 'report membership differs from candidate bank')
    unresolved = [stage for task in admission_report['tasks'] for stage in task['stage_results'] if stage['status'] == 'invalid']
    if unresolved:
        raise TrajectoryConfigError('admission_needs_rerun', 'invalid stage evidence blocks task-cell freeze')
    entries_by_hash = {entry.task_hash: entry for entry in candidate.entries}
    entries_by_id = {entry.task_id: entry for entry in candidate.entries}
    admitted = [task for task in admission_report['tasks'] if task['disposition'] == 'admitted']
    frozen_tasks = []
    cells = []
    for result in admitted:
        entry = entries_by_hash.get(result['task_hash']) or entries_by_id.get(result['task_id'])
        if entry is None:
            raise BundleConflictError('admission_artifact_conflict', 'admitted task is absent from candidate bank')
        if entry.task_id != result['task_id'] or entry.task_hash != result['task_hash']:
            raise BundleConflictError('admission_artifact_conflict', 'admitted task identity differs from candidate bank')
        frozen_tasks.append(json_deep_copy(entry.task))
        cells.extend(json_deep_copy(result['required_cells']))
    artifact = {'schema_version': 'eb-behavior-admitted-task-cells.v1', 'source_task_bank_hash': admission_report['candidate_artifact_hash'], 'protocol_sha256': admission_report['protocol_sha256'], 'common_contract_sha256': admission_report['common_contract_sha256'], 'admission_report_content_hash': admission_report['report_content_hash'], 'tasks': sorted(frozen_tasks, key=lambda item: item['id']), 'admitted_cells': sorted(cells, key=lambda item: item['cell_id'])}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as handle:
            handle.write(canonical_json_line(artifact))
    except FileExistsError as exc:
        raise BundleConflictError('admitted_task_cells_exists', 'output already exists') from exc
    if strict_json_load(path) != artifact:
        raise TrajectoryConfigError('admitted_task_cells_invalid', 'frozen artifact re-read differs')
    return artifact

def validate_admission_evidence_files(admission_report: Mapping[str, Any], *, artifact_root: Path) -> None:
    validate_admission_report(admission_report)
    root = Path(artifact_root).resolve()
    for task in admission_report['tasks']:
        for stage in task['stage_results']:
            for reference in stage['evidence_refs']:
                kind = reference['kind']
                if kind not in {'behavior-static-admission-evidence.v1', 'behavior-deterministic-reset-evidence.v1'}:
                    continue
                path = (root / reference['relative_path_or_record_id']).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise BundleStructuralError('admission_evidence_missing', 'persisted admission evidence is missing')
                if sha256_file(path) != reference['artifact_sha256']:
                    raise BundleStructuralError('admission_evidence_hash_mismatch', 'admission evidence hash differs')
                value = strict_json_load(path)
                if value.get('schema_version') != kind:
                    raise BundleStructuralError('admission_evidence_invalid', 'admission evidence schema differs')
                if value.get('task_id') != task['task_id'] or value.get('task_hash') != task['task_hash']:
                    raise BundleStructuralError('admission_evidence_invalid', 'admission evidence task identity differs')
                if kind == 'behavior-static-admission-evidence.v1':
                    _validate_static_evidence(value, task, stage, admission_report)
                else:
                    _validate_reset_evidence(value, stage)

def _validate_static_evidence(value: Mapping[str, Any], task: Mapping[str, Any], stage: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    required = {'schema_version', 'task_id', 'task_hash', 'candidate_artifact_hash', 'identity_task', 'schema_checks', 'semantic_checks', 'cell_results'}
    if set(value) != required:
        raise BundleStructuralError('admission_evidence_invalid', 'static evidence fields differ')
    if value['candidate_artifact_hash'] != report['source_task_bank_hash']:
        raise BundleStructuralError('admission_evidence_invalid', 'static evidence task-bank hash differs')
    identity_task = value['identity_task']
    source = identity_task.get('source') if isinstance(identity_task, Mapping) else None
    cell_results = value['cell_results']
    if not isinstance(cell_results, list) or len(cell_results) != len(task['required_cells']):
        raise BundleStructuralError('admission_evidence_invalid', 'static evidence cell results are incomplete')
    expected_cell_ids = sorted((cell['cell_id'] for cell in task['required_cells']))
    actual_cell_ids = sorted((row.get('cell_id') for row in cell_results if isinstance(row, Mapping) and isinstance(row.get('cell_id'), str)))
    if actual_cell_ids != expected_cell_ids:
        raise BundleStructuralError('admission_evidence_invalid', 'static evidence cell identities differ')
    expected_schema_checks = {'task_schema_valid': all((row.get('structure_ok') is True for row in cell_results)), 'task_hash_matches': sha256_json(identity_task) == task['task_hash'], 'task_id_matches': isinstance(identity_task, Mapping) and identity_task.get('id') == task['task_id'], 'scene_identity_valid': isinstance(source, Mapping) and isinstance(source.get('scene_model'), str) and bool(source.get('scene_model'))}
    if value['schema_checks'] != expected_schema_checks:
        raise BundleStructuralError('admission_evidence_invalid', 'static schema checks are not derivable')
    semantic_fields = ('init_ok', 'geometry_ok', 'mutation_spec_ok', 'mutation_replay_ok')
    expected_semantic_checks = {field: all((row.get(field) is True for row in cell_results)) for field in semantic_fields}
    if value['semantic_checks'] != expected_semantic_checks:
        raise BundleStructuralError('admission_evidence_invalid', 'static semantic checks are not derivable')
    if stage['stage'] == 'schema_identity':
        expected_status = 'pass' if all(expected_schema_checks.values()) else 'fail'
    elif all(expected_semantic_checks.values()):
        expected_status = 'pass'
    elif any((row.get('failure_code') in {'RESET_FAILED', 'STAGE5_VALIDATION_EXCEPTION'} for row in cell_results)):
        expected_status = 'invalid'
    else:
        expected_status = 'fail'
    if stage['status'] != expected_status:
        raise BundleStructuralError('admission_evidence_invalid', 'static stage disagrees with persisted evidence')

def _validate_reset_evidence(value: Mapping[str, Any], stage: Mapping[str, Any]) -> None:
    first = value.get('first_snapshot')
    second = value.get('second_snapshot')
    if value.get('capture_status') != 'complete':
        if stage['status'] != 'invalid':
            raise BundleStructuralError('admission_evidence_invalid', 'incomplete reset evidence is not invalid')
        return
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise BundleStructuralError('admission_evidence_invalid', 'reset snapshots are missing')
    if sha256_json(first) != value.get('first_snapshot_sha256'):
        raise BundleStructuralError('admission_evidence_invalid', 'first reset snapshot hash differs')
    if sha256_json(second) != value.get('second_snapshot_sha256'):
        raise BundleStructuralError('admission_evidence_invalid', 'second reset snapshot hash differs')
    equal = first == second
    if value.get('equal') is not equal or bool(value.get('diff')) is equal:
        raise BundleStructuralError('admission_evidence_invalid', 'reset equality/diff is inconsistent')
    expected_status = 'pass' if equal else 'fail'
    if stage['status'] != expected_status:
        raise BundleStructuralError('admission_evidence_invalid', 'reset stage disagrees with snapshots')

def _preflight_metrics(preflight: Mapping[str, Any]) -> Mapping[str, Any]:
    roles = preflight.get('required_roles') or []
    return {'required_role_count': len(roles), 'reachable_role_count': sum((bool(role.get('reachable')) for role in roles)), 'interaction_candidate_count': sum((int(role.get('interaction_candidate_count') or 0) for role in roles)), 'free_candidate_count': sum((int(role.get('free_candidate_count') or 0) for role in roles)), 'same_component_candidate_count': sum((int(role.get('same_component_candidate_count') or 0) for role in roles)), 'path_planned_candidate_count': sum((int(role.get('path_planned_candidate_count') or 0) for role in roles))}

def _complete_stage_matrix(*, task_id: str, task_hash: str, cells: Sequence[Mapping[str, Any]], stages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = [json_deep_copy(stage) for stage in stages]
    for row in rows:
        validate_admission_stage(row)
    static_present = {row['stage'] for row in rows if row['stage'] in STATIC_ADMISSION_STAGES}
    for stage in STATIC_ADMISSION_STAGES:
        if stage not in static_present:
            rows.append(_missing_stage(task_id=task_id, task_hash=task_hash, stage=stage, cell=None))
    present = {(row['stage'], (row.get('details') or {}).get('cell', {}).get('cell_id')) for row in rows if row['stage'] in RUNTIME_ADMISSION_STAGES}
    for cell in cells:
        blockers = [row for row in rows if row['stage'] in RUNTIME_ADMISSION_STAGES and row['status'] in {'fail', 'invalid'} and isinstance((row.get('details') or {}).get('cell'), Mapping) and (row['details']['cell']['cell_id'] == cell['cell_id'])]
        blocker = min(blockers, key=lambda row: ADMISSION_STAGE_ORDER.index(row['stage'])) if blockers else None
        for stage in RUNTIME_ADMISSION_STAGES:
            if (stage, cell['cell_id']) in present:
                continue
            if blocker is not None and ADMISSION_STAGE_ORDER.index(stage) > ADMISSION_STAGE_ORDER.index(blocker['stage']):
                rows.append(_not_applicable_stage(task_id=task_id, task_hash=task_hash, stage=stage, cell=cell, reason='upstream_{}_{}'.format(blocker['stage'], 'rejection' if blocker['status'] == 'fail' else 'evidence_invalid')))
            else:
                rows.append(_missing_stage(task_id=task_id, task_hash=task_hash, stage=stage, cell=cell))
    return sorted(rows, key=stage_sort_key)

def _missing_stage(*, task_id: str, task_hash: str, stage: str, cell: Mapping[str, Any] | None) -> Mapping[str, Any]:
    runtime = cell is not None
    row = {'schema_version': 'task-admission-stage.v1', 'task_id': task_id, 'task_hash': task_hash, 'backend': 'scoper.benchmark.behavior', 'stage': stage, 'status': 'invalid', 'failure_code_or_null': 'ADMISSION_MISSING_REQUIRED_CELL', 'error_class': 'evaluator_invalid', 'repeat_id_or_null': cell['repeat_id'] if runtime else None, 'episode_id_or_null': None, 'rollout_arm_or_null': cell['rollout_arm'] if runtime else None, 'evidence_refs': [], 'metrics': {}, 'details': {'cell': json_deep_copy(cell) if runtime else None, 'execution_profile': cell['execution_profile'] if runtime else None, 'disposition': 'needs_rerun', 'evidence': {'status': 'missing', 'reason': 'required_stage_evidence_absent'}}}
    validate_admission_stage(row)
    return row

def _not_applicable_stage(*, task_id: str, task_hash: str, stage: str, cell: Mapping[str, Any], reason: str) -> Mapping[str, Any]:
    row = {'schema_version': 'task-admission-stage.v1', 'task_id': task_id, 'task_hash': task_hash, 'backend': 'scoper.benchmark.behavior', 'stage': stage, 'status': 'not_applicable', 'failure_code_or_null': None, 'error_class': 'none', 'repeat_id_or_null': cell['repeat_id'], 'episode_id_or_null': None, 'rollout_arm_or_null': cell['rollout_arm'], 'evidence_refs': [], 'metrics': {}, 'details': {'cell': json_deep_copy(cell), 'execution_profile': cell['execution_profile'], 'disposition': 'rejected' if 'rejection' in reason else 'needs_rerun', 'evidence': {'status': 'not_applicable', 'reason': reason}}}
    validate_admission_stage(row)
    return row

def _count(rows: Sequence[Mapping[str, Any]], field: str) -> Mapping[str, int]:
    values = Counter((str(row.get(field)) for row in rows))
    return {key: int(values[key]) for key in sorted(values)}

def _failure_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    values = Counter()
    for row in rows:
        for code in [row.get('primary_failure_code_or_null'), *(row.get('secondary_failure_codes') or [])]:
            if code is not None:
                values[str(code)] += 1
    return {key: int(values[key]) for key in sorted(values)}

def _invalid_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    values = Counter((stage['error_class'] for row in rows for stage in row['stage_results'] if stage['status'] == 'invalid'))
    return {key: int(values[key]) for key in sorted(values)}

def _implementation_hash() -> str:
    return sha256_file(Path(__file__).with_name('admission_backend.py'))

def _stage_implementation_hashes() -> Mapping[str, str]:
    backend_hash = _implementation_hash()
    pipeline_hash = sha256_file(Path(__file__))
    return {stage: backend_hash if stage == 'reachability' else pipeline_hash for stage in ADMISSION_STAGE_ORDER}

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
__all__ = ['TaskAdmissionPipeline', 'freeze_admitted_task_cells', 'validate_admission_evidence_files']
