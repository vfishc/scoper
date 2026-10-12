from __future__ import annotations
import copy
import fcntl
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.procthor.export.ids import canonical_json_bytes, sha256_json
from scoper.benchmark.procthor.export.schema import COMMON_CONTRACT_SHA256, SchemaValidationError, TASK_BANK_HASH_SCHEMA_VERSION, TASK_SET_HASH_SCHEMA_VERSION, load_json_strict, load_jsonl_strict, validate_task_set_envelope
from scoper.benchmark.procthor.export.task_identity import _normalize_procthor_task_for_identity_unchecked
from scoper.benchmark.procthor.paths import repo_root as default_repo_root
from scoper.benchmark.procthor.benchmark.admission_schema import ADMISSION_STAGE_ORDER, RUNTIME_ADMISSION_STAGES, STATIC_ADMISSION_STAGES, report_has_unresolved_evidence, stage_sort_key, task_disposition, validate_admission_cell, validate_admission_report, validate_admission_stage, validate_admission_task, validate_reachability_preflight

class TaskAdmissionPipeline:

    def __init__(self, *, protocol: Mapping[str, Any], protocol_sha256: str, candidate_artifact_schema: str, candidate_artifact_hash: str, source_task_bank_hash: str, code_commit: str, dirty_diff_sha256_or_null: str | None, execution_profiles: Sequence[str], stage_implementation_hashes: Mapping[str, str], complete_candidate_set: bool=True) -> None:
        self.protocol = copy.deepcopy(dict(protocol))
        self.protocol_sha256 = protocol_sha256
        self.candidate_artifact_schema = candidate_artifact_schema
        self.candidate_artifact_hash = candidate_artifact_hash
        self.source_task_bank_hash = source_task_bank_hash
        self.code_commit = code_commit
        self.dirty_diff_sha256_or_null = dirty_diff_sha256_or_null
        self.execution_profiles = tuple(execution_profiles)
        self.stage_implementation_hashes = copy.deepcopy(dict(stage_implementation_hashes))
        self.complete_candidate_set = bool(complete_candidate_set)

    def stage_from_preflight(self, *, task_id: str, task_hash: str, episode_id: str | None, evidence_sha256: str, evidence_relative_path: str, preflight: Mapping[str, Any], additional_evidence_refs: Sequence[Mapping[str, Any]]=()) -> Mapping[str, Any]:
        validate_reachability_preflight(preflight)
        cell = preflight['cell']
        row = _stage(task_id=task_id, task_hash=task_hash, name='reachability', status=preflight['status'], error_class=preflight['error_class'], failure_code=preflight['failure_code'], cell=cell, episode_id=episode_id, evidence_refs=[{'artifact_sha256': evidence_sha256, 'relative_path_or_record_id': evidence_relative_path, 'kind': 'trajectory_bundle_admission_preflight' if episode_id is not None else 'run_level_admission_preflight'}] + [copy.deepcopy(dict(item)) for item in additional_evidence_refs], metrics={'required_role_count': len(preflight['required_roles']), 'reachable_role_count': sum((row.get('primary_reachable') is True and row.get('audit_reachable') is True for row in preflight['required_roles'])), 'interaction_pose_count': sum((int(row.get('interaction_pose_count') or 0) for row in preflight['required_roles'])), 'reachable_interaction_pose_count': sum((int(row.get('reachable_interaction_pose_count') or 0) for row in preflight['required_roles']))}, evidence={'status': 'validated' if preflight['status'] in {'pass', 'fail'} else 'invalid', 'disposition': preflight['disposition'], 'alignment': copy.deepcopy(dict(preflight['alignment']))})
        return row

    def task_result(self, *, task: Mapping[str, Any], task_hash: str, lineage_id: str, source_split: str, required_cells: Sequence[Mapping[str, Any]], stage_results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        cells = sorted((copy.deepcopy(dict(cell)) for cell in required_cells), key=lambda item: item['cell_id'])
        stages = _complete_stage_matrix(task_id=str(task['id']), task_hash=task_hash, cells=cells, stages=stage_results)
        disposition, primary = task_disposition(stages)
        secondary = sorted({row['failure_code_or_null'] for row in stages if row['failure_code_or_null'] is not None and row['failure_code_or_null'] != primary})
        source = task.get('source') if isinstance(task.get('source'), Mapping) else {}
        template = task.get('template') if isinstance(task.get('template'), Mapping) else {}
        tracks = []
        if disposition == 'admitted':
            tracks = sorted(('native' if cell['execution_profile'] == 'native_ai2thor' else 'compatibility' for cell in cells))
            tracks = sorted(set(tracks))
        result = {'schema_version': 'task-admission-task.v1', 'task_id': task['id'], 'task_hash': task_hash, 'lineage_id': lineage_id, 'source_split': source_split, 'family': task.get('category'), 'template_id': template.get('template_id'), 'scene_id': source.get('house_id'), 'asset_lineage_ids': sorted(set(task.get('asset_lineage_ids') or [])), 'required_cells': cells, 'stage_results': stages, 'disposition': disposition, 'primary_failure_code_or_null': primary, 'secondary_failure_codes': secondary, 'admitted_tracks': tracks, 'quality_summary': {'required_cell_count': len(cells), 'pass_count': sum((row['status'] == 'pass' for row in stages)), 'fail_count': sum((row['status'] == 'fail' for row in stages)), 'invalid_count': sum((row['status'] == 'invalid' for row in stages))}}
        validate_admission_task(result)
        return result

    def report(self, tasks: Sequence[Mapping[str, Any]], *, input_run_manifest_hashes: Sequence[str], created_at_utc: str) -> Mapping[str, Any]:
        rows = sorted((copy.deepcopy(dict(task)) for task in tasks), key=lambda item: item['task_id'])
        for row in rows:
            validate_admission_task(row)
        benchmark = self.protocol.get('benchmark') if isinstance(self.protocol.get('benchmark'), Mapping) else {}
        lifecycle = self.protocol.get('lifecycle') if isinstance(self.protocol.get('lifecycle'), Mapping) else {}
        admission = self.protocol.get('admission') if isinstance(self.protocol.get('admission'), Mapping) else {}
        seed_policy = self.protocol.get('seed_policy') if isinstance(self.protocol.get('seed_policy'), Mapping) else {}
        report = {'schema_version': 'task-admission-report.v1', 'backend': 'eb_procthor', 'benchmark_id': benchmark.get('id'), 'benchmark_version': benchmark.get('version'), 'protocol_sha256': self.protocol_sha256, 'protocol_status': lifecycle.get('status') or 'provisional', 'candidate_artifact_schema': self.candidate_artifact_schema, 'candidate_artifact_hash': self.candidate_artifact_hash, 'source_task_bank_hash': self.source_task_bank_hash, 'common_contract_sha256': COMMON_CONTRACT_SHA256, 'code_commit': self.code_commit, 'dirty_diff_sha256_or_null': self.dirty_diff_sha256_or_null, 'stage_order': list(ADMISSION_STAGE_ORDER), 'stage_implementation_hashes': copy.deepcopy(self.stage_implementation_hashes), 'required_repeat_ids': list(seed_policy.get('admission_repeat_ids') or []), 'required_arm_rules': copy.deepcopy(admission.get('required_arm_rules') or []), 'execution_profiles': sorted(set(self.execution_profiles)), 'tasks': rows, 'counts_by_disposition': _count(rows, 'disposition'), 'counts_by_failure_code': _failure_counts(rows), 'counts_by_family': _count(rows, 'family'), 'counts_by_template': _count(rows, 'template_id'), 'counts_by_scene': _count(rows, 'scene_id'), 'invalid_episode_count_by_error_class': _invalid_counts(rows), 'input_run_manifest_hashes': sorted(set(input_run_manifest_hashes)), 'complete_candidate_set': self.complete_candidate_set, 'report_content_hash': '', 'created_at_utc': created_at_utc}
        identity = {key: value for key, value in report.items() if key not in {'created_at_utc', 'report_content_hash'}}
        report['report_content_hash'] = sha256_json(identity)
        validate_admission_report(report)
        return report

def write_standalone_preflight_evidence(run_root: Path, preflight: Mapping[str, Any], *, fsync: bool) -> Path:
    validate_reachability_preflight(preflight)
    root = Path(run_root)
    directory = root / 'admission_cells'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (preflight['cell']['cell_id'] + '.json')
    payload = canonical_json_bytes(dict(preflight)) + b'\n'
    lock_path = directory / '.admission_cells.lock'
    lock_path.touch(exist_ok=True)
    with lock_path.open('r+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                if path.read_bytes() != payload:
                    raise SchemaValidationError('admission_evidence_conflict', 'existing cell evidence differs')
                return path
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 292)
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(payload)
                handle.flush()
                if fsync:
                    os.fsync(handle.fileno())
            if fsync:
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return path

def freeze_admitted_task_cells(*, task_bank_path: Path, task_set_path: Path, protocol: Mapping[str, Any], admission_report: Mapping[str, Any], output_path: Path, repo_root: Path | None=None) -> Mapping[str, Any]:
    validate_admission_report(admission_report)
    if sha256_json(dict(protocol)) != admission_report['protocol_sha256']:
        raise SchemaValidationError('ADMISSION_PROTOCOL_MISMATCH', 'freeze protocol differs from admission report')
    if admission_report['complete_candidate_set'] is not True:
        raise SchemaValidationError('admission_freeze_incomplete', 'limited candidate reports cannot be frozen')
    if report_has_unresolved_evidence(admission_report):
        raise SchemaValidationError('admission_needs_rerun', 'needs_rerun evidence blocks task-cell freeze')
    repository = Path(repo_root or default_repo_root()).resolve()
    tasks = load_jsonl_strict(task_bank_path)
    if not all((isinstance(task, dict) for task in tasks)):
        raise SchemaValidationError('task_bank_schema_invalid', 'every task-bank line must be an object')
    bank = {}
    for ordinal, task in enumerate(tasks):
        task_id = task.get('id')
        if not isinstance(task_id, str) or not task_id:
            task_id = '__invalid_task_{:06d}'.format(ordinal)
        bank.setdefault(task_id, task)
    task_set = load_json_strict(task_set_path)
    if not isinstance(task_set, Mapping):
        raise SchemaValidationError('task_set_schema_invalid', 'task set must be an object')
    validate_task_set_envelope(task_set)
    missing = [task_id for task_id in task_set['task_ids'] if task_id not in bank]
    assets = protocol['assets']['procthor_dataset']
    all_identities = {}
    task_hashes = {}
    for task_id, task in bank.items():
        try:
            identity = _normalize_procthor_task_for_identity_unchecked(task, repo_root=repository, dataset_name=assets['name'], dataset_revision=assets['revision'])
            all_identities[task_id] = identity
            task_hashes[task_id] = identity.task_hash
        except (SchemaValidationError, OSError, UnicodeError, ValueError):
            task_hashes[task_id] = sha256_json(task)
    for task_id in missing:
        task_hashes[task_id] = _missing_task_hash(task_id)
    candidate_identities = {task_id: all_identities[task_id] for task_id in task_set['task_ids'] if task_id in all_identities}
    bank_hash = _task_bank_hash_from_hashes({task_id: task_hashes[task_id] for task_id in bank})
    set_hash = _task_set_hash_from_hashes(task_set, task_hashes)
    if bank_hash != admission_report['source_task_bank_hash'] or set_hash != admission_report['candidate_artifact_hash']:
        raise SchemaValidationError('admission_artifact_conflict', 'candidate hashes differ from report')
    result_by_id = {task['task_id']: task for task in admission_report['tasks']}
    if set(result_by_id) != set(task_set['task_ids']):
        raise SchemaValidationError('admission_artifact_conflict', 'report task membership differs from candidate set')
    frozen_tasks = []
    cells = []
    for task_id in task_set['task_ids']:
        result = result_by_id[task_id]
        identity = candidate_identities.get(task_id)
        if result['task_hash'] != task_hashes[task_id]:
            raise SchemaValidationError('admission_artifact_conflict', 'admitted task hash differs')
        if result['disposition'] == 'admitted':
            if identity is None:
                raise SchemaValidationError('admission_artifact_conflict', 'admitted task identity is unavailable')
            frozen_tasks.append(copy.deepcopy(dict(identity.identity_task)))
            cells.extend(copy.deepcopy(list(result['required_cells'])))
    artifact = {'schema_version': 'eb-procthor-admitted-task-cells.v1', 'source_task_bank_hash': bank_hash, 'source_task_set_hash': set_hash, 'admission_report_content_hash': admission_report['report_content_hash'], 'tasks': sorted(frozen_tasks, key=lambda item: item['id']), 'admitted_cells': sorted(cells, key=lambda item: item['cell_id'])}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(artifact) + b'\n'
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 292)
    except FileExistsError as exc:
        raise SchemaValidationError('admitted_task_cells_exists', 'output already exists') from exc
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != payload:
        raise SchemaValidationError('admitted_task_cells_invalid', 'frozen artifact reread differs')
    return artifact

def _task_bank_hash_from_hashes(task_hashes: Mapping[str, str]) -> str:
    return sha256_json({'schema_version': TASK_BANK_HASH_SCHEMA_VERSION, 'backend': 'eb_procthor', 'tasks': [{'task_id': task_id, 'task_hash': task_hashes[task_id]} for task_id in sorted(task_hashes)]})

def _missing_task_hash(task_id: str) -> str:
    return sha256_json({'schema_version': 'missing-task-identity.v1', 'backend': 'eb_procthor', 'task_id': task_id})

def _task_set_hash_from_hashes(task_set: Mapping[str, Any], task_hashes: Mapping[str, str]) -> str:
    return sha256_json({'schema_version': TASK_SET_HASH_SCHEMA_VERSION, 'backend': 'eb_procthor', 'task_set': {key: task_set[key] for key in ('schema_version', 'name', 'version', 'lineage_id', 'benchmark_split')}, 'tasks': sorted(({'task_id': task_id, 'task_hash': task_hashes[task_id]} for task_id in task_set['task_ids']), key=lambda item: item['task_id'])})

def _complete_stage_matrix(*, task_id: str, task_hash: str, cells: Sequence[Mapping[str, Any]], stages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = [copy.deepcopy(dict(stage)) for stage in stages]
    for row in rows:
        validate_admission_stage(row)
    static_present = {row['stage'] for row in rows if row['stage'] in STATIC_ADMISSION_STAGES}
    for stage in STATIC_ADMISSION_STAGES:
        if stage not in static_present:
            rows.append(_missing_stage(task_id, task_hash, stage, None))
    present = {(row['stage'], (row.get('details') or {}).get('cell', {}).get('cell_id')) for row in rows if row['stage'] in RUNTIME_ADMISSION_STAGES}
    reachability = {row['details']['cell']['cell_id']: row for row in rows if row['stage'] == 'reachability' and isinstance(row['details'].get('cell'), Mapping)}
    static_fail = any((row['stage'] in STATIC_ADMISSION_STAGES and row['status'] == 'fail' for row in rows))
    static_invalid = any((row['stage'] in STATIC_ADMISSION_STAGES and row['status'] == 'invalid' for row in rows))
    failed_stage_index_by_cell: dict[str, int] = {}
    for row in rows:
        cell = (row.get('details') or {}).get('cell')
        if not isinstance(cell, Mapping) or row['status'] != 'fail':
            continue
        index = ADMISSION_STAGE_ORDER.index(row['stage'])
        current = failed_stage_index_by_cell.get(cell['cell_id'])
        if current is None or index < current:
            failed_stage_index_by_cell[cell['cell_id']] = index
    reachability_index = ADMISSION_STAGE_ORDER.index('reachability')
    for cell in cells:
        validate_admission_cell(cell)
        upstream = reachability.get(cell['cell_id'])
        for stage in RUNTIME_ADMISSION_STAGES:
            if (stage, cell['cell_id']) in present:
                continue
            stage_index = ADMISSION_STAGE_ORDER.index(stage)
            upstream_failure_index = failed_stage_index_by_cell.get(cell['cell_id'])
            if static_fail or static_invalid:
                rows.append(_not_applicable_stage(task_id, task_hash, stage, cell, 'upstream_static_rejection' if static_fail else 'upstream_static_evidence_invalid'))
            elif upstream_failure_index is not None and stage_index > upstream_failure_index:
                rows.append(_not_applicable_stage(task_id, task_hash, stage, cell, 'upstream_stage_rejection'))
            elif upstream is not None and upstream['status'] in {'fail', 'invalid'} and (stage_index > reachability_index):
                rows.append(_not_applicable_stage(task_id, task_hash, stage, cell, 'upstream_reachability_rejection' if upstream['status'] == 'fail' else 'upstream_reachability_evidence_invalid'))
            else:
                rows.append(_missing_stage(task_id, task_hash, stage, cell))
    return sorted(rows, key=stage_sort_key)

def _stage(*, task_id: str, task_hash: str, name: str, status: str, error_class: str, failure_code: str | None, cell: Mapping[str, Any] | None, episode_id: str | None, evidence_refs: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = cell is not None
    row = {'schema_version': 'task-admission-stage.v1', 'task_id': task_id, 'task_hash': task_hash, 'backend': 'eb_procthor', 'stage': name, 'status': status, 'failure_code_or_null': failure_code, 'error_class': error_class, 'repeat_id_or_null': cell['repeat_id'] if runtime else None, 'episode_id_or_null': episode_id, 'rollout_arm_or_null': cell['rollout_arm'] if runtime else None, 'evidence_refs': copy.deepcopy(list(evidence_refs)), 'metrics': copy.deepcopy(dict(metrics)), 'details': {'cell': copy.deepcopy(dict(cell)) if runtime else None, 'execution_profile': cell['execution_profile'] if runtime else None, 'disposition': 'rejected' if status == 'fail' else 'needs_rerun' if status == 'invalid' else 'admitted', 'evidence': copy.deepcopy(dict(evidence))}}
    validate_admission_stage(row)
    return row

def _missing_stage(task_id: str, task_hash: str, name: str, cell: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _stage(task_id=task_id, task_hash=task_hash, name=name, status='invalid', error_class='evaluator_invalid', failure_code='ADMISSION_MISSING_REQUIRED_CELL', cell=cell, episode_id=None, evidence_refs=[], metrics={}, evidence={'status': 'missing', 'reason': 'required_stage_evidence_absent'})

def _not_applicable_stage(task_id: str, task_hash: str, name: str, cell: Mapping[str, Any], reason: str) -> Mapping[str, Any]:
    return _stage(task_id=task_id, task_hash=task_hash, name=name, status='not_applicable', error_class='none', failure_code=None, cell=cell, episode_id=None, evidence_refs=[], metrics={}, evidence={'status': 'not_applicable', 'reason': reason})

def _count(rows: Sequence[Mapping[str, Any]], field: str) -> Mapping[str, int]:
    values = Counter((str(row.get(field)) for row in rows))
    return {key: int(values[key]) for key in sorted(values)}

def _failure_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    values = Counter()
    for row in rows:
        for code in (row.get('primary_failure_code_or_null'), *(row.get('secondary_failure_codes') or [])):
            if code is not None:
                values[str(code)] += 1
    return {key: int(values[key]) for key in sorted(values)}

def _invalid_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    values = Counter((stage['error_class'] for row in rows for stage in row['stage_results'] if stage['status'] == 'invalid'))
    return {key: int(values[key]) for key in sorted(values)}
__all__ = ['TaskAdmissionPipeline', 'freeze_admitted_task_cells', 'write_standalone_preflight_evidence']
