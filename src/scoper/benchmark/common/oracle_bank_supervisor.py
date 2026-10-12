from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.common.canonical import canonical_json_bytes, sha256_json
ORACLE_BANK_CONFIG_SCHEMA_VERSION = 'oracle-video-bank-config.v1'
ORACLE_BANK_STATE_SCHEMA_VERSION = 'oracle-video-bank-state.v1'
ORACLE_BANK_ATTEMPT_SCHEMA_VERSION = 'oracle-video-bank-attempt.v1'
MASTER_SEED = 20260821
INFRA_FAILURE_CIRCUIT_BREAKER_COUNT = 6
TEMPLATE_INFRA_FAILURE_QUARANTINE_COUNT = 6
RECOVERABLE_QA_BLOCKER_PREFIX = 'qa_attachment_blocked:'
COMPOSITE_REAL_ATTEMPT_LIMIT = 500
COMPOSITE_STATIC_ATTEMPT_LIMIT = 1500

def redundant_composite_child_indices(child_goal_predicates: Sequence[Sequence[Mapping[str, Any]]]) -> list[int]:
    seen: set[bytes] = set()
    redundant: list[int] = []
    for index, predicates in enumerate(child_goal_predicates):
        signatures = {canonical_json_bytes(_goal_obligation_value(predicate)) for predicate in predicates if isinstance(predicate, Mapping)}
        if not signatures or signatures.issubset(seen):
            redundant.append(index)
        seen.update(signatures)
    return redundant

def _goal_obligation_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _goal_obligation_value(item) for key, item in value.items() if key not in {'id', 'owner_subtask_id'}}
    if isinstance(value, list):
        return [_goal_obligation_value(item) for item in value]
    return value

def validate_oracle_bank_config(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {'backend', 'gpu_ids', 'layouts', 'master_seed', 'output_root', 'schema_version', 'source_manifest', 'templates', 'timeout_seconds', 'worker_extra_args', 'worker_python', 'worker_script'}
    if set(value) != required:
        raise ValueError('Oracle bank config fields are invalid')
    if value.get('schema_version') != ORACLE_BANK_CONFIG_SCHEMA_VERSION:
        raise ValueError('Oracle bank config schema is invalid')
    if value.get('backend') not in {'behavior', 'procthor'}:
        raise ValueError('Oracle bank backend is invalid')
    if int(value.get('master_seed', -1)) != MASTER_SEED:
        raise ValueError('Oracle bank master seed must be {}'.format(MASTER_SEED))
    gpu_ids = value.get('gpu_ids')
    if not isinstance(gpu_ids, list) or not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Oracle bank config requires at least one unique GPU id')
    if any((isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in gpu_ids)):
        raise ValueError('Oracle bank GPU ids are invalid')
    templates = value.get('templates')
    if not isinstance(templates, list) or not templates:
        raise ValueError('Oracle bank config requires at least one template')
    template_ids = []
    for item in templates:
        if not isinstance(item, Mapping) or set(item) != {'category', 'template_id'}:
            raise ValueError('Oracle bank template records are invalid')
        template_ids.append(str(item.get('template_id') or ''))
    if any((not item for item in template_ids)) or len(set(template_ids)) != len(template_ids):
        raise ValueError('Oracle bank template ids are invalid or duplicated')
    layouts = value.get('layouts')
    if not isinstance(layouts, list) or len(layouts) < 1:
        raise ValueError('Oracle bank requires at least one layout')
    layout_ids = []
    for item in layouts:
        if not isinstance(item, Mapping) or not isinstance(item.get('layout_id'), str):
            raise ValueError('Oracle bank layout record is invalid')
        layout_ids.append(item['layout_id'])
    if len(set(layout_ids)) != len(layout_ids):
        raise ValueError('Oracle bank layout ids must be unique')
    if not isinstance(value.get('worker_extra_args'), list) or any((not isinstance(item, str) for item in value['worker_extra_args'])):
        raise ValueError('Oracle bank worker_extra_args are invalid')
    if isinstance(value.get('timeout_seconds'), bool) or int(value.get('timeout_seconds', 0)) <= 0:
        raise ValueError('Oracle bank timeout_seconds is invalid')
    normalized = copy.deepcopy(dict(value))
    normalized['gpu_ids'] = [int(item) for item in gpu_ids]
    normalized['timeout_seconds'] = int(value['timeout_seconds'])
    return normalized

def validate_oracle_bank_attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {'artifacts', 'failure_class', 'failure_code', 'job_id', 'layout_id', 'real_oracle_attempted', 'schema_version', 'status'}
    optional = {'selected_template_ids', 'static_diagnostics'}
    if not required.issubset(value) or set(value) - required - optional or value.get('schema_version') != ORACLE_BANK_ATTEMPT_SCHEMA_VERSION:
        raise ValueError('Oracle bank attempt result fields are invalid')
    if value.get('status') not in {'accepted', 'failed', 'qa_attachment_blocked'}:
        raise ValueError('Oracle bank attempt status is invalid')
    if value.get('failure_class') not in {None, 'infra', 'oracle', 'qa'}:
        raise ValueError('Oracle bank attempt failure_class is invalid')
    if not isinstance(value.get('real_oracle_attempted'), bool):
        raise ValueError('Oracle bank real_oracle_attempted is invalid')
    if not isinstance(value.get('artifacts'), Mapping):
        raise ValueError('Oracle bank attempt artifacts are invalid')
    selected_template_ids = value.get('selected_template_ids')
    if selected_template_ids is not None and (not isinstance(selected_template_ids, list) or any((not isinstance(item, str) or not item for item in selected_template_ids)) or len(set(selected_template_ids)) != len(selected_template_ids)):
        raise ValueError('Oracle bank selected template ids are invalid')
    if 'static_diagnostics' in value and (not isinstance(value['static_diagnostics'], Mapping)):
        raise ValueError('Oracle bank static diagnostics are invalid')
    return copy.deepcopy(dict(value))

def rank_layout_composite_candidates(records: Sequence[Mapping[str, Any]], *, size: int, seed: int, limit: int=256) -> list[list[str]]:
    if int(size) not in {2, 3}:
        raise ValueError('Composite size must be two or three')
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for raw in records:
        candidate_id = str(raw.get('candidate_id') or '')
        template_id = str(raw.get('template_id') or '')
        start_pose_key = str(raw.get('start_pose_key') or '')
        if not candidate_id or not template_id or (not start_pose_key) or (raw.get('supported') is False):
            continue
        record = {'candidate_id': candidate_id, 'category': str(raw.get('category') or ''), 'template_id': template_id, 'template_rank': int(raw.get('template_rank', 0))}
        grouped.setdefault(start_pose_key, {}).setdefault(template_id, []).append(record)
    ranked: list[tuple[tuple[Any, ...], list[str]]] = []
    for start_pose_key, by_template in sorted(grouped.items()):
        for candidates in by_template.values():
            candidates.sort(key=lambda item: item['candidate_id'])
        for template_ids in itertools.combinations(sorted(by_template), int(size)):
            candidate_lists = [by_template[template_id][:2] for template_id in template_ids]
            for children in itertools.product(*candidate_lists):
                categories = {item['category'] for item in children}
                candidate_ids = [item['candidate_id'] for item in children]
                digest = hashlib.sha256('{}:{}:{}'.format(int(seed), start_pose_key, ','.join(candidate_ids)).encode()).hexdigest()
                score = (int(size) - len(categories), sum((item['template_rank'] for item in children)), digest, tuple(candidate_ids))
                ranked.append((score, candidate_ids))
    ranked.sort(key=lambda item: item[0])
    return [candidate_ids for _score, candidate_ids in ranked[:max(0, int(limit))]]

class OracleBankSupervisor:

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = validate_oracle_bank_config(config)
        self.root = Path(self.config['output_root']).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.attempts_root = self.root / 'attempts'
        self.workers_root = self.root / 'workers'
        self.attempts_root.mkdir(exist_ok=True)
        self.workers_root.mkdir(exist_ok=True)
        self.state_path = self.root / 'state.json'
        self.status_path = self.root / 'status.json'
        self.manifest_path = self.root / 'run_manifest.json'
        self.templates = [dict(item) for item in self.config['templates']]
        self.layouts = [dict(item) for item in self.config['layouts']]
        self.layout_by_id = {item['layout_id']: item for item in self.layouts}
        self.state = self._load_or_initialize_state()
        self._startup_accepted_count = len(self.state['accepted'])
        self.processes: dict[int, dict[str, Any]] = {}
        self._last_heartbeat = 0.0

    def run(self) -> int:
        self._write_manifest()
        while True:
            self._collect_finished()
            terminal = self._terminal_status()
            if terminal is not None:
                self._write_status(terminal)
                self._write_final_outputs(terminal)
                return 0 if terminal in {'completed', 'completed_with_gaps'} else 2
            if self.state.get('blocked_reason'):
                if time.monotonic() - self._last_heartbeat >= 60 or not self.status_path.exists():
                    self._write_status('running')
                time.sleep(1.0)
                continue
            self._advance_phase()
            launched = False
            for gpu_id in self.config['gpu_ids']:
                if gpu_id not in self.processes:
                    job = self._next_job()
                    if job is None:
                        break
                    self._launch(gpu_id, job)
                    launched = True
            if not self.processes and (not launched):
                previous_phase = self.state['phase']
                self._advance_phase()
                if self.state['phase'] != previous_phase or self._terminal_status() is not None:
                    continue
                self.state['blocked_reason'] = 'no_schedulable_job:{}'.format(self.state['phase'])
                self._write_state(self.state)
                continue
            if time.monotonic() - self._last_heartbeat >= 60 or not self.status_path.exists():
                self._write_status('running')
            time.sleep(1.0)

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding='utf-8'))
            if state.get('schema_version') != ORACLE_BANK_STATE_SCHEMA_VERSION:
                raise ValueError('Oracle bank state schema is invalid')
            if state.get('config_sha256') != sha256_json(self.config):
                raise ValueError('Oracle bank resume config differs')
            blocked_reason = str(state.get('blocked_reason') or '')
            if blocked_reason.startswith(RECOVERABLE_QA_BLOCKER_PREFIX):
                state['blocked_reason'] = None
            elif blocked_reason.startswith('atomic_template_attempt_limit:'):
                state['blocked_reason'] = None
            elif blocked_reason == 'composite_attempt_limit':
                state['blocked_reason'] = None
                state['composite_exhausted'] = True
            state.setdefault('composite_exhausted', False)
            state.setdefault('exhausted_templates', {})
            state['inflight'] = {}
            return state
        state = {'accepted': [], 'attempt_counter': 0, 'attempts': [], 'blocked_reason': None, 'composite_exhausted': False, 'config_sha256': sha256_json(self.config), 'exhausted_templates': {}, 'inflight': {}, 'phase': 'atomic_pilot', 'retries': [], 'schema_version': ORACLE_BANK_STATE_SCHEMA_VERSION}
        self._write_state(state)
        return state

    def _next_job(self) -> dict[str, Any] | None:
        while self.state['retries']:
            retry = self.state['retries'].pop(0)
            if retry.get('job_type') == 'atomic' and retry.get('template_ids', [None])[0] in self.state['exhausted_templates']:
                continue
            return retry
        phase = self.state['phase']
        if phase.startswith('atomic'):
            target = 1 if phase == 'atomic_pilot' else 10
            counts = self._reserved_atomic_counts()
            for item in self.templates:
                template_id = item['template_id']
                if counts.get(template_id, 0) < target and template_id not in self.state['exhausted_templates'] and (self._real_attempt_count(template_id=template_id, job_type='atomic') >= 50):
                    self._exhaust_template(template_id, 'real_oracle_attempt_limit')
            attempt_counts = {item['template_id']: self._job_attempt_count_for_template(item['template_id']) for item in self.templates}
            candidates = sorted((item for item in self.templates if counts.get(item['template_id'], 0) < target and item['template_id'] not in self.state['exhausted_templates']), key=lambda item: (attempt_counts[item['template_id']], counts.get(item['template_id'], 0), item['template_id']))
            for template in candidates:
                template_id = template['template_id']
                selection_salt = 'atomic:{}'.format(template_id)
                layout = self._next_layout(selection_salt)
                if layout is not None:
                    return self._new_job('atomic', layout, [template_id], selection_salt)
                self._exhaust_template(template_id, 'layout_bank_exhausted')
            return None
        target_by_size = {2: 3, 3: 2} if phase == 'composite_pilot' else {2: 50, 3: 50}
        counts = self._reserved_composite_counts()
        attempts = self._composite_attempt_counts(include_reserved=True)
        sizes = sorted(target_by_size, key=lambda size: (attempts.get(size, 0) / max(1, target_by_size[size] - counts.get(size, 0)), attempts.get(size, 0), size))
        for size in sizes:
            if counts.get(size, 0) >= target_by_size[size]:
                continue
            if self._real_attempt_count(job_type='composite') >= COMPOSITE_REAL_ATTEMPT_LIMIT or self._job_attempt_count(job_type='composite', include_reserved=True) >= COMPOSITE_STATIC_ATTEMPT_LIMIT:
                self.state['composite_exhausted'] = True
                self._write_state(self.state)
                continue
            selection_salt = 'composite:{}:layout_first'.format(size)
            layout = self._next_layout(selection_salt)
            if layout is not None:
                return self._new_job('composite', layout, [], selection_salt, candidate_template_ids=self._composite_candidate_template_ids(layout), composite_size=size)
        return None

    def _new_job(self, job_type: str, layout: Mapping[str, Any], template_ids: Sequence[str], selection_salt: str, *, candidate_template_ids: Sequence[str]=(), composite_size: int | None=None) -> dict[str, Any]:
        counter = int(self.state['attempt_counter'])
        self.state['attempt_counter'] = counter + 1
        identity = '{}:{}:{}:{}:{}'.format(self.config['backend'], job_type, layout['layout_id'], int(composite_size or 0), ','.join(template_ids or candidate_template_ids))
        job_id = '{:06d}_{}'.format(counter, hashlib.sha256(identity.encode()).hexdigest()[:12])
        job = {'backend': self.config['backend'], 'job_id': job_id, 'job_type': job_type, 'layout': copy.deepcopy(dict(layout)), 'master_seed': MASTER_SEED, 'retry_count': 0, 'selection_salt': selection_salt, 'seed': int(hashlib.sha256((identity + ':20260821').encode()).hexdigest()[:8], 16) % 2 ** 31, 'template_ids': list(template_ids)}
        if job_type == 'composite':
            job['candidate_template_ids'] = list(candidate_template_ids)
            job['composite_size'] = int(composite_size or len(template_ids))
        self._write_state(self.state)
        return job

    def _launch(self, gpu_id: int, job: Mapping[str, Any]) -> None:
        attempt_dir = self.attempts_root / str(job['job_id'])
        attempt_dir.mkdir(parents=True, exist_ok=True)
        job_path = attempt_dir / 'job.json'
        _atomic_write_json(job_path, job)
        worker_dir = self.workers_root / 'gpu_{}'.format(gpu_id)
        worker_dir.mkdir(exist_ok=True)
        ready_path = worker_dir / 'worker_ready.json'
        ready_path.unlink(missing_ok=True)
        stdout = (attempt_dir / 'worker_stdout.log').open('ab')
        stderr = (attempt_dir / 'worker_stderr.log').open('ab')
        command = [str(self.config['worker_python']), str(self.config['worker_script']), '--job-file', str(job_path), '--output-dir', str(attempt_dir), '--ready-file', str(ready_path), '--gpu-device', str(gpu_id), *self.config['worker_extra_args']]
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        record = {'command': command, 'gpu_id': gpu_id, 'job': copy.deepcopy(dict(job)), 'process': process, 'ready_path': str(ready_path), 'started_at': time.time(), 'stderr': stderr, 'stdout': stdout}
        self.processes[gpu_id] = record
        self.state['inflight'][str(gpu_id)] = {'job_id': job['job_id'], 'layout_id': job['layout']['layout_id'], 'pid': process.pid}
        self._write_state(self.state)

    def _collect_finished(self) -> None:
        for gpu_id, record in list(self.processes.items()):
            process = record['process']
            if process.poll() is None:
                if time.time() - record['started_at'] > int(self.config['timeout_seconds']):
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                else:
                    continue
            record['stdout'].close()
            record['stderr'].close()
            job = record['job']
            attempt_dir = self.attempts_root / job['job_id']
            result_path = attempt_dir / 'attempt_result.json'
            try:
                result = validate_oracle_bank_attempt(json.loads(result_path.read_text(encoding='utf-8')))
                if result['job_id'] != job['job_id'] or result['layout_id'] != job['layout']['layout_id']:
                    raise ValueError('attempt result identity differs')
                if result['status'] == 'accepted' and job.get('job_type') == 'composite':
                    if len(result.get('selected_template_ids') or []) != _composite_size(job):
                        raise ValueError('accepted composite selected template ids differ')
            except Exception as exc:
                result = {'artifacts': {}, 'failure_class': 'infra', 'failure_code': 'ATTEMPT_RESULT_INVALID:{}'.format(type(exc).__name__), 'job_id': job['job_id'], 'layout_id': job['layout']['layout_id'], 'real_oracle_attempted': False, 'schema_version': ORACLE_BANK_ATTEMPT_SCHEMA_VERSION, 'status': 'failed'}
            attempt_job = copy.deepcopy(job)
            if result.get('selected_template_ids'):
                attempt_job['template_ids'] = list(result['selected_template_ids'])
            attempt_record = {'exit_code': process.returncode, 'job': attempt_job, 'provenance': {'mode': 'generated', 'source_manifest': copy.deepcopy(self.config['source_manifest'])}, 'result': result}
            if result['status'] == 'accepted':
                try:
                    if process.returncode != 0:
                        raise ValueError('accepted worker exited nonzero')
                    self._validate_accepted_artifacts(result)
                except Exception as exc:
                    result = {'artifacts': {}, 'failure_class': 'infra', 'failure_code': 'ACCEPTED_ARTIFACT_INVALID:{}'.format(type(exc).__name__), 'job_id': job['job_id'], 'layout_id': job['layout']['layout_id'], 'real_oracle_attempted': True, 'schema_version': ORACLE_BANK_ATTEMPT_SCHEMA_VERSION, 'status': 'failed'}
                    attempt_record['result'] = result
            self.state['attempts'].append(attempt_record)
            template_infra_code = self._repeated_template_infra_failure_code(job)
            if template_infra_code is not None:
                template_id = job['template_ids'][0]
                self._exhaust_template(template_id, 'repeated_template_infra:{}'.format(template_infra_code))
                self.state['retries'] = [retry for retry in self.state['retries'] if retry.get('template_ids') != [template_id]]
            repeated_infra_code = self._repeated_infra_failure_code()
            if repeated_infra_code is not None and (not self.state.get('blocked_reason')):
                self.state['blocked_reason'] = 'infra_failure_circuit_breaker:{}'.format(repeated_infra_code)
            if result['status'] == 'accepted':
                self.state['accepted'].append(attempt_record)
            elif result['failure_class'] == 'infra' and int(job.get('retry_count', 0)) < 1 and (template_infra_code is None):
                retry = copy.deepcopy(job)
                retry['retry_count'] = int(job.get('retry_count', 0)) + 1
                retry['job_id'] = '{}_retry{}'.format(job['job_id'], retry['retry_count'])
                self.state['retries'].append(retry)
            self.state['inflight'].pop(str(gpu_id), None)
            self.processes.pop(gpu_id, None)
            self._write_state(self.state)
            self._write_qa_pending()

    def _validate_accepted_artifacts(self, result: Mapping[str, Any]) -> None:
        required = {'base_task', 'base_trace', 'final_task', 'qa_replay', 'video'}
        artifacts = result['artifacts']
        if not required.issubset(artifacts) or set(artifacts) - required - {'trace_quality'}:
            raise ValueError('accepted attempt artifact set is invalid')
        for name, artifact in artifacts.items():
            if not isinstance(artifact, Mapping) or set(artifact) != {'path', 'sha256'}:
                raise ValueError('accepted artifact record is invalid')
            path = Path(str(artifact['path']))
            if not path.is_file() or _sha256_file(path) != artifact['sha256']:
                raise ValueError('accepted artifact {} is missing or changed'.format(name))

    def _advance_phase(self) -> None:
        phase = self.state['phase']
        atomic = self._atomic_counts()
        composite = self._composite_counts()
        if phase == 'atomic_pilot' and self._atomic_phase_satisfied(atomic, 1):
            self.state['phase'] = 'composite_pilot'
        elif phase == 'composite_pilot' and (composite.get(2, 0) >= 3 and composite.get(3, 0) >= 2 or self.state.get('composite_exhausted')):
            self.state['phase'] = 'atomic_full'
        elif phase == 'atomic_full' and self._atomic_phase_satisfied(atomic, 10):
            self.state['phase'] = 'composite_full'
        self._write_state(self.state)

    def _terminal_status(self) -> str | None:
        if self.state.get('blocked_reason') and (not self.processes):
            return 'blocked'
        atomic = self._atomic_counts()
        composite = self._composite_counts()
        atomic_finished = self._atomic_phase_satisfied(atomic, 10)
        composite_finished = composite.get(2, 0) >= 50 and composite.get(3, 0) >= 50 or bool(self.state.get('composite_exhausted'))
        if atomic_finished and composite_finished and (not self.processes):
            quotas_met = all((atomic.get(item['template_id'], 0) >= 10 for item in self.templates)) and composite.get(2, 0) >= 50 and (composite.get(3, 0) >= 50)
            return 'completed' if quotas_met else 'completed_with_gaps'
        return None

    def _repeated_infra_failure_code(self) -> str | None:
        window = self.state['attempts'][-INFRA_FAILURE_CIRCUIT_BREAKER_COUNT:]
        if len(window) != INFRA_FAILURE_CIRCUIT_BREAKER_COUNT:
            return None
        codes = []
        template_ids = set()
        for item in window:
            result = item['result']
            if result.get('failure_class') != 'infra':
                return None
            codes.append(str(result.get('failure_code') or 'UNKNOWN_INFRA_FAILURE'))
            template_ids.update((str(value) for value in item['job'].get('template_ids') or item['job'].get('candidate_template_ids') or []))
        return codes[0] if len(set(codes)) == 1 and len(template_ids) >= 2 else None

    def _repeated_template_infra_failure_code(self, job: Mapping[str, Any]) -> str | None:
        if job.get('job_type') != 'atomic' or len(job.get('template_ids') or []) != 1:
            return None
        template_id = str(job['template_ids'][0])
        attempts = [item for item in self.state['attempts'] if item['job'].get('job_type') == 'atomic' and item['job'].get('template_ids') == [template_id]][-TEMPLATE_INFRA_FAILURE_QUARANTINE_COUNT:]
        if len(attempts) != TEMPLATE_INFRA_FAILURE_QUARANTINE_COUNT:
            return None
        codes = []
        for item in attempts:
            result = item['result']
            if result.get('failure_class') != 'infra':
                return None
            codes.append(str(result.get('failure_code') or 'UNKNOWN_INFRA_FAILURE'))
        return codes[0] if len(set(codes)) == 1 else None

    def _next_layout(self, salt: str) -> dict[str, Any] | None:
        inflight = {item['layout_id'] for item in self.state['inflight'].values()}
        ordered = sorted(self.layouts, key=lambda item: hashlib.sha256('{}:{}:{}'.format(MASTER_SEED, salt, item['layout_id']).encode()).hexdigest())
        attempted = {item['job']['layout']['layout_id'] for item in self.state['attempts'] if _job_salt(item['job']) == salt}
        return next((item for item in ordered if item['layout_id'] not in inflight | attempted), None)

    def _composite_template_ids(self, size: int, attempt_index: int) -> list[str]:
        atomic = self._atomic_counts()
        eligible = [item for item in self.templates if atomic.get(item['template_id'], 0) > 0 or item['template_id'] not in self.state['exhausted_templates']]
        combinations = list(itertools.combinations(eligible, size))
        if not combinations:
            return []
        combinations.sort(key=lambda group: (len(group) - len({item['category'] for item in group}), hashlib.sha256('{}:{}:{}'.format(MASTER_SEED, size, ','.join((item['template_id'] for item in group))).encode()).hexdigest()))
        return [item['template_id'] for item in combinations[attempt_index % len(combinations)]]

    def _composite_candidate_template_ids(self, layout: Mapping[str, Any]) -> list[str]:
        usage = {item['template_id']: 0 for item in self.templates}
        for accepted in self.state['accepted']:
            job = accepted['job']
            if job.get('job_type') != 'composite':
                continue
            for template_id in job.get('template_ids') or []:
                usage[str(template_id)] = usage.get(str(template_id), 0) + 1
        return [item['template_id'] for item in sorted(self.templates, key=lambda item: (usage.get(item['template_id'], 0), hashlib.sha256('{}:{}:{}'.format(MASTER_SEED, layout['layout_id'], item['template_id']).encode()).hexdigest()))]

    def _real_attempt_count(self, *, template_id: str | None=None, job_type: str | None=None) -> int:
        count = 0
        for item in self.state['attempts']:
            result = item['result']
            if not result.get('real_oracle_attempted'):
                continue
            if result.get('failure_class') in {'infra', 'qa'}:
                continue
            job = item['job']
            if template_id is not None and template_id not in job['template_ids']:
                continue
            if job_type is not None and job['job_type'] != job_type:
                continue
            count += 1
        return count

    def _atomic_phase_satisfied(self, counts: Mapping[str, int], target: int) -> bool:
        exhausted = self.state['exhausted_templates']
        return all((counts.get(item['template_id'], 0) >= target or item['template_id'] in exhausted for item in self.templates))

    def _exhaust_template(self, template_id: str, reason: str) -> None:
        if template_id in self.state['exhausted_templates']:
            return
        self.state['exhausted_templates'][template_id] = {'accepted': self._atomic_counts().get(template_id, 0), 'attempted': self._job_attempt_count_for_template(template_id), 'real_oracle_attempted': self._real_attempt_count(template_id=template_id, job_type='atomic'), 'reason': str(reason)}
        self._write_state(self.state)

    def _job_attempt_count_for_template(self, template_id: str) -> int:
        return sum((1 for item in self.state['attempts'] if item['job'].get('job_type') == 'atomic' and item['job'].get('template_ids') == [template_id]))

    def _job_attempt_count(self, *, job_type: str, include_reserved: bool=False) -> int:
        count = sum((1 for item in self.state['attempts'] if item['job']['job_type'] == job_type))
        if include_reserved:
            count += sum((1 for item in self.processes.values() if item['job']['job_type'] == job_type))
            count += sum((1 for job in self.state['retries'] if job['job_type'] == job_type))
        return count

    def _composite_attempt_counts(self, *, include_reserved: bool=False) -> dict[int, int]:
        result = {2: 0, 3: 0}
        for item in self.state['attempts']:
            job = item['job']
            if job.get('job_type') == 'composite':
                size = _composite_size(job)
                result[size] = result.get(size, 0) + 1
        if include_reserved:
            reserved = [item['job'] for item in self.processes.values()] + list(self.state['retries'])
            for job in reserved:
                if job.get('job_type') == 'composite':
                    size = _composite_size(job)
                    result[size] = result.get(size, 0) + 1
        return result

    def _atomic_counts(self) -> dict[str, int]:
        result = {item['template_id']: 0 for item in self.templates}
        for item in self.state['accepted']:
            job = item['job']
            if job['job_type'] == 'atomic':
                result[job['template_ids'][0]] += 1
        return result

    def _composite_counts(self) -> dict[int, int]:
        result = {2: 0, 3: 0}
        for item in self.state['accepted']:
            job = item['job']
            if job['job_type'] == 'composite':
                result[_composite_size(job)] += 1
        return result

    def _reserved_atomic_counts(self) -> dict[str, int]:
        result = self._atomic_counts()
        reserved = [item['job'] for item in self.processes.values()] + list(self.state['retries'])
        for job in reserved:
            if job['job_type'] == 'atomic':
                result[job['template_ids'][0]] += 1
        return result

    def _reserved_composite_counts(self) -> dict[int, int]:
        result = self._composite_counts()
        reserved = [item['job'] for item in self.processes.values()] + list(self.state['retries'])
        for job in reserved:
            if job['job_type'] == 'composite':
                result[_composite_size(job)] += 1
        return result

    def _write_manifest(self) -> None:
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            if existing.get('config_sha256') != sha256_json(self.config):
                raise ValueError('existing Oracle bank manifest differs')
            return
        manifest = {'config': self.config, 'config_sha256': sha256_json(self.config), 'created_at_unix': time.time(), 'protocol': {'atomic_per_template': 10, 'atomic_template_attempt_limit': 50, 'composite_real_attempt_limit': COMPOSITE_REAL_ATTEMPT_LIMIT, 'composite_static_attempt_limit': COMPOSITE_STATIC_ATTEMPT_LIMIT, 'composite_two_child': 50, 'composite_three_child': 50, 'infra_retry_limit': 1, 'infra_failure_circuit_breaker_count': INFRA_FAILURE_CIRCUIT_BREAKER_COUNT, 'pilot_atomic_per_template': 1, 'pilot_composite_total': 5}, 'schema_version': 'oracle-video-bank-run-manifest.v1'}
        _atomic_write_json(self.manifest_path, manifest)

    def _write_status(self, status: str) -> None:
        startup_accepted_delta = len(self.state['accepted']) - self._startup_accepted_count
        payload = {'accepted_atomic': self._atomic_counts(), 'accepted_composite': self._composite_counts(), 'attempt_count': len(self.state['attempts']), 'blocked_reason': self.state.get('blocked_reason'), 'composite_exhausted': bool(self.state.get('composite_exhausted')), 'exhausted_templates': copy.deepcopy(self.state['exhausted_templates']), 'heartbeat_unix': time.time(), 'phase': self.state['phase'], 'status': status, 'startup_accepted_delta': startup_accepted_delta, 'startup_health': 'passed' if startup_accepted_delta > 0 else 'failed' if status in {'blocked', 'failed'} else 'pending', 'workers': [{'gpu_id': gpu_id, 'job_id': record['job']['job_id'], 'pid': record['process'].pid, 'ready': self._worker_is_ready(gpu_id, record)} for gpu_id, record in sorted(self.processes.items())]}
        _atomic_write_json(self.status_path, payload)
        self._last_heartbeat = time.monotonic()

    @staticmethod
    def _worker_is_ready(gpu_id: int, record: Mapping[str, Any]) -> bool:
        try:
            ready = json.loads(Path(str(record['ready_path'])).read_text(encoding='utf-8'))
            return ready.get('status') == 'simulator_initialized' and ready.get('job_id') == record['job']['job_id'] and (int(ready.get('gpu_device')) == int(gpu_id)) and (int(ready.get('pid')) == int(record['process'].pid))
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return False

    def _write_final_outputs(self, status: str) -> None:
        index_rows = []
        artifact_rows = []
        qa_type_counts: dict[str, int] = {}
        scope_counts: dict[str, int] = {}
        for item in self.state['accepted']:
            job = item['job']
            artifacts = item['result']['artifacts']
            final_task = json.loads(Path(artifacts['final_task']['path']).read_text(encoding='utf-8'))
            question_types = [str(question.get('question_type') or 'unknown') for question in (final_task.get('qa_overlay') or {}).get('questions') or []]
            for question_type in question_types:
                qa_type_counts[question_type] = qa_type_counts.get(question_type, 0) + 1
            scope_label = str(((final_task.get('family_spec') or {}).get('scope') or {}).get('label') or 'unknown')
            scope_counts[scope_label] = scope_counts.get(scope_label, 0) + 1
            index_rows.append({'artifacts': artifacts, 'job_id': job['job_id'], 'job_type': job['job_type'], 'layout_id': job['layout']['layout_id'], 'question_types': question_types, 'provenance': copy.deepcopy(item.get('provenance') or {'mode': 'legacy', 'source_manifest': copy.deepcopy(self.config['source_manifest'])}), 'template_ids': job['template_ids']})
            for name, artifact in sorted(artifacts.items()):
                artifact_rows.append({'job_id': job['job_id'], 'name': name, 'path': artifact['path'], 'sha256': artifact['sha256']})
        failure_counts: dict[str, int] = {}
        template_stats = {item['template_id']: {'accepted': 0, 'attempted': 0, 'real_oracle_attempted': 0} for item in self.templates}
        for item in self.state['attempts']:
            result = item['result']
            code = str(result.get('failure_code') or 'accepted')
            failure_counts[code] = failure_counts.get(code, 0) + 1
            for template_id in item['job']['template_ids']:
                stats = template_stats[template_id]
                stats['attempted'] += 1
                stats['real_oracle_attempted'] += int(bool(result.get('real_oracle_attempted')))
                stats['accepted'] += int(result.get('status') == 'accepted')
        _atomic_write_json(self.root / 'bank_index.json', {'records': index_rows})
        _atomic_write_json(self.root / 'artifact_manifest.json', {'artifacts': artifact_rows})
        self._write_qa_pending()
        _atomic_write_json(self.root / 'final_report.json', {'accepted_atomic': sum((1 for row in index_rows if row['job_type'] == 'atomic')), 'accepted_composite': sum((1 for row in index_rows if row['job_type'] == 'composite')), 'attempt_count': len(self.state['attempts']), 'failure_code_counts': dict(sorted(failure_counts.items())), 'composite_exhausted': bool(self.state.get('composite_exhausted')), 'exhausted_templates': copy.deepcopy(self.state['exhausted_templates']), 'qa_type_counts': dict(sorted(qa_type_counts.items())), 'schema_version': 'oracle-video-bank-final-report.v1', 'scope_label_counts': dict(sorted(scope_counts.items())), 'status': status, 'template_stats': template_stats})

    def _write_qa_pending(self) -> None:
        qa_pending = [{'attempt_dir': str(self.attempts_root / item['job']['job_id']), 'job': item['job'], 'provenance': copy.deepcopy(item.get('provenance') or {}), 'result': item['result']} for item in self.state['attempts'] if item['result'].get('status') == 'qa_attachment_blocked']
        _atomic_write_json(self.root / 'qa_pending.json', {'records': qa_pending})

    def _write_state(self, state: Mapping[str, Any]) -> None:
        _atomic_write_json(self.state_path, state)

def _job_salt(job: Mapping[str, Any]) -> str:
    return str(job['selection_salt'])

def _composite_size(job: Mapping[str, Any]) -> int:
    return int(job.get('composite_size') or len(job.get('template_ids') or []))

def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.{}.{}.tmp'.format(path.name, os.getpid()))
    with temporary.open('wb') as stream:
        stream.write(canonical_json_bytes(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def run_from_config(path: str | Path) -> int:
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    return OracleBankSupervisor(config).run()

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    return run_from_config(args.config)
if __name__ == '__main__':
    raise SystemExit(main())
__all__ = ['INFRA_FAILURE_CIRCUIT_BREAKER_COUNT', 'MASTER_SEED', 'ORACLE_BANK_ATTEMPT_SCHEMA_VERSION', 'ORACLE_BANK_CONFIG_SCHEMA_VERSION', 'ORACLE_BANK_STATE_SCHEMA_VERSION', 'OracleBankSupervisor', 'run_from_config', 'validate_oracle_bank_attempt', 'validate_oracle_bank_config']
