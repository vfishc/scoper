from __future__ import annotations
import copy
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from scoper.benchmark.common.canonical import canonical_json_line
LEGACY_EPISODE_RECORD_SCHEMA = 'embodied-method-evaluation-episode.v1'
EPISODE_RECORD_SCHEMA = 'embodied-method-evaluation-episode.v2'
AGENT_KINDS = ('method', 'vlm_api', 'vlm_local')
EVALUATION_PROTOCOLS = ('agent', 'standard_vlm')
OBSERVATION_MODES = ('multi_frame', 'single_frame', 'streaming')
_PRIVATE_EVALUATION_KEYS = frozenset({'binding_role', 'candidate_id', 'object_id', 'private_id', 'raw_id', 'raw_object_id', 'role', 'role_handle', 'source_role', 'target_role'})
_AI2THOR_OBJECT_ID = re.compile('(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*(?:\\|[-+A-Za-z0-9_.]+)+')
_BEHAVIOR_OBJECT_ID = re.compile('(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*\\.[a-z]\\.[0-9]+_[0-9]+\\b')
_SIMULATOR_PRIM_PATH = re.compile('(?<![A-Za-z0-9_])/World(?:/[A-Za-z0-9_.:-]+)+')

class UnsupportedEvaluationCapability(ValueError):

    def __init__(self, *, agent_kind: str, evaluation_protocol: str, observation_mode: str) -> None:
        self.details = {'code': 'unsupported_evaluation_capability', 'agent_kind': str(agent_kind), 'evaluation_protocol': str(evaluation_protocol), 'observation_mode': str(observation_mode)}
        super().__init__(f'unsupported evaluation capability: {agent_kind}/{evaluation_protocol}/{observation_mode}')

@dataclass(frozen=True)
class EvaluationRunConfig:
    evaluation_protocol: str
    observation_mode: str
    observation_budget: Mapping[str, Any]
    task_binding_sidecar_sha256: str | None = None

    def validate(self, agent_kind: str) -> None:
        if agent_kind not in AGENT_KINDS:
            raise UnsupportedEvaluationCapability(agent_kind=agent_kind, evaluation_protocol=self.evaluation_protocol, observation_mode=self.observation_mode)
        if self.evaluation_protocol not in EVALUATION_PROTOCOLS:
            raise UnsupportedEvaluationCapability(agent_kind=agent_kind, evaluation_protocol=self.evaluation_protocol, observation_mode=self.observation_mode)
        supported = self.observation_mode == 'streaming' if agent_kind == 'method' else self.observation_mode in {'single_frame', 'multi_frame'}
        if not supported:
            raise UnsupportedEvaluationCapability(agent_kind=agent_kind, evaluation_protocol=self.evaluation_protocol, observation_mode=self.observation_mode)
        frame_count = int(self.observation_budget.get('high_level_frame_count', 0))
        if self.observation_mode == 'single_frame' and frame_count != 1:
            raise ValueError('single_frame requires high_level_frame_count=1')
        if self.observation_mode == 'multi_frame' and frame_count != 4:
            raise ValueError('multi_frame requires high_level_frame_count=4')
        if self.observation_mode == 'streaming' and frame_count != 1:
            raise ValueError('Method streaming policy requires high_level_frame_count=1')
        if self.task_binding_sidecar_sha256 is not None:
            _require_sha256(self.task_binding_sidecar_sha256, 'task_binding_sidecar_sha256')

    @classmethod
    def default_for_agent_kind(cls, agent_kind: str) -> 'EvaluationRunConfig':
        if agent_kind == 'method':
            return cls(evaluation_protocol='standard_vlm', observation_mode='streaming', observation_budget={'high_level_frame_count': 1, 'stream_window': 'public_runtime_window'})
        return cls(evaluation_protocol='standard_vlm', observation_mode='single_frame', observation_budget={'high_level_frame_count': 1})

class MetricPlugin(Protocol):
    name: str

    def reset(self, context: Mapping[str, Any]) -> None:
        ...

    def on_transition(self, transition: Mapping[str, Any]) -> None:
        ...

    def finalize(self, result: Any) -> Mapping[str, Any]:
        ...

class EvaluationBackendAdapter(Protocol):
    backend_id: str

    def load_tasks(self) -> Sequence[Mapping[str, Any]]:
        ...

    def run_episode(self, task: Mapping[str, Any], agent: Any, *, seed: int) -> Any:
        ...

    def normalize_result(self, result: Any) -> Mapping[str, Any]:
        ...

class AgentFactory(Protocol):
    agent_kind: str
    agent_id: str
    config_sha256: str
    harness_spec_sha256: str | None

    def build(self) -> Any:
        ...

    def close(self, agent: Any) -> None:
        ...

@dataclass(frozen=True)
class EpisodeRecord:
    schema_version: str
    run_id: str
    backend: str
    task_bank_sha256: str
    task_id: str
    seed: int
    agent_kind: str
    agent_id: str
    agent_config_sha256: str
    harness_spec_sha256_or_null: str | None
    evaluation_protocol: str
    observation_mode: str
    observation_budget: Mapping[str, Any]
    task_binding_spec_sha256: str
    status: str
    termination: str
    task_success: bool
    task_progress: float
    high_level_steps: int
    environment_tool_calls: int
    model_call_count: int
    backend_metrics: Mapping[str, Any] = field(default_factory=dict)
    binding_metrics: Mapping[str, Any] = field(default_factory=dict)
    navigation_metrics: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    token_usage: Mapping[str, Any] | None = None
    cost_or_null: float | None = None
    failure: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))

class EvaluationEngine:

    def __init__(self, *, adapter: EvaluationBackendAdapter, agent_factory: AgentFactory, metrics: Iterable[MetricPlugin]=(), run_config: EvaluationRunConfig | None=None, max_infrastructure_retries: int=0, task_binding_spec_sha256_by_task: Mapping[str, str] | None=None) -> None:
        if agent_factory.agent_kind not in AGENT_KINDS:
            raise ValueError(f'unsupported agent kind: {agent_factory.agent_kind}')
        self.adapter = adapter
        self.agent_factory = agent_factory
        self.metrics = tuple(metrics)
        self.run_config = run_config or EvaluationRunConfig.default_for_agent_kind(agent_factory.agent_kind)
        self.run_config.validate(agent_factory.agent_kind)
        if isinstance(max_infrastructure_retries, bool) or not isinstance(max_infrastructure_retries, int) or max_infrastructure_retries < 0:
            raise ValueError('max_infrastructure_retries must be a non-negative integer')
        self.max_infrastructure_retries = max_infrastructure_retries
        self.task_binding_spec_sha256_by_task = dict(task_binding_spec_sha256_by_task or {})

    def run(self, *, tasks: Sequence[Mapping[str, Any]], run_id: str, task_bank_sha256: str, seed: int, output_jsonl: Path | None=None) -> tuple[EpisodeRecord, ...]:
        records: list[EpisodeRecord] = []
        task_binding_hashes: dict[str, str] = {}
        for task in tasks:
            task_id = str(task.get('id') or task.get('task_id') or '')
            binding_spec_sha256 = self.task_binding_spec_sha256_by_task.get(task_id)
            if binding_spec_sha256 is None:
                binding_spec_sha256 = str(task.get('task_binding_spec_sha256') or '')
            _require_sha256(binding_spec_sha256, 'task_binding_spec_sha256')
            task_binding_hashes[task_id] = binding_spec_sha256
        agent = None
        try:
            agent = self.agent_factory.build()
            for index, task in enumerate(tasks):
                from scoper.benchmark.common.transport_retry import episode_started
                episode_started(str(task.get('id') or task.get('task_id') or ''))
                from scoper.benchmark.common.evaluation_policy import episode_seed, step_budget, PROTOCOL_VERSION
                task_seed = episode_seed(str(task.get('id') or task.get('task_id') or ''), seed)
                context = {'task': task, 'seed': task_seed, 'run_id': run_id}
                for metric in self.metrics:
                    metric.reset(context)
                started = time.monotonic()
                try:
                    retry_count = 0
                    while True:
                        try:
                            result = self.adapter.run_episode(task, agent, seed=task_seed)
                            normalized = dict(self.adapter.normalize_result(result))
                            break
                        except Exception as exc:
                            if isinstance(getattr(exc, 'evaluation_failure_record', None), Mapping):
                                raise
                            if retry_count >= self.max_infrastructure_retries:
                                raise
                            retry_count += 1
                    backend_metrics = dict(normalized.get('backend_metrics') or {})
                    backend_metrics['evaluation_policy'] = PROTOCOL_VERSION
                    backend_metrics['max_high_level_steps'] = step_budget(task, getattr(self.adapter, 'step_budget_override', getattr(self.adapter, 'max_high_level_steps', None)))
                    backend_metrics['seed_policy'] = 'sha256-base-seed-task-id.v1'
                    backend_metrics['infrastructure_retry_count'] = retry_count
                    normalized['backend_metrics'] = backend_metrics
                except Exception as exc:
                    from scoper.benchmark.common.transport_retry import is_transport_failure, transport_error_info
                    result = exc
                    bounded_failure = getattr(exc, 'evaluation_failure_record', None)
                    if not isinstance(bounded_failure, Mapping):
                        bounded_failure = {}
                    backend_metrics = dict(bounded_failure.get('backend_metrics') or {})
                    transport_info = transport_error_info(exc)
                    if transport_info is not None:
                        backend_metrics['transport_failure'] = transport_info
                    qa_overlay = task.get('qa_overlay')
                    if isinstance(qa_overlay, Mapping):
                        questions = qa_overlay.get('questions')
                        if isinstance(questions, Sequence) and (not isinstance(questions, (str, bytes))):
                            backend_metrics.setdefault('num_qa_total', len(questions))
                    backend_metrics['infrastructure_retry_count'] = retry_count
                    normalized = {'task_id': str(task.get('id') or task.get('task_id') or ''), 'status': 'infrastructure_failed' if is_transport_failure(exc) else 'failed', 'termination': 'exception', 'task_success': False, 'task_progress': float(bounded_failure.get('task_progress', 0.0)), 'high_level_steps': int(bounded_failure.get('high_level_steps', 0)), 'environment_tool_calls': int(bounded_failure.get('environment_tool_calls', 0)), 'model_call_count': int(bounded_failure.get('model_call_count', 0)), 'backend_metrics': backend_metrics, 'binding_metrics': dict(bounded_failure.get('binding_metrics') or {}), 'navigation_metrics': dict(bounded_failure.get('navigation_metrics') or {}), 'transitions': bounded_failure.get('transitions', ()), 'token_usage': bounded_failure.get('token_usage'), 'cost_or_null': None, 'failure': dict(bounded_failure.get('failure') or {}) or {'type': type(exc).__name__, 'code': 'evaluation_episode_exception', 'reason': str(exc)[:512]}}
                latency = time.monotonic() - started
                transitions = normalized.pop('transitions', ())
                normalized_task_id = str(normalized.pop('task_id', ''))
                task_id = str(task.get('id') or task.get('task_id') or normalized_task_id)
                binding_spec_sha256 = task_binding_hashes[task_id]
                for transition in transitions:
                    for metric in self.metrics:
                        metric.on_transition(transition)
                plugin_metrics = {metric.name: dict(metric.finalize(result)) for metric in self.metrics}
                record = EpisodeRecord(schema_version=EPISODE_RECORD_SCHEMA, run_id=run_id, backend=self.adapter.backend_id, task_bank_sha256=task_bank_sha256, task_id=task_id, seed=task_seed, agent_kind=self.agent_factory.agent_kind, agent_id=self.agent_factory.agent_id, agent_config_sha256=self.agent_factory.config_sha256, harness_spec_sha256_or_null=self.agent_factory.harness_spec_sha256, evaluation_protocol=self.run_config.evaluation_protocol, observation_mode=self.run_config.observation_mode, observation_budget=copy.deepcopy(dict(self.run_config.observation_budget)), task_binding_spec_sha256=binding_spec_sha256, status=str(normalized.pop('status')), termination=_sanitize_public_evaluation_text(str(normalized.pop('termination'))), task_success=bool(normalized.pop('task_success')), task_progress=float(normalized.pop('task_progress')), high_level_steps=int(normalized.pop('high_level_steps')), environment_tool_calls=int(normalized.pop('environment_tool_calls')), model_call_count=int(normalized.pop('model_call_count', 0)), backend_metrics=dict(sanitize_public_evaluation_payload(normalized.pop('backend_metrics', {}))), binding_metrics=dict(sanitize_public_evaluation_payload(normalized.pop('binding_metrics', {}))), navigation_metrics=dict(sanitize_public_evaluation_payload(normalized.pop('navigation_metrics', {}))), metrics=dict(sanitize_public_evaluation_payload(plugin_metrics)), latency_seconds=latency, token_usage=sanitize_public_evaluation_payload(normalized.pop('token_usage', None)), cost_or_null=normalized.pop('cost_or_null', None), failure=sanitize_public_evaluation_payload(normalized.pop('failure', None)))
                if normalized:
                    raise ValueError(f'unconsumed normalized result fields: {sorted(normalized)}')
                records.append(record)
                if output_jsonl is not None:
                    write_episode_records(output_jsonl, records)
        finally:
            self.agent_factory.close(agent)
        if output_jsonl is not None:
            write_episode_records(output_jsonl, records)
        return tuple(records)

def write_episode_records(path: Path, records: Iterable[EpisodeRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        for record in records:
            stream.write(canonical_json_line(sanitize_public_evaluation_payload(record.to_dict())))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def load_episode_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    result = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, Mapping) or value.get('schema_version') not in {EPISODE_RECORD_SCHEMA, LEGACY_EPISODE_RECORD_SCHEMA}:
            raise ValueError(f'invalid evaluation record at line {number}')
        result.append(value)
    return tuple(result)

def normalize_episode_result(result: Any) -> dict[str, Any]:
    transitions = list(getattr(result, 'transitions', ()) or ())
    final_goal = dict(getattr(result, 'final_goal', {}) or {})
    metrics = dict(getattr(result, 'metrics', {}) or {})
    artifacts = dict(getattr(result, 'hook_artifacts', {}) or {})
    evidence = dict(artifacts.get('runtime_evidence', {}) or {})
    for key in ('stream_frame_count', 'model_facing_window', 'passive_observe'):
        if key in evidence:
            metrics[key] = copy.deepcopy(evidence[key])
    success_value = final_goal.get('task_success', metrics.get('task_success', 0.0))
    try:
        task_success = float(success_value) == 1.0
    except (TypeError, ValueError):
        task_success = bool(success_value)
    progress_value = final_goal.get('task_progress', final_goal.get('predicate_success', metrics.get('task_progress', float(task_success))))
    try:
        task_progress = float(progress_value)
    except (TypeError, ValueError):
        task_progress = float(task_success)
    termination = ''
    if transitions and isinstance(transitions[-1], Mapping):
        termination = str(transitions[-1].get('done_reason') or '')
    if not termination:
        termination = str(getattr(result, 'failure_reason', None) or getattr(result, 'status', 'unknown'))
    failure = None
    if getattr(result, 'status', 'failed') != 'passed':
        failure = {'type': getattr(result, 'failure_type', None), 'code': getattr(result, 'failure_code', None), 'reason': getattr(result, 'failure_reason', None)}
    return {'task_id': str(getattr(result, 'task_id', '')), 'status': str(getattr(result, 'status', 'failed')), 'termination': termination, 'task_success': task_success, 'task_progress': task_progress, 'high_level_steps': len(transitions), 'environment_tool_calls': len(transitions), 'model_call_count': int(evidence.get('model_call_count', len(transitions))), 'backend_metrics': metrics, 'binding_metrics': dict(metrics.get('binding_metrics', {})), 'navigation_metrics': dict(metrics.get('navigation_metrics', {})), 'transitions': transitions, 'token_usage': metrics.get('token_usage'), 'cost_or_null': metrics.get('cost_or_null'), 'failure': failure}

def sanitize_public_evaluation_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        public = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized_key = key.casefold()
            if normalized_key in _PRIVATE_EVALUATION_KEYS or normalized_key.endswith('_object_id') or normalized_key.endswith('_candidate_id') or normalized_key.endswith('_role_handle'):
                continue
            public[key] = sanitize_public_evaluation_payload(item)
        return public
    if isinstance(value, (list, tuple)):
        return [sanitize_public_evaluation_payload(item) for item in value]
    if isinstance(value, str):
        return _sanitize_public_evaluation_text(value)
    return copy.deepcopy(value)

def _sanitize_public_evaluation_text(value: str) -> str:
    text = str(value)
    for pattern in (_AI2THOR_OBJECT_ID, _BEHAVIOR_OBJECT_ID, _SIMULATOR_PRIM_PATH):
        text = pattern.sub('[private_object]', text)
    return text
__all__ = ['AGENT_KINDS', 'EPISODE_RECORD_SCHEMA', 'EVALUATION_PROTOCOLS', 'LEGACY_EPISODE_RECORD_SCHEMA', 'OBSERVATION_MODES', 'AgentFactory', 'EpisodeRecord', 'EvaluationBackendAdapter', 'EvaluationEngine', 'EvaluationRunConfig', 'MetricPlugin', 'UnsupportedEvaluationCapability', 'load_episode_records', 'normalize_episode_result', 'sanitize_public_evaluation_payload', 'write_episode_records']

def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any((character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'{name} must be sha256')
