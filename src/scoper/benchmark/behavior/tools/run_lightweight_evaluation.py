"""Capture-free evaluation for BEHAVIOR VLM/API/Method agents."""
from __future__ import annotations
from scoper.benchmark.common.evaluation_policy import step_budget
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.common import EvaluationEngine, EvaluationRunConfig, EPISODE_RECORD_SCHEMA, SHARED_VOCABULARY_VERSION, load_episode_records, load_binding_sidecar, normalize_episode_result, validate_task_binding_spec
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.benchmark.behavior.episode_runner import EpisodeRunner
from scoper.benchmark.behavior.export.ids import strict_json_load
from scoper.benchmark.behavior.render.particle_segmentation_replay import particle_segmentation_policy
from scoper.benchmark.behavior.runtime_timing import RuntimeTimingRecorder, activate_runtime_timing, deactivate_runtime_timing
from scoper.benchmark.behavior.tools.run_vlm_benchmark import make_stage9_ego_renderer_factory

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class _DiagnosticEvidenceWriter:
    """Persist the exact public RGB frames delivered to the Method bridge."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.episode_dir: Path | None = None

    def on_evidence_reset(self, episode_id: str) -> None:
        safe_id = re.sub('[^A-Za-z0-9_.-]+', '_', episode_id).strip('._')
        self.episode_dir = self.root / (safe_id or hashlib.sha256(episode_id.encode()).hexdigest()[:16])
        self.episode_dir.mkdir(parents=True, exist_ok=False)

    def on_evidence_frame(self, frame: Any) -> None:
        if self.episode_dir is None:
            raise RuntimeError('diagnostic evidence writer was not reset')
        from scoper.benchmark.common import canonical_json_line
        from scoper.benchmark.behavior.json_utils import json_safe
        from scoper.integration.behavior.image_io import encode_public_rgb
        image = encode_public_rgb(frame.rgb)
        filename = f'{frame.frame_id}_{frame.kind}.png'
        (self.episode_dir / filename).write_bytes(image.encoded)
        row = {'schema_version': 'behavior-method-evidence-frame.v1', 'frame_id': frame.frame_id, 'kind': frame.kind, 'high_level_step': frame.high_level_step, 'action_id': frame.action_id, 'logical_time_s': frame.logical_time_s, 'relative_path': filename, **image.metadata()}
        with (self.episode_dir / 'manifest.jsonl').open('ab') as stream:
            stream.write(canonical_json_line(row))

def _task_list(path: Path) -> list[Mapping[str, Any]]:
    value = strict_json_load(path)
    if isinstance(value, list):
        tasks = value
    elif isinstance(value, Mapping):
        tasks = value.get('tasks') or value.get('entries')
    else:
        tasks = None
    if not isinstance(tasks, list) or any((not isinstance(task, Mapping) for task in tasks)):
        raise ValueError('task bank does not contain a task list')
    return [dict(task) for task in tasks]

def _particle_renderer_overrides(task: Mapping[str, Any]) -> dict[str, bool]:
    """Use the same task-aware particle policy as trajectory capture.

    Faucet tasks retain the native state graph but render only the zero-physics
    mesh stream. Other tasks strip the unused water particle system entirely.
    In particular, evaluation must never fall back to the renderer factory's
    native-particle defaults.
    """
    policy = particle_segmentation_policy(task)
    return {'particle_systems_enabled_override': bool(policy['source_particle_systems_enabled']), 'native_particle_source_visuals_enabled_override': bool(policy['source_native_particle_source_visuals_enabled']), 'particle_source_stream_visuals_enabled_override': bool(policy['source_particle_source_stream_visuals_enabled'])}

class BehaviorAgentFactory:

    def __init__(self, *, agent_kind: str, config_path: Path, run_config: EvaluationRunConfig, api_config: Path | None=None) -> None:
        self.agent_kind = agent_kind
        self.config_path = Path(config_path).resolve(strict=True)
        self.api_config = None if api_config is None else Path(api_config).resolve(strict=True)
        self.config_sha256 = _sha256_file(self.config_path)
        self.agent_id = self.config_path.stem
        self.harness_spec_sha256: str | None = None
        self.vlm_config: Any | None = None
        self._api_env_original: tuple[str, str | None] | None = None
        if self.agent_kind in {'vlm_local', 'vlm_api'}:
            from scoper.agents.behavior.config import VLMConfig
            import yaml
            document = yaml.safe_load(self.config_path.read_text(encoding='utf-8')) or {}
            if not isinstance(document, Mapping):
                raise ValueError('VLM config must be a mapping')
            document = dict(document)
            observation = dict(document.get('observation') or {})
            observation.update(max_resolution=[512, 512], max_history_turns=8, image_format='jpeg', jpeg_quality=90)
            observation['mode'] = run_config.observation_mode
            observation['num_frames'] = int(run_config.observation_budget['high_level_frame_count'])
            document['observation'] = observation
            document['eval_mode'] = run_config.evaluation_protocol
            config = VLMConfig.from_mapping(document)
            public_overlay: dict[str, Any] = {}
            if self.agent_kind == 'vlm_api':
                if self.api_config is None:
                    raise ValueError('vlm_api requires --api-config')
                overlay = json.loads(self.api_config.read_text(encoding='utf-8'))
                if not isinstance(overlay, Mapping):
                    raise ValueError('api config must be a JSON object')
                if overlay.get('api_key') or overlay.get('key'):
                    raise ValueError('API credentials must be provided only through an environment variable')
                if overlay.get('base_url'):
                    config.adapter.base_url = str(overlay['base_url'])
                if overlay.get('model'):
                    config.adapter.model = str(overlay['model'])
                adapter_extra = overlay.get('adapter_extra')
                if adapter_extra is not None:
                    if not isinstance(adapter_extra, Mapping):
                        raise ValueError('api adapter_extra must be a JSON object')
                    config.adapter.extra.update(dict(adapter_extra))
                public_overlay = {key: overlay[key] for key in ('base_url', 'model', 'adapter_extra') if overlay.get(key)}
                self.config_sha256 = hashlib.sha256(json.dumps({'agent_config_sha256': self.config_sha256, 'api': public_overlay}, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
            self.vlm_config = config
            self.agent_id = f'{config.adapter.type}:{config.adapter.model}'
        else:
            raise ValueError('Legacy Method integration is not shipped; use direct VLM')

    def build(self) -> Any:
        if self.agent_kind in {'vlm_local', 'vlm_api'}:
            from scoper.agents.behavior.vlm_brain import VLMBrain
            config = self.vlm_config
            brain = VLMBrain(config, io_log_path=None, image_log_dir=None, require_valid_visual_input=True, raise_adapter_errors=True, close_adapter_on_episode_end=False)
            return brain
        raise ValueError('Only direct VLM agents are supported by this runner')

    def close(self, agent: Any) -> None:
        close = getattr(agent, 'close', None)
        if callable(close):
            close()
        if self._api_env_original is not None:
            name, original = self._api_env_original
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original
            self._api_env_original = None

class BehaviorEvaluationAdapter:
    backend_id = 'scoper.benchmark.behavior'

    def __init__(self, *, task_bank: Path, agent_factory: BehaviorAgentFactory, max_high_level_steps: int | None=None, binding_sidecar: Path, run_config: EvaluationRunConfig, segmentation_capture_mode: str='on_demand', single_episode_fast_exit: bool=False, benevolent_task_binding: bool=False, diagnostic_output_jsonl: Path | None=None, diagnostic_frame_dir: Path | None=None) -> None:
        self.task_bank = Path(task_bank)
        self.task_bank_sha256 = _sha256_file(self.task_bank)
        sidecar_value = json.loads(Path(binding_sidecar).read_text(encoding='utf-8'))
        self.binding_specs = load_binding_sidecar(sidecar_value, bank_sha256=self.task_bank_sha256)
        self.binding_sidecar_sha256 = _sha256_file(binding_sidecar)
        self.run_config = run_config
        if run_config.task_binding_sidecar_sha256 != self.binding_sidecar_sha256:
            raise ValueError('evaluation run config Binding sidecar hash differs')
        self.agent_factory = agent_factory
        self.max_high_level_steps = max_high_level_steps
        self.segmentation_capture_mode = str(segmentation_capture_mode)
        self.single_episode_fast_exit = bool(single_episode_fast_exit)
        if benevolent_task_binding and run_config.evaluation_protocol == 'standard_vlm':
            raise ValueError('standard_vlm excludes truth-based benevolent task binding')
        self.benevolent_task_binding = bool(benevolent_task_binding)
        self.diagnostic_output_jsonl = None if diagnostic_output_jsonl is None else Path(diagnostic_output_jsonl)
        self.diagnostic_evidence_writer = None if diagnostic_frame_dir is None else _DiagnosticEvidenceWriter(diagnostic_frame_dir)

    def load_tasks(self) -> Sequence[Mapping[str, Any]]:
        tasks = []
        for task in _task_list(self.task_bank):
            task_id = str(task.get('id') or '')
            spec = self.binding_specs.get(task_id)
            if spec is None:
                raise ValueError(f'Binding sidecar misses task {task_id}')
            validate_task_binding_spec(task, spec)
            value = dict(task)
            value['task_binding_spec'] = spec.to_dict()
            value['task_binding_spec_sha256'] = spec.sha256
            tasks.append(value)
        return tasks

    def run_episode(self, task: Mapping[str, Any], agent: Any, *, seed: int) -> Any:
        session = getattr(self, 'scene_session', None)
        if session is not None:
            from scoper.benchmark.common.evaluation_policy import layout_key
            session.begin_episode(layout_key(task))
        if self.diagnostic_output_jsonl is not None:
            from scoper.benchmark.common import InMemoryMethodEventSink
            configure = getattr(agent, 'configure_method_event_sink', None)
            if callable(configure):

                def _raw_output(component: str) -> Mapping[str, Any]:
                    clients = getattr(agent, 'clients', None)
                    client = getattr(clients, 'memory_proposer', None) if component == 'memory_proposer' else getattr(clients, 'verifier', None)
                    return {'text': getattr(client, 'last_raw_text', None)}
                configure(InMemoryMethodEventSink(model_output_provider=_raw_output, capture_raw_output=True))
        scratch = tempfile.TemporaryDirectory(prefix='behavior-lightweight-eval-')
        renderer = None
        env = None
        result = None
        episode_error = None
        timing_recorder = RuntimeTimingRecorder.from_environment(str(task.get('id') or 'runtime-episode'))
        timing_token = activate_runtime_timing(timing_recorder) if timing_recorder is not None else None
        try:
            particle_overrides = _particle_renderer_overrides(task)
            rendered = make_stage9_ego_renderer_factory(segmentation_capture_mode=self.segmentation_capture_mode)(task_spec=task, logs_dir=Path(scratch.name), rng_seed=seed, **particle_overrides)
            renderer = rendered['renderer']
            env_kwargs = dict(rendered.get('env_kwargs') or {})
            stream_controller = None
            from scoper.agents.behavior.config import ObjectBindingConfig
            config = self.agent_factory.vlm_config
            binding_config = getattr(config, 'binding', None) or ObjectBindingConfig(allow_role_handles=False, allow_visible_aliases=True, allow_memory=True, allow_description_match=True, allow_raw_object_ids=False, allow_task_text_candidates=False)
            pass
            if getattr(self, 'benevolent_task_binding', False):
                binding_config = replace(binding_config, benevolent_task_progress=True)
            env_kwargs.update({'max_high_level_steps': step_budget(task, self.max_high_level_steps), 'eval_mode': self.run_config.evaluation_protocol, 'object_binding_config': binding_config})
            env = BehaviorClosedLoopToolEnv(renderer=renderer, stream_controller=stream_controller, **env_kwargs)
            try:
                result = EpisodeRunner(env, max_high_level_steps=step_budget(task, self.max_high_level_steps), raise_infrastructure_errors=True, runtime_evidence_observers=() if self.diagnostic_evidence_writer is None else (self.diagnostic_evidence_writer,)).run(task, brain=agent, rng_seed=seed, trace_path=None)
            except Exception as exc:
                episode_error = exc
                if self.diagnostic_output_jsonl is not None:
                    self._append_diagnostic(task, agent, None, seed, exception=exc)
                raise
            if self.diagnostic_output_jsonl is not None:
                self._append_diagnostic(task, agent, result, seed)
            return result
        finally:
            if session is not None:
                session.finish_episode(result, error=episode_error)
            try:
                if env is not None:
                    resolver = getattr(env, 'object_resolver', None)
                    close = getattr(resolver, 'close', None)
                    if callable(close):
                        close()
                close = getattr(renderer, 'close', None)
                if renderer is not None and bool(getattr(self, 'single_episode_fast_exit', False)):
                    preserve = getattr(renderer, 'preserve_simulator_for_process_exit', None)
                    if callable(preserve):
                        preserve()
                if callable(close):
                    close()
                scratch.cleanup()
            finally:
                if timing_token is not None:
                    deactivate_runtime_timing(timing_token)

    def _append_diagnostic(self, task: Mapping[str, Any], agent: Any, result: Any, seed: int, exception: Exception | None=None) -> None:
        from scoper.benchmark.common import canonical_json_line
        from scoper.benchmark.behavior.json_utils import json_safe
        events = getattr(getattr(agent, 'event_sink', None), 'events', ()) or ()
        row = {'schema_version': 'behavior-method-diagnostic.v1', 'task_id': str(task.get('id') or ''), 'seed': seed, 'instruction': str(task.get('instruction') or ''), 'method_events': json_safe([dict(event) for event in events if isinstance(event, Mapping)]), 'transitions': json_safe(list(getattr(result, 'transitions', ()) or ())), 'final_status': getattr(result, 'status', 'failed') if result is not None else 'failed', 'final_goal_metrics': json_safe(dict(getattr(result, 'final_goal', {}) or {})) if result is not None else {}, 'episode_metrics': json_safe(dict(getattr(result, 'metrics', {}) or {})) if result is not None else {}, 'hook_artifacts': json_safe(dict(getattr(result, 'hook_artifacts', {}) or {})) if result is not None else {}, 'failure': None if exception is None else {'type': type(exception).__name__, 'reason': str(exception)}}
        self.diagnostic_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.diagnostic_output_jsonl.open('ab') as stream:
            stream.write(canonical_json_line(row))

    def normalize_result(self, result: Any) -> Mapping[str, Any]:
        return normalize_episode_result(result)

def run_lightweight_evaluation(*, task_bank: Path, agent_kind: str, agent_config: Path, output_jsonl: Path, run_id: str, api_config: Path | None=None, task_ids: Sequence[str]=(), limit: int=1, seed: int=0, max_high_level_steps: int | None=None, binding_sidecar: Path, evaluation_protocol: str, observation_mode: str, segmentation_capture_mode: str='on_demand', single_episode_fast_exit: bool=False, reuse_layout_session: bool=False, benevolent_task_binding: bool=False, oracle_reference: Path | None=None, diagnostic_output_jsonl: Path | None=None, diagnostic_frame_dir: Path | None=None) -> tuple[Mapping[str, Any], ...]:
    from scoper.benchmark.common.action_coverage import NecessaryActionCoverage, load_references
    metric_plugins = () if oracle_reference is None else (NecessaryActionCoverage(load_references(oracle_reference, _sha256_file(task_bank))),)
    run_config = _evaluation_run_config(agent_kind=agent_kind, evaluation_protocol=evaluation_protocol, observation_mode=observation_mode, binding_sidecar=binding_sidecar)
    run_config.validate(agent_kind)
    factory = BehaviorAgentFactory(agent_kind=agent_kind, config_path=agent_config, run_config=run_config, api_config=api_config)
    adapter = BehaviorEvaluationAdapter(task_bank=task_bank, agent_factory=factory, max_high_level_steps=max_high_level_steps, binding_sidecar=binding_sidecar, run_config=run_config, segmentation_capture_mode=segmentation_capture_mode, single_episode_fast_exit=single_episode_fast_exit, benevolent_task_binding=benevolent_task_binding, diagnostic_output_jsonl=diagnostic_output_jsonl, diagnostic_frame_dir=diagnostic_frame_dir)
    tasks = list(adapter.load_tasks())
    if task_ids:
        by_id = {str(task['id']): task for task in tasks}
        if len(set(task_ids)) != len(task_ids):
            raise ValueError('duplicate requested task IDs')
        missing = set(task_ids) - by_id.keys()
        if missing:
            raise ValueError('unknown requested task IDs: ' + str(sorted(missing)))
        tasks = [by_id[task_id] for task_id in task_ids]
    elif reuse_layout_session:
        from scoper.benchmark.common.evaluation_policy import layout_key
        tasks.sort(key=lambda task: (layout_key(task), str(task['id'])))
    tasks = tasks[:limit]
    if reuse_layout_session:
        if single_episode_fast_exit:
            raise ValueError('layout reuse and single_episode_fast_exit are mutually exclusive')
    if not tasks:
        raise ValueError('lightweight evaluation selected no tasks')
    _validate_single_episode_fast_exit_selection(tasks, enabled=single_episode_fast_exit)
    engine = EvaluationEngine(adapter=adapter, metrics=metric_plugins, agent_factory=factory, run_config=run_config, max_infrastructure_retries=0 if single_episode_fast_exit or reuse_layout_session else _max_infrastructure_retries(agent_kind), task_binding_spec_sha256_by_task={task_id: spec.sha256 for task_id, spec in adapter.binding_specs.items()})
    engine_output = Path(output_jsonl)
    temporary_output: Path | None = None
    if single_episode_fast_exit:
        temporary_output = engine_output.with_name('.{}.{}.tmp'.format(engine_output.name, os.getpid()))
        engine_output = temporary_output
    try:
        from contextlib import nullcontext
        from scoper.benchmark.behavior.render.evaluation_session import EvaluationSceneSession
        with EvaluationSceneSession() if reuse_layout_session else nullcontext() as session:
            adapter.scene_session = session
            records = engine.run(tasks=tasks, run_id=run_id, task_bank_sha256=_sha256_file(task_bank), seed=seed, output_jsonl=engine_output)
        if temporary_output is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_output, output_jsonl)
    finally:
        if temporary_output is not None and temporary_output.exists():
            temporary_output.unlink()
    return tuple((record.to_dict() for record in records))

def _max_infrastructure_retries(agent_kind: str) -> int:
    return 2

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Run capture-free BEHAVIOR evaluation')
    parser.add_argument('--task-bank', type=Path, required=True)
    parser.add_argument('--agent-kind', choices=('vlm_local', 'vlm_api'), required=True)
    parser.add_argument('--agent-config', type=Path, required=True)
    parser.add_argument('--binding-sidecar', type=Path, required=True)
    parser.add_argument('--evaluation-protocol', choices=('standard_vlm', 'agent'), required=True)
    parser.add_argument('--observation-mode', choices=('single_frame', 'multi_frame', 'streaming'), required=True)
    parser.add_argument('--api-config', type=Path)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--task-id', action='append', default=[])
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--oracle-reference', type=Path)
    parser.add_argument('--reuse-layout-session', action='store_true')
    parser.add_argument('--max-high-level-steps', type=int, default=None)
    parser.add_argument('--benevolent-task-binding', action='store_true')
    parser.add_argument('--diagnostic-output-jsonl', type=Path)
    parser.add_argument('--diagnostic-frame-dir', type=Path)
    parser.add_argument('--single-episode-fast-exit', action='store_true', help='After exactly one episode is durably written, detach the renderer and hard-exit without clearing or shutting down OmniGibson.')
    parser.add_argument('--segmentation-capture-mode', choices=('required', 'on_demand', 'off'), default='on_demand', help='Instance-segmentation policy. Capture-free inference defaults to on_demand; off remains experimental, and formal Oracle/training trajectories use required.')
    args = parser.parse_args(argv)
    records = run_lightweight_evaluation(task_bank=args.task_bank, agent_kind=args.agent_kind, agent_config=args.agent_config, binding_sidecar=args.binding_sidecar, evaluation_protocol=args.evaluation_protocol, observation_mode=args.observation_mode, api_config=args.api_config, output_jsonl=args.output_jsonl, run_id=args.run_id, task_ids=args.task_id, limit=args.limit, seed=args.seed, max_high_level_steps=args.max_high_level_steps, oracle_reference=args.oracle_reference, reuse_layout_session=args.reuse_layout_session, segmentation_capture_mode=args.segmentation_capture_mode, single_episode_fast_exit=args.single_episode_fast_exit, benevolent_task_binding=args.benevolent_task_binding, diagnostic_output_jsonl=args.diagnostic_output_jsonl, diagnostic_frame_dir=args.diagnostic_frame_dir)
    print(json.dumps({'episodes': len(records), 'output': str(args.output_jsonl)}, sort_keys=True), flush=True)
    if args.single_episode_fast_exit:
        _finish_single_episode_fast_exit(args.output_jsonl)
    import omnigibson as og
    og.shutdown()
    return 0

def _fsync_file_and_parent(path: Path) -> None:
    resolved = Path(path)
    with resolved.open('rb') as stream:
        os.fsync(stream.fileno())
    directory_fd = os.open(str(resolved.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

def _validate_single_episode_fast_exit_selection(tasks: Sequence[Mapping[str, Any]], *, enabled: bool) -> None:
    if enabled and len(tasks) != 1:
        raise ValueError('--single-episode-fast-exit requires exactly one selected task')

def _finish_single_episode_fast_exit(output_jsonl: Path, *, exit_fn: Any=None) -> None:
    records = load_episode_records(output_jsonl)
    if len(records) != 1 or records[0].get('schema_version') != EPISODE_RECORD_SCHEMA:
        raise RuntimeError('single-episode fast exit requires one valid v2 episode record')
    _fsync_file_and_parent(output_jsonl)
    timing_path = str(os.environ.get('EB_BEHAVIOR_TIMING_JSONL', '') or '').strip()
    if timing_path and Path(timing_path).is_file():
        _fsync_file_and_parent(Path(timing_path))
    sys.stdout.flush()
    sys.stderr.flush()
    hard_exit = os._exit if exit_fn is None else exit_fn
    failure = records[0].get('failure')
    infrastructure_failure = isinstance(failure, Mapping) and failure.get('code') == 'evaluation_episode_exception'
    hard_exit(1 if infrastructure_failure else 0)

def _evaluation_run_config(*, agent_kind: str, evaluation_protocol: str, observation_mode: str, binding_sidecar: Path) -> EvaluationRunConfig:
    frame_count = 4 if observation_mode == 'multi_frame' else 1
    budget: dict[str, Any] = {'high_level_frame_count': frame_count}
    if observation_mode == 'streaming':
        budget['stream_window'] = 'public_runtime_window'
    return EvaluationRunConfig(evaluation_protocol=evaluation_protocol, observation_mode=observation_mode, observation_budget=budget, task_binding_sidecar_sha256=_sha256_file(binding_sidecar))
if __name__ == '__main__':
    from scoper.benchmark.behavior.render.evaluation_session import FreshProcessRequired
    try:
        raise SystemExit(main())
    except FreshProcessRequired as exc:
        print(json.dumps({'status': 'fresh_process_required', 'reason': str(exc)}), flush=True)
        os._exit(75)
__all__ = ['BehaviorAgentFactory', 'BehaviorEvaluationAdapter', 'main', 'run_lightweight_evaluation']
