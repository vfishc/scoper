from __future__ import annotations
from scoper.benchmark.common.evaluation_policy import step_budget
import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from scoper.benchmark.common import EvaluationEngine, EvaluationRunConfig, SHARED_VOCABULARY_VERSION, UnsupportedEvaluationCapability, load_binding_sidecar, normalize_episode_result, validate_task_binding_spec
from scoper.agents.procthor import VLMBrain, VLMConfig
from scoper.benchmark.procthor.closed_loop_tool_env import ProcTHORClosedLoopToolEnv
from scoper.benchmark.procthor.episode_runner import EpisodeRunner
from scoper.benchmark.procthor.manipulathor_config import manipulathor_env_kwargs
from scoper.benchmark.procthor.online_task_loader import load_task_bank
from scoper.benchmark.procthor.paths import resolve_repo_path
from scoper.benchmark.procthor.nl_target_resolver import ObjectBindingConfig
from scoper.benchmark.procthor.vlm_smoke_runner import select_tasks

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class _DiagnosticEvidenceWriter:

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.episode_id = ''
        self.episode_dir: Path | None = None

    def on_evidence_reset(self, episode_id: str) -> None:
        safe_id = re.sub('[^A-Za-z0-9_.-]+', '_', episode_id).strip('._')
        if not safe_id:
            safe_id = hashlib.sha256(episode_id.encode('utf-8')).hexdigest()[:16]
        self.episode_id = episode_id
        self.episode_dir = self.root / safe_id
        self.episode_dir.mkdir(parents=True, exist_ok=False)

    def on_evidence_frame(self, frame: Any) -> None:
        if self.episode_dir is None:
            raise RuntimeError('diagnostic evidence writer was not reset')
        from scoper.benchmark.common import canonical_json_line
        from scoper.integration.procthor.image_io import encode_public_rgb
        image = encode_public_rgb(frame.rgb)
        filename = f'{frame.frame_id}_{frame.kind}.png'
        path = self.episode_dir / filename
        path.write_bytes(image.encoded)
        row = {'schema_version': 'procthor-method-evidence-frame.v1', 'episode_id': self.episode_id, 'frame_id': frame.frame_id, 'kind': frame.kind, 'high_level_step': frame.high_level_step, 'action_id': frame.action_id, 'logical_time_s': frame.logical_time_s, 'relative_path': filename, **image.metadata()}
        with (self.episode_dir / 'manifest.jsonl').open('ab') as stream:
            stream.write(canonical_json_line(row))

class ProcTHORAgentFactory:

    def __init__(self, *, agent_kind: str, config_path: Path, run_config: EvaluationRunConfig, api_config: Path | None=None, gpu_device: int=0) -> None:
        self.agent_kind = agent_kind
        self.config_path = Path(config_path).resolve(strict=True)
        self.api_config = None if api_config is None else Path(api_config).resolve(strict=True)
        self.config_sha256 = _sha256_file(self.config_path)
        if isinstance(gpu_device, bool) or not isinstance(gpu_device, int) or gpu_device < 0:
            raise ValueError('gpu_device must be a non-negative integer')
        self.gpu_device = gpu_device
        self.agent_id = self.config_path.stem
        self.harness_spec_sha256: str | None = None
        self.vlm_config: VLMConfig | None = None
        self._api_env_original: tuple[str, str | None] | None = None
        if self.agent_kind in {'vlm_local', 'vlm_api'}:
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
            config = self.vlm_config
            brain = VLMBrain(config, close_adapter_on_episode_end=False)
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

class ProcTHOREvaluationAdapter:
    backend_id = 'eb_procthor'

    def __init__(self, *, task_bank: Path, agent_factory: ProcTHORAgentFactory, width: int=320, height: int=240, gpu_device: int | None=0, max_high_level_steps: int | None=None, manipulathor_visual_attach: bool=True, binding_sidecar: Path, run_config: EvaluationRunConfig, diagnostic_output_jsonl: Path | None=None, diagnostic_frame_dir: Path | None=None, benevolent_task_binding: bool=False) -> None:
        if not manipulathor_visual_attach:
            error = UnsupportedEvaluationCapability(agent_kind=agent_factory.agent_kind, evaluation_protocol=run_config.evaluation_protocol, observation_mode=run_config.observation_mode)
            error.details.update({'backend': self.backend_id, 'reason': 'procthor_requires_visual_attach', 'execution_profile': 'semantic_compatibility'})
            raise error
        self.task_bank = Path(task_bank)
        self.task_bank_sha256 = _sha256_file(self.task_bank)
        sidecar_value = json.loads(Path(binding_sidecar).read_text(encoding='utf-8'))
        self.binding_specs = load_binding_sidecar(sidecar_value, bank_sha256=self.task_bank_sha256)
        self.binding_sidecar_sha256 = _sha256_file(binding_sidecar)
        self.run_config = run_config
        self.diagnostic_output_jsonl = None if diagnostic_output_jsonl is None else Path(diagnostic_output_jsonl)
        diagnostic_evidence_writer = None if diagnostic_frame_dir is None else _DiagnosticEvidenceWriter(Path(diagnostic_frame_dir))
        if run_config.task_binding_sidecar_sha256 != self.binding_sidecar_sha256:
            raise ValueError('evaluation run config Binding sidecar hash differs')
        config = agent_factory.vlm_config
        binding_config = getattr(config, 'binding', None) or ObjectBindingConfig(allow_role_handles=False, allow_visible_aliases=True, allow_memory=True, allow_description_match=True, allow_raw_object_ids=False, allow_task_text_candidates=False)
        pass
        if benevolent_task_binding:
            if run_config.evaluation_protocol == 'standard_vlm':
                raise ValueError('standard_vlm excludes truth-based benevolent task binding')
            binding_config = replace(binding_config, benevolent_task_progress=True)
        env_kwargs = {'width': width, 'height': height, 'gpu_device': gpu_device, 'cloud_rendering': True, 'max_nav_steps_per_tool': 1000, 'max_high_level_steps': max_high_level_steps, 'max_total_low_level_steps': 100000, 'eval_mode': run_config.evaluation_protocol, 'object_binding_config': binding_config, 'allow_force_navigation_fallback': True, 'allow_force_open_container_fallback': True, 'allow_logical_put_down_fallback': True}
        env_kwargs.update(manipulathor_env_kwargs(manipulathor_arm=manipulathor_visual_attach, manipulathor_visual_attach=manipulathor_visual_attach))
        self.step_budget_override = max_high_level_steps
        env_kwargs['max_high_level_steps'] = max_high_level_steps or 24
        self.env = ProcTHORClosedLoopToolEnv(**env_kwargs)
        self.runner = EpisodeRunner(self.env, max_high_level_steps=max_high_level_steps, require_passed_status=False, raise_infrastructure_errors=True, runtime_evidence_observers=() if diagnostic_evidence_writer is None else (diagnostic_evidence_writer,))

    def load_tasks(self) -> Sequence[Mapping[str, Any]]:
        bank = load_task_bank(str(self.task_bank))
        tasks = []
        for task_id, task in bank.items():
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
        if self.diagnostic_output_jsonl is not None:
            from scoper.benchmark.common import InMemoryMethodEventSink
            configure = getattr(agent, 'configure_method_event_sink', None)
            if callable(configure):

                def _raw_output(component: str) -> Mapping[str, Any]:
                    client = getattr(agent, 'memory_proposer_client', None) if component == 'memory_proposer' else getattr(agent, 'verifier_client', None)
                    return {'text': getattr(client, 'last_raw_text', None)}
                configure(InMemoryMethodEventSink(model_output_provider=_raw_output, capture_raw_output=True))
        budget = step_budget(task, self.step_budget_override)
        self.runner.max_high_level_steps = budget
        self.env.max_high_level_steps = budget
        try:
            result = self.runner.run(task, brain=agent, trace_path=None)
        except Exception as exc:
            if self.diagnostic_output_jsonl is not None:
                self._append_diagnostic(task, agent, None, seed=seed, exception=exc)
            raise
        if self.diagnostic_output_jsonl is not None:
            self._append_diagnostic(task, agent, result, seed=seed)
        return result

    def _append_diagnostic(self, task: Mapping[str, Any], agent: Any, result: Any | None, *, seed: int, exception: Exception | None=None) -> None:
        events = getattr(getattr(agent, 'event_sink', None), 'events', ()) or ()
        safe_events = []
        for raw in events:
            if not isinstance(raw, Mapping):
                continue
            event = {'event_type': raw.get('event_type'), 'parent_record_id': raw.get('parent_record_id'), 'correlation': dict(raw.get('correlation') or {}), 'payload': dict(raw.get('payload') or {})}
            safe_events.append(event)
        ordered_tools = []
        for index, raw in enumerate(getattr(result, 'transitions', ()) or (), 1):
            if not isinstance(raw, Mapping):
                continue
            execution = raw.get('raw_execution')
            sync = execution.get('sync') if isinstance(execution, Mapping) else None
            private_geometry = sync.get('private_visual_attach_geometry') if isinstance(sync, Mapping) else None
            visual_attach_geometry = None
            if isinstance(private_geometry, Mapping):
                visual_attach_geometry = {key: private_geometry.get(key) for key in ('hand_sphere_radius', 'bbox_height', 'final_offset', 'anchor_contact_gap_m')}
            private_confirmation = raw.get('confirmation')
            confirmation_camera = None
            if isinstance(private_confirmation, Mapping):
                confirmation_camera = {key: private_confirmation.get(key) for key in ('schema_version', 'view_kind', 'used_bound_coordinate', 'requested_yaw', 'requested_horizon', 'actual_yaw', 'actual_horizon', 'video_fps_or_null')}
            ordered_tools.append({'step': index, 'tool_name': raw.get('tool_name'), 'arguments': dict(raw.get('arguments') or {}), 'environment_success': raw.get('success') is True, 'binding': dict(raw.get('binding') or {}), 'execution_status': execution.get('status') if isinstance(execution, Mapping) else None, 'execution_error': execution.get('error_message') if isinstance(execution, Mapping) else None, 'fallback_entries': list(execution.get('fallback_entries') or () if isinstance(execution, Mapping) else ()), 'num_low_level_steps': int(raw.get('num_low_level_steps') or 0), 'visual_attach_geometry_or_null': visual_attach_geometry, 'confirmation_camera_or_null': confirmation_camera})
        row = {'schema_version': 'procthor-method-diagnostic.v1', 'task_id': str(task.get('id') or task.get('task_id') or ''), 'seed': seed, 'harness_spec_sha256': getattr(agent, 'harness_provenance', {}).get('harness_spec_sha256'), 'instruction': str(task.get('instruction') or ''), 'ordered_tool_calls': ordered_tools, 'visual_method_events': safe_events, 'final_status': getattr(result, 'status', 'failed'), 'final_goal_metrics': dict(getattr(result, 'final_goal', {}) or {}), 'episode_metrics': dict(getattr(result, 'metrics', {}) or {}), 'failure': {'type': type(exception).__name__ if exception is not None else getattr(result, 'failure_type', None), 'code': getattr(result, 'failure_code', None), 'reason': str(exception) if exception is not None else getattr(result, 'failure_reason', None)}}
        from scoper.benchmark.common import canonical_json_line
        self.diagnostic_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.diagnostic_output_jsonl.open('ab') as stream:
            stream.write(canonical_json_line(row))

    def normalize_result(self, result: Any) -> Mapping[str, Any]:
        return normalize_episode_result(result)

    def close(self) -> None:
        self.env.close()

def run_lightweight_evaluation(*, task_bank: Path, agent_kind: str, agent_config: Path, output_jsonl: Path, run_id: str, api_config: Path | None=None, task_ids: Sequence[str]=(), limit: int=1, seed: int=0, gpu_device: int | None=0, max_high_level_steps: int | None=None, manipulathor_visual_attach: bool=True, binding_sidecar: Path, evaluation_protocol: str, observation_mode: str, oracle_reference: Path | None=None, diagnostic_output_jsonl: Path | None=None, diagnostic_frame_dir: Path | None=None, benevolent_task_binding: bool=False) -> tuple[Mapping[str, Any], ...]:
    from scoper.benchmark.common.action_coverage import NecessaryActionCoverage, load_references
    metric_plugins = () if oracle_reference is None else (NecessaryActionCoverage(load_references(oracle_reference, _sha256_file(task_bank))),)
    run_config = _evaluation_run_config(agent_kind=agent_kind, evaluation_protocol=evaluation_protocol, observation_mode=observation_mode, binding_sidecar=binding_sidecar)
    run_config.validate(agent_kind)
    factory = ProcTHORAgentFactory(agent_kind=agent_kind, config_path=agent_config, run_config=run_config, api_config=api_config, gpu_device=int(gpu_device or 0))
    adapter = ProcTHOREvaluationAdapter(task_bank=task_bank, agent_factory=factory, gpu_device=gpu_device, max_high_level_steps=max_high_level_steps, manipulathor_visual_attach=manipulathor_visual_attach, binding_sidecar=binding_sidecar, run_config=run_config, diagnostic_output_jsonl=diagnostic_output_jsonl, diagnostic_frame_dir=diagnostic_frame_dir, benevolent_task_binding=benevolent_task_binding)
    try:
        loaded_tasks = adapter.load_tasks()
        tasks_by_id = {str(task.get('id') or task.get('task_id') or ''): task for task in loaded_tasks}
        if '' in tasks_by_id or len(tasks_by_id) != len(loaded_tasks):
            raise ValueError('lightweight evaluation tasks require unique non-empty ids')
        tasks = select_tasks(tasks_by_id, task_ids=list(task_ids), max_tasks=limit)
        records = EvaluationEngine(adapter=adapter, metrics=metric_plugins, agent_factory=factory, run_config=run_config, max_infrastructure_retries=2, task_binding_spec_sha256_by_task={task_id: spec.sha256 for task_id, spec in adapter.binding_specs.items()}).run(tasks=tasks, run_id=run_id, task_bank_sha256=_sha256_file(task_bank), seed=seed, output_jsonl=output_jsonl)
        return tuple((record.to_dict() for record in records))
    finally:
        adapter.close()

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Run capture-free ProcTHOR evaluation')
    parser.add_argument('--task-bank', type=Path, required=True)
    parser.add_argument('--agent-kind', choices=('vlm_local', 'vlm_api'), required=True)
    parser.add_argument('--agent-config', type=Path, required=True)
    parser.add_argument('--binding-sidecar', type=Path, required=True)
    parser.add_argument('--evaluation-protocol', choices=('standard_vlm', 'agent'), required=True)
    parser.add_argument('--observation-mode', choices=('single_frame', 'multi_frame', 'streaming'), required=True)
    parser.add_argument('--api-config', type=Path)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--diagnostic-output-jsonl', type=Path)
    parser.add_argument('--diagnostic-frame-dir', type=Path, help='Save exact Method current/pre/post/stream RGB for visual diagnosis.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--task-id', action='append', default=[])
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--oracle-reference', type=Path)
    parser.add_argument('--max-high-level-steps', type=int, default=None)
    parser.add_argument('--benevolent-task-binding', action='store_true')
    parser.add_argument('--manipulathor-visual-attach', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    records = run_lightweight_evaluation(task_bank=args.task_bank, agent_kind=args.agent_kind, agent_config=args.agent_config, binding_sidecar=args.binding_sidecar, evaluation_protocol=args.evaluation_protocol, observation_mode=args.observation_mode, api_config=args.api_config, output_jsonl=args.output_jsonl, run_id=args.run_id, task_ids=args.task_id, limit=args.limit, seed=args.seed, gpu_device=args.gpu_device, max_high_level_steps=args.max_high_level_steps, oracle_reference=args.oracle_reference, manipulathor_visual_attach=args.manipulathor_visual_attach, diagnostic_output_jsonl=args.diagnostic_output_jsonl, diagnostic_frame_dir=args.diagnostic_frame_dir, benevolent_task_binding=args.benevolent_task_binding)
    print(json.dumps({'episodes': len(records), 'output': str(args.output_jsonl)}, sort_keys=True))
    return 0

def _evaluation_run_config(*, agent_kind: str, evaluation_protocol: str, observation_mode: str, binding_sidecar: Path) -> EvaluationRunConfig:
    frame_count = 4 if observation_mode == 'multi_frame' else 1
    budget: dict[str, Any] = {'high_level_frame_count': frame_count}
    budget['scan_frame_count'] = 8
    if observation_mode == 'streaming':
        budget['stream_window'] = 'public_runtime_window'
    return EvaluationRunConfig(evaluation_protocol=evaluation_protocol, observation_mode=observation_mode, observation_budget=budget, task_binding_sidecar_sha256=_sha256_file(binding_sidecar))
if __name__ == '__main__':
    raise SystemExit(main())
__all__ = ['ProcTHORAgentFactory', 'ProcTHOREvaluationAdapter', 'main', 'run_lightweight_evaluation']
