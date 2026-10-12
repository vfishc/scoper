from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple
from scoper.agents.procthor import VLMBrain, VLMConfig, assert_runtime_requirements, missing_runtime_requirements
from scoper.benchmark.procthor.closed_loop_tool_env import ProcTHORClosedLoopToolEnv
from scoper.benchmark.procthor.episode_runner import EpisodeRunner
from scoper.benchmark.procthor.hooks import ExecutionTimelineHook, HookBus
from scoper.benchmark.procthor.manipulathor_config import manipulathor_env_kwargs
from scoper.benchmark.procthor.metrics import aggregate_metrics
from scoper.benchmark.procthor.online_task_loader import load_task_bank, load_task_set, resolve_task_set
from scoper.benchmark.procthor.paths import resolve_repo_path
from scoper.benchmark.procthor.scheduler import CompositeScheduler, DisturbanceScheduler, FailureScheduler, NoOpScheduler, Scheduler

class VLMSmokeRunnerError(ValueError):
    pass

def run_vlm_smoke(*, task_bank_path: str, vlm_config_path: str, output_dir: str, api_config_path: Optional[str]=None, task_set_path: Optional[str]=None, task_ids: Optional[Sequence[str]]=None, task_categories: Optional[Sequence[str]]=None, first_per_category: bool=False, max_tasks: Optional[int]=1, width: int=320, height: int=240, cloud_rendering: bool=True, gpu_device: Optional[int]=0, max_high_level_steps: Optional[int]=None, max_nav_steps_per_tool: int=1000, max_total_low_level_steps: int=100000, scheduler_config: Optional[Mapping[str, Any]]=None, write_trace: bool=True, save_vlm_input_frames: bool=False, skip_runtime_check: bool=False, manipulathor_arm: bool=False, manipulathor_visual_attach: bool=True, agent_radius: Optional[float]=None, env_factory: Callable[..., ProcTHORClosedLoopToolEnv]=ProcTHORClosedLoopToolEnv) -> Tuple[List[dict], dict]:
    vlm_config = VLMConfig.from_yaml(resolve_repo_path(vlm_config_path))
    apply_api_config(vlm_config, api_config_path)
    if not skip_runtime_check:
        assert_runtime_requirements(vlm_config)
    task_bank = load_task_bank(str(resolve_repo_path(task_bank_path)))
    tasks = select_tasks(task_bank, task_set_path=task_set_path, task_ids=task_ids, task_categories=task_categories, first_per_category=first_per_category, max_tasks=max_tasks)
    if not tasks:
        raise VLMSmokeRunnerError('No tasks selected for VLM smoke run.')
    output = Path(output_dir)
    results_dir = output / 'results'
    traces_dir = output / 'traces'
    vlm_frames_dir = output / 'vlm_input_frames'
    results_dir.mkdir(parents=True, exist_ok=True)
    if write_trace:
        traces_dir.mkdir(parents=True, exist_ok=True)
    if save_vlm_input_frames:
        vlm_frames_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = {'width': width, 'height': height, 'cloud_rendering': cloud_rendering, 'gpu_device': gpu_device, 'max_nav_steps_per_tool': max_nav_steps_per_tool, 'max_total_low_level_steps': max_total_low_level_steps, 'eval_mode': vlm_config.eval_mode, 'object_binding_config': vlm_config.binding}
    env_kwargs.update(manipulathor_env_kwargs(manipulathor_arm=manipulathor_arm, manipulathor_visual_attach=manipulathor_visual_attach, agent_radius=agent_radius))
    env = env_factory(**env_kwargs)
    runner = EpisodeRunner(env, max_high_level_steps=max_high_level_steps, require_passed_status=False)
    scheduler_factory = scheduler_factory_from_config(scheduler_config)
    records: List[dict] = []
    brain: Optional[VLMBrain] = None
    try:
        brain = reusable_vlm_brain(vlm_config)
        for index, task in enumerate(tasks, start=1):
            episode_name = 'episode_{:03d}_{}'.format(index, safe_label(task['id']))
            frame_capture_dir = vlm_frames_dir / episode_name if save_vlm_input_frames else None
            if hasattr(brain, 'set_frame_capture'):
                brain.set_frame_capture(frame_capture_dir)
            trace_path = traces_dir / '{}.jsonl'.format(episode_name) if write_trace else None
            timeline_hook = ExecutionTimelineHook(output_dir=output, episode_name=episode_name)
            result = runner.run(task, brain=brain, trace_path=trace_path, scheduler=scheduler_factory(), hooks=HookBus([timeline_hook]), runner_config_extras=vlm_runner_config(vlm_config, vlm_config_path))
            record = result_to_record(index=index, task=task, result=result, vlm_config=vlm_config, vlm_config_path=vlm_config_path, vlm_input_frames_dir=frame_capture_dir)
            records.append(record)
            (results_dir / 'episode_{}_final_res.json'.format(index)).write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    finally:
        if brain is not None:
            brain.close()
        env.close()
    summary = summarize_vlm_records(records, vlm_config, vlm_config_path)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    (output / 'vlm_config_resolved.json').write_text(json.dumps(vlm_runner_config(vlm_config, vlm_config_path), indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    return (records, summary)

def select_tasks(task_bank: Mapping[str, Mapping[str, Any]], *, task_set_path: Optional[str]=None, task_ids: Optional[Sequence[str]]=None, task_categories: Optional[Sequence[str]]=None, first_per_category: bool=False, max_tasks: Optional[int]=1) -> List[Mapping[str, Any]]:
    if task_set_path:
        task_set = load_task_set(str(resolve_repo_path(task_set_path)), task_bank=task_bank)
        tasks = list(resolve_task_set(task_bank, task_set))
    elif task_ids:
        missing = [task_id for task_id in task_ids if task_id not in task_bank]
        if missing:
            raise VLMSmokeRunnerError('Task ids not found in task bank: {}'.format(', '.join(missing)))
        tasks = [task_bank[task_id] for task_id in task_ids]
    elif first_per_category:
        seen = set()
        tasks = []
        for task in task_bank.values():
            category = task.get('category')
            if category in seen:
                continue
            seen.add(category)
            tasks.append(task)
    else:
        tasks = list(task_bank.values())
    if task_categories:
        wanted = set(task_categories)
        tasks = [task for task in tasks if task.get('category') in wanted]
    if max_tasks is not None:
        tasks = tasks[:int(max_tasks)]
    return tasks

def scheduler_factory_from_config(config: Optional[Mapping[str, Any] | str]) -> Callable[[], Optional[Scheduler]]:
    if not config:
        return lambda: None
    if isinstance(config, str):
        data = json.loads(resolve_repo_path(config).read_text(encoding='utf-8'))
    else:
        data = dict(config)
    seed = int(data.get('seed', 0))
    disturbances = list(data.get('disturbances') or [])
    failures = list(data.get('failures') or [])

    def factory() -> Scheduler:
        children = []
        if disturbances:
            children.append(DisturbanceScheduler(configs=disturbances, seed=seed))
        if failures:
            children.append(FailureScheduler(configs=failures, seed=seed))
        if not children:
            return NoOpScheduler()
        if len(children) == 1:
            return children[0]
        return CompositeScheduler(children)
    return factory

def result_to_record(*, index: int, task: Mapping[str, Any], result, vlm_config: VLMConfig, vlm_config_path: str, vlm_input_frames_dir: Optional[Path]=None) -> dict:
    timeline_manifest = (result.hook_artifacts.get('execution_timeline') if isinstance(result.hook_artifacts, Mapping) else {}) or {}
    return {'episode_index': index, 'task_id': result.task_id, 'category': result.category, 'house_id': result.house_id, 'instruction': task['instruction'], 'brain': 'vlm', 'vlm': vlm_runner_config(vlm_config, vlm_config_path), 'status': result.status, 'metrics': result.metrics, 'final_goal': result.final_goal, 'failure_type': result.failure_type, 'failure_reason': result.failure_reason, 'failure_code': result.failure_code, 'failure_details': json_safe(result.failure_details), 'transitions': json_safe(result.transitions), 'scheduler_events': json_safe(result.scheduler_events), 'hook_events': json_safe(result.hook_events), 'hook_artifacts': json_safe(result.hook_artifacts), 'init_mutations': json_safe(result.init_mutations), 'scene_mutations': json_safe(result.scene_mutations), 'timeline': timeline_manifest.get('timeline'), 'timeline_json': timeline_manifest.get('timeline_json'), 'trace_path': result.trace_path, 'vlm_input_frames': str(vlm_input_frames_dir) if vlm_input_frames_dir else None, 'episode_elapsed_seconds': result.elapsed_seconds}

def summarize_vlm_records(records: Sequence[Mapping[str, Any]], vlm_config: VLMConfig, vlm_config_path: str) -> dict:
    status_counts = Counter((record.get('status', 'unknown') for record in records))
    return {**aggregate_metrics(records), 'brain': 'vlm', 'vlm': vlm_runner_config(vlm_config, vlm_config_path), 'status_counts': dict(sorted(status_counts.items()))}

def vlm_runner_config(vlm_config: VLMConfig, config_path: str) -> dict:
    return {'config_path': str(resolve_repo_path(config_path)), 'adapter_type': vlm_config.adapter.type, 'model': vlm_config.adapter.model, 'base_url': vlm_config.adapter.base_url, 'max_tokens': vlm_config.adapter.max_tokens, 'temperature': vlm_config.adapter.temperature, 'prompt_template': vlm_config.prompt_template, 'eval_mode': vlm_config.eval_mode, 'observation': {'mode': vlm_config.observation.mode, 'num_frames': vlm_config.observation.num_frames, 'max_history_turns': vlm_config.observation.max_history_turns, 'image_format': vlm_config.observation.image_format, 'max_resolution': list(vlm_config.observation.max_resolution), 'jpeg_quality': vlm_config.observation.jpeg_quality}, 'retry': {'max_retries': vlm_config.retry.max_retries, 'backoff_base': vlm_config.retry.backoff_base, 'backoff_cap': vlm_config.retry.backoff_cap, 'timeout_seconds': vlm_config.retry.timeout_seconds}, 'metadata': {'mode': vlm_config.metadata.mode, 'expose_visible_catalog': vlm_config.metadata.expose_visible_catalog, 'expose_raw_object_ids': vlm_config.metadata.expose_raw_object_ids, 'expose_states': vlm_config.metadata.expose_states, 'expose_bbox': vlm_config.metadata.expose_bbox, 'expose_segmentation': vlm_config.metadata.expose_segmentation, 'log_privileged_metadata': vlm_config.metadata.log_privileged_metadata}, 'exposure': dict(vlm_config.exposure.__dict__), 'binding': dict(vlm_config.binding.__dict__)}

def apply_api_config(vlm_config: VLMConfig, api_config_path: Optional[str]) -> None:
    if not api_config_path:
        return
    if vlm_config.adapter.type != 'openai':
        raise VLMSmokeRunnerError("--api-config can only be used with adapter.type='openai'.")
    path = resolve_repo_path(api_config_path)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise VLMSmokeRunnerError("Invalid API config JSON '{}': {}.".format(path, exc)) from exc
    if not isinstance(data, Mapping):
        raise VLMSmokeRunnerError("API config '{}' must contain a JSON object.".format(path))
    api_key = data.get('api_key') or data.get('key')
    if api_key:
        env_name = vlm_config.adapter.api_key_env or 'EB_PROCTHOR_API_KEY'
        vlm_config.adapter.api_key_env = env_name
        os.environ[env_name] = str(api_key)
    base_url = data.get('base_url')
    if base_url:
        vlm_config.adapter.base_url = str(base_url)

def reusable_vlm_brain(vlm_config: VLMConfig) -> VLMBrain:
    brain = VLMBrain(vlm_config)
    if hasattr(brain, 'close_adapter_on_episode_end'):
        brain.close_adapter_on_episode_end = False
    return brain

def safe_label(value: str) -> str:
    label = re.sub('[^A-Za-z0-9_.-]+', '_', str(value).strip())
    return label.strip('_') or 'episode'

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if key != 'head_rgb'}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist') and (not isinstance(value, (str, bytes))):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return value

def _trajectory_wrapper_argv(argv: Sequence[str]) -> list[str]:
    if '--first-per-category' in argv or '--scheduler-config' in argv:
        raise VLMSmokeRunnerError('trajectory mode forbids --first-per-category and --scheduler-config')
    if '--limit' in argv and '--max-tasks' in argv:
        raise VLMSmokeRunnerError('trajectory mode cannot combine --limit and --max-tasks')
    translated = []
    iterator = iter(argv)
    for value in iterator:
        if value == '--output-dir':
            translated.extend(('--output-root', next(iterator)))
        elif value == '--max-tasks':
            translated.extend(('--limit', next(iterator)))
        elif value in ('--allow-failures', '--no-write-trace', '--save-vlm-input-frames', '--skip-runtime-check'):
            continue
        else:
            translated.append(value)
    if '--policy' not in translated:
        translated.extend(('--policy', 'vlm'))
    return translated
__all__ = ['VLMSmokeRunnerError', 'apply_api_config', 'json_safe', 'reusable_vlm_brain', 'run_vlm_smoke', 'safe_label', 'scheduler_factory_from_config', 'select_tasks', 'summarize_vlm_records', 'vlm_runner_config']
