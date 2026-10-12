"""Run a VLM benchmark over a task bank with optional fake streaming."""
from __future__ import annotations
import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.agents.behavior.config import VLMConfig
from scoper.agents.behavior.streaming import StreamController
from scoper.agents.behavior.vlm_brain import VLMBrain
from scoper.benchmark.behavior.episode_runner import EpisodeRunner
from scoper.benchmark.behavior.json_utils import json_safe
from scoper.benchmark.behavior.paths import resolve_repo_path
from scoper.benchmark.behavior.task_generation.profile_task_generator import generate_stage4_tasks
from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates, match_scene_to_templates
from scoper.benchmark.behavior.scene.profiler import profile_scene, profile_scenes
from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec
from scoper.benchmark.behavior.task_generation.observation_conditioned_generator import generate_observation_conditioned_tasks
DEFAULT_OUT_ROOT = 'outputs/eb_behavior_vlm_benchmark'

@contextmanager
def _renderer_factory_lifetime(renderer: Any):
    """Close a renderer if its factory fails before ownership reaches the caller."""
    try:
        yield renderer
    except BaseException:
        try:
            renderer.close()
        except Exception:
            pass
        raise

@dataclass
class BenchmarkResult:
    task_id: str
    status: str
    output_dir: str
    trace_path: str
    stream_log_path: str | None
    vlm_io_path: str | None
    binding_log_path: str | None
    scene_layout_mode: str
    metrics: dict[str, Any]
    final_goal: dict[str, Any]

@dataclass
class TaskBatch:
    tasks: list[dict[str, Any]]
    artifacts: dict[str, Any]
    generation: dict[str, Any]

class FakeBenchmarkAdapter:
    """Deterministic adapter used for smoke/debug without a real model."""

    def __init__(self) -> None:
        self.high_level_calls = 0
        self.stream_calls = 0
        self.last_call_metadata = None

    def call(self, messages: Sequence[Mapping[str, Any]]) -> str:
        self.last_call_metadata = type('_Meta', (), {'elapsed_seconds': 0.0, 'prompt_tokens': None, 'completion_tokens': None, 'finish_reason': 'stop'})()
        last_message = messages[-1] if messages else {}
        text = '\n'.join((str(block.get('text', '')) for block in last_message.get('content', []) if block.get('type') == 'text'))
        if 'current tool execution stream' in text.lower():
            self.stream_calls += 1
            return '{"stream_decision": "continue", "reasoning": "fake stream continue"}'
        self.high_level_calls += 1
        if self.high_level_calls == 1:
            return '{"tool_name": "scan", "arguments": {"num_views": 2}, "reasoning": "fake scan"}'
        return '{"tool_name": "stop", "arguments": {}, "reasoning": "fake stop"}'

    def supports_images(self) -> bool:
        return True

    def close(self) -> None:
        return None

class BlankRenderer:
    """Small renderer used by fake VLM smoke runs."""

    def __init__(self, shape: tuple[int, int, int]=(16, 16, 3)) -> None:
        self.shape = shape
        self.core = None

    def sync_initial_from_core(self, core: Any) -> None:
        self.core = core

    def render_at_pose(self, pose: Any) -> Any:
        return np.zeros(self.shape, dtype=np.uint8)

    def visible_object_names_at_pose(self, pose: Any, object_names: Sequence[str] | None=None, min_pixels: int=8) -> list[str]:
        names = list(object_names or getattr(getattr(self, 'core', None), 'objects', {}).keys())
        return names[:12]

def _should_fast_exit_after_manifest(args: argparse.Namespace) -> bool:
    if bool(getattr(args, 'no_fast_exit_after_manifest', False)):
        return False
    return bool(getattr(args, 'fast_exit_after_manifest', False)) or str(getattr(args, 'renderer', '')) == 'ego'

def run_vlm_benchmark(*, task_bank_path: Path | None=None, task_manifest_path: Path | None=None, vlm_config_path: Path | None=None, out_root: Path, run_id: str, limit: int | None=None, shard_index: int=0, shard_count: int=1, resume: bool=False, fake_vlm: bool=True, metadata_mode: str='config', max_high_level_steps: int=12, max_total_low_level_steps: int=1000000, scene_models: Sequence[str] | None=None, generate_stage4: bool=False, generate_observation_conditioned: bool=False, scene_variant_files: Sequence[str | Path] | None=None, generate_scene_variants: bool=False, num_scene_variants: int=1, scene_variant_seed: int=0, scene_variant_output_root: Path | None=None, scene_variant_config: Mapping[str, Any] | None=None, assets_root: Path | None=None, template_ids: Sequence[str] | None=None, max_bindings_per_template: int=2, max_tasks_per_template_scene: int=1, generate_only: bool=False, renderer_factory: Callable[..., Any] | None=None, brain_adapter_factory: Callable[[], Any] | None=None, validate_task_readiness: bool=True, generation_validation: str='readiness', scene_layout_mode: str='auto', load_doors: bool=False, export_vlm_images: bool=False, export_training_records: bool=False) -> dict[str, Any]:
    output_dir = Path(out_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir = output_dir / 'logs'
    run_logs_dir.mkdir(exist_ok=True)
    episodes_dir = output_dir / 'episodes'
    episodes_dir.mkdir(exist_ok=True)
    generation_dir = output_dir / 'generation'
    generation_dir.mkdir(exist_ok=True)
    task_batch = _load_task_batch(task_bank_path=task_bank_path, task_manifest_path=task_manifest_path, scene_models=scene_models or (), generate_stage4=generate_stage4, generate_observation_conditioned=generate_observation_conditioned, scene_variant_files=scene_variant_files or (), generate_scene_variants=generate_scene_variants, num_scene_variants=num_scene_variants, scene_variant_seed=scene_variant_seed, scene_variant_output_root=scene_variant_output_root, scene_variant_config=scene_variant_config or {}, assets_root=assets_root, template_ids=template_ids or (), max_bindings_per_template=max_bindings_per_template, max_tasks_per_template_scene=max_tasks_per_template_scene, generation_dir=generation_dir)
    tasks = task_batch.tasks
    tasks = _apply_shard(tasks, shard_index=shard_index, shard_count=shard_count)
    if limit is not None:
        tasks = tasks[:max(int(limit), 0)]
    if not tasks:
        raise ValueError('No tasks were loaded or generated for the VLM benchmark.')
    write_json(output_dir / 'task_bank.json', {'tasks': tasks})
    task_readiness = _validate_task_readiness(tasks, scene_layout_mode=scene_layout_mode, load_doors=load_doors) if validate_task_readiness else {'validated_total': 0, 'passed_count': 0, 'failed_count': 0, 'records': []}
    write_json(run_logs_dir / 'task_readiness.json', task_readiness)
    if int(task_readiness.get('failed_count') or 0) > 0:
        failures = [record for record in task_readiness.get('records', []) if not record.get('vlm_ready')]
        raise ValueError('Task readiness validation failed before VLM execution: {}'.format(json.dumps(json_safe(failures[:5]), ensure_ascii=False, sort_keys=True)))
    generation_admission = _validate_generation_admission(tasks, mode=generation_validation, scene_layout_mode=scene_layout_mode, load_doors=load_doors)
    write_json(run_logs_dir / 'generation_admission.json', generation_admission)
    if int(generation_admission.get('failed_count') or 0) > 0:
        failures = [record for record in generation_admission.get('records', []) if not record.get('admitted')]
        raise ValueError('Generation admission validation failed before VLM execution: {}'.format(json.dumps(json_safe(failures[:5]), ensure_ascii=False, sort_keys=True)))
    if generate_only:
        manifest = {'run_id': run_id, 'output_dir': str(output_dir), 'task_count': len(tasks), 'executed_count': 0, 'skipped_count': 0, 'skipped_tasks': [], 'episode_count': 0, 'passed_count': 0, 'failed_count': 0, 'metadata_mode': None, 'scene_layout_mode': scene_layout_mode, 'load_doors': bool(load_doors), 'effective_scene_layout_modes': _scene_layout_mode_counts(tasks, requested_mode=scene_layout_mode), 'generate_only': True, 'generation': task_batch.generation, 'task_readiness': task_readiness, 'generation_admission': generation_admission, 'artifacts': {**task_batch.artifacts, 'task_bank_json': str(output_dir / 'task_bank.json'), 'task_readiness_json': str(run_logs_dir / 'task_readiness.json'), 'generation_admission_json': str(run_logs_dir / 'generation_admission.json'), 'training_records_jsonl': None}, 'episodes': []}
        write_json(output_dir / 'manifest.json', manifest)
        return manifest
    config = _load_vlm_config(vlm_config_path, fake_vlm=fake_vlm)
    resolved_metadata_mode = _resolve_metadata_mode(metadata_mode, config)
    if not fake_vlm and renderer_factory is None:
        raise ValueError("A non-fake VLM benchmark requires a renderer_factory so observation['head_rgb'] is available. Use fake_vlm=True for dry runs, or provide a real RGB renderer factory from a GPU/runtime wrapper.")
    training_records_path = run_logs_dir / 'training_records.jsonl'
    if export_training_records and training_records_path.exists():
        training_records_path.unlink()
    results: list[BenchmarkResult] = []
    skipped: list[str] = []
    for index, task in enumerate(tasks):
        task_id = str(task.get('id') or f'task_{index:05d}')
        episode_scene_layout_mode = _effective_scene_layout_mode(scene_layout_mode, task)
        episode_dir = episodes_dir / f'{index:05d}_{_safe_slug(task_id)}'
        logs_dir = episode_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        summary_path = logs_dir / 'summary.json'
        if resume and summary_path.exists():
            skipped.append(task_id)
            continue
        if fake_vlm:
            adapter = FakeBenchmarkAdapter()
        elif brain_adapter_factory is not None:
            adapter = brain_adapter_factory()
        else:
            adapter = None
        brain = VLMBrain(config, adapter=adapter, io_log_path=logs_dir / 'vlm_io.jsonl', image_log_dir=logs_dir / 'vlm_images' if export_vlm_images else None, require_valid_visual_input=not fake_vlm)
        controller = StreamController(brain=brain, config=config.observation.stream, log_path=logs_dir / 'stream_frames.jsonl')
        renderer_bundle = _build_episode_renderer(fake_vlm=fake_vlm, renderer_factory=renderer_factory, task=task, episode_dir=episode_dir, logs_dir=logs_dir, rng_seed=index, task_index=index, scene_layout_mode=episode_scene_layout_mode, load_doors=load_doors)
        renderer = renderer_bundle['renderer']
        env_kwargs = dict(renderer_bundle.get('env_kwargs') or {})
        env_kwargs.setdefault('scene_layout_mode', episode_scene_layout_mode)
        env_kwargs.setdefault('load_doors', bool(load_doors))
        env = BehaviorClosedLoopToolEnv(renderer=renderer, stream_controller=controller, metadata_mode=resolved_metadata_mode, metadata_config=config.metadata, eval_mode=config.eval_mode, object_binding_config=config.binding, binding_log_path=logs_dir / 'binding_log.jsonl', max_high_level_steps=max_high_level_steps, max_total_low_level_steps=max_total_low_level_steps, **env_kwargs)
        runner = EpisodeRunner(env, max_high_level_steps=max_high_level_steps)
        trace_path = logs_dir / 'trace.jsonl'
        try:
            result = runner.run(task, brain=brain, trace_path=trace_path, rng_seed=index)
        finally:
            close = getattr(renderer, 'close', None)
            if callable(close):
                close()
            resolver_close = getattr(getattr(env, 'object_resolver', None), 'close', None)
            if callable(resolver_close):
                resolver_close()
        metrics = dict(result.metrics or {})
        if hasattr(env, 'binding_metrics'):
            metrics.update(env.binding_metrics())
        episode_manifest = {'task_id': task_id, 'task_index': index, 'status': result.status, 'output_dir': str(episode_dir), 'trace_path': str(trace_path), 'stream_log_path': str(logs_dir / 'stream_frames.jsonl'), 'vlm_io_path': str(logs_dir / 'vlm_io.jsonl'), 'binding_log_path': str(logs_dir / 'binding_log.jsonl'), 'scene_layout_mode': episode_scene_layout_mode, 'load_doors': bool(load_doors), 'metrics': metrics, 'final_goal': dict(result.final_goal), 'transitions': len(result.transitions), 'failure_type': result.failure_type, 'failure_reason': result.failure_reason}
        write_json(summary_path, episode_manifest)
        if export_training_records:
            _append_training_records(training_records_path, task=task, episode_manifest=episode_manifest, vlm_io_path=logs_dir / 'vlm_io.jsonl')
        results.append(BenchmarkResult(task_id=task_id, status=result.status, output_dir=str(episode_dir), trace_path=str(trace_path), stream_log_path=str(logs_dir / 'stream_frames.jsonl'), vlm_io_path=str(logs_dir / 'vlm_io.jsonl'), binding_log_path=str(logs_dir / 'binding_log.jsonl'), scene_layout_mode=episode_scene_layout_mode, metrics=metrics, final_goal=_final_goal_summary(result.final_goal)))
    manifest = {'run_id': run_id, 'output_dir': str(output_dir), 'task_count': len(tasks), 'executed_count': len(results), 'skipped_count': len(skipped), 'skipped_tasks': skipped, 'episode_count': len(results), 'passed_count': sum((1 for item in results if item.status == 'passed')), 'failed_count': sum((1 for item in results if item.status != 'passed')), 'metadata_mode': resolved_metadata_mode, 'scene_layout_mode': scene_layout_mode, 'load_doors': bool(load_doors), 'effective_scene_layout_modes': _scene_layout_mode_counts(tasks, requested_mode=scene_layout_mode), 'generate_only': False, 'generation': task_batch.generation, 'task_readiness': task_readiness, 'generation_admission': generation_admission, 'artifacts': {**task_batch.artifacts, 'task_bank_json': str(output_dir / 'task_bank.json'), 'task_readiness_json': str(run_logs_dir / 'task_readiness.json'), 'generation_admission_json': str(run_logs_dir / 'generation_admission.json'), 'training_records_jsonl': str(training_records_path) if export_training_records else None}, 'episodes': [json_safe(item.__dict__) for item in results]}
    write_json(output_dir / 'manifest.json', manifest)
    return manifest

def _resolve_metadata_mode(metadata_mode: str, config: VLMConfig) -> str:
    mode = str(metadata_mode or 'config').strip().lower()
    if mode == 'config':
        mode = str(getattr(config.metadata, 'mode', 'standard') or 'standard').strip().lower()
    if mode not in {'standard', 'privileged'}:
        raise ValueError("metadata_mode must be 'config', 'standard', or 'privileged'.")
    return mode

def _effective_scene_layout_mode(requested_mode: str, task: Mapping[str, Any]) -> str:
    mode = str(requested_mode or 'auto').strip().lower()
    if mode == 'auto':
        source = task.get('source') if isinstance(task, Mapping) else {}
        return 'source_json_full' if isinstance(source, Mapping) and source.get('source_json') else 'task_roles'
    if mode in {'task_roles', 'source_json_full'}:
        return mode
    raise ValueError("scene_layout_mode must be 'auto', 'task_roles', or 'source_json_full'.")

def _scene_layout_mode_counts(tasks: Sequence[Mapping[str, Any]], *, requested_mode: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        mode = _effective_scene_layout_mode(requested_mode, task)
        counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))

def _build_episode_renderer(*, fake_vlm: bool, renderer_factory: Callable[..., Any] | None, task: Mapping[str, Any], episode_dir: Path, logs_dir: Path, rng_seed: int, task_index: int, scene_layout_mode: str, load_doors: bool=False) -> dict[str, Any]:
    if renderer_factory is None:
        if fake_vlm:
            return {'renderer': BlankRenderer(), 'env_kwargs': {}}
        raise ValueError('renderer_factory is required for non-fake VLM runs.')
    kwargs = {'task_spec': task, 'task': task, 'episode_dir': episode_dir, 'output_dir': episode_dir, 'logs_dir': logs_dir, 'rng_seed': rng_seed, 'task_index': task_index, 'scene_layout_mode': scene_layout_mode, 'load_doors': bool(load_doors)}
    try:
        bundle = renderer_factory(**kwargs)
    except TypeError:
        bundle = renderer_factory()
    renderer_bundle = _coerce_renderer_bundle(bundle)
    if not fake_vlm and isinstance(renderer_bundle.get('renderer'), BlankRenderer):
        raise ValueError('Non-fake VLM evaluation requires real ego RGB; BlankRenderer is only for fake smoke runs.')
    return renderer_bundle

def _coerce_renderer_bundle(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, Mapping):
        renderer = bundle.get('renderer')
        if renderer is None:
            raise ValueError("renderer_factory mapping result must contain a 'renderer' entry.")
        env_kwargs = dict(bundle.get('env_kwargs') or {})
        return {'renderer': renderer, 'env_kwargs': env_kwargs}
    return {'renderer': bundle, 'env_kwargs': {}}

def make_stage9_ego_renderer_factory(*, scene_layout_mode: str='auto', renders_per_frame: int=2, use_rendered_visibility: bool=False, include_navigation_context: bool=True, structural_preset: str='stage9', load_doors: bool=False, allow_navigation_context_fallback: bool=True, require_line_of_sight_navigation: bool=False, formal_trajectory_capture: bool=False, segmentation_capture_mode: str | None=None, particle_systems_enabled: bool=True, native_particle_source_visuals_enabled: bool=False, particle_source_stream_visuals_enabled: bool=True) -> Callable[..., dict[str, Any]]:
    """Create a task-aware EgoRenderer factory for real VLM benchmark runs."""
    resolved_segmentation_capture_mode = _resolve_segmentation_capture_mode(formal_trajectory_capture=formal_trajectory_capture, segmentation_capture_mode=segmentation_capture_mode)
    if resolved_segmentation_capture_mode == 'off' and use_rendered_visibility:
        raise ValueError('segmentation_capture_mode=off conflicts with use_rendered_visibility=True')
    effective_use_rendered_visibility = bool(use_rendered_visibility) or resolved_segmentation_capture_mode == 'on_demand'
    visibility_settings = _trajectory_visibility_settings(formal_trajectory_capture=formal_trajectory_capture, use_rendered_visibility=effective_use_rendered_visibility, segmentation_capture_mode=resolved_segmentation_capture_mode)

    def _factory(*, task_spec: Mapping[str, Any], logs_dir: Path, rng_seed: int, particle_systems_enabled_override: bool | None=None, native_particle_source_visuals_enabled_override: bool | None=None, particle_source_stream_visuals_enabled_override: bool | None=None, include_navigation_context_override: bool | None=None, **_: Any) -> dict[str, Any]:
        from scoper.benchmark.behavior.bddl_adapter import BDDLAdapter
        from scoper.benchmark.behavior.execution_core.occupancy_utils import indoor_mask_as_traversability, mask_traversability_with_reference, scene_wall_slice_occupancy_as_traversability
        from scoper.benchmark.behavior.render.ego_renderer import EgoRenderer
        from scoper.benchmark.behavior.render.ego_arm_visual import ego_arm_visual_config_from_env
        from scoper.benchmark.behavior.render.gripper_visual import gripper_visual_config_from_env
        from scoper.benchmark.behavior.scene.placement_render import build as build_render_scene
        from scoper.benchmark.behavior.tools.stage9_render_video_report import STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS, STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN, _attach_disabled_doorway_openings_to_traversability, make_stage9_render_env_config, render_env_config_for_log, select_stage9_navigation_traversability
        adapter = BDDLAdapter()
        effective_particle_systems_enabled = bool(particle_systems_enabled) if particle_systems_enabled_override is None else bool(particle_systems_enabled_override)
        effective_particle_source_stream_visuals_enabled = bool(particle_source_stream_visuals_enabled) if particle_source_stream_visuals_enabled_override is None else bool(particle_source_stream_visuals_enabled_override)
        effective_native_particle_source_visuals_enabled = bool(native_particle_source_visuals_enabled) if native_particle_source_visuals_enabled_override is None else bool(native_particle_source_visuals_enabled_override)
        if not effective_particle_systems_enabled:
            effective_native_particle_source_visuals_enabled = False
        effective_include_navigation_context = bool(include_navigation_context) if include_navigation_context_override is None else bool(include_navigation_context_override)
        episode_scene_layout_mode = _effective_scene_layout_mode(scene_layout_mode, task_spec)
        render_result = build_render_scene(task_spec, adapter=adapter, rng_seed=rng_seed, scene_layout_mode=episode_scene_layout_mode, load_doors=load_doors, particle_systems_enabled=effective_particle_systems_enabled)
        write_json(logs_dir / 'render_scene_report.json', render_result.render_report)
        env_config = make_stage9_render_env_config(task_spec, scene_layout_mode=episode_scene_layout_mode, include_visibility_modalities=visibility_settings['include_visibility_modalities'], visibility_modality='seg_instance_id' if formal_trajectory_capture else 'seg_instance', load_doors=load_doors)
        _apply_structural_render_preset(env_config, structural_preset, scene_layout_mode=episode_scene_layout_mode)
        gripper_visual_config = gripper_visual_config_from_env()
        ego_arm_visual_config = ego_arm_visual_config_from_env()
        write_json(logs_dir / 'render_env_config.json', render_env_config_for_log(env_config, render_result.og_objects, render_result.render_aliases, render_result.render_alias_metadata, gripper_visual_config=gripper_visual_config, ego_arm_visual_config=ego_arm_visual_config))
        renderer = EgoRenderer(env_config, render_result.og_objects, renders_per_frame=int(renders_per_frame), render_aliases=render_result.render_aliases, render_alias_metadata=render_result.render_alias_metadata, hide_scene_structure_categories=_hidden_structure_categories_for_preset(structural_preset, scene_layout_mode=episode_scene_layout_mode), reset_diagnostics_path=logs_dir / 'renderer_reset_timing.jsonl', gripper_visual_config=gripper_visual_config, ego_arm_visual_config=ego_arm_visual_config, particle_systems_enabled=effective_particle_systems_enabled, native_particle_source_visuals_enabled=effective_native_particle_source_visuals_enabled, particle_source_stream_visuals_enabled=effective_particle_source_stream_visuals_enabled)
        with _renderer_factory_lifetime(renderer):
            renderer.reset()
            write_json(logs_dir / 'render_runtime_scene_report.json', renderer.scene_summary())
            env_kwargs: dict[str, Any] = {'scene_layout_mode': episode_scene_layout_mode, 'occupancy_agent_radius': STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS, 'occupancy_safety_margin': STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN, 'use_rendered_visibility': effective_use_rendered_visibility, 'action_visibility_mode': 'rendered_required' if effective_use_rendered_visibility else 'line_of_sight_required' if require_line_of_sight_navigation else 'disabled', 'load_doors': bool(load_doors), 'capture_rendered_observation_evidence': visibility_settings['capture_rendered_observation_evidence'], 'allow_symbolic_public_observation_evidence': visibility_settings['allow_symbolic_public_observation_evidence']}
            if resolved_segmentation_capture_mode == 'required':
                env_kwargs['rendered_visibility_modality'] = 'seg_instance_id'
            if effective_include_navigation_context:
                try:
                    try:
                        scene_traversability = renderer.scene_traversability(occupancy_resolution=0.1)
                    except Exception as exc:
                        if not allow_navigation_context_fallback:
                            raise
                        scene_traversability = renderer.scene_traversability(occupancy_resolution=0.1, use_indoor_segmentation_mask=False)
                        write_json(logs_dir / 'renderer_navigation_context_warning.json', {'warning': 'scene_traversability_indoor_mask_fallback', 'error': '{}: {}'.format(type(exc).__name__, exc)})
                    scene_traversability = _attach_disabled_doorway_openings_to_traversability(scene_traversability, task_spec, scene_layout_mode=episode_scene_layout_mode, load_doors=load_doors)
                    scene_navigation_objects = renderer.scene_navigation_objects(load_doors=load_doors)
                    interaction_faces = renderer.interaction_faces_by_name(load_doors=load_doors)
                    raw_scene_mesh = renderer.scene_mesh_occupancy(occupancy_resolution=0.1, map_size=getattr(scene_traversability, 'map_size', None))
                    scene_indoor = indoor_mask_as_traversability(scene_traversability)
                    scene_mesh = mask_traversability_with_reference(raw_scene_mesh, scene_indoor, label='floor_insseg')
                    line_of_sight_traversability = mask_traversability_with_reference(raw_scene_mesh, scene_indoor, label='floor_insseg') if bool(getattr(scene_indoor, 'enabled', False)) else scene_mesh
                    wall_slice_resolution = 0.05
                    wall_slice_map_size = None
                    if getattr(scene_traversability, 'map_size', None) and getattr(scene_traversability, 'map_resolution', None):
                        wall_slice_map_size = int(round(float(scene_traversability.map_size) * float(scene_traversability.map_resolution) / wall_slice_resolution))
                    wall_slice = renderer.scene_wall_slice_occupancy(occupancy_resolution=wall_slice_resolution, map_size=wall_slice_map_size, slice_heights=(0.2, 0.8, 1.2), visual_fallback=False, exterior_barriers=scene_navigation_objects, interior_mask=getattr(scene_traversability, 'indoor_mask', None), interior_mask_resolution=getattr(scene_traversability, 'map_resolution', None), interior_mask_size=getattr(scene_traversability, 'map_size', None))
                    wall_slice_traversability = scene_wall_slice_occupancy_as_traversability(wall_slice)
                    wall_slice_traversability = mask_traversability_with_reference(wall_slice_traversability, scene_indoor, label='floor_insseg')
                    navigation_traversability, navigation_selection = select_stage9_navigation_traversability(task_spec, (('wall_slice', wall_slice_traversability), ('scene_traversability', scene_traversability), ('mesh_occupancy', scene_mesh)), adapter=adapter, rng_seed=rng_seed, interaction_faces_by_name=interaction_faces, scene_navigation_objects=scene_navigation_objects, scene_layout_mode=episode_scene_layout_mode, load_doors=load_doors)
                    env_kwargs['scene_traversability'] = navigation_traversability
                    env_kwargs['line_of_sight_traversability'] = line_of_sight_traversability
                    env_kwargs['scene_navigation_objects'] = scene_navigation_objects
                    env_kwargs['interaction_faces_by_name'] = interaction_faces
                    navigation_context_selection = {'selected': navigation_selection.get('selected'), 'reason': navigation_selection.get('reason'), 'candidates': navigation_selection.get('candidates', []), 'line_of_sight_source': getattr(line_of_sight_traversability, 'source', None)}
                    write_json(logs_dir / 'renderer_navigation_context_selection.json', navigation_context_selection)
                    renderer._trajectory_navigation_context_selection = navigation_context_selection
                except Exception as exc:
                    if not allow_navigation_context_fallback:
                        raise RuntimeError('canonical renderer navigation context unavailable') from exc
                    write_json(logs_dir / 'renderer_navigation_context_warning.json', {'warning': '{}: {}'.format(type(exc).__name__, exc)})
            else:
                write_json(logs_dir / 'renderer_navigation_context_warning.json', {'warning': 'renderer_navigation_context_skipped'})
            return {'renderer': renderer, 'env_kwargs': env_kwargs}
    return _factory

def _trajectory_visibility_settings(*, formal_trajectory_capture: bool, use_rendered_visibility: bool, segmentation_capture_mode: str | None=None) -> Mapping[str, bool]:
    mode = _resolve_segmentation_capture_mode(formal_trajectory_capture=formal_trajectory_capture, segmentation_capture_mode=segmentation_capture_mode)
    return {'include_visibility_modalities': False, 'capture_rendered_observation_evidence': mode == 'required', 'allow_symbolic_public_observation_evidence': mode != 'required'}

def _resolve_segmentation_capture_mode(*, formal_trajectory_capture: bool, segmentation_capture_mode: str | None) -> str:
    mode = str(segmentation_capture_mode if segmentation_capture_mode is not None else 'required' if formal_trajectory_capture else 'off').strip().lower()
    if mode not in {'required', 'on_demand', 'off'}:
        raise ValueError('segmentation_capture_mode must be required, on_demand, or off')
    if formal_trajectory_capture and mode != 'required':
        raise ValueError('formal trajectory capture requires segmentation_capture_mode=required')
    return mode

def _apply_structural_render_preset(env_config: dict[str, Any], structural_preset: str, *, scene_layout_mode: str) -> None:
    """Adjust structural categories for lightweight task-role ego rendering.

    ``source_json_full`` intentionally keeps its full scene-file loading behavior.
    ``stage9`` is the normal VLM-eval path and preserves source-scene structure so
    ego RGB is grounded in an indoor scene. ``minimal`` and ``none`` are debug modes
    for isolating task-role assets; they are not suitable for model-quality runs.
    """
    if str(scene_layout_mode) == 'source_json_full':
        return
    scene = env_config.setdefault('scene', {})
    preset = str(structural_preset or 'stage9').strip().lower()
    if preset == 'stage9':
        return
    if preset == 'minimal':
        scene['load_object_categories'] = ['room_light', 'downlight', 'square_light']
        scene['not_load_object_categories'] = _hidden_structure_categories_for_preset(preset, scene_layout_mode=scene_layout_mode)
        return
    if preset == 'none':
        scene['load_object_categories'] = []
        scene['not_load_object_categories'] = _hidden_structure_categories_for_preset(preset, scene_layout_mode=scene_layout_mode)
        return
    raise ValueError('Unsupported renderer structural preset: {}'.format(structural_preset))

def _hidden_structure_categories_for_preset(structural_preset: str, *, scene_layout_mode: str) -> list[str]:
    if str(scene_layout_mode) == 'source_json_full':
        return []
    preset = str(structural_preset or 'stage9').strip().lower()
    if preset not in {'minimal', 'none'}:
        return []
    return ['floors', 'floor', 'ceilings', 'ceiling', 'walls', 'wall', 'door', 'sliding_door', 'fixed_window', 'window', 'window_blind']

def _load_vlm_config(path: Path | None, *, fake_vlm: bool) -> VLMConfig:
    if path is not None:
        return VLMConfig.from_yaml(path)
    if fake_vlm:
        return VLMConfig.from_mapping({'adapter': {'type': 'openai', 'model': 'fake'}, 'observation': {'mode': 'streaming', 'stream': {'input_fps': 2.0, 'recent_window_frames': 2}}})
    raise ValueError('vlm_config_path is required unless fake_vlm is enabled.')

def _load_task_batch(*, task_bank_path: Path | None, task_manifest_path: Path | None, scene_models: Sequence[str], generate_stage4: bool, generate_observation_conditioned: bool, scene_variant_files: Sequence[str | Path], generate_scene_variants: bool, num_scene_variants: int, scene_variant_seed: int, scene_variant_output_root: Path | None, scene_variant_config: Mapping[str, Any], assets_root: Path | None, template_ids: Sequence[str], max_bindings_per_template: int, max_tasks_per_template_scene: int, generation_dir: Path) -> TaskBatch:
    variant_files: list[Path] = [resolve_repo_path(path) for path in scene_variant_files]
    scene_variant_records: list[dict[str, Any]] = []
    if generate_scene_variants:
        generated = _generate_scene_variants(scene_models=scene_models or ('Rs_int',), output_root=scene_variant_output_root or generation_dir / 'scene_variants', assets_root=assets_root, num_variants=num_scene_variants, seed=scene_variant_seed, config=scene_variant_config)
        scene_variant_records.extend(generated)
        variant_files.extend((Path(record['scene_file']) for record in generated if record.get('scene_file')))
    if variant_files:
        return _generate_stage4_variant_task_batch(scene_files=variant_files, generation_dir=generation_dir, assets_root=assets_root, seed=scene_variant_seed, template_ids=template_ids, max_bindings_per_template=max_bindings_per_template, max_tasks_per_template_scene=max_tasks_per_template_scene, scene_variant_records=scene_variant_records)
    if generate_stage4:
        return _generate_stage4_task_batch(scene_models=scene_models or ('Rs_int',), generation_dir=generation_dir, template_ids=template_ids, max_bindings_per_template=max_bindings_per_template, max_tasks_per_template_scene=max_tasks_per_template_scene)
    if generate_observation_conditioned:
        tasks = _filter_tasks_by_template_ids(_generate_observation_conditioned_tasks(scene_models or ('Rs_int',)), template_ids)
        return TaskBatch(tasks=tasks, artifacts={}, generation={'mode': 'observation_conditioned', 'scene_models': list(scene_models or ('Rs_int',)), 'generated_total': len(tasks)})
    if task_bank_path is not None:
        tasks = _filter_tasks_by_template_ids(_read_task_bank(task_bank_path), template_ids)
        return TaskBatch(tasks=tasks, artifacts={'input_task_bank_json': str(task_bank_path)}, generation={'mode': 'task_bank', 'generated_total': len(tasks)})
    if task_manifest_path is not None:
        manifest = _read_json(task_manifest_path)
        if isinstance(manifest, Mapping):
            if 'tasks' in manifest and isinstance(manifest['tasks'], list):
                tasks = _filter_tasks_by_template_ids([dict(item) for item in manifest['tasks'] if isinstance(item, Mapping)], template_ids)
                return TaskBatch(tasks=tasks, artifacts={'input_task_manifest_json': str(task_manifest_path)}, generation={'mode': 'task_manifest_inline', 'generated_total': len(tasks)})
            artifacts = manifest.get('artifacts') or {}
            task_bank_json = artifacts.get('task_bank_json') if isinstance(artifacts, Mapping) else None
            if isinstance(task_bank_json, str) and task_bank_json:
                candidate = task_manifest_path.parent / task_bank_json
                if candidate.exists():
                    tasks = _filter_tasks_by_template_ids(_read_task_bank(candidate), template_ids)
                    return TaskBatch(tasks=tasks, artifacts={'input_task_manifest_json': str(task_manifest_path), 'input_task_bank_json': str(candidate)}, generation={'mode': 'task_manifest_artifact', 'generated_total': len(tasks)})
        return TaskBatch(tasks=[], artifacts={'input_task_manifest_json': str(task_manifest_path)}, generation={'mode': 'task_manifest_empty'})
    return TaskBatch(tasks=[], artifacts={}, generation={'mode': 'empty'})

def _load_tasks(*, task_bank_path: Path | None, task_manifest_path: Path | None, scene_models: Sequence[str], generate_stage4: bool, generate_observation_conditioned: bool) -> list[dict[str, Any]]:
    return _load_task_batch(task_bank_path=task_bank_path, task_manifest_path=task_manifest_path, scene_models=scene_models, generate_stage4=generate_stage4, generate_observation_conditioned=generate_observation_conditioned, scene_variant_files=(), generate_scene_variants=False, num_scene_variants=1, scene_variant_seed=0, scene_variant_output_root=None, scene_variant_config={}, assets_root=None, template_ids=(), max_bindings_per_template=2, max_tasks_per_template_scene=1, generation_dir=Path('outputs/eb_behavior_vlm_benchmark_generation')).tasks

def _generate_scene_variants(*, scene_models: Sequence[str], output_root: Path, assets_root: Path | None, num_variants: int, seed: int, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from scoper.benchmark.behavior.scene_geometry.asset_catalog import AssetCatalog
    from scoper.benchmark.behavior.scene_geometry.category_classifier import CategoryClassifier
    from scoper.benchmark.behavior.scene_geometry.layout_quality import audit_layout_quality, write_layout_quality_report
    from scoper.benchmark.behavior.scene_geometry.paths import default_scene_json_path
    from scoper.benchmark.behavior.scene_geometry.rules import load_rules
    from scoper.benchmark.behavior.scene_geometry.sampler import SceneVariantGenerator, VariantConfig
    from scoper.benchmark.behavior.scene_geometry.scene_io import write_scene_json
    from scoper.benchmark.behavior.scene_geometry.scene_statistics import build_scene_statistics
    from scoper.benchmark.behavior.scene_geometry.taxonomy_index import TaxonomyIndex
    from scoper.benchmark.behavior.scene_geometry.validator import validate_scene_variant, write_validation_report
    assets = Path(assets_root) if assets_root is not None else resolve_repo_path('datasets/behavior-1k-assets')
    output_root = Path(output_root)
    rules = load_rules()
    catalog = AssetCatalog.build(assets)
    taxonomy = TaxonomyIndex.build()
    classifier = CategoryClassifier(catalog, taxonomy, rules)
    generator = SceneVariantGenerator(catalog=catalog, taxonomy=taxonomy, classifier=classifier, statistics=build_scene_statistics(assets), rules=rules, assets_root=assets)
    variant_config = VariantConfig(large_layout_mode=str(config.get('large_layout_mode') or 'balanced'), large_rewrite_fraction=float(config.get('large_rewrite_fraction', 0.6)), space_budget=config.get('space_budget'), diversity_level=float(config.get('diversity_level', 1.0)), room_subset_mode=str(config.get('room_subset_mode') or 'all'), room_subset_size=int(config.get('room_subset_size') or 0), active_rooms=tuple(config.get('active_rooms') or ()), exclude_room_types=tuple(config.get('exclude_room_types') or ()), add_large=bool(config.get('add_large', True)), add_small=bool(config.get('add_small', True)), model_swap=bool(config.get('model_swap', True)), pose_jitter=bool(config.get('pose_jitter', True)), preserve_task_objects=bool(config.get('preserve_task_objects', True)), stationary_category_swap=bool(config.get('stationary_category_swap', True)), stationary_category_swap_probability=float(config.get('stationary_category_swap_probability', 0.65)))
    records: list[dict[str, Any]] = []
    for scene_model in scene_models:
        input_json = Path(config.get('input_json') or default_scene_json_path(str(scene_model), assets=assets))
        scene_out_dir = output_root / 'scenes' / str(scene_model) / 'json'
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        for offset in range(max(0, int(num_variants))):
            variant_seed = int(seed) + offset
            result = generator.generate_from_file(scene_model=str(scene_model), input_json=input_json, seed=variant_seed, config=variant_config)
            stem = '{}_variant_{:04d}'.format(scene_model, variant_seed)
            scene_path = scene_out_dir / '{}.json'.format(stem)
            summary_path = scene_out_dir / '{}_summary.json'.format(stem)
            validation_path = scene_out_dir / '{}_validation.json'.format(stem)
            layout_quality_path = scene_out_dir / '{}_layout_quality.json'.format(stem)
            write_scene_json(scene_path, result.scene)
            write_json(summary_path, result.summary)
            validation = validate_scene_variant(scene_model=str(scene_model), scene_file=scene_path, catalog=catalog, classifier=classifier, rules=rules, source_scene_file=input_json, assets_root=assets)
            write_validation_report(validation_path, validation)
            layout_quality = audit_layout_quality(scene_model=str(scene_model), scene_file=scene_path, source_scene_file=input_json, catalog=catalog, classifier=classifier, rules=rules, assets_root=assets)
            write_layout_quality_report(layout_quality_path, layout_quality)
            status = 'passed' if validation.get('status') == 'passed' and layout_quality.get('status') == 'passed' else 'failed'
            records.append({'scene_model': str(scene_model), 'layout_variant_id': stem, 'seed': variant_seed, 'source_scene_file': str(input_json), 'scene_file': str(scene_path), 'summary_json': str(summary_path), 'validation_json': str(validation_path), 'layout_quality_json': str(layout_quality_path), 'status': status, 'summary': {'added_large': result.summary.get('added_large'), 'rewritten_large': result.summary.get('rewritten_large'), 'room_relayout_rooms': result.summary.get('room_relayout_rooms'), 'added_small': result.summary.get('added_small'), 'model_swaps': result.summary.get('model_swaps'), 'pose_jitters': result.summary.get('pose_jitters')}, 'validation_status': validation.get('status'), 'layout_quality_status': layout_quality.get('status')})
    return records

def _generate_stage4_task_batch(*, scene_models: Sequence[str], generation_dir: Path, template_ids: Sequence[str], max_bindings_per_template: int, max_tasks_per_template_scene: int) -> TaskBatch:
    profiles = profile_scenes(list(scene_models))
    templates = load_procthor_templates()
    matches: list[dict[str, Any]] = []
    for profile in profiles:
        matches.extend(match_scene_to_templates(profile, templates, max_bindings_per_template=int(max_bindings_per_template)))
    generation = generate_stage4_tasks(profiles=profiles, templates=templates, match_records=matches, template_ids=template_ids or None, max_tasks_per_template_scene=int(max_tasks_per_template_scene))
    artifacts = _write_generation_artifacts(generation_dir, profiles=profiles, match_records=matches, generation=generation, scene_variants=[])
    summary = dict(generation.get('summary') or {})
    summary.update({'mode': 'stage4_scene_models', 'scene_models': list(scene_models), 'template_ids': list(template_ids or ()), 'profile_count': len(profiles), 'match_record_count': len(matches), 'generated_total': len(generation.get('tasks') or []), 'rejected_total': len(generation.get('rejected') or [])})
    return TaskBatch(tasks=[dict(task) for task in generation['tasks']], artifacts=artifacts, generation=summary)

def _generate_stage4_tasks(scene_models: Sequence[str]) -> list[dict[str, Any]]:
    return _generate_stage4_task_batch(scene_models=scene_models, generation_dir=Path('outputs/eb_behavior_vlm_benchmark_generation'), template_ids=(), max_bindings_per_template=2, max_tasks_per_template_scene=1).tasks

def _generate_stage4_variant_task_batch(*, scene_files: Sequence[Path], generation_dir: Path, assets_root: Path | None, seed: int, max_bindings_per_template: int, max_tasks_per_template_scene: int, scene_variant_records: Sequence[Mapping[str, Any]], template_ids: Sequence[str] | None=None) -> TaskBatch:
    assets = Path(assets_root) if assets_root is not None else resolve_repo_path('datasets/behavior-1k-assets')
    templates = load_procthor_templates()
    profiles: list[dict[str, Any]] = []
    all_matches: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    per_variant: list[dict[str, Any]] = []
    for scene_file in scene_files:
        scene_path = Path(scene_file)
        scene_model = _scene_model_from_variant_path(scene_path)
        variant_id = scene_path.stem
        profile = profile_scene(scene_model, assets_root=assets, scene_json=scene_path, layout_variant_id=variant_id)
        profiles.append(profile)
        matches = match_scene_to_templates(profile, templates, max_bindings_per_template=int(max_bindings_per_template))
        all_matches.extend(matches)
        generation = generate_stage4_tasks(profiles=[profile], templates=templates, match_records=matches, template_ids=template_ids or None, max_tasks_per_template_scene=int(max_tasks_per_template_scene), seed=int(seed))
        tasks.extend((dict(task) for task in generation.get('tasks') or []))
        rejected.extend((dict(item) for item in generation.get('rejected') or []))
        per_variant.append({'scene_model': scene_model, 'layout_variant_id': variant_id, 'source_json': str(scene_path), 'profile_object_count': profile.get('object_count'), 'match_record_count': len(matches), 'generated_total': len(generation.get('tasks') or []), 'rejected_total': len(generation.get('rejected') or []), 'summary': generation.get('summary') or {}})
    generation_summary = {'mode': 'stage4_scene_variants', 'scene_file_count': len(scene_files), 'template_ids': list(template_ids or ()), 'profile_count': len(profiles), 'match_record_count': len(all_matches), 'generated_total': len(tasks), 'rejected_total': len(rejected), 'per_variant': per_variant}
    generation = {'tasks': tasks, 'rejected': rejected, 'summary': generation_summary}
    artifacts = _write_generation_artifacts(generation_dir, profiles=profiles, match_records=all_matches, generation=generation, scene_variants=list(scene_variant_records))
    return TaskBatch(tasks=tasks, artifacts=artifacts, generation=generation_summary)

def _write_generation_artifacts(generation_dir: Path, *, profiles: Sequence[Mapping[str, Any]], match_records: Sequence[Mapping[str, Any]], generation: Mapping[str, Any], scene_variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    generation_dir.mkdir(parents=True, exist_ok=True)
    paths = {'scene_variants_json': generation_dir / 'scene_variants.json', 'profiles_json': generation_dir / 'profiles.json', 'match_records_json': generation_dir / 'match_records.json', 'generation_summary_json': generation_dir / 'generation_summary.json', 'generation_rejected_json': generation_dir / 'generation_rejected.json', 'raw_generated_task_bank_json': generation_dir / 'raw_task_bank.json'}
    write_json(paths['scene_variants_json'], {'scene_variants': list(scene_variants)})
    write_json(paths['profiles_json'], {'profiles': list(profiles)})
    write_json(paths['match_records_json'], {'match_records': list(match_records)})
    write_json(paths['generation_summary_json'], generation.get('summary') or {})
    write_json(paths['generation_rejected_json'], {'rejected': list(generation.get('rejected') or [])})
    write_json(paths['raw_generated_task_bank_json'], {'tasks': list(generation.get('tasks') or [])})
    return {key: str(path) for key, path in paths.items()}

def _scene_model_from_variant_path(scene_file: Path) -> str:
    parts = scene_file.parts
    if 'scenes' in parts:
        index = parts.index('scenes')
        if index + 1 < len(parts):
            return parts[index + 1]
    stem = scene_file.stem
    if '_variant_' in stem:
        return stem.split('_variant_', 1)[0]
    if stem.endswith('_best'):
        return stem[:-len('_best')]
    return scene_file.parent.parent.name if scene_file.parent.name == 'json' else stem

def _generate_observation_conditioned_tasks(scene_models: Sequence[str]) -> list[dict[str, Any]]:
    profiles = profile_scenes(list(scene_models))
    generation = generate_observation_conditioned_tasks(profiles=profiles)
    return [dict(task) for task in generation['tasks']]

def _filter_tasks_by_template_ids(tasks: Sequence[Mapping[str, Any]], template_ids: Sequence[str]) -> list[dict[str, Any]]:
    allowed = {str(template_id) for template_id in template_ids if str(template_id)}
    if not allowed:
        return [dict(task) for task in tasks]
    return [dict(task) for task in tasks if _task_template_id(task) in allowed]

def _task_template_id(task: Mapping[str, Any]) -> str:
    template = task.get('template')
    if isinstance(template, Mapping) and template.get('template_id'):
        return str(template.get('template_id'))
    source = task.get('source')
    if isinstance(source, Mapping) and source.get('template_id'):
        return str(source.get('template_id'))
    return str(task.get('template_id') or '')

def _validate_task_readiness(tasks: Sequence[Mapping[str, Any]], *, scene_layout_mode: str='auto', load_doors: bool=False) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    passed = 0
    for index, task in enumerate(tasks):
        task_id = str(task.get('id') or f'task_{index:05d}')
        record = {'task_id': task_id, 'task_index': index, 'scene_layout_mode': _effective_scene_layout_mode(scene_layout_mode, task), 'load_doors': bool(load_doors), 'structure_ok': False, 'reset_ok': False, 'prompt_context_ok': False, 'head_rgb_ok': False, 'vlm_ready': False, 'failure': None}
        try:
            validate_behavior_task_spec(task)
            record['structure_ok'] = True
            env = BehaviorClosedLoopToolEnv(renderer=BlankRenderer(), stream_controller=None, scene_layout_mode=str(record['scene_layout_mode']), load_doors=load_doors)
            reset_obs = env.reset_to_task(task, rng_seed=0)
            record['reset_ok'] = True
            record['head_rgb_ok'] = reset_obs.get('head_rgb') is not None
            prompt_context = env.get_prompt_context()
            record['prompt_context_ok'] = bool(prompt_context.get('tool_specs')) and bool(prompt_context.get('tool_contract', {}).get('allowed_model_tools'))
            record['vlm_ready'] = all((bool(record[key]) for key in ('structure_ok', 'reset_ok', 'prompt_context_ok', 'head_rgb_ok')))
            close = getattr(env.renderer, 'close', None)
            if callable(close):
                close()
        except Exception as exc:
            record['failure'] = '{}: {}'.format(type(exc).__name__, exc)
        if record['vlm_ready']:
            passed += 1
        records.append(record)
    return {'validated_total': len(records), 'passed_count': passed, 'failed_count': len(records) - passed, 'records': records}

def _validate_generation_admission(tasks: Sequence[Mapping[str, Any]], *, mode: str, scene_layout_mode: str='auto', load_doors: bool=False) -> dict[str, Any]:
    mode = str(mode or 'readiness').strip().lower()
    if mode == 'readiness':
        return {'mode': 'readiness', 'validated_total': len(tasks), 'admitted_count': len(tasks), 'failed_count': 0, 'records': [{'task_id': str(task.get('id') or f'task_{index:05d}'), 'task_index': index, 'scene_layout_mode': _effective_scene_layout_mode(scene_layout_mode, task), 'load_doors': bool(load_doors), 'admitted': True, 'admission_source': 'task_readiness'} for index, task in enumerate(tasks)]}
    if mode not in {'stage5', 'stage6'}:
        raise ValueError("generation_validation must be 'readiness', 'stage5', or 'stage6'.")
    from scoper.benchmark.behavior.task_generation.stage5_validator import summarize_stage5_results, validate_stage5_reset_mutation
    generation = {'tasks': [dict(task) for task in tasks], 'summary': {}}
    stage5_results = [validate_stage5_reset_mutation(task, rng_seed=0, scene_layout_mode=_effective_scene_layout_mode(scene_layout_mode, task), load_doors=load_doors) for task in tasks]
    stage5_summary = summarize_stage5_results(generation, stage5_results)
    if mode == 'stage5':
        records = [{'task_id': str(result.get('task_id') or tasks[index].get('id') or f'task_{index:05d}'), 'task_index': index, 'scene_layout_mode': _effective_scene_layout_mode(scene_layout_mode, tasks[index]), 'load_doors': bool(load_doors), 'admitted': bool(result.get('stage5_reset_mutation_ok')), 'admission_source': 'stage5_reset_mutation', 'stage5_bucket': result.get('stage5_bucket'), 'failure_code': result.get('failure_code'), 'failure_reason': result.get('failure_reason')} for index, result in enumerate(stage5_results)]
        admitted = sum((1 for record in records if record['admitted']))
        return {'mode': 'stage5', 'validated_total': len(records), 'admitted_count': admitted, 'failed_count': len(records) - admitted, 'stage5_summary': stage5_summary, 'stage5_results': stage5_results, 'records': records}
    from scoper.benchmark.behavior.task_generation.stage6_executor import summarize_stage6_results, validate_stage6_execution
    stage5_by_task = {item.get('task_id'): item for item in stage5_results}
    stage6_results = [validate_stage6_execution(task, stage5_result=stage5_by_task.get(task.get('id')), rng_seed=0, scene_layout_mode=_effective_scene_layout_mode(scene_layout_mode, task), load_doors=load_doors) for task in tasks]
    stage6_summary = summarize_stage6_results(generation, stage5_results, stage6_results)
    records = [{'task_id': str(result.get('task_id') or tasks[index].get('id') or f'task_{index:05d}'), 'task_index': index, 'scene_layout_mode': _effective_scene_layout_mode(scene_layout_mode, tasks[index]), 'load_doors': bool(load_doors), 'admitted': bool(result.get('execution_ok')), 'admission_source': 'stage6_execution', 'stage5_bucket': stage5_results[index].get('stage5_bucket') if index < len(stage5_results) else None, 'stage6_bucket': result.get('stage6_bucket'), 'failure_code': result.get('failure_code'), 'failure_reason': result.get('failure_reason')} for index, result in enumerate(stage6_results)]
    admitted = sum((1 for record in records if record['admitted']))
    return {'mode': 'stage6', 'validated_total': len(records), 'admitted_count': admitted, 'failed_count': len(records) - admitted, 'stage5_summary': stage5_summary, 'stage6_summary': stage6_summary, 'stage5_results': stage5_results, 'stage6_results': stage6_results, 'records': records}

def _image_paths_from_logged_messages(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    paths: list[str] = []
    for message in messages:
        blocks = message.get('content', []) if isinstance(message, Mapping) else []
        for block in blocks:
            if isinstance(block, Mapping) and block.get('type') == 'image' and block.get('path'):
                paths.append(str(block.get('path')))
    return paths

def _read_task_bank(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, Mapping)]
    if isinstance(data, Mapping):
        tasks = data.get('tasks')
        if isinstance(tasks, list):
            return [dict(item) for item in tasks if isinstance(item, Mapping)]
    raise ValueError('Unsupported task bank format at {}.'.format(path))

def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def _apply_shard(tasks: Sequence[Mapping[str, Any]], *, shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count <= 1:
        return [dict(task) for task in tasks]
    shard_index = max(0, min(int(shard_index), shard_count - 1))
    out = []
    for index, task in enumerate(tasks):
        if index % shard_count == shard_index:
            out.append(dict(task))
    return out

def _safe_slug(text: str) -> str:
    chars = []
    for ch in str(text):
        if ch.isalnum() or ch in {'-', '_'}:
            chars.append(ch)
        else:
            chars.append('_')
    slug = ''.join(chars).strip('_')
    return slug or 'task'

def _final_goal_summary(final_goal: Mapping[str, Any]) -> dict[str, Any]:
    return {'task_success': final_goal.get('task_success'), 'task_progress': final_goal.get('task_progress'), 'model_stopped': final_goal.get('model_stopped'), 'require_model_stop': final_goal.get('require_model_stop')}

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
__all__ = ['BenchmarkResult', 'FakeBenchmarkAdapter', 'main', 'make_stage9_ego_renderer_factory', 'run_vlm_benchmark']
