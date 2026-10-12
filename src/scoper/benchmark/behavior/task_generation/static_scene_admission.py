from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import cv2
from PIL import Image
from scoper.benchmark.common import build_static_scene_admission_result
from scoper.benchmark.behavior.execution_core.occupancy_utils import SceneTraversability
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.benchmark.behavior.brains.base import ToolCall
from scoper.benchmark.behavior.brains.teachers.dispatcher import default_dispatcher
from scoper.benchmark.behavior.episode_runner import EpisodeRunner
from scoper.benchmark.behavior.json_utils import json_safe
from scoper.benchmark.behavior.paths import repo_root
from scoper.benchmark.behavior.render.asset_resolver import AssetPreflightError, require_task_render_assets
from scoper.benchmark.behavior.scene import build_scene
from scoper.benchmark.behavior.scene.doors import is_door_like
from scoper.benchmark.behavior.scene.navigation_settings import STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS, STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN
from scoper.benchmark.behavior.scene.room_layout_maps import load_behavior_room_layout_maps
from scoper.benchmark.behavior.task_generation.stage5_validator import validate_reachability, validate_stage5_reset_mutation
from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec
BEHAVIOR_STATIC_SCENE_ADMISSION_IMPLEMENTATION_ID = 'behavior-static-scene-admission.cpu-floor-layout.v1'
BEHAVIOR_STATIC_SCENE_GEOMETRY_SOURCE = 'floor_trav+floor_insseg+source_json_aabb'
_CHECK_NAMES = ('task_schema', 'source_layout', 'render_assets', 'layout_merge', 'spawn_collisions', 'stage5_semantics', 'start_pose', 'required_roles', 'oracle_rollout', 'qa', 'combined')

class _QAAnsweringOracle:

    def __init__(self, delegate: Any, env: BehaviorClosedLoopToolEnv):
        self.delegate = delegate
        self.env = env

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> Any:
        return self.delegate.on_episode_start(env, task_spec)

    def act(self, brain_input: Any) -> ToolCall:
        pending = [message for message in (brain_input.prompt_context or {}).get('human_messages') or [] if message.get('type') == 'qa']
        if pending:
            question_id = str(pending[0].get('id') or '')
            evaluator_keys = dict((getattr(self.env, '_human_qa_state', {}) or {}).get('evaluator_keys') or {})
            if question_id not in evaluator_keys:
                raise RuntimeError(f'Oracle QA evaluator key missing for {question_id!r}.')
            return ToolCall('answer', {'question_id': question_id, 'value': evaluator_keys[question_id]}, reasoning='static admission oracle answers situated QA')
        return self.delegate.act(brain_input)

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any]) -> Any:
        callback = getattr(self.delegate, 'on_episode_end', None)
        return callback(env, final_goal) if callable(callback) else None

def build_static_scene_traversability(task_spec: Mapping[str, Any], *, assets_root: str | Path | None=None, floor: int=0, occupancy_resolution: float=0.1) -> SceneTraversability:
    source = task_spec.get('source') if isinstance(task_spec.get('source'), Mapping) else {}
    scene_model = str(source.get('scene_model') or '')
    if not scene_model:
        raise ValueError('task source.scene_model is required for static scene admission')
    resolution = float(occupancy_resolution)
    if resolution <= 0.0:
        raise ValueError('occupancy_resolution must be positive')
    layout = load_behavior_room_layout_maps(scene_model, assets_root=assets_root, floor=int(floor))
    indoor = np.asarray(layout.room_ins_map) > 0
    traversable = np.where(indoor, np.asarray(layout.trav_map), 0).astype(np.uint8, copy=False)
    target_size = max(1, int(float(layout.map_size) * float(layout.map_resolution) / resolution))
    floor_map = _resize_traversability_like_omnigibson(traversable, target_size)
    indoor_mask = _resize_mask(indoor, target_size)
    floor_map = np.where(indoor_mask, floor_map, False)
    source_paths = dict(layout.source_paths)
    doorway_opening_objects = _disabled_doorway_opening_objects(task_spec)
    metadata = {'enabled': True, 'source': 'scoper.benchmark.behavior.room_layout_maps.floor_trav&floor_insseg', 'scene_model': scene_model, 'floor': int(floor), 'raw_map_resolution': float(layout.map_resolution), 'raw_map_size': int(layout.map_size), 'map_resolution': resolution, 'map_size': int(target_size), 'shape': [int(target_size), int(target_size)], 'free_pixels': int(np.count_nonzero(floor_map)), 'floor_map_sha256': hashlib.sha256(np.ascontiguousarray(np.where(floor_map, 255, 0).astype(np.uint8)).tobytes()).hexdigest(), 'indoor_mask_sha256': hashlib.sha256(np.ascontiguousarray(indoor_mask.astype(np.uint8)).tobytes()).hexdigest(), 'source_paths': source_paths}
    if doorway_opening_objects:
        metadata['disabled_doorway_openings'] = {'object_count': int(len(doorway_opening_objects)), 'objects': [{'name': str(obj.name), 'category': str(obj.category), 'center': [float(value) for value in obj.center], 'extent': [float(value) for value in obj.extent]} for obj in doorway_opening_objects]}
    return SceneTraversability(floor_map=np.where(floor_map, 255, 0).astype(np.uint8), map_resolution=resolution, map_size=int(target_size), floor=int(floor), source=str(metadata['source']), indoor_mask=indoor_mask, doorway_opening_objects=doorway_opening_objects, doorway_opening_allowed_map=indoor_mask, doorway_opening_allowed_map_resolution=resolution, doorway_opening_allowed_map_size=target_size, metadata=metadata)

def _disabled_doorway_opening_objects(task_spec: Mapping[str, Any]) -> tuple[Any, ...]:
    source = task_spec.get('source') if isinstance(task_spec.get('source'), Mapping) else {}
    if not source.get('source_json'):
        return ()
    try:
        from scoper.benchmark.behavior.scene.variant_layout import merge_source_json_layout
        merged = merge_source_json_layout(task_spec, (), {}, include_pickupable_background=True, load_doors=True)
    except Exception:
        return ()
    doorway_objects = []
    for obj in tuple(getattr(merged, 'objects', ()) or ()):
        if not is_door_like(getattr(obj, 'category', ''), getattr(obj, 'name', '')):
            continue
        states = dict(getattr(obj, 'states', {}) or {})
        states['doorway_opening'] = True
        states['doorway_opening_source'] = 'disabled_door_loading'
        doorway_objects.append(type(obj)(name=str(getattr(obj, 'name', '')), category=str(getattr(obj, 'category', 'door') or 'door'), center=tuple((float(value) for value in getattr(obj, 'center'))), extent=tuple((float(value) for value in getattr(obj, 'extent'))), visible=False, abilities=tuple(getattr(obj, 'abilities', ()) or ()), states=states, root_position=getattr(obj, 'root_position', None), root_orientation_xyzw=getattr(obj, 'root_orientation_xyzw', None), root_yaw=getattr(obj, 'root_yaw', None), pose_source=getattr(obj, 'pose_source', None)))
    return tuple(doorway_objects)

def build_symbolic_navigation_context(task_spec: Mapping[str, Any], *, assets_root: str | Path | None=None, occupancy_resolution: float=0.1) -> dict[str, Any]:
    traversability = build_static_scene_traversability(task_spec, assets_root=assets_root, occupancy_resolution=occupancy_resolution)
    if not bool(getattr(traversability, 'enabled', False)):
        raise ValueError('symbolic scene traversability is unavailable')
    return {'scene_traversability': traversability, 'metadata': dict(traversability.metadata or {})}

def evaluate_static_scene_admission(task_spec: Mapping[str, Any], *, rng_seed: int=0, max_high_level_steps: int=120, assets_root: str | Path | None=None) -> dict[str, Any]:
    normalized_assets_root = None if assets_root is None else Path(assets_root).resolve()
    repository_root = normalized_assets_root
    render_assets_root = normalized_assets_root
    if normalized_assets_root is not None and (normalized_assets_root / 'datasets' / 'behavior-1k-assets').is_dir():
        render_assets_root = normalized_assets_root / 'datasets' / 'behavior-1k-assets'
    elif normalized_assets_root is not None and normalized_assets_root.name == 'behavior-1k-assets':
        repository_root = normalized_assets_root.parent.parent
    task = dict(task_spec)
    source = task.get('source') if isinstance(task.get('source'), Mapping) else {}
    scene_id = str(source.get('scene_model') or 'unknown')
    layout_id = source.get('layout_variant_id')
    layout_id = str(layout_id) if layout_id else None
    checks = {name: 'not_run' for name in _CHECK_NAMES}
    diagnostics: dict[str, Any] = {}

    def finish(status: str, failure_code: str | None) -> dict[str, Any]:
        return build_static_scene_admission_result(backend='behavior', task=task, scene_id=scene_id, layout_id_or_null=layout_id, implementation_id=BEHAVIOR_STATIC_SCENE_ADMISSION_IMPLEMENTATION_ID, geometry_source=BEHAVIOR_STATIC_SCENE_GEOMETRY_SOURCE, status=status, failure_code_or_null=failure_code, checks=checks, diagnostics=json_safe(diagnostics))
    try:
        validate_behavior_task_spec(task)
        checks['task_schema'] = 'pass'
    except Exception as exc:
        checks['task_schema'] = 'fail'
        diagnostics['task_schema_error'] = _error_record(exc)
        return finish('rejected', 'STATIC_TASK_SCHEMA_INVALID')
    source_json = source.get('source_json')
    source_path = _resolve_source_path(source_json)
    if not scene_id or scene_id == 'unknown' or source_path is None or (not source_path.is_file()):
        checks['source_layout'] = 'fail'
        diagnostics['source_layout'] = {'scene_model': scene_id, 'source_json': None if source_path is None else str(source_path), 'exists': bool(source_path and source_path.is_file())}
        return finish('rejected', 'STATIC_SOURCE_LAYOUT_REQUIRED')
    checks['source_layout'] = 'pass'
    diagnostics['source_layout'] = {'scene_model': scene_id, 'source_json': str(source_path), 'layout_variant_id': layout_id}
    try:
        asset_result = require_task_render_assets(task, assets_root=render_assets_root, scene_layout_mode='source_json_full', load_doors=False)
    except AssetPreflightError as exc:
        checks['render_assets'] = 'fail'
        diagnostics['render_assets'] = _asset_preflight_summary(exc.result.provenance)
        return finish('rejected', 'STATIC_RENDER_ASSET_UNAVAILABLE')
    except Exception as exc:
        checks['render_assets'] = 'fail'
        diagnostics['render_assets_error'] = _error_record(exc)
        return finish('invalid', 'STATIC_ASSET_CATALOG_UNAVAILABLE')
    resolved_task = dict(asset_result.task)
    checks['render_assets'] = 'pass'
    diagnostics['render_assets'] = _asset_preflight_summary(asset_result.provenance)
    try:
        render_scene = build_scene(resolved_task, mode='render', rng_seed=int(rng_seed), scene_layout_mode='source_json_full', load_doors=False)
    except Exception as exc:
        checks['layout_merge'] = 'fail'
        diagnostics['layout_merge_error'] = _error_record(exc)
        return finish('rejected', 'STATIC_RENDER_LAYOUT_INVALID')
    render_report = dict(render_scene.render_report or {})
    unresolved_aliases = list(render_report.get('unresolved_aliases') or [])
    if unresolved_aliases:
        checks['layout_merge'] = 'fail'
        diagnostics['unresolved_aliases'] = unresolved_aliases
        return finish('rejected', 'STATIC_RENDER_ALIAS_UNRESOLVED')
    checks['layout_merge'] = 'pass'
    diagnostics['render_layout'] = _render_layout_summary(render_report)
    collision_report = dict((render_scene.symbolic.placement_report or {}).get('post_merge_collision_repairs') or {})
    collision_failure = _collision_failure_code(collision_report)
    diagnostics['spawn_collisions'] = collision_report
    if collision_failure is not None:
        checks['spawn_collisions'] = 'fail'
        return finish('rejected', collision_failure)
    checks['spawn_collisions'] = 'pass'
    try:
        stage5 = validate_stage5_reset_mutation(resolved_task, rng_seed=int(rng_seed), run_reachability=False, scene_layout_mode='source_json_full', load_doors=False)
    except Exception as exc:
        checks['stage5_semantics'] = 'fail'
        diagnostics['stage5_error'] = _error_record(exc)
        return finish('invalid', 'STATIC_STAGE5_EVALUATOR_FAILED')
    diagnostics['stage5'] = _stage5_summary(stage5)
    if not bool(stage5.get('stage5_reset_mutation_ok')):
        checks['stage5_semantics'] = 'fail'
        return finish('rejected', 'STATIC_STAGE5_REJECTED')
    checks['stage5_semantics'] = 'pass'
    try:
        traversability = build_static_scene_traversability(resolved_task, assets_root=render_assets_root, occupancy_resolution=0.1)
        env = BehaviorClosedLoopToolEnv(max_high_level_steps=int(max_high_level_steps), use_rendered_visibility=False, scene_traversability=traversability, occupancy_agent_radius=STAGE9_RENDER_OCCUPANCY_AGENT_RADIUS, occupancy_safety_margin=STAGE9_RENDER_OCCUPANCY_SAFETY_MARGIN, scene_layout_mode='source_json_full', action_visibility_mode='disabled', load_doors=False, allow_navigation_candidate_fallbacks=False)
        env.reset_to_task(resolved_task, rng_seed=int(rng_seed))
    except Exception as exc:
        checks['start_pose'] = 'fail'
        diagnostics['static_environment_error'] = _error_record(exc)
        return finish('invalid', 'STATIC_LAYOUT_DATA_UNAVAILABLE')
    occupancy = getattr(getattr(env, 'core', None), 'occupancy', None)
    start_pose = getattr(getattr(env, 'core', None), 'ego_state', None)
    start_pose = getattr(start_pose, 'pose', None)
    if occupancy is None or start_pose is None:
        checks['start_pose'] = 'fail'
        diagnostics['start_pose'] = {'reason': 'occupancy_or_pose_unavailable'}
        return finish('invalid', 'STATIC_OCCUPANCY_UNAVAILABLE')
    start_free = bool(occupancy.is_free(start_pose.xy))
    diagnostics['navigation_map'] = {**dict(traversability.metadata or {}), 'component_count': int(occupancy.metadata.get('component_count') or 0)}
    diagnostics['start_pose'] = {'pose': start_pose.as_dict(), 'free': start_free, 'component': int(occupancy.component_at(start_pose.xy)), 'repair': dict(occupancy.metadata.get('start_pose_repair') or {}), 'room_repair': dict(occupancy.metadata.get('start_pose_room_repair') or {})}
    if not start_free:
        checks['start_pose'] = 'fail'
        return finish('rejected', 'STATIC_START_POSE_OCCUPIED')
    checks['start_pose'] = 'pass'
    reachability = validate_reachability(env, resolved_task)
    diagnostics['required_roles'] = reachability
    if not bool(reachability.get('ok')):
        checks['required_roles'] = 'fail'
        return finish('rejected', 'STATIC_REQUIRED_ROLE_UNREACHABLE')
    checks['required_roles'] = 'pass'
    try:
        delegate, teacher_source = default_dispatcher().select(resolved_task)
        brain = _QAAnsweringOracle(delegate, env)
        episode = EpisodeRunner(env, max_high_level_steps=int(max_high_level_steps)).run(resolved_task, brain=brain, rng_seed=int(rng_seed), policy_source='oracle')
    except Exception as exc:
        checks['oracle_rollout'] = 'fail'
        diagnostics['oracle_error'] = _error_record(exc)
        return finish('invalid', 'STATIC_ORACLE_EVALUATOR_FAILED')
    metrics = dict(episode.metrics or {})
    has_qa_overlay = bool(isinstance(resolved_task.get('qa_overlay'), Mapping) and (resolved_task.get('qa_overlay') or {}).get('questions'))
    base_value = metrics.get('base_task_success', episode.final_goal.get('task_success'))
    base_success = float(base_value or 0.0) == 1.0
    qa_success = int(metrics.get('qa_success') or 0) == 1 if has_qa_overlay else True
    combined_success = int(metrics.get('combined_success') or 0) == 1 if has_qa_overlay else base_success
    diagnostics['oracle'] = {'teacher_source': teacher_source, 'episode_status': episode.status, 'failure_type': episode.failure_type, 'failure_reason': episode.failure_reason, 'failure_code': episode.failure_code, 'transition_count': len(episode.transitions), 'tool_sequence': [str(item.get('tool_name') or '') for item in episode.transitions], 'final_goal': dict(episode.final_goal or {}), 'base_task_success': base_value, 'qa_success': metrics.get('qa_success') if has_qa_overlay else 1, 'combined_success': metrics.get('combined_success') if has_qa_overlay else int(base_success), 'num_qa_total': metrics.get('num_qa_total'), 'num_qa_answered': metrics.get('num_qa_answered'), 'num_qa_correct': metrics.get('num_qa_correct')}
    if episode.status == 'invalid':
        checks['oracle_rollout'] = 'fail'
        return finish('invalid', 'STATIC_ORACLE_RUNTIME_INVALID')
    if episode.status != 'passed' or not base_success:
        checks['oracle_rollout'] = 'fail'
        return finish('rejected', 'STATIC_ORACLE_TASK_FAILED')
    checks['oracle_rollout'] = 'pass'
    checks['qa'] = 'pass' if qa_success else 'fail'
    if not qa_success:
        return finish('rejected', 'STATIC_QA_FAILED')
    checks['combined'] = 'pass' if combined_success else 'fail'
    if not combined_success:
        return finish('rejected', 'STATIC_COMBINED_TASK_FAILED')
    return finish('admitted', None)

def _resize_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
    if image.size != (int(target_size), int(target_size)):
        image = image.resize((int(target_size), int(target_size)), Image.Resampling.NEAREST)
    return np.asarray(image) > 0

def _resize_traversability_like_omnigibson(values: np.ndarray, target_size: int) -> np.ndarray:
    resized = cv2.resize(np.asarray(values, dtype=np.uint8), (int(target_size), int(target_size)))
    return np.asarray(resized) >= 255

def _resolve_source_path(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else repo_root() / path

def _collision_failure_code(report: Mapping[str, Any]) -> str | None:
    if int(report.get('unresolved_collision_count') or 0) > 0:
        return 'STATIC_SPAWN_COLLISION_UNRESOLVED'
    repairs = [item for item in report.get('repairs') or [] if isinstance(item, Mapping)]
    if any((item.get('action') == 'remove_symbolic_background' for item in repairs)):
        return 'STATIC_SYMBOLIC_BACKGROUND_REMOVAL_UNSAFE'
    return None

def _render_layout_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {'render_layout_mode': report.get('render_layout_mode'), 'render_object_count': report.get('render_object_count'), 'render_aliases': dict(report.get('render_aliases') or {}), 'unresolved_aliases': list(report.get('unresolved_aliases') or [])}

def _asset_preflight_summary(provenance: Mapping[str, Any]) -> dict[str, Any]:
    roles = [item for item in provenance.get('roles') or [] if isinstance(item, Mapping)]
    return {'schema_version': provenance.get('schema_version'), 'status': provenance.get('status'), 'failure_code': provenance.get('failure_code'), 'asset_catalog_revision': provenance.get('asset_catalog_revision'), 'asset_catalog_hash': provenance.get('asset_catalog_hash'), 'role_count': len(roles), 'passed_role_count': sum((item.get('status') in {'pass', 'exempt'} for item in roles)), 'failures': list(provenance.get('failures') or [])}

def _stage5_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {'stage5_reset_mutation_ok': result.get('stage5_reset_mutation_ok'), 'stage5_bucket': result.get('stage5_bucket'), 'failure_code': result.get('failure_code'), 'deterministic_reset': result.get('deterministic_reset'), 'init_ok': result.get('init_ok'), 'initial_goal_false': result.get('initial_goal_false'), 'geometry_ok': result.get('geometry_ok'), 'mutation_spec_ok': result.get('mutation_spec_ok'), 'mutation_replay_ok': result.get('mutation_replay_ok'), 'deferred_codes': list(result.get('deferred_codes') or [])}

def _error_record(exc: BaseException) -> dict[str, str]:
    return {'error_type': type(exc).__name__, 'error_message': str(exc)}
__all__ = ['BEHAVIOR_STATIC_SCENE_ADMISSION_IMPLEMENTATION_ID', 'BEHAVIOR_STATIC_SCENE_GEOMETRY_SOURCE', 'build_symbolic_navigation_context', 'build_static_scene_traversability', 'evaluate_static_scene_admission']
