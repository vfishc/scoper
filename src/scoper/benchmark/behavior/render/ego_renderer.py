from __future__ import annotations
from scoper.resources import behavior_assets_root
import atexit
import copy
import hashlib
import json
import math
import os
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, List, Mapping, Optional, Sequence
import numpy as np
from scoper.benchmark.behavior.render.ego_arm_visual import EgoArmVisualConfig, EgoArmVisualSynchronizer
from scoper.benchmark.behavior.render.gripper_visual import GripperAnchor, GripperVisualConfig, GripperVisualSynchronizer, held_pose_at_gripper
from scoper.benchmark.behavior.runtime_timing import record_render, runtime_phase
_FAST_VISIBILITY_ANNOTATOR = 'instance_id_segmentation_fast'
_FAST_VISIBILITY_SHUTDOWN_REGISTERED = False
_RENDER_PROFILE_ENV = 'EB_BEHAVIOR_RENDER_PROFILE'
_RENDERS_PER_FRAME_ENV = 'EB_BEHAVIOR_RENDERS_PER_FRAME'
_WATER_SHUTDOWN_RENDER_STEPS_ENV = 'EB_BEHAVIOR_WATER_SHUTDOWN_RENDER_STEPS'
_DEFAULT_RENDER_PROFILE = {'name': 'omnigibson_default'}
_A100_SAFE_RENDER_PROFILE = {'name': 'a100_safe', 'rendermode': 'RaytracedLighting', 'anti_aliasing': 'FXAA', 'anti_aliasing_op': 2, 'dlss_frame_generation': False, 'rt2': False}
_RTX_HQ_64_RENDER_PROFILE = {'name': 'rtx_hq_64', 'rendermode': 'RealTimePathTracing', 'anti_aliasing': 'RTXAA', 'anti_aliasing_op': 4, 'pathtracing_spp': 64, 'optix_denoiser': True, 'dlss_frame_generation': False, 'rt2': True}

def render_profile_from_env(env: Mapping[str, str] | None=None) -> dict[str, Any]:
    values = os.environ if env is None else env
    raw = str(values.get(_RENDER_PROFILE_ENV, '') or '').strip().lower()
    if raw in {'', 'default', 'omnigibson_default'}:
        return dict(_DEFAULT_RENDER_PROFILE)
    if raw == 'a100_safe':
        return dict(_A100_SAFE_RENDER_PROFILE)
    if raw == 'rtx_hq_64':
        return dict(_RTX_HQ_64_RENDER_PROFILE)
    raise ValueError(f'{_RENDER_PROFILE_ENV} must be default, a100_safe, or rtx_hq_64')

def render_timing_from_env(env: Mapping[str, str] | None=None) -> dict[str, int]:
    values = os.environ if env is None else env

    def positive_int(name: str, default: int) -> int:
        raw = str(values.get(name, '') or '').strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f'{name} must be a positive integer') from exc
        if value < 1:
            raise ValueError(f'{name} must be a positive integer')
        return value
    return {'renders_per_frame': positive_int(_RENDERS_PER_FRAME_ENV, 2), 'water_shutdown_render_steps': positive_int(_WATER_SHUTDOWN_RENDER_STEPS_ENV, 50)}

def _register_fast_visibility_shutdown(og_module: Any) -> None:
    global _FAST_VISIBILITY_SHUTDOWN_REGISTERED
    if _FAST_VISIBILITY_SHUTDOWN_REGISTERED:
        return

    def _shutdown() -> None:
        app = getattr(og_module, 'app', None)
        if app is None:
            return
        try:
            is_running = getattr(app, 'is_running', None)
            if callable(is_running) and (not is_running()):
                return
            og_module.shutdown()
        except Exception:
            return
    atexit.register(_shutdown)
    _FAST_VISIBILITY_SHUTDOWN_REGISTERED = True

class EgoRendererError(RuntimeError):
    pass

class EgoRenderer:

    def __init__(self, env_config: Mapping[str, Any], og_objects: List[dict], renders_per_frame: int=2, render_aliases: Mapping[str, str] | None=None, auto_exposure: bool=True, hide_scene_structure_categories: Sequence[str]=(), reset_diagnostics_path: str | Path | None=None, render_alias_metadata: Mapping[str, Mapping[str, Any]] | None=None, gripper_visual_config: Mapping[str, Any] | GripperVisualConfig | None=None, ego_arm_visual_config: Mapping[str, Any] | EgoArmVisualConfig | None=None, render_profile: Mapping[str, Any] | None=None, particle_systems_enabled: bool=True, native_particle_source_visuals_enabled: bool | None=False, particle_source_stream_visuals_enabled: bool=True, water_shutdown_render_steps: int | None=None):
        self.env_config = copy.deepcopy(dict(env_config))
        self.reset_diagnostics_path = Path(reset_diagnostics_path) if reset_diagnostics_path else None
        self._reset_t0: float | None = None
        self._desired_aabb_centers: dict[str, list[float]] = {}
        self._object_aliases = {_og_safe_name(str(alias)): _og_safe_name(str(target)) for alias, target in dict(render_aliases or {}).items() if alias and target}
        self._render_alias_metadata = self._normalize_render_alias_metadata(render_alias_metadata)
        for alias_safe, target_safe in self._object_aliases.items():
            self._render_alias_metadata.setdefault(str(alias_safe), self._infer_render_alias_metadata(str(alias_safe), str(target_safe)))
        self._raw_name_to_safe_name: dict[str, str] = {}
        for alias, target in dict(render_aliases or {}).items():
            if alias:
                self._raw_name_to_safe_name[str(alias)] = _og_safe_name(str(alias))
            if target:
                self._raw_name_to_safe_name[str(target)] = _og_safe_name(str(target))
        for alias_safe, metadata in self._render_alias_metadata.items():
            alias_safe = str(alias_safe)
            for raw_key in ('alias', 'core_name'):
                raw_name = metadata.get(raw_key)
                if raw_name:
                    self._raw_name_to_safe_name.setdefault(str(raw_name), alias_safe)
            render_name = metadata.get('render_name')
            if render_name:
                render_safe = _og_safe_name(str(render_name))
                for raw_key in ('parent_object', 'source_scene_object_id'):
                    raw_name = metadata.get(raw_key)
                    if raw_name:
                        self._raw_name_to_safe_name.setdefault(str(raw_name), render_safe)
        cleaned_objects = []
        for obj in og_objects:
            obj = dict(obj)
            desired = obj.pop('_desired_aabb_center', None)
            symbolic_name = obj.pop('_symbolic_name', None)
            name = str(obj.get('name', ''))
            if symbolic_name and name:
                self._raw_name_to_safe_name[str(symbolic_name)] = _og_safe_name(name)
            if desired is not None and name:
                self._desired_aabb_centers[name] = [float(v) for v in desired]
            cleaned_objects.append(obj)
        self.env_config['objects'] = cleaned_objects
        self._ego_arm_visual_sync: EgoArmVisualSynchronizer | None = None
        self._last_ego_arm_visual_sync: dict[str, Any] = {}
        arm_config = EgoArmVisualConfig.from_value(ego_arm_visual_config)
        if arm_config.enabled:
            self._ego_arm_visual_sync = EgoArmVisualSynchronizer(arm_config)
            self._ego_arm_visual_sync.inject_env_config(self.env_config)
        render_timing = render_timing_from_env()
        env_renders_per_frame = os.environ.get(_RENDERS_PER_FRAME_ENV)
        self.renders_per_frame = int(render_timing['renders_per_frame'] if env_renders_per_frame else renders_per_frame)
        self.camera_settle_renders = int(os.environ.get('EB_BEHAVIOR_CAMERA_SETTLE_RENDERS', '0'))
        if not 0 <= self.camera_settle_renders <= 8:
            raise ValueError('EB_BEHAVIOR_CAMERA_SETTLE_RENDERS must be between 0 and 8')
        env_water_shutdown_steps = os.environ.get(_WATER_SHUTDOWN_RENDER_STEPS_ENV)
        self.water_shutdown_render_steps = int(render_timing['water_shutdown_render_steps'] if env_water_shutdown_steps else water_shutdown_render_steps if water_shutdown_render_steps is not None else 50)
        self.render_profile = dict(render_profile or render_profile_from_env())
        self.particle_systems_enabled = bool(particle_systems_enabled)
        self.native_particle_source_visuals_enabled = bool(native_particle_source_visuals_enabled)
        if not self.particle_systems_enabled:
            self.native_particle_source_visuals_enabled = False
        self.particle_source_stream_visuals_enabled = bool(particle_source_stream_visuals_enabled)
        self._preserve_simulator_for_process_exit = False
        self.auto_exposure = bool(auto_exposure)
        self.hide_scene_structure_categories = {str(category).strip().lower() for category in hide_scene_structure_categories if str(category).strip()}
        self._og = None
        self._env = None
        self._camera = None
        self._visibility_annotator = None
        self._visibility_render_product = None
        self._last_visibility_evidence: dict[str, Any] | None = None
        self._particle_suppression_report: dict[str, Any] = {'enabled': not self.particle_systems_enabled, 'objects': [], 'removed_state_count': 0, 'removed_ability_count': 0}
        self._object_offsets: dict[str, np.ndarray] = {}
        self._object_offset_orientations: dict[str, tuple[float, float, float, float] | None] = {}
        self._object_pose_catalog: dict[str, dict[str, Any]] = {}
        self._visual_replacements: dict[str, list[str]] = {}
        self._visual_replacement_offsets: dict[str, dict[str, np.ndarray]] = {}
        self._visual_sync = None
        self._gripper_visual_sync: GripperVisualSynchronizer | None = None
        self._last_gripper_visual_sync: dict[str, Any] = {}
        gripper_config = GripperVisualConfig.from_value(gripper_visual_config)
        self._gripper_visual_config = gripper_config
        if gripper_config.enabled:
            self._gripper_visual_sync = GripperVisualSynchronizer(gripper_config)
        self._container_interiors: dict[str, dict[str, list[float]]] = {}
        self._hidden_scene_structures: list[dict[str, str]] = []

    def _normalize_render_alias_metadata(self, render_alias_metadata: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for alias, metadata in dict(render_alias_metadata or {}).items():
            if not alias or not isinstance(metadata, Mapping):
                continue
            alias_safe = _og_safe_name(str(alias))
            record = dict(metadata)
            record.setdefault('alias', alias_safe)
            if record.get('render_name'):
                record['render_name'] = _og_safe_name(str(record['render_name']))
            normalized[alias_safe] = record
        return normalized

    def _infer_render_alias_metadata(self, alias_safe: str, target_safe: str) -> dict[str, Any]:
        alias_text = str(alias_safe)
        target_text = str(target_safe)
        alias_kind = 'derived_parent' if 'faucet' in alias_text.lower() else 'source_scene_object'
        pose_allowed = alias_kind != 'derived_parent'
        return {'alias': alias_text, 'alias_kind': alias_kind, 'render_name': target_text, 'pose_sync_allowed': pose_allowed, 'held_sync_allowed': pose_allowed, 'visual_state_sync_allowed': True, 'inferred': True}

    def _alias_metadata_for(self, alias_safe: str, target_safe: str) -> dict[str, Any]:
        metadata = dict(self._render_alias_metadata.get(str(alias_safe)) or {})
        if not metadata:
            metadata = self._infer_render_alias_metadata(str(alias_safe), str(target_safe))
        metadata.setdefault('alias', str(alias_safe))
        metadata.setdefault('render_name', str(target_safe))
        metadata.setdefault('alias_kind', 'source_scene_object')
        return metadata

    def _alias_operation_allowed(self, metadata: Mapping[str, Any], operation: str) -> bool:
        alias_kind = str(metadata.get('alias_kind') or 'source_scene_object')
        allowed_key = {'held': 'held_sync_allowed', 'pose': 'pose_sync_allowed', 'visual_state': 'visual_state_sync_allowed'}.get(str(operation or 'pose'), 'pose_sync_allowed')
        allowed_default = alias_kind != 'derived_parent' or str(operation or 'pose') == 'visual_state'
        return bool(metadata.get(allowed_key, allowed_default))

    def _resolve_alias_chain(self, safe_name: Any) -> dict[str, Any]:
        start_safe = _og_safe_name(str(safe_name))
        chain = [start_safe]
        metadata_chain: list[dict[str, Any]] = []
        current_safe = start_safe
        seen = {current_safe}
        max_hops = max(1, len(self._object_aliases) + 1)
        for _ in range(max_hops):
            target_safe = self._object_aliases.get(current_safe)
            if not target_safe:
                break
            target_safe = _og_safe_name(str(target_safe))
            metadata_chain.append(self._alias_metadata_for(current_safe, target_safe))
            chain.append(target_safe)
            if target_safe == current_safe:
                break
            if target_safe in seen:
                return {'safe_name': start_safe, 'resolved_safe_name': target_safe, 'alias_chain': chain, 'alias_metadata_chain': metadata_chain, 'has_alias': True, 'alias_cycle': True, 'cycle_at': target_safe}
            seen.add(target_safe)
            current_safe = target_safe
        return {'safe_name': start_safe, 'resolved_safe_name': current_safe, 'alias_chain': chain, 'alias_metadata_chain': metadata_chain, 'has_alias': len(chain) > 1, 'alias_cycle': False}

    def _alias_chain_denial(self, chain_info: Mapping[str, Any], operation: str) -> dict[str, Any] | None:
        chain = list(chain_info.get('alias_chain') or [])
        for idx, metadata in enumerate(chain_info.get('alias_metadata_chain') or []):
            if self._alias_operation_allowed(metadata, operation):
                continue
            alias_safe = str(chain[idx]) if idx < len(chain) else str(metadata.get('alias') or '')
            target_safe = str(chain[idx + 1]) if idx + 1 < len(chain) else str(metadata.get('render_name') or '')
            return {'alias': alias_safe, 'target': target_safe, 'alias_kind': str(metadata.get('alias_kind') or 'source_scene_object'), 'metadata': dict(metadata)}
        return None

    def _resolve_alias_target_safe(self, safe_name: Any) -> str:
        chain_info = self._resolve_alias_chain(safe_name)
        if bool(chain_info.get('alias_cycle')):
            return str(chain_info.get('safe_name') or _og_safe_name(str(safe_name)))
        return str(chain_info.get('resolved_safe_name') or _og_safe_name(str(safe_name)))

    def resolve_render_object_name(self, object_name: Any, *, operation: str='pose') -> dict[str, Any]:
        text = '' if object_name is None else str(object_name)
        operation = str(operation or 'pose')
        safe_name = _og_safe_name(text)
        diagnostic: dict[str, Any] = {'requested_name': text, 'safe_name': safe_name, 'operation': operation, 'resolved_name': text, 'resolved_safe_name': safe_name, 'resolution': 'identity', 'success': bool(text), 'reason': 'identity' if text else 'missing_object_name'}
        if not text:
            return diagnostic
        if text.startswith('spawnable:'):
            diagnostic.update({'success': False, 'resolution': 'spawn_logical_id', 'reason': 'spawn_logical_id_not_renderable'})
            return diagnostic
        if operation in {'pose', 'held'} and self._replacement_names_for_object(text):
            diagnostic.update({'resolution': 'visual_replacement', 'reason': 'visual_replacement', 'success': True, 'has_visual_replacements': True})
            return diagnostic
        alias_chain = self._resolve_alias_chain(safe_name)
        if bool(alias_chain.get('has_alias')):
            metadata_chain = [dict(item) for item in alias_chain.get('alias_metadata_chain') or []]
            first_metadata = dict(metadata_chain[0]) if metadata_chain else {}
            alias_kind = str(first_metadata.get('alias_kind') or 'source_scene_object')
            direct_target = str(alias_chain.get('alias_chain', [safe_name, safe_name])[1])
            resolved_target = str(alias_chain.get('resolved_safe_name') or direct_target)
            diagnostic.update({'resolution': 'renderer_alias', 'alias_kind': alias_kind, 'target_safe_name': direct_target, 'resolved_target_safe_name': resolved_target, 'alias_chain': list(alias_chain.get('alias_chain') or []), 'alias_metadata': first_metadata, 'alias_metadata_chain': metadata_chain})
            if bool(alias_chain.get('alias_cycle')):
                diagnostic.update({'success': False, 'reason': 'alias_cycle', 'resolution': 'renderer_alias_cycle'})
                return diagnostic
            denial = self._alias_chain_denial(alias_chain, operation)
            if denial is not None:
                diagnostic.update({'success': False, 'reason': 'aliased_object', 'resolution': 'renderer_alias_not_syncable', 'denied_alias': denial})
                return diagnostic
            diagnostic.update({'success': True, 'reason': 'renderer_alias', 'resolved_name': resolved_target, 'resolved_safe_name': resolved_target})
            return diagnostic
        mapped_safe = self._raw_name_to_safe_name.get(text)
        if mapped_safe:
            mapped_safe = str(mapped_safe)
            mapped_chain = self._resolve_alias_chain(mapped_safe)
            target_safe = str(mapped_chain.get('resolved_safe_name') or mapped_safe)
            if bool(mapped_chain.get('alias_cycle')):
                diagnostic.update({'resolution': 'raw_name_mapping_alias_cycle', 'reason': 'alias_cycle', 'success': False, 'mapped_safe_name': mapped_safe, 'alias_chain': list(mapped_chain.get('alias_chain') or [])})
                return diagnostic
            denial = self._alias_chain_denial(mapped_chain, operation)
            if denial is not None:
                diagnostic.update({'resolution': 'raw_name_mapping_alias_not_syncable', 'reason': 'aliased_object', 'success': False, 'mapped_safe_name': mapped_safe, 'alias_chain': list(mapped_chain.get('alias_chain') or []), 'denied_alias': denial})
                return diagnostic
            if target_safe != safe_name:
                diagnostic.update({'resolution': 'raw_name_mapping', 'reason': 'raw_name_mapping', 'success': True, 'mapped_safe_name': mapped_safe, 'alias_chain': list(mapped_chain.get('alias_chain') or []), 'resolved_name': str(target_safe), 'resolved_safe_name': str(target_safe)})
            return diagnostic
        return diagnostic

    def _resolve_pose_target_or_diagnostic(self, object_name: Any, *, operation: str, base_diagnostic: dict[str, Any]) -> tuple[str | None, str | None, dict[str, Any]]:
        resolution = self.resolve_render_object_name(object_name, operation=operation)
        base_diagnostic['object_resolution'] = resolution
        if not bool(resolution.get('success')):
            reason = str(resolution.get('reason') or 'object_not_found')
            base_diagnostic['reason'] = 'object_not_found' if reason == 'spawn_logical_id_not_renderable' else reason
            if resolution.get('target_safe_name'):
                base_diagnostic['target_safe_name'] = resolution.get('target_safe_name')
            return (None, None, resolution)
        resolved_name = str(resolution.get('resolved_name') or object_name)
        resolved_safe = str(resolution.get('resolved_safe_name') or _og_safe_name(resolved_name))
        if resolved_name != str(object_name):
            base_diagnostic['requested_object_name'] = str(object_name)
            base_diagnostic['object_name'] = resolved_name
        base_diagnostic['safe_name'] = resolved_safe
        return (resolved_name, resolved_safe, resolution)

    def reset(self) -> 'EgoRenderer':
        self._reset_t0 = time.monotonic()
        self._object_offsets.clear()
        self._object_offset_orientations.clear()
        self._object_pose_catalog.clear()
        self._record_reset_event('reset', 'start', scene_model=(self.env_config.get('scene') or {}).get('scene_model'), scene_type=(self.env_config.get('scene') or {}).get('type'), scene_file=(self.env_config.get('scene') or {}).get('scene_file'), requested_object_count=len(self.env_config.get('objects') or []), renders_per_frame=self.renders_per_frame)
        try:
            with self._reset_phase('import_omnigibson'):
                import omnigibson as og
                from omnigibson.macros import gm
                from scoper.benchmark.behavior.execution_core.env import make_omnigibson_environment
            with self._reset_phase('configure_macros'):
                self._configure_og_macros(gm)
            self._og = og
            if not self.particle_systems_enabled:
                with self._reset_phase('prepare_particle_suppressed_scene'):
                    self._prepare_particle_suppressed_scene()
            with self._reset_phase('make_environment'):
                self._env = make_omnigibson_environment(self.env_config)
            with self._reset_phase('configure_renderer_profile'):
                self._configure_renderer_profile()
            with self._reset_phase('env_reset'):
                self._env.reset()
            with self._reset_phase('sim_play'):
                try:
                    self._og.sim.play()
                except Exception as exc:
                    self._record_reset_event('sim_play', 'ignored_error', error=repr(exc))
            with self._reset_phase('hide_scene_structures'):
                self._hidden_scene_structures = self._hide_scene_structures()
            with self._reset_phase('align_loaded_objects'):
                self._align_loaded_objects_to_desired_centers()
            with self._reset_phase('init_visual_state_sync'):
                self._init_visual_state_sync()
            with self._reset_phase('init_ego_arm_visual'):
                self._init_ego_arm_visual()
            with self._reset_phase('capture_container_interiors'):
                self._container_interiors = self._capture_container_interiors()
            for idx in range(4):
                with self._reset_phase(f'warmup_render_{idx}'):
                    self._og.sim.render()
                    record_render('renderer_reset_warmup', sim_render_calls=1)
            with self._reset_phase('ego_camera'):
                self._camera = self._ego_camera()
        except Exception as exc:
            self._record_reset_event('reset', 'error', error=repr(exc))
            raise
        self._record_reset_event('reset', 'end')
        return self

    def _configure_og_macros(self, gm: Any) -> None:
        if not bool(gm.HEADLESS):
            with gm.unlocked():
                gm.HEADLESS = True
        with gm.unlocked():
            gm.USE_GPU_DYNAMICS = True
            gm.ENABLE_OBJECT_STATES = True
            gm.ENABLE_TRANSITION_RULES = False

    def _prepare_particle_suppressed_scene(self) -> None:
        scene = self.env_config.get('scene') or {}
        scene_file = scene.get('scene_file') if isinstance(scene, Mapping) else None
        if not scene_file:
            self._particle_suppression_report = {'enabled': True, 'mode': 'explicit_object_config_only', 'source_scene_file': None, 'removed_ability_count': 0, 'removed_initial_systems': [], 'objects': []}
            return
        from omnigibson.utils.bddl_utils import get_knowledge_base
        from scoper.benchmark.behavior.render.particle_segmentation_replay import particle_suppressed_scene_file
        knowledge_base = get_knowledge_base()

        def resolve_abilities(category: str) -> Mapping[str, Any] | None:
            knowledge_category = knowledge_base.get_category(category)
            synset = getattr(knowledge_category, 'synset', None) if knowledge_category is not None else None
            abilities = getattr(synset, 'abilities', None) if synset is not None else None
            return abilities if isinstance(abilities, Mapping) else None
        sanitized, report = particle_suppressed_scene_file(scene_file, ability_resolver=resolve_abilities)
        self.env_config['scene']['scene_file'] = str(sanitized)
        self._particle_suppression_report = dict(report)

    def _configure_renderer_profile(self, lazy_module: Any | None=None) -> None:
        if self.render_profile['name'] == 'omnigibson_default':
            return
        if self.render_profile['name'] not in {'a100_safe', 'rtx_hq_64'}:
            raise EgoRendererError(f"Unsupported render profile: {self.render_profile['name']}")
        if lazy_module is None:
            import omnigibson.lazy as lazy_module
        settings = lazy_module.carb.settings.get_settings()
        settings.set_bool('/rtx/rtx/modes/rt/enabled', True)
        settings.set_bool('/rtx/rtx/modes/rt2/enabled', bool(self.render_profile['rt2']))
        settings.set('/rtx/rendermode', self.render_profile['rendermode'])
        settings.set_int('/rtx/post/aa/op', int(self.render_profile['anti_aliasing_op']))
        settings.set_bool('/rtx-transient/dlssg/enabled', False)
        if self.render_profile['name'] == 'rtx_hq_64':
            spp = int(self.render_profile['pathtracing_spp'])
            settings.set_int('/rtx/pathtracing/spp', spp)
            settings.set_int('/rtx/pathtracing/totalSpp', spp)
            settings.set_int('/rtx/pathtracing/clampSpp', spp)
        settings.set_bool('/rtx/pathtracing/optixDenoiser/enabled', bool(self.render_profile.get('optix_denoiser', False)))

    def _record_reset_event(self, phase: str, event: str, **payload: Any) -> None:
        if self.reset_diagnostics_path is None:
            return
        now = time.monotonic()
        record = {'phase': str(phase), 'event': str(event), 'elapsed_seconds': None if self._reset_t0 is None else round(now - self._reset_t0, 6), **payload}
        path = self.reset_diagnostics_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
            f.write('\n')

    @contextmanager
    def _reset_phase(self, phase: str) -> Iterator[None]:
        start = time.monotonic()
        self._record_reset_event(phase, 'start')
        try:
            with runtime_phase(f'renderer_reset.{phase}'):
                yield
        except Exception as exc:
            self._record_reset_event(phase, 'error', duration_seconds=round(time.monotonic() - start, 6), error=repr(exc))
            raise
        self._record_reset_event(phase, 'end', duration_seconds=round(time.monotonic() - start, 6))

    def _ego_camera(self):
        sensors = getattr(self._env, 'external_sensors', None)
        if not sensors or 'ego_camera' not in sensors:
            raise EgoRendererError("env has no 'ego_camera' external sensor.")
        return sensors['ego_camera']

    def _init_ego_arm_visual(self) -> None:
        if self._ego_arm_visual_sync is None or self._env is None:
            return
        self._last_ego_arm_visual_sync = _to_jsonable_value(self._ego_arm_visual_sync.initialize(self._env))

    def _sync_ego_arm_visual_to_pose(self, ego_pose: Any) -> None:
        if self._ego_arm_visual_sync is None:
            return
        try:
            self._last_ego_arm_visual_sync = _to_jsonable_value(self._ego_arm_visual_sync.sync_to_ego_pose(ego_pose))
        except Exception as exc:
            self._last_ego_arm_visual_sync = {'enabled': True, 'success': False, 'reason': 'ego_arm_visual_sync_error', 'error': repr(exc)}

    def render_at_pose(self, ego_pose: Any, renders: Optional[int]=None) -> np.ndarray:
        if self._camera is None:
            raise EgoRendererError('Call reset() before render_at_pose().')
        from scoper.benchmark.behavior.execution_core.render_utils import render_rgb, set_camera_pose
        self._sync_ego_arm_visual_to_pose(ego_pose)
        set_camera_pose(self._camera, _camera_pose_object(ego_pose))
        flush_renders = max(2, int(renders if renders is not None else self.renders_per_frame))
        settle = getattr(self, 'camera_settle_renders', 0)
        if settle:
            render_rgb(self._og, self._camera, renders=flush_renders)
        rgb = _auto_expose_rgb(render_rgb(self._og, self._camera, renders=settle or flush_renders), enabled=self.auto_exposure)
        record_render('rgb', sim_render_calls=flush_renders + settle, rgb=2 if settle else 1)
        return rgb

    def advance_temporal_history_at_pose(self, ego_pose: Any, *, renders: int=1) -> None:
        if self._camera is None:
            raise EgoRendererError('Call reset() before advance_temporal_history_at_pose().')
        render_count = int(renders)
        if render_count <= 0:
            return
        from scoper.benchmark.behavior.execution_core.render_utils import set_camera_pose
        self._sync_ego_arm_visual_to_pose(ego_pose)
        set_camera_pose(self._camera, _camera_pose_object(ego_pose))
        for _ in range(render_count):
            self._og.sim.render()
        record_render('stream_temporal_settle', sim_render_calls=render_count, rgb=0, segmentation=0)

    def render_visibility_at_pose(self, ego_pose: Any, renders: Optional[int]=None, *, modality: str='seg_instance', min_pixels: int=8, object_names: Optional[Sequence[str]]=None) -> dict[str, Any]:
        if self._camera is None:
            raise EgoRendererError('Call reset() before render_visibility_at_pose().')
        modality = str(modality or 'seg_instance')
        if modality not in {'seg_instance', 'seg_instance_id'}:
            raise EgoRendererError("Unsupported visibility modality '{}'. Use seg_instance or seg_instance_id.".format(modality))
        from scoper.benchmark.behavior.execution_core.render_utils import set_camera_pose
        self._ensure_visibility_modality(modality)
        self._sync_ego_arm_visual_to_pose(ego_pose)
        set_camera_pose(self._camera, _camera_pose_object(ego_pose))
        flush_renders = max(2, int(renders if renders is not None else self.renders_per_frame))
        for _ in range(flush_renders):
            self._og.sim.render()
        settle = getattr(self, 'camera_settle_renders', 0)
        if settle:
            self._fast_visibility_data()
            for _ in range(settle):
                self._og.sim.render()
        record_render('segmentation', sim_render_calls=flush_renders + settle, segmentation=2 if settle else 1)
        seg, label_mapping = self._fast_visibility_data()
        return self._visibility_payload(seg, label_mapping, modality=modality, min_pixels=min_pixels, object_names=object_names)

    def capture_observation_at_pose(self, ego_pose: Any, renders: Optional[int]=None, *, modality: str='seg_instance_id', min_pixels: int=8, object_names: Optional[Sequence[str]]=None) -> dict[str, Any]:
        if self._camera is None:
            raise EgoRendererError('Call reset() before capture_observation_at_pose().')
        modality = str(modality or 'seg_instance_id')
        if modality not in {'seg_instance', 'seg_instance_id'}:
            raise EgoRendererError("Unsupported visibility modality '{}'. Use seg_instance or seg_instance_id.".format(modality))
        from scoper.benchmark.behavior.execution_core.render_utils import rgb_uint8, set_camera_pose
        self._ensure_visibility_modality(modality)
        self._sync_ego_arm_visual_to_pose(ego_pose)
        set_camera_pose(self._camera, _camera_pose_object(ego_pose))
        flush_renders = max(2, int(renders if renders is not None else self.renders_per_frame))
        for _ in range(flush_renders):
            self._og.sim.render()
        settle = getattr(self, 'camera_settle_renders', 0)
        if settle:
            self._camera.get_obs()
            self._fast_visibility_data()
            for _ in range(settle):
                self._og.sim.render()
        record_render('rgb_segmentation', sim_render_calls=flush_renders + settle, rgb=2 if settle else 1, segmentation=2 if settle else 1)
        camera_obs, _ = self._camera.get_obs()
        if not isinstance(camera_obs, Mapping) or 'rgb' not in camera_obs:
            raise EgoRendererError('Camera observation is missing RGB data.')
        rgb = _auto_expose_rgb(rgb_uint8(camera_obs['rgb']), enabled=self.auto_exposure)
        seg, label_mapping = self._fast_visibility_data()
        visibility = self._visibility_payload(seg, label_mapping, modality=modality, min_pixels=min_pixels, object_names=object_names)
        return {'head_rgb': rgb, 'visibility': visibility}

    def _visibility_payload(self, seg: np.ndarray, label_mapping: Mapping[Any, Any], *, modality: str, min_pixels: int, object_names: Optional[Sequence[str]]) -> dict[str, Any]:
        label_counts: Counter[str] = Counter()
        instance_rows: list[dict[str, Any]] = []
        instance_ids, instance_counts = np.unique(seg.reshape(-1), return_counts=True)
        for label_id, count in zip(instance_ids, instance_counts):
            label = _segmentation_label(label_mapping, int(label_id))
            if _ignored_visibility_label(label):
                continue
            label_counts[str(label)] += int(count)
            instance_rows.append({'instance_id': int(label_id), 'prim_path': str(label), 'pixel_count': int(count), 'object_ids': self._object_names_from_visibility_labels([str(label)], object_names=object_names)})
        threshold = max(1, int(min_pixels))
        visible_labels = sorted((label for label, count in label_counts.items() if count >= threshold))
        visible_object_ids = self._object_names_from_visibility_labels(visible_labels, object_names=object_names)
        object_pixel_counts: Counter[str] = Counter()
        for row in instance_rows:
            if int(row['pixel_count']) < threshold:
                continue
            for object_id in row['object_ids']:
                object_pixel_counts[str(object_id)] += int(row['pixel_count'])
        canonical_seg = np.ascontiguousarray(seg.astype('<u4', copy=False))
        digest = hashlib.sha256()
        digest.update(b'instance-id-segmentation.v1\x00')
        digest.update('{}x{}:<u4\x00'.format(*canonical_seg.shape).encode('ascii'))
        digest.update(canonical_seg.tobytes(order='C'))
        payload = {'mode': 'rendered_segmentation', 'modality': modality, 'segmentation_backend': _FAST_VISIBILITY_ANNOTATOR, 'segmentation_shape': [int(seg.shape[0]), int(seg.shape[1])], 'segmentation_dtype': 'uint32', 'segmentation_sha256': digest.hexdigest(), 'min_pixels': threshold, 'visible_object_ids': visible_object_ids, 'visible_labels': visible_labels, 'pixel_counts_by_label': dict(sorted(label_counts.items())), 'pixel_counts_by_object_id': dict(sorted(object_pixel_counts.items())), 'instances': sorted(instance_rows, key=lambda row: int(row['instance_id']))}
        self._last_visibility_evidence = copy.deepcopy(payload)
        return payload

    def last_visibility_evidence(self) -> Mapping[str, Any] | None:
        return copy.deepcopy(self._last_visibility_evidence)

    def visible_object_names_at_pose(self, ego_pose: Any, renders: Optional[int]=None, *, modality: str='seg_instance', min_pixels: int=8, object_names: Optional[Sequence[str]]=None) -> List[str]:
        payload = self.render_visibility_at_pose(ego_pose, renders=renders, modality=modality, min_pixels=min_pixels, object_names=object_names)
        return list(payload.get('visible_object_ids') or [])

    def _ensure_visibility_modality(self, modality: str) -> None:
        if self._camera is None:
            raise EgoRendererError('Call reset() before querying visibility.')
        if self._visibility_annotator is not None:
            return
        render_product = getattr(self._camera, 'render_product', None)
        if render_product is None:
            render_product = getattr(self._camera, '_render_product', None)
        if render_product is None:
            raise EgoRendererError('Camera has no render product for rendered visibility.')
        annotator = self._create_fast_visibility_annotator()
        try:
            with self._og.sim.editing_usd():
                annotator.attach([render_product])
            self._visibility_annotator = annotator
            self._visibility_render_product = render_product
            _register_fast_visibility_shutdown(self._og)
            for _ in range(4):
                self._og.sim.render()
        except Exception as exc:
            self._visibility_annotator = None
            self._visibility_render_product = None
            raise EgoRendererError('Failed to initialize fast instance-ID segmentation.') from exc

    def _create_fast_visibility_annotator(self) -> Any:
        from omnigibson import lazy
        return lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(_FAST_VISIBILITY_ANNOTATOR)

    def _fast_visibility_data(self) -> tuple[np.ndarray, Mapping[Any, Any]]:
        annotator = self._visibility_annotator
        if annotator is None:
            raise EgoRendererError('Fast instance-ID segmentation annotator is not initialized.')
        try:
            raw = annotator.get_data()
        except Exception as exc:
            raise EgoRendererError('Fast instance-ID segmentation capture failed.') from exc
        if not isinstance(raw, Mapping):
            raise EgoRendererError('Fast instance-ID segmentation returned a non-mapping payload.')
        data = raw.get('data')
        info = raw.get('info')
        label_mapping = info.get('idToLabels') if isinstance(info, Mapping) else None
        if data is None or not isinstance(label_mapping, Mapping):
            raise EgoRendererError('Fast instance-ID segmentation is missing data or idToLabels.')
        seg = _to_numpy(data)
        if seg.ndim != 2 or seg.size == 0:
            raise EgoRendererError('Fast instance-ID segmentation returned an empty or non-2D image.')
        return (seg, label_mapping)

    def _detach_visibility_annotator(self) -> None:
        annotator = self._visibility_annotator
        render_product = self._visibility_render_product
        self._visibility_annotator = None
        self._visibility_render_product = None
        if annotator is None or render_product is None or self._og is None:
            return
        with self._og.sim.editing_usd():
            annotator.detach(render_product)
        for _ in range(2):
            self._og.sim.render()

    def _object_names_from_visibility_labels(self, labels: Sequence[str], *, object_names: Optional[Sequence[str]]=None) -> List[str]:
        allowed = {str(name) for name in object_names} if object_names is not None else None
        safe_to_raw: dict[str, set[str]] = {}

        def register(raw_name: str, safe_name: str) -> None:
            if allowed is not None and raw_name not in allowed:
                return
            safe = _og_safe_name(str(safe_name))
            target_safe = self._resolve_alias_target_safe(safe)
            safe_to_raw.setdefault(safe, set()).add(str(raw_name))
            safe_to_raw.setdefault(target_safe, set()).add(str(raw_name))
        for raw_name, safe_name in self._raw_name_to_safe_name.items():
            register(str(raw_name), str(safe_name))
        replacement_sources = allowed if allowed is not None else self._visual_replacements.keys()
        for raw_name in replacement_sources:
            for replacement_name in self._replacement_names_for_object(str(raw_name)):
                register(str(raw_name), str(replacement_name))
        if allowed is not None:
            for raw_name in allowed:
                register(raw_name, self._raw_name_to_safe_name.get(raw_name, _og_safe_name(raw_name)))
        else:
            for safe_name in self._desired_aabb_centers:
                register(str(safe_name), str(safe_name))
        visible: set[str] = set()
        for label in labels:
            for candidate in _visibility_label_candidates(str(label)):
                safe = _og_safe_name(candidate)
                target_safe = self._resolve_alias_target_safe(safe)
                visible.update(safe_to_raw.get(safe, set()))
                visible.update(safe_to_raw.get(target_safe, set()))
                if allowed is not None and candidate in allowed:
                    visible.add(candidate)
        if allowed is not None:
            visible &= allowed
        return sorted(visible)

    def set_object_center(self, object_name: str, center: Any, yaw: float | None=None, *, orientation: Any | None=None, verify_readback: bool=True, operation: str='pose') -> dict[str, Any]:
        diagnostic: dict[str, Any] = {'success': False, 'object_name': str(object_name), 'safe_name': _og_safe_name(str(object_name)), 'desired_center': None, 'actual_center': None, 'center_error_m': None, 'desired_root_position': None, 'actual_root_position': None, 'actual_root_orientation_xyzw': None, 'actual_root_yaw': None, 'yaw': None if yaw is None else float(yaw), 'orientation': _jsonable_orientation(orientation), 'verify_readback': bool(verify_readback), 'reason': ''}
        try:
            diagnostic['desired_center'] = _to_numpy(center).astype(float).tolist()
        except Exception as exc:
            diagnostic['reason'] = 'invalid_center'
            diagnostic['error'] = repr(exc)
            return diagnostic
        if self._env is None:
            diagnostic['reason'] = 'renderer_not_reset'
            return diagnostic
        replacement_names = self._replacement_names_for_object(str(object_name))
        if replacement_names:
            diagnostic['object_resolution'] = self.resolve_render_object_name(object_name, operation=operation)
            replacement_diagnostic = self._set_replacement_object_centers(str(object_name), replacement_names, np.array(diagnostic['desired_center'], dtype=float), yaw=yaw, orientation=orientation, verify_readback=verify_readback)
            diagnostic.update(replacement_diagnostic)
            return diagnostic
        _, safe_name, _ = self._resolve_pose_target_or_diagnostic(object_name, operation=operation, base_diagnostic=diagnostic)
        if not safe_name:
            return diagnostic
        obj = self._object_by_name(safe_name)
        if obj is None:
            diagnostic['reason'] = 'object_not_found'
            return diagnostic
        visible = self._object_visible(obj)
        diagnostic['before_visible'] = visible
        if visible is False:
            diagnostic['reason'] = 'object_not_visible'
            return diagnostic
        center_np = np.array(center, dtype=float)
        if safe_name not in self._object_offsets:
            offset = self._object_aabb_offset(safe_name, obj)
            if offset is None:
                diagnostic['reason'] = 'aabb_offset_unavailable'
                return diagnostic
            self._object_offsets[safe_name] = offset
            self._object_offset_orientations[safe_name] = self._current_orientation(obj)
        root_position = center_np - self._object_offsets[safe_name]
        diagnostic['desired_root_position'] = root_position.astype(float).tolist()
        orientation_to_write = self._resolved_orientation(obj, yaw=yaw, orientation=orientation)
        diagnostic['desired_root_orientation_xyzw'] = list(orientation_to_write)
        try:
            obj.set_position_orientation(position=root_position.tolist(), orientation=list(orientation_to_write), frame='scene')
        except Exception as exc:
            diagnostic['reason'] = 'set_position_orientation_error'
            diagnostic['error'] = repr(exc)
            return diagnostic
        if not bool(verify_readback):
            if yaw is not None or orientation is not None:
                self._object_offsets.pop(safe_name, None)
                self._object_offset_orientations.pop(safe_name, None)
            diagnostic['success'] = True
            diagnostic['reason'] = 'set_position_orientation_unverified'
            return diagnostic
        self._flush_render_for_readback()
        record = self._object_pose_record(safe_name, obj)
        if record is None:
            diagnostic['reason'] = 'readback_center_unavailable'
            return diagnostic
        diagnostic.update(_pose_record_diagnostic(record))
        actual_center = np.asarray(record['aabb_center'], dtype=float)
        diagnostic['actual_center'] = actual_center.tolist()
        diagnostic['center_error_m'] = float(np.linalg.norm(actual_center - center_np))
        self._object_offsets[safe_name] = np.asarray(record['aabb_offset'], dtype=float)
        self._object_offset_orientations[safe_name] = tuple(record['root_orientation_xyzw'])
        diagnostic['success'] = True
        diagnostic['reason'] = 'set_position_orientation'
        return diagnostic

    def set_object_pose(self, object_name: str, root_position: Any, root_orientation_xyzw: Any | None=None, *, yaw: float | None=None, verify_readback: bool=True) -> dict[str, Any]:
        safe_name = _og_safe_name(str(object_name))
        diagnostic: dict[str, Any] = {'success': False, 'object_name': str(object_name), 'safe_name': safe_name, 'desired_root_position': None, 'desired_root_orientation_xyzw': _jsonable_orientation(root_orientation_xyzw), 'actual_root_position': None, 'actual_root_orientation_xyzw': None, 'actual_root_yaw': None, 'actual_center': None, 'actual_aabb_extent': None, 'yaw': None if yaw is None else float(yaw), 'verify_readback': bool(verify_readback), 'reason': ''}
        try:
            root_np = _to_numpy(root_position).astype(float)
            if root_np.shape[0] < 3:
                raise ValueError('root_position must have length 3')
            root_np = root_np[:3]
            diagnostic['desired_root_position'] = root_np.tolist()
        except Exception as exc:
            diagnostic['reason'] = 'invalid_root_position'
            diagnostic['error'] = repr(exc)
            return diagnostic
        if self._env is None:
            diagnostic['reason'] = 'renderer_not_reset'
            return diagnostic
        _, safe_name, _ = self._resolve_pose_target_or_diagnostic(object_name, operation='pose', base_diagnostic=diagnostic)
        if not safe_name:
            return diagnostic
        obj = self._object_by_name(safe_name)
        if obj is None:
            diagnostic['reason'] = 'object_not_found'
            return diagnostic
        visible = self._object_visible(obj)
        diagnostic['before_visible'] = visible
        if visible is False:
            diagnostic['reason'] = 'object_not_visible'
            return diagnostic
        orientation_to_write = self._resolved_orientation(obj, yaw=yaw, orientation=root_orientation_xyzw)
        diagnostic['desired_root_orientation_xyzw'] = list(orientation_to_write)
        try:
            obj.set_position_orientation(position=root_np.tolist(), orientation=list(orientation_to_write), frame='scene')
        except Exception as exc:
            diagnostic['reason'] = 'set_position_orientation_error'
            diagnostic['error'] = repr(exc)
            return diagnostic
        self._object_offsets.pop(safe_name, None)
        self._object_offset_orientations.pop(safe_name, None)
        if not bool(verify_readback):
            diagnostic['success'] = True
            diagnostic['reason'] = 'set_position_orientation_unverified'
            return diagnostic
        self._flush_render_for_readback()
        record = self._object_pose_record(safe_name, obj)
        if record is None:
            diagnostic['reason'] = 'readback_pose_unavailable'
            return diagnostic
        diagnostic.update(_pose_record_diagnostic(record))
        diagnostic['actual_center'] = list(record['aabb_center'])
        diagnostic['actual_aabb_extent'] = list(record['aabb_extent'])
        self._object_offsets[safe_name] = np.asarray(record['aabb_offset'], dtype=float)
        self._object_offset_orientations[safe_name] = tuple(record['root_orientation_xyzw'])
        diagnostic['success'] = True
        diagnostic['reason'] = 'set_position_orientation'
        return diagnostic

    def _flush_render_for_readback(self, renders: int=2) -> None:
        sim = getattr(getattr(self, '_og', None), 'sim', None)
        render = getattr(sim, 'render', None)
        if not callable(render):
            return
        for _ in range(max(1, int(renders))):
            try:
                render()
            except Exception:
                return

    def _sync_gripper_visual_for_held_pose(self, held_pose: Any, *, ego_pose: Any=None, state: str='holding') -> tuple[Any, dict[str, Any] | None]:
        if held_pose is None:
            return (held_pose, None)
        arm_anchor = self._ego_arm_held_anchor_for_pose(ego_pose)
        if arm_anchor is not None:
            return self._sync_held_pose_to_gripper_anchor(held_pose, arm_anchor, state=state)
        if self._gripper_visual_sync is None:
            return (held_pose, None)
        try:
            if ego_pose is not None:
                gripper_diagnostic = self._gripper_visual_sync.sync_from_ego_pose(self, ego_pose, held_pose=held_pose, state=state)
            else:
                gripper_diagnostic = self._gripper_visual_sync.sync_from_held_pose(self, held_pose, state=state)
        except Exception as exc:
            gripper_diagnostic = {'enabled': True, 'success': False, 'reason': 'gripper_visual_sync_error', 'error': repr(exc)}
        gripper_diagnostic = _to_jsonable_value(gripper_diagnostic)
        self._last_gripper_visual_sync = dict(gripper_diagnostic) if isinstance(gripper_diagnostic, Mapping) else {}
        adjusted_held_pose = gripper_diagnostic.get('held_pose_for_renderer') if isinstance(gripper_diagnostic, Mapping) else None
        return (adjusted_held_pose if adjusted_held_pose is not None else held_pose, gripper_diagnostic)

    def _ego_arm_held_anchor_for_pose(self, ego_pose: Any) -> GripperAnchor | None:
        if self._ego_arm_visual_sync is None or ego_pose is None:
            return None
        self._sync_ego_arm_visual_to_pose(ego_pose)
        anchor_payload = self._last_ego_arm_visual_sync.get('held_anchor') if isinstance(self._last_ego_arm_visual_sync, Mapping) else None
        if not isinstance(anchor_payload, Mapping):
            return None
        center = anchor_payload.get('center')
        if center is None:
            return None
        try:
            return GripperAnchor(center=tuple((float(v) for v in _to_numpy(center).astype(float).tolist()[:3])), yaw=float(anchor_payload.get('yaw', 0.0)), pitch=float(anchor_payload.get('pitch', 0.0)))
        except Exception:
            return None

    def _sync_held_pose_to_gripper_anchor(self, held_pose: Any, anchor: GripperAnchor, *, state: str) -> tuple[Any, dict[str, Any]]:
        try:
            if self._gripper_visual_sync is not None:
                gripper_diagnostic = self._gripper_visual_sync.sync_from_anchor(self, anchor, held_pose=held_pose, state=state)
            else:
                gripper_diagnostic = {'enabled': False, 'success': True, 'reason': 'ego_arm_visual_eef_anchor', 'state': str(state), 'anchor': anchor.as_dict(), 'proxy_object_names': [], 'proxy_mutations': [], 'held_pose_for_renderer': held_pose_at_gripper(held_pose, anchor, self._gripper_visual_config)}
        except Exception as exc:
            gripper_diagnostic = {'enabled': True, 'success': False, 'reason': 'gripper_visual_sync_error', 'error': repr(exc)}
        gripper_diagnostic = _to_jsonable_value(gripper_diagnostic)
        self._last_gripper_visual_sync = dict(gripper_diagnostic) if isinstance(gripper_diagnostic, Mapping) else {}
        adjusted_held_pose = gripper_diagnostic.get('held_pose_for_renderer') if isinstance(gripper_diagnostic, Mapping) else None
        return (adjusted_held_pose if adjusted_held_pose is not None else held_pose, gripper_diagnostic)

    def set_held_pose(self, held_pose: Any, *, verify_readback: bool=True, ego_pose: Any=None, sync_gripper_visual: bool=True) -> dict[str, Any]:
        if held_pose is None:
            return {'success': False, 'reason': 'held_pose_absent'}
        original_held_pose = held_pose
        gripper_diagnostic = None
        if bool(sync_gripper_visual):
            held_pose, gripper_diagnostic = self._sync_gripper_visual_for_held_pose(held_pose, ego_pose=ego_pose, state='holding')
        object_name = _held_pose_field(held_pose, 'object_name', '')
        center = _held_pose_field(held_pose, 'center', (0.0, 0.0, 0.0))
        yaw_value = _held_pose_field(held_pose, 'yaw', None)
        yaw = None if yaw_value is None else float(yaw_value)
        diagnostic = self.set_object_center(object_name, center, yaw=yaw, verify_readback=verify_readback, operation='held')
        diagnostic['held_pose'] = to_jsonable_held_pose(held_pose)
        if gripper_diagnostic is not None:
            diagnostic['gripper_visual'] = _to_jsonable_value(gripper_diagnostic)
            diagnostic['gripper_visual_status'] = dict(self._last_gripper_visual_sync or {})
            if to_jsonable_held_pose(original_held_pose) != to_jsonable_held_pose(held_pose):
                diagnostic['original_held_pose'] = to_jsonable_held_pose(original_held_pose)
        return diagnostic

    def gripper_visual_status(self) -> dict[str, Any]:
        return dict(self._last_gripper_visual_sync or {})

    def ego_arm_visual_status(self) -> dict[str, Any]:
        return dict(self._last_ego_arm_visual_sync or {})

    def sync_from_core(self, core: Any, *, prefer_root_pose: bool=True, update_core_from_readback: bool=False) -> None:
        if core is None:
            return
        self.sync_container_interiors_to_core(core)
        held = getattr(getattr(core, 'ego_state', None), 'held_object_name', None)
        held_pose = core.current_held_pose() if held and hasattr(core, 'current_held_pose') else None
        if held_pose is not None:
            held_pose, _ = self._sync_gripper_visual_for_held_pose(held_pose, ego_pose=getattr(getattr(core, 'ego_state', None), 'pose', None), state='holding')
        elif self._gripper_visual_sync is not None:
            if held_pose is None:
                try:
                    gripper_diagnostic = self._gripper_visual_sync.sync_from_core(self, core, held_pose=None)
                except Exception as exc:
                    gripper_diagnostic = {'enabled': True, 'success': False, 'reason': 'gripper_visual_sync_error', 'error': repr(exc)}
                self._last_gripper_visual_sync = _to_jsonable_value(gripper_diagnostic)
        for name, obj in getattr(core, 'objects', {}).items():
            if str(name) == str(held) and held_pose is not None:
                diagnostic = self.set_held_pose(held_pose, verify_readback=bool(update_core_from_readback), sync_gripper_visual=False)
            else:
                root_position = getattr(obj, 'root_position', None)
                root_orientation = getattr(obj, 'root_orientation_xyzw', None)
                root_yaw = getattr(obj, 'root_yaw', None)
                if bool(prefer_root_pose) and root_position is not None:
                    diagnostic = self.set_object_pose(str(name), root_position, root_orientation_xyzw=root_orientation, yaw=root_yaw if root_orientation is None else None, verify_readback=bool(update_core_from_readback))
                else:
                    diagnostic = self.set_object_center(str(name), getattr(obj, 'center', (0.0, 0.0, 0.0)), yaw=root_yaw if root_yaw is not None else None, verify_readback=bool(update_core_from_readback))
            if bool(update_core_from_readback):
                _update_core_object_from_pose_diagnostic(obj, diagnostic)

    def sync_initial_from_core(self, core: Any) -> None:
        self.sync_from_core(core)
        self._sync_initial_boolean_states(core)

    def apply_action_result(self, action_result: Any) -> list[dict[str, Any]]:
        mutations: list[dict[str, Any]] = []
        canonical_toggle_target, canonical_toggle_value = self._toggle_action_target_and_value(action_result)
        if not self.particle_systems_enabled and canonical_toggle_target is not None and (canonical_toggle_value is not None):
            mutations = [self.set_toggle_visual(canonical_toggle_target, canonical_toggle_value, reason='toggle_result')]
        elif self._visual_sync is not None:
            try:
                mutations = list(self._visual_sync.apply_action_result(action_result) or [])
            except Exception as exc:
                fallback_mutation = self._toggle_visual_fallback_from_action_result(action_result, native_error=repr(exc), reason='apply_action_result_fallback')
                if fallback_mutation is not None:
                    mutations = [fallback_mutation]
                else:
                    mutations = [{'success': False, 'reason': 'apply_action_result_error', 'action': getattr(action_result, 'action', None), 'error': repr(exc)}]
            else:
                failed_toggle_mutations = self._failed_toggle_visual_mutations(action_result, mutations)
                if failed_toggle_mutations:
                    fallback_mutation = self._toggle_visual_fallback_from_action_result(action_result, failed_mutations=failed_toggle_mutations, reason='failed_toggle_visual_mutation_fallback')
                    if fallback_mutation is not None and bool(fallback_mutation.get('success', False)):
                        mutations = [fallback_mutation]
        self._register_cut_visual_replacements(mutations)
        if bool(getattr(action_result, 'success', False)) and getattr(action_result, 'action', None) == 'PICK':
            held_delta = (getattr(action_result, 'state_delta', None) or {}).get('held_object', {})
            held_pose = held_delta.get('held_pose') if isinstance(held_delta, Mapping) else None
            held_pose_to_write, aliased_from = self._action_held_pose_for_renderer(held_pose)
            diagnostic = self.set_held_pose(held_pose_to_write, verify_readback=False, ego_pose=getattr(action_result, 'end_pose', None))
            gripper_diagnostic = diagnostic.get('gripper_visual') if isinstance(diagnostic, Mapping) else None
            if isinstance(gripper_diagnostic, Mapping):
                mutations.extend(list(gripper_diagnostic.get('proxy_mutations') or []))
            diagnostic['action'] = 'PICK'
            diagnostic['mutation'] = 'held_pose'
            if aliased_from is not None:
                diagnostic['core_object_name'] = str(aliased_from)
                diagnostic['aliased_from_object_name'] = str(aliased_from)
                diagnostic['original_held_pose'] = to_jsonable_held_pose(held_pose)
            mutations.append(diagnostic)
        if bool(getattr(action_result, 'success', False)) and getattr(action_result, 'action', None) in {'PLACE', 'RELEASE'}:
            if self._gripper_visual_sync is not None:
                end_pose = getattr(action_result, 'end_pose', None)
                if end_pose is not None:
                    try:
                        self._last_gripper_visual_sync = _to_jsonable_value(self._gripper_visual_sync.sync_from_ego_pose(self, end_pose, state='idle'))
                    except Exception as exc:
                        self._last_gripper_visual_sync = {'enabled': True, 'success': False, 'reason': 'gripper_visual_sync_error', 'error': repr(exc)}
            pose_delta = (getattr(action_result, 'state_delta', None) or {}).get('object_pose', {})
            target = pose_delta.get('target')
            center = pose_delta.get('after_center')
            if target and center:
                render_target, aliased_from = self._action_render_name_for_object(str(target), operation='pose')
                diagnostic = self.set_object_center(render_target, center, verify_readback=True)
                diagnostic['action'] = getattr(action_result, 'action', None)
                diagnostic['mutation'] = 'object_pose'
                if aliased_from is not None:
                    diagnostic['core_object_name'] = str(aliased_from)
                    diagnostic['aliased_from_object_name'] = str(aliased_from)
                mutations.append(diagnostic)
        return mutations

    def _toggle_action_target_and_value(self, action_result: Any) -> tuple[str | None, bool | None]:
        action = getattr(action_result, 'action', None)
        if action not in {'TOGGLE_ON', 'TOGGLE_OFF'}:
            return (None, None)
        state_delta = getattr(action_result, 'state_delta', None) or {}
        target_name = None
        if isinstance(state_delta, Mapping):
            for key in ('object_state', 'toggle'):
                delta = state_delta.get(key)
                if isinstance(delta, Mapping) and delta.get('target'):
                    target_name = delta.get('target')
                    break
        if not target_name:
            target_name = (getattr(action_result, 'args', {}) or {}).get('target')
        if not target_name:
            return (None, None)
        requested_value = action == 'TOGGLE_ON'
        try:
            value = self._state_delta_after(action_result, 'toggled_on')
            if value is not None:
                requested_value = bool(value)
        except Exception:
            pass
        return (str(target_name), bool(requested_value))

    def _failed_toggle_visual_mutations(self, action_result: Any, mutations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        target_name, _ = self._toggle_action_target_and_value(action_result)
        if not target_name:
            return []
        failed: list[dict[str, Any]] = []
        target_safe = _og_safe_name(target_name)
        for mutation in mutations or ():
            if not isinstance(mutation, Mapping) or bool(mutation.get('success', False)):
                continue
            mutation_state = str(mutation.get('state') or '')
            mutation_action = str(mutation.get('action') or getattr(action_result, 'action', '') or '')
            mutation_object = mutation.get('object')
            mutation_object_safe = _og_safe_name(str(mutation_object)) if mutation_object else ''
            looks_like_toggle = mutation_state == 'toggled_on' or mutation_action in {'TOGGLE_ON', 'TOGGLE_OFF'}
            if mutation_object_safe and mutation_object_safe != target_safe and (not looks_like_toggle):
                continue
            if looks_like_toggle or mutation_object is None or mutation_object_safe == target_safe:
                failed.append(dict(mutation))
        return failed

    def _toggle_visual_fallback_from_action_result(self, action_result: Any, *, native_error: str | None=None, failed_mutations: Sequence[Mapping[str, Any]] | None=None, reason: str) -> dict[str, Any] | None:
        target_name, requested_value = self._toggle_action_target_and_value(action_result)
        if target_name is None or requested_value is None:
            return None
        try:
            fallback_mutation = self.set_toggle_visual(str(target_name), bool(requested_value), reason=reason)
            if not bool(fallback_mutation.get('success', False)):
                stream_mutation = self._set_toggle_liquid_stream_visual(str(target_name), bool(requested_value), reason=reason)
                if bool(stream_mutation.get('success', False)):
                    stream_mutation['failed_set_toggle_visual'] = _to_jsonable_value(fallback_mutation)
                    fallback_mutation = stream_mutation
            if native_error is not None:
                fallback_mutation['native_apply_action_result_error'] = str(native_error)
            if failed_mutations:
                fallback_mutation['replaced_failed_visual_mutations'] = _to_jsonable_value(list(failed_mutations))
            return fallback_mutation
        except Exception as fallback_exc:
            payload: dict[str, Any] = {'success': False, 'reason': 'apply_action_result_error', 'action': getattr(action_result, 'action', None), 'object': str(target_name), 'state': 'toggled_on', 'requested': bool(requested_value), 'fallback_error': repr(fallback_exc)}
            if native_error is not None:
                payload['error'] = str(native_error)
            if failed_mutations:
                payload['replaced_failed_visual_mutations'] = _to_jsonable_value(list(failed_mutations))
            return payload

    def _action_held_pose_for_renderer(self, held_pose: Any) -> tuple[Any, str | None]:
        object_name = _held_pose_field(held_pose, 'object_name', None)
        render_name, aliased_from = self._action_render_name_for_object(object_name, operation='held')
        if aliased_from is None or held_pose is None:
            return (held_pose, aliased_from)
        payload = to_jsonable_held_pose(held_pose)
        payload['object_name'] = str(render_name)
        return (payload, aliased_from)

    def _action_render_name_for_object(self, object_name: Any, *, operation: str='pose') -> tuple[str, str | None]:
        if object_name is None:
            return ('', None)
        resolution = self.resolve_render_object_name(object_name, operation=operation)
        text = str(object_name)
        render_name = str(resolution.get('resolved_name') or text)
        if bool(resolution.get('success')) and render_name != text:
            return (render_name, text)
        return (text, None)

    def _state_delta_after(self, action_result: Any, state_name: str) -> Any | None:
        state_delta = getattr(action_result, 'state_delta', None) or {}
        object_state = state_delta.get('object_state') if isinstance(state_delta, Mapping) else None
        if isinstance(object_state, Mapping) and object_state.get('state') == state_name:
            return object_state.get('after')
        return None

    def _visual_state_target_for_object(self, object_name: str) -> dict[str, Any]:
        resolution = self.resolve_render_object_name(object_name, operation='visual_state')
        resolved_name = str(resolution.get('resolved_name') or object_name)
        resolved_safe_name = str(resolution.get('resolved_safe_name') or _og_safe_name(resolved_name))
        og_obj = self._object_by_name(resolved_safe_name)
        if og_obj is None:
            mapping = getattr(self._visual_sync, 'og_object_by_name', None)
            if isinstance(mapping, Mapping):
                candidates = [resolved_name, resolved_safe_name, str(object_name), _og_safe_name(str(object_name))]
                for candidate in dict.fromkeys(candidates):
                    og_obj = mapping.get(str(candidate))
                    if og_obj is not None:
                        break
        return {'resolution': resolution, 'resolved_name': resolved_name, 'resolved_safe_name': resolved_safe_name, 'og_obj': og_obj}

    def _set_toggle_liquid_stream_visual(self, object_name: str, value: bool, *, reason: str) -> dict[str, Any]:
        if not self.particle_source_stream_visuals_enabled:
            return {'success': False, 'object': str(object_name), 'state': 'toggled_on', 'requested': bool(value), 'reason': reason, 'error': 'particle_source_stream_visuals_disabled', 'particle_systems_enabled': self.particle_systems_enabled, 'particle_source_stream_visuals_enabled': False}
        if self._visual_sync is None:
            return {'success': False, 'object': str(object_name), 'state': 'toggled_on', 'requested': bool(value), 'reason': reason, 'error': 'visual_sync_unavailable'}
        helper = getattr(self._visual_sync, '_set_missing_toggle_water_stream_fallback', None)
        if not callable(helper):
            return {'success': False, 'object': str(object_name), 'state': 'toggled_on', 'requested': bool(value), 'reason': reason, 'error': 'missing_toggle_water_stream_fallback_helper'}
        before_count = len(self.visual_mutations)
        target_info = self._visual_state_target_for_object(str(object_name))
        info = helper(str(object_name), target_info.get('og_obj'), bool(value), reason, fallback_reason='apply_action_result_fallback')
        mutations = self.visual_mutations[before_count:]
        success = bool(info and info.get('success'))
        return {'success': success, 'object': str(object_name), 'state': 'toggled_on', 'requested': bool(value), 'reason': reason, 'fallback': 'water_source_stream' if info is not None else None, 'resolved_object_name': str(target_info.get('resolved_name') or ''), 'resolved_safe_name': str(target_info.get('resolved_safe_name') or ''), 'object_resolution': dict(target_info.get('resolution') or {}), 'stream_visual': dict(info or {}), 'mutations': mutations}

    def set_open_openness(self, object_name: str, openness: float, *, symbolic_open: bool | None=None, reason: str='partial_open_close') -> dict[str, Any]:
        if self._visual_sync is None:
            return {'success': False, 'object': str(object_name), 'openness': float(openness), 'symbolic_open': symbolic_open, 'reason': reason, 'error': 'visual_sync_unavailable'}
        try:
            return dict(self._visual_sync.set_open_openness(str(object_name), float(openness), symbolic_open=symbolic_open, reason=reason) or {})
        except Exception as exc:
            return {'success': False, 'object': str(object_name), 'openness': float(openness), 'symbolic_open': symbolic_open, 'reason': reason, 'error': repr(exc)}

    def set_open_visual(self, object_name: str, value: bool, *, reason: str='open_result') -> dict[str, Any]:
        if self._visual_sync is None:
            return {'success': False, 'object': str(object_name), 'state': 'open', 'requested': bool(value), 'reason': reason, 'error': 'visual_sync_unavailable'}
        before_count = len(self.visual_mutations)
        try:
            self._visual_sync.set_open_visual(str(object_name), bool(value), reason=reason)
        except Exception as exc:
            return {'success': False, 'object': str(object_name), 'state': 'open', 'requested': bool(value), 'reason': reason, 'error': repr(exc)}
        mutations = self.visual_mutations[before_count:]
        success = bool(mutations) and all((bool(item.get('success', False)) for item in mutations))
        return {'success': success, 'object': str(object_name), 'state': 'open', 'requested': bool(value), 'reason': reason, 'mutations': mutations}

    def set_toggle_visual(self, object_name: str, value: bool, *, reason: str='toggle_result') -> dict[str, Any]:
        if not self.particle_systems_enabled:
            return self._set_particle_suppressed_toggle_visual(str(object_name), bool(value), reason=reason)
        if self._visual_sync is None:
            return {'success': False, 'object': str(object_name), 'state': 'toggled_on', 'requested': bool(value), 'reason': reason, 'error': 'visual_sync_unavailable'}
        target_info = self._visual_state_target_for_object(str(object_name))
        before_count = len(self.visual_mutations)
        try:
            self._visual_sync.set_toggle_visual(str(object_name), bool(value), reason=reason, og_obj=target_info.get('og_obj'))
        except Exception as exc:
            return {'success': False, 'object': str(object_name), 'state': 'toggled_on', 'requested': bool(value), 'reason': reason, 'resolved_object_name': str(target_info.get('resolved_name') or ''), 'resolved_safe_name': str(target_info.get('resolved_safe_name') or ''), 'object_resolution': dict(target_info.get('resolution') or {}), 'error': repr(exc)}
        mutations = self.visual_mutations[before_count:]
        success = bool(mutations) and all((bool(item.get('success', False)) for item in mutations))
        return {'success': success, 'object': str(object_name), 'state': 'toggled_on', 'requested': bool(value), 'reason': reason, 'resolved_object_name': str(target_info.get('resolved_name') or ''), 'resolved_safe_name': str(target_info.get('resolved_safe_name') or ''), 'object_resolution': dict(target_info.get('resolution') or {}), 'mutations': mutations}

    def _set_particle_suppressed_toggle_visual(self, object_name: str, value: bool, *, reason: str) -> dict[str, Any]:
        target_info = self._visual_state_target_for_object(str(object_name))
        obj = target_info.get('og_obj')
        if obj is None:
            return {'success': False, 'object': str(object_name), 'state': 'toggled_on', 'requested': bool(value), 'reason': reason, 'error': 'particle_suppressed_toggle_object_missing', 'particle_systems_enabled': False}
        native_toggle_result = None
        native_toggle_error = None
        try:
            from omnigibson.object_states import ToggledOn
            toggled = (getattr(obj, 'states', {}) or {}).get(ToggledOn)
            if toggled is None:
                raise EgoRendererError('particle-suppressed replay target has no ToggledOn state')
            native_toggle_result = toggled.set_value(bool(value))
        except Exception as exc:
            native_toggle_error = repr(exc)
        stream_visual = None
        stream_success = True
        if self.particle_source_stream_visuals_enabled:
            stream_visual = self._set_toggle_liquid_stream_visual(str(object_name), bool(value), reason=reason)
            stream_success = bool(stream_visual.get('success', False))
        native_toggle_success = native_toggle_error is None and native_toggle_result is not False
        visual_fallback_success = self.particle_source_stream_visuals_enabled and stream_success
        diagnostic = {'success': native_toggle_success or visual_fallback_success, 'object': str(object_name), 'state': 'toggled_on', 'requested': bool(value), 'reason': reason, 'resolved_object_name': str(target_info.get('resolved_name') or ''), 'resolved_safe_name': str(target_info.get('resolved_safe_name') or ''), 'object_resolution': dict(target_info.get('resolution') or {}), 'particle_systems_enabled': False, 'particle_source_stream_visuals_enabled': self.particle_source_stream_visuals_enabled, 'water_visual_mode': 'continuous_mesh' if self.particle_source_stream_visuals_enabled else 'excluded', 'stream_visual': stream_visual, 'native_toggle_success': native_toggle_success, 'native_toggle_result': native_toggle_result, 'native_toggle_error': native_toggle_error, 'particle_fallback_forbidden': not self.particle_source_stream_visuals_enabled}
        self._record_reset_event('toggle_visual', 'mutation', diagnostic=_to_jsonable_value(diagnostic))
        return diagnostic

    def set_dirty_visual(self, object_name: str, value: bool, *, reason: str='dirty_result') -> dict[str, Any]:
        if self._visual_sync is None:
            return {'success': False, 'object': str(object_name), 'state': 'dirty', 'requested': bool(value), 'reason': reason, 'error': 'visual_sync_unavailable'}
        before_count = len(self.visual_mutations)
        try:
            self._visual_sync.set_dirty_visual(str(object_name), bool(value), reason=reason)
        except Exception as exc:
            return {'success': False, 'object': str(object_name), 'state': 'dirty', 'requested': bool(value), 'reason': reason, 'error': repr(exc)}
        mutations = self.visual_mutations[before_count:]
        success = bool(mutations) and all((bool(item.get('success', False)) for item in mutations))
        return {'success': success, 'object': str(object_name), 'state': 'dirty', 'requested': bool(value), 'reason': reason, 'mutations': mutations}

    @property
    def visual_mutations(self) -> list[dict[str, Any]]:
        if self._visual_sync is None:
            return []
        return list(getattr(self._visual_sync, 'visual_mutations', []) or [])

    @property
    def container_interiors(self) -> dict[str, dict[str, list[float]]]:
        return dict(self._container_interiors)

    def sync_container_interiors_to_core(self, core: Any) -> None:
        if not self._container_interiors:
            return
        objects = getattr(core, 'objects', {}) or {}
        for name, interior in self._container_interiors.items():
            obj = objects.get(name)
            if obj is None:
                continue
            states = dict(getattr(obj, 'states', {}) or {})
            states['container_interior_center'] = list(interior['center'])
            states['container_interior_half_extent'] = list(interior['half_extent'])
            obj.states = states

    def scene_summary(self) -> dict[str, Any]:
        scene = getattr(self._env, 'scene', None)
        objects = list(getattr(scene, 'objects', []) or [])
        categories = Counter((str(getattr(obj, 'category', 'unknown')) for obj in objects))
        loaded = []
        for obj in objects:
            name = str(getattr(obj, 'name', ''))
            if name not in self._desired_aabb_centers:
                continue
            try:
                center = _to_numpy(getattr(obj, 'aabb_center')).astype(float).tolist()
                extent = _to_numpy(getattr(obj, 'aabb_extent')).astype(float).tolist()
            except Exception:
                center = None
                extent = None
            loaded.append({'name': name, 'category': str(getattr(obj, 'category', '')), 'desired_aabb_center': self._desired_aabb_centers.get(name), 'actual_aabb_center': center, 'actual_aabb_extent': extent, 'state_names': _og_state_names(obj)})
        return {'scene_model': self.env_config.get('scene', {}).get('scene_model'), 'scene_type': self.env_config.get('scene', {}).get('type'), 'headless_requested': True, 'render_aliases': dict(sorted(self._object_aliases.items())), 'render_alias_metadata': dict(sorted(self._render_alias_metadata.items())), 'render_identity_map': self.render_identity_map(), 'scene_object_count': len(objects), 'scene_category_counts': dict(sorted(categories.items())), 'external_sensors': list((getattr(self._env, 'external_sensors', {}) or {}).keys()) if self._env is not None else [], 'requested_loaded_objects': loaded, 'hidden_scene_structures': list(self._hidden_scene_structures), 'hidden_scene_structure_categories': sorted(self.hide_scene_structure_categories), 'container_interiors': dict(sorted(self._container_interiors.items())), 'ego_arm_visual': dict(self._last_ego_arm_visual_sync or {}), 'gripper_visual': dict(self._last_gripper_visual_sync or {}), 'particle_systems_enabled': self.particle_systems_enabled, 'excluded_particle_systems': [] if self.particle_systems_enabled else ['water'], 'native_particle_source_visuals_enabled': self.native_particle_source_visuals_enabled, 'particle_source_stream_visuals_enabled': self.particle_source_stream_visuals_enabled, 'water_visual_mode': 'native_particle_plus_continuous_mesh' if self.native_particle_source_visuals_enabled and self.particle_source_stream_visuals_enabled else 'native_particle' if self.native_particle_source_visuals_enabled else 'continuous_mesh' if self.particle_source_stream_visuals_enabled else 'excluded', 'particle_suppression_report': dict(self._particle_suppression_report), 'object_pose_catalog': self.object_pose_catalog()}

    def render_identity_map(self) -> dict[str, Any]:
        objects_by_safe = {str(getattr(obj, 'name', '')): str(getattr(obj, 'category', '')) for obj in getattr(getattr(self._env, 'scene', None), 'objects', []) or [] if str(getattr(obj, 'name', ''))}
        aliases: dict[str, dict[str, Any]] = {}
        for alias_safe, target_safe in sorted(self._object_aliases.items()):
            metadata = dict(self._render_alias_metadata.get(alias_safe) or {})
            chain_info = self._resolve_alias_chain(alias_safe)
            final_target_safe = str(chain_info.get('resolved_safe_name') or target_safe)
            aliases[str(alias_safe)] = {'alias': str(alias_safe), 'target': str(target_safe), 'target_loaded': str(target_safe) in objects_by_safe, 'target_category': objects_by_safe.get(str(target_safe)), 'final_target': final_target_safe, 'final_target_loaded': final_target_safe in objects_by_safe, 'final_target_category': objects_by_safe.get(final_target_safe), 'alias_chain': list(chain_info.get('alias_chain') or []), 'alias_cycle': bool(chain_info.get('alias_cycle')), 'alias_kind': metadata.get('alias_kind'), 'pose_sync_allowed': metadata.get('pose_sync_allowed'), 'held_sync_allowed': metadata.get('held_sync_allowed'), 'visual_state_sync_allowed': metadata.get('visual_state_sync_allowed'), 'metadata': metadata}
        raw_names = {str(raw_name): {'safe_name': str(safe_name), 'alias_target': str(self._resolve_alias_target_safe(str(safe_name))), 'loaded': str(self._resolve_alias_target_safe(str(safe_name))) in objects_by_safe} for raw_name, safe_name in sorted(self._raw_name_to_safe_name.items())}
        return {'aliases': aliases, 'raw_name_to_safe_name': raw_names, 'loaded_object_names': sorted(objects_by_safe)}

    def _hide_scene_structures(self) -> list[dict[str, str]]:
        if self._env is None or not self.hide_scene_structure_categories:
            return []
        hidden: list[dict[str, str]] = []
        desired = set(self._desired_aabb_centers)
        for obj in getattr(getattr(self._env, 'scene', None), 'objects', []) or []:
            name = str(getattr(obj, 'name', '') or '')
            category = str(getattr(obj, 'category', '') or '').lower()
            if not name or name in desired or category not in self.hide_scene_structure_categories:
                continue
            try:
                setattr(obj, 'visible', False)
                hidden.append({'name': name, 'category': category, 'method': 'object_visible_false'})
                continue
            except Exception:
                pass
            try:
                prim = getattr(obj, 'prim', None)
                if prim is not None:
                    from pxr import UsdGeom
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden.append({'name': name, 'category': category, 'method': 'usd_imageable_invisible'})
            except Exception:
                continue
        return hidden

    def scene_traversability(self, *, floor: int=0, occupancy_resolution: float=0.1, assets_root: str | Path | None=None, use_indoor_segmentation_mask: bool=True) -> Any:
        from scoper.benchmark.behavior.execution_core.occupancy_utils import SceneTraversability, scene_traversability_from_scene
        if self._env is None:
            return SceneTraversability(floor=floor, metadata={'enabled': False, 'reason': 'renderer_not_reset'})
        scene_model = str((self.env_config.get('scene') or {}).get('scene_model') or '')
        if not scene_model:
            return SceneTraversability(floor=floor, metadata={'enabled': False, 'reason': 'missing_scene_model'})
        resolved_assets_root = assets_root or _infer_og_assets_root()
        return scene_traversability_from_scene(self._env.scene, scene_model=scene_model, assets_root=resolved_assets_root or Path(''), floor=floor, occupancy_resolution=occupancy_resolution, use_indoor_segmentation_mask=bool(use_indoor_segmentation_mask and resolved_assets_root))

    def scene_mesh_occupancy(self, *, floor: int=0, floor_height: float=0.0, occupancy_resolution: float=0.1, map_size: int | None=None, navigation_height_band: tuple[float, float]=(0.05, 2.0)) -> Any:
        from scoper.benchmark.behavior.execution_core.occupancy_utils import SceneTraversability, scene_mesh_occupancy_from_scene
        if self._env is None:
            return SceneTraversability(floor=floor, metadata={'enabled': False, 'reason': 'renderer_not_reset'})
        return scene_mesh_occupancy_from_scene(self._env.scene, floor=floor, floor_height=floor_height, occupancy_resolution=occupancy_resolution, map_size=map_size, navigation_height_band=navigation_height_band)

    def scene_wall_slice_occupancy(self, *, floor: int=0, floor_height: float=0.0, occupancy_resolution: float=0.05, map_size: int | None=None, slice_heights: tuple[float, ...]=(0.2, 0.8, 1.2), visual_fallback: bool=False, exterior_barriers: Sequence[Any]=(), interior_mask: Any | None=None, interior_mask_resolution: float | None=None, interior_mask_size: int | None=None) -> dict[str, Any]:
        from scoper.benchmark.behavior.execution_core.occupancy_utils import scene_wall_slice_occupancy_from_scene
        if self._env is None:
            return {'enabled': False, 'reason': 'renderer_not_reset', 'floor': floor}
        return scene_wall_slice_occupancy_from_scene(self._env.scene, floor=floor, floor_height=floor_height, occupancy_resolution=occupancy_resolution, map_size=map_size, slice_heights=slice_heights, visual_fallback=visual_fallback, exterior_barriers=exterior_barriers, interior_mask=interior_mask, interior_mask_resolution=interior_mask_resolution, interior_mask_size=interior_mask_size)

    def render_mesh_occupancy_alignment(self, mesh_occupancy: Any, output_dir: str | Path, *, navigation_objects: list[Any] | tuple[Any, ...]=(), reference_traversability: Any | None=None, floor_height: float=0.0, overlay_z: float=2.6) -> dict[str, Any]:
        if self._env is None or self._camera is None or self._og is None:
            return {'enabled': False, 'reason': 'renderer_not_reset'}
        floor_map = getattr(mesh_occupancy, 'floor_map', None)
        if floor_map is None:
            return {'enabled': False, 'reason': 'missing_mesh_occupancy'}
        from PIL import Image
        from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        free = np.asarray(floor_map) > 0
        if free.ndim != 2 or free.size == 0:
            return {'enabled': False, 'reason': f'invalid_mesh_occupancy_shape_{tuple(free.shape)}'}
        resolution = float(getattr(mesh_occupancy, 'map_resolution', None) or 0.1)
        map_size = int(getattr(mesh_occupancy, 'map_size', None) or free.shape[0])
        extent_m = float(map_size) * resolution
        topdown_z = max(float(floor_height) + 7.0, extent_m * 1.25)
        topdown_pose = EgoPose(0.0, 0.0, z=topdown_z, yaw=0.0, pitch=-math.pi / 2.0)
        plain_topdown = self.render_at_pose(topdown_pose, renders=max(4, self.renders_per_frame))
        plain_topdown_path = output_dir / 'scene_topdown_plain.png'
        Image.fromarray(plain_topdown).save(plain_topdown_path)
        stage = self._og.sim.stage
        root_path = '/World/OccupancyDebug'
        self._remove_debug_prim(stage, root_path)
        overlay_info: dict[str, Any] = {}
        try:
            self._install_mesh_occupancy_overlay(stage, root_path, free=free, resolution=resolution, map_size=map_size, z=float(overlay_z), navigation_objects=navigation_objects)
            overlay_topdown = self.render_at_pose(topdown_pose, renders=max(4, self.renders_per_frame))
            overlay_topdown_path = output_dir / 'scene_topdown_mesh_occupancy_overlay.png'
            Image.fromarray(overlay_topdown).save(overlay_topdown_path)
            oblique_pose = _mesh_alignment_oblique_pose(extent_m, float(overlay_z))
            overlay_oblique = self.render_at_pose(oblique_pose, renders=max(4, self.renders_per_frame))
            overlay_oblique_path = output_dir / 'scene_oblique_mesh_occupancy_overlay.png'
            Image.fromarray(overlay_oblique).save(overlay_oblique_path)
            overlay_info.update({'mesh_overlay_topdown_png': str(overlay_topdown_path), 'mesh_overlay_oblique_png': str(overlay_oblique_path), 'occupied_cells': int(np.count_nonzero(~free)), 'free_cells': int(np.count_nonzero(free))})
        finally:
            self._remove_debug_prim(stage, root_path)
        if reference_traversability is not None and getattr(reference_traversability, 'floor_map', None) is not None:
            reference = np.asarray(reference_traversability.floor_map) > 0
            if reference.shape == free.shape:
                diff_root_path = '/World/OccupancyDiffDebug'
                self._remove_debug_prim(stage, diff_root_path)
                try:
                    self._install_mesh_traversability_diff_overlay(stage, diff_root_path, mesh_free=free, reference_free=reference, resolution=resolution, map_size=map_size, z=float(overlay_z))
                    diff_topdown = self.render_at_pose(topdown_pose, renders=max(4, self.renders_per_frame))
                    diff_topdown_path = output_dir / 'scene_topdown_mesh_vs_traversability_diff_overlay.png'
                    Image.fromarray(diff_topdown).save(diff_topdown_path)
                    overlay_info['mesh_vs_traversability_diff_topdown_png'] = str(diff_topdown_path)
                    overlay_info['mesh_free_scene_blocked_cells'] = int(np.count_nonzero(free & ~reference))
                    overlay_info['mesh_blocked_scene_free_cells'] = int(np.count_nonzero(~free & reference))
                finally:
                    self._remove_debug_prim(stage, diff_root_path)
        return {'enabled': True, 'plain_topdown_png': str(plain_topdown_path), 'map_resolution': resolution, 'map_size': map_size, 'map_extent_m': extent_m, 'topdown_z': topdown_z, 'overlay_z': float(overlay_z), 'door_overlay_count': int(len(list(navigation_objects or []))), **overlay_info}

    def scene_navigation_objects(self, *, load_doors: bool=False) -> list[Any]:
        from scoper.benchmark.behavior.execution_core.occupancy import AABBObject
        if not load_doors:
            return []
        if self._env is None:
            return []
        try:
            from omnigibson import object_states
        except Exception:
            object_states = None
        out: list[AABBObject] = []
        for og_obj in getattr(getattr(self._env, 'scene', None), 'objects', []) or []:
            if not _is_door_like_og_object(og_obj):
                continue
            try:
                center = _to_numpy(getattr(og_obj, 'aabb_center')).astype(float)
                extent = _to_numpy(getattr(og_obj, 'aabb_extent')).astype(float)
            except Exception:
                continue
            if center.shape[0] < 3 or extent.shape[0] < 3:
                continue
            states = {'navigation_barrier': 'door', 'blocks_navigation_when_closed': True, 'navigation_opening_margin': 0.12}
            abilities: tuple[str, ...] = ()
            if object_states is not None and object_states.Open in getattr(og_obj, 'states', {}):
                states['open'] = bool(_og_open_value(og_obj, object_states))
                abilities = ('openable',)
            out.append(AABBObject(name=str(getattr(og_obj, 'name', '')), category=str(getattr(og_obj, 'category', 'door') or 'door'), center=(float(center[0]), float(center[1]), float(center[2])), extent=(float(extent[0]), float(extent[1]), float(extent[2])), visible=True, abilities=abilities, states=states))
        return out

    def interaction_faces_by_name(self, *, load_doors: bool=False) -> dict[str, list[dict[str, Any]]]:
        if self._env is None:
            return {}
        try:
            from omnigibson import object_states
        except Exception:
            return {}
        faces_by_safe: dict[str, list[dict[str, Any]]] = {}
        for safe_name, og_obj in self._visual_object_by_name().items():
            if safe_name != _og_safe_name(safe_name):
                continue
            if not load_doors and _is_door_like_og_object(og_obj):
                continue
            if object_states.Open not in getattr(og_obj, 'states', {}):
                continue
            faces = _openable_joint_interaction_faces_for_og_object(og_obj)
            if not faces:
                faces = _aabb_interaction_faces_for_og_object(og_obj)
            if faces:
                faces_by_safe[str(safe_name)] = faces
        out: dict[str, list[dict[str, Any]]] = dict(faces_by_safe)
        for raw_name, safe_name in self._raw_name_to_safe_name.items():
            target_safe = self._resolve_alias_target_safe(safe_name)
            faces = faces_by_safe.get(target_safe)
            if faces:
                out[str(raw_name)] = [dict(face) for face in faces]
                out[str(safe_name)] = [dict(face) for face in faces]
        return out

    def render_action_frames(self, core: Any, trace_id: str, max_frames: int=6) -> List[np.ndarray]:
        payloads = [event.payload for event in core.trace.events if event.trace_id == trace_id and event.event == 'frame_pose']
        if not payloads:
            return [self.render_at_pose(core.ego_state.pose)]
        poses = [p['pose'] for p in _downsample(payloads, max_frames)]
        return [self.render_at_pose(pose) for pose in poses]

    def close(self) -> None:
        with runtime_phase('teardown'):
            self._detach_visibility_annotator()
            if self._preserve_simulator_for_process_exit or os.environ.get('EB_EGO_RENDERER_SKIP_CLEAR', '').strip().lower() in {'1', 'true', 'yes'}:
                self._env = None
                self._camera = None
                return
            if self._og is not None:
                try:
                    self._og.clear()
                except Exception:
                    if self._env is not None:
                        try:
                            self._env.close()
                        except Exception:
                            pass
            self._env = None
            self._camera = None

    def preserve_simulator_for_process_exit(self) -> None:
        self._preserve_simulator_for_process_exit = True

    def _align_loaded_objects_to_desired_centers(self) -> None:
        for name, center in self._desired_aabb_centers.items():
            self.set_object_center(name, center, verify_readback=False)

    def _object_by_name(self, safe_name: str) -> Any:
        if self._env is None:
            return None
        for obj in getattr(getattr(self._env, 'scene', None), 'objects', []) or []:
            if str(getattr(obj, 'name', '')) == safe_name:
                return obj
        return None

    def _object_visible(self, obj: Any) -> bool | None:
        try:
            return bool(getattr(obj, 'visible'))
        except Exception:
            return None

    def _current_orientation(self, obj: Any) -> tuple[float, float, float, float] | None:
        try:
            _, orientation = obj.get_position_orientation(frame='scene')
            values = _to_numpy(orientation).astype(float).tolist()
            if len(values) >= 4:
                return tuple((float(v) for v in values[:4]))
        except Exception:
            return None
        return None

    def _resolved_orientation(self, obj: Any, *, yaw: float | None=None, orientation: Any | None=None) -> tuple[float, float, float, float]:
        if orientation is not None:
            try:
                values = _to_numpy(orientation).astype(float).tolist()
                if len(values) >= 4:
                    return tuple((float(v) for v in values[:4]))
            except Exception:
                pass
        if yaw is not None:
            try:
                from scoper.benchmark.behavior.execution_core.ego_agent import yaw_to_quat_xyzw
                return tuple((float(v) for v in yaw_to_quat_xyzw(float(yaw))))
            except Exception:
                pass
        current = self._current_orientation(obj)
        return current if current is not None else (0.0, 0.0, 0.0, 1.0)

    def _object_aabb_offset(self, safe_name: str, obj: Any) -> np.ndarray | None:
        try:
            root_pos, orientation = obj.get_position_orientation(frame='scene')
            root_np = _to_numpy(root_pos).astype(float)
            center_np = _to_numpy(getattr(obj, 'aabb_center')).astype(float)
            try:
                values = _to_numpy(orientation).astype(float).tolist()
                self._object_offset_orientations[safe_name] = tuple((float(v) for v in values[:4]))
            except Exception:
                self._object_offset_orientations[safe_name] = None
            return center_np - root_np
        except Exception:
            return None

    def _object_pose_record(self, safe_name: str, obj: Any) -> dict[str, Any] | None:
        try:
            root_pos, orientation = obj.get_position_orientation(frame='scene')
            root_np = _to_numpy(root_pos).astype(float)[:3]
            orientation_values = _to_numpy(orientation).astype(float).tolist()
            if len(orientation_values) < 4:
                orientation_values = [0.0, 0.0, 0.0, 1.0]
            orientation_tuple = tuple((float(v) for v in orientation_values[:4]))
            center_np = _to_numpy(getattr(obj, 'aabb_center')).astype(float)[:3]
            extent_np = _object_aabb_extent(obj)
        except Exception:
            return None
        record = {'name': str(safe_name), 'root_position': root_np.astype(float).tolist(), 'root_orientation_xyzw': list(orientation_tuple), 'root_yaw': _quat_xyzw_to_yaw(orientation_tuple), 'aabb_center': center_np.astype(float).tolist(), 'aabb_extent': extent_np.astype(float).tolist(), 'aabb_offset': (center_np - root_np).astype(float).tolist(), 'visible': self._object_visible(obj), 'category': str(getattr(obj, 'category', ''))}
        self._object_pose_catalog[str(safe_name)] = dict(record)
        return record

    def capture_object_pose_catalog(self, object_names: Sequence[str] | None=None) -> dict[str, dict[str, Any]]:
        if self._env is None:
            return {}
        wanted = {_og_safe_name(str(name)) for name in object_names} if object_names is not None else None
        catalog: dict[str, dict[str, Any]] = {}
        for obj in getattr(getattr(self._env, 'scene', None), 'objects', []) or []:
            safe_name = str(getattr(obj, 'name', '') or '')
            if not safe_name or (wanted is not None and safe_name not in wanted):
                continue
            record = self._object_pose_record(safe_name, obj)
            if record is not None:
                catalog[safe_name] = dict(record)
        self._object_pose_catalog.update(catalog)
        return dict(catalog)

    def object_pose_catalog(self) -> dict[str, dict[str, Any]]:
        if self._env is not None:
            self.capture_object_pose_catalog()
        return {name: dict(record) for name, record in self._object_pose_catalog.items()}

    def capture_render_replay_state(self, core: Any, ego_pose: Any) -> dict[str, Any]:
        if self._env is None:
            raise EgoRendererError('Call reset() before capture_render_replay_state().')
        camera_pose = _ego_pose_mapping(ego_pose)
        if camera_pose is None:
            raise EgoRendererError('render replay requires a complete ego camera pose')
        object_names = sorted((str(name) for name in getattr(core, 'objects', {}) or {}))
        visual_states: dict[str, dict[str, bool]] = {}
        for name, obj in sorted((getattr(core, 'objects', {}) or {}).items(), key=lambda item: str(item[0])):
            states = getattr(obj, 'states', {}) or {}
            values = {key: bool(states[key]) for key in ('open', 'toggled_on', 'dirty') if isinstance(states.get(key), (bool, np.bool_))}
            if values:
                visual_states[str(name)] = values
        held_pose = None
        held_name = getattr(getattr(core, 'ego_state', None), 'held_object_name', None)
        current_held_pose = getattr(core, 'current_held_pose', None)
        if held_name and callable(current_held_pose):
            held_pose = to_jsonable_held_pose(current_held_pose())
        identity = self.render_identity_map()
        return {'camera_pose': camera_pose, 'object_names': object_names, 'object_poses': dict(sorted(self.capture_object_pose_catalog().items())), 'visual_states': visual_states, 'held_pose': held_pose, 'render_identity_sha256': _json_sha256(identity)}

    def apply_render_replay_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        from scoper.benchmark.behavior.render.particle_segmentation_replay import validate_render_replay_state
        validate_render_replay_state(state)
        if self.particle_systems_enabled:
            raise EgoRendererError('render replay renderer must suppress particle systems')
        pose_mutations = []
        for name, record in sorted(dict(state['object_poses']).items()):
            if not isinstance(record, Mapping):
                continue
            root_position = record.get('root_position')
            if root_position is None:
                continue
            pose_mutations.append(self.set_object_pose(str(name), root_position, root_orientation_xyzw=record.get('root_orientation_xyzw'), verify_readback=False))
        visual_mutations = []
        for name, states in sorted(dict(state['visual_states']).items()):
            if not isinstance(states, Mapping):
                continue
            if 'open' in states:
                visual_mutations.append(self.set_open_visual(str(name), bool(states['open']), reason='particle_segmentation_replay'))
            if 'toggled_on' in states:
                visual_mutations.append(self.set_toggle_visual(str(name), bool(states['toggled_on']), reason='particle_segmentation_replay'))
            if 'dirty' in states:
                visual_mutations.append(self.set_dirty_visual(str(name), bool(states['dirty']), reason='particle_segmentation_replay'))
        held_pose = state.get('held_pose')
        held_mutation = None
        if isinstance(held_pose, Mapping) and held_pose:
            held_mutation = self.set_held_pose(held_pose, verify_readback=False, ego_pose=state['camera_pose'])
        return {'pose_mutations': _to_jsonable_value(pose_mutations), 'visual_mutations': _to_jsonable_value(visual_mutations), 'held_mutation': _to_jsonable_value(held_mutation)}

    def render_replay_visibility(self, state: Mapping[str, Any], *, min_pixels: int=8) -> dict[str, Any]:
        diagnostics = self.apply_render_replay_state(state)
        safety = self._particle_segmentation_safety_check()
        visibility = self.render_visibility_at_pose(state['camera_pose'], modality='seg_instance_id', min_pixels=min_pixels, object_names=state['object_names'])
        visibility['particle_systems_enabled'] = False
        visibility['excluded_particle_systems'] = ['water']
        visibility['render_replay_state_sha256'] = state['state_sha256']
        visibility['particle_suppression'] = safety
        visibility['replay_diagnostics'] = diagnostics
        return visibility

    def _particle_segmentation_safety_check(self) -> dict[str, Any]:
        if self.particle_systems_enabled:
            raise EgoRendererError('particle segmentation safety check requires suppression mode')
        from omnigibson.object_states import ParticleSink, ParticleSource
        remaining_states = []
        remaining_abilities = []
        for obj in getattr(getattr(self._env, 'scene', None), 'objects', []) or []:
            name = str(getattr(obj, 'name', ''))
            states = getattr(obj, 'states', {}) or {}
            abilities = getattr(obj, 'abilities', {}) or {}
            if ParticleSource in states or ParticleSink in states:
                remaining_states.append(name)
            if 'particleSource' in abilities or 'particleSink' in abilities:
                remaining_abilities.append(name)
        water_instancers = self._water_particle_instancer_paths()
        if remaining_states or remaining_abilities or water_instancers:
            raise EgoRendererError('particle-suppressed replay is unsafe: states={}, abilities={}, instancers={}'.format(remaining_states, remaining_abilities, water_instancers))
        return {'safe': True, 'remaining_particle_state_objects': [], 'remaining_particle_ability_objects': [], 'water_instancer_paths': []}

    def _water_particle_instancer_paths(self) -> list[str]:
        stage = getattr(getattr(self, '_og', None), 'sim', None)
        stage = getattr(stage, 'stage', None)
        if stage is None:
            return []
        paths = []
        try:
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                lowered = path.casefold()
                if 'waterinstancer' in lowered or ('water' in lowered and 'pointinstancer' in lowered):
                    paths.append(path)
        except Exception as exc:
            raise EgoRendererError('failed to inspect replay particle prims') from exc
        return sorted(set(paths))

    def _init_visual_state_sync(self) -> None:
        if self._env is None or self._og is None:
            return
        try:
            from scoper.benchmark.behavior.execution_core.visual_state_sync import OmniGibsonVisualStateSynchronizer, VisualStateSyncConfig
            self._visual_sync = OmniGibsonVisualStateSynchronizer(og_module=self._og, env=self._env, og_object_by_name=self._visual_object_by_name(), config=VisualStateSyncConfig(require_visual_state_sync=False, water_shutdown_render_steps=self.water_shutdown_render_steps, enable_particle_source_visuals=self.native_particle_source_visuals_enabled, enable_particle_source_stream_visuals=self.particle_source_stream_visuals_enabled), on_object_added=self._on_visual_sync_object_added)
        except Exception:
            self._visual_sync = None

    def _on_visual_sync_object_added(self, name: str, obj: Any) -> None:
        safe_name = _og_safe_name(str(name))
        self._raw_name_to_safe_name[str(name)] = safe_name
        if obj is not None:
            try:
                self._object_offsets.pop(safe_name, None)
            except Exception:
                pass

    def _register_cut_visual_replacements(self, mutations: Sequence[Mapping[str, Any]]) -> None:
        for mutation in mutations or ():
            if not isinstance(mutation, Mapping):
                continue
            if mutation.get('state') != 'sliced' or not bool(mutation.get('success')):
                continue
            source_name = str(mutation.get('object') or '')
            replacement_names = [str(name) for name in mutation.get('registered_added_objects') or [] if str(name)]
            if not source_name or not replacement_names:
                continue
            self._register_visual_replacement(source_name, replacement_names)

    def _register_visual_replacement(self, source_name: str, replacement_names: Sequence[str]) -> None:
        source_keys = self._replacement_keys_for_object(source_name)
        deduped = list(dict.fromkeys((str(name) for name in replacement_names if str(name))))
        if not deduped:
            return
        for key in source_keys:
            self._visual_replacements[key] = list(deduped)
        for name in deduped:
            self._raw_name_to_safe_name[str(name)] = _og_safe_name(str(name))
        offsets = self._replacement_offsets(deduped)
        for key in source_keys:
            self._visual_replacement_offsets[key] = dict(offsets)

    def _replacement_keys_for_object(self, object_name: str) -> list[str]:
        raw = str(object_name)
        keys = [raw, _og_safe_name(raw)]
        return list(dict.fromkeys((key for key in keys if key)))

    def _replacement_names_for_object(self, object_name: str) -> list[str]:
        for key in self._replacement_keys_for_object(object_name):
            names = self._visual_replacements.get(key)
            if names:
                return list(names)
        return []

    def _replacement_offsets(self, replacement_names: Sequence[str]) -> dict[str, np.ndarray]:
        centers: dict[str, np.ndarray] = {}
        for replacement_name in replacement_names:
            obj = self._object_by_name(_og_safe_name(str(replacement_name)))
            if obj is None:
                continue
            try:
                centers[str(replacement_name)] = _to_numpy(getattr(obj, 'aabb_center')).astype(float)
            except Exception:
                continue
        if not centers:
            return {}
        centroid = np.mean(np.stack(list(centers.values()), axis=0), axis=0)
        return {name: center - centroid for name, center in centers.items()}

    def _set_replacement_object_centers(self, source_name: str, replacement_names: Sequence[str], center_np: np.ndarray, *, yaw: float | None, orientation: Any | None=None, verify_readback: bool) -> dict[str, Any]:
        offsets = self._visual_replacement_offsets.get(source_name) or self._visual_replacement_offsets.get(_og_safe_name(source_name), {})
        replacement_diagnostics: list[dict[str, Any]] = []
        for replacement_name in replacement_names:
            offset = offsets.get(str(replacement_name))
            desired_center = center_np + (offset if offset is not None else np.zeros(3, dtype=float))
            diagnostic = self._set_single_object_center(str(replacement_name), desired_center, yaw=yaw, orientation=orientation, verify_readback=verify_readback)
            replacement_diagnostics.append(diagnostic)
        success = bool(replacement_diagnostics) and all((bool(item.get('success')) for item in replacement_diagnostics))
        return {'success': success, 'reason': 'visual_replacements_set_position_orientation' if success else 'visual_replacement_sync_failed', 'visual_replacement_for': str(source_name), 'visual_replacements': list(replacement_names), 'replacement_diagnostics': replacement_diagnostics, 'actual_center': _mean_actual_center(replacement_diagnostics), 'center_error_m': _mean_center_error(center_np, replacement_diagnostics) if success else None}

    def _set_single_object_center(self, object_name: str, center: Any, *, yaw: float | None, orientation: Any | None=None, verify_readback: bool) -> dict[str, Any]:
        safe_name = _og_safe_name(str(object_name))
        diagnostic: dict[str, Any] = {'success': False, 'object_name': str(object_name), 'safe_name': safe_name, 'desired_center': None, 'actual_center': None, 'center_error_m': None, 'desired_root_position': None, 'actual_root_position': None, 'actual_root_orientation_xyzw': None, 'actual_root_yaw': None, 'yaw': None if yaw is None else float(yaw), 'orientation': _jsonable_orientation(orientation), 'verify_readback': bool(verify_readback), 'reason': ''}
        try:
            center_np = _to_numpy(center).astype(float)
            diagnostic['desired_center'] = center_np.tolist()
        except Exception as exc:
            diagnostic['reason'] = 'invalid_center'
            diagnostic['error'] = repr(exc)
            return diagnostic
        obj = self._object_by_name(safe_name)
        if obj is None:
            diagnostic['reason'] = 'object_not_found'
            return diagnostic
        visible = self._object_visible(obj)
        diagnostic['before_visible'] = visible
        if visible is False:
            diagnostic['reason'] = 'object_not_visible'
            return diagnostic
        if safe_name not in self._object_offsets:
            offset = self._object_aabb_offset(safe_name, obj)
            if offset is None:
                diagnostic['reason'] = 'aabb_offset_unavailable'
                return diagnostic
            self._object_offsets[safe_name] = offset
            self._object_offset_orientations[safe_name] = self._current_orientation(obj)
        root_position = center_np - self._object_offsets[safe_name]
        diagnostic['desired_root_position'] = root_position.astype(float).tolist()
        orientation_to_write = self._resolved_orientation(obj, yaw=yaw, orientation=orientation)
        diagnostic['desired_root_orientation_xyzw'] = list(orientation_to_write)
        try:
            obj.set_position_orientation(position=root_position.tolist(), orientation=list(orientation_to_write), frame='scene')
        except Exception as exc:
            diagnostic['reason'] = 'set_position_orientation_error'
            diagnostic['error'] = repr(exc)
            return diagnostic
        if not bool(verify_readback):
            if yaw is not None or orientation is not None:
                self._object_offsets.pop(safe_name, None)
                self._object_offset_orientations.pop(safe_name, None)
            diagnostic['success'] = True
            diagnostic['reason'] = 'set_position_orientation_unverified'
            return diagnostic
        self._flush_render_for_readback()
        record = self._object_pose_record(safe_name, obj)
        if record is None:
            diagnostic['reason'] = 'readback_center_unavailable'
            return diagnostic
        diagnostic.update(_pose_record_diagnostic(record))
        actual_center = np.asarray(record['aabb_center'], dtype=float)
        diagnostic['actual_center'] = actual_center.tolist()
        diagnostic['center_error_m'] = float(np.linalg.norm(actual_center - center_np))
        self._object_offsets[safe_name] = np.asarray(record['aabb_offset'], dtype=float)
        self._object_offset_orientations[safe_name] = tuple(record['root_orientation_xyzw'])
        diagnostic['success'] = True
        diagnostic['reason'] = 'set_position_orientation'
        return diagnostic

    def _remove_debug_prim(self, stage: Any, prim_path: str) -> None:
        try:
            context = self._og.sim.editing_usd() if self._og is not None and hasattr(self._og.sim, 'editing_usd') else nullcontext()
            with context:
                prim = stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    stage.RemovePrim(prim_path)
        except Exception:
            pass

    def _install_mesh_occupancy_overlay(self, stage: Any, root_path: str, *, free: np.ndarray, resolution: float, map_size: int, z: float, navigation_objects: list[Any] | tuple[Any, ...]) -> None:
        context = self._og.sim.editing_usd() if hasattr(self._og.sim, 'editing_usd') else nullcontext()
        with context:
            _define_xform(stage, root_path)
            _define_quad_overlay(stage, f'{root_path}/mesh_occupied_cells', _cell_quads(~free, resolution=resolution, map_size=map_size, z=z), color=(1.0, 0.05, 0.02), opacity=0.34)
            _define_quad_overlay(stage, f'{root_path}/map_extent_border', _map_border_quads(resolution=resolution, map_size=map_size, z=z + 0.04), color=(1.0, 0.92, 0.0), opacity=0.78)
            closed_door_quads = []
            open_door_quads = []
            for obj in navigation_objects or ():
                states = getattr(obj, 'states', {}) or {}
                quad = _aabb_quad(obj, z=z + 0.08)
                if not quad:
                    continue
                if isinstance(states, dict) and states.get('open') is True:
                    open_door_quads.append(quad)
                else:
                    closed_door_quads.append(quad)
            _define_quad_overlay(stage, f'{root_path}/closed_door_aabbs', closed_door_quads, color=(0.0, 0.2, 1.0), opacity=0.55)
            _define_quad_overlay(stage, f'{root_path}/open_door_aabbs', open_door_quads, color=(0.0, 0.85, 1.0), opacity=0.5)

    def _install_mesh_traversability_diff_overlay(self, stage: Any, root_path: str, *, mesh_free: np.ndarray, reference_free: np.ndarray, resolution: float, map_size: int, z: float) -> None:
        context = self._og.sim.editing_usd() if hasattr(self._og.sim, 'editing_usd') else nullcontext()
        with context:
            _define_xform(stage, root_path)
            _define_quad_overlay(stage, f'{root_path}/mesh_free_scene_blocked', _cell_quads(mesh_free & ~reference_free, resolution=resolution, map_size=map_size, z=z), color=(0.0, 0.72, 1.0), opacity=0.42)
            _define_quad_overlay(stage, f'{root_path}/mesh_blocked_scene_free', _cell_quads(~mesh_free & reference_free, resolution=resolution, map_size=map_size, z=z + 0.02), color=(1.0, 0.45, 0.0), opacity=0.45)
            _define_quad_overlay(stage, f'{root_path}/map_extent_border', _map_border_quads(resolution=resolution, map_size=map_size, z=z + 0.04), color=(1.0, 0.92, 0.0), opacity=0.78)

    def _visual_object_by_name(self) -> dict[str, Any]:
        if self._env is None:
            return {}
        objects_by_safe = {str(getattr(obj, 'name', '')): obj for obj in getattr(getattr(self._env, 'scene', None), 'objects', []) or [] if str(getattr(obj, 'name', ''))}
        out = dict(objects_by_safe)
        for alias_safe, target_safe in self._object_aliases.items():
            target_obj = objects_by_safe.get(self._resolve_alias_target_safe(alias_safe))
            if target_obj is not None:
                out[alias_safe] = target_obj
        for raw_name, safe_name in self._raw_name_to_safe_name.items():
            target_safe = self._resolve_alias_target_safe(safe_name)
            target_obj = objects_by_safe.get(target_safe)
            if target_obj is not None:
                out[raw_name] = target_obj
                out[safe_name] = target_obj
        return out

    def _sync_initial_boolean_states(self, core: Any) -> None:
        if self._visual_sync is None or core is None:
            return
        for name, obj in getattr(core, 'objects', {}).items():
            states = getattr(obj, 'states', {}) or {}
            if 'open' in states:
                self._visual_sync.set_open_visual(str(name), bool(states.get('open')), reason='initial_open_state')
            if 'toggled_on' in states:
                self.set_toggle_visual(str(name), bool(states.get('toggled_on')), reason='initial_toggle_state')
            if states.get('dirty') is True:
                self._visual_sync.set_dirty_visual(str(name), bool(states.get('dirty')), reason='initial_dirty_state')

    def _capture_container_interiors(self) -> dict[str, dict[str, list[float]]]:
        try:
            from omnigibson.object_states import ContainedParticles
        except Exception:
            return {}
        interiors_by_safe: dict[str, dict[str, list[float]]] = {}
        for safe_name, og_obj in self._visual_object_by_name().items():
            if safe_name != _og_safe_name(safe_name):
                continue
            if og_obj is None or ContainedParticles not in getattr(og_obj, 'states', {}):
                continue
            try:
                link = og_obj.states[ContainedParticles].link
                lo, hi = link.aabb
                lo_np = _to_numpy(lo).astype(float)
                hi_np = _to_numpy(hi).astype(float)
            except Exception:
                continue
            center = ((lo_np + hi_np) * 0.5).tolist()
            half_extent = ((hi_np - lo_np) * 0.5).tolist()
            interiors_by_safe[str(safe_name)] = {'center': [float(v) for v in center], 'half_extent': [float(v) for v in half_extent]}
        interiors: dict[str, dict[str, list[float]]] = {}
        for raw_name, safe_name in self._raw_name_to_safe_name.items():
            target_safe = self._resolve_alias_target_safe(safe_name)
            interior = interiors_by_safe.get(target_safe)
            if interior is not None:
                interiors[str(raw_name)] = interior
        for safe_name, interior in interiors_by_safe.items():
            interiors.setdefault(safe_name, interior)
        return interiors

def _define_xform(stage: Any, prim_path: str) -> None:
    from pxr import UsdGeom
    UsdGeom.Xform.Define(stage, prim_path)

def _define_quad_overlay(stage: Any, prim_path: str, quads: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]], *, color: tuple[float, float, float], opacity: float) -> None:
    if not quads:
        return
    from pxr import Gf, UsdGeom, UsdShade
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    points = []
    face_counts = []
    face_indices = []
    for quad in quads:
        base = len(points)
        points.extend((Gf.Vec3f(*point) for point in quad))
        face_counts.append(4)
        face_indices.extend([base, base + 1, base + 2, base + 3])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    gprim = UsdGeom.Gprim(mesh.GetPrim())
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
    material_path = f'{prim_path}_material'
    _define_preview_material(stage, material_path, color=color, opacity=opacity)
    material_prim = stage.GetPrimAtPath(material_path)
    if material_prim and material_prim.IsValid():
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(UsdShade.Material(material_prim), UsdShade.Tokens.strongerThanDescendants)

def _define_preview_material(stage: Any, material_path: str, *, color: tuple[float, float, float], opacity: float) -> None:
    from pxr import Gf, Sdf, UsdShade
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f'{material_path}/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')

    def set_input(name: str, value_type: Any, value: Any) -> None:
        inp = shader.GetInput(name)
        if not inp:
            inp = shader.CreateInput(name, value_type)
        inp.Set(value)
    set_input('diffuseColor', Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color))
    set_input('opacity', Sdf.ValueTypeNames.Float, float(opacity))
    set_input('roughness', Sdf.ValueTypeNames.Float, 0.45)
    set_input('metallic', Sdf.ValueTypeNames.Float, 0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')

def _cell_quads(mask: np.ndarray, *, resolution: float, map_size: int, z: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    rows, cols = np.where(mask)
    half = float(map_size) * float(resolution) * 0.5
    quads = []
    for row, col in zip(rows, cols):
        x0 = float(col) * resolution - half
        x1 = x0 + resolution
        y0 = float(row) * resolution - half
        y1 = y0 + resolution
        quads.append(((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)))
    return quads

def _map_border_quads(*, resolution: float, map_size: int, z: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    half = float(map_size) * float(resolution) * 0.5
    width = max(float(resolution) * 0.5, 0.04)
    return [((-half, -half, z), (half, -half, z), (half, -half + width, z), (-half, -half + width, z)), ((-half, half - width, z), (half, half - width, z), (half, half, z), (-half, half, z)), ((-half, -half, z), (-half + width, -half, z), (-half + width, half, z), (-half, half, z)), ((half - width, -half, z), (half, -half, z), (half, half, z), (half - width, half, z))]

def _aabb_quad(obj: Any, *, z: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    try:
        center = getattr(obj, 'center')
        extent = getattr(obj, 'extent')
        x0 = float(center[0]) - float(extent[0]) * 0.5
        x1 = float(center[0]) + float(extent[0]) * 0.5
        y0 = float(center[1]) - float(extent[1]) * 0.5
        y1 = float(center[1]) + float(extent[1]) * 0.5
    except Exception:
        return None
    return ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))

def _mesh_alignment_oblique_pose(extent_m: float, overlay_z: float) -> Any:
    from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose, yaw_towards
    distance = max(float(extent_m), 5.0)
    x = -0.65 * distance
    y = -0.85 * distance
    z = max(4.8, 0.72 * distance + overlay_z)
    target_xy = (0.0, 0.0)
    yaw = yaw_towards((x, y), target_xy)
    horizontal = math.dist((x, y), target_xy)
    pitch = math.atan2(float(overlay_z) - z, horizontal)
    return EgoPose(x, y, z=z, yaw=yaw, pitch=pitch)

def _downsample(items: List[Any], max_count: int) -> List[Any]:
    if max_count <= 0 or len(items) <= max_count:
        return list(items)
    idx = np.linspace(0, len(items) - 1, max_count).round().astype(int)
    seen = set()
    out = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out

def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.array(value)

def _quat_xyzw_to_yaw(orientation: Sequence[float] | None) -> float | None:
    if orientation is None or len(orientation) < 4:
        return None
    x, y, z, w = (float(v) for v in orientation[:4])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))

def _jsonable_orientation(orientation: Any | None) -> list[float] | None:
    if orientation is None:
        return None
    try:
        values = _to_numpy(orientation).astype(float).tolist()
        if len(values) >= 4:
            return [float(v) for v in values[:4]]
    except Exception:
        return None
    return None

def _to_jsonable_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return repr(value)

def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def _ego_pose_mapping(pose: Any) -> dict[str, float] | None:
    fields = ('x', 'y', 'z', 'yaw', 'pitch')
    try:
        if isinstance(pose, Mapping):
            return {'x': float(pose.get('x', 0.0)), 'y': float(pose.get('y', 0.0)), 'z': float(pose.get('z', 1.55)), 'yaw': float(pose.get('yaw', 0.0)), 'pitch': float(pose.get('pitch', 0.0))}
        return {field: float(getattr(pose, field)) for field in fields}
    except (AttributeError, TypeError, ValueError):
        return None

def _object_aabb_extent(obj: Any) -> np.ndarray:
    try:
        return _to_numpy(getattr(obj, 'aabb_extent')).astype(float)[:3]
    except Exception:
        pass
    aabb = getattr(obj, 'aabb', None)
    if callable(aabb):
        try:
            aabb = aabb()
        except Exception:
            aabb = None
    if aabb is not None:
        try:
            arr = _to_numpy(aabb).astype(float)
            if arr.shape[0] >= 2:
                return (arr[1][:3] - arr[0][:3]).astype(float)
        except Exception:
            pass
    return np.zeros(3, dtype=float)

def _pose_record_diagnostic(record: Mapping[str, Any]) -> dict[str, Any]:
    return {'actual_root_position': list(record.get('root_position') or []), 'actual_root_orientation_xyzw': list(record.get('root_orientation_xyzw') or []), 'actual_root_yaw': record.get('root_yaw'), 'actual_aabb_center': list(record.get('aabb_center') or []), 'actual_aabb_extent': list(record.get('aabb_extent') or []), 'actual_aabb_offset': list(record.get('aabb_offset') or [])}

def _renderer_name_for_core_object(name: str, obj: Any) -> str:
    states = getattr(obj, 'states', {}) or {}
    if isinstance(states, Mapping):
        for key in ('source_scene_object_id', 'source_object_id'):
            source_object_id = states.get(key)
            if source_object_id and (not str(source_object_id).startswith('spawnable:')):
                return str(source_object_id)
    return str(name)

def _update_core_object_from_pose_diagnostic(obj: Any, diagnostic: Mapping[str, Any] | None) -> None:
    if obj is None or not isinstance(diagnostic, Mapping) or (not bool(diagnostic.get('success'))):
        return
    center = diagnostic.get('actual_aabb_center') or diagnostic.get('actual_center')
    extent = diagnostic.get('actual_aabb_extent')
    root_position = diagnostic.get('actual_root_position')
    root_orientation = diagnostic.get('actual_root_orientation_xyzw')
    if center is not None:
        try:
            obj.center = tuple((float(v) for v in center[:3]))
        except Exception:
            pass
    if extent is not None:
        try:
            obj.extent = tuple((float(v) for v in extent[:3]))
        except Exception:
            pass
    if root_position is not None:
        try:
            obj.root_position = tuple((float(v) for v in root_position[:3]))
        except Exception:
            pass
    if root_orientation is not None:
        try:
            obj.root_orientation_xyzw = tuple((float(v) for v in root_orientation[:4]))
        except Exception:
            pass
    if diagnostic.get('actual_root_yaw') is not None:
        try:
            obj.root_yaw = float(diagnostic['actual_root_yaw'])
        except Exception:
            pass
    try:
        obj.pose_source = 'renderer_readback'
    except Exception:
        pass

def _mean_actual_center(diagnostics: Sequence[Mapping[str, Any]]) -> list[float] | None:
    centers = []
    for diagnostic in diagnostics:
        actual = diagnostic.get('actual_center')
        if actual is None:
            continue
        try:
            centers.append(np.asarray(actual, dtype=float))
        except Exception:
            continue
    if not centers:
        return None
    return np.mean(np.stack(centers, axis=0), axis=0).astype(float).tolist()

def _mean_center_error(center: np.ndarray, diagnostics: Sequence[Mapping[str, Any]]) -> float | None:
    actual_center = _mean_actual_center(diagnostics)
    if actual_center is None:
        return None
    try:
        return float(np.linalg.norm(np.asarray(actual_center, dtype=float) - np.asarray(center, dtype=float)))
    except Exception:
        return None

def _held_pose_field(held_pose: Any, key: str, default: Any=None) -> Any:
    if isinstance(held_pose, Mapping):
        return held_pose.get(key, default)
    return getattr(held_pose, key, default)

def to_jsonable_held_pose(held_pose: Any) -> dict[str, Any]:
    if held_pose is None:
        return {}
    if isinstance(held_pose, Mapping):
        items = held_pose.items()
    else:
        keys = ('object_name', 'visual_name', 'center', 'yaw', 'pitch', 'distance', 'lateral_offset', 'vertical_offset')
        items = ((key, getattr(held_pose, key)) for key in keys if hasattr(held_pose, key))
    out: dict[str, Any] = {}
    for key, value in items:
        if hasattr(value, 'tolist'):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        out[str(key)] = value
    return out

def _segmentation_label(label_mapping: Mapping[Any, Any], label_id: int) -> str:
    for key in (label_id, str(label_id)):
        if key in label_mapping:
            return str(label_mapping[key])
    return ''

def _ignored_visibility_label(label: str) -> bool:
    text = str(label or '').strip()
    lowered = text.lower()
    if lowered in {'', 'background', 'invalid', 'unlabelled', 'unlabeled', 'groundplane', 'ground_plane'}:
        return True
    return lowered.startswith('/world/ground_plane')

def _camera_pose_object(pose: Any) -> Any:
    if not isinstance(pose, Mapping):
        return pose
    return SimpleNamespace(x=float(pose.get('x', 0.0)), y=float(pose.get('y', 0.0)), z=float(pose.get('z', 1.55)), yaw=float(pose.get('yaw', 0.0)), pitch=float(pose.get('pitch', 0.0)))

def _visibility_label_candidates(label: str) -> set[str]:
    text = str(label or '').strip()
    candidates = {text} if text else set()
    parts = [part for part in text.replace('\\', '/').split('/') if part]
    candidates.update(parts)
    for part in parts:
        if 'Instancer' in part:
            candidates.add(part.split('Instancer', 1)[0])
        if 'Particle' in part:
            candidates.add(part.split('Particle', 1)[0])
        if '.' in part:
            candidates.add(part.split('.', 1)[0])
    return {candidate for candidate in candidates if candidate}

def _og_safe_name(instance_name: str) -> str:
    return instance_name.replace('.', '_').replace('-', '_')
__all__ = ['EgoRenderer', 'EgoRendererError']

def _infer_og_assets_root() -> Path | None:
    for env_name in ('OMNIGIBSON_ASSET_PATH', 'OMNIGIBSON_DATASET_PATH', 'OG_DATASET_PATH'):
        value = os.environ.get(env_name)
        if value:
            path = _normalize_og_assets_root(Path(value))
            if path is not None:
                return path
    try:
        from omnigibson.macros import gm
    except Exception:
        gm = None
    if gm is not None:
        for attr in ('DATASET_PATH', 'ASSET_PATH', 'DATA_PATH'):
            value = getattr(gm, attr, None)
            if value:
                path = _normalize_og_assets_root(Path(str(value)))
                if path is not None:
                    return path
    repo_assets = behavior_assets_root()
    if _valid_og_assets_root(repo_assets):
        return repo_assets
    for attr in ('DATASET_PATH', 'ASSET_PATH', 'DATA_PATH'):
        value = os.environ.get(attr)
        if value:
            path = _normalize_og_assets_root(Path(value))
            if path is not None:
                return path
    return None

def _normalize_og_assets_root(path: Path) -> Path | None:
    candidates = [path, path / 'behavior-1k-assets', path / 'datasets' / 'behavior-1k-assets']
    for candidate in candidates:
        if _valid_og_assets_root(candidate):
            return candidate
    return None

def _valid_og_assets_root(path: Path) -> bool:
    return path.exists() and (path / 'scenes').is_dir() and (path / 'objects').is_dir()

def _is_door_like_og_object(og_obj: Any) -> bool:
    category = str(getattr(og_obj, 'category', '') or '').lower()
    name = str(getattr(og_obj, 'name', '') or '').lower()
    return 'door' in category or 'door' in name

def _og_open_value(og_obj: Any, object_states: Any) -> bool:
    try:
        return bool(og_obj.states[object_states.Open].get_value())
    except Exception:
        return False

def _openable_joint_interaction_faces_for_og_object(og_obj: Any) -> list[dict[str, Any]]:
    joints_info = _openable_relevant_joints_info(og_obj)
    if not joints_info:
        return []
    both_sides, relevant_joints, joint_directions = joints_info
    if not relevant_joints:
        return []
    try:
        obj_center = _to_numpy(getattr(og_obj, 'aabb_center')).astype(float)
    except Exception:
        return []
    if obj_center.shape[0] < 3:
        return []
    faces: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float, float]] = set()
    for idx, joint in enumerate(relevant_joints):
        link = _child_link_for_joint(og_obj, joint)
        if link is None:
            continue
        bounds = _link_aabb_bounds(link)
        if bounds is None:
            continue
        joint_direction = 1
        if idx < len(joint_directions):
            try:
                joint_direction = int(joint_directions[idx])
            except Exception:
                joint_direction = 1
        for face in _interaction_faces_from_link_bounds(og_obj=og_obj, joint=joint, link=link, obj_center=obj_center, link_bounds=bounds, both_sides=bool(both_sides), joint_direction=joint_direction):
            key = (round(float(face['center'][0]), 4), round(float(face['center'][1]), 4), round(float(face['center'][2]), 4), round(float(face['normal_xy'][0]), 4), round(float(face['normal_xy'][1]), 4))
            if key in seen:
                continue
            seen.add(key)
            faces.append(face)
    return faces

def _openable_relevant_joints_info(og_obj: Any) -> tuple[bool, list[Any], list[int]] | None:
    try:
        from omnigibson import object_states
        state = getattr(og_obj, 'states', {}).get(object_states.Open)
        info = getattr(state, 'relevant_joints_info', None)
        if info is not None and len(info) >= 3 and info[1]:
            return (bool(info[0]), list(info[1]), list(info[2]))
    except Exception:
        pass
    try:
        from omnigibson.object_states.open_state import _get_relevant_joints
        both_sides, relevant_joints, joint_directions = _get_relevant_joints(og_obj)
    except Exception:
        return None
    if not relevant_joints:
        return None
    return (bool(both_sides), list(relevant_joints), list(joint_directions))

def _child_link_for_joint(og_obj: Any, joint: Any) -> Any | None:
    body1 = str(getattr(joint, 'body1', '') or '')
    if not body1:
        return None
    links = getattr(og_obj, 'links', {}) or {}
    body1_name = body1.rsplit('/', 1)[-1]
    for link_name, link in links.items():
        link_path = str(getattr(link, 'prim_path', '') or '')
        if link_path == body1 or str(link_name) == body1_name:
            return link
    return links.get(body1_name)

def _link_aabb_bounds(link: Any) -> tuple[np.ndarray, np.ndarray] | None:
    for attr in ('visual_aabb', 'aabb'):
        try:
            lower, upper = getattr(link, attr)
            lower_np = _to_numpy(lower).astype(float)
            upper_np = _to_numpy(upper).astype(float)
        except Exception:
            continue
        if lower_np.shape[0] >= 3 and upper_np.shape[0] >= 3 and np.all(np.isfinite(lower_np[:3])) and np.all(np.isfinite(upper_np[:3])):
            return (lower_np[:3], upper_np[:3])
    return None

def _interaction_faces_from_link_bounds(*, og_obj: Any, joint: Any, link: Any, obj_center: np.ndarray, link_bounds: tuple[np.ndarray, np.ndarray], both_sides: bool, joint_direction: int) -> list[dict[str, Any]]:
    lower, upper = link_bounds
    center = 0.5 * (lower + upper)
    extent = np.maximum(upper - lower, 0.0)
    if extent.shape[0] < 3 or max(float(extent[0]), float(extent[1])) <= 1e-06:
        return []
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    candidates = [{'center': [float(upper[0]), cy, cz], 'normal_xy': [1.0, 0.0], 'axis': 'x', 'sign': 1}, {'center': [float(lower[0]), cy, cz], 'normal_xy': [-1.0, 0.0], 'axis': 'x', 'sign': -1}, {'center': [cx, float(upper[1]), cz], 'normal_xy': [0.0, 1.0], 'axis': 'y', 'sign': 1}, {'center': [cx, float(lower[1]), cz], 'normal_xy': [0.0, -1.0], 'axis': 'y', 'sign': -1}]
    keep: list[dict[str, Any]]
    if both_sides or _is_door_like_og_object(og_obj):
        axis = 'x' if float(extent[0]) <= float(extent[1]) else 'y'
        keep = [face for face in candidates if face['axis'] == axis]
    else:
        prismatic_face = _prismatic_axis_aligned_face(candidates, og_obj=og_obj, joint=joint, joint_direction=joint_direction)
        keep = [prismatic_face] if prismatic_face is not None else [max(candidates, key=lambda face: _face_outward_score(face, obj_center))]
    joint_name = str(getattr(joint, 'joint_name', '') or getattr(joint, 'name', '') or '')
    link_name = str(getattr(link, 'name', '') or '')
    joint_type = str(getattr(joint, 'joint_type', '') or '')
    out = []
    for face in keep:
        out.append({'center': face['center'], 'normal_xy': face['normal_xy'], 'source': 'og_openable_joint_link_face', 'joint_name': joint_name, 'joint_type': joint_type, 'joint_direction': int(joint_direction), 'link_name': link_name})
    return out

def _prismatic_axis_aligned_face(candidates: list[dict[str, Any]], *, og_obj: Any, joint: Any, joint_direction: int) -> dict[str, Any] | None:
    if 'Prismatic' not in str(getattr(joint, 'joint_type', '') or ''):
        return None
    axis_xy = _joint_axis_xy_world(og_obj, joint, joint_direction=joint_direction)
    if axis_xy is None:
        return None
    return max(candidates, key=lambda face: float(np.dot(np.asarray(face['normal_xy'], dtype=float), axis_xy)))

def _joint_axis_xy_world(og_obj: Any, joint: Any, *, joint_direction: int) -> np.ndarray | None:
    axis_token = str(getattr(joint, 'axis', 'X') or 'X').upper()
    axis_local = {'X': np.array([1.0, 0.0, 0.0], dtype=float), 'Y': np.array([0.0, 1.0, 0.0], dtype=float), 'Z': np.array([0.0, 0.0, 1.0], dtype=float)}.get(axis_token)
    if axis_local is None:
        return None
    try:
        from omnigibson import lazy
        from omnigibson.utils.transform_utils_np import quat_apply
        joint_quat = lazy.isaacsim.core.utils.rotations.gf_quat_to_np_array(joint.get_attribute('physics:localRot0'))[[1, 2, 3, 0]]
        parent_link = _parent_link_for_joint(og_obj, joint)
        if parent_link is None:
            parent_link = getattr(og_obj, 'root_link', None)
        if parent_link is None:
            return None
        _, parent_quat_raw = parent_link.get_position_orientation()
        parent_quat = _to_numpy(parent_quat_raw).astype(float)
        axis_world = quat_apply(parent_quat, quat_apply(joint_quat, axis_local))
    except Exception:
        return None
    axis_xy = np.asarray(axis_world[:2], dtype=float) * (1.0 if int(joint_direction) >= 0 else -1.0)
    norm = float(np.linalg.norm(axis_xy))
    if norm <= 1e-09:
        return None
    return axis_xy / norm

def _parent_link_for_joint(og_obj: Any, joint: Any) -> Any | None:
    body0 = str(getattr(joint, 'body0', '') or '')
    if not body0:
        return None
    links = getattr(og_obj, 'links', {}) or {}
    body0_name = body0.rsplit('/', 1)[-1]
    for link_name, link in links.items():
        link_path = str(getattr(link, 'prim_path', '') or '')
        if link_path == body0 or str(link_name) == body0_name:
            return link
    return links.get(body0_name)

def _og_state_names(og_obj: Any) -> list[str]:
    names = []
    for state_cls in getattr(og_obj, 'states', {}) or {}:
        name = getattr(state_cls, '__name__', None) or str(state_cls)
        names.append(str(name))
    return sorted(names)

def _auto_expose_rgb(rgb: Any, *, enabled: bool=True) -> np.ndarray:
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] < 3:
        return array
    if not enabled:
        return np.ascontiguousarray(array[:, :, :3])
    rgb3 = np.ascontiguousarray(array[:, :, :3])
    mean = float(np.mean(rgb3))
    p1 = float(np.percentile(rgb3, 1.0))
    p99 = float(np.percentile(rgb3, 99.0))
    dynamic = p99 - p1
    if mean >= 45.0 and dynamic >= 24.0:
        return rgb3
    gain = min(6.0, max(1.0, 95.0 / max(mean, 1.0)))
    exposed = rgb3.astype(np.float32) * gain
    if dynamic < 18.0 and p99 > p1 + 1e-06:
        low = p1
        high = max(p99, p1 + 1.0)
        stretched = (rgb3.astype(np.float32) - low) * (120.0 / (high - low)) + 40.0
        exposed = 0.65 * exposed + 0.35 * stretched
    return np.clip(exposed, 0, 255).astype(np.uint8)

def _face_outward_score(face: dict[str, Any], obj_center: np.ndarray) -> float:
    center = np.asarray(face['center'], dtype=float)
    normal = np.asarray(face['normal_xy'], dtype=float)
    vector = center[:2] - obj_center[:2]
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-09:
        return 0.0
    return float(np.dot(normal, vector / distance)) + 0.02 * distance

def _aabb_interaction_faces_for_og_object(og_obj: Any) -> list[dict[str, Any]]:
    try:
        center = _to_numpy(getattr(og_obj, 'aabb_center')).astype(float)
        extent = _to_numpy(getattr(og_obj, 'aabb_extent')).astype(float)
    except Exception:
        return []
    if center.shape[0] < 3 or extent.shape[0] < 3:
        return []
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    hx, hy = (max(float(extent[0]) * 0.5, 0.0), max(float(extent[1]) * 0.5, 0.0))
    faces = [{'center': [cx + hx, cy, cz], 'normal_xy': [1.0, 0.0], 'source': 'og_openable_aabb_face'}, {'center': [cx - hx, cy, cz], 'normal_xy': [-1.0, 0.0], 'source': 'og_openable_aabb_face'}, {'center': [cx, cy + hy, cz], 'normal_xy': [0.0, 1.0], 'source': 'og_openable_aabb_face'}, {'center': [cx, cy - hy, cz], 'normal_xy': [0.0, -1.0], 'source': 'og_openable_aabb_face'}]
    if not _is_door_like_og_object(og_obj):
        return faces
    return faces[:2] if hx <= hy else faces[2:]
