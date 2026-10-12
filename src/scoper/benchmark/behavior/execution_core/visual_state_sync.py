from __future__ import annotations
from scoper.resources import behavior_assets_root
import json
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import traceback
import numpy as np
from scoper.benchmark.behavior.execution_core.trace import ActionResult
from scoper.benchmark.behavior.execution_core.visual_mesh_water import ContainerWaterVisual, VisualMeshWaterConfig, create_visual_mesh_water, set_visual_mesh_water_visibility
REALISTIC_WATER_COLOR: tuple[float, float, float] = (0.16, 0.45, 0.78)
DEEP_FIRE_COLORMAP: tuple[tuple[float, float, float, float], ...] = ((0.2, 0.02, 0.004, 0.2), (0.85, 0.09, 0.01, 0.75), (2.2, 0.32, 0.03, 0.9), (6.5, 1.1, 0.09, 0.95), (24.0, 6.0, 0.6, 0.92), (90.0, 45.0, 7.0, 0.85))

class _CutMetadataObjectProxy:

    def __init__(self, obj: Any, metadata: Mapping[str, Any]) -> None:
        self._obj = obj
        self.metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

@dataclass
class VisualStateSyncConfig:
    dirty_visual_system: str = 'stain'
    visual_state_sync_steps: int = 4
    open_visual_settle_renders: int = 8
    require_visual_state_sync: bool = True
    fire_visual_steps: int = 18
    fire_shutdown_steps: int = 8
    fire_shutdown_render_steps: int = 4
    deepen_fire_color: bool = True
    fire_force_clear_on_shutdown: bool = True
    water_source_system: str = 'water'
    water_visual_steps: int = 30
    water_shutdown_steps: int = 6
    water_shutdown_render_steps: int = 50
    water_shutdown_min_render_steps: int = 8
    enable_particle_source_visuals: bool = True
    enable_particle_source_stream_visuals: bool = True
    particle_source_stream_style: str = 'continuous_mesh'
    particle_source_stream_length: float = 0.42
    particle_source_stream_radius: float = 0.025
    particle_source_stream_color: tuple[float, float, float] = (0.38, 0.82, 1.0)
    particle_source_stream_opacity: float = 0.72
    particle_source_stream_segments: int = 18
    particle_source_stream_radial_segments: int = 12
    particle_source_stream_wobble: float = 0.006
    particle_source_stream_taper: float = 0.42
    particle_source_stream_droplets: int = 5
    liquid_visual_radius_scale: float = 0.34
    liquid_visual_height_scale: float = 0.7
    liquid_visual_bottom_margin_scale: float = 0.18
    liquid_visual_top_margin_scale: float = 0.14
    liquid_visual_min_radius: float = 0.012
    liquid_visual_min_height: float = 0.006
    liquid_visual_color: tuple[float, float, float] = (0.0, 0.82, 1.0)
    liquid_visual_opacity: float = 0.55
    liquid_visual_body_opacity: float = 0.48
    liquid_visual_surface_opacity: float = 0.88
    liquid_visual_roughness: float = 0.03
    liquid_visual_surface_height: float = 0.004
    liquid_visual_surface_lift: float = 0.001
    liquid_visual_surface_radius_scale: float = 1.02
    liquid_visual_ior: float = 1.333
    liquid_visual_depth: float = 0.06
    container_liquid_visual_mode: str = 'visual_mesh_water'
    liquid_particle_color: tuple[float, float, float] = REALISTIC_WATER_COLOR
    fluid_hq_rendering: bool = False
    dirty_overlay_color: tuple[float, float, float] = (0.23, 0.12, 0.045)
    dirty_overlay_opacity: float = 0.86
    dirty_overlay_radius_scale: float = 0.48
    dirty_overlay_height: float = 0.014
    dirty_overlay_top_lift: float = 0.004

    @classmethod
    def from_object(cls, obj: Any) -> 'VisualStateSyncConfig':
        liquid_color = getattr(obj, 'liquid_visual_color', (0.0, 0.82, 1.0))
        if isinstance(liquid_color, str):
            liquid_color = tuple((float(value) for value in liquid_color.split(',')))
        particle_color = getattr(obj, 'liquid_particle_color', REALISTIC_WATER_COLOR)
        if isinstance(particle_color, str):
            particle_color = tuple((float(value) for value in particle_color.split(',')))
        stream_color = getattr(obj, 'particle_source_stream_color', (0.0, 0.86, 1.0))
        if isinstance(stream_color, str):
            stream_color = tuple((float(value) for value in stream_color.split(',')))
        dirty_overlay_color = getattr(obj, 'dirty_overlay_color', (0.23, 0.12, 0.045))
        if isinstance(dirty_overlay_color, str):
            dirty_overlay_color = tuple((float(value) for value in dirty_overlay_color.split(',')))
        return cls(dirty_visual_system=str(getattr(obj, 'dirty_visual_system', 'stain')), visual_state_sync_steps=int(getattr(obj, 'visual_state_sync_steps', 4)), open_visual_settle_renders=int(getattr(obj, 'open_visual_settle_renders', 8)), require_visual_state_sync=bool(getattr(obj, 'require_visual_state_sync', True)), fire_visual_steps=int(getattr(obj, 'fire_visual_steps', 18)), fire_shutdown_steps=int(getattr(obj, 'fire_shutdown_steps', 24)), fire_shutdown_render_steps=int(getattr(obj, 'fire_shutdown_render_steps', 6)), deepen_fire_color=bool(getattr(obj, 'deepen_fire_color', True)), fire_force_clear_on_shutdown=bool(getattr(obj, 'fire_force_clear_on_shutdown', True)), water_source_system=str(getattr(obj, 'water_source_system', 'water')), water_visual_steps=int(getattr(obj, 'water_visual_steps', 30)), water_shutdown_steps=int(getattr(obj, 'water_shutdown_steps', 6)), water_shutdown_render_steps=int(getattr(obj, 'water_shutdown_render_steps', 50)), water_shutdown_min_render_steps=int(getattr(obj, 'water_shutdown_min_render_steps', 8)), enable_particle_source_visuals=bool(getattr(obj, 'enable_particle_source_visuals', True)), enable_particle_source_stream_visuals=bool(getattr(obj, 'enable_particle_source_stream_visuals', True)), particle_source_stream_style=str(getattr(obj, 'particle_source_stream_style', 'continuous_mesh')), particle_source_stream_length=float(getattr(obj, 'particle_source_stream_length', 0.42)), particle_source_stream_radius=float(getattr(obj, 'particle_source_stream_radius', 0.025)), particle_source_stream_color=tuple((float(value) for value in stream_color)), particle_source_stream_opacity=float(getattr(obj, 'particle_source_stream_opacity', 0.72)), particle_source_stream_segments=int(getattr(obj, 'particle_source_stream_segments', 18)), particle_source_stream_radial_segments=int(getattr(obj, 'particle_source_stream_radial_segments', 12)), particle_source_stream_wobble=float(getattr(obj, 'particle_source_stream_wobble', 0.006)), particle_source_stream_taper=float(getattr(obj, 'particle_source_stream_taper', 0.42)), particle_source_stream_droplets=int(getattr(obj, 'particle_source_stream_droplets', 5)), liquid_visual_radius_scale=float(getattr(obj, 'liquid_visual_radius_scale', 0.34)), liquid_visual_height_scale=float(getattr(obj, 'liquid_visual_height_scale', 0.7)), liquid_visual_bottom_margin_scale=float(getattr(obj, 'liquid_visual_bottom_margin_scale', 0.18)), liquid_visual_top_margin_scale=float(getattr(obj, 'liquid_visual_top_margin_scale', 0.14)), liquid_visual_min_radius=float(getattr(obj, 'liquid_visual_min_radius', 0.012)), liquid_visual_min_height=float(getattr(obj, 'liquid_visual_min_height', 0.006)), liquid_visual_color=tuple((float(value) for value in liquid_color)), liquid_visual_opacity=float(getattr(obj, 'liquid_visual_opacity', 0.55)), liquid_visual_body_opacity=float(getattr(obj, 'liquid_visual_body_opacity', 0.48)), liquid_visual_surface_opacity=float(getattr(obj, 'liquid_visual_surface_opacity', 0.88)), liquid_visual_roughness=float(getattr(obj, 'liquid_visual_roughness', 0.03)), liquid_visual_surface_height=float(getattr(obj, 'liquid_visual_surface_height', 0.004)), liquid_visual_surface_lift=float(getattr(obj, 'liquid_visual_surface_lift', 0.001)), liquid_visual_surface_radius_scale=float(getattr(obj, 'liquid_visual_surface_radius_scale', 1.02)), liquid_visual_ior=float(getattr(obj, 'liquid_visual_ior', 1.333)), liquid_visual_depth=float(getattr(obj, 'liquid_visual_depth', 0.06)), container_liquid_visual_mode=str(getattr(obj, 'container_liquid_visual_mode', 'visual_mesh_water')), liquid_particle_color=tuple((float(value) for value in particle_color)), fluid_hq_rendering=bool(getattr(obj, 'fluid_hq_rendering', False)), dirty_overlay_color=tuple((float(value) for value in dirty_overlay_color)), dirty_overlay_opacity=float(getattr(obj, 'dirty_overlay_opacity', 0.86)), dirty_overlay_radius_scale=float(getattr(obj, 'dirty_overlay_radius_scale', 0.48)), dirty_overlay_height=float(getattr(obj, 'dirty_overlay_height', 0.014)), dirty_overlay_top_lift=float(getattr(obj, 'dirty_overlay_top_lift', 0.004)))

class OmniGibsonVisualStateSynchronizer:

    def __init__(self, *, og_module: Any, env: Any, og_object_by_name: MutableMapping[str, Any], config: VisualStateSyncConfig | None=None, before_fire_step: Callable[[], None] | None=None, on_object_added: Callable[[str, Any], None] | None=None) -> None:
        self.og = og_module
        self.env = env
        self.og_object_by_name = og_object_by_name
        self.config = config or VisualStateSyncConfig()
        self.before_fire_step = before_fire_step
        self.on_object_added = on_object_added
        self.visual_mutations: list[dict[str, Any]] = []
        self.dirty_visual_system_by_object: dict[str, str] = {}
        self.container_liquid_visuals: dict[tuple[str, str], dict[str, Any]] = {}
        self.particle_source_stream_visuals: dict[tuple[str, str], dict[str, Any]] = {}
        self.dirty_overlay_visuals: dict[str, dict[str, Any]] = {}
        self._last_fire_diagnostics: dict[str, Any] = {}
        self._cut_object_parts_by_asset: dict[tuple[str, str], dict[str, Any] | None] = {}

    def sync_initial_symbolic_states(self, objects: Mapping[str, Any]) -> None:
        for name, obj in objects.items():
            states = getattr(obj, 'states', {}) or {}
            if states.get('dirty') is True:
                self.set_dirty_visual(name, True, reason='initial_dirty')
            initial_substances = self._substances_from_container_state(states.get('contains')) | self._substances_from_container_state(states.get('filled'))
            for substance in sorted(initial_substances):
                self.set_container_substance_visual(name, substance, True, reason='initial_container_substance')
            saturated = states.get('saturated')
            if isinstance(saturated, Mapping):
                substances = {str(substance) for substance in saturated}
                if not substances:
                    substances.add('water')
                for substance in sorted(substances):
                    self.set_saturated_visual(name, substance, bool(saturated.get(substance, False)), reason='initial_saturated')

    def apply_action_result(self, result: ActionResult) -> list[dict[str, Any]]:
        before_count = len(self.visual_mutations)
        if not result.success:
            return []
        if result.action in {'OPEN', 'CLOSE'}:
            target_name = result.args['target']
            value = self._state_delta_after(result, 'open')
            self.set_open_visual(target_name, result.action == 'OPEN' if value is None else bool(value))
        elif result.action == 'NAVIGATE_TO' and result.state_delta:
            for item in result.state_delta.get('navigation_opened_doors', []) or []:
                if isinstance(item, Mapping) and item.get('name'):
                    self.set_open_visual(str(item['name']), True)
        elif result.action in {'TOGGLE_ON', 'TOGGLE_OFF'}:
            target_name = result.args['target']
            value = self._state_delta_after(result, 'toggled_on')
            self.set_toggle_visual(target_name, result.action == 'TOGGLE_ON' if value is None else bool(value))
        elif result.action == 'HEAT' and result.state_delta:
            target_name = result.args.get('target')
            value = self._nested_toggle_delta_after(result)
            if target_name and value is not None and bool(value):
                self.set_toggle_visual(str(target_name), True, reason='heat_toggle_result')
        elif result.action == 'WIPE' and result.state_delta:
            state_delta = result.state_delta.get('object_state', {})
            if state_delta.get('state') == 'dirty':
                self.set_dirty_visual(result.args['target'], bool(state_delta.get('after')), reason='wipe_result')
        elif result.action == 'CUT' and result.state_delta:
            state_delta = result.state_delta.get('object_state', {})
            if state_delta.get('state') == 'sliced' and state_delta.get('after') is True:
                self.set_cut_visual(result.args['target'])
        elif result.action == 'SOAK' and result.state_delta:
            state_delta = result.state_delta.get('object_state', {})
            if state_delta.get('state') == 'saturated':
                target_name = state_delta.get('target')
                substance = state_delta.get('substance')
                if target_name and substance:
                    after = state_delta.get('after')
                    saturated = True
                    if isinstance(after, Mapping):
                        saturated = bool(after.get(substance, True))
                    self.set_saturated_visual(str(target_name), str(substance), saturated, reason=f'{result.action.lower()}_result')
            cleaning_delta = result.state_delta.get('cleaning', {})
            if cleaning_delta.get('state') == 'dirty':
                target_name = cleaning_delta.get('target')
                if target_name:
                    self.set_dirty_visual(str(target_name), bool(cleaning_delta.get('after')), reason='soak_cleaning_result')
        elif result.action in {'FILL', 'SETUP_FILLED_FOR_POUR'} and result.state_delta:
            state_delta = result.state_delta.get('object_state', {})
            if state_delta.get('state') == 'contains/filled':
                target_name = state_delta.get('target') or result.args.get('target')
                substance = state_delta.get('substance') or result.args.get('substance')
                if target_name and substance:
                    self.set_container_substance_visual(str(target_name), str(substance), True, reason=f'{result.action.lower()}_result')
        elif result.action == 'POUR' and result.state_delta:
            state_delta = result.state_delta.get('object_state', {})
            if state_delta.get('state') == 'contains/filled':
                source_name = state_delta.get('source') or result.args.get('source')
                target_name = state_delta.get('target') or result.args.get('target')
                substances = state_delta.get('substances') or result.args.get('substances') or []
                for substance in substances:
                    if source_name and substance:
                        self.set_container_substance_visual(str(source_name), str(substance), False, reason='pour_source_depleted')
                    if target_name and substance:
                        self.set_container_substance_visual(str(target_name), str(substance), True, reason='pour_target_filled')
        elif result.action == 'EMPTY' and result.state_delta:
            state_delta = result.state_delta.get('object_state', {})
            if state_delta.get('state') == 'contains/filled':
                target_name = state_delta.get('target') or result.args.get('target')
                substances = state_delta.get('removed_substances') or self._substances_from_container_delta(state_delta.get('before'))
                for substance in substances:
                    if target_name and substance:
                        self.set_container_substance_visual(str(target_name), str(substance), False, reason='empty_result')
        return self.visual_mutations[before_count:]

    def set_action_liquid_stream_visual(self, stream_name: str, substance: str, start: tuple[float, float, float] | list[float], end: tuple[float, float, float] | list[float], visible: bool, reason: str) -> dict[str, Any]:
        prim_path = self._action_liquid_stream_prim_path(stream_name, substance)
        before_visible = self._liquid_visual_visible(prim_path)
        if not visible:
            if before_visible is not None:
                self._force_hide_liquid_visual_for_render(prim_path)
            after_visible = self._liquid_visual_visible(prim_path)
            render_steps = self._advance_sim_renders(max(int(self.config.water_shutdown_render_steps), 2))
            info = {'success': after_visible is not True, 'object': str(stream_name), 'substance': str(substance), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible, 'removed': False, 'retained_hidden': before_visible is not None, 'water_shutdown_render_steps': render_steps, 'visual_mode': 'action_liquid_stream'}
            self._record_visual_mutation(str(stream_name), 'liquid_stream', False, bool(info['success']), reason, **self._visual_info_payload(info))
            if not info['success']:
                self._require_visual_state_sync(str(stream_name), 'liquid_stream')
            return info
        try:
            geometry = self._action_liquid_stream_geometry(start, end)
            material_info = self._ensure_particle_source_stream_material(substance)
            stream_info = self._create_or_update_particle_source_stream_visual(prim_path, geometry, material_info, visible=True)
            self._set_liquid_visual_visibility(prim_path, True)
            after_visible = self._liquid_visual_visible(prim_path)
            success = bool(material_info.get('success')) and bool(stream_info.get('success')) and (after_visible is True)
            info = {'success': success, 'object': str(stream_name), 'substance': str(substance), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible, 'created': before_visible is None, 'visual_mode': str(stream_info.get('visual_mode', 'action_liquid_stream')), 'geometry': geometry, 'material': self._visual_info_payload(material_info), 'stream_mesh': self._visual_info_payload(stream_info)}
        except Exception as exc:
            info = {'success': False, 'object': str(stream_name), 'substance': str(substance), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': self._liquid_visual_visible(prim_path), 'visual_mode': 'action_liquid_stream', 'error': repr(exc)}
        self._record_visual_mutation(str(stream_name), 'liquid_stream', True, bool(info['success']), reason, **self._visual_info_payload(info))
        if not info['success']:
            self._require_visual_state_sync(str(stream_name), 'liquid_stream')
        return info

    def set_open_visual(self, name: str, value: bool, *, reason: str='open_result') -> None:
        try:
            from omnigibson import object_states
        except Exception as exc:
            self._record_visual_mutation(name, 'open', value, False, reason, error=repr(exc))
            self._require_visual_state_sync(name, 'open')
            return
        og_obj = self._resolve_og_object_by_name(name)
        if og_obj is None or not hasattr(og_obj, 'states') or object_states.Open not in og_obj.states:
            self._record_visual_mutation(name, 'open', value, False, reason, error='missing_open_state')
            self._require_visual_state_sync(name, 'open')
            return
        state = og_obj.states[object_states.Open]
        sim_ready = self._ensure_open_visual_sim_ready()
        state_ready = self._ensure_open_state_ready(og_obj, state)
        before = self._boolean_state_value(state)
        before_joints = self._open_joint_snapshot(og_obj, state, requested_value=bool(value))
        fallback_info: dict[str, Any] | None = None
        primary_info = self._set_open_state_value_like_demo(state, bool(value), fully=True)
        self._refresh_object_visuals(name)
        primary_settle_renders = self._advance_open_visual_settle_renders()
        after = self._boolean_state_value(state)
        after_joints = self._open_joint_snapshot(og_obj, state, requested_value=bool(value))
        ok = self._open_visual_request_satisfied(after, after_joints, bool(value))
        if not ok:
            fallback_info = self._set_open_joints_directly(og_obj, state, bool(value), error=primary_info.get('error'))
            self._refresh_object_visuals(name)
            fallback_settle_renders = self._advance_open_visual_settle_renders()
            after = self._boolean_state_value(state)
            after_joints = self._open_joint_snapshot(og_obj, state, requested_value=bool(value))
            ok = self._open_visual_request_satisfied(after, after_joints, bool(value))
        else:
            fallback_settle_renders = 0
        self._record_visual_mutation(name, 'open', value, ok, reason, before=before, after=after, before_joints=before_joints, after_joints=after_joints, sim_ready=sim_ready, state_ready=state_ready, primary=primary_info, fallback=fallback_info, primary_settle_renders=primary_settle_renders, fallback_settle_renders=fallback_settle_renders)
        if not ok:
            self._require_visual_state_sync(name, 'open')

    def set_open_openness(self, name: str, openness: float, *, symbolic_open: bool | None=None, reason: str='partial_open_close') -> dict[str, Any]:
        requested = max(0.0, min(1.0, float(openness)))
        try:
            from omnigibson import object_states
        except Exception as exc:
            info = {'success': False, 'object': str(name), 'openness': requested, 'symbolic_open': symbolic_open, 'reason': reason, 'error': repr(exc)}
            self._record_visual_mutation(name, 'openness', requested, False, reason, symbolic_open=symbolic_open, error=repr(exc))
            self._require_visual_state_sync(name, 'openness')
            return info
        og_obj = self._resolve_og_object_by_name(name)
        if og_obj is None or not hasattr(og_obj, 'states') or object_states.Open not in og_obj.states:
            info = {'success': False, 'object': str(name), 'openness': requested, 'symbolic_open': symbolic_open, 'reason': reason, 'error': 'missing_open_state'}
            self._record_visual_mutation(name, 'openness', requested, False, reason, symbolic_open=symbolic_open, error='missing_open_state')
            self._require_visual_state_sync(name, 'openness')
            return info
        state = og_obj.states[object_states.Open]
        sim_ready = self._ensure_open_visual_sim_ready()
        state_ready = self._ensure_open_state_ready(og_obj, state)
        before = self._boolean_state_value(state)
        before_joints = self._open_joint_snapshot(og_obj, state, requested_openness=requested)
        joint_info = self._set_open_joints_to_openness(og_obj, state, requested)
        self._refresh_object_visuals(name)
        settle_renders = self._advance_open_visual_settle_renders()
        after = self._boolean_state_value(state)
        after_joints = self._open_joint_snapshot(og_obj, state, requested_openness=requested)
        ok = self._open_openness_request_satisfied(after_joints)
        info = {'success': ok, 'object': str(name), 'openness': requested, 'symbolic_open': symbolic_open, 'reason': reason, 'before': before, 'after': after, 'before_joints': before_joints, 'after_joints': after_joints, 'sim_ready': sim_ready, 'state_ready': state_ready, 'direct_joint': joint_info, 'settle_renders': settle_renders}
        self._record_visual_mutation(name, 'openness', requested, ok, reason, **self._visual_info_payload({key: value for key, value in info.items() if key not in {'object', 'openness', 'reason'}}))
        if not ok:
            self._require_visual_state_sync(name, 'openness')
        return info

    def set_toggle_visual(self, name: str, value: bool, *, reason: str='toggle_result', og_obj: Any | None=None) -> None:
        og_obj = og_obj if og_obj is not None else self._resolve_og_object_by_name(name)
        try:
            from omnigibson import object_states
        except Exception as exc:
            fallback_info = self._set_missing_toggle_water_stream_fallback(name, og_obj, bool(value), reason)
            if fallback_info is not None:
                ok = bool(fallback_info.get('success'))
                self._record_visual_mutation(name, 'toggled_on', value, ok, reason, before=None, after=bool(value) if ok else None, fallback='water_source_stream', import_error=repr(exc), **self._visual_info_payload(fallback_info))
                if not ok:
                    self._require_visual_state_sync(name, 'toggled_on')
                return
            self._record_visual_mutation(name, 'toggled_on', value, False, reason, error=repr(exc))
            self._require_visual_state_sync(name, 'toggled_on')
            return
        if og_obj is None or not hasattr(og_obj, 'states') or object_states.ToggledOn not in og_obj.states:
            fallback_info = self._set_missing_toggle_water_stream_fallback(name, og_obj, bool(value), reason)
            if fallback_info is not None:
                ok = bool(fallback_info.get('success'))
                self._record_visual_mutation(name, 'toggled_on', value, ok, reason, before=None, after=bool(value) if ok else None, fallback='water_source_stream', **self._visual_info_payload(fallback_info))
                if not ok:
                    self._require_visual_state_sync(name, 'toggled_on')
                return
            if self._is_appliance_heat_visual_optional(name, og_obj, reason) and (not self._has_fire_emitter(og_obj)):
                self._record_heat_visual_not_required(name, value, reason, og_obj=og_obj, before=None, after=None, fire_emitter=False, fire_steps=0, fire_diagnostics=None, particle_source_info={'success': True, 'reason': 'missing_particle_source_state'}, extra={'error': 'missing_toggle_state'})
                return
            self._record_visual_mutation(name, 'toggled_on', value, False, reason, error='missing_toggle_state')
            self._require_visual_state_sync(name, 'toggled_on')
            return
        state = og_obj.states[object_states.ToggledOn]
        before = self._boolean_state_value(state)
        try:
            state.set_value(bool(value))
        except Exception as exc:
            fallback_info = self._set_missing_toggle_water_stream_fallback(name, og_obj, bool(value), reason, fallback_reason=repr(exc))
            if fallback_info is not None:
                ok = bool(fallback_info.get('success'))
                self._record_visual_mutation(name, 'toggled_on', value, ok, reason, before=before, after=bool(value) if ok else None, native_toggle_error=repr(exc), fallback='water_source_stream', **self._visual_info_payload(fallback_info))
                if not ok:
                    self._require_visual_state_sync(name, 'toggled_on')
                return
            has_fire_emitter = self._has_fire_emitter(og_obj)
            if self._is_appliance_heat_visual_optional(name, og_obj, reason) and (not has_fire_emitter):
                self._record_heat_visual_not_required(name, value, reason, og_obj=og_obj, before=before, after=None, fire_emitter=has_fire_emitter, fire_steps=0, fire_diagnostics=None, particle_source_info={'success': True, 'reason': 'not_evaluated'}, extra={'native_toggle_error': repr(exc)})
                return
            self._record_visual_mutation(name, 'toggled_on', value, False, reason, before=before, error=repr(exc))
            self._require_visual_state_sync(name, 'toggled_on')
            return
        self._refresh_object_visuals(name)
        has_fire_emitter = self._has_fire_emitter(og_obj)
        self._last_fire_diagnostics = {}
        fire_steps = self._advance_fire_visuals(og_obj, bool(value)) if has_fire_emitter else 0
        fire_diagnostics = dict(self._last_fire_diagnostics)
        has_particle_source = self._has_particle_source(og_obj, object_states)
        particle_source_visual_enabled = bool(self.config.enable_particle_source_visuals)
        fallback_info: dict[str, Any] | None = None
        if has_particle_source and particle_source_visual_enabled:
            particle_source_info = self._advance_particle_source_visuals(og_obj, bool(value))
            if not bool(particle_source_info.get('success', False)):
                fallback_info = self._set_missing_toggle_water_stream_fallback(name, og_obj, bool(value), reason, fallback_reason=str(particle_source_info.get('error') or 'particle_source_visual_failed'))
                if fallback_info is not None:
                    particle_source_info['fallback'] = self._visual_info_payload(fallback_info)
                    particle_source_info['fallback_success'] = bool(fallback_info.get('success'))
                    particle_source_info['success'] = bool(fallback_info.get('success'))
        elif has_particle_source:
            particle_source_info = {'success': True, 'skipped': True, 'reason': 'particle_source_visuals_disabled'}
            if self.config.enable_particle_source_stream_visuals:
                if value:
                    particle_source_state = getattr(object_states, 'ParticleSource', None)
                    states = getattr(og_obj, 'states', {})
                    try:
                        source_state = states[particle_source_state]
                    except (KeyError, TypeError, AttributeError):
                        source_state = None
                    if source_state is not None:
                        fallback_info = self._set_particle_source_stream_visual(og_obj, source_state, str(self.config.water_source_system or 'water'), visible=True)
                        fallback_info['fallback_reason'] = 'particle_source_visuals_disabled'
                    else:
                        fallback_info = self._set_missing_toggle_water_stream_fallback(name, og_obj, True, reason, fallback_reason='particle_source_state_lookup_failed')
                        if fallback_info is None:
                            particle_source_info['success'] = False
                            particle_source_info['error'] = 'particle_source_state_lookup_failed'
                else:
                    fallback_info = self._remove_toggle_water_stream_visuals(name, og_obj, reason)
                if fallback_info is not None:
                    particle_source_info['fallback'] = self._visual_info_payload(fallback_info)
                    particle_source_info['fallback_success'] = bool(fallback_info.get('success'))
                    particle_source_info['success'] = bool(fallback_info.get('success'))
        else:
            particle_source_info = {'success': True, 'reason': 'missing_particle_source_state'}
            fallback_info = self._set_missing_toggle_water_stream_fallback(name, og_obj, bool(value), reason, fallback_reason='missing_particle_source_state')
            if fallback_info is not None:
                particle_source_info['fallback'] = self._visual_info_payload(fallback_info)
                particle_source_info['fallback_success'] = bool(fallback_info.get('success'))
                particle_source_info['success'] = bool(fallback_info.get('success'))
        after = self._boolean_state_value(state)
        heat_visual_optional = self._is_appliance_heat_visual_optional(name, og_obj, reason)
        ok = bool(after) == bool(value) if after is not None else True
        ok = ok and bool(particle_source_info.get('success', True))
        if heat_visual_optional and (not has_fire_emitter) and (not ok):
            self._record_heat_visual_not_required(name, value, reason, og_obj=og_obj, before=before, after=after, fire_emitter=has_fire_emitter, fire_steps=fire_steps, fire_diagnostics=fire_diagnostics if fire_diagnostics else None, particle_source_info=particle_source_info)
            return
        fallback_payload = {'fallback': 'water_source_stream'} if fallback_info is not None else {}
        self._record_visual_mutation(name, 'toggled_on', value, ok, reason, before=before, after=after, fire_emitter=has_fire_emitter, fire_visual_steps=fire_steps, fire_diagnostics=fire_diagnostics if fire_diagnostics else None, particle_source=has_particle_source, particle_source_visual_enabled=particle_source_visual_enabled, **fallback_payload, **self._particle_source_payload(particle_source_info))
        if not ok:
            self._require_visual_state_sync(name, 'toggled_on')

    def _is_appliance_heat_visual_optional(self, name: str, og_obj: Any | None, reason: str) -> bool:
        if reason != 'heat_toggle_result':
            return False
        tokens = [str(name), str(getattr(og_obj, 'name', '')), str(getattr(og_obj, 'category', '')), str(getattr(og_obj, 'model_name', ''))]
        text = ' '.join(tokens).lower()
        appliance_tokens = ('microwave', 'microwave_oven', 'oven', 'toaster_oven')
        return any((token in text for token in appliance_tokens))

    def _record_heat_visual_not_required(self, name: str, value: bool, reason: str, *, og_obj: Any | None, before: Any, after: Any, fire_emitter: bool, fire_steps: int, fire_diagnostics: Mapping[str, Any] | None, particle_source_info: Mapping[str, Any], extra: Mapping[str, Any] | None=None) -> None:
        payload = {'before': before, 'after': after, 'status': 'unsupported_visual_state', 'unsupported_visual_state': True, 'unsupported_reason': 'appliance_heat_visual_not_required', 'appliance_heat_visual_not_required': True, 'error': 'unsupported_visual_state', 'fire_emitter': bool(fire_emitter), 'fire_visual_steps': int(fire_steps), 'fire_diagnostics': fire_diagnostics, 'particle_source': False, 'particle_source_visual_enabled': bool(self.config.enable_particle_source_visuals), 'heat_source_category': getattr(og_obj, 'category', None), 'heat_source_name': getattr(og_obj, 'name', None)}
        payload.update(self._particle_source_payload(particle_source_info))
        if extra:
            payload.update(dict(extra))
        self._record_visual_mutation(name, 'toggled_on', value, None, reason, **payload)

    def _set_missing_toggle_water_stream_fallback(self, name: str, og_obj: Any | None, value: bool, reason: str, *, fallback_reason: str='missing_toggle_state') -> dict[str, Any] | None:
        if not self.config.enable_particle_source_stream_visuals:
            return None
        og_obj = og_obj or self._resolve_og_object_by_name(name)
        if og_obj is None:
            og_obj = self._find_water_source_like_object(name)
        if og_obj is None or not self._looks_like_water_source(name, og_obj):
            return None
        if not value:
            return self._remove_toggle_water_stream_visuals(name, og_obj, reason)
        aabb = self._object_aabb_low_high(og_obj)
        if aabb is None:
            return None
        low, high = aabb
        start, end, geometry_hint = self._toggle_water_fallback_stream_endpoints(name, og_obj, low, high)
        try:
            info = self.set_action_liquid_stream_visual(f'{name}_toggle_fallback', str(self.config.water_source_system or 'water'), start, end, bool(value), reason)
        except Exception as exc:
            info = {'success': False, 'object': str(name), 'visual_mode': 'action_liquid_stream', 'error': repr(exc)}
        info['fallback_reason'] = fallback_reason
        info['fallback_geometry'] = geometry_hint
        return info

    def _remove_toggle_water_stream_visuals(self, name: str, og_obj: Any, reason: str) -> dict[str, Any]:
        inventory_before = self._liquid_stream_prim_inventory()
        system_name = str(self.config.water_source_system or 'water')
        object_names = {str(name), str(getattr(og_obj, 'name', ''))}
        object_names.update((str(candidate_name) for candidate_name, candidate_obj in self.og_object_by_name.items() if candidate_obj is og_obj))
        object_names.discard('')
        prim_paths = {self._action_liquid_stream_prim_path(f'{object_name}_toggle_fallback', system_name) for object_name in object_names}
        prim_paths.add(self._particle_source_stream_prim_path(og_obj, system_name))
        for (object_name, tracked_system), info in self.particle_source_stream_visuals.items():
            if str(tracked_system) != system_name or str(object_name) not in object_names:
                continue
            tracked_path = info.get('prim_path') if isinstance(info, Mapping) else None
            if tracked_path:
                prim_paths.add(str(tracked_path))
        removals = []
        for prim_path in sorted(prim_paths):
            before_visible = self._liquid_visual_visible(prim_path)
            if before_visible is not None:
                self._force_hide_liquid_visual_for_render(prim_path)
            after_visible = self._liquid_visual_visible(prim_path)
            removals.append({'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible, 'removed': False, 'retained_hidden': before_visible is not None})
        requested_render_steps = max(int(self.config.water_shutdown_render_steps), int(self.config.water_shutdown_min_render_steps), 0)
        render_steps = self._advance_sim_renders(requested_render_steps)
        inventory_after = self._liquid_stream_prim_inventory()
        return {'success': all((item['after_visible'] is not True for item in removals)), 'object': str(name), 'substance': system_name, 'visible': False, 'reason': reason, 'visual_mode': 'water_source_stream_cleanup', 'candidate_object_names': sorted(object_names), 'removals': removals, 'water_shutdown_render_steps_configured': int(self.config.water_shutdown_render_steps), 'water_shutdown_min_render_steps': int(self.config.water_shutdown_min_render_steps), 'water_shutdown_render_steps_requested': requested_render_steps, 'water_shutdown_render_steps': render_steps, 'water_shutdown_remove_render_steps': 0, 'stream_inventory_before': inventory_before, 'stream_inventory_after': inventory_after, 'fallback_reason': 'toggle_off_cleanup'}

    def _force_hide_liquid_visual_for_render(self, prim_path: str) -> None:
        try:
            stage = self._usd_stage()
        except Exception:
            self._set_liquid_visual_visibility(prim_path, False)
            return
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            return
        try:
            from pxr import Usd, UsdGeom
            prims = list(Usd.PrimRange(root))
            with self._usd_edit_context():
                for index, prim in enumerate(prims):
                    if not prim.IsValid():
                        continue
                    try:
                        if index == 0:
                            UsdGeom.Imageable(prim).MakeInvisible()
                    except Exception:
                        continue
                    try:
                        if prim.IsA(UsdGeom.Gprim):
                            UsdGeom.Gprim(prim).CreateDisplayOpacityAttr().Set([0.0])
                    except Exception:
                        pass
        except Exception:
            self._set_liquid_visual_visibility(prim_path, False)

    def _liquid_stream_prim_inventory(self) -> list[dict[str, Any]]:
        stage = self._usd_stage()
        rows = []
        for root_path in ('/World/VisualActionLiquidStreams', '/World/VisualParticleSourceStreams'):
            root = stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                continue
            for prim in root.GetChildren():
                path = str(prim.GetPath())
                rows.append({'prim_path': path, 'type_name': str(prim.GetTypeName() or ''), 'active': bool(prim.IsActive()), 'visible': self._liquid_visual_visible(path), 'child_paths': sorted((str(child.GetPath()) for child in prim.GetChildren()))})
        return sorted(rows, key=lambda item: item['prim_path'])

    def _toggle_water_fallback_stream_endpoints(self, name: str, og_obj: Any, low: np.ndarray, high: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], dict[str, Any]]:
        low = np.array(low, dtype=float)
        high = np.array(high, dtype=float)
        center = (low + high) * 0.5
        extent = np.maximum(high - low, 1e-06)
        link_position, link_hint = self._water_source_link_position(og_obj)
        if link_position is not None:
            start_np = np.array(link_position, dtype=float)
            start_np[2] = max(float(start_np[2]), float(low[2] + extent[2] * 0.68))
            source = 'water_source_link'
        else:
            axis = 1 if float(extent[1]) >= float(extent[0]) else 0
            offset = min(max(float(extent[axis]) * 0.3, 0.08), float(extent[axis]) * 0.42)
            start_np = center.copy()
            start_np[axis] += offset
            start_np[2] = float(high[2] + min(max(float(extent[2]) * 0.08, 0.055), 0.12))
            link_hint = {'axis': 'y' if axis == 1 else 'x', 'side_offset': float(offset)}
            source = 'aabb_top_back_spout_heuristic'
        length = self._toggle_water_fallback_stream_length(low, high, start_np)
        end_np = start_np + np.array([0.0, 0.0, -length], dtype=float)
        hint = {'source': source, 'aabb_low': [float(value) for value in low], 'aabb_high': [float(value) for value in high], 'aabb_center': [float(value) for value in center], 'aabb_extent': [float(value) for value in extent], 'stream_length': float(length), **link_hint}
        return ((float(start_np[0]), float(start_np[1]), float(start_np[2])), (float(end_np[0]), float(end_np[1]), float(end_np[2])), hint)

    def _toggle_water_fallback_stream_length(self, low: np.ndarray, high: np.ndarray, start: np.ndarray) -> float:
        default_length = max(float(self.config.particle_source_stream_length), 0.28)
        extent_z = max(float(high[2] - low[2]), 1e-06)
        basin_z = float(low[2] + extent_z * 0.42)
        candidate = float(start[2] - basin_z)
        min_length = max(default_length * 0.72, 0.28)
        max_length = max(default_length * 1.45, 0.58)
        return float(np.clip(candidate, min_length, max_length))

    def _water_source_link_position(self, og_obj: Any) -> tuple[tuple[float, float, float] | None, dict[str, Any]]:
        links = getattr(og_obj, 'links', {}) or {}
        if not isinstance(links, Mapping):
            return (None, {})
        tokens = ('faucet', 'spout', 'tap', 'water', 'source')
        for link_name, link in links.items():
            link_fields = [str(link_name)]
            for attr in ('name', 'prim_path', 'meta_link_type'):
                try:
                    link_fields.append(str(getattr(link, attr, '')))
                except Exception:
                    link_fields.append('')
            text = ' '.join((str(value).lower() for value in link_fields))
            if not any((token in text for token in tokens)):
                continue
            get_pose = getattr(link, 'get_position_orientation', None)
            if get_pose is None:
                continue
            try:
                position, _ = get_pose(frame='scene')
            except TypeError:
                position, _ = get_pose()
            except Exception:
                continue
            try:
                return (self._as_float_triplet(position), {'link_name': str(link_name), 'link_hint': text})
            except Exception:
                continue
        return (None, {})

    def _resolve_og_object_by_name(self, name: str) -> Any | None:
        raw = str(name)
        candidates = [raw, self._safe_prim_token(raw)]
        if '.' in raw:
            candidates.append(raw.replace('.', '_'))
        for candidate in dict.fromkeys(candidates):
            og_obj = self.og_object_by_name.get(candidate)
            if og_obj is not None:
                return og_obj
        safe_raw = self._safe_prim_token(raw)
        for key, og_obj in self.og_object_by_name.items():
            if self._safe_prim_token(key) == safe_raw:
                return og_obj
        return None

    def _find_water_source_like_object(self, requested_name: str) -> Any | None:
        requested_safe = self._safe_prim_token(requested_name)
        for key, og_obj in self.og_object_by_name.items():
            key_text = str(key)
            if self._safe_prim_token(key_text) == requested_safe:
                return og_obj
            if self._looks_like_water_source(key_text, og_obj):
                return og_obj
        return None

    def _looks_like_water_source(self, name: str, og_obj: Any) -> bool:
        text = ' '.join((str(value).lower() for value in (name, getattr(og_obj, 'name', ''), getattr(og_obj, 'category', ''), getattr(og_obj, 'model_name', ''))))
        if any((token in text for token in ('faucet', 'sink', 'shower', 'water_source'))):
            return True
        abilities = getattr(og_obj, 'abilities', ()) or ()
        return any((str(ability) in {'water_source', 'fluidSource', 'fluid_source', 'particleSource', 'particle_source'} for ability in abilities))

    def set_dirty_visual(self, name: str, dirty: bool, reason: str) -> None:
        og_obj = self._resolve_og_object_by_name(name)
        native_info: dict[str, Any] | None = None
        try:
            from omnigibson import object_states
        except Exception as exc:
            native_info = {'success': False, 'error': repr(exc), 'stage': 'import_object_states'}
            object_states = None
        if og_obj is None or object_states is None or (not hasattr(og_obj, 'states')) or (object_states.Covered not in og_obj.states):
            if native_info is None:
                native_info = {'success': False, 'error': 'missing_covered_state'}
            self._record_native_dirty_failure(name, dirty, reason, native_info=native_info)
            return
        system = self._dirty_visual_system_for(name)
        if system is None:
            native_info = {'success': False, 'error': 'missing_dirty_visual_system'}
            self._record_native_dirty_failure(name, dirty, reason, native_info=native_info)
            return
        state = og_obj.states[object_states.Covered]
        before = self._covered_value(state, system)
        before_particles = self._covered_particle_count(state, system)
        try:
            success = bool(state.set_value(system, bool(dirty)))
        except Exception as exc:
            native_info = {'success': False, 'system': system.name, 'before': before, 'before_particles': before_particles, 'error': repr(exc)}
            self._record_native_dirty_failure(name, dirty, reason, native_info=native_info)
            return
        self._clear_state_cache(state)
        self._refresh_object_visuals(name)
        self._advance_visual_state_renders()
        self._clear_state_cache(state)
        after = self._covered_value(state, system)
        after_particles = self._covered_particle_count(state, system)
        if after_particles is None:
            ok = success and after is bool(dirty)
        elif dirty:
            ok = success and after_particles > 0
        else:
            ok = success and after_particles == 0
        native_info = {'success': ok, 'set_value_success': success, 'system': system.name, 'before': before, 'after': after, 'before_particles': before_particles, 'after_particles': after_particles}
        needs_visible_enhancement = self._dirty_native_particles_need_enhancement(og_obj, dirty, after_particles)
        if not ok or needs_visible_enhancement:
            manual_info = self._set_manual_native_dirty_particles(og_obj, state, system, dirty)
            self._clear_state_cache(state)
            self._refresh_object_visuals(name)
            self._advance_visual_state_renders()
            self._clear_state_cache(state)
            after = self._covered_value(state, system)
            after_particles = self._covered_particle_count(state, system)
            ok = bool(manual_info.get('success')) and (after_particles is None or (after_particles > 0 if dirty else after_particles == 0))
            native_info = {**native_info, 'success': ok, 'after': after, 'after_particles': after_particles, 'manual_particle_fallback': self._visual_info_payload(manual_info)}
            if not ok:
                self._record_native_dirty_failure(name, dirty, reason, native_info=native_info)
                return
        self._record_visual_mutation(name, 'dirty', dirty, ok, reason, system=system.name, before=before, after=after, before_particles=before_particles, after_particles=after_particles, native_particle_enhancement=needs_visible_enhancement, manual_particle_fallback=self._visual_info_payload(native_info.get('manual_particle_fallback')) if native_info.get('manual_particle_fallback') is not None else None)
        if not ok:
            self._require_visual_state_sync(name, 'dirty')

    def _record_native_dirty_failure(self, name: str, dirty: bool, reason: str, *, native_info: Mapping[str, Any]) -> None:
        self._record_visual_mutation(name, 'dirty', dirty, False, reason, visual_mode='covered_particle_system', native_dirty=self._visual_info_payload(native_info), error=str(native_info.get('error') or 'native_dirty_visual_failed'))
        self._require_visual_state_sync(name, 'dirty')

    def _record_dirty_overlay_fallback(self, name: str, og_obj: Any | None, dirty: bool, reason: str, *, native_info: Mapping[str, Any]) -> None:
        overlay_info = self._set_dirty_overlay_visual(name, og_obj, dirty, reason)
        ok = bool(overlay_info.get('success'))
        self._record_visual_mutation(name, 'dirty', dirty, ok, reason, fallback='dirty_overlay', native_dirty=self._visual_info_payload(native_info), **self._visual_info_payload(overlay_info, exclude={'object', 'state', 'requested', 'reason'}))
        if not ok:
            self._require_visual_state_sync(name, 'dirty')

    def set_cut_visual(self, name: str) -> None:
        og_obj = self._resolve_og_object_by_name(name)
        if og_obj is None:
            self._record_visual_mutation(name, 'sliced', True, False, 'cut_result', error='target_not_found')
            self._require_visual_state_sync(name, 'sliced')
            return
        try:
            from omnigibson.transition_rules import SlicingRule
        except Exception as exc:
            self._record_visual_mutation(name, 'sliced', True, False, 'cut_result', error=repr(exc))
            self._require_visual_state_sync(name, 'sliced')
            return
        before_visible = self._object_visible(og_obj)
        try:
            rule = SlicingRule(scene=self.env.scene)
            transition_obj = self._cut_transition_object_with_parts_metadata(og_obj)
            output = rule.transition({'sliceable': [transition_obj]})
            add_report = self._add_sliced_part_objects(getattr(output, 'add', None))
            if not add_report['registered_added_objects'] and (not add_report['step_errors']):
                add_report['step_errors'].append({'stage': 'generate_sliced_parts', 'object': name, 'error': 'no_sliced_parts_generated'})
                add_report['cleanup_errors'].extend(self._cleanup_sliced_part_objects(add_report['added_objects']))
            if not add_report['step_errors'] and add_report['registered_added_objects']:
                og_obj.visible = False
                self._refresh_object_visuals(name)
                self._advance_visual_state_renders()
                add_report['callback_errors'].extend(self._run_deferred_sliced_part_inits(add_report.get('_deferred_inits') or []))
            after_visible = self._object_visible(og_obj)
            registered_added_names = [added_name for added_name in add_report['registered_added_objects'] if self._scene_has_object(added_name)]
            if len(registered_added_names) != len(add_report['registered_added_objects']):
                registered_set = set(registered_added_names)
                for added_name in add_report['registered_added_objects']:
                    if added_name not in registered_set:
                        add_report['step_errors'].append({'stage': 'verify_sliced_part_registered', 'object': added_name, 'error': 'sliced_part_not_registered'})
        except Exception as exc:
            self._record_visual_mutation(name, 'sliced', True, False, 'cut_result', before_visible=before_visible, after_visible=self._object_visible(og_obj), error=repr(exc), traceback=self._exception_traceback(exc))
            self._require_visual_state_sync(name, 'sliced')
            return
        ok = bool(registered_added_names) and after_visible is False and (not add_report['step_errors'])
        self._record_visual_mutation(name, 'sliced', True, ok, 'cut_result', before_visible=before_visible, after_visible=after_visible, added_objects=list(add_report['added_objects']), registered_added_objects=registered_added_names, step_errors=list(add_report['step_errors']), callback_errors=list(add_report['callback_errors']), cleanup_errors=list(add_report.get('cleanup_errors') or []))
        if not ok:
            self._require_visual_state_sync(name, 'sliced')

    def set_saturated_visual(self, name: str, substance: str, saturated: bool, reason: str) -> None:
        try:
            from omnigibson import object_states
            from omnigibson.object_states.saturated import ModifiedParticles
        except Exception as exc:
            self._record_visual_mutation(name, 'saturated', saturated, False, reason, substance=substance, error=repr(exc))
            self._require_visual_state_sync(name, 'saturated')
            return
        og_obj = self._resolve_og_object_by_name(name)
        if og_obj is None or not hasattr(og_obj, 'states') or object_states.Saturated not in og_obj.states:
            self._record_visual_mutation(name, 'saturated', saturated, None, reason, substance=substance, status='unsupported_visual_state', unsupported_visual_state=True, unsupported_reason='missing_saturated_state', error='unsupported_visual_state')
            return
        try:
            system = self._get_system(substance, force_init=False)
        except Exception as exc:
            self._record_visual_mutation(name, 'saturated', saturated, False, reason, substance=substance, error=repr(exc))
            self._require_visual_state_sync(name, 'saturated')
            return
        state = og_obj.states[object_states.Saturated]
        before = self._relative_boolean_state_value(state, system)
        before_particles = self._modified_particle_count(og_obj, ModifiedParticles, system)
        try:
            result = state.set_value(system, bool(saturated))
            success = True if result is None else bool(result)
        except Exception as exc:
            self._record_visual_mutation(name, 'saturated', saturated, False, reason, substance=system.name, before=before, before_particles=before_particles, error=repr(exc))
            self._require_visual_state_sync(name, 'saturated')
            return
        self._clear_state_cache(state)
        self._refresh_object_visuals(name)
        texture_applied = self._apply_texture_change(og_obj, state if saturated else None)
        visual_steps = self._advance_visual_state_renders()
        self._clear_state_cache(state)
        after = self._relative_boolean_state_value(state, system)
        after_particles = self._modified_particle_count(og_obj, ModifiedParticles, system)
        if after is None:
            ok = success
        else:
            ok = success and after is bool(saturated)
        if after_particles is not None:
            ok = ok and (after_particles > 0 if saturated else after_particles == 0)
        ok = ok and texture_applied
        self._record_visual_mutation(name, 'saturated', saturated, ok, reason, substance=system.name, before=before, after=after, before_particles=before_particles, after_particles=after_particles, texture_applied=texture_applied, visual_steps=visual_steps)
        if not ok:
            self._require_visual_state_sync(name, 'saturated')

    def set_container_substance_visual(self, name: str, substance: str, present: bool, reason: str) -> None:
        og_obj = self._resolve_og_object_by_name(name)
        visual_mode = str(self.config.container_liquid_visual_mode)
        if og_obj is None:
            self._record_visual_mutation(name, 'contains/filled', present, False, reason, substance=substance, visual_mode=visual_mode, error='target_not_found')
            self._require_visual_state_sync(name, 'contains/filled')
            return
        try:
            if visual_mode == 'official_filled_particles':
                info = self._set_container_liquid_official_filled_particles(og_obj, name, substance, bool(present))
            elif visual_mode in {'visual_mesh_water', 'transparent_visual_cylinder'}:
                info = self._set_container_visual_mesh_water(og_obj, name, substance, bool(present))
            else:
                raise ValueError(f'Unknown container liquid visual mode: {visual_mode}')
        except Exception as exc:
            self._record_visual_mutation(name, 'contains/filled', present, False, reason, substance=substance, visual_mode=visual_mode, error=repr(exc))
            self._require_visual_state_sync(name, 'contains/filled')
            return
        self._refresh_object_visuals(name)
        visual_steps = self._advance_visual_state_renders()
        after_visible = info.get('after_visible')
        if 'prim_path' in info:
            try:
                after_visible = self._liquid_visual_visible(info['prim_path'])
            except Exception:
                after_visible = info.get('after_visible')
        ok = bool(info.get('success'))
        if visual_mode in {'visual_mesh_water', 'transparent_visual_cylinder'} and after_visible is not None:
            ok = ok and after_visible is bool(present)
        self._record_visual_mutation(name, 'contains/filled', present, ok, reason, substance=str(substance), visual_mode=str(info.get('visual_mode', visual_mode)), before_visible=info.get('before_visible'), after_visible=after_visible, visual_primitive=info, visual_steps=visual_steps)
        if not ok:
            self._require_visual_state_sync(name, 'contains/filled')

    def _state_delta_after(self, result: ActionResult, state_name: str) -> Any | None:
        state_delta = (result.state_delta or {}).get('object_state', {})
        if state_delta.get('state') == state_name:
            return state_delta.get('after')
        return None

    def _nested_toggle_delta_after(self, result: ActionResult) -> Any | None:
        toggle_delta = (result.state_delta or {}).get('toggle', {})
        if not isinstance(toggle_delta, Mapping):
            return None
        object_state = toggle_delta.get('object_state', {})
        if isinstance(object_state, Mapping) and object_state.get('state') == 'toggled_on':
            return object_state.get('after')
        return None

    def _ensure_open_visual_sim_ready(self) -> dict[str, Any]:
        info: dict[str, Any] = {'success': True, 'was_playing': None, 'play_called': False, 'update_handles_called': False, 'refresh_physics_sim_view_called': False, 'physics_sim_view_before': None, 'physics_sim_view_after': None, 'errors': []}
        sim = getattr(self.og, 'sim', None)
        if sim is None:
            info['success'] = False
            info['errors'].append('missing_sim')
            return info
        try:
            info['was_playing'] = bool(sim.is_playing())
        except Exception as exc:
            info['errors'].append(f'is_playing:{repr(exc)}')
        try:
            info['physics_sim_view_before'] = getattr(sim, 'physics_sim_view', None) is not None
        except Exception as exc:
            info['errors'].append(f'physics_sim_view_before:{repr(exc)}')
        if info['was_playing'] is not True:
            try:
                sim.play()
                info['play_called'] = True
            except Exception as exc:
                info['success'] = False
                info['errors'].append(f'play:{repr(exc)}')
        update_handles = getattr(sim, 'update_handles', None)
        if callable(update_handles):
            try:
                update_handles()
                info['update_handles_called'] = True
            except Exception as exc:
                info['errors'].append(f'update_handles:{repr(exc)}')
        try:
            info['physics_sim_view_after'] = getattr(sim, 'physics_sim_view', None) is not None
        except Exception as exc:
            info['errors'].append(f'physics_sim_view_after:{repr(exc)}')
        if info['physics_sim_view_after'] is not True:
            refresh = getattr(sim, '_refresh_physics_sim_view', None)
            if callable(refresh):
                try:
                    refresh()
                    info['refresh_physics_sim_view_called'] = True
                    info['physics_sim_view_after'] = getattr(sim, 'physics_sim_view', None) is not None
                except Exception as exc:
                    info['errors'].append(f'refresh_physics_sim_view:{repr(exc)}')
        if info['physics_sim_view_after'] is False and (not info['play_called']):
            info['success'] = False
        return info

    def _ensure_open_state_ready(self, og_obj: Any, state: Any) -> dict[str, Any]:
        info: dict[str, Any] = {'success': True, 'was_initialized': None, 'initialized_now': False, 'rebuilt_relevant_joints': False, 'relevant_joint_count': None, 'errors': []}
        try:
            info['was_initialized'] = bool(getattr(state, '_initialized', False))
        except Exception as exc:
            info['errors'].append(f'initialized:{repr(exc)}')
        if info['was_initialized'] is not True:
            initialize = getattr(state, 'initialize', None)
            if callable(initialize):
                try:
                    initialize()
                    info['initialized_now'] = True
                except Exception as exc:
                    info['errors'].append(f'initialize:{repr(exc)}')
        joints_info = getattr(state, 'relevant_joints_info', None)
        if not joints_info:
            try:
                from omnigibson.object_states.open_state import _get_relevant_joints
                joints_info = _get_relevant_joints(og_obj)
                try:
                    state.relevant_joints_info = joints_info
                    info['rebuilt_relevant_joints'] = True
                except Exception as assign_exc:
                    info['errors'].append(f'assign_relevant_joints:{repr(assign_exc)}')
            except Exception as exc:
                info['errors'].append(f'get_relevant_joints:{repr(exc)}')
        try:
            info['relevant_joint_count'] = len(joints_info[1]) if joints_info else 0
        except Exception as exc:
            info['errors'].append(f'count_relevant_joints:{repr(exc)}')
        info['success'] = bool(getattr(state, '_initialized', False)) and bool(info['relevant_joint_count'])
        return info

    def _set_open_state_value_like_demo(self, state: Any, value: bool, *, fully: bool) -> dict[str, Any]:
        info: dict[str, Any] = {'success': False, 'mode': 'object_states.Open.set_value', 'requested': bool(value), 'fully': bool(fully), 'used_fully_arg': True}
        try:
            try:
                ok = state.set_value(bool(value), fully=fully)
            except TypeError:
                info['used_fully_arg'] = False
                ok = state.set_value(bool(value))
            info['set_ok'] = bool(ok)
            info['success'] = bool(ok)
        except Exception as exc:
            info['error'] = repr(exc)
        return info

    def _set_open_joints_directly(self, og_obj: Any, state: Any, value: bool, *, error: Any) -> dict[str, Any]:
        try:
            joints_info = getattr(state, 'relevant_joints_info', None)
            if not joints_info:
                from omnigibson.object_states.open_state import _get_relevant_joints
                joints_info = _get_relevant_joints(og_obj)
            both_sides, relevant_joints, joint_directions = joints_info
        except Exception as fallback_exc:
            return {'success': False, 'mode': 'direct_joint_fallback', 'error': repr(error), 'fallback_error': repr(fallback_exc)}
        records: list[dict[str, Any]] = []
        try:
            for idx, joint in enumerate(list(relevant_joints or [])):
                direction = 1
                if idx < len(joint_directions or []):
                    direction = 1 if int(joint_directions[idx]) >= 0 else -1
                open_end = float(joint.upper_limit if direction >= 0 else joint.lower_limit)
                closed_end = float(joint.lower_limit if direction >= 0 else joint.upper_limit)
                target_pos = open_end if bool(value) else closed_end
                before = self._joint_position(joint)
                joint.set_pos(target_pos)
                records.append({'joint_name': str(getattr(joint, 'joint_name', '') or getattr(joint, 'name', '')), 'joint_direction': int(direction), 'before': before, 'after': self._joint_position(joint), 'target': float(target_pos)})
        except Exception as fallback_exc:
            return {'success': False, 'mode': 'direct_joint_fallback', 'error': repr(error), 'fallback_error': repr(fallback_exc), 'joints': records}
        snapshot = self._open_joint_snapshot(og_obj, state, requested_value=bool(value))
        return {'success': self._open_visual_request_satisfied(self._boolean_state_value(state), snapshot, bool(value)), 'mode': 'direct_joint_fallback', 'error': repr(error), 'both_sides': bool(both_sides), 'joints': records, 'verification': snapshot}

    def _set_open_joints_to_openness(self, og_obj: Any, state: Any, openness: float) -> dict[str, Any]:
        requested = max(0.0, min(1.0, float(openness)))
        try:
            joints_info = getattr(state, 'relevant_joints_info', None)
            if not joints_info:
                from omnigibson.object_states.open_state import _get_relevant_joints
                joints_info = _get_relevant_joints(og_obj)
            both_sides, relevant_joints, joint_directions = joints_info
        except Exception as exc:
            return {'success': False, 'mode': 'direct_joint_openness', 'openness': requested, 'error': repr(exc)}
        records: list[dict[str, Any]] = []
        try:
            for idx, joint in enumerate(list(relevant_joints or [])):
                direction = 1
                if idx < len(joint_directions or []):
                    direction = 1 if int(joint_directions[idx]) >= 0 else -1
                open_end = float(joint.upper_limit if direction >= 0 else joint.lower_limit)
                closed_end = float(joint.lower_limit if direction >= 0 else joint.upper_limit)
                target_pos = closed_end + requested * (open_end - closed_end)
                before = self._joint_position(joint)
                joint.set_pos(float(target_pos))
                after = self._joint_position(joint)
                records.append({'joint_name': str(getattr(joint, 'joint_name', '') or getattr(joint, 'name', '')), 'joint_direction': int(direction), 'before': before, 'after': after, 'target': float(target_pos), 'matches_target': self._joint_position_matches_target(after, target_pos)})
        except Exception as exc:
            return {'success': False, 'mode': 'direct_joint_openness', 'openness': requested, 'error': repr(exc), 'joints': records}
        readable = [record for record in records if record.get('after') is not None]
        return {'success': bool(readable) and all((bool(record.get('matches_target')) for record in readable)), 'mode': 'direct_joint_openness', 'openness': requested, 'both_sides': bool(both_sides), 'joints': records}

    def _open_joint_snapshot(self, og_obj: Any, state: Any, *, requested_value: bool | None=None, requested_openness: float | None=None) -> dict[str, Any]:
        snapshot: dict[str, Any] = {'success': False, 'requested_open': None if requested_value is None else bool(requested_value), 'requested_openness': None if requested_openness is None else max(0.0, min(1.0, float(requested_openness))), 'both_sides': None, 'joints': [], 'readable_joint_count': 0, 'matching_joint_count': 0, 'relevant_joint_count': 0, 'error': None}
        try:
            joints_info = getattr(state, 'relevant_joints_info', None)
            if not joints_info:
                from omnigibson.object_states.open_state import _get_relevant_joints
                joints_info = _get_relevant_joints(og_obj)
            both_sides, relevant_joints, joint_directions = joints_info
        except Exception as exc:
            snapshot['error'] = repr(exc)
            return snapshot
        snapshot['both_sides'] = bool(both_sides)
        relevant_joints = list(relevant_joints or [])
        joint_directions = list(joint_directions or [])
        snapshot['relevant_joint_count'] = len(relevant_joints)
        for idx, joint in enumerate(relevant_joints):
            direction = 1
            if idx < len(joint_directions):
                try:
                    direction = 1 if int(joint_directions[idx]) >= 0 else -1
                except Exception:
                    direction = 1
            try:
                lower = float(getattr(joint, 'lower_limit', 0.0))
                upper = float(getattr(joint, 'upper_limit', 0.0))
                open_end = upper if direction >= 0 else lower
                closed_end = lower if direction >= 0 else upper
            except Exception:
                lower = None
                upper = None
                open_end = None
                closed_end = None
            position = self._joint_position(joint)
            if requested_openness is not None and open_end is not None and (closed_end is not None):
                target = closed_end + snapshot['requested_openness'] * (open_end - closed_end)
            else:
                target = open_end if requested_value is True else closed_end if requested_value is False else None
            readable = position is not None
            matches_target = self._joint_position_matches_target(position, target)
            snapshot['joints'].append({'joint_name': str(getattr(joint, 'joint_name', '') or getattr(joint, 'name', '')), 'joint_direction': int(direction), 'position': position, 'lower_limit': lower, 'upper_limit': upper, 'target': target, 'matches_target': matches_target})
            if readable:
                snapshot['readable_joint_count'] += 1
            if matches_target:
                snapshot['matching_joint_count'] += 1
        snapshot['success'] = bool(snapshot['readable_joint_count'])
        return snapshot

    def _open_visual_request_satisfied(self, open_value: bool | None, joint_snapshot: Mapping[str, Any], requested_value: bool) -> bool:
        if open_value is bool(requested_value):
            return True
        joints = list(joint_snapshot.get('joints') or [])
        if not joints:
            return False
        readable = [joint for joint in joints if joint.get('position') is not None]
        if not readable:
            return False
        return all((bool(joint.get('matches_target')) for joint in readable))

    def _open_openness_request_satisfied(self, joint_snapshot: Mapping[str, Any]) -> bool:
        joints = list(joint_snapshot.get('joints') or [])
        readable = [joint for joint in joints if joint.get('position') is not None]
        if not readable:
            return False
        return all((bool(joint.get('matches_target')) for joint in readable))

    @staticmethod
    def _joint_position_matches_target(position: float | None, target: float | None) -> bool:
        if position is None or target is None:
            return False
        try:
            return abs(float(position) - float(target)) <= 0.001
        except Exception:
            return False

    def _joint_position(self, joint: Any) -> float | None:
        try:
            pos = joint.get_state()[0]
            if hasattr(pos, 'detach'):
                pos = pos.detach()
            if hasattr(pos, 'cpu'):
                pos = pos.cpu()
            if hasattr(pos, 'item'):
                return float(pos.item())
            return float(pos)
        except Exception:
            return None

    def _dirty_visual_system_for(self, object_name: str) -> Any | None:
        existing_name = self.dirty_visual_system_by_object.get(object_name)
        candidates = [existing_name] if existing_name else []
        candidates.extend((name for name in (self.config.dirty_visual_system, 'stain', 'mud', 'dust') if name and name not in candidates))
        scene = self.env.scene
        for system_name in candidates:
            try:
                system = scene.get_system(system_name)
            except Exception:
                continue
            try:
                is_visual = bool(scene.is_visual_particle_system(system.name))
            except Exception:
                is_visual = False
            if is_visual or system_name == existing_name:
                self.dirty_visual_system_by_object[object_name] = system.name
                return system
        return None

    def _covered_value(self, state: Any, system: Any) -> bool | None:
        try:
            return bool(state.get_value(system))
        except Exception:
            return None

    def _covered_particle_count(self, state: Any, system: Any) -> int | None:
        group = getattr(state, '_visual_particle_group', None) or getattr(getattr(state, 'obj', None), 'name', None)
        if not group:
            return None
        try:
            if group not in system.groups:
                return 0
            return int(system.num_group_particles(group=group))
        except Exception:
            return None

    def _set_manual_native_dirty_particles(self, og_obj: Any, state: Any, system: Any, dirty: bool) -> dict[str, Any]:
        group = getattr(state, '_visual_particle_group', None)
        if not group:
            return {'success': False, 'error': 'missing_visual_particle_group'}
        try:
            if group not in system.groups:
                system.create_attachment_group(obj=og_obj)
            system.remove_all_group_particles(group=group)
            if not dirty:
                return {'success': True, 'mode': 'manual_native_particles', 'group': group, 'particles': 0}
            positions = self._manual_dirty_particle_positions(og_obj)
            link_path = getattr(getattr(og_obj, 'root_link', None), 'prim_path', None) or self._object_prim_path(og_obj)
            if not positions or not link_path:
                return {'success': False, 'error': 'missing_particle_positions_or_link', 'group': group}
            try:
                import torch as th
                tensor_positions = [th.tensor(position, dtype=th.float32) for position in positions]
                orientations = th.zeros((len(tensor_positions), 4), dtype=th.float32)
                orientations[:, -1] = 1.0
            except Exception:
                tensor_positions = [np.asarray(position, dtype=float) for position in positions]
                orientations = np.zeros((len(tensor_positions), 4), dtype=float)
                orientations[:, -1] = 1.0
            scales = self._manual_dirty_particle_scales(system, og_obj, len(tensor_positions))
            system.generate_group_particles(group=group, positions=tensor_positions, orientations=orientations, scales=scales, link_prim_paths=[str(link_path)] * len(tensor_positions))
            count = self._covered_particle_count(state, system)
            return {'success': bool(count and count > 0), 'mode': 'manual_native_particles', 'group': group, 'particles': count, 'link_prim_path': str(link_path), 'scale_mode': 'visible_small_object_stain' if scales is not None else 'system_default', 'scale_count': len(scales) if scales is not None else None, 'positions': [[float(value) for value in position] for position in positions]}
        except Exception as exc:
            return {'success': False, 'error': repr(exc), 'mode': 'manual_native_particles', 'group': group}

    def _manual_dirty_particle_positions(self, og_obj: Any) -> list[list[float]]:
        aabb = self._object_aabb_low_high(og_obj)
        if aabb is None:
            return []
        low, high = aabb
        low = np.array(low, dtype=float)
        high = np.array(high, dtype=float)
        center = (low + high) * 0.5
        extent = np.maximum(high - low, 1e-06)
        radius_x = max(min(float(extent[0]) * 0.36, 0.085), 0.025)
        radius_y = max(min(float(extent[1]) * 0.36, 0.085), 0.025)
        z = float(high[2] + max(float(extent[2]) * 0.045, 0.006))
        offsets = ((0.0, 0.0), (radius_x * 0.42, 0.0), (-radius_x * 0.42, 0.0), (0.0, radius_y * 0.42), (0.0, -radius_y * 0.42))
        return [[float(center[0] + dx), float(center[1] + dy), z] for dx, dy in offsets]

    def _manual_dirty_particle_scales(self, system: Any, og_obj: Any, count: int) -> Any | None:
        if count <= 0:
            return None
        aabb = self._object_aabb_low_high(og_obj)
        if aabb is None:
            return None
        try:
            particle_extent = np.array(getattr(system.particle_object, 'aabb_extent'), dtype=float)
        except Exception:
            return None
        if particle_extent.size < 2:
            return None
        low, high = aabb
        extent = np.maximum(np.array(high, dtype=float) - np.array(low, dtype=float), 1e-06)
        min_xy = float(max(min(extent[0], extent[1]), 1e-06))
        base_xy = float(max(particle_extent[0], particle_extent[1], 1e-06))
        desired_xy = float(np.clip(min_xy * 0.12, 0.015, 0.035))
        scale_xy = float(np.clip(desired_xy / base_xy, 0.2, 3.5))
        variations = np.linspace(0.88, 1.14, count, dtype=float)
        try:
            import torch as th
            scales = th.ones((count, 3), dtype=th.float32)
            scales[:, 0] = th.tensor(scale_xy * variations, dtype=th.float32)
            scales[:, 1] = th.tensor(scale_xy * variations[::-1], dtype=th.float32)
            scales[:, 2] = 1.0
            return scales
        except Exception:
            scales = np.ones((count, 3), dtype=float)
            scales[:, 0] = scale_xy * variations
            scales[:, 1] = scale_xy * variations[::-1]
            scales[:, 2] = 1.0
            return scales

    def _dirty_native_particles_need_enhancement(self, og_obj: Any, dirty: bool, particle_count: int | None) -> bool:
        if not dirty or particle_count is None:
            return False
        aabb = self._object_aabb_low_high(og_obj)
        if aabb is None:
            return False
        low, high = aabb
        extent = np.maximum(np.array(high, dtype=float) - np.array(low, dtype=float), 1e-06)
        footprint = float(extent[0] * extent[1])
        min_xy = float(min(extent[0], extent[1]))
        if min_xy > 0.35 and footprint > 0.16:
            return False
        return int(particle_count) < 5

    def _boolean_state_value(self, state: Any) -> bool | None:
        try:
            return bool(state.get_value())
        except Exception:
            return None

    def _relative_boolean_state_value(self, state: Any, system: Any) -> bool | None:
        try:
            return bool(state.get_value(system))
        except Exception:
            return None

    def _modified_particle_count(self, og_obj: Any, modified_particles_state: Any, system: Any) -> int | None:
        try:
            state = og_obj.states[modified_particles_state]
            return int(state.get_value(system))
        except Exception:
            return None

    def _container_contains_value(self, og_obj: Any, contains_state: Any, system: Any) -> bool | None:
        try:
            return bool(og_obj.states[contains_state].get_value(system))
        except Exception:
            return None

    def _container_filled_value(self, og_obj: Any, filled_state: Any, system: Any) -> bool | None:
        try:
            return bool(og_obj.states[filled_state].get_value(system))
        except Exception:
            return None

    def _contained_particle_count(self, og_obj: Any, contained_particles_state: Any, system: Any) -> int | None:
        try:
            value = og_obj.states[contained_particles_state].get_value(system)
            return int(value.n_in_volume)
        except Exception:
            return None

    def _set_container_liquid_visual_mesh(self, og_obj: Any, name: str, substance: str, present: bool) -> dict[str, Any]:
        prim_path = self._container_liquid_prim_path(og_obj, substance)
        before_visible = self._liquid_visual_visible(prim_path)
        if not present:
            if before_visible is not None:
                self._set_liquid_visual_visibility(prim_path, False)
            info = {'success': self._liquid_visual_visible(prim_path) is not True, 'object': name, 'substance': str(substance), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': self._liquid_visual_visible(prim_path), 'created': False, 'visual_mode': 'transparent_visual_cylinder'}
            self.container_liquid_visuals[name, str(substance)] = info
            return info
        geometry = self._container_liquid_geometry(og_obj)
        material_info = self._ensure_liquid_visual_material(substance)
        cylinder_info = self._create_or_update_liquid_cylinder(prim_path, geometry, material_info, visible=True)
        self._set_liquid_visual_visibility(prim_path, True)
        after_visible = self._liquid_visual_visible(prim_path)
        info = {'success': bool(material_info.get('success')) and bool(cylinder_info.get('success')) and (after_visible is True), 'object': name, 'substance': str(substance), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible, 'created': before_visible is None, 'visual_mode': 'visual_mesh_water', 'geometry': geometry, 'material': self._visual_info_payload(material_info), 'cylinder': cylinder_info}
        self.container_liquid_visuals[name, str(substance)] = info
        return info

    def _set_container_visual_mesh_water(self, og_obj: Any, name: str, substance: str, present: bool) -> dict[str, Any]:
        paths = self._container_liquid_visual_mesh_paths(og_obj, substance)
        before_body_visible = self._liquid_visual_visible(paths['body'])
        before_surface_visible = self._liquid_visual_visible(paths['surface'])
        if not present:
            set_visual_mesh_water_visibility(self._usd_stage(), (paths['body'], paths['surface']), False, edit_context=self._usd_edit_context())
            after_body_visible = self._liquid_visual_visible(paths['body'])
            after_surface_visible = self._liquid_visual_visible(paths['surface'])
            info = {'success': after_body_visible is not True and after_surface_visible is not True, 'object': name, 'substance': str(substance), 'visual_mode': 'hq_scene_with_visual_mesh_water', 'prim_path': paths['body'], 'body_prim_path': paths['body'], 'surface_prim_path': paths['surface'], 'before_visible': before_body_visible or before_surface_visible, 'after_visible': after_body_visible or after_surface_visible, 'before_body_visible': before_body_visible, 'before_surface_visible': before_surface_visible, 'after_body_visible': after_body_visible, 'after_surface_visible': after_surface_visible, 'created': False}
            self.container_liquid_visuals[name, str(substance)] = info
            return info
        water = create_visual_mesh_water(self._usd_stage(), og_obj, VisualMeshWaterConfig(color=tuple((float(value) for value in self.config.liquid_visual_color)), body_opacity=float(self.config.liquid_visual_body_opacity), surface_opacity=float(self.config.liquid_visual_surface_opacity), roughness=float(self.config.liquid_visual_roughness), radius_scale=float(self.config.liquid_visual_radius_scale), height_scale=float(self.config.liquid_visual_height_scale), bottom_margin_scale=float(self.config.liquid_visual_bottom_margin_scale), top_margin_scale=float(self.config.liquid_visual_top_margin_scale), min_radius=float(self.config.liquid_visual_min_radius), min_height=float(self.config.liquid_visual_min_height), surface_height=float(self.config.liquid_visual_surface_height), surface_lift=float(self.config.liquid_visual_surface_lift), surface_radius_scale=float(self.config.liquid_visual_surface_radius_scale)), body_path=paths['body'], surface_path=paths['surface'], body_material_path=paths['body_material'], surface_material_path=paths['surface_material'], edit_context=self._usd_edit_context())
        geometry = water['geometry']
        body = water['body']
        surface = water['surface']
        body_material = water['body_material']
        surface_material = water['surface_material']
        try:
            self.og.sim.step()
        except Exception:
            pass
        after_body_visible = self._liquid_visual_visible(paths['body'])
        after_surface_visible = self._liquid_visual_visible(paths['surface'])
        info = {'success': after_body_visible is True and after_surface_visible is True, 'object': name, 'substance': str(substance), 'visual_mode': 'hq_scene_with_visual_mesh_water', 'prim_path': paths['body'], 'body_prim_path': paths['body'], 'surface_prim_path': paths['surface'], 'before_visible': before_body_visible or before_surface_visible, 'after_visible': after_body_visible or after_surface_visible, 'before_body_visible': before_body_visible, 'before_surface_visible': before_surface_visible, 'after_body_visible': after_body_visible, 'after_surface_visible': after_surface_visible, 'created': before_body_visible is None and before_surface_visible is None, 'geometry': geometry, 'body': body, 'surface': surface, 'body_material': body_material, 'surface_material': surface_material}
        self.container_liquid_visuals[name, str(substance)] = info
        return info

    def _set_container_liquid_compare_visual_mesh(self, og_obj: Any, name: str, substance: str, present: bool) -> dict[str, Any]:
        return self._set_container_visual_mesh_water(og_obj, name, substance, present)

    def _set_container_liquid_official_filled_particles(self, og_obj: Any, name: str, substance: str, present: bool) -> dict[str, Any]:
        try:
            from omnigibson import object_states
        except Exception as exc:
            return self._container_liquid_visual_mesh_fallback(og_obj, name, substance, present, official_error=repr(exc), official_stage='import_object_states')
        states = getattr(og_obj, 'states', {}) or {}
        Filled = getattr(object_states, 'Filled', None)
        Contains = getattr(object_states, 'Contains', None)
        ContainedParticles = getattr(object_states, 'ContainedParticles', None)
        if Filled is None or Filled not in states:
            return self._container_liquid_visual_mesh_fallback(og_obj, name, substance, present, official_error='missing_filled_state', official_stage='state_validation', states=sorted((state.__name__ for state in states)))
        if Contains is None or Contains not in states:
            return self._container_liquid_visual_mesh_fallback(og_obj, name, substance, present, official_error='missing_contains_state', official_stage='state_validation', states=sorted((state.__name__ for state in states)))
        try:
            system = self._get_fluid_system_for_container_visual(substance)
        except Exception as exc:
            return self._container_liquid_visual_mesh_fallback(og_obj, name, substance, present, official_error=repr(exc), official_stage='get_fluid_system')
        particle_material = self._tint_fluid_particle_material(system, substance)
        self._clear_container_state_caches()
        before_contains = self._container_contains_value(og_obj, Contains, system)
        before_filled = self._container_filled_value(og_obj, Filled, system)
        before_particles = self._contained_particle_count(og_obj, ContainedParticles, system) if ContainedParticles is not None and ContainedParticles in states else None
        before_system_particles = self._system_particle_count(system)
        operation_result: Any = None
        operation_error: str | None = None
        if present:
            try:
                operation_result = states[Filled].set_value(system, True)
            except Exception as exc:
                operation_error = repr(exc)
        else:
            try:
                operation_result = states[Contains].set_value(system, False)
            except Exception as exc:
                operation_error = repr(exc)
        self._clear_container_state_caches()
        after_contains = self._container_contains_value(og_obj, Contains, system)
        after_filled = self._container_filled_value(og_obj, Filled, system)
        after_particles = self._contained_particle_count(og_obj, ContainedParticles, system) if ContainedParticles is not None and ContainedParticles in states else None
        after_system_particles = self._system_particle_count(system)
        try:
            visual_mesh = self._set_container_visual_mesh_water(og_obj, name, substance, present)
        except Exception as exc:
            visual_mesh = {'success': False, 'object': name, 'substance': str(substance), 'visual_mode': 'hq_scene_with_visual_mesh_water', 'error': repr(exc)}
        operation_success = operation_error is None and (operation_result is None or bool(operation_result))
        if present:
            state_success = (after_particles is None or after_particles > 0) and (after_filled is True if after_filled is not None else True)
        else:
            state_success = (after_particles == 0 if after_particles is not None else after_contains is False) and (after_filled is False if after_filled is not None else True)
        visual_success = bool(visual_mesh.get('success'))
        info = {'success': bool(operation_success and state_success and visual_success), 'object': name, 'substance': str(substance), 'system': str(getattr(system, 'name', substance)), 'visual_mode': 'official_filled_particles', 'visual_overlay_mode': 'hq_scene_with_visual_mesh_water', 'method': 'object_states.Filled.set_value(water, True)' if present else 'object_states.Contains.set_value(water, False)', 'prim_path': visual_mesh.get('prim_path'), 'before_visible': visual_mesh.get('before_visible'), 'after_visible': visual_mesh.get('after_visible'), 'before_contains': before_contains, 'after_contains': after_contains, 'before_filled': before_filled, 'after_filled': after_filled, 'before_particles': before_particles, 'after_particles': after_particles, 'before_system_particles': before_system_particles, 'after_system_particles': after_system_particles, 'particle_material': particle_material, 'visual_mesh': visual_mesh}
        if operation_error is not None:
            info['error'] = operation_error
        self.container_liquid_visuals[name, str(substance)] = info
        return info

    def _container_liquid_visual_mesh_fallback(self, og_obj: Any, name: str, substance: str, present: bool, *, official_error: str, official_stage: str, **payload: Any) -> dict[str, Any]:
        try:
            visual_mesh = self._set_container_visual_mesh_water(og_obj, name, substance, bool(present))
        except Exception as exc:
            visual_mesh = {'success': False, 'object': name, 'substance': str(substance), 'visual_mode': 'hq_scene_with_visual_mesh_water', 'error': repr(exc)}
        info = {'success': bool(visual_mesh.get('success')), 'object': name, 'substance': str(substance), 'visual_mode': 'official_filled_particles', 'visual_overlay_mode': 'hq_scene_with_visual_mesh_water', 'method': 'visual_mesh_fallback_after_official_failure', 'official_error': official_error, 'official_stage': official_stage, 'prim_path': visual_mesh.get('prim_path'), 'before_visible': visual_mesh.get('before_visible'), 'after_visible': visual_mesh.get('after_visible'), 'visual_mesh': visual_mesh, **payload}
        self.container_liquid_visuals[name, str(substance)] = info
        return info

    def _get_fluid_system_for_container_visual(self, substance: str) -> Any:
        system = self._get_system(substance, force_init=False)
        if bool(getattr(system, 'initialized', False)):
            return system
        if bool(self.config.fluid_hq_rendering):
            return self._get_system(substance, force_init=True)
        from omnigibson.macros import gm
        with gm.unlocked():
            original_hq_rendering = bool(gm.ENABLE_HQ_RENDERING)
            gm.ENABLE_HQ_RENDERING = False
            try:
                return self._get_system(substance, force_init=True)
            finally:
                gm.ENABLE_HQ_RENDERING = original_hq_rendering

    def _container_liquid_geometry(self, og_obj: Any) -> dict[str, Any]:
        extent = self._as_float_triplet(getattr(og_obj, 'aabb_extent', (0.12, 0.12, 0.14)))
        local_center = self._object_aabb_local_center(og_obj)
        min_xy_extent = max(min(extent[0], extent[1]), 1e-06)
        z_extent = max(extent[2], self.config.liquid_visual_min_height)
        radius = max(min_xy_extent * self.config.liquid_visual_radius_scale, self.config.liquid_visual_min_radius)
        bottom_margin = max(z_extent * self.config.liquid_visual_bottom_margin_scale, self.config.liquid_visual_min_height)
        top_margin = max(z_extent * self.config.liquid_visual_top_margin_scale, self.config.liquid_visual_min_height)
        available_height = max(z_extent - bottom_margin - top_margin, self.config.liquid_visual_min_height)
        height = min(max(z_extent * self.config.liquid_visual_height_scale, self.config.liquid_visual_min_height), available_height)
        bottom_z = local_center[2] - z_extent * 0.5 + bottom_margin
        local_translation = (local_center[0], local_center[1], bottom_z + height * 0.5)
        surface_translation = (local_center[0], local_center[1], bottom_z + height + float(self.config.liquid_visual_surface_lift))
        return {'radius': float(radius), 'height': float(height), 'local_translation': [float(value) for value in local_translation], 'body_translation': [float(value) for value in local_translation], 'surface_translation': [float(value) for value in surface_translation], 'surface_height': float(self.config.liquid_visual_surface_height), 'aabb_extent': [float(value) for value in extent], 'aabb_local_center': [float(value) for value in local_center], 'bottom_margin': float(bottom_margin), 'top_margin': float(top_margin)}

    def _object_aabb_local_center(self, og_obj: Any) -> tuple[float, float, float]:
        try:
            aabb_center = np.array(self._as_float_triplet(getattr(og_obj, 'aabb_center')), dtype=float)
            root_position, root_orientation = og_obj.get_position_orientation(frame='scene')
            root_position_np = np.array(self._as_float_triplet(root_position), dtype=float)
            root_orientation_np = np.array(self._as_float_tuple(root_orientation, expected_len=4), dtype=float)
            local = self._quat_xyzw_inverse_rotate(aabb_center - root_position_np, root_orientation_np)
            return (float(local[0]), float(local[1]), float(local[2]))
        except Exception:
            return (0.0, 0.0, 0.0)

    def _ensure_liquid_visual_material(self, substance: str) -> dict[str, Any]:
        color = np.clip(np.array(self.config.liquid_visual_color, dtype=float), 0.0, 1.0)
        opacity = self._liquid_visual_opacity()
        material_path = f'/World/Looks/visual_{self._safe_prim_token(substance)}_liquid_preview'
        try:
            from isaacsim.core.api.materials import PreviewSurface
            from pxr import Sdf
            with self._usd_edit_context():
                material = PreviewSurface(prim_path=material_path, name=f'visual_{self._safe_prim_token(substance)}_liquid', color=color, roughness=0.02, metallic=0.0)
                self._set_shader_input(material.shaders_list[0], 'opacity', Sdf.ValueTypeNames.Float, opacity)
            return {'success': True, 'method': 'isaacsim.core.api.materials.PreviewSurface', 'material_path': material_path, 'visual_material': material, 'color': color.tolist(), 'opacity': opacity, 'roughness': 0.02, 'metallic': 0.0}
        except Exception as wrapper_exc:
            return self._ensure_liquid_visual_material_usd(substance, material_path, color, opacity, wrapper_error=repr(wrapper_exc))

    def _ensure_liquid_visual_material_usd(self, substance: str, material_path: str, color: np.ndarray, opacity: float, *, wrapper_error: str) -> dict[str, Any]:
        try:
            from pxr import Gf, Sdf, UsdShade
            stage = self._usd_stage()
            created = False
            material_prim = stage.GetPrimAtPath(material_path)
            with self._usd_edit_context():
                if not material_prim.IsValid():
                    UsdShade.Material.Define(stage, material_path)
                    created = True
                    material_prim = stage.GetPrimAtPath(material_path)
                material = UsdShade.Material(material_prim)
                shader = UsdShade.Shader.Define(stage, f'{material_path}/shader')
                shader.CreateIdAttr('UsdPreviewSurface')
                self._set_shader_input(shader, 'diffuseColor', Sdf.ValueTypeNames.Float3, Gf.Vec3f(*color.tolist()))
                self._set_shader_input(shader, 'opacity', Sdf.ValueTypeNames.Float, opacity)
                self._set_shader_input(shader, 'roughness', Sdf.ValueTypeNames.Float, 0.02)
                self._set_shader_input(shader, 'metallic', Sdf.ValueTypeNames.Float, 0.0)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            return {'success': True, 'method': 'usd_preview_surface_material', 'material_path': material_path, 'created': created, 'color': color.tolist(), 'opacity': opacity, 'roughness': 0.02, 'metallic': 0.0, 'wrapper_error': wrapper_error}
        except Exception as exc:
            return {'success': False, 'method': 'usd_preview_surface_material', 'material_path': material_path, 'color': color.tolist(), 'opacity': opacity, 'wrapper_error': wrapper_error, 'error': repr(exc)}

    def _create_or_update_liquid_cylinder(self, prim_path: str, geometry: Mapping[str, Any], material_info: Mapping[str, Any], *, visible: bool) -> dict[str, Any]:
        visual_material = material_info.get('visual_material')
        errors: dict[str, str] = {}
        if visual_material is not None:
            try:
                from isaacsim.core.api.objects import VisualCylinder
                with self._usd_edit_context():
                    VisualCylinder(prim_path=prim_path, name=self._safe_prim_token(prim_path.rsplit('/', 1)[-1]).lower(), translation=np.array(geometry['local_translation'], dtype=float), radius=float(geometry['radius']), height=float(geometry['height']), visible=visible, visual_material=visual_material)
                    display_attrs = self._set_liquid_visual_display_attrs(prim_path)
                return {'success': True, 'method': 'isaacsim.core.api.objects.VisualCylinder', 'prim_path': prim_path, 'radius': float(geometry['radius']), 'height': float(geometry['height']), 'local_translation': list(geometry['local_translation']), 'display_attrs': display_attrs}
            except Exception as exc:
                errors['visual_cylinder'] = repr(exc)
        try:
            stage = self._usd_stage()
            from pxr import Gf, UsdGeom, UsdShade
            with self._usd_edit_context():
                cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
                prim = cylinder.GetPrim()
                cylinder.GetRadiusAttr().Set(float(geometry['radius']))
                cylinder.GetHeightAttr().Set(float(geometry['height']))
                cylinder.GetExtentAttr().Set([Gf.Vec3f(-float(geometry['radius']), -float(geometry['radius']), -float(geometry['height']) * 0.5), Gf.Vec3f(float(geometry['radius']), float(geometry['radius']), float(geometry['height']) * 0.5)])
                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                xformable.AddTranslateOp().Set(Gf.Vec3d(*geometry['local_translation']))
                display_attrs = self._set_liquid_visual_display_attrs(prim_path)
                material_path = material_info.get('material_path')
                if material_path:
                    material_prim = stage.GetPrimAtPath(str(material_path))
                    if material_prim.IsValid():
                        UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(material_prim), UsdShade.Tokens.strongerThanDescendants)
                self._set_liquid_visual_visibility(prim_path, visible)
            return {'success': True, 'method': 'usd_geom_cylinder', 'prim_path': prim_path, 'radius': float(geometry['radius']), 'height': float(geometry['height']), 'local_translation': list(geometry['local_translation']), 'display_attrs': display_attrs, 'fallback_errors': errors}
        except Exception as exc:
            errors['usd_geom_cylinder'] = repr(exc)
            return {'success': False, 'method': 'usd_geom_cylinder', 'prim_path': prim_path, 'fallback_errors': errors}

    def _set_particle_source_stream_visual(self, og_obj: Any, source_state: Any, system_name: str, visible: bool) -> dict[str, Any]:
        prim_path = self._particle_source_stream_prim_path(og_obj, system_name)
        object_name = str(getattr(og_obj, 'name', 'particle_source'))
        if not self.config.enable_particle_source_stream_visuals:
            return {'success': True, 'skipped': True, 'reason': 'particle_source_stream_visuals_disabled', 'object': object_name, 'system': str(system_name), 'prim_path': prim_path, 'before_visible': None, 'after_visible': None}
        before_visible = self._liquid_visual_visible(prim_path)
        if not visible:
            if before_visible is not None:
                self._force_hide_liquid_visual_for_render(prim_path)
            after_visible = self._liquid_visual_visible(prim_path)
            info = {'success': after_visible is not True, 'object': object_name, 'system': str(system_name), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible, 'removed': False, 'retained_hidden': before_visible is not None, 'created': False, 'visual_mode': 'particle_source_stream_continuous_mesh'}
            self.particle_source_stream_visuals[object_name, str(system_name)] = info
            return info
        try:
            geometry = self._particle_source_stream_geometry(og_obj, source_state)
            material_info = self._ensure_particle_source_stream_material(system_name)
            stream_info = self._create_or_update_particle_source_stream_visual(prim_path, geometry, material_info, visible=True)
            self._set_liquid_visual_visibility(prim_path, True)
            after_visible = self._liquid_visual_visible(prim_path)
            info = {'success': bool(material_info.get('success')) and bool(stream_info.get('success')) and (after_visible is True), 'object': object_name, 'system': str(system_name), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible, 'created': before_visible is None, 'visual_mode': str(stream_info.get('visual_mode', 'particle_source_stream_continuous_mesh')), 'geometry': geometry, 'material': self._visual_info_payload(material_info), 'stream_mesh': stream_info}
        except Exception as exc:
            info = {'success': False, 'object': object_name, 'system': str(system_name), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': self._liquid_visual_visible(prim_path), 'visual_mode': 'particle_source_stream_continuous_mesh', 'error': repr(exc)}
        self.particle_source_stream_visuals[object_name, str(system_name)] = info
        return info

    def _particle_source_stream_geometry(self, og_obj: Any, source_state: Any) -> dict[str, Any]:
        start = np.array(self._particle_source_world_position(og_obj, source_state), dtype=float)
        radius = max(float(self.config.particle_source_stream_radius), 0.002)
        length = self._particle_source_stream_length(og_obj, start)
        end = start + np.array([0.0, 0.0, -length], dtype=float)
        translation = (float(start[0]), float(start[1]), float(start[2] - length * 0.5))
        return {'radius': float(radius), 'height': float(length), 'world_translation': list(translation), 'source_position': [float(value) for value in start], 'end_position': [float(value) for value in end], 'direction': [0.0, 0.0, -1.0], 'style': self._particle_source_stream_style(), 'segments': int(np.clip(int(self.config.particle_source_stream_segments), 4, 64)), 'radial_segments': int(np.clip(int(self.config.particle_source_stream_radial_segments), 6, 32)), 'wobble': float(max(float(self.config.particle_source_stream_wobble), 0.0)), 'taper': float(np.clip(float(self.config.particle_source_stream_taper), 0.0, 0.85)), 'droplets': int(np.clip(int(self.config.particle_source_stream_droplets), 0, 16))}

    def _particle_source_stream_style(self) -> str:
        style = str(getattr(self.config, 'particle_source_stream_style', 'continuous_mesh')).lower().strip()
        return style if style in {'continuous_mesh', 'cylinder'} else 'continuous_mesh'

    def _particle_source_world_position(self, og_obj: Any, source_state: Any) -> tuple[float, float, float]:
        for holder in (getattr(source_state, 'link', None), getattr(source_state, 'projection_mesh', None)):
            if holder is None:
                continue
            get_pose = getattr(holder, 'get_position_orientation', None)
            if get_pose is None:
                continue
            try:
                position, _ = get_pose(frame='scene')
            except TypeError:
                position, _ = get_pose()
            return self._as_float_triplet(position)
        try:
            return self._as_float_triplet(getattr(og_obj, 'aabb_center'))
        except Exception:
            return (0.0, 0.0, 0.75)

    def _particle_source_stream_length(self, og_obj: Any, start: np.ndarray) -> float:
        default_length = max(float(self.config.particle_source_stream_length), 0.05)
        radius = max(float(self.config.particle_source_stream_radius), 0.002)
        min_length = max(radius * 4.0, 0.08)
        max_length = max(default_length * 1.8, 0.65)
        length = default_length
        aabb = self._object_aabb_low_high(og_obj)
        if aabb is not None:
            low, high = aabb
            z_extent = max(float(high[2] - low[2]), 1e-06)
            basin_z = float(low[2] + z_extent * 0.36)
            candidate = float(start[2] - basin_z)
            if min_length <= candidate <= max_length:
                length = candidate
        return float(np.clip(length, min_length, max_length))

    def _object_aabb_low_high(self, og_obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
        aabb = getattr(og_obj, 'aabb', None)
        if callable(aabb):
            aabb = aabb()
        if aabb is not None:
            try:
                if hasattr(aabb, 'detach'):
                    aabb = aabb.detach().cpu().numpy()
                arr = np.array(aabb, dtype=float)
                if arr.shape == (2, 3):
                    return (arr[0], arr[1])
            except Exception:
                pass
        try:
            center = np.array(self._as_float_triplet(getattr(og_obj, 'aabb_center')), dtype=float)
            extent = np.array(self._as_float_triplet(getattr(og_obj, 'aabb_extent')), dtype=float)
            return (center - extent * 0.5, center + extent * 0.5)
        except Exception:
            return None

    def _ensure_particle_source_stream_material(self, system_name: str) -> dict[str, Any]:
        color = np.clip(np.array(self.config.particle_source_stream_color, dtype=float), 0.0, 1.0)
        opacity = float(np.clip(float(self.config.particle_source_stream_opacity), 0.0, 1.0))
        material_path = f'/World/Looks/visual_{self._safe_prim_token(system_name)}_particle_source_stream'
        try:
            from isaacsim.core.api.materials import PreviewSurface
            from pxr import Gf, Sdf
            with self._usd_edit_context():
                material = PreviewSurface(prim_path=material_path, name=f'visual_{self._safe_prim_token(system_name)}_particle_source_stream', color=color, roughness=0.0, metallic=0.0)
                shader = material.shaders_list[0]
                self._set_shader_input(shader, 'opacity', Sdf.ValueTypeNames.Float, opacity)
                self._set_shader_input(shader, 'emissiveColor', Sdf.ValueTypeNames.Float3, Gf.Vec3f(*color.tolist()))
            return {'success': True, 'method': 'isaacsim.core.api.materials.PreviewSurface', 'material_path': material_path, 'visual_material': material, 'color': color.tolist(), 'opacity': opacity, 'roughness': 0.0, 'metallic': 0.0}
        except Exception as wrapper_exc:
            return self._ensure_particle_source_stream_material_usd(system_name, material_path, color, opacity, wrapper_error=repr(wrapper_exc))

    def _ensure_particle_source_stream_material_usd(self, system_name: str, material_path: str, color: np.ndarray, opacity: float, *, wrapper_error: str) -> dict[str, Any]:
        try:
            from pxr import Gf, Sdf, UsdShade
            stage = self._usd_stage()
            created = False
            with self._usd_edit_context():
                material_prim = stage.GetPrimAtPath(material_path)
                if not material_prim.IsValid():
                    UsdShade.Material.Define(stage, material_path)
                    created = True
                    material_prim = stage.GetPrimAtPath(material_path)
                material = UsdShade.Material(material_prim)
                shader = UsdShade.Shader.Define(stage, f'{material_path}/shader')
                shader.CreateIdAttr('UsdPreviewSurface')
                self._set_shader_input(shader, 'diffuseColor', Sdf.ValueTypeNames.Float3, Gf.Vec3f(*color.tolist()))
                self._set_shader_input(shader, 'opacity', Sdf.ValueTypeNames.Float, opacity)
                self._set_shader_input(shader, 'roughness', Sdf.ValueTypeNames.Float, 0.0)
                self._set_shader_input(shader, 'metallic', Sdf.ValueTypeNames.Float, 0.0)
                self._set_shader_input(shader, 'emissiveColor', Sdf.ValueTypeNames.Float3, Gf.Vec3f(*color.tolist()))
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            return {'success': True, 'method': 'usd_preview_surface_material', 'material_path': material_path, 'created': created, 'color': color.tolist(), 'opacity': opacity, 'wrapper_error': wrapper_error}
        except Exception as exc:
            return {'success': False, 'method': 'usd_preview_surface_material', 'material_path': material_path, 'color': color.tolist(), 'opacity': opacity, 'wrapper_error': wrapper_error, 'error': repr(exc)}

    def _create_or_update_particle_source_stream_cylinder(self, prim_path: str, geometry: Mapping[str, Any], material_info: Mapping[str, Any], *, visible: bool) -> dict[str, Any]:
        visual_material = material_info.get('visual_material')
        color = np.clip(np.array(self.config.particle_source_stream_color, dtype=float), 0.0, 1.0)
        opacity = float(np.clip(float(self.config.particle_source_stream_opacity), 0.0, 1.0))
        errors: dict[str, str] = {}
        if visual_material is not None:
            try:
                from isaacsim.core.api.objects import VisualCylinder
                with self._usd_edit_context():
                    VisualCylinder(prim_path=prim_path, name=self._safe_prim_token(prim_path.rsplit('/', 1)[-1]).lower(), translation=np.array(geometry['world_translation'], dtype=float), radius=float(geometry['radius']), height=float(geometry['height']), visible=visible, visual_material=visual_material)
                    display_attrs = self._set_visual_display_attrs(prim_path, color, opacity)
                return {'success': True, 'method': 'isaacsim.core.api.objects.VisualCylinder', 'prim_path': prim_path, 'radius': float(geometry['radius']), 'height': float(geometry['height']), 'world_translation': list(geometry['world_translation']), 'display_attrs': display_attrs}
            except Exception as exc:
                errors['visual_cylinder'] = repr(exc)
        try:
            stage = self._usd_stage()
            from pxr import Gf, UsdGeom, UsdShade
            parent_path = prim_path.rsplit('/', 1)[0]
            with self._usd_edit_context():
                if not stage.GetPrimAtPath(parent_path).IsValid():
                    UsdGeom.Xform.Define(stage, parent_path)
                cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
                prim = cylinder.GetPrim()
                cylinder.GetRadiusAttr().Set(float(geometry['radius']))
                cylinder.GetHeightAttr().Set(float(geometry['height']))
                cylinder.GetExtentAttr().Set([Gf.Vec3f(-float(geometry['radius']), -float(geometry['radius']), -float(geometry['height']) * 0.5), Gf.Vec3f(float(geometry['radius']), float(geometry['radius']), float(geometry['height']) * 0.5)])
                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                xformable.AddTranslateOp().Set(Gf.Vec3d(*geometry['world_translation']))
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color.tolist())])
                gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
                display_attrs = {'success': True, 'display_color': color.tolist(), 'display_opacity': float(opacity)}
                material_path = material_info.get('material_path')
                if material_path:
                    material_prim = stage.GetPrimAtPath(str(material_path))
                    if material_prim.IsValid():
                        UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(material_prim), UsdShade.Tokens.strongerThanDescendants)
                imageable = UsdGeom.Imageable(prim)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()
            return {'success': True, 'method': 'usd_geom_cylinder', 'prim_path': prim_path, 'radius': float(geometry['radius']), 'height': float(geometry['height']), 'world_translation': list(geometry['world_translation']), 'display_attrs': display_attrs, 'fallback_errors': errors}
        except Exception as exc:
            errors['usd_geom_cylinder'] = repr(exc)
            return {'success': False, 'method': 'usd_geom_cylinder', 'prim_path': prim_path, 'fallback_errors': errors}

    def _create_or_update_particle_source_stream_visual(self, prim_path: str, geometry: Mapping[str, Any], material_info: Mapping[str, Any], *, visible: bool) -> dict[str, Any]:
        if str(geometry.get('style', self._particle_source_stream_style())) == 'cylinder':
            info = self._create_or_update_particle_source_stream_cylinder(prim_path, geometry, material_info, visible=visible)
            info['visual_mode'] = 'particle_source_stream_cylinder'
            return info
        mesh_info = self._create_or_update_particle_source_stream_continuous_mesh(prim_path, geometry, material_info, visible=visible)
        if mesh_info.get('success'):
            return mesh_info
        fallback = self._create_or_update_particle_source_stream_cylinder(prim_path, geometry, material_info, visible=visible)
        fallback['visual_mode'] = 'particle_source_stream_cylinder_fallback'
        fallback['mesh_error'] = self._visual_info_payload(mesh_info)
        return fallback

    def _create_or_update_particle_source_stream_continuous_mesh(self, prim_path: str, geometry: Mapping[str, Any], material_info: Mapping[str, Any], *, visible: bool) -> dict[str, Any]:
        color = np.clip(np.array(self.config.particle_source_stream_color, dtype=float), 0.0, 1.0)
        opacity = float(np.clip(float(self.config.particle_source_stream_opacity), 0.0, 1.0))
        try:
            from pxr import Gf, Sdf, UsdGeom, UsdShade
            stage = self._usd_stage()
            parent_path = prim_path.rsplit('/', 1)[0]
            main_path = f'{prim_path}/MainColumn'
            droplets_path = f'{prim_path}/Droplets'
            material_path = material_info.get('material_path')
            points, face_counts, face_indices = self._particle_source_stream_mesh_arrays(geometry)
            droplet_specs = self._particle_source_stream_droplet_specs(geometry)
            with self._usd_edit_context():
                if not stage.GetPrimAtPath(parent_path).IsValid():
                    UsdGeom.Xform.Define(stage, parent_path)
                root = UsdGeom.Xform.Define(stage, prim_path)
                mesh = UsdGeom.Mesh.Define(stage, main_path)
                mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
                mesh.CreateFaceVertexCountsAttr(face_counts)
                mesh.CreateFaceVertexIndicesAttr(face_indices)
                mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                mesh.CreateDoubleSidedAttr(True)
                self._set_visual_display_attrs(main_path, color, opacity)
                if material_path:
                    material_prim = stage.GetPrimAtPath(str(material_path))
                    if material_prim.IsValid():
                        material = UsdShade.Material(material_prim)
                        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants)
                expected_droplet_paths = {f'{droplets_path}/Droplet_{idx:02d}' for idx in range(len(droplet_specs))}
                existing_droplets = stage.GetPrimAtPath(droplets_path)
                if existing_droplets.IsValid():
                    for child in list(existing_droplets.GetChildren()):
                        if str(child.GetPath()) not in expected_droplet_paths:
                            stage.RemovePrim(str(child.GetPath()))
                if droplet_specs:
                    UsdGeom.Xform.Define(stage, droplets_path)
                elif existing_droplets.IsValid():
                    stage.RemovePrim(droplets_path)
                for idx, spec in enumerate(droplet_specs):
                    droplet_path = f'{droplets_path}/Droplet_{idx:02d}'
                    sphere = UsdGeom.Sphere.Define(stage, droplet_path)
                    sphere.GetRadiusAttr().Set(float(spec['radius']))
                    xformable = UsdGeom.Xformable(sphere.GetPrim())
                    xformable.ClearXformOpOrder()
                    xformable.AddTranslateOp().Set(Gf.Vec3d(*spec['position']))
                    xformable.AddScaleOp().Set(Gf.Vec3f(0.82, 0.82, float(spec['z_scale'])))
                    self._set_visual_display_attrs(droplet_path, color, min(1.0, opacity + 0.08))
                    if material_path:
                        material_prim = stage.GetPrimAtPath(str(material_path))
                        if material_prim.IsValid():
                            UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(UsdShade.Material(material_prim), UsdShade.Tokens.strongerThanDescendants)
                for path in [prim_path, main_path, *[f'{droplets_path}/Droplet_{idx:02d}' for idx in range(len(droplet_specs))]]:
                    prim = stage.GetPrimAtPath(path)
                    if prim.IsValid():
                        prim.CreateAttribute('primvars:doNotCastShadows', Sdf.ValueTypeNames.Bool).Set(True)
                if visible:
                    for path in [prim_path, main_path, *sorted(expected_droplet_paths)]:
                        prim = stage.GetPrimAtPath(path)
                        if prim.IsValid():
                            UsdGeom.Imageable(prim).MakeVisible()
                else:
                    UsdGeom.Imageable(root.GetPrim()).MakeInvisible()
            return {'success': True, 'method': 'usd_geom_mesh_with_droplets', 'visual_mode': 'particle_source_stream_continuous_mesh', 'prim_path': prim_path, 'main_mesh_path': main_path, 'radius': float(geometry['radius']), 'height': float(geometry['height']), 'segments': int(geometry['segments']), 'radial_segments': int(geometry['radial_segments']), 'droplets': len(droplet_specs), 'points': len(points), 'faces': len(face_counts)}
        except Exception as exc:
            return {'success': False, 'method': 'usd_geom_mesh_with_droplets', 'visual_mode': 'particle_source_stream_continuous_mesh', 'prim_path': prim_path, 'error': repr(exc)}

    def _particle_source_stream_mesh_arrays(self, geometry: Mapping[str, Any]) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
        start = np.array(geometry['source_position'], dtype=float)
        length = float(geometry['height'])
        base_radius = float(geometry['radius'])
        segments = int(geometry['segments'])
        radial_segments = int(geometry['radial_segments'])
        wobble = float(geometry['wobble'])
        taper = float(geometry['taper'])
        points: list[tuple[float, float, float]] = []
        for i in range(segments + 1):
            t = i / max(segments, 1)
            z = start[2] - length * t
            center_x = start[0] + wobble * np.sin(t * np.pi * 2.15)
            center_y = start[1] + wobble * 0.65 * np.sin(t * np.pi * 3.4 + 0.7)
            radius = base_radius * (1.0 - taper * t)
            radius *= 1.0 + 0.1 * np.sin(t * np.pi * 5.0)
            radius = max(radius, base_radius * 0.34)
            for j in range(radial_segments):
                theta = 2.0 * np.pi * j / radial_segments
                ripple = 1.0 + 0.07 * np.sin(theta * 3.0 + t * np.pi * 5.0)
                x = center_x + radius * ripple * np.cos(theta)
                y = center_y + radius * ripple * np.sin(theta)
                points.append((float(x), float(y), float(z)))
        top_center_idx = len(points)
        points.append((float(start[0]), float(start[1]), float(start[2])))
        bottom_center = np.mean(np.array(points[-radial_segments - 1:-1], dtype=float), axis=0)
        bottom_center_idx = len(points)
        points.append((float(bottom_center[0]), float(bottom_center[1]), float(bottom_center[2])))
        face_counts: list[int] = []
        face_indices: list[int] = []
        for i in range(segments):
            row = i * radial_segments
            next_row = (i + 1) * radial_segments
            for j in range(radial_segments):
                a = row + j
                b = row + (j + 1) % radial_segments
                c = next_row + (j + 1) % radial_segments
                d = next_row + j
                face_counts.append(4)
                face_indices.extend([a, b, c, d])
        for j in range(radial_segments):
            face_counts.append(3)
            face_indices.extend([top_center_idx, (j + 1) % radial_segments, j])
            bottom_row = segments * radial_segments
            face_counts.append(3)
            face_indices.extend([bottom_center_idx, bottom_row + j, bottom_row + (j + 1) % radial_segments])
        return (points, face_counts, face_indices)

    def _particle_source_stream_droplet_specs(self, geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
        start = np.array(geometry['source_position'], dtype=float)
        length = float(geometry['height'])
        radius = float(geometry['radius'])
        wobble = float(geometry['wobble'])
        count = int(geometry['droplets'])
        specs: list[dict[str, Any]] = []
        for idx in range(count):
            t = 0.72 + 0.22 * (idx / max(count - 1, 1))
            angle = idx * 2.399963229728653
            spread = radius * (0.45 + 0.1 * idx)
            pos = np.array([start[0] + wobble * np.sin(t * np.pi * 2.15) + spread * np.cos(angle), start[1] + wobble * 0.65 * np.sin(t * np.pi * 3.4 + 0.7) + spread * np.sin(angle), start[2] - length * t - radius * 0.45 * (idx % 2)], dtype=float)
            specs.append({'position': [float(value) for value in pos], 'radius': float(radius * (0.28 + 0.06 * (idx % 3))), 'z_scale': float(1.3 + 0.18 * (idx % 2))})
        return specs

    def _particle_source_stream_prim_path(self, og_obj: Any, system_name: str) -> str:
        object_name = self._safe_prim_token(getattr(og_obj, 'name', 'particle_source'))
        system_token = self._safe_prim_token(system_name)
        return f'/World/VisualParticleSourceStreams/{object_name}_{system_token}_Stream'

    def _action_liquid_stream_prim_path(self, stream_name: str, substance: str) -> str:
        stream_token = self._safe_prim_token(stream_name)
        substance_token = self._safe_prim_token(substance)
        return f'/World/VisualActionLiquidStreams/{stream_token}_{substance_token}_Stream'

    def _action_liquid_stream_geometry(self, start: tuple[float, float, float] | list[float], end: tuple[float, float, float] | list[float]) -> dict[str, Any]:
        start_np = np.array(self._as_float_triplet(start), dtype=float)
        end_np = np.array(self._as_float_triplet(end), dtype=float)
        if start_np[2] < end_np[2]:
            start_np, end_np = (end_np, start_np)
        length = max(float(start_np[2] - end_np[2]), 0.08)
        radius = max(float(self.config.particle_source_stream_radius) * 0.86, 0.006)
        return {'radius': float(radius), 'height': float(length), 'world_translation': [float(start_np[0]), float(start_np[1]), float(start_np[2] - length * 0.5)], 'source_position': [float(value) for value in start_np], 'end_position': [float(start_np[0]), float(start_np[1]), float(start_np[2] - length)], 'requested_end_position': [float(value) for value in end_np], 'direction': [0.0, 0.0, -1.0], 'style': self._particle_source_stream_style(), 'segments': int(np.clip(int(self.config.particle_source_stream_segments), 4, 64)), 'radial_segments': int(np.clip(int(self.config.particle_source_stream_radial_segments), 6, 32)), 'wobble': float(max(float(self.config.particle_source_stream_wobble), 0.0)), 'taper': float(np.clip(float(self.config.particle_source_stream_taper), 0.0, 0.85)), 'droplets': int(np.clip(int(self.config.particle_source_stream_droplets), 0, 16))}

    def _container_liquid_prim_path(self, og_obj: Any, substance: str) -> str:
        object_path = self._container_visual_parent_path(og_obj)
        return f'{object_path}/VisualLiquid_{self._safe_prim_token(substance)}'

    def _container_liquid_visual_mesh_paths(self, og_obj: Any, substance: str) -> dict[str, str]:
        object_path = self._container_visual_parent_path(og_obj)
        token = self._safe_prim_token(substance)
        return {'body': f'{object_path}/VisualWaterBody_{token}', 'surface': f'{object_path}/VisualWaterSurface_{token}', 'body_material': f'/World/Looks/visual_mesh_{token}_water_body', 'surface_material': f'/World/Looks/visual_mesh_{token}_water_surface'}

    def _make_raw_usd_preview_material(self, stage: Any, material_path: str, *, color: list[float], opacity: float, roughness: float) -> dict[str, Any]:
        from pxr import Gf, Sdf, UsdShade
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f'{material_path}/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')

        def set_input(name: str, value_type: Any, value: Any) -> None:
            shader_input = shader.GetInput(name)
            if not shader_input:
                shader_input = shader.CreateInput(name, value_type)
            shader_input.Set(value)
        set_input('diffuseColor', Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color))
        set_input('opacity', Sdf.ValueTypeNames.Float, float(opacity))
        set_input('roughness', Sdf.ValueTypeNames.Float, float(roughness))
        set_input('metallic', Sdf.ValueTypeNames.Float, 0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        return {'material_path': material_path, 'shader_path': f'{material_path}/Shader', 'color': list(color), 'opacity': float(opacity), 'roughness': float(roughness), 'method': 'UsdPreviewSurface'}

    def _create_raw_usd_liquid_cylinder(self, stage: Any, prim_path: str, *, radius: float, height: float, translation: list[float], material_path: str, color: list[float], opacity: float) -> dict[str, Any]:
        from pxr import Gf, UsdGeom, UsdShade
        cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
        prim = cylinder.GetPrim()
        cylinder.GetRadiusAttr().Set(float(radius))
        cylinder.GetHeightAttr().Set(float(height))
        cylinder.GetExtentAttr().Set([Gf.Vec3f(-float(radius), -float(radius), -float(height) * 0.5), Gf.Vec3f(float(radius), float(radius), float(height) * 0.5)])
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
        material_prim = stage.GetPrimAtPath(material_path)
        if material_prim.IsValid():
            UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(material_prim), UsdShade.Tokens.strongerThanDescendants)
        UsdGeom.Imageable(prim).MakeVisible()
        return {'success': True, 'method': 'UsdGeom.Cylinder', 'prim_path': prim_path, 'radius': float(radius), 'height': float(height), 'translation': [float(value) for value in translation], 'material_path': material_path, 'display_color': list(color), 'display_opacity': float(opacity)}

    def _object_prim_path(self, og_obj: Any) -> str:
        candidates = [getattr(og_obj, 'prim_path', None), getattr(getattr(og_obj, 'root_link', None), 'prim_path', None)]
        prim = getattr(og_obj, 'prim', None)
        if prim is not None:
            try:
                candidates.append(prim.GetPath().pathString)
            except Exception:
                pass
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return f"/World/{self._safe_prim_token(getattr(og_obj, 'name', 'container'))}"

    def _container_visual_parent_path(self, og_obj: Any) -> str:
        root_link_path = getattr(getattr(og_obj, 'root_link', None), 'prim_path', None)
        if root_link_path:
            return str(root_link_path)
        return self._object_prim_path(og_obj)

    def _set_dirty_overlay_visual(self, name: str, og_obj: Any | None, dirty: bool, reason: str) -> dict[str, Any]:
        prim_path = self._dirty_overlay_prim_path(name)
        before_visible = self._liquid_visual_visible(prim_path)
        if not dirty:
            if before_visible is not None:
                self._set_liquid_visual_visibility(prim_path, False)
            removed = self._remove_visual_prim(prim_path)
            after_visible = self._liquid_visual_visible(prim_path)
            info = {'success': after_visible is not True, 'object': str(name), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible, 'removed': removed, 'visual_mode': 'dirty_overlay_disc'}
            self.dirty_overlay_visuals[str(name)] = info
            return info
        if og_obj is None:
            return {'success': False, 'object': str(name), 'prim_path': prim_path, 'error': 'target_not_found', 'visual_mode': 'dirty_overlay_disc'}
        aabb = self._object_aabb_low_high(og_obj)
        if aabb is None:
            return {'success': False, 'object': str(name), 'prim_path': prim_path, 'error': 'missing_aabb', 'visual_mode': 'dirty_overlay_disc'}
        try:
            geometry = self._dirty_overlay_geometry(aabb)
            material_info = self._ensure_dirty_overlay_material()
            disc_info = self._create_or_update_dirty_overlay_disc(prim_path, geometry, material_info, visible=True)
            self._set_liquid_visual_visibility(prim_path, True)
            self._advance_visual_state_renders()
            after_visible = self._liquid_visual_visible(prim_path)
            success = bool(material_info.get('success')) and bool(disc_info.get('success')) and (after_visible is True)
            info = {'success': success, 'object': str(name), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible, 'created': before_visible is None, 'visual_mode': 'dirty_overlay_disc', 'geometry': geometry, 'material': self._visual_info_payload(material_info), 'disc': self._visual_info_payload(disc_info)}
        except Exception as exc:
            info = {'success': False, 'object': str(name), 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': self._liquid_visual_visible(prim_path), 'error': repr(exc), 'visual_mode': 'dirty_overlay_disc'}
        self.dirty_overlay_visuals[str(name)] = info
        return info

    def _dirty_overlay_geometry(self, aabb: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        low, high = aabb
        low = np.array(low, dtype=float)
        high = np.array(high, dtype=float)
        extent = np.maximum(high - low, 1e-06)
        center = (low + high) * 0.5
        min_xy = max(float(min(extent[0], extent[1])), 1e-06)
        radius = max(min_xy * float(self.config.dirty_overlay_radius_scale), 0.035)
        radius = min(radius, max(min_xy * 0.72, 0.035))
        height = max(float(self.config.dirty_overlay_height), min(float(extent[2]) * 0.14, 0.035))
        top_lift = float(self.config.dirty_overlay_top_lift)
        translation = (float(center[0]), float(center[1]), float(high[2] + height * 0.5 + top_lift))
        return {'radius': float(radius), 'height': float(height), 'world_translation': [float(value) for value in translation], 'aabb_low': [float(value) for value in low], 'aabb_high': [float(value) for value in high]}

    def _dirty_overlay_prim_path(self, name: str) -> str:
        return f'/World/VisualDirtyOverlays/{self._safe_prim_token(name)}_DirtyOverlay'

    def _ensure_dirty_overlay_material(self) -> dict[str, Any]:
        color = np.clip(np.array(self.config.dirty_overlay_color, dtype=float), 0.0, 1.0)
        opacity = float(np.clip(float(self.config.dirty_overlay_opacity), 0.0, 1.0))
        material_path = '/World/Looks/visual_dirty_overlay_preview'
        try:
            from isaacsim.core.api.materials import PreviewSurface
            from pxr import Sdf
            with self._usd_edit_context():
                material = PreviewSurface(prim_path=material_path, name='visual_dirty_overlay', color=color, roughness=0.92, metallic=0.0)
                self._set_shader_input(material.shaders_list[0], 'opacity', Sdf.ValueTypeNames.Float, opacity)
            return {'success': True, 'method': 'isaacsim.core.api.materials.PreviewSurface', 'material_path': material_path, 'visual_material': material, 'color': color.tolist(), 'opacity': opacity, 'roughness': 0.92, 'metallic': 0.0}
        except Exception as wrapper_exc:
            try:
                stage = self._usd_stage()
                with self._usd_edit_context():
                    material_info = self._make_raw_usd_preview_material(stage, material_path, color=color.tolist(), opacity=opacity, roughness=0.92)
                return {'success': True, 'method': 'usd_preview_surface_material', 'material_path': material_path, 'color': color.tolist(), 'opacity': opacity, 'roughness': 0.92, 'wrapper_error': repr(wrapper_exc), 'usd_material': material_info}
            except Exception as exc:
                return {'success': False, 'method': 'dirty_overlay_material', 'material_path': material_path, 'color': color.tolist(), 'opacity': opacity, 'wrapper_error': repr(wrapper_exc), 'error': repr(exc)}

    def _create_or_update_dirty_overlay_disc(self, prim_path: str, geometry: Mapping[str, Any], material_info: Mapping[str, Any], *, visible: bool) -> dict[str, Any]:
        color = np.clip(np.array(self.config.dirty_overlay_color, dtype=float), 0.0, 1.0)
        opacity = float(np.clip(float(self.config.dirty_overlay_opacity), 0.0, 1.0))
        try:
            stage = self._usd_stage()
            from pxr import Gf, UsdGeom, UsdShade
            parent_path = prim_path.rsplit('/', 1)[0]
            with self._usd_edit_context():
                if not stage.GetPrimAtPath(parent_path).IsValid():
                    UsdGeom.Xform.Define(stage, parent_path)
                cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
                prim = cylinder.GetPrim()
                cylinder.GetRadiusAttr().Set(float(geometry['radius']))
                cylinder.GetHeightAttr().Set(float(geometry['height']))
                cylinder.GetExtentAttr().Set([Gf.Vec3f(-float(geometry['radius']), -float(geometry['radius']), -float(geometry['height']) * 0.5), Gf.Vec3f(float(geometry['radius']), float(geometry['radius']), float(geometry['height']) * 0.5)])
                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                xformable.AddTranslateOp().Set(Gf.Vec3d(*geometry['world_translation']))
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color.tolist())])
                gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
                display_attrs = {'success': True, 'display_color': color.tolist(), 'display_opacity': float(opacity)}
                material_path = material_info.get('material_path')
                if material_path:
                    material_prim = stage.GetPrimAtPath(str(material_path))
                    if material_prim.IsValid():
                        UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(material_prim), UsdShade.Tokens.strongerThanDescendants)
                imageable = UsdGeom.Imageable(prim)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()
            return {'success': True, 'method': 'usd_geom_cylinder', 'prim_path': prim_path, 'radius': float(geometry['radius']), 'height': float(geometry['height']), 'world_translation': list(geometry['world_translation']), 'display_attrs': display_attrs}
        except Exception as exc:
            return {'success': False, 'method': 'usd_geom_cylinder', 'prim_path': prim_path, 'error': repr(exc)}

    def _liquid_visual_visible(self, prim_path: str) -> bool | None:
        stage = self._usd_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        from pxr import UsdGeom
        return UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible

    def _set_liquid_visual_visibility(self, prim_path: str, visible: bool) -> None:
        stage = self._usd_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        from pxr import UsdGeom
        imageable = UsdGeom.Imageable(prim)
        with self._usd_edit_context():
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def _remove_visual_prim(self, prim_path: str) -> bool:
        stage = self._usd_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False
        try:
            with self._usd_edit_context():
                return bool(stage.RemovePrim(prim_path))
        except Exception:
            return False

    def _set_liquid_visual_display_attrs(self, prim_path: str) -> dict[str, Any]:
        color = np.clip(np.array(self.config.liquid_visual_color, dtype=float), 0.0, 1.0)
        opacity = self._liquid_visual_opacity()
        return self._set_visual_display_attrs(prim_path, color, opacity)

    def _set_visual_display_attrs(self, prim_path: str, color: np.ndarray, opacity: float) -> dict[str, Any]:
        stage = self._usd_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {'success': False, 'error': 'prim_not_found'}
        try:
            from pxr import Gf, UsdGeom
            color = np.clip(np.array(color, dtype=float), 0.0, 1.0)
            opacity = float(np.clip(float(opacity), 0.0, 1.0))
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color.tolist())])
            gprim.CreateDisplayOpacityAttr().Set([opacity])
            return {'success': True, 'display_color': color.tolist(), 'display_opacity': opacity}
        except Exception as exc:
            return {'success': False, 'error': repr(exc)}

    def _liquid_visual_opacity(self) -> float:
        return float(np.clip(float(self.config.liquid_visual_opacity), 0.0, 1.0))

    def _set_shader_input(self, shader: Any, name: str, value_type: Any, value: Any) -> None:
        shader_input = shader.GetInput(name)
        try:
            valid = bool(shader_input) and bool(shader_input.GetAttr().IsValid())
        except Exception:
            valid = False
        if not valid:
            shader_input = shader.CreateInput(name, value_type)
        shader_input.Set(value)

    def _usd_stage(self) -> Any:
        try:
            import omni.usd
            return omni.usd.get_context().get_stage()
        except Exception:
            sim = getattr(self.og, 'sim', None)
            stage = getattr(sim, 'stage', None)
            if stage is not None:
                return stage
            raise

    def _usd_edit_context(self) -> Any:
        sim = getattr(self.og, 'sim', None)
        editing_usd = getattr(sim, 'editing_usd', None)
        if callable(editing_usd):
            return editing_usd()
        return nullcontext()

    def _safe_prim_token(self, value: Any) -> str:
        token = ''.join((ch if ch.isalnum() else '_' for ch in str(value)))
        return token.strip('_') or 'liquid'

    def _as_float_triplet(self, value: Any) -> tuple[float, float, float]:
        return self._as_float_tuple(value, expected_len=3)

    def _as_float_tuple(self, value: Any, *, expected_len: int) -> tuple[float, ...]:
        if callable(value):
            value = value()
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        if hasattr(value, 'tolist'):
            value = value.tolist()
        if len(value) != expected_len:
            raise ValueError(f'Expected length-{expected_len} value, got {value!r}')
        return tuple((float(item) for item in value))

    def _quat_xyzw_inverse_rotate(self, vector: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        quat_norm = float(np.linalg.norm(quat_xyzw))
        if quat_norm <= 1e-12:
            return vector
        x, y, z, w = (quat_xyzw / quat_norm).tolist()
        q_vec = np.array([-x, -y, -z], dtype=float)
        uv = np.cross(q_vec, vector)
        uuv = np.cross(q_vec, uv)
        return vector + 2.0 * (w * uv + uuv)

    def clear_substance_visuals(self, substance: str, *, names: set[str] | None=None, reason: str='scenario_reset', record: bool=True) -> dict[str, Any]:
        if str(self.config.container_liquid_visual_mode) == 'official_filled_particles':
            return self._clear_official_filled_particles(substance, names=names, reason=reason, record=record)
        requested_names = sorted(names or set(self.og_object_by_name))
        cleared: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        for name in requested_names:
            og_obj = self._resolve_og_object_by_name(name)
            if og_obj is None:
                continue
            try:
                prim_paths_to_clear = [self._container_liquid_prim_path(og_obj, substance)]
                mesh_paths = self._container_liquid_visual_mesh_paths(og_obj, substance)
                prim_paths_to_clear.extend([mesh_paths['body'], mesh_paths['surface']])
                for prim_path in prim_paths_to_clear:
                    before_visible = self._liquid_visual_visible(prim_path)
                    if before_visible is not None:
                        self._set_liquid_visual_visibility(prim_path, False)
                    after_visible = self._liquid_visual_visible(prim_path)
                    if before_visible is not None or after_visible is not None:
                        cleared.append({'object': name, 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible})
            except Exception as exc:
                errors[name] = repr(exc)
        success = not errors and all((item.get('after_visible') is not True for item in cleared))
        info = {'success': success, 'substance': str(substance), 'visual_mode': 'visual_mesh_water', 'target_count': len(requested_names), 'cleared_visuals': cleared}
        if errors:
            info['errors'] = errors
        if record:
            self._record_visual_mutation(str(substance), 'substance_visuals', False, success, reason, **self._visual_info_payload(info))
        return info

    def _clear_official_filled_particles(self, substance: str, *, names: set[str] | None=None, reason: str, record: bool) -> dict[str, Any]:
        requested_names = sorted(names or set(self.og_object_by_name))
        cleared_mesh_visuals: list[dict[str, Any]] = []
        mesh_errors: dict[str, str] = {}
        for name in requested_names:
            og_obj = self._resolve_og_object_by_name(name)
            if og_obj is None:
                continue
            try:
                prim_paths = [self._container_liquid_prim_path(og_obj, substance)]
                mesh_paths = self._container_liquid_visual_mesh_paths(og_obj, substance)
                prim_paths.extend([mesh_paths['body'], mesh_paths['surface']])
                for prim_path in prim_paths:
                    before_visible = self._liquid_visual_visible(prim_path)
                    if before_visible is not None:
                        self._set_liquid_visual_visibility(prim_path, False)
                    after_visible = self._liquid_visual_visible(prim_path)
                    if before_visible is not None or after_visible is not None:
                        cleared_mesh_visuals.append({'object': name, 'prim_path': prim_path, 'before_visible': before_visible, 'after_visible': after_visible})
            except Exception as exc:
                mesh_errors[name] = repr(exc)
        before_particles: int | None = None
        after_particles: int | None = None
        particle_error: str | None = None
        try:
            system = self._get_system(substance, force_init=False)
            if bool(getattr(system, 'initialized', False)):
                before_particles = self._system_particle_count(system)
                self._remove_all_particles(system)
                self._clear_container_state_caches()
                after_particles = self._system_particle_count(system)
            else:
                before_particles = 0
                after_particles = 0
        except Exception as exc:
            particle_error = repr(exc)
        success = particle_error is None and (not mesh_errors) and (after_particles == 0 if after_particles is not None else True) and all((item.get('after_visible') is not True for item in cleared_mesh_visuals))
        info: dict[str, Any] = {'success': success, 'substance': str(substance), 'visual_mode': 'official_filled_particles', 'target_count': len(requested_names), 'before_particles': before_particles, 'after_particles': after_particles, 'cleared_mesh_visuals': cleared_mesh_visuals}
        if particle_error is not None:
            info['error'] = particle_error
        if mesh_errors:
            info['mesh_errors'] = mesh_errors
        if record:
            self._record_visual_mutation(str(substance), 'substance_visuals', False, success, reason, **self._visual_info_payload(info))
        return info

    def _clear_container_state_caches(self) -> int:
        try:
            from omnigibson import object_states
        except Exception:
            return 0
        state_classes = (getattr(object_states, 'Contains', None), getattr(object_states, 'Filled', None), getattr(object_states, 'ContainedParticles', None))
        cleared = 0
        for og_obj in self.og_object_by_name.values():
            states = getattr(og_obj, 'states', None)
            if not states:
                continue
            for state_cls in state_classes:
                if state_cls is None:
                    continue
                state = states.get(state_cls)
                if state is None:
                    continue
                self._clear_state_cache(state)
                cleared += 1
        return cleared

    def _tint_fluid_particle_material(self, system: Any, substance: str) -> dict[str, Any]:
        material = getattr(system, 'material', None)
        info = {'success': False, 'substance': str(substance), 'system': str(getattr(system, 'name', substance)), 'material_path': str(getattr(material, 'prim_path', '')) if material is not None else None}
        if material is None:
            info['error'] = 'missing_particle_material'
            return info
        target_color = tuple((float(value) for value in self.config.liquid_particle_color))
        input_names = ('diffuse_reflection_weight', 'diffuse_reflection_color', 'enable_specular_transmission', 'specular_transmission_weight', 'specular_transmission_color', 'specular_transmission_scattering_color', 'emission_color', 'emission_intensity')
        info['target_color'] = list(target_color)
        info['before_inputs'] = self._material_input_snapshot(material, input_names)
        rgb = [float(value) for value in target_color]
        updates: tuple[tuple[str, Any], ...] = (('diffuse_reflection_weight', 1.0), ('diffuse_reflection_color', rgb), ('enable_specular_transmission', False), ('specular_transmission_weight', 0.0), ('specular_transmission_color', rgb), ('specular_transmission_scattering_color', rgb), ('emission_color', rgb), ('emission_intensity', 0.0))
        applied: list[str] = []
        errors: dict[str, str] = {}
        for input_name, value in updates:
            try:
                setattr(material, input_name, value)
                applied.append(input_name)
            except Exception as exc:
                errors[input_name] = repr(exc)
        try:
            import torch as th
            system._color = th.tensor(rgb, dtype=th.float32)
        except Exception as exc:
            errors['system._color'] = repr(exc)
        bound_paths: list[str] = []
        for bind_path in [getattr(system, 'system_prim_path', None)] + [getattr(prototype, 'prim_path', None) for prototype in getattr(system, 'particle_prototypes', [])]:
            if bind_path is None:
                continue
            try:
                material.bind(bind_path)
                bound_paths.append(str(bind_path))
            except Exception as exc:
                errors[f'bind:{bind_path}'] = repr(exc)
        info['applied_inputs'] = applied
        info['bound_paths'] = bound_paths
        info['after_inputs'] = self._material_input_snapshot(material, input_names)
        if errors:
            info['errors'] = errors
        info['success'] = bool(applied)
        return info

    def _material_input_snapshot(self, material: Any, input_names: tuple[str, ...]) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for input_name in input_names:
            try:
                snapshot[input_name] = self._serializable_material_value(material.get_input(inp=input_name))
            except Exception:
                continue
        return snapshot

    def _serializable_material_value(self, value: Any) -> Any:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        if hasattr(value, 'tolist'):
            value = value.tolist()
        if isinstance(value, (str, bool, int, float)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(key): self._serializable_material_value(item) for key, item in value.items()}
        try:
            return [self._serializable_material_value(item) for item in value]
        except TypeError:
            return str(value)

    def _visual_info_payload(self, info: Mapping[str, Any], *, exclude: set[str] | None=None) -> dict[str, Any]:
        excluded = {'success', 'visual_material'} | set(exclude or ())
        return {key: self._serializable_material_value(value) for key, value in info.items() if key not in excluded}

    def _substances_from_container_state(self, value: Any) -> set[str]:
        if isinstance(value, Mapping):
            return {str(substance) for substance, present in value.items() if bool(present)}
        if isinstance(value, (set, list, tuple)):
            return {str(substance) for substance in value}
        return set()

    def _substances_from_container_delta(self, value: Any) -> set[str]:
        if not isinstance(value, Mapping):
            return set()
        return self._substances_from_container_state(value.get('contains')) | self._substances_from_container_state(value.get('filled'))

    def _get_system(self, system_name: str, *, force_init: bool=True) -> Any:
        try:
            return self.env.scene.get_system(system_name, force_init=force_init)
        except TypeError:
            return self.env.scene.get_system(system_name)

    def _apply_texture_change(self, og_obj: Any, object_state: Any | None) -> bool:
        update_texture_change = getattr(og_obj, '_update_texture_change', None)
        if update_texture_change is None:
            return False
        try:
            update_texture_change(object_state)
            if hasattr(og_obj, '_current_texture_state'):
                og_obj._current_texture_state = object_state
            return True
        except Exception:
            return False

    def _clear_state_cache(self, state: Any) -> None:
        clear_cache = getattr(state, 'clear_cache', None)
        if clear_cache is None:
            return
        try:
            clear_cache()
        except Exception:
            pass

    def _add_sliced_part_objects(self, added_obj_attrs: Any) -> dict[str, Any]:
        report: dict[str, Any] = {'added_objects': [], 'registered_added_objects': [], 'step_errors': [], 'callback_errors': [], 'cleanup_errors': [], '_deferred_inits': []}
        for index, added_obj_attr in enumerate(added_obj_attrs or []):
            new_obj = None
            name = ''
            try:
                new_obj = added_obj_attr.obj
                name = str(getattr(new_obj, 'name', ''))
                if not name:
                    raise ValueError('sliced_part_missing_name')
                self.env.scene.add_object(new_obj)
                report['added_objects'].append(name)
                pos = getattr(added_obj_attr, 'pos', None)
                orn = getattr(added_obj_attr, 'orn', None)
                bb_pos = getattr(added_obj_attr, 'bb_pos', None)
                bb_orn = getattr(added_obj_attr, 'bb_orn', None)
                if pos is not None or orn is not None:
                    current_pos, current_orn = self._current_sliced_part_orientation(new_obj)
                    new_obj.set_position_orientation(position=self._coerce_sliced_pose_value(current_pos if pos is None else pos), orientation=self._coerce_sliced_pose_value(current_orn if orn is None else orn))
                elif bb_pos is not None or bb_orn is not None:
                    self._set_sliced_part_bbox_pose(new_obj, position=bb_pos, orientation=bb_orn)
                self.og_object_by_name[name] = new_obj
                if self.on_object_added is not None:
                    self.on_object_added(name, new_obj)
                report['registered_added_objects'].append(name)
                states = getattr(added_obj_attr, 'states', None)
                callback = getattr(added_obj_attr, 'callback', None)
                if states is not None or callback is not None:
                    report['_deferred_inits'].append({'object': new_obj, 'name': name, 'states': states, 'callback': callback})
            except Exception as exc:
                report['step_errors'].append({'stage': 'add_sliced_part', 'index': index, 'object': name or str(getattr(new_obj, 'name', '')), 'error': repr(exc), 'traceback': self._exception_traceback(exc)})
                report['cleanup_errors'].extend(self._cleanup_sliced_part_objects(report['added_objects']))
                return report
        for name in list(report['registered_added_objects']):
            if not self._scene_has_object(name):
                report['step_errors'].append({'stage': 'verify_sliced_part_registered', 'object': name, 'error': 'sliced_part_not_registered'})
        if report['step_errors']:
            report['cleanup_errors'].extend(self._cleanup_sliced_part_objects(report['added_objects']))
        return report

    def _run_deferred_sliced_part_inits(self, deferred_inits: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        for item in deferred_inits:
            new_obj = item.get('object')
            name = str(item.get('name') or getattr(new_obj, 'name', ''))
            states = item.get('states')
            callback = item.get('callback')
            if states is not None:
                try:
                    for state, args in dict(states).items():
                        new_obj.states[state].set_value(*args)
                except Exception as exc:
                    errors.append({'stage': 'sliced_part_state_init', 'object': name, 'error': repr(exc), 'traceback': self._exception_traceback(exc)})
            if callback is not None:
                try:
                    callback(new_obj)
                except Exception as exc:
                    errors.append({'stage': 'sliced_part_callback', 'object': name, 'error': repr(exc), 'traceback': self._exception_traceback(exc)})
        return errors

    def _cleanup_sliced_part_objects(self, names: list[str]) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        for name in list(dict.fromkeys((str(item) for item in names if item))):
            obj = self.og_object_by_name.pop(name, None)
            if obj is None:
                try:
                    obj = self.env.scene.object_registry('name', name, default_val=None)
                except TypeError:
                    try:
                        obj = self.env.scene.object_registry('name', name)
                    except Exception:
                        obj = None
                except Exception:
                    obj = None
            try:
                if obj is not None:
                    setattr(obj, 'visible', False)
            except Exception:
                pass
            remove_error = None
            for method_name in ('remove_object', 'remove_obj', 'delete_object'):
                method = getattr(self.env.scene, method_name, None)
                if not callable(method):
                    continue
                try:
                    method(obj if obj is not None else name)
                    remove_error = None
                    break
                except TypeError:
                    try:
                        method(name)
                        remove_error = None
                        break
                    except Exception as exc:
                        remove_error = exc
                except Exception as exc:
                    remove_error = exc
            if remove_error is not None:
                errors.append({'stage': 'cleanup_sliced_part', 'object': name, 'error': repr(remove_error)})
        return errors

    def _cut_transition_object_with_parts_metadata(self, og_obj: Any) -> Any:
        metadata = dict(getattr(og_obj, 'metadata', {}) or {})
        object_parts = self._normalize_cut_object_parts(metadata.get('object_parts'))
        if object_parts is None:
            category = str(getattr(og_obj, 'category', '') or '')
            model = str(getattr(og_obj, 'model_name', '') or getattr(og_obj, 'model', '') or '')
            object_parts = self._cut_object_parts_from_assets(category, model)
        if object_parts is None:
            return og_obj
        patched = dict(metadata)
        patched['object_parts'] = object_parts
        return _CutMetadataObjectProxy(og_obj, patched)

    def _normalize_cut_object_parts(self, object_parts: Any) -> dict[str, Any] | None:
        if object_parts is None:
            return None
        if isinstance(object_parts, Mapping):
            items = list(object_parts.items())
        elif isinstance(object_parts, (str, bytes)):
            return None
        else:
            try:
                items = [(str(index), part) for index, part in enumerate(object_parts)]
            except TypeError:
                return None
        normalized: dict[str, Any] = {}
        for key, part in items:
            if not isinstance(part, Mapping) or not self._valid_cut_object_part(part):
                continue
            normalized[str(key)] = {str(part_key): self._plain_sliced_pose_value(part_value) for part_key, part_value in dict(part).items()}
        return normalized or None

    def _valid_cut_object_part(self, part: Mapping[str, Any]) -> bool:
        for key in ('category', 'model', 'bb_pos', 'bb_orn', 'bb_size'):
            if key not in part:
                return False
        try:
            return len(self._plain_sliced_pose_value(part['bb_pos'])) == 3 and len(self._plain_sliced_pose_value(part['bb_orn'])) == 4 and (len(self._plain_sliced_pose_value(part['bb_size'])) == 3)
        except Exception:
            return False

    def _cut_object_parts_from_assets(self, category: str, model: str='') -> dict[str, Any] | None:
        key = (str(category), str(model or ''))
        if key not in self._cut_object_parts_by_asset:
            self._cut_object_parts_by_asset[key] = self._load_cut_object_parts_from_assets(category, model)
        return self._cut_object_parts_by_asset[key]

    def _load_cut_object_parts_from_assets(self, category: str, model: str='') -> dict[str, Any] | None:
        for path in self._cut_metadata_asset_paths(category, model):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                continue
            object_parts = self._normalize_cut_object_parts(data.get('object_parts'))
            if object_parts is not None:
                return object_parts
        return None

    def _cut_metadata_asset_paths(self, category: str, model: str='') -> list[Path]:
        category = str(category or '').strip()
        model = str(model or '').strip()
        if not category:
            return []
        paths: list[Path] = []

        def add_category_dir(category_dir: Path) -> None:
            if model:
                paths.append(category_dir / model / 'misc' / 'metadata.json')
            paths.append(category_dir / 'misc' / 'metadata.json')
            try:
                paths.extend(sorted(category_dir.glob('*/misc/metadata.json')))
            except Exception:
                pass
        try:
            from omnigibson.utils.asset_utils import get_category_path
            add_category_dir(Path(get_category_path(category)))
        except Exception:
            pass
        repo_root = behavior_assets_root().parent.parent
        add_category_dir(repo_root / 'datasets' / 'behavior-1k-assets' / 'objects' / category)
        deduped: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                deduped.append(path)
        return deduped

    def _set_sliced_part_bbox_pose(self, obj: Any, *, position: Any, orientation: Any) -> None:
        current_position, current_orientation = self._current_sliced_part_orientation(obj)
        target_position = np.asarray(self._plain_sliced_pose_value(position) if position is not None else current_position, dtype=float)
        target_orientation = np.asarray(self._plain_sliced_pose_value(orientation) if orientation is not None else current_orientation, dtype=float)
        bbox_center_in_base = self._sliced_part_bbox_center_in_base_frame(obj)
        root_position = target_position - self._rotate_sliced_vector_by_quat(bbox_center_in_base, target_orientation)
        obj.set_position_orientation(position=self._coerce_sliced_pose_value(root_position), orientation=self._coerce_sliced_pose_value(target_orientation))

    def _current_sliced_part_orientation(self, obj: Any) -> tuple[list[float], list[float]]:
        try:
            position, orientation = obj.get_position_orientation(frame='scene')
        except TypeError:
            position, orientation = obj.get_position_orientation()
        except Exception:
            position, orientation = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        return (list(self._plain_sliced_pose_value(position)), list(self._plain_sliced_pose_value(orientation)))

    def _sliced_part_bbox_center_in_base_frame(self, obj: Any) -> np.ndarray:
        try:
            base_position, base_orientation = self._current_sliced_part_orientation(obj)
            bbox_center = getattr(obj, 'aabb_center', None)
            if bbox_center is None and hasattr(obj, 'aabb'):
                aabb = np.asarray(obj.aabb, dtype=float)
                bbox_center = (aabb[0] + aabb[1]) * 0.5
            if bbox_center is None:
                return np.zeros(3, dtype=float)
            offset_world = np.asarray(self._plain_sliced_pose_value(bbox_center), dtype=float) - np.asarray(base_position, dtype=float)
            return self._quat_xyzw_inverse_rotate(offset_world, np.asarray(base_orientation, dtype=float))
        except Exception:
            return np.zeros(3, dtype=float)

    def _rotate_sliced_vector_by_quat(self, vector: Any, quat_xyzw: Any) -> np.ndarray:
        vector_np = np.asarray(self._plain_sliced_pose_value(vector), dtype=float)
        quat_np = np.asarray(self._plain_sliced_pose_value(quat_xyzw), dtype=float)
        quat_norm = float(np.linalg.norm(quat_np))
        if quat_norm <= 1e-12:
            return vector_np
        x, y, z, w = (quat_np / quat_norm).tolist()
        q_vec = np.array([x, y, z], dtype=float)
        uv = np.cross(q_vec, vector_np)
        uuv = np.cross(q_vec, uv)
        return vector_np + 2.0 * (w * uv + uuv)

    def _coerce_sliced_pose_value(self, value: Any) -> Any:
        plain = self._plain_sliced_pose_value(value)
        try:
            import torch as th
            return th.tensor(plain, dtype=th.float32)
        except Exception:
            return plain

    def _plain_sliced_pose_value(self, value: Any) -> Any:
        if callable(value):
            value = value()
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        if hasattr(value, 'tolist'):
            value = value.tolist()
        while isinstance(value, (list, tuple)) and len(value) == 1 and (not isinstance(value[0], (str, bytes))) and hasattr(value[0], '__iter__'):
            value = value[0]
            if hasattr(value, 'tolist'):
                value = value.tolist()
        if isinstance(value, Mapping):
            return {str(key): self._plain_sliced_pose_value(item) for key, item in value.items()}
        if isinstance(value, (str, bytes)) or value is None:
            return value
        try:
            return [self._plain_sliced_pose_value(item) for item in value]
        except TypeError:
            try:
                return float(value)
            except Exception:
                return value

    def _scene_has_object(self, name: str) -> bool:
        try:
            return self.env.scene.object_registry('name', name) is not None
        except Exception:
            return any((getattr(obj, 'name', '') == name for obj in getattr(self.env.scene, 'objects', [])))

    def _object_visible(self, og_obj: Any) -> bool | None:
        try:
            return bool(og_obj.visible)
        except Exception:
            return None

    def _exception_traceback(self, exc: BaseException) -> str:
        return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=6))

    def _refresh_object_visuals(self, name: str) -> None:
        og_obj = self._resolve_og_object_by_name(name)
        if og_obj is None:
            return
        for state in getattr(og_obj, 'states', {}).values():
            self._clear_state_cache(state)
        update_visuals = getattr(og_obj, 'update_visuals', None)
        if update_visuals is None:
            return
        try:
            update_visuals()
        except Exception:
            pass

    def _has_fire_emitter(self, og_obj: Any) -> bool:
        try:
            from omnigibson.utils.constants import EmitterType
        except Exception:
            return False
        return EmitterType.FIRE in getattr(og_obj, '_emitters', {})

    def _advance_fire_visuals(self, og_obj: Any, enabled: bool) -> int:
        diagnostics: dict[str, Any] = {'enabled': bool(enabled)}
        self._last_fire_diagnostics = diagnostics
        if enabled:
            diagnostics['emitter_set'] = self._set_fire_emitter_enabled(og_obj, True)
            if self.config.deepen_fire_color:
                diagnostics['colormap'] = self._apply_fire_colormap(og_obj)
            steps = self._advance_sim_steps(self.config.fire_visual_steps)
            diagnostics['sim_steps'] = steps
            return steps
        diagnostics['emitter_set'] = self._set_fire_emitter_enabled(og_obj, False)
        if self.config.fire_force_clear_on_shutdown:
            steps, clear = self._clear_fire_grid_over_steps(og_obj, self.config.fire_shutdown_steps, self.config.fire_shutdown_render_steps)
            diagnostics['force_clear'] = clear
            diagnostics['sim_steps'] = steps
            return steps
        steps = self._advance_sim_steps(self.config.fire_shutdown_steps)
        render_steps = self._advance_sim_renders(self.config.fire_shutdown_render_steps)
        diagnostics['sim_steps'] = steps
        diagnostics['render_steps'] = render_steps
        return steps

    def _set_fire_emitter_enabled(self, og_obj: Any, enabled: bool) -> dict[str, Any]:
        result: dict[str, Any] = {'requested': bool(enabled), 'success': False}
        try:
            from omnigibson.utils.constants import EmitterType
        except Exception as exc:
            result['error'] = repr(exc)
            return result
        set_emitter_enabled = getattr(og_obj, 'set_emitter_enabled', None)
        if set_emitter_enabled is None:
            result['error'] = 'object_has_no_set_emitter_enabled'
            return result
        try:
            set_emitter_enabled(EmitterType.FIRE, bool(enabled))
            result['success'] = True
        except Exception as exc:
            result['error'] = repr(exc)
            return result
        info = self._fire_emitter_info(og_obj)
        if info is not None:
            try:
                result['enabled_attr'] = bool(info['emitter'].GetAttribute('enabled').Get())
            except Exception:
                pass
        return result

    def _fire_emitter_info(self, og_obj: Any) -> dict[str, Any] | None:
        try:
            from omnigibson.utils.constants import EmitterType
        except Exception:
            return None
        emitters = getattr(og_obj, '_emitters', {})
        return emitters.get(EmitterType.FIRE)

    def _fire_mesh_prim_path(self, og_obj: Any) -> str | None:
        info = self._fire_emitter_info(og_obj)
        if not info:
            return None
        mesh = info.get('mesh')
        prim_path = getattr(mesh, 'prim_path', None)
        return str(prim_path) if prim_path else None

    def _apply_fire_colormap(self, og_obj: Any) -> dict[str, Any]:
        result: dict[str, Any] = {'success': False}
        mesh_path = self._fire_mesh_prim_path(og_obj)
        if mesh_path is None:
            result['error'] = 'missing_fire_mesh'
            return result
        try:
            from pxr import Gf, Sdf
        except Exception as exc:
            result['error'] = repr(exc)
            return result
        colormap_path = f'{mesh_path}/flowOffscreen/colormap'
        try:
            stage = self.og.sim.stage
            prim = stage.GetPrimAtPath(colormap_path)
            if not prim or not prim.IsValid():
                result['error'] = 'missing_colormap_prim'
                return result
            rgba_points = [Gf.Vec4f(*point) for point in DEEP_FIRE_COLORMAP]
            with self._usd_edit_context():
                attr = prim.GetAttribute('rgbaPoints')
                if not attr or not attr.IsValid():
                    attr = prim.CreateAttribute('rgbaPoints', Sdf.ValueTypeNames.Float4Array, False)
                attr.Set(rgba_points)
            result.update({'success': True, 'colormap_path': colormap_path, 'points': len(rgba_points)})
        except Exception as exc:
            result['error'] = repr(exc)
        return result

    def _force_clear_fire_grid(self, og_obj: Any) -> dict[str, Any]:
        result: dict[str, Any] = {'success': False}
        mesh_path = self._fire_mesh_prim_path(og_obj)
        if mesh_path is None:
            result['error'] = 'missing_fire_mesh'
            return result
        try:
            from pxr import Sdf
        except Exception as exc:
            result['error'] = repr(exc)
            return result
        simulate_path = f'{mesh_path}/flowSimulate'
        try:
            stage = self.og.sim.stage
            prim = stage.GetPrimAtPath(simulate_path)
            if not prim or not prim.IsValid():
                result['error'] = 'missing_flow_simulate_prim'
                return result

            def _set_bool(name: str, value: bool) -> None:
                with self._usd_edit_context():
                    attr = prim.GetAttribute(name)
                    if not attr or not attr.IsValid():
                        attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool, False)
                    attr.Set(value)
            _set_bool('forceClear', True)
            _set_bool('forceDisableEmitters', True)
            self._advance_sim_steps(1)
            _set_bool('forceClear', False)
            _set_bool('forceDisableEmitters', False)
            result.update({'success': True, 'simulate_path': simulate_path})
        except Exception as exc:
            result['error'] = repr(exc)
        return result

    def _clear_fire_grid_over_steps(self, og_obj: Any, sim_steps: int, render_steps: int) -> tuple[int, dict[str, Any]]:
        result: dict[str, Any] = {'success': False, 'held': True}
        mesh_path = self._fire_mesh_prim_path(og_obj)
        if mesh_path is None:
            result['error'] = 'missing_fire_mesh'
            return (0, result)
        try:
            from pxr import Sdf
        except Exception as exc:
            result['error'] = repr(exc)
            return (0, result)
        simulate_path = f'{mesh_path}/flowSimulate'
        try:
            stage = self.og.sim.stage
            prim = stage.GetPrimAtPath(simulate_path)
            if not prim or not prim.IsValid():
                result['error'] = 'missing_flow_simulate_prim'
                return (0, result)

            def _set_bool(name: str, value: bool) -> None:
                with self._usd_edit_context():
                    attr = prim.GetAttribute(name)
                    if not attr or not attr.IsValid():
                        attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool, False)
                    attr.Set(value)
            _set_bool('forceClear', True)
            _set_bool('forceDisableEmitters', True)
            steps = max(1, int(sim_steps))
            renders_per_step = max(0, int(render_steps))
            completed = 0
            for _ in range(steps):
                if self.before_fire_step is not None:
                    self.before_fire_step()
                try:
                    self.og.sim.step()
                except Exception:
                    break
                completed += 1
                for _ in range(renders_per_step):
                    try:
                        self.og.sim.render()
                    except Exception:
                        break
            _set_bool('forceClear', False)
            _set_bool('forceDisableEmitters', False)
            result.update({'success': True, 'simulate_path': simulate_path, 'held_steps': completed})
            return (completed, result)
        except Exception as exc:
            result['error'] = repr(exc)
            return (0, result)

    def _usd_edit_context(self):
        editing_usd = getattr(self.og.sim, 'editing_usd', None)
        if callable(editing_usd):
            try:
                return editing_usd()
            except Exception:
                return nullcontext()
        return nullcontext()

    def _has_particle_source(self, og_obj: Any, object_states: Any | None=None) -> bool:
        if object_states is None:
            try:
                from omnigibson import object_states
            except Exception:
                return False
        particle_source_state = getattr(object_states, 'ParticleSource', None)
        return bool(particle_source_state is not None and particle_source_state in getattr(og_obj, 'states', {}))

    def _advance_particle_source_visuals(self, og_obj: Any, enabled: bool) -> dict[str, Any]:
        system_name = self.config.water_source_system
        info: dict[str, Any] = {'success': False, 'system': system_name, 'enabled': bool(enabled)}
        try:
            from omnigibson import object_states
        except Exception as exc:
            info['error'] = repr(exc)
            return info
        particle_source_state = getattr(object_states, 'ParticleSource', None)
        if particle_source_state is None or particle_source_state not in getattr(og_obj, 'states', {}):
            info['error'] = 'missing_particle_source_state'
            return info
        source_state = og_obj.states[particle_source_state]
        if not self._particle_source_supports_system(source_state, system_name):
            info['error'] = 'particle_source_does_not_support_system'
            return info
        try:
            system = self._get_system(system_name, force_init=True)
        except Exception as exc:
            info['error'] = repr(exc)
            return info
        info['particle_material'] = self._tint_fluid_particle_material(system, system_name)
        before_particles = self._system_particle_count(system)
        info['before_particles'] = before_particles
        if enabled:
            try:
                source_period = int(getattr(source_state, 'n_steps_per_modification', 0) or 0)
                requested_steps = max(self.config.water_visual_steps, source_period)
                steps = self._advance_sim_steps(requested_steps)
                after_particles = self._system_particle_count(system)
                particle_success = after_particles is not None and before_particles is not None and (after_particles > before_particles or after_particles > 0)
                stream_visual = self._set_particle_source_stream_visual(og_obj, source_state, system_name, visible=particle_success)
                info.update({'after_particles': after_particles, 'water_visual_steps': steps, 'source_period_steps': source_period, 'stream_visual_success': bool(stream_visual.get('success', True)), 'stream_visual': self._visual_info_payload(stream_visual), 'success': particle_success and bool(stream_visual.get('success', True))})
            except Exception as exc:
                info['error'] = repr(exc)
                info['success'] = False
        else:
            try:
                stream_visual = self._set_particle_source_stream_visual(og_obj, source_state, system_name, visible=False)
                cleared_particles = self._remove_all_particles(system)
                after_clear_particles = self._system_particle_count(system)
                steps = self._advance_sim_steps(self.config.water_shutdown_steps)
                render_steps = self._advance_sim_renders(self.config.water_shutdown_render_steps)
                after_particles = self._system_particle_count(system)
                info.update({'cleared_particles': cleared_particles, 'after_clear_particles': after_clear_particles, 'after_particles': after_particles, 'water_visual_steps': steps, 'water_shutdown_render_steps': render_steps, 'stream_visual_success': bool(stream_visual.get('success', True)), 'stream_visual': self._visual_info_payload(stream_visual), 'success': after_particles == 0 and bool(stream_visual.get('success', True))})
            except Exception as exc:
                info['error'] = repr(exc)
                info['success'] = False
        return info

    def _particle_source_supports_system(self, source_state: Any, system_name: str) -> bool:
        supports_system = getattr(source_state, 'supports_system', None)
        if supports_system is not None:
            try:
                return bool(supports_system(system_name))
            except Exception:
                pass
        conditions = getattr(source_state, 'conditions', None)
        if isinstance(conditions, Mapping):
            return system_name in conditions
        return True

    def _system_particle_count(self, system: Any) -> int | None:
        try:
            return int(getattr(system, 'n_particles'))
        except Exception:
            return None

    def _remove_all_particles(self, system: Any) -> int | None:
        before_particles = self._system_particle_count(system)
        remove_all_particles = getattr(system, 'remove_all_particles', None)
        if remove_all_particles is None:
            return before_particles
        try:
            remove_all_particles()
        except Exception:
            return before_particles
        return before_particles

    def _particle_source_payload(self, info: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in info.items():
            if key == 'success':
                payload['particle_source_success'] = bool(value)
            else:
                payload[f'particle_source_{key}'] = value
        return payload

    def _advance_sim_steps(self, steps: int) -> int:
        completed = 0
        for _ in range(max(0, steps)):
            if self.before_fire_step is not None:
                self.before_fire_step()
            try:
                self.og.sim.step()
            except Exception:
                break
            completed += 1
        return completed

    def _advance_sim_renders(self, steps: int) -> int:
        completed = 0
        for _ in range(max(0, steps)):
            try:
                self.og.sim.render()
            except Exception:
                break
            completed += 1
        return completed

    def _advance_visual_state_renders(self) -> int:
        completed = 0
        for _ in range(max(0, self.config.visual_state_sync_steps)):
            try:
                self.og.sim.render()
            except Exception:
                break
            completed += 1
        return completed

    def _advance_open_visual_settle_renders(self) -> int:
        steps = max(int(getattr(self.config, 'open_visual_settle_renders', 8)), int(getattr(self.config, 'visual_state_sync_steps', 0)))
        return self._advance_sim_renders(steps)

    def _record_visual_mutation(self, name: str, state: str, requested: Any, success: Any, reason: str, **payload: Any) -> None:
        success_value = None if success is None else bool(success)
        self.visual_mutations.append({'object': name, 'state': state, 'requested': requested, 'success': success_value, 'reason': reason, **payload})

    def _require_visual_state_sync(self, name: str, state: str) -> None:
        if self.config.require_visual_state_sync:
            raise AssertionError(f'Visual state sync failed for {name}.{state}')
