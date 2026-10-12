from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional
from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.execution_core.navigation import NavigationTarget, plan_navigation
from scoper.benchmark.behavior.goal_checker import _is_inside, _is_ontop
from scoper.benchmark.behavior.scheduler.base import Scheduler, SchedulerEvent
KNOWN_DISTURBANCE_TYPES = frozenset({'close_object', 'open_object', 'toggle_on', 'toggle_off', 'dirty_object', 'relocate_object', 'restore_initial_state'})
_MAX_RELOCATION_CENTER_HEIGHT_ABOVE_FLOOR_M = 2.2

@dataclass
class DisturbanceConfig:
    type: str
    target_role: str | None = None
    target_object_id: str | None = None
    after_step: int = 0
    once: bool = True
    activation: dict | None = None
    requires_out_of_view: bool = True
    before_step: int = 10 ** 9
    probability: float = 1.0
    cooldown: int = 0
    max_count: int = 10 ** 9
    notify_brain: bool = False
    notify_message: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in KNOWN_DISTURBANCE_TYPES:
            raise ValueError("Unknown disturbance type '{}'.".format(self.type))
        if self.target_role is None and self.target_object_id is None:
            raise ValueError('DisturbanceConfig requires target_role or target_object_id.')
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError('DisturbanceConfig.probability must be in [0, 1].')

class DisturbanceScheduler(Scheduler):
    name = 'disturbance'

    def __init__(self, configs: list, seed: int=0) -> None:
        self._configs = [DisturbanceConfig(**_disturbance_config_kwargs(cfg)) if isinstance(cfg, Mapping) else cfg for cfg in configs]
        self._fired: set = set()
        self._initial_snapshot: dict = {}
        self._activation_state: dict[int, dict] = {}
        self._rng = random.Random(seed)
        self._fire_counts: dict[int, int] = {i: 0 for i in range(len(self._configs))}
        self._last_fire_step: dict[int, int] = {}

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> Optional[List[SchedulerEvent]]:
        self._initial_snapshot = {}
        self._fired = set()
        self._activation_state = {i: {} for i in range(len(self._configs))}
        self._fire_counts = {i: 0 for i in range(len(self._configs))}
        self._last_fire_step = {}
        setattr(env, '_scheduler_notices', [])
        for role_name, obj_name in env.role_map.items():
            obj = env.core.objects.get(obj_name)
            if obj is not None:
                self._initial_snapshot[role_name] = _snapshot(obj)
                self._initial_snapshot[obj_name] = _snapshot(obj)
        return [SchedulerEvent('disturbance_armed', 'episode_start', {'num_configs': len(self._configs)})]

    def pre_step(self, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]) -> Optional[List[SchedulerEvent]]:
        pending_qa = bool((getattr(env, '_human_qa_state', {}) or {}).get('pending'))
        events: List[SchedulerEvent] = []
        for i, cfg in enumerate(self._configs):
            if not self._eligible(i, cfg, high_level_step):
                continue
            if not self._activation_ready(i, cfg, env, high_level_step, last_transition):
                continue
            if pending_qa:
                continue
            if float(cfg.probability) < 1.0 and self._rng.random() > float(cfg.probability):
                continue
            event = self._fire(env, i, cfg, phase='pre_step', high_level_step=high_level_step)
            if event is not None:
                events.append(event)
                self._mark_fired(env, i, cfg, high_level_step, event)
        return events or None

    def post_step(self, env: Any, high_level_step: int, transition: Mapping[str, Any]) -> Optional[List[SchedulerEvent]]:
        if transition.get('tool_name') == 'answer':
            return None
        return None

    def _eligible(self, index: int, cfg: DisturbanceConfig, high_level_step: int) -> bool:
        count = int(self._fire_counts.get(index, 0))
        if cfg.once and count > 0:
            return False
        if count >= int(cfg.max_count):
            return False
        if high_level_step < int(cfg.after_step):
            return False
        if high_level_step >= int(cfg.before_step):
            return False
        last = self._last_fire_step.get(index)
        if last is not None and int(cfg.cooldown) > 0 and (high_level_step - int(last) < int(cfg.cooldown)):
            return False
        return True

    def _activation_ready(self, index: int, cfg: DisturbanceConfig, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]) -> bool:
        activation = cfg.activation if isinstance(cfg.activation, Mapping) else None
        if not _subtask_gate_matches(env, activation):
            return False
        if not activation:
            return True
        activation_type = str(activation.get('type') or 'high_level_step')
        if activation_type == 'high_level_step':
            return True
        if activation_type != 'after_tool_result':
            return False
        state = self._activation_state.setdefault(index, {})
        if not state.get('armed') and self._transition_matches_activation(activation, last_transition, env):
            armed_step = _transition_step(last_transition, high_level_step - 1)
            state['armed'] = True
            state['armed_step'] = armed_step
            state['distinct_tool_count'] = 0
            state['last_counted_step'] = armed_step
            state['armed_by'] = _transition_brief(last_transition)
        if not state.get('armed'):
            return False
        self._maybe_count_distinct_work(index, cfg, activation, last_transition, env)
        armed_step = int(state.get('armed_step', high_level_step))
        min_delay_steps = _nonnegative_int(activation.get('min_delay_steps'), 0)
        required_distinct = _nonnegative_int(activation.get('after_distinct_tool_count'), _nonnegative_int(activation.get('distinct_tool_count'), 0))
        if high_level_step < armed_step + min_delay_steps:
            return False
        if int(state.get('distinct_tool_count', 0)) < required_distinct:
            return False
        if activation.get('requires_target_not_held') and self._target_held(env, cfg):
            return False
        required_parent_role = activation.get('required_parent_role')
        if isinstance(required_parent_role, str) and required_parent_role:
            if not self._target_has_parent_role(env, cfg, required_parent_role):
                return False
        return True

    def _mark_fired(self, env: Any, index: int, cfg: DisturbanceConfig, high_level_step: int, event: SchedulerEvent) -> None:
        if event.name == 'disturbance_out_of_view_wait':
            return
        self._fire_counts[index] = int(self._fire_counts.get(index, 0)) + 1
        self._last_fire_step[index] = int(high_level_step)
        if cfg.once:
            self._fired.add(index)
        if cfg.notify_brain:
            notices = list(getattr(env, '_scheduler_notices', []) or [])
            notices.append(_disturbance_notice(cfg, event))
            setattr(env, '_scheduler_notices', notices)

    def _transition_matches_activation(self, activation: Mapping[str, Any], transition: Optional[Mapping[str, Any]], env: Any) -> bool:
        if not isinstance(transition, Mapping):
            return False
        expected_tools = activation.get('tools')
        if expected_tools is None and activation.get('tool') is not None:
            expected_tools = [activation.get('tool')]
        if expected_tools is not None:
            if str(transition.get('tool_name') or transition.get('tool')) not in {str(tool) for tool in expected_tools}:
                return False
        if 'success' in activation and bool(transition.get('success')) is not bool(activation.get('success')):
            return False
        arguments = transition.get('arguments') if isinstance(transition.get('arguments'), Mapping) else {}
        target = arguments.get('target')
        target_roles = activation.get('target_roles')
        if target_roles is None and activation.get('target_role') is not None:
            target_roles = [activation.get('target_role')]
        from scoper.benchmark.common.transition_roles import target_roles as resolved_roles
        if target_roles is not None and (not resolved_roles(env, transition).intersection((str(role) for role in target_roles))):
            return False
        state_delta_object_role = activation.get('state_delta_object_role')
        if isinstance(state_delta_object_role, str) and state_delta_object_role:
            state_delta = transition.get('state_delta')
            object_pose = state_delta.get('object_pose') if isinstance(state_delta, Mapping) else None
            placed_object = object_pose.get('target') if isinstance(object_pose, Mapping) else None
            expected_object = env.role_map.get(state_delta_object_role, state_delta_object_role)
            if expected_object is None or str(placed_object) != expected_object:
                return False
        return True

    def _maybe_count_distinct_work(self, index: int, cfg: DisturbanceConfig, activation: Mapping[str, Any], transition: Optional[Mapping[str, Any]], env: Any=None) -> None:
        if not isinstance(transition, Mapping):
            return
        state = self._activation_state.setdefault(index, {})
        step = _transition_step(transition, -1)
        if step <= int(state.get('armed_step', -1)):
            return
        if step == state.get('last_counted_step'):
            return
        tool = str(transition.get('tool_name') or transition.get('tool') or '')
        ignore_tools = {str(tool_name) for tool_name in activation.get('ignore_tools', ['stop', 'answer']) or []}
        if tool in ignore_tools:
            state['last_counted_step'] = step
            return
        if activation.get('distinct_success_only', True) and (not bool(transition.get('success', False))):
            state['last_counted_step'] = step
            return
        arguments = transition.get('arguments') if isinstance(transition.get('arguments'), Mapping) else {}
        target = arguments.get('target')
        exclude_targets = {str(role) for role in activation.get('exclude_target_roles', []) or []}
        if cfg.target_role:
            exclude_targets.add(str(cfg.target_role))
        from scoper.benchmark.common.transition_roles import target_roles as resolved_roles
        if resolved_roles(env, transition).intersection(exclude_targets):
            state['last_counted_step'] = step
            return
        state['distinct_tool_count'] = int(state.get('distinct_tool_count', 0)) + 1
        state['last_counted_step'] = step

    def _target_held(self, env: Any, cfg: DisturbanceConfig) -> bool:
        _, obj_name = _resolve_target(env, cfg)
        return bool(obj_name and obj_name == getattr(env.core.ego_state, 'held_object_name', None))

    def _target_has_parent_role(self, env: Any, cfg: DisturbanceConfig, role_name: str) -> bool:
        _, obj_name = _resolve_target(env, cfg)
        parent_name = env.role_map.get(str(role_name), str(role_name))
        if not obj_name or not parent_name:
            return False
        obj = env.core.objects.get(obj_name)
        parent = env.core.objects.get(parent_name)
        if obj is None or parent is None:
            return False
        try:
            return bool(_is_inside(obj, parent) or _is_ontop(obj, parent))
        except Exception:
            return False

    def _fire(self, env: Any, index: int, cfg: DisturbanceConfig, *, phase: str, high_level_step: int) -> Optional[SchedulerEvent]:
        target_key, obj_name = _resolve_target(env, cfg)
        if obj_name is None:
            return None
        obj = env.core.objects.get(obj_name)
        if obj is None:
            return None
        if cfg.requires_out_of_view:
            visibility = _visibility_payload(env, target_key, obj_name)
            if visibility.get('target_visible'):
                return SchedulerEvent('disturbance_out_of_view_wait', phase, {'config_index': index, 'disturbance_type': cfg.type, 'target_role': cfg.target_role, 'target_object_id': cfg.target_object_id, 'target': obj_name, 'step': high_level_step, 'reason': 'target_visible', 'requires_out_of_view': True, 'activation': self._activation_payload(index, cfg), 'visibility': visibility})
        before = _snapshot(obj)
        mutation = self._apply(cfg, obj, env)
        after = _snapshot(obj)
        if before == after:
            return None
        visibility = _visibility_payload(env, target_key, obj_name)
        payload = {'config_index': index, 'target_role': cfg.target_role, 'target_object_id': cfg.target_object_id, 'target': obj_name, 'step': high_level_step, 'before': before, 'after': after, 'after_step': cfg.after_step, 'requires_out_of_view': bool(getattr(cfg, 'requires_out_of_view', False)), 'activation': self._activation_payload(index, cfg), 'visibility': visibility}
        owner = _activation_subtask_id(cfg.activation)
        if owner is not None:
            payload['owner_subtask_id'] = owner
        if mutation:
            payload['mutation'] = mutation
        return SchedulerEvent(cfg.type, phase, payload)

    def _activation_payload(self, index: int, cfg: DisturbanceConfig) -> dict:
        activation = cfg.activation if isinstance(cfg.activation, Mapping) else None
        state = self._activation_state.get(index) or {}
        if not activation and (not state):
            return {}
        payload = {'type': activation.get('type') if activation else None, 'armed': bool(state.get('armed', not activation))}
        if state.get('armed_step') is not None:
            payload['armed_step'] = state.get('armed_step')
        if state.get('distinct_tool_count') is not None:
            payload['distinct_tool_count'] = state.get('distinct_tool_count')
        if state.get('armed_by') is not None:
            payload['armed_by'] = dict(state.get('armed_by') or {})
        if activation:
            for key in ('tool', 'tools', 'target_role', 'target_roles', 'success', 'min_delay_steps', 'after_distinct_tool_count', 'distinct_tool_count', 'required_parent_role', 'requires_target_not_held', 'state_delta_object_role', 'ignore_tools', 'exclude_target_roles', 'distinct_success_only', 'subtask_id', 'owner_subtask_id'):
                if key in activation:
                    payload[key] = activation[key]
        return payload

    def _apply(self, cfg: DisturbanceConfig, obj: Any, env: Any) -> Optional[dict[str, Any]]:
        mutation: dict[str, Any] | None = None
        if cfg.type == 'close_object':
            _force_state(obj, 'open', False, actions.set_open_state)
            mutation = _sync_boolean_disturbance_renderer(env, obj, 'open', False, reason='disturbance_close_object')
        elif cfg.type == 'open_object':
            _force_state(obj, 'open', True, actions.set_open_state)
            mutation = _sync_boolean_disturbance_renderer(env, obj, 'open', True, reason='disturbance_open_object')
        elif cfg.type == 'toggle_on':
            _force_state(obj, 'toggled_on', True, actions.set_toggle_state)
            mutation = _sync_boolean_disturbance_renderer(env, obj, 'toggled_on', True, reason='disturbance_toggle_on')
        elif cfg.type == 'toggle_off':
            _force_state(obj, 'toggled_on', False, actions.set_toggle_state)
            mutation = _sync_boolean_disturbance_renderer(env, obj, 'toggled_on', False, reason='disturbance_toggle_off')
        elif cfg.type == 'dirty_object':
            _force_state(obj, 'dirty', True, actions.set_dirty_state)
            mutation = _sync_boolean_disturbance_renderer(env, obj, 'dirty', True, reason='disturbance_dirty_object')
        elif cfg.type == 'relocate_object':
            return _relocate_object_to_recoverable_ground(obj, env, cfg.extra)
        elif cfg.type == 'restore_initial_state':
            target_key = cfg.target_role or cfg.target_object_id
            snap = self._initial_snapshot.get(target_key, {})
            if 'open' in snap and snap['open'] is not None:
                obj.states['open'] = snap['open']
                mutation = _sync_boolean_disturbance_renderer(env, obj, 'open', bool(snap['open']), reason='disturbance_restore_initial_state')
            if 'toggled_on' in snap and snap['toggled_on'] is not None:
                obj.states['toggled_on'] = snap['toggled_on']
                toggle_mutation = _sync_boolean_disturbance_renderer(env, obj, 'toggled_on', bool(snap['toggled_on']), reason='disturbance_restore_initial_state')
                if mutation is None:
                    mutation = toggle_mutation
                else:
                    mutation = {'states': [mutation, toggle_mutation]}
        return mutation

def _force_state(obj: Any, state_key: str, value: bool, setter) -> None:
    try:
        setter(obj, value)
    except (ValueError, AssertionError):
        states = getattr(obj, 'states', None)
        if isinstance(states, dict):
            states[state_key] = bool(value)

def _sync_boolean_disturbance_renderer(env: Any, obj: Any, state_key: str, value: bool, *, reason: str) -> dict[str, Any]:
    object_name = str(getattr(obj, 'name', ''))
    report: dict[str, Any] = {'state': str(state_key), 'value': bool(value), 'object': object_name, 'renderer_synced': False}
    renderer = getattr(env, 'renderer', None)
    if renderer is None:
        report['reason'] = 'renderer_unavailable'
        return report
    try:
        if state_key == 'open':
            sync = getattr(renderer, 'set_open_visual', None)
            if callable(sync):
                payload = dict(sync(object_name, bool(value), reason=reason) or {})
            else:
                report['reason'] = 'renderer_missing_set_open_visual'
                return report
        elif state_key == 'toggled_on':
            sync = getattr(renderer, 'set_toggle_visual', None)
            if callable(sync):
                payload = dict(sync(object_name, bool(value), reason=reason) or {})
            else:
                report['reason'] = 'renderer_missing_set_toggle_visual'
                return report
        elif state_key == 'dirty':
            sync = getattr(renderer, 'set_dirty_visual', None)
            if callable(sync):
                payload = dict(sync(object_name, bool(value), reason=reason) or {})
            else:
                report['reason'] = 'renderer_missing_set_dirty_visual'
                return report
        else:
            report['reason'] = 'unsupported_renderer_state'
            return report
    except Exception as exc:
        report['reason'] = 'renderer_sync_error'
        report['error'] = repr(exc)
        return report
    report.update(payload)
    report['renderer_synced'] = bool(payload.get('success', False))
    return report

def _snapshot(obj: Any) -> dict:
    states = getattr(obj, 'states', {}) or {}
    return {'open': states.get('open'), 'toggled_on': states.get('toggled_on'), 'dirty': states.get('dirty'), 'center': _vector3(getattr(obj, 'center', None))}

def _relocate_object_to_recoverable_ground(obj: Any, env: Any, extra: Mapping[str, Any] | None=None) -> Optional[dict[str, Any]]:
    if str(getattr(obj, 'category', '') or '').lower() in {'fridge', 'refrigerator', 'freezer', 'microwave', 'microwave_oven'}:
        return None
    before_center = tuple((float(value) for value in getattr(obj, 'center', (0.0, 0.0, 0.0))))
    objects = list(getattr(env.core, 'objects', {}).values())
    options = _relocation_options(env, obj, before_center, extra)
    for candidate in _relocation_ground_candidates(env, obj, before_center, objects, options):
        center = candidate['center']
        if _relocation_navigation_plan(obj, center, env) is None:
            continue
        commit = _commit_relocation(env, obj, center)
        if not commit.get('success'):
            continue
        return {'mode': 'recoverable_ground', 'before_center': list(before_center), 'after_center': list(center), 'before_room_id': candidate.get('before_room_id'), 'after_room_id': candidate.get('after_room_id'), 'different_room': bool(candidate.get('different_room')), 'distance_m': float(candidate.get('distance_m', 0.0)), 'connected_component': candidate.get('connected_component'), 'commit': commit}
    return None

def _relocation_options(env: Any, obj: Any, before_center: tuple[float, float, float], extra: Mapping[str, Any] | None) -> dict[str, Any]:
    extra = extra if isinstance(extra, Mapping) else {}
    exclude_object_names = {str(getattr(obj, 'name', ''))}
    for role_name in extra.get('exclude_roles') or []:
        object_name = getattr(env, 'role_map', {}).get(str(role_name), str(role_name))
        if object_name:
            exclude_object_names.add(str(object_name))
    for object_name in extra.get('exclude_object_ids') or extra.get('exclude_objects') or []:
        if object_name:
            exclude_object_names.add(str(object_name))
    try:
        min_distance = float(extra.get('min_distance', 0.25))
    except Exception:
        min_distance = 0.25
    try:
        max_distance = extra.get('max_distance')
        max_distance = float(max_distance) if max_distance is not None else None
    except Exception:
        max_distance = None
    return {'exclude_object_names': exclude_object_names, 'min_distance': max(0.0, min_distance), 'max_distance': max_distance if max_distance is None or max_distance >= 0.0 else None, 'avoid_current_view': bool(extra.get('avoid_current_view', False)), 'avoid_last_seen_view': bool(extra.get('avoid_last_seen_view', False)), 'prefer_different_room': bool(extra.get('prefer_different_room', True)), 'max_candidates': max(1, min(8, int(extra.get('max_candidates', 8) or 8))), 'before_center': before_center}

def _relocation_ground_candidates(env: Any, obj: Any, before_center: tuple[float, float, float], objects: List[Any], options: Mapping[str, Any]) -> list[dict[str, Any]]:
    if getattr(env.core, 'occupancy', None) is None:
        env.core.build_occupancy('disturbance_relocation_candidates')
    occupancy = getattr(env.core, 'occupancy', None)
    if occupancy is None:
        return []
    start_component = int(occupancy.component_at(env.core.ego_state.pose.xy))
    if start_component < 0:
        return []
    min_distance = float(options.get('min_distance', 1.0))
    max_distance = options.get('max_distance')
    max_distance = float(max_distance) if max_distance is not None else 4.0
    if max_distance < min_distance:
        return []
    resolution = max(float(getattr(occupancy, 'resolution', 0.1)), 1e-06)
    origin_cell = occupancy.world_to_cell(before_center[:2])
    radius_cells = max(1, int(math.ceil(max_distance / resolution)))
    room_query = getattr(env, '_private_room_id_for_xy', None)
    before_room_id = room_query(before_center[:2]) if callable(room_query) else None
    floor_height = float(getattr(getattr(env.core.config, 'occupancy', None), 'floor_height', 0.0))
    center_z = floor_height + float(getattr(obj, 'extent', (0.2, 0.2, 0.2))[2]) * 0.5 + 0.005
    candidates: list[dict[str, Any]] = []
    for row in range(int(origin_cell[0]) - radius_cells, int(origin_cell[0]) + radius_cells + 1):
        for col in range(int(origin_cell[1]) - radius_cells, int(origin_cell[1]) + radius_cells + 1):
            cell = (row, col)
            if not occupancy.in_bounds_cell(cell):
                continue
            try:
                free = occupancy.is_cell_free(cell, eroded=True)
            except TypeError:
                free = occupancy.is_cell_free(cell)
            if not free:
                continue
            xy = occupancy.cell_to_world(cell)
            distance = _xy_distance(before_center, xy)
            if distance < min_distance or distance > max_distance:
                continue
            component = int(occupancy.component_at(xy))
            if component != start_component:
                continue
            center = (float(xy[0]), float(xy[1]), float(center_z))
            if not _relocation_center_allowed(env, obj, before_center, center, options):
                continue
            if _center_collides(obj, center, objects, ignore_names={str(getattr(obj, 'name', ''))}):
                continue
            after_room_id = room_query(xy) if callable(room_query) else None
            different_room = bool(before_room_id and after_room_id and (before_room_id != after_room_id))
            preferred_distance = 0.5 * (min_distance + max_distance)
            candidates.append({'center': center, 'before_room_id': before_room_id, 'after_room_id': after_room_id, 'different_room': different_room, 'distance_m': float(distance), 'connected_component': component, 'score': (0 if bool(options.get('prefer_different_room')) and different_room else 1, abs(float(distance) - preferred_distance), row, col)})
    candidates.sort(key=lambda item: item['score'])
    limit = int(options.get('max_candidates', 8))
    if not bool(options.get('prefer_different_room')):
        return candidates[:limit]
    preferred = [candidate for candidate in candidates if candidate['different_room']]
    fallback = [candidate for candidate in candidates if not candidate['different_room']]
    if not preferred or not fallback or limit <= 1:
        return candidates[:limit]
    preferred_limit = max(1, (limit + 1) // 2)
    selected = preferred[:preferred_limit] + fallback[:limit - preferred_limit]
    if len(selected) < limit:
        selected.extend(preferred[preferred_limit:preferred_limit + limit - len(selected)])
    if len(selected) < limit:
        selected.extend(fallback[limit - preferred_limit:limit - preferred_limit + limit - len(selected)])
    return selected[:limit]

def _activation_subtask_id(activation: Mapping[str, Any] | None) -> str | None:
    if not isinstance(activation, Mapping):
        return None
    value = activation.get('subtask_id') or activation.get('owner_subtask_id')
    return str(value) if value is not None else None

def _subtask_gate_matches(env: Any, activation: Mapping[str, Any] | None) -> bool:
    expected = _activation_subtask_id(activation)
    if expected is None:
        return True
    current = getattr(env, 'current_composite_subtask_id', None)
    return str(current) == str(expected)

def _relocation_center_allowed(env: Any, obj: Any, before_center: tuple[float, float, float], center: tuple[float, float, float], options: Mapping[str, Any]) -> bool:
    floor_height = float(getattr(getattr(env.core.config, 'occupancy', None), 'floor_height', 0.0))
    if float(center[2]) > floor_height + _MAX_RELOCATION_CENTER_HEIGHT_ABOVE_FLOOR_M:
        return False
    distance = _xy_distance(before_center, center)
    if distance < float(options.get('min_distance', 0.25)):
        return False
    max_distance = options.get('max_distance')
    if max_distance is not None and distance > float(max_distance):
        return False
    if bool(options.get('avoid_current_view')) and _relocation_center_visible_from_current_pose(env, obj, center):
        return False
    if bool(options.get('avoid_last_seen_view')) and _relocation_center_visible_from_last_seen_pose(env, obj, center):
        return False
    return True

def _relocation_center_visible_from_current_pose(env: Any, obj: Any, center: tuple[float, float, float]) -> bool:
    try:
        pose = dict(env._agent_pose())
    except Exception:
        return False
    return _relocation_center_visible_from_pose(env, obj, center, pose)

def _relocation_center_visible_from_last_seen_pose(env: Any, obj: Any, center: tuple[float, float, float]) -> bool:
    tracker = getattr(env, '_belief_tracker', None)
    if tracker is None or not hasattr(tracker, 'get'):
        return False
    try:
        record = tracker.get(str(getattr(obj, 'name', '')))
    except Exception:
        record = None
    pose = getattr(record, 'last_seen_pose', None)
    if not isinstance(pose, Mapping):
        return False
    return _relocation_center_visible_from_pose(env, obj, center, pose)

def _relocation_center_visible_from_pose(env: Any, obj: Any, center: tuple[float, float, float], pose: Mapping[str, Any]) -> bool:
    visible_query = getattr(env, '_object_visible_from_pose', None)
    if not callable(visible_query):
        return False
    try:
        proxy = _AABBProxy(obj, center)
        return bool(visible_query(proxy, pose))
    except Exception:
        return False

def _relocation_reachable(obj: Any, center: tuple[float, float, float], env: Any) -> bool:
    return _relocation_navigation_plan(obj, center, env) is not None

def _relocation_navigation_plan(obj: Any, center: tuple[float, float, float], env: Any) -> Any | None:
    original = getattr(obj, 'center', None)
    try:
        obj.center = center
        occupancy = env.core._build_occupancy_snapshot()
        plan = plan_navigation(env.core.ego_state.pose.copy(), NavigationTarget.from_aabb_object(obj), occupancy, env.core.config.navigation)
        return plan if plan.success and getattr(plan, 'selected_goal_xy', None) is not None else None
    except Exception:
        return None
    finally:
        if original is not None:
            obj.center = original

def _center_collides(obj: Any, center: tuple[float, float, float], objects: List[Any], *, ignore_names: set[str]) -> bool:
    proxy = _AABBProxy(obj, center)
    for other in objects:
        other_name = str(getattr(other, 'name', ''))
        if other_name in ignore_names or _skip_collision_object(other):
            continue
        if _aabb_overlap(proxy, other, tolerance=0.01):
            return True
    return False

class _AABBProxy:

    def __init__(self, obj: Any, center: tuple[float, float, float]) -> None:
        self.center = center
        self.extent = getattr(obj, 'extent', (0.2, 0.2, 0.2))

def _aabb_overlap(a: Any, b: Any, *, tolerance: float) -> bool:
    for axis in range(3):
        delta = abs(float(a.center[axis]) - float(b.center[axis]))
        limit = float(a.extent[axis]) * 0.5 + float(b.extent[axis]) * 0.5 - float(tolerance)
        if delta >= limit:
            return False
    return True

def _skip_collision_object(obj: Any) -> bool:
    return str(getattr(obj, 'category', '')).lower() in {'floors', 'ceilings', 'walls'}

def _xy_distance(a: Any, b: Any) -> float:
    try:
        return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5
    except Exception:
        return 0.0

def _commit_relocation(env: Any, obj: Any, center: tuple[float, float, float]) -> dict[str, Any]:
    before_pose = {name: getattr(obj, name, None) for name in ('center', 'extent', 'root_position', 'root_orientation_xyzw', 'root_yaw', 'pose_source')}
    before_center = tuple((float(value) for value in obj.center))
    renderer = getattr(env, 'renderer', None)
    report: dict[str, Any] = {'success': False, 'renderer_synced': False, 'rendered_visible_at_interaction_pose': None, 'occupancy_rebuilt': False, 'public_navigation_ready': False, 'rollback_success': False}
    current_pose = env.core.ego_state.pose.copy()
    obj.center = center
    try:
        if renderer is not None:
            setter = getattr(renderer, 'set_object_center', None)
            if not callable(setter):
                raise RuntimeError('renderer_set_object_center_unavailable')
            diagnostic = setter(str(getattr(obj, 'name', '')), center, verify_readback=True)
            if not isinstance(diagnostic, Mapping) or not bool(diagnostic.get('success')):
                raise RuntimeError('renderer_readback_failed:{!r}'.format(diagnostic))
            pose_mutation = dict(diagnostic)
            pose_mutation['core_object_name'] = str(getattr(obj, 'name', ''))
            env._update_core_object_pose_from_visual_mutation(pose_mutation)
            report['renderer_synced'] = True
        env.core.build_occupancy('disturbance_relocate_object')
        report['occupancy_rebuilt'] = True
        if renderer is not None:
            selection = env._validate_default_pickup_route(str(getattr(obj, 'name', '')), start_from_last_seen_pose=True)
            report['navigation_selection'] = _navigation_selection_summary(selection)
            report['stable_visibility_passes'] = selection.get('stable_visibility_passes')
            report['stable_visibility_required'] = selection.get('stable_visibility_required')
            if not bool(selection.get('success')):
                raise RuntimeError('relocated_object_not_publicly_navigable')
            report['public_navigation_ready'] = True
            report['rendered_visible_at_interaction_pose'] = bool(selection.get('rendered_visible'))
            env._sync_renderer_for_observation_pose(env._pose_to_dict(current_pose))
        else:
            report['navigation_validation_mode'] = 'symbolic_prefilter_without_renderer'
        if hasattr(env, 'invalidate_private_parent_placement'):
            env.invalidate_private_parent_placement(str(getattr(obj, 'name', '')))
        staged = getattr(env, '_staged_object_ledger', None)
        if isinstance(staged, dict):
            staged.pop(str(getattr(obj, 'name', '')), None)
        report['success'] = True
        from scoper.benchmark.common.evaluation_navigation import relocated
        relocated(env, str(getattr(obj, 'name', '')), before_center)
        return report
    except Exception as exc:
        report['error'] = repr(exc)
        for name, value in before_pose.items():
            setattr(obj, name, value)
        try:
            if renderer is not None and callable(getattr(renderer, 'set_object_center', None)):
                renderer.set_object_center(str(getattr(obj, 'name', '')), before_center, verify_readback=True)
                env._sync_renderer_for_observation_pose(env._pose_to_dict(current_pose))
            env.core.build_occupancy('disturbance_relocate_object_rollback')
            report['rollback_success'] = True
        except Exception as rollback_exc:
            report['rollback_error'] = repr(rollback_exc)
        return report

def _navigation_selection_summary(selection: Any) -> dict[str, Any]:
    if not isinstance(selection, Mapping):
        return {'success': False, 'error_type': 'invalid_navigation_selection'}
    return {'success': bool(selection.get('success')), 'error_type': selection.get('error_type'), 'selected_goal_xy': selection.get('selected_goal_xy'), 'reachable_candidate_count': selection.get('reachable_candidate_count'), 'line_of_sight_clear_candidate_count': selection.get('line_of_sight_clear_candidate_count')}

def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return [float(item) for item in value]
    except Exception:
        return None

def _resolve_target(env: Any, cfg: DisturbanceConfig) -> tuple[str, Optional[str]]:
    if cfg.target_object_id:
        object_id = str(cfg.target_object_id)
        if object_id in getattr(env.core, 'objects', {}):
            return (object_id, object_id)
    role = str(cfg.target_role) if cfg.target_role is not None else str(cfg.target_object_id or '')
    return (role, env.role_map.get(role))

def _disturbance_notice(cfg: DisturbanceConfig, event: SchedulerEvent) -> str:
    if cfg.notify_message:
        return str(cfg.notify_message)
    payload = event.payload if isinstance(event.payload, Mapping) else {}
    return 'Disturbance {} applied to {}.'.format(cfg.type, payload.get('target_role') or payload.get('target_object_id') or payload.get('target') or 'object')

def _visibility_payload(env: Any, target_role: str, obj_name: str) -> dict:
    try:
        visible_ids = list(env._visible_object_names_at_pose(env.core.ego_state.pose))
    except Exception:
        visible_ids = []
    try:
        visible_roles = list(env._roles_for_object_names(visible_ids))
    except Exception:
        visible_roles = []
    try:
        agent_pose = dict(env._agent_pose())
    except Exception:
        agent_pose = {}
    return {'mode': str(getattr(env, '_last_visibility_mode', 'symbolic_fov') or 'symbolic_fov'), 'target_role': target_role, 'target': obj_name, 'target_visible': obj_name in set(visible_ids), 'visible_object_ids': sorted((str(item) for item in visible_ids)), 'visible_object_roles': sorted((str(item) for item in visible_roles)), 'agent_pose': agent_pose}

def _disturbance_config_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {'type': cfg.get('type'), 'target_role': cfg.get('target_role'), 'target_object_id': cfg.get('target_object_id'), 'after_step': int(cfg.get('after_step', 0) or 0), 'once': bool(cfg.get('once', True)), 'activation': dict(cfg.get('activation') or {}) if isinstance(cfg.get('activation'), Mapping) else None, 'requires_out_of_view': bool(cfg.get('requires_out_of_view', True)), 'before_step': int(cfg.get('before_step', 10 ** 9) or 10 ** 9), 'probability': float(cfg.get('probability', 1.0) if cfg.get('probability', 1.0) is not None else 1.0), 'cooldown': int(cfg.get('cooldown', 0) or 0), 'max_count': int(cfg.get('max_count', 10 ** 9) or 10 ** 9), 'notify_brain': bool(cfg.get('notify_brain', False)), 'notify_message': cfg.get('notify_message'), 'extra': dict(cfg.get('extra') or {}) if isinstance(cfg.get('extra'), Mapping) else {}}

def _transition_step(transition: Optional[Mapping[str, Any]], fallback: int) -> int:
    if not isinstance(transition, Mapping):
        return int(fallback)
    value = transition.get('high_level_step')
    try:
        return int(value)
    except Exception:
        return int(fallback)

def _transition_brief(transition: Optional[Mapping[str, Any]]) -> dict:
    if not isinstance(transition, Mapping):
        return {}
    return {'tool': transition.get('tool_name') or transition.get('tool'), 'target': (transition.get('arguments') or {}).get('target') if isinstance(transition.get('arguments'), Mapping) else None, 'success': bool(transition.get('success', False)), 'step': _transition_step(transition, -1)}

def _nonnegative_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return int(default)
__all__ = ['DisturbanceConfig', 'DisturbanceScheduler', 'KNOWN_DISTURBANCE_TYPES']
