from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from scoper.benchmark.procthor.scheduler.base import Scheduler, SchedulerEvent
KNOWN_DISTURBANCE_TYPES = ('close_object', 'open_object', 'toggle_on', 'toggle_off', 'dirty_object', 'relocate_object', 'restore_initial_state')

@dataclass
class DisturbanceConfig:
    type: str
    target_role: Optional[str] = None
    target_object_id: Optional[str] = None
    after_step: int = 0
    before_step: int = 10 ** 9
    probability: float = 1.0
    requires_out_of_view: bool = True
    cooldown: int = 0
    once: bool = True
    notify_brain: bool = False
    notify_message: Optional[str] = None
    activation: Optional[dict] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in KNOWN_DISTURBANCE_TYPES:
            raise ValueError("Unknown disturbance type '{}'. Known: {}.".format(self.type, ', '.join(KNOWN_DISTURBANCE_TYPES)))
        if self.target_role is None and self.target_object_id is None:
            raise ValueError('DisturbanceConfig requires target_role or target_object_id.')
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError('DisturbanceConfig.probability must be in [0, 1].')

class DisturbanceScheduler(Scheduler):
    name = 'disturbance'

    def __init__(self, configs: Sequence[Mapping[str, Any] | DisturbanceConfig], seed: int=0) -> None:
        self._configs: List[DisturbanceConfig] = [cfg if isinstance(cfg, DisturbanceConfig) else DisturbanceConfig(**dict(cfg)) for cfg in configs]
        self._rng = random.Random(seed)
        self._fire_counts: Dict[int, int] = {i: 0 for i in range(len(self._configs))}
        self._last_fire_step: Dict[int, int] = {}
        self._initial_state_by_object_id: Dict[str, dict] = {}
        self._activation_state: Dict[int, dict] = {}
        self._pending_notices: List[str] = []
        self._notice_callback_installed: bool = False
        self._current_visible_ids: set = set()
        self._last_seen_visible_ids: set = set()

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> Optional[List[SchedulerEvent]]:
        self._fire_counts = {i: 0 for i in range(len(self._configs))}
        self._last_fire_step = {}
        self._initial_state_by_object_id = self._snapshot_targets(env)
        self._activation_state = {i: {} for i in range(len(self._configs))}
        self._pending_notices = []
        self._current_visible_ids = set()
        self._last_seen_visible_ids = set()
        self._update_visible_tracking(env)
        self._install_prompt_notice_hook(env)
        return [SchedulerEvent('disturbance_armed', 'episode_start', {'num_configs': len(self._configs)})]

    def pre_step(self, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]) -> Optional[List[SchedulerEvent]]:
        pending_qa = bool((getattr(env, '_human_qa_state', {}) or {}).get('pending'))
        self._update_visible_tracking(env)
        events: List[SchedulerEvent] = []
        for i, cfg in enumerate(self._configs):
            if not self._eligible(i, cfg, env, high_level_step, last_transition):
                continue
            if pending_qa:
                continue
            if cfg.probability < 1.0 and self._rng.random() > cfg.probability:
                continue
            event = self._apply(env, i, cfg, high_level_step)
            if event is not None:
                invalidate = getattr(env, 'invalidate_public_placed_container_navigation', None)
                if callable(invalidate):
                    invalidate()
                events.append(event)
                self._fire_counts[i] += 1
                self._last_fire_step[i] = high_level_step
                if cfg.notify_brain:
                    self._pending_notices.append(self._build_notice(cfg, event))
        return events or None

    def _eligible(self, index: int, cfg: DisturbanceConfig, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]) -> bool:
        if not _active_subtask_matches(env, cfg.extra.get('active_subtask_id')):
            return False
        if cfg.once and self._fire_counts.get(index, 0) > 0:
            return False
        if high_level_step < cfg.after_step:
            return False
        if high_level_step >= cfg.before_step:
            return False
        last = self._last_fire_step.get(index)
        if last is not None and high_level_step - last < cfg.cooldown:
            return False
        if self._resolve_object_id(env, cfg) is None:
            return False
        if not self._activation_ready(index, cfg, env, high_level_step, last_transition):
            return False
        if cfg.requires_out_of_view and self._target_visible(env, cfg):
            return False
        return True

    def _activation_ready(self, index: int, cfg: DisturbanceConfig, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]) -> bool:
        activation = cfg.activation if isinstance(cfg.activation, Mapping) else None
        if not activation:
            return True
        activation_type = activation.get('type') or 'high_level_step'
        if activation_type == 'high_level_step':
            return True
        if activation_type != 'after_tool_result':
            return False
        state = self._activation_state.setdefault(index, {})
        if not state.get('armed') and self._transition_matches_activation(activation, last_transition, env):
            state['armed'] = True
            armed_step = _transition_step(last_transition, high_level_step - 1)
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

    def _transition_matches_activation(self, activation: Mapping[str, Any], transition: Optional[Mapping[str, Any]], env: Any=None) -> bool:
        if not isinstance(transition, Mapping):
            return False
        tool = transition.get('tool_name') or transition.get('tool')
        expected_tools = activation.get('tools')
        if expected_tools is None and activation.get('tool') is not None:
            expected_tools = [activation.get('tool')]
        if expected_tools is not None and str(tool) not in {str(t) for t in expected_tools}:
            return False
        if 'success' in activation and bool(transition.get('success')) != bool(activation.get('success')):
            return False
        arguments = transition.get('arguments')
        target = arguments.get('target') if isinstance(arguments, Mapping) else None
        target_roles = activation.get('target_roles')
        if target_roles is None and activation.get('target_role') is not None:
            target_roles = [activation.get('target_role')]
        from scoper.benchmark.common.transition_roles import target_roles as resolved_roles
        if target_roles is not None and (not resolved_roles(env, transition).intersection((str(t) for t in target_roles))):
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
        ignore_tools = {str(t) for t in activation.get('ignore_tools', ['stop', 'answer'])}
        if tool in ignore_tools:
            state['last_counted_step'] = step
            return
        if activation.get('distinct_success_only', True) and (not bool(transition.get('success', False))):
            state['last_counted_step'] = step
            return
        arguments = transition.get('arguments')
        target = arguments.get('target') if isinstance(arguments, Mapping) else None
        exclude_targets = set((str(t) for t in activation.get('exclude_target_roles', [])))
        if cfg.target_role:
            exclude_targets.add(str(cfg.target_role))
        from scoper.benchmark.common.transition_roles import target_roles as resolved_roles
        if resolved_roles(env, transition).intersection(exclude_targets):
            state['last_counted_step'] = step
            return
        state['distinct_tool_count'] = int(state.get('distinct_tool_count', 0)) + 1
        state['last_counted_step'] = step

    def _target_held(self, env: Any, cfg: DisturbanceConfig) -> bool:
        object_id = self._resolve_object_id(env, cfg)
        if not object_id:
            return False
        inventory = (env.controller.last_event.metadata or {}).get('inventoryObjects') or []
        return any((isinstance(item, Mapping) and item.get('objectId') == object_id for item in inventory))

    def _target_has_parent_role(self, env: Any, cfg: DisturbanceConfig, role_name: str) -> bool:
        object_id = self._resolve_object_id(env, cfg)
        parent_id = getattr(env, 'role_object_ids', {}).get(role_name)
        if not object_id or not parent_id:
            return False
        obj = env._objects_by_id().get(object_id, {})
        parents = obj.get('parentReceptacles') or obj.get('parent_receptacles') or []
        return parent_id in parents

    def _resolve_object_id(self, env: Any, cfg: DisturbanceConfig) -> Optional[str]:
        if cfg.target_object_id:
            return cfg.target_object_id
        return env.role_object_ids.get(cfg.target_role) if cfg.target_role else None

    def _target_visible(self, env: Any, cfg: DisturbanceConfig) -> bool:
        object_id = self._resolve_object_id(env, cfg)
        if not object_id:
            return False
        obj = env._objects_by_id().get(object_id, {})
        return bool(obj.get('visible', False))

    def _snapshot_targets(self, env: Any) -> Dict[str, dict]:
        snapshot: Dict[str, dict] = {}
        objects = env._objects_by_id()
        for cfg in self._configs:
            object_id = self._resolve_object_id(env, cfg)
            if not object_id or object_id in snapshot:
                continue
            obj = objects.get(object_id)
            if obj is None:
                continue
            snapshot[object_id] = {'isOpen': bool(obj.get('isOpen', False)), 'isToggled': bool(obj.get('isToggled', False)), 'openness': obj.get('openness')}
        return snapshot

    def _apply(self, env: Any, index: int, cfg: DisturbanceConfig, high_level_step: int) -> Optional[SchedulerEvent]:
        object_id = self._resolve_object_id(env, cfg)
        if object_id is None:
            return None
        before = self._object_state(env, object_id)
        action: Optional[str] = None
        kwargs: Dict[str, Any] = {'objectId': object_id, 'forceAction': True}
        if cfg.type == 'close_object':
            if not before.get('openable') or before.get('isOpen') is False:
                return None
            action = 'CloseObject'
        elif cfg.type == 'open_object':
            if not before.get('openable') or before.get('isOpen') is True:
                return None
            action = 'OpenObject'
        elif cfg.type == 'toggle_off':
            if not before.get('toggleable') or before.get('isToggled') is False:
                return None
            action = 'ToggleObjectOff'
        elif cfg.type == 'toggle_on':
            if not before.get('toggleable') or before.get('isToggled') is True:
                return None
            action = 'ToggleObjectOn'
        elif cfg.type == 'dirty_object':
            return self._apply_dirty_object(env, index, cfg, high_level_step, object_id, before)
        elif cfg.type == 'relocate_object':
            return self._apply_relocate_object(env, index, cfg, high_level_step, object_id, before)
        elif cfg.type == 'restore_initial_state':
            initial = self._initial_state_by_object_id.get(object_id)
            if initial is None:
                return None
            actions: List[str] = []
            if before.get('openable') and bool(before.get('isOpen', False)) != bool(initial.get('isOpen', False)):
                actions.append('OpenObject' if initial.get('isOpen') else 'CloseObject')
            if before.get('toggleable') and bool(before.get('isToggled', False)) != bool(initial.get('isToggled', False)):
                actions.append('ToggleObjectOn' if initial.get('isToggled') else 'ToggleObjectOff')
            if not actions:
                return None
            for a in actions:
                env.controller.step(action=a, objectId=object_id, forceAction=True)
            after = self._object_state(env, object_id)
            return SchedulerEvent(cfg.type, 'pre_step', {'object_id': object_id, 'target_role': cfg.target_role, 'step': high_level_step, 'actions': actions, 'before': before, 'after': after, 'config_index': index, 'requires_out_of_view': cfg.requires_out_of_view, 'activation': self._activation_payload(index, cfg)})
        else:
            return None
        env.controller.step(action=action, **kwargs)
        after = self._object_state(env, object_id)
        return SchedulerEvent(cfg.type, 'pre_step', {'object_id': object_id, 'target_role': cfg.target_role, 'step': high_level_step, 'action': action, 'before': before, 'after': after, 'config_index': index, 'requires_out_of_view': cfg.requires_out_of_view, 'activation': self._activation_payload(index, cfg)})

    def _apply_dirty_object(self, env: Any, index: int, cfg: DisturbanceConfig, high_level_step: int, object_id: str, before: Mapping[str, Any]) -> Optional[SchedulerEvent]:
        if before.get('isDirty') is True:
            return None
        objects = env._objects_by_id()
        obj = objects.get(object_id, {})
        method = None
        last_error = ''
        try:
            event = env.controller.step(action='DirtyObject', objectId=object_id, forceAction=True)
            if bool(event.metadata.get('lastActionSuccess', False)):
                method = 'DirtyObject'
            else:
                last_error = event.metadata.get('errorMessage', '')
        except Exception as exc:
            last_error = str(exc)
        if method is None:
            try:
                event = env.controller.step(action='SetObjectStates', SetObjectStates={'objectType': obj.get('objectType'), 'isDirty': True})
                if bool(event.metadata.get('lastActionSuccess', False)):
                    method = 'SetObjectStates'
                else:
                    last_error = event.metadata.get('errorMessage', last_error)
            except Exception as exc:
                last_error = str(exc)
        if method is None:
            sticky_setter = getattr(env, '_set_sticky_object_state', None)
            if callable(sticky_setter):
                sticky_setter(object_id, isDirty=True)
                method = 'sticky_state'
        if method is None:
            return None
        after = self._object_state(env, object_id)
        if after.get('isDirty') is not True:
            after['isDirty'] = True
        return SchedulerEvent(cfg.type, 'pre_step', {'object_id': object_id, 'target_role': cfg.target_role, 'step': high_level_step, 'action': 'DirtyObject', 'method': method, 'before': before, 'after': after, 'config_index': index, 'requires_out_of_view': cfg.requires_out_of_view, 'error_message': last_error, 'activation': self._activation_payload(index, cfg)})

    def _apply_relocate_object(self, env: Any, index: int, cfg: DisturbanceConfig, high_level_step: int, object_id: str, before: Mapping[str, Any]) -> Optional[SchedulerEvent]:
        objects = env._objects_by_id()
        obj = objects.get(object_id, {})
        old_position = dict(obj.get('position') or {})
        candidates = self._relocation_candidates(env, object_id, cfg)
        if not candidates:
            return None
        object_rotation = dict(obj.get('rotation') or {'x': 0.0, 'y': 0.0, 'z': 0.0})
        chosen = None
        chosen_receptacle = None
        last_error = ''
        for receptacle_id, coordinate in candidates:
            try:
                event = env.controller.step(action='TeleportObject', objectId=object_id, position=dict(coordinate), rotation=object_rotation, forceAction=True)
            except Exception as exc:
                last_error = str(exc)
                continue
            if bool(event.metadata.get('lastActionSuccess', False)):
                chosen = dict(coordinate)
                chosen_receptacle = receptacle_id
                break
            last_error = event.metadata.get('errorMessage', '')
        if chosen is None:
            return None
        try:
            env.controller.step(action='Pass')
        except Exception:
            pass
        sticky_setter = getattr(env, '_set_sticky_object_state', None)
        if callable(sticky_setter) and chosen_receptacle:
            sticky_setter(object_id, parentReceptacles=[chosen_receptacle], position=dict(chosen))
        after = self._object_state(env, object_id)
        if chosen:
            after['position'] = dict(chosen)
        from scoper.benchmark.common.evaluation_navigation import relocated
        relocated(env, object_id, old_position)
        return SchedulerEvent(cfg.type, 'pre_step', {'object_id': object_id, 'target_role': cfg.target_role, 'step': high_level_step, 'action': 'TeleportObject', 'before': before, 'after': after, 'old_position': old_position, 'new_position': chosen, 'new_receptacle_object_id': chosen_receptacle, 'distance_moved': _position_distance(old_position, chosen or {}), 'config_index': index, 'requires_out_of_view': cfg.requires_out_of_view, 'error_message': last_error, 'activation': self._activation_payload(index, cfg)})

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
            for key in ('tool', 'tools', 'target_role', 'target_roles', 'success', 'min_delay_steps', 'after_distinct_tool_count', 'required_parent_role', 'requires_target_not_held'):
                if key in activation:
                    payload[key] = activation[key]
        return payload

    def _relocation_candidates(self, env: Any, object_id: str, cfg: DisturbanceConfig) -> List[tuple[str, Mapping[str, Any]]]:
        objects = env._objects_by_id()
        target = objects.get(object_id, {})
        target_pos = target.get('position') or {}
        target_room = _room_id(target)
        min_distance = float(cfg.extra.get('min_distance', 0.5))
        max_distance = float(cfg.extra.get('max_distance', 3.0))
        exclude_roles = set(cfg.extra.get('exclude_roles') or [])
        excluded_ids = {object_id}
        for role_name in exclude_roles:
            role_id = getattr(env, 'role_object_ids', {}).get(role_name)
            if role_id:
                excluded_ids.add(role_id)
        receptacles = []
        for candidate in objects.values():
            receptacle_id = candidate.get('objectId')
            if not isinstance(receptacle_id, str) or receptacle_id in excluded_ids:
                continue
            if not bool(candidate.get('receptacle', False)):
                continue
            if candidate.get('openable') and candidate.get('isOpen') is False:
                continue
            if target_room and _room_id(candidate) and (_room_id(candidate) != target_room):
                continue
            candidate_pos = candidate.get('position') or {}
            distance = _position_distance(target_pos, candidate_pos)
            if distance < min_distance or distance > max_distance:
                continue
            receptacles.append((distance, receptacle_id))
        receptacles.sort(key=lambda item: item[0])
        out: List[tuple[str, Mapping[str, Any]]] = []
        for _, receptacle_id in receptacles:
            try:
                event = env.controller.step(action='GetSpawnCoordinatesAboveReceptacle', objectId=receptacle_id, anywhere=True)
            except Exception:
                continue
            coordinates = event.metadata.get('actionReturn') or []
            for coordinate in coordinates:
                if isinstance(coordinate, Mapping):
                    out.append((receptacle_id, dict(coordinate)))
        self._rng.shuffle(out)
        avoid_current = cfg.extra.get('avoid_current_view', False)
        avoid_last = cfg.extra.get('avoid_last_seen_view', False)
        if avoid_current or avoid_last:
            out = self._filter_avoid_view(env, out, avoid_current, avoid_last)
        return out

    def _object_state(self, env: Any, object_id: str) -> dict:
        obj = env._objects_by_id().get(object_id, {})
        return {'objectId': object_id, 'objectType': obj.get('objectType'), 'isOpen': obj.get('isOpen'), 'isToggled': obj.get('isToggled'), 'isDirty': obj.get('isDirty'), 'openable': bool(obj.get('openable', False)), 'toggleable': bool(obj.get('toggleable', False)), 'dirtyable': bool(obj.get('dirtyable', False)), 'visible': bool(obj.get('visible', False)), 'openness': obj.get('openness'), 'position': dict(obj.get('position') or {}), 'parentReceptacles': list(obj.get('parentReceptacles') or [])}

    def _update_visible_tracking(self, env: Any) -> None:
        if self._current_visible_ids:
            self._last_seen_visible_ids = set(self._current_visible_ids)
        self._current_visible_ids = set()
        for obj in env._objects_by_id().values():
            if obj.get('visible'):
                object_id = obj.get('objectId')
                if object_id:
                    self._current_visible_ids.add(object_id)

    def _filter_avoid_view(self, env: Any, candidates: List[tuple[str, Mapping[str, Any]]], avoid_current: bool, avoid_last: bool) -> List[tuple[str, Mapping[str, Any]]]:
        if not avoid_current and (not avoid_last):
            return candidates
        avoid_ids = set()
        if avoid_current:
            avoid_ids.update(self._current_visible_ids)
        if avoid_last:
            avoid_ids.update(self._last_seen_visible_ids)
        filtered = []
        for receptacle_id, coordinate in candidates:
            if receptacle_id not in avoid_ids:
                filtered.append((receptacle_id, coordinate))
        return filtered

    def _install_prompt_notice_hook(self, env: Any) -> None:
        if self._notice_callback_installed:
            return
        original_get_prompt_context = env.get_prompt_context

        def _patched_get_prompt_context(*args, **kwargs):
            ctx = dict(original_get_prompt_context(*args, **kwargs))
            if self._pending_notices:
                ctx.setdefault('disturbance_notices', []).extend(self._pending_notices)
                self._pending_notices = []
            return ctx
        env.get_prompt_context = _patched_get_prompt_context
        self._notice_callback_installed = True

    def _build_notice(self, cfg: DisturbanceConfig, event: SchedulerEvent) -> str:
        if cfg.notify_message:
            return cfg.notify_message
        target = cfg.target_role or event.payload.get('object_id') or 'object'
        if cfg.type == 'close_object':
            return 'Disturbance: {} was closed.'.format(target)
        if cfg.type == 'open_object':
            return 'Disturbance: {} was opened.'.format(target)
        if cfg.type == 'toggle_off':
            return 'Disturbance: {} was switched off.'.format(target)
        if cfg.type == 'toggle_on':
            return 'Disturbance: {} was switched on.'.format(target)
        if cfg.type == 'restore_initial_state':
            return 'Disturbance: {} was reset to its initial state.'.format(target)
        if cfg.type == 'dirty_object':
            return 'Disturbance: {} was dirtied.'.format(target)
        if cfg.type == 'relocate_object':
            return 'Disturbance: {} was moved.'.format(target)
        return 'Disturbance applied to {}.'.format(target)

def _room_id(obj: Mapping[str, Any]) -> Optional[str]:
    value = obj.get('room_id') or obj.get('roomId')
    return str(value) if isinstance(value, str) and value else None

def _position_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    try:
        dx = float(a.get('x', 0.0)) - float(b.get('x', 0.0))
        dz = float(a.get('z', 0.0)) - float(b.get('z', 0.0))
    except (TypeError, ValueError):
        return 0.0
    return math.sqrt(dx * dx + dz * dz)

def _transition_step(transition: Optional[Mapping[str, Any]], fallback: int) -> int:
    if not isinstance(transition, Mapping):
        return int(fallback)
    value = transition.get('high_level_step')
    if isinstance(value, bool):
        return int(fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)

def _transition_brief(transition: Optional[Mapping[str, Any]]) -> dict:
    if not isinstance(transition, Mapping):
        return {}
    arguments = transition.get('arguments')
    target = arguments.get('target') if isinstance(arguments, Mapping) else None
    return {'step': _transition_step(transition, -1), 'tool': transition.get('tool_name') or transition.get('tool'), 'target': target, 'success': bool(transition.get('success', False))}

def _nonnegative_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, ivalue)

def _active_subtask_matches(env: Any, expected: Any) -> bool:
    if not expected:
        return True
    active = getattr(env, '_evaluation_active_subtask_spec', getattr(env, '_composite_active_task_spec', None))
    if not isinstance(active, Mapping):
        return False
    subtask_id = active.get('subtask_id')
    if not isinstance(subtask_id, str) or not subtask_id:
        task_id = active.get('id')
        if isinstance(task_id, str) and '::' in task_id:
            subtask_id = task_id.rsplit('::', 1)[1]
    return subtask_id == str(expected)
__all__ = ['DisturbanceConfig', 'DisturbanceScheduler', 'KNOWN_DISTURBANCE_TYPES']
