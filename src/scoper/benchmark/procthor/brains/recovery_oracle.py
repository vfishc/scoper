from __future__ import annotations
from typing import Any, List, Mapping, Optional, Tuple
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
_MAX_SAME_CALL = 2
_MAX_FAILED_CALLS = 2
_MAX_TOTAL_ACTS = 40

class RecoveryOracleBrain(BaseBrain):

    def __init__(self) -> None:
        self._task_spec: Optional[Mapping[str, Any]] = None
        self._env: Any = None
        self._acts = 0
        self._last_call_signature: Optional[Tuple[str, str]] = None
        self._same_call_count = 0
        self._failed_call_counts: dict[Tuple[str, str], int] = {}
        self._last_observed_transition_step: Optional[int] = None

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._task_spec = task_spec
        self._env = env
        self._acts = 0
        self._last_call_signature = None
        self._same_call_count = 0
        self._failed_call_counts = {}
        self._last_observed_transition_step = None

    def act(self, brain_input: BrainInput) -> ToolCall:
        self._remember_failed_transition(brain_input.last_transition)
        self._acts += 1
        if self._acts > _MAX_TOTAL_ACTS:
            return ToolCall('stop', {}, reasoning='recovery oracle: act budget exhausted')
        env = self._env
        task_spec = self._task_spec
        if env is None or task_spec is None:
            return ToolCall('stop', {}, reasoning='no task bound')
        goal = env.check_goal(model_stopped=getattr(env, 'model_stopped', False))
        if float(goal.get('predicate_success', 0.0)) >= 1.0:
            cleanup = self._plan_cleanup()
            if cleanup is not None:
                return self._track_and_return(cleanup)
        if float(goal.get('task_success', 0.0)) >= 1.0:
            return ToolCall('stop', {}, reasoning='goal satisfied')
        for predicate in task_spec['goal']['predicates']:
            call = self._plan_for_predicate(predicate)
            if call is not None:
                return self._track_and_return(call)
        return self._track_and_return(ToolCall('stop', {}, reasoning='no remaining work'))

    def _track_and_return(self, call: ToolCall) -> ToolCall:
        signature = _call_signature(call.tool_name, call.arguments)
        failed_count = self._failed_call_counts.get(signature, 0)
        if failed_count >= _MAX_FAILED_CALLS:
            return ToolCall('stop', {}, reasoning='recovery oracle: action {}({}) already failed {} times — giving up'.format(signature[0], signature[1], failed_count))
        if signature == self._last_call_signature:
            self._same_call_count += 1
        else:
            self._same_call_count = 1
            self._last_call_signature = signature
        if self._same_call_count > _MAX_SAME_CALL:
            return ToolCall('stop', {}, reasoning='recovery oracle: same action {}({}) failed {} times — giving up'.format(signature[0], signature[1], self._same_call_count))
        return call

    def _remember_failed_transition(self, transition: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(transition, Mapping):
            return
        transition_step = transition.get('high_level_step')
        if transition_step == self._last_observed_transition_step:
            return
        self._last_observed_transition_step = transition_step
        signature = _call_signature(str(transition.get('tool_name', '')), transition.get('arguments') if isinstance(transition.get('arguments'), Mapping) else {})
        if not transition.get('success', True):
            self._failed_call_counts[signature] = self._failed_call_counts.get(signature, 0) + 1
        else:
            self._failed_call_counts.pop(signature, None)

    def _plan_for_predicate(self, predicate: Mapping[str, Any]) -> Optional[ToolCall]:
        ptype = predicate.get('type')
        if ptype == 'object_in_receptacle':
            return self._plan_object_in_receptacle(predicate)
        if ptype in {'object_toggled_on', 'object_toggled_off'}:
            return self._plan_toggle(predicate)
        if ptype in {'object_open', 'object_closed'}:
            return self._plan_open_close(predicate)
        if ptype == 'object_sliced':
            return self._plan_sliced(predicate)
        if ptype == 'object_clean':
            return self._plan_clean(predicate)
        if ptype == 'object_hot':
            return self._plan_hot(predicate)
        if ptype == 'object_cool':
            return self._plan_cool(predicate)
        if ptype == 'object_filled':
            return self._plan_filled(predicate)
        raise BrainError('Unsupported predicate for RecoveryOracleBrain: {}'.format(ptype))

    def _plan_object_in_receptacle(self, predicate: Mapping[str, Any]) -> Optional[ToolCall]:
        env = self._env
        roles = self._task_spec['roles']
        object_role = predicate['object_role']
        receptacle_role = predicate['receptacle_role']
        source_role = 'source_container' if 'source_container' in roles else None
        target_object_id = env.role_object_ids.get(object_role)
        target_receptacle_id = env.role_object_ids.get(receptacle_role)
        source_container_id = env.role_object_ids.get(source_role) if source_role else None
        objects = env._objects_by_id()
        target_obj = objects.get(target_object_id, {}) if target_object_id else {}
        target_recep = objects.get(target_receptacle_id, {}) if target_receptacle_id else {}
        source_obj = objects.get(source_container_id, {}) if source_container_id else {}
        effective_object_role: str = object_role
        effective_object_id: str = target_object_id
        effective_obj = target_obj
        if target_obj.get('isSliced'):
            slice_children = [o for o in objects.values() if o.get('objectId', '').startswith(target_object_id + '|') and 'Sliced' in o.get('objectType', '')]
            if slice_children:
                target_slice = next((o for o in slice_children if target_receptacle_id not in (o.get('parentReceptacles') or [])), slice_children[0])
                effective_object_id = target_slice['objectId']
                effective_object_role = effective_object_id
                effective_obj = target_slice
        held_id = _held_object_id(env)
        if target_receptacle_id and held_id != effective_object_id and _object_in_receptacle(effective_obj, target_receptacle_id):
            return None
        if held_id == effective_object_id:
            if target_recep.get('openable'):
                if self._half_open(target_recep):
                    return ToolCall('open', {'target': receptacle_role}, reasoning='receptacle is half-open (jammed); retry open')
                if not target_recep.get('isOpen', True):
                    return self._nav_then('open', receptacle_role, target_recep)
            return self._nav_then('put_down', receptacle_role, target_recep)
        if held_id and held_id != target_object_id:
            agent_pos = (env.controller.last_event.metadata.get('agent') or {}).get('position') or {}
            ax = float(agent_pos.get('x', 0.0))
            az = float(agent_pos.get('z', 0.0))
            import math
            candidates = []
            for o in objects.values():
                if not o.get('receptacle'):
                    continue
                if o.get('pickupable'):
                    continue
                if o.get('openable') and (not o.get('isOpen', True)):
                    continue
                if o.get('objectId') == target_receptacle_id:
                    continue
                pos = o.get('position') or {}
                if 'x' not in pos or 'z' not in pos:
                    continue
                d = math.hypot(float(pos['x']) - ax, float(pos['z']) - az)
                candidates.append((d, o))
            candidates.sort(key=lambda c: c[0])
            if candidates:
                _, nearest_recep = candidates[0]
                return self._nav_then('put_down', nearest_recep['objectId'], nearest_recep)
            return self._nav_then('put_down', receptacle_role, target_recep)
        parent_recs = effective_obj.get('parentReceptacles') or effective_obj.get('parent_receptacles') or []
        if source_container_id and source_container_id in parent_recs:
            if source_obj.get('openable'):
                if self._half_open(source_obj):
                    return ToolCall('open', {'target': source_role}, reasoning='source container jammed half-open; retry open')
                if not source_obj.get('isOpen', True):
                    return self._nav_then('open', source_role, source_obj)
            return self._nav_then('pick_up', effective_object_role, effective_obj)
        for parent_id in parent_recs:
            parent_obj = objects.get(parent_id, {})
            if parent_obj.get('openable') and (not parent_obj.get('isOpen', True)):
                role_by_id = {v: k for k, v in env.role_object_ids.items()}
                parent_target = role_by_id.get(parent_id, parent_id)
                return self._nav_then('open', parent_target, parent_obj)
        return self._nav_then('pick_up', effective_object_role, effective_obj)

    def _plan_toggle(self, predicate: Mapping[str, Any]) -> Optional[ToolCall]:
        env = self._env
        role = predicate['object_role']
        object_id = env.role_object_ids.get(role)
        obj = env._objects_by_id().get(object_id, {}) if object_id else {}
        desired = predicate['type'] == 'object_toggled_on'
        if bool(obj.get('isToggled', False)) == desired:
            return None
        tool = 'toggle_on' if desired else 'toggle_off'
        return self._nav_then(tool, role, obj)

    def _plan_open_close(self, predicate: Mapping[str, Any]) -> Optional[ToolCall]:
        env = self._env
        role = predicate['object_role']
        object_id = env.role_object_ids.get(role)
        obj = env._objects_by_id().get(object_id, {}) if object_id else {}
        desired_open = predicate['type'] == 'object_open'
        openness = obj.get('openness')
        if desired_open:
            if self._half_open(obj):
                return ToolCall('open', {'target': role}, reasoning='half-open (jammed); retry open')
            if not bool(obj.get('isOpen', False)):
                return self._nav_then('open', role, obj)
            return None
        else:
            if self._half_open(obj):
                return ToolCall('close', {'target': role}, reasoning='half-closed (jammed); retry close')
            if bool(obj.get('isOpen', False)):
                return self._nav_then('close', role, obj)
            return None

    def _plan_sliced(self, predicate: Mapping[str, Any]) -> Optional[ToolCall]:
        env = self._env
        role = predicate['object_role']
        roles = self._task_spec['roles']
        object_id = env.role_object_ids.get(role)
        obj = env._objects_by_id().get(object_id, {}) if object_id else {}
        if bool(obj.get('isSliced', False)):
            return None
        knife_role = 'knife' if 'knife' in roles else None
        held = _held_object_id(env)
        if knife_role:
            knife_id = env.role_object_ids.get(knife_role)
            if held != knife_id:
                knife_obj = env._objects_by_id().get(knife_id, {})
                return self._nav_then('pick_up', knife_role, knife_obj)
        return self._nav_then('slice', role, obj)

    def _plan_clean(self, predicate: Mapping[str, Any]) -> Optional[ToolCall]:
        env = self._env
        role = predicate['object_role']
        roles = self._task_spec['roles']
        object_id = env.role_object_ids.get(role)
        obj = env._objects_by_id().get(object_id, {}) if object_id else {}
        if not bool(obj.get('isDirty', False)):
            cleanup = self._plan_cleanup()
            if cleanup is not None:
                return cleanup
            return None
        sink_role = 'sink_basin' if 'sink_basin' in roles else None
        faucet_role = 'faucet' if 'faucet' in roles else None
        sink_id = env.role_object_ids.get(sink_role) if sink_role else None
        faucet_id = env.role_object_ids.get(faucet_role) if faucet_role else None
        held = _held_object_id(env)
        in_sink = sink_id and held != object_id and _object_in_receptacle(obj, sink_id)
        if not in_sink:
            if held == object_id and sink_role:
                return self._nav_then('put_down', sink_role, env._objects_by_id().get(sink_id, {}))
            if held is None:
                return self._nav_then('pick_up', role, obj)
            if sink_role:
                return self._nav_then('put_down', sink_role, env._objects_by_id().get(sink_id, {}))
        if faucet_role and faucet_id:
            faucet_obj = env._objects_by_id().get(faucet_id, {})
            if not bool(faucet_obj.get('isToggled', False)):
                return self._nav_then('toggle_on', faucet_role, faucet_obj)
        return None

    def _plan_hot(self, predicate: Mapping[str, Any]) -> Optional[ToolCall]:
        env = self._env
        role = predicate['object_role']
        roles = self._task_spec['roles']
        object_id = env.role_object_ids.get(role)
        obj = env._objects_by_id().get(object_id, {}) if object_id else {}
        if bool(obj.get('isHeated', False)) or obj.get('temperature') == 'Hot':
            return None
        microwave_role = 'microwave' if 'microwave' in roles else None
        microwave_id = env.role_object_ids.get(microwave_role) if microwave_role else None
        if not microwave_role or not microwave_id:
            return None
        microwave_obj = env._objects_by_id().get(microwave_id, {})
        held = _held_object_id(env)
        in_microwave = held != object_id and _object_in_receptacle(obj, microwave_id)
        if not in_microwave:
            if not bool(microwave_obj.get('isOpen', False)):
                return self._nav_then('open', microwave_role, microwave_obj)
            if held == object_id:
                return self._nav_then('put_down', microwave_role, microwave_obj)
            return self._nav_then('pick_up', role, obj)
        if bool(microwave_obj.get('isOpen', False)):
            return self._nav_then('close', microwave_role, microwave_obj)
        if not bool(microwave_obj.get('isToggled', False)):
            return self._nav_then('toggle_on', microwave_role, microwave_obj)
        return None

    def _plan_cool(self, predicate: Mapping[str, Any]) -> Optional[ToolCall]:
        env = self._env
        role = predicate['object_role']
        roles = self._task_spec['roles']
        object_id = env.role_object_ids.get(role)
        obj = env._objects_by_id().get(object_id, {}) if object_id else {}
        cool_done = bool(obj.get('isCooled', False)) or obj.get('temperature') == 'Cold'
        if cool_done:
            return None
        fridge_role = 'fridge' if 'fridge' in roles else None
        fridge_id = env.role_object_ids.get(fridge_role) if fridge_role else None
        if not fridge_role or not fridge_id:
            return None
        fridge_obj = env._objects_by_id().get(fridge_id, {})
        held = _held_object_id(env)
        in_fridge = held != object_id and _object_in_receptacle(obj, fridge_id)
        if not in_fridge:
            if not bool(fridge_obj.get('isOpen', False)):
                return self._nav_then('open', fridge_role, fridge_obj)
            if held == object_id:
                return self._nav_then('put_down', fridge_role, fridge_obj)
            return self._nav_then('pick_up', role, obj)
        if bool(fridge_obj.get('isOpen', False)):
            return self._nav_then('close', fridge_role, fridge_obj)
        return None

    def _plan_filled(self, predicate: Mapping[str, Any]) -> Optional[ToolCall]:
        env = self._env
        role = predicate['object_role']
        roles = self._task_spec['roles']
        object_id = env.role_object_ids.get(role)
        obj = env._objects_by_id().get(object_id, {}) if object_id else {}
        if _object_filled_satisfies(obj, predicate):
            cleanup = self._plan_cleanup()
            if cleanup is not None:
                return cleanup
            return None
        sink_role = 'sink_basin' if 'sink_basin' in roles else None
        faucet_role = 'faucet' if 'faucet' in roles else None
        sink_id = env.role_object_ids.get(sink_role) if sink_role else None
        faucet_id = env.role_object_ids.get(faucet_role) if faucet_role else None
        held = _held_object_id(env)
        in_sink = sink_id and held != object_id and _object_in_receptacle(obj, sink_id)
        if not in_sink:
            if held == object_id and sink_role:
                return self._nav_then('put_down', sink_role, env._objects_by_id().get(sink_id, {}))
            if held is None:
                return self._nav_then('pick_up', role, obj)
            if sink_role:
                return self._nav_then('put_down', sink_role, env._objects_by_id().get(sink_id, {}))
        if faucet_role and faucet_id:
            faucet_obj = env._objects_by_id().get(faucet_id, {})
            if not bool(faucet_obj.get('isToggled', False)):
                return self._nav_then('toggle_on', faucet_role, faucet_obj)
        return None

    def _plan_cleanup(self) -> Optional[ToolCall]:
        if self._has_pending_faucet_effect():
            return None
        return self._plan_faucet_off_cleanup()

    def _plan_faucet_off_cleanup(self) -> Optional[ToolCall]:
        env = self._env
        task_spec = self._task_spec or {}
        roles = task_spec.get('roles', {}) if isinstance(task_spec, Mapping) else {}
        objects = env._objects_by_id()
        for role in roles:
            object_id = env.role_object_ids.get(role)
            obj = objects.get(object_id, {}) if object_id else {}
            if obj.get('objectType') != 'Faucet':
                continue
            if not bool(obj.get('isToggled', False)):
                continue
            if self._goal_requires_toggled_on(object_id):
                continue
            return self._nav_then('toggle_off', role, obj)
        return None

    def _goal_requires_toggled_on(self, object_id: Optional[str]) -> bool:
        if not object_id:
            return False
        env = self._env
        task_spec = self._task_spec or {}
        goal = task_spec.get('goal', {}) if isinstance(task_spec, Mapping) else {}
        for predicate in goal.get('predicates', []):
            if not isinstance(predicate, Mapping):
                continue
            if predicate.get('type') != 'object_toggled_on':
                continue
            role = predicate.get('object_role')
            if env.role_object_ids.get(role) == object_id:
                return True
        return False

    def _has_pending_faucet_effect(self) -> bool:
        env = self._env
        task_spec = self._task_spec or {}
        goal = task_spec.get('goal', {}) if isinstance(task_spec, Mapping) else {}
        objects = env._objects_by_id()
        for predicate in goal.get('predicates', []):
            if not isinstance(predicate, Mapping):
                continue
            ptype = predicate.get('type')
            if ptype not in {'object_clean', 'object_filled'}:
                continue
            role = predicate.get('object_role')
            object_id = env.role_object_ids.get(role)
            obj = objects.get(object_id, {}) if object_id else {}
            if ptype == 'object_clean' and bool(obj.get('isDirty', False)):
                return True
            if ptype == 'object_filled' and (not _object_filled_satisfies(obj, predicate)):
                return True
        return False

    def _nav_then(self, tool: str, role: str, obj: Mapping[str, Any]) -> ToolCall:
        if self._needs_navigation_retry(role):
            return ToolCall('navigate_to', {'target': role}, reasoning='retrying navigation to {} after a failed attempt'.format(role))
        ready_target = getattr(self._env, '_ready_target', None) or {}
        ready_for_role = False
        obj_id = obj.get('objectId') if obj else None
        if ready_target and ready_target.get('object_id') == obj_id:
            ready_for_role = bool(ready_target.get('interaction_ready', False))
        if tool == 'put_down' and (not ready_for_role):
            return ToolCall('navigate_to', {'target': role}, reasoning='navigating to {} before put_down'.format(role))
        if not obj.get('visible', False) and (not ready_for_role):
            return ToolCall('navigate_to', {'target': role}, reasoning='target not visible; navigating to {}'.format(role))
        return ToolCall(tool, {'target': role})

    def _needs_navigation_retry(self, target: str) -> bool:
        if not target:
            return False
        return self._failed_call_counts.get(('navigate_to', str(target)), 0) > 0

    def _half_open(self, obj: Mapping[str, Any]) -> bool:
        if not obj or not obj.get('openable'):
            return False
        openness = obj.get('openness')
        if openness is None:
            return False
        try:
            v = float(openness)
        except (TypeError, ValueError):
            return False
        return 0.05 < v < 0.95 and v not in (0.0, 1.0)

def _object_in_receptacle(obj: Mapping[str, Any], receptacle_id: str) -> bool:
    parent_recs = obj.get('parentReceptacles') or obj.get('parent_receptacles') or []
    return any((receptacle_id == p for p in parent_recs))

def _object_filled_satisfies(obj: Mapping[str, Any], predicate: Mapping[str, Any]) -> bool:
    if not bool(obj.get('isFilledWithLiquid', False)):
        return False
    wanted_liquid = predicate.get('liquid')
    if wanted_liquid is None:
        return True
    return str(obj.get('fillLiquid', '') or '').lower() == str(wanted_liquid).lower()

def _held_object_id(env: Any) -> Optional[str]:
    meta = env.controller.last_event.metadata
    inv = meta.get('inventoryObjects') or []
    for entry in inv:
        if isinstance(entry, Mapping) and entry.get('objectId'):
            return entry['objectId']
    return None

def _call_signature(tool_name: str, arguments: Mapping[str, Any]) -> Tuple[str, str]:
    return (str(tool_name), str(arguments.get('target', '')))
__all__ = ['RecoveryOracleBrain']
