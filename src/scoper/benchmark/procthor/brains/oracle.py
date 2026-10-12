from __future__ import annotations
from typing import Any, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class OracleBrain(BaseBrain):

    def __init__(self, max_replans: Optional[int]=None, *, harness_definition: Any=None) -> None:
        self._queue: List[ToolCall] = []
        self._env: Any = None
        self._task_spec: Mapping[str, Any] = {}
        self._replan_count = 0
        self._max_replans = max_replans
        self.harness_definition = harness_definition

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        self._task_spec = task_spec
        self._replan_count = 0
        self._queue = list(_build_tool_sequence(env, task_spec))
        if self._max_replans is None:
            num_predicates = len(task_spec.get('goal', {}).get('predicates', []))
            self._max_replans = max(num_predicates * 3, 9)

    def act(self, brain_input: BrainInput) -> ToolCall:
        last_transition = brain_input.last_transition
        if last_transition is not None:
            success = last_transition.get('success', True)
            if not success:
                if self._replan_count < self._max_replans:
                    self._replan_count += 1
                    self._queue = list(_build_tool_sequence(self._env, self._task_spec))
                else:
                    return ToolCall('stop', {}, reasoning=f'replan_budget_exceeded ({self._replan_count} replans)')
        if self._queue:
            return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='oracle plan complete')

def _role_needs_open(env: Any, role_name: str) -> bool:
    object_id = env.role_object_ids.get(role_name)
    if object_id is None:
        return False
    obj = env._objects_by_id().get(object_id)
    return bool(obj and obj.get('openable', False) and (obj.get('isOpen') is False))

def _find_closed_parent_role(env: Any, object_role: str, roles: Mapping[str, Any]) -> Optional[str]:
    obj_id = env.role_object_ids.get(object_role)
    if obj_id is None:
        return None
    obj = env._objects_by_id().get(obj_id, {})
    parent_ids = obj.get('parentReceptacles') or obj.get('parent_receptacles') or []
    if not parent_ids:
        return None
    role_by_id = {v: k for k, v in env.role_object_ids.items()}
    for parent_id in parent_ids:
        parent_obj = env._objects_by_id().get(parent_id, {})
        if not (parent_obj.get('openable') and parent_obj.get('isOpen') is False):
            continue
        parent_role = role_by_id.get(parent_id)
        if parent_role:
            return parent_role
    return None

def _ensure_target_accessible(env: Any, object_role: str, roles: Mapping[str, Any], calls: List[ToolCall]) -> None:
    parent_role = _find_closed_parent_role(env, object_role, roles)
    if parent_role:
        calls.append(ToolCall('navigate_to', {'target': parent_role}))
        calls.append(ToolCall('open', {'target': parent_role}))

def _build_tool_sequence(env: Any, task_spec: Mapping[str, Any]) -> List[ToolCall]:
    calls: List[ToolCall] = []
    roles = task_spec['roles']
    predicates = task_spec['goal']['predicates']
    for predicate in predicates:
        predicate_type = predicate.get('type')
        if predicate_type == 'object_in_receptacle':
            object_role = predicate['object_role']
            receptacle_role = predicate['receptacle_role']
            source_role = 'source_container' if 'source_container' in roles else None
            if source_role and _role_needs_open(env, source_role):
                calls.append(ToolCall('navigate_to', {'target': source_role}))
                calls.append(ToolCall('open', {'target': source_role}))
            calls.append(ToolCall('navigate_to', {'target': object_role}))
            calls.append(ToolCall('pick_up', {'target': object_role}))
            if _role_needs_open(env, receptacle_role):
                calls.append(ToolCall('navigate_to', {'target': receptacle_role}))
                calls.append(ToolCall('open', {'target': receptacle_role}))
            calls.append(ToolCall('navigate_to', {'target': receptacle_role}))
            calls.append(ToolCall('put_down', {'target': receptacle_role}))
        elif predicate_type in {'object_toggled_on', 'object_toggled_off'}:
            role = predicate['object_role']
            tool_name = 'toggle_on' if predicate_type == 'object_toggled_on' else 'toggle_off'
            calls.append(ToolCall('navigate_to', {'target': role}))
            calls.append(ToolCall(tool_name, {'target': role}))
        elif predicate_type in {'object_open', 'object_closed'}:
            role = predicate['object_role']
            tool_name = 'open' if predicate_type == 'object_open' else 'close'
            calls.append(ToolCall('navigate_to', {'target': role}))
            calls.append(ToolCall(tool_name, {'target': role}))
        elif predicate_type == 'object_sliced':
            role = predicate['object_role']
            knife_role = 'knife' if 'knife' in roles else None
            if knife_role:
                calls.append(ToolCall('navigate_to', {'target': knife_role}))
                calls.append(ToolCall('pick_up', {'target': knife_role}))
            _ensure_target_accessible(env, role, roles, calls)
            calls.append(ToolCall('navigate_to', {'target': role}))
            calls.append(ToolCall('slice', {'target': role}))
            has_follow_up_manipulation = any((p.get('type') in {'object_in_receptacle', 'object_clean', 'object_hot', 'object_cool', 'object_filled'} and p.get('object_role') == role for p in predicates))
            if knife_role and has_follow_up_manipulation:
                drop_target = knife_role
                calls.append(ToolCall('navigate_to', {'target': drop_target}))
                calls.append(ToolCall('put_down', {'target': drop_target}))
        elif predicate_type == 'object_clean':
            role = predicate['object_role']
            sink_role = 'sink_basin' if 'sink_basin' in roles else None
            faucet_role = 'faucet' if 'faucet' in roles else None
            _ensure_target_accessible(env, role, roles, calls)
            calls.append(ToolCall('navigate_to', {'target': role}))
            calls.append(ToolCall('pick_up', {'target': role}))
            if sink_role:
                calls.append(ToolCall('navigate_to', {'target': sink_role}))
                calls.append(ToolCall('put_down', {'target': sink_role}))
            if faucet_role:
                calls.append(ToolCall('navigate_to', {'target': faucet_role}))
                calls.append(ToolCall('toggle_on', {'target': faucet_role}))
                calls.append(ToolCall('toggle_off', {'target': faucet_role}))
        elif predicate_type == 'object_hot':
            role = predicate['object_role']
            microwave_role = 'microwave' if 'microwave' in roles else None
            _ensure_target_accessible(env, role, roles, calls)
            calls.append(ToolCall('navigate_to', {'target': role}))
            calls.append(ToolCall('pick_up', {'target': role}))
            if microwave_role:
                calls.append(ToolCall('navigate_to', {'target': microwave_role}))
                calls.append(ToolCall('open', {'target': microwave_role}))
                calls.append(ToolCall('put_down', {'target': microwave_role}))
                calls.append(ToolCall('close', {'target': microwave_role}))
                calls.append(ToolCall('toggle_on', {'target': microwave_role}))
                calls.append(ToolCall('toggle_off', {'target': microwave_role}))
                calls.append(ToolCall('open', {'target': microwave_role}))
                calls.append(ToolCall('pick_up', {'target': role}))
        elif predicate_type == 'object_cool':
            role = predicate['object_role']
            fridge_role = 'fridge' if 'fridge' in roles else None
            _ensure_target_accessible(env, role, roles, calls)
            calls.append(ToolCall('navigate_to', {'target': role}))
            calls.append(ToolCall('pick_up', {'target': role}))
            if fridge_role:
                calls.append(ToolCall('navigate_to', {'target': fridge_role}))
                calls.append(ToolCall('open', {'target': fridge_role}))
                calls.append(ToolCall('put_down', {'target': fridge_role}))
                calls.append(ToolCall('close', {'target': fridge_role}))
                calls.append(ToolCall('open', {'target': fridge_role}))
                calls.append(ToolCall('pick_up', {'target': role}))
        elif predicate_type == 'object_filled':
            role = predicate['object_role']
            sink_role = 'sink_basin' if 'sink_basin' in roles else None
            faucet_role = 'faucet' if 'faucet' in roles else None
            _ensure_target_accessible(env, role, roles, calls)
            calls.append(ToolCall('navigate_to', {'target': role}))
            calls.append(ToolCall('pick_up', {'target': role}))
            if sink_role:
                calls.append(ToolCall('navigate_to', {'target': sink_role}))
                calls.append(ToolCall('put_down', {'target': sink_role}))
            if faucet_role:
                calls.append(ToolCall('navigate_to', {'target': faucet_role}))
                calls.append(ToolCall('toggle_on', {'target': faucet_role}))
                calls.append(ToolCall('toggle_off', {'target': faucet_role}))
        else:
            raise BrainError('Unsupported predicate for OracleBrain: {}'.format(predicate_type))
    if bool(task_spec['goal'].get('require_model_stop', False)):
        calls.append(ToolCall('stop', {}, reasoning='task requires model stop'))
    return calls
__all__ = ['OracleBrain']
