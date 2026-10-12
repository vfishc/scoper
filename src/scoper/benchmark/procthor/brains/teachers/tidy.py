from __future__ import annotations
from typing import Any, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class TidyTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: List[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        violations = list(task_spec['family_spec']['tidy']['violations'])
        ordered = sorted(enumerate(violations), key=lambda item: (0 if item[1].get('urgency') == 'immediate' else 1, item[0]))
        calls: List[ToolCall] = []
        for _, violation in ordered:
            calls.extend(self._repair_plan(env, task_spec, violation))
        calls.append(ToolCall('stop', {}, reasoning='tidy plan complete'))
        self._queue = calls

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='tidy plan exhausted')

    def _repair_plan(self, env: Any, task_spec: Mapping[str, Any], violation: Mapping[str, Any]) -> List[ToolCall]:
        v_type = violation['type']
        object_role = violation['object_role']
        if v_type == 'faucet_on':
            return [_call('navigate_to', object_role), _call('toggle_off', object_role)]
        if v_type == 'open_container':
            return [_call('navigate_to', object_role), _call('close', object_role)]
        if v_type == 'dirty_dish':
            return self._wash_and_store(env, task_spec, violation)
        raise BrainError("Unknown tidy violation type '{}'.".format(v_type))

    def _wash_and_store(self, env: Any, task_spec: Mapping[str, Any], violation: Mapping[str, Any]) -> List[ToolCall]:
        object_role = violation['object_role']
        support = violation['support_roles']
        sink_role = support['sink']
        faucet_role = support['faucet']
        storage_role = support['storage']
        calls: List[ToolCall] = []
        _ensure_target_accessible(env, object_role, task_spec['roles'], calls)
        calls.extend([_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role), _call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', storage_role)])
        if _role_is_openable(env, storage_role):
            calls.append(_call('open', storage_role))
        calls.append(_call('put_down', storage_role))
        if _role_is_openable(env, storage_role):
            calls.append(_call('close', storage_role))
        return calls

def _call(tool_name: str, target: str) -> ToolCall:
    return ToolCall(tool_name, {'target': target})

def _role_object_id(env: Any, role_name: str) -> Optional[str]:
    return getattr(env, 'role_object_ids', {}).get(role_name)

def _object_by_role(env: Any, role_name: str) -> Mapping[str, Any]:
    object_id = _role_object_id(env, role_name)
    if object_id is None:
        return {}
    return env._objects_by_id().get(object_id, {})

def _role_is_openable(env: Any, role_name: str) -> bool:
    return bool(_object_by_role(env, role_name).get('openable', False))

def _find_closed_parent_role(env: Any, object_role: str, roles: Mapping[str, Any]) -> Optional[str]:
    object_id = _role_object_id(env, object_role)
    if object_id is None:
        return None
    objects = env._objects_by_id()
    obj = objects.get(object_id, {})
    parents = obj.get('parentReceptacles') or obj.get('parent_receptacles') or []
    role_by_id = {object_id: role for role, object_id in env.role_object_ids.items()}
    for parent_id in parents:
        parent = objects.get(parent_id, {})
        if not (parent.get('openable') and parent.get('isOpen') is False):
            continue
        parent_role = role_by_id.get(parent_id)
        if parent_role in roles:
            return parent_role
        return parent_id
    return None

def _ensure_target_accessible(env: Any, object_role: str, roles: Mapping[str, Any], calls: List[ToolCall]) -> None:
    parent = _find_closed_parent_role(env, object_role, roles)
    if parent:
        calls.append(_call('navigate_to', parent))
        calls.append(_call('open', parent))
from scoper.benchmark.procthor.brains.teachers import get_dispatcher
get_dispatcher().register('tidy_by_policy', TidyTeacher)
__all__ = ['TidyTeacher']
