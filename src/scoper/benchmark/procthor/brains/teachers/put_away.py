from __future__ import annotations
from typing import Any, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainInput, ToolCall

class PutAwayTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: List[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        put_away = task_spec['family_spec']['put_away']
        target_roles = list(put_away['target_roles'])
        storage_role = put_away['storage_roles'][0]
        roles = task_spec['roles']
        calls: List[ToolCall] = []
        storage_openable = _role_is_openable(env, storage_role)
        for target_role in target_roles:
            _ensure_target_accessible(env, target_role, roles, calls)
            calls.extend([_call('navigate_to', target_role), _call('pick_up', target_role), _call('navigate_to', storage_role)])
            if storage_openable:
                calls.append(_call('open', storage_role))
            calls.append(_call('put_down', storage_role))
            if storage_openable:
                calls.append(_call('close', storage_role))
        calls.append(ToolCall('stop', {}, reasoning='put_away plan complete'))
        self._queue = calls

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='put_away plan exhausted')

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

def _preferred_staging_target(env: Any, storage_role: str, excluded_roles: List[str]) -> Optional[str]:
    storage_id = _role_object_id(env, storage_role)
    objects = env._objects_by_id()
    if storage_id and '___' in storage_id:
        parent_id = storage_id.split('___', 1)[0]
        parent = objects.get(parent_id, {})
        if parent.get('receptacle') and (not parent.get('pickupable')):
            return parent_id
    storage = objects.get(storage_id, {}) if storage_id else {}
    storage_position = storage.get('position') or {}
    excluded_ids = {object_id for role, object_id in getattr(env, 'role_object_ids', {}).items() if role in set(excluded_roles + [storage_role])}
    candidates = []
    for object_id, obj in objects.items():
        if object_id in excluded_ids:
            continue
        if obj.get('pickupable') or not obj.get('receptacle'):
            continue
        if obj.get('openable'):
            continue
        position = obj.get('position') or {}
        candidates.append((_xz_distance(position, storage_position), object_id))
    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None

def _xz_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax = float(a.get('x', 0.0))
    az = float(a.get('z', 0.0))
    bx = float(b.get('x', 0.0))
    bz = float(b.get('z', 0.0))
    return ((ax - bx) ** 2 + (az - bz) ** 2) ** 0.5

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
get_dispatcher().register('put_away_by_category', PutAwayTeacher)
get_dispatcher().register('put_away_visual_judgment', PutAwayTeacher)
get_dispatcher().register('put_away_cross_room', PutAwayTeacher)
__all__ = ['PutAwayTeacher']
