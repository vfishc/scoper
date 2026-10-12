from __future__ import annotations
from typing import Any, List, Mapping
DEFAULT_SUBSTANCE = 'water'
_BOOL_STATE_WRITES: dict[str, tuple[str, bool]] = {'open': ('open', True), 'closed': ('open', False), 'toggled_on': ('toggled_on', True), 'toggled_off': ('toggled_on', False), 'dirty': ('dirty', True), 'clean': ('dirty', False), 'sliced': ('sliced', True), 'cooled': ('cooled', True)}
_SUBSTANCE_STATE_WRITES = {'filled': 'filled', 'saturated': 'saturated'}

def apply_init_states(objects_by_name: Mapping[str, Any], init: List[Mapping[str, Any]], role_map: Mapping[str, str]) -> List[dict]:
    applied: List[dict] = []
    for entry in init:
        predicate = entry['predicate']
        if predicate not in _BOOL_STATE_WRITES and predicate not in _SUBSTANCE_STATE_WRITES:
            continue
        role = entry['args'][0]
        obj_name = role_map.get(role, role)
        obj = objects_by_name.get(obj_name)
        if obj is None:
            continue
        if obj.states is None:
            obj.states = {}
        if predicate in _BOOL_STATE_WRITES:
            key, value = _BOOL_STATE_WRITES[predicate]
            obj.states[key] = value
            applied.append({'object': obj_name, 'state': key, 'value': value})
        else:
            key = _SUBSTANCE_STATE_WRITES[predicate]
            substance = _substance_arg(entry)
            obj.states[key] = {substance: True}
            applied.append({'object': obj_name, 'state': key, 'substance': substance})
    return applied

def apply_role_init_states(objects_by_name: Mapping[str, Any], roles: Mapping[str, Any], role_map: Mapping[str, str]) -> List[dict]:
    applied: List[dict] = []
    for role_name, role in roles.items():
        init_states = role.get('init_states') if isinstance(role, Mapping) else None
        if not isinstance(init_states, Mapping):
            continue
        obj = objects_by_name.get(role_map.get(role_name, role_name))
        if obj is None:
            continue
        if obj.states is None:
            obj.states = {}
        for key, value in init_states.items():
            obj.states[key] = value
            applied.append({'object': obj.name, 'state': key, 'value': value})
    return applied

def _substance_arg(entry: Mapping[str, Any]) -> str:
    args = entry.get('args', [])
    if len(args) >= 2 and isinstance(args[1], str):
        return args[1]
    return DEFAULT_SUBSTANCE
__all__ = ['apply_init_states', 'apply_role_init_states', 'DEFAULT_SUBSTANCE']
