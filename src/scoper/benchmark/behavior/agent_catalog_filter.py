from __future__ import annotations
import math
import re
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

class AgentCatalogTracker:

    def __init__(self) -> None:
        self._object_to_alias: dict[str, str] = {}
        self._alias_to_object: dict[str, str] = {}
        self._type_counts: defaultdict[str, int] = defaultdict(int)
        self._entries_by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def reset(self) -> None:
        self._object_to_alias.clear()
        self._alias_to_object.clear()
        self._type_counts.clear()
        self._entries_by_type.clear()

    def alias_for(self, object_id: str, object_type: Optional[str]=None) -> str:
        object_id = str(object_id)
        if object_id in self._object_to_alias:
            return self._object_to_alias[object_id]
        base = _safe_alias_base(object_type or object_id or 'object')
        self._type_counts[base] += 1
        alias = '{}_{}'.format(base, self._type_counts[base])
        while alias in self._alias_to_object:
            self._type_counts[base] += 1
            alias = '{}_{}'.format(base, self._type_counts[base])
        self._object_to_alias[object_id] = alias
        self._alias_to_object[alias] = object_id
        return alias

    def aliases(self) -> list[str]:
        return sorted(self._alias_to_object.keys())

    def resolve(self, alias_or_object_id: Any) -> str:
        text = str(alias_or_object_id)
        return self._alias_to_object.get(text, text)

    def entries_for_type(self, object_type: str) -> list[dict[str, Any]]:
        return [dict(item) for item in self._entries_by_type.get(str(object_type), [])]

    def record_entries(self, entries: Sequence[Mapping[str, Any]]) -> None:
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            object_type = str(entry.get('objectType') or '')
            if object_type:
                grouped[object_type].append(dict(entry))
        self._entries_by_type = grouped

def build_prompt_safe_catalog(*, objects: Mapping[str, Any], pose: Mapping[str, Any], visible_object_ids: Sequence[str], tracker: AgentCatalogTracker, include_raw_ids: bool=False, include_states: bool=False, include_bbox: bool=False) -> list[dict[str, Any]]:
    visible = {str(item) for item in visible_object_ids}
    entries: list[dict[str, Any]] = []
    for object_id in sorted((str(name) for name in objects.keys())):
        if object_id not in visible:
            continue
        obj = objects.get(object_id)
        if obj is None:
            continue
        object_type = str(getattr(obj, 'category', '') or 'object')
        alias = tracker.alias_for(object_id, object_type)
        abilities = set((str(item) for item in getattr(obj, 'abilities', ()) or ()))
        states = getattr(obj, 'states', {}) or {}
        center = tuple((float(v) for v in getattr(obj, 'center', (0.0, 0.0, 0.0))))
        entry = {'objectType': object_type, 'alias': alias, 'position': {'x': center[0], 'y': center[1], 'z': center[2]}, 'distance': _distance(pose, center), 'room_id': str(states.get('room_id') or 'unknown'), 'pickupable': 'pickupable' in abilities, 'receptacle': bool({'receptacle', 'support_surface', 'container'} & abilities), 'openable': 'openable' in abilities or 'open' in states, 'toggleable': 'toggleable' in abilities or 'toggled_on' in states, 'sliceable': bool({'sliceable', 'cuttable'} & abilities)}
        if include_raw_ids:
            entry['objectId'] = object_id
            entry['name'] = object_id
        if include_states:
            entry['states'] = dict(states)
        if include_bbox:
            extent = tuple((float(v) for v in getattr(obj, 'extent', (0.0, 0.0, 0.0))))
            entry['aabb_center'] = [center[0], center[1], center[2]]
            entry['aabb_extent'] = [extent[0], extent[1], extent[2]]
        entries.append(entry)
    tracker.record_entries(entries)
    return entries

def build_prompt_target_handles(*, role_map: Mapping[str, str], visible_object_ids: Sequence[str], tracker: AgentCatalogTracker) -> dict[str, Any]:
    visible = {str(item) for item in visible_object_ids}
    role_aliases = {}
    for role_name, object_id in sorted(role_map.items()):
        object_id = str(object_id)
        if object_id in visible:
            role_aliases[str(role_name)] = tracker.alias_for(object_id)
    return {'role_handles': sorted((str(role) for role in role_map.keys())), 'role_object_aliases': role_aliases, 'object_aliases': tracker.aliases()}

def prompt_safe_tool_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = ('tool_name', 'arguments', 'high_level_step', 'success', 'feedback', 'env_feedback', 'done', 'done_reason', 'num_low_level_steps', 'binding')
    safe: list[dict[str, Any]] = []
    for item in history or []:
        safe.append({key: item.get(key) for key in keys if key in item})
    return safe

def _safe_alias_base(value: str) -> str:
    base = re.sub('[^a-zA-Z0-9]+', '_', str(value).lower()).strip('_')
    return base or 'object'

def _distance(pose: Mapping[str, Any], center: Sequence[float]) -> float:
    px = float(pose.get('x', 0.0))
    py = float(pose.get('y', 0.0))
    pz = float(pose.get('z', 0.0))
    return float(math.sqrt((center[0] - px) ** 2 + (center[1] - py) ** 2 + (center[2] - pz) ** 2))
__all__ = ['AgentCatalogTracker', 'build_prompt_safe_catalog', 'build_prompt_target_handles', 'prompt_safe_tool_history']
