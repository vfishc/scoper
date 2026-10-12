from __future__ import annotations
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence
AGENT_CATALOG_FIELDS = ('objectId', 'objectType', 'alias', 'position', 'distance', 'room_id', 'pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable')
HIDDEN_AGENT_CATALOG_FIELDS = frozenset({'parentReceptacles', 'receptacleObjectIds', 'isOpen', 'openness', 'isToggled', 'isSliced', 'isDirty', 'isHeated', 'isCooled', 'temperature', 'isFilledWithLiquid', 'fillLiquid', 'visible'})

class AgentCatalogTracker:

    def __init__(self) -> None:
        self.entries_by_object_id: Dict[str, dict] = {}
        self.alias_to_object_id: Dict[str, str] = {}
        self._next_index_by_type: Dict[str, int] = {}

    def reset(self) -> None:
        self.entries_by_object_id.clear()
        self.alias_to_object_id.clear()
        self._next_index_by_type.clear()

    def aliases(self) -> List[str]:
        return [entry['alias'] for entry in sorted(self.entries_by_object_id.values(), key=lambda item: (str(item.get('object_type')), str(item.get('alias'))))]

    def resolve(self, target: str) -> Optional[dict]:
        if target in self.entries_by_object_id:
            return self.entries_by_object_id[target]
        object_id = self.alias_to_object_id.get(target)
        if object_id:
            return self.entries_by_object_id.get(object_id)
        return None

    def entries_for_type(self, object_type: str) -> List[dict]:
        lowered = object_type.lower()
        return [entry for entry in self.entries_by_object_id.values() if str(entry.get('object_type') or '').lower() == lowered]

    def update_visible_object(self, obj: Mapping[str, Any], *, high_level_step: int, alias: Optional[str]=None) -> dict:
        object_id = _object_id(obj)
        object_type = _object_type(obj)
        entry = self.entries_by_object_id.get(object_id)
        if entry is None:
            resolved_alias = alias or self._assign_alias(object_id, object_type)
            entry = {'alias': resolved_alias, 'object_type': object_type, 'first_seen_step': int(high_level_step), 'last_seen_step': int(high_level_step), 'last_known_position': {}, 'last_known_room_id': None}
            self.entries_by_object_id[object_id] = entry
            self.alias_to_object_id[resolved_alias] = object_id
        else:
            entry['last_seen_step'] = int(high_level_step)
            if alias and alias != entry.get('alias'):
                old_alias = entry.get('alias')
                if old_alias in self.alias_to_object_id:
                    del self.alias_to_object_id[old_alias]
                entry['alias'] = alias
                self.alias_to_object_id[alias] = object_id
            entry['object_type'] = object_type
        entry['last_known_position'] = dict(obj.get('position') or {})
        entry['last_known_room_id'] = _room_id(obj)
        return entry

    def _assign_alias(self, object_id: str, object_type: str) -> str:
        next_index = self._next_index_by_type.get(object_type, 0) + 1
        while True:
            alias = object_type if next_index == 1 else '{}_{}'.format(object_type, next_index)
            if alias not in self.alias_to_object_id:
                break
            next_index += 1
        self._next_index_by_type[object_type] = next_index
        self.alias_to_object_id[alias] = object_id
        return alias

def build_agent_catalog_view(metadata: Mapping[str, Any], *, tracker: AgentCatalogTracker, high_level_step: int, alias_by_object_id: Optional[Mapping[str, str]]=None) -> List[dict]:
    objects_by_id = _objects_by_id(metadata)
    blocked = _objects_blocked_by_closed_container(objects_by_id)
    entries: List[dict] = []
    for object_id, obj in sorted(objects_by_id.items()):
        if object_id in blocked:
            continue
        if not bool(obj.get('visible', False)):
            continue
        tracker_entry = tracker.update_visible_object(obj, high_level_step=high_level_step, alias=(alias_by_object_id or {}).get(object_id))
        entries.append(_agent_entry(obj, tracker_entry))
    return entries

def build_inventory_view(metadata: Mapping[str, Any]) -> List[dict]:
    inventory = []
    objects_by_id = _objects_by_id(metadata)
    for item in metadata.get('inventoryObjects', []) or []:
        if not isinstance(item, Mapping):
            continue
        object_id = item.get('objectId')
        if not isinstance(object_id, str) or not object_id:
            continue
        inventory.append({'objectId': object_id, 'objectType': _object_type(item, object_id)})
    seen_object_ids = {item['objectId'] for item in inventory}
    arm = metadata.get('arm')
    if isinstance(arm, Mapping):
        for item in arm.get('heldObjects', []) or []:
            if isinstance(item, Mapping):
                object_id = item.get('objectId')
            else:
                object_id = item
            if not isinstance(object_id, str) or not object_id:
                continue
            if object_id in seen_object_ids:
                continue
            obj = objects_by_id.get(object_id, {})
            inventory.append({'objectId': object_id, 'objectType': _object_type(obj, object_id)})
            seen_object_ids.add(object_id)
    return inventory

def _agent_entry(obj: Mapping[str, Any], tracker_entry: Mapping[str, Any]) -> dict:
    entry = {'objectId': _object_id(obj), 'objectType': _object_type(obj), 'alias': tracker_entry['alias'], 'position': dict(obj.get('position') or {}), 'distance': _optional_float(obj.get('distance')), 'room_id': _room_id(obj), 'pickupable': bool(obj.get('pickupable', False)), 'receptacle': bool(obj.get('receptacle', False)), 'openable': bool(obj.get('openable', False)), 'toggleable': bool(obj.get('toggleable', False)), 'sliceable': bool(obj.get('sliceable', False))}
    return {key: entry[key] for key in AGENT_CATALOG_FIELDS}

def _objects_blocked_by_closed_container(objects_by_id: Mapping[str, Mapping[str, Any]]) -> set[str]:
    blocked = set()
    for object_id, obj in objects_by_id.items():
        if _has_closed_openable_parent(obj, objects_by_id, seen=set()):
            blocked.add(object_id)
    return blocked

def _has_closed_openable_parent(obj: Mapping[str, Any], objects_by_id: Mapping[str, Mapping[str, Any]], seen: set[str]) -> bool:
    for parent_id in obj.get('parentReceptacles') or []:
        if not isinstance(parent_id, str) or parent_id in seen:
            continue
        seen.add(parent_id)
        parent = objects_by_id.get(parent_id)
        if parent is None:
            continue
        if bool(parent.get('openable', False)) and parent.get('isOpen') is False:
            return True
        if _has_closed_openable_parent(parent, objects_by_id, seen):
            return True
    return False

def _objects_by_id(metadata: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {obj['objectId']: obj for obj in metadata.get('objects', []) or [] if isinstance(obj, Mapping) and isinstance(obj.get('objectId'), str) and bool(obj.get('objectId'))}

def _object_id(obj: Mapping[str, Any]) -> str:
    object_id = obj.get('objectId')
    if not isinstance(object_id, str) or not object_id:
        raise ValueError('Object metadata is missing objectId.')
    return object_id

def _object_type(obj: Mapping[str, Any], object_id: Optional[str]=None) -> str:
    object_type = obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    return (object_id or _object_id(obj)).split('|', 1)[0]

def _room_id(obj: Mapping[str, Any]) -> Optional[str]:
    room_id = obj.get('roomId') or obj.get('room_id')
    return room_id if isinstance(room_id, str) and room_id else None

def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
__all__ = ['AGENT_CATALOG_FIELDS', 'HIDDEN_AGENT_CATALOG_FIELDS', 'AgentCatalogTracker', 'build_agent_catalog_view', 'build_inventory_view']
