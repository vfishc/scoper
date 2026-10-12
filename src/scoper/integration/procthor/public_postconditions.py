from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.common import PublicActionState, PublicPropertyValue, PublicRelationValue
_INSIDE_TYPES = frozenset({'bathtubbasin', 'bowl', 'box', 'cabinet', 'cup', 'drawer', 'fridge', 'garbagecan', 'microwave', 'mug', 'pan', 'pot', 'safe', 'sink', 'sinkbasin', 'toilet'})
_ON_TOP_TYPES = frozenset({'armchair', 'bed', 'chair', 'coffeetable', 'countertop', 'desk', 'diningtable', 'dresser', 'ottoman', 'shelf', 'shelvingunit', 'sidetable', 'sofa', 'stool', 'tvstand'})

def _metadata(env: Any) -> Mapping[str, Any]:
    builder = getattr(env, '_metadata_with_sticky_object_states', None)
    if callable(builder):
        value = builder()
    else:
        value = getattr(getattr(env, 'controller', None), 'last_event', None)
        value = getattr(value, 'metadata', {})
    return value if isinstance(value, Mapping) else {}

def _alias_by_object_id(env: Any) -> dict[str, str]:
    aliases: dict[str, str] = {}
    tracker = getattr(env, '_agent_catalog_tracker', None)
    entries = getattr(tracker, 'entries_by_object_id', {})
    if isinstance(entries, Mapping):
        for object_id, entry in entries.items():
            alias = entry.get('alias') if isinstance(entry, Mapping) else None
            if isinstance(object_id, str) and isinstance(alias, str) and alias:
                aliases[object_id] = alias
    catalog = getattr(env, 'object_catalog', None)
    for entry in getattr(catalog, 'entries', ()) or ():
        object_id = getattr(entry, 'object_id', None)
        alias = getattr(entry, 'alias', None)
        if isinstance(object_id, str) and isinstance(alias, str) and alias:
            aliases.setdefault(object_id, alias)
    return aliases

def capture_public_action_state(env: Any) -> Mapping[str, Any]:
    metadata = _metadata(env)
    objects = {item['objectId']: item for item in metadata.get('objects', ()) or () if isinstance(item, Mapping) and isinstance(item.get('objectId'), str)}
    aliases = _alias_by_object_id(env)
    visible_ids = {object_id for object_id, item in objects.items() if item.get('visible') is True and object_id in aliases}
    held_ids = []
    for item in metadata.get('inventoryObjects', ()) or ():
        object_id = item.get('objectId') if isinstance(item, Mapping) else None
        if isinstance(object_id, str):
            held_ids.append(object_id)
    arm = metadata.get('arm')
    if isinstance(arm, Mapping):
        for item in arm.get('heldObjects', ()) or ():
            object_id = item.get('objectId') if isinstance(item, Mapping) else item
            if isinstance(object_id, str):
                held_ids.append(object_id)
    held_refs = tuple(sorted({aliases[object_id] for object_id in held_ids if object_id in aliases}, key=lambda value: value.encode('utf-8')))
    properties = []
    for object_id in sorted(visible_ids, key=lambda value: aliases[value].encode('utf-8')):
        item = objects[object_id]
        public_ref = aliases[object_id]
        for metadata_name, public_name in (('isOpen', 'open'), ('isToggled', 'toggled_on'), ('isSliced', 'sliced'), ('isDirty', 'dirty'), ('isHeated', 'heated'), ('isCooled', 'cooled'), ('isFilledWithLiquid', 'filled')):
            value = item.get(metadata_name)
            if isinstance(value, bool):
                properties.append(PublicPropertyValue(public_ref, public_name, value))
    relations = []
    for subject_id in visible_ids:
        subject_ref = aliases[subject_id]
        for parent_id in objects[subject_id].get('parentReceptacles', ()) or ():
            if parent_id not in aliases:
                continue
            object_type = str(objects.get(parent_id, {}).get('objectType') or '').casefold()
            relation = 'inside' if object_type in _INSIDE_TYPES else 'on_top' if object_type in _ON_TOP_TYPES else None
            if relation is not None:
                relations.append(PublicRelationValue(subject_ref, relation, aliases[parent_id]))
    observed_refs = tuple(sorted((aliases[object_id] for object_id in visible_ids), key=lambda value: value.encode('utf-8')))
    state = PublicActionState(True, held_refs, observed_refs, observed_refs, tuple(sorted(properties, key=lambda item: (item.public_ref.encode('utf-8'), item.property_name.encode('utf-8')))), tuple(sorted(set(relations), key=lambda item: (item.subject_public_ref.encode('utf-8'), item.relation.encode('utf-8'), item.object_public_ref.encode('utf-8')))))
    return state.to_mapping()

def build_public_postcondition_evidence(pre_state: Mapping[str, Any] | None, post_state: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if pre_state is None or post_state is None:
        return None
    return {'schema_version': 'method-public-postcondition-evidence.v1', 'pre': dict(pre_state), 'post': dict(post_state)}
__all__ = ['build_public_postcondition_evidence', 'capture_public_action_state']
