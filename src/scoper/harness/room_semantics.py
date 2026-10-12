from __future__ import annotations
import copy
import re
from typing import Any, Mapping, Sequence
CANONICAL_ROOM_TYPES = frozenset({'bar', 'bathroom', 'bedroom', 'biology_lab', 'break_room', 'chemistry_lab', 'childs_room', 'classroom', 'closet', 'computer_lab', 'conference_hall', 'copy_room', 'dining_room', 'empty_room', 'entryway', 'exercise_room', 'garage', 'garden', 'grocery_store', 'gym', 'hallway', 'hammam', 'infirmary', 'kitchen', 'laundry_room', 'living_room', 'lobby', 'locker_room', 'meeting_room', 'office', 'pantry_room', 'phone_room', 'playroom', 'sauna', 'shared_office', 'spa', 'staircase', 'storage_room', 'television_room', 'utility_room'})

class RoomSemanticError(ValueError):

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

def canonical_room_type(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RoomSemanticError('ROOM_TYPE_UNKNOWN', 'room type is missing')
    normalized = re.sub('[^a-z0-9]+', '_', value.strip().casefold()).strip('_')
    aliases = {'bath_room': 'bathroom', 'bed_room': 'bedroom', 'corridor': 'hallway', 'diningroom': 'dining_room', 'entrance': 'entryway', 'laundry': 'laundry_room', 'livingroom': 'living_room', 'private_office': 'office'}
    normalized = aliases.get(normalized, normalized)
    if normalized not in CANONICAL_ROOM_TYPES:
        raise RoomSemanticError('ROOM_TYPE_UNKNOWN', 'unsupported room type {!r}'.format(value))
    return normalized

def room_display_label(value: Any) -> str:
    return canonical_room_type(value).replace('_', ' ')

def build_room_type_catalog(room_records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for record in room_records:
        if not isinstance(record, Mapping):
            raise RoomSemanticError('ROOM_TYPE_UNKNOWN', 'room record is not an object')
        room_id = record.get('room_id', record.get('roomId', record.get('id')))
        room_type = record.get('room_type', record.get('roomType'))
        if not isinstance(room_id, str) or not room_id:
            raise RoomSemanticError('ROOM_TYPE_UNKNOWN', 'room record is missing room_id')
        canonical = canonical_room_type(room_type)
        existing = catalog.get(room_id)
        if existing is not None and existing != canonical:
            raise RoomSemanticError('ROOM_TYPE_CONFLICT', 'room {!r} has conflicting types'.format(room_id))
        catalog[room_id] = canonical
    return catalog

def materialize_room_scope(scope_template: Mapping[str, Any], *, roles: Mapping[str, Mapping[str, Any]], room_catalog: Mapping[str, str]) -> dict[str, Any]:
    scope = copy.deepcopy(dict(scope_template))
    scope_type = scope.get('type')
    if scope_type == 'whole_house':
        scope.pop('room_ids_from_role', None)
        scope.pop('room_ids_from_roles', None)
        scope['room_ids'] = []
        scope['label'] = 'house'
        return scope
    role_names: list[str] = []
    role_name = scope.pop('room_ids_from_role', None)
    if isinstance(role_name, str) and role_name:
        role_names.append(role_name)
    role_list = scope.pop('room_ids_from_roles', None)
    if isinstance(role_list, list):
        role_names.extend((str(item) for item in role_list if item))
    if role_names:
        room_ids = []
        for name in role_names:
            role = roles.get(name)
            room_id = role.get('room_id') if isinstance(role, Mapping) else None
            if not isinstance(room_id, str) or not room_id:
                raise RoomSemanticError('ROOM_SCOPE_ROLE_UNBOUND', 'scope role {!r} has no room binding'.format(name))
            if room_id not in room_ids:
                room_ids.append(room_id)
    else:
        raw_room_ids = scope.get('room_ids')
        room_ids = [str(item) for item in raw_room_ids or [] if item]
        if not room_ids:
            room_ids = sorted({str(role.get('room_id')) for role in roles.values() if isinstance(role, Mapping) and role.get('room_id')})
    if not room_ids:
        raise RoomSemanticError('ROOM_SCOPE_ROLE_UNBOUND', 'room scope is empty')
    if scope_type == 'single_room' and len(room_ids) != 1:
        raise RoomSemanticError('SINGLE_ROOM_SCOPE_CONFLICT', 'single-room scope resolved multiple rooms')
    try:
        room_types = [canonical_room_type(room_catalog[room_id]) for room_id in room_ids]
    except KeyError as exc:
        raise RoomSemanticError('ROOM_TYPE_UNKNOWN', 'room {!r} is absent from the room catalog'.format(exc.args[0])) from exc
    required = scope.pop('required_room_types', None)
    if required is not None:
        if not isinstance(required, list) or not required:
            raise RoomSemanticError('ROOM_TYPE_UNKNOWN', 'required_room_types must be a non-empty list')
        required_types = {canonical_room_type(item) for item in required}
        if any((room_type not in required_types for room_type in room_types)):
            raise RoomSemanticError('ROOM_TYPE_MISMATCH', 'resolved room types {!r} do not satisfy {!r}'.format(sorted(set(room_types)), sorted(required_types)))
        scope['required_room_types'] = sorted(required_types)
    scope['room_ids'] = room_ids
    scope['label'] = room_display_label(room_types[0]) if scope_type == 'single_room' else 'connected area'
    return scope

def audit_room_scope(scope: Mapping[str, Any], room_catalog: Mapping[str, str]) -> list[str]:
    issues: list[str] = []
    room_ids = [str(item) for item in scope.get('room_ids') or [] if item]
    if scope.get('type') == 'whole_house':
        if room_ids or scope.get('label') != 'house':
            issues.append('WHOLE_HOUSE_SCOPE_MISMATCH')
        return issues
    if scope.get('type') == 'single_room' and len(room_ids) != 1:
        issues.append('SINGLE_ROOM_SCOPE_CONFLICT')
    room_types: list[str] = []
    for room_id in room_ids:
        try:
            room_types.append(canonical_room_type(room_catalog[room_id]))
        except (KeyError, RoomSemanticError):
            issues.append('ROOM_TYPE_UNKNOWN')
    if len(room_types) == 1 and scope.get('label') != room_display_label(room_types[0]):
        issues.append('ROOM_SCOPE_LABEL_MISMATCH')
    required = scope.get('required_room_types')
    if isinstance(required, list):
        try:
            required_types = {canonical_room_type(item) for item in required}
            if any((room_type not in required_types for room_type in room_types)):
                issues.append('ROOM_TYPE_MISMATCH')
        except RoomSemanticError:
            issues.append('ROOM_TYPE_UNKNOWN')
    return sorted(set(issues))

def audit_task_room_semantics(task_spec: Mapping[str, Any], room_catalog: Mapping[str, str]) -> list[str]:
    family = task_spec.get('family_spec')
    scope = family.get('scope') if isinstance(family, Mapping) else None
    if not isinstance(scope, Mapping):
        return []
    issues = audit_room_scope(scope, room_catalog)
    if scope.get('type') == 'single_room':
        room_ids = list(scope.get('room_ids') or [])
        if len(room_ids) == 1 and room_ids[0] in room_catalog:
            actual = room_display_label(room_catalog[room_ids[0]])
            instruction = str(task_spec.get('instruction') or '').casefold()
            advertised = {room_display_label(room_type) for room_type in CANONICAL_ROOM_TYPES if re.search('\\b{}\\b'.format(re.escape(room_display_label(room_type))), instruction)}
            if advertised and advertised != {actual}:
                issues.append('ROOM_INSTRUCTION_LABEL_MISMATCH')
    return sorted(set(issues))

def audit_task_bank_room_semantics(tasks: Sequence[Mapping[str, Any]], room_catalogs: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for task in tasks:
        source = task.get('source') if isinstance(task.get('source'), Mapping) else {}
        source_id = source.get('house_id') or source.get('scene_model')
        catalog = room_catalogs.get(str(source_id)) if source_id is not None else None
        issues = audit_task_room_semantics(task, catalog) if isinstance(catalog, Mapping) else ['ROOM_CATALOG_MISSING']
        if issues:
            report.append({'issues': issues, 'task_id': str(task.get('id') or '')})
    return report
__all__ = ['CANONICAL_ROOM_TYPES', 'RoomSemanticError', 'audit_room_scope', 'audit_task_bank_room_semantics', 'audit_task_room_semantics', 'build_room_type_catalog', 'canonical_room_type', 'materialize_room_scope', 'room_display_label']
