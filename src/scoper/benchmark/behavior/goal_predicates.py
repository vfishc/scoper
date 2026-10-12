from __future__ import annotations
from typing import Any, Mapping
ROLE_ARG_KEYS = frozenset({'object', 'receptacle', 'target', 'source', 'tool', 'heatsource', 'coldsource'})
LITERAL_ARG_KEYS = frozenset({'substance', 'threshold', 'liquid'})

def normalize_goal_predicate(predicate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(predicate, Mapping):
        raise TypeError('predicate must be a mapping')
    ptype = str(predicate.get('type') or '')
    args = dict(predicate.get('args') or {})
    object_role = predicate.get('object_role') or predicate.get('role') or args.get('object')
    receptacle_role = predicate.get('receptacle_role') or args.get('receptacle')
    receptacle_roles = predicate.get('receptacle_roles') or args.get('receptacle_roles')
    if receptacle_roles is None:
        receptacle_roles = []
    elif isinstance(receptacle_roles, (str, bytes)):
        receptacle_roles = [str(receptacle_roles)]
    else:
        receptacle_roles = [str(role) for role in receptacle_roles if role is not None]
    if object_role is not None:
        args.setdefault('object', object_role)
    if receptacle_role is not None:
        args.setdefault('receptacle', receptacle_role)
    if receptacle_roles:
        args.setdefault('receptacle_roles', list(receptacle_roles))
    literal_args: dict[str, Any] = {}
    for key, value in args.items():
        if key in LITERAL_ARG_KEYS:
            literal_args[key] = value
    for key, value in predicate.items():
        if key in LITERAL_ARG_KEYS and key not in literal_args:
            literal_args[key] = value
    record = {'type': ptype, 'args': args, 'object_role': str(object_role) if object_role is not None else None, 'receptacle_role': str(receptacle_role) if receptacle_role is not None else None, 'receptacle_roles': list(receptacle_roles), 'literal_args': literal_args}
    for key in ('satisfied', 'expected', 'actual', 'reason', 'owner_subtask_id'):
        if key in predicate:
            record[key] = predicate[key]
    return record

def format_goal_predicate(predicate: Mapping[str, Any]) -> str:
    record = normalize_goal_predicate(predicate)
    ptype = record['type'] or 'predicate'
    obj = record.get('object_role')
    rec = record.get('receptacle_role')
    recs = record.get('receptacle_roles') or []
    if obj and rec:
        return '{}({} -> {})'.format(ptype, obj, rec)
    if obj and recs:
        return '{}({} -> [{}])'.format(ptype, obj, ', '.join(recs))
    if obj:
        return '{}({})'.format(ptype, obj)
    return ptype

def predicate_records_from_goal(goal: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [normalize_goal_predicate(item) for item in goal.get('predicates') or []]
__all__ = ['normalize_goal_predicate', 'format_goal_predicate', 'predicate_records_from_goal']
