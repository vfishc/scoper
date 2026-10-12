from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
from scoper.benchmark.behavior.export.types import TrajectoryConfigError
PUBLIC_ENTITY_BINDING_DELTA_VERSION = 'public-entity-binding-delta.v1'
PUBLIC_ENTITY_BINDING_DELTA_FIELDS = frozenset({'schema_version', 'operation', 'public_role', 'public_alias', 'source'})
PUBLIC_ENTITY_BINDING_SOURCES = frozenset({'action_end_unique_held_delta', 'legacy_next_decision_unique_held_delta'})

def canonical_public_handles(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    handles = tuple(value)
    if any((not isinstance(item, str) or not item for item in handles)):
        return None
    if len(handles) != len(set(handles)):
        return None
    return tuple(sorted(handles, key=lambda item: item.encode('utf-8')))

def build_public_entity_binding_delta(tool_call: Mapping[str, Any], *, executor_status: Any, held_before: Any, held_after: Any, source: str) -> Mapping[str, Any] | None:
    if source not in PUBLIC_ENTITY_BINDING_SOURCES:
        raise TrajectoryConfigError('public_entity_binding_delta_invalid', 'public entity binding source is invalid')
    if executor_status != 'succeeded' or not isinstance(tool_call, Mapping) or tool_call.get('tool_name') != 'pick_up' or (not isinstance(tool_call.get('arguments'), Mapping)):
        return None
    role = tool_call['arguments'].get('target')
    before = canonical_public_handles(held_before)
    after = canonical_public_handles(held_after)
    if not isinstance(role, str) or not role or before is None or (after is None):
        return None
    before_set = set(before)
    after_set = set(after)
    added = after_set - before_set
    if not before_set.issubset(after_set) or len(added) != 1:
        return None
    value = {'schema_version': PUBLIC_ENTITY_BINDING_DELTA_VERSION, 'operation': 'bind', 'public_role': role, 'public_alias': next(iter(added)), 'source': source}
    validate_public_entity_binding_delta(value)
    return value

def validate_public_entity_binding_delta(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PUBLIC_ENTITY_BINDING_DELTA_FIELDS:
        raise TrajectoryConfigError('public_entity_binding_delta_invalid', 'public entity binding fields differ')
    if value['schema_version'] != PUBLIC_ENTITY_BINDING_DELTA_VERSION or value['operation'] != 'bind' or value['source'] not in PUBLIC_ENTITY_BINDING_SOURCES or (not isinstance(value['public_role'], str)) or (not value['public_role']) or (not isinstance(value['public_alias'], str)) or (not value['public_alias']):
        raise TrajectoryConfigError('public_entity_binding_delta_invalid', 'public entity binding value is invalid')
    return value
__all__ = ['PUBLIC_ENTITY_BINDING_DELTA_FIELDS', 'PUBLIC_ENTITY_BINDING_DELTA_VERSION', 'PUBLIC_ENTITY_BINDING_SOURCES', 'build_public_entity_binding_delta', 'canonical_public_handles', 'validate_public_entity_binding_delta']
