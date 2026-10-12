from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from scoper.harness.canonical import canonical_json_bytes
from scoper.harness.contracts import MethodContractError
from scoper.harness.leakage import assert_no_method_leakage
MEMORY_DELTA_FIELDS = frozenset({'schema_version', 'base_revision', 'evidence_frame_ids', 'proposal_mode', 'fact_mutations', 'incident_mutations', 'obligation_mutations', 'abstain_reason_or_null'})
FACT_MUTATION_FIELDS = frozenset({'operation', 'public_entity_ref', 'relation_or_property', 'object_public_role_or_value', 'confidence'})
INCIDENT_MUTATION_FIELDS = frozenset({'operation', 'failure_type', 'public_roles', 'confidence'})
OBLIGATION_MUTATION_FIELDS = frozenset({'operation', 'type', 'public_roles', 'source_incident_id_or_null'})

def _exact_keys(value: Any, expected: frozenset[str], *, pointer: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodContractError('method_schema_invalid', 'value must be a mapping', pointer=pointer)
    actual = set(value)
    if actual != expected:
        raise MethodContractError('method_schema_invalid', f'keys must be exactly {sorted(expected)!r}; got {sorted(actual)!r}', pointer=pointer)
    return value

def _non_empty_string(value: Any, *, pointer: str) -> str:
    if not isinstance(value, str) or not value:
        raise MethodContractError('method_schema_invalid', 'value must be a non-empty string', pointer=pointer)
    return value

def _int(value: Any, *, pointer: str, minimum: int=0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MethodContractError('method_schema_invalid', f'value must be an integer >= {minimum}', pointer=pointer)
    return value

def _number_01(value: Any, *, pointer: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not 0 <= value <= 1):
        raise MethodContractError('method_schema_invalid', 'confidence must be in [0, 1]', pointer=pointer)
    return float(value)

def _string_list(value: Any, *, pointer: str) -> list[str]:
    if not isinstance(value, list) or any((not isinstance(item, str) or not item for item in value)):
        raise MethodContractError('method_schema_invalid', 'value must be a list of non-empty strings', pointer=pointer)
    return list(value)

def validate_memory_delta(value: Mapping[str, Any], *, pointer: str='/labels/memory_delta') -> None:
    delta = _exact_keys(value, MEMORY_DELTA_FIELDS, pointer=pointer)
    if delta['schema_version'] != 'memory-delta.v1':
        raise MethodContractError('method_schema_invalid', 'invalid memory delta version', pointer=f'{pointer}/schema_version')
    _int(delta['base_revision'], pointer=f'{pointer}/base_revision')
    _string_list(delta['evidence_frame_ids'], pointer=f'{pointer}/evidence_frame_ids')
    if delta['proposal_mode'] not in {'update', 'reconcile'}:
        raise MethodContractError('method_schema_invalid', 'invalid proposal_mode', pointer=f'{pointer}/proposal_mode')
    fact_keys: dict[bytes, set[str]] = {}
    for index, mutation in enumerate(delta['fact_mutations']):
        item_pointer = f'{pointer}/fact_mutations/{index}'
        mutation = _exact_keys(mutation, FACT_MUTATION_FIELDS, pointer=item_pointer)
        if mutation['operation'] not in {'add', 'confirm', 'invalidate', 'delete'}:
            raise MethodContractError('method_schema_invalid', 'invalid fact operation', pointer=f'{item_pointer}/operation')
        for field in ('public_entity_ref', 'relation_or_property'):
            _non_empty_string(mutation[field], pointer=f'{item_pointer}/{field}')
        _number_01(mutation['confidence'], pointer=f'{item_pointer}/confidence')
        key = canonical_json_bytes({'public_entity_ref': mutation['public_entity_ref'], 'relation_or_property': mutation['relation_or_property'], 'object_public_role_or_value': mutation['object_public_role_or_value']})
        fact_keys.setdefault(key, set()).add(mutation['operation'])
    if any(({'confirm', 'invalidate'}.issubset(operations) for operations in fact_keys.values())):
        raise MethodContractError('method_schema_invalid', 'same fact cannot be confirmed and invalidated', pointer=pointer)
    fact_sort = lambda item: (str(item['public_entity_ref']).encode('utf-8'), str(item['relation_or_property']).encode('utf-8'), canonical_json_bytes(item['object_public_role_or_value']), str(item['operation']).encode('ascii'))
    if delta['fact_mutations'] != sorted(delta['fact_mutations'], key=fact_sort):
        raise MethodContractError('method_schema_invalid', 'fact mutations are not canonical sorted', pointer=pointer)
    for index, mutation in enumerate(delta['incident_mutations']):
        item_pointer = f'{pointer}/incident_mutations/{index}'
        mutation = _exact_keys(mutation, INCIDENT_MUTATION_FIELDS, pointer=item_pointer)
        if mutation['operation'] not in {'create', 'update', 'resolve'}:
            raise MethodContractError('method_schema_invalid', 'invalid incident operation', pointer=item_pointer)
        _non_empty_string(mutation['failure_type'], pointer=f'{item_pointer}/failure_type')
        _string_list(mutation['public_roles'], pointer=f'{item_pointer}/public_roles')
        _number_01(mutation['confidence'], pointer=f'{item_pointer}/confidence')
    incident_sort = lambda item: (str(item['failure_type']).encode('utf-8'), canonical_json_bytes(item['public_roles']), str(item['operation']).encode('ascii'))
    if delta['incident_mutations'] != sorted(delta['incident_mutations'], key=incident_sort):
        raise MethodContractError('method_schema_invalid', 'incident mutations are not canonical sorted', pointer=pointer)
    for index, mutation in enumerate(delta['obligation_mutations']):
        item_pointer = f'{pointer}/obligation_mutations/{index}'
        mutation = _exact_keys(mutation, OBLIGATION_MUTATION_FIELDS, pointer=item_pointer)
        if mutation['operation'] not in {'create', 'resolve'} or mutation['type'] not in {'observe', 'act'}:
            raise MethodContractError('method_schema_invalid', 'invalid obligation mutation', pointer=item_pointer)
        _string_list(mutation['public_roles'], pointer=f'{item_pointer}/public_roles')
        if mutation['source_incident_id_or_null'] is not None:
            _non_empty_string(mutation['source_incident_id_or_null'], pointer=f'{item_pointer}/source_incident_id_or_null')
    obligation_sort = lambda item: (str(item['type']).encode('ascii'), canonical_json_bytes(item['public_roles']), str(item['operation']).encode('ascii'))
    if delta['obligation_mutations'] != sorted(delta['obligation_mutations'], key=obligation_sort):
        raise MethodContractError('method_schema_invalid', 'obligation mutations are not canonical sorted', pointer=pointer)
    abstain = delta['abstain_reason_or_null']
    mutations_exist = bool(delta['fact_mutations'] or delta['incident_mutations'] or delta['obligation_mutations'])
    if abstain is not None:
        _non_empty_string(abstain, pointer=f'{pointer}/abstain_reason_or_null')
        if mutations_exist:
            raise MethodContractError('method_schema_invalid', 'abstain cannot include commit mutations', pointer=pointer)
    assert_no_method_leakage(value, root_pointer=pointer, include_future_action=False)
__all__ = ['validate_memory_delta']
