from __future__ import annotations
import copy
from typing import Any, Mapping
from scoper.benchmark.common.vocabulary import SHARED_VOCABULARY_REGISTRY

def materialize_observation_conditioned_spec(spec_template: Mapping[str, Any], *, roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    spec = copy.deepcopy(dict(spec_template))
    rooms_from_roles = spec.pop('rooms_from_roles', None)
    room_from_role = spec.pop('room_from_role', None)
    if isinstance(rooms_from_roles, list):
        spec['rooms'] = _room_ids(rooms_from_roles, roles)
    elif isinstance(room_from_role, str):
        spec['rooms'] = _room_ids([room_from_role], roles)
    else:
        spec.setdefault('rooms', [])
    payload_templates = spec.pop('latent_payload_templates', None)
    if payload_templates is None:
        return spec
    if not isinstance(payload_templates, list):
        raise ValueError('observation_conditioned.latent_payload_templates must be a list')
    spec['latent_payloads'] = [_materialize_payload(payload, roles, index, spec['rooms']) for index, payload in enumerate(payload_templates) if isinstance(payload, Mapping)]
    return spec

def _materialize_payload(payload_template: Mapping[str, Any], roles: Mapping[str, Mapping[str, Any]], index: int, rooms: list[str]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(payload_template))
    if not payload.get('latent_id') and (not payload.get('issue_id')):
        payload['latent_id'] = 'issue_{}'.format(index)
    room_from_role = payload.pop('room_from_role', None)
    if isinstance(room_from_role, str):
        room_ids = _room_ids([room_from_role], roles)
        if room_ids:
            payload['room_id'] = room_ids[0]
    if not payload.get('room_id') and rooms:
        payload['room_id'] = rooms[0]
    family_from_role = payload.pop('object_family_from_role', None)
    if isinstance(family_from_role, str):
        role = roles.get(family_from_role)
        raw_type = role.get('object_type') if isinstance(role, Mapping) else None
        family = SHARED_VOCABULARY_REGISTRY.family(raw_type)
        if family:
            payload['object_family'] = family
    return payload

def _room_ids(role_names: list[str], roles: Mapping[str, Mapping[str, Any]]) -> list[str]:
    room_ids: list[str] = []
    for role_name in role_names:
        role = roles.get(str(role_name))
        room_id = role.get('room_id') if isinstance(role, Mapping) else None
        if isinstance(room_id, str) and room_id and (room_id not in room_ids):
            room_ids.append(room_id)
    return room_ids
__all__ = ['materialize_observation_conditioned_spec']
