from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence
from scoper.benchmark.procthor.object_taxonomy import ObjectTaxonomy

@dataclass(frozen=True)
class ViolationCandidate:
    v_type: str
    object_id: str
    object_type: str
    room_id: str
    urgency: str
    support_roles: Dict[str, str] = field(default_factory=dict)

def enumerate_violation_candidates(profile: Mapping[str, Any], scope_room_ids: Sequence[str], taxonomy: ObjectTaxonomy) -> List[ViolationCandidate]:
    scope_ids = set(scope_room_ids or [])
    objects = [obj for obj in profile.get('objects', []) or [] if not scope_ids or obj.get('room_id') in scope_ids]
    objects_by_id = {_object_id(obj): obj for obj in objects}
    candidates: List[ViolationCandidate] = []
    for obj in _sorted_objects(objects):
        if _object_type_in_group(obj, taxonomy, 'faucets') and obj.get('toggleable'):
            candidates.append(ViolationCandidate(v_type='faucet_on', object_id=_object_id(obj), object_type=_object_type(obj), room_id=str(obj.get('room_id') or ''), urgency='immediate'))
    for obj in _sorted_objects(objects):
        if _object_type_in_group(obj, taxonomy, 'policy_open_containers') and obj.get('openable'):
            candidates.append(ViolationCandidate(v_type='open_container', object_id=_object_id(obj), object_type=_object_type(obj), room_id=str(obj.get('room_id') or ''), urgency='before_stop'))
    sinks = [obj for obj in _sorted_objects(objects) if _object_type_in_group(obj, taxonomy, 'sink_basins')]
    faucets = [obj for obj in _sorted_objects(objects) if _object_type_in_group(obj, taxonomy, 'faucets') and obj.get('toggleable')]
    storages = [obj for obj in _sorted_objects(objects) if _object_type_in_group(obj, taxonomy, 'dishware_storage') and obj.get('receptacle') and (not obj.get('pickupable'))]
    if sinks and faucets and storages:
        sink = sinks[0]
        faucet = _paired_faucet(faucets, sink) or faucets[0]
        storage = storages[0]
        for obj in _sorted_objects(objects):
            if not (_object_type_in_group(obj, taxonomy, 'dirtyable_task_objects') and obj.get('pickupable')):
                continue
            if _has_closed_parent(obj, objects_by_id):
                continue
            candidates.append(ViolationCandidate(v_type='dirty_dish', object_id=_object_id(obj), object_type=_object_type(obj), room_id=str(obj.get('room_id') or ''), urgency='before_stop', support_roles={'sink': _object_id(sink), 'faucet': _object_id(faucet), 'storage': _object_id(storage)}))
    return candidates

def sample_violations(profile: Mapping[str, Any], scope_room_ids: Sequence[str], taxonomy: ObjectTaxonomy, seed: int, min_violations: int=2, max_violations: int=3) -> Optional[List[ViolationCandidate]]:
    candidates = enumerate_violation_candidates(profile, scope_room_ids, taxonomy)
    by_type: Dict[str, List[ViolationCandidate]] = {}
    for candidate in candidates:
        by_type.setdefault(candidate.v_type, []).append(candidate)
    if len(by_type) < int(min_violations):
        return None
    rng = random.Random(seed)
    selected_types = sorted(by_type)
    count = min(rng.randint(int(min_violations), int(max_violations)), len(selected_types))
    chosen_types = rng.sample(selected_types, count)
    return [rng.choice(sorted(by_type[v_type], key=_candidate_key)) for v_type in chosen_types]

def _paired_faucet(faucets: Sequence[Mapping[str, Any]], sink: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    sink_id = _object_id(sink)
    sink_base = sink_id.split('___', 1)[0]
    for faucet in faucets:
        parents = set(faucet.get('parent_receptacles') or [])
        if sink_id in parents or sink_base in parents:
            return faucet
    return None

def _has_closed_parent(obj: Mapping[str, Any], objects_by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    for parent_id in obj.get('parent_receptacles') or []:
        parent = objects_by_id.get(parent_id)
        if parent and parent.get('openable') and (parent.get('is_open') is False):
            return True
    return False

def _object_type_in_group(obj: Mapping[str, Any], taxonomy: ObjectTaxonomy, group_name: str) -> bool:
    return taxonomy.object_type_in_group(_object_type(obj), group_name)

def _candidate_key(candidate: ViolationCandidate) -> tuple:
    return (candidate.v_type, candidate.object_type, candidate.object_id, tuple(sorted(candidate.support_roles.items())))

def _sorted_objects(objects: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(objects, key=lambda obj: (_object_type(obj), _object_id(obj)))

def _object_id(obj: Mapping[str, Any]) -> str:
    object_id = obj.get('object_id') or obj.get('objectId')
    if not isinstance(object_id, str) or not object_id:
        raise ValueError('Profile object is missing object_id.')
    return object_id

def _object_type(obj: Mapping[str, Any]) -> str:
    object_type = obj.get('object_type') or obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    return _object_id(obj).split('|', 1)[0]
__all__ = ['ViolationCandidate', 'enumerate_violation_candidates', 'sample_violations']
