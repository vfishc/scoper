from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence
from scoper.benchmark.procthor.object_taxonomy import ObjectTaxonomy
from scoper.benchmark.procthor.route_visibility_estimator import DEFAULT_GRID_STEP, estimate_route_visibility, _estimate_route_astar, _interpolated_route
from scoper.benchmark.procthor.scene_profile_utils import profile_object_room_id
OPEN_SURFACE_GROUP = 'open_surface_receptacles'

@dataclass(frozen=True)
class PoseCandidate:
    receptacle_object_id: str
    receptacle_room_id: Optional[str]
    distance_to_anchor: float
    rationale: dict

class PoseCandidateError(ValueError):
    pass

def find_pose_candidates_unblock_from_closed_container(profile: Mapping[str, Any], target_object_id: str, scope_room_ids: Sequence[str], taxonomy: ObjectTaxonomy, agent_initial_pos: Mapping[str, Any]) -> List[PoseCandidate]:
    objects = _objects_by_id(profile)
    target = objects.get(target_object_id)
    if target is None:
        return []
    closed_parent_ids = _closed_openable_parent_ids(target, objects)
    if not closed_parent_ids:
        return []
    return _surface_candidates(profile, taxonomy, scope_room_ids=scope_room_ids, excluded_object_ids=closed_parent_ids | {target_object_id}, anchor_pos=agent_initial_pos, rationale_base={'mode': 'unblock_from_closed_container', 'closed_parent_object_ids': sorted(closed_parent_ids)})

def find_pose_candidates_evacuate_from_goal_receptacle(profile: Mapping[str, Any], target_object_id: str, goal_receptacle_object_id: str, scope_room_ids: Sequence[str], taxonomy: ObjectTaxonomy, agent_initial_pos: Mapping[str, Any]) -> List[PoseCandidate]:
    objects = _objects_by_id(profile)
    target = objects.get(target_object_id)
    if target is None:
        return []
    if goal_receptacle_object_id not in set(_parent_receptacles(target)):
        return []
    return _surface_candidates(profile, taxonomy, scope_room_ids=scope_room_ids, excluded_object_ids={target_object_id, goal_receptacle_object_id}, anchor_pos=agent_initial_pos, rationale_base={'mode': 'evacuate_from_goal_receptacle', 'goal_receptacle_object_id': goal_receptacle_object_id})

def find_pose_candidates_route_visible_surface(profile: Mapping[str, Any], primary_object_pos: Mapping[str, Any], primary_receptacle_pos: Mapping[str, Any], agent_initial_pose: Mapping[str, Any], scope_room_ids: Sequence[str], taxonomy: ObjectTaxonomy) -> List[PoseCandidate]:
    reachable = _reachable_positions(profile)
    large_bboxes = _large_occluders(profile, taxonomy)
    candidates = []
    for obj in _surface_objects(profile, taxonomy, scope_room_ids=scope_room_ids, excluded_object_ids=set()):
        result = estimate_route_visibility(agent_initial_pose, primary_object_pos, primary_receptacle_pos, obj.get('position') or {}, reachable, large_bboxes)
        if not result.get('visible'):
            continue
        distance = float(result.get('distance_to_route', 0.0))
        candidates.append(PoseCandidate(receptacle_object_id=_object_id(obj), receptacle_room_id=_room_id(obj), distance_to_anchor=distance, rationale={'mode': 'route_visible_surface', 'predicted_visibility_source': 'navigation_stream_to_primary_receptacle', 'estimated_route_step_index': result.get('first_visible_step_index'), 'estimated_distance_to_route': distance}))
    return sorted(candidates, key=lambda item: (item.distance_to_anchor, item.receptacle_room_id or '', item.receptacle_object_id))

def find_pose_candidates_route_near_surface(profile: Mapping[str, Any], primary_object_pos: Mapping[str, Any], primary_receptacle_pos: Mapping[str, Any], agent_initial_pose: Mapping[str, Any], scope_room_ids: Sequence[str], taxonomy: ObjectTaxonomy, max_route_distance: float=4.0) -> List[PoseCandidate]:
    route_points = _estimate_route_astar(_reachable_positions(profile), agent_initial_pose.get('position') or {}, primary_object_pos, primary_receptacle_pos)
    if not route_points:
        route_points = _interpolated_route(agent_initial_pose.get('position') or {}, primary_object_pos, primary_receptacle_pos, DEFAULT_GRID_STEP)
    if not route_points:
        return []
    candidates = []
    for obj in _surface_objects(profile, taxonomy, scope_room_ids=scope_room_ids, excluded_object_ids=set()):
        distance = min((_xz_distance(point, obj.get('position') or {}) for point in route_points))
        if distance > max_route_distance:
            continue
        candidates.append(PoseCandidate(receptacle_object_id=_object_id(obj), receptacle_room_id=_room_id(obj), distance_to_anchor=distance, rationale={'mode': 'route_near_surface', 'predicted_visibility_source': 'route_adjacent_surface_fallback', 'estimated_distance_to_route': distance, 'max_route_distance': max_route_distance}))
    return sorted(candidates, key=lambda item: (item.distance_to_anchor, item.receptacle_room_id or '', item.receptacle_object_id))

def find_pose_candidates_same_room_as(profile: Mapping[str, Any], anchor_role_object_id: str, target_object_id: str, taxonomy: ObjectTaxonomy) -> List[PoseCandidate]:
    objects = _objects_by_id(profile)
    anchor = objects.get(anchor_role_object_id)
    target = objects.get(target_object_id)
    if anchor is None or target is None:
        return []
    anchor_room_id = _room_id(anchor)
    if not anchor_room_id or _room_id(target) == anchor_room_id:
        return []
    return _surface_candidates(profile, taxonomy, scope_room_ids=[anchor_room_id], excluded_object_ids={target_object_id, anchor_role_object_id}, anchor_pos=anchor.get('position') or {}, rationale_base={'mode': 'same_room_as', 'anchor_object_id': anchor_role_object_id, 'anchor_room_id': anchor_room_id})

def _surface_candidates(profile: Mapping[str, Any], taxonomy: ObjectTaxonomy, scope_room_ids: Sequence[str], excluded_object_ids: set[str], anchor_pos: Mapping[str, Any], rationale_base: Mapping[str, Any]) -> List[PoseCandidate]:
    candidates = []
    for obj in _surface_objects(profile, taxonomy, scope_room_ids=scope_room_ids, excluded_object_ids=excluded_object_ids):
        distance = _xz_distance(anchor_pos, obj.get('position') or {})
        candidates.append(PoseCandidate(receptacle_object_id=_object_id(obj), receptacle_room_id=_room_id(obj), distance_to_anchor=distance, rationale={**dict(rationale_base), 'target_receptacle_object_id': _object_id(obj), 'distance_to_anchor': distance}))
    return sorted(candidates, key=lambda item: (item.distance_to_anchor, item.receptacle_room_id or '', item.receptacle_object_id))

def _surface_objects(profile: Mapping[str, Any], taxonomy: ObjectTaxonomy, scope_room_ids: Sequence[str], excluded_object_ids: set[str]) -> List[Mapping[str, Any]]:
    room_filter = {room_id for room_id in scope_room_ids if room_id}
    out = []
    for obj in profile.get('objects', []) or []:
        if not isinstance(obj, Mapping):
            continue
        object_id = _object_id(obj)
        if object_id in excluded_object_ids:
            continue
        if room_filter and _room_id(obj) not in room_filter:
            continue
        if not bool(obj.get('receptacle', False)):
            continue
        if bool(obj.get('pickupable', False)):
            continue
        if bool(obj.get('openable', False)):
            continue
        if not taxonomy.object_type_in_group(_object_type(obj), OPEN_SURFACE_GROUP):
            continue
        if not _has_xz(obj.get('position') or {}):
            continue
        out.append(obj)
    return out

def _closed_openable_parent_ids(obj: Mapping[str, Any], objects_by_id: Mapping[str, Mapping[str, Any]]) -> set[str]:
    parent_ids = set()
    for parent_id in _parent_receptacles(obj):
        parent = objects_by_id.get(parent_id, {})
        if parent.get('openable') and parent.get('is_open') is False:
            parent_ids.add(parent_id)
    return parent_ids

def _large_occluders(profile: Mapping[str, Any], taxonomy: ObjectTaxonomy) -> List[Mapping[str, Any]]:
    occluders = []
    for obj in profile.get('objects', []) or []:
        if not isinstance(obj, Mapping):
            continue
        if bool(obj.get('pickupable', False)):
            continue
        if taxonomy.object_type_in_group(_object_type(obj), OPEN_SURFACE_GROUP):
            continue
        if _has_xz(obj.get('position') or {}):
            occluders.append(obj)
    return occluders

def _reachable_positions(profile: Mapping[str, Any]) -> List[dict]:
    reachable = profile.get('reachable') or {}
    positions = reachable.get('positions') or reachable.get('reachable_positions') or []
    if positions:
        return [dict(position) for position in positions if isinstance(position, Mapping)]
    poses = reachable.get('sampled_agent_poses') or []
    return [dict(pose.get('position') or {}) for pose in poses if isinstance(pose, Mapping) and isinstance(pose.get('position'), Mapping)]

def _objects_by_id(profile: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {_object_id(obj): obj for obj in profile.get('objects', []) or [] if isinstance(obj, Mapping)}

def _parent_receptacles(obj: Mapping[str, Any]) -> List[str]:
    parents = obj.get('parent_receptacles') or obj.get('parentReceptacles') or []
    return [parent for parent in parents if isinstance(parent, str) and parent]

def _object_id(obj: Mapping[str, Any]) -> str:
    object_id = obj.get('object_id') or obj.get('objectId')
    if not isinstance(object_id, str) or not object_id:
        raise PoseCandidateError('Profile object is missing object_id.')
    return object_id

def _object_type(obj: Mapping[str, Any]) -> str:
    object_type = obj.get('object_type') or obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    return _object_id(obj).split('|', 1)[0]

def _room_id(obj: Mapping[str, Any]) -> Optional[str]:
    return profile_object_room_id(obj)

def _xz_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax = float(a.get('x', 0.0))
    az = float(a.get('z', 0.0))
    bx = float(b.get('x', 0.0))
    bz = float(b.get('z', 0.0))
    return ((ax - bx) ** 2 + (az - bz) ** 2) ** 0.5

def _has_xz(value: Mapping[str, Any]) -> bool:
    return 'x' in value and 'z' in value
__all__ = ['PoseCandidate', 'PoseCandidateError', 'find_pose_candidates_evacuate_from_goal_receptacle', 'find_pose_candidates_route_near_surface', 'find_pose_candidates_route_visible_surface', 'find_pose_candidates_same_room_as', 'find_pose_candidates_unblock_from_closed_container']
