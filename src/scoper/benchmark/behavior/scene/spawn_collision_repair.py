from __future__ import annotations
from typing import Any, Mapping, Sequence
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject
SPAWN_SUPPORT_CLEARANCE = 0.02
SPAWN_SUPPORT_MARGIN = 0.06

def repair_task_spawn_collisions_after_merge(objects: Sequence[AABBObject], *, role_map: Mapping[str, str], placements: dict[str, dict[str, Any]], max_slots: int=48) -> tuple[list[AABBObject], dict[str, Any]]:
    role_object_names = {str(name) for name in role_map.values()}
    by_name = {str(obj.name): obj for obj in objects}
    removed_background: set[str] = set()
    repairs: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    checked = 0
    for obj_name, placement in list(placements.items()):
        spawn = placement.get('spawn_placement') if isinstance(placement, dict) else None
        if not isinstance(spawn, dict) or spawn.get('mode') != 'on_support_surface':
            continue
        obj = by_name.get(str(obj_name))
        if obj is None:
            continue
        checked += 1
        support_name = str(spawn.get('support_object') or spawn.get('support_role') or '')
        support = by_name.get(support_name)
        blockers = _collision_blockers(obj, objects, removed_background, support_name=support_name)
        if not blockers:
            continue
        before = tuple(obj.center)
        if support is not None and _try_reslot_spawn(obj, support, objects, removed_background, spawn, start_slot=int(spawn.get('slot') or 0) + 1, max_slots=max_slots):
            placement['center'] = [float(v) for v in obj.center]
            placement['source'] = 'post_merge_collision_repair_support_slot'
            repairs.append({'object': obj.name, 'action': 'reslot_spawn', 'support_object': support.name, 'overlapped_with': [item.name for item in blockers], 'before': list(before), 'after': list(obj.center), 'slot': int(spawn.get('slot') or 0)})
            continue
        removable = [item for item in blockers if _is_removable_background(item, role_object_names)]
        if removable:
            for item in removable:
                removed_background.add(str(item.name))
            blockers_after_remove = _collision_blockers(obj, objects, removed_background, support_name=support_name)
            repairs.append({'object': obj.name, 'action': 'remove_symbolic_background', 'removed_background': [item.name for item in removable], 'remaining_blockers': [item.name for item in blockers_after_remove], 'before': list(before), 'after': list(obj.center)})
            if not blockers_after_remove:
                continue
            blockers = blockers_after_remove
        unresolved.append({'object': obj.name, 'support_object': support_name or None, 'blockers': [item.name for item in blockers], 'center': list(obj.center)})
    final_objects = [obj for obj in objects if str(obj.name) not in removed_background]
    report = {'enabled': True, 'checked_spawn_object_count': checked, 'repair_count': len(repairs), 'removed_background_object_count': len(removed_background), 'unresolved_collision_count': len(unresolved), 'repairs': repairs, 'removed_background_objects': sorted(removed_background), 'unresolved_collisions': unresolved, 'final_object_count': len(final_objects)}
    return (final_objects, report)

def _try_reslot_spawn(obj: AABBObject, support: AABBObject, objects: Sequence[AABBObject], removed_background: set[str], spawn: dict[str, Any], *, start_slot: int, max_slots: int) -> bool:
    clearance = float(spawn.get('clearance', SPAWN_SUPPORT_CLEARANCE))
    for slot in range(int(start_slot), int(start_slot) + int(max_slots)):
        center = _support_surface_center(obj.extent, support, slot, clearance=clearance)
        old_center = obj.center
        obj.center = center
        if not _collision_blockers(obj, objects, removed_background, support_name=str(support.name)):
            spawn['slot'] = int(slot)
            spawn['offset'] = [float(obj.center[0] - support.center[0]), float(obj.center[1] - support.center[1]), float(obj.center[2] - support.center[2])]
            _clear_pose_metadata(obj)
            return True
        obj.center = old_center
    return False

def _support_surface_center(extent: Sequence[float], support: AABBObject, slot: int, *, clearance: float) -> tuple[float, float, float]:
    usable_x = max(float(support.extent[0]) * 0.5 - float(extent[0]) * 0.5 - SPAWN_SUPPORT_MARGIN, 0.0)
    usable_y = max(float(support.extent[1]) * 0.5 - float(extent[1]) * 0.5 - SPAWN_SUPPORT_MARGIN, 0.0)
    direction = int(slot) % 4
    lane = int(slot) // 4
    lane_shift_x = min(usable_x, lane * max(float(extent[0]) + SPAWN_SUPPORT_MARGIN, 0.08))
    lane_shift_y = min(usable_y, lane * max(float(extent[1]) + SPAWN_SUPPORT_MARGIN, 0.08))
    if direction == 0:
        offset_x, offset_y = (usable_x, -lane_shift_y)
    elif direction == 1:
        offset_x, offset_y = (-usable_x, lane_shift_y)
    elif direction == 2:
        offset_x, offset_y = (-lane_shift_x, usable_y)
    else:
        offset_x, offset_y = (lane_shift_x, -usable_y)
    top_z = float(support.center[2]) + float(support.extent[2]) * 0.5
    return (float(support.center[0] + offset_x), float(support.center[1] + offset_y), float(top_z + float(extent[2]) * 0.5 + float(clearance)))

def _collision_blockers(obj: AABBObject, objects: Sequence[AABBObject], removed_background: set[str], *, support_name: str) -> list[AABBObject]:
    blockers: list[AABBObject] = []
    for other in objects:
        if other is obj:
            continue
        if str(other.name) in removed_background:
            continue
        if support_name and str(other.name) == str(support_name):
            continue
        if _aabb_overlap(obj, other, tolerance=0.01):
            blockers.append(other)
    return blockers

def _is_removable_background(obj: AABBObject, role_object_names: set[str]) -> bool:
    if str(obj.name) in role_object_names:
        return False
    states = dict(obj.states or {})
    abilities = {str(item) for item in obj.abilities or []}
    return bool(states.get('occupancy_ignored') or (states.get('scene_layout_source') == 'source_json_full' and 'pickupable' in abilities))

def _aabb_overlap(a: AABBObject, b: AABBObject, *, tolerance: float) -> bool:
    for axis in range(3):
        delta = abs(float(a.center[axis]) - float(b.center[axis]))
        limit = float(a.extent[axis]) * 0.5 + float(b.extent[axis]) * 0.5 - float(tolerance)
        if delta >= limit:
            return False
    return True

def _clear_pose_metadata(obj: AABBObject) -> None:
    obj.root_position = None
    obj.root_orientation_xyzw = None
    obj.root_yaw = None
    obj.pose_source = None
__all__ = ['repair_task_spawn_collisions_after_merge']
