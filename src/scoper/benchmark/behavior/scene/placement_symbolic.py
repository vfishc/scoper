from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any, List, Mapping
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose, EgoState
from scoper.benchmark.behavior.execution_core.object_placement import aabb_top_z
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject
from scoper.benchmark.behavior.scene import scene_state
from scoper.benchmark.behavior.scene import surface_placement
_RECEPTACLE_ABILITIES = frozenset({'support_surface', 'receptacle', 'surface', 'placeable_surface', 'container', 'fillable'})
_PICKABLE_ABILITIES = frozenset({'pickupable', 'pickable'})
_DEFAULT_START_XY = (2.0, 2.5)
_DEFAULT_FLOOR_Z = 0.0
_PICKABLE_BAND_X = (4.5, 6.5)
_PICKABLE_BAND_Y = (0.8, 1.6)
_SPAWN_SUPPORT_CLEARANCE = 0.02
_SPAWN_SUPPORT_MARGIN = 0.06

@dataclass
class SceneInitResult:
    aabb_objects: List[AABBObject]
    ego_state: EgoState
    role_map: dict
    placement_report: dict = field(default_factory=dict)

def build(task_spec: Mapping[str, Any], rng_seed: int=0) -> SceneInitResult:
    rng = random.Random(rng_seed)
    placement_geometry_mode = surface_placement.placement_geometry_mode(task_spec)
    placement_assets_root = surface_placement.assets_root_for_task(task_spec)
    surface_placement_report: list[dict[str, Any]] = []
    if placement_geometry_mode == 'aabb':
        roles = task_spec['roles']
    else:
        roles, normalization_report = surface_placement.normalize_roles_for_surface_placement(task_spec['roles'], mode=placement_geometry_mode, assets_root=placement_assets_root)
        surface_placement_report.extend(normalization_report)
    role_map: dict[str, str] = {}
    objects_by_name: dict[str, AABBObject] = {}
    placements: dict[str, dict] = {}
    pickable_slot = 0
    for role_name, role in roles.items():
        inst = role.get('bddl_inst') or role_name
        role_map[role_name] = inst
        category = _category_for(role)
        extent = _as_tuple3(role.get('extent'), default=(0.2, 0.2, 0.2))
        abilities = tuple(role.get('abilities', ()))
        if 'center' in role:
            center = _as_tuple3(role['center'])
            source = 'explicit_center'
        elif _is_pickable(abilities):
            center = _standalone_pickable_center(extent, pickable_slot, rng)
            pickable_slot += 1
            source = 'standalone_pickable'
        else:
            center = _standalone_pickable_center(extent, pickable_slot, rng)
            pickable_slot += 1
            source = 'floor_default'
        states = dict(role.get('init_states', {}) or {})
        interaction_faces = role.get('interaction_faces')
        if isinstance(interaction_faces, (list, tuple)) and interaction_faces:
            states.setdefault('interaction_faces', [dict(face) for face in interaction_faces if isinstance(face, Mapping)])
        if role.get('room_id'):
            states['room_id'] = str(role['room_id'])
        existing_source_object_id = states.get('source_object_id')
        if existing_source_object_id and str(existing_source_object_id).startswith('spawnable:'):
            states.setdefault('spawn_logical_id', str(existing_source_object_id))
            states.pop('source_object_id', None)
        elif existing_source_object_id:
            states.setdefault('source_scene_object_id', str(existing_source_object_id))
        object_id = str(role.get('object_id') or '')
        if object_id:
            if bool(role.get('requires_spawn')) or object_id.startswith('spawnable:'):
                states['spawn_logical_id'] = object_id
                catalog_object_id = str(role.get('catalog_object_id') or '')
                if catalog_object_id:
                    states['spawn_catalog_object_id'] = catalog_object_id
            else:
                states['source_scene_object_id'] = object_id
                states['source_object_id'] = object_id
        elif not existing_source_object_id and (bool(role.get('requires_spawn')) or ('requires_spawn' not in role and source == 'standalone_pickable')):
            states['spawn_logical_id'] = 'spawnable:{}:{}'.format(role_name, category)
        obj = AABBObject(name=inst, category=category, center=center, extent=extent, abilities=abilities, states=states, root_position=_root_position_from_role(role), root_orientation_xyzw=_root_orientation_from_role(role), root_yaw=_root_yaw_from_role(role), pose_source='role' if role.get('root_position') or role.get('root_orientation_xyzw') else None)
        objects_by_name[inst] = obj
        placement = {'center': list(center), 'extent': list(extent), 'source': source}
        if isinstance(role.get('spawn_placement'), Mapping):
            placement['spawn_placement'] = dict(role['spawn_placement'])
        placements[inst] = placement
    init = task_spec.get('init', [])
    applied_states = scene_state.apply_role_init_states(objects_by_name, roles, role_map)
    applied_states += scene_state.apply_init_states(objects_by_name, init, role_map)
    applied_spatial_relations = _apply_spatial_init_relations(objects_by_name, init, role_map, placements)
    _resolve_spawn_support_objects(roles, role_map, placements)
    protected_pairs = {frozenset((item['object'], item['target'])) for item in applied_spatial_relations if item.get('object') and item.get('target')}
    if placement_geometry_mode != 'aabb':
        protected_pairs.update(_spawn_support_pairs(roles, role_map))
    collision_repairs = _repair_pickable_overlaps(objects_by_name, placements, protected_pairs=protected_pairs, rng=rng, start_slot=pickable_slot, bounds=(task_spec.get('source') or {}).get('occupancy_bounds'))
    if placement_geometry_mode != 'aabb':
        surface_placement_report.extend(_apply_surface_spawn_adjustments(objects_by_name, roles, role_map, placements, mode=placement_geometry_mode, assets_root=placement_assets_root))
    spatial_relations = [{'predicate': e['predicate'], 'args': list(e['args'])} for e in init if e.get('predicate') in ('ontop', 'inside', 'inroom')]
    start_pose = (task_spec.get('source') or {}).get('start_pose') or task_spec.get('agent_start_pose')
    if start_pose:
        ego_pose = EgoPose(float(start_pose.get('x', _DEFAULT_START_XY[0])), float(start_pose.get('y', _DEFAULT_START_XY[1])), z=float(start_pose.get('z', 1.55)), yaw=float(start_pose.get('yaw', 0.0)), pitch=float(start_pose.get('pitch', 0.0)))
    else:
        ego_pose = EgoPose(*_DEFAULT_START_XY, yaw=0.0)
    ego_state = EgoState(pose=ego_pose)
    report = {'rng_seed': rng_seed, 'placements': placements, 'applied_states': applied_states, 'spatial_relations': spatial_relations, 'applied_spatial_relations': applied_spatial_relations, 'collision_repairs': collision_repairs}
    if placement_geometry_mode != 'aabb':
        report['placement_geometry'] = placement_geometry_mode
        report['surface_placement'] = surface_placement_report
    return SceneInitResult(aabb_objects=list(objects_by_name.values()), ego_state=ego_state, role_map=role_map, placement_report=report)

def _category_for(role: Mapping[str, Any]) -> str:
    if role.get('category'):
        return str(role['category'])
    synset = str(role.get('synset', 'object'))
    return synset.split('.', 1)[0]

def _is_pickable(abilities: tuple) -> bool:
    return any((a in _PICKABLE_ABILITIES for a in abilities))

def _as_tuple3(value, default=(0.0, 0.0, 0.0)) -> tuple:
    if value is None:
        return tuple((float(v) for v in default))
    return (float(value[0]), float(value[1]), float(value[2]))

def _root_position_from_role(role: Mapping[str, Any]) -> tuple[float, float, float] | None:
    value = role.get('root_position')
    if isinstance(value, Mapping):
        try:
            return (float(value['x']), float(value['y']), float(value['z']))
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return None
    return None

def _root_orientation_from_role(role: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ('root_orientation_xyzw', 'orientation'):
        value = role.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                return tuple((float(item) for item in value[:4]))
            except Exception:
                return None
    return None

def _root_yaw_from_role(role: Mapping[str, Any]) -> float | None:
    value = role.get('root_yaw')
    if isinstance(value, (int, float)):
        return float(value)
    return None

def _clear_pose_metadata(obj: AABBObject) -> None:
    obj.root_position = None
    obj.root_orientation_xyzw = None
    obj.root_yaw = None
    obj.pose_source = None

def _standalone_pickable_center(extent: tuple, slot: int, rng: random.Random) -> tuple:
    span_x = _PICKABLE_BAND_X[1] - _PICKABLE_BAND_X[0]
    x = _PICKABLE_BAND_X[0] + slot * 0.6 % max(span_x, 1e-06) + rng.uniform(-0.02, 0.02)
    y = _PICKABLE_BAND_Y[0] + slot * 0.37 % max(_PICKABLE_BAND_Y[1] - _PICKABLE_BAND_Y[0], 1e-06)
    z = _DEFAULT_FLOOR_Z + extent[2] * 0.5
    return (float(x), float(y), float(z))

def _apply_spatial_init_relations(objects_by_name: Mapping[str, AABBObject], init: List[Mapping[str, Any]], role_map: Mapping[str, str], placements: dict[str, dict]) -> list[dict]:
    applied: list[dict] = []
    for entry in init:
        predicate = entry.get('predicate')
        if predicate not in ('ontop', 'inside'):
            continue
        args = list(entry.get('args') or [])
        if len(args) < 2:
            continue
        obj_name = role_map.get(str(args[0]), str(args[0]))
        target_name = role_map.get(str(args[1]), str(args[1]))
        obj = objects_by_name.get(obj_name)
        target = objects_by_name.get(target_name)
        if obj is None or target is None:
            continue
        before = tuple(obj.center)
        if predicate == 'ontop':
            obj.center = (float(target.center[0]), float(target.center[1]), float(aabb_top_z(target) + obj.extent[2] * 0.5))
        else:
            obj.center = _inside_center(obj, target)
            clearance = float(entry.get('interior_floor_clearance', 0.0))
            if not 0.0 <= clearance <= 0.1:
                raise ValueError('interior_floor_clearance must be between 0 and 0.10 metres')
            if clearance:
                top = float(target.center[2] + target.extent[2] * 0.5 - obj.extent[2] * 0.5)
                if obj.center[2] + clearance > top:
                    raise ValueError('Object does not fit above container interior floor')
                obj.center = (obj.center[0], obj.center[1], obj.center[2] + clearance)
            offset = entry.get('interior_offset_xy', [0.0, 0.0])
            if len(offset) != 2 or any((abs(float(v)) > 0.5 for v in offset)):
                raise ValueError('interior_offset_xy must contain two offsets within 0.5 metres')
            if 'interior_offset_xy' in entry and any((abs(float(offset[i])) + obj.extent[i] * 0.5 > target.extent[i] * 0.5 for i in (0, 1))):
                raise ValueError('Object does not fit within container interior bounds')
            obj.center = (obj.center[0] + float(offset[0]), obj.center[1] + float(offset[1]), obj.center[2])
        _clear_pose_metadata(obj)
        placements.setdefault(obj_name, {})['center'] = list(obj.center)
        placements[obj_name]['source'] = 'spatial_init_{}'.format(predicate)
        placements[obj_name]['spatial_init_target'] = target_name
        applied.append({'predicate': str(predicate), 'object': obj_name, 'target': target_name, 'before': list(before), 'after': list(obj.center)})
    return applied

def _resolve_spawn_support_objects(roles: Mapping[str, Mapping[str, Any]], role_map: Mapping[str, str], placements: dict[str, dict]) -> None:
    for role_name, role in roles.items():
        inst = role_map.get(str(role_name), str(role.get('bddl_inst') or role_name))
        placement = placements.get(inst)
        if not placement:
            continue
        spawn = placement.get('spawn_placement')
        if not isinstance(spawn, dict):
            continue
        support_role = spawn.get('support_role')
        if isinstance(support_role, str) and support_role in role_map:
            spawn['support_object'] = role_map[support_role]

def _spawn_support_pairs(roles: Mapping[str, Mapping[str, Any]], role_map: Mapping[str, str]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for role_name, role in roles.items():
        spawn = role.get('spawn_placement')
        if not isinstance(spawn, Mapping) or spawn.get('mode') != 'on_support_surface':
            continue
        support_role = spawn.get('support_role')
        if not isinstance(support_role, str):
            continue
        obj_name = role_map.get(str(role_name), str(role.get('bddl_inst') or role_name))
        support_name = role_map.get(str(support_role), str(support_role))
        if obj_name and support_name and (obj_name != support_name):
            pairs.add(frozenset((str(obj_name), str(support_name))))
    return pairs

def _inside_center(obj: AABBObject, container: AABBObject) -> tuple[float, float, float]:
    min_z = float(container.center[2] - container.extent[2] * 0.5)
    max_z = float(container.center[2] + container.extent[2] * 0.5)
    low = min_z + float(obj.extent[2]) * 0.5
    high = max_z - float(obj.extent[2]) * 0.5
    z = float(container.center[2]) if low > high else max(min(low, high), min(low, max_z))
    return (float(container.center[0]), float(container.center[1]), z)

def _apply_surface_spawn_adjustments(objects_by_name: Mapping[str, AABBObject], roles: Mapping[str, Mapping[str, Any]], role_map: Mapping[str, str], placements: dict[str, dict], *, mode: str, assets_root: Any) -> list[dict]:
    reports: list[dict] = []
    for role_name, role in roles.items():
        spawn = role.get('spawn_placement')
        if not isinstance(spawn, Mapping) or spawn.get('mode') != 'on_support_surface':
            continue
        support_role_name = spawn.get('support_role')
        if not isinstance(support_role_name, str):
            continue
        obj_name = role_map.get(str(role_name), str(role.get('bddl_inst') or role_name))
        support_name = role_map.get(str(support_role_name), str(support_role_name))
        obj = objects_by_name.get(obj_name)
        support = objects_by_name.get(support_name)
        support_role = roles.get(support_role_name)
        if obj is None or support is None or (not isinstance(support_role, Mapping)):
            continue
        before = tuple(obj.center)
        new_center, diagnostic = surface_placement.resolve_on_support_surface(role_name=str(role_name), role=role, support_role_name=str(support_role_name), support_role=support_role, object_center=obj.center, object_extent=obj.extent, support_center=support.center, support_extent=support.extent, mode=mode, assets_root=assets_root, clearance=float(spawn.get('clearance', _SPAWN_SUPPORT_CLEARANCE)))
        obj.center = new_center
        _clear_pose_metadata(obj)
        placements.setdefault(obj_name, {})['center'] = list(obj.center)
        placements[obj_name]['source'] = 'surface_placement_{}'.format(mode)
        placements[obj_name]['surface_placement'] = dict(diagnostic)
        diagnostic['object'] = obj_name
        diagnostic['support_object'] = support_name
        diagnostic['before'] = list(before)
        diagnostic['after'] = list(obj.center)
        reports.append(diagnostic)
    return reports

def _repair_pickable_overlaps(objects_by_name: Mapping[str, AABBObject], placements: dict[str, dict], *, protected_pairs: set[frozenset[str]], rng: random.Random, start_slot: int, bounds: Any) -> list[dict]:
    repairs: list[dict] = []
    placed: list[AABBObject] = []
    next_slot = int(start_slot)
    for obj in sorted(objects_by_name.values(), key=lambda item: item.name):
        if not _is_pickable(tuple(obj.abilities)):
            placed.append(obj)
            continue
        overlapping = [other.name for other in placed if frozenset((obj.name, other.name)) not in protected_pairs and _aabb_overlap(obj, other, tolerance=0.01)]
        if overlapping:
            before = tuple(obj.center)
            repaired_on_support = _repair_spawn_on_support(obj, objects_by_name, placements, placed, protected_pairs)
            if repaired_on_support:
                _clear_pose_metadata(obj)
                placements.setdefault(obj.name, {})['center'] = list(obj.center)
                placements[obj.name]['source'] = 'collision_repair_support_slot'
            else:
                for _ in range(32):
                    obj.center = _repair_pickable_center(obj.extent, next_slot, rng, bounds)
                    next_slot += 1
                    if not any((frozenset((obj.name, other.name)) not in protected_pairs and _aabb_overlap(obj, other, tolerance=0.01) for other in placed)):
                        break
                _clear_pose_metadata(obj)
                placements.setdefault(obj.name, {})['center'] = list(obj.center)
                placements[obj.name]['source'] = 'collision_repair_floor_slot'
            repairs.append({'object': obj.name, 'overlapped_with': overlapping, 'before': list(before), 'after': list(obj.center), 'source': placements[obj.name]['source']})
        placed.append(obj)
    return repairs

def _repair_spawn_on_support(obj: AABBObject, objects_by_name: Mapping[str, AABBObject], placements: dict[str, dict], placed: list[AABBObject], protected_pairs: set[frozenset[str]]) -> bool:
    spawn = (placements.get(obj.name) or {}).get('spawn_placement')
    if not isinstance(spawn, Mapping) or spawn.get('mode') != 'on_support_surface':
        return False
    support_name = spawn.get('support_object')
    if not isinstance(support_name, str):
        return False
    support = objects_by_name.get(support_name)
    if support is None:
        return False
    start_slot = int(spawn.get('slot') or 0) + 1
    for attempt in range(32):
        obj.center = _support_surface_center(obj.extent, support, start_slot + attempt)
        _clear_pose_metadata(obj)
        if not any((frozenset((obj.name, other.name)) not in protected_pairs and _aabb_overlap(obj, other, tolerance=0.01) for other in placed)):
            spawn['slot'] = start_slot + attempt
            spawn['offset'] = [float(obj.center[0] - support.center[0]), float(obj.center[1] - support.center[1]), float(obj.center[2] - support.center[2])]
            return True
    return False

def _support_surface_center(extent: tuple, support: AABBObject, slot: int) -> tuple[float, float, float]:
    usable_x = max(float(support.extent[0]) * 0.5 - float(extent[0]) * 0.5 - _SPAWN_SUPPORT_MARGIN, 0.0)
    usable_y = max(float(support.extent[1]) * 0.5 - float(extent[1]) * 0.5 - _SPAWN_SUPPORT_MARGIN, 0.0)
    direction = int(slot) % 4
    lane = int(slot) // 4
    lane_shift_x = min(usable_x, lane * max(float(extent[0]) + _SPAWN_SUPPORT_MARGIN, 0.08))
    lane_shift_y = min(usable_y, lane * max(float(extent[1]) + _SPAWN_SUPPORT_MARGIN, 0.08))
    if direction == 0:
        offset_x, offset_y = (usable_x, -lane_shift_y)
    elif direction == 1:
        offset_x, offset_y = (-usable_x, lane_shift_y)
    elif direction == 2:
        offset_x, offset_y = (-lane_shift_x, usable_y)
    else:
        offset_x, offset_y = (lane_shift_x, -usable_y)
    top_z = float(support.center[2]) + float(support.extent[2]) * 0.5
    return (float(support.center[0] + offset_x), float(support.center[1] + offset_y), float(top_z + float(extent[2]) * 0.5 + _SPAWN_SUPPORT_CLEARANCE))

def _repair_pickable_center(extent: tuple, slot: int, rng: random.Random, bounds: Any) -> tuple:
    parsed = _bounds4(bounds)
    if parsed is None:
        return _standalone_pickable_center(extent, slot, rng)
    min_x, min_y, max_x, max_y = parsed
    margin = 0.7
    usable_x = max(max_x - min_x - margin * 2.0, 0.5)
    usable_y = max(max_y - min_y - margin * 2.0, 0.5)
    x = min_x + margin + slot * 0.57 % usable_x + rng.uniform(-0.015, 0.015)
    y = min_y + margin + slot * 0.41 % usable_y
    z = _DEFAULT_FLOOR_Z + extent[2] * 0.5
    return (float(x), float(y), float(z))

def _bounds4(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        min_x, min_y, max_x, max_y = [float(v) for v in value]
    except Exception:
        return None
    if max_x <= min_x or max_y <= min_y:
        return None
    return (min_x, min_y, max_x, max_y)

def _aabb_overlap(a: AABBObject, b: AABBObject, *, tolerance: float) -> bool:
    for axis in range(3):
        delta = abs(float(a.center[axis]) - float(b.center[axis]))
        limit = float(a.extent[axis]) * 0.5 + float(b.extent[axis]) * 0.5 - float(tolerance)
        if delta >= limit:
            return False
    return True
__all__ = ['SceneInitResult', 'build']
