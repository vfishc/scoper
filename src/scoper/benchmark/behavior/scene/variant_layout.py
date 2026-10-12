from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject
from scoper.benchmark.behavior.paths import repo_root, resolve_repo_path
from scoper.benchmark.behavior.scene.doors import is_door_like
from scoper.benchmark.behavior.scene.profiler import profile_scene
from scoper.benchmark.behavior.scene_geometry.geometry import footprint_extent_for_yaw
from scoper.benchmark.behavior.scene_geometry.scene_io import quat_to_yaw
STRUCTURE_CATEGORIES = frozenset({'floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'})

@dataclass
class VariantLayoutMergeResult:
    objects: list[AABBObject]
    report: dict[str, Any] = field(default_factory=dict)

def merge_source_json_layout(task_spec: Mapping[str, Any], role_objects: Sequence[AABBObject], role_map: Mapping[str, str], *, include_pickupable_background: bool=False, load_doors: bool=False) -> VariantLayoutMergeResult:
    role_objects = list(role_objects)
    source = task_spec.get('source') or {}
    source_json = source.get('source_json')
    if not source_json:
        return VariantLayoutMergeResult(objects=role_objects, report={'enabled': False, 'reason': 'missing_source_json', 'role_object_count': len(role_objects), 'final_object_count': len(role_objects)})
    path = _resolve_source_json(source_json)
    scene_model = str(source.get('scene_model') or '')
    if not scene_model:
        scene_model = _scene_model_from_path(path)
    profile = profile_scene(scene_model, scene_json=path, layout_variant_id=source.get('layout_variant_id'))
    role_object_id_to_name = {str(role.get('object_id')): str(role_map.get(str(role_name), role.get('bddl_inst') or role_name)) for role_name, role in (task_spec.get('roles') or {}).items() if isinstance(role, Mapping) and role.get('object_id')}
    role_object_ids = set(role_object_id_to_name)
    role_names = {str(obj.name) for obj in role_objects}
    role_objects_by_name = {str(obj.name): obj for obj in role_objects}
    merged = list(role_objects)
    imported: list[dict[str, Any]] = []
    updated_roles: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for obj in profile.get('objects') or []:
        object_id = str(obj.get('object_id') or '')
        if not object_id:
            skipped.append({'object_id': object_id, 'reason': 'missing_object_id'})
            continue
        category = str(obj.get('category') or '')
        if category in STRUCTURE_CATEGORIES:
            skipped.append({'object_id': object_id, 'category': category, 'reason': 'structure_category'})
            continue
        if not load_doors and is_door_like(category, object_id):
            skipped.append({'object_id': object_id, 'category': category, 'reason': 'door_loading_disabled'})
            continue
        root_position = _root_position_from_profile_object(obj)
        root_orientation_xyzw = _root_orientation_from_profile_object(obj)
        root_yaw = _root_yaw_from_profile_object(obj, root_orientation_xyzw)
        center, center_source = _aabb_center_from_profile_object(obj, root_position, root_orientation_xyzw)
        extent = _extent_from_profile_object(obj)
        if object_id in role_object_ids or object_id in role_names:
            role_name = role_object_id_to_name.get(object_id, object_id)
            role_obj = role_objects_by_name.get(role_name)
            if role_obj is not None and center is not None and (root_position is not None):
                extent_source = 'profile'
                if extent is None:
                    extent = _fallback_extent(category)
                    extent_source = 'fallback'
                role_obj.center = center
                role_obj.extent = extent
                role_obj.root_position = root_position
                role_obj.root_orientation_xyzw = root_orientation_xyzw
                role_obj.root_yaw = root_yaw
                role_obj.pose_source = 'source_json'
                states = dict(role_obj.states or {})
                states.setdefault('source_scene_object_id', object_id)
                states.setdefault('source_object_id', object_id)
                states.setdefault('scene_layout_source', 'source_json_full')
                role_obj.states = states
                updated_roles.append({'object_id': object_id, 'name': role_name, 'category': category, 'root_position': list(root_position), 'root_orientation_xyzw': list(root_orientation_xyzw) if root_orientation_xyzw is not None else None, 'root_yaw': root_yaw, 'aabb_center': list(center), 'aabb_extent': list(extent), 'aabb_center_source': center_source, 'extent_source': extent_source, 'pose_source': 'source_json'})
            skipped.append({'object_id': object_id, 'category': category, 'reason': 'already_represented_by_role'})
            continue
        if object_id.endswith('::faucet'):
            skipped.append({'object_id': object_id, 'category': category, 'reason': 'derived_alias'})
            continue
        if center is None or root_position is None:
            skipped.append({'object_id': object_id, 'category': category, 'reason': 'missing_geometry'})
            continue
        extent_source = 'profile'
        if extent is None:
            extent = _fallback_extent(category)
            extent_source = 'fallback'
        abilities = tuple((str(item) for item in obj.get('abilities') or []))
        pickupable = bool(obj.get('pickupable')) or 'pickupable' in abilities
        states = _states_from_profile_object(obj)
        if pickupable and (not include_pickupable_background):
            states['occupancy_ignored'] = True
        states.setdefault('scene_layout_source', 'source_json_full')
        if obj.get('room_id'):
            states.setdefault('room_id', str(obj.get('room_id')))
        name = _background_name(object_id, existing=role_names | {item.name for item in merged})
        merged.append(AABBObject(name=name, category=category or str(obj.get('object_type') or 'object'), center=center, extent=extent, abilities=abilities, states=states, root_position=root_position, root_orientation_xyzw=root_orientation_xyzw, root_yaw=root_yaw, pose_source='source_json'))
        imported.append({'object_id': object_id, 'name': name, 'category': category, 'room_id': obj.get('room_id'), 'pickupable': pickupable, 'occupancy_ignored': bool(states.get('occupancy_ignored')), 'extent_source': extent_source, 'root_position': list(root_position), 'root_orientation_xyzw': list(root_orientation_xyzw) if root_orientation_xyzw is not None else None, 'root_yaw': root_yaw, 'aabb_center': list(center), 'aabb_extent': list(extent), 'aabb_center_source': center_source, 'pose_source': 'source_json'})
    report = {'enabled': True, 'source_json': str(path.relative_to(repo_root()) if path.is_relative_to(repo_root()) else path), 'layout_variant_id': source.get('layout_variant_id'), 'scene_model': scene_model, 'role_object_count': len(role_objects), 'profile_object_count': int(profile.get('object_count') or 0), 'imported_background_object_count': len(imported), 'updated_role_object_count': len(updated_roles), 'skipped_background_object_count': len(skipped), 'final_object_count': len(merged), 'include_pickupable_background': bool(include_pickupable_background), 'load_doors': bool(load_doors), 'imported_background_objects': imported[:200], 'updated_role_objects': updated_roles[:200], 'skipped_background_objects': skipped[:200]}
    return VariantLayoutMergeResult(objects=merged, report=report)

def _resolve_source_json(source_json: Any) -> Path:
    return resolve_repo_path(str(source_json))

def _scene_model_from_path(path: Path) -> str:
    parts = path.parts
    if 'scenes' in parts:
        index = parts.index('scenes')
        if index + 1 < len(parts):
            return parts[index + 1]
    stem = path.stem
    if '_variant_' in stem:
        return stem.split('_variant_', 1)[0]
    if stem.endswith('_best'):
        return stem[:-len('_best')]
    return path.parent.parent.name if path.parent.name == 'json' else stem

def _root_position_from_profile_object(obj: Mapping[str, Any]) -> tuple[float, float, float] | None:
    root_position = obj.get('root_position')
    if isinstance(root_position, Mapping):
        try:
            return (float(root_position['x']), float(root_position['y']), float(root_position['z']))
        except Exception:
            pass
    position = obj.get('position')
    if not isinstance(position, Mapping):
        return None
    try:
        return (float(position['x']), float(position['z']), float(position['y']))
    except Exception:
        return None

def _root_orientation_from_profile_object(obj: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ('root_orientation_xyzw', 'orientation'):
        orientation = obj.get(key)
        if isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
            try:
                return tuple((float(v) for v in orientation[:4]))
            except Exception:
                return None
    return None

def _root_yaw_from_profile_object(obj: Mapping[str, Any], root_orientation_xyzw: tuple[float, float, float, float] | None) -> float | None:
    value = obj.get('root_yaw')
    if isinstance(value, (int, float)):
        return float(value)
    if root_orientation_xyzw is None:
        return None
    try:
        return float(quat_to_yaw(root_orientation_xyzw))
    except Exception:
        return None

def _aabb_center_from_profile_object(obj: Mapping[str, Any], root_position: tuple[float, float, float] | None, root_orientation_xyzw: tuple[float, float, float, float] | None) -> tuple[tuple[float, float, float] | None, str]:
    if root_position is None:
        return (None, 'missing_root_position')
    offset = _scaled_base_link_offset_from_profile_object(obj)
    if offset is not None:
        world_offset = _rotate_xyzw(offset, root_orientation_xyzw)
        return ((float(root_position[0]) + float(world_offset[0]), float(root_position[1]) + float(world_offset[1]), float(root_position[2]) + float(world_offset[2])), 'metadata_base_link_offset')
    return (root_position, 'fallback_root_position_legacy')

def _scaled_base_link_offset_from_profile_object(obj: Mapping[str, Any]) -> tuple[float, float, float] | None:
    offset = obj.get('base_link_offset')
    if not isinstance(offset, (list, tuple)) or len(offset) < 3:
        return None
    try:
        raw = [float(value) for value in offset[:3]]
    except Exception:
        return None
    scale = obj.get('scale')
    if isinstance(scale, (list, tuple)) and len(scale) >= 3:
        try:
            return tuple((float(value) * float(factor) for value, factor in zip(raw, scale[:3])))
        except Exception:
            pass
    return (raw[0], raw[1], raw[2])

def _rotate_xyzw(vector: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float] | None) -> tuple[float, float, float]:
    if quat_xyzw is None:
        return (float(vector[0]), float(vector[1]), float(vector[2]))
    x, y, z, w = (float(value) for value in quat_xyzw)
    vx, vy, vz = (float(value) for value in vector)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return (vx, vy, vz)
    x, y, z, w = (x / norm, y / norm, z / norm, w / norm)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx))

def _extent_from_profile_object(obj: Mapping[str, Any]) -> tuple[float, float, float] | None:
    extent = obj.get('extent')
    if not isinstance(extent, list) or len(extent) < 3:
        return None
    try:
        x_extent = float(extent[0])
        y_extent = float(extent[1])
        z_extent = float(extent[2])
    except Exception:
        return None
    orientation = obj.get('root_orientation_xyzw')
    if not isinstance(orientation, (list, tuple)):
        orientation = obj.get('orientation')
    if isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
        try:
            yaw = quat_to_yaw(tuple((float(v) for v in orientation[:4])))
            x_extent, y_extent = footprint_extent_for_yaw((x_extent, y_extent), yaw)
        except Exception:
            pass
    return (float(x_extent), float(y_extent), float(z_extent))

def _fallback_extent(category: str) -> tuple[float, float, float]:
    text = str(category).lower()
    if any((token in text for token in ('table', 'counter', 'desk'))):
        return (0.9, 0.8, 0.75)
    if any((token in text for token in ('cabinet', 'shelf', 'bookcase', 'wardrobe'))):
        return (0.8, 0.5, 1.2)
    if any((token in text for token in ('sofa', 'couch'))):
        return (1.8, 0.9, 0.8)
    if 'bed' in text:
        return (2.0, 1.5, 0.6)
    if 'chair' in text:
        return (0.5, 0.5, 0.9)
    return (0.4, 0.4, 0.4)

def _states_from_profile_object(obj: Mapping[str, Any]) -> dict[str, Any]:
    states: dict[str, Any] = {}
    raw = obj.get('states')
    if isinstance(raw, Mapping):
        states.update({str(key): value for key, value in raw.items()})
    for source_key, state_key in (('is_open', 'open'), ('is_toggled', 'toggled_on'), ('is_sliced', 'sliced'), ('is_dirty', 'dirty')):
        value = obj.get(source_key)
        if isinstance(value, bool):
            states[state_key] = value
    return states

def _background_name(object_id: str, *, existing: set[str]) -> str:
    name = str(object_id)
    if name not in existing:
        return name
    base = 'scene__{}'.format(name)
    candidate = base
    index = 1
    while candidate in existing:
        index += 1
        candidate = '{}_{}'.format(base, index)
    return candidate
__all__ = ['VariantLayoutMergeResult', 'merge_source_json_layout']
