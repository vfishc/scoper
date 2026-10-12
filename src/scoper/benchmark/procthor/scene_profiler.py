from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from scoper.benchmark.procthor.house_registry import load_house_registry
from scoper.benchmark.procthor.paths import resolve_repo_path
from scoper.benchmark.procthor.procthor_dataset_assets import is_prior_house_uri, load_prior_house
PROFILE_VERSION = '0.1'
AFFORDANCE_FIELDS = ('pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable', 'dirtyable', 'breakable', 'cookable', 'heatable', 'coolable')
REQUIRED_SCENE_PROFILE_FIELDS = ('house_id', 'source', 'profile_version', 'rooms', 'objects', 'affordance_index', 'object_type_index', 'reachable')
REQUIRED_SCENE_OBJECT_FIELDS = ('object_id', 'object_type', 'room_id', 'position', 'pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable', 'parent_receptacles')

class SceneProfileError(ValueError):
    pass

def profile_house(registry_entry: Mapping[str, Any], width: int=320, height: int=240, cloud_rendering: bool=True, gpu_device: Optional[int]=None, sample_poses: int=20, seed: int=0) -> dict:
    house = load_house_json(registry_entry['house_path'])
    controller = _create_controller(house=house, width=width, height=height, cloud_rendering=cloud_rendering, gpu_device=gpu_device)
    try:
        event = controller.step(action='Pass')
        if not bool(event.metadata.get('lastActionSuccess', True)):
            raise SceneProfileError("Initial Pass action failed for house '{}': {}".format(registry_entry['house_id'], event.metadata.get('errorMessage', '')))
        reachable_positions = query_reachable_positions(controller)
        return build_scene_profile(registry_entry=registry_entry, house=house, metadata=event.metadata, reachable_positions=reachable_positions, sample_poses=sample_poses, seed=seed)
    finally:
        controller.stop()

def load_house_json(path: str) -> dict:
    if is_prior_house_uri(path):
        return load_prior_house(path)
    try:
        house = json.loads(resolve_repo_path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise SceneProfileError("Invalid house JSON '{}': {}.".format(path, exc)) from exc
    if not isinstance(house, dict):
        raise SceneProfileError("House JSON '{}' must contain one object.".format(path))
    return house

def query_reachable_positions(controller: Any) -> List[dict]:
    event = controller.step(action='GetReachablePositions')
    if not bool(event.metadata.get('lastActionSuccess', True)):
        raise SceneProfileError('GetReachablePositions failed: {}'.format(event.metadata.get('errorMessage', '')))
    positions = event.metadata.get('actionReturn') or []
    if not isinstance(positions, list):
        raise SceneProfileError('GetReachablePositions returned a non-list value.')
    return [dict(position) for position in positions]

def build_scene_profile(registry_entry: Mapping[str, Any], house: Mapping[str, Any], metadata: Mapping[str, Any], reachable_positions: Sequence[Mapping[str, Any]], sample_poses: int=20, seed: int=0) -> dict:
    rooms = _profile_rooms(house, reachable_positions=reachable_positions)
    objects = [_profile_object(obj, house=house, rooms=rooms, reachable_positions=reachable_positions) for obj in metadata.get('objects', []) if _has_object_id(obj)]
    profile = {'house_id': registry_entry['house_id'], 'source': 'procthor', 'profile_version': PROFILE_VERSION, 'split': registry_entry.get('split'), 'house_path': registry_entry.get('house_path'), 'rooms': rooms, 'objects': objects, 'affordance_index': _build_affordance_index(objects), 'object_type_index': _build_object_type_index(objects), 'reachable': {'num_positions': len(reachable_positions), 'positions': [dict(position) for position in reachable_positions], 'sampled_agent_poses': sample_agent_poses(reachable_positions, limit=sample_poses, seed=seed)}}
    validate_scene_profile(profile)
    return profile

def sample_agent_poses(reachable_positions: Sequence[Mapping[str, Any]], limit: int=20, seed: int=0) -> List[dict]:
    positions = [dict(position) for position in reachable_positions]
    if limit <= 0 or not positions:
        return []
    rng = random.Random(seed)
    sample_size = min(limit, len(positions))
    indices = sorted(rng.sample(range(len(positions)), sample_size))
    poses = []
    for index in indices:
        position = positions[index]
        poses.append({'position': {'x': float(position.get('x', 0.0)), 'y': float(position.get('y', 0.0)), 'z': float(position.get('z', 0.0))}, 'rotation': {'y': 0.0}, 'horizon': 0.0})
    return poses

def validate_scene_profile(profile: Mapping[str, Any]) -> None:
    if not isinstance(profile, Mapping):
        raise SceneProfileError('SceneProfile must be a mapping.')
    _require_fields(profile, REQUIRED_SCENE_PROFILE_FIELDS, 'SceneProfile')
    if not isinstance(profile['house_id'], str) or not profile['house_id']:
        raise SceneProfileError("SceneProfile field 'house_id' must be a string.")
    if profile['source'] != 'procthor':
        raise SceneProfileError("SceneProfile field 'source' must be 'procthor'.")
    if not isinstance(profile['profile_version'], str) or not profile['profile_version']:
        raise SceneProfileError("SceneProfile field 'profile_version' must be a string.")
    if not isinstance(profile['rooms'], list):
        raise SceneProfileError("SceneProfile field 'rooms' must be a list.")
    if not isinstance(profile['objects'], list):
        raise SceneProfileError("SceneProfile field 'objects' must be a list.")
    if not isinstance(profile['affordance_index'], Mapping):
        raise SceneProfileError("SceneProfile field 'affordance_index' must be a mapping.")
    if not isinstance(profile['object_type_index'], Mapping):
        raise SceneProfileError("SceneProfile field 'object_type_index' must be a mapping.")
    if not isinstance(profile['reachable'], Mapping):
        raise SceneProfileError("SceneProfile field 'reachable' must be a mapping.")
    if not isinstance(profile['reachable'].get('num_positions'), int):
        raise SceneProfileError('SceneProfile reachable.num_positions must be an int.')
    if not isinstance(profile['reachable'].get('sampled_agent_poses', []), list):
        raise SceneProfileError('SceneProfile reachable.sampled_agent_poses must be a list.')
    for index, obj in enumerate(profile['objects']):
        _validate_profile_object(obj, index)
    for index, room in enumerate(profile['rooms']):
        _validate_profile_room(room, index)

def write_profiles_jsonl(profiles: Iterable[Mapping[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        for profile in profiles:
            validate_scene_profile(profile)
            json.dump(profile, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')

def main(argv: Optional[Sequence[str]]=None) -> None:
    parser = argparse.ArgumentParser(description='Profile ProcTHOR houses.')
    parser.add_argument('--house-registry', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--sample-poses', type=int, default=20)
    parser.add_argument('--no-cloud-rendering', action='store_true', help="Use AI2-THOR's default platform instead of CloudRendering.")
    parser.add_argument('--gpu-device', type=int, default=None, help='Optional CUDA GPU index to pass to AI2-THOR. With CloudRendering, AI2-THOR uses vulkaninfo to map this to a Vulkan device index.')
    args = parser.parse_args(argv)
    entries = load_house_registry(args.house_registry)
    if args.limit is not None:
        entries = entries[:args.limit]
    profiles = []
    failures = []
    for entry in entries:
        try:
            profiles.append(profile_house(entry, width=args.width, height=args.height, cloud_rendering=not args.no_cloud_rendering, gpu_device=args.gpu_device, sample_poses=args.sample_poses, seed=args.seed))
        except Exception as exc:
            failures.append({'house_id': entry.get('house_id'), 'error': str(exc)})
    write_profiles_jsonl(profiles, args.output)
    summary = {'num_requested': len(entries), 'num_profiled': len(profiles), 'num_failed': len(failures), 'failures': failures, 'output': args.output}
    summary_path = str(Path(args.output).with_suffix(Path(args.output).suffix + '.summary.json'))
    Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)

def _create_controller(house: Mapping[str, Any], width: int, height: int, cloud_rendering: bool, gpu_device: Optional[int]=None) -> Any:
    from ai2thor.controller import Controller
    kwargs = {'scene': dict(house), 'width': width, 'height': height, 'renderDepthImage': False, 'renderInstanceSegmentation': False}
    if cloud_rendering:
        from ai2thor.platform import CloudRendering
        kwargs['platform'] = CloudRendering
    if gpu_device is not None:
        kwargs['gpu_device'] = gpu_device
    executable = os.environ.get('EB_PROCTHOR_LOCAL_EXECUTABLE')
    if executable:
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError('Configured local Unity executable is unavailable')
        kwargs['local_executable_path'] = executable
        if cloud_rendering:

            class LocalCloudRenderingController(Controller):

                def start(controller, *args, **start_kwargs):
                    controller._build.platform = CloudRendering
                    return super().start(*args, **start_kwargs)
            return LocalCloudRenderingController(**kwargs)
    elif os.environ.get('EB_TASK_BANK_DISTINCT_REFERENCES') == '1':
        raise RuntimeError('Identity revision requires an explicit offline Unity executable')
    return Controller(**kwargs)

def _profile_object(obj: Mapping[str, Any], house: Optional[Mapping[str, Any]]=None, rooms: Optional[Sequence[Mapping[str, Any]]]=None, reachable_positions: Optional[Sequence[Mapping[str, Any]]]=None) -> dict:
    object_id = _object_id(obj)
    object_type = _object_type(obj, object_id)
    position = dict(obj.get('position') or {})
    pickupable = bool(obj.get('pickupable', False))
    receptacle = bool(obj.get('receptacle', False))
    cookable = _optional_bool(obj.get('cookable'))
    heatable_types = {'Apple', 'Bread', 'Egg', 'Potato', 'Tomato', 'Lettuce', 'Mug', 'Cup', 'Pot', 'Pan', 'Plate', 'Bowl'}
    heatable = cookable is True or object_type in heatable_types
    coolable_types = {'Apple', 'Bread', 'Egg', 'Potato', 'Tomato', 'Lettuce', 'Mug', 'Cup', 'Bottle', 'Bowl', 'Pot', 'Pan'}
    coolable = receptacle or object_type in coolable_types
    profile_obj = {'object_id': object_id, 'object_type': object_type, 'room_id': _infer_room_id(obj, position=position, house=house, rooms=rooms, reachable_positions=reachable_positions), 'position': position, 'pickupable': pickupable, 'receptacle': receptacle, 'openable': bool(obj.get('openable', False)), 'toggleable': bool(obj.get('toggleable', False)), 'sliceable': bool(obj.get('sliceable', False)), 'dirtyable': _optional_bool(obj.get('dirtyable')), 'breakable': _optional_bool(obj.get('breakable')), 'cookable': cookable, 'heatable': heatable, 'coolable': coolable, 'visible_from_initial_pose': _optional_bool(obj.get('visible')), 'distance_from_initial_pose': _optional_float(obj.get('distance')), 'parent_receptacles': list(obj.get('parentReceptacles') or []), 'is_open': _optional_bool(obj.get('isOpen')), 'is_toggled': _optional_bool(obj.get('isToggled')), 'is_sliced': _optional_bool(obj.get('isSliced')), 'is_dirty': _optional_bool(obj.get('isDirty'))}
    asset_id = obj.get('assetId')
    if isinstance(asset_id, str) and asset_id:
        profile_obj['assetId'] = asset_id
    return profile_obj

def _profile_rooms(house: Mapping[str, Any], reachable_positions: Optional[Sequence[Mapping[str, Any]]]=None) -> List[dict]:
    rooms = []
    for index, room in enumerate(house.get('rooms', []) or []):
        if not isinstance(room, Mapping):
            continue
        room_id = room.get('id') or room.get('roomId') or room.get('room_id') or 'room_{}'.format(index)
        representative_pose = _representative_reachable_pose(room_id, room, reachable_positions or [])
        entry = {'room_id': room_id, 'room_type': room.get('roomType') or room.get('room_type') or room.get('type'), 'representative_reachable_pose': representative_pose}
        floor_polygon = room.get('floorPolygon') or room.get('floor_polygon')
        if isinstance(floor_polygon, list) and len(floor_polygon) >= 3:
            entry['floor_polygon'] = [{'x': float(vertex['x']), 'z': float(vertex['z'])} for vertex in floor_polygon if isinstance(vertex, Mapping) and 'x' in vertex and ('z' in vertex)]
            if len(entry['floor_polygon']) < 3:
                entry.pop('floor_polygon', None)
        rooms.append(entry)
    return rooms

def _infer_room_id(obj: Mapping[str, Any], position: Mapping[str, Any], house: Optional[Mapping[str, Any]]=None, rooms: Optional[Sequence[Mapping[str, Any]]]=None, reachable_positions: Optional[Sequence[Mapping[str, Any]]]=None) -> Optional[str]:
    metadata_room_id = obj.get('roomId') or obj.get('room_id')
    if isinstance(metadata_room_id, str) and metadata_room_id:
        return metadata_room_id
    if 'x' in position and 'z' in position and (house is not None):
        for index, room in enumerate(house.get('rooms', []) or []):
            if not isinstance(room, Mapping):
                continue
            polygon = room.get('floorPolygon') or room.get('floor_polygon')
            if _point_in_floor_polygon(position, polygon):
                room_id = room.get('id') or room.get('roomId') or room.get('room_id') or 'room_{}'.format(index)
                return str(room_id)
    nearest = _nearest_reachable_position(position, reachable_positions or [])
    if nearest is not None:
        room_id = nearest.get('roomId') or nearest.get('room_id')
        if isinstance(room_id, str) and room_id:
            return room_id
        if rooms:
            for room in rooms:
                raw_room = _room_by_id(house or {}, room.get('room_id'))
                if raw_room and _point_in_floor_polygon(nearest, raw_room.get('floorPolygon') or raw_room.get('floor_polygon')):
                    return str(room.get('room_id'))
    return None

def _representative_reachable_pose(room_id: str, room: Mapping[str, Any], reachable_positions: Sequence[Mapping[str, Any]]) -> Optional[dict]:
    room_positions = [position for position in reachable_positions if _reachable_position_in_room(position, room_id, room)]
    if not room_positions:
        return None
    center_x = sum((float(position.get('x', 0.0)) for position in room_positions)) / len(room_positions)
    center_z = sum((float(position.get('z', 0.0)) for position in room_positions)) / len(room_positions)
    center = {'x': center_x, 'z': center_z}
    best = min(room_positions, key=lambda position: _squared_xz_distance(position, center))
    return {'position': {'x': float(best.get('x', 0.0)), 'y': float(best.get('y', 0.0)), 'z': float(best.get('z', 0.0))}, 'rotation': {'y': 0.0}, 'horizon': 0.0}

def _reachable_position_in_room(position: Mapping[str, Any], room_id: str, room: Mapping[str, Any]) -> bool:
    position_room_id = position.get('roomId') or position.get('room_id')
    if isinstance(position_room_id, str) and position_room_id == room_id:
        return True
    polygon = room.get('floorPolygon') or room.get('floor_polygon')
    return _point_in_floor_polygon(position, polygon)

def _point_in_floor_polygon(point: Mapping[str, Any], polygon: Any) -> bool:
    if 'x' not in point or 'z' not in point:
        return False
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    vertices = []
    for vertex in polygon:
        if not isinstance(vertex, Mapping):
            return False
        if 'x' not in vertex or 'z' not in vertex:
            return False
        vertices.append((float(vertex['x']), float(vertex['z'])))
    x = float(point['x'])
    z = float(point['z'])
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, zi = vertices[i]
        xj, zj = vertices[j]
        if _point_on_segment(x, z, xi, zi, xj, zj):
            return True
        crosses = (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi or 1e-12) + xi
        if crosses:
            inside = not inside
        j = i
    return inside

def _point_on_segment(x: float, z: float, x1: float, z1: float, x2: float, z2: float) -> bool:
    cross = (x - x1) * (z2 - z1) - (z - z1) * (x2 - x1)
    if abs(cross) > 1e-09:
        return False
    dot = (x - x1) * (x2 - x1) + (z - z1) * (z2 - z1)
    if dot < -1e-09:
        return False
    squared_len = (x2 - x1) ** 2 + (z2 - z1) ** 2
    return dot <= squared_len + 1e-09

def _nearest_reachable_position(position: Mapping[str, Any], reachable_positions: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if 'x' not in position or 'z' not in position or (not reachable_positions):
        return None
    return min(reachable_positions, key=lambda item: _squared_xz_distance(position, item))

def _room_by_id(house: Mapping[str, Any], room_id: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(room_id, str):
        return None
    for index, room in enumerate(house.get('rooms', []) or []):
        if not isinstance(room, Mapping):
            continue
        candidate_id = room.get('id') or room.get('roomId') or room.get('room_id') or 'room_{}'.format(index)
        if candidate_id == room_id:
            return room
    return None

def _squared_xz_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return (float(a.get('x', 0.0)) - float(b.get('x', 0.0))) ** 2 + (float(a.get('z', 0.0)) - float(b.get('z', 0.0))) ** 2

def _build_affordance_index(objects: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {field: [] for field in AFFORDANCE_FIELDS}
    for obj in objects:
        object_id = obj['object_id']
        for field in AFFORDANCE_FIELDS:
            if bool(obj.get(field, False)):
                index[field].append(object_id)
    return index

def _build_object_type_index(objects: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for obj in objects:
        index.setdefault(obj['object_type'], []).append(obj['object_id'])
    return index

def _validate_profile_object(obj: Any, index: int) -> None:
    if not isinstance(obj, Mapping):
        raise SceneProfileError('SceneProfile object at index {} must be a mapping.'.format(index))
    _require_fields(obj, REQUIRED_SCENE_OBJECT_FIELDS, 'SceneProfile object {}'.format(index))
    for field in ('object_id', 'object_type'):
        if not isinstance(obj[field], str) or not obj[field]:
            raise SceneProfileError("SceneProfile object {} field '{}' must be a string.".format(index, field))
    if obj['room_id'] is not None and (not isinstance(obj['room_id'], str)):
        raise SceneProfileError("SceneProfile object {} field 'room_id' must be a string or null.".format(index))
    if not isinstance(obj['position'], Mapping):
        raise SceneProfileError("SceneProfile object {} field 'position' must be a mapping.".format(index))
    for field in ('pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable'):
        if not isinstance(obj[field], bool):
            raise SceneProfileError("SceneProfile object {} field '{}' must be a bool.".format(index, field))
    if not isinstance(obj['parent_receptacles'], list):
        raise SceneProfileError("SceneProfile object {} field 'parent_receptacles' must be a list.".format(index))

def _validate_profile_room(room: Any, index: int) -> None:
    if not isinstance(room, Mapping):
        raise SceneProfileError('SceneProfile room at index {} must be a mapping.'.format(index))
    _require_fields(room, ('room_id',), 'SceneProfile room {}'.format(index))
    if not isinstance(room['room_id'], str) or not room['room_id']:
        raise SceneProfileError("SceneProfile room {} field 'room_id' must be a string.".format(index))
    pose = room.get('representative_reachable_pose')
    if pose is None:
        return
    if not isinstance(pose, Mapping):
        raise SceneProfileError('SceneProfile room {} representative_reachable_pose must be a mapping or null.'.format(index))
    if not isinstance(pose.get('position'), Mapping):
        raise SceneProfileError('SceneProfile room {} representative_reachable_pose.position must be a mapping.'.format(index))
    for key in ('x', 'y', 'z'):
        if key not in pose['position']:
            raise SceneProfileError('SceneProfile room {} representative pose missing position.{}.'.format(index, key))

def _require_fields(mapping: Mapping[str, Any], required_fields: Iterable[str], context: str) -> None:
    missing = [field for field in required_fields if field not in mapping]
    if missing:
        raise SceneProfileError('{} missing required fields: {}.'.format(context, ', '.join(missing)))

def _has_object_id(obj: Any) -> bool:
    if not isinstance(obj, Mapping):
        return False
    return isinstance(obj.get('objectId'), str) and bool(obj.get('objectId'))

def _object_id(obj: Mapping[str, Any]) -> str:
    object_id = obj.get('objectId')
    if not isinstance(object_id, str) or not object_id:
        raise SceneProfileError('Object metadata is missing objectId.')
    return object_id

def _object_type(obj: Mapping[str, Any], object_id: str) -> str:
    object_type = obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    return object_id.split('|', 1)[0]

def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)

def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
if __name__ == '__main__':
    main()
__all__ = ['AFFORDANCE_FIELDS', 'PROFILE_VERSION', 'SceneProfileError', 'build_scene_profile', 'load_house_json', 'profile_house', 'query_reachable_positions', 'sample_agent_poses', 'validate_scene_profile', 'write_profiles_jsonl']
