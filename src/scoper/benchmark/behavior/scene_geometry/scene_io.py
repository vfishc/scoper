from __future__ import annotations
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

@dataclass(frozen=True)
class SceneObjectRecord:
    name: str
    category: str
    model: str
    rooms: tuple[str, ...]
    pos: tuple[float, float, float]
    ori: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    fixed_base: bool
    visual_only: bool
    joint_pos: tuple[float, ...]
    joint_vel: tuple[float, ...]

    @property
    def primary_room(self) -> str | None:
        return self.rooms[0] if self.rooms else None

    @property
    def room_type(self) -> str | None:
        room = self.primary_room
        if room is None:
            return None
        return room_type_from_instance(room)

def load_scene_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_scene_json(path: str | Path, scene: Mapping[str, Any]) -> None:
    scene = ensure_og_state_velocity_fields(clone_scene(scene))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, indent=2, sort_keys=False), encoding='utf-8')

def ensure_og_state_velocity_fields(scene: dict[str, Any]) -> dict[str, Any]:
    registry = mutable_object_registry(scene)
    for state in registry.values():
        if not isinstance(state, dict):
            continue
        root = state.get('root_link')
        if not isinstance(root, dict):
            continue
        root.setdefault('lin_vel', [0.0, 0.0, 0.0])
        root.setdefault('ang_vel', [0.0, 0.0, 0.0])
    return scene

def parse_scene_objects(scene: Mapping[str, Any]) -> dict[str, SceneObjectRecord]:
    init_info = scene.get('objects_info', {}).get('init_info', {}) or {}
    registry = scene.get('state', {}).get('registry', {}).get('object_registry', {}) or {}
    records: dict[str, SceneObjectRecord] = {}
    for name, info in init_info.items():
        args = dict(info.get('args') or {})
        state = dict(registry.get(name) or {})
        root = dict(state.get('root_link') or {})
        category = str(args.get('category') or '')
        model = str(args.get('model') or '')
        pos = _float_tuple(root.get('pos'), 3, default=(0.0, 0.0, 0.0))
        ori = _float_tuple(root.get('ori'), 4, default=(0.0, 0.0, 0.0, 1.0))
        scale = _float_tuple(args.get('scale'), 3, default=(1.0, 1.0, 1.0))
        joint_pos = _float_sequence(state.get('joint_pos'))
        joint_vel = _float_sequence(state.get('joint_vel'))
        rooms = tuple((str(room) for room in args.get('in_rooms') or [] if str(room)))
        records[str(name)] = SceneObjectRecord(name=str(name), category=category, model=model, rooms=rooms, pos=pos, ori=ori, scale=scale, fixed_base=bool(args.get('fixed_base', False)), visual_only=bool(args.get('visual_only', False)), joint_pos=joint_pos, joint_vel=joint_vel)
    return records

def clone_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(scene))

def init_info(scene: Mapping[str, Any]) -> Mapping[str, Any]:
    return scene.get('objects_info', {}).get('init_info', {}) or {}

def mutable_init_info(scene: dict[str, Any]) -> dict[str, Any]:
    return scene.setdefault('objects_info', {}).setdefault('init_info', {})

def mutable_object_registry(scene: dict[str, Any]) -> dict[str, Any]:
    return scene.setdefault('state', {}).setdefault('registry', {}).setdefault('object_registry', {})

def add_dataset_object(scene: dict[str, Any], *, name: str, category: str, model: str, room: str, pos: tuple[float, float, float], ori: tuple[float, float, float, float], fixed_base: bool=False, visual_only: bool=False, scale: tuple[float, float, float]=(1.0, 1.0, 1.0)) -> None:
    mutable_init_info(scene)[name] = {'class_module': 'omnigibson.objects.dataset_object', 'class_name': 'DatasetObject', 'args': {'name': name, 'category': category, 'model': model, 'scale': [float(v) for v in scale], 'fixed_base': bool(fixed_base), 'visual_only': bool(visual_only), 'in_rooms': [str(room)], 'bddl_object_scope': None}}
    mutable_object_registry(scene)[name] = {'is_asleep': False, 'root_link': {'pos': [float(v) for v in pos], 'ori': [float(v) for v in ori], 'lin_vel': [0.0, 0.0, 0.0], 'ang_vel': [0.0, 0.0, 0.0]}, 'joint_pos': [], 'joint_vel': [], 'non_kin': {'Temperature': {'temperature': 23.0}}}

def remove_object(scene: dict[str, Any], name: str) -> None:
    mutable_init_info(scene).pop(str(name), None)
    mutable_object_registry(scene).pop(str(name), None)

def set_object_model(scene: dict[str, Any], name: str, model: str) -> None:
    args = mutable_init_info(scene)[name].setdefault('args', {})
    args['model'] = str(model)
    args.pop('expected_file_hash', None)

def set_object_pose(scene: dict[str, Any], name: str, *, pos: tuple[float, float, float] | None=None, ori: tuple[float, float, float, float] | None=None) -> None:
    root = mutable_object_registry(scene)[name].setdefault('root_link', {})
    if pos is not None:
        root['pos'] = [float(v) for v in pos]
    if ori is not None:
        root['ori'] = [float(v) for v in ori]

def set_object_room(scene: dict[str, Any], name: str, room: str) -> None:
    args = mutable_init_info(scene)[name].setdefault('args', {})
    args['in_rooms'] = [str(room)]

def ensure_scene_augmentation_metadata(scene: dict[str, Any]) -> dict[str, Any]:
    metadata = scene.setdefault('metadata', {})
    return metadata.setdefault('scoper.benchmark.behavior.scene_geometry', {})

def room_type_from_instance(room: str) -> str:
    text = str(room)
    head, sep, tail = text.rpartition('_')
    if sep and tail.isdigit():
        return head
    return text

def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = float(yaw) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))

def quat_to_yaw(ori: tuple[float, float, float, float]) -> float:
    x, y, z, w = ori
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def _float_tuple(value: Any, length: int, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return default
    try:
        return tuple((float(v) for v in value))
    except (TypeError, ValueError):
        return default

def _float_sequence(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        return tuple((float(v) for v in value))
    except (TypeError, ValueError):
        return ()
__all__ = ['SceneObjectRecord', 'add_dataset_object', 'clone_scene', 'ensure_scene_augmentation_metadata', 'ensure_og_state_velocity_fields', 'init_info', 'load_scene_json', 'mutable_init_info', 'mutable_object_registry', 'parse_scene_objects', 'quat_to_yaw', 'remove_object', 'room_type_from_instance', 'set_object_model', 'set_object_pose', 'set_object_room', 'write_scene_json', 'yaw_to_quat']
