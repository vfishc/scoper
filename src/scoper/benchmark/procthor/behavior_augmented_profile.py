from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Optional
from scoper.benchmark.procthor.scene_profiler import validate_scene_profile
BEHAVIOR_TO_PROCTHOR_TYPES = {'apple': 'Apple', 'bowl': 'Bowl', 'cabinet': 'Cabinet', 'bottom_cabinet': 'Cabinet', 'chair': 'Chair', 'countertop': 'CounterTop', 'desk': 'Desk', 'drawer': 'Drawer', 'faucet': 'Faucet', 'fridge': 'Fridge', 'knife': 'Knife', 'microwave': 'Microwave', 'mug': 'Mug', 'plate': 'Plate', 'shelf': 'Shelf', 'sink': 'Sink', 'sink_basin': 'SinkBasin', 'sofa': 'Sofa', 'table': 'DiningTable'}

def profile_from_behavior_augmented_labels(labels_json: str | Path, *, scene_json: str | Path, house_path: str, house_id: Optional[str]=None, split: str='behavior_augmented', occupancy_map_path: Optional[str | Path]=None, validate: bool=True) -> dict:
    labels_path = Path(labels_json)
    data = json.loads(labels_path.read_text(encoding='utf-8'))
    scene_path = Path(scene_json)
    scene_model = str(data.get('scene_model') or scene_path.stem.split('_variant_', 1)[0])
    variant_id = scene_path.stem
    objects = []
    rooms_by_id = {}
    for label in data.get('labels') or []:
        if not isinstance(label, Mapping):
            continue
        category = str(label.get('category') or '')
        object_type = BEHAVIOR_TO_PROCTHOR_TYPES.get(category.lower())
        if object_type is None:
            continue
        room_ids = [str(room) for room in label.get('rooms') or [] if str(room)]
        room_id = room_ids[0] if room_ids else 'room_0'
        room_type = str(label.get('room_type') or room_id.split('_', 1)[0] or 'Room')
        if room_type.endswith('_'):
            room_type = room_type[:-1]
        rooms_by_id.setdefault(room_id, {'room_id': room_id, 'room_type': room_type})
        pos = label.get('pos') or [0.0, 0.0, 0.0]
        pickupable = object_type in {'Apple', 'Bowl', 'Knife', 'Mug', 'Plate'}
        interactive = pickupable or object_type in {'Cabinet', 'Drawer', 'Faucet', 'Fridge', 'Microwave'}
        objects.append({'object_id': str(label.get('name') or '{}_{}'.format(category, len(objects))), 'object_type': object_type, 'room_id': room_id, 'position': {'x': float(pos[0]) if len(pos) > 0 else 0.0, 'y': float(pos[2]) if len(pos) > 2 else 0.9, 'z': float(pos[1]) if len(pos) > 1 else 0.0}, 'pickupable': pickupable, 'receptacle': object_type in {'Bowl', 'Cabinet', 'CounterTop', 'Desk', 'DiningTable', 'Drawer', 'Fridge', 'Microwave', 'Shelf', 'Sink', 'SinkBasin'}, 'openable': object_type in {'Cabinet', 'Drawer', 'Fridge', 'Microwave'}, 'toggleable': object_type in {'Faucet', 'Microwave'}, 'sliceable': object_type in {'Apple'}, 'dirtyable': object_type in {'Bowl', 'Mug', 'Plate'}, 'breakable': object_type in {'Bowl', 'Mug', 'Plate'}, 'cookable': object_type in {'Apple'}, 'parent_receptacles': [], 'visible_from_initial_pose': bool(interactive), 'is_open': False if object_type in {'Cabinet', 'Drawer', 'Fridge', 'Microwave'} else None, 'is_toggled': False if object_type in {'Faucet', 'Microwave'} else None, 'is_dirty': False if object_type in {'Bowl', 'Mug', 'Plate'} else None, 'is_sliced': False if object_type in {'Apple'} else None})
    profile = {'house_id': house_id or '{}:{}'.format(scene_model, variant_id), 'source': 'procthor', 'profile_version': '0.1', 'split': split, 'house_path': str(house_path), 'rooms': sorted(rooms_by_id.values(), key=lambda room: room['room_id']), 'objects': objects, 'affordance_index': _affordance_index(objects), 'object_type_index': _object_type_index(objects), 'reachable': {'num_positions': 1, 'positions': [{'x': 0.0, 'y': 0.9, 'z': 0.0}], 'sampled_agent_poses': [{'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'rotation': {'y': 0.0}, 'horizon': 0.0}]}, 'source_json': str(scene_path), 'layout_variant_id': variant_id, 'profile_source': 'behavior_scene_augmentation', 'bev_map_path': str(data.get('bev_png') or labels_path.with_name(labels_path.name.replace('_labels.json', '_bev.png'))), 'labels_json_path': str(labels_path)}
    if occupancy_map_path is not None:
        profile['occupancy_map_path'] = str(occupancy_map_path)
    if validate:
        validate_scene_profile(profile)
    return profile

def _affordance_index(objects: list[Mapping[str, Any]]) -> dict:
    fields = ('pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable', 'dirtyable', 'breakable', 'cookable')
    return {field: [obj['object_id'] for obj in objects if bool(obj.get(field, False))] for field in fields}

def _object_type_index(objects: list[Mapping[str, Any]]) -> dict:
    index = {}
    for obj in objects:
        index.setdefault(str(obj['object_type']), []).append(str(obj['object_id']))
    return {key: sorted(value) for key, value in sorted(index.items())}
__all__ = ['profile_from_behavior_augmented_labels']
