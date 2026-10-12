from __future__ import annotations
import csv
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from scoper.benchmark.behavior.paths import repo_root, resolve_repo_path
from scoper.benchmark.behavior.scene.taxonomy import normalize_behavior_object_type, taxonomy_groups_for_object_type
DEFAULT_SCENE_MODELS = ('Rs_int', 'Beechwood_0_int', 'Wainscott_0_int', 'house_single_floor')
UNKNOWN = 'unknown'
PROFILE_VERSION = 'behavior_profile_v0.1'
AFFORDANCE_FIELDS = ('pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable', 'dirtyable', 'breakable', 'cookable', 'heatable', 'coolable')
REQUIRED_PROFILE_FIELDS = ('house_id', 'source', 'profile_version', 'rooms', 'objects', 'affordance_index', 'object_type_index', 'reachable', 'spawnable_catalog', 'coordinate_frame')

class SceneProfileError(ValueError):
    pass

def default_assets_root() -> Path:
    return resolve_repo_path('datasets/behavior-1k-assets')

def scene_json_path(scene_model: str, *, assets_root: Path | None=None) -> Path:
    root = Path(assets_root) if assets_root else default_assets_root()
    return root / 'scenes' / scene_model / 'json' / '{}_best.json'.format(scene_model)

def profile_scenes(scene_models: list[str] | tuple[str, ...]=DEFAULT_SCENE_MODELS, *, assets_root: Path | None=None) -> list[dict[str, Any]]:
    return [profile_scene(scene_model, assets_root=assets_root) for scene_model in scene_models]

def profile_scene(scene_model: str, *, assets_root: Path | None=None, scene_json: Path | str | None=None, layout_variant_id: str | None=None) -> dict[str, Any]:
    assets_root = Path(assets_root) if assets_root else default_assets_root()
    path = Path(scene_json) if scene_json is not None else scene_json_path(scene_model, assets_root=assets_root)
    if not path.exists():
        raise FileNotFoundError('Scene JSON not found: {}'.format(path))
    scene_data = json.loads(path.read_text(encoding='utf-8'))
    init_info = (scene_data.get('objects_info') or {}).get('init_info') or {}
    registry = ((scene_data.get('state') or {}).get('registry') or {}).get('object_registry') or {}
    category_to_synset = load_category_mapping()
    synset_abilities = load_synset_abilities()
    objects = []
    unknown_counts = Counter()
    for name in sorted(init_info):
        info = init_info[name] or {}
        args = info.get('args') or {}
        category = str(args.get('category') or UNKNOWN)
        model = str(args.get('model') or UNKNOWN)
        synset = category_to_synset.get(category, UNKNOWN)
        metadata = load_object_metadata(category, model, assets_root=assets_root)
        raw_abilities = set((synset_abilities.get(synset) or {}).keys())
        abilities = sorted(raw_abilities | infer_ability_aliases(category, raw_abilities, metadata, scene_object=True))
        root_link = (registry.get(name) or {}).get('root_link') or {}
        pos = _number_list(root_link.get('pos'), 3)
        ori = _number_list(root_link.get('ori'), 4)
        bbox_size = _number_list((metadata or {}).get('bbox_size'), 3)
        base_link_offset = _number_list((metadata or {}).get('base_link_offset'), 3)
        scale = _number_list(args.get('scale'), 3)
        if pos is None:
            unknown_counts['position'] += 1
        if synset == UNKNOWN:
            unknown_counts['synset'] += 1
        obj = build_profile_object(object_id=str(name), category=category, model=model, synset=synset, rooms=[str(room) for room in args.get('in_rooms') or []], raw_position=pos, orientation=ori, scale=scale, bbox_size=bbox_size, base_link_offset=base_link_offset, abilities=abilities, registry_entry=registry.get(name) or {}, metadata_status='loaded' if metadata is not None else UNKNOWN, source_kind='scene_instance', requires_spawn=False)
        objects.append(obj)
        derived = maybe_build_derived_faucet(obj)
        if derived is not None:
            objects.append(derived)
    asset_catalog = build_asset_catalog(assets_root=assets_root)
    spawnable_catalog = build_spawnable_catalog(asset_catalog)
    room_records = build_room_records(objects)
    reachable = build_reachable_placeholder(room_records)
    profile = {'house_id': scene_model, 'source': 'behavior', 'profile_version': PROFILE_VERSION, 'scene_model': scene_model, 'source_json': str(path.relative_to(repo_root()) if path.is_relative_to(repo_root()) else path), 'layout_variant_id': layout_variant_id, 'object_count': len(objects), 'room_count': len(room_records), 'rooms': room_records, 'category_counts': dict(sorted(Counter((obj['category'] for obj in objects)).items())), 'ability_counts': dict(sorted(Counter((ability for obj in objects for ability in obj['abilities'])).items())), 'unknown_counts': dict(sorted(unknown_counts.items())), 'bounds': compute_xy_bounds(objects), 'objects': objects, 'affordance_index': build_affordance_index(objects), 'object_type_index': build_object_type_index(objects), 'reachable': reachable, 'spawnable_catalog': spawnable_catalog, 'coordinate_frame': {'raw_scene_position': 'BEHAVIOR/OmniGibson [x, y, z] with z vertical', 'task_position': 'AI2-THOR-like dict: x=raw_x, y=raw_z(vertical), z=raw_y', 'horizontal_axes': ['x', 'z'], 'vertical_axis': 'y', 'source': 'file_profile_coordinate_view'}, 'asset_catalog': asset_catalog}
    validate_scene_profile(profile)
    return profile

def build_profile_object(*, object_id: str, category: str, model: str, synset: str, rooms: list[str], raw_position: list[float] | None, orientation: list[float] | None, scale: list[float] | None, bbox_size: list[float] | None, base_link_offset: list[float] | None, abilities: list[str], registry_entry: Mapping[str, Any], metadata_status: str, source_kind: str, requires_spawn: bool) -> dict[str, Any]:
    object_type = normalize_behavior_object_type(category=category, synset=synset, abilities=abilities)
    ability_set = set(abilities)
    position = to_task_position(raw_position)
    root_position = to_root_position(raw_position)
    root_orientation_xyzw = to_root_orientation_xyzw(orientation)
    root_yaw = quat_to_yaw_xyzw(root_orientation_xyzw)
    room_id = rooms[0] if rooms else None
    pickupable = infer_pickupable(category, ability_set, source_kind=source_kind)
    receptacle = 'receptacle' in ability_set or is_receptacle_category(category)
    openable = 'openable' in ability_set
    toggleable = 'toggleable' in ability_set
    sliceable = 'sliceable' in ability_set
    dirtyable = 'dirtyable' in ability_set or 'wetable' in ability_set
    cookable = 'cookable' in ability_set
    heatable = 'heatable' in ability_set or cookable
    coolable = 'freezable' in ability_set or 'coolable' in ability_set
    extent = scaled_extent(bbox_size, scale)
    return {'object_id': object_id, 'object_type': object_type, 'category': category, 'synset': synset, 'model': model, 'room_id': room_id, 'rooms': rooms, 'position': position, 'raw_position': raw_position if raw_position is not None else UNKNOWN, 'root_position': root_position if root_position is not None else UNKNOWN, 'orientation': orientation if orientation is not None else UNKNOWN, 'root_orientation_xyzw': root_orientation_xyzw if root_orientation_xyzw is not None else UNKNOWN, 'root_yaw': root_yaw if root_yaw is not None else UNKNOWN, 'scale': scale if scale is not None else UNKNOWN, 'bbox_size': bbox_size if bbox_size is not None else UNKNOWN, 'base_link_offset': base_link_offset if base_link_offset is not None else UNKNOWN, 'extent': extent if extent is not None else UNKNOWN, 'abilities': abilities, 'taxonomy_groups': taxonomy_groups_for_object_type(object_type), 'pickupable': pickupable, 'receptacle': receptacle, 'openable': openable, 'toggleable': toggleable, 'sliceable': sliceable, 'dirtyable': dirtyable, 'breakable': 'breakable' in ability_set, 'cookable': cookable, 'heatable': heatable, 'coolable': coolable, 'parent_receptacles': [], 'parent_relation_status': 'not_available_in_scene_json', 'visible_from_initial_pose': None, 'distance_from_initial_pose': None, 'visibility_status': 'unknown_without_runtime_renderer', 'is_open': infer_open_state(registry_entry, openable=openable), 'is_toggled': None, 'is_sliced': None, 'is_dirty': None, 'states': infer_state_readability(registry_entry, abilities), 'source_kind': source_kind, 'requires_spawn': requires_spawn, 'metadata_status': metadata_status, 'state_readability': infer_state_readability(registry_entry, abilities)}

def maybe_build_derived_faucet(obj: Mapping[str, Any]) -> dict[str, Any] | None:
    abilities = set(obj.get('abilities') or [])
    if obj.get('object_type') != 'SinkBasin':
        return None
    if 'water_source' not in abilities and 'faucet_like' not in abilities and (not obj.get('toggleable')):
        return None
    faucet = dict(obj)
    faucet['object_id'] = '{}::faucet'.format(obj['object_id'])
    faucet['object_type'] = 'Faucet'
    faucet['category'] = 'derived_faucet'
    faucet['synset'] = 'faucet.n.01'
    faucet['pickupable'] = False
    faucet['receptacle'] = False
    faucet['openable'] = False
    faucet['toggleable'] = True
    faucet['sliceable'] = False
    faucet['parent_receptacles'] = [obj['object_id']]
    faucet['parent_relation_status'] = 'derived_from_sink_water_source'
    faucet['source_kind'] = 'derived_scene_part'
    faucet['requires_spawn'] = False
    faucet['abilities'] = sorted(set(obj.get('abilities') or []) | {'toggleable', 'faucet_like', 'water_source'})
    faucet['taxonomy_groups'] = taxonomy_groups_for_object_type('Faucet')
    faucet['metadata_status'] = 'derived_from_sink_water_source'
    return faucet

def to_task_position(raw_position: list[float] | None) -> dict[str, float] | None:
    if raw_position is None:
        return None
    return {'x': float(raw_position[0]), 'y': float(raw_position[2]), 'z': float(raw_position[1])}

def to_root_position(raw_position: list[float] | None) -> dict[str, float] | None:
    if raw_position is None:
        return None
    return {'x': float(raw_position[0]), 'y': float(raw_position[1]), 'z': float(raw_position[2])}

def to_root_orientation_xyzw(orientation: list[float] | None) -> list[float] | None:
    if orientation is None or len(orientation) < 4:
        return None
    return [float(value) for value in orientation[:4]]

def quat_to_yaw_xyzw(orientation: list[float] | tuple[float, ...] | None) -> float | None:
    if orientation is None or len(orientation) < 4:
        return None
    x, y, z, w = (float(value) for value in orientation[:4])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def scaled_extent(bbox_size: list[float] | None, scale: list[float] | None) -> list[float] | None:
    if bbox_size is None:
        return None
    if scale is None:
        return list(bbox_size)
    return [float(size) * float(factor) for size, factor in zip(bbox_size, scale)]

def infer_pickupable(category: str, abilities: set[str], *, source_kind: str) -> bool:
    if source_kind == 'spawnable_asset':
        return 'pickupable' in abilities or 'rigidBody' in abilities
    if 'sceneObject' in abilities:
        return False
    if is_fixture_category(category):
        return False
    return 'rigidBody' in abilities or 'pickupable' in abilities

def infer_open_state(registry_entry: Mapping[str, Any], *, openable: bool) -> bool | None:
    if not openable:
        return None
    joint_pos = registry_entry.get('joint_pos') if isinstance(registry_entry, Mapping) else None
    if not isinstance(joint_pos, list) or not joint_pos:
        return None
    try:
        return any((abs(float(value)) > 0.001 for value in joint_pos))
    except Exception:
        return None

def build_room_records(objects: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_room: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for obj in objects:
        for room in obj.get('rooms') or []:
            by_room[str(room)].append(obj)
    records = []
    for room_id in sorted(by_room):
        room_objects = by_room[room_id]
        records.append({'room_id': room_id, 'room_type': room_id.rsplit('_', 1)[0], 'object_count': len(room_objects), 'representative_reachable_pose': representative_pose_from_objects(room_objects), 'geometry_status': 'estimated_from_object_centers'})
    return records

def representative_pose_from_objects(objects: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    positions = [obj.get('position') for obj in objects if isinstance(obj.get('position'), Mapping)]
    if not positions:
        return None
    x = sum((float(pos['x']) for pos in positions)) / len(positions)
    z = sum((float(pos['z']) for pos in positions)) / len(positions)
    return {'position': {'x': round(x, 3), 'y': 0.0, 'z': round(z, 3)}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'source': 'object_center_centroid_placeholder'}

def build_reachable_placeholder(rooms: list[Mapping[str, Any]]) -> dict[str, Any]:
    sampled = [room['representative_reachable_pose'] for room in rooms if room.get('representative_reachable_pose')]
    return {'status': 'not_computed_file_profile', 'num_positions': 0, 'positions': [], 'sampled_agent_poses': sampled[:20], 'notes': ['BEHAVIOR scene JSON profiling is file-only; navigable occupancy requires OmniGibson/runtime.', 'Room representative poses are object-center placeholders and are not reset validation evidence.']}

def build_affordance_index(objects: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {field: [] for field in AFFORDANCE_FIELDS}
    for obj in objects:
        for field in AFFORDANCE_FIELDS:
            if bool(obj.get(field)):
                index[field].append(str(obj['object_id']))
    return {field: sorted(ids) for field, ids in index.items()}

def build_object_type_index(objects: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for obj in objects:
        index[str(obj['object_type'])].append(str(obj['object_id']))
    return {object_type: sorted(ids) for object_type, ids in sorted(index.items())}

def build_spawnable_catalog(asset_catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for item in asset_catalog.get('categories') or []:
        if not item.get('spawnable'):
            continue
        category = str(item.get('category') or UNKNOWN)
        synset = str(item.get('synset') or UNKNOWN)
        abilities = sorted(set(item.get('abilities') or []) | {'pickupable'})
        object_type = normalize_behavior_object_type(category=category, synset=synset, abilities=abilities)
        if object_type == 'Unknown':
            continue
        records.append(build_profile_object(object_id='spawnable:{}'.format(category), category=category, model=UNKNOWN, synset=synset, rooms=[], raw_position=None, orientation=None, scale=None, bbox_size=None, base_link_offset=None, abilities=abilities, registry_entry={}, metadata_status='asset_catalog', source_kind='spawnable_asset', requires_spawn=True) | {'model_count': int(item.get('model_count') or 0), 'spawnable': True})
    return sorted(records, key=lambda obj: (obj['object_type'], obj['category']))

def validate_scene_profile(profile: Mapping[str, Any]) -> None:
    if not isinstance(profile, Mapping):
        raise SceneProfileError('SceneProfile must be a mapping.')
    missing = [field for field in REQUIRED_PROFILE_FIELDS if field not in profile]
    if missing:
        raise SceneProfileError('SceneProfile missing fields: {}.'.format(', '.join(missing)))
    if profile.get('source') != 'behavior':
        raise SceneProfileError("SceneProfile source must be 'behavior'.")
    if not isinstance(profile.get('house_id'), str) or not profile['house_id']:
        raise SceneProfileError('SceneProfile house_id must be a non-empty string.')
    if not isinstance(profile.get('objects'), list):
        raise SceneProfileError('SceneProfile objects must be a list.')
    if not isinstance(profile.get('rooms'), list):
        raise SceneProfileError('SceneProfile rooms must be a list.')
    if not isinstance(profile.get('affordance_index'), Mapping):
        raise SceneProfileError('SceneProfile affordance_index must be a mapping.')
    if not isinstance(profile.get('object_type_index'), Mapping):
        raise SceneProfileError('SceneProfile object_type_index must be a mapping.')
    if not isinstance(profile.get('reachable'), Mapping):
        raise SceneProfileError('SceneProfile reachable must be a mapping.')
    for index, obj in enumerate(profile['objects']):
        validate_profile_object(obj, index)
    for index, obj in enumerate(profile['spawnable_catalog']):
        validate_profile_object(obj, index, spawnable=True)

def validate_profile_object(obj: Any, index: int, *, spawnable: bool=False) -> None:
    if not isinstance(obj, Mapping):
        raise SceneProfileError('Profile object {} must be a mapping.'.format(index))
    required = ('object_id', 'object_type', 'category', 'synset', 'room_id', 'position', 'pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable', 'parent_receptacles', 'source_kind', 'requires_spawn')
    missing = [field for field in required if field not in obj]
    if missing:
        raise SceneProfileError('Profile object {} missing fields: {}.'.format(index, ', '.join(missing)))
    if not isinstance(obj['object_id'], str) or not obj['object_id']:
        raise SceneProfileError('Profile object {} has invalid object_id.'.format(index))
    if not isinstance(obj['object_type'], str) or not obj['object_type']:
        raise SceneProfileError('Profile object {} has invalid object_type.'.format(index))
    for field in ('pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable', 'requires_spawn'):
        if not isinstance(obj[field], bool):
            raise SceneProfileError('Profile object {} field {} must be bool.'.format(index, field))
    if not isinstance(obj['parent_receptacles'], list):
        raise SceneProfileError('Profile object {} parent_receptacles must be a list.'.format(index))
    if spawnable and (not obj['requires_spawn']):
        raise SceneProfileError('Spawnable catalog object {} must require spawn.'.format(index))

@lru_cache(maxsize=1)
def load_category_mapping() -> dict[str, str]:
    path = resolve_repo_path('bddl3/bddl/generated_data/category_mapping.csv')
    mapping = {}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            category = row.get('category')
            synset = row.get('synset')
            if category and synset:
                mapping[category] = synset
    return mapping

@lru_cache(maxsize=1)
def load_synset_abilities() -> dict[str, dict[str, Any]]:
    for rel in ('bddl3/bddl/generated_data/propagated_annots_canonical.json', 'bddl3/bddl/generated_data/syn_prop_annots_canonical.json'):
        path = resolve_repo_path(rel)
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
    return {}

def build_asset_catalog(*, assets_root: Path | None=None) -> dict[str, Any]:
    assets_root = Path(assets_root) if assets_root else default_assets_root()
    return _build_asset_catalog_cached(str(assets_root))

@lru_cache(maxsize=8)
def _build_asset_catalog_cached(assets_root_value: str) -> dict[str, Any]:
    assets_root = Path(assets_root_value)
    category_to_synset = load_category_mapping()
    synset_abilities = load_synset_abilities()
    objects_root = assets_root / 'objects'
    categories = []
    if objects_root.exists():
        for category_dir in sorted((p for p in objects_root.iterdir() if p.is_dir())):
            category = category_dir.name
            synset = category_to_synset.get(category, UNKNOWN)
            raw_abilities = set((synset_abilities.get(synset) or {}).keys())
            models = [p.name for p in category_dir.iterdir() if p.is_dir()]
            abilities = sorted(raw_abilities | infer_ability_aliases(category, raw_abilities, None, scene_object=False))
            categories.append({'category': category, 'synset': synset, 'model_count': len(models), 'abilities': abilities, 'spawnable': is_spawnable_asset(category, raw_abilities)})
    return {'category_count': len(categories), 'spawnable_category_count': sum((1 for item in categories if item['spawnable'])), 'categories': categories}

def load_object_metadata(category: str, model: str, *, assets_root: Path | None=None) -> dict[str, Any] | None:
    assets_root = Path(assets_root) if assets_root else default_assets_root()
    return _load_object_metadata_cached(str(category), str(model), str(assets_root))

@lru_cache(maxsize=8192)
def _load_object_metadata_cached(category: str, model: str, assets_root_value: str) -> dict[str, Any] | None:
    if category == UNKNOWN or model == UNKNOWN:
        return None
    assets_root = Path(assets_root_value)
    path = assets_root / 'objects' / category / model / 'misc' / 'metadata.json'
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def infer_ability_aliases(category: str, raw_abilities: set[str], metadata: Mapping[str, Any] | None, *, scene_object: bool) -> set[str]:
    aliases = set()
    category_l = category.lower()
    if 'openable' in raw_abilities or (metadata and metadata.get('openable_joint_ids')):
        aliases.add('openable')
    if 'toggleable' in raw_abilities or any((token in category_l for token in ('switch', 'light'))):
        aliases.add('toggleable')
    if 'heatSource' in raw_abilities or any((token in category_l for token in ('microwave', 'oven', 'burner', 'stove'))):
        aliases.update({'heatSource', 'heat_source'})
    if 'coldSource' in raw_abilities or any((token in category_l for token in ('fridge', 'freezer'))):
        aliases.update({'coldSource', 'cool_source'})
    if 'slicer' in raw_abilities or 'knife' in category_l:
        aliases.update({'cutting_tool', 'slicer'})
    if 'cookable' in raw_abilities:
        aliases.update({'cookable', 'heatable'})
    if 'sliceable' in raw_abilities:
        aliases.add('sliceable')
    if 'wetable' in raw_abilities or 'disinfectable' in raw_abilities:
        aliases.add('dirtyable')
    if 'fillable' in raw_abilities or 'sceneObject' in raw_abilities or is_receptacle_category(category):
        aliases.update({'receptacle', 'support_surface'})
    if 'sink' in category_l:
        aliases.update({'sink_like', 'receptacle'})
        if 'toggleable' in raw_abilities or 'waterSource' in raw_abilities:
            aliases.add('faucet_like')
    if 'waterSource' in raw_abilities:
        aliases.add('water_source')
    if is_storage_category(category):
        aliases.add('storage')
    if not scene_object and is_spawnable_asset(category, raw_abilities):
        aliases.add('pickupable')
    return aliases

def infer_state_readability(registry_entry: Mapping[str, Any], abilities: list[str]) -> dict[str, str]:
    non_kin = registry_entry.get('non_kin') if isinstance(registry_entry, Mapping) else {}
    return {'open_closed': 'joint_pos' if 'openable' in abilities and registry_entry.get('joint_pos') is not None else UNKNOWN, 'toggle': UNKNOWN if 'toggleable' in abilities else 'not_applicable', 'temperature': 'readable' if isinstance(non_kin, Mapping) and 'Temperature' in non_kin else UNKNOWN, 'dirty_clean': UNKNOWN if 'dirtyable' in abilities else 'not_applicable', 'sliced': UNKNOWN if 'sliceable' in abilities else 'not_applicable'}

def compute_xy_bounds(objects: list[Mapping[str, Any]]) -> dict[str, float | str]:
    xs = []
    zs = []
    for obj in objects:
        pos = obj.get('position')
        if isinstance(pos, Mapping) and 'x' in pos and ('z' in pos):
            xs.append(float(pos['x']))
            zs.append(float(pos['z']))
    if not xs or not zs:
        return {'min_x': UNKNOWN, 'min_z': UNKNOWN, 'max_x': UNKNOWN, 'max_z': UNKNOWN}
    return {'min_x': min(xs), 'min_z': min(zs), 'max_x': max(xs), 'max_z': max(zs)}

def is_spawnable_asset(category: str, raw_abilities: set[str]) -> bool:
    category_l = category.lower()
    if 'rigidBody' not in raw_abilities:
        return False
    if 'sceneObject' in raw_abilities:
        return False
    if 'openable' in raw_abilities and any((token in category_l for token in ('cabinet', 'fridge', 'oven', 'door', 'window'))):
        return False
    return not any((token in category_l for token in ('wall', 'floor', 'ceiling', 'countertop', 'sink')))

def is_receptacle_category(category: str) -> bool:
    category_l = category.lower()
    return any((token in category_l for token in ('table', 'counter', 'cabinet', 'shelf', 'sink', 'fridge', 'freezer', 'dishwasher', 'oven', 'microwave', 'dresser', 'nightstand', 'wardrobe', 'drawer', 'storage', 'rack')))

def is_storage_category(category: str) -> bool:
    category_l = category.lower()
    return any((token in category_l for token in ('cabinet', 'shelf', 'fridge', 'freezer', 'dishwasher', 'drawer', 'storage', 'wardrobe')))

def is_fixture_category(category: str) -> bool:
    category_l = category.lower()
    return any((token in category_l for token in ('wall', 'floor', 'ceiling', 'countertop', 'sink', 'cabinet', 'fridge', 'oven', 'microwave', 'table', 'chair', 'sofa', 'door', 'window', 'shelf', 'lamp', 'switch', 'bed', 'toilet')))

def _number_list(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    try:
        return [float(v) for v in value]
    except Exception:
        return None
__all__ = ['DEFAULT_SCENE_MODELS', 'SceneProfileError', 'build_asset_catalog', 'build_spawnable_catalog', 'profile_scene', 'profile_scenes', 'scene_json_path', 'validate_scene_profile']
