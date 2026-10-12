from __future__ import annotations
import copy
import math
import re
from typing import Any, Mapping, Optional, Sequence
DEFAULT_EXTENTS = {'Apple': [0.12, 0.12, 0.12], 'Bowl': [0.2, 0.2, 0.12], 'Cup': [0.14, 0.14, 0.16], 'Mug': [0.14, 0.14, 0.16], 'Plate': [0.24, 0.24, 0.04], 'Potato': [0.12, 0.1, 0.1], 'Tomato': [0.12, 0.12, 0.1], 'Knife': [0.26, 0.04, 0.03], 'SprayBottle': [0.12, 0.1, 0.24], 'SoapBottle': [0.12, 0.1, 0.22], 'CounterTop': [0.9, 0.8, 0.18], 'DiningTable': [0.9, 0.9, 0.75], 'CoffeeTable': [0.8, 0.8, 0.45], 'Shelf': [0.8, 0.35, 1.2], 'Cabinet': [0.8, 0.6, 1.2], 'Fridge': [0.8, 0.7, 1.7], 'Microwave': [0.45, 0.45, 0.35], 'SinkBasin': [0.7, 0.75, 0.9], 'LightSwitch': [0.08, 0.04, 0.12], 'Faucet': [0.16, 0.16, 0.2], 'FloorLamp': [0.25, 0.25, 1.4], 'DeskLamp': [0.25, 0.25, 0.5]}
CONTAINER_TYPES = {'Cabinet', 'Drawer', 'Fridge', 'Microwave', 'GarbageCan', 'Bowl', 'Cup', 'Mug', 'Plate'}
CUTTABLE_FOOD_TYPES = {'Apple', 'Potato', 'Tomato'}
FRIDGE_TARGET_OBJECT_TYPES = {'Apple', 'Bread', 'Egg', 'Lettuce', 'Potato', 'Tomato'}
SOAKABLE_OBJECT_TYPES = CUTTABLE_FOOD_TYPES | {'Bowl', 'Cup', 'Mug', 'Plate'}
WIPEABLE_OBJECT_TYPES = SOAKABLE_OBJECT_TYPES | {'SprayBottle', 'SoapBottle'}
CLEANING_TOOL_TYPES = {'Rag', 'Sponge', 'Cloth', 'Towel'}
SPAWN_SUPPORT_CLEARANCE = 0.02
SPAWN_SUPPORT_MARGIN = 0.06
SPAWN_SUPPORT_MIN_CENTER_DISTANCE = 0.35
SPAWN_TABLE_OBJECT_TYPES = {'DiningTable', 'CoffeeTable', 'SideTable', 'CounterTop'}
SPAWN_SUPPORT_TYPE_PRIORITY = {'DiningTable': 0, 'CoffeeTable': 1, 'SideTable': 2, 'CounterTop': 3, 'Shelf': 20, 'Cabinet': 30, 'Microwave': 40, 'SinkBasin': 50}
SPAWN_SUPPORT_OBJECT_TYPES = set(SPAWN_SUPPORT_TYPE_PRIORITY)
RELOCATION_STAGING_SUPPORT_PRIORITY = {'DiningTable': 0, 'CoffeeTable': 1, 'SideTable': 2}
NON_SPAWN_SUPPORT_CATEGORIES = {'floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'}
RECLOSED_TEMPLATE_ID = 'mid_execution_container_re_closed_001'
RE_DIRTY_TEMPLATE_ID = 'mid_execution_state_re_dirty_001'
OBJECT_RELOCATED_TEMPLATE_ID = 'mid_execution_object_relocated_001'
SEMANTIC_RELOCATE_TEMPLATE_ID = 'semantic_relocate_after_interruption_001'
RECOVER_FAUCET_AND_PICK_TEMPLATE_ID = 'recover_faucet_and_pick_001'
RECOVER_FAUCET_AND_PICK_PREFERRED_TYPE = 'Bowl'

def _object_id(obj: Mapping[str, Any]) -> Optional[str]:
    object_id = obj.get('object_id') or obj.get('objectId')
    return object_id if isinstance(object_id, str) and object_id else None

def _object_type(obj: Mapping[str, Any]) -> str:
    object_type = obj.get('object_type') or obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    object_id = _object_id(obj)
    return object_id.split('|', 1)[0] if object_id else ''

def _object_label(obj: Mapping[str, Any], role_name: str) -> str:
    object_type = _object_type(obj)
    source = object_type or role_name
    words = re.sub('(?<!^)(?=[A-Z])', ' ', source).replace('_', ' ')
    return words.lower()

def number_vector(value: Any, length: int) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        vector = [float(v) for v in value]
    except Exception:
        return None
    if not all((math.isfinite(v) for v in vector)):
        return None
    return vector

def role_center(role: Mapping[str, Any]) -> Optional[list[float]]:
    if role.get('requires_spawn'):
        return None
    pos = role.get('position')
    if not isinstance(pos, Mapping):
        return None
    try:
        return [float(pos['x']), float(pos['z']), float(pos['y'])]
    except Exception:
        return None

def role_center_or_center(role: Mapping[str, Any]) -> Optional[list[float]]:
    return number_vector(role.get('center'), 3) or role_center(role)

def role_extent(role: Mapping[str, Any]) -> list[float]:
    object_type = _object_type(role)
    default = DEFAULT_EXTENTS.get(object_type, [0.2, 0.2, 0.2])
    raw_extent = number_vector(role.get('extent'), 3) or number_vector(role.get('bbox_size'), 3)
    if raw_extent is not None:
        return [max(0.04, float(raw_extent[0])), max(0.04, float(raw_extent[1])), max(0.04, float(raw_extent[2]))]
    return list(default)

def xy_distance(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if a is None or b is None:
        return float('inf')
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except Exception:
        return float('inf')

def aabb_vectors_overlap(center_a: Sequence[float], extent_a: Sequence[float], center_b: Any, extent_b: Any, *, tolerance: float) -> bool:
    center_b_vec = number_vector(center_b, 3)
    extent_b_vec = number_vector(extent_b, 3)
    if center_b_vec is None or extent_b_vec is None:
        return False
    for axis in range(3):
        delta = abs(float(center_a[axis]) - float(center_b_vec[axis]))
        limit = float(extent_a[axis]) * 0.5 + float(extent_b_vec[axis]) * 0.5 - float(tolerance)
        if delta >= limit:
            return False
    return True

def spawn_support_can_hold(support: Mapping[str, Any], role: Mapping[str, Any]) -> bool:
    object_type = _object_type(support)
    category = str(support.get('category') or object_type or '').lower()
    if category in NON_SPAWN_SUPPORT_CATEGORIES:
        return False
    if support.get('pickupable'):
        return False
    support_extent = role_extent(support)
    role_extent_value = role_extent(role)
    support_center = role_center_or_center(support)
    if support_center is None:
        return False
    if support_extent[0] <= 0.0 or support_extent[1] <= 0.0 or support_extent[2] <= 0.0:
        return False
    usable_x = support_extent[0] * 0.5 - role_extent_value[0] * 0.5 - SPAWN_SUPPORT_MARGIN
    usable_y = support_extent[1] * 0.5 - role_extent_value[1] * 0.5 - SPAWN_SUPPORT_MARGIN
    if usable_x < 0.0 or usable_y < 0.0:
        return False
    abilities = set(role_abilities(support))
    return object_type in SPAWN_SUPPORT_OBJECT_TYPES or 'support_surface' in abilities or 'placeable_surface' in abilities or ('heatSource' in abilities)

def support_overlaps_role(support: Mapping[str, Any], role: Optional[Mapping[str, Any]]) -> bool:
    if not role:
        return False
    support_center = role_center_or_center(support)
    role_center_value = role_center_or_center(role)
    if support_center is None or role_center_value is None:
        return False
    support_extent = role_extent(support)
    role_extent_value = role_extent(role)
    return aabb_vectors_overlap(role_center_value, role_extent_value, support_center, support_extent, tolerance=0.01)

def is_table_like_spawn_support(support: Mapping[str, Any]) -> bool:
    return _object_type(support) in SPAWN_TABLE_OBJECT_TYPES

def role_abilities(role: Mapping[str, Any]) -> list[str]:
    abilities = set()
    if role.get('pickupable'):
        abilities.add('pickupable')
    if role.get('receptacle'):
        object_type = _object_type(role)
        abilities.add('container' if object_type in CONTAINER_TYPES else 'support_surface')
        abilities.add('receptacle')
    if role.get('openable'):
        abilities.update({'openable', 'container'})
    if role.get('toggleable'):
        abilities.add('toggleable')
    object_type = _object_type(role)
    if object_type in {'Microwave', 'Oven', 'StoveBurner'}:
        abilities.update({'heatSource', 'toggleable', 'support_surface'})
    if object_type == 'Fridge':
        abilities.add('coldSource')
    if object_type == 'Faucet':
        abilities.update({'toggleable', 'water_source'})
    if object_type == 'Knife':
        abilities.add('cutting_tool')
    if object_type in CLEANING_TOOL_TYPES:
        abilities.add('cleaning_tool')
    if object_type in CUTTABLE_FOOD_TYPES:
        abilities.add('cuttable')
    if object_type in SOAKABLE_OBJECT_TYPES:
        abilities.add('soakable')
    if object_type in WIPEABLE_OBJECT_TYPES:
        abilities.add('wipeable')
    return sorted(abilities or {'support_surface'})

def _used_object_ids(binding: Mapping[str, Mapping[str, Any]]) -> set[str]:
    ids = set()
    for obj in binding.values():
        object_id = _object_id(obj) if isinstance(obj, Mapping) else None
        if object_id:
            ids.add(object_id)
    return ids

def _clone_existing_secondary_object(candidate: Mapping[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(dict(candidate))
    clone['source_kind'] = str(clone.get('source_kind') or 'scene_existing_secondary')
    clone.pop('requires_spawn', None)
    clone['init_states'] = {'dirty': False}
    notes = list(clone.get('notes') or [])
    if 'selected_existing_secondary_re_dirty_work_item' not in notes:
        notes.append('selected_existing_secondary_re_dirty_work_item')
    clone['notes'] = notes
    return clone

def _clone_existing_sink_pick_object(candidate: Mapping[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(dict(candidate))
    clone['source_kind'] = str(clone.get('source_kind') or 'scene_existing_sink_pick')
    clone.pop('requires_spawn', None)
    notes = list(clone.get('notes') or [])
    if 'selected_existing_recover_faucet_and_pick_object' not in notes:
        notes.append('selected_existing_recover_faucet_and_pick_object')
    clone['notes'] = notes
    return clone

def _same_secondary_family(candidate_type: str, target_type: str) -> bool:
    if candidate_type == target_type:
        return True
    return candidate_type in SOAKABLE_OBJECT_TYPES and target_type in SOAKABLE_OBJECT_TYPES

def select_re_dirty_secondary_object(binding: Mapping[str, Mapping[str, Any]], target_object: Mapping[str, Any], scene_profile: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not scene_profile:
        return None
    target_type = _object_type(target_object)
    target_center = role_center_or_center(target_object)
    sink_center = role_center_or_center(binding.get('sink_basin') or {})
    target_room = str(target_object.get('room_id') or '')
    used_object_ids = _used_object_ids(binding)
    target_parents = set(target_object.get('parent_receptacles') or [])
    same_type_candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    same_family_candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    for candidate in scene_profile.get('objects') or []:
        object_id = _object_id(candidate)
        if not object_id or object_id in used_object_ids:
            continue
        if candidate.get('requires_spawn'):
            continue
        candidate_type = _object_type(candidate)
        if not _same_secondary_family(candidate_type, target_type):
            continue
        if not bool(candidate.get('pickupable', False)):
            continue
        center = role_center_or_center(candidate)
        if center is None:
            continue
        parents = set(candidate.get('parent_receptacles') or [])
        same_parent_penalty = 0 if parents and parents != target_parents else 1
        same_type_penalty = 0 if candidate_type == target_type else 1
        candidate_room = str(candidate.get('room_id') or '')
        same_room_penalty = 0 if target_room and candidate_room == target_room else 1
        score = (same_room_penalty, same_parent_penalty, xy_distance(center, target_center), xy_distance(center, sink_center), candidate_type, object_id)
        if same_type_penalty == 0:
            same_type_candidates.append((score, candidate))
        else:
            same_family_candidates.append((score, candidate))
    from scoper.benchmark.common.task_identity_policy import enabled
    candidates = same_family_candidates if enabled() else same_type_candidates or same_family_candidates
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]

def select_recover_faucet_pick_object(binding: Mapping[str, Mapping[str, Any]], scene_profile: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not scene_profile:
        return None
    sink = binding.get('incident_sink')
    sink_id = _object_id(sink or {})
    if not sink_id:
        return None
    used_object_ids = _used_object_ids(binding)
    used_object_ids.discard(_object_id(binding.get('primary_object') or {}))
    candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    sink_center = role_center_or_center(sink or {})
    for candidate in scene_profile.get('objects') or []:
        object_id = _object_id(candidate)
        if not object_id or object_id in used_object_ids:
            continue
        if candidate.get('requires_spawn'):
            continue
        if not bool(candidate.get('pickupable', False)):
            continue
        if sink_id not in set(candidate.get('parent_receptacles') or []):
            continue
        center = role_center_or_center(candidate)
        candidates.append(((0 if _object_type(candidate) == RECOVER_FAUCET_AND_PICK_PREFERRED_TYPE else 1, xy_distance(center, sink_center), _object_type(candidate), object_id), candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]

def reclosed_binding_is_compatible(binding: Mapping[str, Mapping[str, Any]]) -> bool:
    target_object = binding.get('target_object') or {}
    target_receptacle = binding.get('target_receptacle') or {}
    if _object_type(target_receptacle) == 'Fridge':
        return _object_type(target_object) in FRIDGE_TARGET_OBJECT_TYPES
    return True

def select_reclosed_second_target(binding: Mapping[str, Mapping[str, Any]], scene_profile: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not scene_profile:
        return None
    target_object = binding.get('target_object') or {}
    target_receptacle = binding.get('target_receptacle') or {}
    target_type = _object_type(target_object)
    target_id = _object_id(target_object)
    receptacle_id = _object_id(target_receptacle)
    target_room = str(target_object.get('room_id') or '')
    receptacle_room = str(target_receptacle.get('room_id') or '')
    require_food = _object_type(target_receptacle) == 'Fridge'
    if require_food and target_type not in FRIDGE_TARGET_OBJECT_TYPES:
        return None
    target_center = role_center_or_center(target_object)
    receptacle_center = role_center_or_center(target_receptacle)
    used_object_ids = _used_object_ids(binding)
    candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    for candidate in scene_profile.get('objects') or []:
        object_id = _object_id(candidate)
        if not object_id or object_id == target_id or object_id in used_object_ids:
            continue
        if candidate.get('requires_spawn'):
            continue
        if not bool(candidate.get('pickupable', False)):
            continue
        candidate_type = _object_type(candidate)
        from scoper.benchmark.common.task_identity_policy import enabled
        if enabled():
            families = (set(FRIDGE_TARGET_OBJECT_TYPES), {'Bowl', 'Plate', 'Mug', 'Cup'}, {'Book', 'Pencil', 'Pen'}, {'Fork', 'Spoon', 'Knife', 'ButterKnife'})
            if candidate_type == target_type or not any(({candidate_type, target_type} <= group for group in families)):
                continue
        elif candidate_type != target_type:
            continue
        if require_food and candidate_type not in FRIDGE_TARGET_OBJECT_TYPES:
            continue
        parent_ids = set(candidate.get('parent_receptacles') or [])
        if receptacle_id in parent_ids:
            continue
        candidate_room = str(candidate.get('room_id') or '')
        center = role_center_or_center(candidate)
        candidates.append(((0 if target_room and candidate_room == target_room else 1, 0 if receptacle_room and candidate_room == receptacle_room else 1, xy_distance(center, target_center), xy_distance(center, receptacle_center), object_id), candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]

def select_interrupt_faucet_role(binding: Mapping[str, Mapping[str, Any]], scene_profile: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for role_name in ('faucet', 'interrupt_role'):
        role = binding.get(role_name)
        if isinstance(role, Mapping) and _object_type(role) == 'Faucet':
            return role
    if not scene_profile:
        return None
    sink_center = role_center_or_center(binding.get('sink_basin') or {})
    target_center = role_center_or_center(binding.get('target_object') or {}) or role_center_or_center(binding.get('target_receptacle') or {})
    candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    used_object_ids = _used_object_ids(binding)
    for candidate in scene_profile.get('objects') or []:
        object_id = _object_id(candidate)
        if not object_id or object_id in used_object_ids:
            continue
        if _object_type(candidate) != 'Faucet':
            continue
        if not candidate.get('toggleable'):
            continue
        center = role_center_or_center(candidate)
        score = (xy_distance(center, sink_center), xy_distance(center, target_center), str(candidate.get('room_id') or ''), object_id)
        candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]

def select_object_relocated_staging_receptacle(binding: Mapping[str, Mapping[str, Any]], target_object: Optional[Mapping[str, Any]], scene_profile: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not target_object:
        return None
    candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    used_object_ids = _used_object_ids(binding)
    excluded_roles = {'target_object', 'target_receptacle', 'faucet', 'sink_basin'}
    excluded_object_ids = {_object_id(binding.get(role_name) or {}) for role_name in excluded_roles if isinstance(binding.get(role_name), Mapping) and _object_id(binding[role_name])}
    anchor_center = role_center_or_center(binding.get('target_object') or {}) or role_center_or_center(binding.get('target_receptacle') or {})
    final_center = role_center_or_center(binding.get('target_receptacle') or {})
    target_room = str(target_object.get('room_id') or '')

    def staging_score(role: Mapping[str, Any], source_rank: int, role_id: str) -> Optional[tuple[Any, ...]]:
        object_type = _object_type(role)
        type_rank = RELOCATION_STAGING_SUPPORT_PRIORITY.get(object_type)
        if type_rank is None:
            return None
        candidate_room = str(role.get('room_id') or '')
        if target_room and candidate_room and (candidate_room != target_room):
            return None
        center = role_center_or_center(role)
        return (source_rank, type_rank, xy_distance(center, anchor_center), -xy_distance(center, final_center), object_type, role_id)
    for role_name, role in binding.items():
        if role_name in excluded_roles or not isinstance(role, Mapping):
            continue
        if role.get('requires_spawn'):
            continue
        if not spawn_support_can_hold(role, target_object):
            continue
        if support_overlaps_role(role, target_object):
            continue
        score = staging_score(role, 0, _object_id(role) or role_name)
        if score is not None:
            candidates.append((score, role))
    if scene_profile:
        for candidate in scene_profile.get('objects') or []:
            object_id = _object_id(candidate)
            if not object_id or object_id in used_object_ids or object_id in excluded_object_ids:
                continue
            if candidate.get('requires_spawn'):
                continue
            if not spawn_support_can_hold(candidate, target_object):
                continue
            if support_overlaps_role(candidate, target_object):
                continue
            score = staging_score(candidate, 1, object_id)
            if score is not None:
                candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]

def select_re_dirty_secondary_receptacle(binding: Mapping[str, Mapping[str, Any]], secondary_object: Mapping[str, Any], scene_profile: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    used_object_ids = _used_object_ids(binding)
    excluded_roles = {'target_object', 'secondary_object', 'target_receptacle', 'sink_basin', 'faucet'}
    excluded_object_ids = {_object_id(binding.get(role_name) or {}) for role_name in excluded_roles if isinstance(binding.get(role_name), Mapping) and _object_id(binding[role_name])}
    sink = binding.get('sink_basin') or {}
    target_receptacle = binding.get('target_receptacle') or {}
    sink_center = role_center_or_center(sink)
    target_center = role_center_or_center(target_receptacle) or sink_center
    sink_room = str(sink.get('room_id') or '')
    for candidate in (scene_profile or {}).get('objects') or []:
        object_id = _object_id(candidate)
        if not object_id or object_id in used_object_ids or object_id in excluded_object_ids:
            continue
        if candidate.get('requires_spawn'):
            continue
        if not spawn_support_can_hold(candidate, secondary_object):
            continue
        if support_overlaps_role(candidate, secondary_object):
            continue
        center = role_center_or_center(candidate)
        if center is None:
            continue
        sink_distance = xy_distance(center, sink_center)
        target_distance = xy_distance(center, target_center)
        candidate_room = str(candidate.get('room_id') or '')
        same_room_penalty = 1 if sink_room and candidate_room == sink_room else 0
        near_penalty = 1 if sink_distance < 2.0 else 0
        table_penalty = 0 if is_table_like_spawn_support(candidate) else 1
        score = (near_penalty, same_room_penalty, table_penalty, -sink_distance, -target_distance, SPAWN_SUPPORT_TYPE_PRIORITY.get(_object_type(candidate), 100), _object_type(candidate), object_id)
        candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]

def effective_binding_for_template(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], *, scene_profile: Optional[Mapping[str, Any]]=None) -> dict[str, Any]:
    template_id = str(template.get('template_id'))
    effective = dict(binding)
    if template_id == RECOVER_FAUCET_AND_PICK_TEMPLATE_ID:
        sink_object = select_recover_faucet_pick_object(effective, scene_profile)
        if sink_object is None:
            raise ValueError('recover_faucet_and_pick_001 requires a real pickupable object already inside incident_sink; scene {} has none.'.format((scene_profile or {}).get('house_id', 'unknown')))
        effective['primary_object'] = _clone_existing_sink_pick_object(sink_object)
        return effective
    if template_id == RE_DIRTY_TEMPLATE_ID and 'target_object' in effective and ('secondary_object' not in effective):
        secondary = select_re_dirty_secondary_object(effective, effective['target_object'], scene_profile)
        if secondary is None:
            raise ValueError('mid_execution_state_re_dirty_001 requires a second real scene object compatible with target_object; scene {} has none.'.format((scene_profile or {}).get('house_id', 'unknown')))
        effective['secondary_object'] = _clone_existing_secondary_object(secondary)
    return effective

def effective_roles_for_template(template: Mapping[str, Any], binding: dict[str, Mapping[str, Any]], *, scene_profile: Optional[Mapping[str, Any]]=None) -> None:
    template_id = str(template.get('template_id'))
    if template_id == RECLOSED_TEMPLATE_ID:
        if not reclosed_binding_is_compatible(binding):
            raise ValueError('mid_execution_container_re_closed_001 with a fridge target requires food objects.')
        if 'target_object_2' not in binding:
            candidate = select_reclosed_second_target(binding, scene_profile)
            if candidate is None:
                raise ValueError('mid_execution_container_re_closed_001 requires a second real scene object matching target_object and compatible with target_receptacle; scene {} has none.'.format((scene_profile or {}).get('house_id', 'unknown')))
            binding['target_object_2'] = candidate
        return
    if template_id == OBJECT_RELOCATED_TEMPLATE_ID:
        if 'faucet' not in binding:
            faucet = select_interrupt_faucet_role(binding, scene_profile)
            if faucet is None:
                raise ValueError('mid_execution_object_relocated_001 requires a toggleable faucet; scene {} has none.'.format((scene_profile or {}).get('house_id', 'unknown')))
            binding['faucet'] = faucet
        if 'staging_receptacle' not in binding:
            candidate = select_object_relocated_staging_receptacle(binding, binding.get('target_object'), scene_profile)
            if candidate is None:
                raise ValueError('mid_execution_object_relocated_001 requires a staging receptacle; scene {} has none suitable.'.format((scene_profile or {}).get('house_id', 'unknown')))
            binding['staging_receptacle'] = candidate
        return
    if template_id != RE_DIRTY_TEMPLATE_ID:
        return
    if 'secondary_receptacle' in binding:
        return
    secondary = binding.get('secondary_object') or binding.get('target_object')
    if not secondary:
        return
    candidate = select_re_dirty_secondary_receptacle(binding, secondary, scene_profile)
    if candidate is not None:
        binding['secondary_receptacle'] = candidate

def _receptacle_relation_type(binding: Mapping[str, Mapping[str, Any]], role_name: str) -> str:
    role = binding.get(role_name) or {}
    return 'object_in_receptacle' if _object_type(role) in CONTAINER_TYPES else 'object_on_receptacle'

def effective_goal_predicates_for_template(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], goal_predicates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = [copy.deepcopy(dict(predicate)) for predicate in goal_predicates]
    template_id = str(template.get('template_id'))
    if template_id == OBJECT_RELOCATED_TEMPLATE_ID:
        if 'faucet' in binding:
            faucet_predicate = {'type': 'object_toggled_off', 'object_role': 'faucet'}
            if not any((p.get('type') == 'object_toggled_off' and p.get('object_role') == 'faucet' for p in out)):
                out.append(faucet_predicate)
    if template_id == RECLOSED_TEMPLATE_ID:
        if 'target_object_2' not in binding:
            return out
        second_goal = {'type': _receptacle_relation_type(binding, 'target_receptacle'), 'object_role': 'target_object_2', 'receptacle_role': 'target_receptacle'}
        if any((predicate == second_goal for predicate in out)):
            return out
        insert_at = len(out)
        for idx, predicate in enumerate(out):
            if predicate.get('type') == 'object_closed':
                insert_at = idx
                break
        out.insert(insert_at, second_goal)
    return out

def effective_family_spec_for_template(template: Mapping[str, Any], family_spec: Optional[Mapping[str, Any]], *, binding: Optional[Mapping[str, Mapping[str, Any]]]=None) -> Optional[Mapping[str, Any]]:
    template_id = str(template.get('template_id'))
    if family_spec is None:
        return family_spec
    effective = copy.deepcopy(dict(family_spec))
    primary = (effective.get('mid_execution') or {}).get('primary_task')
    if not isinstance(primary, dict):
        return effective
    roles = binding or {}
    if template_id == RECLOSED_TEMPLATE_ID:
        if 'target_object_2' in roles:
            primary['target_objects'] = ['target_object', 'target_object_2']
    elif template_id == OBJECT_RELOCATED_TEMPLATE_ID:
        primary['belief_navigation_enabled'] = True
        primary['scenario'] = str(primary.get('scenario') or 'staging_interrupt')
        primary['relocation_activation'] = 'after_staging_interrupt'
        if 'staging_receptacle' in roles:
            primary['staging_receptacle_role'] = str(primary.get('staging_receptacle_role') or 'staging_receptacle')
        if 'faucet' in roles:
            primary['interrupt_role'] = str(primary.get('interrupt_role') or 'faucet')
            if 'interrupt_tools' not in primary:
                primary['interrupt_tools'] = ['navigate_to', 'toggle_on', 'toggle_off']
    elif template_id == SEMANTIC_RELOCATE_TEMPLATE_ID:
        primary['belief_navigation_enabled'] = True
        primary['scenario'] = str(primary.get('scenario') or 'staging_interrupt')
        primary['relocation_activation'] = 'after_staging_interrupt'
    elif template_id == RE_DIRTY_TEMPLATE_ID:
        sink_role = str(primary.get('sink_role') or 'sink_basin')
        if sink_role in roles:
            primary['staging_receptacle_role'] = sink_role
        if 'secondary_object' in roles:
            secondary_receptacle_role = 'secondary_receptacle' if 'secondary_receptacle' in roles else str(primary.get('receptacle_role') or 'target_receptacle')
            if secondary_receptacle_role in roles:
                primary['secondary_task'] = {'type': 'place_object', 'object_role': 'secondary_object', 'receptacle_role': secondary_receptacle_role}
    return effective

def _disturbance_items(disturbance_spec: Any) -> list[dict[str, Any]]:
    if isinstance(disturbance_spec, Mapping):
        return [copy.deepcopy(dict(item)) for item in disturbance_spec.get('disturbances') or []]
    if isinstance(disturbance_spec, Sequence):
        return [copy.deepcopy(dict(item)) for item in disturbance_spec]
    return []

def effective_disturbance_spec_for_template(template: Mapping[str, Any], disturbance_spec: Any, *, binding: Optional[Mapping[str, Mapping[str, Any]]]=None) -> Optional[dict[str, Any]]:
    items = _disturbance_items(disturbance_spec)
    if not items:
        return copy.deepcopy(dict(disturbance_spec)) if isinstance(disturbance_spec, Mapping) else None
    template_id = str(template.get('template_id'))
    roles = binding or {}
    seed = disturbance_spec.get('seed', 0) if isinstance(disturbance_spec, Mapping) else 0

    def _passthrough():
        return copy.deepcopy(dict(disturbance_spec)) if isinstance(disturbance_spec, Mapping) else None
    if template_id == RE_DIRTY_TEMPLATE_ID:
        if 'sink_basin' not in roles:
            return _passthrough()
        secondary_receptacle_role = 'secondary_receptacle' if 'secondary_receptacle' in roles else 'target_receptacle'
        if secondary_receptacle_role not in roles:
            return _passthrough()
        for item in items:
            if item.get('type') != 'dirty_object':
                continue
            item['requires_out_of_view'] = True
            item['after_step'] = 0
            item['activation'] = {'type': 'after_tool_result', 'tool': 'put_down', 'target_role': secondary_receptacle_role, 'success': True, 'state_delta_object_role': 'secondary_object', 'requires_target_not_held': True, 'required_parent_role': 'sink_basin'}
    elif template_id == OBJECT_RELOCATED_TEMPLATE_ID:
        if 'staging_receptacle' not in roles:
            return _passthrough()
        for item in items:
            if item.get('type') != 'relocate_object':
                continue
            item['requires_out_of_view'] = True
            item['after_step'] = 0
            item['activation'] = {'type': 'after_tool_result', 'tool': 'put_down', 'target_role': 'staging_receptacle', 'success': True, 'after_distinct_tool_count': 3, 'requires_target_not_held': True, 'required_parent_role': 'staging_receptacle'}
            extra = dict(item.get('extra') or {})
            exclude_roles = list(extra.get('exclude_roles') or [])
            for role in ('target_receptacle', 'staging_receptacle', 'target_object', 'faucet'):
                if role in roles and role not in exclude_roles:
                    exclude_roles.append(role)
            extra['exclude_roles'] = exclude_roles
            extra['avoid_current_view'] = True
            extra['avoid_last_seen_view'] = True
            item['extra'] = extra
    elif template_id == SEMANTIC_RELOCATE_TEMPLATE_ID:
        if 'staging_receptacle' not in roles:
            return _passthrough()
        for item in items:
            if item.get('type') != 'relocate_object':
                continue
            item['requires_out_of_view'] = True
            item['after_step'] = 0
            activation = dict(item.get('activation') or {})
            activation.setdefault('type', 'after_tool_result')
            activation.setdefault('tool', 'put_down')
            activation.setdefault('target_role', 'staging_receptacle')
            activation.setdefault('success', True)
            activation.setdefault('after_distinct_tool_count', 2)
            activation.setdefault('requires_target_not_held', True)
            activation.setdefault('required_parent_role', 'staging_receptacle')
            item['activation'] = activation
            extra = dict(item.get('extra') or {})
            exclude_roles = list(extra.get('exclude_roles') or [])
            for role in ('target_receptacle', 'staging_receptacle', 'target_object', 'faucet'):
                if role in roles and role not in exclude_roles:
                    exclude_roles.append(role)
            extra['exclude_roles'] = exclude_roles
            extra['avoid_current_view'] = True
            extra['avoid_last_seen_view'] = True
            item['extra'] = extra
    elif template_id == RECLOSED_TEMPLATE_ID:
        for item in items:
            if item.get('type') != 'close_object':
                continue
            item['requires_out_of_view'] = True
            item['after_step'] = 0
            item['activation'] = {'type': 'after_tool_result', 'tool': 'put_down', 'target_role': 'target_receptacle', 'success': True, 'min_delay_steps': 1, 'after_distinct_tool_count': 1, 'exclude_target_roles': ['target_receptacle'], 'requires_target_not_held': False}
    else:
        return copy.deepcopy(dict(disturbance_spec)) if isinstance(disturbance_spec, Mapping) else None
    return {'seed': seed, 'disturbances': items}

def effective_instruction_for_template(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], instruction: str) -> str:
    template_id = str(template.get('template_id'))
    labels = {role_name: _object_label(role, role_name) for role_name, role in binding.items()}
    if template_id == OBJECT_RELOCATED_TEMPLATE_ID:
        target_object = labels.get('target_object', 'object')
        staging_receptacle = labels.get('staging_receptacle', 'staging receptacle')
        target_receptacle = labels.get('target_receptacle', 'target receptacle')
        if 'faucet' in binding:
            return 'Find the {target_object}, stage it on the {staging_receptacle}, turn the faucet on and back off, then return and put the {target_object} in the {target_receptacle} even if it has moved.'.format(target_object=target_object, staging_receptacle=staging_receptacle, target_receptacle=target_receptacle)
        return 'Find the {target_object}, stage it on the {staging_receptacle}, then return and put the {target_object} in the {target_receptacle} even if it has moved.'.format(target_object=target_object, staging_receptacle=staging_receptacle, target_receptacle=target_receptacle)
    if template_id == RE_DIRTY_TEMPLATE_ID:
        target_object = labels.get('target_object', 'object')
        secondary_object = labels.get('secondary_object', 'object')
        target_receptacle = labels.get('target_receptacle', 'target receptacle')
        secondary_receptacle = labels.get('secondary_receptacle', target_receptacle)
        secondary_phrase = 'the second {}'.format(secondary_object) if secondary_object == target_object else 'the {}'.format(secondary_object)
        return 'Clean the {target_object} at the sink, leave it staged there, put {secondary_phrase} in the {secondary_receptacle}, then return to check the {target_object}; if it has become dirty again, clean it again before putting it in the {target_receptacle}.'.format(target_object=target_object, secondary_phrase=secondary_phrase, secondary_receptacle=secondary_receptacle, target_receptacle=target_receptacle)
    if template_id == RECLOSED_TEMPLATE_ID:
        target_object = labels.get('target_object', 'object')
        target_object_2 = labels.get('target_object_2', target_object)
        receptacle = labels.get('target_receptacle', 'receptacle')
        if target_object == target_object_2:
            raise ValueError('RECLOSED_REQUIRES_DISTINGUISHABLE_OBJECTS')
        return 'Put the {} and the {} in the {}, then close it.'.format(target_object, target_object_2, receptacle)
    return instruction
__all__ = ['effective_binding_for_template', 'effective_roles_for_template', 'effective_goal_predicates_for_template', 'effective_family_spec_for_template', 'effective_disturbance_spec_for_template', 'effective_instruction_for_template', 'reclosed_binding_is_compatible', 'select_interrupt_faucet_role', 'select_object_relocated_staging_receptacle', 'select_reclosed_second_target', 'select_re_dirty_secondary_object', 'select_re_dirty_secondary_receptacle']
