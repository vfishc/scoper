from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
from scoper.benchmark.behavior.task_generation import profile_task_generator as stage4
from scoper.benchmark.behavior.scene.room_layout_maps import load_behavior_room_layout_maps

class LatentMaterializationError(ValueError):
    pass
_OBSERVATION_VESSEL_FAMILIES = ('plate', 'bowl', 'drinking_vessel')
_OBSERVATION_VESSEL_SPECS = {'plate': {'category': 'plate', 'object_type': 'Plate', 'synset': 'plate.n.04', 'extent': [0.24, 0.24, 0.04]}, 'bowl': {'category': 'bowl', 'object_type': 'Bowl', 'synset': 'bowl.n.01', 'extent': [0.22, 0.22, 0.12]}, 'drinking_vessel': {'category': 'cup', 'object_type': 'Cup', 'synset': 'cup.n.01', 'extent': [0.1, 0.1, 0.16]}}

@dataclass
class LatentMaterialization:
    roles: dict[str, dict[str, Any]] = field(default_factory=dict)
    init: list[dict[str, Any]] = field(default_factory=list)
    goal_predicates: list[dict[str, Any]] = field(default_factory=list)
    latent_payload: dict[str, Any] = field(default_factory=dict)
    generation_notes: list[str] = field(default_factory=list)
    room_constraints: dict[str, str] = field(default_factory=dict)
Materializer = Callable[[Mapping[str, Any], Mapping[str, Any], int], LatentMaterialization]

def materialize_latent_issue(latent_spec: Mapping[str, Any], *, scene_profile: Mapping[str, Any], latent_index: int=0) -> LatentMaterialization:
    issue_type = str(latent_spec.get('issue_type') or '')
    materializer = LATENT_MATERIALIZERS.get(issue_type)
    if materializer is None:
        raise LatentMaterializationError("Unsupported latent issue_type '{}'.".format(issue_type))
    return materializer(latent_spec, scene_profile, latent_index)

def materialize_faucet_on(latent_spec: Mapping[str, Any], scene_profile: Mapping[str, Any], latent_index: int) -> LatentMaterialization:
    room_id = _required_room(latent_spec)
    faucet = _select_room_object(scene_profile, room_id, object_types={'Faucet'}, categories_contains=('faucet',), toggleable=True)
    if faucet is None:
        raise LatentMaterializationError("No faucet candidate found in room '{}'.".format(room_id))
    role_name = str(latent_spec.get('object_role') or 'latent_faucet_{}'.format(latent_index))
    role = _materialize_profile_role(faucet, role_name)
    role.setdefault('init_states', {})['toggled_on'] = True
    roles = {role_name: role}
    init = [{'predicate': 'toggled_on', 'args': [role_name]}, {'predicate': 'inroom', 'args': [role_name, room_id]}]
    room_constraints = {role_name: room_id}
    support_roles: dict[str, str] = {}
    generation_notes = ['bound_existing_faucet:{}'.format(faucet.get('object_id'))]
    parent_sink = _select_derived_faucet_parent(scene_profile, faucet)
    if parent_sink is not None:
        sink_role = str(latent_spec.get('sink_role') or 'latent_sink_{}'.format(latent_index))
        roles[sink_role] = _materialize_profile_role(parent_sink, sink_role)
        init.append({'predicate': 'inroom', 'args': [sink_role, room_id]})
        room_constraints[sink_role] = room_id
        support_roles['sink'] = sink_role
        generation_notes.append('bound_existing_faucet_parent_sink:{}'.format(parent_sink.get('object_id')))
    goal = {'type': 'object_toggled_off', 'args': {'object': role_name}}
    payload = {'latent_id': str(latent_spec.get('latent_id') or 'latent_faucet_on_{}'.format(latent_index)), 'room_id': room_id, 'source_family': str(latent_spec.get('source_family') or 'recover'), 'issue_type': 'faucet_on', 'object_role': role_name, 'repair_plan_template': 'toggle_off', 'suggested_affordances': ['toggle_off'], 'goal_predicates': [goal], 'public_after': 'scan'}
    if support_roles:
        payload['support_roles'] = support_roles
    return LatentMaterialization(roles=roles, init=init, goal_predicates=[goal], latent_payload=payload, generation_notes=generation_notes, room_constraints=room_constraints)

def materialize_open_container(latent_spec: Mapping[str, Any], scene_profile: Mapping[str, Any], latent_index: int) -> LatentMaterialization:
    room_id = _required_room(latent_spec)
    container = _select_room_object(scene_profile, room_id, object_types={'Cabinet', 'Drawer', 'Fridge', 'Microwave'}, openable=True, receptacle=True)
    if container is None:
        raise LatentMaterializationError("No openable container candidate found in room '{}'.".format(room_id))
    role_name = str(latent_spec.get('object_role') or 'latent_container_{}'.format(latent_index))
    role = _materialize_profile_role(container, role_name)
    role.setdefault('init_states', {})['open'] = True
    goal = {'type': 'object_closed', 'args': {'object': role_name}}
    return LatentMaterialization(roles={role_name: role}, init=[{'predicate': 'open', 'args': [role_name]}, {'predicate': 'inroom', 'args': [role_name, room_id]}], goal_predicates=[goal], latent_payload={'latent_id': str(latent_spec.get('latent_id') or 'latent_open_container_{}'.format(latent_index)), 'room_id': room_id, 'source_family': str(latent_spec.get('source_family') or 'recover'), 'issue_type': 'open_container', 'object_role': role_name, 'repair_plan_template': 'close', 'suggested_affordances': ['close'], 'goal_predicates': [goal], 'public_after': 'scan'}, generation_notes=['bound_existing_openable_container:{}'.format(container.get('object_id'))], room_constraints={role_name: room_id})

def materialize_dirty_dish(latent_spec: Mapping[str, Any], scene_profile: Mapping[str, Any], latent_index: int) -> LatentMaterialization:
    room_id = _required_room(latent_spec)
    faucet = _select_room_object(scene_profile, room_id, object_types={'Faucet'}, categories_contains=('faucet',), toggleable=True)
    storage = _select_room_object(scene_profile, room_id, object_types={'Cabinet', 'Drawer', 'Fridge'}, openable=True, receptacle=True)
    if faucet is None:
        raise LatentMaterializationError("No water source candidate found in room '{}'.".format(room_id))
    if storage is None:
        raise LatentMaterializationError("No storage receptacle candidate found in room '{}'.".format(room_id))
    dish_role = str(latent_spec.get('object_role') or 'latent_dish_{}'.format(latent_index))
    faucet_role = str(latent_spec.get('faucet_role') or 'latent_faucet_{}'.format(latent_index))
    storage_role = str(latent_spec.get('storage_role') or 'latent_storage_{}'.format(latent_index))
    object_family = str(latent_spec.get('object_family') or _OBSERVATION_VESSEL_FAMILIES[latent_index % len(_OBSERVATION_VESSEL_FAMILIES)])
    vessel = _OBSERVATION_VESSEL_SPECS.get(object_family)
    if vessel is None:
        raise LatentMaterializationError("Unsupported observation vessel family '{}'".format(object_family))
    parent_sink = _select_derived_faucet_parent(scene_profile, faucet)
    sink_role = str(latent_spec.get('sink_role') or 'latent_sink_{}'.format(latent_index)) if parent_sink is not None else None
    roles = {dish_role: {'synset': vessel['synset'], 'bddl_inst': '{}_latent_{}'.format(vessel['synset'], latent_index), 'category': vessel['category'], 'object_type': vessel['object_type'], 'source_kind': 'spawnable_asset', 'requires_spawn': True, 'abilities': ['pickupable', 'soakable', 'wipeable'], 'extent': list(vessel['extent']), 'init_states': {'dirty': True, 'saturated': {}}, 'room_id': room_id}, faucet_role: _materialize_profile_role(faucet, faucet_role), storage_role: _materialize_profile_role(storage, storage_role)}
    if parent_sink is not None and sink_role:
        roles[sink_role] = _materialize_profile_role(parent_sink, sink_role)
    goals = [{'type': 'object_clean', 'args': {'object': dish_role, 'source': faucet_role}}, {'type': 'object_in_receptacle', 'args': {'object': dish_role, 'receptacle': storage_role}}]
    assign_observation_conditioned_spawn_centers(roles, goals, scene_profile=scene_profile, latent_room_constraints={dish_role: room_id}, floor_anchor_role=sink_role or faucet_role)
    if 'center' not in roles[dish_role]:
        raise LatentMaterializationError("Could not place spawned dish inside room '{}'.".format(room_id))
    placement = roles[dish_role].setdefault('spawn_placement', {})
    placement['room_id'] = room_id
    placement['room_constraint_status'] = 'satisfied'
    return LatentMaterialization(roles=roles, init=[{'predicate': 'dirty', 'args': [dish_role]}, {'predicate': 'inroom', 'args': [dish_role, room_id]}, {'predicate': 'inroom', 'args': [faucet_role, room_id]}, {'predicate': 'inroom', 'args': [storage_role, room_id]}, *([{'predicate': 'inroom', 'args': [sink_role, room_id]}] if sink_role else [])], goal_predicates=goals, latent_payload={'latent_id': str(latent_spec.get('latent_id') or 'latent_dirty_dish_{}'.format(latent_index)), 'room_id': room_id, 'source_family': str(latent_spec.get('source_family') or 'recover'), 'issue_type': 'dirty_dish', 'object_family': object_family, 'object_role': dish_role, 'support_roles': {'faucet': faucet_role, 'storage': storage_role, **({'sink': sink_role} if sink_role else {})}, 'repair_plan_template': 'wash_then_place', 'suggested_affordances': ['pick_up', 'toggle_on', 'soak_under', 'toggle_off', 'put_down'], 'goal_predicates': goals, 'public_after': 'scan'}, generation_notes=['spawned_dirty_dish_in_room:{}'.format(room_id), 'bound_existing_faucet:{}'.format(faucet.get('object_id')), 'bound_existing_storage:{}'.format(storage.get('object_id')), *(['bound_existing_faucet_parent_sink:{}'.format(parent_sink.get('object_id'))] if parent_sink is not None else [])], room_constraints={dish_role: room_id, faucet_role: room_id, storage_role: room_id, **({sink_role: room_id} if sink_role else {})})

def assign_observation_conditioned_spawn_centers(roles: dict[str, Any], goal_predicates: Sequence[Mapping[str, Any]], *, scene_profile: Mapping[str, Any] | None=None, latent_room_constraints: Mapping[str, str] | None=None, floor_anchor_role: str | None=None) -> None:
    constraints = {str(role): str(room) for role, room in (latent_room_constraints or {}).items() if room}
    staged_roles = {name: copy.deepcopy(role) for name, role in roles.items()}
    constrained_roles = set(constraints)
    profile = _filtered_scene_profile(scene_profile, set(constraints.values()))
    stage4.assign_spawn_centers(staged_roles, goal_predicates, scene_profile=profile)
    if floor_anchor_role:
        anchor = roles.get(str(floor_anchor_role)) or staged_roles.get(str(floor_anchor_role))
        for role_name in constrained_roles:
            role = staged_roles.get(role_name)
            if not isinstance(role, Mapping) or not role.get('requires_spawn') or (not isinstance(anchor, Mapping)):
                continue
            center, offset = stage4.relocation_floor_center(role, anchor, roles=staged_roles, scene_profile=scene_profile, candidate_validator=lambda candidate_center, candidate_extent, room=constraints.get(role_name), current_role=role_name: _relocation_footprint_in_room({'center': candidate_center, 'extent': candidate_extent}, room, scene_profile) and (not any((stage4.aabb_vectors_overlap(candidate_center, candidate_extent, blocker_center, blocker_extent, tolerance=0.01) for blocker_center, blocker_extent in stage4.relocation_floor_blockers({**dict(staged_roles.get(current_role) or {}), 'object_id': '__relocation_candidate__'}, staged_roles, scene_profile)))))
            role['center'] = [float(value) for value in center]
            role['spawn_placement'] = {'mode': 'near_anchor_floor', 'anchor_role': str(floor_anchor_role), 'offset': [float(value) for value in offset], 'room_id': constraints.get(role_name), 'room_constraint_status': 'satisfied'}
    for role_name in constrained_roles:
        generated = staged_roles.get(role_name) or {}
        if 'center' not in generated:
            continue
        room_id = constraints[role_name]
        placement = dict(generated.get('spawn_placement') or {})
        support_role = placement.get('support_role') or placement.get('anchor_role')
        if support_role and support_role in staged_roles:
            support_room = str((staged_roles.get(str(support_role)) or {}).get('room_id') or '')
            if support_room and support_room != room_id:
                continue
        generated['room_id'] = room_id
        if isinstance(generated.get('spawn_placement'), Mapping):
            generated['spawn_placement'] = dict(generated['spawn_placement'])
            generated['spawn_placement']['room_id'] = room_id
            generated['spawn_placement']['room_constraint_status'] = 'satisfied'
        roles[role_name] = generated
    for role_name, generated in staged_roles.items():
        if role_name in roles:
            if role_name in constrained_roles:
                continue
            roles[role_name] = generated
        else:
            room_id = str(generated.get('room_id') or '')
            if not constraints or room_id in set(constraints.values()):
                roles[role_name] = generated

def _relocation_footprint_in_room(role: Mapping[str, Any], room_id: str, scene_profile: Mapping[str, Any] | None) -> bool:
    if not scene_profile or not role.get('center'):
        return True
    scene_model = scene_profile.get('scene_model')
    if not scene_model:
        return True
    try:
        maps = load_behavior_room_layout_maps(str(scene_model))
    except Exception:
        return True
    mask = maps.room_mask(str(room_id))
    if mask is None:
        return True
    center = [float(v) for v in role.get('center')]
    extent = stage4.number_vector(role.get('extent'), 3) or [0.2, 0.2, 0.2]
    free = maps.traversable_mask()
    for x in (center[0] - extent[0] * 0.5, center[0] + extent[0] * 0.5):
        for y in (center[1] - extent[1] * 0.5, center[1] + extent[1] * 0.5):
            row, col = maps.world_to_map((x, y))
            if not (0 <= row < maps.map_size and 0 <= col < maps.map_size and bool(mask[row, col]) and bool(free[row, col])):
                return False
    return True

def materialize_misplaced_object(latent_spec: Mapping[str, Any], scene_profile: Mapping[str, Any], latent_index: int) -> LatentMaterialization:
    room_id = _required_room(latent_spec)
    storage = _select_room_object(scene_profile, room_id, object_types={'Cabinet', 'Drawer', 'Shelf', 'CounterTop'}, receptacle=True)
    if storage is None:
        raise LatentMaterializationError("No storage receptacle candidate found in room '{}'".format(room_id))
    object_family = str(latent_spec.get('object_family') or _OBSERVATION_VESSEL_FAMILIES[latent_index % len(_OBSERVATION_VESSEL_FAMILIES)])
    vessel = _OBSERVATION_VESSEL_SPECS.get(object_family)
    if vessel is None:
        raise LatentMaterializationError("Unsupported observation vessel family '{}'".format(object_family))
    object_role = str(latent_spec.get('object_role') or 'latent_object_{}'.format(latent_index))
    storage_role = str(latent_spec.get('storage_role') or 'latent_storage_{}'.format(latent_index))
    roles = {object_role: {'synset': vessel['synset'], 'bddl_inst': '{}_latent_{}'.format(vessel['synset'], latent_index), 'category': vessel['category'], 'object_type': vessel['object_type'], 'source_kind': 'spawnable_asset', 'requires_spawn': True, 'abilities': ['pickupable'], 'extent': list(vessel['extent']), 'room_id': room_id}, storage_role: _materialize_profile_role(storage, storage_role)}
    goal = {'type': 'object_in_receptacle', 'args': {'object': object_role, 'receptacle': storage_role}}
    assign_observation_conditioned_spawn_centers(roles, [goal], scene_profile=scene_profile, latent_room_constraints={object_role: room_id}, floor_anchor_role=storage_role)
    if 'center' not in roles[object_role]:
        raise LatentMaterializationError("Could not place object on floor in room '{}'".format(room_id))
    return LatentMaterialization(roles=roles, init=[{'predicate': 'inroom', 'args': [object_role, room_id]}], goal_predicates=[goal], latent_payload={'latent_id': str(latent_spec.get('latent_id') or 'latent_misplaced_object_{}'.format(latent_index)), 'room_id': room_id, 'source_family': str(latent_spec.get('source_family') or 'observation'), 'issue_type': 'misplaced_object', 'object_role': object_role, 'object_family': object_family, 'support_roles': {'storage': storage_role}, 'repair_plan_template': 'place', 'suggested_affordances': ['pick_up', 'put_down'], 'goal_predicates': [goal], 'public_after': 'scan'}, generation_notes=['spawned_misplaced_object_on_floor:{}'.format(room_id), 'bound_existing_storage:{}'.format(storage.get('object_id'))], room_constraints={object_role: room_id, storage_role: room_id})

def _required_room(latent_spec: Mapping[str, Any]) -> str:
    room_id = str(latent_spec.get('room_id') or '')
    if not room_id:
        raise LatentMaterializationError('Latent issue requires room_id.')
    return room_id

def _select_room_object(scene_profile: Mapping[str, Any], room_id: str, *, object_types: set[str] | None=None, categories_contains: Sequence[str]=(), openable: bool | None=None, receptacle: bool | None=None, toggleable: bool | None=None) -> Mapping[str, Any] | None:
    candidates = []
    for candidate in scene_profile.get('objects') or []:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get('room_id') or '') != str(room_id):
            continue
        if candidate.get('requires_spawn'):
            continue
        if object_types and str(candidate.get('object_type') or '') not in object_types:
            continue
        haystack = '{} {}'.format(candidate.get('object_type') or '', candidate.get('category') or '').lower()
        if categories_contains and (not any((token.lower() in haystack for token in categories_contains))):
            continue
        if openable is not None and bool(candidate.get('openable')) is not bool(openable):
            continue
        if receptacle is not None and bool(candidate.get('receptacle')) is not bool(receptacle):
            continue
        if toggleable is not None and bool(candidate.get('toggleable')) is not bool(toggleable):
            continue
        center = stage4.role_center_or_center(candidate)
        if center is None:
            continue
        candidates.append((str(candidate.get('object_type') or ''), str(candidate.get('object_id') or ''), candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:2])
    selection_seed = scene_profile.get('_selection_seed')
    if selection_seed is not None:
        random.Random(int(selection_seed)).shuffle(candidates)
    return candidates[0][2]

def _select_derived_faucet_parent(scene_profile: Mapping[str, Any], faucet: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if str(faucet.get('source_kind') or '') != 'derived_scene_part' and str(faucet.get('category') or '') != 'derived_faucet':
        return None
    object_id = str(faucet.get('object_id') or '')
    parent_id = object_id.split('::', 1)[0] if '::' in object_id else ''
    if not parent_id:
        parents = faucet.get('parent_receptacles') or []
        parent_id = str(parents[0]) if parents else ''
    if not parent_id:
        return None
    for candidate in scene_profile.get('objects') or []:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get('object_id') or '') == parent_id:
            return candidate
    return None

def _materialize_profile_role(candidate: Mapping[str, Any], role_name: str) -> dict[str, Any]:
    role = stage4.materialize_profile_support_role(candidate, role_name)
    role.pop('auxiliary_spawn_support', None)
    role['source_kind'] = candidate.get('source_kind') or 'scene_instance'
    role['requires_spawn'] = False
    role['abilities'] = stage4.role_abilities(candidate)
    if candidate.get('toggleable') and 'toggleable' not in role['abilities']:
        role['abilities'] = sorted(set(role['abilities']) | {'toggleable'})
    if str(candidate.get('object_type') or '') == 'Faucet':
        role['abilities'] = sorted(set(role['abilities']) | {'toggleable', 'water_source'})
    if candidate.get('openable'):
        role['abilities'] = sorted(set(role['abilities']) | {'openable', 'container'})
        role.setdefault('init_states', {}).setdefault('open', False)
    role['room_id'] = candidate.get('room_id')
    return role

def _filtered_scene_profile(scene_profile: Mapping[str, Any] | None, room_ids: set[str]) -> dict[str, Any]:
    if not scene_profile:
        return {}
    out = copy.deepcopy(dict(scene_profile))
    if room_ids:
        out['objects'] = [copy.deepcopy(dict(item)) for item in scene_profile.get('objects') or [] if isinstance(item, Mapping) and str(item.get('room_id') or '') in room_ids]
    return out
LATENT_MATERIALIZERS: dict[str, Materializer] = {'faucet_on': materialize_faucet_on, 'open_container': materialize_open_container, 'dirty_dish': materialize_dirty_dish, 'misplaced_object': materialize_misplaced_object}
__all__ = ['LATENT_MATERIALIZERS', 'LatentMaterialization', 'LatentMaterializationError', 'assign_observation_conditioned_spawn_centers', 'materialize_misplaced_object', 'materialize_latent_issue']
