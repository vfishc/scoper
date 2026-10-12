from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.behavior.scene.taxonomy import normalize_task_object_type, object_type_in_group
from scoper.benchmark.behavior.scene.placement_target_policy import role_requires_generic_placement, supports_generic_surface_placement
from scoper.benchmark.common import RoomSemanticError, build_room_type_catalog, materialize_room_scope
DEFAULT_PROCTHOR_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / 'procthor/data/task_templates'
REJECTED = 'rejected'
MATCHED_PROFILE_ONLY = 'matched_profile_only'
MATCHED_REQUIRES_SPAWN = 'matched_requires_spawn'
MATCHED_REQUIRES_RUNTIME_VALIDATION = 'matched_requires_runtime_validation'
MATCHED_STATUSES = {MATCHED_PROFILE_ONLY, MATCHED_REQUIRES_SPAWN, MATCHED_REQUIRES_RUNTIME_VALIDATION}
SUPPORTED_MATCH_PREDICATES = {'object_in_receptacle', 'object_not_in_receptacle', 'object_open', 'object_closed', 'object_toggled_on', 'object_toggled_off', 'object_clean', 'object_dirty', 'object_sliced', 'object_hot', 'object_cool', 'object_filled', 'object_in_any_receptacle', 'object_not_in_any_receptacle'}
OBJECT_TYPE_PRIORITY = {'CounterTop': 0, 'DiningTable': 1, 'CoffeeTable': 2, 'SideTable': 3, 'Fridge': 4, 'Microwave': 5, 'Cabinet': 6, 'Drawer': 7, 'Shelf': 8, 'SinkBasin': 9, 'GarbageCan': 10, 'Bowl': 20, 'Plate': 21, 'Mug': 22, 'Cup': 23, 'Apple': 30, 'Potato': 31, 'Tomato': 32, 'Bread': 33, 'Lettuce': 34, 'Knife': 40, 'SoapBottle': 50, 'SprayBottle': 51, 'FloorLamp': 60, 'DeskLamp': 61, 'LightSwitch': 62, 'Faucet': 63, 'Sofa': 90, 'ArmChair': 91, 'Ottoman': 92, 'TVStand': 93}

def load_procthor_templates(path: str | Path=DEFAULT_PROCTHOR_TEMPLATE_DIR) -> list[dict[str, Any]]:
    template_path = Path(path)
    if template_path.is_file():
        data = json.loads(template_path.read_text(encoding='utf-8'))
        data['_template_file'] = template_path.name
        return [data]
    templates = []
    for child in sorted(template_path.glob('*.json')):
        data = json.loads(child.read_text(encoding='utf-8'))
        data['_template_file'] = child.name
        templates.append(data)
    return templates

def match_scene_to_templates(scene_profile: Mapping[str, Any], templates: Sequence[Mapping[str, Any]], *, max_bindings_per_template: int=8, candidate_seed: int | None=None) -> list[dict[str, Any]]:
    return [match_template_to_scene(scene_profile, template, max_bindings=max_bindings_per_template, candidate_seed=candidate_seed) for template in templates]

def match_template_to_scene(scene_profile: Mapping[str, Any], template: Mapping[str, Any], *, max_bindings: int=8, candidate_seed: int | None=None) -> dict[str, Any]:
    requires = template.get('requires') if isinstance(template.get('requires'), Mapping) else {}
    candidate_counts: dict[str, int] = {}
    candidates_by_role: dict[str, list[Mapping[str, Any]]] = {}
    rejection_reasons: list[str] = []
    notes: list[str] = []
    room_scope = (template.get('family_spec_template') or {}).get('scope')
    try:
        room_catalog = build_room_type_catalog(scene_profile.get('rooms') or [])
    except RoomSemanticError as exc:
        room_catalog = {}
        rejection_reasons.append('{}:{}'.format(exc.code, exc))
    for role_name, requirement in requires.items():
        candidates = role_candidates(scene_profile, requirement)
        if candidate_seed is not None:
            candidates = diversify_role_candidates(candidates, candidate_seed)
        if role_requires_generic_placement(template, str(role_name)):
            allowed = [candidate for candidate in candidates if supports_generic_surface_placement(candidate)]
            if len(allowed) != len(candidates):
                notes.append('unsupported_surface_geometry:{}:{}'.format(role_name, len(candidates) - len(allowed)))
                if candidates and (not allowed) and (not requirement.get('optional', False)):
                    rejection_reasons.append('unsupported_surface_geometry:{}'.format(role_name))
            candidates = allowed
        candidate_counts[str(role_name)] = len(candidates)
        candidates_by_role[str(role_name)] = candidates
    required_roles = [str(role) for role, requirement in requires.items() if not requirement.get('optional', False) and (not requirement.get('relocate_target_only', False))]
    optional_roles = [str(role) for role, requirement in requires.items() if requirement.get('optional', False) and (not requirement.get('relocate_target_only', False))]
    for role in required_roles:
        if not candidates_by_role.get(role):
            rejection_reasons.append('no_candidates:{}'.format(role))
    unsupported_predicates = unsupported_template_predicates(template)
    runtime_capabilities = template_runtime_capabilities(template, scene_profile)
    bindings: list[dict[str, Any]] = []
    if not rejection_reasons:
        raw_bindings: list[dict[str, Mapping[str, Any]]] = []
        collection_limit = _binding_collection_limit(max_bindings=max_bindings, role_order=required_roles, candidates_by_role=candidates_by_role, template=template)
        _collect_bindings(template=template, role_order=required_roles, candidates_by_role=candidates_by_role, role_index=0, current={}, bindings=raw_bindings, limit=collection_limit)
        if isinstance(room_scope, Mapping):
            accepted_bindings = []
            for raw_binding in raw_bindings:
                try:
                    materialize_room_scope(room_scope, roles=raw_binding, room_catalog=room_catalog)
                except RoomSemanticError:
                    continue
                accepted_bindings.append(raw_binding)
            raw_bindings = accepted_bindings
        if required_roles and (not raw_bindings):
            rejection_reasons.append('room_scope_mismatch' if isinstance(room_scope, Mapping) else 'no_compatible_role_binding')
        elif not required_roles:
            raw_bindings = [{}]
        raw_bindings = _select_diverse_bindings(raw_bindings, max_bindings=max_bindings, role_order=required_roles, candidates_by_role=candidates_by_role, template=template)
        for raw in raw_bindings:
            extended = _extend_optional_roles(template, raw, optional_roles, candidates_by_role)
            bindings.append(serialize_binding(extended, template, scene_profile))
    pair_caps = combined_runtime_capabilities(runtime_capabilities, bindings)
    pair_mutations = combined_init_mutations(bindings)
    status = match_status(bindings, pair_caps, pair_mutations)
    profile_match = status in MATCHED_STATUSES
    if unsupported_predicates:
        notes.append('unsupported_or_deferred_predicates:{}'.format(','.join(unsupported_predicates)))
    if pair_caps:
        notes.append('requires_runtime_capabilities:{}'.format(','.join(pair_caps)))
    if pair_mutations:
        notes.append('requires_init_mutations:{}'.format(','.join(pair_mutations)))
    if scene_profile.get('reachable', {}).get('status') != 'computed':
        notes.append('reachable_map:not_computed_file_profile')
    return {'scene_model': scene_profile.get('scene_model') or scene_profile.get('house_id'), 'house_id': scene_profile.get('house_id'), 'template_id': template.get('template_id'), 'template_file': template.get('_template_file'), 'category': template.get('category'), 'status': status, 'profile_match': profile_match, 'requires_spawn': 'spawn_object' in pair_mutations or 'spawn_placement' in pair_caps, 'requires_runtime_validation': status == MATCHED_REQUIRES_RUNTIME_VALIDATION, 'candidate_counts_by_role': candidate_counts, 'rejection_reasons': rejection_reasons, 'bindings': bindings, 'binding_count': len(bindings), 'unsupported_predicates': unsupported_predicates, 'runtime_capabilities_required': pair_caps, 'required_init_mutations': pair_mutations, 'notes': notes}

def combined_runtime_capabilities(template_caps: Sequence[str], bindings: Sequence[Mapping[str, Any]]) -> list[str]:
    caps = set(template_caps)
    for binding in bindings:
        caps.update(binding.get('runtime_capabilities_required') or [])
    return sorted(caps)

def combined_init_mutations(bindings: Sequence[Mapping[str, Any]]) -> list[str]:
    mutations = set()
    for binding in bindings:
        mutations.update(binding.get('required_init_mutations') or [])
    return sorted(mutations)

def match_status(bindings: Sequence[Mapping[str, Any]], runtime_capabilities: Sequence[str], init_mutations: Sequence[str]) -> str:
    if not bindings:
        return REJECTED
    runtime_set = set(runtime_capabilities)
    mutation_set = set(init_mutations)
    non_spawn_caps = runtime_set - {'spawn_placement'}
    non_spawn_mutations = mutation_set - {'spawn_object'}
    if non_spawn_caps or non_spawn_mutations:
        return MATCHED_REQUIRES_RUNTIME_VALIDATION
    if 'spawn_placement' in runtime_set or 'spawn_object' in mutation_set:
        return MATCHED_REQUIRES_SPAWN
    return MATCHED_PROFILE_ONLY

def role_candidates(scene_profile: Mapping[str, Any], requirement: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    pool = list(scene_profile.get('objects') or [])
    if spawnable_role_allowed(requirement):
        pool.extend(scene_profile.get('spawnable_catalog') or [])
    candidates = [candidate for candidate in pool if candidate_satisfies_requirement(candidate, requirement, scene_profile)]
    return sorted(candidates, key=lambda obj: candidate_sort_key(obj, requirement))

def diversify_role_candidates(candidates, seed):
    import hashlib
    groups = {}
    for candidate in candidates:
        groups.setdefault(str(candidate.get('object_type') or candidate.get('category') or ''), []).append(candidate)
    keys = sorted(groups)
    if not keys:
        return []
    offset = int(seed) % len(keys)
    keys = keys[offset:] + keys[:offset]
    return [candidate for key in keys for candidate in sorted(groups[key], key=lambda obj: hashlib.sha256((str(seed) + ':' + str(obj.get('object_id') or obj.get('model'))).encode()).hexdigest())]

def candidate_sort_key(obj: Mapping[str, Any], requirement: Mapping[str, Any]) -> tuple[Any, ...]:
    object_type = str(obj.get('object_type') or '')
    source_rank = 0
    if obj.get('requires_spawn') and (not requirement.get('must_be_pickupable')):
        source_rank = 1
    receptacle_bonus = 0 if requirement.get('must_be_receptacle') and obj.get('receptacle') else 1
    return (source_rank, receptacle_bonus, OBJECT_TYPE_PRIORITY.get(object_type, 500), object_type, str(obj.get('object_id')))

def spawnable_role_allowed(requirement: Mapping[str, Any]) -> bool:
    if requirement.get('must_not_be_pickupable'):
        return False
    return bool(requirement.get('must_be_pickupable') or requirement.get('must_have_parent_receptacle') or requirement.get('taxonomy_group') in {'pickupable_task_objects', 'dirtyable_task_objects', 'heatable_task_objects', 'coolable_task_objects', 'sliceable_task_objects', 'knife_objects', 'cold_food', 'dishware', 'cleaning_supplies', 'dirtyable_heatable_task_objects'})

def candidate_satisfies_requirement(obj: Mapping[str, Any], requirement: Mapping[str, Any], scene_profile: Mapping[str, Any]) -> bool:
    if obj.get('_bank_ambiguous_concrete'):
        return False
    from scoper.benchmark.behavior.scene.concrete_categories import CONCRETE_CATEGORY_TYPES
    object_type = CONCRETE_CATEGORY_TYPES.get(str(obj.get('category') or ''), normalize_task_object_type(str(obj.get('object_type') or '')))
    if object_type == 'Unknown':
        return False
    allowlist = requirement.get('object_type_allowlist')
    if allowlist and object_type not in set(allowlist):
        return False
    blocklist = set(requirement.get('object_type_blocklist') or [])
    if object_type in blocklist:
        return False
    group_name = requirement.get('taxonomy_group')
    if group_name and (not object_type_in_group(object_type, str(group_name))):
        return False
    bool_checks = {'must_be_pickupable': 'pickupable', 'must_be_receptacle': 'receptacle', 'must_be_openable': 'openable', 'must_be_toggleable': 'toggleable', 'must_be_sliceable': 'sliceable'}
    for requirement_field, object_field in bool_checks.items():
        if requirement.get(requirement_field) and (not bool(obj.get(object_field, False))):
            return False
    if requirement.get('must_not_be_pickupable') and bool(obj.get('pickupable', False)):
        return False
    if requirement.get('must_have_parent_receptacle'):
        if not obj.get('parent_receptacles') and (not obj.get('requires_spawn')):
            return False
    if requirement.get('must_not_have_parent_receptacle') and obj.get('parent_receptacles'):
        return False
    if requirement.get('must_not_have_closed_parent_receptacle'):
        if has_closed_parent(obj, scene_profile) and (not requirement_allows_parent_relocation(requirement)):
            return False
    if has_visibility_requirement(requirement):
        if obj.get('requires_spawn'):
            return True
        expected = expected_visibility(requirement)
        actual = obj.get('visible_from_initial_pose')
        if actual is None:
            return True
        if expected is not None and actual is not expected:
            return False
    return True

def has_closed_parent(obj: Mapping[str, Any], scene_profile: Mapping[str, Any]) -> bool:
    parent_ids = obj.get('parent_receptacles') or []
    if not parent_ids:
        return False
    by_id = {item.get('object_id'): item for item in scene_profile.get('objects') or []}
    for parent_id in parent_ids:
        parent = by_id.get(parent_id)
        if parent and parent.get('openable') and (parent.get('is_open') is False):
            return True
    return False

def requirement_allows_parent_relocation(requirement: Mapping[str, Any]) -> bool:
    relocate_if = requirement.get('relocate_if')
    if not isinstance(relocate_if, Mapping):
        return False
    modes = set(relocate_if.get('modes') or [])
    fallback = relocate_if.get('fallback_to')
    if fallback:
        modes.add(fallback)
    return bool(modes & {'unblock_from_closed_container', 'evacuate_from_goal_receptacle'})

def has_visibility_requirement(requirement: Mapping[str, Any]) -> bool:
    return requirement.get('must_be_visible_from_start') or requirement.get('must_not_be_visible_from_start') or 'visible_from_initial_pose' in requirement

def expected_visibility(requirement: Mapping[str, Any]) -> bool | None:
    if requirement.get('must_be_visible_from_start'):
        return True
    if requirement.get('must_not_be_visible_from_start'):
        return False
    if 'visible_from_initial_pose' in requirement:
        return bool(requirement['visible_from_initial_pose'])
    return None

def _collect_bindings(*, template: Mapping[str, Any], role_order: Sequence[str], candidates_by_role: Mapping[str, Sequence[Mapping[str, Any]]], role_index: int, current: dict[str, Mapping[str, Any]], bindings: list[dict[str, Mapping[str, Any]]], limit: int) -> None:
    if len(bindings) >= limit:
        return
    if role_index >= len(role_order):
        if binding_satisfies_initial_constraints(template, current):
            bindings.append(dict(current))
        return
    role_name = role_order[role_index]
    requirement = template['requires'][role_name]
    for candidate in candidates_by_role.get(role_name, []):
        if len(bindings) >= limit:
            return
        if not candidate_compatible_with_partial_binding(role_name, candidate, requirement, template, current):
            continue
        current[role_name] = candidate
        _collect_bindings(template=template, role_order=role_order, candidates_by_role=candidates_by_role, role_index=role_index + 1, current=current, bindings=bindings, limit=limit)
        del current[role_name]

def _binding_collection_limit(*, max_bindings: int, role_order: Sequence[str], candidates_by_role: Mapping[str, Sequence[Mapping[str, Any]]], template: Mapping[str, Any]) -> int:
    limit = max(0, int(max_bindings))
    if limit <= 0:
        return 0
    sensitive_indexes = [index for index, role_name in enumerate(role_order) if _role_requires_interaction_face(template.get('requires', {}).get(role_name, {}), candidates_by_role.get(role_name, ()))]
    if not sensitive_indexes:
        return limit
    required = limit
    for index in sensitive_indexes:
        block_size = 1
        for role_name in role_order[index + 1:]:
            block_size *= max(1, len(candidates_by_role.get(role_name, ())))
            if block_size >= 4096:
                block_size = 4096
                break
        required = max(required, (len(candidates_by_role.get(role_order[index], ())) or 1) * block_size)
    return min(4096, required)

def _role_requires_interaction_face(requirement: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> bool:
    if bool(requirement.get('interaction_faces_required')) or bool(requirement.get('must_be_openable')):
        return True
    return any((bool(candidate.get('openable')) or bool(candidate.get('interaction_faces')) or 'openable' in {str(value).lower() for value in candidate.get('abilities') or []} for candidate in candidates))

def _select_diverse_bindings(bindings: Sequence[Mapping[str, Mapping[str, Any]]], *, max_bindings: int, role_order: Sequence[str], candidates_by_role: Mapping[str, Sequence[Mapping[str, Any]]], template: Mapping[str, Any]) -> list[Mapping[str, Mapping[str, Any]]]:
    limit = max(0, int(max_bindings))
    if limit == 0:
        return []
    sensitive_roles = [role_name for role_name in role_order if _role_requires_interaction_face(template.get('requires', {}).get(role_name, {}), candidates_by_role.get(role_name, ()))]
    selected: list[Mapping[str, Mapping[str, Any]]] = []
    selected_indexes: set[int] = set()
    for role_name in sensitive_roles:
        seen: set[str] = set()
        for index, binding in enumerate(bindings):
            candidate = binding.get(role_name) or {}
            object_id = str(candidate.get('object_id') or candidate.get('category') or '')
            if not object_id or object_id in seen:
                continue
            seen.add(object_id)
            if index in selected_indexes:
                continue
            selected_indexes.add(index)
            selected.append(binding)
            if len(selected) >= limit:
                return selected
    for index, binding in enumerate(bindings):
        if index in selected_indexes:
            continue
        selected_indexes.add(index)
        selected.append(binding)
        if len(selected) >= limit:
            break
    return selected

def _extend_optional_roles(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], optional_roles: Sequence[str], candidates_by_role: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Mapping[str, Any]]:
    extended = dict(binding)
    for role_name in optional_roles:
        requirement = template['requires'][role_name]
        for candidate in candidates_by_role.get(role_name, []):
            if candidate_compatible_with_partial_binding(role_name, candidate, requirement, template, extended):
                extended[role_name] = candidate
                break
    return extended

def candidate_compatible_with_partial_binding(role_name: str, candidate: Mapping[str, Any], requirement: Mapping[str, Any], template: Mapping[str, Any], current: Mapping[str, Mapping[str, Any]]) -> bool:
    for other_role, other_candidate in current.items():
        from scoper.benchmark.common.task_identity_policy import enabled
        if enabled() and (not requirement.get('same_object_type_as')) and (not template['requires'][other_role].get('same_object_type_as')) and (candidate.get('object_type') == other_candidate.get('object_type')) and (not same_runtime_object(candidate, other_candidate)):
            return False
        if same_runtime_object(candidate, other_candidate) and (not same_object_allowed(role_name, other_role, requirement, template['requires'][other_role])):
            return False
    for other_role in requirement.get('different_from') or []:
        if other_role in current and same_runtime_object(candidate, current[other_role]):
            return False
    same_room_as = requirement.get('same_room_as')
    if same_room_as in current and (not compatible_same_room(candidate, current[same_room_as])):
        return False
    different_room_from = requirement.get('different_room_from')
    if different_room_from in current and (not compatible_different_room(candidate, current[different_room_from])):
        return False
    same_object_type_as = requirement.get('same_object_type_as')
    if same_object_type_as in current:
        if candidate.get('object_type') != current[same_object_type_as].get('object_type'):
            return False
    near_role = requirement.get('near_role')
    if near_role in current and (not compatible_near(candidate, current[near_role])):
        return False
    parent_of_role_base = requirement.get('parent_of_role_base')
    if parent_of_role_base in current:
        parent_id = str(current[parent_of_role_base].get('object_id'))
        if parent_id not in set(candidate.get('parent_receptacles') or []):
            return False
    return True

def same_runtime_object(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a.get('requires_spawn') or b.get('requires_spawn'):
        return False
    return a.get('object_id') == b.get('object_id')

def same_object_allowed(role_name: str, other_role: str, requirement: Mapping[str, Any], other_requirement: Mapping[str, Any]) -> bool:
    return other_role in set(requirement.get('allow_same_object_as') or []) or role_name in set(other_requirement.get('allow_same_object_as') or [])

def compatible_same_room(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a.get('requires_spawn') or b.get('requires_spawn'):
        return True
    return bool(a.get('room_id') and a.get('room_id') == b.get('room_id'))

def compatible_different_room(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a.get('requires_spawn') or b.get('requires_spawn'):
        return True
    return bool(a.get('room_id') and b.get('room_id') and (a.get('room_id') != b.get('room_id')))

def compatible_near(a: Mapping[str, Any], b: Mapping[str, Any], threshold: float=1.5) -> bool:
    if a.get('requires_spawn') or b.get('requires_spawn'):
        return True
    a_pos = a.get('position')
    b_pos = b.get('position')
    if not isinstance(a_pos, Mapping) or not isinstance(b_pos, Mapping):
        return False
    if 'x' not in a_pos or 'z' not in a_pos or 'x' not in b_pos or ('z' not in b_pos):
        return False
    return math.hypot(float(a_pos['x']) - float(b_pos['x']), float(a_pos['z']) - float(b_pos['z'])) <= threshold

def binding_satisfies_initial_constraints(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> bool:
    for constraint in template.get('initial_state_constraints') or []:
        if not isinstance(constraint, Mapping):
            continue
        ctype = constraint.get('type')
        if ctype not in {'object_in_receptacle', 'object_not_in_receptacle'}:
            continue
        obj = binding.get(str(constraint.get('object_role')))
        receptacle = binding.get(str(constraint.get('receptacle_role')))
        if obj is None or receptacle is None:
            continue
        if obj.get('requires_spawn'):
            continue
        in_receptacle = receptacle.get('object_id') in set(obj.get('parent_receptacles') or [])
        if ctype == 'object_in_receptacle' and (not in_receptacle):
            return False
        if ctype == 'object_not_in_receptacle' and (in_receptacle or profile_aabbs_overlap(obj, receptacle)):
            return False
    return True

def profile_aabbs_overlap(a: Mapping[str, Any], b: Mapping[str, Any], *, tolerance: float=0.01) -> bool:
    center_a = profile_center(a)
    center_b = profile_center(b)
    extent_a = profile_extent(a)
    extent_b = profile_extent(b)
    if center_a is None or center_b is None or extent_a is None or (extent_b is None):
        return False
    for axis in range(3):
        delta = abs(center_a[axis] - center_b[axis])
        limit = extent_a[axis] * 0.5 + extent_b[axis] * 0.5 - float(tolerance)
        if delta >= limit:
            return False
    return True

def profile_center(obj: Mapping[str, Any]) -> tuple[float, float, float] | None:
    center = obj.get('center')
    if isinstance(center, Sequence) and (not isinstance(center, (str, bytes))) and (len(center) == 3):
        try:
            values = tuple((float(value) for value in center))
        except Exception:
            return None
        if all((math.isfinite(value) for value in values)):
            return values
    position = obj.get('position')
    if isinstance(position, Mapping) and {'x', 'y', 'z'}.issubset(position):
        try:
            values = (float(position['x']), float(position['z']), float(position['y']))
        except Exception:
            return None
        if all((math.isfinite(value) for value in values)):
            return values
    return None

def profile_extent(obj: Mapping[str, Any]) -> tuple[float, float, float] | None:
    for key in ('extent', 'bbox_size'):
        extent = obj.get(key)
        if not isinstance(extent, Sequence) or isinstance(extent, (str, bytes)) or len(extent) != 3:
            continue
        try:
            values = tuple((max(0.0, float(value)) for value in extent))
        except Exception:
            continue
        if all((math.isfinite(value) for value in values)):
            return values
    return None

def serialize_binding(binding: Mapping[str, Mapping[str, Any]], template: Mapping[str, Any], scene_profile: Mapping[str, Any]) -> dict[str, Any]:
    roles = {role_name: serialize_role_binding(role_name, candidate) for role_name, candidate in sorted(binding.items())}
    mutations = set()
    runtime_caps = set(template_runtime_capabilities(template, scene_profile))
    for candidate in binding.values():
        if candidate.get('requires_spawn'):
            mutations.add('spawn_object')
            runtime_caps.add('spawn_placement')
    for constraint in template.get('initial_state_constraints') or []:
        if isinstance(constraint, Mapping) and constraint.get('type') not in {'object_in_receptacle', 'object_not_in_receptacle'}:
            mutations.add('state_override')
    if any((requires.get('relocate_if') for requires in (template.get('requires') or {}).values())):
        mutations.add('object_pose_relocation')
    return {'roles': roles, 'required_init_mutations': sorted(mutations), 'runtime_capabilities_required': sorted(runtime_caps), 'notes': binding_notes(binding, template)}

def serialize_role_binding(role_name: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    object_id = str(candidate.get('object_id'))
    source_kind = str(candidate.get('source_kind'))
    if candidate.get('requires_spawn'):
        object_id = 'spawnable:{}:{}'.format(role_name, candidate.get('category'))
    entry = {'object_id': object_id, 'catalog_object_id': candidate.get('object_id') if candidate.get('requires_spawn') else None, 'object_type': candidate.get('object_type'), 'category': candidate.get('category'), 'synset': candidate.get('synset'), 'room_id': candidate.get('room_id'), 'source_kind': source_kind, 'requires_spawn': bool(candidate.get('requires_spawn')), 'pickupable': bool(candidate.get('pickupable')), 'receptacle': bool(candidate.get('receptacle')), 'openable': bool(candidate.get('openable')), 'toggleable': bool(candidate.get('toggleable')), 'position': candidate.get('position')}
    for field in ('extent', 'bbox_size', 'scale'):
        vector = _number_vector(candidate.get(field), 3)
        if vector is not None:
            entry[field] = vector
    for field in ('root_position', 'root_orientation_xyzw'):
        vector = _number_vector(candidate.get(field), 3 if field == 'root_position' else 4)
        if vector is not None:
            entry[field] = vector
        elif field == 'root_position' and isinstance(candidate.get(field), Mapping):
            position = candidate[field]
            if all((key in position for key in ('x', 'y', 'z'))):
                try:
                    entry[field] = {key: float(position[key]) for key in ('x', 'y', 'z')}
                except (TypeError, ValueError):
                    pass
    root_yaw = candidate.get('root_yaw')
    if isinstance(root_yaw, (int, float)):
        entry['root_yaw'] = float(root_yaw)
    interaction_faces = candidate.get('interaction_faces')
    if isinstance(interaction_faces, Sequence) and (not isinstance(interaction_faces, (str, bytes))):
        entry['interaction_faces'] = [copy.deepcopy(dict(face)) for face in interaction_faces if isinstance(face, Mapping)]
    model = candidate.get('model')
    if isinstance(model, str) and model and (model != 'unknown'):
        entry['model'] = model
    return entry

def _number_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        vector = [float(v) for v in value]
    except Exception:
        return None
    if not all((math.isfinite(v) for v in vector)):
        return None
    return vector

def binding_notes(binding: Mapping[str, Mapping[str, Any]], template: Mapping[str, Any]) -> list[str]:
    notes = []
    for role_name, candidate in sorted(binding.items()):
        if candidate.get('requires_spawn'):
            notes.append('{}:spawnable_asset_requires_reset_spawn'.format(role_name))
        if candidate.get('visible_from_initial_pose') is None and has_visibility_requirement((template.get('requires') or {}).get(role_name, {})):
            notes.append('{}:visibility_requires_runtime_validation'.format(role_name))
    return notes

def template_runtime_capabilities(template: Mapping[str, Any], scene_profile: Mapping[str, Any]) -> list[str]:
    caps = set()
    for role_name, requirement in (template.get('requires') or {}).items():
        if has_visibility_requirement(requirement):
            caps.add('visibility_query_or_controlled_spawn_visibility')
        relocate_if = requirement.get('relocate_if')
        if isinstance(relocate_if, Mapping):
            for mode in relocate_if.get('modes') or []:
                caps.add('relocation:{}'.format(mode))
            if relocate_if.get('fallback_to'):
                caps.add('relocation:{}'.format(relocate_if['fallback_to']))
    if scene_profile.get('reachable', {}).get('status') != 'computed':
        if any((cap.startswith('relocation:') for cap in caps)):
            caps.add('reachable_map')
    if unsupported_template_predicates(template):
        caps.add('extended_goal_checker')
    return sorted(caps)

def unsupported_template_predicates(template: Mapping[str, Any]) -> list[str]:
    predicates = []
    for group in ('initial_state_constraints', 'goal_predicates'):
        for predicate in template.get(group) or []:
            if isinstance(predicate, Mapping):
                ptype = str(predicate.get('type') or '')
                if ptype and ptype not in SUPPORTED_MATCH_PREDICATES:
                    predicates.append(ptype)
    return sorted(set(predicates))
__all__ = ['DEFAULT_PROCTHOR_TEMPLATE_DIR', 'MATCHED_PROFILE_ONLY', 'MATCHED_REQUIRES_RUNTIME_VALIDATION', 'MATCHED_REQUIRES_SPAWN', 'MATCHED_STATUSES', 'REJECTED', 'load_procthor_templates', 'match_scene_to_templates', 'match_template_to_scene', 'role_candidates']
