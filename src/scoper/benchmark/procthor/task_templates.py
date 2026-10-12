from __future__ import annotations
import json
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from scoper.benchmark.procthor.object_taxonomy import ObjectTaxonomy, ObjectTaxonomyError
from scoper.benchmark.procthor.online_task_spec import ALLOWED_CATEGORIES, ALLOWED_RELOCATE_MODES
from scoper.benchmark.procthor.scene_profile_utils import profile_object_room_id
REQUIRED_TEMPLATE_FIELDS = ('template_id', 'template_version', 'category', 'requires', 'initial_state_constraints', 'goal_predicates', 'instruction_templates', 'expected_events', 'expected_brain_behaviors')
ROLE_REQUIREMENT_BOOL_FIELDS = ('optional', 'must_be_pickupable', 'must_not_be_pickupable', 'must_be_receptacle', 'must_be_openable', 'must_be_toggleable', 'must_be_sliceable', 'must_be_dirtyable', 'must_be_cookable', 'must_have_parent_receptacle', 'must_not_have_parent_receptacle', 'must_not_have_closed_parent_receptacle', 'must_be_visible_from_start', 'must_not_be_visible_from_start', 'visible_from_initial_pose', 'relocate_target_only')
ROLE_REQUIREMENT_STRING_FIELDS = ('taxonomy_group', 'same_room_as', 'different_room_from', 'same_object_type_as', 'near_role', 'parent_of_role_base')
ROLE_REQUIREMENT_STRING_LIST_FIELDS = ('object_type_allowlist', 'object_type_blocklist', 'different_from', 'allow_same_object_as')
SUPPORTED_TEMPLATE_CONSTRAINTS = frozenset({'object_in_receptacle', 'object_not_in_receptacle', 'object_open', 'object_closed', 'object_toggled_on', 'object_toggled_off', 'object_clean', 'object_dirty', 'object_sliced', 'object_hot', 'object_cool', 'object_filled'})
DISABLED_TEMPLATE_IDS = frozenset({'recover_faucet_and_dirty_dish_001', 'recover_faucet_on_then_food_to_fridge_001'})

class TaskTemplateError(ValueError):
    pass

def load_task_template(path: str) -> dict:
    try:
        template = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise TaskTemplateError("Invalid task template JSON '{}': {}.".format(path, exc)) from exc
    validate_task_template(template)
    return template

def is_task_template_disabled(template: Mapping[str, Any]) -> bool:
    return template.get('template_id') in DISABLED_TEMPLATE_IDS

def load_task_templates(path: str, *, include_disabled: bool=False) -> List[dict]:
    template_path = Path(path)
    if template_path.is_file():
        template = load_task_template(str(template_path))
        if not include_disabled and is_task_template_disabled(template):
            return []
        return [template]
    if not template_path.is_dir():
        raise TaskTemplateError('Task template path does not exist: {}'.format(path))
    templates = []
    for child in sorted(template_path.glob('*.json')):
        template = load_task_template(str(child))
        if not include_disabled and is_task_template_disabled(template):
            continue
        templates.append(template)
    if not templates:
        raise TaskTemplateError("No task templates found in '{}'.".format(path))
    return templates

def validate_task_template(template: Mapping[str, Any]) -> None:
    if not isinstance(template, Mapping):
        raise TaskTemplateError('Task template must be a mapping.')
    _require_fields(template, REQUIRED_TEMPLATE_FIELDS, 'task template')
    for field in ('template_id', 'template_version', 'category'):
        _require_non_empty_string(template, field, 'task template')
    if template['category'] not in ALLOWED_CATEGORIES:
        raise TaskTemplateError("Unknown template category '{}'. Allowed categories: {}.".format(template['category'], ', '.join(sorted(ALLOWED_CATEGORIES))))
    role_names = _validate_requires(template['requires'])
    _require_string_list(template['instruction_templates'], 'instruction_templates')
    _require_string_list(template['expected_events'], 'expected_events')
    _require_string_list(template['expected_brain_behaviors'], 'expected_brain_behaviors')
    _validate_constraints(template['initial_state_constraints'], role_names, 'initial_state_constraints', allow_empty=True)
    _validate_constraints(template['goal_predicates'], role_names, 'goal_predicates', allow_empty=False)
    _validate_instruction_placeholders(template, role_names)
    if 'disturbance_spec' in template:
        if not isinstance(template['disturbance_spec'], Mapping):
            raise TaskTemplateError('Template disturbance_spec must be a mapping.')

def role_candidates(scene_profile: Mapping[str, Any], role_requirement: Mapping[str, Any], taxonomy: ObjectTaxonomy) -> List[Mapping[str, Any]]:
    candidates = []
    for obj in scene_profile.get('objects', []):
        if _object_satisfies_requirement(obj, role_requirement, taxonomy, scene_profile):
            candidates.append(obj)
    return candidates

def find_first_role_binding(template: Mapping[str, Any], scene_profile: Mapping[str, Any], taxonomy: ObjectTaxonomy) -> Tuple[Optional[Dict[str, Mapping[str, Any]]], Optional[str]]:
    bindings, reason = enumerate_role_bindings(template, scene_profile, taxonomy, limit=1)
    if not bindings:
        return (None, reason)
    return (bindings[0], None)

def enumerate_role_bindings(template: Mapping[str, Any], scene_profile: Mapping[str, Any], taxonomy: ObjectTaxonomy, limit: Optional[int]=None) -> Tuple[List[Dict[str, Mapping[str, Any]]], Optional[str]]:
    validate_task_template(template)
    required_role_order = [role_name for role_name, requirement in template['requires'].items() if not requirement.get('optional', False) and (not requirement.get('relocate_target_only', False))]
    optional_role_order = [role_name for role_name, requirement in template['requires'].items() if requirement.get('optional', False) and (not requirement.get('relocate_target_only', False))]
    candidates_by_role = {}
    for role_name in required_role_order:
        candidates = role_candidates(scene_profile, template['requires'][role_name], taxonomy)
        if not candidates:
            return ([], "No candidates for role '{}'.".format(role_name))
        candidates_by_role[role_name] = sorted(candidates, key=_object_id)
    optional_candidates_by_role = {}
    for role_name in optional_role_order:
        candidates = role_candidates(scene_profile, template['requires'][role_name], taxonomy)
        optional_candidates_by_role[role_name] = sorted(candidates, key=_object_id)
    base_bindings: List[Dict[str, Mapping[str, Any]]] = []
    _collect_bindings(template=template, role_order=required_role_order, candidates_by_role=candidates_by_role, role_index=0, current={}, bindings=base_bindings, limit=limit)
    if not base_bindings:
        return ([], 'No compatible role binding satisfies template constraints.')
    bindings = [_extend_binding_with_optional_roles(template, base_binding, optional_role_order, optional_candidates_by_role) for base_binding in base_bindings]
    return (bindings, None)

def _extend_binding_with_optional_roles(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], optional_role_order: Sequence[str], candidates_by_role: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Mapping[str, Any]]:
    extended: Dict[str, Mapping[str, Any]] = dict(binding)
    for role_name in optional_role_order:
        role_requirement = template['requires'][role_name]
        for candidate in candidates_by_role.get(role_name, []):
            if not _candidate_compatible_with_partial_binding(role_name, candidate, role_requirement, template, extended):
                continue
            extended[role_name] = candidate
            break
    return extended

def _search_binding(template: Mapping[str, Any], role_order: Sequence[str], candidates_by_role: Mapping[str, Sequence[Mapping[str, Any]]], role_index: int, current: Dict[str, Mapping[str, Any]]) -> Optional[Dict[str, Mapping[str, Any]]]:
    if role_index >= len(role_order):
        if _binding_satisfies_constraints(template, current):
            return dict(current)
        return None
    role_name = role_order[role_index]
    role_requirement = template['requires'][role_name]
    for candidate in candidates_by_role[role_name]:
        if not _candidate_compatible_with_partial_binding(role_name, candidate, role_requirement, template, current):
            continue
        current[role_name] = candidate
        result = _search_binding(template, role_order, candidates_by_role, role_index + 1, current)
        if result is not None:
            return result
        del current[role_name]
    return None

def _collect_bindings(template: Mapping[str, Any], role_order: Sequence[str], candidates_by_role: Mapping[str, Sequence[Mapping[str, Any]]], role_index: int, current: Dict[str, Mapping[str, Any]], bindings: List[Dict[str, Mapping[str, Any]]], limit: Optional[int]) -> None:
    if limit is not None and len(bindings) >= limit:
        return
    if role_index >= len(role_order):
        if _binding_satisfies_constraints(template, current):
            bindings.append(dict(current))
        return
    role_name = role_order[role_index]
    role_requirement = template['requires'][role_name]
    for candidate in candidates_by_role[role_name]:
        if limit is not None and len(bindings) >= limit:
            return
        if not _candidate_compatible_with_partial_binding(role_name, candidate, role_requirement, template, current):
            continue
        current[role_name] = candidate
        _collect_bindings(template=template, role_order=role_order, candidates_by_role=candidates_by_role, role_index=role_index + 1, current=current, bindings=bindings, limit=limit)
        del current[role_name]

def _candidate_compatible_with_partial_binding(role_name: str, candidate: Mapping[str, Any], role_requirement: Mapping[str, Any], template: Mapping[str, Any], current: Mapping[str, Mapping[str, Any]]) -> bool:
    candidate_id = _object_id(candidate)
    for other_role, other_obj in current.items():
        other_id = _object_id(other_obj)
        from scoper.benchmark.common.task_identity_policy import enabled
        other_requirement = template['requires'][other_role]
        if enabled() and candidate_id != other_id and (_object_type(candidate) == _object_type(other_obj)) and (not role_requirement.get('relocate_target_only')) and (not other_requirement.get('relocate_target_only')) and (not role_requirement.get('same_object_type_as')) and (not other_requirement.get('same_object_type_as')):
            return False
        if candidate_id == other_id and (not _same_object_allowed(role_name, other_role, role_requirement, template['requires'][other_role])):
            return False
    different_from = set(role_requirement.get('different_from', []))
    for other_role in different_from:
        if other_role in current and candidate_id == _object_id(current[other_role]):
            return False
    same_room_as = role_requirement.get('same_room_as')
    if same_room_as in current:
        if not _room_constraint_deferred_to_relocation(template, role_requirement, 'same_room_as', same_room_as) and profile_object_room_id(candidate) != profile_object_room_id(current[same_room_as]):
            return False
    different_room_from = role_requirement.get('different_room_from')
    if different_room_from in current:
        candidate_room = profile_object_room_id(candidate)
        anchor_room = profile_object_room_id(current[different_room_from])
        if not _room_constraint_deferred_to_relocation(template, role_requirement, 'different_room_from', different_room_from) and (candidate_room is None or anchor_room is None or candidate_room == anchor_room):
            return False
    same_object_type_as = role_requirement.get('same_object_type_as')
    if same_object_type_as in current:
        if _object_type(candidate) != _object_type(current[same_object_type_as]):
            return False
    near_role = role_requirement.get('near_role')
    if near_role in current:
        if not _objects_near(candidate, current[near_role]):
            return False
    parent_of_role_base = role_requirement.get('parent_of_role_base')
    if parent_of_role_base in current:
        other_id = _object_id(current[parent_of_role_base])
        base_id = other_id.split('___')[0] if '___' in other_id else other_id
        cand_parents = candidate.get('parent_receptacles') or []
        if base_id not in cand_parents:
            return False
    return True

def _same_object_allowed(role_name: str, other_role: str, role_requirement: Mapping[str, Any], other_requirement: Mapping[str, Any]) -> bool:
    return other_role in set(role_requirement.get('allow_same_object_as', [])) or role_name in set(other_requirement.get('allow_same_object_as', []))

def _binding_satisfies_constraints(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> bool:
    if not _binding_satisfies_role_constraints(template, binding):
        return False
    for constraint in template.get('initial_state_constraints', []):
        if not _constraint_satisfied(constraint, binding, template):
            return False
    return True

def _binding_satisfies_role_constraints(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> bool:
    for role_name, role_requirement in template.get('requires', {}).items():
        if role_name not in binding:
            continue
        candidate = binding[role_name]
        same_room_as = role_requirement.get('same_room_as')
        if same_room_as in binding:
            if not _room_constraint_deferred_to_relocation(template, role_requirement, 'same_room_as', same_room_as) and profile_object_room_id(candidate) != profile_object_room_id(binding[same_room_as]):
                return False
        different_room_from = role_requirement.get('different_room_from')
        if different_room_from in binding:
            candidate_room = profile_object_room_id(candidate)
            anchor_room = profile_object_room_id(binding[different_room_from])
            if not _room_constraint_deferred_to_relocation(template, role_requirement, 'different_room_from', different_room_from) and (candidate_room is None or anchor_room is None or candidate_room == anchor_room):
                return False
        same_object_type_as = role_requirement.get('same_object_type_as')
        if same_object_type_as in binding:
            if _object_type(candidate) != _object_type(binding[same_object_type_as]):
                return False
        near_role = role_requirement.get('near_role')
        if near_role in binding and (not _objects_near(candidate, binding[near_role])):
            return False
        parent_of_role_base = role_requirement.get('parent_of_role_base')
        if parent_of_role_base in binding:
            other_id = _object_id(binding[parent_of_role_base])
            base_id = other_id.split('___')[0] if '___' in other_id else other_id
            if base_id not in set(candidate.get('parent_receptacles') or []):
                return False
    return True

def _room_constraint_deferred_to_relocation(template: Mapping[str, Any], role_requirement: Mapping[str, Any], constraint_field: str, referenced_role: Any) -> bool:
    if not isinstance(referenced_role, str) or not referenced_role:
        return False
    relocate_if = role_requirement.get('relocate_if')
    if not isinstance(relocate_if, Mapping):
        return False
    modes = _requirement_relocate_modes(role_requirement)
    if constraint_field == 'same_room_as':
        if 'same_room_as' in modes:
            anchor_role = relocate_if.get('anchor_role') or role_requirement.get('same_room_as')
            return anchor_role == referenced_role
        if 'near_primary_object_floor' in modes:
            anchor_role = relocate_if.get('to_floor_anchor_role')
            return anchor_role == referenced_role
        return False
    if constraint_field != 'different_room_from':
        return False
    anchor_role = None
    if 'near_primary_object_floor' in modes:
        anchor_role = relocate_if.get('to_floor_anchor_role')
    elif 'same_room_as' in modes:
        anchor_role = relocate_if.get('anchor_role') or role_requirement.get('same_room_as')
    if not isinstance(anchor_role, str) or not anchor_role:
        return False
    anchor_requirement = template.get('requires', {}).get(anchor_role, {})
    if not isinstance(anchor_requirement, Mapping):
        return False
    return anchor_requirement.get('different_room_from') == referenced_role

def _constraint_satisfied(constraint: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], template: Optional[Mapping[str, Any]]=None) -> bool:
    constraint_type = constraint.get('type')
    if constraint_type not in SUPPORTED_TEMPLATE_CONSTRAINTS:
        return True
    if constraint_type in {'object_in_receptacle', 'object_not_in_receptacle'}:
        obj = binding[constraint['object_role']]
        receptacle = binding[constraint['receptacle_role']]
        in_receptacle = _object_id(receptacle) in set(obj.get('parent_receptacles') or [])
        if constraint_type == 'object_not_in_receptacle' and in_receptacle and (template is not None) and ('evacuate_from_goal_receptacle' in _role_relocate_modes(template, constraint['object_role'])):
            return True
        return in_receptacle if constraint_type == 'object_in_receptacle' else not in_receptacle
    return True

def _object_satisfies_requirement(obj: Mapping[str, Any], role_requirement: Mapping[str, Any], taxonomy: ObjectTaxonomy, scene_profile: Optional[Mapping[str, Any]]=None) -> bool:
    object_type = taxonomy.normalize_type(_object_type(obj))
    if taxonomy.is_excluded(object_type):
        return False
    allowlist = role_requirement.get('object_type_allowlist')
    if allowlist and object_type not in set(allowlist):
        return False
    taxonomy_group = role_requirement.get('taxonomy_group')
    if taxonomy_group and (not taxonomy.object_type_in_group(object_type, taxonomy_group)):
        return False
    blocklist = set(role_requirement.get('object_type_blocklist', []))
    if object_type in blocklist:
        return False
    bool_checks = {'must_be_pickupable': 'pickupable', 'must_be_receptacle': 'receptacle', 'must_be_openable': 'openable', 'must_be_toggleable': 'toggleable', 'must_be_sliceable': 'sliceable', 'must_be_dirtyable': 'dirtyable', 'must_be_cookable': 'cookable'}
    for requirement_field, object_field in bool_checks.items():
        if role_requirement.get(requirement_field) and (not bool(obj.get(object_field, False))):
            return False
    negated_bool_checks = {'must_not_be_pickupable': 'pickupable'}
    for requirement_field, object_field in negated_bool_checks.items():
        if role_requirement.get(requirement_field) and bool(obj.get(object_field, False)):
            return False
    if role_requirement.get('must_have_parent_receptacle'):
        if not obj.get('parent_receptacles'):
            return False
    if role_requirement.get('must_not_have_parent_receptacle'):
        if obj.get('parent_receptacles'):
            return False
    if role_requirement.get('must_not_have_closed_parent_receptacle'):
        parents = obj.get('parent_receptacles') or []
        if parents and scene_profile is not None:
            obj_by_id = {o.get('object_id'): o for o in scene_profile.get('objects', [])}
            for parent_id in parents:
                parent = obj_by_id.get(parent_id, {})
                if parent.get('openable') and parent.get('is_open') is False:
                    relocate_modes = _requirement_relocate_modes(role_requirement)
                    if not ('unblock_from_closed_container' in relocate_modes or 'evacuate_from_goal_receptacle' in relocate_modes):
                        return False
    if role_requirement.get('must_be_visible_from_start'):
        if obj.get('visible_from_initial_pose') is not True:
            return False
    if role_requirement.get('must_not_be_visible_from_start'):
        if obj.get('visible_from_initial_pose') is not False:
            return False
    if 'visible_from_initial_pose' in role_requirement:
        expected = bool(role_requirement['visible_from_initial_pose'])
        actual = bool(obj.get('visible_from_initial_pose', False))
        if actual != expected:
            return False
    return True

def _role_relocate_modes(template: Mapping[str, Any], role_name: str) -> Set[str]:
    requirement = template.get('requires', {}).get(role_name, {})
    if not isinstance(requirement, Mapping):
        return set()
    return _requirement_relocate_modes(requirement)

def _requirement_relocate_modes(requirement: Mapping[str, Any]) -> Set[str]:
    relocate_if = requirement.get('relocate_if')
    if not isinstance(relocate_if, Mapping):
        return set()
    modes = set((mode for mode in relocate_if.get('modes', []) if isinstance(mode, str)))
    fallback = relocate_if.get('fallback_to')
    if isinstance(fallback, str):
        modes.add(fallback)
    return modes

def _validate_requires(requires: Any) -> Set[str]:
    if not isinstance(requires, Mapping) or not requires:
        raise TaskTemplateError("Template field 'requires' must be a non-empty mapping.")
    role_names: Set[str] = set()
    for role_name, role_requirement in requires.items():
        if not isinstance(role_name, str) or not role_name:
            raise TaskTemplateError('Template role names must be non-empty strings.')
        if not isinstance(role_requirement, Mapping):
            raise TaskTemplateError("Template role '{}' requirement must be a mapping.".format(role_name))
        _validate_role_requirement(role_name, role_requirement)
        role_names.add(role_name)
    for role_name, role_requirement in requires.items():
        _validate_role_requirement_references(role_name, role_requirement, role_names)
    return role_names

def _validate_role_requirement(role_name: str, role_requirement: Mapping[str, Any]) -> None:
    allowed_fields = set(ROLE_REQUIREMENT_BOOL_FIELDS) | set(ROLE_REQUIREMENT_STRING_FIELDS) | set(ROLE_REQUIREMENT_STRING_LIST_FIELDS) | {'relocate_if'}
    unknown = sorted((field for field in role_requirement if field not in allowed_fields))
    if unknown:
        raise TaskTemplateError("Template role '{}' has unknown requirement fields: {}.".format(role_name, ', '.join(unknown)))
    for field in ROLE_REQUIREMENT_BOOL_FIELDS:
        if field in role_requirement and (not isinstance(role_requirement[field], bool)):
            raise TaskTemplateError("Template role '{}' field '{}' must be a bool.".format(role_name, field))
    for field in ROLE_REQUIREMENT_STRING_FIELDS:
        if field in role_requirement:
            value = role_requirement[field]
            if not isinstance(value, str) or not value:
                raise TaskTemplateError("Template role '{}' field '{}' must be a string.".format(role_name, field))
    for field in ROLE_REQUIREMENT_STRING_LIST_FIELDS:
        if field in role_requirement:
            _require_string_list(role_requirement[field], "Template role '{}' field '{}'".format(role_name, field))
    if 'relocate_if' in role_requirement and (not isinstance(role_requirement['relocate_if'], Mapping)):
        raise TaskTemplateError("Template role '{}' field 'relocate_if' must be a mapping.".format(role_name))

def _validate_role_requirement_references(role_name: str, role_requirement: Mapping[str, Any], role_names: Set[str]) -> None:
    for field in ('different_from', 'allow_same_object_as'):
        for referenced_role in role_requirement.get(field, []):
            if referenced_role not in role_names:
                raise TaskTemplateError("Template role '{}' field '{}' references unknown role '{}'.".format(role_name, field, referenced_role))
    same_room_as = role_requirement.get('same_room_as')
    if same_room_as is not None and same_room_as not in role_names:
        raise TaskTemplateError("Template role '{}' field 'same_room_as' references unknown role '{}'.".format(role_name, same_room_as))
    different_room_from = role_requirement.get('different_room_from')
    if different_room_from is not None and different_room_from not in role_names:
        raise TaskTemplateError("Template role '{}' field 'different_room_from' references unknown role '{}'.".format(role_name, different_room_from))
    same_object_type_as = role_requirement.get('same_object_type_as')
    if same_object_type_as is not None and same_object_type_as not in role_names:
        raise TaskTemplateError("Template role '{}' field 'same_object_type_as' references unknown role '{}'.".format(role_name, same_object_type_as))
    near_role = role_requirement.get('near_role')
    if near_role is not None and near_role not in role_names:
        raise TaskTemplateError("Template role '{}' field 'near_role' references unknown role '{}'.".format(role_name, near_role))
    parent_of_role_base = role_requirement.get('parent_of_role_base')
    if parent_of_role_base is not None and parent_of_role_base not in role_names:
        raise TaskTemplateError("Template role '{}' field 'parent_of_role_base' references unknown role '{}'.".format(role_name, parent_of_role_base))
    relocate_if = role_requirement.get('relocate_if')
    if isinstance(relocate_if, Mapping):
        _validate_relocate_if(role_name, relocate_if, role_names)

def _validate_relocate_if(role_name: str, relocate_if: Mapping[str, Any], role_names: Set[str]) -> None:
    modes = relocate_if.get('modes')
    if not isinstance(modes, list) or not modes:
        raise TaskTemplateError("Template role '{}' relocate_if.modes must be a non-empty list.".format(role_name))
    for mode in modes:
        if mode not in ALLOWED_RELOCATE_MODES:
            raise TaskTemplateError("Template role '{}' relocate_if.modes contains invalid mode '{}'.".format(role_name, mode))
    floor_modes = {'near_primary_object_floor'}
    has_floor_mode = any((mode in floor_modes for mode in modes))
    has_receptacle_mode = any((mode not in floor_modes for mode in modes))
    if has_receptacle_mode:
        target = relocate_if.get('to_receptacle_role')
        if not isinstance(target, str) or not target:
            raise TaskTemplateError("Template role '{}' relocate_if.to_receptacle_role must be a string.".format(role_name))
        if target not in role_names:
            raise TaskTemplateError("Template role '{}' relocate_if.to_receptacle_role '{}' not in roles.".format(role_name, target))
    if has_floor_mode:
        anchor = relocate_if.get('to_floor_anchor_role')
        if not isinstance(anchor, str) or not anchor:
            raise TaskTemplateError("Template role '{}' relocate_if.to_floor_anchor_role must be a string.".format(role_name))
        if anchor not in role_names:
            raise TaskTemplateError("Template role '{}' relocate_if.to_floor_anchor_role '{}' not in roles.".format(role_name, anchor))
        validation = relocate_if.get('validation')
        if validation != 'interactable_poses':
            raise TaskTemplateError("Template role '{}' relocate_if.validation must be 'interactable_poses'.".format(role_name))
    fallback = relocate_if.get('fallback_to')
    if fallback is not None and fallback not in ALLOWED_RELOCATE_MODES:
        raise TaskTemplateError("Template role '{}' relocate_if.fallback_to '{}' is invalid.".format(role_name, fallback))

def _validate_constraints(constraints: Any, role_names: Set[str], context: str, allow_empty: bool) -> None:
    if not isinstance(constraints, list):
        raise TaskTemplateError("Template field '{}' must be a list.".format(context))
    if not allow_empty and (not constraints):
        raise TaskTemplateError("Template field '{}' must not be empty.".format(context))
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, Mapping):
            raise TaskTemplateError('Template {} item {} must be a mapping.'.format(context, index))
        _require_non_empty_string(constraint, 'type', '{} item {}'.format(context, index))
        _validate_role_references_in_mapping(constraint, role_names, '{} item {}'.format(context, index))

def _validate_instruction_placeholders(template: Mapping[str, Any], role_names: Set[str]) -> None:
    role_names = set(role_names)
    if template.get('template_id') == 'mid_execution_container_re_closed_001':
        role_names.add('target_object_2')
    formatter = string.Formatter()
    for instruction in template['instruction_templates']:
        for _, field_name, _, _ in formatter.parse(instruction):
            if field_name is None:
                continue
            root_name = field_name.split('.', 1)[0].split('[', 1)[0]
            if root_name not in role_names and root_name not in {'scope_room_label', 'scope_room_labels'}:
                raise TaskTemplateError("Instruction template references unknown role '{}'.".format(root_name))

def _validate_role_references_in_mapping(value: Any, role_names: Set[str], context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and (key == 'role' or key.endswith('_role')):
                if child not in role_names:
                    raise TaskTemplateError("{} references unknown role '{}'.".format(context, child))
            else:
                _validate_role_references_in_mapping(child, role_names, context)
    elif isinstance(value, list):
        for child in value:
            _validate_role_references_in_mapping(child, role_names, context)

def _object_id(obj: Mapping[str, Any]) -> str:
    object_id = obj.get('object_id') or obj.get('objectId')
    if not isinstance(object_id, str) or not object_id:
        raise TaskTemplateError('Profile object is missing object_id.')
    return object_id

def _object_type(obj: Mapping[str, Any]) -> str:
    object_type = obj.get('object_type') or obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    return _object_id(obj).split('|', 1)[0]

def _objects_near(a: Mapping[str, Any], b: Mapping[str, Any], threshold: float=1.5) -> bool:
    a_pos = a.get('position') or {}
    b_pos = b.get('position') or {}
    if not all((key in a_pos and key in b_pos for key in ('x', 'z'))):
        return False
    import math
    return math.hypot(float(a_pos['x']) - float(b_pos['x']), float(a_pos['z']) - float(b_pos['z'])) <= threshold

def _require_fields(mapping: Mapping[str, Any], required_fields: Iterable[str], context: str) -> None:
    missing = [field for field in required_fields if field not in mapping]
    if missing:
        raise TaskTemplateError('{} missing required fields: {}.'.format(context, ', '.join(missing)))

def _require_non_empty_string(mapping: Mapping[str, Any], field: str, context: str) -> None:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise TaskTemplateError("{} field '{}' must be a non-empty string.".format(context, field))

def _require_string_list(value: Any, context: str) -> None:
    if not isinstance(value, list):
        raise TaskTemplateError('{} must be a list.'.format(context))
    if not all((isinstance(item, str) and item for item in value)):
        raise TaskTemplateError('{} must contain only non-empty strings.'.format(context))
__all__ = ['DISABLED_TEMPLATE_IDS', 'TaskTemplateError', 'enumerate_role_bindings', 'find_first_role_binding', 'is_task_template_disabled', 'load_task_template', 'load_task_templates', 'role_candidates', 'validate_task_template']
