from __future__ import annotations
from typing import Any, Iterable, Mapping, Optional, Sequence, Set
from scoper.benchmark.common import validate_qa_overlay
ALLOWED_CATEGORIES = frozenset({'observation_conditioned_container_inspection', 'static_simple', 'hidden_precondition', 'mid_execution_discovery', 'dynamic_state_change', 'completion_timing', 'disturbance_recovery', 'failure_recovery', 'put_away_by_category', 'prepare_then_place', 'tidy_by_policy', 'recover_and_store_incident_object', 'verify_after_action', 'human_interactive_execution', 'mid_execution_state_disturbance', 'prepare_multi_stage', 'put_away_visual_judgment', 'recover_multi_incident', 'put_away_cross_room', 'explore_then_act', 'observation_conditioned_room_inspection', 'observation_conditioned_task', 'composite'})
V1_FAMILY_CATEGORIES = frozenset({'put_away_by_category', 'prepare_then_place', 'tidy_by_policy', 'recover_and_store_incident_object', 'verify_after_action', 'human_interactive_execution'})
M8_FAMILY_CATEGORIES = frozenset({'mid_execution_state_disturbance'})
M9_FAMILY_CATEGORIES = frozenset({'prepare_multi_stage', 'put_away_visual_judgment', 'recover_multi_incident', 'put_away_cross_room', 'explore_then_act', 'composite'})
M10_FAMILY_CATEGORIES = frozenset({'observation_conditioned_room_inspection', 'observation_conditioned_task'})
FAMILY_CATEGORIES = V1_FAMILY_CATEGORIES | M8_FAMILY_CATEGORIES | M9_FAMILY_CATEGORIES | M10_FAMILY_CATEGORIES
COMPOSITE_SUBTASK_CATEGORIES = frozenset(ALLOWED_CATEGORIES - {'composite'})
ALLOWED_SCOPE_TYPES = frozenset({'single_room', 'connected_area', 'whole_house'})
ALLOWED_HUMAN_PRIMARY_TYPES = frozenset({'simple_place', 'put_away_by_category'})
ALLOWED_HUMAN_EVENT_TYPES = frozenset({'qa', 'insert'})
ALLOWED_QA_QUESTION_TYPES = frozenset({'scene_yes_no', 'progress_status', 'post_place_next_action'})
ALLOWED_INSERT_PRIORITIES = frozenset({'before_stop'})
ALLOWED_HUMAN_ACTIVATION_TYPES = frozenset({'high_level_step', 'milestone'})
ALLOWED_HUMAN_MILESTONES = frozenset({'after_successful_navigate_to_role', 'after_successful_tool_on_role'})
ALLOWED_RELOCATE_MODES = frozenset({'unblock_from_closed_container', 'evacuate_from_goal_receptacle', 'route_visible_surface', 'route_near_surface', 'near_primary_object_floor', 'same_room_as'})
MAX_HUMAN_EVENTS_PER_TASK = 2
FAMILY_SPEC_BLOCK_BY_CATEGORY = {'put_away_by_category': 'put_away', 'prepare_then_place': 'prepare', 'tidy_by_policy': 'tidy', 'recover_and_store_incident_object': 'recover', 'verify_after_action': 'verify', 'human_interactive_execution': 'human', 'mid_execution_state_disturbance': 'mid_execution', 'prepare_multi_stage': 'prepare', 'put_away_visual_judgment': 'put_away', 'recover_multi_incident': 'recover', 'put_away_cross_room': 'put_away', 'explore_then_act': 'explore', 'observation_conditioned_room_inspection': 'observation_conditioned', 'observation_conditioned_task': 'observation_conditioned', 'composite': 'composite'}
ALLOWED_VALIDATION_STATUSES = frozenset({'candidate', 'validated', 'rejected'})
REQUIRED_TOP_LEVEL_FIELDS = ('id', 'version', 'backend', 'category', 'source', 'template', 'instruction', 'roles', 'init', 'goal', 'online_requirements', 'validation', 'tags')
REQUIRED_SOURCE_FIELDS = ('house_id', 'house_path', 'split')
REQUIRED_TEMPLATE_FIELDS = ('template_id', 'template_version')
REQUIRED_ROLE_FIELDS = ('object_id', 'object_type')
REQUIRED_INIT_FIELDS = ('agent_pose', 'object_state_overrides')
REQUIRED_GOAL_FIELDS = ('type', 'predicates', 'require_model_stop')
REQUIRED_ONLINE_REQUIREMENTS_FIELDS = ('expected_events', 'expected_brain_behaviors')
REQUIRED_VALIDATION_FIELDS = ('status', 'checks', 'failure_reason')

class TaskSpecValidationError(ValueError):
    pass

def validate_online_task_spec(task_spec: Mapping[str, Any]) -> None:
    if not isinstance(task_spec, Mapping):
        raise TaskSpecValidationError('Task spec must be a mapping.')
    _require_fields(task_spec, REQUIRED_TOP_LEVEL_FIELDS, 'task spec')
    _require_non_empty_string(task_spec, 'id', 'task spec')
    _require_non_empty_string(task_spec, 'version', 'task spec')
    _require_non_empty_string(task_spec, 'instruction', 'task spec')
    if task_spec['backend'] != 'procthor':
        raise TaskSpecValidationError("Task spec backend must be 'procthor'.")
    if task_spec['category'] not in ALLOWED_CATEGORIES:
        raise TaskSpecValidationError("Unknown task category '{}'. Allowed categories: {}.".format(task_spec['category'], ', '.join(sorted(ALLOWED_CATEGORIES))))
    _validate_source(task_spec['source'])
    _validate_template(task_spec['template'])
    role_names = _validate_roles(task_spec['roles'])
    _validate_init(task_spec['init'], role_names)
    _validate_goal(task_spec['goal'])
    _validate_online_requirements(task_spec['online_requirements'])
    _validate_validation_block(task_spec['validation'])
    _require_string_list(task_spec, 'tags', 'task spec')
    if 'family_spec' in task_spec:
        _validate_family_spec(task_spec, role_names)
    if 'qa_overlay' in task_spec:
        composite = (task_spec.get('family_spec') or {}).get('composite') or {}
        subtask_ids = [str(subtask.get('subtask_id')) for subtask in composite.get('subtasks') or [] if isinstance(subtask, Mapping) and subtask.get('subtask_id')]
        try:
            normalized_overlay = validate_qa_overlay(task_spec['qa_overlay'], role_names=sorted(role_names), subtask_ids=subtask_ids)
            overlay_ids = {question['id'] for question in normalized_overlay['questions']}
            collisions = overlay_ids.intersection(_human_event_ids(task_spec.get('family_spec')))
            if collisions:
                raise ValueError('qa_overlay IDs collide with legacy human events: {}'.format(', '.join(sorted(collisions))))
        except ValueError as exc:
            raise TaskSpecValidationError('Invalid qa_overlay: {}'.format(exc)) from exc
    _validate_role_references(task_spec, role_names)
    if 'disturbance_spec' in task_spec:
        _validate_disturbance_spec(task_spec['disturbance_spec'], role_names)

def _human_event_ids(value: Any) -> Set[str]:
    ids: Set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {'human_events', 'human_events_template'} and isinstance(child, list):
                ids.update((str(event.get('id')) for event in child if isinstance(event, Mapping) and event.get('id')))
            else:
                ids.update(_human_event_ids(child))
    elif isinstance(value, list):
        for child in value:
            ids.update(_human_event_ids(child))
    return ids

def validate_task_bank(tasks: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        raise TaskSpecValidationError('Task bank must be a sequence of task specs.')
    seen_ids: Set[str] = set()
    for index, task_spec in enumerate(tasks):
        validate_online_task_spec(task_spec)
        task_id = task_spec['id']
        if task_id in seen_ids:
            raise TaskSpecValidationError("Duplicate task id '{}' in task bank at index {}.".format(task_id, index))
        seen_ids.add(task_id)

def validate_task_set(task_set: Mapping[str, Any], task_ids: Optional[Iterable[str]]=None) -> None:
    if not isinstance(task_set, Mapping):
        raise TaskSpecValidationError('Task set must be a mapping.')
    _require_fields(task_set, ('name', 'version', 'task_ids'), 'task set')
    _require_non_empty_string(task_set, 'name', 'task set')
    _require_non_empty_string(task_set, 'version', 'task set')
    ids = task_set['task_ids']
    if not isinstance(ids, list):
        raise TaskSpecValidationError("Task set field 'task_ids' must be a list.")
    if not all((isinstance(task_id, str) and task_id for task_id in ids)):
        raise TaskSpecValidationError("Task set field 'task_ids' must contain non-empty strings.")
    seen_ids: Set[str] = set()
    for task_id in ids:
        if task_id in seen_ids:
            raise TaskSpecValidationError("Duplicate task id '{}' in task set '{}'.".format(task_id, task_set['name']))
        seen_ids.add(task_id)
    if task_ids is not None:
        valid_ids = set(task_ids)
        missing = sorted((task_id for task_id in ids if task_id not in valid_ids))
        if missing:
            raise TaskSpecValidationError("Task set '{}' references missing task ids: {}.".format(task_set['name'], ', '.join(missing)))

def collect_role_references(task_spec: Mapping[str, Any]) -> Set[str]:
    references: Set[str] = set()
    for section_name in ('init', 'goal', 'family_spec'):
        section = task_spec.get(section_name)
        _collect_role_references(section, references)
    return references

def _validate_source(source: Any) -> None:
    if not isinstance(source, Mapping):
        raise TaskSpecValidationError("Task spec field 'source' must be a mapping.")
    _require_fields(source, REQUIRED_SOURCE_FIELDS, 'source')
    for field in REQUIRED_SOURCE_FIELDS:
        _require_non_empty_string(source, field, 'source')

def _validate_template(template: Any) -> None:
    if not isinstance(template, Mapping):
        raise TaskSpecValidationError("Task spec field 'template' must be a mapping.")
    _require_fields(template, REQUIRED_TEMPLATE_FIELDS, 'template')
    for field in REQUIRED_TEMPLATE_FIELDS:
        _require_non_empty_string(template, field, 'template')

def _validate_roles(roles: Any) -> Set[str]:
    if not isinstance(roles, Mapping):
        raise TaskSpecValidationError("Task spec field 'roles' must be a mapping.")
    if not roles:
        raise TaskSpecValidationError("Task spec field 'roles' must not be empty.")
    role_names: Set[str] = set()
    for role_name, role in roles.items():
        if not isinstance(role_name, str) or not role_name:
            raise TaskSpecValidationError('Role names must be non-empty strings.')
        if not isinstance(role, Mapping):
            raise TaskSpecValidationError("Role '{}' must be a mapping.".format(role_name))
        _require_fields(role, REQUIRED_ROLE_FIELDS, "role '{}'".format(role_name))
        for field in REQUIRED_ROLE_FIELDS:
            _require_non_empty_string(role, field, "role '{}'".format(role_name))
        role_names.add(role_name)
    return role_names

def _validate_init(init: Any, role_names: Set[str]) -> None:
    if not isinstance(init, Mapping):
        raise TaskSpecValidationError("Task spec field 'init' must be a mapping.")
    _require_fields(init, REQUIRED_INIT_FIELDS, 'init')
    if not isinstance(init['agent_pose'], Mapping):
        raise TaskSpecValidationError("Init field 'agent_pose' must be a mapping.")
    if not isinstance(init['object_state_overrides'], list):
        raise TaskSpecValidationError("Init field 'object_state_overrides' must be a list.")
    for index, override in enumerate(init['object_state_overrides']):
        if not isinstance(override, Mapping):
            raise TaskSpecValidationError('Init override at index {} must be a mapping.'.format(index))
        _require_non_empty_string(override, 'type', 'init override {}'.format(index))
    _validate_object_pose_overrides(init.get('object_pose_overrides', []), role_names)
    if 'spawn_objects' in init:
        raise TaskSpecValidationError('init.spawn_objects is no longer supported; generated tasks must use real scene objects.')

def _validate_object_pose_overrides(overrides: Any, role_names: Set[str]) -> None:
    if not isinstance(overrides, list):
        raise TaskSpecValidationError('init.object_pose_overrides must be a list.')
    for index, override in enumerate(overrides):
        context = 'init.object_pose_overrides[{}]'.format(index)
        if not isinstance(override, Mapping):
            raise TaskSpecValidationError('{} must be a mapping.'.format(context))
        override_type = override.get('type')
        if override_type == 'relocate_object':
            _validate_relocate_object_pose_override(override, role_names, context)
        elif override_type == 'clear_receptacle_clutter':
            _validate_clear_receptacle_clutter_override(override, role_names, context)
        else:
            raise TaskSpecValidationError('{}.type must be one of: clear_receptacle_clutter, relocate_object.'.format(context))

def _validate_relocate_object_pose_override(override: Mapping[str, Any], role_names: Set[str], context: str) -> None:
    for field in ('object_role', 'mode'):
        if field not in override:
            raise TaskSpecValidationError("{} missing field '{}'.".format(context, field))
    mode = override.get('mode')
    if mode not in ALLOWED_RELOCATE_MODES:
        raise TaskSpecValidationError("{}.mode '{}' not in allowed relocate modes.".format(context, mode))
    for field in ('object_role',):
        value = override.get(field)
        if not isinstance(value, str) or not value:
            raise TaskSpecValidationError('{}.{} must be a non-empty string.'.format(context, field))
        if value not in role_names:
            raise TaskSpecValidationError("{}.{} '{}' not in roles.".format(context, field, value))
    if mode == 'near_primary_object_floor':
        _validate_floor_relocate_object_pose_override(override, role_names, context)
    else:
        _validate_receptacle_relocate_object_pose_override(override, role_names, context)
    for field in ('open_parent_during_relocate', 'open_target_during_relocate', 'keep_object_kinematic', 'prefer_receptacle_center', 'collision_checked_relocation', 'require_inside_receptacle', 'restore_parent_state_after_relocate'):
        if field in override and (not isinstance(override[field], bool)):
            raise TaskSpecValidationError('{}.{} must be a bool.'.format(context, field))

def _validate_receptacle_relocate_object_pose_override(override: Mapping[str, Any], role_names: Set[str], context: str) -> None:
    field = 'to_receptacle_role'
    if field not in override:
        raise TaskSpecValidationError("{} missing field '{}'.".format(context, field))
    value = override.get(field)
    if not isinstance(value, str) or not value:
        raise TaskSpecValidationError('{}.{} must be a non-empty string.'.format(context, field))
    if value not in role_names:
        raise TaskSpecValidationError("{}.{} '{}' not in roles.".format(context, field, value))

def _validate_floor_relocate_object_pose_override(override: Mapping[str, Any], role_names: Set[str], context: str) -> None:
    if 'to_receptacle_role' in override:
        raise TaskSpecValidationError('{}.to_receptacle_role is not allowed for near_primary_object_floor.'.format(context))
    anchor_role = override.get('to_floor_anchor_role')
    if not isinstance(anchor_role, str) or not anchor_role:
        raise TaskSpecValidationError('{}.to_floor_anchor_role must be a non-empty string.'.format(context))
    if anchor_role not in role_names:
        raise TaskSpecValidationError("{}.to_floor_anchor_role '{}' not in roles.".format(context, anchor_role))
    validation = override.get('validation')
    if validation != 'interactable_poses':
        raise TaskSpecValidationError("{}.validation must be 'interactable_poses'.".format(context))
    candidates = override.get('floor_position_candidates')
    if not isinstance(candidates, list) or not candidates:
        raise TaskSpecValidationError('{}.floor_position_candidates must be a non-empty list.'.format(context))
    for candidate_index, candidate in enumerate(candidates):
        candidate_context = '{}.floor_position_candidates[{}]'.format(context, candidate_index)
        if not isinstance(candidate, Mapping):
            raise TaskSpecValidationError('{} must be a mapping.'.format(candidate_context))
        for axis in ('x', 'y', 'z'):
            value = candidate.get(axis)
            if not isinstance(value, (int, float)):
                raise TaskSpecValidationError('{}.{} must be a number.'.format(candidate_context, axis))

def _validate_clear_receptacle_clutter_override(override: Mapping[str, Any], role_names: Set[str], context: str) -> None:
    for field in ('receptacle_role', 'object_ids'):
        if field not in override:
            raise TaskSpecValidationError("{} missing field '{}'.".format(context, field))
    receptacle_role = override.get('receptacle_role')
    if not isinstance(receptacle_role, str) or not receptacle_role:
        raise TaskSpecValidationError('{}.receptacle_role must be a non-empty string.'.format(context))
    if receptacle_role not in role_names:
        raise TaskSpecValidationError("{}.receptacle_role '{}' not in roles.".format(context, receptacle_role))
    object_ids = override.get('object_ids')
    if not isinstance(object_ids, list) or not object_ids:
        raise TaskSpecValidationError('{}.object_ids must be a non-empty list.'.format(context))
    for item in object_ids:
        if not isinstance(item, str) or not item:
            raise TaskSpecValidationError('{}.object_ids must contain only non-empty strings.'.format(context))
    if 'to_receptacle_object_id' in override:
        _require_non_empty_string(override, 'to_receptacle_object_id', context)
    if 'to_receptacle_object_type' in override:
        _require_non_empty_string(override, 'to_receptacle_object_type', context)

def _validate_goal(goal: Any) -> None:
    if not isinstance(goal, Mapping):
        raise TaskSpecValidationError("Task spec field 'goal' must be a mapping.")
    _require_fields(goal, REQUIRED_GOAL_FIELDS, 'goal')
    if goal['type'] != 'predicate':
        raise TaskSpecValidationError("Goal field 'type' must be 'predicate'.")
    if not isinstance(goal['predicates'], list):
        raise TaskSpecValidationError("Goal field 'predicates' must be a list.")
    if len(goal['predicates']) == 0:
        raise TaskSpecValidationError("Goal field 'predicates' must not be empty.")
    for index, predicate in enumerate(goal['predicates']):
        if not isinstance(predicate, Mapping):
            raise TaskSpecValidationError('Goal predicate at index {} must be a mapping.'.format(index))
        _require_non_empty_string(predicate, 'type', 'goal predicate {}'.format(index))
    if not isinstance(goal['require_model_stop'], bool):
        raise TaskSpecValidationError("Goal field 'require_model_stop' must be a bool.")

def _validate_online_requirements(online_requirements: Any) -> None:
    if not isinstance(online_requirements, Mapping):
        raise TaskSpecValidationError("Task spec field 'online_requirements' must be a mapping.")
    _require_fields(online_requirements, REQUIRED_ONLINE_REQUIREMENTS_FIELDS, 'online_requirements')
    for field in REQUIRED_ONLINE_REQUIREMENTS_FIELDS:
        _require_string_list(online_requirements, field, 'online_requirements')

def _validate_validation_block(validation: Any) -> None:
    if not isinstance(validation, Mapping):
        raise TaskSpecValidationError("Task spec field 'validation' must be a mapping.")
    _require_fields(validation, REQUIRED_VALIDATION_FIELDS, 'validation')
    if validation['status'] not in ALLOWED_VALIDATION_STATUSES:
        raise TaskSpecValidationError("Unknown validation status '{}'. Allowed statuses: {}.".format(validation['status'], ', '.join(sorted(ALLOWED_VALIDATION_STATUSES))))
    if not isinstance(validation['checks'], Mapping):
        raise TaskSpecValidationError("Validation field 'checks' must be a mapping.")
    failure_reason = validation['failure_reason']
    if failure_reason is not None and (not isinstance(failure_reason, str)):
        raise TaskSpecValidationError("Validation field 'failure_reason' must be a string or null.")

def _validate_role_references(task_spec: Mapping[str, Any], role_names: Set[str]) -> None:
    referenced_roles = collect_role_references(task_spec)
    missing = sorted((role for role in referenced_roles if role not in role_names))
    if missing:
        raise TaskSpecValidationError('Task spec references missing roles: {}.'.format(', '.join(missing)))

def _validate_family_spec(task_spec: Mapping[str, Any], role_names: Set[str]) -> None:
    family_spec = task_spec.get('family_spec')
    if not isinstance(family_spec, Mapping):
        raise TaskSpecValidationError('family_spec must be a mapping.')
    family = family_spec.get('family')
    if family not in FAMILY_CATEGORIES:
        raise TaskSpecValidationError("family_spec.family '{}' is not a registered family category.".format(family))
    category = task_spec.get('category')
    if family != category:
        raise TaskSpecValidationError("family_spec.family '{}' must match task category '{}'.".format(family, category))
    _validate_family_scope(family_spec.get('scope'))
    expected_block = FAMILY_SPEC_BLOCK_BY_CATEGORY[family]
    family_blocks = set(FAMILY_SPEC_BLOCK_BY_CATEGORY.values())
    present_blocks = [block for block in family_blocks if block in family_spec]
    if expected_block not in present_blocks:
        raise TaskSpecValidationError("family_spec for '{}' must contain '{}'.".format(family, expected_block))
    extra_blocks = sorted((block for block in present_blocks if block != expected_block))
    if extra_blocks:
        raise TaskSpecValidationError("family_spec for '{}' contains blocks for other families: {}.".format(family, ', '.join(extra_blocks)))
    if not isinstance(family_spec[expected_block], Mapping):
        raise TaskSpecValidationError('family_spec.{} must be a mapping.'.format(expected_block))
    if family == 'human_interactive_execution':
        _validate_family_human(family_spec[expected_block])
    if family == 'mid_execution_state_disturbance':
        _validate_family_mid_execution(family_spec[expected_block])
    if family == 'explore_then_act':
        _validate_family_explore(family_spec[expected_block])
    if family in {'observation_conditioned_room_inspection', 'observation_conditioned_task'}:
        _validate_family_observation_conditioned(family_spec[expected_block])
    if family == 'composite':
        _validate_family_composite(family_spec[expected_block], role_names)
    references: Set[str] = set()
    _collect_role_references(family_spec, references)
    missing = sorted((role for role in references if role not in role_names))
    if missing:
        raise TaskSpecValidationError('family_spec references missing roles: {}.'.format(', '.join(missing)))

def _validate_family_scope(scope: Any) -> None:
    if not isinstance(scope, Mapping):
        raise TaskSpecValidationError('family_spec.scope must be a mapping.')
    _require_fields(scope, ('type', 'label', 'room_ids'), 'family_spec.scope')
    scope_type = scope.get('type')
    if scope_type not in ALLOWED_SCOPE_TYPES:
        raise TaskSpecValidationError("family_spec.scope.type '{}' is invalid. Allowed: {}.".format(scope_type, ', '.join(sorted(ALLOWED_SCOPE_TYPES))))
    _require_non_empty_string(scope, 'label', 'family_spec.scope')
    room_ids = scope.get('room_ids')
    if not isinstance(room_ids, list) or not all((isinstance(item, str) for item in room_ids)):
        raise TaskSpecValidationError('family_spec.scope.room_ids must be a list of strings.')
    if scope_type != 'whole_house' and (not room_ids):
        raise TaskSpecValidationError("family_spec.scope.room_ids must not be empty unless scope.type is 'whole_house'.")

def _validate_family_human(human: Mapping[str, Any]) -> None:
    primary = human.get('primary_task')
    if not isinstance(primary, Mapping):
        raise TaskSpecValidationError('family_spec.human.primary_task must be a mapping.')
    primary_type = primary.get('type')
    if primary_type not in ALLOWED_HUMAN_PRIMARY_TYPES:
        raise TaskSpecValidationError("family_spec.human.primary_task.type '{}' is invalid. Allowed: {}.".format(primary_type, ', '.join(sorted(ALLOWED_HUMAN_PRIMARY_TYPES))))
    if primary_type == 'simple_place':
        _require_non_empty_string(primary, 'object_role', 'family_spec.human.primary_task')
        _require_non_empty_string(primary, 'receptacle_role', 'family_spec.human.primary_task')
    elif primary_type == 'put_away_by_category':
        put_away_ref = primary.get('put_away_ref')
        if not isinstance(put_away_ref, Mapping):
            raise TaskSpecValidationError('family_spec.human.primary_task.put_away_ref must be a mapping.')
        _require_non_empty_string(put_away_ref, 'semantic_category', 'family_spec.human.primary_task.put_away_ref')
        _require_string_list(put_away_ref, 'target_roles', 'family_spec.human.primary_task.put_away_ref')
        if not put_away_ref['target_roles']:
            raise TaskSpecValidationError('family_spec.human.primary_task.put_away_ref.target_roles must not be empty.')
        _require_string_list(put_away_ref, 'storage_roles', 'family_spec.human.primary_task.put_away_ref')
        _optional_string_list(put_away_ref, 'protected_distractor_roles', 'family_spec.human.primary_task.put_away_ref')
    events = human.get('human_events')
    if not isinstance(events, list):
        raise TaskSpecValidationError('family_spec.human.human_events must be a list.')
    if not events:
        raise TaskSpecValidationError('family_spec.human.human_events must not be empty.')
    if len(events) > MAX_HUMAN_EVENTS_PER_TASK:
        raise TaskSpecValidationError('family_spec.human.human_events may contain at most {} events.'.format(MAX_HUMAN_EVENTS_PER_TASK))
    seen_ids: Set[str] = set()
    before_stop_inserts = 0
    for index, event in enumerate(events):
        context = 'family_spec.human.human_events[{}]'.format(index)
        if not isinstance(event, Mapping):
            raise TaskSpecValidationError('{} must be a mapping.'.format(context))
        _require_non_empty_string(event, 'id', context)
        event_id = event['id']
        if event_id in seen_ids:
            raise TaskSpecValidationError("Duplicate human event id '{}'.".format(event_id))
        seen_ids.add(event_id)
        event_type = event.get('type')
        if event_type not in ALLOWED_HUMAN_EVENT_TYPES:
            raise TaskSpecValidationError("{}.type '{}' is invalid. Allowed: {}.".format(context, event_type, ', '.join(sorted(ALLOWED_HUMAN_EVENT_TYPES))))
        _validate_human_activation(event.get('activation'), context)
        if event_type == 'qa':
            _validate_human_qa_event(event, context)
        else:
            before_stop_inserts += _validate_human_insert_event(event, context)
    if before_stop_inserts > 1:
        raise TaskSpecValidationError('family_spec.human may contain at most one before_stop insert event.')

def _validate_human_activation(activation: Any, context: str) -> None:
    if not isinstance(activation, Mapping):
        raise TaskSpecValidationError('{}.activation must be a mapping.'.format(context))
    activation_type = activation.get('type')
    if activation_type not in ALLOWED_HUMAN_ACTIVATION_TYPES:
        raise TaskSpecValidationError("{}.activation.type '{}' is invalid. Allowed: {}.".format(context, activation_type, ', '.join(sorted(ALLOWED_HUMAN_ACTIVATION_TYPES))))
    if activation_type == 'milestone':
        milestone = activation.get('milestone')
        if milestone not in ALLOWED_HUMAN_MILESTONES:
            raise TaskSpecValidationError("{}.activation.milestone '{}' is invalid. Allowed: {}.".format(context, milestone, ', '.join(sorted(ALLOWED_HUMAN_MILESTONES))))
        _require_non_empty_string(activation, 'role', '{}.activation'.format(context))
        if milestone == 'after_successful_tool_on_role':
            _require_non_empty_string(activation, 'tool', '{}.activation'.format(context))
        return
    after_step = activation.get('after_step')
    if isinstance(after_step, bool) or not isinstance(after_step, int) or after_step < 0:
        raise TaskSpecValidationError('{}.activation.after_step must be a non-negative integer.'.format(context))

def _validate_human_qa_event(event: Mapping[str, Any], context: str) -> None:
    question_type = event.get('question_type')
    if question_type not in ALLOWED_QA_QUESTION_TYPES:
        raise TaskSpecValidationError("{}.question_type '{}' is invalid. Allowed: {}.".format(context, question_type, ', '.join(sorted(ALLOWED_QA_QUESTION_TYPES))))
    _require_non_empty_string(event, 'question', context)
    _require_string_list(event, 'answer_set', context)
    if not event['answer_set']:
        raise TaskSpecValidationError('{}.answer_set must not be empty.'.format(context))
    _require_non_empty_string(event, 'evaluator_answer', context)
    if event['evaluator_answer'] not in set(event['answer_set']):
        raise TaskSpecValidationError('{}.evaluator_answer must be one of answer_set.'.format(context))
    pred = event.get('evaluator_answer_predicate')
    if pred is not None and (not isinstance(pred, Mapping)):
        raise TaskSpecValidationError('{}.evaluator_answer_predicate must be a mapping.'.format(context))
    _optional_string_list(event, 'requires_visible_roles', context)

def _validate_human_insert_event(event: Mapping[str, Any], context: str) -> int:
    priority = event.get('priority')
    if priority not in ALLOWED_INSERT_PRIORITIES:
        raise TaskSpecValidationError("{}.priority '{}' is invalid. Allowed: {}.".format(context, priority, ', '.join(sorted(ALLOWED_INSERT_PRIORITIES))))
    _require_non_empty_string(event, 'instruction', context)
    goal_predicates = event.get('goal_predicates')
    if not isinstance(goal_predicates, list) or not goal_predicates:
        raise TaskSpecValidationError('{}.goal_predicates must be a non-empty list.'.format(context))
    for pred_index, predicate in enumerate(goal_predicates):
        pred_context = '{}.goal_predicates[{}]'.format(context, pred_index)
        if not isinstance(predicate, Mapping):
            raise TaskSpecValidationError('{} must be a mapping.'.format(pred_context))
        _require_non_empty_string(predicate, 'type', pred_context)
    return 1 if priority == 'before_stop' else 0
ALLOWED_MID_EXECUTION_PRIMARY_TYPES = frozenset({'state_re_dirty', 'object_relocated', 'container_re_closed'})
ALLOWED_MID_EXECUTION_DISTURBANCE_TYPES = frozenset({'dirty_object', 'relocate_object', 'close_object'})
ALLOWED_MID_EXECUTION_ACTIVATION_TYPES = frozenset({'high_level_step', 'after_tool_result'})
ALLOWED_EXPLORE_POST_DISCOVERY_TASK_TYPES = frozenset({'pickup_and_place', 'toggle', 'open', 'clean'})
ALLOWED_OBSERVATION_CONDITIONED_ISSUE_TYPES = frozenset({'faucet_on', 'open_container', 'dirty_dish', 'misplaced_object'})
ALLOWED_OBSERVATION_CONDITIONED_REPAIR_TEMPLATES = frozenset({'toggle_off', 'close', 'wash_then_place', 'place'})
ALLOWED_COMPOSITE_ORDERINGS = frozenset({'sequential'})

def _validate_family_mid_execution(mid_execution: Mapping[str, Any]) -> None:
    primary = mid_execution.get('primary_task')
    if not isinstance(primary, Mapping):
        raise TaskSpecValidationError('family_spec.mid_execution.primary_task must be a mapping.')
    primary_type = primary.get('type')
    if primary_type not in ALLOWED_MID_EXECUTION_PRIMARY_TYPES:
        raise TaskSpecValidationError("family_spec.mid_execution.primary_task.type '{}' is invalid. Allowed: {}.".format(primary_type, ', '.join(sorted(ALLOWED_MID_EXECUTION_PRIMARY_TYPES))))
    _require_non_empty_string(primary, 'object_role', 'family_spec.mid_execution.primary_task')
    if primary_type in {'state_re_dirty', 'object_relocated', 'container_re_closed'}:
        _require_non_empty_string(primary, 'receptacle_role', 'family_spec.mid_execution.primary_task')
    for optional_role in ('sink_role', 'faucet_role', 'container_role', 'staging_receptacle_role', 'interrupt_role', 'receptacle_open_role', 'receptacle_close_role'):
        if optional_role in primary:
            value = primary[optional_role]
            if value is not None and (not isinstance(value, str) or not value):
                raise TaskSpecValidationError('family_spec.mid_execution.primary_task.{} must be a non-empty string.'.format(optional_role))
    if 'interrupt_tools' in primary:
        tools = primary.get('interrupt_tools')
        if not isinstance(tools, list) or not tools:
            raise TaskSpecValidationError('family_spec.mid_execution.primary_task.interrupt_tools must be a non-empty list.')
        for index, tool in enumerate(tools):
            if not isinstance(tool, str) or not tool:
                raise TaskSpecValidationError('family_spec.mid_execution.primary_task.interrupt_tools[{}] must be a non-empty string.'.format(index))
    disturbance = mid_execution.get('disturbance')
    if not isinstance(disturbance, Mapping):
        raise TaskSpecValidationError('family_spec.mid_execution.disturbance must be a mapping.')
    dtype = disturbance.get('type')
    if dtype not in ALLOWED_MID_EXECUTION_DISTURBANCE_TYPES:
        raise TaskSpecValidationError("family_spec.mid_execution.disturbance.type '{}' is invalid. Allowed: {}.".format(dtype, ', '.join(sorted(ALLOWED_MID_EXECUTION_DISTURBANCE_TYPES))))
    _require_non_empty_string(disturbance, 'target_role', 'family_spec.mid_execution.disturbance')
    activation = disturbance.get('activation')
    if not isinstance(activation, Mapping):
        raise TaskSpecValidationError('family_spec.mid_execution.disturbance.activation must be a mapping.')
    activation_type = activation.get('type')
    if activation_type not in ALLOWED_MID_EXECUTION_ACTIVATION_TYPES:
        raise TaskSpecValidationError("family_spec.mid_execution.disturbance.activation.type '{}' is invalid. Allowed: {}.".format(activation_type, ', '.join(sorted(ALLOWED_MID_EXECUTION_ACTIVATION_TYPES))))
    _validate_disturbance_activation(activation, set(), 'family_spec.mid_execution.disturbance.activation', require_high_level_after_step=True)
    if 'requires_out_of_view' in disturbance and (not isinstance(disturbance['requires_out_of_view'], bool)):
        raise TaskSpecValidationError('family_spec.mid_execution.disturbance.requires_out_of_view must be a bool.')
    if 'once' in disturbance and (not isinstance(disturbance['once'], bool)):
        raise TaskSpecValidationError('family_spec.mid_execution.disturbance.once must be a bool.')

def _validate_family_explore(explore: Mapping[str, Any]) -> None:
    discovery = explore.get('discovery_spec')
    if not isinstance(discovery, Mapping):
        raise TaskSpecValidationError('family_spec.explore.discovery_spec must be a mapping.')
    _require_non_empty_string(discovery, 'target_role', 'family_spec.explore.discovery_spec')
    initially_visible = discovery.get('initially_visible')
    if not isinstance(initially_visible, bool):
        raise TaskSpecValidationError('family_spec.explore.discovery_spec.initially_visible must be a bool.')
    if initially_visible:
        raise TaskSpecValidationError('family_spec.explore.discovery_spec.initially_visible must be false.')
    hint_room = discovery.get('hint_room')
    if hint_room is not None and (not isinstance(hint_room, str) or not hint_room):
        raise TaskSpecValidationError('family_spec.explore.discovery_spec.hint_room must be a string or null.')
    max_steps = discovery.get('max_exploration_steps')
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise TaskSpecValidationError('family_spec.explore.discovery_spec.max_exploration_steps must be a positive integer.')
    post_task = explore.get('post_discovery_task')
    if not isinstance(post_task, Mapping):
        raise TaskSpecValidationError('family_spec.explore.post_discovery_task must be a mapping.')
    task_type = post_task.get('task_type')
    if task_type not in ALLOWED_EXPLORE_POST_DISCOVERY_TASK_TYPES:
        raise TaskSpecValidationError("family_spec.explore.post_discovery_task.task_type '{}' is invalid. Allowed: {}.".format(task_type, ', '.join(sorted(ALLOWED_EXPLORE_POST_DISCOVERY_TASK_TYPES))))
    if task_type == 'pickup_and_place':
        _require_non_empty_string(post_task, 'target_receptacle_role', 'family_spec.explore.post_discovery_task')
    if task_type == 'clean':
        for field in ('sink_role', 'faucet_role'):
            if field in post_task and (not isinstance(post_task[field], str) or not post_task[field]):
                raise TaskSpecValidationError('family_spec.explore.post_discovery_task.{} must be a string.'.format(field))

def _validate_family_observation_conditioned(observation_conditioned: Mapping[str, Any]) -> None:
    public_mode = observation_conditioned.get('public_discovery_mode')
    if public_mode not in {None, 'structured', 'visual_only'}:
        raise TaskSpecValidationError("family_spec.observation_conditioned.public_discovery_mode must be 'structured' or 'visual_only'.")
    require_visual = observation_conditioned.get('require_visual_evidence')
    if require_visual is not None and (not isinstance(require_visual, bool)):
        raise TaskSpecValidationError('family_spec.observation_conditioned.require_visual_evidence must be a bool.')
    rooms = observation_conditioned.get('rooms')
    if not isinstance(rooms, list) or not rooms:
        raise TaskSpecValidationError('family_spec.observation_conditioned.rooms must be a non-empty list.')
    for index, room in enumerate(rooms):
        if not isinstance(room, str) or not room:
            raise TaskSpecValidationError('family_spec.observation_conditioned.rooms[{}] must be a non-empty string.'.format(index))
    scan_policy = observation_conditioned.get('scan_policy')
    if scan_policy is not None:
        if not isinstance(scan_policy, Mapping):
            raise TaskSpecValidationError('family_spec.observation_conditioned.scan_policy must be a mapping.')
        if 'num_views' in scan_policy:
            _require_positive_int_value(scan_policy.get('num_views'), 'family_spec.observation_conditioned.scan_policy.num_views')
        value = scan_policy.get('require_room_entry_before_scan')
        if value is not None and (not isinstance(value, bool)):
            raise TaskSpecValidationError('family_spec.observation_conditioned.scan_policy.require_room_entry_before_scan must be a bool.')
    payloads = observation_conditioned.get('latent_payloads')
    if not isinstance(payloads, list) or not payloads:
        raise TaskSpecValidationError('family_spec.observation_conditioned.latent_payloads must be a non-empty list.')
    seen_ids: Set[str] = set()
    room_set = set(rooms)
    for index, payload in enumerate(payloads):
        context = 'family_spec.observation_conditioned.latent_payloads[{}]'.format(index)
        if not isinstance(payload, Mapping):
            raise TaskSpecValidationError('{} must be a mapping.'.format(context))
        issue_id = payload.get('latent_id') or payload.get('issue_id')
        if not isinstance(issue_id, str) or not issue_id:
            raise TaskSpecValidationError('{} requires non-empty latent_id or issue_id.'.format(context))
        if issue_id in seen_ids:
            raise TaskSpecValidationError("Duplicate observation-conditioned issue id '{}'.".format(issue_id))
        seen_ids.add(issue_id)
        room_id = payload.get('room_id')
        if not isinstance(room_id, str) or not room_id:
            raise TaskSpecValidationError('{}.room_id must be a non-empty string.'.format(context))
        if room_id not in room_set:
            raise TaskSpecValidationError("{}.room_id '{}' is not listed in observation_conditioned.rooms.".format(context, room_id))
        issue_type = payload.get('issue_type')
        if issue_type not in ALLOWED_OBSERVATION_CONDITIONED_ISSUE_TYPES:
            raise TaskSpecValidationError("{}.issue_type '{}' is invalid. Allowed: {}.".format(context, issue_type, ', '.join(sorted(ALLOWED_OBSERVATION_CONDITIONED_ISSUE_TYPES))))
        _require_non_empty_string(payload, 'object_role', context)
        repair_template = payload.get('repair_plan_template')
        if repair_template not in ALLOWED_OBSERVATION_CONDITIONED_REPAIR_TEMPLATES:
            raise TaskSpecValidationError("{}.repair_plan_template '{}' is invalid. Allowed: {}.".format(context, repair_template, ', '.join(sorted(ALLOWED_OBSERVATION_CONDITIONED_REPAIR_TEMPLATES))))
        support_roles = payload.get('support_roles')
        if support_roles is not None and (not isinstance(support_roles, Mapping)):
            raise TaskSpecValidationError('{}.support_roles must be a mapping.'.format(context))
        goal_predicates = payload.get('goal_predicates')
        if not isinstance(goal_predicates, list) or not goal_predicates:
            raise TaskSpecValidationError('{}.goal_predicates must be a non-empty list.'.format(context))
        for pred_index, predicate in enumerate(goal_predicates):
            pred_context = '{}.goal_predicates[{}]'.format(context, pred_index)
            if not isinstance(predicate, Mapping):
                raise TaskSpecValidationError('{} must be a mapping.'.format(pred_context))
            _require_non_empty_string(predicate, 'type', pred_context)

def _validate_family_composite(composite: Mapping[str, Any], role_names: Set[str]) -> None:
    ordering = composite.get('ordering')
    if ordering not in ALLOWED_COMPOSITE_ORDERINGS:
        raise TaskSpecValidationError("family_spec.composite.ordering '{}' is invalid. Allowed: {}.".format(ordering, ', '.join(sorted(ALLOWED_COMPOSITE_ORDERINGS))))
    subtasks = composite.get('subtasks')
    if not isinstance(subtasks, list):
        raise TaskSpecValidationError('family_spec.composite.subtasks must be a list.')
    if len(subtasks) < 2:
        raise TaskSpecValidationError('family_spec.composite.subtasks must contain at least 2 subtasks.')
    seen_ids: Set[str] = set()
    for index, subtask in enumerate(subtasks):
        context = 'family_spec.composite.subtasks[{}]'.format(index)
        if not isinstance(subtask, Mapping):
            raise TaskSpecValidationError('{} must be a mapping.'.format(context))
        _require_non_empty_string(subtask, 'subtask_id', context)
        subtask_id = subtask['subtask_id']
        if subtask_id in seen_ids:
            raise TaskSpecValidationError("family_spec.composite.subtasks has duplicate subtask_id '{}'.".format(subtask_id))
        seen_ids.add(subtask_id)
        _require_non_empty_string(subtask, 'category', context)
        category = subtask['category']
        if category not in COMPOSITE_SUBTASK_CATEGORIES:
            raise TaskSpecValidationError("{} category '{}' is not supported in composite subtasks.".format(context, category))
        if category == 'composite':
            raise TaskSpecValidationError('{} recursive composite subtasks are not supported.'.format(context))
        family_spec = subtask.get('family_spec')
        if category in FAMILY_CATEGORIES:
            if not isinstance(family_spec, Mapping):
                raise TaskSpecValidationError("{}.family_spec must be a mapping for family child '{}'.".format(context, category))
            if family_spec.get('family') != category:
                raise TaskSpecValidationError("{}.family_spec.family '{}' must match subtask category '{}'.".format(context, family_spec.get('family'), category))
            _validate_family_spec({'category': category, 'family_spec': family_spec}, role_names)
        elif family_spec is not None:
            raise TaskSpecValidationError('{}.family_spec is only valid for registered family children.'.format(context))
        goal_predicates = subtask.get('goal_predicates')
        if not isinstance(goal_predicates, list) or not goal_predicates:
            raise TaskSpecValidationError('{}.goal_predicates must be a non-empty list.'.format(context))
        for pred_index, predicate in enumerate(goal_predicates):
            if not isinstance(predicate, Mapping):
                raise TaskSpecValidationError('{}.goal_predicates[{}] must be a mapping.'.format(context, pred_index))
            _require_non_empty_string(predicate, 'type', '{}.goal_predicates[{}]'.format(context, pred_index))
        disturbance_spec = subtask.get('disturbance_spec')
        if disturbance_spec is not None:
            _validate_disturbance_spec(disturbance_spec, role_names)

def _collect_role_references(value: Any, references: Set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and (key == 'role' or key.endswith('_role')):
                if isinstance(child, str) and child:
                    references.add(child)
                elif child is not None:
                    raise TaskSpecValidationError("Role reference '{}' must be a non-empty string.".format(key))
            elif isinstance(key, str) and (key == 'roles' or key.endswith('_roles')):
                _collect_role_list_references(key, child, references)
            else:
                _collect_role_references(child, references)
    elif isinstance(value, list):
        for child in value:
            _collect_role_references(child, references)

def _collect_role_list_references(key: str, value: Any, references: Set[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if not value:
            raise TaskSpecValidationError("Role reference '{}' must be a non-empty string.".format(key))
        references.add(value)
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                references.add(item)
            elif isinstance(item, (Mapping, list)):
                _collect_role_references(item, references)
            else:
                raise TaskSpecValidationError("Role reference list '{}' must contain role strings.".format(key))
        return
    if isinstance(value, Mapping):
        for item in value.values():
            if isinstance(item, str) and item:
                references.add(item)
            elif isinstance(item, list):
                _collect_role_list_references(key, item, references)
            elif isinstance(item, Mapping):
                _collect_role_references(item, references)
            elif item is not None:
                raise TaskSpecValidationError("Role reference mapping '{}' must contain role strings.".format(key))
        return
    raise TaskSpecValidationError("Role reference '{}' must be a string, list, or mapping.".format(key))

def _require_fields(mapping: Mapping[str, Any], required_fields: Iterable[str], context: str) -> None:
    missing = [field for field in required_fields if field not in mapping]
    if missing:
        raise TaskSpecValidationError('{} missing required fields: {}.'.format(context, ', '.join(missing)))

def _require_non_empty_string(mapping: Mapping[str, Any], field: str, context: str) -> None:
    if field not in mapping:
        raise TaskSpecValidationError("{} missing required field '{}'.".format(context, field))
    value = mapping[field]
    if not isinstance(value, str) or not value:
        raise TaskSpecValidationError("{} field '{}' must be a non-empty string.".format(context, field))

def _require_string_list(mapping: Mapping[str, Any], field: str, context: str) -> None:
    if field not in mapping:
        raise TaskSpecValidationError("{} missing required field '{}'.".format(context, field))
    value = mapping[field]
    if not isinstance(value, list):
        raise TaskSpecValidationError("{} field '{}' must be a list.".format(context, field))
    if not all((isinstance(item, str) for item in value)):
        raise TaskSpecValidationError("{} field '{}' must contain only strings.".format(context, field))

def _optional_string_list(mapping: Mapping[str, Any], field: str, context: str) -> None:
    if field not in mapping:
        return
    _require_string_list(mapping, field, context)
ALLOWED_DISTURBANCE_TYPES = frozenset({'close_object', 'open_object', 'toggle_on', 'toggle_off', 'restore_initial_state', 'dirty_object', 'relocate_object'})
ALLOWED_FAILURE_MODES = frozenset({'block', 'report_fail', 'delayed_success', 'partial_execution', 'wrong_target', 'grasp_slip', 'drop_beside', 'partial_open', 'partial_close', 'nav_blocked_partway', 'nav_wrong_target'})

def _validate_disturbance_spec(spec: Any, role_names: Set[str]) -> None:
    if not isinstance(spec, Mapping):
        raise TaskSpecValidationError('disturbance_spec must be a mapping.')
    disturbances = spec.get('disturbances')
    if disturbances is not None:
        if not isinstance(disturbances, list):
            raise TaskSpecValidationError('disturbance_spec.disturbances must be a list.')
        for i, d in enumerate(disturbances):
            if not isinstance(d, Mapping):
                raise TaskSpecValidationError('disturbance_spec.disturbances[{}] must be a mapping.'.format(i))
            dtype = d.get('type')
            if dtype not in ALLOWED_DISTURBANCE_TYPES:
                raise TaskSpecValidationError("disturbance_spec.disturbances[{}].type '{}' is not allowed.".format(i, dtype))
            target_role = d.get('target_role')
            if target_role and target_role not in role_names:
                raise TaskSpecValidationError("disturbance_spec.disturbances[{}].target_role '{}' not in roles.".format(i, target_role))
            activation = d.get('activation')
            if activation is not None:
                if not isinstance(activation, Mapping):
                    raise TaskSpecValidationError('disturbance_spec.disturbances[{}].activation must be a mapping.'.format(i))
                _validate_disturbance_activation(activation, role_names, 'disturbance_spec.disturbances[{}].activation'.format(i), require_high_level_after_step=False)
    failures = spec.get('failures')
    if failures is not None:
        if not isinstance(failures, list):
            raise TaskSpecValidationError('disturbance_spec.failures must be a list.')
        for i, f in enumerate(failures):
            if not isinstance(f, Mapping):
                raise TaskSpecValidationError('disturbance_spec.failures[{}] must be a mapping.'.format(i))
            mode = f.get('mode')
            if mode not in ALLOWED_FAILURE_MODES:
                raise TaskSpecValidationError("disturbance_spec.failures[{}].mode '{}' is not allowed.".format(i, mode))
            target_role = f.get('target_role')
            if target_role and target_role not in role_names:
                raise TaskSpecValidationError("disturbance_spec.failures[{}].target_role '{}' not in roles.".format(i, target_role))

def _validate_disturbance_activation(activation: Mapping[str, Any], role_names: Set[str], context: str, require_high_level_after_step: bool) -> None:
    activation_type = activation.get('type')
    if activation_type not in ALLOWED_MID_EXECUTION_ACTIVATION_TYPES:
        raise TaskSpecValidationError("{}.type '{}' is invalid. Allowed: {}.".format(context, activation_type, ', '.join(sorted(ALLOWED_MID_EXECUTION_ACTIVATION_TYPES))))
    if activation_type == 'high_level_step':
        if require_high_level_after_step or 'after_step' in activation:
            _require_nonnegative_int_value(activation.get('after_step'), '{}.after_step'.format(context))
        return
    tool = activation.get('tool')
    tools = activation.get('tools')
    if tool is None and tools is None:
        raise TaskSpecValidationError('{} must include tool or tools for after_tool_result.'.format(context))
    if tool is not None and (not isinstance(tool, str) or not tool):
        raise TaskSpecValidationError('{}.tool must be a non-empty string.'.format(context))
    if tools is not None:
        if not isinstance(tools, list) or not tools:
            raise TaskSpecValidationError('{}.tools must be a non-empty list.'.format(context))
        for index, value in enumerate(tools):
            if not isinstance(value, str) or not value:
                raise TaskSpecValidationError('{}.tools[{}] must be a non-empty string.'.format(context, index))
    for key in ('target_role', 'required_parent_role'):
        role = activation.get(key)
        if role is not None:
            if not isinstance(role, str) or not role:
                raise TaskSpecValidationError('{}.{} must be a non-empty string.'.format(context, key))
            if role_names and role not in role_names:
                raise TaskSpecValidationError("{}.{} '{}' not in roles.".format(context, key, role))
    for key in ('target_roles', 'exclude_target_roles'):
        values = activation.get(key)
        if values is None:
            continue
        if not isinstance(values, list):
            raise TaskSpecValidationError('{}.{} must be a list.'.format(context, key))
        for index, role in enumerate(values):
            if not isinstance(role, str) or not role:
                raise TaskSpecValidationError('{}.{}[{}] must be a non-empty string.'.format(context, key, index))
            if role_names and role not in role_names:
                raise TaskSpecValidationError("{}.{}[{}] '{}' not in roles.".format(context, key, index, role))
    for key in ('success', 'requires_target_not_held', 'distinct_success_only'):
        value = activation.get(key)
        if value is not None and (not isinstance(value, bool)):
            raise TaskSpecValidationError('{}.{} must be a bool.'.format(context, key))
    for key in ('min_delay_steps', 'after_distinct_tool_count', 'distinct_tool_count'):
        if key in activation:
            _require_nonnegative_int_value(activation.get(key), '{}.{}'.format(context, key))

def _require_nonnegative_int_value(value: Any, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TaskSpecValidationError('{} must be a non-negative integer.'.format(context))

def _require_positive_int_value(value: Any, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TaskSpecValidationError('{} must be a positive integer.'.format(context))
__all__ = ['ALLOWED_CATEGORIES', 'ALLOWED_HUMAN_ACTIVATION_TYPES', 'ALLOWED_HUMAN_EVENT_TYPES', 'ALLOWED_HUMAN_MILESTONES', 'ALLOWED_HUMAN_PRIMARY_TYPES', 'ALLOWED_INSERT_PRIORITIES', 'ALLOWED_QA_QUESTION_TYPES', 'ALLOWED_SCOPE_TYPES', 'ALLOWED_VALIDATION_STATUSES', 'COMPOSITE_SUBTASK_CATEGORIES', 'FAMILY_SPEC_BLOCK_BY_CATEGORY', 'FAMILY_CATEGORIES', 'M8_FAMILY_CATEGORIES', 'M9_FAMILY_CATEGORIES', 'M10_FAMILY_CATEGORIES', 'MAX_HUMAN_EVENTS_PER_TASK', 'TaskSpecValidationError', 'V1_FAMILY_CATEGORIES', 'collect_role_references', 'validate_online_task_spec', 'validate_task_bank', 'validate_task_set']
