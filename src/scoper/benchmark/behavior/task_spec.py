from __future__ import annotations
from typing import Any, Iterable, Mapping, Set
from scoper.benchmark.common import validate_qa_overlay
SUPPORTED_GOAL_PREDICATES = frozenset({'object_on_receptacle', 'object_in_receptacle', 'object_not_in_receptacle', 'object_in_any_receptacle', 'object_not_in_any_receptacle', 'object_held', 'object_not_held', 'object_open', 'object_closed', 'object_toggled_on', 'object_toggled_off', 'object_clean', 'object_dirty', 'object_sliced', 'object_filled', 'object_soaked', 'object_cooked', 'object_hot', 'object_cool', 'agent_near'})
ROLE_ARG_KEYS = ('object', 'receptacle')
NON_ROLE_ARG_KEYS = ('substance', 'threshold')
INIT_SPATIAL_PREDICATES = frozenset({'ontop', 'inside', 'inroom'})
INIT_STATE_PREDICATES = frozenset({'open', 'closed', 'toggled_on', 'toggled_off', 'dirty', 'clean', 'sliced', 'cooled', 'filled', 'saturated'})
INIT_PREDICATES = INIT_SPATIAL_PREDICATES | INIT_STATE_PREDICATES
INIT_BINARY_ROLE_PREDICATES = frozenset({'ontop', 'inside'})
REQUIRED_TOP_LEVEL_FIELDS = ('id', 'version', 'backend', 'category', 'source', 'instruction', 'roles', 'init', 'goal')
REQUIRED_SOURCE_FIELDS = ('type', 'scene_model')
REQUIRED_ROLE_FIELDS = ('synset',)
REQUIRED_GOAL_FIELDS = ('type', 'predicates', 'require_model_stop')
ALLOWED_SOURCE_TYPES = frozenset({'template', 'bddl_problem'})

class TaskSpecValidationError(ValueError):
    pass

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

def validate_behavior_task_spec(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping):
        raise TaskSpecValidationError('Task spec must be a mapping.')
    _require_fields(spec, REQUIRED_TOP_LEVEL_FIELDS, 'task spec')
    _require_non_empty_string(spec, 'id', 'task spec')
    _require_non_empty_string(spec, 'version', 'task spec')
    _require_non_empty_string(spec, 'instruction', 'task spec')
    _require_non_empty_string(spec, 'category', 'task spec')
    if 'qa_overlay' in spec:
        role_names = sorted((str(role) for role in spec.get('roles') or {}))
        composite = (spec.get('family_spec') or {}).get('composite') or {}
        subtask_ids = [str(subtask.get('subtask_id')) for subtask in composite.get('subtasks') or [] if isinstance(subtask, Mapping) and subtask.get('subtask_id')]
        try:
            normalized_overlay = validate_qa_overlay(spec['qa_overlay'], role_names=role_names, subtask_ids=subtask_ids)
            overlay_ids = {question['id'] for question in normalized_overlay['questions']}
            collisions = overlay_ids.intersection(_human_event_ids(spec.get('family_spec')))
            if collisions:
                raise ValueError('qa_overlay IDs collide with legacy human events: {}'.format(', '.join(sorted(collisions))))
        except ValueError as exc:
            raise TaskSpecValidationError('Invalid qa_overlay: {}'.format(exc)) from exc
    if spec['backend'] != 'behavior':
        raise TaskSpecValidationError("Task spec backend must be 'behavior'.")
    _validate_source(spec['source'])
    role_names = _validate_roles(spec['roles'])
    _validate_init(spec['init'], role_names)
    _validate_goal(spec['goal'], role_names)
    _validate_family_spec(spec, role_names)
    _validate_disturbance_spec(spec)
    _validate_failure_spec(spec)
PREPARE_STAGE_TYPES = frozenset({'heat', 'cool', 'slice', 'wash', 'store', 'place'})

def _validate_family_spec(spec: Mapping[str, Any], role_names: set) -> None:
    fam = spec.get('family_spec')
    if fam is None:
        return
    if not isinstance(fam, Mapping):
        raise TaskSpecValidationError('family_spec must be a mapping.')
    prepare = fam.get('prepare')
    if prepare is not None:
        if not isinstance(prepare, Mapping):
            raise TaskSpecValidationError('family_spec.prepare must be a mapping.')
        stages = prepare.get('stages')
        if stages is not None:
            if not isinstance(stages, list) or not stages:
                raise TaskSpecValidationError('family_spec.prepare.stages must be a non-empty list when present.')
            for st in stages:
                if not isinstance(st, Mapping):
                    raise TaskSpecValidationError('Each prepare stage must be a mapping.')
                if st.get('type') not in PREPARE_STAGE_TYPES:
                    raise TaskSpecValidationError("Unknown prepare stage type '{}'. Allowed: {}".format(st.get('type'), sorted(PREPARE_STAGE_TYPES)))
    observation_conditioned = fam.get('observation_conditioned')
    if observation_conditioned is not None:
        _validate_observation_conditioned_family_spec(spec, observation_conditioned, role_names)
    composite = fam.get('composite')
    if composite is not None:
        _validate_composite_family_spec(composite, role_names)

def _validate_composite_family_spec(composite: Any, role_names: set) -> None:
    if not isinstance(composite, Mapping):
        raise TaskSpecValidationError('family_spec.composite must be a mapping.')
    if composite.get('ordering') != 'sequential':
        raise TaskSpecValidationError("family_spec.composite.ordering must be 'sequential'.")
    subtasks = composite.get('subtasks')
    if not isinstance(subtasks, list) or len(subtasks) < 2:
        raise TaskSpecValidationError('family_spec.composite.subtasks must contain at least two subtasks.')
    subtask_ids = set()
    for index, subtask in enumerate(subtasks):
        if not isinstance(subtask, Mapping):
            raise TaskSpecValidationError('Composite subtask {} must be a mapping.'.format(index))
        subtask_id = subtask.get('subtask_id')
        if not isinstance(subtask_id, str) or not subtask_id:
            raise TaskSpecValidationError('Composite subtask {} requires non-empty subtask_id.'.format(index))
        if subtask_id in subtask_ids:
            raise TaskSpecValidationError("Composite subtask_id '{}' is duplicated.".format(subtask_id))
        subtask_ids.add(subtask_id)
        category = subtask.get('category') or subtask.get('family')
        if not isinstance(category, str) or not category:
            raise TaskSpecValidationError("Composite subtask '{}' requires category.".format(subtask_id))
        family_spec = subtask.get('family_spec')
        if family_spec is not None and (not isinstance(family_spec, Mapping)):
            raise TaskSpecValidationError("Composite subtask '{}' family_spec must be a mapping.".format(subtask_id))
        goal_predicates = subtask.get('goal_predicates')
        if not isinstance(goal_predicates, list) or not goal_predicates:
            raise TaskSpecValidationError("Composite subtask '{}' goal_predicates must be a non-empty list.".format(subtask_id))
        for predicate in goal_predicates:
            if not isinstance(predicate, Mapping):
                raise TaskSpecValidationError("Composite subtask '{}' goal predicate must be a mapping.".format(subtask_id))
            ptype = predicate.get('type')
            if ptype not in SUPPORTED_GOAL_PREDICATES:
                raise TaskSpecValidationError("Composite subtask '{}' has unsupported goal predicate '{}'.".format(subtask_id, ptype))
            for role_ref in _iter_role_refs(predicate):
                if role_ref not in role_names:
                    raise TaskSpecValidationError("Composite subtask '{}' predicate references unknown role '{}'.".format(subtask_id, role_ref))

def _validate_observation_conditioned_family_spec(spec: Mapping[str, Any], observation_conditioned: Any, role_names: set) -> None:
    if not isinstance(observation_conditioned, Mapping):
        raise TaskSpecValidationError('family_spec.observation_conditioned must be a mapping.')
    public_mode = observation_conditioned.get('public_discovery_mode')
    if public_mode not in {None, 'structured', 'visual_only'}:
        raise TaskSpecValidationError("family_spec.observation_conditioned.public_discovery_mode must be 'structured' or 'visual_only'.")
    require_visual = observation_conditioned.get('require_visual_evidence')
    if require_visual is not None and (not isinstance(require_visual, bool)):
        raise TaskSpecValidationError('family_spec.observation_conditioned.require_visual_evidence must be a bool.')
    rooms = observation_conditioned.get('rooms')
    if not isinstance(rooms, list) or not rooms or (not all((isinstance(room, str) and room for room in rooms))):
        raise TaskSpecValidationError('family_spec.observation_conditioned.rooms must be a non-empty string list.')
    catalog_room_ids = {str(room.get('room_id')) for room in (spec.get('source') or {}).get('room_catalog') or spec.get('room_catalog') or [] if isinstance(room, Mapping) and room.get('room_id')}
    missing_rooms = sorted(set(rooms) - catalog_room_ids) if catalog_room_ids else []
    if missing_rooms:
        raise TaskSpecValidationError('family_spec.observation_conditioned references rooms not in source.room_catalog: {}.'.format(', '.join(missing_rooms)))
    payloads = observation_conditioned.get('latent_payloads')
    if not isinstance(payloads, list) or not payloads:
        raise TaskSpecValidationError('family_spec.observation_conditioned.latent_payloads must be a non-empty list.')
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise TaskSpecValidationError('Each observation-conditioned latent payload must be a mapping.')
        room_id = payload.get('room_id')
        if not isinstance(room_id, str) or not room_id:
            raise TaskSpecValidationError('Observation-conditioned latent payload requires room_id.')
        if catalog_room_ids and room_id not in catalog_room_ids:
            raise TaskSpecValidationError("Observation-conditioned latent payload room '{}' is not in source.room_catalog.".format(room_id))
        object_role = payload.get('object_role')
        if object_role is not None and object_role not in role_names:
            raise TaskSpecValidationError("Observation-conditioned latent payload references unknown object_role '{}'.".format(object_role))
        support_roles = payload.get('support_roles')
        if isinstance(support_roles, Mapping):
            for support_role in support_roles.values():
                if support_role is not None and support_role not in role_names:
                    raise TaskSpecValidationError("Observation-conditioned latent payload references unknown support role '{}'.".format(support_role))
        for predicate in payload.get('goal_predicates') or []:
            if isinstance(predicate, Mapping):
                for role_ref in _iter_role_refs(predicate):
                    if role_ref not in role_names:
                        raise TaskSpecValidationError("Observation-conditioned payload predicate references unknown role '{}'.".format(role_ref))

def _validate_source(source: Any) -> None:
    if not isinstance(source, Mapping):
        raise TaskSpecValidationError("Task spec field 'source' must be a mapping.")
    _require_fields(source, REQUIRED_SOURCE_FIELDS, 'source')
    for field in REQUIRED_SOURCE_FIELDS:
        _require_non_empty_string(source, field, 'source')
    if source['type'] not in ALLOWED_SOURCE_TYPES:
        raise TaskSpecValidationError('source.type must be one of {}.'.format(sorted(ALLOWED_SOURCE_TYPES)))

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
        _require_non_empty_string(role, 'synset', "role '{}'".format(role_name))
        _validate_optional_role_geometry(role_name, role)
        role_names.add(role_name)
    return role_names

def _validate_optional_role_geometry(role_name: str, role: Mapping[str, Any]) -> None:
    for key in ('extent', 'center'):
        if key in role:
            value = role[key]
            if not isinstance(value, (list, tuple)) or len(value) != 3 or (not all((isinstance(v, (int, float)) for v in value))):
                raise TaskSpecValidationError("Role '{}' field '{}' must be a 3-number list.".format(role_name, key))
    if 'abilities' in role and (not isinstance(role['abilities'], (list, tuple))):
        raise TaskSpecValidationError("Role '{}' field 'abilities' must be a list.".format(role_name))
    if 'init_states' in role and (not isinstance(role['init_states'], Mapping)):
        raise TaskSpecValidationError("Role '{}' field 'init_states' must be a mapping.".format(role_name))

def _validate_init(init: Any, role_names: Set[str]) -> None:
    if not isinstance(init, list):
        raise TaskSpecValidationError("Task spec field 'init' must be a list.")
    for entry in init:
        if not isinstance(entry, Mapping):
            raise TaskSpecValidationError('Each init entry must be a mapping.')
        predicate = entry.get('predicate')
        if predicate not in INIT_PREDICATES:
            raise TaskSpecValidationError("Unsupported init predicate '{}'. Supported: {}".format(predicate, sorted(INIT_PREDICATES)))
        args = entry.get('args')
        if not isinstance(args, (list, tuple)) or not args:
            raise TaskSpecValidationError("init entry '{}' requires a non-empty 'args' list.".format(predicate))
        if predicate in INIT_STATE_PREDICATES:
            if args[0] not in role_names:
                raise TaskSpecValidationError("init '{}' references unknown role '{}'.".format(predicate, args[0]))
        elif predicate in INIT_BINARY_ROLE_PREDICATES:
            if len(args) < 2:
                raise TaskSpecValidationError("init '{}' requires two role args.".format(predicate))
            for ref in args[:2]:
                if ref not in role_names:
                    raise TaskSpecValidationError("init '{}' references unknown role '{}'.".format(predicate, ref))
        elif predicate == 'inroom':
            if args[0] not in role_names:
                raise TaskSpecValidationError("init 'inroom' references unknown role '{}'.".format(args[0]))

def _validate_goal(goal: Any, role_names: Set[str]) -> None:
    if not isinstance(goal, Mapping):
        raise TaskSpecValidationError('goal must be a mapping.')
    _require_fields(goal, REQUIRED_GOAL_FIELDS, 'goal')
    if not isinstance(goal['require_model_stop'], bool):
        raise TaskSpecValidationError('goal.require_model_stop must be a boolean.')
    preds = goal['predicates']
    if not isinstance(preds, list) or not preds:
        raise TaskSpecValidationError('goal.predicates must be a non-empty list.')
    for predicate in preds:
        if not isinstance(predicate, Mapping):
            raise TaskSpecValidationError('Each goal predicate must be a mapping.')
        ptype = predicate.get('type')
        if ptype not in SUPPORTED_GOAL_PREDICATES:
            raise TaskSpecValidationError("Unsupported goal predicate '{}'. Stage-1 supports: {}".format(ptype, sorted(SUPPORTED_GOAL_PREDICATES)))
        args = predicate.get('args', {})
        if not isinstance(args, Mapping):
            raise TaskSpecValidationError("Predicate '{}' field 'args' must be a mapping.".format(ptype))
        for role_ref in _iter_role_refs(predicate):
            if role_ref not in role_names:
                raise TaskSpecValidationError("Predicate '{}' references unknown role '{}'.".format(ptype, role_ref))
    if goal.get('bddl_goal') is not None and (not isinstance(goal['bddl_goal'], (list, dict))):
        raise TaskSpecValidationError('goal.bddl_goal must be a list or dict when present.')

def _iter_role_refs(predicate: Mapping[str, Any]) -> Iterable[str]:
    for key, value in predicate.items():
        if key == 'role' or key.endswith('_role'):
            if isinstance(value, str):
                yield value
        elif key == 'roles' or key.endswith('_roles'):
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        yield item
    args = predicate.get('args', {})
    if not isinstance(args, Mapping):
        return
    for key, value in args.items():
        if key in NON_ROLE_ARG_KEYS:
            continue
        if isinstance(value, str):
            yield value

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
__all__ = ['SUPPORTED_GOAL_PREDICATES', 'TaskSpecValidationError', 'validate_behavior_task_spec']
from scoper.benchmark.behavior.scheduler.disturbance import KNOWN_DISTURBANCE_TYPES
from scoper.benchmark.behavior.scheduler.failure_injection import KNOWN_FAILURE_MODES

def _validate_disturbance_spec(spec: Mapping[str, Any]) -> None:
    entries = _disturbance_entries(spec.get('disturbance_spec'))
    if not entries:
        return
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TaskSpecValidationError('Each disturbance_spec entry must be a mapping.')
        dtype = entry.get('type')
        if dtype not in KNOWN_DISTURBANCE_TYPES:
            raise TaskSpecValidationError("Unknown disturbance type '{}'. Allowed: {}".format(dtype, sorted(KNOWN_DISTURBANCE_TYPES)))
        if not entry.get('target_role') and (not entry.get('target_object_id')):
            raise TaskSpecValidationError("disturbance_spec entry requires 'target_role' or 'target_object_id'.")

def _validate_failure_spec(spec: Mapping[str, Any]) -> None:
    entries = _failure_entries(spec)
    if not entries:
        return
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TaskSpecValidationError('Each failure_spec entry must be a mapping.')
        mode = entry.get('mode')
        if mode not in KNOWN_FAILURE_MODES:
            raise TaskSpecValidationError("Unknown failure mode '{}'. Allowed: {}".format(mode, sorted(KNOWN_FAILURE_MODES)))

def _disturbance_entries(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, Mapping):
        if isinstance(value.get('disturbances'), list):
            return list(value.get('disturbances'))
        if value.get('type') is not None:
            return [value]
        if isinstance(value.get('failures'), list):
            return []
    raise TaskSpecValidationError("disturbance_spec must be a list or a mapping with 'disturbances'/'failures'.")

def _failure_entries(spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    nested = spec.get('disturbance_spec')
    if isinstance(nested, Mapping) and isinstance(nested.get('failures'), list):
        entries.extend(list(nested.get('failures')))
    value = spec.get('failure_spec')
    if value is None:
        return entries
    if isinstance(value, list):
        entries.extend(list(value))
        return entries
    if isinstance(value, Mapping):
        if isinstance(value.get('failures'), list):
            entries.extend(list(value.get('failures')))
        elif isinstance(value.get('configs'), list):
            entries.extend(list(value.get('configs')))
        elif value.get('mode') is not None:
            entries.append(value)
        else:
            raise TaskSpecValidationError("failure_spec mapping must contain 'failures', 'configs', or 'mode'.")
        return entries
    raise TaskSpecValidationError('failure_spec must be a list or mapping.')
