from __future__ import annotations
import math
from typing import Any, Callable, Dict, Mapping, Optional
from scoper.benchmark.procthor.online_task_spec import validate_online_task_spec
DEFAULT_AGENT_NEAR_THRESHOLD = 1.5

class PredicateGoalCheckerError(ValueError):
    pass

def check_goal(task_spec: Mapping[str, Any], metadata: Mapping[str, Any], model_stopped: bool=False) -> dict:
    validate_online_task_spec(task_spec)
    predicates = []
    for predicate in task_spec['goal']['predicates']:
        predicates.append({'type': predicate['type'], 'satisfied': bool(_evaluate_predicate(predicate, task_spec, metadata))})
    total = len(predicates)
    satisfied_count = sum((1 for predicate in predicates if predicate['satisfied']))
    predicate_success = total > 0 and satisfied_count == total
    task_progress = satisfied_count / total if total else 0.0
    require_model_stop = bool(task_spec['goal'].get('require_model_stop', False))
    task_success = predicate_success and (not require_model_stop or bool(model_stopped))
    return {'task_success': 1.0 if task_success else 0.0, 'task_progress': float(task_progress), 'predicate_success': 1.0 if predicate_success else 0.0, 'completion_candidate': 1.0 if predicate_success else 0.0, 'require_model_stop': require_model_stop, 'model_stopped': bool(model_stopped), 'predicates': predicates}

def validate_goal_predicates(task_spec: Mapping[str, Any]) -> None:
    validate_online_task_spec(task_spec)
    role_names = set(task_spec['roles'].keys())
    for predicate in task_spec['goal']['predicates']:
        predicate_type = predicate.get('type')
        if predicate_type not in _PREDICATE_EVALUATORS:
            raise PredicateGoalCheckerError("Unsupported goal predicate type '{}'.".format(predicate_type))
        for key, value in predicate.items():
            if key == 'role' or key.endswith('_role'):
                if value not in role_names:
                    raise PredicateGoalCheckerError("Predicate '{}' references unknown role '{}'.".format(predicate_type, value))
            elif key == 'roles' or key.endswith('_roles'):
                _validate_role_list(predicate_type, key, value, role_names)

def _evaluate_predicate(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    predicate_type = predicate.get('type')
    evaluator = _PREDICATE_EVALUATORS.get(predicate_type)
    if evaluator is None:
        raise PredicateGoalCheckerError("Unsupported goal predicate type '{}'.".format(predicate_type))
    return evaluator(predicate, task_spec, metadata)

def _object_in_receptacle(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    receptacle_id = _role_object_id(predicate, task_spec, 'receptacle_role')
    return _object_or_slice_child_in_any_receptacle(obj, {receptacle_id}, metadata)

def _object_in_any_receptacle(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    receptacle_ids = _role_object_ids(predicate, task_spec, 'receptacle_roles')
    return _object_or_slice_child_in_any_receptacle(obj, receptacle_ids, metadata)

def _object_or_slice_child_in_any_receptacle(obj: Mapping[str, Any], receptacle_ids: set[str], metadata: Mapping[str, Any]) -> bool:
    held_ids = _inventory_object_ids(metadata)
    object_id = obj.get('objectId')
    if object_id in held_ids:
        return False
    if receptacle_ids.intersection(set(obj.get('parentReceptacles') or [])):
        return True
    if obj.get('isSliced'):
        slice_prefix = f'{object_id}|'
        for candidate in metadata.get('objects', []):
            if not isinstance(candidate, Mapping):
                continue
            cand_id = candidate.get('objectId', '')
            if not cand_id.startswith(slice_prefix):
                continue
            if cand_id in held_ids:
                continue
            if receptacle_ids.intersection(set(candidate.get('parentReceptacles') or [])):
                return True
    return False

def _object_not_in_receptacle(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    return not _object_in_receptacle(predicate, task_spec, metadata)

def _object_not_in_any_receptacle(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    return not _object_in_any_receptacle(predicate, task_spec, metadata)

def _object_held(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    object_id = _role_object_id(predicate, task_spec, 'object_role')
    return object_id in _inventory_object_ids(metadata)

def _object_not_held(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    return not _object_held(predicate, task_spec, metadata)

def _object_open(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    _require_affordance(obj, 'openable')
    openness = obj.get('openness')
    if isinstance(openness, (int, float)) and (not isinstance(openness, bool)):
        return math.isfinite(float(openness)) and float(openness) >= 0.9
    return _required_metadata_bool(obj, 'isOpen')

def _object_closed(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    _require_affordance(obj, 'openable')
    openness = obj.get('openness')
    if isinstance(openness, (int, float)) and (not isinstance(openness, bool)):
        return math.isfinite(float(openness)) and float(openness) <= 0.1
    return not _required_metadata_bool(obj, 'isOpen')

def _object_toggled_on(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    _require_affordance(obj, 'toggleable')
    return _required_metadata_bool(obj, 'isToggled')

def _object_toggled_off(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    _require_affordance(obj, 'toggleable')
    return not _required_metadata_bool(obj, 'isToggled')

def _object_clean(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    return not _required_metadata_bool(obj, 'isDirty')

def _object_dirty(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    return _required_metadata_bool(obj, 'isDirty')

def _object_sliced(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    _require_affordance(obj, 'sliceable')
    return _required_metadata_bool(obj, 'isSliced')

def _object_hot(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    return _object_or_slice_child_has_temperature_state(obj, metadata, sticky_field='isHeated', temperature_value='Hot')

def _object_cool(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    return _object_or_slice_child_has_temperature_state(obj, metadata, sticky_field='isCooled', temperature_value='Cold')

def _object_or_slice_child_has_temperature_state(obj: Mapping[str, Any], metadata: Mapping[str, Any], *, sticky_field: str, temperature_value: str) -> bool:
    if bool(obj.get(sticky_field, False)) or bool(obj.get('temperature') == temperature_value):
        return True
    object_id = obj.get('objectId')
    if not object_id or not obj.get('isSliced'):
        return False
    slice_prefix = f'{object_id}|'
    for candidate in metadata.get('objects', []):
        if not isinstance(candidate, Mapping):
            continue
        cand_id = candidate.get('objectId', '')
        if not cand_id.startswith(slice_prefix):
            continue
        if bool(candidate.get(sticky_field, False)) or bool(candidate.get('temperature') == temperature_value):
            return True
    return False

def _object_filled(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    if not bool(obj.get('isFilledWithLiquid', False)):
        return False
    wanted_liquid = predicate.get('liquid')
    if wanted_liquid is None:
        return True
    return str(obj.get('fillLiquid', '') or '').lower() == str(wanted_liquid).lower()

def _agent_near(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    agent = metadata.get('agent') or {}
    agent_position = agent.get('position') or {}
    object_position = obj.get('position') or {}
    for key in ('x', 'z'):
        if key not in agent_position or key not in object_position:
            raise PredicateGoalCheckerError('agent_near requires agent and object x/z positions.')
    threshold = float(predicate.get('threshold', DEFAULT_AGENT_NEAR_THRESHOLD))
    distance = math.hypot(float(agent_position['x']) - float(object_position['x']), float(agent_position['z']) - float(object_position['z']))
    return distance <= threshold

def _object_on_receptacle(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    receptacle_id = _role_object_id(predicate, task_spec, 'receptacle_role')
    held_ids = _inventory_object_ids(metadata)
    object_id = obj.get('objectId')
    if object_id in held_ids:
        return False
    parent_receptacles = obj.get('parentReceptacles') or []
    if receptacle_id not in parent_receptacles:
        return False
    objects_by_id = _objects_by_id(metadata)
    receptacle = objects_by_id.get(receptacle_id)
    if receptacle is None:
        return False
    if bool(receptacle.get('openable', False)):
        return False
    return True

def _object_soaked(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    if bool(obj.get('isSoaked', False)):
        return True
    if bool(obj.get('isFilledWithLiquid', False)):
        wanted_liquid = predicate.get('liquid')
        if wanted_liquid is None:
            return True
        return str(obj.get('fillLiquid', '') or '').lower() == str(wanted_liquid).lower()
    return False

def _object_cooked(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    obj = _role_object(predicate, task_spec, metadata, 'object_role')
    if bool(obj.get('isCooked', False)):
        return True
    return _object_or_slice_child_has_temperature_state(obj, metadata, sticky_field='isHeated', temperature_value='Hot')

def _role_object(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], metadata: Mapping[str, Any], role_field: str) -> Mapping[str, Any]:
    object_id = _role_object_id(predicate, task_spec, role_field)
    objects = _objects_by_id(metadata)
    obj = objects.get(object_id)
    if obj is None:
        raise PredicateGoalCheckerError("Role object '{}' not found in current metadata.".format(object_id))
    return obj

def _role_object_id(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], role_field: str) -> str:
    role_name = predicate.get(role_field)
    if not isinstance(role_name, str) or not role_name:
        raise PredicateGoalCheckerError("Predicate '{}' missing role field '{}'.".format(predicate.get('type'), role_field))
    role = task_spec['roles'].get(role_name)
    if role is None:
        raise PredicateGoalCheckerError("Predicate '{}' references unknown role '{}'.".format(predicate.get('type'), role_name))
    return role['object_id']

def _role_object_ids(predicate: Mapping[str, Any], task_spec: Mapping[str, Any], role_field: str) -> set[str]:
    role_names = predicate.get(role_field)
    if not isinstance(role_names, list) or not role_names:
        raise PredicateGoalCheckerError("Predicate '{}' missing role list field '{}'.".format(predicate.get('type'), role_field))
    object_ids = set()
    for role_name in role_names:
        if not isinstance(role_name, str) or not role_name:
            raise PredicateGoalCheckerError("Predicate '{}' role list '{}' must contain non-empty strings.".format(predicate.get('type'), role_field))
        role = task_spec['roles'].get(role_name)
        if role is None:
            raise PredicateGoalCheckerError("Predicate '{}' references unknown role '{}'.".format(predicate.get('type'), role_name))
        object_ids.add(role['object_id'])
    return object_ids

def _validate_role_list(predicate_type: Any, key: str, value: Any, role_names: set[str]) -> None:
    if not isinstance(value, list):
        raise PredicateGoalCheckerError("Predicate '{}' field '{}' must be a list of roles.".format(predicate_type, key))
    for role_name in value:
        if role_name not in role_names:
            raise PredicateGoalCheckerError("Predicate '{}' references unknown role '{}'.".format(predicate_type, role_name))

def _objects_by_id(metadata: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {obj['objectId']: obj for obj in metadata.get('objects', []) if isinstance(obj, Mapping) and isinstance(obj.get('objectId'), str)}

def _inventory_object_ids(metadata: Mapping[str, Any]) -> set[str]:
    object_ids = {obj['objectId'] for obj in metadata.get('inventoryObjects', []) if isinstance(obj, Mapping) and isinstance(obj.get('objectId'), str)}
    arm = metadata.get('arm')
    if isinstance(arm, Mapping):
        for item in arm.get('heldObjects', []) or []:
            if isinstance(item, Mapping):
                item = item.get('objectId')
            if isinstance(item, str) and item:
                object_ids.add(item)
    return object_ids

def _require_affordance(obj: Mapping[str, Any], affordance: str) -> None:
    if not bool(obj.get(affordance, False)):
        raise PredicateGoalCheckerError("Object '{}' does not support affordance '{}'.".format(obj.get('objectId'), affordance))

def _required_metadata_bool(obj: Mapping[str, Any], field: str) -> bool:
    value = obj.get(field)
    if not isinstance(value, bool):
        raise PredicateGoalCheckerError("Object '{}' is missing bool metadata field '{}'.".format(obj.get('objectId'), field))
    return value
_PREDICATE_EVALUATORS: Dict[str, Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], bool]] = {'object_in_receptacle': _object_in_receptacle, 'object_not_in_receptacle': _object_not_in_receptacle, 'object_in_any_receptacle': _object_in_any_receptacle, 'object_not_in_any_receptacle': _object_not_in_any_receptacle, 'object_held': _object_held, 'object_not_held': _object_not_held, 'object_open': _object_open, 'object_closed': _object_closed, 'object_toggled_on': _object_toggled_on, 'object_toggled_off': _object_toggled_off, 'object_clean': _object_clean, 'object_dirty': _object_dirty, 'object_sliced': _object_sliced, 'object_hot': _object_hot, 'object_cool': _object_cool, 'object_filled': _object_filled, 'agent_near': _agent_near, 'object_on_receptacle': _object_on_receptacle, 'object_soaked': _object_soaked, 'object_cooked': _object_cooked}
__all__ = ['DEFAULT_AGENT_NEAR_THRESHOLD', 'PredicateGoalCheckerError', 'check_goal', 'validate_goal_predicates']
