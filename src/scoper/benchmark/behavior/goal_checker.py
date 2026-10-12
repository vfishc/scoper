from __future__ import annotations
import math
from typing import Any, Callable, Mapping
from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.execution_core.object_placement import aabb_bottom_z, aabb_top_z
from scoper.benchmark.behavior.bddl_adapter import DEFAULT_SUBSTANCE
from scoper.benchmark.behavior.goal_predicates import normalize_goal_predicate
DEFAULT_AGENT_NEAR_THRESHOLD = 1.5
ONTOP_Z_TOLERANCE_M = 0.06

def _xy_inside(a, b) -> bool:
    return abs(a.center[0] - b.center[0]) <= b.extent[0] * 0.5 and abs(a.center[1] - b.center[1]) <= b.extent[1] * 0.5

def _is_ontop(a, b) -> bool:
    if not _xy_inside(a, b):
        return False
    return abs(aabb_bottom_z(a) - aabb_top_z(b)) <= ONTOP_Z_TOLERANCE_M

def _eval_ontop(core, objs) -> bool:
    return bool(_is_ontop(objs[0], objs[1]))

def _is_inside(a, b) -> bool:
    return _xy_inside(a, b) and b.center[2] - b.extent[2] * 0.5 <= a.center[2] <= b.center[2] + b.extent[2] * 0.5

def _eval_inside(core, objs) -> bool:
    return bool(_is_inside(objs[0], objs[1]))

class BehaviorGoalChecker:

    def __init__(self, adapter, core, role_map: Mapping[str, str]):
        self.adapter = adapter
        self.core = core
        self.role_map = dict(role_map)

    def check(self, task_spec: Mapping[str, Any], model_stopped: bool=False) -> dict:
        goal = task_spec['goal']
        predicate_records = [self._evaluate_predicate(predicate) for predicate in goal.get('predicates') or []]
        predicates = [{key: value for key, value in record.items() if key in {'type', 'satisfied', 'args', 'expected', 'actual', 'reason', 'object_role', 'receptacle_role', 'receptacle_roles'}} for record in predicate_records]
        total = len(predicate_records)
        satisfied_count = sum((1 for pp in predicate_records if pp['satisfied']))
        predicate_success = total > 0 and satisfied_count == total
        task_progress = satisfied_count / total if total else 0.0
        require_model_stop = bool(goal.get('require_model_stop', False))
        task_success = predicate_success and (not require_model_stop or bool(model_stopped))
        return {'task_success': 1.0 if task_success else 0.0, 'task_progress': float(task_progress), 'predicate_success': 1.0 if predicate_success else 0.0, 'completion_candidate': 1.0 if predicate_success else 0.0, 'require_model_stop': require_model_stop, 'model_stopped': bool(model_stopped), 'predicates': predicates, 'predicate_records': predicate_records}

    def _evaluate_predicate(self, predicate: Mapping[str, Any]) -> dict:
        record = normalize_goal_predicate(predicate)
        evaluator = _PREDICATE_EVALUATORS.get(record['type'])
        if evaluator is None:
            raise KeyError("No symbolic evaluator for goal predicate '{}'.".format(record['type']))
        update = evaluator(self, record)
        record.update(update)
        record['satisfied'] = bool(record.get('satisfied', False))
        return record

    def _object_for_role(self, role: Any):
        if role is None:
            return None
        name = self.role_map.get(str(role), str(role))
        return self.core.objects.get(name)

    def _role_for_object_name(self, object_name: str | None) -> str | None:
        if object_name is None:
            return None
        for role, name in self.role_map.items():
            if name == object_name:
                return role
        return object_name

def _placement(checker: BehaviorGoalChecker, record: Mapping[str, Any], *, relation: str, negate: bool=False) -> dict:
    obj_role = record.get('object_role')
    receptacle_role = record.get('receptacle_role')
    obj = checker._object_for_role(obj_role)
    receptacle = checker._object_for_role(receptacle_role)
    expected = '{} {} {}'.format(obj_role, 'not in' if negate and relation == 'inside' else 'not on' if negate else 'inside' if relation == 'inside' else 'on', receptacle_role)
    if obj is None:
        return _result(False, expected, "object role '{}' is missing".format(obj_role), 'missing_object')
    if receptacle is None:
        return _result(False, expected, "receptacle role '{}' is missing".format(receptacle_role), 'missing_receptacle')
    held_name = getattr(checker.core.ego_state, 'held_object_name', None)
    is_held = held_name == obj.name
    relation_true = _is_inside(obj, receptacle) if relation == 'inside' else _is_ontop(obj, receptacle)
    if relation == 'inside' and is_held:
        relation_true = False
    satisfied = not relation_true if negate else relation_true
    actual = _placement_actual(checker, obj, receptacle, relation, is_held)
    reason = 'predicate satisfied' if satisfied else _placement_reason(obj, receptacle, relation, is_held)
    return _result(satisfied, expected, actual, reason)

def _placement_any(checker: BehaviorGoalChecker, record: Mapping[str, Any], *, negate: bool=False) -> dict:
    obj_role = record.get('object_role')
    receptacle_roles = list(record.get('receptacle_roles') or [])
    obj = checker._object_for_role(obj_role)
    expected = '{} {} any of [{}]'.format(obj_role, 'not in' if negate else 'inside', ', '.join(receptacle_roles))
    if obj is None:
        return _result(False, expected, "object role '{}' is missing".format(obj_role), 'missing_object')
    held_name = getattr(checker.core.ego_state, 'held_object_name', None)
    if held_name == obj.name:
        relation_true = False
        matched_role = None
    else:
        matched_role = None
        for role in receptacle_roles:
            receptacle = checker._object_for_role(role)
            if receptacle is not None and _is_inside(obj, receptacle):
                matched_role = role
                break
        relation_true = matched_role is not None
    satisfied = not relation_true if negate else relation_true
    if matched_role:
        actual = '{} is inside {}'.format(obj_role, matched_role)
    elif held_name == obj.name:
        actual = '{} is held'.format(obj_role)
    else:
        actual = '{} is not inside any listed receptacle'.format(obj_role)
    reason = 'predicate satisfied' if satisfied else 'required any-receptacle containment was not met'
    out = _result(satisfied, expected, actual, reason)
    out['any_receptacle_resolved_to'] = matched_role
    return out

def _held(checker: BehaviorGoalChecker, record: Mapping[str, Any], *, negate: bool=False) -> dict:
    obj_role = record.get('object_role')
    obj = checker._object_for_role(obj_role)
    held_name = getattr(checker.core.ego_state, 'held_object_name', None)
    held_role = checker._role_for_object_name(held_name)
    expected = '{} {}'.format(obj_role, 'not held' if negate else 'held')
    if obj is None:
        return _result(False, expected, "object role '{}' is missing".format(obj_role), 'missing_object')
    relation_true = held_name == obj.name
    satisfied = not relation_true if negate else relation_true
    actual = 'held object is {}'.format(held_role or 'none')
    reason = 'predicate satisfied' if satisfied else 'held-object state does not match expectation'
    return _result(satisfied, expected, actual, reason)

def _open_state(checker: BehaviorGoalChecker, record: Mapping[str, Any], *, want_open: bool) -> dict:
    return _boolean_state(checker, record, state_name='open', getter=actions.get_open_state, supports=actions.supports_open_state, want_value=want_open, source='symbolic_open_state')

def _toggle_state(checker: BehaviorGoalChecker, record: Mapping[str, Any], *, want_on: bool) -> dict:
    return _boolean_state(checker, record, state_name='toggled_on', getter=actions.get_toggle_state, supports=actions.supports_toggle_state, want_value=want_on, source='symbolic_toggle_state')

def _dirty_state(checker: BehaviorGoalChecker, record: Mapping[str, Any], *, want_dirty: bool) -> dict:
    return _boolean_state(checker, record, state_name='dirty', getter=actions.get_dirty_state, supports=None, want_value=want_dirty, source='symbolic_dirty_state', unknown_is_false=True)

def _sliced_state(checker: BehaviorGoalChecker, record: Mapping[str, Any]) -> dict:
    return _boolean_state(checker, record, state_name='sliced', getter=actions.get_sliced_state, supports=actions.supports_cut_target, want_value=True, source='symbolic_sliced_state', unknown_is_false=True)

def _cooked_state(checker: BehaviorGoalChecker, record: Mapping[str, Any], *, hot: bool=False) -> dict:
    out = _boolean_state(checker, record, state_name='cooked' if not hot else 'hot', getter=actions.get_cooked_state, supports=None, want_value=True, source='symbolic_cooked_state', unknown_is_false=True)
    if hot:
        out['temperature_model'] = 'approximated_as_cooked'
        out['source'] = 'symbolic_cooked_state'
    return out

def _cool_state(checker: BehaviorGoalChecker, record: Mapping[str, Any]) -> dict:
    out = _boolean_state(checker, record, state_name='cooled', getter=actions.get_cooled_state, supports=None, want_value=True, source='symbolic_cooled_state', unknown_is_false=True)
    out['temperature_model'] = 'symbolic_cooled_flag'
    return out

def _substance_state(checker: BehaviorGoalChecker, record: Mapping[str, Any], *, state_name: str, label: str) -> dict:
    obj_role = record.get('object_role')
    obj = checker._object_for_role(obj_role)
    literal_args = record.get('literal_args') or {}
    substance = str(literal_args.get('substance') or literal_args.get('liquid') or DEFAULT_SUBSTANCE)
    expected = '{} {} with {}'.format(obj_role, label, substance)
    if obj is None:
        return _result(False, expected, "object role '{}' is missing".format(obj_role), 'missing_object')
    actual_substances = actions.get_substances(obj, state_name)
    satisfied = substance in set(actual_substances)
    actual = '{} has {}={}'.format(obj_role, state_name, actual_substances)
    reason = 'predicate satisfied' if satisfied else '{} does not contain required substance {}'.format(state_name, substance)
    out = _result(satisfied, expected, actual, reason)
    out['source'] = 'symbolic_{}_state'.format(state_name)
    out['substance'] = substance
    return out

def _agent_near(checker: BehaviorGoalChecker, record: Mapping[str, Any]) -> dict:
    obj_role = record.get('object_role')
    obj = checker._object_for_role(obj_role)
    threshold = float((record.get('literal_args') or {}).get('threshold', DEFAULT_AGENT_NEAR_THRESHOLD))
    expected = 'agent within {:.2f}m of {}'.format(threshold, obj_role)
    if obj is None:
        return _result(False, expected, "object role '{}' is missing".format(obj_role), 'missing_object')
    ex, ey = (checker.core.ego_state.pose.x, checker.core.ego_state.pose.y)
    distance = math.hypot(ex - obj.center[0], ey - obj.center[1])
    satisfied = distance <= threshold
    actual = 'distance={:.3f}m threshold={:.3f}m'.format(distance, threshold)
    reason = 'predicate satisfied' if satisfied else 'agent is outside the required distance threshold'
    out = _result(satisfied, expected, actual, reason)
    out['distance'] = distance
    out['threshold'] = threshold
    return out

def _boolean_state(checker: BehaviorGoalChecker, record: Mapping[str, Any], *, state_name: str, getter: Callable[[Any], Any], supports: Callable[[Any], bool] | None, want_value: bool, source: str, unknown_is_false: bool=False) -> dict:
    obj_role = record.get('object_role')
    obj = checker._object_for_role(obj_role)
    expected = '{} {}={}'.format(obj_role, state_name, bool(want_value))
    if obj is None:
        return _result(False, expected, "object role '{}' is missing".format(obj_role), 'missing_object')
    raw_value = getter(obj)
    actual_value = False if raw_value is None and unknown_is_false else raw_value
    satisfied = raw_value is not None and bool(actual_value) is bool(want_value)
    support_ok = True if supports is None else bool(supports(obj))
    actual = '{} {}={}'.format(obj_role, state_name, actual_value)
    if not support_ok:
        reason = 'object does not advertise support for {} state'.format(state_name)
    elif raw_value is None and (not unknown_is_false):
        reason = '{} state is unavailable'.format(state_name)
    elif satisfied:
        reason = 'predicate satisfied'
    else:
        reason = '{} state does not match expectation'.format(state_name)
    out = _result(satisfied, expected, actual, reason)
    out['source'] = source
    out['affordance_ok'] = support_ok
    out['actual_value'] = actual_value
    return out

def _placement_actual(checker: BehaviorGoalChecker, obj: Any, receptacle: Any, relation: str, is_held: bool) -> str:
    obj_role = checker._role_for_object_name(obj.name) or obj.name
    receptacle_role = checker._role_for_object_name(receptacle.name) or receptacle.name
    if relation == 'inside' and is_held:
        return '{} is held'.format(obj_role)
    relation_true = _is_inside(obj, receptacle) if relation == 'inside' else _is_ontop(obj, receptacle)
    if relation_true:
        return '{} is {} {}'.format(obj_role, 'inside' if relation == 'inside' else 'on', receptacle_role)
    current = _current_spatial_roles(checker, obj, relation)
    if current:
        return '{} is {} {}'.format(obj_role, 'inside' if relation == 'inside' else 'on', ', '.join(current))
    return '{} is not {} {}'.format(obj_role, 'inside' if relation == 'inside' else 'on', receptacle_role)

def _placement_reason(obj: Any, receptacle: Any, relation: str, is_held: bool) -> str:
    if relation == 'inside' and is_held:
        return 'held objects cannot satisfy receptacle containment'
    if not _xy_inside(obj, receptacle):
        return 'object xy center is outside the receptacle footprint'
    if relation == 'on_top':
        return 'object bottom is not near the receptacle top surface'
    return 'object center is not inside receptacle vertical bounds'

def _current_spatial_roles(checker: BehaviorGoalChecker, obj: Any, relation: str) -> list[str]:
    roles = []
    for role, name in checker.role_map.items():
        if name == obj.name:
            continue
        candidate = checker.core.objects.get(name)
        if candidate is None:
            continue
        if relation == 'inside' and _is_inside(obj, candidate):
            roles.append(role)
        elif relation == 'on_top' and _is_ontop(obj, candidate):
            roles.append(role)
    return roles

def _result(satisfied: bool, expected: str, actual: str, reason: str) -> dict:
    return {'satisfied': bool(satisfied), 'expected': str(expected), 'actual': str(actual), 'reason': str(reason)}
_PREDICATE_EVALUATORS: dict[str, Callable[[BehaviorGoalChecker, Mapping[str, Any]], dict]] = {'object_on_receptacle': lambda checker, record: _placement(checker, record, relation='on_top'), 'object_in_receptacle': lambda checker, record: _placement(checker, record, relation='inside'), 'object_not_in_receptacle': lambda checker, record: _placement(checker, record, relation='inside', negate=True), 'object_in_any_receptacle': lambda checker, record: _placement_any(checker, record), 'object_not_in_any_receptacle': lambda checker, record: _placement_any(checker, record, negate=True), 'object_held': lambda checker, record: _held(checker, record), 'object_not_held': lambda checker, record: _held(checker, record, negate=True), 'object_open': lambda checker, record: _open_state(checker, record, want_open=True), 'object_closed': lambda checker, record: _open_state(checker, record, want_open=False), 'object_toggled_on': lambda checker, record: _toggle_state(checker, record, want_on=True), 'object_toggled_off': lambda checker, record: _toggle_state(checker, record, want_on=False), 'object_dirty': lambda checker, record: _dirty_state(checker, record, want_dirty=True), 'object_clean': lambda checker, record: _dirty_state(checker, record, want_dirty=False), 'object_sliced': lambda checker, record: _sliced_state(checker, record), 'object_cooked': lambda checker, record: _cooked_state(checker, record), 'object_hot': lambda checker, record: _cooked_state(checker, record, hot=True), 'object_cool': lambda checker, record: _cool_state(checker, record), 'object_filled': lambda checker, record: _substance_state(checker, record, state_name=actions.FILLED_STATE_NAME, label='filled'), 'object_soaked': lambda checker, record: _substance_state(checker, record, state_name=actions.SATURATED_STATE_NAME, label='soaked'), 'agent_near': _agent_near}
__all__ = ['BehaviorGoalChecker', 'DEFAULT_AGENT_NEAR_THRESHOLD', '_eval_inside', '_eval_ontop']
