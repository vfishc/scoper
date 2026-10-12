from __future__ import annotations
from typing import Any, List, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class OracleBrain(BaseBrain):

    def __init__(self, max_replans: int | None=None) -> None:
        self._queue: List[ToolCall] = []
        self._env: Any = None
        self._task_spec: Mapping[str, Any] = {}
        self._replan_count = 0
        self._max_replans = max_replans

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        self._task_spec = task_spec
        self._replan_count = 0
        self._queue = list(_build_tool_sequence(env, task_spec))
        if self._max_replans is None:
            self._max_replans = len(task_spec['goal']['predicates']) * 3

    def act(self, brain_input: BrainInput) -> ToolCall:
        last = brain_input.last_transition
        if last is not None and (not last.get('success', True)):
            if self._replan_count < (self._max_replans or 99):
                self._replan_count += 1
                self._queue = list(_build_tool_sequence(self._env, self._task_spec))
            else:
                return ToolCall('stop', {}, reasoning='replan_budget_exceeded')
        if self._queue:
            return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='oracle plan complete')

def _role_needs_open(env: Any, role_name: str) -> bool:
    obj_name = env.role_map.get(role_name, role_name)
    obj = env.core.objects.get(obj_name)
    if obj is None:
        return False
    abilities = tuple(getattr(obj, 'abilities', ()))
    states = getattr(obj, 'states', {}) or {}
    is_openable = 'openable' in abilities or 'open' in states
    return bool(is_openable and states.get('open') is not True)

def _role_is_held(env: Any, role_name: str) -> bool:
    obj_name = env.role_map.get(role_name, role_name)
    held = getattr(getattr(env.core, 'ego_state', None), 'held_object_name', None)
    return held is not None and held == obj_name

def _ensure_open(env: Any, receptacle_role: str) -> List[ToolCall]:
    if _role_needs_open(env, receptacle_role):
        return [ToolCall('navigate_to', {'target': receptacle_role}), ToolCall('open', {'target': receptacle_role})]
    return []

def _plan_placement(env: Any, args: Mapping[str, Any]) -> List[ToolCall]:
    object_role = args['object']
    receptacle_role = args['receptacle']
    calls: List[ToolCall] = []
    if not _role_is_held(env, object_role):
        calls.append(ToolCall('navigate_to', {'target': object_role}))
        calls.append(ToolCall('pick_up', {'target': object_role}))
    calls.extend(_ensure_open(env, receptacle_role))
    calls.append(ToolCall('navigate_to', {'target': receptacle_role}))
    calls.append(ToolCall('put_down', {'target': receptacle_role}))
    return calls

def _plan_bool(tool_name: str, args: Mapping[str, Any]) -> List[ToolCall]:
    role = args['object']
    return [ToolCall('navigate_to', {'target': role}), ToolCall(tool_name, {'target': role})]

def _plan_cook(env: Any, args: Mapping[str, Any]) -> List[ToolCall]:
    food = args['object']
    heatsource = args['heatsource']
    calls: List[ToolCall] = [ToolCall('navigate_to', {'target': food}), ToolCall('pick_up', {'target': food})]
    calls.extend(_ensure_open(env, heatsource))
    calls.extend([ToolCall('navigate_to', {'target': heatsource}), ToolCall('heat', {'target': heatsource})])
    return calls

def _plan_cool(env: Any, args: Mapping[str, Any]) -> List[ToolCall]:
    food = args['object']
    coldsource = args['coldsource']
    calls: List[ToolCall] = [ToolCall('navigate_to', {'target': food}), ToolCall('pick_up', {'target': food})]
    calls.extend(_ensure_open(env, coldsource))
    calls.extend([ToolCall('navigate_to', {'target': coldsource}), ToolCall('cool', {'target': coldsource})])
    return calls

def _plan_sliced(env: Any, args: Mapping[str, Any]) -> List[ToolCall]:
    obj = args['object']
    tool = args.get('tool', 'knife')
    return [ToolCall('navigate_to', {'target': tool}), ToolCall('pick_up', {'target': tool}), ToolCall('navigate_to', {'target': obj}), ToolCall('cut', {'target': obj})]

def _plan_clean(env: Any, args: Mapping[str, Any]) -> List[ToolCall]:
    obj = args['object']
    tool = args.get('tool', 'rag')
    return [ToolCall('navigate_to', {'target': tool}), ToolCall('pick_up', {'target': tool}), ToolCall('navigate_to', {'target': obj}), ToolCall('wipe', {'target': obj})]

def _plan_soaked(env: Any, args: Mapping[str, Any]) -> List[ToolCall]:
    obj = args['object']
    source = args.get('source', 'sink')
    return [ToolCall('navigate_to', {'target': obj}), ToolCall('pick_up', {'target': obj}), ToolCall('navigate_to', {'target': source}), ToolCall('soak_under', {'target': source})]

def _plan_filled(env: Any, args: Mapping[str, Any]) -> List[ToolCall]:
    obj = args['object']
    source = args.get('source', 'sink')
    return [ToolCall('navigate_to', {'target': obj}), ToolCall('pick_up', {'target': obj}), ToolCall('navigate_to', {'target': source}), ToolCall('fill', {'target': obj, 'source': source})]

def _plan_held(env: Any, args: Mapping[str, Any]) -> List[ToolCall]:
    obj = args['object']
    if _role_is_held(env, obj):
        return []
    return [ToolCall('navigate_to', {'target': obj}), ToolCall('pick_up', {'target': obj})]

def _plan_not_held(env: Any, args: Mapping[str, Any]) -> List[ToolCall]:
    obj = args['object']
    if _role_is_held(env, obj):
        return [ToolCall('drop', {})]
    return []
PREDICATE_PLAN_TEMPLATES = {'object_on_receptacle': _plan_placement, 'object_in_receptacle': _plan_placement, 'object_open': lambda env, args: _plan_bool('open', args), 'object_closed': lambda env, args: _plan_bool('close', args), 'object_toggled_on': lambda env, args: _plan_bool('toggle_on', args), 'object_toggled_off': lambda env, args: _plan_bool('toggle_off', args), 'object_cooked': _plan_cook, 'object_hot': _plan_cook, 'object_cool': _plan_cool, 'object_sliced': _plan_sliced, 'object_clean': _plan_clean, 'object_soaked': _plan_soaked, 'object_filled': _plan_filled, 'object_held': _plan_held, 'object_not_held': _plan_not_held, 'agent_near': lambda env, args: [ToolCall('navigate_to', {'target': args['object']})]}

def unsupported_goal_predicates(task_spec: Mapping[str, Any]) -> List[str]:
    unsupported: List[str] = []
    for predicate in (task_spec.get('goal') or {}).get('predicates') or []:
        ptype = str(predicate.get('type'))
        if ptype not in PREDICATE_PLAN_TEMPLATES:
            unsupported.append(ptype)
    return sorted(set(unsupported))

def can_solve_goal_predicates(task_spec: Mapping[str, Any]) -> bool:
    predicates = (task_spec.get('goal') or {}).get('predicates') or []
    return bool(predicates) and (not unsupported_goal_predicates(task_spec))

def _build_tool_sequence(env: Any, task_spec: Mapping[str, Any]) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for predicate in task_spec['goal']['predicates']:
        ptype = predicate.get('type')
        args = predicate.get('args', {})
        planner = PREDICATE_PLAN_TEMPLATES.get(ptype)
        if planner is None:
            raise BrainError("OracleBrain has no plan template for predicate '{}'.".format(ptype))
        calls.extend(planner(env, args))
    if bool(task_spec['goal'].get('require_model_stop', False)):
        calls.append(ToolCall('stop', {}, reasoning='task requires model stop'))
    return calls
__all__ = ['OracleBrain', 'PREDICATE_PLAN_TEMPLATES', 'can_solve_goal_predicates', 'unsupported_goal_predicates']
