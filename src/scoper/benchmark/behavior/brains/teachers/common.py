from __future__ import annotations
from typing import Any, Iterable, Mapping
from scoper.benchmark.behavior.brains.base import ToolCall
from scoper.benchmark.behavior.goal_predicates import normalize_goal_predicate

def call(tool_name: str, target: str | None=None, *, reasoning: str='') -> ToolCall:
    args = {'target': target} if target is not None else {}
    return ToolCall(tool_name, args, reasoning=reasoning)

def stop(reasoning: str) -> ToolCall:
    return ToolCall('stop', {}, reasoning=reasoning)

def object_name(env: Any, role_name: str | None) -> str | None:
    if not role_name:
        return None
    return env.role_map.get(str(role_name), str(role_name))

def object_for_role(env: Any, role_name: str | None) -> Any | None:
    name = object_name(env, role_name)
    if not name:
        return None
    return env.core.objects.get(name)

def role_has_ability(env: Any, role_name: str | None, ability: str) -> bool:
    obj = object_for_role(env, role_name)
    if obj is None:
        return False
    return ability in tuple(getattr(obj, 'abilities', ()) or ())

def role_is_openable(env: Any, role_name: str | None) -> bool:
    obj = object_for_role(env, role_name)
    if obj is None:
        return False
    states = getattr(obj, 'states', {}) or {}
    abilities = tuple(getattr(obj, 'abilities', ()) or ())
    return getattr(obj, 'renderer_openable', True) is not False and ('openable' in abilities or 'open' in states)

def role_is_open(env: Any, role_name: str | None) -> bool | None:
    obj = object_for_role(env, role_name)
    if obj is None:
        return None
    states = getattr(obj, 'states', {}) or {}
    value = states.get('open')
    return bool(value) if value is not None else None

def currently_holding_role(env: Any, role_name: str | None) -> bool:
    name = object_name(env, role_name)
    return bool(name and getattr(env.core.ego_state, 'held_object_name', None) == name)

def ensure_open(env: Any, role_name: str | None) -> list[ToolCall]:
    if not role_is_openable(env, role_name):
        return []
    if role_is_open(env, role_name) is True:
        return []
    return [call('navigate_to', str(role_name)), call('open', str(role_name))]

def close_if_openable(env: Any, role_name: str | None) -> list[ToolCall]:
    if not role_is_openable(env, role_name):
        return []
    return [call('navigate_to', str(role_name)), call('close', str(role_name))]

def simple_place_plan(env: Any, object_role: str, receptacle_role: str, *, close_after: bool=True, start_with_object_held: bool | None=None) -> list[ToolCall]:
    calls: list[ToolCall] = []
    held = currently_holding_role(env, object_role) if start_with_object_held is None else bool(start_with_object_held)
    if not held:
        calls.extend([call('navigate_to', object_role), call('pick_up', object_role)])
    calls.extend(ensure_open(env, receptacle_role))
    calls.extend([call('navigate_to', receptacle_role), call('put_down', receptacle_role)])
    if close_after:
        calls.extend(close_if_openable(env, receptacle_role))
    return calls

def wash_plan(env: Any, object_role: str, *, faucet_role: str | None=None, keep_held: bool=True, start_with_object_held: bool | None=None) -> list[ToolCall]:
    faucet = faucet_role or find_role_with_ability(env, 'water_source') or find_role_with_ability(env, 'toggleable') or 'faucet'
    calls: list[ToolCall] = []
    held = currently_holding_role(env, object_role) if start_with_object_held is None else bool(start_with_object_held)
    if not held:
        calls.extend([call('navigate_to', object_role), call('pick_up', object_role)])
    calls.extend([call('navigate_to', str(faucet)), call('toggle_on', str(faucet)), call('soak_under', str(faucet)), call('toggle_off', str(faucet))])
    if not keep_held:
        calls.append(call('drop'))
    return calls

def staged_sink_wash_plan(env: Any, object_role: str, sink_role: str, *, faucet_role: str | None=None, object_already_staged: bool=False, pick_up_after: bool=True) -> list[ToolCall]:
    faucet = faucet_role or find_role_with_ability(env, 'water_source') or find_role_with_ability(env, 'toggleable') or 'faucet'
    calls: list[ToolCall] = []
    if not object_already_staged:
        if not currently_holding_role(env, object_role):
            calls.extend([call('navigate_to', object_role), call('pick_up', object_role)])
        calls.extend([call('navigate_to', str(sink_role)), call('put_down', str(sink_role))])
    calls.extend([call('navigate_to', str(faucet)), call('toggle_on', str(faucet)), call('soak_under', str(faucet)), call('toggle_off', str(faucet))])
    if pick_up_after:
        calls.extend([call('navigate_to', object_role), call('pick_up', object_role)])
    return calls

def wash_then_place_plan(env: Any, object_role: str, receptacle_role: str, *, faucet_role: str | None=None, close_after: bool=True) -> list[ToolCall]:
    calls = wash_plan(env, object_role, faucet_role=faucet_role, keep_held=True)
    calls.extend(simple_place_plan(env, object_role, receptacle_role, close_after=close_after, start_with_object_held=True))
    return calls

def find_role_with_ability(env: Any, ability: str, *, excluded: Iterable[str]=()) -> str | None:
    excluded_set = {str(item) for item in excluded}
    for role_name in sorted(env.role_map):
        if role_name in excluded_set:
            continue
        if role_has_ability(env, role_name, ability):
            return role_name
    return None

def plan_predicate(env: Any, predicate: Mapping[str, Any]) -> list[ToolCall]:
    record = normalize_goal_predicate(predicate)
    ptype = record['type']
    args = record.get('args') or {}
    obj = args.get('object') or record.get('object_role')
    rec = args.get('receptacle') or record.get('receptacle_role')
    if ptype in {'object_in_receptacle', 'object_on_receptacle'} and obj and rec:
        return simple_place_plan(env, str(obj), str(rec), close_after=False)
    if ptype == 'object_open' and obj:
        return [call('navigate_to', str(obj)), call('open', str(obj))]
    if ptype == 'object_closed' and obj:
        return [call('navigate_to', str(obj)), call('close', str(obj))]
    if ptype == 'object_toggled_on' and obj:
        return [call('navigate_to', str(obj)), call('toggle_on', str(obj))]
    if ptype == 'object_toggled_off' and obj:
        return [call('navigate_to', str(obj)), call('toggle_off', str(obj))]
    if ptype == 'object_clean' and obj:
        return wash_plan(env, str(obj), faucet_role=args.get('source'), keep_held=False)
    if ptype == 'object_sliced' and obj:
        tool = str(args.get('tool') or find_role_with_ability(env, 'cutting_tool') or 'knife')
        return [call('navigate_to', tool), call('pick_up', tool), call('navigate_to', str(obj)), call('cut', str(obj)), call('drop')]
    return []

def goal_cleanup_plan(env: Any, task_spec: Mapping[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for predicate in (task_spec.get('goal') or {}).get('predicates') or []:
        ptype = predicate.get('type')
        if ptype in {'object_open', 'object_closed', 'object_toggled_on', 'object_toggled_off'}:
            calls.extend(plan_predicate(env, predicate))
    return calls

def predicates_to_plan(env: Any, predicates: Iterable[Mapping[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for predicate in predicates:
        calls.extend(plan_predicate(env, predicate))
    return calls
__all__ = ['call', 'stop', 'object_name', 'object_for_role', 'role_has_ability', 'role_is_openable', 'role_is_open', 'currently_holding_role', 'ensure_open', 'close_if_openable', 'simple_place_plan', 'wash_plan', 'staged_sink_wash_plan', 'wash_then_place_plan', 'find_role_with_ability', 'plan_predicate', 'goal_cleanup_plan', 'predicates_to_plan']
