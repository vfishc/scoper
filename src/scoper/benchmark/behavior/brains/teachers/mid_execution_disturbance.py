from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.behavior.brains.teachers.common import call, close_if_openable, simple_place_plan, stop, wash_plan

class MidExecutionDisturbanceTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: list[ToolCall] = []
        self._primary: Mapping[str, Any] = {}
        self._env: Any = None

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        mid = (task_spec.get('family_spec') or {}).get('mid_execution')
        if not isinstance(mid, Mapping):
            raise BrainError('MidExecutionDisturbanceTeacher requires family_spec.mid_execution.')
        primary = mid.get('primary_task')
        if not isinstance(primary, Mapping):
            raise BrainError('MidExecutionDisturbanceTeacher requires mid_execution.primary_task.')
        self._env = env
        self._primary = primary
        self._queue = _primary_plan(env, primary) + [stop('mid-execution disturbance plan complete')]

    def act(self, brain_input: BrainInput) -> ToolCall:
        last = brain_input.last_transition or {}
        feedback = last.get('env_feedback') or {}
        if last.get('success') is False and feedback.get('error_type') == 'container_closed':
            object_role = str(self._primary.get('object_role') or 'target_object')
            receptacle_role = str(self._primary.get('receptacle_role') or 'target_receptacle')
            self._queue = []
            if not _holding_any(self._env):
                self._queue.append(call('pick_up', object_role))
            self._queue.extend([call('navigate_to', receptacle_role), call('open', receptacle_role), call('navigate_to', receptacle_role), call('put_down', receptacle_role), call('navigate_to', receptacle_role), call('close', receptacle_role), stop('recovered from container re-close')])
        if self._queue:
            return self._queue.pop(0)
        return stop('mid-execution disturbance plan exhausted')

def _primary_plan(env: Any, primary: Mapping[str, Any]) -> list[ToolCall]:
    ptype = str(primary.get('type') or '')
    object_role = str(primary.get('object_role') or 'target_object')
    receptacle_role = str(primary.get('receptacle_role') or 'target_receptacle')
    staging_role = primary.get('staging_receptacle_role')
    if ptype == 'state_re_dirty':
        if staging_role:
            sink_role = str(primary.get('sink_role') or staging_role)
            faucet_role = primary.get('faucet_role')
            calls = wash_plan(env, object_role, faucet_role=faucet_role, keep_held=True)
            calls.extend([call('navigate_to', sink_role), call('put_down', sink_role)])
            calls.extend(_secondary_task_plan(env, primary))
            calls.append(call('navigate_to', sink_role, reasoning='return to staged object after secondary task'))
            calls.extend([call('navigate_to', object_role, reasoning='return to the staged object'), call('pick_up', object_role)])
            calls.extend(wash_plan(env, object_role, faucet_role=faucet_role, keep_held=True, start_with_object_held=True))
            calls.extend(simple_place_plan(env, object_role, receptacle_role, close_after=True, start_with_object_held=True))
            return calls
        calls = wash_plan(env, object_role, faucet_role=primary.get('faucet_role'), keep_held=True)
        calls.extend(_interrupt_plan(primary))
        calls.append(ToolCall('look_for', {'target': object_role}, reasoning='verify object is still clean'))
        calls.extend(wash_plan(env, object_role, faucet_role=primary.get('faucet_role'), keep_held=True, start_with_object_held=True))
        calls.extend(simple_place_plan(env, object_role, receptacle_role, close_after=True, start_with_object_held=True))
        return calls
    if ptype == 'object_relocated':
        if staging_role:
            calls = simple_place_plan(env, object_role, str(staging_role), close_after=False)
            calls.extend(_interrupt_plan(primary))
            calls.append(call('navigate_to', object_role, reasoning="return to the staged object's last-seen location"))
            calls.append(call('navigate_to', object_role, reasoning='search for the moved object after staged-location miss'))
            calls.append(call('pick_up', object_role))
            calls.extend(simple_place_plan(env, object_role, receptacle_role, close_after=True, start_with_object_held=True))
            return calls
        return simple_place_plan(env, object_role, receptacle_role, close_after=True)
    if ptype == 'container_re_closed':
        target_roles = [str(role) for role in primary.get('target_objects') or primary.get('object_roles') or [] if isinstance(role, str) and role]
        if not target_roles:
            target_roles = [object_role]
        calls: list[ToolCall] = []
        for index, target_role in enumerate(target_roles):
            if index == 0:
                calls.extend(simple_place_plan(env, target_role, receptacle_role, close_after=False))
                continue
            calls.extend([call('navigate_to', target_role), call('pick_up', target_role), call('navigate_to', receptacle_role), call('put_down', receptacle_role)])
        calls.extend(close_if_openable(env, receptacle_role))
        return calls
    raise BrainError("Unknown mid-execution primary task type '{}'.".format(ptype))

def _interrupt_plan(primary: Mapping[str, Any]) -> list[ToolCall]:
    role = primary.get('interrupt_role')
    if not isinstance(role, str) or not role:
        return []
    from scoper.benchmark.behavior.behavior_closed_loop_tool_env import MODEL_VISIBLE_TOOLS
    allowed_tools = set(MODEL_VISIBLE_TOOLS)
    calls: list[ToolCall] = []
    for tool_name in primary.get('interrupt_tools') or ['navigate_to']:
        if isinstance(tool_name, str) and tool_name:
            if tool_name not in allowed_tools:
                raise BrainError("Unsupported interrupt tool '{}'.".format(tool_name))
            calls.append(call(tool_name, role))
    return calls

def _secondary_task_plan(env: Any, primary: Mapping[str, Any]) -> list[ToolCall]:
    task = primary.get('secondary_task')
    if not isinstance(task, Mapping):
        return _interrupt_plan(primary)
    task_type = str(task.get('type') or '')
    if task_type == 'place_object':
        secondary_role = str(task.get('object_role') or 'secondary_object')
        receptacle_role = str(task.get('receptacle_role') or primary.get('receptacle_role') or 'target_receptacle')
        calls = [call('navigate_to', secondary_role), call('pick_up', secondary_role)]
        calls.extend(simple_place_plan(env, secondary_role, receptacle_role, close_after=False, start_with_object_held=True))
        return calls
    raise BrainError("Unsupported secondary task type '{}'.".format(task_type))

def _holding_role(env: Any, role_name: str) -> bool:
    if env is None:
        return False
    obj_name = env.role_map.get(role_name, role_name)
    return getattr(env.core.ego_state, 'held_object_name', None) == obj_name

def _holding_any(env: Any) -> bool:
    if env is None:
        return False
    return bool(getattr(env.core.ego_state, 'held_object_name', None))
__all__ = ['MidExecutionDisturbanceTeacher']
