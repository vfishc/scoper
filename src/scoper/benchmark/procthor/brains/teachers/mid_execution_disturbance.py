from __future__ import annotations
from typing import Any, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class MidExecutionDisturbanceTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: List[ToolCall] = []
        self._env: Any = None
        self._task_spec: Optional[Mapping[str, Any]] = None
        self._max_recoveries = 3
        self._recovery_attempts = 0
        self._primary_target_role: Optional[str] = None
        self._last_issued_call: Optional[ToolCall] = None
        self._retried_failure: Optional[tuple[str, str]] = None

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        self._task_spec = task_spec
        self._recovery_attempts = 0
        self._last_issued_call = None
        self._retried_failure = None
        mid = task_spec['family_spec']['mid_execution']
        primary = mid['primary_task']
        self._primary_target_role = primary.get('object_role')
        self._queue = self._build_plan(env, task_spec)

    def act(self, brain_input: BrainInput) -> ToolCall:
        env_feedback = getattr(brain_input, 'env_feedback', None)
        if env_feedback:
            error_type = env_feedback.get('error_type')
            if error_type == 'container_closed':
                container_role = env_feedback.get('container_role')
                if container_role and self._primary_target_role:
                    self._rebuild_queue_with_reopen(container_role)
            elif error_type == 'object_relocated':
                object_role = env_feedback.get('object_role')
                if object_role:
                    self._rebuild_queue_after_relocation(object_role)
        last_transition = brain_input.last_transition
        if last_transition and (not last_transition.get('success', True)):
            if self._primary_task_type() == 'object_relocated':
                tool_name = str(last_transition.get('tool_name') or last_transition.get('tool') or (self._last_issued_call.tool_name if self._last_issued_call is not None else '') or '')
                arguments = dict(last_transition.get('arguments') or (self._last_issued_call.arguments if self._last_issued_call is not None else {}))
                if tool_name in {'scan', 'look_for'}:
                    self._retried_failure = None
                else:
                    fingerprint = (tool_name, str(arguments.get('target') or ''))
                    if tool_name and self._retried_failure != fingerprint:
                        self._retried_failure = fingerprint
                        return self._issue(ToolCall(tool_name, arguments, reasoning='retry the failed relocation step once from the current state'))
                    return self._issue(ToolCall('stop', {}, reasoning='relocation step failed after one local retry'))
            elif self._recovery_attempts < self._max_recoveries:
                self._recovery_attempts += 1
                self._rebuild_queue_from_current_state()
            else:
                return self._issue(ToolCall('stop', {}, reasoning='max_recoveries_exceeded'))
        elif last_transition:
            self._retried_failure = None
        if self._queue:
            return self._issue(self._queue.pop(0))
        return self._issue(ToolCall('stop', {}, reasoning='mid-execution task complete'))

    def _rebuild_queue_with_reopen(self, container_role: str) -> None:
        recovery = [_call('navigate_to', container_role), _call('open', container_role)]
        if self._primary_target_role:
            recovery.extend([_call('navigate_to', self._primary_target_role), _call('pick_up', self._primary_target_role)])
        self._queue = recovery + self._queue

    def _rebuild_queue_after_relocation(self, object_role: str) -> None:
        recovery = [_call('navigate_to', object_role)]
        if self._task_spec:
            mid = self._task_spec['family_spec']['mid_execution']
            primary = mid['primary_task']
            disturbance_type = primary.get('type')
            if disturbance_type == 'state_re_dirty':
                sink_role = primary.get('sink_role') or 'sink_basin'
                faucet_role = primary.get('faucet_role') or 'faucet'
                recovery.extend([_call('pick_up', object_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role)])
            elif disturbance_type == 'container_re_closed':
                receptacle_role = primary.get('receptacle_role')
                if receptacle_role:
                    recovery.extend([_call('pick_up', object_role), _call('navigate_to', receptacle_role)])
        self._queue = recovery + self._queue

    def _rebuild_queue_from_current_state(self) -> None:
        if self._env and self._task_spec:
            self._queue = self._build_plan(self._env, self._task_spec)

    def _build_plan(self, env: Any, task_spec: Mapping[str, Any]) -> List[ToolCall]:
        mid = task_spec['family_spec']['mid_execution']
        primary = mid['primary_task']
        ptype = primary['type']
        object_role = primary['object_role']
        receptacle_role = primary['receptacle_role']
        if ptype == 'state_re_dirty':
            if primary.get('staging_receptacle_role') or primary.get('interrupt_role'):
                return self._state_re_dirty_after_staging_plan(primary, object_role, receptacle_role)
            return self._state_re_dirty_plan(primary, object_role, receptacle_role)
        if ptype == 'object_relocated':
            if primary.get('staging_receptacle_role') or primary.get('interrupt_role'):
                return self._object_relocated_after_staging_plan(object_role, receptacle_role, primary)
            return self._object_relocated_plan(object_role, receptacle_role)
        if ptype == 'container_re_closed':
            return self._container_re_closed_plan(env, primary, object_role, receptacle_role)
        raise BrainError("Unknown mid-execution primary task type '{}'.".format(ptype))

    def _state_re_dirty_plan(self, primary: Mapping[str, Any], object_role: str, receptacle_role: str) -> List[ToolCall]:
        sink_role = primary.get('sink_role') or 'sink_basin'
        faucet_role = primary.get('faucet_role') or 'faucet'
        calls: List[ToolCall] = [_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role), _call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role), _call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', receptacle_role)]
        if primary.get('receptacle_open_role'):
            calls.append(_call('open', receptacle_role))
        calls.append(_call('put_down', receptacle_role))
        if primary.get('receptacle_close_role'):
            calls.append(_call('close', receptacle_role))
        calls.append(ToolCall('stop', {}, reasoning='verified recovered clean placement'))
        return calls

    def _state_re_dirty_after_staging_plan(self, primary: Mapping[str, Any], object_role: str, receptacle_role: str) -> List[ToolCall]:
        sink_role = primary.get('sink_role') or 'sink_basin'
        faucet_role = primary.get('faucet_role') or 'faucet'
        staging_role = primary.get('staging_receptacle_role') or receptacle_role
        calls: List[ToolCall] = [_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role), _call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', staging_role), _call('put_down', staging_role)]
        secondary_task = primary.get('secondary_task')
        if isinstance(secondary_task, Mapping) and secondary_task.get('type') == 'place_object':
            secondary_object_role = secondary_task.get('object_role')
            secondary_receptacle_role = secondary_task.get('receptacle_role')
            if secondary_object_role and secondary_receptacle_role:
                calls.extend([_call('navigate_to', secondary_object_role), _call('pick_up', secondary_object_role), _call('navigate_to', secondary_receptacle_role), _call('put_down', secondary_receptacle_role)])
        calls.extend(_interrupt_plan(primary))
        calls.extend([_call('navigate_to', staging_role), ToolCall('look_for', {'target': object_role}, reasoning='look for the staged object before recovering it'), _call('pick_up', object_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role), _call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', receptacle_role)])
        if primary.get('receptacle_open_role'):
            calls.append(_call('open', receptacle_role))
        calls.append(_call('put_down', receptacle_role))
        if primary.get('receptacle_close_role'):
            calls.append(_call('close', receptacle_role))
        calls.append(ToolCall('stop', {}, reasoning='verified staged object was recovered after semantic disturbance'))
        return calls

    def _object_relocated_plan(self, object_role: str, receptacle_role: str) -> List[ToolCall]:
        return [_call('navigate_to', object_role), ToolCall('scan', {'num_views': 4}, reasoning='inspect the old location before re-localizing the moved object'), _call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', receptacle_role), _call('put_down', receptacle_role), ToolCall('stop', {}, reasoning='verified relocated object was placed')]

    def _object_relocated_after_staging_plan(self, object_role: str, receptacle_role: str, primary: Mapping[str, Any]) -> List[ToolCall]:
        staging_role = primary.get('staging_receptacle_role') or receptacle_role
        calls: List[ToolCall] = [_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', staging_role), _call('put_down', staging_role)]
        interrupt_calls = _interrupt_plan(primary)
        if not interrupt_calls and self._task_spec:
            roles = self._task_spec.get('roles', {})
            faucet_role = None
            for role_name, role_data in roles.items():
                if isinstance(role_data, Mapping):
                    obj_type = role_data.get('object_type', '')
                    if obj_type == 'Faucet':
                        faucet_role = role_name
                        break
            if faucet_role:
                interrupt_calls = [_call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role)]
        calls.extend(interrupt_calls)
        calls.append(_call('navigate_to', staging_role))
        calls.append(ToolCall('scan', {'num_views': 4}, reasoning='inspect the staged location before re-localizing the moved object'))
        calls.extend([_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', receptacle_role), _call('put_down', receptacle_role), ToolCall('stop', {}, reasoning='verified object was relocated after interruption and placed')])
        return calls

    def _primary_task_type(self) -> Optional[str]:
        if not self._task_spec:
            return None
        primary = self._task_spec['family_spec']['mid_execution']['primary_task']
        value = primary.get('type')
        return str(value) if value else None

    def _issue(self, call: ToolCall) -> ToolCall:
        self._last_issued_call = call
        return call

    def _container_re_closed_plan(self, env: Any, primary: Mapping[str, Any], object_role: str, receptacle_role: str) -> List[ToolCall]:
        source_container = primary.get('container_role')
        target_roles = primary.get('target_objects') or [object_role]
        if not isinstance(target_roles, list):
            target_roles = [object_role]
        calls: List[ToolCall] = []
        if source_container:
            calls.append(_call('navigate_to', source_container))
            if _role_is_openable(env, source_container):
                calls.append(_call('open', source_container))
        for idx, target in enumerate(target_roles):
            calls.extend([_call('navigate_to', target), _call('pick_up', target), _call('navigate_to', receptacle_role)])
            if _role_is_openable(env, receptacle_role):
                calls.append(_call('open', receptacle_role))
            calls.append(_call('put_down', receptacle_role))
        if _role_is_openable(env, receptacle_role):
            calls.append(_call('close', receptacle_role))
        calls.append(ToolCall('stop', {}, reasoning='verified recovered container placement'))
        return calls

def _call(tool_name: str, target: str) -> ToolCall:
    return ToolCall(tool_name, {'target': target})

def _interrupt_plan(primary: Mapping[str, Any]) -> List[ToolCall]:
    role = primary.get('interrupt_role')
    if not isinstance(role, str) or not role:
        return []
    tools = list(primary.get('interrupt_tools') or ['navigate_to'])
    calls: List[ToolCall] = []
    for tool in tools:
        if not isinstance(tool, str) or not tool:
            continue
        calls.append(_call(tool, role))
    return calls

def _role_is_openable(env: Any, role_name: Optional[str]) -> bool:
    if not role_name:
        return False
    object_id = getattr(env, 'role_object_ids', {}).get(role_name)
    if not object_id:
        return False
    obj = env._objects_by_id().get(object_id, {})
    return bool(obj.get('openable', False))
from scoper.benchmark.procthor.brains.teachers import get_dispatcher
get_dispatcher().register('mid_execution_state_disturbance', MidExecutionDisturbanceTeacher)
__all__ = ['MidExecutionDisturbanceTeacher']
