from __future__ import annotations
from typing import Any, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class ObservationConditionedTeacher(BaseBrain):

    def __init__(self) -> None:
        self._env: Any = None
        self._task_spec: Optional[Mapping[str, Any]] = None
        self._queue: List[ToolCall] = []
        self._rooms: List[str] = []
        self._next_room_index = 0
        self._repair_started = False

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        obs = task_spec['family_spec']['observation_conditioned']
        self._env = env
        self._task_spec = task_spec
        self._rooms = [str(room) for room in obs.get('rooms') or [] if str(room)]
        self._next_room_index = 0
        self._repair_started = False
        self._queue = []
        self._enqueue_next_room_scan(obs)

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        if self._env is None or self._task_spec is None:
            return ToolCall('stop', {}, reasoning='observation-conditioned teacher unbound')
        goal = self._env.check_goal(model_stopped=getattr(self._env, 'model_stopped', False))
        if float(goal.get('predicate_success', 0.0)) >= 1.0:
            return ToolCall('stop', {}, reasoning='all visible issues repaired')
        obs = self._task_spec['family_spec']['observation_conditioned']
        if self._next_room_index < len(self._rooms):
            self._enqueue_next_room_scan(obs)
            if self._queue:
                return self._queue.pop(0)
        if not self._repair_started:
            self._repair_started = True
            self._queue = self._repair_plan_from_visible_issues()
            if self._queue:
                return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='no observation-conditioned repair remains')

    def _enqueue_next_room_scan(self, obs: Mapping[str, Any]) -> None:
        if self._next_room_index >= len(self._rooms):
            return
        room_id = self._rooms[self._next_room_index]
        self._next_room_index += 1
        scan_policy = obs.get('scan_policy') if isinstance(obs.get('scan_policy'), Mapping) else {}
        num_views = int(scan_policy.get('num_views', 4) or 4)
        require_entry = bool(scan_policy.get('require_room_entry_before_scan', True))
        if require_entry:
            room_target = next((str(room.get('label')) for room in getattr(self._env, '_rooms_for_navigation', []) if str(room.get('room_id')) == room_id and room.get('label')), room_id)
            self._queue.append(ToolCall('navigate_to', {'target': room_target}, reasoning='enter room before inspection'))
        self._queue.append(ToolCall('scan', {'num_views': num_views}, reasoning='inspect room for latent issues'))

    def _repair_plan_from_visible_issues(self) -> List[ToolCall]:
        visible = []
        getter = getattr(self._env, 'observation_conditioned_visible_issues', None)
        if callable(getter):
            visible = list(getter())
        elif hasattr(self._env, '_observation_conditioned_visible_issues'):
            visible = list(getattr(self._env, '_observation_conditioned_visible_issues') or [])
        if not visible and self._task_spec is not None:
            visible = list(self._task_spec['family_spec']['observation_conditioned'].get('latent_payloads') or [])
        calls: List[ToolCall] = []
        seen = set()
        for issue in visible:
            if not isinstance(issue, Mapping):
                continue
            issue_id = str(issue.get('latent_id') or issue.get('issue_id') or '')
            if issue_id and issue_id in seen:
                continue
            if issue_id:
                seen.add(issue_id)
            calls.extend(_repair_plan_for_issue(self._env, issue))
        calls.append(ToolCall('stop', {}, reasoning='observation-conditioned repair plan complete'))
        return calls

def _repair_plan_for_issue(env: Any, issue: Mapping[str, Any]) -> List[ToolCall]:
    template = issue.get('repair_plan_template')
    object_role = issue.get('object_role')
    if not isinstance(object_role, str) or not object_role:
        raise BrainError('Observation-conditioned issue missing object_role.')
    if template == 'toggle_off':
        return [_call('navigate_to', object_role), _call('toggle_off', object_role)]
    if template == 'close':
        return [_call('navigate_to', object_role), _call('close', object_role)]
    if template == 'wash_then_place':
        support = issue.get('support_roles') if isinstance(issue.get('support_roles'), Mapping) else {}
        sink_role = str(support.get('sink') or 'sink_basin')
        faucet_role = str(support.get('faucet') or 'faucet')
        storage_role = support.get('storage')
        calls = [_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role)]
        if isinstance(storage_role, str) and storage_role:
            calls.extend([_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', storage_role)])
            if _role_is_openable(env, storage_role):
                calls.append(_call('open', storage_role))
            calls.append(_call('put_down', storage_role))
            if _role_is_openable(env, storage_role):
                calls.append(_call('close', storage_role))
        return calls
    if template == 'place':
        support = issue.get('support_roles') if isinstance(issue.get('support_roles'), Mapping) else {}
        storage_role = support.get('storage')
        if not isinstance(storage_role, str) or not storage_role:
            raise BrainError('Observation-conditioned place issue missing storage role.')
        calls = [_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', storage_role)]
        if _role_is_openable(env, storage_role):
            calls.append(_call('open', storage_role))
        calls.append(_call('put_down', storage_role))
        if _role_is_openable(env, storage_role):
            calls.append(_call('close', storage_role))
        return calls
    raise BrainError("Unknown observation-conditioned repair template '{}'.".format(template))

def _call(tool_name: str, target: str) -> ToolCall:
    return ToolCall(tool_name, {'target': target})

def _role_is_openable(env: Any, role_name: str) -> bool:
    object_id = getattr(env, 'role_object_ids', {}).get(role_name)
    if not object_id:
        return False
    return bool(env._objects_by_id().get(object_id, {}).get('openable', False))
from scoper.benchmark.procthor.brains.teachers import get_dispatcher
get_dispatcher().register('observation_conditioned_room_inspection', ObservationConditionedTeacher)
get_dispatcher().register('observation_conditioned_task', ObservationConditionedTeacher)
__all__ = ['ObservationConditionedTeacher']
