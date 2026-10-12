from __future__ import annotations
from typing import Any, List, Mapping
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class ExploreTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: List[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        explore = task_spec['family_spec']['explore']
        discovery = explore['discovery_spec']
        target_role = discovery.get('target_role') or 'target_object'
        if not isinstance(target_role, str) or not target_role:
            raise BrainError('explore.discovery_spec.target_role must be a role name.')
        post_task = explore['post_discovery_task']
        calls: List[ToolCall] = [ToolCall('scan', {}), _call('look_for', target_role), _call('navigate_to', target_role)]
        calls.extend(self._post_discovery_plan(env, target_role, post_task))
        calls.append(ToolCall('stop', {}, reasoning='explore-then-act plan complete'))
        self._queue = calls

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='explore-then-act plan exhausted')

    def _post_discovery_plan(self, env: Any, target_role: str, post_task: Mapping[str, Any]) -> List[ToolCall]:
        task_type = post_task.get('task_type')
        if task_type == 'pickup_and_place':
            receptacle_role = post_task.get('target_receptacle_role') or post_task.get('target_receptacle')
            if not isinstance(receptacle_role, str) or not receptacle_role:
                raise BrainError('pickup_and_place requires a target receptacle role.')
            calls = [_call('pick_up', target_role), _call('navigate_to', receptacle_role)]
            if _role_is_openable(env, receptacle_role):
                calls.append(_call('open', receptacle_role))
            calls.append(_call('put_down', receptacle_role))
            if _role_is_openable(env, receptacle_role):
                calls.append(_call('close', receptacle_role))
            return calls
        if task_type == 'toggle':
            return [_call('toggle_on', target_role)]
        if task_type == 'open':
            return [_call('open', target_role)]
        if task_type == 'clean':
            sink_role = str(post_task.get('sink_role') or 'sink_basin')
            faucet_role = str(post_task.get('faucet_role') or 'faucet')
            return [_call('pick_up', target_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role), _call('navigate_to', target_role), _call('pick_up', target_role)]
        raise BrainError("Unknown explore post_discovery task_type '{}'.".format(task_type))

def _call(tool_name: str, target: str) -> ToolCall:
    return ToolCall(tool_name, {'target': target})

def _role_is_openable(env: Any, role_name: str) -> bool:
    object_id = getattr(env, 'role_object_ids', {}).get(role_name)
    if not object_id:
        return False
    return bool(env._objects_by_id().get(object_id, {}).get('openable', False))
from scoper.benchmark.procthor.brains.teachers import get_dispatcher
get_dispatcher().register('explore_then_act', ExploreTeacher)
__all__ = ['ExploreTeacher']
