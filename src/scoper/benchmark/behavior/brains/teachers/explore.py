from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.behavior.brains.teachers.common import call, simple_place_plan, stop, wash_plan

class ExploreTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: list[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        explore = (task_spec.get('family_spec') or {}).get('explore')
        if not isinstance(explore, Mapping):
            raise BrainError('ExploreTeacher requires family_spec.explore.')
        discovery = explore.get('discovery_spec') or {}
        target_role = str(discovery.get('target_role') or 'target_object')
        post_task = explore.get('post_discovery_task')
        if not isinstance(post_task, Mapping):
            raise BrainError('ExploreTeacher requires post_discovery_task.')
        calls = [call('scan', reasoning='oracle discovery scan'), ToolCall('look_for', {'target': target_role}, reasoning='privileged target localization'), call('navigate_to', target_role)]
        calls.extend(_post_task_plan(env, target_role, post_task))
        calls.append(stop('explore-then-act plan complete'))
        self._queue = calls

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return stop('explore-then-act plan exhausted')

def _post_task_plan(env: Any, target_role: str, post_task: Mapping[str, Any]) -> list[ToolCall]:
    task_type = str(post_task.get('task_type') or post_task.get('type') or '')
    if task_type == 'pickup_and_place':
        receptacle_role = str(post_task.get('target_receptacle_role') or post_task.get('receptacle_role') or 'target_receptacle')
        return simple_place_plan(env, target_role, receptacle_role, close_after=True)
    if task_type == 'toggle':
        return [call('toggle_on', target_role)]
    if task_type == 'open':
        return [call('open', target_role)]
    if task_type == 'clean':
        return wash_plan(env, target_role, faucet_role=post_task.get('faucet_role'), keep_held=False)
    raise BrainError("Unknown explore post-discovery task type '{}'.".format(task_type))
__all__ = ['ExploreTeacher']
