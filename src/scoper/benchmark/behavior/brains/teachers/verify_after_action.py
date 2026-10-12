from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.behavior.brains.teachers.common import call, simple_place_plan, stop

class VerifyAfterActionTeacher(BaseBrain):
    _MAX_RECOVERIES = 4

    def __init__(self) -> None:
        self._env: Any = None
        self._task_spec: Mapping[str, Any] = {}
        self._queue: list[ToolCall] = []
        self._recoveries = 0

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        self._task_spec = task_spec
        self._queue = self._base_plan(env, task_spec)
        self._recoveries = 0

    def act(self, brain_input: BrainInput) -> ToolCall:
        last = brain_input.last_transition or {}
        if last.get('success') is False and self._recoveries < self._MAX_RECOVERIES:
            self._recoveries += 1
            self._queue = self._recovery_plan(self._env, self._task_spec)
        if self._queue:
            return self._queue.pop(0)
        goal = self._env.check_goal(model_stopped=getattr(self._env, 'model_stopped', False))
        if float(goal.get('predicate_success', 0.0) or 0.0) >= 1.0:
            return stop('verified final state')
        if self._recoveries < self._MAX_RECOVERIES:
            self._recoveries += 1
            self._queue = self._recovery_plan(self._env, self._task_spec)
            if self._queue:
                return self._queue.pop(0)
        return stop('verify retry budget exhausted')

    def _base_plan(self, env: Any, task_spec: Mapping[str, Any]) -> list[ToolCall]:
        task = _verification_task(task_spec)
        task_type = task.get('type')
        if task_type == 'place_object':
            return simple_place_plan(env, str(task['object_role']), str(task['receptacle_role']), close_after=False)
        if task_type == 'open_container':
            role = str(task['container_role'])
            return [call('navigate_to', role), call('open', role)]
        if task_type == 'close_container':
            role = str(task['container_role'])
            return [call('navigate_to', role), call('close', role)]
        raise BrainError("Unknown verify task type '{}'.".format(task_type))

    def _recovery_plan(self, env: Any, task_spec: Mapping[str, Any]) -> list[ToolCall]:
        verify = (task_spec.get('family_spec') or {}).get('verify') or {}
        anomaly = verify.get('execution_anomaly') or {}
        mode = anomaly.get('mode')
        task = _verification_task(task_spec)
        task_type = task.get('type')
        if task_type == 'place_object':
            object_role = str(task['object_role'])
            receptacle_role = str(task['receptacle_role'])
            prefix = []
            if mode in {'drop_beside', 'grasp_slip', 'nav_wrong_target', 'nav_blocked_partway'}:
                prefix.append(ToolCall('look_for', {'target': object_role}, reasoning='re-localize the target object'))
            return prefix + simple_place_plan(env, object_role, receptacle_role, close_after=False)
        if task_type == 'open_container':
            role = str(task.get('container_role') or anomaly.get('target_role'))
            return [ToolCall('look_for', {'target': role}, reasoning='verify container state'), call('navigate_to', role), call('open', role)]
        if task_type == 'close_container':
            role = str(task.get('container_role') or anomaly.get('target_role'))
            return [ToolCall('look_for', {'target': role}, reasoning='verify container state'), call('navigate_to', role), call('close', role)]
        return []

def _verification_task(task_spec: Mapping[str, Any]) -> Mapping[str, Any]:
    family = task_spec.get('family_spec') or {}
    verify = family.get('verify')
    if not isinstance(verify, Mapping):
        raise BrainError('VerifyAfterActionTeacher requires family_spec.verify.')
    task = verify.get('verification_task')
    if not isinstance(task, Mapping):
        raise BrainError('VerifyAfterActionTeacher requires verify.verification_task.')
    return task
__all__ = ['VerifyAfterActionTeacher']
