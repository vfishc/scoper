from __future__ import annotations
from typing import Any, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class VerifyAfterActionTeacher(BaseBrain):
    _MAX_RETRIES = 3

    def __init__(self) -> None:
        self._env: Any = None
        self._task_spec: Optional[Mapping[str, Any]] = None
        self._queue: List[ToolCall] = []
        self._retries = 0

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        self._task_spec = task_spec
        self._queue = self._base_plan(task_spec)
        self._retries = 0

    def act(self, brain_input: BrainInput) -> ToolCall:
        last_transition = brain_input.last_transition
        if last_transition is not None:
            success = last_transition.get('success', True)
            if not success:
                if self._retries < self._MAX_RETRIES:
                    self._retries += 1
                    return self._build_recovery_action(brain_input)
                else:
                    return ToolCall('stop', {}, reasoning='max_retries_exceeded')
        if self._queue:
            return self._queue.pop(0)
        env = self._env
        task_spec = self._task_spec
        if env is None or task_spec is None:
            return ToolCall('stop', {}, reasoning='verify teacher is unbound')
        goal = env.check_goal(model_stopped=getattr(env, 'model_stopped', False))
        if float(goal.get('predicate_success', 0.0)) >= 1.0:
            return ToolCall('stop', {}, reasoning='verified final state')
        if self._retries >= self._MAX_RETRIES:
            return ToolCall('stop', {}, reasoning='verify retry budget exhausted')
        self._retries += 1
        self._queue = self._recovery_plan(task_spec)
        if self._queue:
            return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='no verify recovery plan')

    def _base_plan(self, task_spec: Mapping[str, Any]) -> List[ToolCall]:
        verify = task_spec['family_spec']['verify']
        verification_task = verify['verification_task']
        task_type = verification_task['type']
        if task_type == 'place_object':
            object_role = verification_task['object_role']
            receptacle_role = verification_task['receptacle_role']
            return [_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', receptacle_role), _call('put_down', receptacle_role)]
        if task_type == 'open_container':
            container_role = verification_task['container_role']
            return [_call('navigate_to', container_role), _call('open', container_role)]
        if task_type == 'close_container':
            container_role = verification_task['container_role']
            return [_call('navigate_to', container_role), _call('close', container_role)]
        raise BrainError("Unknown verify task type '{}'.".format(task_type))

    def _build_recovery_action(self, brain_input: BrainInput) -> ToolCall:
        last_transition = brain_input.last_transition
        if not last_transition:
            self._rebuild_queue_from_goal()
            if self._queue:
                return self._queue.pop(0)
            return ToolCall('stop', {}, reasoning='cannot_recover')
        anomaly_mode = last_transition.get('anomaly_mode')
        last_action = last_transition.get('action', {})
        last_tool = last_action.get('tool_name')
        last_target = last_action.get('args', {}).get('target')
        if anomaly_mode == 'drop_beside':
            return ToolCall('look_for', {'target': last_target})
        elif anomaly_mode == 'grasp_slip':
            return ToolCall('pick_up', {'target': last_target})
        elif anomaly_mode == 'nav_blocked_partway':
            return ToolCall('navigate_to', {'target': last_target})
        elif anomaly_mode == 'partial_open':
            return ToolCall('open', {'target': last_target})
        elif anomaly_mode == 'partial_close':
            return ToolCall('close', {'target': last_target})
        else:
            self._rebuild_queue_from_goal()
            if self._queue:
                return self._queue.pop(0)
            return ToolCall('stop', {}, reasoning='unknown_failure')

    def _rebuild_queue_from_goal(self) -> None:
        if self._task_spec:
            self._queue = self._base_plan(self._task_spec)

    def _recovery_plan(self, task_spec: Mapping[str, Any]) -> List[ToolCall]:
        verify = task_spec['family_spec']['verify']
        anomaly = verify['execution_anomaly']
        mode = anomaly['mode']
        verification_task = verify['verification_task']
        if mode in {'drop_beside', 'grasp_slip'}:
            object_role = verification_task['object_role']
            receptacle_role = verification_task['receptacle_role']
            return [ToolCall('look_for', {'target': object_role}, reasoning='find the dropped object'), _call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', receptacle_role), _call('put_down', receptacle_role)]
        if mode == 'partial_open':
            container_role = anomaly.get('target_role') or verification_task.get('container_role')
            return [_call('navigate_to', container_role), _call('open', container_role)]
        if mode == 'partial_close':
            container_role = anomaly.get('target_role') or verification_task.get('container_role')
            return [_call('navigate_to', container_role), _call('close', container_role)]
        if mode in {'nav_blocked_partway', 'nav_wrong_target'}:
            target_role = anomaly.get('target_role')
            task_type = verification_task.get('type')
            if task_type == 'place_object':
                object_role = verification_task['object_role']
                receptacle_role = verification_task['receptacle_role']
                if target_role == receptacle_role:
                    return [_call('navigate_to', receptacle_role), _call('put_down', receptacle_role)]
                return [_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', receptacle_role), _call('put_down', receptacle_role)]
            container_role = target_role or verification_task.get('container_role')
            if task_type == 'open_container':
                return [_call('navigate_to', container_role), _call('open', container_role)]
            if task_type == 'close_container':
                return [_call('navigate_to', container_role), _call('close', container_role)]
            return [_call('navigate_to', container_role)] if container_role else []
        raise BrainError("Unknown verify anomaly mode '{}'.".format(mode))

def _call(tool_name: str, target: str) -> ToolCall:
    return ToolCall(tool_name, {'target': target})
from scoper.benchmark.procthor.brains.teachers import get_dispatcher
get_dispatcher().register('verify_after_action', VerifyAfterActionTeacher)
__all__ = ['VerifyAfterActionTeacher']
