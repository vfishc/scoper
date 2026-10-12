from __future__ import annotations
from typing import Any, Mapping, Optional
from scoper.benchmark.procthor.scheduler.base import Scheduler, SchedulerEvent

class ActiveSubtaskScheduler(Scheduler):
    name = 'active_subtask'

    def __init__(self, scheduler: Scheduler, *, subtask_id: str, task_spec: Mapping[str, Any]) -> None:
        self.scheduler = scheduler
        self.subtask_id = str(subtask_id)
        self.task_spec = task_spec
        self._initialized = False
        self._pending_events: list[SchedulerEvent] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]):
        del task_spec
        events = self._initialize_if_active(env)
        return events or None

    def pre_step(self, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]):
        if not self._is_active(env):
            return None
        events = self._initialize_if_active(env)
        events.extend(self.scheduler.pre_step(env, high_level_step, last_transition) or [])
        return events or None

    def intercept_tool_call(self, env: Any, tool_call: Any, high_level_step: int, default_call: Any):
        if not self._is_active(env):
            return None
        init_events = self._initialize_if_active(env)
        if init_events:
            self._pending_events.extend(init_events)
        return self.scheduler.intercept_tool_call(env, tool_call, high_level_step, default_call)

    def post_step(self, env: Any, high_level_step: int, transition: Mapping[str, Any]):
        if not self._is_active(env):
            return None
        events = list(self._pending_events)
        self._pending_events = []
        events.extend(self._initialize_if_active(env))
        events.extend(self.scheduler.post_step(env, high_level_step, transition) or [])
        return events or None

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any]):
        if not self._initialized:
            return None
        return self.scheduler.on_episode_end(env, final_goal)

    def _initialize_if_active(self, env: Any) -> list[SchedulerEvent]:
        if self._initialized or not self._is_active(env):
            return []
        self._initialized = True
        return list(self.scheduler.on_episode_start(env, self.task_spec) or [])

    def _is_active(self, env: Any) -> bool:
        active = getattr(env, '_evaluation_active_subtask_spec', getattr(env, '_composite_active_task_spec', None))
        if not isinstance(active, Mapping):
            return False
        subtask_id = active.get('subtask_id')
        if not isinstance(subtask_id, str) or not subtask_id:
            task_id = active.get('id')
            if isinstance(task_id, str) and '::' in task_id:
                subtask_id = task_id.rsplit('::', 1)[1]
        return subtask_id == self.subtask_id
__all__ = ['ActiveSubtaskScheduler']
