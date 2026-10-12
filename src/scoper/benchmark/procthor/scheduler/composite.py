from __future__ import annotations
from typing import Any, Callable, List, Mapping, Optional, Sequence
from scoper.benchmark.procthor.scheduler.base import Scheduler, SchedulerEvent

class CompositeScheduler(Scheduler):
    name = 'composite'

    def __init__(self, children: Sequence[Scheduler]):
        self._children: List[Scheduler] = list(children)

    @property
    def children(self) -> List[Scheduler]:
        return list(self._children)

    def on_episode_start(self, env, task_spec):
        return self._collect('on_episode_start', env, task_spec)

    def pre_step(self, env, high_level_step, last_transition):
        return self._collect('pre_step', env, high_level_step, last_transition)

    def post_step(self, env, high_level_step, transition):
        return self._collect('post_step', env, high_level_step, transition)

    def on_episode_end(self, env, final_goal):
        return self._collect('on_episode_end', env, final_goal)

    def intercept_tool_call(self, env: Any, tool_call: Any, high_level_step: int, default_call: Callable[[], dict]) -> Optional[dict]:
        for child in self._children:
            override = child.intercept_tool_call(env, tool_call, high_level_step, default_call)
            if override is not None:
                return override
        return None

    def _collect(self, method_name: str, *args) -> Optional[List[SchedulerEvent]]:
        events: List[SchedulerEvent] = []
        for child in self._children:
            method = getattr(child, method_name, None)
            if method is None:
                continue
            result = method(*args)
            if not result:
                continue
            for event in result:
                if isinstance(event, SchedulerEvent):
                    events.append(event)
                elif isinstance(event, Mapping):
                    events.append(SchedulerEvent(name=str(event.get('name', 'event')), phase=str(event.get('phase', method_name)), payload=dict(event.get('payload', {}))))
        return events or None
__all__ = ['CompositeScheduler']
