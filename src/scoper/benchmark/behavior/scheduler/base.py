from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional

@dataclass
class SchedulerEvent:
    name: str
    phase: str
    payload: dict = field(default_factory=dict)

class Scheduler:
    name: str = 'scheduler'

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> Optional[List[SchedulerEvent]]:
        return None

    def pre_step(self, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]) -> Optional[List[SchedulerEvent]]:
        return None

    def intercept_tool_call(self, env: Any, tool_call: Any, high_level_step: int, default_call: Callable[[], dict]) -> Optional[dict]:
        return None

    def post_step(self, env: Any, high_level_step: int, transition: Mapping[str, Any]) -> Optional[List[SchedulerEvent]]:
        return None

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any]) -> Optional[List[SchedulerEvent]]:
        return None

class NoOpScheduler(Scheduler):
    name = 'noop'
__all__ = ['NoOpScheduler', 'Scheduler', 'SchedulerEvent']
