from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import ToolCall

@dataclass
class HookEvent:
    name: str
    phase: str
    payload: dict = field(default_factory=dict)

class Hook:
    name: str = 'hook'

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any], reset_observation: Mapping[str, Any]) -> Optional[Iterable[HookEvent]]:
        return None

    def before_tool(self, env: Any, tool_call: ToolCall, high_level_step: int) -> Optional[Iterable[HookEvent]]:
        return None

    def on_decision_input(self, env: Any, brain_input: Any, high_level_step: int, decision_context: Mapping[str, Any]) -> Optional[Iterable[HookEvent]]:
        return None

    def after_tool(self, env: Any, tool_call: ToolCall, transition: Mapping[str, Any], high_level_step: int) -> Optional[Iterable[HookEvent]]:
        return None

    def capture_immediate_post_action(self, env: Any, tool_call: ToolCall, transition: Mapping[str, Any], high_level_step: int) -> Optional[Iterable[HookEvent]]:
        return None

    def on_low_level(self, env: Any, action_record: Mapping[str, Any], event: Any) -> Optional[Iterable[HookEvent]]:
        return None

    def on_scheduler_events(self, env: Any, events: Iterable[Mapping[str, Any]], phase: str, high_level_step: int) -> Optional[Iterable[HookEvent]]:
        return None

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any], status: str) -> Optional[Iterable[HookEvent]]:
        return None

class HookBus:

    def __init__(self, hooks: Optional[Iterable[Hook]]=None):
        self._hooks: List[Hook] = list(hooks or [])

    @property
    def hooks(self) -> List[Hook]:
        return list(self._hooks)

    def _dispatch(self, method_name: str, *args, **kwargs) -> List[dict]:
        events: List[dict] = []
        for hook in self._hooks:
            method = getattr(hook, method_name, None)
            if method is None:
                continue
            result = method(*args, **kwargs)
            if not result:
                continue
            for event in result:
                if isinstance(event, HookEvent):
                    events.append({'name': event.name, 'phase': event.phase, 'payload': dict(event.payload), 'hook': hook.name})
                elif isinstance(event, Mapping):
                    events.append({**event, 'hook': hook.name})
        return events

    def on_episode_start(self, env, task_spec, reset_observation) -> List[dict]:
        return self._dispatch('on_episode_start', env, task_spec, reset_observation)

    def before_tool(self, env, tool_call, high_level_step) -> List[dict]:
        return self._dispatch('before_tool', env, tool_call, high_level_step)

    def on_decision_input(self, env, brain_input, high_level_step, decision_context) -> List[dict]:
        return self._dispatch('on_decision_input', env, brain_input, high_level_step, decision_context)

    def after_tool(self, env, tool_call, transition, high_level_step) -> List[dict]:
        return self._dispatch('after_tool', env, tool_call, transition, high_level_step)

    def capture_immediate_post_action(self, env, tool_call, transition, high_level_step) -> List[dict]:
        return self._dispatch('capture_immediate_post_action', env, tool_call, transition, high_level_step)

    def on_low_level(self, env, action_record, event) -> List[dict]:
        return self._dispatch('on_low_level', env, action_record, event)

    def on_scheduler_events(self, env, events, phase, high_level_step) -> List[dict]:
        return self._dispatch('on_scheduler_events', env, events, phase, high_level_step)

    def on_episode_end(self, env, final_goal, status) -> List[dict]:
        return self._dispatch('on_episode_end', env, final_goal, status)
__all__ = ['Hook', 'HookBus', 'HookEvent']
