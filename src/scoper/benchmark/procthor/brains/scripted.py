from __future__ import annotations
from typing import Any, List, Mapping, Sequence
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainInput, ToolCall

class ScriptedBrain(BaseBrain):

    def __init__(self, script: Sequence[ToolCall]):
        self._script: List[ToolCall] = [ToolCall(**tc) if isinstance(tc, Mapping) else tc for tc in script]
        self._cursor = 0

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._cursor = 0

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._cursor < len(self._script):
            call = self._script[self._cursor]
            self._cursor += 1
            return call
        return ToolCall('stop', {}, reasoning='script exhausted')
__all__ = ['ScriptedBrain']
