from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

@dataclass
class ToolCall:
    tool_name: str
    arguments: dict = field(default_factory=dict)
    reasoning: str = ''

@dataclass
class BrainInput:
    observation: Mapping[str, Any]
    prompt_context: Mapping[str, Any]
    last_transition: Optional[Mapping[str, Any]]
    high_level_step: int

@dataclass(frozen=True)
class ActionResultContext:
    step_id: int
    decision_id: str
    action_id: str
    tool_call: ToolCall
    public_transition: Mapping[str, Any]
    pre_public_observation: Mapping[str, Any]
    post_public_observation: Mapping[str, Any]
    public_postcondition_evidence_or_null: Optional[Mapping[str, Any]] = None
    immediate_post_public_observation_or_null: Optional[Mapping[str, Any]] = None
    confirmation_public_observation_or_null: Optional[Mapping[str, Any]] = None
    private_binding_roles_or_null: Optional[tuple[str, ...]] = None

class BrainError(RuntimeError):
    pass

@runtime_checkable
class Brain(Protocol):

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        ...

    def act(self, brain_input: BrainInput) -> ToolCall:
        ...

    def prepare_input(self, brain_input: BrainInput) -> BrainInput:
        ...

    def on_action_result(self, context: ActionResultContext) -> None:
        ...

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any]) -> None:
        ...

class BaseBrain:

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        return None

    def act(self, brain_input: BrainInput) -> ToolCall:
        raise NotImplementedError

    def prepare_input(self, brain_input: BrainInput) -> BrainInput:
        return brain_input

    def on_action_result(self, context: ActionResultContext) -> None:
        return None

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any]) -> None:
        return None
__all__ = ['ActionResultContext', 'Brain', 'BaseBrain', 'BrainError', 'BrainInput', 'ToolCall']
