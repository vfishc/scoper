from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelCallContext:
    episode_id: str
    model_call_id: str
    call_type: str
    decision_id: str | None
    action_id: str | None
    stream_obs_id: str | None
    verification_id: str | None
__all__ = ['ModelCallContext']
