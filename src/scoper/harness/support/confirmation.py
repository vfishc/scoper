from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Mapping
from scoper.harness.support.method_contract import MethodContractError
CONFIRMABLE_ACTIONS = frozenset({'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'clean', 'slice', 'fill', 'saturate', 'heat', 'cool'})
BOUND_TARGET_CONFIRMATION_ACTIONS = CONFIRMABLE_ACTIONS - {'pick_up'}
TRANSITION_IMAGE_ROLES = ('pre_rgb', 'immediate_post_rgb', 'confirmation_rgb')

@dataclass(frozen=True)
class ConfirmationIntent:
    canonical_action: str
    view_kind: str
    target_public_ref_or_null: str | None

    def __post_init__(self) -> None:
        if self.canonical_action not in CONFIRMABLE_ACTIONS:
            raise MethodContractError('method_schema_invalid', 'confirmation action is not visually verifiable')
        if self.view_kind not in {'held_region', 'bound_target', 'neutral_level'}:
            raise MethodContractError('method_schema_invalid', 'confirmation view kind is invalid')
        if self.target_public_ref_or_null is not None and (not isinstance(self.target_public_ref_or_null, str) or not self.target_public_ref_or_null):
            raise MethodContractError('method_schema_invalid', 'confirmation public target is invalid')

    def public_record(self) -> dict[str, Any]:
        return {'schema_version': 'confirmation-intent.v1', 'canonical_action': self.canonical_action, 'view_kind': self.view_kind, 'target_public_ref_or_null': self.target_public_ref_or_null}

def confirmation_intent_for_action(canonical_action: str, *, target_public_ref_or_null: str | None, has_bound_target: bool) -> ConfirmationIntent:
    if canonical_action not in CONFIRMABLE_ACTIONS:
        raise MethodContractError('method_schema_invalid', 'action has no visual confirmation contract')
    if canonical_action == 'pick_up':
        view_kind = 'held_region'
    elif canonical_action in BOUND_TARGET_CONFIRMATION_ACTIONS and has_bound_target:
        view_kind = 'bound_target'
    else:
        view_kind = 'neutral_level'
    return ConfirmationIntent(canonical_action=canonical_action, view_kind=view_kind, target_public_ref_or_null=target_public_ref_or_null)

@dataclass(frozen=True)
class TransitionEvidence:
    pre_frame_id: str
    immediate_post_frame_id: str
    confirmation_frame_id: str
    pre_rgb: Any
    immediate_post_rgb: Any
    confirmation_rgb: Any

    def __post_init__(self) -> None:
        frame_ids = self.frame_ids
        if any((not isinstance(item, str) or not item for item in frame_ids)):
            raise MethodContractError('required_frame_missing', 'transition evidence frame IDs must be non-empty')
        if len(set(frame_ids)) != 3:
            raise MethodContractError('method_event_correlation_invalid', 'transition evidence requires three distinct frame IDs')

    @property
    def frame_ids(self) -> tuple[str, str, str]:
        return (self.pre_frame_id, self.immediate_post_frame_id, self.confirmation_frame_id)

    def images(self) -> Mapping[str, Any]:
        return {'pre_rgb': self.pre_rgb, 'immediate_post_rgb': self.immediate_post_rgb, 'confirmation_rgb': self.confirmation_rgb}

@dataclass(frozen=True)
class LogicalEvidenceDecision:
    emit: bool
    forced: bool
    satisfies_periodic_tick: bool
    logical_time_s: float

class LogicalEvidenceSampler:

    def __init__(self, *, rate_hz: float=1.0) -> None:
        if isinstance(rate_hz, bool) or not isinstance(rate_hz, (int, float)):
            raise ValueError('rate_hz must be a positive finite number')
        rate = float(rate_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError('rate_hz must be a positive finite number')
        self.rate_hz = rate
        self.interval_s = 1.0 / rate
        self.reset()

    def reset(self, *, start_time_s: float=0.0) -> None:
        start = float(start_time_s)
        if not math.isfinite(start):
            raise ValueError('start_time_s must be finite')
        self._last_time_s: float | None = None
        self._next_tick_s = start

    def consider(self, logical_time_s: float, *, force: bool=False) -> LogicalEvidenceDecision:
        timestamp = float(logical_time_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError('logical_time_s must be finite and non-negative')
        if self._last_time_s is not None and timestamp < self._last_time_s - 1e-09:
            raise ValueError('logical evidence time must be monotonic')
        self._last_time_s = timestamp
        due = timestamp + 1e-09 >= self._next_tick_s
        emit = bool(force or due)
        if due:
            skipped = math.floor((timestamp - self._next_tick_s) / self.interval_s) + 1
            self._next_tick_s += max(1, skipped) * self.interval_s
        return LogicalEvidenceDecision(emit=emit, forced=bool(force), satisfies_periodic_tick=bool(due), logical_time_s=timestamp)
__all__ = ['BOUND_TARGET_CONFIRMATION_ACTIONS', 'CONFIRMABLE_ACTIONS', 'TRANSITION_IMAGE_ROLES', 'ConfirmationIntent', 'LogicalEvidenceDecision', 'LogicalEvidenceSampler', 'TransitionEvidence', 'confirmation_intent_for_action']
