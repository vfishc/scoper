from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.behavior.brains.teachers.common import plan_predicate, stop, wash_then_place_plan

class TidyTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: list[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        tidy = (task_spec.get('family_spec') or {}).get('tidy')
        if not isinstance(tidy, Mapping):
            raise BrainError('TidyTeacher requires family_spec.tidy.')
        violations = tidy.get('violations')
        calls: list[ToolCall] = []
        if isinstance(violations, list):
            ordered = sorted([item for item in violations if isinstance(item, Mapping)], key=lambda item: 0 if str(item.get('urgency')) in {'immediate', 'high'} else 1)
            for violation in ordered:
                calls.extend(_violation_plan(env, violation))
        else:
            for predicate in (task_spec.get('goal') or {}).get('predicates') or []:
                calls.extend(plan_predicate(env, predicate))
        calls.append(stop('tidy plan complete'))
        self._queue = calls

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return stop('tidy plan exhausted')

def _violation_plan(env: Any, violation: Mapping[str, Any]) -> list[ToolCall]:
    from scoper.benchmark.behavior.brains.teachers.common import call
    vtype = str(violation.get('type') or '')
    object_role = str(violation.get('object_role') or '')
    if not object_role:
        raise BrainError('Tidy violation is missing object_role.')
    if vtype == 'faucet_on':
        return [call('navigate_to', object_role), call('toggle_off', object_role)]
    if vtype == 'open_container':
        return [call('navigate_to', object_role), call('close', object_role)]
    if vtype == 'dirty_dish':
        support = violation.get('support_roles') or {}
        storage = support.get('storage')
        if storage:
            return wash_then_place_plan(env, object_role, str(storage), faucet_role=support.get('faucet'), close_after=True)
        return plan_predicate(env, {'type': 'object_clean', 'args': {'object': object_role, 'source': support.get('faucet')}})
    raise BrainError("Unknown tidy violation type '{}'.".format(vtype))
__all__ = ['TidyTeacher']
