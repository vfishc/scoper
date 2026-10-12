from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.behavior.brains.teachers.common import simple_place_plan, stop

class PutAwayTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: list[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        family = task_spec.get('family_spec') or {}
        put_away = family.get('put_away')
        if not isinstance(put_away, Mapping):
            raise BrainError('PutAwayTeacher requires family_spec.put_away.')
        target_roles = [str(role) for role in put_away.get('target_roles') or [] if role]
        storage_roles = [str(role) for role in put_away.get('storage_roles') or [] if role]
        if not target_roles or not storage_roles:
            raise BrainError('PutAwayTeacher requires target_roles and storage_roles.')
        storage_role = storage_roles[0]
        calls: list[ToolCall] = []
        for target_role in target_roles:
            calls.extend(simple_place_plan(env, target_role, storage_role, close_after=False))
        calls.extend(_final_storage_cleanup(env, storage_role, task_spec))
        calls.append(stop('put-away plan complete'))
        self._queue = calls

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return stop('put-away plan exhausted')

def _final_storage_cleanup(env: Any, storage_role: str, task_spec: Mapping[str, Any]) -> list[ToolCall]:
    from scoper.benchmark.behavior.brains.teachers.common import close_if_openable
    needs_closed = any((predicate.get('type') == 'object_closed' and (predicate.get('args') or {}).get('object') == storage_role for predicate in (task_spec.get('goal') or {}).get('predicates') or []))
    return close_if_openable(env, storage_role) if needs_closed else []
__all__ = ['PutAwayTeacher']
