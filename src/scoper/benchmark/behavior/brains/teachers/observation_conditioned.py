from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.behavior.brains.teachers.common import call, predicates_to_plan, simple_place_plan, stop, wash_then_place_plan

class ObservationConditionedTeacher(BaseBrain):

    def __init__(self) -> None:
        self._rooms: list[str] = []
        self._entered_rooms: set[str] = set()
        self._scanned_rooms: set[str] = set()
        self._planned_issue_ids: set[str] = set()
        self._repair_queue: list[ToolCall] = []
        self._env: Any | None = None
        self._task_spec: Mapping[str, Any] | None = None

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        obs = _observation_conditioned_spec(task_spec)
        rooms = [str(room) for room in obs.get('rooms') or [] if room]
        if not rooms:
            rooms = sorted({str(item.get('room_id')) for item in obs.get('latent_payloads') or [] if item.get('room_id')})
        if not rooms:
            raise BrainError('ObservationConditionedTeacher requires rooms or latent payload room_ids.')
        self._env = env
        self._task_spec = task_spec
        self._rooms = rooms
        self._entered_rooms = set()
        self._scanned_rooms = set()
        self._planned_issue_ids = set()
        self._repair_queue = []

    def act(self, brain_input: BrainInput) -> ToolCall:
        context = brain_input.prompt_context or {}
        last = brain_input.last_transition or {}
        self._update_state_from_transition(last)
        self._enqueue_repairs_from_context(context)
        if self._repair_queue:
            return self._repair_queue.pop(0)
        for room in self._rooms:
            if room not in self._entered_rooms:
                return ToolCall('navigate_to_room', {'room': room}, reasoning='enter inspection room before observing')
            if room not in self._scanned_rooms:
                return ToolCall('scan', {'num_views': 4}, reasoning='scan room before choosing repair actions')
        return stop('observation-conditioned inspection complete')

    def _update_state_from_transition(self, transition: Mapping[str, Any]) -> None:
        if not transition.get('success'):
            return
        tool_name = transition.get('tool_name')
        if tool_name == 'navigate_to_room':
            room = (transition.get('state_delta') or {}).get('navigate_to_room') or {}
            room_id = room.get('room_id') or (transition.get('arguments') or {}).get('room')
            if room_id:
                self._entered_rooms.add(str(room_id))
        elif tool_name == 'scan':
            scan = (transition.get('state_delta') or {}).get('scan') or {}
            room_id = scan.get('current_room') or scan.get('room_instance')
            if room_id:
                self._scanned_rooms.add(str(room_id))

    def _enqueue_repairs_from_context(self, context: Mapping[str, Any]) -> None:
        env = getattr(self, '_env', None)
        task_spec = getattr(self, '_task_spec', None)
        if env is None or task_spec is None:
            return
        payloads = _payloads_by_issue_id(task_spec)
        getter = getattr(env, 'observation_conditioned_visible_issues', None)
        visible_issues = getter() if callable(getter) else context.get('visible_issues') or []
        for issue in visible_issues:
            issue_id = str(issue.get('issue_id') or '')
            if not issue_id or issue_id in self._planned_issue_ids:
                continue
            payload = payloads.get(issue_id)
            if payload is None:
                continue
            calls = _repair_calls_for_payload(env, payload)
            self._repair_queue.extend(calls)
            self._planned_issue_ids.add(issue_id)

def _repair_calls_for_payload(env: Any, payload: Mapping[str, Any]) -> list[ToolCall]:
    template = str(payload.get('repair_plan_template') or '')
    object_role = str(payload.get('object_role') or '')
    support_roles = payload.get('support_roles') if isinstance(payload.get('support_roles'), Mapping) else {}
    if template == 'toggle_off' and object_role:
        return [call('navigate_to', object_role), call('toggle_off', object_role)]
    if template == 'close' and object_role:
        return [call('navigate_to', object_role), call('close', object_role)]
    if template == 'wash_then_place' and object_role:
        faucet_role = support_roles.get('faucet') if isinstance(support_roles, Mapping) else None
        storage_role = support_roles.get('storage') if isinstance(support_roles, Mapping) else None
        if storage_role:
            return wash_then_place_plan(env, object_role, str(storage_role), faucet_role=str(faucet_role) if faucet_role else None, close_after=False)
    if template == 'place' and object_role:
        storage_role = support_roles.get('storage') if isinstance(support_roles, Mapping) else None
        if storage_role:
            return simple_place_plan(env, object_role, str(storage_role), close_after=True)
    predicates = payload.get('goal_predicates') or []
    return predicates_to_plan(env, predicates)

def _observation_conditioned_spec(task_spec: Mapping[str, Any]) -> Mapping[str, Any]:
    family = task_spec.get('family_spec') or {}
    obs = family.get('observation_conditioned') if isinstance(family, Mapping) else None
    if not isinstance(obs, Mapping):
        raise BrainError('ObservationConditionedTeacher requires family_spec.observation_conditioned.')
    return obs

def _payloads_by_issue_id(task_spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    obs = _observation_conditioned_spec(task_spec)
    out: dict[str, Mapping[str, Any]] = {}
    for payload in obs.get('latent_payloads') or []:
        if not isinstance(payload, Mapping):
            continue
        issue_id = str(payload.get('latent_id') or payload.get('issue_id') or '')
        if issue_id:
            out[issue_id] = payload
    return out
__all__ = ['ObservationConditionedTeacher']
