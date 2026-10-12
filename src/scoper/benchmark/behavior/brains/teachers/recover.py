from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall
from scoper.benchmark.behavior.brains.teachers.common import call, simple_place_plan, stop, wash_then_place_plan, wash_plan

class RecoverTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: list[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        family = task_spec.get('family_spec') or {}
        recover = family.get('recover')
        if not isinstance(recover, Mapping):
            raise BrainError('RecoverTeacher requires family_spec.recover.')
        primary = recover.get('primary_task')
        incidents = []
        if isinstance(recover.get('incident'), Mapping):
            incidents.append(recover['incident'])
        incidents.extend((item for item in recover.get('incidents') or [] if isinstance(item, Mapping)))
        immediate = [item for item in incidents if str(item.get('urgency')) in {'immediate', 'high'}]
        later = [item for item in incidents if item not in immediate]
        calls = _visual_discovery_probe_plan(recover, incidents)
        for incident in immediate:
            calls.extend(_incident_plan(env, incident))
        if isinstance(primary, Mapping):
            calls.extend(_primary_plan(env, primary))
        for incident in later:
            calls.extend(_incident_plan(env, incident))
        calls.append(stop('recover plan complete'))
        self._queue = calls

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return stop('recover plan exhausted')

def _primary_plan(env: Any, primary: Mapping[str, Any]) -> list[ToolCall]:
    object_role = str(primary['object_role'])
    receptacle_role = str(primary['receptacle_role'])
    return simple_place_plan(env, object_role, receptacle_role, close_after=True)

def _incident_plan(env: Any, incident: Mapping[str, Any]) -> list[ToolCall]:
    incident_type = str(incident.get('type') or '')
    object_role = str(incident.get('object_role') or incident.get('target_role') or incident.get('target') or '')
    if not object_role:
        raise BrainError('Recover incident is missing object_role.')
    if incident_type == 'faucet_on':
        return [call('navigate_to', object_role), call('toggle_off', object_role)]
    if incident_type == 'open_container':
        return [call('navigate_to', object_role), call('close', object_role)]
    if incident_type in {'dirty_dish', 'dirty_dish_on_floor_or_surface'}:
        support = incident.get('support_roles') or {}
        faucet = support.get('faucet') or incident.get('faucet_role')
        storage = support.get('storage') or incident.get('storage_role')
        if storage:
            return wash_then_place_plan(env, object_role, str(storage), faucet_role=faucet, close_after=True)
        return wash_plan(env, object_role, faucet_role=faucet, keep_held=False)
    raise BrainError("Unknown recover incident type '{}'.".format(incident_type))

def _visual_discovery_probe_plan(recover: Mapping[str, Any], incidents: list[Mapping[str, Any]]) -> list[ToolCall]:
    visual_incident = next((incident for incident in incidents if isinstance(incident.get('activation'), Mapping) and incident['activation'].get('type') == 'visual_discovery'), None)
    if visual_incident is None:
        return []
    primary = recover.get('primary_task')
    anchor_role = primary.get('object_role') if isinstance(primary, Mapping) else None
    if not isinstance(anchor_role, str) or not anchor_role:
        anchor_role = next((str(incident.get('object_role') or incident.get('target_role') or incident.get('target') or '') for incident in incidents if incident is not visual_incident), '')
    if not anchor_role:
        return []
    return [call('navigate_to', anchor_role), call('scan')]
__all__ = ['RecoverTeacher']
