from __future__ import annotations
from typing import Any, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class RecoverTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: List[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        recover = task_spec['family_spec']['recover']
        if 'incidents' in recover:
            self._queue = self._multi_incident_plan(env, task_spec, recover)
            return
        primary = recover['primary_task']
        incident = recover['incident']
        primary_plan = self._build_primary_plan(env, task_spec, primary)
        incident_plan = self._build_incident_plan(env, task_spec, incident)
        discovery_plan = _visual_discovery_probe_plan(recover)
        if incident.get('urgency') == 'immediate':
            calls = list(discovery_plan)
            calls.extend(incident_plan)
            calls.extend(primary_plan)
            calls.append(ToolCall('stop', {}, reasoning='recover immediate plan complete'))
        elif incident.get('urgency') == 'before_stop':
            calls = list(primary_plan)
            calls.extend(incident_plan)
            calls.append(ToolCall('stop', {}, reasoning='recover plan complete'))
        else:
            raise BrainError("Unknown recover incident urgency '{}'.".format(incident.get('urgency')))
        self._queue = calls

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='recover plan exhausted')

    def _multi_incident_plan(self, env: Any, task_spec: Mapping[str, Any], recover: Mapping[str, Any]) -> List[ToolCall]:
        calls: List[ToolCall] = _visual_discovery_probe_plan(recover)
        primary = recover.get('primary_task')
        if isinstance(primary, Mapping):
            calls.extend(self._build_primary_plan(env, task_spec, primary))
        incidents = list(recover.get('incidents') or [])
        for incident in incidents:
            if not isinstance(incident, Mapping):
                raise BrainError('recover.incidents items must be mappings.')
            calls.extend(self._build_incident_plan(env, task_spec, incident))
        calls.append(ToolCall('stop', {}, reasoning='recover multi-incident plan complete'))
        return calls

    def _build_primary_plan(self, env: Any, task_spec: Mapping[str, Any], primary: Mapping[str, Any]) -> List[ToolCall]:
        object_role = primary['object_role']
        receptacle_role = primary['receptacle_role']
        calls: List[ToolCall] = []
        _ensure_target_accessible(env, object_role, task_spec['roles'], calls)
        calls.extend([_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', receptacle_role)])
        if _role_is_openable(env, receptacle_role):
            calls.append(_call('open', receptacle_role))
        calls.append(_call('put_down', receptacle_role))
        if _role_is_openable(env, receptacle_role):
            calls.append(_call('close', receptacle_role))
        return calls

    def _build_incident_plan(self, env: Any, task_spec: Mapping[str, Any], incident: Mapping[str, Any]) -> List[ToolCall]:
        incident_type = incident['type']
        object_role = _incident_object_role(incident)
        if incident_type == 'faucet_on':
            if not _role_is_toggled_on(env, object_role):
                return []
            return [_call('navigate_to', object_role), _call('toggle_off', object_role)]
        if incident_type == 'open_container':
            return [_call('navigate_to', object_role), _call('close', object_role)]
        if incident_type == 'dirty_dish_on_floor_or_surface':
            return self._dirty_dish_plan(env, task_spec, incident)
        if incident_type == 'dirty_dish':
            return self._dirty_dish_recovery_plan(env, task_spec, incident)
        raise BrainError("Unknown recover incident type '{}'.".format(incident_type))

    def _dirty_dish_plan(self, env: Any, task_spec: Mapping[str, Any], incident: Mapping[str, Any]) -> List[ToolCall]:
        object_role = incident['object_role']
        support_roles = incident.get('support_roles') or {}
        sink_role = support_roles['sink']
        faucet_role = support_roles['faucet']
        storage_role = support_roles['storage']
        calls: List[ToolCall] = []
        _ensure_target_accessible(env, object_role, task_spec['roles'], calls)
        calls.extend([_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role), _call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', storage_role)])
        if _role_is_openable(env, storage_role):
            calls.append(_call('open', storage_role))
        calls.append(_call('put_down', storage_role))
        if _role_is_openable(env, storage_role):
            calls.append(_call('close', storage_role))
        return calls

    def _dirty_dish_recovery_plan(self, env: Any, task_spec: Mapping[str, Any], incident: Mapping[str, Any]) -> List[ToolCall]:
        object_role = _incident_object_role(incident)
        support_roles = incident.get('support_roles') or {}
        sink_role = support_roles.get('sink') or incident.get('sink_role') or incident.get('sink') or 'sink_basin'
        faucet_role = support_roles.get('faucet') or incident.get('faucet_role') or incident.get('faucet') or 'faucet'
        storage_role = support_roles.get('storage') or incident.get('storage_role')
        calls: List[ToolCall] = []
        _ensure_target_accessible(env, object_role, task_spec['roles'], calls)
        calls.extend([_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', str(sink_role)), _call('put_down', str(sink_role)), _call('navigate_to', str(faucet_role)), _call('toggle_on', str(faucet_role)), _call('toggle_off', str(faucet_role))])
        if storage_role:
            calls.extend([_call('navigate_to', object_role), _call('pick_up', object_role), _call('navigate_to', str(storage_role))])
            if _role_is_openable(env, str(storage_role)):
                calls.append(_call('open', str(storage_role)))
            calls.append(_call('put_down', str(storage_role)))
            if _role_is_openable(env, str(storage_role)):
                calls.append(_call('close', str(storage_role)))
        else:
            calls.extend([_call('navigate_to', object_role), _call('pick_up', object_role)])
        return calls

    def compute_reference_metrics(self, trace: List[Mapping[str, Any]]) -> dict:
        faucet_step = _first_tool_step(trace, 'toggle_off', incident_type='faucet_on')
        dirty_step = _first_tool_step(trace, 'pick_up', incident_type='dirty_dish')
        faucet_urgency = faucet_step is not None and dirty_step is not None and (faucet_step < dirty_step)
        return {'faucet_urgency_success': faucet_urgency, 'incidents_resolved_count': int(faucet_step is not None) + int(dirty_step is not None), 'total_incidents': 2}

def _call(tool_name: str, target: str) -> ToolCall:
    return ToolCall(tool_name, {'target': target})

def _visual_discovery_probe_plan(recover: Mapping[str, Any]) -> List[ToolCall]:
    incidents: List[Mapping[str, Any]] = []
    incident = recover.get('incident')
    if isinstance(incident, Mapping):
        incidents.append(incident)
    incidents.extend((item for item in recover.get('incidents') or [] if isinstance(item, Mapping)))
    visual_incident = next((item for item in incidents if isinstance(item.get('activation'), Mapping) and item['activation'].get('type') == 'visual_discovery'), None)
    if visual_incident is None:
        return []
    primary = recover.get('primary_task')
    anchor_role = primary.get('object_role') if isinstance(primary, Mapping) else None
    if not isinstance(anchor_role, str) or not anchor_role:
        anchor_role = next((_incident_object_role(item) for item in incidents if item is not visual_incident), None)
    if not isinstance(anchor_role, str) or not anchor_role:
        return []
    return [_call('navigate_to', anchor_role), ToolCall('scan', {})]

def _incident_object_role(incident: Mapping[str, Any]) -> str:
    role = incident.get('object_role') or incident.get('target_role') or incident.get('target')
    if not isinstance(role, str) or not role:
        raise BrainError('Recover incident is missing an object role.')
    return role

def _first_tool_step(trace: List[Mapping[str, Any]], tool_name: str, *, incident_type: str) -> Optional[int]:
    for index, step in enumerate(trace):
        if step.get('tool_name') == tool_name and step.get('incident_type') == incident_type:
            return index
    return None

def _role_object_id(env: Any, role_name: str) -> Optional[str]:
    return getattr(env, 'role_object_ids', {}).get(role_name)

def _object_by_role(env: Any, role_name: str) -> Mapping[str, Any]:
    object_id = _role_object_id(env, role_name)
    if object_id is None:
        return {}
    return env._objects_by_id().get(object_id, {})

def _role_is_openable(env: Any, role_name: str) -> bool:
    return bool(_object_by_role(env, role_name).get('openable', False))

def _role_is_toggled_on(env: Any, role_name: str) -> bool:
    return bool(_object_by_role(env, role_name).get('isToggled', False))

def _find_closed_parent_role(env: Any, object_role: str, roles: Mapping[str, Any]) -> Optional[str]:
    object_id = _role_object_id(env, object_role)
    if object_id is None:
        return None
    objects = env._objects_by_id()
    obj = objects.get(object_id, {})
    parents = obj.get('parentReceptacles') or obj.get('parent_receptacles') or []
    role_by_id = {object_id: role for role, object_id in env.role_object_ids.items()}
    for parent_id in parents:
        parent = objects.get(parent_id, {})
        if not (parent.get('openable') and parent.get('isOpen') is False):
            continue
        parent_role = role_by_id.get(parent_id)
        if parent_role in roles:
            return parent_role
        return parent_id
    return None

def _ensure_target_accessible(env: Any, object_role: str, roles: Mapping[str, Any], calls: List[ToolCall]) -> None:
    parent = _find_closed_parent_role(env, object_role, roles)
    if parent:
        calls.append(_call('navigate_to', parent))
        calls.append(_call('open', parent))
from scoper.benchmark.procthor.brains.teachers import get_dispatcher
get_dispatcher().register('recover_and_store_incident_object', RecoverTeacher)
get_dispatcher().register('recover_multi_incident', RecoverTeacher)
__all__ = ['RecoverTeacher']
