from __future__ import annotations
from typing import Any, List, Mapping, Optional
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class PrepareTeacher(BaseBrain):

    def __init__(self) -> None:
        self._env: Any = None
        self._task_spec: Mapping[str, Any] = {}
        self._queue: List[ToolCall] = []

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        self._task_spec = task_spec
        prepare = task_spec['family_spec']['prepare']
        stages = prepare.get('stages')
        if isinstance(stages, list) and stages:
            self._queue = self._multi_stage_plan(env, task_spec, prepare, stages)
            return
        subtemplate = prepare['subtemplate']
        if subtemplate == 'wash_then_store':
            self._queue = self._wash_then_store(env, task_spec)
        elif subtemplate == 'cool_then_place':
            self._queue = self._cool_then_place(env, task_spec)
        elif subtemplate == 'heat_then_place':
            self._queue = self._heat_then_place(env, task_spec)
        elif subtemplate == 'slice_then_place':
            self._queue = self._slice_then_place(env, task_spec)
        else:
            raise BrainError("Unknown prepare subtemplate '{}'.".format(subtemplate))

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='prepare plan complete')

    def _multi_stage_plan(self, env: Any, task_spec: Mapping[str, Any], prepare: Mapping[str, Any], stages: List[Mapping[str, Any]]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        current_target_ref = str(prepare.get('target_role') or 'target_object')
        for index, stage in enumerate(stages):
            stage_spec = dict(stage)
            stage_spec.setdefault('object_ref', current_target_ref)
            stage_type = stage_spec.get('type')
            is_final = index == len(stages) - 1
            if stage_type == 'wash':
                calls.extend(self._wash_stage(env, task_spec, prepare, stage_spec, is_final))
            elif stage_type == 'heat':
                calls.extend(self._heat_stage(env, prepare, stage_spec, is_final))
            elif stage_type == 'cool':
                calls.extend(self._cool_stage(env, prepare, stage_spec, is_final))
            elif stage_type == 'slice':
                calls.extend(self._slice_stage(env, task_spec, prepare, stage_spec, is_final))
                if not is_final:
                    current_target_ref = _prepare_target_role(prepare, stage_spec)
            elif stage_type in {'store', 'place'}:
                calls.extend(self._store_stage(env, prepare, stage_spec))
            else:
                raise BrainError("Unknown prepare stage type '{}'.".format(stage_type))
        cleanup_calls = _goal_cleanup_plan(env, task_spec)
        calls.extend(cleanup_calls)
        calls.append(ToolCall('stop', {}, reasoning='multi-stage prepare plan complete'))
        return calls

    def _wash_stage(self, env: Any, task_spec: Mapping[str, Any], prepare: Mapping[str, Any], stage: Mapping[str, Any], is_final: bool) -> List[ToolCall]:
        target_role = _prepare_target_role(prepare, stage)
        sink_role = _stage_receptacle_role(stage, 'sink_basin')
        faucet_role = stage.get('faucet_role') or _first_role(prepare.get('device_roles'), preferred='faucet', fallback='faucet')
        calls: List[ToolCall] = []
        _ensure_target_accessible(env, target_role, task_spec['roles'], calls)
        calls.extend([_call('navigate_to', target_role), _call('pick_up', target_role), _call('navigate_to', sink_role), _call('put_down', sink_role), _call('navigate_to', faucet_role), _call('toggle_on', faucet_role), _call('toggle_off', faucet_role)])
        if not is_final:
            calls.extend([_call('navigate_to', target_role), _call('pick_up', target_role)])
        return calls

    def _heat_stage(self, env: Any, prepare: Mapping[str, Any], stage: Mapping[str, Any], is_final: bool) -> List[ToolCall]:
        target_role = _prepare_target_role(prepare, stage)
        heater_role = _stage_receptacle_role(stage, 'microwave')
        calls = [_call('navigate_to', heater_role), _call('open', heater_role), _call('put_down', heater_role), _call('close', heater_role), _call('toggle_on', heater_role), _call('toggle_off', heater_role)]
        if not is_final:
            calls.extend([_call('open', heater_role), _call('navigate_to', target_role), _call('pick_up', target_role), _call('close', heater_role)])
        return calls

    def _cool_stage(self, env: Any, prepare: Mapping[str, Any], stage: Mapping[str, Any], is_final: bool) -> List[ToolCall]:
        target_ref = stage.get('object_ref') or _prepare_target_role(prepare, stage)
        fridge_role = _stage_receptacle_role(stage, 'fridge')
        calls = [_call('navigate_to', fridge_role), _call('open', fridge_role), _call('put_down', fridge_role), _call('close', fridge_role)]
        if not is_final:
            calls.extend([_call('open', fridge_role), _call('navigate_to', target_ref), _call('pick_up', target_ref), _call('close', fridge_role)])
        return calls

    def _slice_stage(self, env: Any, task_spec: Mapping[str, Any], prepare: Mapping[str, Any], stage: Mapping[str, Any], is_final: bool) -> List[ToolCall]:
        target_role = _prepare_target_role(prepare, stage)
        knife_role = stage.get('tool_role') or _first_role(prepare.get('tool_roles'), preferred='knife', fallback='knife')
        knife_drop_target = stage.get('drop_role') or stage.get('receptacle_role') or _preferred_drop_target_for_role(env, knife_role, excluded_roles={target_role, _stage_receptacle_role(stage, 'target_receptacle')})
        calls: List[ToolCall] = [_call('navigate_to', knife_role), _call('pick_up', knife_role)]
        _ensure_target_accessible(env, target_role, task_spec['roles'], calls)
        calls.extend([_call('navigate_to', target_role), _call('slice', target_role), _call('navigate_to', knife_drop_target), _call('put_down', knife_drop_target)])
        if not is_final:
            calls.extend([_call('navigate_to', target_role), _call('pick_up', target_role)])
        return calls

    def _store_stage(self, env: Any, prepare: Mapping[str, Any], stage: Mapping[str, Any]) -> List[ToolCall]:
        receptacle_role = stage.get('receptacle_role') or prepare.get('final_receptacle_role') or 'target_receptacle'
        calls: List[ToolCall] = [_call('navigate_to', receptacle_role)]
        if _role_is_openable(env, receptacle_role):
            calls.append(_call('open', receptacle_role))
        calls.append(_call('put_down', receptacle_role))
        if _role_is_openable(env, receptacle_role):
            calls.append(_call('close', receptacle_role))
        return calls

    def _wash_then_store(self, env: Any, task_spec: Mapping[str, Any]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        roles = task_spec['roles']
        _ensure_target_accessible(env, 'target_object', roles, calls)
        calls.extend([_call('navigate_to', 'target_object'), _call('pick_up', 'target_object'), _call('navigate_to', 'sink_basin'), _call('put_down', 'sink_basin'), _call('navigate_to', 'faucet'), _call('toggle_on', 'faucet'), _call('toggle_off', 'faucet'), _call('navigate_to', 'target_object'), _call('pick_up', 'target_object'), _call('navigate_to', 'target_receptacle')])
        if _role_is_openable(env, 'target_receptacle'):
            calls.append(_call('open', 'target_receptacle'))
        calls.append(_call('put_down', 'target_receptacle'))
        if _role_is_openable(env, 'target_receptacle'):
            calls.append(_call('close', 'target_receptacle'))
        cleanup_calls = _goal_cleanup_plan(env, task_spec)
        calls.extend(cleanup_calls)
        calls.append(ToolCall('stop', {}, reasoning='prepare plan complete'))
        return calls

    def _cool_then_place(self, env: Any, task_spec: Mapping[str, Any]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        roles = task_spec['roles']
        _ensure_target_accessible(env, 'target_object', roles, calls)
        calls.extend([_call('navigate_to', 'target_object'), _call('pick_up', 'target_object'), _call('navigate_to', 'fridge'), _call('open', 'fridge'), _call('put_down', 'fridge'), _call('close', 'fridge'), _call('navigate_to', 'fridge'), _call('open', 'fridge'), _call('navigate_to', 'target_object'), _call('pick_up', 'target_object'), _call('close', 'fridge'), _call('navigate_to', 'target_receptacle')])
        if _role_is_openable(env, 'target_receptacle'):
            calls.append(_call('open', 'target_receptacle'))
        calls.append(_call('put_down', 'target_receptacle'))
        if _role_is_openable(env, 'target_receptacle'):
            calls.append(_call('close', 'target_receptacle'))
        cleanup_calls = _goal_cleanup_plan(env, task_spec)
        calls.extend(cleanup_calls)
        calls.append(ToolCall('stop', {}, reasoning='prepare plan complete'))
        return calls

    def _heat_then_place(self, env: Any, task_spec: Mapping[str, Any]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        roles = task_spec['roles']
        _ensure_target_accessible(env, 'target_object', roles, calls)
        calls.extend([_call('navigate_to', 'target_object'), _call('pick_up', 'target_object'), _call('navigate_to', 'microwave'), _call('open', 'microwave'), _call('put_down', 'microwave'), _call('close', 'microwave'), _call('toggle_on', 'microwave'), _call('toggle_off', 'microwave'), _call('open', 'microwave'), _call('navigate_to', 'target_object'), _call('pick_up', 'target_object'), _call('close', 'microwave'), _call('navigate_to', 'target_receptacle')])
        if _role_is_openable(env, 'target_receptacle'):
            calls.append(_call('open', 'target_receptacle'))
        calls.append(_call('put_down', 'target_receptacle'))
        if _role_is_openable(env, 'target_receptacle'):
            calls.append(_call('close', 'target_receptacle'))
        cleanup_calls = _goal_cleanup_plan(env, task_spec)
        calls.extend(cleanup_calls)
        calls.append(ToolCall('stop', {}, reasoning='prepare plan complete'))
        return calls

    def _slice_then_place(self, env: Any, task_spec: Mapping[str, Any]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        roles = task_spec['roles']
        knife_drop_target = _preferred_drop_target_for_role(env, 'knife', excluded_roles={'target_object', 'target_receptacle'})
        calls.extend([_call('navigate_to', 'knife'), _call('pick_up', 'knife')])
        _ensure_target_accessible(env, 'target_object', roles, calls)
        calls.extend([_call('navigate_to', 'target_object'), _call('slice', 'target_object'), _call('navigate_to', knife_drop_target), _call('put_down', knife_drop_target), _call('navigate_to', 'target_object'), _call('pick_up', 'target_object'), _call('navigate_to', 'target_receptacle')])
        if _role_is_openable(env, 'target_receptacle'):
            calls.append(_call('open', 'target_receptacle'))
        calls.append(_call('put_down', 'target_receptacle'))
        if _role_is_openable(env, 'target_receptacle'):
            calls.append(_call('close', 'target_receptacle'))
        cleanup_calls = _goal_cleanup_plan(env, task_spec)
        calls.extend(cleanup_calls)
        calls.append(ToolCall('stop', {}, reasoning='prepare plan complete'))
        return calls

def _goal_cleanup_plan(env: Any, task_spec: Mapping[str, Any]) -> List[ToolCall]:
    calls: List[ToolCall] = []
    goal_predicates = task_spec.get('goal', {}).get('predicates', [])
    for predicate in goal_predicates:
        ptype = predicate.get('type')
        args = predicate.get('args', {})
        if ptype == 'object_closed':
            role = args.get('object')
            if role:
                calls.append(_call('navigate_to', role))
                calls.append(_call('close', role))
        elif ptype == 'object_open':
            role = args.get('object')
            if role:
                calls.append(_call('navigate_to', role))
                calls.append(_call('open', role))
        elif ptype == 'object_toggled_off':
            role = args.get('object')
            if role:
                calls.append(_call('navigate_to', role))
                calls.append(_call('toggle_off', role))
        elif ptype == 'object_toggled_on':
            role = args.get('object')
            if role:
                calls.append(_call('navigate_to', role))
                calls.append(_call('toggle_on', role))
    return calls

def _call(tool_name: str, target: str) -> ToolCall:
    return ToolCall(tool_name, {'target': target})

def _prepare_target_role(prepare: Mapping[str, Any], stage: Mapping[str, Any]) -> str:
    return str(stage.get('target_role') or prepare.get('target_role') or 'target_object')

def _stage_receptacle_role(stage: Mapping[str, Any], fallback: str) -> str:
    return str(stage.get('receptacle_role') or fallback)

def _first_role(roles: Any, *, preferred: str, fallback: str) -> str:
    if isinstance(roles, list):
        for role in roles:
            if role == preferred:
                return str(role)
        for role in roles:
            if isinstance(role, str) and role:
                return role
    return fallback

def _role_object_id(env: Any, role_name: str) -> Optional[str]:
    return getattr(env, 'role_object_ids', {}).get(role_name)

def _object_by_role(env: Any, role_name: str) -> Mapping[str, Any]:
    object_id = _role_object_id(env, role_name)
    if object_id is None:
        return {}
    return env._objects_by_id().get(object_id, {})

def _role_is_openable(env: Any, role_name: str) -> bool:
    return bool(_object_by_role(env, role_name).get('openable', False))

def _find_closed_parent_role(env: Any, object_role: str, roles: Mapping[str, Any]) -> Optional[str]:
    object_id = _role_object_id(env, object_role)
    if object_id is None:
        return None
    obj = env._objects_by_id().get(object_id, {})
    parents = obj.get('parentReceptacles') or obj.get('parent_receptacles') or []
    if not parents:
        return None
    role_by_id = {object_id: role for role, object_id in env.role_object_ids.items()}
    for parent_id in parents:
        parent_obj = env._objects_by_id().get(parent_id, {})
        if not (parent_obj.get('openable') and parent_obj.get('isOpen') is False):
            continue
        parent_role = role_by_id.get(parent_id)
        if parent_role in roles:
            return parent_role
    return None

def _ensure_target_accessible(env: Any, object_role: str, roles: Mapping[str, Any], calls: List[ToolCall]) -> None:
    parent_role = _find_closed_parent_role(env, object_role, roles)
    if parent_role:
        calls.append(_call('navigate_to', parent_role))
        calls.append(_call('open', parent_role))

def _preferred_drop_target_for_role(env: Any, role_name: str, excluded_roles: set[str]) -> str:
    obj = _object_by_role(env, role_name)
    parent_ids = obj.get('parentReceptacles') or obj.get('parent_receptacles') or []
    role_by_id = {object_id: role for role, object_id in env.role_object_ids.items()}
    for parent_id in parent_ids:
        parent_role = role_by_id.get(parent_id)
        if parent_role and parent_role not in excluded_roles:
            return parent_role
    objects = env._objects_by_id()
    preferred_role = 'countertop'
    preferred_id = env.role_object_ids.get(preferred_role)
    preferred = objects.get(preferred_id, {}) if preferred_id else {}
    if preferred_role not in excluded_roles and preferred_role != role_name and preferred.get('receptacle') and (not preferred.get('pickupable')):
        return preferred_role
    for candidate_role, object_id in sorted(env.role_object_ids.items()):
        if candidate_role in excluded_roles or candidate_role == role_name:
            continue
        candidate = objects.get(object_id, {})
        if candidate.get('receptacle') and (not candidate.get('pickupable')):
            return candidate_role
    return 'target_receptacle'
from scoper.benchmark.procthor.brains.teachers import get_dispatcher
get_dispatcher().register('prepare_then_place', PrepareTeacher)
get_dispatcher().register('prepare_multi_stage', PrepareTeacher)
__all__ = ['PrepareTeacher']
