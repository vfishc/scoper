from __future__ import annotations
import math
from typing import Any, Mapping, Optional, Tuple
from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.execution_core.object_placement import aabb_bottom_z, aabb_top_z
from scoper.benchmark.behavior.bddl_adapter import DEFAULT_SUBSTANCE
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainInput, ToolCall
from scoper.benchmark.behavior.goal_checker import DEFAULT_AGENT_NEAR_THRESHOLD, ONTOP_Z_TOLERANCE_M
from scoper.benchmark.behavior.goal_predicates import normalize_goal_predicate
_MAX_SAME_CALL = 2
_MAX_FAILED_CALLS = 2
_MAX_TOTAL_ACTS = 40

class RecoveryOracleBrain(BaseBrain):

    def __init__(self) -> None:
        self._task_spec: Mapping[str, Any] | None = None
        self._env: Any = None
        self._acts = 0
        self._last_call_signature: tuple[str, str, str] | None = None
        self._same_call_count = 0
        self._failed_call_counts: dict[tuple[str, str, str], int] = {}
        self._last_observed_transition_id: int | None = None
        self._last_successful_nav_target: str | None = None

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        self._task_spec = task_spec
        self._acts = 0
        self._last_call_signature = None
        self._same_call_count = 0
        self._failed_call_counts = {}
        self._last_observed_transition_id = None
        self._last_successful_nav_target = None

    def act(self, brain_input: BrainInput) -> ToolCall:
        self._remember_failed_transition(brain_input.last_transition)
        self._acts += 1
        if self._acts > _MAX_TOTAL_ACTS:
            return ToolCall('stop', {}, reasoning='recovery oracle: act budget exhausted')
        if self._env is None or self._task_spec is None:
            return ToolCall('stop', {}, reasoning='recovery oracle: no task bound')
        goal = self._env.check_goal(model_stopped=getattr(self._env, 'model_stopped', False))
        if float(goal.get('predicate_success', 0.0)) >= 1.0:
            cleanup = self._plan_cleanup()
            if cleanup is not None:
                return self._track_and_return(cleanup)
        if float(goal.get('task_success', 0.0)) >= 1.0:
            return ToolCall('stop', {}, reasoning='goal satisfied')
        for predicate in (self._task_spec.get('goal') or {}).get('predicates') or []:
            call = self._plan_for_predicate(predicate)
            if call is not None:
                return self._track_and_return(call)
        return self._track_and_return(ToolCall('stop', {}, reasoning='no remaining recovery action'))

    def _remember_failed_transition(self, transition: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(transition, Mapping):
            return
        marker = id(transition)
        if marker == self._last_observed_transition_id:
            return
        self._last_observed_transition_id = marker
        signature = _call_signature(str(transition.get('tool_name', '')), transition.get('arguments') if isinstance(transition.get('arguments'), Mapping) else {})
        if not transition.get('success', True):
            self._failed_call_counts[signature] = self._failed_call_counts.get(signature, 0) + 1
        else:
            self._failed_call_counts.pop(signature, None)
            if str(transition.get('tool_name', '')) == 'navigate_to':
                args = transition.get('arguments') if isinstance(transition.get('arguments'), Mapping) else {}
                self._last_successful_nav_target = str(args.get('target', '')) or None

    def _track_and_return(self, call: ToolCall) -> ToolCall:
        signature = _call_signature(call.tool_name, call.arguments)
        failed_count = self._failed_call_counts.get(signature, 0)
        if failed_count >= _MAX_FAILED_CALLS:
            return ToolCall('stop', {}, reasoning='recovery oracle: {}({}) failed {} times'.format(signature[0], signature[1] or signature[2], failed_count))
        if signature == self._last_call_signature:
            self._same_call_count += 1
        else:
            self._last_call_signature = signature
            self._same_call_count = 1
        if self._same_call_count > _MAX_SAME_CALL:
            return ToolCall('stop', {}, reasoning='recovery oracle: repeated {}({}) too often'.format(signature[0], signature[1] or signature[2]))
        return call

    def _plan_for_predicate(self, predicate: Mapping[str, Any]) -> ToolCall | None:
        record = normalize_goal_predicate(predicate)
        ptype = record['type']
        if ptype == 'object_in_receptacle':
            return self._plan_placement(record, relation='inside')
        if ptype == 'object_on_receptacle':
            return self._plan_placement(record, relation='on_top')
        if ptype in {'object_toggled_on', 'object_toggled_off'}:
            return self._plan_toggle(record)
        if ptype in {'object_open', 'object_closed'}:
            return self._plan_open_close(record)
        if ptype == 'object_sliced':
            return self._plan_sliced(record)
        if ptype == 'object_clean':
            return self._plan_clean(record)
        if ptype in {'object_cooked', 'object_hot'}:
            return self._plan_hot(record)
        if ptype == 'object_cool':
            return self._plan_cool(record)
        if ptype == 'object_filled':
            return self._plan_filled(record)
        if ptype == 'object_soaked':
            return self._plan_soaked(record)
        if ptype == 'object_held':
            return self._plan_held(record)
        if ptype == 'object_not_held':
            return self._plan_not_held(record)
        if ptype == 'agent_near':
            role = _object_role(record)
            obj = _object_for_role(self._env, role)
            if role is None or obj is None or self._agent_close_to(obj):
                return None
            return ToolCall('navigate_to', {'target': role}, reasoning='move near {}'.format(role))
        return ToolCall('stop', {}, reasoning="unsupported recovery predicate '{}'".format(ptype))

    def _plan_placement(self, record: Mapping[str, Any], *, relation: str) -> ToolCall | None:
        env = self._env
        object_role = _object_role(record)
        receptacle_role = _receptacle_role(record)
        if object_role is None or receptacle_role is None:
            return ToolCall('stop', {}, reasoning='placement predicate missing object/receptacle role')
        obj_name = _object_name(env, object_role)
        receptacle = _object_for_role(env, receptacle_role)
        obj = env.core.objects.get(obj_name) if obj_name else None
        if obj is None or receptacle is None:
            return ToolCall('stop', {}, reasoning='placement predicate references missing object')
        held = _held_name(env)
        relation_satisfied = _relation_holds(obj, receptacle, relation) and held != obj_name
        if relation_satisfied:
            return None
        if held == obj_name:
            if _needs_open(receptacle):
                return self._nav_then('open', receptacle_role, receptacle)
            return self._nav_then('put_down', receptacle_role, receptacle)
        if held and held != obj_name:
            safe_role, safe_obj = self._safe_drop_target(exclude_role=receptacle_role)
            if safe_role is not None and safe_obj is not None:
                return self._nav_then('put_down', safe_role, safe_obj)
            return ToolCall('drop', {}, reasoning='free hand before recovering {}'.format(object_role))
        container_role, container_obj = self._closed_container_holding(obj)
        if container_role is not None and container_obj is not None:
            return self._nav_then('open', container_role, container_obj)
        return self._nav_then('pick_up', object_role, obj)

    def _plan_toggle(self, record: Mapping[str, Any]) -> ToolCall | None:
        role = _object_role(record)
        obj = _object_for_role(self._env, role)
        if role is None or obj is None:
            return ToolCall('stop', {}, reasoning='toggle predicate references missing object')
        desired = record['type'] == 'object_toggled_on'
        if bool(actions.get_toggle_state(obj)) == desired:
            return None
        return self._nav_then('toggle_on' if desired else 'toggle_off', role, obj)

    def _plan_open_close(self, record: Mapping[str, Any]) -> ToolCall | None:
        role = _object_role(record)
        obj = _object_for_role(self._env, role)
        if role is None or obj is None:
            return ToolCall('stop', {}, reasoning='open/close predicate references missing object')
        desired_open = record['type'] == 'object_open'
        if bool(actions.get_open_state(obj)) == desired_open:
            return None
        return self._nav_then('open' if desired_open else 'close', role, obj)

    def _plan_sliced(self, record: Mapping[str, Any]) -> ToolCall | None:
        env = self._env
        role = _object_role(record)
        obj = _object_for_role(env, role)
        if role is None or obj is None:
            return ToolCall('stop', {}, reasoning='sliced predicate references missing object')
        if bool(actions.get_sliced_state(obj)):
            return None
        tool_role = str((record.get('args') or {}).get('tool') or self._role_with_affordance(actions.supports_cut_tool) or 'knife')
        tool_obj = _object_for_role(env, tool_role)
        tool_name = _object_name(env, tool_role)
        held = _held_name(env)
        if held != tool_name:
            if held is not None:
                return ToolCall('drop', {}, reasoning='free hand before picking cutting tool')
            if tool_obj is None:
                return ToolCall('stop', {}, reasoning="missing cutting tool role '{}'".format(tool_role))
            return self._nav_then('pick_up', tool_role, tool_obj)
        return self._nav_then('cut', role, obj)

    def _plan_clean(self, record: Mapping[str, Any]) -> ToolCall | None:
        env = self._env
        role = _object_role(record)
        obj = _object_for_role(env, role)
        if role is None or obj is None:
            return ToolCall('stop', {}, reasoning='clean predicate references missing object')
        source_role = self._cleaning_fluid_source_role(record, role)
        source_obj = _object_for_role(env, source_role)
        if actions.get_dirty_state(obj) is not True:
            if source_role is not None and source_obj is not None and bool(actions.get_toggle_state(source_obj)) and (not _goal_requires_toggle_on(self._task_spec or {}, source_role)):
                return self._nav_then('toggle_off', source_role, source_obj)
            return None
        has_explicit_wipe_tool = bool((record.get('args') or {}).get('tool'))
        if not has_explicit_wipe_tool and source_role is not None and (source_obj is not None) and actions.supports_soakable(obj):
            return self._plan_fluid_clean(role, obj, source_role, source_obj)
        tool_role = str((record.get('args') or {}).get('tool') or self._role_with_affordance(actions.supports_wipe_tool) or 'rag')
        tool_obj = _object_for_role(env, tool_role)
        tool_name = _object_name(env, tool_role)
        held = _held_name(env)
        if held != tool_name:
            if held is not None:
                return ToolCall('drop', {}, reasoning='free hand before picking cleaning tool')
            if tool_obj is None:
                return ToolCall('stop', {}, reasoning="missing cleaning tool role '{}'".format(tool_role))
            return self._nav_then('pick_up', tool_role, tool_obj)
        return self._nav_then('wipe', role, obj)

    def _plan_hot(self, record: Mapping[str, Any]) -> ToolCall | None:
        env = self._env
        role = _object_role(record)
        obj = _object_for_role(env, role)
        if role is None or obj is None:
            return ToolCall('stop', {}, reasoning='hot/cooked predicate references missing object')
        if bool(actions.get_cooked_state(obj)):
            return None
        source_role = str((record.get('args') or {}).get('heatsource') or self._role_with_affordance(actions.supports_heatsource) or 'heatsource')
        source_obj = _object_for_role(env, source_role)
        if source_obj is not None and _needs_open(source_obj):
            return self._nav_then('open', source_role, source_obj)
        return self._plan_held_object_wrapper(role, obj, source_role, 'heat')

    def _plan_cool(self, record: Mapping[str, Any]) -> ToolCall | None:
        env = self._env
        role = _object_role(record)
        obj = _object_for_role(env, role)
        if role is None or obj is None:
            return ToolCall('stop', {}, reasoning='cool predicate references missing object')
        if bool(actions.get_cooled_state(obj)):
            return None
        source_role = str((record.get('args') or {}).get('coldsource') or self._role_with_affordance(actions.supports_coldsource) or 'coldsource')
        source_obj = _object_for_role(env, source_role)
        if source_obj is not None and _needs_open(source_obj):
            return self._nav_then('open', source_role, source_obj)
        return self._plan_held_object_wrapper(role, obj, source_role, 'cool')

    def _plan_filled(self, record: Mapping[str, Any]) -> ToolCall | None:
        env = self._env
        role = _object_role(record)
        obj = _object_for_role(env, role)
        if role is None or obj is None:
            return ToolCall('stop', {}, reasoning='filled predicate references missing object')
        substance = _wanted_substance(record)
        if substance in set(actions.get_substances(obj, actions.FILLED_STATE_NAME)):
            return None
        source_role = str((record.get('args') or {}).get('source') or self._role_with_affordance(actions.supports_fluid_source) or 'source')
        source_obj = _object_for_role(env, source_role)
        held = _held_name(env)
        obj_name = _object_name(env, role)
        if held != obj_name:
            if held is not None:
                return ToolCall('drop', {}, reasoning='free hand before picking fillable object')
            return self._nav_then('pick_up', role, obj)
        if source_obj is None:
            return ToolCall('stop', {}, reasoning="missing fill source role '{}'".format(source_role))
        if _needs_toggle_on(source_obj):
            return self._nav_then('toggle_on', source_role, source_obj)
        if not self._agent_close_to(source_obj):
            return ToolCall('navigate_to', {'target': source_role}, reasoning='move to fill source')
        return ToolCall('fill', {'target': role, 'source': source_role}, reasoning='fill {}'.format(role))

    def _plan_soaked(self, record: Mapping[str, Any]) -> ToolCall | None:
        env = self._env
        role = _object_role(record)
        obj = _object_for_role(env, role)
        if role is None or obj is None:
            return ToolCall('stop', {}, reasoning='soaked predicate references missing object')
        substance = _wanted_substance(record)
        if substance in set(actions.get_substances(obj, actions.SATURATED_STATE_NAME)):
            return None
        source_role = str((record.get('args') or {}).get('source') or self._role_with_affordance(actions.supports_fluid_source) or 'source')
        source_obj = _object_for_role(env, source_role)
        held = _held_name(env)
        obj_name = _object_name(env, role)
        if held != obj_name:
            if held is not None:
                return ToolCall('drop', {}, reasoning='free hand before picking soakable object')
            return self._nav_then('pick_up', role, obj)
        if source_obj is None:
            return ToolCall('stop', {}, reasoning="missing soak source role '{}'".format(source_role))
        if _needs_toggle_on(source_obj):
            return self._nav_then('toggle_on', source_role, source_obj)
        return self._nav_then('soak_under', source_role, source_obj)

    def _plan_held(self, record: Mapping[str, Any]) -> ToolCall | None:
        role = _object_role(record)
        obj = _object_for_role(self._env, role)
        if role is None or obj is None:
            return ToolCall('stop', {}, reasoning='held predicate references missing object')
        obj_name = _object_name(self._env, role)
        held = _held_name(self._env)
        if held == obj_name:
            return None
        if held is not None:
            return ToolCall('drop', {}, reasoning='free hand before picking {}'.format(role))
        return self._nav_then('pick_up', role, obj)

    def _plan_not_held(self, record: Mapping[str, Any]) -> ToolCall | None:
        role = _object_role(record)
        obj_name = _object_name(self._env, role)
        if obj_name is None or _held_name(self._env) != obj_name:
            return None
        return ToolCall('drop', {}, reasoning='release {}'.format(role))

    def _plan_held_object_wrapper(self, object_role: str, obj: Any, source_role: str, tool: str) -> ToolCall:
        env = self._env
        held = _held_name(env)
        obj_name = _object_name(env, object_role)
        source_obj = _object_for_role(env, source_role)
        if held != obj_name:
            if held is not None:
                return ToolCall('drop', {}, reasoning='free hand before picking {}'.format(object_role))
            return self._nav_then('pick_up', object_role, obj)
        if source_obj is None:
            return ToolCall('stop', {}, reasoning="missing {} source role '{}'".format(tool, source_role))
        return self._nav_then(tool, source_role, source_obj)

    def _plan_fluid_clean(self, object_role: str, obj: Any, source_role: str, source_obj: Any) -> ToolCall:
        env = self._env
        held = _held_name(env)
        obj_name = _object_name(env, object_role)
        if held != obj_name:
            if held is not None:
                return ToolCall('drop', {}, reasoning='free hand before washing {}'.format(object_role))
            return self._nav_then('pick_up', object_role, obj)
        if _needs_toggle_on(source_obj):
            return self._nav_then('toggle_on', source_role, source_obj)
        return self._nav_then('soak_under', source_role, source_obj)

    def _plan_cleanup(self) -> ToolCall | None:
        env = self._env
        task_spec = self._task_spec or {}
        for role, obj_name in sorted(env.role_map.items()):
            obj = env.core.objects.get(obj_name)
            if obj is None or not _looks_like_faucet_or_fluid_source(role, obj):
                continue
            if not bool(actions.get_toggle_state(obj)):
                continue
            if _goal_requires_toggle_on(task_spec, role):
                continue
            return self._nav_then('toggle_off', role, obj)
        return None

    def _nav_then(self, tool: str, role: str, obj: Any) -> ToolCall:
        if self._last_successful_nav_target == role:
            return ToolCall(tool, {'target': role})
        if self._failed_call_counts.get(_call_signature(tool, {'target': role}), 0) > 0:
            return ToolCall('navigate_to', {'target': role}, reasoning='renavigate after failed {}'.format(tool))
        if not self._agent_close_to(obj):
            return ToolCall('navigate_to', {'target': role}, reasoning='navigate before {}'.format(tool))
        return ToolCall(tool, {'target': role})

    def _agent_close_to(self, obj: Any) -> bool:
        if self._env is None or obj is None:
            return False
        pose = self._env.core.ego_state.pose
        return math.hypot(float(pose.x) - float(obj.center[0]), float(pose.y) - float(obj.center[1])) <= DEFAULT_AGENT_NEAR_THRESHOLD

    def _closed_container_holding(self, obj: Any) -> tuple[str | None, Any | None]:
        env = self._env
        if env is None or obj is None:
            return (None, None)
        for role, name in sorted(env.role_map.items()):
            candidate = env.core.objects.get(name)
            if candidate is None or candidate.name == obj.name:
                continue
            if not _is_inside(obj, candidate):
                continue
            if _needs_open(candidate):
                return (role, candidate)
        return (None, None)

    def _safe_drop_target(self, *, exclude_role: str | None=None) -> tuple[str | None, Any | None]:
        env = self._env
        if env is None:
            return (None, None)
        pose = env.core.ego_state.pose
        candidates: list[tuple[float, str, Any]] = []
        for role, name in sorted(env.role_map.items()):
            if exclude_role is not None and role == exclude_role:
                continue
            obj = env.core.objects.get(name)
            if obj is None or actions.supports_pick_action(obj):
                continue
            if actions.supports_place_inside_container(obj) and _needs_open(obj):
                continue
            if not (actions.supports_place_on_surface(obj) or actions.supports_place_inside_container(obj)):
                continue
            dist = math.hypot(float(pose.x) - float(obj.center[0]), float(pose.y) - float(obj.center[1]))
            candidates.append((dist, role, obj))
        if not candidates:
            return (None, None)
        _, role, obj = sorted(candidates, key=lambda item: item[0])[0]
        return (role, obj)

    def _role_with_affordance(self, predicate) -> str | None:
        env = self._env
        if env is None:
            return None
        for role, name in sorted(env.role_map.items()):
            obj = env.core.objects.get(name)
            if obj is not None and predicate(obj):
                return role
        return None

    def _cleaning_fluid_source_role(self, record: Mapping[str, Any], object_role: str) -> str | None:
        args = record.get('args') or {}
        explicit = self._first_fluid_source_role((args.get(key) for key in ('source', 'faucet', 'water_source', 'sink')))
        if explicit is not None:
            return explicit
        recover = ((self._task_spec or {}).get('family_spec') or {}).get('recover')
        if isinstance(recover, Mapping):
            incidents = []
            if isinstance(recover.get('incident'), Mapping):
                incidents.append(recover['incident'])
            incidents.extend((item for item in recover.get('incidents') or [] if isinstance(item, Mapping)))
            for incident in incidents:
                incident_role = incident.get('object_role') or incident.get('target_role') or incident.get('target')
                if str(incident_role or '') != object_role:
                    continue
                support = incident.get('support_roles') or {}
                family_role = self._first_fluid_source_role((support.get('faucet'), incident.get('faucet_role'), support.get('sink'), incident.get('sink_role')))
                if family_role is not None:
                    return family_role
        return self._role_with_affordance(actions.supports_fluid_source)

    def _first_fluid_source_role(self, candidates) -> str | None:
        for candidate in candidates:
            if candidate is None:
                continue
            role = str(candidate)
            obj = _object_for_role(self._env, role)
            if obj is not None and actions.supports_fluid_source(obj):
                return role
        return None

def _object_role(record: Mapping[str, Any]) -> str | None:
    role = record.get('object_role') or (record.get('args') or {}).get('object')
    return str(role) if role is not None else None

def _receptacle_role(record: Mapping[str, Any]) -> str | None:
    role = record.get('receptacle_role') or (record.get('args') or {}).get('receptacle')
    return str(role) if role is not None else None

def _object_name(env: Any, role: str | None) -> str | None:
    if env is None or role is None:
        return None
    return env.role_map.get(str(role), str(role))

def _object_for_role(env: Any, role: str | None) -> Any | None:
    name = _object_name(env, role)
    if env is None or name is None:
        return None
    return env.core.objects.get(name)

def _held_name(env: Any) -> str | None:
    if env is None or getattr(env, 'core', None) is None:
        return None
    return getattr(env.core.ego_state, 'held_object_name', None)

def _needs_open(obj: Any) -> bool:
    return actions.supports_open_state(obj) and actions.get_open_state(obj) is not True

def _needs_toggle_on(obj: Any) -> bool:
    return actions.supports_toggle_state(obj) and actions.get_toggle_state(obj) is not True

def _relation_holds(obj: Any, receptacle: Any, relation: str) -> bool:
    if relation == 'inside':
        return _is_inside(obj, receptacle)
    return _is_ontop(obj, receptacle)

def _xy_inside(a: Any, b: Any) -> bool:
    return abs(float(a.center[0]) - float(b.center[0])) <= float(b.extent[0]) * 0.5 and abs(float(a.center[1]) - float(b.center[1])) <= float(b.extent[1]) * 0.5

def _is_inside(a: Any, b: Any) -> bool:
    return _xy_inside(a, b) and float(b.center[2]) - float(b.extent[2]) * 0.5 <= float(a.center[2]) <= float(b.center[2]) + float(b.extent[2]) * 0.5

def _is_ontop(a: Any, b: Any) -> bool:
    return _xy_inside(a, b) and abs(aabb_bottom_z(a) - aabb_top_z(b)) <= ONTOP_Z_TOLERANCE_M

def _wanted_substance(record: Mapping[str, Any]) -> str:
    literal_args = record.get('literal_args') or {}
    return str(literal_args.get('substance') or literal_args.get('liquid') or DEFAULT_SUBSTANCE)

def _goal_requires_toggle_on(task_spec: Mapping[str, Any], role: str) -> bool:
    for predicate in (task_spec.get('goal') or {}).get('predicates') or []:
        record = normalize_goal_predicate(predicate)
        if record.get('type') == 'object_toggled_on' and _object_role(record) == role:
            return True
    return False

def _looks_like_faucet_or_fluid_source(role: str, obj: Any) -> bool:
    role_l = str(role).lower()
    category = str(getattr(obj, 'category', '')).lower()
    abilities = tuple(getattr(obj, 'abilities', ()) or ())
    return 'faucet' in role_l or 'faucet' in category or (any((ability in abilities for ability in actions.FLUID_SOURCE_ABILITY_NAMES)) and actions.supports_toggle_state(obj))

def _call_signature(tool_name: str, arguments: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (str(tool_name), str(arguments.get('target', '')), str(arguments.get('source', '')))
__all__ = ['RecoveryOracleBrain']
