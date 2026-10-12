from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.goal_checker import _eval_inside, _eval_ontop
from scoper.benchmark.behavior.export.ids import json_deep_copy, state_snapshot_id
STATE_VALUE_STATUSES = {'known', 'unknown', 'not_applicable', 'conflict', 'error'}
STATE_VALUE_SOURCES = {'symbolic_core', 'omnigibson_state', 'renderer', 'goal_evaluator', 'scheduler', 'derived_rule', 'unavailable'}
STATE_BOUNDARIES = {'reset', 'pre_action', 'post_action', 'intervention_before', 'intervention_after', 'episode_end'}
PROPERTY_NAMES = ('open', 'toggled_on', 'dirty', 'sliced', 'cooked', 'heated', 'cooled', 'filled', 'saturated')

def state_value(value: Any=None, *, status: str='unknown', source: str='unavailable', source_detail: str | None=None, error_code: str | None=None) -> dict[str, Any]:
    if status not in STATE_VALUE_STATUSES:
        raise ValueError(f'invalid StateValue status {status!r}')
    if source not in STATE_VALUE_SOURCES:
        raise ValueError(f'invalid StateValue source {source!r}')
    if status == 'known' and value is None:
        raise ValueError('known StateValue requires a non-null value')
    if status not in {'known', 'conflict'} and value is not None:
        raise ValueError(f'{status} StateValue requires a null value')
    return {'value': json_deep_copy(value), 'status': status, 'source': source, 'source_detail': source_detail, 'error_code': error_code}

def known_state(value: Any, source: str, source_detail: str | None=None) -> dict[str, Any]:
    return state_value(value, status='known', source=source, source_detail=source_detail)

def unknown_state(source_detail: str | None=None) -> dict[str, Any]:
    return state_value(source_detail=source_detail)

def not_applicable_state(source_detail: str | None=None) -> dict[str, Any]:
    return state_value(status='not_applicable', source='unavailable', source_detail=source_detail)

def error_state(error_code: str, source_detail: str | None=None) -> dict[str, Any]:
    return state_value(status='error', source='unavailable', source_detail=source_detail, error_code=error_code)

class BehaviorStateNormalizer:

    def __init__(self, episode_id: str | None=None) -> None:
        self.episode_id = episode_id
        self._snapshot_ordinal = 0

    def snapshot(self, env: Any, *, boundary: str, step_id: int, action_id: str | None, record_id: str, target_object_ids: Sequence[str]=(), scheduler_object_ids: Sequence[str]=(), full_world: bool=False) -> Mapping[str, Any]:
        if boundary not in STATE_BOUNDARIES:
            raise ValueError(f'invalid state boundary {boundary!r}')
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
            raise ValueError('step_id must be a non-negative integer')
        if not isinstance(record_id, str) or not record_id:
            raise ValueError('record_id must be a non-empty string')
        episode_id = self.episode_id or getattr(env, 'trajectory_episode_id', None)
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('BehaviorStateNormalizer requires an episode_id')
        self._snapshot_ordinal += 1
        identifier = state_snapshot_id(self._snapshot_ordinal)
        errors: list[dict[str, Any]] = []
        core = getattr(env, 'core', None)
        objects = getattr(core, 'objects', None)
        if not isinstance(objects, Mapping):
            objects = {}
            errors.append(_error_info('state_objects_unavailable', 'environment core objects are unavailable'))
        role_map = getattr(env, 'role_map', None)
        if not isinstance(role_map, Mapping):
            role_map = {}
            errors.append(_error_info('state_role_map_unavailable', 'environment role map is unavailable'))
        normalized_roles = {str(role): str(object_id) for role, object_id in role_map.items() if isinstance(role, str) and isinstance(object_id, str)}
        agent, agent_ok, held_ids = self._agent_state(env, errors)
        goal_predicates, goal_ok, goal_object_ids = self._goal_state(env, normalized_roles, errors)
        selected_ids = self._selected_object_ids(objects, normalized_roles, held_ids, target_object_ids, scheduler_object_ids, goal_object_ids, full_world=full_world)
        visible_ids, visibility_source, visibility_detail = self._visibility(env, errors)
        object_rows = []
        for object_id in selected_ids:
            obj = objects.get(object_id)
            if obj is None:
                errors.append(_error_info('state_object_missing', f'selected object {object_id!r} is missing'))
                continue
            object_rows.append(self._object_state(env, object_id, obj, selected_ids, objects, normalized_roles, visible_ids, visibility_source, visibility_detail, errors))
        core_fields_ok = agent_ok and goal_ok
        if not core_fields_ok and (not agent_ok) and (not goal_ok):
            capture_status = 'failed'
        elif errors:
            capture_status = 'partial'
        else:
            capture_status = 'complete'
        return {'schema_version': 'eb-behavior-state.v1', 'snapshot_id': identifier, 'episode_id': episode_id, 'step_id': step_id, 'action_id': action_id, 'record_id': record_id, 'boundary': boundary, 'scope': 'full_world' if full_world else 'task_relevant', 'capture_status': capture_status, 'agent': agent, 'objects': object_rows, 'goal_predicates': goal_predicates, 'snapshot_errors': errors}

    def diff(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> Mapping[str, Any]:
        before_id = _snapshot_identifier(before)
        after_id = _snapshot_identifier(after)
        events: list[dict[str, Any]] = []
        before_agent = before.get('agent') if isinstance(before.get('agent'), Mapping) else {}
        after_agent = after.get('agent') if isinstance(after.get('agent'), Mapping) else {}
        self._append_value_change(events, event_type='relocated', subject_id='agent', object_id=None, field='agent.pose', before=before_agent.get('pose'), after=after_agent.get('pose'), only_boolean_transition=None)
        before_held = _known_list(before_agent.get('held_object_ids'))
        after_held = _known_list(after_agent.get('held_object_ids'))
        if before_held is not None and after_held is not None:
            for object_id in sorted(set(after_held) - set(before_held)):
                events.append(_delta_event('picked_up', object_id, None, 'agent.held_object_ids', before_agent['held_object_ids'], after_agent['held_object_ids']))
            for object_id in sorted(set(before_held) - set(after_held)):
                events.append(_delta_event('released', object_id, None, 'agent.held_object_ids', before_agent['held_object_ids'], after_agent['held_object_ids']))
        before_objects = _rows_by_id(before.get('objects'), 'object_id')
        after_objects = _rows_by_id(after.get('objects'), 'object_id')
        for object_id in sorted(set(before_objects) & set(after_objects)):
            previous = before_objects[object_id]
            current = after_objects[object_id]
            self._append_value_change(events, event_type='relocated', subject_id=object_id, object_id=None, field='pose', before=previous.get('pose'), after=current.get('pose'), only_boolean_transition=None)
            self._append_boolean_event(events, object_id, 'visible', previous.get('visible'), current.get('visible'), true_event='became_visible', false_event='became_hidden')
            previous_properties = previous.get('properties') if isinstance(previous.get('properties'), Mapping) else {}
            current_properties = current.get('properties') if isinstance(current.get('properties'), Mapping) else {}
            property_events = {'open': ('opened', 'closed'), 'toggled_on': ('toggled_on', 'toggled_off'), 'dirty': ('dirtied', 'cleaned'), 'sliced': ('sliced', None), 'cooked': ('heated', None), 'heated': ('heated', None), 'cooled': ('cooled', None), 'filled': ('filled', None), 'saturated': ('saturated', None)}
            for field, (true_event, false_event) in property_events.items():
                self._append_boolean_event(events, object_id, f'properties.{field}', previous_properties.get(field), current_properties.get(field), true_event=true_event, false_event=false_event)
            previous_relations = _relations_by_key(previous.get('relations'))
            current_relations = _relations_by_key(current.get('relations'))
            for key in sorted(set(previous_relations) & set(current_relations)):
                previous_relation = previous_relations[key]
                current_relation = current_relations[key]
                before_value = _known_bool(previous_relation.get('value'))
                after_value = _known_bool(current_relation.get('value'))
                if before_value is None or after_value is None or before_value == after_value:
                    continue
                predicate, related_id = key
                events.append(_delta_event('relation_added' if after_value else 'relation_removed', object_id, related_id, f'relations.{predicate}', previous_relation['value'], current_relation['value']))
        before_goals = _rows_by_id(before.get('goal_predicates'), 'predicate_id')
        after_goals = _rows_by_id(after.get('goal_predicates'), 'predicate_id')
        for predicate_id in sorted(set(before_goals) & set(after_goals)):
            previous = before_goals[predicate_id]
            current = after_goals[predicate_id]
            before_value = _known_bool(previous.get('value'))
            after_value = _known_bool(current.get('value'))
            if before_value is None or after_value is None or before_value == after_value:
                continue
            events.append(_delta_event('predicate_satisfied' if after_value else 'predicate_regressed', predicate_id, None, 'goal_predicate.value', previous['value'], current['value']))
        events.sort(key=lambda item: (item['event_type'], item['subject_id'], item['object_id'] or '', item['field']))
        return {'before_snapshot_id': before_id, 'after_snapshot_id': after_id, 'events': events}

    def _agent_state(self, env: Any, errors: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, list[str]]:
        core = getattr(env, 'core', None)
        ego_state = getattr(core, 'ego_state', None)
        pose_ok = held_ok = True
        try:
            pose = getattr(ego_state, 'pose')
            pose_value = known_state(_pose_mapping(pose), 'symbolic_core', 'env.core.ego_state.pose')
        except Exception as exc:
            pose_ok = False
            pose_value = error_state('agent_pose_getter_error', type(exc).__name__)
            errors.append(_error_info('agent_pose_getter_error', str(exc)))
        try:
            held = getattr(ego_state, 'held_object_name')
            held_ids = [] if held is None else [str(held)]
            held_value = known_state(held_ids, 'symbolic_core', 'env.core.ego_state.held_object_name')
        except Exception as exc:
            held_ok = False
            held_ids = []
            held_value = error_state('held_objects_getter_error', type(exc).__name__)
            errors.append(_error_info('held_objects_getter_error', str(exc)))
        try:
            room_getter = getattr(env, '_current_room_for_pose')
            room = room_getter(getattr(ego_state, 'pose'))
            room_value = known_state(str(room), 'derived_rule', 'env._current_room_for_pose') if room is not None else unknown_state('room lookup returned null')
        except Exception as exc:
            room_value = error_state('agent_room_getter_error', type(exc).__name__)
            errors.append(_error_info('agent_room_getter_error', str(exc)))
        return ({'pose': pose_value, 'room': room_value, 'held_object_ids': held_value}, pose_ok and held_ok, held_ids)

    def _goal_state(self, env: Any, role_map: Mapping[str, str], errors: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool, set[str]]:
        try:
            result = env.check_goal(model_stopped=bool(getattr(env, 'model_stopped', False)))
            records = result.get('predicate_records') if isinstance(result, Mapping) else None
            if not isinstance(records, list):
                raise TypeError('check_goal predicate_records must be a list')
        except Exception as exc:
            errors.append(_error_info('goal_getter_error', str(exc), source='evaluator'))
            return ([], False, set())
        task = getattr(env, 'current_task_spec', None)
        raw_predicates = []
        if isinstance(task, Mapping):
            goal = task.get('goal')
            if isinstance(goal, Mapping) and isinstance(goal.get('predicates'), list):
                raw_predicates = goal['predicates']
        rows: list[dict[str, Any]] = []
        object_ids: set[str] = set()
        for index, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                errors.append(_error_info('goal_record_invalid', f'predicate record {index} is not a mapping', source='evaluator'))
                continue
            expression = raw_predicates[index - 1] if index <= len(raw_predicates) else _goal_expression(record)
            object_ids.update(_object_ids_from_expression(expression, role_map))
            satisfied = record.get('satisfied')
            if isinstance(satisfied, bool):
                value = known_state(satisfied, 'goal_evaluator', f'predicate_records[{index - 1}]')
            else:
                value = error_state('goal_value_invalid', f'predicate_records[{index - 1}].satisfied')
                errors.append(_error_info('goal_value_invalid', f'predicate record {index} satisfied is not boolean', source='evaluator'))
            rows.append({'predicate_id': f'predicate-{index:04d}', 'expression': json_deep_copy(expression), 'value': value})
        return (rows, True, object_ids)

    def _selected_object_ids(self, objects: Mapping[str, Any], role_map: Mapping[str, str], held_ids: Sequence[str], target_object_ids: Sequence[str], scheduler_object_ids: Sequence[str], goal_object_ids: set[str], *, full_world: bool) -> list[str]:
        if full_world:
            return sorted((str(item) for item in objects if isinstance(item, str)))
        selected = set(role_map.values()) | set(held_ids) | goal_object_ids
        selected.update((str(item) for item in target_object_ids))
        selected.update((str(item) for item in scheduler_object_ids))
        selected.intersection_update((str(item) for item in objects))
        closure = set(selected)
        for subject_id in sorted(selected):
            subject = objects.get(subject_id)
            if subject is None:
                continue
            for object_id, candidate in objects.items():
                object_id = str(object_id)
                if object_id == subject_id:
                    continue
                try:
                    if _eval_inside(None, [subject, candidate]) or _eval_ontop(None, [subject, candidate]):
                        closure.add(object_id)
                except Exception:
                    continue
        return sorted(closure)

    def _visibility(self, env: Any, errors: list[dict[str, Any]]) -> tuple[set[str] | None, str, str | None]:
        core = getattr(env, 'core', None)
        ego_state = getattr(core, 'ego_state', None)
        pose = getattr(ego_state, 'pose', None)
        rendered = getattr(env, '_rendered_visible_object_names_at_pose', None)
        if callable(rendered):
            try:
                visible = rendered(pose)
                if visible is not None:
                    return ({str(item) for item in visible}, 'renderer', 'rendered_segmentation')
            except Exception as exc:
                errors.append(_error_info('rendered_visibility_error', str(exc), source='renderer'))
        symbolic = getattr(env, '_symbolic_visible_object_names_at_pose', None)
        if callable(symbolic):
            try:
                return ({str(item) for item in symbolic(pose)}, 'derived_rule', 'symbolic_fov')
            except Exception as exc:
                errors.append(_error_info('symbolic_visibility_error', str(exc)))
        return (None, 'unavailable', None)

    def _object_state(self, env: Any, object_id: str, obj: Any, selected_ids: Sequence[str], objects: Mapping[str, Any], role_map: Mapping[str, str], visible_ids: set[str] | None, visibility_source: str, visibility_detail: str | None, errors: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            pose = known_state(_object_pose_mapping(obj), 'symbolic_core', 'env.core.objects')
        except Exception as exc:
            pose = error_state('object_pose_getter_error', type(exc).__name__)
            errors.append(_error_info('object_pose_getter_error', f'{object_id}: {exc}'))
        try:
            room = env._current_room_for_pose(_object_pose_proxy(obj))
            room_value = known_state(str(room), 'derived_rule', 'env._current_room_for_pose') if room is not None else unknown_state('room lookup returned null')
        except Exception as exc:
            room_value = error_state('object_room_getter_error', type(exc).__name__)
            errors.append(_error_info('object_room_getter_error', f'{object_id}: {exc}'))
        visible = known_state(object_id in visible_ids, visibility_source, visibility_detail) if visible_ids is not None else unknown_state('visibility adapter unavailable')
        interaction_ready = self._interaction_ready(env, object_id, obj, errors)
        properties = self._properties(object_id, obj, errors)
        relations = []
        for related_id in selected_ids:
            if related_id == object_id:
                continue
            related = objects.get(related_id)
            if related is None:
                continue
            for predicate, getter in (('inside', _eval_inside), ('on_top', _eval_ontop)):
                try:
                    value = known_state(bool(getter(None, [obj, related])), 'symbolic_core', f'BehaviorGoalChecker.{predicate}')
                except Exception as exc:
                    value = error_state('relation_getter_error', type(exc).__name__)
                    errors.append(_error_info('relation_getter_error', f'{object_id} {predicate} {related_id}: {exc}'))
                relations.append({'predicate': predicate, 'object_id': related_id, 'value': value})
        relations.sort(key=lambda item: (item['predicate'], item['object_id']))
        return {'object_id': object_id, 'role_names': sorted({role for role, bound in role_map.items() if bound == object_id}), 'category': str(getattr(obj, 'category')) if getattr(obj, 'category', None) is not None else None, 'pose': pose, 'room': room_value, 'visible': visible, 'interaction_ready': interaction_ready, 'properties': properties, 'relations': relations}

    def _interaction_ready(self, env: Any, object_id: str, obj: Any, errors: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            core = env.core
            pose = core.ego_state.pose
            config = core.config.navigation
            min_distance, max_distance = config.interaction_distance
            dx = float(pose.x) - float(obj.center[0])
            dy = float(pose.y) - float(obj.center[1])
            distance = (dx * dx + dy * dy) ** 0.5
            ready = float(min_distance) <= distance <= float(max_distance)
            return known_state(ready, 'derived_rule', 'navigation.interaction_distance')
        except Exception as exc:
            errors.append(_error_info('interaction_ready_getter_error', f'{object_id}: {exc}'))
            return error_state('interaction_ready_getter_error', type(exc).__name__)

    def _properties(self, object_id: str, obj: Any, errors: list[dict[str, Any]]) -> dict[str, Any]:
        states = getattr(obj, 'states', None)
        state_keys = {str(key) for key in states} if isinstance(states, Mapping) else set()
        abilities = {str(item) for item in getattr(obj, 'abilities', ()) or ()}
        specifications: dict[str, tuple[Callable[[Any], Any], Callable[[Any], bool] | bool, str]] = {'open': (actions.get_open_state, actions.supports_open_state, 'actions.get_open_state'), 'toggled_on': (actions.get_toggle_state, actions.supports_toggle_state, 'actions.get_toggle_state'), 'dirty': (actions.get_dirty_state, 'dirty' in state_keys or bool(abilities & {'wipeable', 'cleanable', 'stainable'}), 'actions.get_dirty_state'), 'sliced': (actions.get_sliced_state, 'sliced' in state_keys or bool(abilities & {'cuttable', 'sliceable'}), 'actions.get_sliced_state'), 'cooked': (actions.get_cooked_state, 'cooked' in state_keys or 'cookable' in abilities, 'actions.get_cooked_state'), 'heated': (lambda item: _mapping_boolean_state(item, ('heated', 'hot')), bool(state_keys & {'heated', 'hot'}) or 'heatable' in abilities, 'states.heated'), 'cooled': (actions.get_cooled_state, 'cooled' in state_keys or 'coolable' in abilities, 'actions.get_cooled_state'), 'filled': (lambda item: _substance_boolean_state(item, actions.FILLED_STATE_NAME), actions.supports_fillable_container, 'actions.get_substances(filled)'), 'saturated': (lambda item: _substance_boolean_state(item, actions.SATURATED_STATE_NAME), actions.supports_soakable, 'actions.get_substances(saturated)')}
        result: dict[str, Any] = {}
        for name in PROPERTY_NAMES:
            getter, supports, detail = specifications[name]
            try:
                supported = supports(obj) if callable(supports) else supports
            except Exception as exc:
                result[name] = error_state('property_support_getter_error', type(exc).__name__)
                errors.append(_error_info('property_support_getter_error', f'{object_id}.{name}: {exc}'))
                continue
            if not supported:
                result[name] = not_applicable_state('object does not support property')
                continue
            try:
                value = getter(obj)
                if isinstance(value, bool):
                    result[name] = known_state(value, 'symbolic_core', detail)
                elif value is None:
                    result[name] = unknown_state(f'{detail} returned null')
                else:
                    result[name] = error_state('property_value_invalid', detail)
                    errors.append(_error_info('property_value_invalid', f'{object_id}.{name} returned {type(value).__name__}'))
            except Exception as exc:
                result[name] = error_state('property_getter_error', type(exc).__name__)
                errors.append(_error_info('property_getter_error', f'{object_id}.{name}: {exc}'))
        return result

    @staticmethod
    def _append_boolean_event(events: list[dict[str, Any]], subject_id: str, field: str, before: Any, after: Any, *, true_event: str | None, false_event: str | None) -> None:
        before_value = _known_bool(before)
        after_value = _known_bool(after)
        if before_value is None or after_value is None or before_value == after_value:
            return
        event_type = true_event if after_value else false_event
        if event_type is not None:
            events.append(_delta_event(event_type, subject_id, None, field, before, after))

    @staticmethod
    def _append_value_change(events: list[dict[str, Any]], *, event_type: str, subject_id: str, object_id: str | None, field: str, before: Any, after: Any, only_boolean_transition: bool | None) -> None:
        del only_boolean_transition
        if _is_known(before) and _is_known(after) and (before.get('value') != after.get('value')):
            events.append(_delta_event(event_type, subject_id, object_id, field, before, after))

def _pose_mapping(pose: Any) -> Mapping[str, Any]:
    if isinstance(pose, Mapping):
        return json_deep_copy(pose)
    as_dict = getattr(pose, 'as_dict', None)
    if callable(as_dict):
        return json_deep_copy(as_dict())
    return {'x': float(getattr(pose, 'x')), 'y': float(getattr(pose, 'y')), 'z': float(getattr(pose, 'z')), 'yaw': float(getattr(pose, 'yaw')), 'pitch': float(getattr(pose, 'pitch'))}

def _object_pose_mapping(obj: Any) -> Mapping[str, Any]:
    center = getattr(obj, 'center')
    root_position = getattr(obj, 'root_position', None)
    root_orientation = getattr(obj, 'root_orientation_xyzw', None)
    return {'position': [float(item) for item in (root_position if root_position is not None else center)], 'orientation_xyzw': [float(item) for item in root_orientation] if root_orientation is not None else None, 'yaw': float(getattr(obj, 'root_yaw')) if getattr(obj, 'root_yaw', None) is not None else None, 'source': str(getattr(obj, 'pose_source')) if getattr(obj, 'pose_source', None) is not None else 'aabb_center'}

class _PositionProxy:

    def __init__(self, position: Sequence[float]) -> None:
        self.x = float(position[0])
        self.y = float(position[1])
        self.z = float(position[2])

def _object_pose_proxy(obj: Any) -> _PositionProxy:
    position = getattr(obj, 'root_position', None) or getattr(obj, 'center')
    return _PositionProxy(position)

def _mapping_boolean_state(obj: Any, names: Sequence[str]) -> bool | None:
    states = getattr(obj, 'states', None)
    if not isinstance(states, Mapping):
        return None
    for name in names:
        if name in states:
            value = states[name]
            return value if isinstance(value, bool) else None
    return None

def _substance_boolean_state(obj: Any, state_name: str) -> bool | None:
    states = getattr(obj, 'states', None)
    if not isinstance(states, Mapping) or state_name not in states:
        return None
    value = states[state_name]
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return any((present is True for present in value.values()))
    if isinstance(value, (list, tuple, set, str)):
        return len(value) > 0
    return None

def _goal_expression(record: Mapping[str, Any]) -> Mapping[str, Any]:
    keys = ('type', 'args', 'object_role', 'receptacle_role', 'receptacle_roles', 'literal_args')
    return {key: json_deep_copy(record[key]) for key in keys if key in record}

def _object_ids_from_expression(value: Any, role_map: Mapping[str, str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {'object', 'target', 'receptacle', 'source', 'destination', 'object_role', 'receptacle_role'}:
                if isinstance(item, str):
                    found.add(role_map.get(item, item))
            else:
                found.update(_object_ids_from_expression(item, role_map))
    elif isinstance(value, list):
        for item in value:
            found.update(_object_ids_from_expression(item, role_map))
    return found

def _error_info(code: str, message: str, *, source: str='capture') -> dict[str, Any]:
    return {'error_class': 'infra_error', 'error_code': code, 'message': message or None, 'retryable': None, 'source': source}

def _snapshot_identifier(snapshot: Mapping[str, Any]) -> str:
    identifier = snapshot.get('snapshot_id')
    if not isinstance(identifier, str) or not identifier:
        raise ValueError('snapshot_id must be a non-empty string')
    return identifier

def _rows_by_id(value: Any, key: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        return {}
    return {row[key]: row for row in value if isinstance(row, Mapping) and isinstance(row.get(key), str)}

def _relations_by_key(value: Any) -> dict[tuple[str, str], Mapping[str, Any]]:
    if not isinstance(value, list):
        return {}
    return {(row['predicate'], row['object_id']): row for row in value if isinstance(row, Mapping) and isinstance(row.get('predicate'), str) and isinstance(row.get('object_id'), str)}

def _is_known(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get('status') == 'known' and (value.get('value') is not None)

def _known_bool(value: Any) -> bool | None:
    if not _is_known(value) or not isinstance(value.get('value'), bool):
        return None
    return value['value']

def _known_list(value: Any) -> list[str] | None:
    if not _is_known(value) or not isinstance(value.get('value'), list):
        return None
    if not all((isinstance(item, str) for item in value['value'])):
        return None
    return list(value['value'])

def _delta_event(event_type: str, subject_id: str, object_id: str | None, field: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {'event_type': event_type, 'subject_id': subject_id, 'object_id': object_id, 'field': field, 'before': json_deep_copy(before), 'after': json_deep_copy(after), 'provenance': {'source': 'derived_rule', 'source_ref': None, 'confidence': 0.8, 'review_status': 'unreviewed'}}
__all__ = ['BehaviorStateNormalizer', 'PROPERTY_NAMES', 'STATE_BOUNDARIES', 'STATE_VALUE_SOURCES', 'STATE_VALUE_STATUSES', 'error_state', 'known_state', 'not_applicable_state', 'state_value', 'unknown_state']
