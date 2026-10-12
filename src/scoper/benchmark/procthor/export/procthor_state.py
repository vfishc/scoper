from __future__ import annotations
import copy
import math
from typing import Any, Mapping, Sequence
from scoper.benchmark.procthor.predicate_goal_checker import check_goal
from scoper.benchmark.procthor.spatial_state import interaction_ready_from_metadata, room_id_for_position
from scoper.benchmark.procthor.export.schema import STATE_SCHEMA_VERSION
STATE_PROPERTIES = ('isOpen', 'openness', 'isToggled', 'isDirty', 'isSliced', 'isCooked', 'isHeated', 'isCooled', 'fillLiquid')
AFFORDANCES = ('pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable', 'dirtyable')

def state_value(value: Any, *, source: str, status: str='known', source_detail: str | None=None, error_code: str | None=None) -> dict[str, Any]:
    if status == 'known' and value is None:
        status = 'unknown'
        source = 'unavailable'
    return {'value': copy.deepcopy(value) if status == 'known' else None, 'status': status, 'source': source, 'source_detail': source_detail, 'error_code': error_code}

def unavailable(*, status: str='unknown', error_code: str | None=None) -> dict[str, Any]:
    return state_value(None, source='unavailable', status=status, error_code=error_code)

class ProcTHORStateNormalizer:

    def snapshot(self, env: Any, *, boundary: str, step_id: int, action_id: str | None, record_id: str, target_object_ids: Sequence[str]=(), scheduler_object_ids: Sequence[str]=(), full_world: bool=False) -> Mapping[str, Any]:
        errors: list[dict[str, Any]] = []
        try:
            native_metadata = copy.deepcopy(dict(env._metadata()))
        except Exception as exc:
            native_metadata = {}
            errors.append(_error('native_metadata_unavailable', exc))
        try:
            effective_metadata = copy.deepcopy(dict(env._metadata_with_sticky_object_states()))
        except Exception as exc:
            effective_metadata = {}
            errors.append(_error('effective_metadata_unavailable', exc))
        native_objects = _objects_by_id(native_metadata)
        effective_objects = _objects_by_id(effective_metadata)
        inventory_native = _inventory_ids(native_metadata)
        inventory_effective = _inventory_ids(effective_metadata)
        house = getattr(env, 'current_house', None)
        relevant = set((str(item) for item in target_object_ids if item))
        relevant.update((str(item) for item in scheduler_object_ids if item))
        relevant.update((str(item) for item in getattr(env, 'role_object_ids', {}).values()))
        relevant.update(inventory_native)
        relevant.update(inventory_effective)
        if full_world:
            relevant.update(native_objects)
            relevant.update(effective_objects)
        else:
            _expand_one_hop(relevant, native_objects)
            _expand_one_hop(relevant, effective_objects)
        role_names_by_id: dict[str, list[str]] = {}
        for role, object_id in getattr(env, 'role_object_ids', {}).items():
            role_names_by_id.setdefault(str(object_id), []).append(str(role))
        objects = []
        for object_id in sorted(relevant):
            native = native_objects.get(object_id)
            effective = effective_objects.get(object_id)
            source = native or effective or {}
            native_view = _object_view(native, native_metadata, house=house, source='ai2thor_metadata')
            effective_source = _effective_overlay_source(env, object_id) if native is not None and effective is not None and (native != effective) else 'ai2thor_metadata'
            effective_view = _object_view(effective, effective_metadata, house=house, source=effective_source, inventory_source=_effective_overlay_source(env, object_id) if (object_id in inventory_native) != (object_id in inventory_effective) else effective_source)
            objects.append({'object_id': object_id, 'role_names': sorted(role_names_by_id.get(object_id, [])), 'object_type': source.get('objectType'), 'native': native_view, 'effective': effective_view, 'affordances': {key: _metadata_value(source, key, source='ai2thor_metadata') for key in AFFORDANCES}, 'overlay_deltas': _overlay_deltas(object_id, native_view, effective_view)})
        task_spec = _active_task_spec(env)
        native_goal = _goal_values(task_spec, native_metadata, env, errors, 'native')
        effective_goal = _goal_values(task_spec, effective_metadata, env, errors, 'effective')
        event_ordinal = getattr(env, '_trajectory_event_ordinal', None)
        capture_status = 'complete' if not errors else 'partial'
        if not native_metadata and (not effective_metadata):
            capture_status = 'failed'
        return {'schema_version': STATE_SCHEMA_VERSION, 'episode_id': _episode_id(env), 'step_id': int(step_id), 'action_id': action_id, 'record_id': record_id, 'boundary': boundary, 'scope': 'full_world' if full_world else 'task_relevant', 'capture_status': capture_status, 'agent': {'native_pose': _agent_pose_value(native_metadata, 'ai2thor_metadata'), 'effective_pose': _agent_pose_value(effective_metadata, 'ai2thor_metadata'), 'room': _agent_room_value(native_metadata, house), 'native_inventory_object_ids': state_value(inventory_native, source='ai2thor_metadata'), 'effective_inventory_object_ids': state_value(inventory_effective, source='visual_attachment_overlay' if inventory_native != inventory_effective else 'ai2thor_metadata')}, 'objects': objects, 'protocol_state': {'scan_views_captured': state_value(int(getattr(env, '_scan_views_captured_total', 0)), source='derived_rule'), 'accepted_question_ids': state_value(sorted((str(question_id) for question_id in getattr(env, '_human_qa_state', {}).get('answers', {}))), source='derived_rule'), 'accepted_stop_requests': state_value(int(getattr(env, '_accepted_stop_requests', 0)), source='derived_rule')}, 'native_goal_predicates': native_goal, 'effective_goal_predicates': effective_goal, 'snapshot_errors': errors, 'extensions': {'event_ordinal': event_ordinal, 'native_effective_same_event': event_ordinal is not None}}

    def diff(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> Mapping[str, Any]:
        return {'before_snapshot_id': before.get('snapshot_id'), 'after_snapshot_id': after.get('snapshot_id'), 'native_events': _diff_track(before, after, 'native'), 'effective_events': _diff_track(before, after, 'effective'), 'overlay_only_events': _overlay_only_events(before, after)}

def _episode_id(env: Any) -> str:
    context = getattr(env, '_trajectory_episode_context', None)
    return str(getattr(context, 'episode_id', 'unknown'))

def _active_task_spec(env: Any) -> Mapping[str, Any]:
    getter = getattr(env, '_active_task_spec', None)
    if callable(getter):
        value = getter()
        if isinstance(value, Mapping):
            return value
    value = getattr(env, 'current_task_spec', None)
    return value if isinstance(value, Mapping) else {}

def _effective_overlay_source(env: Any, object_id: str) -> str:
    if object_id == getattr(env, '_visual_attached_object_id', None):
        return 'visual_attachment_overlay'
    return 'sticky_overlay'

def _objects_by_id(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item['objectId']): copy.deepcopy(dict(item)) for item in metadata.get('objects', []) or [] if isinstance(item, Mapping) and isinstance(item.get('objectId'), str)}

def _inventory_ids(metadata: Mapping[str, Any]) -> list[str]:
    return sorted({str(item['objectId']) for item in metadata.get('inventoryObjects', []) or [] if isinstance(item, Mapping) and isinstance(item.get('objectId'), str)})

def _expand_one_hop(relevant: set[str], objects: Mapping[str, Mapping[str, Any]]) -> None:
    initial = set(relevant)
    for object_id in initial:
        item = objects.get(object_id, {})
        relevant.update((str(value) for value in item.get('parentReceptacles', []) or []))
        relevant.update((str(value) for value in item.get('receptacleObjectIds', []) or []))

def _agent_pose_value(metadata: Mapping[str, Any], source: str) -> dict[str, Any]:
    agent = metadata.get('agent')
    if not isinstance(agent, Mapping):
        return unavailable(error_code='agent_metadata_missing')
    pose = {'position': copy.deepcopy(agent.get('position')), 'rotation': copy.deepcopy(agent.get('rotation')), 'horizon': agent.get('cameraHorizon'), 'standing': agent.get('isStanding')}
    return state_value(pose, source=source)

def _object_view(item: Mapping[str, Any] | None, metadata: Mapping[str, Any], *, house: Mapping[str, Any] | None, source: str, inventory_source: str | None=None) -> dict[str, Any]:
    if item is None:
        missing = unavailable(error_code='object_metadata_missing')
        return {'pose': missing, 'room': unavailable(error_code='object_room_unavailable'), 'interaction_ready': unavailable(error_code='interaction_ready_unavailable'), 'visible': missing, 'distance': missing, 'properties': {key: missing for key in STATE_PROPERTIES}, 'parentReceptacles': missing, 'receptacleObjectIds': missing, 'inventory_member': missing}
    object_id = item.get('objectId')
    inventory = set(_inventory_ids(metadata))
    return {'pose': state_value({'position': copy.deepcopy(item.get('position')), 'rotation': copy.deepcopy(item.get('rotation'))}, source=source), 'room': _room_value(item, house), 'interaction_ready': _interaction_ready_value(metadata, str(object_id), source), 'visible': _metadata_value(item, 'visible', source=source), 'distance': _metadata_value(item, 'distance', source=source), 'properties': {key: _metadata_value(item, key, source=source, not_applicable=not _property_applicable(item, key)) for key in STATE_PROPERTIES}, 'parentReceptacles': state_value(sorted(set(item.get('parentReceptacles') or [])), source=source), 'receptacleObjectIds': state_value(sorted(set(item.get('receptacleObjectIds') or [])), source=source), 'inventory_member': state_value(object_id in inventory, source=inventory_source or source)}

def _agent_room_value(metadata: Mapping[str, Any], house: Mapping[str, Any] | None) -> dict[str, Any]:
    agent = metadata.get('agent')
    position = agent.get('position') if isinstance(agent, Mapping) else None
    room_id = room_id_for_position(position, house)
    if room_id is None:
        return unavailable(error_code='agent_room_unavailable')
    return state_value(room_id, source='derived_rule')

def _room_value(item: Mapping[str, Any], house: Mapping[str, Any] | None) -> dict[str, Any]:
    room_id = room_id_for_position(item.get('position'), house, metadata_room_id=item.get('roomId') or item.get('room_id'))
    if room_id is None:
        return unavailable(error_code='object_room_unavailable')
    return state_value(room_id, source='derived_rule')

def _interaction_ready_value(metadata: Mapping[str, Any], object_id: str, source: str) -> dict[str, Any]:
    ready = interaction_ready_from_metadata(metadata, object_id)
    if ready is None:
        return unavailable(error_code='interaction_ready_unavailable')
    return state_value(ready, source=source)

def _metadata_value(item: Mapping[str, Any], key: str, *, source: str, not_applicable: bool=False) -> dict[str, Any]:
    if not_applicable:
        return unavailable(status='not_applicable')
    if key not in item or item[key] is None:
        return unavailable(error_code='{}_missing'.format(key))
    return state_value(item[key], source=source)

def _property_applicable(item: Mapping[str, Any], field: str) -> bool:
    affordance = {'isOpen': 'openable', 'openness': 'openable', 'isToggled': 'toggleable', 'isDirty': 'dirtyable', 'isSliced': 'sliceable', 'isCooked': 'cookable', 'isHeated': 'canBeUsedUp', 'isCooled': 'canBeUsedUp', 'fillLiquid': 'canFillWithLiquid'}.get(field)
    return field in item or affordance is None or bool(item.get(affordance, False))

def _goal_values(task_spec: Mapping[str, Any], metadata: Mapping[str, Any], env: Any, errors: list[dict[str, Any]], track: str) -> list[dict[str, Any]]:
    if not task_spec or not metadata:
        return []
    try:
        result = check_goal(task_spec, metadata, model_stopped=bool(getattr(env, 'model_stopped', False)))
        return [{'index': index, 'predicate_type': predicate.get('type'), 'status': state_value(bool(predicate.get('satisfied')), source='goal_evaluator')} for index, predicate in enumerate(result.get('predicates', []))]
    except Exception as exc:
        errors.append(_error('{}_goal_evaluation_failed'.format(track), exc))
        return []

def _overlay_deltas(object_id: str, native: Mapping[str, Any], effective: Mapping[str, Any]) -> list[dict[str, Any]]:
    deltas = []
    for path, native_value in _flatten_state_values(native):
        effective_value = _get_path(effective, path)
        if not isinstance(effective_value, Mapping):
            continue
        if native_value.get('value') != effective_value.get('value'):
            deltas.append({'field': path, 'native_value': copy.deepcopy(native_value), 'effective_value': copy.deepcopy(effective_value), 'source': effective_value.get('source', 'sticky_overlay'), 'reason': None})
    return deltas

def _flatten_state_values(value: Any, prefix: str=''):
    if isinstance(value, Mapping) and {'value', 'status', 'source'}.issubset(value):
        yield (prefix, value)
    elif isinstance(value, Mapping):
        for key in sorted(value):
            path = '{}/{}'.format(prefix, key) if prefix else str(key)
            yield from _flatten_state_values(value[key], path)

def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split('/'):
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current

def _diff_track(before: Mapping[str, Any], after: Mapping[str, Any], track: str) -> list[dict[str, Any]]:
    before_objects = {item['object_id']: item for item in before.get('objects', [])}
    after_objects = {item['object_id']: item for item in after.get('objects', [])}
    events = []
    for object_id in sorted(set(before_objects) | set(after_objects)):
        prior = before_objects.get(object_id, {}).get(track, {})
        current = after_objects.get(object_id, {}).get(track, {})
        for path, prior_value in _flatten_state_values(prior):
            current_value = _get_path(current, path)
            if not isinstance(current_value, Mapping):
                continue
            if prior_value.get('value') == current_value.get('value'):
                continue
            event_type = _event_type(path, prior_value.get('value'), current_value.get('value'))
            events.append({'event_type': event_type, 'subject_id': object_id, 'object_id': None, 'field': path, 'before': copy.deepcopy(prior_value), 'after': copy.deepcopy(current_value), 'provenance': _provenance('ai2thor_metadata' if track == 'native' else 'effective_overlay')})
    return events

def _event_type(path: str, before: Any, after: Any) -> str:
    if path.endswith('inventory_member'):
        return 'picked_up' if after is True else 'released'
    if path.endswith('parentReceptacles'):
        return 'relation_added' if len(after or []) >= len(before or []) else 'relation_removed'
    mapping = {'isOpen': ('opened', 'closed'), 'isToggled': ('toggled_on', 'toggled_off'), 'isDirty': ('dirtied', 'cleaned'), 'isSliced': ('sliced', 'sliced'), 'isHeated': ('heated', 'heated'), 'isCooled': ('cooled', 'cooled'), 'fillLiquid': ('filled', 'filled'), 'visible': ('became_visible', 'became_hidden')}
    for field, pair in mapping.items():
        if path.endswith(field):
            return pair[0] if bool(after) else pair[1]
    if path.endswith('pose'):
        return 'relocated'
    return 'relocated'

def _overlay_only_events(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    native = _diff_track(before, after, 'native')
    effective = _diff_track(before, after, 'effective')
    native_keys = {(item['event_type'], item['subject_id'], item['field']) for item in native}
    return [item for item in effective if (item['event_type'], item['subject_id'], item['field']) not in native_keys]

def _provenance(source: str) -> dict[str, Any]:
    return {'source': source, 'source_ref': None, 'confidence': 0.8, 'review_status': 'unreviewed'}

def _error(code: str, exc: Exception) -> dict[str, Any]:
    return {'error_class': 'infra_error', 'error_code': code, 'message': type(exc).__name__, 'retryable': None, 'source': 'capture'}
__all__ = ['ProcTHORStateNormalizer', 'state_value', 'unavailable']
