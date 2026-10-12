from __future__ import annotations
import copy
import math
from typing import Any, Mapping, Sequence
import numpy as np
from scoper.benchmark.procthor.export.ids import sha256_json
from scoper.benchmark.procthor.benchmark.admission_schema import build_admission_cell, validate_reachability_preflight
_START_TOLERANCE_M = 0.075
_POSE_GRAPH_TOLERANCE_M = 0.075

class AdmissionEvaluationError(RuntimeError):
    pass

def canonical_reachability_preflight(env: Any, task_spec: Mapping[str, Any], *, episode_context: Any) -> Mapping[str, Any]:
    cell = build_admission_cell(episode_context)
    try:
        metadata = _metadata(env)
        objects = _objects_by_id(metadata)
        if not objects:
            raise AdmissionEvaluationError('controller metadata contains no objects')
        reachable = _reachable_positions(env)
        if not reachable:
            raise AdmissionEvaluationError('GetReachablePositions returned no positions')
        start_pose = _agent_pose(metadata)
        nearest_start = _nearest_xz_distance(start_pose, reachable)
        start_on_graph = nearest_start <= _START_TOLERANCE_M
        roles = []
        for role_name in sorted(task_spec.get('roles', {})):
            roles.append(_evaluate_role(env, role_name, task_spec['roles'][role_name], objects=objects, reachable=reachable))
        primary = start_on_graph and all((bool(row['primary_reachable']) for row in roles))
        audit = start_on_graph and all((bool(row['audit_reachable']) for row in roles))
        house = _house_identity(env, cell.house_content_sha256)
        controller = _controller_identity(env, metadata, objects)
        renderer = _renderer_identity(env, metadata, objects)
        disagreements = [row['role'] for row in roles if bool(row['primary_reachable']) != bool(row['audit_reachable'])]
        alignment = {'consistent': bool(house['content_matches_cell'] and controller['role_bindings_match_metadata'] and renderer['identity_consistent'] and (not disagreements)), 'evaluator_disagreement_roles': disagreements, 'house_matches_cell': bool(house['content_matches_cell']), 'controller_matches_task': bool(controller['role_bindings_match_metadata']), 'renderer_matches_controller': bool(renderer['identity_consistent'])}
        if not alignment['consistent'] or primary != audit:
            result = _result(cell, status='invalid', error_class='evaluator_invalid', failure_code='ADMISSION_EVALUATOR_DISAGREEMENT', disposition='needs_rerun', house=house, controller=controller, renderer=renderer, start_pose=start_pose, reachable_count=len(reachable), nearest_start=nearest_start, start_on_graph=start_on_graph, roles=roles, primary=primary, audit=audit, alignment=alignment)
        else:
            passed = primary and audit
            result = _result(cell, status='pass' if passed else 'fail', error_class='none' if passed else 'task_invalid', failure_code=None if passed else 'ADMISSION_UNREACHABLE', disposition='admitted' if passed else 'rejected', house=house, controller=controller, renderer=renderer, start_pose=start_pose, reachable_count=len(reachable), nearest_start=nearest_start, start_on_graph=start_on_graph, roles=roles, primary=primary, audit=audit, alignment=alignment)
    except Exception as exc:
        result = _invalid_infra(cell, exc)
    validate_reachability_preflight(result)
    return result

def reset_failure_evidence(episode_context: Any, exc: BaseException) -> Mapping[str, Any]:
    cell = build_admission_cell(episode_context)
    result = _invalid_infra(cell, exc, failure_code='ADMISSION_INFRA_RESET_ERROR')
    validate_reachability_preflight(result)
    return result

def infrastructure_failure_evidence(episode_context: Any, exc: BaseException, *, failure_code: str) -> Mapping[str, Any]:
    if failure_code not in {'ADMISSION_INFRA_RESET_ERROR', 'ADMISSION_INFRA_RENDERER_ERROR', 'ADMISSION_INFRA_CONTROLLER_ERROR', 'ADMISSION_INFRA_CAPTURE_ERROR'}:
        raise ValueError('unsupported admission infrastructure failure code')
    cell = build_admission_cell(episode_context)
    result = _invalid_infra(cell, exc, failure_code=failure_code)
    validate_reachability_preflight(result)
    return result

def _evaluate_role(env: Any, role_name: str, role: Mapping[str, Any], *, objects: Mapping[str, Mapping[str, Any]], reachable: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    object_id = str(role.get('object_id') or '')
    obj = objects.get(object_id)
    result: dict[str, Any] = {'role': role_name, 'object_id': object_id, 'object_type': role.get('object_type'), 'object_exists': obj is not None, 'interaction_target_object_id': object_id, 'interaction_pose_count': 0, 'reachable_interaction_pose_count': 0, 'primary_reachable': False, 'audit_reachable': False, 'activation_chain': [], 'rejection_reasons': [], 'selected_pose': None}
    if obj is None:
        result['rejection_reasons'] = ['required_object_missing']
        return result
    target = obj
    activation_chain = _activation_chain(obj, objects)
    result['activation_chain'] = activation_chain
    illegal = [row for row in activation_chain if row['legally_activatable'] is not True]
    if illegal:
        result['rejection_reasons'] = ['contained_parent_activation_illegal']
        return result
    closed = [row for row in activation_chain if row['requires_open'] is True]
    if closed:
        parent_id = closed[0]['parent_object_id']
        target = objects[parent_id]
        result['interaction_target_object_id'] = parent_id
    poses = _interactable_poses(env, str(target['objectId']))
    graph_poses = [pose for pose in poses if _nearest_xz_distance(pose, reachable) <= _POSE_GRAPH_TOLERANCE_M]
    result['interaction_pose_count'] = len(poses)
    result['reachable_interaction_pose_count'] = len(graph_poses)
    result['primary_reachable'] = bool(poses)
    result['audit_reachable'] = bool(graph_poses)
    if graph_poses:
        result['selected_pose'] = _portable_pose(graph_poses[0])
    reasons = []
    if not poses:
        reasons.append('no_native_interaction_pose')
    elif not graph_poses:
        reasons.append('interaction_poses_outside_reachable_graph')
    result['rejection_reasons'] = sorted(set(reasons))
    return result

def _activation_chain(obj: Mapping[str, Any], objects: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    seen = set()
    frontier = list(obj.get('parentReceptacles') or [])
    while frontier:
        parent_id = frontier.pop(0)
        if not isinstance(parent_id, str) or parent_id in seen:
            continue
        seen.add(parent_id)
        parent = objects.get(parent_id)
        exists = parent is not None
        openable = bool((parent or {}).get('openable', False))
        is_open = (parent or {}).get('isOpen') if openable else None
        rows.append({'parent_object_id': parent_id, 'exists': exists, 'openable': openable, 'is_open': is_open if isinstance(is_open, bool) else None, 'requires_open': bool(exists and openable and (is_open is False)), 'legally_activatable': bool(exists and (not openable or isinstance(is_open, bool)))})
        if parent is not None:
            frontier.extend(parent.get('parentReceptacles') or [])
    return rows

def _reachable_positions(env: Any) -> list[Mapping[str, Any]]:
    controller = getattr(env, 'controller', None)
    if controller is None:
        raise AdmissionEvaluationError('controller is unavailable')
    event = controller.step(action='GetReachablePositions')
    metadata = getattr(event, 'metadata', {}) or {}
    if metadata.get('lastActionSuccess') is not True:
        raise AdmissionEvaluationError('GetReachablePositions failed: {}'.format(metadata.get('errorMessage') or ''))
    positions = metadata.get('actionReturn')
    if not isinstance(positions, list):
        raise AdmissionEvaluationError('GetReachablePositions returned a non-list')
    return [dict(item) for item in positions if _has_xz(item)]

def _interactable_poses(env: Any, object_id: str) -> list[Mapping[str, Any]]:
    event = env.controller.step(action='GetInteractablePoses', objectId=object_id)
    metadata = getattr(event, 'metadata', {}) or {}
    if metadata.get('lastActionSuccess') is not True:
        raise AdmissionEvaluationError('GetInteractablePoses failed: {}'.format(metadata.get('errorMessage') or ''))
    poses = metadata.get('actionReturn') or []
    if not isinstance(poses, list):
        raise AdmissionEvaluationError('GetInteractablePoses returned a non-list')
    return [dict(item) for item in poses if _has_xz(item)]

def _metadata(env: Any) -> Mapping[str, Any]:
    getter = getattr(env, '_metadata', None)
    if callable(getter):
        value = getter()
    else:
        value = getattr(getattr(env, 'controller', None), 'last_event', None)
        value = getattr(value, 'metadata', None)
    if not isinstance(value, Mapping):
        raise AdmissionEvaluationError('controller metadata is unavailable')
    return value

def _objects_by_id(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item['objectId']): item for item in metadata.get('objects', []) or [] if isinstance(item, Mapping) and isinstance(item.get('objectId'), str) and item['objectId']}

def _agent_pose(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    agent = metadata.get('agent')
    position = agent.get('position') if isinstance(agent, Mapping) else None
    if not _has_xz(position):
        raise AdmissionEvaluationError('agent start pose is unavailable')
    return {'position': {'x': float(position['x']), 'y': float(position.get('y', 0.0)), 'z': float(position['z'])}, 'rotation': copy.deepcopy((agent or {}).get('rotation') or {}), 'camera_horizon': (agent or {}).get('cameraHorizon'), 'is_standing': (agent or {}).get('isStanding')}

def _house_identity(env: Any, expected_sha256: str) -> Mapping[str, Any]:
    house = getattr(env, 'current_house', None)
    actual = sha256_json(dict(house)) if isinstance(house, Mapping) else None
    return {'expected_content_sha256': expected_sha256, 'loaded_content_sha256': actual, 'content_matches_cell': actual == expected_sha256}

def _controller_identity(env: Any, metadata: Mapping[str, Any], objects: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    bindings = getattr(env, 'role_object_ids', {})
    bindings = bindings if isinstance(bindings, Mapping) else {}
    missing = sorted((str(role) for role, object_id in bindings.items() if object_id not in objects))
    return {'scene_name': metadata.get('sceneName'), 'object_count': len(objects), 'role_binding_count': len(bindings), 'missing_role_bindings': missing, 'role_bindings_match_metadata': not missing, 'last_action': metadata.get('lastAction'), 'last_action_success': metadata.get('lastActionSuccess')}

def _renderer_identity(env: Any, metadata: Mapping[str, Any], objects: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    event = getattr(getattr(env, 'controller', None), 'last_event', None)
    frame = getattr(event, 'frame', None)
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2:] != (3,):
        raise AdmissionEvaluationError('renderer frame is unavailable or malformed')
    if frame.dtype != np.uint8:
        raise AdmissionEvaluationError('renderer frame dtype is not uint8')
    expected_shape = [int(getattr(env, 'height', frame.shape[0])), int(getattr(env, 'width', frame.shape[1])), 3]
    shape = list(frame.shape)
    if shape != expected_shape:
        raise AdmissionEvaluationError('renderer frame shape differs from configured camera')
    masks = getattr(event, 'instance_masks', None)
    mask_ids = sorted((str(key) for key in masks)) if isinstance(masks, Mapping) else []
    renderer_only_ids = sorted(set(mask_ids) - set(objects))
    structural_renderer_ids = [object_id for object_id in renderer_only_ids if object_id.split('_', 1)[0].split('|', 1)[0].casefold() in {'floor', 'wall', 'ceiling'}]
    unknown_mask_ids = sorted(set(renderer_only_ids) - set(structural_renderer_ids))
    return {'frame_shape': shape, 'frame_dtype': str(frame.dtype), 'metadata_frame_available': bool(metadata), 'segmentation_available': isinstance(masks, Mapping), 'segmentation_object_count': len(mask_ids), 'renderer_only_structural_object_ids': structural_renderer_ids, 'unknown_segmentation_object_ids': unknown_mask_ids, 'identity_consistent': not unknown_mask_ids}

def _result(cell: Any, *, status: str, error_class: str, failure_code: str | None, disposition: str, house: Mapping[str, Any], controller: Mapping[str, Any], renderer: Mapping[str, Any], start_pose: Mapping[str, Any], reachable_count: int, nearest_start: float, start_on_graph: bool, roles: Sequence[Mapping[str, Any]], primary: bool, audit: bool, alignment: Mapping[str, Any]) -> Mapping[str, Any]:
    return {'schema_version': 'procthor-admission-preflight.v1', 'stage': 'reachability', 'status': status, 'error_class': error_class, 'failure_code': failure_code, 'disposition': disposition, 'cell': cell.to_mapping(), 'house': copy.deepcopy(dict(house)), 'controller': copy.deepcopy(dict(controller)), 'renderer': copy.deepcopy(dict(renderer)), 'start': {'pose': copy.deepcopy(dict(start_pose)), 'reachable_position_count': int(reachable_count), 'nearest_distance_m': _stable_float(nearest_start), 'on_reachable_graph': bool(start_on_graph)}, 'required_roles': copy.deepcopy(list(roles)), 'evaluators': {'native_interaction_poses': bool(primary), 'independent_reachable_graph_audit': bool(audit)}, 'alignment': copy.deepcopy(dict(alignment))}

def _invalid_infra(cell: Any, exc: BaseException, *, failure_code: str | None=None) -> Mapping[str, Any]:
    text = '{} {}'.format(type(exc).__name__, exc).casefold()
    code = failure_code or ('ADMISSION_INFRA_RENDERER_ERROR' if 'render' in text or 'frame' in text or 'segmentation' in text else 'ADMISSION_INFRA_CONTROLLER_ERROR')
    return {'schema_version': 'procthor-admission-preflight.v1', 'stage': 'reachability', 'status': 'invalid', 'error_class': 'infra_error', 'failure_code': code, 'disposition': 'needs_rerun', 'cell': cell.to_mapping(), 'house': {}, 'controller': {}, 'renderer': {}, 'start': {'pose': {}, 'reachable_position_count': 0, 'nearest_distance_m': None, 'on_reachable_graph': None}, 'required_roles': [], 'evaluators': {'native_interaction_poses': None, 'independent_reachable_graph_audit': None}, 'alignment': {'consistent': False, 'error_type': type(exc).__name__, 'error_message': _redact_error(str(exc))}}

def _portable_pose(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {'x': _stable_float(float(value['x'])), 'y': _stable_float(float(value.get('y', 0.0))), 'z': _stable_float(float(value['z'])), 'rotation': _stable_float(float(value.get('rotation', 0.0))), 'horizon': _stable_float(float(value.get('horizon', 0.0))), 'standing': bool(value.get('standing', True))}

def _nearest_xz_distance(value: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> float:
    position = value.get('position') if isinstance(value.get('position'), Mapping) else value
    if not _has_xz(position) or not candidates:
        return math.inf
    return min((math.hypot(float(position['x']) - float(item['x']), float(position['z']) - float(item['z'])) for item in candidates if _has_xz(item)))

def _has_xz(value: Any) -> bool:
    return isinstance(value, Mapping) and all((isinstance(value.get(axis), (int, float)) and (not isinstance(value.get(axis), bool)) for axis in ('x', 'z')))

def _stable_float(value: float) -> float:
    return float(round(value, 8))

def _redact_error(message: str) -> str:
    text = str(message)
    lowered = text.casefold()
    for marker in ('api_key=', 'api-key=', 'authorization:', 'bearer ', 'token='):
        index = lowered.find(marker)
        if index >= 0:
            return text[:index + len(marker)] + '<redacted>'
    return text
__all__ = ['AdmissionEvaluationError', 'canonical_reachability_preflight', 'infrastructure_failure_evidence', 'reset_failure_evidence']
