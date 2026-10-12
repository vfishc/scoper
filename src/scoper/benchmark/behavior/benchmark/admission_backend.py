from __future__ import annotations
from collections import Counter
import math
from typing import Any, Mapping, Sequence
import numpy as np
from scoper.benchmark.behavior.execution_core.navigation import NavigationTarget, plan_navigation_to_pose, sample_interaction_candidates
from scoper.benchmark.behavior.export.ids import canonical_json_bytes, json_deep_copy, sha256_bytes
from scoper.benchmark.behavior.task_generation.stage5_validator import key_roles_for_validation
from scoper.benchmark.behavior.benchmark.admission_schema import build_admission_cell, validate_reachability_preflight
_STRUCTURAL_ROLE_CATEGORIES = frozenset({'floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'})
_REJECT_SAMPLE_LIMIT = 24

class AdmissionEvaluationError(RuntimeError):
    pass

def canonical_reachability_preflight(env: Any, task_spec: Mapping[str, Any], *, episode_context: Any) -> Mapping[str, Any]:
    cell = build_admission_cell(episode_context)
    try:
        occupancy = getattr(getattr(env, 'core', None), 'occupancy', None)
        if occupancy is None:
            raise AdmissionEvaluationError('canonical navigation occupancy is unavailable')
        map_diagnostics = _navigation_map_diagnostics(env, occupancy)
        start_pose = env.core.ego_state.pose.copy()
        start_cell = occupancy.world_to_cell(start_pose.xy)
        start_component = occupancy.component_at(start_pose.xy)
        start_free = bool(occupancy.is_free(start_pose.xy))
        roles = _required_roles(task_spec)
        parent_by_child = _contained_parent_roles(task_spec)
        role_reports = []
        for role in roles:
            role_reports.append(_evaluate_role(env, task_spec, role, occupancy=occupancy, start_pose=start_pose, start_component=start_component, parent_by_child=parent_by_child))
        evaluator_primary = start_free and all((bool(report['primary_reachable']) for report in role_reports))
        evaluator_audit = start_free and all((bool(report['audit_reachable']) for report in role_reports))
        alignment = _alignment_diagnostics(env, occupancy, map_diagnostics)
        evaluator_disagreements = [report['role'] for report in role_reports if bool(report['primary_reachable']) != bool(report['audit_reachable'])]
        alignment = dict(alignment)
        alignment['evaluator_disagreement_roles'] = evaluator_disagreements
        if not alignment['consistent'] or evaluator_disagreements:
            result = _result(cell, status='invalid', error_class='evaluator_invalid', failure_code='ADMISSION_EVALUATOR_DISAGREEMENT', disposition='needs_rerun', map_diagnostics=map_diagnostics, alignment=alignment, start_pose=start_pose.as_dict(), start_cell=start_cell, start_component=start_component, start_free=start_free, roles=role_reports, primary=evaluator_primary, audit=evaluator_audit)
            validate_reachability_preflight(result)
            return result
        passed = evaluator_primary and evaluator_audit
        result = _result(cell, status='pass' if passed else 'fail', error_class='none' if passed else 'task_invalid', failure_code=None if passed else 'ADMISSION_UNREACHABLE', disposition='admitted' if passed else 'rejected', map_diagnostics=map_diagnostics, alignment=alignment, start_pose=start_pose.as_dict(), start_cell=start_cell, start_component=start_component, start_free=start_free, roles=role_reports, primary=evaluator_primary, audit=evaluator_audit)
        validate_reachability_preflight(result)
        return result
    except AdmissionEvaluationError as exc:
        result = _invalid_infra(cell, env, exc)
    except Exception as exc:
        result = _invalid_infra(cell, env, exc)
    validate_reachability_preflight(result)
    return result

def _evaluate_role(env: Any, task_spec: Mapping[str, Any], role: str, *, occupancy: Any, start_pose: Any, start_component: int, parent_by_child: Mapping[str, str]) -> Mapping[str, Any]:
    resolved = getattr(env, 'role_map', {}).get(role)
    report: dict[str, Any] = {'role': role, 'resolved_object': resolved, 'object_exists': False, 'access_mode': 'direct', 'parent_role_or_null': None, 'parent_object_or_null': None, 'parent_activation': None, 'interaction_candidate_count': 0, 'free_candidate_count': 0, 'same_component_candidate_count': 0, 'path_planned_candidate_count': 0, 'reachable_candidate_count': 0, 'visible_interaction_candidate_count': 0, 'rejection_reason_histogram': {}, 'rejected_candidate_samples': [], 'rejected_candidate_samples_truncated': 0, 'primary_reachable': False, 'audit_reachable': False, 'reachable': False, 'selected_candidate': None}
    objects = getattr(getattr(env, 'core', None), 'objects', {})
    obj = objects.get(resolved) if resolved else None
    if obj is None:
        report['rejection_reason_histogram'] = {'missing_object': 1}
        return report
    report['object_exists'] = True
    parent_role = parent_by_child.get(role)
    target_obj = obj
    if parent_role is not None:
        parent_object = getattr(env, 'role_map', {}).get(parent_role)
        parent = objects.get(parent_object) if parent_object else None
        report.update({'access_mode': 'contained_via_parent', 'parent_role_or_null': parent_role, 'parent_object_or_null': parent_object, 'parent_activation': _parent_activation(parent)})
        if parent is None or not report['parent_activation']['legally_accessible']:
            reason = 'missing_parent' if parent is None else 'closed_parent_without_open_semantics'
            report['rejection_reason_histogram'] = {reason: 1}
            return report
        target_obj = parent
    navigation_obj, anchor = env._navigation_anchor_object_for_target(str(target_obj.name), target_obj)
    target = NavigationTarget.from_aabb_object(navigation_obj)
    config = env.core.config.navigation
    candidates = sample_interaction_candidates(target, config.candidate_count, config.interaction_distance)
    report['navigation_anchor'] = json_deep_copy(anchor)
    report['interaction_face_count'] = len(target.interaction_faces)
    report['interaction_candidate_count'] = len(candidates)
    reasons: Counter[str] = Counter()
    samples: list[Mapping[str, Any]] = []
    selected = None
    for ordinal, goal_xy in enumerate(candidates):
        sample = {'ordinal': ordinal, 'goal_xy': [float(goal_xy[0]), float(goal_xy[1])], 'goal_free': False, 'same_component': False, 'path_planned': False, 'reason': None}
        if not occupancy.is_free(goal_xy):
            sample['reason'] = 'goal_occupied'
        else:
            sample['goal_free'] = True
            report['free_candidate_count'] += 1
            goal_component = occupancy.component_at(goal_xy)
            sample['goal_component'] = int(goal_component)
            if start_component != -1 and goal_component != -1 and (start_component != goal_component):
                sample['reason'] = 'different_component'
            else:
                sample['same_component'] = True
                report['same_component_candidate_count'] += 1
                facing_xy = target.facing_xy_for_goal(goal_xy)
                plan = plan_navigation_to_pose(start_pose, goal_xy, occupancy, config, facing_xy=facing_xy)
                if not plan.success:
                    sample['reason'] = str((plan.check.reasons or ['path_planning_failed'])[0])
                else:
                    sample['path_planned'] = True
                    sample['path_waypoints'] = len(plan.path_xy or [])
                    report['path_planned_candidate_count'] += 1
                    report['reachable_candidate_count'] += 1
                    sample['target_facing_xy'] = [float(facing_xy[0]), float(facing_xy[1])]
                    sample['interaction_face_source'] = _interaction_face_source(target, facing_xy)
                    observation_target = _interaction_target_point(target, facing_xy)
                    observation_pose = env._pose_looking_at_center({'x': float(goal_xy[0]), 'y': float(goal_xy[1]), 'z': start_pose.z, 'yaw': start_pose.yaw, 'pitch': start_pose.pitch}, observation_target)
                    visibility = env._navigation_target_visibility_check(str(target_obj.name), observation_pose)
                    sample['visible_interaction'] = bool(visibility.get('visible'))
                    if sample['visible_interaction']:
                        sample['reason'] = 'reachable'
                        report['visible_interaction_candidate_count'] += 1
                        if selected is None:
                            selected = sample
                    else:
                        sample['reason'] = str(visibility.get('reason') or 'navigation_target_not_visible')
        if sample['reason'] != 'reachable':
            reasons[str(sample['reason'] or 'unknown')] += 1
            if len(samples) < _REJECT_SAMPLE_LIMIT:
                samples.append(sample)
    primary = env._select_visible_navigation_goal(str(target_obj.name))
    strict_primary_reachable = bool(primary.get('success'))
    strict_audit_reachable = report['visible_interaction_candidate_count'] > 0
    audit_topology_reachable = bool(report['path_planned_candidate_count'] > 0 and report['reachable_candidate_count'] > 0)
    primary_topology_reachable = bool(int(primary.get('reachable_candidate_count') or 0) > 0)
    topology_reachable = bool(audit_topology_reachable or primary_topology_reachable)
    initial_visibility_soft_fallback = bool(topology_reachable and (not strict_primary_reachable) and (not strict_audit_reachable))
    audit_candidate_coverage_fallback = bool(strict_primary_reachable and (not strict_audit_reachable))
    admission_reachable = bool(strict_primary_reachable)
    admission_audit_reachable = bool(strict_audit_reachable or audit_candidate_coverage_fallback or initial_visibility_soft_fallback)
    report.update({'rejection_reason_histogram': {key: int(reasons[key]) for key in sorted(reasons)}, 'rejected_candidate_samples': samples, 'rejected_candidate_samples_truncated': max(0, sum(reasons.values()) - len(samples)), 'strict_primary_reachable': strict_primary_reachable, 'strict_audit_reachable': strict_audit_reachable, 'primary_topology_reachable': primary_topology_reachable, 'audit_topology_reachable': audit_topology_reachable, 'topology_reachable': topology_reachable, 'visibility_soft_fallback': initial_visibility_soft_fallback, 'audit_candidate_coverage_fallback': audit_candidate_coverage_fallback, 'primary_reachable': admission_reachable, 'audit_reachable': admission_audit_reachable, 'reachable': admission_reachable and admission_audit_reachable, 'selected_candidate': selected, 'primary_evaluator_summary': {'success': strict_primary_reachable, 'error_type': primary.get('error_type'), 'candidate_count': primary.get('candidate_count'), 'reachable_candidate_count': primary.get('reachable_candidate_count'), 'line_of_sight_clear_candidate_count': primary.get('line_of_sight_clear_candidate_count'), 'candidate_filter_summary': primary.get('candidate_filter_summary')}})
    return report

def _interaction_face_source(target: NavigationTarget, facing_xy: Sequence[float]) -> str | None:
    if not target.interaction_faces:
        return None
    face = min(target.interaction_faces, key=lambda item: math.dist(item.xy_center, (float(facing_xy[0]), float(facing_xy[1]))))
    return str(face.source)

def _interaction_target_point(target: NavigationTarget, facing_xy: Sequence[float]) -> tuple[float, float, float]:
    if not target.interaction_faces:
        return tuple((float(value) for value in target.center))
    face = min(target.interaction_faces, key=lambda item: math.dist(item.xy_center, (float(facing_xy[0]), float(facing_xy[1]))))
    return tuple((float(value) for value in face.center))

def _navigation_map_diagnostics(env: Any, occupancy: Any) -> Mapping[str, Any]:
    occupied = np.asarray(occupancy.occupied, dtype=np.uint8)
    eroded = np.asarray(occupancy.eroded_free, dtype=np.uint8)
    labels = np.asarray(occupancy.component_labels, dtype=np.int32)
    source_map = getattr(getattr(env, 'scene_traversability', None), 'floor_map', None)
    current_source_hash = None if source_map is None else sha256_bytes(np.ascontiguousarray(np.asarray(source_map)).tobytes())
    source_map_sha256 = getattr(env, '_canonical_navigation_source_map_sha256', current_source_hash)
    payload = {'source': str((getattr(occupancy, 'metadata', {}) or {}).get('traversability', {}).get('source') or occupancy.source), 'resolution': float(occupancy.resolution), 'origin': [float(occupancy.origin_xy[0]), float(occupancy.origin_xy[1])], 'shape': [int(occupancy.height), int(occupancy.width)], 'component_count': int((getattr(occupancy, 'metadata', {}) or {}).get('component_count') or 0), 'source_map_sha256': source_map_sha256, 'occupied_sha256': sha256_bytes(np.ascontiguousarray(occupied).tobytes()), 'eroded_free_sha256': sha256_bytes(np.ascontiguousarray(eroded).tobytes()), 'component_labels_sha256': sha256_bytes(np.ascontiguousarray(labels).tobytes()), 'renderer_navigation_context_selection': json_deep_copy(getattr(getattr(env, 'renderer', None), '_trajectory_navigation_context_selection', {}) or {})}
    payload['navigation_map_hash'] = sha256_bytes(canonical_json_bytes(payload))
    return payload

def _alignment_diagnostics(env: Any, occupancy: Any, map_diagnostics: Mapping[str, Any]) -> Mapping[str, Any]:
    reset = env.reset_pose_diagnostics() if callable(getattr(env, 'reset_pose_diagnostics', None)) else {}
    renderer_map = getattr(getattr(env, 'scene_traversability', None), 'floor_map', None)
    renderer_hash = None if renderer_map is None else sha256_bytes(np.ascontiguousarray(np.asarray(renderer_map)).tobytes())
    source_hash = map_diagnostics['source_map_sha256']
    map_consistent = renderer_hash is None and source_hash is None or renderer_hash == source_hash
    world = _renderer_world_alignment(env)
    consistent = bool(map_consistent and world['consistent'])
    return {'consistent': bool(consistent), 'map_consistent': bool(map_consistent), 'env_navigation_source_map_sha256': source_hash, 'renderer_navigation_source_map_sha256': renderer_hash, 'renderer_pose_readback': json_deep_copy(reset.get('renderer_pose_readback') or {}), 'reset_pose_repair': json_deep_copy(reset.get('start_pose_repair') or {}), 'start_pose_room_repair': json_deep_copy(reset.get('start_pose_room_repair') or {}), 'scene_layout_mode': reset.get('scene_layout_mode'), 'world': world}

def _renderer_world_alignment(env: Any) -> Mapping[str, Any]:
    renderer = getattr(env, 'renderer', None)
    catalog_fn = getattr(renderer, 'object_pose_catalog', None)
    if not callable(catalog_fn):
        return {'consistent': True, 'status': 'not_available', 'checked_object_count': 0, 'mismatches': []}
    try:
        catalog = catalog_fn()
    except Exception as exc:
        raise AdmissionEvaluationError(f'renderer pose catalog error: {type(exc).__name__}: {exc}') from exc
    if not isinstance(catalog, Mapping) or not catalog:
        return {'consistent': False, 'status': 'renderer_pose_catalog_empty', 'checked_object_count': 0, 'mismatches': []}
    normalized = {str(key).replace('.', '_').replace('-', '_'): value for key, value in catalog.items()}
    mismatches = []
    checked = 0
    role_objects = sorted(set((str(value) for value in getattr(env, 'role_map', {}).values())))
    for object_id in role_objects:
        obj = getattr(getattr(env, 'core', None), 'objects', {}).get(object_id)
        if str(getattr(obj, 'category', '')).casefold() in _STRUCTURAL_ROLE_CATEGORIES:
            continue
        record = next((normalized[candidate] for candidate in _renderer_pose_match_candidates(object_id, obj) if candidate in normalized), None)
        if obj is None or not isinstance(record, Mapping):
            mismatches.append({'object_id': object_id, 'reason': 'object_missing_on_one_side'})
            continue
        renderer_center = record.get('center')
        comparison_center = getattr(obj, 'center', None)
        if not isinstance(renderer_center, Sequence) or isinstance(renderer_center, (str, bytes)) or len(renderer_center) < 3:
            root = record.get('root_position')
            renderer_center = root if isinstance(root, Sequence) and (not isinstance(root, (str, bytes))) else None
            comparison_center = getattr(obj, 'root_position', None)
        if renderer_center is None or len(renderer_center) < 3:
            mismatches.append({'object_id': object_id, 'reason': 'renderer_center_missing'})
            continue
        if not isinstance(comparison_center, Sequence) or len(comparison_center) < 3:
            mismatches.append({'object_id': object_id, 'reason': 'env_comparison_pose_missing'})
            continue
        checked += 1
        delta = max((abs(float(comparison_center[index]) - float(renderer_center[index])) for index in range(3)))
        if delta > 0.1:
            mismatches.append({'object_id': object_id, 'reason': 'pose_mismatch', 'max_abs_delta_m': delta})
    return {'consistent': not mismatches, 'status': 'matched' if not mismatches else 'mismatch', 'checked_object_count': checked, 'mismatches': mismatches[:24], 'mismatches_truncated': max(0, len(mismatches) - 24)}

def _renderer_pose_match_candidates(object_id: str, obj: Any) -> tuple[str, ...]:
    values = [object_id]
    states = getattr(obj, 'states', {}) or {}
    if isinstance(states, Mapping):
        for field in ('source_scene_object_id', 'source_object_id', 'renderer_name', 'object_id'):
            value = states.get(field)
            if value and (not str(value).startswith('spawnable:')):
                text = str(value)
                values.append(text)
                if '::' in text:
                    values.append(text.split('::', 1)[0])
    return tuple(dict.fromkeys((value.replace('.', '_').replace('-', '_') for value in values)))

def _required_roles(task_spec: Mapping[str, Any]) -> list[str]:
    roles = task_spec.get('roles') if isinstance(task_spec.get('roles'), Mapping) else {}
    required = key_roles_for_validation(task_spec)
    if not required:
        required = set(roles)
    return [role for role in sorted(required) if str((roles.get(role) or {}).get('category') or (roles.get(role) or {}).get('object_type') or '').lower() not in _STRUCTURAL_ROLE_CATEGORIES]

def _contained_parent_roles(task_spec: Mapping[str, Any]) -> Mapping[str, str]:
    out = {}
    for entry in task_spec.get('init') or []:
        if not isinstance(entry, Mapping) or entry.get('predicate') != 'inside':
            continue
        args = entry.get('args') or []
        if isinstance(args, Sequence) and (not isinstance(args, (str, bytes))) and (len(args) >= 2):
            out[str(args[0])] = str(args[1])
    return out

def _parent_activation(parent: Any) -> Mapping[str, Any]:
    if parent is None:
        return {'legally_accessible': False, 'requires_open': False, 'open_now': None, 'open_supported': False}
    abilities = {str(item) for item in getattr(parent, 'abilities', ()) or ()}
    states = getattr(parent, 'states', {}) or {}
    open_supported = 'openable' in abilities or 'open' in states
    open_now = states.get('open') if open_supported else None
    return {'legally_accessible': not open_supported or open_now is True or open_supported, 'requires_open': bool(open_supported and open_now is not True), 'open_now': open_now if isinstance(open_now, bool) else None, 'open_supported': bool(open_supported)}

def _result(cell: Any, *, status: str, error_class: str, failure_code: str | None, disposition: str, map_diagnostics: Mapping[str, Any], alignment: Mapping[str, Any], start_pose: Mapping[str, Any], start_cell: Sequence[int], start_component: int, start_free: bool, roles: Sequence[Mapping[str, Any]], primary: bool, audit: bool) -> Mapping[str, Any]:
    return {'schema_version': 'behavior-admission-preflight.v1', 'stage': 'reachability', 'status': status, 'error_class': error_class, 'failure_code': failure_code, 'disposition': disposition, 'cell': cell.to_mapping(), 'navigation_map': json_deep_copy(map_diagnostics), 'start': {'pose': json_deep_copy(start_pose), 'cell': [int(start_cell[0]), int(start_cell[1])], 'component': int(start_component), 'free': bool(start_free)}, 'required_roles': json_deep_copy(list(roles)), 'evaluators': {'executor_semantics': bool(primary), 'independent_candidate_audit': bool(audit)}, 'alignment': json_deep_copy(alignment)}

def _invalid_infra(cell: Any, env: Any, exc: BaseException) -> Mapping[str, Any]:
    reset = env.reset_pose_diagnostics() if callable(getattr(env, 'reset_pose_diagnostics', None)) else {}
    text = f'{type(exc).__name__} {exc}'.casefold()
    failure_code = 'ADMISSION_INFRA_RENDERER_ERROR' if 'render' in text else 'ADMISSION_INFRA_CONTROLLER_ERROR'
    return {'schema_version': 'behavior-admission-preflight.v1', 'stage': 'reachability', 'status': 'invalid', 'error_class': 'infra_error', 'failure_code': failure_code, 'disposition': 'needs_rerun', 'cell': cell.to_mapping(), 'navigation_map': {}, 'start': {'pose': {}, 'cell': [], 'component': None, 'free': None}, 'required_roles': [], 'evaluators': {'executor_semantics': None, 'independent_candidate_audit': None}, 'alignment': {'consistent': False, 'reset_pose_repair': json_deep_copy(reset.get('start_pose_repair') or {}), 'renderer_pose_readback': json_deep_copy(reset.get('renderer_pose_readback') or {}), 'error_type': type(exc).__name__, 'error_message': _redact_error_message(str(exc))}}

def _redact_error_message(message: str) -> str:
    text = str(message)
    lowered = text.casefold()
    for marker in ('api_key=', 'api-key=', 'authorization:', 'bearer ', 'token='):
        index = lowered.find(marker)
        if index >= 0:
            return text[:index + len(marker)] + '<redacted>'
    return text
__all__ = ['AdmissionEvaluationError', 'canonical_reachability_preflight']
