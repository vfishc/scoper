from __future__ import annotations
import heapq
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from scoper.benchmark.procthor.online_task_spec import validate_online_task_spec
from scoper.benchmark.procthor.spatial_state import DEFAULT_INTERACTION_DISTANCE_M, interaction_ready_from_metadata, object_visible_from_metadata
GridKey = Tuple[int, int]
CandidateKey = Tuple[GridKey, int, int, bool]
StreamCallback = Callable[[dict, Any], Optional[Any]]
PHYSICAL_ACTIONS = {'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject', 'Crouch', 'Stand'}
NAVIGATION_ACTIONS = {'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Crouch', 'Stand'}
INTERACTION_ACTIONS = PHYSICAL_ACTIONS - NAVIGATION_ACTIONS
TRANSLATION_ACTIONS = {'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight'}
_MAX_ORIENTATIONS_PER_KEY = 4
_MAX_GRID_POSITIONS_PER_NAVIGATE = 4
_MAX_LAST_KNOWN_POSITION_FALLBACKS = 3
_MAX_LAST_KNOWN_POSITION_FALLBACK_GRID_STEPS = 3
_INTERACTION_POSE_MARGIN_M = 0.15
_HELD_NAVIGATION_CLEARANCE_RADIUS = 3
_HELD_NAVIGATION_CLEARANCE_MAX_PATH_STEPS = 3
_HELD_NAVIGATION_CLEARANCE_SCORE_MARGIN = 0.5
_HELD_NAVIGATION_CLEARANCE_IGNORED_TYPES = {'Ceiling', 'Doorway', 'Floor', 'Rug', 'Wall'}
_PUT_DOWN_VIEWPORT_PERTURBATIONS = ('look_down_30', 'look_down_60', 'look_up_30', 'look_up_30_back', 'look_down_30_back')
_PUT_DOWN_USE_FORCE_FALLBACK = True
_PUT_DOWN_FORCE_FALLBACK_DISTANCE_MARGIN = 0.25
_FORCE_NAVIGATION_FALLBACK_ACTIONS = {'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft', 'RotateRight'}
_FORCE_OPEN_CONTAINER_FALLBACK_ACTIONS = {'OpenObject'}
_FORCE_ARM_OPEN_CLOSE_COLLISION_ACTIONS = {'OpenObject', 'CloseObject'}
_ARM_COLLIDER_TOKENS = ('robot_arm', 'robot_wrist', 'robot_shoulder', 'robot_hand', 'gripper')
STATUS_SUCCESS = 'success'
STATUS_FAILED = 'failed'
STATUS_SKIPPED = 'skipped'

class ProcTHORLowLevelExecutionError(RuntimeError):
    pass

@dataclass(frozen=True)
class InteractionPose:
    key: GridKey
    position: Dict[str, float]
    rotation_y: int
    horizon: int
    standing: bool
    distance: float
    path_length: int
    score: float
    source: str = 'heuristic_visible_pose'
    turns: int = 0
    target_object_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

class ProcTHORLowLevelExecutor:

    def __init__(self, env: Any, task_spec: Mapping[str, Any], max_nav_steps: int=240, step_size: float=0.25, interaction_distance: float=DEFAULT_INTERACTION_DISTANCE_M, max_pose_candidates: int=80, stream_callback: Optional[StreamCallback]=None, move_execution_step: Optional[float]=None, rotate_execution_degrees: Optional[float]=None, look_execution_degrees: Optional[float]=None, turn_penalty: float=2.0, max_natural_pose_probes: int=2, allow_force_navigation_fallback: bool=True, max_force_navigation_fallbacks_per_tool: int=80, allow_force_open_container_fallback: bool=True, max_force_open_container_fallbacks_per_tool: int=8, allow_force_arm_open_close_collision_fallback: bool=False, max_force_arm_open_close_collision_fallbacks_per_tool: int=8):
        validate_online_task_spec(task_spec)
        self.env = env
        self.task_spec = task_spec
        self.max_nav_steps = int(max_nav_steps)
        self.step_size = float(step_size)
        self.interaction_distance = float(interaction_distance)
        self.max_pose_candidates = int(max_pose_candidates)
        self.stream_callback = stream_callback
        self.move_execution_step = move_execution_step
        self.rotate_execution_degrees = rotate_execution_degrees
        self.look_execution_degrees = look_execution_degrees
        self.turn_penalty = float(turn_penalty)
        self.max_natural_pose_probes = int(max_natural_pose_probes)
        self.allow_force_navigation_fallback = bool(allow_force_navigation_fallback)
        self.max_force_navigation_fallbacks_per_tool = int(max_force_navigation_fallbacks_per_tool)
        self.allow_force_open_container_fallback = bool(allow_force_open_container_fallback)
        self.max_force_open_container_fallbacks_per_tool = int(max_force_open_container_fallbacks_per_tool)
        self.allow_force_arm_open_close_collision_fallback = bool(allow_force_arm_open_close_collision_fallback)
        self.max_force_arm_open_close_collision_fallbacks_per_tool = int(max_force_arm_open_close_collision_fallbacks_per_tool)
        self._force_navigation_fallback_active = False
        self._force_navigation_fallback_count = 0
        self._force_open_container_fallback_count = 0
        self._force_arm_open_close_collision_fallback_count = 0
        self.blocked_edges = set()
        self.action_records: List[dict] = []

    def execute_role_action(self, action: str, role_name: str, label: Optional[str]=None, allow_failure: bool=False) -> dict:
        return self.execute_object_action(action, self.role_object_id(role_name), label=label or '{} {}'.format(action, role_name), target_ref=role_name, allow_failure=allow_failure)

    def execute_object_action(self, action: str, object_id: str, label: Optional[str]=None, target_ref: Optional[str]=None, allow_failure: bool=False) -> dict:
        if action not in INTERACTION_ACTIONS:
            raise ProcTHORLowLevelExecutionError('Unsupported interaction action: {}.'.format(action))
        target_name = target_ref or object_id
        action_label = label or '{} {}'.format(action, target_name)
        tried: set[CandidateKey] = set()
        attempts: List[dict] = []
        prefer_standing = True
        navigated_keys: set[GridKey] = set()
        self._force_navigation_fallback_count = 0
        self._force_open_container_fallback_count = 0
        self._force_arm_open_close_collision_fallback_count = 0
        while len(tried) < self.max_pose_candidates and len(navigated_keys) < _MAX_GRID_POSITIONS_PER_NAVIGATE:
            candidates = self.rank_interaction_poses(object_id, exclude=tried, prefer_standing=prefer_standing)
            if not candidates:
                break
            best = candidates[0]
            same_key_group = [c for c in candidates if c.key == best.key]
            same_key_group = self._sort_by_rotation_cost(same_key_group)[:_MAX_ORIENTATIONS_PER_KEY]
            already_at_key = best.key in navigated_keys
            if not already_at_key:
                tried.add(_candidate_key(best))
                attempt = {'pose': best.to_dict(), 'navigation': None, 'visible_before_action': False, 'action_success': False, 'error_message': ''}
                attempts.append(attempt)
                try:
                    attempt['navigation'] = self._navigate_to_grid_key(best, label='navigate for {}'.format(action_label))
                except ProcTHORLowLevelExecutionError as exc:
                    attempt['error_message'] = str(exc)
                    for c in same_key_group:
                        tried.add(_candidate_key(c))
                    continue
                navigated_keys.add(best.key)
            for pose in same_key_group:
                if _candidate_key(pose) in tried and (not (pose is best and (not already_at_key))):
                    continue
                tried.add(_candidate_key(pose))
                if already_at_key or pose is not best:
                    attempt = {'pose': pose.to_dict(), 'navigation': {'status': 'reused_position'}, 'visible_before_action': False, 'action_success': False, 'error_message': ''}
                    attempts.append(attempt)
                else:
                    attempt = attempts[-1]
                view_clearance = self._prepare_held_view_clearance(action_label)
                if view_clearance is not None:
                    attempt['held_view_clearance'] = view_clearance
                try:
                    self._rotate_to_yaw(pose.rotation_y, action_label)
                    self._set_standing(pose.standing, action_label)
                    self._look_to_horizon(pose.horizon, action_label)
                except ProcTHORLowLevelExecutionError:
                    attempt['error_message'] = 'orientation adjustment failed'
                    continue
                visible = self.is_object_visible(object_id)
                attempt['visible_before_action'] = visible
                if not visible:
                    attempt['error_message'] = 'target object is not visible from candidate pose'
                    continue
                event = self._step_physical({'action': action, 'objectId': object_id}, label=action_label, role_name=target_ref, allow_failure=True)
                success = bool(event.metadata.get('lastActionSuccess', False))
                if not success and action == 'PutObject':
                    event, success = self._retry_put_object_after_failure(event=event, object_id=object_id, label=action_label, role_name=target_ref)
                attempt['action_success'] = success
                attempt['error_message'] = event.metadata.get('errorMessage') or ''
                if success:
                    return {'status': STATUS_SUCCESS, 'action': action, 'target': target_name, 'object_id': object_id, 'label': action_label, 'attempts': attempts, 'final_pose': self.agent_pose()}
        result = {'status': STATUS_FAILED, 'action': action, 'target': target_name, 'object_id': object_id, 'label': action_label, 'attempts': attempts, 'final_pose': self.agent_pose(), 'message': "No candidate pose produced a successful '{}' for target '{}'.".format(action, target_name)}
        if allow_failure:
            return result
        raise ProcTHORLowLevelExecutionError(result['message'])

    def navigate_to_role(self, role_name: str, label: Optional[str]=None, allow_failure: bool=False) -> dict:
        return self.navigate_to_object_id(self.role_object_id(role_name), label=label or 'navigate to {}'.format(role_name), target_ref=role_name, allow_failure=allow_failure)

    def navigate_to_object_id(self, object_id: str, label: Optional[str]=None, target_ref: Optional[str]=None, allow_failure: bool=False, prefer_standing: Optional[bool]=None) -> dict:
        target_label = label or 'navigate to {}'.format(target_ref or object_id)
        if self.is_object_interaction_ready(object_id):
            return {'status': STATUS_SUCCESS, 'action': 'navigate_to', 'target': target_ref or object_id, 'object_id': object_id, 'interaction_ready': True, 'interaction_pose_source': 'current_pose', 'attempts': [], 'final_pose': self.agent_pose()}
        tried: set[CandidateKey] = set()
        attempts: List[dict] = []
        navigated_keys: set[GridKey] = set()
        self._force_navigation_fallback_count = 0
        self._force_open_container_fallback_count = 0
        self._force_arm_open_close_collision_fallback_count = 0
        while len(tried) < self.max_pose_candidates and len(navigated_keys) < _MAX_GRID_POSITIONS_PER_NAVIGATE:
            resolved_prefer_standing = prefer_standing
            if resolved_prefer_standing is None:
                resolved_prefer_standing = True
            candidates = self.rank_interaction_poses(object_id, exclude=tried, prefer_standing=resolved_prefer_standing)
            if not candidates:
                break
            best = candidates[0]
            same_key_group = [c for c in candidates if c.key == best.key]
            same_key_group = self._sort_by_rotation_cost(same_key_group)[:_MAX_ORIENTATIONS_PER_KEY]
            already_at_key = best.key in navigated_keys
            if not already_at_key:
                tried.add(_candidate_key(best))
                attempt = {'pose': best.to_dict(), 'navigation': None, 'visible_after_navigation': False, 'interaction_ready': False, 'error_message': ''}
                attempts.append(attempt)
                try:
                    attempt['navigation'] = self._navigate_to_grid_key(best, label=target_label)
                except ProcTHORLowLevelExecutionError as exc:
                    attempt['error_message'] = str(exc)
                    for c in same_key_group:
                        tried.add(_candidate_key(c))
                    continue
                navigated_keys.add(best.key)
            for pose in same_key_group:
                if _candidate_key(pose) in tried and (not (pose is best and (not already_at_key))):
                    continue
                tried.add(_candidate_key(pose))
                if already_at_key or pose is not best:
                    attempt = {'pose': pose.to_dict(), 'navigation': {'status': 'reused_position'}, 'visible_after_navigation': False, 'interaction_ready': False, 'error_message': ''}
                    attempts.append(attempt)
                else:
                    attempt = attempts[-1]
                view_clearance = self._prepare_held_view_clearance(target_label)
                if view_clearance is not None:
                    attempt['held_view_clearance'] = view_clearance
                try:
                    self._rotate_to_yaw(pose.rotation_y, target_label)
                    self._set_standing(pose.standing, target_label)
                    self._look_to_horizon(pose.horizon, target_label)
                except ProcTHORLowLevelExecutionError:
                    attempt['error_message'] = 'orientation adjustment failed'
                    continue
                visible = self.is_object_visible(object_id)
                interaction_ready = self.is_object_interaction_ready(object_id)
                attempt['visible_after_navigation'] = visible
                attempt['interaction_ready'] = interaction_ready
                if interaction_ready:
                    return {'status': STATUS_SUCCESS, 'action': 'navigate_to', 'target': target_ref or object_id, 'object_id': object_id, 'interaction_ready': True, 'interaction_pose_source': pose.source, 'attempts': attempts, 'final_pose': self.agent_pose()}
                attempt['error_message'] = 'target object is visible but outside the interaction radius' if visible else 'target object is not visible from candidate pose'
        result = {'status': STATUS_FAILED, 'action': 'navigate_to', 'target': target_ref or object_id, 'object_id': object_id, 'interaction_ready': False, 'attempts': attempts, 'final_pose': self.agent_pose(), 'message': "No candidate pose made target '{}' visible.".format(target_ref or object_id)}
        if allow_failure:
            return result
        raise ProcTHORLowLevelExecutionError(result['message'])

    def navigate_to_position(self, position: Mapping[str, Any], label: str='navigate to position', target_ref: Optional[str]=None, resolved_via: str='position', object_id: Optional[str]=None, allow_failure: bool=False) -> dict:
        if 'x' not in position or 'z' not in position:
            raise ProcTHORLowLevelExecutionError('Position navigation requires x/z coordinates.')
        self._force_navigation_fallback_count = 0
        self._force_open_container_fallback_count = 0
        self._force_arm_open_close_collision_fallback_count = 0
        reachable = self.reachable_positions()
        points_by_key = _points_by_key(reachable, self.step_size)
        target_key = _nearest_key(position, points_by_key, self.step_size)
        if target_key is None:
            result = {'status': STATUS_FAILED, 'action': 'navigate_to', 'target': target_ref or label, 'resolved_via': resolved_via, 'interaction_ready': False, 'object_visibility_guaranteed': False, 'final_pose': self.agent_pose(), 'message': "No reachable position found for target '{}'.".format(target_ref or label)}
            if allow_failure:
                return result
            raise ProcTHORLowLevelExecutionError(result['message'])
        candidate_keys = [target_key]
        if resolved_via == 'last_known_position' and object_id is not None:
            nearby_keys = sorted((key for key in points_by_key if key != target_key and _manhattan(key, target_key) <= _MAX_LAST_KNOWN_POSITION_FALLBACK_GRID_STEPS), key=lambda key: (_squared_xz_distance(position, points_by_key[key]), _manhattan(key, target_key), key))
            candidate_keys.extend(nearby_keys[:_MAX_LAST_KNOWN_POSITION_FALLBACKS])
        navigation = None
        navigation_attempts = []
        first_error = None
        for candidate_index, candidate_key in enumerate(candidate_keys):
            target_point = points_by_key[candidate_key]
            pose = InteractionPose(key=candidate_key, position={'x': float(target_point.get('x', 0.0)), 'y': float(target_point.get('y', self.agent_position().get('y', 0.0))), 'z': float(target_point.get('z', 0.0))}, rotation_y=_nearest_cardinal_yaw(_yaw_towards(target_point, position)), horizon=_desired_horizon(target_point, position) if object_id is not None else 0, standing=True, distance=float(_xz_distance(target_point, position)), path_length=0, score=0.0, source=resolved_via)
            try:
                navigation = self.navigate_to_pose(pose, label=label, stream_source='navigation_stream')
                navigation_attempts.append({'candidate_key': list(candidate_key), 'status': STATUS_SUCCESS})
                break
            except ProcTHORLowLevelExecutionError as exc:
                error_message = str(exc)
                navigation_attempts.append({'candidate_key': list(candidate_key), 'status': STATUS_FAILED, 'message': error_message})
                if candidate_index == 0:
                    first_error = exc
                    if not _looks_like_static_final_rotation_collision(error_message):
                        break
                elif not _looks_like_static_final_rotation_collision(error_message):
                    break
        if navigation is None:
            error = first_error or ProcTHORLowLevelExecutionError('Position navigation failed.')
            result = {'status': STATUS_FAILED, 'action': 'navigate_to', 'target': target_ref or label, 'resolved_via': resolved_via, 'interaction_ready': False, 'object_visibility_guaranteed': False, 'final_pose': self.agent_pose(), 'message': str(error)}
            if len(navigation_attempts) > 1:
                result['last_known_position_navigation_attempts'] = navigation_attempts
            if allow_failure:
                return result
            raise error
        object_visible = bool(object_id and self.is_object_visible(object_id))
        object_ready = bool(object_id and self.is_object_interaction_ready(object_id))
        result = {'status': STATUS_SUCCESS, 'action': 'navigate_to', 'target': target_ref or label, 'resolved_via': resolved_via, 'interaction_ready': object_ready, 'object_visibility_guaranteed': False, 'object_visible': object_visible, 'target_position': {'x': float(position.get('x', 0.0)), 'y': float(position.get('y', self.agent_position().get('y', 0.0))), 'z': float(position.get('z', 0.0))}, 'navigation': navigation, 'final_pose': self.agent_pose()}
        if len(navigation_attempts) > 1:
            result['last_known_position_navigation_attempts'] = navigation_attempts
        return result

    def scan_in_place(self, num_views: int=8, label: str='scan') -> dict:
        num_views = int(num_views)
        if num_views <= 0:
            raise ProcTHORLowLevelExecutionError('scan num_views must be positive.')
        from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding
        evidence_scan = task_uses_evidence_binding(getattr(self.env, 'current_task_spec', None))
        if evidence_scan:
            num_views = min(num_views, 8)
        clearance_directions = iter(('MoveBack', 'MoveLeft', 'MoveRight', 'MoveAhead'))
        horizon_delta = round((45.0 - _normalize_horizon(self.agent_pose().get('cameraHorizon', 0.0))) * 10.0) / 10.0
        if abs(horizon_delta) > 1e-06:
            self._step_physical({'action': 'LookDown' if horizon_delta > 0 else 'LookUp', 'degrees': abs(horizon_delta)}, label=label, source='scan', interpolate=False)
        degrees = 360.0 / float(num_views)
        frame_count = 0
        visible_object_ids: set[str] = set()
        clearance_used = False
        clearance_attempts: List[dict] = []
        for _index in range(num_views):
            event = self._step_physical({'action': 'RotateRight', 'degrees': degrees}, label=label, source='scan', allow_failure=True)
            success = bool(event.metadata.get('lastActionSuccess', False))
            visible_object_ids.update(_visible_object_ids(event))
            rotation_error = event.metadata.get('errorMessage') or ''
            if evidence_scan:
                while not success and _looks_like_viewpoint_collision(rotation_error):
                    direction = next(clearance_directions, None)
                    if direction is None:
                        break
                    clearance_used = True
                    moved, attempts = self._try_scan_clearance(label, directions=(direction,))
                    clearance_attempts.extend(attempts)
                    if not moved:
                        continue
                    event = self._step_physical({'action': 'RotateRight', 'degrees': degrees}, label=label, source='scan', allow_failure=True)
                    success = bool(event.metadata.get('lastActionSuccess', False))
                    rotation_error = event.metadata.get('errorMessage') or ''
                    visible_object_ids.update(_visible_object_ids(event))
            if not success and (not evidence_scan) and (not clearance_used) and _looks_like_viewpoint_collision(rotation_error):
                clearance_used = True
                moved, clearance_attempts = self._try_scan_clearance(label)
                if moved:
                    event = self._step_physical({'action': 'RotateRight', 'degrees': degrees}, label=label, source='scan', allow_failure=True)
                    success = bool(event.metadata.get('lastActionSuccess', False))
            if not success:
                error_message = event.metadata.get('errorMessage') or 'RotateRight during scan failed.'
                return {'status': STATUS_SUCCESS if evidence_scan and frame_count else STATUS_FAILED, 'action': 'scan', 'partial_observation': bool(evidence_scan and frame_count), 'motion_failed': True, 'num_views': num_views, 'num_views_completed': frame_count, 'frame_count': frame_count, 'final_pose': self.agent_pose(), 'error_message': error_message, 'clearance_attempts': clearance_attempts, 'clearance_used': clearance_used, 'metadata_visible_object_ids': sorted(visible_object_ids)}
            frame_count += 1
        result = {'status': STATUS_SUCCESS, 'action': 'scan', 'num_views': num_views, 'num_views_completed': frame_count, 'frame_count': frame_count, 'final_pose': self.agent_pose(), 'metadata_visible_object_ids': sorted(visible_object_ids)}
        if clearance_used:
            result['clearance_attempts'] = clearance_attempts
            result['clearance_used'] = True
        return result

    def _try_scan_clearance(self, label: str, *, directions=('MoveBack', 'MoveLeft', 'MoveRight', 'MoveAhead')) -> tuple[bool, List[dict]]:
        attempts: List[dict] = []
        force_navigation_active = self._force_navigation_fallback_active
        self._force_navigation_fallback_active = False
        try:
            for action in directions:
                event = self._step_physical({'action': action, 'moveMagnitude': self.step_size}, label='{} clearance'.format(label), source='scan_clearance', allow_failure=True)
                success = bool(event.metadata.get('lastActionSuccess', False))
                attempts.append({'action': action, 'success': success, 'error_message': event.metadata.get('errorMessage') or ''})
                if success:
                    return (True, attempts)
        finally:
            self._force_navigation_fallback_active = force_navigation_active
        return (False, attempts)

    def look_for_object(self, object_id: str, label: str='look_for') -> dict:
        obj = self._objects_by_id_safe().get(object_id)
        if obj is None:
            return {'status': STATUS_FAILED, 'action': 'look_for', 'object_id': object_id, 'error_message': "Object '{}' not found in scene.".format(object_id), 'final_pose': self.agent_pose(), 'object_visible': False}
        obj_pos = obj.get('position') or {}
        agent_pose = self.agent_pose()
        agent_pos = agent_pose.get('position') or {}
        if not all((k in obj_pos for k in ('x', 'y', 'z'))):
            return {'status': STATUS_FAILED, 'action': 'look_for', 'object_id': object_id, 'error_message': 'Object position incomplete.', 'final_pose': agent_pose, 'object_visible': False}
        dx = float(obj_pos['x']) - float(agent_pos.get('x', 0.0))
        dz = float(obj_pos['z']) - float(agent_pos.get('z', 0.0))
        desired_yaw = math.degrees(math.atan2(dx, dz)) % 360.0
        current_yaw = float(agent_pose.get('rotation', {}).get('y', 0.0)) % 360.0
        yaw_diff = (desired_yaw - current_yaw + 180.0) % 360.0 - 180.0
        if abs(yaw_diff) > 5.0:
            if yaw_diff > 0:
                yaw_event = self._step_physical({'action': 'RotateRight', 'degrees': abs(yaw_diff)}, label='{}_yaw'.format(label), source='look_for', allow_failure=True)
            else:
                yaw_event = self._step_physical({'action': 'RotateLeft', 'degrees': abs(yaw_diff)}, label='{}_yaw'.format(label), source='look_for', allow_failure=True)
            if not bool(yaw_event.metadata.get('lastActionSuccess', False)):
                return {'status': STATUS_FAILED, 'action': 'look_for', 'object_id': object_id, 'object_visible': False, 'motion_failed': True, 'error_message': yaw_event.metadata.get('errorMessage') or 'look_for yaw motion failed.', 'final_pose': self.agent_pose(), 'desired_yaw': desired_yaw}
        camera_y = float(agent_pos.get('y', 0.9)) + 0.675
        xz_dist = max(0.01, math.hypot(dx, dz))
        desired_horizon = -math.degrees(math.atan2(float(obj_pos['y']) - camera_y, xz_dist))
        desired_horizon = max(-30.0, min(60.0, desired_horizon))
        current_horizon = float(agent_pose.get('cameraHorizon', 0.0))
        horizon_diff = desired_horizon - current_horizon
        if abs(horizon_diff) > 5.0:
            if horizon_diff > 0:
                horizon_event = self._step_physical({'action': 'LookDown', 'degrees': abs(horizon_diff)}, label='{}_horizon'.format(label), source='look_for', allow_failure=True)
            else:
                horizon_event = self._step_physical({'action': 'LookUp', 'degrees': abs(horizon_diff)}, label='{}_horizon'.format(label), source='look_for', allow_failure=True)
            if not bool(horizon_event.metadata.get('lastActionSuccess', False)):
                return {'status': STATUS_FAILED, 'action': 'look_for', 'object_id': object_id, 'object_visible': False, 'motion_failed': True, 'error_message': horizon_event.metadata.get('errorMessage') or 'look_for horizon motion failed.', 'final_pose': self.agent_pose(), 'desired_yaw': desired_yaw, 'desired_horizon': desired_horizon}
        obj_after = self._objects_by_id_safe().get(object_id, {})
        visible = bool(obj_after.get('visible', False))
        return {'status': STATUS_SUCCESS if visible else STATUS_FAILED, 'action': 'look_for', 'object_id': object_id, 'object_visible': visible, 'motion_failed': False, 'final_pose': self.agent_pose(), 'desired_yaw': desired_yaw, 'desired_horizon': desired_horizon}

    def _objects_by_id_safe(self) -> Dict[str, Mapping[str, Any]]:
        try:
            meta = self.metadata()
            return {obj['objectId']: obj for obj in meta.get('objects', []) if isinstance(obj, Mapping) and isinstance(obj.get('objectId'), str)}
        except Exception:
            return {}

    def _navigate_to_grid_key(self, pose: InteractionPose, label: str, stream_source: str='navigation_stream') -> dict:
        old_force_active = self._force_navigation_fallback_active
        self._force_navigation_fallback_active = True
        nav_steps = 0
        clearance_moves: List[dict] = []
        waypoint_fallbacks: List[dict] = []
        try:
            reachable = self.reachable_positions()
            points_by_key = _points_by_key(reachable, self.step_size)
            clearance = self._move_to_held_navigation_clearance(points_by_key, target_key=pose.key, label=label, source=stream_source)
            if clearance is not None:
                clearance_moves.append(clearance)
                nav_steps += int(clearance.get('num_steps', 0))
            nav_steps += self._look_to_horizon(0, label, source=stream_source)
            while nav_steps < self.max_nav_steps:
                current_key = _nearest_key(self.agent_position(), points_by_key, self.step_size)
                if current_key is None:
                    raise ProcTHORLowLevelExecutionError('Agent pose cannot be matched to reachable positions.')
                if current_key == pose.key:
                    break
                if pose.key not in points_by_key:
                    raise ProcTHORLowLevelExecutionError('Candidate pose is no longer reachable.')
                path = _astar_path(points_by_key, current_key, pose.key, self._planning_blocked_edges(points_by_key, exclude_object_id=pose.target_object_id), turn_penalty=self.turn_penalty)
                if path is None or len(path) < 2:
                    raise ProcTHORLowLevelExecutionError('No unblocked path remains to candidate pose.')
                segments = _merge_straight_segments(path)
                blocked = False
                for segment_yaw, segment_keys in segments:
                    if nav_steps >= self.max_nav_steps:
                        break
                    try:
                        nav_steps += self._rotate_to_yaw(segment_yaw, label, source=stream_source)
                    except ProcTHORLowLevelExecutionError as exc:
                        clearance = None
                        if self._is_held_navigation_collision(str(exc)):
                            clearance = self._move_to_held_navigation_clearance(points_by_key, target_key=pose.key, label=label, source=stream_source, force=True)
                        if clearance is not None:
                            clearance_moves.append(clearance)
                            nav_steps += int(clearance.get('num_steps', 0))
                            blocked = True
                            break
                        fallback_current = _nearest_key(self.agent_position(), points_by_key, self.step_size)
                        fallback = self._teleport_to_adjacent_waypoint(points_by_key, current_key=fallback_current, next_key=segment_keys[0], destination_key=pose.key, rotation_y=segment_yaw, label=label, stream_source=stream_source, physical_error=str(exc))
                        if fallback is None:
                            raise
                        waypoint_fallbacks.append(fallback)
                        nav_steps += int(fallback.get('num_nav_steps', 0))
                        if fallback.get('status') != STATUS_SUCCESS:
                            self.blocked_edges.add(_edge_key(fallback_current, segment_keys[0]))
                        blocked = True
                        break
                    if nav_steps >= self.max_nav_steps:
                        break
                    for i, step_key in enumerate(segment_keys):
                        if nav_steps >= self.max_nav_steps:
                            break
                        prev_key = _prev_key_for_segment(path, step_key) if i == 0 else segment_keys[i - 1]
                        event = self._step_physical({'action': 'MoveAhead', 'moveMagnitude': self.step_size}, label=label, source=stream_source, allow_failure=True)
                        nav_steps += 1
                        if not bool(event.metadata.get('lastActionSuccess', False)):
                            actual_current = _nearest_key(self.agent_position(), points_by_key, self.step_size)
                            if actual_current == step_key:
                                continue
                            fallback_current = actual_current or prev_key
                            fallback = self._teleport_to_adjacent_waypoint(points_by_key, current_key=fallback_current, next_key=step_key, destination_key=pose.key, rotation_y=segment_yaw, label=label, stream_source=stream_source, physical_error=event.metadata.get('errorMessage') or 'MoveAhead failed')
                            if fallback is not None:
                                waypoint_fallbacks.append(fallback)
                                nav_steps += int(fallback.get('num_nav_steps', 0))
                            if fallback is None or fallback.get('status') != STATUS_SUCCESS:
                                self.blocked_edges.add(_edge_key(fallback_current, step_key))
                            blocked = True
                            break
                    if blocked:
                        break
            current_key = _nearest_key(self.agent_position(), points_by_key, self.step_size)
            if current_key != pose.key:
                raise ProcTHORLowLevelExecutionError('Navigation step budget exhausted before reaching candidate pose.')
            return {'status': STATUS_SUCCESS, 'num_nav_steps': nav_steps, 'target_pose': pose.to_dict(), 'final_pose': self.agent_pose(), **({'held_navigation_clearance': clearance_moves} if clearance_moves else {}), **({'waypoint_fallbacks': waypoint_fallbacks} if waypoint_fallbacks else {})}
        finally:
            self._force_navigation_fallback_active = old_force_active

    def _teleport_to_adjacent_waypoint(self, points_by_key: Mapping[GridKey, Mapping[str, Any]], *, current_key: Optional[GridKey], next_key: GridKey, destination_key: GridKey, rotation_y: float, label: str, stream_source: str, physical_error: str) -> Optional[dict]:
        if not self._manipulathor_visual_attach_enabled():
            return None
        if current_key is None or current_key not in points_by_key or next_key not in points_by_key or (_manhattan(current_key, next_key) != 1):
            return None
        current_pose = self.agent_pose()
        preserved_horizon = _normalize_horizon(current_pose.get('cameraHorizon', 0.0))
        current_standing = current_pose.get('isStanding')
        preserved_standing = True if current_standing is None else bool(current_standing)
        waypoint_position = dict(points_by_key[next_key])
        from scoper.benchmark.procthor.export.execution_audit import audit_controller_classifications
        with audit_controller_classifications(self.env, 'visual_attach_navigation_fallback'):
            event = self.env.controller.step(action='TeleportFull', position=waypoint_position, rotation={'x': 0.0, 'y': float(rotation_y), 'z': 0.0}, horizon=float(preserved_horizon), standing=preserved_standing, forceAction=True)
        success = bool(event.metadata.get('lastActionSuccess', False))
        actual_key = _nearest_key(self.agent_position(), points_by_key, self.step_size)
        if success and actual_key != next_key:
            success = False
            event.metadata['lastActionSuccess'] = False
            event.metadata['errorMessage'] = 'TeleportFull did not arrive at the requested adjacent waypoint.'
        action_record = {'action': 'TeleportFull', 'executed_action': 'TeleportFull', 'label': '{} [visual_navigation_teleport]'.format(label), 'success': success, 'error_message': event.metadata.get('errorMessage') or '', 'agent_pose': self.agent_pose(), 'execution_substep': 1, 'execution_substeps': 1, 'source': stream_source, 'forceAction': True, 'visual_attach_navigation_fallback': True, 'adjacent_waypoint_fallback': True, 'from_waypoint': list(current_key), 'to_waypoint': list(next_key), 'destination_is_waypoint': next_key == destination_key, 'physical_navigation_error': physical_error, 'preserved_camera_horizon': preserved_horizon, 'metadata_visible_object_ids': _visible_object_ids(event)}
        self.action_records.append(action_record)
        if self.stream_callback is not None:
            callback_result = self.stream_callback(action_record, event)
            if isinstance(callback_result, Mapping):
                action_record.update(callback_result)
            elif isinstance(callback_result, str):
                action_record['image_path'] = callback_result
        return {'status': STATUS_SUCCESS if success else STATUS_FAILED, 'num_nav_steps': 1, 'final_pose': self.agent_pose(), 'visual_attach_navigation_waypoint_teleport': True, 'adjacent_waypoint': True, 'from_waypoint': list(current_key), 'to_waypoint': list(next_key), 'destination_is_waypoint': next_key == destination_key, 'physical_navigation_error': physical_error, 'preserved_camera_horizon': preserved_horizon, **({} if success else {'error_message': action_record['error_message']})}

    def interact_with_role(self, action: str, role_name: str, label: Optional[str]=None, allow_failure: bool=False) -> dict:
        return self.interact_with_object_id(action, self.role_object_id(role_name), label=label or '{} {}'.format(action, role_name), target_ref=role_name, allow_failure=allow_failure)

    def interact_with_object_id(self, action: str, object_id: str, label: Optional[str]=None, target_ref: Optional[str]=None, allow_failure: bool=False) -> dict:
        if action not in INTERACTION_ACTIONS:
            raise ProcTHORLowLevelExecutionError('Unsupported interaction action: {}.'.format(action))
        action_label = label or '{} {}'.format(action, target_ref or object_id)
        event = self._step_physical({'action': action, 'objectId': object_id}, label=action_label, role_name=target_ref, allow_failure=True)
        success = bool(event.metadata.get('lastActionSuccess', False))
        if not success and action == 'PutObject':
            event, success = self._retry_put_object_after_failure(event=event, object_id=object_id, label=action_label, role_name=target_ref)
        result = {'status': STATUS_SUCCESS if success else STATUS_FAILED, 'action': action, 'target': target_ref or object_id, 'object_id': object_id, 'success': success, 'error_message': event.metadata.get('errorMessage') or '', 'final_pose': self.agent_pose()}
        if not success and (not allow_failure):
            raise ProcTHORLowLevelExecutionError("Action '{}' failed: {}".format(action_label, result['error_message'] or 'unknown error'))
        return result

    def open_role_if_needed(self, role_name: str, label: Optional[str]=None) -> dict:
        obj = self.object_by_role(role_name)
        if not bool(obj.get('openable', False)):
            return {'status': STATUS_SKIPPED, 'action': 'OpenObject', 'role': role_name, 'object_id': obj.get('objectId'), 'message': 'object is not openable'}
        if obj.get('isOpen') is not False:
            return {'status': STATUS_SKIPPED, 'action': 'OpenObject', 'role': role_name, 'object_id': obj.get('objectId'), 'message': 'object is already open'}
        return self.execute_role_action('OpenObject', role_name, label=label or 'open {}'.format(role_name))

    def navigate_to_pose(self, pose: InteractionPose, label: str='navigate', stream_source: str='navigation_stream') -> dict:
        navigation = self._navigate_to_grid_key(pose, label=label, stream_source=stream_source)
        nav_steps = int(navigation.get('num_nav_steps', 0))
        nav_steps += self._rotate_to_yaw(pose.rotation_y, label, source=stream_source)
        nav_steps += self._set_standing(pose.standing, label, source=stream_source)
        nav_steps += self._look_to_horizon(pose.horizon, label, source=stream_source)
        navigation['num_nav_steps'] = nav_steps
        navigation['final_pose'] = self.agent_pose()
        return navigation

    def rank_interaction_poses(self, object_id: str, exclude: Optional[Iterable[CandidateKey]]=None, prefer_standing: Optional[bool]=None) -> List[InteractionPose]:
        obj = self.object_by_id(object_id)
        object_position = obj.get('position') or {}
        reachable = self.reachable_positions()
        points_by_key = _points_by_key(reachable, self.step_size)
        start_key = _nearest_key(self.agent_position(), points_by_key, self.step_size)
        if start_key is None:
            return []
        excluded = set(exclude or [])
        interactable = self._get_interactable_poses(object_id)
        simulator_poses: List[InteractionPose] = []
        if interactable:
            simulator_poses = self._rank_simulator_interactable_poses(interactable, points_by_key, start_key, object_position, excluded, prefer_standing=prefer_standing, target_object_id=object_id)
            simulator_poses = [pose for pose in simulator_poses if pose.distance <= max(0.0, self.interaction_distance - _INTERACTION_POSE_MARGIN_M)]
        keyed_candidates = []
        pose_distance_limit = max(0.0, self.interaction_distance - _INTERACTION_POSE_MARGIN_M)
        for key, point in points_by_key.items():
            distance = _xz_distance(point, object_position)
            if distance <= pose_distance_limit:
                keyed_candidates.append((key, point, distance))
        if not keyed_candidates:
            nearest = sorted(((key, point, _xz_distance(point, object_position)) for key, point in points_by_key.items()), key=lambda item: item[2])
            keyed_candidates = nearest[:min(len(nearest), self.max_pose_candidates)]
        heuristic_poses: List[InteractionPose] = []
        seen_candidates = {_candidate_key(pose) for pose in simulator_poses}
        used_natural_probe_keys = set()
        for key, point, distance in keyed_candidates:
            path = _astar_path(points_by_key, start_key, key, self._planning_blocked_edges(points_by_key, exclude_object_id=object_id), turn_penalty=self.turn_penalty)
            if path is None:
                continue
            base_yaw = _nearest_cardinal_yaw(_yaw_towards(point, object_position))
            yaws = _ordered_yaws(base_yaw)
            desired_horizon = _desired_horizon(point, object_position)
            horizons = _ordered_horizons(desired_horizon)
            path_length = max(0, len(path) - 1)
            path_turns = _path_num_turns(path)
            for yaw_index, yaw in enumerate(yaws):
                for horizon_index, horizon in enumerate(horizons):
                    candidate = InteractionPose(key=key, position={'x': float(point.get('x', 0.0)), 'y': float(point.get('y', self.agent_position().get('y', 0.0))), 'z': float(point.get('z', 0.0))}, rotation_y=int(yaw), horizon=int(horizon), standing=True, distance=float(distance), path_length=path_length, turns=path_turns, score=path_length * 10.0 + path_turns * 4.0 + distance + yaw_index * 8.0 + horizon_index * 0.1 + _natural_horizon_penalty(horizon, desired_horizon) + _source_penalty('heuristic_visible_pose'), source='heuristic_visible_pose', target_object_id=object_id)
                    candidate_key = _candidate_key(candidate)
                    if candidate_key in excluded:
                        used_natural_probe_keys.add(candidate_key)
                    elif candidate_key not in seen_candidates:
                        heuristic_poses.append(candidate)
                        seen_candidates.add(candidate_key)
        if simulator_poses and len(used_natural_probe_keys) >= self.max_natural_pose_probes:
            heuristic_poses = []
        poses = simulator_poses + heuristic_poses
        poses.sort(key=lambda pose: pose.score)
        return poses[:min(16, self.max_pose_candidates)]

    def _infer_opening_yaw(self, interactable_poses: Sequence[Mapping[str, Any]], object_position: Optional[Mapping[str, Any]]=None) -> Optional[float]:
        if not interactable_poses or len(interactable_poses) < 8:
            return None
        if object_position is None or 'x' not in object_position or 'z' not in object_position:
            return None
        mean_x = sum((float(p.get('x', 0.0)) for p in interactable_poses)) / len(interactable_poses)
        mean_z = sum((float(p.get('z', 0.0)) for p in interactable_poses)) / len(interactable_poses)
        dx = mean_x - float(object_position['x'])
        dz = mean_z - float(object_position['z'])
        if abs(dx) < 0.15 and abs(dz) < 0.15:
            return None
        import math
        agent_face_yaw = math.degrees(math.atan2(-dx, -dz))
        return float(_nearest_cardinal_yaw(agent_face_yaw))

    def _get_interactable_poses(self, object_id: str) -> List[dict]:
        event = self.env.controller.step(action='GetInteractablePoses', objectId=object_id)
        if not bool(event.metadata.get('lastActionSuccess', True)):
            return []
        poses = event.metadata.get('actionReturn') or []
        return [dict(pose) for pose in poses if isinstance(pose, Mapping)]

    def _rank_simulator_interactable_poses(self, interactable_poses: Sequence[Mapping[str, Any]], points_by_key: Mapping[GridKey, Mapping[str, Any]], start_key: GridKey, object_position: Mapping[str, Any], excluded: Iterable[CandidateKey], prefer_standing: Optional[bool]=None, target_object_id: Optional[str]=None) -> List[InteractionPose]:
        excluded_keys = set(excluded)
        opening_yaw = self._infer_opening_yaw(interactable_poses, object_position)
        if opening_yaw is not None:
            cardinal_opening = _nearest_cardinal_yaw(opening_yaw)
            interactable_poses = [p for p in interactable_poses if _nearest_cardinal_yaw(float(p.get('rotation', 0.0))) == cardinal_opening]
        poses: List[InteractionPose] = []
        for raw_index, raw_pose in enumerate(interactable_poses):
            if 'x' not in raw_pose or 'z' not in raw_pose:
                continue
            key = _nearest_key(raw_pose, points_by_key, self.step_size)
            if key is None:
                continue
            path = _astar_path(points_by_key, start_key, key, self._planning_blocked_edges(points_by_key, exclude_object_id=target_object_id), turn_penalty=self.turn_penalty)
            if path is None:
                continue
            point = points_by_key[key]
            rotation_y = _nearest_cardinal_yaw(float(raw_pose.get('rotation', 0.0)))
            horizon = int(_clamp(round(float(raw_pose.get('horizon', 0.0)) / 30.0) * 30, -30, 60))
            standing = bool(raw_pose.get('standing', True))
            path_length = max(0, len(path) - 1)
            desired_h = _desired_horizon(point, object_position)
            path_turns = _path_num_turns(path)
            if opening_yaw is not None:
                desired_yaw = _nearest_cardinal_yaw(opening_yaw)
            else:
                desired_yaw = _nearest_cardinal_yaw(_yaw_towards(point, object_position))
            facing_penalty = _yaw_deviation_penalty(rotation_y, desired_yaw)
            candidate = InteractionPose(key=key, position={'x': float(point.get('x', raw_pose.get('x', 0.0))), 'y': float(point.get('y', raw_pose.get('y', self.agent_position().get('y', 0.0)))), 'z': float(point.get('z', raw_pose.get('z', 0.0)))}, rotation_y=rotation_y, horizon=horizon, standing=standing, distance=float(_xz_distance(point, object_position)), path_length=path_length, turns=path_turns, score=float(_xz_distance(point, object_position)) * 100.0 + facing_penalty + _standing_penalty(standing, prefer_standing) + _natural_horizon_penalty(horizon, desired_h) + raw_index * 0.001, source='simulator_interactable_pose', target_object_id=target_object_id)
            if _candidate_key(candidate) not in excluded_keys:
                poses.append(candidate)
        poses.sort(key=lambda pose: pose.score)
        return poses

    def reachable_positions(self) -> List[dict]:
        event = self.env.controller.step(action='GetReachablePositions')
        if not bool(event.metadata.get('lastActionSuccess', True)):
            raise ProcTHORLowLevelExecutionError('GetReachablePositions failed: {}'.format(event.metadata.get('errorMessage') or ''))
        return [dict(position) for position in event.metadata.get('actionReturn') or []]

    def _planning_blocked_edges(self, points_by_key: Mapping[GridKey, Mapping[str, Any]], exclude_object_id: Optional[str]=None) -> Iterable[Tuple[GridKey, GridKey]]:
        return self.blocked_edges

    def _prepare_held_view_clearance(self, label: str) -> Optional[dict]:
        if not self._held_inventory_object():
            return None
        reachable = self.reachable_positions()
        points_by_key = _points_by_key(reachable, self.step_size)
        return self._move_to_held_navigation_clearance(points_by_key, target_key=None, label=label, source='held_view_clearance')

    def _move_to_held_navigation_clearance(self, points_by_key: Mapping[GridKey, Mapping[str, Any]], *, target_key: Optional[GridKey], label: str, source: Optional[str], force: bool=False) -> Optional[dict]:
        held = self._held_inventory_object()
        if not held:
            return None
        current_key = _nearest_key(self.agent_position(), points_by_key, self.step_size)
        if current_key is None:
            return None
        if target_key is not None and current_key == target_key and (not force):
            return None
        blocked_edges = set(self._planning_blocked_edges(points_by_key))
        obstacle_bounds = self._held_clearance_obstacle_bounds(held)
        current_score = self._held_navigation_clearance_score(current_key, points_by_key, obstacle_bounds)
        current_yaw = _nearest_cardinal_yaw(float(self.agent_pose().get('rotation', {}).get('y', 0.0)))
        backoff_candidates = self._held_clearance_backoff_candidates(current_key, current_yaw, points_by_key, blocked_edges, target_key, obstacle_bounds)
        if backoff_candidates:
            best_backoff_score = max((-item[1] for item in backoff_candidates))
            if force or best_backoff_score > current_score:
                moved = self._try_held_clearance_candidates(backoff_candidates, points_by_key, label=label, source=source, current_key=current_key, current_score=current_score, held=held, strategy='backoff')
                if moved is not None:
                    return moved
        candidates: List[Tuple[int, float, int, int, GridKey, List[GridKey]]] = []
        for key in points_by_key:
            if key == current_key:
                continue
            if _manhattan(current_key, key) > _HELD_NAVIGATION_CLEARANCE_RADIUS:
                continue
            path = _astar_path(points_by_key, current_key, key, blocked_edges, turn_penalty=self.turn_penalty)
            if path is None or len(path) < 2:
                continue
            path_steps = len(path) - 1
            if path_steps > _HELD_NAVIGATION_CLEARANCE_MAX_PATH_STEPS:
                continue
            if not all((_clearance_translation_action_for_step(a, b, current_yaw) is not None for a, b in zip(path, path[1:]))):
                continue
            score = self._held_navigation_clearance_score(key, points_by_key, obstacle_bounds)
            if target_key is not None:
                score -= 0.05 * float(_manhattan(key, target_key))
            first_action_cost = _initial_clearance_rotation_cost(path, current_yaw)
            candidates.append((first_action_cost, -score, path_steps, _path_num_turns(path), key, path))
        if not candidates:
            return None
        best_score = max((-item[1] for item in candidates))
        if not force and best_score <= current_score + _HELD_NAVIGATION_CLEARANCE_SCORE_MARGIN:
            return None
        candidates.sort()
        return self._try_held_clearance_candidates(candidates, points_by_key, label=label, source=source, current_key=current_key, current_score=current_score, held=held, strategy='lateral_clearance')

    def _held_clearance_backoff_candidates(self, current_key: GridKey, current_yaw: int, points_by_key: Mapping[GridKey, Mapping[str, Any]], blocked_edges: Iterable[Tuple[GridKey, GridKey]], target_key: Optional[GridKey], obstacle_bounds: Sequence[Tuple[float, float, float, float]]) -> List[Tuple[int, float, int, int, GridKey, List[GridKey]]]:
        candidates: List[Tuple[int, float, int, int, GridKey, List[GridKey]]] = []
        path = [current_key]
        key = current_key
        for _step in range(_HELD_NAVIGATION_CLEARANCE_MAX_PATH_STEPS):
            next_key = _translation_next_key(key, 'MoveBack', current_yaw)
            if next_key not in points_by_key or _edge_key(key, next_key) in set(blocked_edges):
                break
            path = path + [next_key]
            score = self._held_navigation_clearance_score(next_key, points_by_key, obstacle_bounds)
            if target_key is not None:
                score -= 0.05 * float(_manhattan(next_key, target_key))
            candidates.append((0, -score, len(path) - 1, 0, next_key, list(path)))
            key = next_key
        candidates.sort()
        return candidates

    def _try_held_clearance_candidates(self, candidates: Sequence[Tuple[int, float, int, int, GridKey, List[GridKey]]], points_by_key: Mapping[GridKey, Mapping[str, Any]], *, label: str, source: Optional[str], current_key: GridKey, current_score: float, held: Mapping[str, Any], strategy: str) -> Optional[dict]:
        for _first_action_cost, _neg_score, _path_steps, _turns, key, path in candidates[:8]:
            moved = self._execute_held_clearance_path(path, points_by_key, label=label, source=source)
            if moved is None:
                continue
            moved.update({'from_key': list(current_key), 'to_key': list(key), 'score_before': current_score, 'score_after': -_neg_score, 'held_object_id': held.get('objectId'), 'held_object_type': held.get('objectType'), 'strategy': strategy})
            return moved
        return None

    def _execute_held_clearance_path(self, path: Sequence[GridKey], points_by_key: Mapping[GridKey, Mapping[str, Any]], *, label: str, source: Optional[str]) -> Optional[dict]:
        if len(path) < 2:
            return None
        old_force_active = self._force_navigation_fallback_active
        self._force_navigation_fallback_active = False
        num_steps = 0
        clearance_label = '{} [held navigation clearance]'.format(label)
        try:
            for next_key in path[1:]:
                current_key = _nearest_key(self.agent_position(), points_by_key, self.step_size)
                if current_key is None or _manhattan(current_key, next_key) != 1:
                    return None
                current_yaw = _nearest_cardinal_yaw(float(self.agent_pose().get('rotation', {}).get('y', 0.0)))
                move_action = _clearance_translation_action_for_step(current_key, next_key, current_yaw)
                if move_action is None:
                    return None
                event = self._step_physical({'action': move_action, 'moveMagnitude': self.step_size}, label=clearance_label, source=source, allow_failure=True)
                num_steps += 1
                if not bool(event.metadata.get('lastActionSuccess', False)):
                    return None
        finally:
            self._force_navigation_fallback_active = old_force_active
        return {'status': STATUS_SUCCESS, 'num_steps': num_steps, 'path': [list(key) for key in path], 'final_pose': self.agent_pose()}

    def _held_navigation_clearance_score(self, key: GridKey, points_by_key: Mapping[GridKey, Mapping[str, Any]], obstacle_bounds: Sequence[Tuple[float, float, float, float]]) -> float:
        point = points_by_key[key]
        neighbor_count = sum((1 for _ in _neighbors(key, points_by_key, self.blocked_edges)))
        local_count = sum((1 for other_key in points_by_key if _manhattan(key, other_key) <= _HELD_NAVIGATION_CLEARANCE_RADIUS))
        clearance = _point_obstacle_clearance(point, obstacle_bounds)
        return float(neighbor_count) * 2.0 + float(local_count) * 0.25 + min(clearance, 1.5) * 4.0

    def _held_clearance_obstacle_bounds(self, held: Mapping[str, Any]) -> List[Tuple[float, float, float, float]]:
        held_id = str(held.get('objectId') or '')
        bounds: List[Tuple[float, float, float, float]] = []
        for obj in self.metadata().get('objects', []):
            if not isinstance(obj, Mapping):
                continue
            object_id = str(obj.get('objectId') or '')
            if object_id and object_id == held_id:
                continue
            object_type = str(obj.get('objectType') or '')
            if object_type in _HELD_NAVIGATION_CLEARANCE_IGNORED_TYPES:
                continue
            if obj.get('pickupable') and (not obj.get('receptacle')):
                continue
            xz_bounds = _clearance_object_xz_bounds(obj)
            if xz_bounds is not None:
                bounds.append(xz_bounds)
        return bounds

    def _is_held_navigation_collision(self, error_message: str) -> bool:
        held = self._held_inventory_object()
        return bool(held) and _looks_like_held_object_collision(error_message, held)

    def object_by_role(self, role_name: str) -> Mapping[str, Any]:
        return self.object_by_id(self.role_object_id(role_name))

    def object_by_id(self, object_id: str) -> Mapping[str, Any]:
        for obj in self.metadata().get('objects', []):
            if obj.get('objectId') == object_id:
                return obj
        raise ProcTHORLowLevelExecutionError('Object not found in metadata: {}'.format(object_id))

    def role_object_id(self, role_name: str) -> str:
        if hasattr(self.env, 'role_object_ids') and role_name in self.env.role_object_ids:
            return self.env.role_object_ids[role_name]
        role = self.task_spec['roles'].get(role_name)
        if role is None:
            raise ProcTHORLowLevelExecutionError('Unknown role: {}.'.format(role_name))
        return role['object_id']

    def is_object_visible(self, object_id: str) -> bool:
        return object_visible_from_metadata(self.metadata(), object_id) is True

    def is_object_interaction_ready(self, object_id: str) -> bool:
        return interaction_ready_from_metadata(self.metadata(), object_id, max_distance_m=self.interaction_distance) is True

    def metadata(self) -> Mapping[str, Any]:
        return self.env.controller.last_event.metadata

    def agent_pose(self) -> dict:
        agent = self.metadata().get('agent') or {}
        return {'position': dict(agent.get('position') or {}), 'rotation': dict(agent.get('rotation') or {}), 'cameraHorizon': agent.get('cameraHorizon', 0.0), 'isStanding': agent.get('isStanding')}

    def agent_position(self) -> Mapping[str, Any]:
        return self.agent_pose()['position']

    def _sort_by_rotation_cost(self, poses: List[InteractionPose]) -> List[InteractionPose]:
        current_yaw = _nearest_cardinal_yaw(float(self.agent_pose().get('rotation', {}).get('y', 0.0)))
        current_horizon = _normalize_horizon(self.agent_pose().get('cameraHorizon', 0.0))

        def _cost(pose: InteractionPose) -> float:
            yaw_diff = min(abs(pose.rotation_y - current_yaw) % 360, abs(current_yaw - pose.rotation_y) % 360)
            horizon_diff = abs(pose.horizon - current_horizon)
            return yaw_diff + horizon_diff * 0.5 + pose.score * 0.01
        return sorted(poses, key=_cost)

    def _rotate_to_yaw(self, target_yaw: float, label: str, source: Optional[str]=None) -> int:
        current_yaw = _nearest_cardinal_yaw(float(self.agent_pose().get('rotation', {}).get('y', 0.0)))
        target_yaw = _nearest_cardinal_yaw(target_yaw)
        total_steps = 0
        errors: List[str] = []
        max_rotation_attempts = 8
        while current_yaw != target_yaw and total_steps < max_rotation_attempts:
            progressed = False
            for actions in _ordered_rotation_sequences(current_yaw, target_yaw):
                if not actions:
                    return total_steps
                sequence_start_yaw = current_yaw
                sequence_failed = False
                for action in actions:
                    event = self._step_physical({'action': action}, label=label, source=source, allow_failure=True)
                    total_steps += 1
                    if not bool(event.metadata.get('lastActionSuccess', False)):
                        errors.append(event.metadata.get('errorMessage') or action)
                        sequence_failed = True
                        break
                    progressed = True
                    current_yaw = _nearest_cardinal_yaw(float(self.agent_pose().get('rotation', {}).get('y', 0.0)))
                    if current_yaw == target_yaw:
                        return total_steps
                if not sequence_failed:
                    return total_steps
                current_yaw = _nearest_cardinal_yaw(float(self.agent_pose().get('rotation', {}).get('y', 0.0)))
                if current_yaw == sequence_start_yaw:
                    continue
                break
            if not progressed:
                raise ProcTHORLowLevelExecutionError('Could not rotate to yaw {}: {}'.format(target_yaw, '; '.join(errors) or 'all rotation attempts failed'))
        if current_yaw != target_yaw:
            raise ProcTHORLowLevelExecutionError('Could not rotate to yaw {} within {} attempts: {}'.format(target_yaw, max_rotation_attempts, '; '.join(errors) or 'rotation did not converge'))
        return total_steps

    def _look_to_horizon(self, target_horizon: float, label: str, source: Optional[str]=None) -> int:
        current = _normalize_horizon(self.agent_pose().get('cameraHorizon', 0.0))
        target = _clamp(round(float(target_horizon) / 30.0) * 30, -30, 60)
        steps = 0
        while abs(current - target) >= 15:
            if target > current:
                self._step_physical({'action': 'LookDown', 'degrees': 30}, label=label, source=source)
                current += 30
            else:
                self._step_physical({'action': 'LookUp', 'degrees': 30}, label=label, source=source)
                current -= 30
            current = _clamp(current, -30, 60)
            steps += 1
            if steps > 6:
                break
        return steps

    def _set_standing(self, standing: bool, label: str, source: Optional[str]=None) -> int:
        agent = self.metadata().get('agent') or {}
        currently_standing = bool(agent.get('isStanding', True))
        if currently_standing == bool(standing):
            return 0
        self._step_physical({'action': 'Stand' if standing else 'Crouch'}, label=label, source=source)
        return 1

    def _apply_viewport_perturbation(self, perturbation: str, label: str) -> bool:
        label = '{} [put retry: {}]'.format(label, perturbation)
        try:
            if perturbation == 'look_down_30':
                self._step_physical({'action': 'LookDown', 'degrees': 30}, label=label, allow_failure=True)
            elif perturbation == 'look_down_60':
                self._step_physical({'action': 'LookDown', 'degrees': 30}, label=label, allow_failure=True)
                self._step_physical({'action': 'LookDown', 'degrees': 30}, label=label, allow_failure=True)
            elif perturbation == 'look_up_30':
                self._step_physical({'action': 'LookUp', 'degrees': 30}, label=label, allow_failure=True)
            elif perturbation == 'look_up_30_back':
                self._step_physical({'action': 'LookUp', 'degrees': 30}, label=label, allow_failure=True)
                self._step_physical({'action': 'MoveBack'}, label=label, allow_failure=True)
            elif perturbation == 'look_down_30_back':
                self._step_physical({'action': 'LookDown', 'degrees': 30}, label=label, allow_failure=True)
                self._step_physical({'action': 'MoveBack'}, label=label, allow_failure=True)
            else:
                return False
            return True
        except ProcTHORLowLevelExecutionError:
            return False

    def _retry_put_object_after_failure(self, *, event: Any, object_id: str, label: str, role_name: Optional[str]) -> Tuple[Any, bool]:
        success = bool(event.metadata.get('lastActionSuccess', False))
        if success:
            return (event, True)
        if _PUT_DOWN_USE_FORCE_FALLBACK:
            force_event = self._try_force_put_object_fallback(object_id=object_id, label=label, role_name=role_name)
            if force_event is not None:
                event = force_event
                success = bool(force_event.metadata.get('lastActionSuccess', False))
                if success:
                    return (event, True)
        for perturb in _PUT_DOWN_VIEWPORT_PERTURBATIONS:
            if not self._apply_viewport_perturbation(perturb, label):
                continue
            event = self._step_physical({'action': 'PutObject', 'objectId': object_id}, label='{} [perturb={}]'.format(label, perturb), role_name=role_name, allow_failure=True)
            success = bool(event.metadata.get('lastActionSuccess', False))
            if success:
                return (event, True)
            if not _PUT_DOWN_USE_FORCE_FALLBACK:
                continue
            force_event = self._try_force_put_object_fallback(object_id=object_id, label=label, role_name=role_name)
            if force_event is not None:
                event = force_event
                success = bool(force_event.metadata.get('lastActionSuccess', False))
                if success:
                    return (event, True)
        return (event, success)

    def _put_object_payload(self, receptacle_object_id: str, *, force_action: bool=False) -> dict:
        held = self._held_inventory_object()
        held_id = held.get('objectId') if isinstance(held, Mapping) else None
        if isinstance(held_id, str) and held_id:
            payload = {'action': 'PutObject', 'objectId': held_id, 'receptacleObjectId': receptacle_object_id, 'placeStationary': True}
        else:
            payload = {'action': 'PutObject', 'objectId': receptacle_object_id, 'placeStationary': True}
        if force_action:
            payload['forceAction'] = True
        return payload

    def _put_object_receptacle_candidates(self, receptacle_object_id: str) -> List[str]:
        candidates = [receptacle_object_id]
        if '___' in receptacle_object_id:
            parent_id = receptacle_object_id.split('___', 1)[0]
            if parent_id and parent_id != receptacle_object_id:
                parent = self._objects_by_id_safe().get(parent_id)
                if not parent or bool(parent.get('receptacle', False)):
                    candidates.append(parent_id)
        deduped: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _put_object_payload_variants(self, receptacle_object_id: str, *, force_action: bool=False) -> List[dict]:
        payloads: List[dict] = []
        for candidate_id in self._put_object_receptacle_candidates(receptacle_object_id):
            explicit = self._put_object_payload(candidate_id, force_action=force_action)
            payloads.append(explicit)
            if explicit.get('receptacleObjectId'):
                without_stationary = dict(explicit)
                without_stationary.pop('placeStationary', None)
                payloads.append(without_stationary)
            legacy_stationary = {'action': 'PutObject', 'objectId': candidate_id, 'placeStationary': True}
            legacy = {'action': 'PutObject', 'objectId': candidate_id}
            if force_action:
                legacy_stationary['forceAction'] = True
                legacy['forceAction'] = True
            payloads.extend([legacy_stationary, legacy])
        deduped: List[dict] = []
        seen = set()
        for payload in payloads:
            key = tuple(sorted(payload.items()))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(payload)
        return deduped

    def _try_force_put_object_fallback(self, *, object_id: str, label: str, role_name: Optional[str]=None) -> Optional[Any]:
        allowed, context = self._force_put_object_fallback_context(object_id)
        if not allowed:
            return None
        held = self._held_inventory_object()
        payloads = self._put_object_payload_variants(object_id, force_action=True)
        event = None
        executed_payload: Optional[Mapping[str, Any]] = None
        for payload in payloads:
            try:
                event = self.env.controller.step(**dict(payload))
            except Exception:
                continue
            executed_payload = payload
            if bool(event.metadata.get('lastActionSuccess', False)):
                break
        if event is None:
            return None
        success = bool(event.metadata.get('lastActionSuccess', False))
        action_record = {'action': 'PutObject', 'executed_action': 'PutObject', 'label': '{} [force_put_object_fallback]'.format(label), 'success': success, 'error_message': event.metadata.get('errorMessage') or '', 'agent_pose': self.agent_pose(), 'execution_substep': 1, 'execution_substeps': 1, 'object_id': object_id, 'requested_receptacle_object_id': object_id, 'forceAction': True, 'force_put_object_fallback': True, 'force_put_object_fallback_applied': success, 'force_put_object_fallback_context': dict(context)}
        if executed_payload is not None:
            effective_receptacle_id = executed_payload.get('receptacleObjectId') or executed_payload.get('objectId')
            action_record['put_object_payload_style'] = 'explicit_receptacle' if executed_payload.get('receptacleObjectId') else 'legacy_receptacle_object_id'
            action_record['effective_receptacle_object_id'] = effective_receptacle_id
            if isinstance(effective_receptacle_id, str) and effective_receptacle_id != object_id:
                action_record['receptacle_fallback'] = 'parent_object'
            if executed_payload.get('receptacleObjectId'):
                action_record['receptacle_object_id'] = executed_payload.get('receptacleObjectId')
        if role_name is not None:
            action_record['role'] = role_name
        if isinstance(held, Mapping):
            if held.get('objectId'):
                action_record['held_object_id'] = held.get('objectId')
            if held.get('objectType'):
                action_record['held_object_type'] = held.get('objectType')
        action_record['metadata_visible_object_ids'] = _visible_object_ids(event)
        self.action_records.append(action_record)
        if self.stream_callback is not None:
            callback_result = self.stream_callback(action_record, event)
            if isinstance(callback_result, Mapping):
                action_record.update(callback_result)
            elif isinstance(callback_result, str):
                action_record['image_path'] = callback_result
        return event

    def _force_put_object_fallback_context(self, object_id: str) -> Tuple[bool, dict]:
        obj = self.object_by_id(object_id)
        visible = bool(obj.get('visible', False))
        distance = self._object_xz_distance_to_agent(obj)
        max_distance = float(self.interaction_distance) + _PUT_DOWN_FORCE_FALLBACK_DISTANCE_MARGIN
        context = {'target_visible': visible, 'target_distance': distance, 'max_allowed_distance': max_distance}
        if not visible:
            context['reason'] = 'target_not_visible_for_force_put_object_fallback'
            return (False, context)
        if distance is None or distance > max_distance:
            context['reason'] = 'target_too_far_for_force_put_object_fallback'
            return (False, context)
        context['reason'] = 'target_visible_and_near'
        return (True, context)

    def _object_xz_distance_to_agent(self, obj: Mapping[str, Any]) -> Optional[float]:
        obj_pos = obj.get('position') or {}
        agent_pos = self.agent_position()
        if not all((key in obj_pos for key in ('x', 'z'))):
            return None
        if not all((key in agent_pos for key in ('x', 'z'))):
            return None
        try:
            return _xz_distance(obj_pos, agent_pos)
        except (TypeError, ValueError):
            return None

    def _held_inventory_object(self) -> Optional[Mapping[str, Any]]:
        if self._manipulathor_visual_attach_enabled():
            return None
        inventory = self.metadata().get('inventoryObjects') or []
        for entry in inventory:
            if isinstance(entry, Mapping):
                return entry
        arm = self.metadata().get('arm')
        if not isinstance(arm, Mapping):
            return None
        objects_by_id = {obj.get('objectId'): obj for obj in self.metadata().get('objects', []) or [] if isinstance(obj, Mapping) and isinstance(obj.get('objectId'), str)}
        for entry in arm.get('heldObjects', []) or []:
            object_id = entry.get('objectId') if isinstance(entry, Mapping) else entry
            if not isinstance(object_id, str) or not object_id:
                continue
            obj = objects_by_id.get(object_id, {})
            return {'objectId': object_id, 'objectType': obj.get('objectType') or object_id.split('|', 1)[0]}
        return None

    def _can_force_navigation_fallback(self, action: Any, error_message: str) -> bool:
        if not self.allow_force_navigation_fallback:
            return False
        if not self._force_navigation_fallback_active:
            return False
        if action not in _FORCE_NAVIGATION_FALLBACK_ACTIONS:
            return False
        if self._force_navigation_fallback_count >= self.max_force_navigation_fallbacks_per_tool:
            return False
        if self._manipulathor_visual_attach_enabled():
            return True
        held = self._held_inventory_object()
        if held:
            return _looks_like_held_object_collision(error_message, held)
        return self._logical_put_down_blocker_from_error(error_message) is not None

    def _can_force_viewpoint_fallback(self, action: Any, error_message: str, source: Optional[str]) -> bool:
        return bool(self._manipulathor_visual_attach_enabled() and source in {'look_for', 'scan'} and (action in {'RotateLeft', 'RotateRight', 'LookUp', 'LookDown'}) and _looks_like_viewpoint_collision(error_message))

    def _logical_put_down_blocker_from_error(self, error_message: str) -> Optional[str]:
        logical_ids = getattr(self.env, '_logical_put_down_object_ids', set()) or set()
        if not isinstance(error_message, str) or not logical_ids:
            return None
        for object_id in sorted((str(item) for item in logical_ids)):
            if object_id and object_id in error_message:
                return object_id
        return None

    def _can_force_open_container_fallback(self, action: Any, payload: Mapping[str, Any], error_message: str) -> bool:
        if not self.allow_force_open_container_fallback:
            return False
        if action not in _FORCE_OPEN_CONTAINER_FALLBACK_ACTIONS:
            return False
        if self._force_open_container_fallback_count >= self.max_force_open_container_fallbacks_per_tool:
            return False
        object_id = payload.get('objectId')
        if not isinstance(object_id, str):
            return False
        try:
            target = self.object_by_id(object_id)
        except ProcTHORLowLevelExecutionError:
            return False
        if not (target.get('openable') and target.get('receptacle')):
            return False
        if self._manipulathor_visual_attach_enabled() and getattr(self.env, 'eval_mode', None) == 'standard_vlm' and (target.get('objectType') == 'Fridge') and ('collided with wall|' in str(error_message)):
            return True
        held = self._held_inventory_object()
        if not held:
            return False
        return _looks_like_held_object_collision(error_message, held)

    def _can_force_arm_open_close_collision_fallback(self, action: Any, payload: Mapping[str, Any], error_message: str) -> bool:
        if not self.allow_force_arm_open_close_collision_fallback:
            return False
        if action not in _FORCE_ARM_OPEN_CLOSE_COLLISION_ACTIONS:
            return False
        if not self._manipulathor_arm_enabled():
            return False
        if self._force_arm_open_close_collision_fallback_count >= self.max_force_arm_open_close_collision_fallbacks_per_tool:
            return False
        if _low_level_budget_exhausted(self.env):
            return False
        object_id = payload.get('objectId')
        if not isinstance(object_id, str) or not object_id:
            return False
        matches = [item for item in self.metadata().get('objects', []) or [] if isinstance(item, Mapping) and item.get('objectId') == object_id]
        if len(matches) != 1:
            return False
        target = matches[0]
        if target.get('openable') is not True:
            return False
        return _looks_like_arm_collider_error(error_message)

    def _step_physical(self, payload: Mapping[str, Any], label: str, role_name: Optional[str]=None, allow_failure: bool=False, source: Optional[str]=None, interpolate: bool=True) -> Any:
        action = payload.get('action')
        if action not in PHYSICAL_ACTIONS:
            raise ProcTHORLowLevelExecutionError('Unsupported low-level action: {}.'.format(action))
        if action == 'TeleportFull' or payload.get('forceAction'):
            raise ProcTHORLowLevelExecutionError('Low-level executor does not allow TeleportFull or forceAction.')
        if action == 'PickupObject' and self._manipulathor_arm_enabled() and (not self._manipulathor_visual_attach_enabled()):
            self._prepare_manipulathor_arm_pickup(payload, label, role_name)
        payloads = self._execution_payloads(payload) if interpolate else [dict(payload)]
        last_event = None
        for substep_index, action_to_execute in enumerate(payloads, start=1):
            pose_before = self.agent_pose()
            last_event = self.env.controller.step(**dict(action_to_execute))
            success = bool(last_event.metadata.get('lastActionSuccess', False))
            error_message = last_event.metadata.get('errorMessage') or ''
            if success and action in TRANSLATION_ACTIONS:
                moved_distance = _xz_distance(pose_before.get('position') or {}, self.agent_pose().get('position') or {})
                requested_distance = abs(float(action_to_execute.get('moveMagnitude', 0.0)))
                if requested_distance > 0 and moved_distance <= 1e-06:
                    success = False
                    error_message = '{} produced no agent displacement; disable snapToGrid when using execution substeps.'.format(action)
                    last_event.metadata['lastActionSuccess'] = False
                    last_event.metadata['errorMessage'] = error_message
            force_attempted = False
            force_applied = False
            force_fallback_kind = ''
            force_error_message = ''
            pre_force_error_message = ''
            held = self._held_inventory_object()
            if not success and self._can_force_viewpoint_fallback(action, error_message, source):
                from scoper.benchmark.procthor.export.execution_audit import audit_controller_classifications
                force_attempted = True
                force_fallback_kind = 'viewpoint'
                pre_force_error_message = error_message
                force_payload = dict(action_to_execute)
                force_payload['forceAction'] = True
                with audit_controller_classifications(self.env, 'force_viewpoint_fallback', details={'source': source, 'requested_action': action, 'requested_degrees': force_payload.get('degrees')}):
                    forced_event = self.env.controller.step(**force_payload)
                forced_success = bool(forced_event.metadata.get('lastActionSuccess', False))
                force_error_message = forced_event.metadata.get('errorMessage') or ''
                if forced_success and (not _viewpoint_pose_matches(pose_before, self.agent_pose(), action=str(action), degrees=float(force_payload.get('degrees', 0.0)))):
                    forced_success = False
                    force_error_message = 'Forced viewpoint action did not reach the requested pose.'
                    forced_event.metadata['lastActionSuccess'] = False
                    forced_event.metadata['errorMessage'] = force_error_message
                last_event = forced_event
                success = forced_success
                error_message = '' if forced_success else force_error_message
                force_applied = forced_success
                setattr(self.env, '_force_viewpoint_fallbacks_total', int(getattr(self.env, '_force_viewpoint_fallbacks_total', 0)) + 1)
            elif not success and self._can_force_navigation_fallback(action, error_message):
                force_attempted = True
                force_fallback_kind = 'navigation'
                pre_force_error_message = error_message
                force_payload = dict(action_to_execute)
                force_payload['forceAction'] = True
                self._force_navigation_fallback_count += 1
                forced_event = self.env.controller.step(**force_payload)
                forced_success = bool(forced_event.metadata.get('lastActionSuccess', False))
                force_error_message = forced_event.metadata.get('errorMessage') or ''
                if forced_success and action in TRANSLATION_ACTIONS:
                    moved_distance = _xz_distance(pose_before.get('position') or {}, self.agent_pose().get('position') or {})
                    requested_distance = abs(float(force_payload.get('moveMagnitude', 0.0)))
                    if requested_distance > 0 and moved_distance <= 1e-06:
                        forced_success = False
                        force_error_message = 'Forced {} produced no agent displacement.'.format(action)
                        forced_event.metadata['lastActionSuccess'] = False
                        forced_event.metadata['errorMessage'] = force_error_message
                last_event = forced_event
                success = forced_success
                error_message = '' if forced_success else force_error_message
                force_applied = forced_success
            elif not success and self._can_force_open_container_fallback(action, action_to_execute, error_message):
                force_attempted = True
                force_fallback_kind = 'open_container'
                pre_force_error_message = error_message
                force_payload = dict(action_to_execute)
                force_payload['forceAction'] = True
                self._force_open_container_fallback_count += 1
                forced_event = self.env.controller.step(**force_payload)
                forced_success = bool(forced_event.metadata.get('lastActionSuccess', False))
                force_error_message = forced_event.metadata.get('errorMessage') or ''
                post_target = next((obj for obj in forced_event.metadata.get('objects', []) if obj.get('objectId') == action_to_execute.get('objectId')), {})
                if post_target.get('isOpen') is not True:
                    forced_success = False
                    force_error_message = force_error_message or 'OpenObject did not produce an open container.'
                last_event = forced_event
                success = forced_success
                error_message = '' if forced_success else force_error_message
                force_applied = forced_success
            elif not success and self._can_force_arm_open_close_collision_fallback(action, action_to_execute, error_message):
                from scoper.benchmark.procthor.export.execution_audit import audit_controller_classifications
                force_attempted = True
                force_fallback_kind = 'arm_open_close_collision'
                pre_force_error_message = error_message
                force_payload = dict(action_to_execute)
                force_payload['forceAction'] = True
                self._force_arm_open_close_collision_fallback_count += 1
                setattr(self.env, '_force_arm_open_close_collision_fallbacks_total', int(getattr(self.env, '_force_arm_open_close_collision_fallbacks_total', 0)) + 1)
                with audit_controller_classifications(self.env, 'force_arm_open_close_collision_fallback', details={'normal_attempt_error': pre_force_error_message, 'arm_mode': 'visual_attach' if self._manipulathor_visual_attach_enabled() else 'official', 'target_object_id': action_to_execute.get('objectId'), 'desired_is_open': action == 'OpenObject', 'fallback_budget_used': self._force_arm_open_close_collision_fallback_count, 'fallback_budget_limit': self.max_force_arm_open_close_collision_fallbacks_per_tool}):
                    forced_event = self.env.controller.step(**force_payload)
                forced_success = bool(forced_event.metadata.get('lastActionSuccess', False))
                force_error_message = forced_event.metadata.get('errorMessage') or ''
                post_open = _object_open_state(forced_event.metadata, action_to_execute.get('objectId'))
                desired_open = action == 'OpenObject'
                force_applied = forced_success and post_open is desired_open
                last_event = forced_event
                success = force_applied
                error_message = '' if force_applied else force_error_message or 'Forced action post isOpen state did not match desired state.'
                if not force_applied:
                    forced_event.metadata['lastActionSuccess'] = False
                    forced_event.metadata['errorMessage'] = error_message
            action_record = {'action': action, 'executed_action': action_to_execute.get('action'), 'label': label, 'success': success, 'error_message': error_message, 'agent_pose': self.agent_pose(), 'execution_substep': substep_index, 'execution_substeps': len(payloads)}
            if source is not None:
                action_record['source'] = source
            if role_name is not None:
                action_record['role'] = role_name
            if 'objectId' in payload:
                action_record['object_id'] = payload['objectId']
            if 'objectIdCandidates' in action_to_execute:
                action_record['object_id_candidates'] = list(action_to_execute.get('objectIdCandidates') or [])
            if 'moveMagnitude' in action_to_execute:
                action_record['move_magnitude'] = action_to_execute['moveMagnitude']
            if 'degrees' in action_to_execute:
                action_record['degrees'] = action_to_execute['degrees']
            if force_attempted:
                action_record['forceAction'] = True
                if force_fallback_kind == 'navigation':
                    action_record['force_navigation_fallback'] = True
                    action_record['force_navigation_fallback_applied'] = force_applied
                elif force_fallback_kind == 'open_container':
                    action_record['force_open_container_fallback'] = True
                    action_record['force_open_container_fallback_applied'] = force_applied
                elif force_fallback_kind == 'arm_open_close_collision':
                    action_record['force_arm_open_close_collision_fallback'] = True
                    action_record['force_arm_open_close_collision_fallback_applied'] = force_applied
                    action_record['desired_is_open'] = action == 'OpenObject'
                    action_record['post_is_open'] = _object_open_state(last_event.metadata, action_to_execute.get('objectId'))
                    action_record['arm_mode'] = 'visual_attach' if self._manipulathor_visual_attach_enabled() else 'official'
                    action_record['fallback_budget_used'] = self._force_arm_open_close_collision_fallback_count
                    action_record['fallback_budget_limit'] = self.max_force_arm_open_close_collision_fallbacks_per_tool
                elif force_fallback_kind == 'viewpoint':
                    action_record['force_viewpoint_fallback'] = True
                    action_record['force_viewpoint_fallback_applied'] = force_applied
                action_record['pre_force_error_message'] = pre_force_error_message
                if force_error_message:
                    action_record['force_error_message'] = force_error_message
                if isinstance(held, Mapping):
                    if held.get('objectId'):
                        action_record['held_object_id'] = held.get('objectId')
                    if held.get('objectType'):
                        action_record['held_object_type'] = held.get('objectType')
                elif force_fallback_kind == 'navigation':
                    logical_blocker = self._logical_put_down_blocker_from_error(pre_force_error_message)
                    if logical_blocker:
                        action_record['logical_blocking_object_id'] = logical_blocker
            action_record['metadata_visible_object_ids'] = _visible_object_ids(last_event)
            self.action_records.append(action_record)
            if self.stream_callback is not None:
                callback_result = self.stream_callback(action_record, last_event)
                if isinstance(callback_result, Mapping):
                    action_record.update(callback_result)
                elif isinstance(callback_result, str):
                    action_record['image_path'] = callback_result
            if not success:
                if not allow_failure:
                    raise ProcTHORLowLevelExecutionError("Action '{}' failed: {}".format(label, last_event.metadata.get('errorMessage') or 'unknown error'))
                return last_event
        return last_event

    def _execution_payloads(self, payload: Mapping[str, Any]) -> List[dict]:
        action = payload.get('action')
        if action == 'PickupObject' and self._manipulathor_arm_enabled():
            object_id = payload.get('objectId')
            pickup_payload = dict(payload)
            pickup_payload.pop('objectId', None)
            if isinstance(object_id, str) and object_id:
                pickup_payload['objectIdCandidates'] = [object_id]
            return [pickup_payload]
        if action in TRANSLATION_ACTIONS and self.move_execution_step is not None:
            total = float(payload.get('moveMagnitude', self.step_size))
            return [_payload_with(payload, moveMagnitude=amount) for amount in _split_amount(total, float(self.move_execution_step))]
        if action == 'PutObject' and 'placeStationary' not in payload:
            return [_payload_with(payload, placeStationary=True)]
        if action in {'RotateLeft', 'RotateRight'} and self.rotate_execution_degrees is not None:
            degrees = float(payload.get('degrees', 90.0))
            return [_payload_with(payload, degrees=_thor_degrees(amount)) for amount in _split_amount(degrees, float(self.rotate_execution_degrees))]
        if action in {'LookUp', 'LookDown'} and self.look_execution_degrees is not None:
            degrees = float(payload.get('degrees', 30.0))
            return [_payload_with(payload, degrees=_thor_degrees(amount)) for amount in _split_amount(degrees, float(self.look_execution_degrees))]
        normalized = dict(payload)
        if action in {'RotateLeft', 'RotateRight', 'LookUp', 'LookDown'}:
            normalized['degrees'] = _thor_degrees(float(payload.get('degrees', 90.0 if action.startswith('Rotate') else 30.0)))
        return [normalized]

    def _manipulathor_arm_enabled(self) -> bool:
        return bool(getattr(self.env, 'manipulathor_arm', False))

    def _manipulathor_visual_attach_enabled(self) -> bool:
        return bool(getattr(self.env, 'manipulathor_visual_attach_pickup', False))

    def _prepare_manipulathor_arm_pickup(self, payload: Mapping[str, Any], label: str, role_name: Optional[str]) -> None:
        if not bool(getattr(self.env, 'manipulathor_prepare_arm_for_pickup', True)):
            return
        object_id = payload.get('objectId')
        if not isinstance(object_id, str) or not object_id:
            return
        radius = getattr(self.env, 'manipulathor_pickup_hand_sphere_radius', 0.08)
        if radius is not None:
            self._step_manipulathor_arm_action({'action': 'SetHandSphereRadius', 'radius': float(radius)}, label=label, role_name=role_name, object_id=object_id, phase='pickup_prepare')
        pickupable = self._manipulathor_pickupable_object_ids()
        if object_id in pickupable:
            return
        for position in self._manipulathor_arm_pickup_positions(object_id):
            event = self._step_manipulathor_arm_action({'action': 'MoveArm', 'position': position, 'coordinateSpace': 'world', 'restrictMovement': False, 'speed': 100, 'returnToStart': False, 'fixedDeltaTime': 0.2}, label=label, role_name=role_name, object_id=object_id, phase='pickup_prepare')
            if not bool(event.metadata.get('lastActionSuccess', False)):
                continue
            if object_id in self._manipulathor_pickupable_object_ids():
                return

    def _manipulathor_pickupable_object_ids(self) -> set[str]:
        arm = self.metadata().get('arm')
        if not isinstance(arm, Mapping):
            return set()
        return {item for item in arm.get('pickupableObjects', []) or [] if isinstance(item, str)}

    def _manipulathor_arm_pickup_positions(self, object_id: str) -> List[dict]:
        try:
            obj = self.object_by_id(object_id)
        except ProcTHORLowLevelExecutionError:
            return []
        obj_position = obj.get('position') or {}
        if not _xyz_mapping(obj_position):
            return []
        x = float(obj_position['x'])
        y = float(obj_position['y'])
        z = float(obj_position['z'])
        offsets = ((0.0, 0.04, 0.0), (0.0, 0.1, 0.0), (0.0, 0.18, 0.0), (0.06, 0.08, 0.0), (-0.06, 0.08, 0.0), (0.0, 0.08, 0.06), (0.0, 0.08, -0.06))
        return [{'x': x + dx, 'y': y + dy, 'z': z + dz} for dx, dy, dz in offsets]

    def _step_manipulathor_arm_action(self, payload: Mapping[str, Any], *, label: str, role_name: Optional[str], object_id: Optional[str], phase: str) -> Any:
        event = self.env.controller.step(**dict(payload))
        action = str(payload.get('action'))
        action_record = {'action': action, 'executed_action': action, 'label': '{} [manipulathor_arm]'.format(label), 'success': bool(event.metadata.get('lastActionSuccess', False)), 'error_message': event.metadata.get('errorMessage') or '', 'agent_pose': self.agent_pose(), 'execution_substep': 1, 'execution_substeps': 1, 'manipulathor_arm': True, 'manipulathor_arm_phase': phase, 'metadata_visible_object_ids': _visible_object_ids(event)}
        if role_name is not None:
            action_record['role'] = role_name
        if object_id:
            action_record['object_id'] = object_id
        if 'radius' in payload:
            action_record['radius'] = payload['radius']
        if 'position' in payload:
            action_record['position'] = dict(payload.get('position') or {})
        self.action_records.append(action_record)
        if self.stream_callback is not None:
            callback_result = self.stream_callback(action_record, event)
            if isinstance(callback_result, Mapping):
                action_record.update(callback_result)
            elif isinstance(callback_result, str):
                action_record['image_path'] = callback_result
        return event

def _candidate_key(pose: InteractionPose) -> CandidateKey:
    return (pose.key, int(pose.rotation_y), int(pose.horizon), bool(pose.standing))

def _xyz_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and {'x', 'y', 'z'}.issubset(value)

def _points_by_key(reachable_positions: Sequence[Mapping[str, Any]], step_size: float) -> Dict[GridKey, Mapping[str, Any]]:
    points = {}
    for point in reachable_positions:
        if 'x' not in point or 'z' not in point:
            continue
        points[_key_for_position(point, step_size)] = point
    return points

def _nearest_key(position: Mapping[str, Any], points_by_key: Mapping[GridKey, Mapping[str, Any]], step_size: float) -> Optional[GridKey]:
    if 'x' not in position or 'z' not in position or (not points_by_key):
        return None
    exact_key = _key_for_position(position, step_size)
    if exact_key in points_by_key:
        return exact_key
    return min(points_by_key, key=lambda key: _squared_xz_distance(position, points_by_key[key]))

def _key_for_position(position: Mapping[str, Any], step_size: float) -> GridKey:
    return (int(round(float(position['x']) / step_size)), int(round(float(position['z']) / step_size)))

def _astar_path(points_by_key: Mapping[GridKey, Mapping[str, Any]], start: GridKey, goal: GridKey, blocked_edges: Optional[Iterable[Tuple[GridKey, GridKey]]]=None, turn_penalty: float=0.0) -> Optional[List[GridKey]]:
    if start not in points_by_key or goal not in points_by_key:
        return None
    blocked = set(blocked_edges or [])
    start_state = (start, None)
    frontier = [(0.0, 0, 0, 0, start_state)]
    came_from: Dict[Tuple[GridKey, Optional[Tuple[int, int]]], Optional[Tuple[GridKey, Optional[Tuple[int, int]]]]] = {start_state: None}
    cost_so_far = {start_state: 0.0}
    turns_so_far = {start_state: 0}
    counter = 0
    while frontier:
        _, turns, steps, _, state = heapq.heappop(frontier)
        current, previous_direction = state
        if current == goal:
            return _reconstruct_state_path(came_from, state)
        for neighbor in _neighbors(current, points_by_key, blocked):
            direction = _direction_between(current, neighbor)
            turned = previous_direction is not None and direction != previous_direction
            neighbor_state = (neighbor, direction)
            new_steps = steps + 1
            new_turns = turns + int(turned)
            new_cost = cost_so_far[state] + 1.0 + (float(turn_penalty) if turned else 0.0)
            should_update = neighbor_state not in cost_so_far or new_cost < cost_so_far[neighbor_state] or (abs(new_cost - cost_so_far[neighbor_state]) <= 1e-09 and new_turns < turns_so_far.get(neighbor_state, 10 ** 9))
            if should_update:
                cost_so_far[neighbor_state] = new_cost
                turns_so_far[neighbor_state] = new_turns
                priority = new_cost + _manhattan(neighbor, goal)
                counter += 1
                heapq.heappush(frontier, (priority, new_turns, new_steps, counter, neighbor_state))
                came_from[neighbor_state] = state
    return None

def _neighbors(key: GridKey, points_by_key: Mapping[GridKey, Mapping[str, Any]], blocked_edges: Iterable[Tuple[GridKey, GridKey]]) -> Iterable[GridKey]:
    for offset in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        neighbor = (key[0] + offset[0], key[1] + offset[1])
        if neighbor in points_by_key and _edge_key(key, neighbor) not in blocked_edges:
            yield neighbor

def _edge_key(a: GridKey, b: GridKey) -> Tuple[GridKey, GridKey]:
    return tuple(sorted((a, b)))

def _reconstruct_path(came_from: Mapping[GridKey, Optional[GridKey]], current: GridKey) -> List[GridKey]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def _reconstruct_state_path(came_from: Mapping[Tuple[GridKey, Optional[Tuple[int, int]]], Optional[Tuple[GridKey, Optional[Tuple[int, int]]]]], current: Tuple[GridKey, Optional[Tuple[int, int]]]) -> List[GridKey]:
    path = [current[0]]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current[0])
    path.reverse()
    return path

def _direction_between(current: GridKey, neighbor: GridKey) -> Tuple[int, int]:
    return (neighbor[0] - current[0], neighbor[1] - current[1])

def _path_num_turns(path: Sequence[GridKey]) -> int:
    turns = 0
    previous_direction = None
    for current, neighbor in zip(path, path[1:]):
        direction = _direction_between(current, neighbor)
        if previous_direction is not None and direction != previous_direction:
            turns += 1
        previous_direction = direction
    return turns

def _merge_straight_segments(path: List[GridKey]) -> List[Tuple[int, List[GridKey]]]:
    if len(path) < 2:
        return []
    segments: List[Tuple[int, List[GridKey]]] = []
    current_yaw: Optional[int] = None
    current_keys: List[GridKey] = []
    for i in range(len(path) - 1):
        yaw = _yaw_for_step(path[i], path[i + 1])
        if yaw == current_yaw:
            current_keys.append(path[i + 1])
        else:
            if current_keys:
                segments.append((current_yaw, current_keys))
            current_yaw = yaw
            current_keys = [path[i + 1]]
    if current_keys and current_yaw is not None:
        segments.append((current_yaw, current_keys))
    return segments

def _prev_key_for_segment(path: List[GridKey], first_key_in_segment: GridKey) -> GridKey:
    for i, key in enumerate(path):
        if key == first_key_in_segment and i > 0:
            return path[i - 1]
    return path[0]

def _yaw_for_step(current: GridKey, next_key: GridKey) -> int:
    dx = next_key[0] - current[0]
    dz = next_key[1] - current[1]
    if dx == 1 and dz == 0:
        return 90
    if dx == -1 and dz == 0:
        return 270
    if dx == 0 and dz == 1:
        return 0
    if dx == 0 and dz == -1:
        return 180
    raise ValueError('Path contains non-cardinal step: {} -> {}'.format(current, next_key))

def _clearance_translation_action_for_step(current: GridKey, next_key: GridKey, current_yaw: int) -> Optional[str]:
    desired_yaw = _yaw_for_step(current, next_key)
    current_yaw = _nearest_cardinal_yaw(current_yaw)
    if desired_yaw == current_yaw:
        return 'MoveAhead'
    if desired_yaw == (current_yaw + 180) % 360:
        return 'MoveBack'
    if desired_yaw == (current_yaw + 90) % 360:
        return 'MoveRight'
    if desired_yaw == (current_yaw - 90) % 360:
        return 'MoveLeft'
    return None

def _translation_next_key(key: GridKey, action: str, current_yaw: int) -> GridKey:
    yaw = _nearest_cardinal_yaw(current_yaw)
    if action == 'MoveBack':
        yaw = (yaw + 180) % 360
    elif action == 'MoveLeft':
        yaw = (yaw - 90) % 360
    elif action == 'MoveRight':
        yaw = (yaw + 90) % 360
    elif action != 'MoveAhead':
        raise ValueError('Unsupported translation action: {}'.format(action))
    dx, dz = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}[yaw]
    return (key[0] + dx, key[1] + dz)

def _yaw_towards(current_position: Mapping[str, Any], target_position: Mapping[str, Any]) -> float:
    dx = float(target_position.get('x', 0.0)) - float(current_position.get('x', 0.0))
    dz = float(target_position.get('z', 0.0)) - float(current_position.get('z', 0.0))
    return math.degrees(math.atan2(dx, dz))

def _nearest_cardinal_yaw(yaw: float) -> int:
    return int(round(float(yaw) / 90.0) * 90) % 360

def _yaw_deviation_penalty(actual_yaw: float, desired_yaw: float) -> float:
    delta = abs(float(actual_yaw) - float(desired_yaw)) % 360
    if delta > 180:
        delta = 360 - delta
    return delta / 90.0 * 100.0

def _ordered_rotation_sequences(current_yaw: int, target_yaw: int) -> List[List[str]]:
    right_steps = int((target_yaw - current_yaw) % 360 // 90)
    left_steps = int((current_yaw - target_yaw) % 360 // 90)
    if right_steps == 0:
        return []
    right = ['RotateRight'] * right_steps
    left = ['RotateLeft'] * left_steps
    if right_steps < left_steps:
        return [right, left]
    return [left, right]

def _initial_clearance_rotation_cost(path: Sequence[GridKey], current_yaw: int) -> int:
    if len(path) < 2:
        return 0
    desired_yaw = _yaw_for_step(path[0], path[1])
    current_yaw = _nearest_cardinal_yaw(current_yaw)
    if desired_yaw == current_yaw or desired_yaw == (current_yaw + 180) % 360:
        return 0
    return min(int((desired_yaw - current_yaw) % 360 // 90), int((current_yaw - desired_yaw) % 360 // 90))

def _looks_like_held_object_collision(error_message: str, held: Mapping[str, Any]) -> bool:
    message = str(error_message or '')
    if not message:
        return False
    if 'held item' in message.lower():
        return True
    held_id = str(held.get('objectId') or '')
    if held_id and held_id in message:
        return True
    held_type = str(held.get('objectType') or '')
    if held_type and 'with {}'.format(held_type) in message:
        return True
    return False

def _looks_like_viewpoint_collision(error_message: str) -> bool:
    lowered = str(error_message or '').lower()
    return any((token in lowered for token in ('collid', 'collision', 'intersect', 'obstruct', 'blocked')))

def _looks_like_static_final_rotation_collision(error_message: str) -> bool:
    message = str(error_message or '')
    return message.startswith('Could not rotate to yaw ') and 'Collided with static sim object:' in message

def _viewpoint_pose_matches(before: Mapping[str, Any], after: Mapping[str, Any], *, action: str, degrees: float, tolerance_degrees: float=1.0) -> bool:
    if action in {'RotateLeft', 'RotateRight'}:
        before_yaw = float((before.get('rotation') or {}).get('y', 0.0)) % 360.0
        after_yaw = float((after.get('rotation') or {}).get('y', 0.0)) % 360.0
        direction = 1.0 if action == 'RotateRight' else -1.0
        expected = (before_yaw + direction * degrees) % 360.0
        error = abs((after_yaw - expected + 180.0) % 360.0 - 180.0)
        return error <= tolerance_degrees
    before_horizon = float(before.get('cameraHorizon', 0.0))
    after_horizon = float(after.get('cameraHorizon', 0.0))
    direction = 1.0 if action == 'LookDown' else -1.0
    expected = before_horizon + direction * degrees
    return abs(after_horizon - expected) <= tolerance_degrees

def _ordered_yaws(base_yaw: int) -> List[int]:
    return [int(yaw) % 360 for yaw in (base_yaw, base_yaw + 90, base_yaw - 90, base_yaw + 180)]

def _desired_horizon(agent_position: Mapping[str, Any], target_position: Mapping[str, Any]) -> int:
    xz_dist = max(1e-06, _xz_distance(agent_position, target_position))
    camera_y = float(agent_position.get('y', 0.0)) + 0.675
    desired = -math.degrees(math.atan2(float(target_position.get('y', camera_y)) - camera_y, xz_dist))
    return int(_clamp(round(desired / 30.0) * 30, -30, 60))

def _ordered_horizons(desired_horizon: int) -> List[int]:
    return sorted([-30, 0, 30, 60], key=lambda item: abs(item - desired_horizon))

def _standing_penalty(standing: bool, prefer_standing: Optional[bool]) -> float:
    if prefer_standing is None:
        return 0.0
    return 0.0 if bool(standing) == bool(prefer_standing) else 100.0

def _natural_horizon_penalty(horizon: float, desired_horizon: Optional[float]=None) -> float:
    if desired_horizon is None:
        return abs(float(horizon)) / 30.0 * 8.0
    return abs(float(horizon) - float(desired_horizon)) / 30.0 * 4.0

def _source_penalty(source: str) -> float:
    return 15.0 if source == 'heuristic_visible_pose' else 0.0

def _visible_object_ids(event: Any) -> List[str]:
    metadata = getattr(event, 'metadata', {}) or {}
    return sorted((obj.get('objectId') for obj in metadata.get('objects', []) or [] if isinstance(obj, Mapping) and isinstance(obj.get('objectId'), str) and bool(obj.get('visible', False))))

def _normalize_horizon(horizon: float) -> float:
    horizon = float(horizon)
    if horizon > 180:
        horizon -= 360
    return horizon

def _looks_like_arm_collider_error(error_message: Any) -> bool:
    if not isinstance(error_message, str) or not error_message.strip():
        return False
    lowered = error_message.casefold().replace('-', '_').replace(' ', '_')
    return any((token in lowered for token in _ARM_COLLIDER_TOKENS))

def _object_open_state(metadata: Any, object_id: Any) -> Optional[bool]:
    if not isinstance(metadata, Mapping) or not isinstance(object_id, str):
        return None
    matches = [item for item in metadata.get('objects', []) or [] if isinstance(item, Mapping) and item.get('objectId') == object_id]
    if len(matches) != 1 or not isinstance(matches[0].get('isOpen'), bool):
        return None
    return matches[0]['isOpen']

def _low_level_budget_exhausted(env: Any) -> bool:
    used = getattr(env, 'total_low_level_steps', 0)
    pending = getattr(env, '_pending_low_level_records', ())
    limit = getattr(env, 'max_total_low_level_steps', None)
    if isinstance(used, bool) or not isinstance(used, int):
        return True
    if not isinstance(pending, Sequence) or isinstance(pending, (str, bytes)):
        return True
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        return True
    return used + len(pending) >= limit

def _xz_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return math.hypot(float(a.get('x', 0.0)) - float(b.get('x', 0.0)), float(a.get('z', 0.0)) - float(b.get('z', 0.0)))

def _point_obstacle_clearance(point: Mapping[str, Any], bounds: Sequence[Tuple[float, float, float, float]]) -> float:
    if 'x' not in point or 'z' not in point:
        return 0.0
    if not bounds:
        return 1.5
    x = float(point['x'])
    z = float(point['z'])
    return min((_point_rect_distance_xz(x, z, rect) for rect in bounds))

def _point_rect_distance_xz(x: float, z: float, rect: Tuple[float, float, float, float]) -> float:
    min_x, max_x, min_z, max_z = rect
    dx = max(min_x - x, 0.0, x - max_x)
    dz = max(min_z - z, 0.0, z - max_z)
    return math.hypot(dx, dz)

def _clearance_object_xz_bounds(obj: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    for bbox_key in ('objectOrientedBoundingBox', 'axisAlignedBoundingBox'):
        bbox = obj.get(bbox_key)
        if not isinstance(bbox, Mapping):
            continue
        bounds = _clearance_corner_points_xz_bounds(bbox.get('cornerPoints'))
        if bounds is not None:
            return bounds
        bounds = _clearance_center_size_xz_bounds(bbox.get('center'), bbox.get('size'))
        if bounds is not None:
            return bounds
    position = obj.get('position')
    if isinstance(position, Mapping) and 'x' in position and ('z' in position):
        x = float(position['x'])
        z = float(position['z'])
        return (x - 0.08, x + 0.08, z - 0.08, z + 0.08)
    return None

def _clearance_corner_points_xz_bounds(points: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return None
    xs: List[float] = []
    zs: List[float] = []
    for point in points:
        x = _clearance_vector_component(point, 'x', 0)
        z = _clearance_vector_component(point, 'z', 2)
        if x is None or z is None:
            continue
        xs.append(x)
        zs.append(z)
    if not xs or not zs:
        return None
    return (min(xs), max(xs), min(zs), max(zs))

def _clearance_center_size_xz_bounds(center: Any, size: Any) -> Optional[Tuple[float, float, float, float]]:
    cx = _clearance_vector_component(center, 'x', 0)
    cz = _clearance_vector_component(center, 'z', 2)
    sx = _clearance_vector_component(size, 'x', 0)
    sz = _clearance_vector_component(size, 'z', 2)
    if cx is None or cz is None or sx is None or (sz is None):
        return None
    half_x = abs(sx) / 2.0
    half_z = abs(sz) / 2.0
    return (cx - half_x, cx + half_x, cz - half_z, cz + half_z)

def _clearance_vector_component(value: Any, key: str, index: int) -> Optional[float]:
    try:
        if isinstance(value, Mapping):
            if key not in value:
                return None
            return float(value[key])
        if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
            if len(value) <= index:
                return None
            return float(value[index])
    except (TypeError, ValueError):
        return None
    return None

def _squared_xz_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return (float(a['x']) - float(b['x'])) ** 2 + (float(a['z']) - float(b['z'])) ** 2

def _manhattan(a: GridKey, b: GridKey) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def _split_amount(total: float, max_chunk: float) -> List[float]:
    if total <= 0:
        return [total]
    if max_chunk <= 0 or max_chunk >= total:
        return [total]
    chunks = []
    remaining = total
    epsilon = 1e-09
    while remaining > max_chunk + epsilon:
        chunks.append(max_chunk)
        remaining -= max_chunk
    if remaining > epsilon:
        chunks.append(remaining)
    return chunks or [total]

def _thor_degrees(value: float) -> float:
    return round(float(value) + 0.0, 1)

def _payload_with(payload: Mapping[str, Any], **updates: Any) -> dict:
    updated = dict(payload)
    updated.update(updates)
    return updated
__all__ = ['InteractionPose', 'ProcTHORLowLevelExecutionError', 'ProcTHORLowLevelExecutor', 'STATUS_FAILED', 'STATUS_SKIPPED', 'STATUS_SUCCESS']
