from __future__ import annotations
import copy
import math
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from scoper.benchmark.common import ConfirmationIntent, TaskBindingSpec, derive_benevolent_role_order, derive_public_binding_state, reference_forms, room_display_label
from scoper.benchmark.common.object_catalog import build_object_catalog
from scoper.benchmark.procthor.agent_catalog_filter import AgentCatalogTracker, build_agent_catalog_view, build_inventory_view
from scoper.benchmark.procthor.navigation_executor import ProcTHORLowLevelExecutionError, ProcTHORLowLevelExecutor, STATUS_FAILED, STATUS_SUCCESS
from scoper.benchmark.procthor.nl_target_resolver import NLTargetResolverError, ObjectBindingConfig, ResolvedTarget, extract_tool_target, resolve_navigation_target, resolve_operation_target
from scoper.benchmark.procthor.online_env import ProcTHOROnlineEnv, ProcTHOROnlineEnvError
from scoper.benchmark.procthor.online_task_spec import validate_online_task_spec
from scoper.benchmark.procthor.predicate_goal_checker import check_goal as evaluate_predicate_goal
TOOL_TO_AI2THOR_ACTION = {'pick_up': 'PickupObject', 'put_down': 'PutObject', 'open': 'OpenObject', 'close': 'CloseObject', 'toggle_on': 'ToggleObjectOn', 'toggle_off': 'ToggleObjectOff', 'slice': 'SliceObject'}
PUT_DOWN_FALLBACK_DISTANCE_MARGIN = 0.25
DEFAULT_PUT_DOWN_FALLBACK_MAX_DISTANCE = 1.7
MODEL_VISIBLE_TOOLS = ('navigate_to', 'scan', 'look_for', 'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'slice', 'answer', 'stop')
EVALUATOR_TOOLS = ('get_context', 'check_goal')

class ProcTHORClosedLoopToolEnvError(RuntimeError):
    pass

class ProcTHORClosedLoopToolEnv(ProcTHOROnlineEnv):

    def __init__(self, width: int=320, height: int=240, cloud_rendering: bool=False, gpu_device: Optional[int]=None, render_depth_image: bool=False, render_instance_segmentation: bool=False, max_high_level_steps: int=80, max_total_low_level_steps: int=800, max_invalid_tool_calls: int=10, max_nav_steps_per_tool: int=240, max_pose_candidates: int=80, move_execution_step: Optional[float]=None, rotate_execution_degrees: Optional[float]=None, look_execution_degrees: Optional[float]=None, snap_to_grid: Optional[bool]=None, turn_penalty: float=2.0, allow_force_navigation_fallback: bool=True, max_force_navigation_fallbacks_per_tool: int=80, allow_force_open_container_fallback: bool=True, max_force_open_container_fallbacks_per_tool: int=8, allow_force_arm_open_close_collision_fallback: bool=False, max_force_arm_open_close_collision_fallbacks_per_tool: int=8, allow_public_placed_open_container_navigation: bool=False, allow_logical_put_down_fallback: bool=True, eval_mode: str='agent', object_binding_config: Optional[Any]=None, agent_radius: Optional[float]=None, manipulathor_arm: bool=False, manipulathor_prepare_arm_for_pickup: bool=True, manipulathor_pickup_hand_sphere_radius: Optional[float]=0.08, manipulathor_visual_attach_pickup: bool=False, manipulathor_visual_attach_offset: Optional[Mapping[str, Any]]=None, manipulathor_visual_arm_hand_sphere_radius: Optional[float]=0.04, manipulathor_visual_arm_base_y: Optional[float]=None, manipulathor_visual_arm_wrist_position: Optional[Mapping[str, Any]]=None, manipulathor_visual_arm_move_speed: float=100000.0, manipulathor_visual_arm_fixed_delta_time: Optional[float]=0.0001, manipulathor_visual_arm_restore_object_poses: bool=True):
        resolved_snap_to_grid = move_execution_step is None if snap_to_grid is None else bool(snap_to_grid)
        resolved_agent_radius = agent_radius
        if resolved_agent_radius is None and bool(manipulathor_arm) and bool(manipulathor_visual_attach_pickup):
            resolved_agent_radius = 0.3
        self._sticky_object_states: Dict[str, Dict[str, Any]] = {}
        super().__init__(width=width, height=height, cloud_rendering=cloud_rendering, gpu_device=gpu_device, render_depth_image=render_depth_image, render_instance_segmentation=render_instance_segmentation, snap_to_grid=resolved_snap_to_grid, agent_radius=resolved_agent_radius, manipulathor_arm=bool(manipulathor_arm))
        self.max_high_level_steps = int(max_high_level_steps)
        self.max_total_low_level_steps = int(max_total_low_level_steps)
        self.max_invalid_tool_calls = int(max_invalid_tool_calls)
        self.max_nav_steps_per_tool = int(max_nav_steps_per_tool)
        self.max_pose_candidates = int(max_pose_candidates)
        self.move_execution_step = move_execution_step
        self.rotate_execution_degrees = rotate_execution_degrees
        self.look_execution_degrees = look_execution_degrees
        self.snap_to_grid = resolved_snap_to_grid
        self.turn_penalty = float(turn_penalty)
        self.allow_force_navigation_fallback = bool(allow_force_navigation_fallback)
        self.max_force_navigation_fallbacks_per_tool = int(max_force_navigation_fallbacks_per_tool)
        self.allow_force_open_container_fallback = bool(allow_force_open_container_fallback)
        self.max_force_open_container_fallbacks_per_tool = int(max_force_open_container_fallbacks_per_tool)
        self.allow_force_arm_open_close_collision_fallback = bool(allow_force_arm_open_close_collision_fallback)
        self.max_force_arm_open_close_collision_fallbacks_per_tool = int(max_force_arm_open_close_collision_fallbacks_per_tool)
        self.allow_public_placed_open_container_navigation = bool(allow_public_placed_open_container_navigation)
        self.allow_logical_put_down_fallback = bool(allow_logical_put_down_fallback)
        self.eval_mode = str(eval_mode or 'agent').strip().lower()
        if self.eval_mode not in {'agent', 'standard_vlm'}:
            self.eval_mode = 'agent'
        self.object_binding_config = _coerce_object_binding_config(object_binding_config, eval_mode=self.eval_mode)
        self.manipulathor_prepare_arm_for_pickup = bool(manipulathor_prepare_arm_for_pickup)
        self.manipulathor_pickup_hand_sphere_radius = None if manipulathor_pickup_hand_sphere_radius is None else float(manipulathor_pickup_hand_sphere_radius)
        self.manipulathor_visual_attach_pickup = bool(manipulathor_visual_attach_pickup)
        self.manipulathor_visual_attach_offset = _coerce_xyz(manipulathor_visual_attach_offset, default={'x': 0.0, 'y': 0.0, 'z': 0.0})
        self.manipulathor_visual_arm_hand_sphere_radius = None if manipulathor_visual_arm_hand_sphere_radius is None else float(manipulathor_visual_arm_hand_sphere_radius)
        self.manipulathor_visual_arm_base_y = None if manipulathor_visual_arm_base_y is None else float(manipulathor_visual_arm_base_y)
        self.manipulathor_visual_arm_wrist_position = _coerce_xyz(manipulathor_visual_arm_wrist_position, default={'x': -0.22, 'y': 0.48, 'z': 0.2})
        self.manipulathor_visual_arm_move_speed = float(manipulathor_visual_arm_move_speed)
        self.manipulathor_visual_arm_fixed_delta_time = None if manipulathor_visual_arm_fixed_delta_time is None else float(manipulathor_visual_arm_fixed_delta_time)
        self.manipulathor_visual_arm_restore_object_poses = bool(manipulathor_visual_arm_restore_object_poses)
        self.executor: Optional[ProcTHORLowLevelExecutor] = None
        self.model_stopped = False
        from scoper.benchmark.common.evaluation_lifecycle import reset as reset_evaluation_lifecycle
        reset_evaluation_lifecycle(self)
        self.high_level_step = 0
        self.task_action_step = 0
        self.total_low_level_steps = 0
        self.invalid_tool_calls = 0
        self.tool_history: List[dict] = []
        self._pending_low_level_records: List[dict] = []
        self._ready_target: Optional[dict] = None
        self._local_interaction_context_active = False
        self._binding_attempts = 0
        self._binding_successes = 0
        self._binding_failures: Dict[str, int] = {}
        self._binding_failure_reasons = {}
        self._receptacle_binding_attempts = 0
        self._receptacle_binding_successes = 0
        self._receptacle_binding_failures = 0
        self.seen_object_navigation_steps = 0
        self.stale_object_navigation_steps = 0
        self.unseen_object_exploration_steps = 0
        self.target_missing_at_last_seen_count = 0
        self._visual_attached_object_id: Optional[str] = None
        self._visual_attached_object_type: Optional[str] = None
        self._visual_attached_rotation_delta_y: float = 0.0
        self._visual_attached_position: Optional[dict] = None
        self._visual_attached_rotation: Optional[dict] = None
        self._visual_hand_anchor_local_offset: Optional[dict] = None
        self._logical_put_down_object_ids: set[str] = set()
        self._public_placed_container_by_object: Dict[str, str] = {}
        self._custom_container_relative_placements: Dict[str, dict] = {}
        self._public_open_container_ids: set[str] = set()
        self._agent_catalog_tracker = AgentCatalogTracker()
        self._agent_catalog_view: List[dict] = []
        self._rooms_for_navigation: List[dict] = []
        self._human_qa_state: Dict[str, Any] = {'pending': {}, 'answers': {}}
        self._current_room_id: Optional[str] = None
        self._observation_conditioned_visible_issues: List[dict] = []
        self._public_binding_human_messages: Tuple[Mapping[str, Any], ...] = ()
        self._public_binding_effective_facts: Tuple[Mapping[str, Any], ...] = ()
        self._public_binding_scene_observations: Tuple[Mapping[str, Any], ...] = ()
        self._private_processed_binding_roles: Tuple[str, ...] = ()
        self._current_tool_binding_roles: set[str] = set()
        self._observation_conditioned_scan_events: List[dict] = []
        self.scan_steps = 0
        self.look_for_steps = 0
        self.answer_steps = 0
        self._scan_views_captured_total = 0
        self._accepted_stop_requests = 0
        self._force_arm_open_close_collision_fallbacks_total = 0
        self._force_viewpoint_fallbacks_total = 0
        self._last_confirmation_bound_target: Optional[dict] = None
        self._locked_recovery_admission_bypass: Optional[dict] = None
        self._suppressed_completed_tool_call: Optional[dict] = None
        self._harness_directed_tool_call: Optional[dict] = None
        self._public_precondition_rejection: Optional[dict] = None

    def reset_to_task(self, task_spec: Mapping[str, Any]) -> dict:
        validate_online_task_spec(task_spec)
        from scoper.benchmark.common.evaluation_navigation import reset as reset_navigation
        reset_navigation(self)
        self._sticky_object_states = {}
        self._clear_visual_attachment_state(update_metadata=False)
        obs = super().reset_to_task(task_spec)
        self._apply_visual_arm_pose(label='reset [visual_arm_pose]', append_record=False)
        if self.controller is not None:
            obs['head_rgb'] = self.controller.last_event.frame
        self.model_stopped = False
        from scoper.benchmark.common.evaluation_lifecycle import reset as reset_evaluation_lifecycle
        reset_evaluation_lifecycle(self)
        self.high_level_step = 0
        self.task_action_step = 0
        self.total_low_level_steps = 0
        self.invalid_tool_calls = 0
        self.tool_history = []
        self._pending_low_level_records = []
        self._ready_target: Optional[dict] = None
        self._local_interaction_context_active = False
        self._binding_attempts = 0
        self._binding_successes = 0
        self._binding_failures = {}
        self._binding_failure_reasons = {}
        self._receptacle_binding_attempts = 0
        self._receptacle_binding_successes = 0
        self._receptacle_binding_failures = 0
        self.seen_object_navigation_steps = 0
        self.stale_object_navigation_steps = 0
        self.unseen_object_exploration_steps = 0
        self.target_missing_at_last_seen_count = 0
        self._logical_put_down_object_ids = set()
        self._public_placed_container_by_object = {}
        self._custom_container_relative_placements = {}
        self._public_open_container_ids = set()
        self._agent_catalog_tracker.reset()
        from scoper.benchmark.common.binding_observation_ledger import enable_binding_delivery
        enable_binding_delivery(self, {})
        self._agent_catalog_view = []
        self._human_qa_state = {'pending': {}, 'answers': {}}
        self._current_room_id = None
        self._observation_conditioned_visible_issues = []
        self._public_binding_human_messages = ()
        self._public_binding_effective_facts = ()
        self._public_binding_scene_observations = ()
        self._private_processed_binding_roles = ()
        self._current_tool_binding_roles = set()
        self._observation_conditioned_scan_events = []
        self.scan_steps = 0
        self.look_for_steps = 0
        self.answer_steps = 0
        self._scan_views_captured_total = 0
        self._accepted_stop_requests = 0
        self._force_arm_open_close_collision_fallbacks_total = 0
        self._force_viewpoint_fallbacks_total = 0
        self._last_confirmation_bound_target = None
        self._locked_recovery_admission_bypass = None
        self._suppressed_completed_tool_call = None
        self._harness_directed_tool_call = None
        self._public_precondition_rejection = None
        self._refresh_object_catalog()
        self._refresh_agent_catalog_view()
        self.executor = self._make_executor(task_spec)
        self._rooms_for_navigation = self._build_navigation_rooms()
        obs['prompt_context'] = self.get_prompt_context()
        return obs

    def get_prompt_context(self, include_role_handles: bool=True, include_object_ids: bool=False) -> dict:
        if self.current_task_spec is None:
            raise ProcTHORClosedLoopToolEnvError('Call reset_to_task before get_prompt_context.')
        self._refresh_object_catalog()
        active_task = self._active_task_spec()
        if active_task is self.current_task_spec:
            base_context = super().get_task_context()
        else:
            parent_task = self.current_task_spec
            self.current_task_spec = active_task
            try:
                base_context = super().get_task_context()
            finally:
                self.current_task_spec = parent_task
        self._refresh_agent_catalog_view()
        base_context['object_catalog'] = [dict(entry) for entry in self._agent_catalog_view]
        base_context['held_objects'] = build_inventory_view(self._metadata_with_sticky_object_states())
        base_context['inventoryObjects'] = list(base_context['held_objects'])
        base_context['tool_specs'] = get_tool_specs(model_visible_only=True)
        base_context['tool_contract'] = {'navigation_contract': 'navigate_to(target) resolves only spatial-layer references: role handles, currently or historically seen object ids/aliases, unique or nearest object types, and room labels. Object navigation leaves the target visible and ready for direct operation; room and historical-position navigation do not guarantee object visibility.', 'operation_contract': 'Operation tools execute one real unforced AI2-THOR interaction at the current pose. A well-formed plan should call navigate_to on the same target immediately before an operation.', 'allowed_model_tools': list(MODEL_VISIBLE_TOOLS)}
        base_context['tool_target_handles'] = {'object_aliases': self._agent_catalog_tracker.aliases()}
        base_context['room_catalog'] = [{'label': room['label'], 'room_type': room['room_type']} for room in self._rooms_for_navigation]
        if include_role_handles:
            base_context['tool_target_handles']['role_handles'] = self._role_handles(include_object_ids=include_object_ids)
        if self.tool_history:
            base_context['tool_history'] = self._prompt_safe_tool_history()
        if self._current_room_id is not None:
            base_context['current_room_id'] = self._public_room_id(self._current_room_id)
        if self._observation_conditioned_visible_issues and (not self._observation_conditioned_visual_only()):
            base_context['visible_issues'] = [dict(issue) for issue in self._observation_conditioned_visible_issues]
        return base_context

    def public_task_scope(self) -> dict[str, Any]:
        task = self._active_task_spec()
        family = task.get('family_spec') if isinstance(task, Mapping) else None
        scope = family.get('scope') if isinstance(family, Mapping) else None
        if not isinstance(scope, Mapping):
            return {}
        room_ids = scope.get('room_ids')
        raw_room_ids = room_ids if isinstance(room_ids, (list, tuple)) else ()
        resolved: set[str] = {str(item) for item in raw_room_ids if isinstance(item, str) and item}
        roles = task.get('roles') if isinstance(task, Mapping) else None
        role_refs = scope.get('room_ids_from_roles')
        if isinstance(role_refs, str):
            role_refs = [role_refs]
        if isinstance(role_refs, (list, tuple)) and isinstance(roles, Mapping):
            for role_name in role_refs:
                role = roles.get(role_name) if isinstance(role_name, str) else None
                room_id = role.get('room_id') if isinstance(role, Mapping) else None
                if isinstance(room_id, str) and room_id:
                    resolved.add(room_id)
        single_ref = scope.get('room_ids_from_role')
        if isinstance(single_ref, str) and isinstance(roles, Mapping):
            role = roles.get(single_ref)
            room_id = role.get('room_id') if isinstance(role, Mapping) else None
            if isinstance(room_id, str) and room_id:
                resolved.add(room_id)
        public_room_ids = {self._public_room_id(room_id) for room_id in resolved}
        public_room_ids.discard(None)
        return {'type': str(scope.get('type') or ''), 'label': str(scope.get('label') or ''), 'room_ids': sorted(public_room_ids, key=lambda item: item.encode('utf-8'))}

    def check_goal(self, model_stopped: bool=False) -> dict:
        from scoper.benchmark.common.evaluation_lifecycle import frozen_goal
        snapshot = frozen_goal(self) if model_stopped else None
        if snapshot is not None:
            return snapshot
        if self.current_task_spec is None:
            raise ProcTHORClosedLoopToolEnvError('Call reset_to_task before check_goal.')
        return evaluate_predicate_goal(self._active_task_spec(), self._metadata_with_sticky_object_states(), model_stopped=model_stopped)

    def _qa_overlay_goal_result(self) -> dict:
        if self.current_task_spec is None:
            raise ProcTHORClosedLoopToolEnvError('Call reset_to_task before QA resolution.')
        return evaluate_predicate_goal(self.current_task_spec, self._metadata_with_sticky_object_states(), model_stopped=False)

    def _active_task_spec(self) -> Mapping[str, Any]:
        active = getattr(self, '_composite_active_task_spec', None)
        if isinstance(active, Mapping):
            return active
        return self.current_task_spec or {}

    def _call_scheduler_navigation(self, object_id: str, arguments: Mapping[str, Any], reasoning: str='') -> dict:
        if object_id not in self._raw_objects_by_id():
            raise ProcTHORClosedLoopToolEnvError('Scheduler navigation target does not exist')
        if getattr(self, '_scheduler_navigation_target', None) is not None:
            raise ProcTHORClosedLoopToolEnvError('Nested scheduler navigation is unsupported')
        self._scheduler_navigation_target = ResolvedTarget(target_ref=extract_tool_target(arguments), object_id=object_id, resolved_via='scheduler_injection')
        try:
            return self.call_tool('navigate_to', arguments, reasoning=reasoning)
        finally:
            self._scheduler_navigation_target = None

    def call_tool(self, tool_name: str, arguments: Optional[Mapping[str, Any]]=None, reasoning: str='') -> dict:
        if self.current_task_spec is None or self.executor is None:
            raise ProcTHORClosedLoopToolEnvError('Call reset_to_task before call_tool.')
        tool_name = str(tool_name)
        arguments = dict(arguments or {})
        from scoper.benchmark.common.evaluation_lifecycle import guard_tool
        guarded = guard_tool(self, tool_name, arguments)
        if guarded is not None:
            return guarded
        directed_call = self._harness_directed_tool_call
        harness_directed = bool(isinstance(directed_call, Mapping) and directed_call.get('tool_name') == tool_name and (directed_call.get('arguments') == arguments))
        self._harness_directed_tool_call = None
        suppressed_call = self._suppressed_completed_tool_call
        suppress_execution = bool(isinstance(suppressed_call, Mapping) and suppressed_call.get('tool_name') == tool_name and (suppressed_call.get('arguments') == arguments))
        if suppress_execution:
            self._suppressed_completed_tool_call = None
        rejected_call = self._public_precondition_rejection
        reject_for_public_precondition = bool(isinstance(rejected_call, Mapping) and rejected_call.get('tool_name') == tool_name and (rejected_call.get('arguments') == arguments))
        self._public_precondition_rejection = None
        bypass_interaction_readiness = False
        locked_bypass = self._locked_recovery_admission_bypass
        if isinstance(locked_bypass, Mapping) and (locked_bypass.get('tool_name') == tool_name and locked_bypass.get('arguments') == arguments):
            bypass_interaction_readiness = True
            self._locked_recovery_admission_bypass = None
        if tool_name == 'get_context':
            return self._non_actuating_tool_result(tool_name, success=True, feedback='Context returned.', raw_execution=self.get_prompt_context())
        if tool_name == 'check_goal':
            return self._non_actuating_tool_result(tool_name, success=True, feedback='Goal checked.', raw_execution=self.check_goal(model_stopped=self.model_stopped))
        start_time = time.time()
        self.high_level_step += 1
        if tool_name != 'answer':
            self.task_action_step += 1
        self._pending_low_level_records = []
        self._current_tool_binding_roles = set()
        self._last_confirmation_bound_target = None
        success = False
        feedback = ''
        raw_execution: Optional[dict] = None
        count_failure_as_invalid = not harness_directed
        resolved_object_id_for_state: Optional[str] = None
        held_object_id_before_action: Optional[str] = None
        binding_public_record: Optional[dict] = None
        try:
            if suppress_execution:
                success = True
                count_failure_as_invalid = False
                raw_execution = {'status': 'already_completed', 'action': tool_name, 'action_executed': False}
                feedback = 'This exact action is already completed for planning; continue with the next task step.'
            elif reject_for_public_precondition:
                count_failure_as_invalid = False
                binding_public_record = {'status': 'rejected', 'reason': 'public_precondition_failed', 'source': 'public_precondition', 'match_kind': None, 'canonical_public_ref': None, 'navigation_class': None}
                raw_execution = {'status': 'rejected', 'action': tool_name, 'action_executed': False, 'error_type': 'public_precondition_failed'}
                feedback = str(rejected_call.get('message') or 'Public precondition failed.')
            elif tool_name not in MODEL_VISIBLE_TOOLS:
                raise ProcTHORClosedLoopToolEnvError("Unknown tool '{}'. Valid tools: {}.".format(tool_name, ', '.join(MODEL_VISIBLE_TOOLS)))
            elif tool_name == 'stop':
                premature_stop_reason = self._standard_vlm_premature_stop_reason()
                if premature_stop_reason:
                    success = False
                    count_failure_as_invalid = False
                    raw_execution = {'status': STATUS_FAILED, 'action': 'stop', 'error_type': 'premature_stop', 'error_message': premature_stop_reason}
                    feedback = 'Stop was not accepted because required requested actions are missing.'
                else:
                    self.model_stopped = True
                    self._accepted_stop_requests += 1
                    success = True
                    raw_execution = {'status': STATUS_SUCCESS, 'action': 'stop'}
                    feedback = 'Agent stopped.'
            elif tool_name == 'scan':
                raw_execution = self.executor.scan_in_place(num_views=max(1, min(8, int(arguments.get('num_views', 8)))), label='scan')
                self.scan_steps += 1
                self._scan_views_captured_total += int(raw_execution.get('num_views_completed') or 0)
                scan_visible_ids = raw_execution.pop('metadata_visible_object_ids', [])
                success = raw_execution.get('status') == STATUS_SUCCESS
                if success:
                    visible_issues = self._materialize_observation_conditioned_scan_issues(scan_visible_ids)
                    if visible_issues and (not self._observation_conditioned_visual_only()):
                        raw_execution['visible_issues'] = [dict(issue) for issue in visible_issues]
                feedback = 'Scan completed.' if success else raw_execution.get('message', 'Scan failed.')
                from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding, scan_public_feedback
                if task_uses_evidence_binding(self.current_task_spec):
                    feedback = scan_public_feedback(int(raw_execution.get('num_views_completed') or 0), int(raw_execution.get('num_views') or 8))
            elif tool_name == 'look_for':
                resolved = self._resolve_operation_target(arguments, tool_name='look_for')
                binding_public_record = resolved.binding_public_record
                if resolved.object_id is None:
                    raise ProcTHORClosedLoopToolEnvError('look_for requires a target that resolves to an object.')
                raw_execution = self.executor.look_for_object(resolved.object_id, label='look_for {}'.format(resolved.target_ref))
                self.look_for_steps += 1
                success = raw_execution.get('status') == STATUS_SUCCESS
                raw_execution['interaction_ready'] = bool(success and self.executor.is_object_interaction_ready(resolved.object_id))
                feedback = 'Object found in view.' if success else 'Could not bring object into view from current position.'
            elif tool_name == 'answer':
                success, raw_execution, feedback = self._answer_pending_question(arguments)
                count_failure_as_invalid = False
            elif tool_name == 'navigate_to':
                self._local_interaction_context_active = False
                resolved = getattr(self, '_scheduler_navigation_target', None)
                self._scheduler_navigation_target = None
                if resolved is None:
                    resolved = self._resolve_navigation_target(arguments)
                binding_public_record = resolved.binding_public_record
                raw_execution = self._execute_resolved_navigation(resolved)
                success = raw_execution.get('status') == STATUS_SUCCESS
                self._record_navigation_binding_outcome(resolved)
                if success:
                    self._ready_target = self._ready_target_from_navigation(resolved, raw_execution)
                    self._local_interaction_context_active = bool(resolved.object_id is not None and raw_execution.get('interaction_ready') is True)
                    if raw_execution.get('room_id') is not None:
                        self._current_room_id = str(raw_execution['room_id'])
                feedback = 'Navigation succeeded.' if success else raw_execution.get('message', 'Navigation failed.')
            else:
                resolved = self._resolve_operation_target(arguments, tool_name=tool_name, bypass_interaction_readiness=bypass_interaction_readiness)
                binding_public_record = resolved.binding_public_record
                target_ref = resolved.target_ref
                object_id = resolved.object_id
                resolved_object_id_for_state = object_id
                if object_id is None:
                    raise ProcTHORClosedLoopToolEnvError("Operation target '{}' did not resolve to an object.".format(target_ref))
                target_metadata = self._raw_objects_by_id().get(object_id, {})
                target_position = target_metadata.get('position')
                if _has_xyz(target_position):
                    self._last_confirmation_bound_target = {'position': {'x': float(target_position['x']), 'y': float(target_position['y']), 'z': float(target_position['z'])}}
                action = TOOL_TO_AI2THOR_ACTION[tool_name]
                if tool_name == 'put_down' and (not self._agent_holding()):
                    raw_execution = {'status': STATUS_FAILED, 'action': action, 'target': target_ref, 'object_id': object_id, 'success': False, 'error_message': 'Agent is not holding any object.'}
                    success = False
                    feedback = "Tool 'put_down' failed — agent is not holding any object."
                else:
                    if tool_name == 'put_down':
                        held_object_id_before_action = self._held_inventory_object_id()
                    if self._use_visual_attachment_for_operation(tool_name):
                        if tool_name == 'pick_up':
                            raw_execution = self._execute_visual_attach_pickup(target_ref=target_ref, object_id=object_id)
                        else:
                            raw_execution = self._execute_visual_attach_put_down(target_ref=target_ref, receptacle_object_id=object_id, held_object_id=held_object_id_before_action)
                        success = raw_execution.get('status') == STATUS_SUCCESS
                    else:
                        navigation_precondition = self._navigation_precondition(target_ref, object_id)
                        raw_execution = self.executor.interact_with_object_id(action, object_id, label='{} {}'.format(tool_name, target_ref), target_ref=target_ref, allow_failure=True)
                        raw_execution['navigation_precondition'] = navigation_precondition
                        success = bool(raw_execution.get('success', False))
                        should_retry_with_navigation = bool(navigation_precondition.get('satisfied', False))
                        if not success and should_retry_with_navigation:
                            recovery = self.executor.execute_object_action(action, object_id, label='{} {} retry'.format(tool_name, target_ref), target_ref=target_ref, allow_failure=True)
                            recovery['initial_direct_attempt'] = raw_execution
                            recovery['navigation_precondition'] = navigation_precondition
                            if tool_name == 'pick_up':
                                recovery['pickup_recovery_applied'] = True
                            raw_execution = recovery
                            success = raw_execution.get('status') == STATUS_SUCCESS
                        if not success and tool_name == 'put_down' and self.allow_logical_put_down_fallback:
                            logical_recovery = self._try_logical_put_down_fallback(target_ref=target_ref, receptacle_object_id=object_id, held_object_id=held_object_id_before_action, failed_execution=raw_execution)
                            if logical_recovery is not None:
                                raw_execution = logical_recovery
                                success = True
                        if not success and tool_name == 'pick_up':
                            container_recovery = self._try_open_container_pickup_fallback(target_ref=target_ref, object_id=object_id, failed_execution=raw_execution)
                            if container_recovery is not None:
                                raw_execution = container_recovery
                                success = True
                    if success:
                        self._clear_navigation_block_cache()
                    feedback = "Tool '{}' succeeded.".format(tool_name) if success else raw_execution.get('error_message') or raw_execution.get('message') or "Tool '{}' failed.".format(tool_name)
        except Exception as exc:
            from scoper.benchmark.procthor.export.execution_audit import ExecutionAuditError
            if isinstance(exc, ExecutionAuditError):
                raise
            public_binding = getattr(exc, 'binding_public_record', None)
            if isinstance(public_binding, Mapping):
                binding_public_record = copy.deepcopy(dict(public_binding))
                count_failure_as_invalid = False
            raw_execution = {'status': 'invalid', 'error_type': type(exc).__name__, 'error_message': str(exc)}
            feedback = 'Tool call is invalid. {}'.format(exc)
        state_transition_events: List[dict] = []
        if success:
            try:
                if tool_name == 'slice' and resolved_object_id_for_state:
                    self._rebind_roles_to_sliced_child(resolved_object_id_for_state)
                state_arguments = dict(arguments)
                if resolved_object_id_for_state:
                    state_arguments['_resolved_object_id'] = resolved_object_id_for_state
                if held_object_id_before_action:
                    state_arguments['_held_object_id'] = held_object_id_before_action
                state_transition_events = self._apply_state_transitions(tool_name, state_arguments)
            except Exception as exc:
                from scoper.benchmark.procthor.export.execution_audit import ExecutionAuditError
                if isinstance(exc, ExecutionAuditError):
                    raise
                pass
        if self._visual_attached_object_id:
            try:
                self._sync_visual_attached_object_to_hand(label='visual_attach_sync', append_record=False)
            except Exception as exc:
                from scoper.benchmark.procthor.export.execution_audit import ExecutionAuditError
                if isinstance(exc, ExecutionAuditError):
                    raise
                pass
        self._refresh_object_catalog()
        self._refresh_agent_catalog_view()
        low_level_records = list(self._pending_low_level_records)
        self.total_low_level_steps += len(low_level_records)
        if not success and count_failure_as_invalid:
            self.invalid_tool_calls += 1
        goal_result = self.check_goal(model_stopped=self.model_stopped)
        done, done_reason = self._done_status(goal_result)
        parent_goal_result = None
        if isinstance(getattr(self, '_composite_active_task_spec', None), Mapping):
            parent_goal_result = evaluate_predicate_goal(self.current_task_spec, self._metadata_with_sticky_object_states(), model_stopped=self.model_stopped)
        transition = {'tool_name': tool_name, 'arguments': arguments, 'reasoning': reasoning, 'high_level_step': self.high_level_step, 'task_action_step': self.task_action_step, 'success': success, 'env_feedback': feedback, 'done': done, 'done_reason': done_reason, 'observation': self._observation(), 'goal_result': goal_result, 'parent_goal_result': parent_goal_result, 'raw_execution': raw_execution, 'binding': binding_public_record, 'low_level_actions': low_level_records, 'num_low_level_steps': len(low_level_records), 'episode_metrics': self._episode_metrics(), 'ready_target': dict(self._ready_target) if self._ready_target else None, 'state_transitions': list(state_transition_events), 'elapsed_seconds': time.time() - start_time}
        preparation_result = self._public_preparation_result(tool_name, raw_execution)
        if preparation_result is not None:
            if isinstance(binding_public_record, Mapping) and binding_public_record.get('status') != 'success':
                preparation_result = {**preparation_result, 'status': 'binding_failed', 'interaction_ready': False}
            transition['preparation_result'] = preparation_result
        self._mask_verify_feedback_if_needed(transition)
        if success and tool_name == 'slice' and resolved_object_id_for_state:
            if (self._ready_target or {}).get('object_id') == resolved_object_id_for_state:
                self._ready_target = None
        self.tool_history.append(_json_safe(transition))
        return transition

    @staticmethod
    def _public_preparation_result(tool_name: str, raw_execution: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
        raw = raw_execution if isinstance(raw_execution, Mapping) else {}
        if tool_name == 'look_for':
            if raw.get('motion_failed') is True:
                status = 'motion_failed'
            elif raw.get('status') == STATUS_SUCCESS and raw.get('object_visible') is True:
                status = 'found'
            elif raw.get('status') == STATUS_FAILED and raw.get('object_visible') is False:
                status = 'not_found'
            else:
                status = 'motion_failed'
            return {'tool_name': 'look_for', 'status': status, 'interaction_ready': bool(raw.get('interaction_ready')), 'frame_id_or_null': None}
        if tool_name == 'navigate_to':
            ready = raw.get('interaction_ready') is True
            if raw.get('status') == STATUS_SUCCESS:
                status = 'navigation_ready' if ready else 'navigation_not_ready'
            else:
                status = 'navigation_failed'
            return {'tool_name': 'navigate_to', 'status': status, 'interaction_ready': ready, 'frame_id_or_null': None}
        if tool_name == 'scan':
            return {'tool_name': 'scan', 'status': 'scan_completed' if raw.get('status') == STATUS_SUCCESS else 'motion_failed', 'interaction_ready': False, 'frame_id_or_null': None}
        return None

    def last_private_bound_target(self) -> Optional[dict]:
        return copy.deepcopy(self._last_confirmation_bound_target)

    def record_intercepted_transition(self, transition: Mapping[str, Any]) -> bool:
        if not isinstance(transition, Mapping):
            raise ProcTHORClosedLoopToolEnvError('intercepted transition must be a mapping')
        if transition.get('scheduler_intercept'):
            self.invalidate_public_placed_container_navigation()
        key = (transition.get('high_level_step'), transition.get('tool_name'), _json_safe(transition.get('arguments') or {}))
        for existing in self.tool_history:
            existing_key = (existing.get('high_level_step'), existing.get('tool_name'), _json_safe(existing.get('arguments') or {}))
            if existing_key == key:
                return False
        self.tool_history.append(_json_safe(dict(transition)))
        return True

    def invalidate_public_placed_container_navigation(self) -> None:
        self._public_placed_container_by_object.clear()
        self._public_open_container_ids.clear()

    def prebind_synthetic_operation(self, tool_name: str, arguments: Mapping[str, Any]) -> ResolvedTarget:
        self._last_confirmation_bound_target = None
        resolved = self._resolve_operation_target(arguments, tool_name=str(tool_name))
        if resolved.object_id is not None:
            target_metadata = self._raw_objects_by_id().get(resolved.object_id, {})
            target_position = target_metadata.get('position')
            if _has_xyz(target_position):
                self._last_confirmation_bound_target = {'position': {'x': float(target_position['x']), 'y': float(target_position['y']), 'z': float(target_position['z'])}}
        return resolved

    def capture_confirmation(self, intent: ConfirmationIntent, private_bound_target: Optional[Mapping[str, Any]]=None, *, video_fps: Optional[float]=None) -> Mapping[str, Any]:
        if self.executor is None or self.controller is None:
            raise ProcTHORClosedLoopToolEnvError('capture_confirmation requires an active ProcTHOR episode')
        start_record_count = len(self._pending_low_level_records)
        pose = copy.deepcopy(self.executor.agent_pose())
        current_yaw = float((pose.get('rotation') or {}).get('y', 0.0))
        target_yaw = current_yaw
        target_horizon = 0.0
        used_bound_coordinate = False
        target = private_bound_target.get('position') if isinstance(private_bound_target, Mapping) else None
        if intent.view_kind == 'bound_target' and _has_xyz(target):
            agent_position = pose.get('position') or {}
            if _has_xyz(agent_position):
                dx = float(target['x']) - float(agent_position['x'])
                dz = float(target['z']) - float(agent_position['z'])
                horizontal = math.hypot(dx, dz)
                if horizontal > 1e-06:
                    target_yaw = math.degrees(math.atan2(dx, dz)) % 360.0
                    camera_y = float(agent_position['y']) + 0.675
                    target_horizon = -math.degrees(math.atan2(float(target['y']) - camera_y, horizontal))
                    used_bound_coordinate = True
        target_horizon = max(-30.0, min(60.0, float(target_horizon)))
        label = 'confirmation {}'.format(intent.canonical_action)
        if video_fps is not None and float(video_fps) > 0.0:
            self._smooth_confirmation_camera(target_yaw=target_yaw, target_horizon=target_horizon, video_fps=float(video_fps), label=label)
        else:
            self._set_exact_confirmation_camera(target_yaw=target_yaw, target_horizon=target_horizon, label=label)
        actual_pose = self.executor.agent_pose()
        actual_yaw = float((actual_pose.get('rotation') or {}).get('y', 0.0) or 0.0) % 360.0
        actual_horizon = float(actual_pose.get('cameraHorizon', 0.0) or 0.0)
        confirmation_rgb = self.controller.last_event.frame
        if callable(getattr(confirmation_rgb, 'copy', None)):
            confirmation_rgb = confirmation_rgb.copy()
        confirmation_records = copy.deepcopy(self._pending_low_level_records[start_record_count:])
        restore_record_start = len(self._pending_low_level_records)
        restore_yaw = float((pose.get('rotation') or {}).get('y', current_yaw))
        restore_horizon = float(pose.get('cameraHorizon', 0.0) or 0.0)
        if video_fps is not None and float(video_fps) > 0.0:
            self._smooth_confirmation_camera(target_yaw=restore_yaw, target_horizon=restore_horizon, video_fps=float(video_fps), label='confirmation restore {}'.format(intent.canonical_action))
        else:
            self._set_exact_confirmation_camera(target_yaw=restore_yaw, target_horizon=restore_horizon, label='confirmation restore {}'.format(intent.canonical_action))
        restore_records = copy.deepcopy(self._pending_low_level_records[restore_record_start:])
        restored_pose = self.executor.agent_pose()
        all_records = [*confirmation_records, *restore_records]
        self.total_low_level_steps += len(all_records)
        return {'observation': {'head_rgb': confirmation_rgb}, 'restored_observation': {'head_rgb': self.controller.last_event.frame}, 'low_level_actions': all_records, 'camera': {'schema_version': 'confirmation-camera.v1', 'view_kind': intent.view_kind, 'used_bound_coordinate': used_bound_coordinate, 'translation_speed_mps': 0.4, 'rotation_speed_radps': 0.8, 'video_fps_or_null': video_fps, 'requested_yaw': float(target_yaw) % 360.0, 'requested_horizon': target_horizon, 'actual_yaw': actual_yaw, 'actual_horizon': actual_horizon, 'restore_status': 'success', 'restored_yaw': float((restored_pose.get('rotation') or {}).get('y', 0.0) or 0.0) % 360.0, 'restored_horizon': float(restored_pose.get('cameraHorizon', 0.0) or 0.0), 'restore_frame_evidence_eligible': False}}

    def _set_exact_confirmation_camera(self, *, target_yaw: float, target_horizon: float, label: str) -> None:
        pose = self.executor.agent_pose()
        current_yaw = float((pose.get('rotation') or {}).get('y', 0.0) or 0.0) % 360.0
        yaw_delta = (float(target_yaw) - current_yaw + 180.0) % 360.0 - 180.0
        if abs(yaw_delta) > 0.001:
            self.executor._step_physical({'action': 'RotateRight' if yaw_delta > 0.0 else 'RotateLeft', 'degrees': abs(yaw_delta)}, label=label, source='confirmation')
        current_horizon = float(self.executor.agent_pose().get('cameraHorizon', 0.0) or 0.0)
        clamped_horizon = max(-30.0, min(60.0, float(target_horizon)))
        horizon_delta = clamped_horizon - current_horizon
        if abs(horizon_delta) > 0.001:
            self.executor._step_physical({'action': 'LookDown' if horizon_delta > 0.0 else 'LookUp', 'degrees': abs(horizon_delta)}, label=label, source='confirmation')

    def _smooth_confirmation_camera(self, *, target_yaw: float, target_horizon: float, video_fps: float, label: str) -> None:
        max_step_degrees = max(0.25, math.degrees(0.8) / video_fps)
        pose = self.executor.agent_pose()
        yaw = float((pose.get('rotation') or {}).get('y', 0.0)) % 360.0
        signed_yaw = (float(target_yaw) - yaw + 180.0) % 360.0 - 180.0
        yaw_steps = max(1, int(math.ceil(abs(signed_yaw) / max_step_degrees)))
        for _ in range(yaw_steps):
            remaining = (float(target_yaw) - yaw + 180.0) % 360.0 - 180.0
            if abs(remaining) < 0.001:
                break
            degrees = min(abs(remaining), max_step_degrees)
            action = 'RotateRight' if remaining > 0.0 else 'RotateLeft'
            self.executor._step_physical({'action': action, 'degrees': degrees}, label=label, source='confirmation_video')
            yaw = float((self.executor.agent_pose().get('rotation') or {}).get('y', yaw)) % 360.0
        current_horizon = float(self.executor.agent_pose().get('cameraHorizon', 0.0) or 0.0)
        target = max(-30.0, min(60.0, float(target_horizon)))
        horizon_steps = max(1, int(math.ceil(abs(target - current_horizon) / max_step_degrees)))
        for _ in range(horizon_steps):
            remaining = target - current_horizon
            if abs(remaining) < 0.001:
                break
            degrees = min(abs(remaining), max_step_degrees)
            self.executor._step_physical({'action': 'LookDown' if remaining > 0.0 else 'LookUp', 'degrees': degrees}, label=label, source='confirmation_video')
            current_horizon = float(self.executor.agent_pose().get('cameraHorizon', current_horizon) or current_horizon)

    def get_available_tools(self) -> List[str]:
        return list(MODEL_VISIBLE_TOOLS)

    def _make_executor(self, task_spec: Mapping[str, Any]) -> ProcTHORLowLevelExecutor:
        return ProcTHORLowLevelExecutor(env=self, task_spec=task_spec, max_nav_steps=self.max_nav_steps_per_tool, max_pose_candidates=self.max_pose_candidates, stream_callback=self._record_low_level_action, move_execution_step=self.move_execution_step, rotate_execution_degrees=self.rotate_execution_degrees, look_execution_degrees=self.look_execution_degrees, turn_penalty=self.turn_penalty, allow_force_navigation_fallback=self.allow_force_navigation_fallback, max_force_navigation_fallbacks_per_tool=self.max_force_navigation_fallbacks_per_tool, allow_force_open_container_fallback=self.allow_force_open_container_fallback, max_force_open_container_fallbacks_per_tool=self.max_force_open_container_fallbacks_per_tool, allow_force_arm_open_close_collision_fallback=self.allow_force_arm_open_close_collision_fallback, max_force_arm_open_close_collision_fallbacks_per_tool=self.max_force_arm_open_close_collision_fallbacks_per_tool)

    def _objects_by_id(self) -> Dict[str, Mapping[str, Any]]:
        metadata = self._metadata_with_sticky_object_states()
        return {obj['objectId']: obj for obj in metadata.get('objects', []) if isinstance(obj, Mapping) and isinstance(obj.get('objectId'), str)}

    def _raw_objects_by_id(self) -> Dict[str, Mapping[str, Any]]:
        metadata = self._metadata()
        return {obj['objectId']: obj for obj in metadata.get('objects', []) if isinstance(obj, Mapping) and isinstance(obj.get('objectId'), str)}

    def _metadata_with_sticky_object_states(self) -> Mapping[str, Any]:
        metadata = self._metadata()
        sticky = getattr(self, '_sticky_object_states', None) or {}
        merged: Mapping[str, Any] = metadata
        if sticky:
            updated_metadata = dict(metadata)
            merged_objects = []
            for obj in metadata.get('objects', []):
                if not isinstance(obj, Mapping):
                    merged_objects.append(obj)
                    continue
                object_id = obj.get('objectId')
                if isinstance(object_id, str) and object_id in sticky:
                    updated = dict(obj)
                    updated.update(sticky[object_id])
                    merged_objects.append(updated)
                else:
                    merged_objects.append(obj)
            updated_metadata['objects'] = merged_objects
            merged = updated_metadata
        return self._metadata_with_visual_attachment(merged)

    def _set_sticky_object_state(self, object_id: str, **state: Any) -> None:
        if not object_id:
            return
        object_id = str(object_id)
        if 'parentReceptacles' in state:
            self._public_placed_container_by_object.pop(object_id, None)
        if 'isOpen' in state:
            self._public_open_container_ids.discard(object_id)
        from scoper.benchmark.procthor.export.execution_audit import emit_overlay_audit
        emit_overlay_audit(self, kind='sticky_write', details={'object_id': object_id, 'fields': sorted(state), 'values': copy.deepcopy(dict(state))})
        current = self._sticky_object_states.setdefault(object_id, {})
        current.update(dict(state))

    def _record_low_level_action(self, action_record: Mapping[str, Any], event: Any) -> dict:
        record = dict(action_record)
        record.setdefault('high_level_step', self.high_level_step)
        sync_record = self._sync_visual_attachment_after_low_level_action(record)
        if sync_record is not None:
            record['visual_attach_sync'] = sync_record
        metadata = getattr(event, 'metadata', None)
        if isinstance(metadata, Mapping):
            self._agent_catalog_view = build_agent_catalog_view(metadata, tracker=self._agent_catalog_tracker, high_level_step=self.high_level_step)
        self._pending_low_level_records.append(record)
        return record

    def _sync_visual_attachment_after_low_level_action(self, action_record: Mapping[str, Any]) -> Optional[dict]:
        if not self._visual_attached_object_id:
            return None
        try:
            return self._sync_visual_attached_object_to_hand(label='{} [visual_attach_follow]'.format(action_record.get('label') or action_record.get('action') or 'low_level'), append_record=False)
        except Exception as exc:
            return {'success': False, 'error_message': str(exc), 'visual_attach_follow': True}

    def _agent_holding(self) -> bool:
        return bool(build_inventory_view(self._metadata_with_sticky_object_states()))

    def _held_inventory_object_id(self) -> Optional[str]:
        metadata = self._metadata_with_sticky_object_states()
        metadata = self._metadata_with_visual_attachment(metadata)
        inventory = build_inventory_view(metadata)
        for item in inventory:
            if isinstance(item, Mapping):
                object_id = item.get('objectId')
                if isinstance(object_id, str) and object_id:
                    return object_id
        return None

    def _use_visual_attachment_for_operation(self, tool_name: str) -> bool:
        return bool(self.manipulathor_arm) and bool(self.manipulathor_visual_attach_pickup) and (tool_name in {'pick_up', 'put_down'})

    def _execute_visual_attach_pickup(self, *, target_ref: str, object_id: str) -> dict:
        from scoper.benchmark.procthor.export.execution_audit import emit_overlay_audit
        emit_overlay_audit(self, kind='visual_attachment_pickup', details={'object_id': object_id, 'target_ref': target_ref}, applied=False, success=False)
        objects = self._objects_by_id()
        obj = objects.get(object_id)
        if obj is None:
            return {'status': STATUS_FAILED, 'action': 'PickupObject', 'target': target_ref, 'object_id': object_id, 'success': False, 'visual_attach_pickup': True, 'error_message': 'Object not found: {}.'.format(object_id)}
        if not bool(obj.get('pickupable', False)):
            return {'status': STATUS_FAILED, 'action': 'PickupObject', 'target': target_ref, 'object_id': object_id, 'success': False, 'visual_attach_pickup': True, 'error_message': 'Object is not pickupable: {}.'.format(object_id)}
        held_id = self._held_inventory_object_id()
        if held_id and held_id != object_id:
            return {'status': STATUS_FAILED, 'action': 'PickupObject', 'target': target_ref, 'object_id': object_id, 'held_object_id': held_id, 'success': False, 'visual_attach_pickup': True, 'error_message': 'Agent is already holding {}.'.format(held_id)}
        agent_yaw = _metadata_agent_yaw(self._metadata())
        rotation = dict(obj.get('rotation') or {'x': 0.0, 'y': agent_yaw, 'z': 0.0})
        self._visual_attached_object_id = object_id
        self._visual_attached_object_type = obj.get('objectType') or object_id.split('|', 1)[0]
        self._visual_attached_rotation_delta_y = float(rotation.get('y', agent_yaw)) - agent_yaw
        self._set_sticky_object_state(object_id, parentReceptacles=[])
        self._apply_visual_arm_pose(label='pick_up {} [visual_arm_pose]'.format(target_ref), append_record=False)
        sync_record = self._sync_visual_attached_object_to_hand(label='pick_up {} [visual_attach]'.format(target_ref), append_record=True)
        success = bool(sync_record.get('success', False))
        return {'status': STATUS_SUCCESS if success else STATUS_FAILED, 'action': 'PickupObject', 'target': target_ref, 'object_id': object_id, 'success': success, 'visual_attach_pickup': True, 'navigation_precondition': {'satisfied': True, 'skipped': True, 'reason': 'visual_attach_mode'}, 'held_object_id': object_id, 'sync': dict(sync_record), 'final_pose': self.executor.agent_pose() if self.executor is not None else {}, **({} if success else {'error_message': sync_record.get('error_message') or 'Visual attach failed.'})}

    def _execute_visual_attach_put_down(self, *, target_ref: str, receptacle_object_id: str, held_object_id: Optional[str]) -> dict:
        from scoper.benchmark.procthor.export.execution_audit import emit_overlay_audit
        held_id = self._visual_attached_object_id or held_object_id
        emit_overlay_audit(self, kind='visual_attachment_release', details={'object_id': held_id, 'receptacle_object_id': receptacle_object_id, 'target_ref': target_ref}, applied=False, success=False)
        if not held_id:
            return {'status': STATUS_FAILED, 'action': 'PutObject', 'target': target_ref, 'object_id': receptacle_object_id, 'success': False, 'visual_attach_put_down': True, 'error_message': 'Agent is not holding any object.'}
        objects = self._objects_by_id()
        receptacle = objects.get(receptacle_object_id)
        if receptacle is None:
            return {'status': STATUS_FAILED, 'action': 'PutObject', 'target': target_ref, 'object_id': receptacle_object_id, 'held_object_id': held_id, 'success': False, 'visual_attach_put_down': True, 'error_message': 'Receptacle not found: {}.'.format(receptacle_object_id)}
        position = self._visual_put_down_position(receptacle)
        if position is None:
            return {'status': STATUS_FAILED, 'action': 'PutObject', 'target': target_ref, 'object_id': receptacle_object_id, 'held_object_id': held_id, 'success': False, 'visual_attach_put_down': True, 'error_message': 'Target has no usable position: {}.'.format(receptacle_object_id)}
        rotation = self._visual_attached_rotation or dict(objects.get(held_id, {}).get('rotation') or {'x': 0.0, 'y': _metadata_agent_yaw(self._metadata()), 'z': 0.0})
        record = self._teleport_visual_object(object_id=held_id, position=position, rotation=rotation, label='put_down {} [visual_attach]'.format(target_ref), record_action='VisualPutObject', append_record=True)
        success = bool(record.get('success', False))
        if success:
            self._set_sticky_object_state(held_id, parentReceptacles=[receptacle_object_id])
            self._mark_visual_object_parent_in_metadata(held_object_id=held_id, receptacle_object_id=receptacle_object_id, position=position, rotation=rotation)
            self._remember_container_relative_placement(held_object_id=held_id, receptacle_object_id=receptacle_object_id, position=position, rotation=rotation)
            self._clear_visual_attachment_state(update_metadata=True)
        return {'status': STATUS_SUCCESS if success else STATUS_FAILED, 'action': 'PutObject', 'target': target_ref, 'object_id': receptacle_object_id, 'held_object_id': held_id, 'success': success, 'visual_attach_put_down': True, 'navigation_precondition': {'satisfied': True, 'skipped': True, 'reason': 'visual_attach_mode'}, 'sync': dict(record), 'final_pose': self.executor.agent_pose() if self.executor is not None else {}, **({} if success else {'error_message': record.get('error_message') or 'Visual put_down failed.'})}

    def _sync_visual_attached_object_to_hand(self, *, label: str, append_record: bool) -> dict:
        from scoper.benchmark.procthor.export.execution_audit import emit_overlay_audit
        object_id = self._visual_attached_object_id
        if not object_id:
            return {'success': False, 'error_message': 'No visual object is attached.'}
        emit_overlay_audit(self, kind='visual_attachment_follow', details={'object_id': object_id, 'label': label}, applied=False, success=False)
        metadata = self._metadata()
        anchor = self._visual_hand_anchor(metadata)
        if anchor is None:
            return {'success': False, 'error_message': 'Arm hand anchor is unavailable.'}
        attach_offset = self._visual_attach_offset_for_object(object_id, metadata)
        position = _offset_from_agent_yaw(anchor, attach_offset, _metadata_agent_yaw(metadata))
        rotation = {'x': 0.0, 'y': (_metadata_agent_yaw(metadata) + float(self._visual_attached_rotation_delta_y)) % 360.0, 'z': 0.0}
        record = self._teleport_visual_object(object_id=object_id, position=position, rotation=rotation, label=label, record_action='VisualAttachObject', append_record=append_record)
        record['visual_attach_offset'] = dict(attach_offset)
        obj = self._object_from_metadata(object_id, metadata) or {}
        bbox_size = None
        for bbox_key in ('objectOrientedBoundingBox', 'axisAlignedBoundingBox'):
            bbox_size = _bbox_size_from_metadata(obj.get(bbox_key))
            if bbox_size is not None:
                break
        bbox_height = None if bbox_size is None else abs(float(bbox_size.get('y', 0.0)))
        arm = metadata.get('arm') if isinstance(metadata, Mapping) else {}
        object_bottom_y = None if bbox_height is None else float(position['y']) - 0.5 * bbox_height
        center_offset_from_hand = {axis: float(position[axis]) - float(anchor[axis]) for axis in ('x', 'y', 'z')}
        record['private_visual_attach_geometry'] = {'hand_anchor': dict(anchor), 'hand_sphere_radius': arm.get('handSphereRadius') if isinstance(arm, Mapping) else None, 'bbox_height': bbox_height, 'final_offset': dict(attach_offset), 'object_bottom_y': object_bottom_y, 'object_center_offset_from_hand': center_offset_from_hand, 'anchor_contact_gap_m': None if object_bottom_y is None else object_bottom_y - float(anchor['y'])}
        if bool(record.get('success', False)):
            self._visual_attached_position = dict(position)
            self._visual_attached_rotation = dict(rotation)
            self._mark_visual_attached_in_metadata(object_id=object_id, position=position, rotation=rotation)
        return record

    def _apply_visual_arm_pose(self, *, label: str, append_record: bool) -> List[dict]:
        if not (bool(self.manipulathor_arm) and bool(self.manipulathor_visual_attach_pickup) and (self.controller is not None)):
            return []
        records: List[dict] = []
        object_pose_snapshot = self._snapshot_visual_arm_restorable_object_poses() if self.manipulathor_visual_arm_restore_object_poses else []
        radius = self.manipulathor_visual_arm_hand_sphere_radius
        if radius is not None:
            records.append(self._step_visual_arm_pose_action({'action': 'SetHandSphereRadius', 'radius': float(radius)}, label=label, append_record=append_record))
        base_y = self.manipulathor_visual_arm_base_y
        if base_y is not None:
            payload = {'action': 'MoveArmBase', 'y': float(base_y), 'speed': self.manipulathor_visual_arm_move_speed, 'returnToStart': False}
            if self.manipulathor_visual_arm_fixed_delta_time is not None:
                payload['fixedDeltaTime'] = self.manipulathor_visual_arm_fixed_delta_time
            records.append(self._step_visual_arm_pose_action(payload, label=label, append_record=append_record))
        wrist_position = self.manipulathor_visual_arm_wrist_position
        if wrist_position is not None:
            payload = {'action': 'MoveArm', 'position': dict(wrist_position), 'coordinateSpace': 'armBase', 'restrictMovement': False, 'speed': self.manipulathor_visual_arm_move_speed, 'returnToStart': False}
            if self.manipulathor_visual_arm_fixed_delta_time is not None:
                payload['fixedDeltaTime'] = self.manipulathor_visual_arm_fixed_delta_time
            records.append(self._step_visual_arm_pose_action(payload, label=label, append_record=append_record))
        if object_pose_snapshot:
            restore_record = self._restore_visual_arm_object_poses(object_pose_snapshot, label=label, append_record=append_record)
            if restore_record is not None:
                records.append(restore_record)
        self._remember_visual_hand_anchor_local_offset()
        return records

    def _step_visual_arm_pose_action(self, payload: Mapping[str, Any], *, label: str, append_record: bool) -> dict:
        event = self.controller.step(**dict(payload))
        record = {'action': 'VisualArmPose', 'executed_action': payload.get('action'), 'label': label, 'success': bool(event.metadata.get('lastActionSuccess', False)), 'error_message': event.metadata.get('errorMessage') or '', 'agent_pose': _agent_pose(event.metadata), 'execution_substep': 1, 'execution_substeps': 1, 'visual_attach': True, 'hidden_state_sync': True}
        if 'radius' in payload:
            record['radius'] = payload['radius']
        if 'y' in payload:
            record['arm_base_y'] = payload['y']
        if 'position' in payload:
            record['position'] = dict(payload.get('position') or {})
        if 'speed' in payload:
            record['speed'] = payload['speed']
        if 'fixedDeltaTime' in payload:
            record['fixed_delta_time'] = payload['fixedDeltaTime']
        if append_record:
            record.setdefault('high_level_step', self.high_level_step)
            self._pending_low_level_records.append(record)
        return record

    def _snapshot_visual_arm_restorable_object_poses(self) -> List[dict]:
        if self.controller is None:
            return []
        object_poses: List[dict] = []
        for obj in self.controller.last_event.metadata.get('objects', []) or []:
            if not isinstance(obj, Mapping):
                continue
            if not (bool(obj.get('pickupable')) or bool(obj.get('moveable'))):
                continue
            object_id = obj.get('objectId')
            object_name = obj.get('name') or object_id
            position = obj.get('position')
            rotation = obj.get('rotation')
            if not isinstance(object_name, str) or not object_name:
                continue
            if not _has_xyz(position):
                continue
            object_poses.append({'objectName': object_name.split('(Clone)', 1)[0], 'position': dict(position), 'rotation': dict(rotation) if isinstance(rotation, Mapping) else {'x': 0.0, 'y': 0.0, 'z': 0.0}})
        return object_poses

    def _restore_visual_arm_object_poses(self, object_poses: Sequence[Mapping[str, Any]], *, label: str, append_record: bool) -> Optional[dict]:
        if self.controller is None or not object_poses:
            return None
        event = self.controller.step(action='SetObjectPoses', objectPoses=[dict(pose) for pose in object_poses])
        record = {'action': 'VisualArmPoseRestore', 'executed_action': 'SetObjectPoses', 'label': label, 'success': bool(event.metadata.get('lastActionSuccess', False)), 'error_message': event.metadata.get('errorMessage') or '', 'agent_pose': _agent_pose(event.metadata), 'execution_substep': 1, 'execution_substeps': 1, 'object_pose_count': len(object_poses), 'visual_attach': True, 'hidden_state_sync': True}
        if append_record:
            record.setdefault('high_level_step', self.high_level_step)
            self._pending_low_level_records.append(record)
        return record

    def _teleport_visual_object(self, *, object_id: str, position: Mapping[str, Any], rotation: Mapping[str, Any], label: str, record_action: str, append_record: bool) -> dict:
        event = self.controller.step(action='TeleportObject', objectId=object_id, position={'x': float(position['x']), 'y': float(position['y']), 'z': float(position['z'])}, rotation=dict(rotation), forceAction=True)
        success = bool(event.metadata.get('lastActionSuccess', False))
        error_message = event.metadata.get('errorMessage') or ''
        if not success:
            fallback_event = self._set_visual_object_pose(object_id=object_id, position=position, rotation=rotation)
            if fallback_event is not None:
                event = fallback_event
                success = bool(event.metadata.get('lastActionSuccess', False))
                error_message = event.metadata.get('errorMessage') or ''
        record = {'action': record_action, 'executed_action': 'TeleportObject', 'label': label, 'success': success, 'error_message': error_message, 'agent_pose': self.executor.agent_pose() if self.executor is not None else {}, 'execution_substep': 1, 'execution_substeps': 1, 'object_id': object_id, 'position': dict(position), 'rotation': dict(rotation), 'visual_attach': True, 'hidden_state_sync': True}
        if append_record:
            record.setdefault('high_level_step', self.high_level_step)
            self._pending_low_level_records.append(record)
        return record

    def _set_visual_object_pose(self, *, object_id: str, position: Mapping[str, Any], rotation: Mapping[str, Any]) -> Optional[Any]:
        obj = self._raw_objects_by_id().get(object_id, {})
        object_name = obj.get('name') or object_id
        if not isinstance(object_name, str) or not object_name:
            return None
        return self.controller.step(action='SetObjectPoses', objectPoses=[{'objectName': object_name.split('(Clone)', 1)[0], 'position': dict(position), 'rotation': dict(rotation)}])

    def _visual_hand_anchor(self, metadata: Mapping[str, Any]) -> Optional[dict]:
        local_offset = self._visual_hand_anchor_local_offset
        agent = metadata.get('agent') if isinstance(metadata, Mapping) else {}
        if isinstance(local_offset, Mapping) and isinstance(agent, Mapping):
            position = agent.get('position')
            if _has_xyz(position):
                return _offset_from_agent_yaw(position, local_offset, _metadata_agent_yaw(metadata))
        arm = metadata.get('arm')
        if isinstance(arm, Mapping):
            center = arm.get('handSphereCenter')
            if _has_xyz(center):
                return {'x': float(center['x']), 'y': float(center['y']), 'z': float(center['z'])}
        agent = metadata.get('agent') if isinstance(metadata, Mapping) else {}
        if isinstance(agent, Mapping):
            position = agent.get('position')
            if _has_xyz(position):
                return _offset_from_agent_yaw(position, {'x': 0.0, 'y': 0.0, 'z': 0.45}, _metadata_agent_yaw(metadata))
        return None

    def _remember_visual_hand_anchor_local_offset(self, metadata: Optional[Mapping[str, Any]]=None) -> None:
        source = metadata if isinstance(metadata, Mapping) else self._metadata()
        arm = source.get('arm') if isinstance(source, Mapping) else {}
        agent = source.get('agent') if isinstance(source, Mapping) else {}
        if not isinstance(arm, Mapping) or not isinstance(agent, Mapping):
            return
        center = arm.get('handSphereCenter')
        position = agent.get('position')
        if not _has_xyz(center) or not _has_xyz(position):
            return
        self._visual_hand_anchor_local_offset = _local_offset_from_agent_yaw(origin=position, point=center, yaw_degrees=_metadata_agent_yaw(source))

    def _visual_attach_offset_for_object(self, object_id: str, metadata: Optional[Mapping[str, Any]]=None) -> dict:
        obj = self._object_from_metadata(object_id, metadata) or {}
        bbox_offset = _bbox_aware_visual_attach_offset(obj)
        manual = self.manipulathor_visual_attach_offset
        return {'x': bbox_offset['x'] + float(manual.get('x', 0.0)), 'y': bbox_offset['y'] + float(manual.get('y', 0.0)), 'z': bbox_offset['z'] + float(manual.get('z', 0.0))}

    def _object_from_metadata(self, object_id: str, metadata: Optional[Mapping[str, Any]]=None) -> Optional[Mapping[str, Any]]:
        source = metadata if isinstance(metadata, Mapping) else self._metadata()
        for obj in source.get('objects', []) or []:
            if isinstance(obj, Mapping) and obj.get('objectId') == object_id:
                return obj
        return None

    def _visual_put_down_position(self, receptacle: Mapping[str, Any]) -> Optional[dict]:
        position = receptacle.get('position')
        if not _has_xyz(position):
            return None
        y_offset = 0.25
        bbox = receptacle.get('axisAlignedBoundingBox')
        if isinstance(bbox, Mapping):
            center = bbox.get('center')
            size = bbox.get('size')
            if _has_xyz(center) and _has_xyz(size):
                y_offset = max(0.08, float(size.get('y', 0.0)) * 0.5 + 0.08)
        return {'x': float(position['x']), 'y': float(position['y']) + y_offset, 'z': float(position['z'])}

    def _mark_visual_attached_in_metadata(self, *, object_id: str, position: Mapping[str, Any], rotation: Mapping[str, Any]) -> None:
        metadata = self.controller.last_event.metadata
        arm = metadata.setdefault('arm', {})
        held = [item for item in arm.get('heldObjects', []) or [] if item != object_id]
        arm['heldObjects'] = [object_id] + held
        self._mark_object_pose_in_metadata(object_id=object_id, position=position, rotation=rotation, parent_receptacles=[])

    def _mark_visual_object_parent_in_metadata(self, *, held_object_id: str, receptacle_object_id: str, position: Mapping[str, Any], rotation: Mapping[str, Any]) -> None:
        self._mark_object_pose_in_metadata(object_id=held_object_id, position=position, rotation=rotation, parent_receptacles=[receptacle_object_id])
        metadata = self.controller.last_event.metadata
        for obj in metadata.get('objects', []) or []:
            if not isinstance(obj, dict):
                continue
            if obj.get('objectId') == receptacle_object_id:
                ids = list(obj.get('receptacleObjectIds') or [])
                if held_object_id not in ids:
                    ids.append(held_object_id)
                obj['receptacleObjectIds'] = ids
                break

    def _mark_object_pose_in_metadata(self, *, object_id: str, position: Mapping[str, Any], rotation: Mapping[str, Any], parent_receptacles: Sequence[str]) -> None:
        metadata = self.controller.last_event.metadata
        for obj in metadata.get('objects', []) or []:
            if not isinstance(obj, dict) or obj.get('objectId') != object_id:
                continue
            obj['position'] = dict(position)
            obj['rotation'] = dict(rotation)
            obj['parentReceptacles'] = list(parent_receptacles)
            if parent_receptacles:
                obj['isPickedUp'] = False
            else:
                obj['isPickedUp'] = True
            break

    def _metadata_with_visual_attachment(self, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        object_id = self._visual_attached_object_id
        if not object_id:
            return metadata
        merged = dict(metadata)
        arm = dict(merged.get('arm') or {})
        held = [item for item in arm.get('heldObjects', []) or [] if item != object_id]
        arm['heldObjects'] = [object_id] + held
        merged['arm'] = arm
        inventory = [dict(item) if isinstance(item, Mapping) else item for item in merged.get('inventoryObjects', []) or [] if not isinstance(item, Mapping) or item.get('objectId') != object_id]
        inventory.insert(0, {'objectId': object_id, 'objectType': self._visual_attached_object_type or object_id.split('|', 1)[0]})
        merged['inventoryObjects'] = inventory
        if self._visual_attached_position is not None:
            objects = []
            for obj in merged.get('objects', []) or []:
                if not isinstance(obj, Mapping) or obj.get('objectId') != object_id:
                    objects.append(obj)
                    continue
                updated = dict(obj)
                updated['position'] = dict(self._visual_attached_position)
                if self._visual_attached_rotation is not None:
                    updated['rotation'] = dict(self._visual_attached_rotation)
                updated['parentReceptacles'] = []
                updated['isPickedUp'] = True
                objects.append(updated)
            merged['objects'] = objects
        return merged

    def _clear_visual_attachment_state(self, *, update_metadata: bool=True) -> None:
        object_id = self._visual_attached_object_id
        self._visual_attached_object_id = None
        self._visual_attached_object_type = None
        self._visual_attached_rotation_delta_y = 0.0
        self._visual_attached_position = None
        self._visual_attached_rotation = None
        self._visual_hand_anchor_local_offset = None
        if update_metadata and object_id and (self.controller is not None):
            metadata = self.controller.last_event.metadata
            arm = metadata.setdefault('arm', {})
            arm['heldObjects'] = [item for item in arm.get('heldObjects', []) or [] if item != object_id]

    def _clear_navigation_block_cache(self) -> None:
        if self.executor is None:
            return
        blocked_edges = getattr(self.executor, 'blocked_edges', None)
        if hasattr(blocked_edges, 'clear'):
            blocked_edges.clear()

    def _try_logical_put_down_fallback(self, *, target_ref: str, receptacle_object_id: str, held_object_id: Optional[str], failed_execution: Mapping[str, Any]) -> Optional[dict]:
        if not held_object_id:
            return None
        if self._held_inventory_object_id() != held_object_id:
            return None
        receptacle = self._objects_by_id().get(receptacle_object_id, {})
        if not bool(receptacle.get('receptacle', False)):
            return None
        if not self._put_down_failed_due_to_no_valid_positions(failed_execution):
            return None
        allowed, fallback_context = self._put_down_fallback_context(receptacle_object_id, failed_execution)
        if not allowed:
            self._annotate_logical_put_down_rejection(failed_execution, fallback_context)
            return None
        from scoper.benchmark.procthor.export.execution_audit import emit_overlay_audit
        emit_overlay_audit(self, kind='logical_relation', details={'held_object_id': held_object_id, 'receptacle_object_id': receptacle_object_id, 'target_ref': target_ref}, applied=False, success=False)
        success = False
        for force_action in (False, True):
            payload = {'action': 'DropHandObject'}
            if force_action:
                payload['forceAction'] = True
            event = self.controller.step(**payload)
            success = bool(event.metadata.get('lastActionSuccess', False))
            action_record = {'action': 'DropHandObject', 'executed_action': 'DropHandObject', 'label': 'put_down {} [logical_put_down_fallback]'.format(target_ref), 'success': success, 'error_message': event.metadata.get('errorMessage') or '', 'agent_pose': self.executor.agent_pose() if self.executor is not None else {}, 'execution_substep': 1, 'execution_substeps': 1, 'object_id': held_object_id, 'held_object_id': held_object_id, 'target': target_ref, 'receptacle_object_id': receptacle_object_id, 'logical_put_down_fallback': True, 'logical_put_down_fallback_context': dict(fallback_context)}
            if force_action:
                action_record['forceAction'] = True
                action_record['force_put_object_fallback'] = True
                action_record['force_put_object_fallback_applied'] = success
            self._pending_low_level_records.append(action_record)
            if success:
                break
        if not success:
            return None
        self._move_logical_put_object_to_receptacle(held_object_id=held_object_id, receptacle_object_id=receptacle_object_id, target_ref=target_ref)
        recovered = dict(failed_execution)
        recovered.update({'status': STATUS_SUCCESS, 'success': True, 'logical_put_down_fallback': True, 'action': 'PutObject', 'target': target_ref, 'object_id': receptacle_object_id, 'held_object_id': held_object_id, 'message': 'PutObject had no valid spawn position; dropped held object and recorded intended receptacle relation.', 'logical_put_down_fallback_context': dict(fallback_context)})
        self._logical_put_down_object_ids.add(held_object_id)
        return recovered

    def _try_open_container_pickup_fallback(self, *, target_ref: str, object_id: str, failed_execution: Mapping[str, Any]) -> Optional[dict]:
        if self._agent_holding():
            return None
        objects = self._objects_by_id()
        obj = objects.get(object_id, {})
        if not bool(obj.get('pickupable', False)):
            return None
        if not self._pickup_failed_due_to_visibility(failed_execution):
            return None
        parent_id, context = self._open_container_pickup_context(object_id)
        if parent_id is None:
            if isinstance(failed_execution, dict):
                failed_execution['open_container_pickup_fallback_rejected'] = context
            return None
        event = self.controller.step(action='PickupObject', objectId=object_id, forceAction=True)
        success = bool(event.metadata.get('lastActionSuccess', False))
        action_record = {'action': 'PickupObject', 'executed_action': 'PickupObject', 'label': 'pick_up {} [force_open_container_pickup_fallback]'.format(target_ref), 'success': success, 'error_message': event.metadata.get('errorMessage') or '', 'agent_pose': self.executor.agent_pose() if self.executor is not None else {}, 'execution_substep': 1, 'execution_substeps': 1, 'object_id': object_id, 'target': target_ref, 'parent_container_id': parent_id, 'forceAction': True, 'force_pickup_open_container_fallback': True, 'force_pickup_open_container_fallback_applied': success, 'force_pickup_open_container_fallback_context': dict(context)}
        self._pending_low_level_records.append(action_record)
        if not success:
            return None
        recovered = dict(failed_execution)
        recovered.update({'status': STATUS_SUCCESS, 'success': True, 'action': 'PickupObject', 'target': target_ref, 'object_id': object_id, 'parent_container_id': parent_id, 'force_pickup_open_container_fallback': True, 'force_pickup_open_container_fallback_context': dict(context), 'message': 'PickupObject target was hidden inside a visible open container; applied controlled force pickup fallback.'})
        return recovered

    def _pickup_failed_due_to_visibility(self, failed_execution: Mapping[str, Any]) -> bool:
        texts: List[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key in ('error_message', 'message'):
                    item = value.get(key)
                    if isinstance(item, str):
                        texts.append(item)
                for item in value.get('attempts') or []:
                    visit(item)
                initial = value.get('initial_direct_attempt')
                if isinstance(initial, Mapping):
                    visit(initial)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(failed_execution)
        lowered = '\n'.join(texts).lower()
        return 'not found within the specified visibility' in lowered or 'target object is not visible' in lowered or "no candidate pose produced a successful 'pickupobject'" in lowered

    def _open_container_pickup_context(self, object_id: str) -> Tuple[Optional[str], dict]:
        objects = self._objects_by_id()
        obj = objects.get(object_id, {})
        parent_ids = [parent_id for parent_id in obj.get('parentReceptacles') or [] if isinstance(parent_id, str)]
        max_distance = self._put_down_fallback_max_distance()
        context = {'candidate_parent_ids': list(parent_ids), 'max_allowed_distance': max_distance}
        for parent_id in parent_ids:
            parent = objects.get(parent_id, {})
            if not bool(parent.get('openable', False)):
                continue
            if parent.get('isOpen') is not True:
                continue
            if not bool(parent.get('visible', False)):
                continue
            distance = self._object_xz_distance_to_agent(parent_id)
            if distance is None or distance > max_distance:
                continue
            context.update({'selected_parent_id': parent_id, 'selected_parent_type': parent.get('objectType'), 'selected_parent_distance': distance, 'reason': 'target_inside_visible_open_near_container'})
            return (parent_id, context)
        context['reason'] = 'no_visible_open_near_parent_container'
        return (None, context)

    def _put_down_fallback_context(self, receptacle_object_id: str, failed_execution: Mapping[str, Any]) -> Tuple[bool, dict]:
        max_distance = self._put_down_fallback_max_distance()
        visible_distances = self._visible_put_down_attempt_distances(failed_execution)
        current_distance = self._object_xz_distance_to_agent(receptacle_object_id)
        current_visible = bool(self._objects_by_id().get(receptacle_object_id, {}).get('visible', False))
        if current_visible and current_distance is not None:
            visible_distances.append(float(current_distance))
        best_distance = min(visible_distances) if visible_distances else None
        context = {'target_visible': bool(visible_distances), 'min_visible_distance': best_distance, 'max_allowed_distance': max_distance}
        if best_distance is None:
            context['reason'] = 'target_not_visible_for_put_down_fallback'
            return (False, context)
        if best_distance > max_distance:
            context['reason'] = 'target_too_far_for_put_down_fallback'
            return (False, context)
        context['reason'] = 'target_visible_and_near'
        return (True, context)

    def _put_down_fallback_max_distance(self) -> float:
        executor = getattr(self, 'executor', None)
        interaction_distance = getattr(executor, 'interaction_distance', None)
        try:
            return float(interaction_distance) + PUT_DOWN_FALLBACK_DISTANCE_MARGIN
        except (TypeError, ValueError):
            return DEFAULT_PUT_DOWN_FALLBACK_MAX_DISTANCE

    def _visible_put_down_attempt_distances(self, execution: Any) -> List[float]:
        distances: List[float] = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for attempt in value.get('attempts') or []:
                    if not isinstance(attempt, Mapping):
                        continue
                    visible = bool(attempt.get('visible_before_action') or attempt.get('visible_after_navigation'))
                    if visible:
                        distance = self._attempt_distance(attempt)
                        if distance is not None:
                            distances.append(distance)
                    visit(attempt)
                initial = value.get('initial_direct_attempt')
                if isinstance(initial, Mapping):
                    visit(initial)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(execution)
        return distances

    def _attempt_distance(self, attempt: Mapping[str, Any]) -> Optional[float]:
        pose = attempt.get('pose')
        if isinstance(pose, Mapping):
            distance = pose.get('distance')
            try:
                return float(distance)
            except (TypeError, ValueError):
                return None
        return None

    def _object_xz_distance_to_agent(self, object_id: str) -> Optional[float]:
        obj = self._objects_by_id().get(object_id, {})
        obj_pos = obj.get('position') or {}
        agent = self.controller.last_event.metadata.get('agent') or {}
        agent_pos = agent.get('position') or {}
        if not all((key in obj_pos for key in ('x', 'z'))):
            return None
        if not all((key in agent_pos for key in ('x', 'z'))):
            return None
        try:
            return math.hypot(float(obj_pos.get('x', 0.0)) - float(agent_pos.get('x', 0.0)), float(obj_pos.get('z', 0.0)) - float(agent_pos.get('z', 0.0)))
        except (TypeError, ValueError):
            return None

    def _annotate_logical_put_down_rejection(self, failed_execution: Mapping[str, Any], fallback_context: Mapping[str, Any]) -> None:
        if isinstance(failed_execution, dict):
            failed_execution['logical_put_down_fallback_rejected'] = dict(fallback_context)

    def _move_logical_put_object_to_receptacle(self, *, held_object_id: str, receptacle_object_id: str, target_ref: str) -> None:
        objects = self._raw_objects_by_id()
        held = objects.get(held_object_id, {})
        receptacle = objects.get(receptacle_object_id, {})
        object_name = held.get('name') or held.get('objectId')
        receptacle_position = receptacle.get('position') or {}
        if not isinstance(object_name, str) or not object_name:
            return
        if not {'x', 'y', 'z'}.issubset(receptacle_position):
            return
        target_position = {'x': float(receptacle_position.get('x', 0.0)), 'y': float(receptacle_position.get('y', 0.0)) + 0.25, 'z': float(receptacle_position.get('z', 0.0))}
        object_poses = []
        for obj in objects.values():
            if not (bool(obj.get('pickupable', False)) or bool(obj.get('moveable', False))):
                continue
            obj_name = obj.get('name') or obj.get('objectId')
            obj_position = obj.get('position') or {}
            if not isinstance(obj_name, str) or not obj_name:
                continue
            if not {'x', 'y', 'z'}.issubset(obj_position):
                continue
            pose_position = dict(obj_position)
            if obj.get('objectId') == held_object_id:
                pose_position = dict(target_position)
            object_poses.append({'objectName': obj_name.split('(Clone)', 1)[0], 'position': pose_position, 'rotation': dict(obj.get('rotation') or {'x': 0.0, 'y': 0.0, 'z': 0.0})})
        if not object_poses:
            return
        event = self.controller.step(action='SetObjectPoses', objectPoses=object_poses)
        placement_success = bool(event.metadata.get('lastActionSuccess', False))
        self._pending_low_level_records.append({'action': 'SetObjectPoses', 'executed_action': 'SetObjectPoses', 'label': 'put_down {} [logical_put_down_reposition]'.format(target_ref), 'success': placement_success, 'error_message': event.metadata.get('errorMessage') or '', 'agent_pose': self.executor.agent_pose() if self.executor is not None else {}, 'execution_substep': 1, 'execution_substeps': 1, 'object_id': held_object_id, 'target': target_ref, 'receptacle_object_id': receptacle_object_id, 'logical_put_down_fallback': True})
        if placement_success:
            child_rotation = dict(held.get('rotation') or {'x': 0.0, 'y': 0.0, 'z': 0.0})
            self._remember_container_relative_placement(held_object_id=held_object_id, receptacle_object_id=receptacle_object_id, position=target_position, rotation=child_rotation)

    def _remember_container_relative_placement(self, *, held_object_id: str, receptacle_object_id: str, position: Mapping[str, Any], rotation: Mapping[str, Any]) -> None:
        receptacle = self._objects_by_id().get(receptacle_object_id, {})
        if not bool(receptacle.get('openable', False)):
            return
        parent_position = receptacle.get('position') or {}
        if not _has_xyz(parent_position) or not _has_xyz(position):
            return
        parent_rotation = receptacle.get('rotation') or {}
        try:
            parent_yaw = float(parent_rotation.get('y', 0.0))
            child_yaw = float(rotation.get('y', 0.0))
        except (TypeError, ValueError):
            parent_yaw = 0.0
            child_yaw = 0.0
        self._custom_container_relative_placements[str(held_object_id)] = {'parent_object_id': str(receptacle_object_id), 'local_position': _local_offset_from_agent_yaw(parent_position, position, parent_yaw), 'local_rotation_y': child_yaw - parent_yaw}

    def _sync_custom_container_children(self, receptacle_object_id: str) -> List[dict]:
        parent_id = str(receptacle_object_id)
        parent = self._objects_by_id().get(parent_id, {})
        parent_position = parent.get('position') or {}
        if not _has_xyz(parent_position):
            return []
        parent_rotation = parent.get('rotation') or {}
        try:
            parent_yaw = float(parent_rotation.get('y', 0.0))
        except (TypeError, ValueError):
            parent_yaw = 0.0
        records: List[dict] = []
        for child_id, placement in list(self._custom_container_relative_placements.items()):
            if placement.get('parent_object_id') != parent_id:
                continue
            local_position = placement.get('local_position')
            if not isinstance(local_position, Mapping):
                continue
            target_position = _offset_from_agent_yaw(parent_position, local_position, parent_yaw)
            child = self._raw_objects_by_id().get(child_id, {})
            if not child:
                self._custom_container_relative_placements.pop(child_id, None)
                continue
            child_rotation = dict(child.get('rotation') or {'x': 0.0, 'y': 0.0, 'z': 0.0})
            try:
                child_rotation['y'] = (parent_yaw + float(placement.get('local_rotation_y', 0.0))) % 360.0
            except (TypeError, ValueError):
                pass
            event = self.controller.step(action='TeleportObject', objectId=child_id, position=target_position, rotation=child_rotation, forceAction=True)
            success = bool(event.metadata.get('lastActionSuccess', False))
            record = {'action': 'TeleportObject', 'executed_action': 'TeleportObject', 'label': 'sync {} with container {}'.format(child_id, parent_id), 'success': success, 'error_message': event.metadata.get('errorMessage') or '', 'object_id': child_id, 'receptacle_object_id': parent_id, 'container_pose_sync': True, 'agent_pose': self.executor.agent_pose() if self.executor is not None else {}, 'execution_substep': 1, 'execution_substeps': 1}
            self._pending_low_level_records.append(record)
            records.append({'name': 'container_pose_sync', 'target': parent_id, 'affected_objects': [child_id], 'details': record})
            if success:
                self._mark_object_pose_in_metadata(object_id=child_id, position=target_position, rotation=child_rotation, parent_receptacles=[parent_id])
        return records

    def _put_down_failed_due_to_no_valid_positions(self, failed_execution: Mapping[str, Any]) -> bool:
        texts: List[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key in ('error_message', 'message'):
                    item = value.get(key)
                    if isinstance(item, str):
                        texts.append(item)
                for item in value.get('attempts') or []:
                    visit(item)
                initial = value.get('initial_direct_attempt')
                if isinstance(initial, Mapping):
                    visit(initial)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(failed_execution)
        return any(('No valid positions to place object found' in text for text in texts))

    def _apply_state_transitions(self, tool_name: str, arguments: Mapping[str, Any]) -> List[dict]:
        events: List[dict] = []
        target_ref = arguments.get('target') if isinstance(arguments, Mapping) else None
        object_id = arguments.get('_resolved_object_id') if isinstance(arguments, Mapping) else None
        if target_ref:
            object_id = object_id or self.role_object_ids.get(target_ref) or (target_ref if target_ref in self._objects_by_id() else None)
        if tool_name == 'toggle_on' and object_id:
            events.extend(self._transition_on_faucet_toggle(object_id))
            objects = self._objects_by_id()
            obj = objects.get(object_id, {})
            if obj.get('objectType') == 'Microwave' and obj.get('isOpen') is False:
                events.extend(self._transition_microwave_heat(object_id))
        elif tool_name == 'pick_up' and object_id:
            self._logical_put_down_object_ids.discard(object_id)
            self._custom_container_relative_placements.pop(str(object_id), None)
            self._set_sticky_object_state(object_id, parentReceptacles=[])
            events.append({'name': 'sticky_parent_clear', 'target': object_id, 'affected_objects': [object_id]})
        elif tool_name == 'put_down' and object_id:
            held_id = arguments.get('_held_object_id')
            if isinstance(held_id, str) and held_id:
                self._set_sticky_object_state(held_id, parentReceptacles=[object_id])
                receptacle = self._objects_by_id().get(object_id, {})
                child = self._raw_objects_by_id().get(held_id, {})
                child_position = child.get('position') or {}
                if bool(receptacle.get('openable', False)) and bool(receptacle.get('isOpen', False)) and _has_xyz(child_position):
                    self._remember_container_relative_placement(held_object_id=held_id, receptacle_object_id=object_id, position=child_position, rotation=child.get('rotation') or {'x': 0.0, 'y': 0.0, 'z': 0.0})
                events.append({'name': 'sticky_parent_set', 'target': object_id, 'affected_objects': [held_id]})
            objects = self._objects_by_id()
            obj = objects.get(object_id, {})
            if obj.get('objectType') == 'Fridge':
                events.extend(self._transition_fridge_cool(object_id))
        elif tool_name == 'close' and object_id:
            objects = self._objects_by_id()
            obj = objects.get(object_id, {})
            object_type = obj.get('objectType', '')
            events.extend(self._sync_custom_container_children(object_id))
            if object_type == 'Microwave' and bool(obj.get('isToggled', False)):
                events.extend(self._transition_microwave_heat(object_id))
            elif object_type == 'Fridge':
                events.extend(self._transition_fridge_cool(object_id))
        if tool_name == 'open' and object_id:
            events.extend(self._sync_custom_container_children(object_id))
        if tool_name == 'pick_up' and object_id:
            self._public_placed_container_by_object.pop(object_id, None)
        elif tool_name == 'put_down' and object_id:
            held_id = arguments.get('_held_object_id')
            if isinstance(held_id, str) and held_id:
                self._public_placed_container_by_object[held_id] = object_id
        elif tool_name == 'open' and object_id:
            self._public_open_container_ids.add(object_id)
        elif tool_name == 'close' and object_id:
            self._public_open_container_ids.discard(object_id)
        return events

    def _rebind_roles_to_sliced_child(self, source_object_id: str) -> None:
        prefix = source_object_id + '|'
        sliced_children = sorted((object_id for object_id, obj in self._objects_by_id().items() if object_id.startswith(prefix) and bool(obj.get('pickupable', False))))
        if not sliced_children:
            return
        sliced_object_id = sliced_children[0]
        for role_name, object_id in list(self.role_object_ids.items()):
            if object_id == source_object_id:
                self.role_object_ids[role_name] = sliced_object_id

    def _transition_on_faucet_toggle(self, faucet_id: str) -> List[dict]:
        objects = self._objects_by_id()
        faucet = objects.get(faucet_id, {})
        if faucet.get('objectType') != 'Faucet':
            return []
        best_sink = None
        task_sink_id = self.role_object_ids.get('sink_basin')
        if task_sink_id and task_sink_id in objects:
            best_sink = objects[task_sink_id]
        else:
            faucet_pos = faucet.get('position') or {}
            best_dist = float('inf')
            for obj in objects.values():
                if obj.get('objectType') != 'SinkBasin':
                    continue
                sink_pos = obj.get('position') or {}
                dx = float(faucet_pos.get('x', 0)) - float(sink_pos.get('x', 0))
                dz = float(faucet_pos.get('z', 0)) - float(sink_pos.get('z', 0))
                d = (dx * dx + dz * dz) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_sink = obj
            if best_sink is None or best_dist > 2.0:
                return []
        fill_targets = self._goal_fill_targets()
        affected_clean: List[str] = []
        affected_fill: List[str] = []
        fill_liquids: Dict[str, str] = {}
        sticky_states = getattr(self, '_sticky_object_states', {}) or {}
        child_ids = list(best_sink.get('receptacleObjectIds') or [])
        sink_id = best_sink.get('objectId')
        if isinstance(sink_id, str):
            for candidate in objects.values():
                candidate_id = candidate.get('objectId')
                if not isinstance(candidate_id, str):
                    continue
                if sink_id in (candidate.get('parentReceptacles') or []):
                    child_ids.append(candidate_id)
        seen_child_ids = set()
        for child_id in child_ids:
            if child_id in seen_child_ids:
                continue
            seen_child_ids.add(child_id)
            child = objects.get(child_id, {})
            sticky_dirty = isinstance(child_id, str) and sticky_states.get(child_id, {}).get('isDirty') is True
            if (bool(child.get('dirtyable', False)) or sticky_dirty) and bool(child.get('isDirty', False)):
                try:
                    self.controller.step(action='CleanObject', objectId=child_id, forceAction=True)
                except Exception:
                    pass
                self._set_sticky_object_state(child_id, isDirty=False)
                affected_clean.append(child_id)
            wanted_liquid = fill_targets.get(child_id)
            if wanted_liquid and bool(child.get('canFillWithLiquid', False)) and (not bool(child.get('isFilledWithLiquid', False))):
                try:
                    event = self.controller.step(action='FillObjectWithLiquid', objectId=child_id, fillLiquid=wanted_liquid, forceAction=True)
                    if event.metadata.get('lastActionSuccess', False):
                        self._set_sticky_object_state(child_id, isFilledWithLiquid=True, fillLiquid=wanted_liquid)
                        affected_fill.append(child_id)
                        fill_liquids[child_id] = wanted_liquid
                except Exception:
                    pass
        events: List[dict] = []
        if affected_clean:
            events.append({'name': 'faucet_clean', 'target': faucet_id, 'sink_id': best_sink.get('objectId'), 'affected_objects': affected_clean})
        if affected_fill:
            liquid_values = sorted(set(fill_liquids.values()))
            events.append({'name': 'faucet_fill', 'target': faucet_id, 'sink_id': best_sink.get('objectId'), 'affected_objects': affected_fill, 'liquid': liquid_values[0] if len(liquid_values) == 1 else None, 'liquids': dict(fill_liquids)})
        return events

    def _goal_fill_targets(self) -> Dict[str, str]:
        task_spec = self._active_task_spec()
        goal = task_spec.get('goal', {}) if isinstance(task_spec, Mapping) else {}
        objects = self._objects_by_id()
        targets: Dict[str, str] = {}
        for predicate in goal.get('predicates', []):
            if not isinstance(predicate, Mapping):
                continue
            if predicate.get('type') != 'object_filled':
                continue
            role = predicate.get('object_role')
            object_id = self.role_object_ids.get(role) or (role if isinstance(role, str) and role in objects else None)
            if object_id:
                targets[object_id] = str(predicate.get('liquid') or 'water')
        return targets

    def _transition_microwave_heat(self, microwave_id: str) -> List[dict]:
        objects = self._objects_by_id()
        microwave = objects.get(microwave_id, {})
        if microwave.get('objectType') != 'Microwave':
            return []
        goal_hot_ids = self._goal_object_ids_for_predicate('object_hot')
        affected: List[str] = []
        child_ids = set(microwave.get('receptacleObjectIds') or [])
        sticky_states = getattr(self, '_sticky_object_states', None) or {}
        for object_id, sticky_state in sticky_states.items():
            if microwave_id in (sticky_state.get('parentReceptacles') or []):
                child_ids.add(object_id)
        for child_id in child_ids:
            if goal_hot_ids and (not _object_or_slice_child_matches_goal(child_id, goal_hot_ids)):
                continue
            child = objects.get(child_id, {})
            if not (bool(child.get('canChangeTempToHot', False)) or bool(child.get('cookable', False)) or bool(child.get('heatable', False)) or bool(goal_hot_ids)):
                continue
            try:
                self.controller.step(action='SetObjectStates', SetObjectStates={'objectType': child.get('objectType'), 'isHeated': True})
            except Exception:
                pass
            self._set_sticky_object_state(child_id, isHeated=True)
            affected.append(child_id)
        if affected:
            return [{'name': 'microwave_heat', 'target': microwave_id, 'affected_objects': affected}]
        return []

    def _transition_fridge_cool(self, fridge_id: str) -> List[dict]:
        objects = self._objects_by_id()
        fridge = objects.get(fridge_id, {})
        if fridge.get('objectType') != 'Fridge':
            return []
        goal_cool_ids = self._goal_object_ids_for_predicate('object_cool')
        if not goal_cool_ids:
            return []
        affected: List[str] = []
        for child_id in fridge.get('receptacleObjectIds') or []:
            if not _object_or_slice_child_matches_goal(child_id, goal_cool_ids):
                continue
            child = objects.get(child_id, {})
            if not (bool(child.get('canChangeTempToCold', False)) or bool(child.get('coolable', False)) or bool(child.get('pickupable', False))):
                continue
            try:
                self.controller.step(action='SetObjectStates', SetObjectStates={'objectType': child.get('objectType'), 'isCooled': True})
            except Exception:
                pass
            self._set_sticky_object_state(child_id, isCooled=True)
            affected.append(child_id)
        if affected:
            return [{'name': 'fridge_cool', 'target': fridge_id, 'affected_objects': affected}]
        return []

    def _goal_object_ids_for_predicate(self, predicate_type: str) -> set[str]:
        task_spec = self._active_task_spec()
        roles = task_spec.get('roles') or {}
        out: set[str] = set()
        for predicate in (task_spec.get('goal') or {}).get('predicates', []):
            if not isinstance(predicate, Mapping):
                continue
            if predicate.get('type') != predicate_type:
                continue
            role_name = predicate.get('object_role')
            role = roles.get(role_name) if isinstance(role_name, str) else None
            object_id = role.get('object_id') if isinstance(role, Mapping) else None
            if isinstance(object_id, str):
                out.add(object_id)
        return out

    def _navigation_precondition(self, target_ref: str, object_id: str) -> dict:
        ready = self._ready_target or {}
        satisfied = ready.get('object_id') == object_id and bool(ready.get('interaction_ready', False))
        return {'satisfied': bool(satisfied), 'ready_target': dict(ready) if ready else None, 'expected_target': target_ref, 'expected_object_id': object_id}

    def _resolve_target(self, arguments: Mapping[str, Any]) -> Tuple[str, str]:
        resolved = self._resolve_operation_target(arguments)
        if resolved.object_id is None:
            raise ProcTHORClosedLoopToolEnvError("Target '{}' did not resolve to an object.".format(resolved.target_ref))
        return (resolved.target_ref, resolved.object_id)

    def _resolve_navigation_target(self, arguments: Mapping[str, Any]) -> ResolvedTarget:
        if self.current_task_spec is None:
            raise ProcTHORClosedLoopToolEnvError('No current task.')
        target = extract_tool_target(arguments)
        receptacle_intent = self._binding_reference_is_receptacle(target)
        try:
            resolved = resolve_navigation_target(target, role_object_ids=self.role_object_ids, objects_by_id=self._raw_objects_by_id(), tracker=self._agent_catalog_tracker, agent_position=_agent_pose(self._metadata()).get('position') or {}, rooms=self._rooms_for_navigation, binding_config=self.object_binding_config, task_spec=self.current_task_spec, ready_target=self._ready_target, publicly_processed_roles=self._publicly_processed_binding_roles(), active_public_event_roles=self._active_public_event_binding_roles(), scene_observation_roles=self._scene_observation_binding_roles(), local_interaction_context_active=self._local_interaction_context_active, benevolent_role_order=self._benevolent_binding_role_order())
            self._record_binding_attempt(resolved, receptacle_intent=receptacle_intent)
            return resolved
        except NLTargetResolverError as exc:
            self._record_binding_failure(exc, receptacle_intent=receptacle_intent)
            error = ProcTHORClosedLoopToolEnvError(str(exc))
            error.binding_public_record = exc.binding_public_record
            raise error from exc

    def _resolve_operation_target(self, arguments: Mapping[str, Any], *, tool_name: str='operation', bypass_interaction_readiness: bool=False) -> ResolvedTarget:
        if self.current_task_spec is None:
            raise ProcTHORClosedLoopToolEnvError('No current task.')
        target = extract_tool_target(arguments)
        receptacle_intent = self._binding_reference_is_receptacle(target)
        try:
            resolved = resolve_operation_target(target, role_object_ids=self.role_object_ids, objects_by_id=self._raw_objects_by_id(), tracker=self._agent_catalog_tracker, agent_position=_agent_pose(self._metadata()).get('position') or {}, binding_config=self.object_binding_config, task_spec=self.current_task_spec, tool_name=tool_name, ready_target=self._ready_target, publicly_processed_roles=self._publicly_processed_binding_roles(), active_public_event_roles=self._active_public_event_binding_roles(), scene_observation_roles=self._scene_observation_binding_roles(), local_interaction_context_active=self._local_interaction_context_active, bypass_interaction_readiness=bypass_interaction_readiness, benevolent_role_order=self._benevolent_binding_role_order())
            self._record_binding_attempt(resolved, receptacle_intent=receptacle_intent)
            return resolved
        except NLTargetResolverError as exc:
            self._record_binding_failure(exc, receptacle_intent=receptacle_intent)
            error = ProcTHORClosedLoopToolEnvError(str(exc))
            error.binding_public_record = exc.binding_public_record
            raise error from exc

    def enable_locked_recovery_admission_bypass(self, tool_call: Mapping[str, Any]) -> None:
        tool_name = str(tool_call.get('tool_name') or '')
        arguments = tool_call.get('arguments')
        if not tool_name or not isinstance(arguments, Mapping):
            raise ProcTHORClosedLoopToolEnvError('Locked recovery bypass requires a public ToolCall.')
        self._locked_recovery_admission_bypass = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments))}

    def suppress_completed_tool_call(self, tool_call: Mapping[str, Any]) -> None:
        tool_name = str(tool_call.get('tool_name') or '')
        arguments = tool_call.get('arguments')
        if not tool_name or not isinstance(arguments, Mapping):
            raise ProcTHORClosedLoopToolEnvError('Completed-action suppression requires a public ToolCall.')
        self._suppressed_completed_tool_call = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments))}

    def reject_public_precondition_tool_call(self, tool_call: Mapping[str, Any], message: str) -> None:
        tool_name = str(tool_call.get('tool_name') or '')
        arguments = tool_call.get('arguments')
        if not tool_name or not isinstance(arguments, Mapping):
            raise ProcTHORClosedLoopToolEnvError('Public-precondition rejection requires a public ToolCall.')
        self._public_precondition_rejection = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments)), 'message': str(message)}

    def mark_harness_directed_tool_call(self, tool_call: Mapping[str, Any]) -> None:
        tool_name = str(tool_call.get('tool_name') or '')
        arguments = tool_call.get('arguments')
        if not tool_name or not isinstance(arguments, Mapping):
            raise ProcTHORClosedLoopToolEnvError('Harness-directed recovery requires a public ToolCall.')
        self._harness_directed_tool_call = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments))}

    def _execute_resolved_navigation(self, resolved: ResolvedTarget) -> dict:
        from scoper.benchmark.common import evaluation_navigation
        if evaluation_navigation.enabled(self) and resolved.object_id is not None:
            previous = evaluation_navigation.consume_previous_position(self, resolved.object_id)
            if previous is not None:
                return self.executor.navigate_to_position(previous, label='navigate to {}'.format(resolved.target_ref), target_ref=resolved.target_ref, resolved_via='last_known_position', object_id=resolved.object_id, allow_failure=True)
        parent_id = self._public_placed_open_container_parent(resolved)
        if parent_id is not None:
            raw_execution = self.executor.navigate_to_object_id(parent_id, label='navigate to {} via open container'.format(resolved.target_ref), target_ref=resolved.target_ref, allow_failure=True)
            raw_execution.update({'object_id': resolved.object_id, 'parent_container_id': parent_id, 'resolved_via': 'public_placed_open_container', 'child_visibility_postcondition_skipped': raw_execution.get('status') == STATUS_SUCCESS, 'object_visibility_guaranteed': False})
            return raw_execution
        if resolved.object_id is not None and (evaluation_navigation.enabled(self) or resolved.resolved_via != 'last_known_position'):
            return self.executor.navigate_to_object_id(resolved.object_id, label='navigate to {}'.format(resolved.target_ref), target_ref=resolved.target_ref, allow_failure=True)
        if resolved.position is None:
            raise ProcTHORClosedLoopToolEnvError("Resolved navigation target '{}' has no position.".format(resolved.target_ref))
        raw_execution = self.executor.navigate_to_position(resolved.position, label='navigate to {}'.format(resolved.target_ref), target_ref=resolved.target_ref, resolved_via=resolved.resolved_via, object_id=resolved.object_id, allow_failure=True)
        if resolved.object_id is not None:
            raw_execution['object_id'] = resolved.object_id
        if resolved.last_seen_step is not None:
            raw_execution['last_seen_step'] = resolved.last_seen_step
        if resolved.last_known_position is not None:
            raw_execution['last_known_position'] = dict(resolved.last_known_position)
        if resolved.room_id is not None:
            raw_execution['room_id'] = resolved.room_id
        return raw_execution

    def _public_placed_open_container_parent(self, resolved: ResolvedTarget) -> Optional[str]:
        from scoper.benchmark.common import evaluation_navigation
        if not self.allow_public_placed_open_container_navigation and (not evaluation_navigation.enabled(self)) or resolved.object_id is None or resolved.binding_role is None or (self.role_object_ids.get(resolved.binding_role) != resolved.object_id):
            return None
        parent_id = self._public_placed_container_by_object.get(resolved.object_id)
        if parent_id not in self._public_open_container_ids:
            return None
        return parent_id

    def _ready_target_from_navigation(self, resolved: ResolvedTarget, raw_execution: Mapping[str, Any]) -> Optional[dict]:
        if resolved.object_id is None:
            return None
        return {'target': resolved.target_ref, 'object_id': resolved.object_id, 'interaction_ready': bool(raw_execution.get('interaction_ready')), 'interaction_pose_source': raw_execution.get('interaction_pose_source'), 'resolved_via': resolved.resolved_via, 'canonical_public_ref': (resolved.binding_public_record or {}).get('canonical_public_ref') or resolved.target_ref, 'binding_role': resolved.binding_role}

    def _role_handles(self, include_object_ids: bool=False) -> dict:
        handles = {}
        if self.current_task_spec is None:
            return handles
        task_spec = self._active_task_spec()
        objects_by_id = self._objects_by_id()
        entries_by_id = self.object_catalog.entries_by_object_id if self.object_catalog is not None else {}
        for role_name, role in task_spec['roles'].items():
            object_id = role['object_id']
            obj = objects_by_id.get(object_id, {})
            entry = entries_by_id.get(object_id)
            handle = {'handle': role_name, 'object_alias': entry.alias if entry is not None else None, 'object_type': role.get('object_type'), 'visible': bool(obj.get('visible', False)), 'distance': obj.get('distance'), 'pickupable': bool(obj.get('pickupable', False)), 'receptacle': bool(obj.get('receptacle', False)), 'openable': bool(obj.get('openable', False)), 'toggleable': bool(obj.get('toggleable', False))}
            if include_object_ids:
                handle['object_id'] = object_id
            handles[role_name] = handle
        return handles

    def _refresh_object_catalog(self) -> None:
        self.object_catalog = build_object_catalog(self._metadata(), previous_catalog=self.object_catalog)

    def _refresh_agent_catalog_view(self) -> None:
        self._agent_catalog_view = build_agent_catalog_view(self._metadata(), tracker=self._agent_catalog_tracker, high_level_step=self.high_level_step)
        for entry in self._agent_catalog_view:
            if isinstance(entry, dict) and entry.get('room_id') is not None:
                entry['room_id'] = self._public_room_id(entry.get('room_id'))

    def _build_navigation_rooms(self) -> List[dict]:
        house = getattr(self, 'current_house', None) or {}
        try:
            reachable_positions = self.executor.reachable_positions() if self.executor is not None else []
        except Exception:
            reachable_positions = []
        rooms = []
        for index, room in enumerate(house.get('rooms', []) or []):
            if not isinstance(room, Mapping):
                continue
            room_id = room.get('id') or room.get('roomId') or room.get('room_id') or 'room_{}'.format(index)
            room_type = room.get('roomType') or room.get('room_type') or room.get('type')
            representative_pose = _representative_reachable_pose_for_room(room, reachable_positions)
            from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding
            issue_pose = None if task_uses_evidence_binding(self.current_task_spec) else self._issue_observation_pose_for_room(room, reachable_positions)
            if issue_pose is not None:
                representative_pose = issue_pose
            try:
                public_type = room_display_label(room_type)
            except ValueError:
                public_type = re.sub('(?<!^)(?=[A-Z])', ' ', str(room_type or 'room')).lower()
            rooms.append({'room_id': room_id, 'room_type': public_type, 'representative_reachable_pose': representative_pose})
        type_counts = {room['room_type']: 0 for room in rooms}
        from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding, public_room_records
        if task_uses_evidence_binding(self.current_task_spec):
            return public_room_records(rooms)
        for room in rooms:
            type_counts[room['room_type']] += 1
        type_indices: Dict[str, int] = {}
        for room in rooms:
            room_type = room['room_type']
            type_indices[room_type] = type_indices.get(room_type, 0) + 1
            room['label'] = room_type if type_counts[room_type] == 1 else '{} {}'.format(room_type, type_indices[room_type])
        return rooms

    def _issue_observation_pose_for_room(self, room: Mapping[str, Any], reachable_positions: Sequence[Mapping[str, Any]]):
        task = getattr(self, 'current_task_spec', None)
        if not isinstance(task, Mapping):
            return None
        family = task.get('family_spec') if isinstance(task, Mapping) else {}
        obs = family.get('observation_conditioned') if isinstance(family, Mapping) else {}
        payloads = obs.get('latent_payloads') if isinstance(obs, Mapping) else None
        if not isinstance(payloads, list):
            return None
        room_id = str(room.get('id') or room.get('roomId') or room.get('room_id') or '')
        visible = {str(i.get('latent_id') or i.get('issue_id') or '') for i in self._observation_conditioned_visible_issues}
        objects = self._raw_objects_by_id()
        targets = []
        for payload in payloads:
            if not isinstance(payload, Mapping) or str(payload.get('room_id') or '') != room_id:
                continue
            if str(payload.get('latent_id') or payload.get('issue_id') or '') in visible:
                continue
            oid = self.role_object_ids.get(str(payload.get('object_role') or ''))
            pos = objects.get(str(oid), {}).get('position') if oid else None
            if isinstance(pos, Mapping) and 'x' in pos and ('z' in pos):
                targets.append((float(pos['x']), float(pos['z'])))
        if not targets or not reachable_positions:
            return None
        valid = []
        for p in reachable_positions:
            if not isinstance(p, Mapping) or 'x' not in p or 'z' not in p:
                continue
            px, pz = (float(p['x']), float(p['z']))
            distance = min((math.hypot(px - tx, pz - tz) for tx, tz in targets))
            if 0.8 <= distance <= 2.0:
                valid.append((distance, p))
        if not valid:
            return None
        _, selected = min(valid, key=lambda row: row[0])
        return {'position': {'x': float(selected['x']), 'y': float(selected.get('y', 0.0)), 'z': float(selected['z'])}, 'rotation': {'y': 0.0}, 'horizon': 0.0}

    def _public_room_id(self, room_id: Any) -> Optional[str]:
        if not isinstance(room_id, str) or not room_id:
            return None
        for room in self._rooms_for_navigation:
            if str(room.get('room_id') or '') == room_id:
                label = room.get('label')
                return str(label) if isinstance(label, str) and label else None
        if '|' not in room_id:
            return room_id
        return None

    def public_room_catalog(self) -> list[dict[str, str]]:
        return [{'room_id': str(room['label']), 'label': str(room['label']), 'room_type': str(room['room_type'])} for room in self._rooms_for_navigation if room.get('label')]

    def _observation(self) -> dict:
        event = self.controller.last_event
        metadata = self._metadata_with_sticky_object_states()
        inventory = build_inventory_view(metadata)
        task_spec = self._active_task_spec()
        return {'head_rgb': event.frame, 'task_id': task_spec['id'] if task_spec else None, 'instruction': task_spec['instruction'] if task_spec else None, 'agent_pose': _agent_pose(metadata), 'current_room_id': self._public_room_id(self._current_room_id), 'visible_issues': [dict(issue) for issue in self._observation_conditioned_visible_issues], 'held_objects': [dict(obj) for obj in inventory], 'object_catalog': [dict(entry) for entry in self._agent_catalog_view]}

    def _non_actuating_tool_result(self, tool_name: str, success: bool, feedback: str, raw_execution: Mapping[str, Any]) -> dict:
        goal_result = self.check_goal(model_stopped=self.model_stopped)
        done, done_reason = self._done_status(goal_result)
        return {'tool_name': tool_name, 'arguments': {}, 'high_level_step': self.high_level_step, 'task_action_step': self.task_action_step, 'success': success, 'env_feedback': feedback, 'done': done, 'done_reason': done_reason, 'observation': self._observation(), 'goal_result': goal_result, 'raw_execution': dict(raw_execution), 'low_level_actions': [], 'num_low_level_steps': 0, 'episode_metrics': self._episode_metrics(), 'ready_target': dict(self._ready_target) if self._ready_target else None}

    def _done_status(self, goal_result: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
        if self._human_qa_state.get('pending') or {}:
            return (False, None)
        if self.model_stopped:
            return (True, 'model_stop')
        if self.task_action_step >= self.max_high_level_steps:
            return (True, 'max_high_level_steps')
        if self.total_low_level_steps >= self.max_total_low_level_steps:
            return (True, 'max_total_low_level_steps')
        if self.invalid_tool_calls >= self.max_invalid_tool_calls:
            return (True, 'max_invalid_tool_calls')
        if isinstance(getattr(self, '_composite_active_task_spec', None), Mapping):
            return (False, None)
        if float(goal_result.get('task_success', 0.0)) == 1.0 and (not bool(goal_result.get('require_model_stop', False))):
            return (True, 'task_success')
        return (False, None)

    def _episode_metrics(self) -> dict:
        return {'high_level_steps': self.high_level_step, 'task_action_steps': self.task_action_step, 'total_low_level_steps': self.total_low_level_steps, 'invalid_tool_calls': self.invalid_tool_calls, 'model_stopped': self.model_stopped, 'scan_steps': self.scan_steps, 'look_for_steps': self.look_for_steps, 'answer_steps': self.answer_steps, 'force_arm_open_close_collision_fallback_count': int(self._force_arm_open_close_collision_fallbacks_total), 'force_viewpoint_fallback_count': int(self._force_viewpoint_fallbacks_total), 'visible_issue_count': len(self._observation_conditioned_visible_issues), 'observation_conditioned_scan_count': len(self._observation_conditioned_scan_events), 'binding_metrics': {'attempts': self._binding_attempts, 'successes': self._binding_successes, 'failures': dict(sorted(self._binding_failures.items())), 'failures_by_reason': dict(sorted(self._binding_failure_reasons.items())), 'receptacle_binding_attempts': self._receptacle_binding_attempts, 'receptacle_binding_successes': self._receptacle_binding_successes, 'receptacle_binding_failures': self._receptacle_binding_failures}, 'navigation_metrics': {'seen_object_navigation': self.seen_object_navigation_steps, 'stale_object_navigation': self.stale_object_navigation_steps, 'unseen_object_exploration': self.unseen_object_exploration_steps, 'target_missing_at_last_seen': self.target_missing_at_last_seen_count}}

    def _record_binding_attempt(self, resolved: ResolvedTarget, *, receptacle_intent: bool=False) -> None:
        if resolved.binding_public_record is None:
            return
        self._binding_attempts += 1
        self._binding_successes += 1
        if resolved.binding_role:
            self._current_tool_binding_roles.add(str(resolved.binding_role))
        is_receptacle = receptacle_intent or self._resolved_binding_is_receptacle(resolved)
        if is_receptacle:
            self._receptacle_binding_attempts += 1
            self._receptacle_binding_successes += 1

    def _record_binding_failure(self, exc: Exception, *, receptacle_intent: bool=False) -> None:
        self._binding_attempts += 1
        reason = str(exc).split(':', 1)[0].strip().lower().replace(' ', '_')
        self._binding_failures[reason] = self._binding_failures.get(reason, 0) + 1
        detail = (getattr(exc, 'binding_public_record', None) or {}).get('reason', 'unresolved_reference')
        self._binding_failure_reasons[detail] = self._binding_failure_reasons.get(detail, 0) + 1
        if receptacle_intent:
            self._receptacle_binding_attempts += 1
            self._receptacle_binding_failures += 1

    def _resolved_binding_is_receptacle(self, resolved: ResolvedTarget) -> bool:
        if resolved.binding_role and self._binding_role_is_receptacle(str(resolved.binding_role)):
            return True
        if resolved.object_id:
            obj = self._raw_objects_by_id().get(str(resolved.object_id), {})
            return bool(obj.get('receptacle', False))
        return False

    def _binding_reference_is_receptacle(self, reference: str) -> bool:
        task_spec = self._active_task_spec()
        binding_spec = task_spec.get('task_binding_spec') if isinstance(task_spec, Mapping) else None
        role_specs = binding_spec.get('roles') if isinstance(binding_spec, Mapping) else None
        if not isinstance(role_specs, Mapping):
            return False
        query_forms = reference_forms(reference)
        matching_roles = set()
        for role_name, role_spec in role_specs.items():
            if not isinstance(role_spec, Mapping):
                continue
            public_forms: set[str] = set()
            terms = (*tuple(role_spec.get('public_terms') or ()), *tuple(role_spec.get('generic_terms') or ()))
            for term in terms:
                public_forms.update(reference_forms(term))
            if query_forms & public_forms:
                matching_roles.add(str(role_name))
        return len(matching_roles) == 1 and self._binding_role_is_receptacle(next(iter(matching_roles)))

    def _binding_role_is_receptacle(self, role_name: str) -> bool:
        object_id = self.role_object_ids.get(role_name)
        if not object_id:
            return False
        obj = self._raw_objects_by_id().get(str(object_id), {})
        return bool(obj.get('receptacle', False))

    def _publicly_processed_binding_roles(self) -> Tuple[str, ...]:
        from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding, public_placed_roles
        if task_uses_evidence_binding(self.current_task_spec):
            return public_placed_roles(self.current_task_spec, self.tool_history, self.role_object_ids)
        return self._private_processed_binding_roles

    def _benevolent_binding_role_order(self) -> Tuple[str, ...]:
        if not self.object_binding_config.benevolent_task_progress:
            return ()
        try:
            task = self._active_task_spec()
            result = evaluate_predicate_goal(task, self._metadata_with_sticky_object_states(), model_stopped=False)
            statuses = tuple((bool(item.get('satisfied')) for item in result.get('predicates', ()) if isinstance(item, Mapping)))
            held_ids = {str(item.get('objectId')) for item in self._metadata().get('inventoryObjects', ()) if isinstance(item, Mapping) and item.get('objectId')}
            held_roles = tuple((role for role, object_id in self.role_object_ids.items() if str(object_id) in held_ids))
            return derive_benevolent_role_order(task, predicate_satisfied=statuses, held_roles=held_roles)
        except (KeyError, TypeError, ValueError):
            return ()

    def _active_public_event_binding_roles(self) -> Tuple[str, ...]:
        state = self._current_public_binding_state()
        return () if state is None else state.active_public_event_roles

    def _scene_observation_binding_roles(self) -> Tuple[str, ...]:
        state = self._current_public_binding_state()
        return () if state is None else state.scene_observation_roles

    def sync_public_binding_inputs(self, safe_human_messages: Sequence[Mapping[str, Any]], effective_facts: Sequence[Mapping[str, Any]], scene_observations: Sequence[Mapping[str, Any]]=()) -> None:
        self._public_binding_human_messages = tuple((copy.deepcopy(dict(item)) for item in safe_human_messages))
        self._public_binding_effective_facts = tuple((copy.deepcopy(dict(item)) for item in effective_facts))
        self._public_binding_scene_observations = tuple((copy.deepcopy(dict(item)) for item in scene_observations))

    def sync_private_processed_binding_roles(self, roles: Sequence[str]) -> None:
        self._private_processed_binding_roles = tuple(sorted({str(role) for role in roles if isinstance(role, str) and role}, key=str.encode))

    def last_private_binding_roles(self) -> Tuple[str, ...]:
        return tuple(sorted(self._current_tool_binding_roles, key=str.encode))

    def _current_public_binding_state(self):
        binding_spec = self._active_task_spec().get('task_binding_spec')
        if not isinstance(binding_spec, Mapping):
            return None
        spec = TaskBindingSpec.from_dict(binding_spec)
        return derive_public_binding_state(spec, safe_human_messages=self._public_binding_human_messages, effective_facts=self._public_binding_effective_facts, scene_observations=self._public_binding_scene_observations, task_instruction=str(self._active_task_spec().get('instruction') or ''))

    def _record_navigation_binding_outcome(self, resolved: ResolvedTarget) -> None:
        category = resolved.navigation_class
        if category == 'unseen_object_exploration':
            self.unseen_object_exploration_steps += 1
            return
        if category == 'seen_object_navigation':
            self.seen_object_navigation_steps += 1
            return
        if category != 'stale_object_navigation':
            return
        obj = self._raw_objects_by_id().get(resolved.object_id or '', {})
        if bool(obj.get('visible', False)):
            self.seen_object_navigation_steps += 1
        else:
            self.stale_object_navigation_steps += 1
            self.target_missing_at_last_seen_count += 1

    def observation_conditioned_visible_issues(self) -> List[dict]:
        return [dict(issue) for issue in self._observation_conditioned_visible_issues]

    def _materialize_observation_conditioned_scan_issues(self, visible_object_ids: Sequence[str]=()) -> List[dict]:
        task = self._active_task_spec()
        family = task.get('family_spec') if isinstance(task, Mapping) else None
        obs = family.get('observation_conditioned') if isinstance(family, Mapping) else None
        if not isinstance(obs, Mapping):
            return []
        payloads = obs.get('latent_payloads')
        if not isinstance(payloads, list):
            return []
        require_visual_evidence = bool(obs.get('require_visual_evidence', False))
        visible_id_set = {str(item) for item in visible_object_ids}
        current_room = self._current_room_id
        existing_ids = {str(issue.get('latent_id') or issue.get('issue_id') or '') for issue in self._observation_conditioned_visible_issues}
        objects = self._objects_by_id()
        revealed: List[dict] = []
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            payload_room = payload.get('room_id')
            if current_room is not None and payload_room != current_room:
                continue
            issue_id = str(payload.get('latent_id') or payload.get('issue_id') or '')
            if issue_id and issue_id in existing_ids:
                continue
            object_role = payload.get('object_role')
            object_id = self.role_object_ids.get(object_role) if isinstance(object_role, str) else None
            obj = objects.get(object_id, {}) if object_id else {}
            if require_visual_evidence and (not object_id or object_id not in visible_id_set):
                continue
            issue = copy_observation_conditioned_issue(payload)
            issue['discovered_step'] = int(self.high_level_step)
            if current_room is not None:
                issue['discovered_from_room_id'] = current_room
            if object_id:
                issue['object_id'] = object_id
                issue['object_type'] = obj.get('objectType') or obj.get('object_type')
                issue['visible'] = bool(obj.get('visible', False))
            revealed.append(issue)
            if issue_id:
                existing_ids.add(issue_id)
        if revealed:
            self._observation_conditioned_visible_issues.extend(revealed)
        event = {'high_level_step': int(self.high_level_step), 'room_id': current_room, 'revealed_issue_ids': [str(issue.get('latent_id') or issue.get('issue_id') or '') for issue in revealed], 'num_revealed': len(revealed)}
        self._observation_conditioned_scan_events.append(event)
        return revealed

    def _observation_conditioned_visual_only(self) -> bool:
        task = self._active_task_spec()
        family = task.get('family_spec') if isinstance(task, Mapping) else None
        obs = family.get('observation_conditioned') if isinstance(family, Mapping) else None
        return isinstance(obs, Mapping) and obs.get('public_discovery_mode') == 'visual_only'

    def _answer_pending_question(self, arguments: Mapping[str, Any]) -> Tuple[bool, dict, str]:
        question_id = arguments.get('question_id')
        value = arguments.get('value')
        if not isinstance(question_id, str) or not question_id:
            raise ProcTHORClosedLoopToolEnvError("answer requires non-empty 'question_id'.")
        pending = self._human_qa_state.setdefault('pending', {})
        answers = self._human_qa_state.setdefault('answers', {})
        self.answer_steps += 1
        if question_id not in pending:
            return (False, {'status': STATUS_FAILED, 'action': 'answer', 'question_id': question_id, 'message': 'No pending QA with that question_id.'}, "No pending question '{}'.".format(question_id))
        question = pending.pop(question_id)
        answers[question_id] = {'question': question, 'value': value, 'high_level_step': self.high_level_step}
        self._human_qa_state.setdefault('answered_steps', {})[question_id] = int(self.high_level_step)
        return (True, {'status': STATUS_SUCCESS, 'action': 'answer', 'question_id': question_id, 'question': question, 'value': value}, 'Answer recorded.')

    def _qa_overlay_rejected_stop_transition(self, scheduler_events: Sequence[Mapping[str, Any]]) -> dict:
        self.high_level_step += 1
        transition = self._non_actuating_tool_result('stop', success=False, feedback='A situated question is pending; answer it before requesting stop again.', raw_execution={'action': 'stop', 'action_executed': False, 'status': 'qa_pending'})
        transition['scheduler_intercept'] = [{'name': str(event.get('name') or 'qa_overlay_event'), 'phase': 'intercept', 'payload': dict(event.get('payload') or {})} for event in scheduler_events]
        return transition

    def _mask_verify_feedback_if_needed(self, transition: dict) -> None:
        task = self._active_task_spec()
        if task.get('category') != 'verify_after_action':
            return
        transition['env_feedback'] = 'Execution finished. Check the scene to verify the result.'

    def _prompt_safe_tool_history(self) -> List[dict]:
        safe = []
        for transition in self.tool_history:
            safe.append({'tool_name': transition.get('tool_name'), 'arguments': transition.get('arguments'), 'high_level_step': transition.get('high_level_step'), 'success': transition.get('success'), 'env_feedback': transition.get('env_feedback'), 'done': transition.get('done'), 'done_reason': transition.get('done_reason'), 'num_low_level_steps': transition.get('num_low_level_steps'), 'binding': transition.get('binding')})
        return safe

    def _standard_vlm_premature_stop_reason(self) -> Optional[str]:
        from scoper.benchmark.common.evaluation_lifecycle import enabled
        if enabled(self):
            return None
        if self.eval_mode != 'standard_vlm':
            return None
        required_tools = self._standard_vlm_required_tool_requests()
        if not required_tools:
            return None
        requested_tools = {str(item.get('tool_name') or '') for item in self.tool_history if isinstance(item, Mapping)}
        missing = [tool for tool in required_tools if tool not in requested_tools]
        if not missing:
            return None
        return 'standard_vlm stop requires prior requested action(s): {}'.format(', '.join(missing))

    def _standard_vlm_required_tool_requests(self) -> Tuple[str, ...]:
        task = self._active_task_spec()
        instruction = str(task.get('instruction') or '').lower()
        required: List[str] = []

        def add(tool_name: str) -> None:
            if tool_name not in required:
                required.append(tool_name)
        if re.search('\\b(put|place|move|set)\\b', instruction) and re.search('\\b(on|onto|in|into)\\b', instruction):
            add('pick_up')
            add('put_down')
        if re.search('\\bpick\\s+up\\b|\\bgrab\\b', instruction):
            add('pick_up')
        if re.search('\\bopen\\b', instruction):
            add('open')
        if re.search('\\bclose\\b', instruction):
            add('close')
        if re.search('\\b(turn|switch)\\s+on\\b|\\btoggle\\s+on\\b', instruction):
            add('toggle_on')
        if re.search('\\b(turn|switch)\\s+off\\b|\\btoggle\\s+off\\b', instruction):
            add('toggle_off')
        if re.search('\\b(cut|slice)\\b', instruction):
            add('slice')
        return tuple(required)

def get_tool_specs(model_visible_only: bool=True) -> List[dict]:
    specs = [_target_tool_spec('navigate_to', 'Navigate with real movement primitives to a spatial target.'), {'name': 'scan', 'description': 'Rotate in place to collect multiple RGB views. This tool accepts no arguments; always pass an empty object.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}, _target_tool_spec('look_for', 'Adjust camera orientation to bring a known object into view without moving. Use after an action may have displaced an object to an unexpected position.'), _target_tool_spec('pick_up', 'Pick up a visible and reachable target object from the current pose.'), _target_tool_spec('put_down', 'Put the held object into/on a visible and reachable receptacle target.'), _target_tool_spec('open', 'Open a visible and reachable openable target.'), _target_tool_spec('close', 'Close a visible and reachable openable target.'), _target_tool_spec('toggle_on', 'Turn on a visible and reachable toggleable target.'), _target_tool_spec('toggle_off', 'Turn off a visible and reachable toggleable target.'), _target_tool_spec('slice', 'Slice a visible and reachable sliceable target. Requires holding a knife.'), {'name': 'answer', 'description': 'Answer a pending human/evaluator QA request.', 'parameters': {'type': 'object', 'properties': {'question_id': {'type': 'string'}, 'value': {'type': 'string'}}, 'required': ['question_id', 'value']}}, {'name': 'stop', 'description': 'Stop the episode when the task is complete.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}]
    if not model_visible_only:
        specs.extend([{'name': 'get_context', 'description': 'Evaluator/debug helper that returns current prompt context.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}, {'name': 'check_goal', 'description': 'Evaluator/debug helper that returns current goal status.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}])
    return specs

def _coerce_object_binding_config(value: Optional[Any], *, eval_mode: str) -> ObjectBindingConfig:
    if isinstance(value, ObjectBindingConfig):
        return value
    defaults = ObjectBindingConfig(allow_role_handles=eval_mode != 'standard_vlm', allow_visible_aliases=True, allow_memory=True, allow_description_match=True, allow_raw_object_ids=False, allow_task_text_candidates=False, ambiguous_score_margin=0.05, benevolent_task_progress=False, vocabulary_version=None)
    if value is None:
        return defaults
    data = value if isinstance(value, Mapping) else getattr(value, '__dict__', {})
    if not isinstance(data, Mapping):
        return defaults
    merged = dict(defaults.__dict__)
    for key in merged:
        if key in data:
            merged[key] = data[key]
    return ObjectBindingConfig(allow_role_handles=bool(merged['allow_role_handles']), allow_visible_aliases=bool(merged['allow_visible_aliases']), allow_memory=bool(merged['allow_memory']), allow_description_match=bool(merged['allow_description_match']), allow_raw_object_ids=bool(merged['allow_raw_object_ids']), allow_task_text_candidates=bool(merged['allow_task_text_candidates']), ambiguous_score_margin=float(merged['ambiguous_score_margin']), benevolent_task_progress=bool(merged['benevolent_task_progress']), vocabulary_version=merged['vocabulary_version'])

def _target_tool_spec(name: str, description: str) -> dict:
    return {'name': name, 'description': description, 'parameters': {'type': 'object', 'properties': {'target': {'type': 'string', 'description': 'Role handle or object alias from get_context.'}}, 'required': ['target']}}

def copy_observation_conditioned_issue(payload: Mapping[str, Any]) -> dict:
    return copy.deepcopy(dict(payload))

def _representative_reachable_pose_for_room(room: Mapping[str, Any], reachable_positions: Sequence[Mapping[str, Any]]) -> Optional[dict]:
    room_positions = [position for position in reachable_positions if _point_in_floor_polygon(position, room.get('floorPolygon') or room.get('floor_polygon'))]
    if not room_positions:
        return None
    center_x = sum((float(position.get('x', 0.0)) for position in room_positions)) / len(room_positions)
    center_z = sum((float(position.get('z', 0.0)) for position in room_positions)) / len(room_positions)
    best = min(room_positions, key=lambda position: (float(position.get('x', 0.0)) - center_x) ** 2 + (float(position.get('z', 0.0)) - center_z) ** 2)
    return {'position': {'x': float(best.get('x', 0.0)), 'y': float(best.get('y', 0.0)), 'z': float(best.get('z', 0.0))}, 'rotation': {'y': 0.0}, 'horizon': 0.0}

def _point_in_floor_polygon(point: Mapping[str, Any], polygon: Any) -> bool:
    if 'x' not in point or 'z' not in point:
        return False
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    vertices = []
    for vertex in polygon:
        if not isinstance(vertex, Mapping) or 'x' not in vertex or 'z' not in vertex:
            return False
        vertices.append((float(vertex['x']), float(vertex['z'])))
    x = float(point['x'])
    z = float(point['z'])
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, zi = vertices[i]
        xj, zj = vertices[j]
        if _point_on_segment(x, z, xi, zi, xj, zj):
            return True
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi or 1e-12) + xi:
            inside = not inside
        j = i
    return inside

def _point_on_segment(x: float, z: float, x1: float, z1: float, x2: float, z2: float) -> bool:
    cross = (x - x1) * (z2 - z1) - (z - z1) * (x2 - x1)
    if abs(cross) > 1e-09:
        return False
    dot = (x - x1) * (x2 - x1) + (z - z1) * (z2 - z1)
    if dot < -1e-09:
        return False
    squared_len = (x2 - x1) ** 2 + (z2 - z1) ** 2
    return dot <= squared_len + 1e-09

def _agent_pose(metadata: Mapping[str, Any]) -> dict:
    agent = metadata.get('agent') or {}
    return {'position': dict(agent.get('position') or {}), 'rotation': dict(agent.get('rotation') or {}), 'cameraHorizon': agent.get('cameraHorizon', 0.0), 'isStanding': agent.get('isStanding')}

def _object_or_slice_child_matches_goal(object_id: str, goal_object_ids: set[str]) -> bool:
    if object_id in goal_object_ids:
        return True
    return any((object_id.startswith(goal_id + '|') for goal_id in goal_object_ids))

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != 'head_rgb'}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value

def _coerce_xyz(value: Optional[Mapping[str, Any]], *, default: Mapping[str, Any]) -> dict:
    source = value if isinstance(value, Mapping) else default
    return {'x': float(source.get('x', default.get('x', 0.0))), 'y': float(source.get('y', default.get('y', 0.0))), 'z': float(source.get('z', default.get('z', 0.0)))}

def _bbox_aware_visual_attach_offset(obj: Mapping[str, Any]) -> dict:
    size = None
    for bbox_key in ('objectOrientedBoundingBox', 'axisAlignedBoundingBox'):
        size = _bbox_size_from_metadata(obj.get(bbox_key))
        if size is not None:
            break
    if size is None:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}
    return {'x': 0.0, 'y': 0.0, 'z': 0.0}

def _bbox_size_from_metadata(bbox: Any) -> Optional[dict]:
    if not isinstance(bbox, Mapping):
        return None
    size = bbox.get('size')
    if _has_xyz(size):
        return {'x': size['x'], 'y': size['y'], 'z': size['z']}
    corners = bbox.get('cornerPoints')
    if not isinstance(corners, Sequence) or isinstance(corners, (str, bytes)):
        return None
    parsed = []
    for corner in corners:
        if not _has_xyz(corner):
            continue
        try:
            parsed.append((float(corner['x']), float(corner['y']), float(corner['z'])))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None
    xs, ys, zs = zip(*parsed)
    return {'x': max(xs) - min(xs), 'y': max(ys) - min(ys), 'z': max(zs) - min(zs)}

def _clamp_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))

def _has_xyz(value: Any) -> bool:
    return isinstance(value, Mapping) and {'x', 'y', 'z'}.issubset(value)

def _metadata_agent_yaw(metadata: Mapping[str, Any]) -> float:
    agent = metadata.get('agent') if isinstance(metadata, Mapping) else {}
    if not isinstance(agent, Mapping):
        return 0.0
    rotation = agent.get('rotation') or {}
    try:
        return float(rotation.get('y', 0.0)) % 360.0
    except (TypeError, ValueError):
        return 0.0

def _offset_from_agent_yaw(origin: Mapping[str, Any], offset: Mapping[str, Any], yaw_degrees: float) -> dict:
    yaw = math.radians(float(yaw_degrees))
    local_right = float(offset.get('x', 0.0))
    local_up = float(offset.get('y', 0.0))
    local_forward = float(offset.get('z', 0.0))
    return {'x': float(origin['x']) + math.cos(yaw) * local_right + math.sin(yaw) * local_forward, 'y': float(origin['y']) + local_up, 'z': float(origin['z']) - math.sin(yaw) * local_right + math.cos(yaw) * local_forward}

def _local_offset_from_agent_yaw(origin: Mapping[str, Any], point: Mapping[str, Any], yaw_degrees: float) -> dict:
    yaw = math.radians(float(yaw_degrees))
    dx = float(point['x']) - float(origin['x'])
    dz = float(point['z']) - float(origin['z'])
    return {'x': math.cos(yaw) * dx - math.sin(yaw) * dz, 'y': float(point['y']) - float(origin['y']), 'z': math.sin(yaw) * dx + math.cos(yaw) * dz}
__all__ = ['MODEL_VISIBLE_TOOLS', 'ProcTHORClosedLoopToolEnv', 'ProcTHORClosedLoopToolEnvError', 'get_tool_specs']
