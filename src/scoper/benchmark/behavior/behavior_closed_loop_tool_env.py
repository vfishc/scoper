from __future__ import annotations
import hashlib
import math
import os
import re
import copy
from collections import Counter
from collections.abc import Iterable
from typing import Any, List, Mapping, Optional, Sequence
import numpy as np
from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.execution_core.api import make_core
from scoper.benchmark.behavior.execution_core.constraints import pose_looking_at_object
from scoper.benchmark.behavior.execution_core.actions import COLDSOURCE_ABILITY_NAMES, CUT_TARGET_ABILITY_NAMES, FILLABLE_ABILITY_NAMES, FLUID_SOURCE_ABILITY_NAMES, HEATSOURCE_ABILITY_NAMES, OPEN_ABILITY_NAME, PICK_ABILITY_NAMES, PLACE_INSIDE_CONTAINER_ABILITY_NAMES, PLACE_ON_SURFACE_ABILITY_NAMES, POURABLE_ABILITY_NAMES, TOGGLE_ABILITY_NAME, WIPE_TARGET_ABILITY_NAMES
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose, normalize_angle
from scoper.benchmark.behavior.execution_core.navigation import NavigationConfig, NavigationTarget, plan_navigation, plan_navigation_to_pose, sample_interaction_candidates
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject, OccupancyConfig, cells_on_segment
from scoper.benchmark.behavior.execution_core.trace import ActionResult, CheckResult, to_jsonable
from scoper.benchmark.behavior.agent_belief import AgentBeliefTracker
from scoper.benchmark.behavior.agent_catalog_filter import AgentCatalogTracker, build_prompt_safe_catalog, build_prompt_target_handles, prompt_safe_tool_history
from scoper.benchmark.behavior.bddl_adapter import BDDLAdapter
from scoper.benchmark.behavior.behavior_executor import ACTION_ROUTING, BehaviorExecutor
from scoper.agents.behavior.streaming import StreamDecision, StreamFrameObservation, build_logical_timeline_from_poses, resample_timeline_by_fps, stream_frame_log_record
from scoper.agents.behavior.config import ObjectBindingConfig
from scoper.agents.behavior.object_resolver import ObjectBindingResult, ObjectReferenceResolver
from scoper.benchmark.behavior.goal_checker import BehaviorGoalChecker, _is_inside
from scoper.benchmark.common import ConfirmationIntent
from scoper.benchmark.behavior.scene import build_scene
from scoper.benchmark.behavior.scene.doors import is_door_like
from scoper.benchmark.behavior.scene.room_layout_maps import RoomLayoutMaps, load_behavior_room_layout_maps
from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec
from scoper.benchmark.behavior.runtime_timing import runtime_phase
SHARED_CORE_TOOLS = ('navigate_to', 'navigate_to_room', 'pick_up', 'put_down', 'drop', 'open', 'close', 'toggle_on', 'toggle_off', 'scan', 'look_for', 'answer', 'stop')
BEHAVIOR_EXTENDED_TOOLS = ('wipe', 'cut', 'soak_under', 'fill', 'pour', 'heat', 'cool')
DEFERRED_TOOLS = ()
MODEL_VISIBLE_TOOLS = SHARED_CORE_TOOLS + BEHAVIOR_EXTENDED_TOOLS
_TARGET_TOOLS = frozenset(ACTION_ROUTING.keys())
_SYMBOLIC_HORIZONTAL_FOV_DEGREES = 90.0
_SYMBOLIC_VERTICAL_FOV_DEGREES = 90.0
_SYMBOLIC_VISIBILITY_MAX_DISTANCE = 12.0
_RENDER_VISIBILITY_MIN_PIXELS = 8
_ACTION_VISIBILITY_MODES = frozenset({'disabled', 'rendered_required', 'rendered_or_symbolic', 'line_of_sight_required', 'symbolic_fov'})
_VISIBILITY_GATED_TOOLS = frozenset({'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'wipe', 'cut', 'soak_under', 'fill', 'pour', 'heat', 'cool'})
_VISIBILITY_EQUIVALENT_CENTER_TOLERANCE_M = 0.08
_LINE_OF_SIGHT_START_CLEARANCE_M = 0.2
_LINE_OF_SIGHT_MIN_TARGET_CLEARANCE_M = 0.28
_LINE_OF_SIGHT_MAX_TARGET_CLEARANCE_M = 0.65
_LINE_OF_SIGHT_CAMERA_HEIGHT_FALLBACK_M = 1.55
_LINE_OF_SIGHT_HEIGHT_CLEARANCE_M = 0.08
_LINE_OF_SIGHT_STRUCTURAL_BLOCKER_CATEGORIES = frozenset({'walls', 'wall', 'ceilings', 'ceiling', 'door', 'doors', 'window', 'windows', 'fixed_window'})
_LINE_OF_SIGHT_STRUCTURAL_BLOCKER_KEYWORDS = ('wall', 'ceiling', 'door', 'window')
_NAVIGATION_VISIBILITY_REJECT_LOG_LIMIT = 24
_NAVIGATION_SMALL_OBJECT_MAX_FOOTPRINT_M = 0.55
_NAVIGATION_SUPPORT_VERTICAL_TOLERANCE_M = 0.35
_NAVIGATION_SUPPORT_HORIZONTAL_TOLERANCE_M = 0.25
_NAVIGATION_GROUND_SUPPORT_MAX_HEIGHT_M = 0.12
_NAVIGATION_SUPPORT_CATEGORIES = frozenset({'countertop', 'countertops', 'breakfast_table', 'dining_table', 'coffee_table', 'table', 'desk', 'console_table', 'shelf', 'shelves', 'bookcase', 'cabinet', 'bottom_cabinet', 'top_cabinet', 'sink', 'sink_basin', 'furniture_sink', 'stove', 'microwave', 'fridge', 'refrigerator'})
_RGB_OBSERVATION_YAW_DELTAS = (0.0, math.pi * 0.25, -math.pi * 0.25, math.pi * 0.5, -math.pi * 0.5, math.pi * 0.75, -math.pi * 0.75, math.pi)
_RGB_OBSERVATION_PITCHES = (-0.25, 0.0)
_OBSERVATION_STRUCTURAL_CATEGORIES = frozenset({'floor', 'floors', 'ceiling', 'ceilings', 'wall', 'walls', 'door', 'sliding_door', 'fixed_window', 'window', 'window_blind', 'room_light', 'downlight', 'square_light'})
_TOOL_DESCRIPTIONS = {'navigate_to': 'Navigate the agent next to the target object.', 'navigate_to_room': 'Navigate the agent to a reachable scan pose inside the target room.', 'pick_up': 'Pick up the target object (must be adjacent).', 'put_down': 'Place the held object onto/into the target receptacle.', 'drop': 'Release the held object in place (frees the hand).', 'open': 'Open the target openable object.', 'close': 'Close the target openable object.', 'toggle_on': 'Toggle the target on.', 'toggle_off': 'Toggle the target off.', 'scan': 'Look around in place to observe nearby objects.', 'look_for': 'Turn the camera toward a target role and report whether it is locatable.', 'answer': 'Answer a pending human QA message.', 'stop': 'Declare the task complete and stop.', 'wipe': 'Wipe the target with the held cleaning tool.', 'cut': 'Cut the target with the held tool.', 'soak_under': 'Soak the held object under the target water source.', 'fill': 'Fill the target container from a source.', 'pour': "Pour the held container's contents into the target.", 'heat': 'Place the held food on the target heatsource, turn it on, and cook the food.', 'cool': 'Place the held food in the target cold source and cool the food.'}

class BehaviorClosedLoopToolEnvError(RuntimeError):
    pass

class BehaviorClosedLoopToolEnv:

    def __init__(self, adapter: Optional[BDDLAdapter]=None, max_high_level_steps: int=50, max_total_low_level_steps: int=10 ** 9, max_invalid_tool_calls: int=10, renderer: Optional[Any]=None, use_rendered_visibility: bool=True, scene_traversability: Optional[Any]=None, line_of_sight_traversability: Optional[Any]=None, interaction_faces_by_name: Optional[Mapping[str, Sequence[Mapping[str, Any]]]]=None, scene_navigation_objects: Optional[Sequence[AABBObject]]=None, occupancy_agent_radius: float | None=None, occupancy_safety_margin: float | None=None, scene_layout_mode: str='task_roles', stream_controller: Optional[Any]=None, metadata_mode: str='standard', metadata_config: Optional[Any]=None, eval_mode: str='agent', object_binding_config: Optional[ObjectBindingConfig]=None, binding_log_path: Optional[str]=None, action_visibility_mode: str='disabled', action_visibility_min_pixels: int=_RENDER_VISIBILITY_MIN_PIXELS, require_rendered_held_pose: bool=False, held_pose_center_tolerance_m: float=0.08, held_pose_min_pixels: int=12, load_doors: bool=False, allow_task_relevant_start_pose_repair: bool=True, allow_occupied_start_pose_repair: bool=True, allow_navigation_candidate_fallbacks: bool=True, defer_stream_start_until_admission: bool=False, capture_rendered_observation_evidence: bool=False, expose_role_bindings_to_policy: bool=True, allow_symbolic_public_observation_evidence: bool=True, rendered_visibility_modality: str='seg_instance'):
        self.adapter = adapter or BDDLAdapter()
        self.max_high_level_steps = int(max_high_level_steps)
        self.max_total_low_level_steps = int(max_total_low_level_steps)
        self.max_invalid_tool_calls = int(max_invalid_tool_calls)
        self.renderer = renderer
        self.use_rendered_visibility = bool(use_rendered_visibility)
        self.action_visibility_mode = str(action_visibility_mode or 'disabled').strip().lower()
        if self.action_visibility_mode not in _ACTION_VISIBILITY_MODES:
            self.action_visibility_mode = 'disabled'
        self.action_visibility_min_pixels = max(1, int(action_visibility_min_pixels))
        self.require_rendered_held_pose = bool(require_rendered_held_pose)
        self.held_pose_center_tolerance_m = max(0.0, float(held_pose_center_tolerance_m))
        self.held_pose_min_pixels = max(1, int(held_pose_min_pixels))
        self.scene_traversability = scene_traversability
        self.line_of_sight_traversability = line_of_sight_traversability
        self.occupancy_agent_radius = occupancy_agent_radius
        self.occupancy_safety_margin = occupancy_safety_margin
        self.interaction_faces_by_name = {str(name): [dict(face) for face in faces] for name, faces in dict(interaction_faces_by_name or {}).items()}
        self.scene_navigation_objects = list(scene_navigation_objects or [])
        self.scene_layout_mode = str(scene_layout_mode or 'task_roles')
        self.load_doors = bool(load_doors)
        self.allow_task_relevant_start_pose_repair = bool(allow_task_relevant_start_pose_repair)
        self.allow_occupied_start_pose_repair = bool(allow_occupied_start_pose_repair)
        self.allow_navigation_candidate_fallbacks = bool(allow_navigation_candidate_fallbacks)
        self.defer_stream_start_until_admission = bool(defer_stream_start_until_admission)
        self.capture_rendered_observation_evidence = bool(capture_rendered_observation_evidence)
        self._rendered_observation_evidence_required = bool(capture_rendered_observation_evidence)
        self.expose_role_bindings_to_policy = bool(expose_role_bindings_to_policy)
        self.allow_symbolic_public_observation_evidence = bool(allow_symbolic_public_observation_evidence)
        self.rendered_visibility_modality = str(rendered_visibility_modality or 'seg_instance')
        if self.rendered_visibility_modality not in {'seg_instance', 'seg_instance_id'}:
            raise ValueError('rendered_visibility_modality must be seg_instance or seg_instance_id')
        self._stream_runtime_started = False
        self.stream_controller = stream_controller
        self.metadata_mode = str(metadata_mode or 'standard').strip().lower()
        if self.metadata_mode not in {'standard', 'privileged'}:
            self.metadata_mode = 'standard'
        self.metadata_config = metadata_config
        self.eval_mode = str(eval_mode or 'agent').strip().lower()
        self.object_resolver = ObjectReferenceResolver(object_binding_config or ObjectBindingConfig(), log_path=binding_log_path)
        self.current_task_spec: Optional[Mapping[str, Any]] = None
        self.core = None
        self.executor: Optional[BehaviorExecutor] = None
        self.goal_checker: Optional[BehaviorGoalChecker] = None
        self.role_map: dict[str, str] = {}
        self.scene_result = None
        self.high_level_step = 0
        self.task_action_step = 0
        self.total_frames = 0
        self.total_low_level_steps = 0
        self.scan_steps = 0
        self.look_for_steps = 0
        self.answer_steps = 0
        self.invalid_tool_calls = 0
        self.model_stopped = False
        from scoper.benchmark.common.evaluation_lifecycle import reset as reset_evaluation_lifecycle
        reset_evaluation_lifecycle(self)
        self.tool_history: List[dict] = []
        self.human_messages: List[dict] = []
        self.human_answers: List[dict] = []
        self._human_qa_state: dict[str, Any] = self._empty_human_qa_state()
        self._human_event_templates: List[dict] = []
        self._activated_human_event_ids: set[str] = set()
        self._answered_human_question_ids: set[str] = set()
        self._last_visibility_mode = 'symbolic_fov'
        self._last_render_visibility_status = 'renderer_absent'
        self._last_render_visibility_error: Optional[str] = None
        self._belief_tracker = AgentBeliefTracker()
        self.seen_object_navigation_steps = 0
        self.stale_object_navigation_steps = 0
        self.unseen_object_exploration_steps = 0
        self.target_missing_at_last_seen_count = 0
        self._catalog_tracker = AgentCatalogTracker()
        self.room_catalog: list[dict[str, Any]] = []
        self.current_room: Optional[str] = None
        self.visited_rooms: set[str] = set()
        self.room_visit_counts: dict[str, int] = {}
        self.room_scan_count = 0
        self._room_layout_maps: Optional[RoomLayoutMaps] = None
        self._room_layout_error: Optional[str] = None
        self._discovered_issue_ids: set[str] = set()
        self._visible_issues: list[dict[str, Any]] = []
        self._stream_frame_index = 0
        self._stream_logical_time_s = 0.0
        self._stream_id = 'stream_uninitialized'
        self._last_resolved_targets: dict[str, dict[str, Any]] = {}
        self._local_interaction_context_active = False
        self._last_confirmation_bound_target: dict[str, Any] | None = None
        self._processed_binding_roles: set[str] = set()
        self._public_binding_human_messages: tuple[Mapping[str, Any], ...] | None = None
        self._public_binding_effective_facts: tuple[Mapping[str, Any], ...] = ()
        self._public_binding_scene_observations: tuple[Mapping[str, Any], ...] = ()
        self._current_tool_binding_roles: set[str] = set()
        self._private_processed_binding_roles: tuple[str, ...] = ()
        self._current_tool_binding_records: list[dict[str, Any]] = []
        self._staged_object_ledger: dict[str, dict[str, Any]] = {}
        self._inside_parent_ledger: dict[str, dict[str, Any]] = {}
        self._locked_recovery_admission_bypass: dict[str, Any] | None = None
        self._suppressed_completed_tool_call: dict[str, Any] | None = None
        self._harness_directed_tool_call: dict[str, Any] | None = None
        self._public_precondition_rejection: dict[str, Any] | None = None
        self._bypass_interaction_readiness_active = False
        self._reset_observation_pose: dict[str, Any] | None = None
        self._last_renderer_pose_readback: dict[str, Any] = {}
        self._canonical_navigation_source_map_sha256: str | None = None
        self._last_observation_privileged_evidence: dict[str, Any] = {}
        self._pending_visual_sensor_settle = False
        self._last_visual_sensor_settle: dict[str, Any] = {'required': False, 'status': 'not_requested'}

    def defer_rendered_observation_evidence_until_admission(self) -> None:
        self._rendered_observation_evidence_required = True
        self.capture_rendered_observation_evidence = False

    def activate_rendered_observation_evidence(self) -> None:
        if self._rendered_observation_evidence_required:
            self.capture_rendered_observation_evidence = True

    def reset_to_task(self, task_spec: Mapping[str, Any], rng_seed: int=0) -> dict:
        validate_behavior_task_spec(task_spec)
        from scoper.benchmark.common.evaluation_navigation import reset as reset_navigation
        reset_navigation(self)
        self._last_environment_seed = int(rng_seed)
        self._stream_runtime_started = False
        self._pending_visual_sensor_settle = False
        self._last_visual_sensor_settle = {'required': False, 'status': 'not_requested'}
        scene = build_scene(task_spec, mode='symbolic', rng_seed=rng_seed, scene_layout_mode=self.scene_layout_mode, load_doors=self.load_doors)
        self.current_task_spec = task_spec
        self.scene_result = scene
        self.role_map = dict(scene.role_map)
        objects = self._objects_with_navigation_overrides(scene.aabb_objects)
        source = task_spec.get('source') or {}
        self.room_catalog = self._build_room_catalog(task_spec)
        occupancy_bounds = source.get('occupancy_bounds') or (-0.5, -0.5, 10.5, 6.5)
        occupancy_kwargs = {}
        if self.scene_traversability is not None:
            try:
                occupancy_kwargs.update(self.scene_traversability.occupancy_kwargs())
            except Exception:
                occupancy_kwargs = {}
        if self.line_of_sight_traversability is not None:
            occupancy_kwargs.update(self._line_of_sight_traversability_occupancy_kwargs(self.line_of_sight_traversability))
        occupancy_bounds = self._clip_occupancy_bounds_to_traversability(occupancy_bounds, occupancy_kwargs)
        occupancy_resolution = float(occupancy_kwargs.get('traversability_map_resolution') or 0.1)
        self.core = make_core(objects=objects, start_pose=scene.ego_state.pose, occupancy_config=OccupancyConfig(resolution=occupancy_resolution, agent_radius=0.2 if self.occupancy_agent_radius is None else float(self.occupancy_agent_radius), safety_margin=0.05 if self.occupancy_safety_margin is None else float(self.occupancy_safety_margin), bounds=tuple((float(v) for v in occupancy_bounds)), **occupancy_kwargs), navigation_config=NavigationConfig(frame_spacing=0.2, candidate_count=192))
        self._last_renderer_pose_readback = self._sync_core_from_renderer_runtime()
        self.core.build_occupancy('reset_occupancy')
        source_map = getattr(self.scene_traversability, 'floor_map', None)
        self._canonical_navigation_source_map_sha256 = None if source_map is None else hashlib.sha256(np.ascontiguousarray(np.asarray(source_map)).tobytes()).hexdigest()
        if self.allow_task_relevant_start_pose_repair:
            self._repair_start_pose_to_task_relevant_room()
        else:
            self.core.occupancy.metadata['start_pose_room_repair'] = {'attempted': False, 'success': False, 'reason': 'disabled_by_execution_profile'}
        if self.allow_occupied_start_pose_repair:
            self._repair_start_pose_after_occupancy()
        else:
            start_cell = self.core.occupancy.world_to_cell(self.core.ego_state.pose.xy)
            self.core.occupancy.metadata['start_pose_repair'] = {'attempted': False, 'success': False, 'reason': 'disabled_by_execution_profile', 'start_pose': self.core.ego_state.pose.as_dict(), 'start_cell': [int(start_cell[0]), int(start_cell[1])], 'start_free': bool(self.core.occupancy.is_cell_free(start_cell))}
        if self.renderer is not None and hasattr(self.renderer, 'sync_initial_from_core'):
            before_mutation_count = len(getattr(self.renderer, 'visual_mutations', []) or [])
            try:
                self.renderer.sync_initial_from_core(self.core)
            except Exception:
                pass
            initial_mutations = list(getattr(self.renderer, 'visual_mutations', []) or [])[before_mutation_count:]
            self._mark_visual_sensor_settle(initial_mutations, reason='initial_visual_state')
        self._apply_renderer_openability_overrides()
        self.executor = BehaviorExecutor(self.core)
        self.goal_checker = BehaviorGoalChecker(self.adapter, self.core, self.role_map)
        self.high_level_step = 0
        self.task_action_step = 0
        self.total_frames = 0
        self.total_low_level_steps = 0
        self.scan_steps = 0
        self.look_for_steps = 0
        self.answer_steps = 0
        self.invalid_tool_calls = 0
        self.model_stopped = False
        from scoper.benchmark.common.evaluation_lifecycle import reset as reset_evaluation_lifecycle
        reset_evaluation_lifecycle(self)
        self.tool_history = []
        self.human_messages = []
        self.human_answers = []
        self._human_qa_state = self._empty_human_qa_state()
        self._human_event_templates = self._human_events_from_task(task_spec)
        self._activated_human_event_ids = set()
        self._answered_human_question_ids = set()
        self._last_visibility_mode = 'symbolic_fov'
        self._last_render_visibility_status = 'renderer_absent' if self.renderer is None else 'not_queried'
        self._last_render_visibility_error = None
        self._belief_tracker.reset(self.core.objects)
        self.seen_object_navigation_steps = 0
        self.stale_object_navigation_steps = 0
        self.unseen_object_exploration_steps = 0
        self.target_missing_at_last_seen_count = 0
        self._catalog_tracker.reset()
        from scoper.benchmark.common.binding_observation_ledger import enable_binding_delivery
        enable_binding_delivery(self, {})
        self.object_resolver.reset()
        self.current_room = self._current_room_for_pose(self.core.ego_state.pose)
        self.visited_rooms = {self.current_room} if self.current_room else set()
        self.room_visit_counts = {self.current_room: 1} if self.current_room else {}
        self.room_scan_count = 0
        self._room_layout_maps = None
        self._room_layout_error = None
        self._discovered_issue_ids = set()
        self._visible_issues = []
        self._stream_frame_index = 0
        self._stream_logical_time_s = 0.0
        self._stream_id = '{}_seed_{}'.format(str(task_spec.get('id') or 'task'), int(rng_seed))
        self._last_resolved_targets = {}
        self._local_interaction_context_active = False
        self._last_confirmation_bound_target = None
        self._processed_binding_roles = set()
        self._public_binding_human_messages = None
        self._public_binding_effective_facts = ()
        self._public_binding_scene_observations = ()
        self._current_tool_binding_roles = set()
        self._private_processed_binding_roles = ()
        self._current_tool_binding_records = []
        self._staged_object_ledger = {}
        self._inside_parent_ledger = {}
        self._locked_recovery_admission_bypass = None
        self._suppressed_completed_tool_call = None
        self._harness_directed_tool_call = None
        self._public_precondition_rejection = None
        self._bypass_interaction_readiness_active = False
        self._reset_observation_pose = None
        self._last_observation_privileged_evidence = {}
        self._prepare_initial_observation_pose()
        if self.stream_controller is not None and (not self.defer_stream_start_until_admission):
            self.activate_stream_runtime(emit_reset_keyframe=True)
        self.object_resolver.observe(self)
        return self._observation()

    def _apply_renderer_openability_overrides(self) -> None:
        sync = getattr(self.renderer, '_visual_sync', None) if self.renderer is not None else None
        resolver = getattr(sync, '_resolve_og_object_by_name', None)
        if not callable(resolver) or self.core is None:
            return
        try:
            from omnigibson import object_states
        except Exception:
            return
        for name, obj in (getattr(self.core, 'objects', {}) or {}).items():
            abilities = set((str(v) for v in getattr(obj, 'abilities', ()) or ()))
            if 'openable' not in abilities:
                continue
            og_obj = resolver(str(name))
            if og_obj is None or object_states.Open not in getattr(og_obj, 'states', {}):
                setattr(obj, 'renderer_openable', False)
                self._demote_open_storage_goal(name)

    def _demote_open_storage_goal(self, object_name: str) -> None:
        roles = [role for role, name in (self.role_map or {}).items() if name == object_name]
        if not roles or not isinstance(self.current_task_spec, Mapping):
            return
        role = roles[0]
        goal = self.current_task_spec.get('goal')
        if isinstance(goal, dict):
            for pred in goal.get('predicates') or []:
                if not isinstance(pred, dict):
                    continue
                args = pred.get('args') if isinstance(pred.get('args'), dict) else {}
                receptacle = pred.get('receptacle') or pred.get('receptacle_role') or args.get('receptacle')
                if (pred.get('object_role') or args.get('object')) and (role in (pred.get('receptacle_roles') or []) or receptacle == role):
                    pred['type'] = 'object_on_receptacle'
                    pred.pop('receptacle_roles', None)
                    pred['receptacle_role'] = role
                    args['receptacle'] = role
                    pred['args'] = args

    def activate_stream_runtime(self, *, emit_reset_keyframe: bool=False) -> None:
        if self.stream_controller is None or self.current_task_spec is None or self._stream_runtime_started:
            return
        self._stream_runtime_started = True
        task_spec = self.current_task_spec
        rng_seed = int(getattr(self, '_last_environment_seed', 0))
        try:
            prompt_context = self.get_prompt_context()
            self.stream_controller.on_stream_start({'stream_id': self._stream_id, 'task_id': task_spec.get('id'), 'rng_seed': rng_seed, 'task_spec': dict(task_spec), 'prompt_context': prompt_context, 'tool_specs': prompt_context.get('tool_specs') or [], 'tool_contract': prompt_context.get('tool_contract') or {}})
        except Exception:
            self._stream_runtime_started = False
            raise
        if emit_reset_keyframe:
            try:
                self.emit_stream_event_keyframe(source='reset', keyframe_reason='reset', active_tool_name='reset', active_tool_args={'rng_seed': rng_seed}, prompt_metadata={'reset': {'rng_seed': rng_seed}}, privileged_metadata={'scene_model': (task_spec.get('source') or {}).get('scene_model')})
            except Exception:
                pass

    def _line_of_sight_traversability_occupancy_kwargs(self, traversability: Any) -> dict[str, Any]:
        floor_map = getattr(traversability, 'floor_map', None)
        if floor_map is None:
            return {}
        observation = {'line_of_sight_traversability_map': floor_map, 'line_of_sight_traversability_map_resolution': getattr(traversability, 'map_resolution', None), 'line_of_sight_traversability_map_size': getattr(traversability, 'map_size', None), 'line_of_sight_traversability_source': getattr(traversability, 'source', None) or 'line_of_sight_traversability'}
        return observation

    def check_goal(self, model_stopped: bool=False) -> dict:
        from scoper.benchmark.common.evaluation_lifecycle import frozen_goal
        snapshot = frozen_goal(self) if model_stopped else None
        if snapshot is not None:
            return snapshot
        if self.goal_checker is None:
            raise BehaviorClosedLoopToolEnvError('Call reset_to_task before check_goal.')
        with runtime_phase('goal_check'):
            return self.goal_checker.check(self.current_task_spec, model_stopped=model_stopped)

    def call_tool(self, tool_name: str, arguments: Optional[Mapping[str, Any]]=None, reasoning: str='') -> dict:
        if self.executor is None or self.current_task_spec is None:
            raise BehaviorClosedLoopToolEnvError('Call reset_to_task before call_tool.')
        tool_name = str(tool_name)
        arguments = dict(arguments or {})
        from scoper.benchmark.common.instruction_contract import room_navigation_arguments
        room_navigation_args = room_navigation_arguments(tool_name, arguments, self._public_room_catalog()) if tool_name == 'navigate_to' else None
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
        locked_bypass = self._locked_recovery_admission_bypass
        bypass_interaction_readiness = bool(isinstance(locked_bypass, Mapping) and locked_bypass.get('tool_name') == tool_name and (locked_bypass.get('arguments') == arguments))
        if bypass_interaction_readiness:
            self._locked_recovery_admission_bypass = None
        self._bypass_interaction_readiness_active = bypass_interaction_readiness
        self.high_level_step += 1
        if tool_name != 'answer':
            self.task_action_step += 1
        self._current_tool_binding_roles = set()
        self._current_tool_binding_records = []
        self._last_confirmation_bound_target = None
        if tool_name in {'navigate_to', 'navigate_to_room'}:
            self._local_interaction_context_active = False
        success = False
        frames = 0
        feedback = ''
        error_type: Optional[str] = None
        error_message: Optional[str] = None
        state_delta = None
        raw_execution: Optional[dict] = None
        low_level_actions: List[dict] = []
        stream_interrupt_record: Optional[dict] = None
        stream_frame_index_before_tool = self._stream_frame_index
        if suppress_execution:
            success = True
            feedback = 'This exact action is already completed for planning; continue with the next task step.'
            raw_execution = {'status': 'already_completed', 'action': tool_name, 'action_executed': False}
        elif reject_for_public_precondition:
            feedback = str(rejected_call.get('message') or 'Public precondition failed.')
            error_type = 'public_precondition_failed'
            error_message = feedback
            raw_execution = {'status': 'rejected', 'action': tool_name, 'action_executed': False, 'error_type': error_type}
            self._current_tool_binding_records.append({'status': 'rejected', 'reason': 'public_precondition_failed', 'source': 'public_precondition', 'match_kind': None, 'canonical_public_ref': None, 'navigation_class': None})
        elif tool_name not in MODEL_VISIBLE_TOOLS:
            error_type = 'unknown_tool'
            error_message = "Unknown tool '{}'. Valid: {}.".format(tool_name, ', '.join(MODEL_VISIBLE_TOOLS))
            feedback = error_message
        elif tool_name == 'stop':
            self.model_stopped = True
            success = True
            feedback = 'Agent stopped.'
            raw_execution = {'status': 'success', 'action': 'stop'}
        elif tool_name == 'scan':
            if self._streaming_enabled_for_tool('scan'):
                raw_execution = self._scan_in_place_streaming(arguments)
            else:
                raw_execution = self._scan_in_place(arguments)
            stream_interrupt_record = raw_execution.pop('_stream_interrupt_record', None)
            success = raw_execution.get('status') == 'success'
            frames = int(raw_execution.get('frame_count') or 0)
            error_type = raw_execution.get('error_type')
            error_message = raw_execution.get('error_message')
            state_delta = raw_execution.get('state_delta')
            low_level_actions = list(raw_execution.get('low_level_actions') or [])
            self.scan_steps += 1
            feedback = 'Scan completed.' if success else raw_execution.get('message', 'Scan failed.')
            from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding, scan_public_feedback
            if task_uses_evidence_binding(self.current_task_spec):
                completed = int(raw_execution.get('num_views_completed') or 0)
                requested = int(raw_execution.get('num_views') or arguments.get('num_views') or 8)
                feedback = scan_public_feedback(completed, requested)
                if completed and (not success):
                    raw_execution['partial_observation'] = True
                    raw_execution['status'] = 'success'
                    success = True
        elif tool_name == 'navigate_to_room' or room_navigation_args is not None:
            room_arguments = room_navigation_args if room_navigation_args is not None else arguments
            if self._streaming_enabled_for_tool('navigate_to_room'):
                raw_execution = self._navigate_to_room_streaming(room_arguments)
            else:
                raw_execution = self._navigate_to_room(room_arguments)
            stream_interrupt_record = raw_execution.pop('_stream_interrupt_record', None)
            success = raw_execution.get('status') == 'success'
            frames = int(raw_execution.get('frame_count') or 0)
            error_type = raw_execution.get('error_type')
            error_message = raw_execution.get('error_message')
            state_delta = raw_execution.get('state_delta')
            low_level_actions = list(raw_execution.get('low_level_actions') or [])
            feedback = 'Room navigation completed.' if success else raw_execution.get('message', 'Room navigation failed.')
        elif tool_name == 'answer':
            self.answer_steps += 1
            success, feedback, error_type, error_message, state_delta = self._answer_human_question(arguments)
        else:
            with runtime_phase('binding', tool_name=tool_name):
                target_binding_count = len(self._current_tool_binding_records)
                target = self._resolve_target(arguments, tool_name=tool_name, argument_name='target')
                if tool_name == 'fill':
                    self._ensure_fill_binding_record(argument_name='target', public_ref=arguments.get('target', arguments.get('object_ref')), resolved=target, previous_count=target_binding_count)
                kwargs, arg_error = self._extra_kwargs(tool_name, arguments)
            binding_rejection = next((record for record in reversed(self._current_tool_binding_records) if record.get('status') != 'success'), None)
            if target is not None and tool_name in _TARGET_TOOLS:
                self._last_confirmation_bound_target = self._confirmation_target_for_object(target)
            self._bypass_interaction_readiness_active = False
            if binding_rejection is not None:
                error_type = 'binding_rejected'
                error_message = str(binding_rejection.get('reason') or 'public_binding_rejected')
                feedback = '{} rejected by public Binding: {}.'.format(tool_name, error_message)
                raw_execution = {'status': 'rejected', 'action': tool_name, 'tool_name': tool_name, 'action_executed': False, 'error_type': error_type, 'error_message': error_message}
            elif arg_error is not None:
                error_type = 'missing_argument'
                error_message = arg_error
                feedback = '{} failed: {}'.format(tool_name, arg_error)
                raw_execution = {'status': 'rejected', 'action': tool_name, 'tool_name': tool_name, 'action_executed': False, 'error_type': error_type, 'error_message': error_message}
            elif tool_name == 'navigate_to' and self._strict_navigation_visibility_enabled():
                navigation_belief_before = self._navigation_belief_before(target)
                parent_name = self._private_inside_parent_for_navigation(target)
                from scoper.benchmark.common import evaluation_navigation
                previous = evaluation_navigation.consume_previous_position(self, target)
                if previous is not None:
                    raw_execution = self._navigate_to_last_seen_object(arguments, target, self._belief_tracker.ensure(target), previous)
                elif parent_name is not None:
                    raw_execution = self._navigate_to_visible_object(arguments, parent_name, navigation_belief_before)
                    raw_execution = self._retarget_inside_parent_navigation(raw_execution, target_name=target, parent_name=parent_name)
                else:
                    raw_execution = self._navigate_to_visible_object(arguments, target, navigation_belief_before)
                success = raw_execution.get('status') == 'success'
                frames = int(raw_execution.get('frame_count') or 0)
                error_type = raw_execution.get('error_type')
                error_message = raw_execution.get('error_message')
                state_delta = raw_execution.get('state_delta')
                low_level_actions = list(raw_execution.get('low_level_actions') or [])
                feedback = raw_execution.get('message', 'navigate_to succeeded.' if success else 'navigate_to failed: {}'.format(error_message))
            elif tool_name == 'navigate_to' and self._streaming_enabled_for_tool('navigate_to') and (not self._belief_navigation_enabled_for_target(arguments, target)):
                navigation_belief_before = self._navigation_belief_before(target)
                raw_execution = self._navigate_to_streaming(arguments, target, navigation_belief_before)
                stream_interrupt_record = raw_execution.pop('_stream_interrupt_record', None)
                success = raw_execution.get('status') == 'success'
                frames = int(raw_execution.get('frame_count') or 0)
                error_type = raw_execution.get('error_type')
                error_message = raw_execution.get('error_message')
                state_delta = raw_execution.get('state_delta')
                low_level_actions = list(raw_execution.get('low_level_actions') or [])
                feedback = raw_execution.get('message', 'navigate_to succeeded.' if success else 'navigate_to failed: {}'.format(error_message))
            elif tool_name == 'navigate_to' and self._belief_navigation_enabled_for_target(arguments, target):
                raw_execution = self._navigate_to_object_with_belief(arguments, target)
                success = raw_execution.get('status') == 'success'
                frames = int(raw_execution.get('frame_count') or 0)
                error_type = raw_execution.get('error_type')
                error_message = raw_execution.get('error_message')
                state_delta = raw_execution.get('state_delta')
                low_level_actions = list(raw_execution.get('low_level_actions') or [])
                feedback = raw_execution.get('message', 'navigate_to succeeded.' if success else 'navigate_to failed: {}'.format(error_message))
            else:
                navigation_belief_before = self._navigation_belief_before(target) if tool_name == 'navigate_to' and target else None
                if tool_name == 'navigate_to':
                    kwargs = dict(kwargs)
                    kwargs.setdefault('trace_id', 'step_{:04d}_navigate_to'.format(self.high_level_step))
                if tool_name == 'look_for' and self._streaming_enabled_for_tool('look_for'):
                    self.look_for_steps += 1
                    raw_execution = self._look_for_streaming(arguments, target)
                    stream_interrupt_record = raw_execution.pop('_stream_interrupt_record', None)
                    success = raw_execution.get('status') == 'success'
                    frames = int(raw_execution.get('frame_count') or 0)
                    error_type = raw_execution.get('error_type')
                    error_message = raw_execution.get('error_message')
                    state_delta = raw_execution.get('state_delta')
                    low_level_actions = list(raw_execution.get('low_level_actions') or [])
                else:
                    staged_admission_failure = None
                    if tool_name == 'soak_under' and getattr(getattr(self.core, 'ego_state', None), 'held_object_name', None) is None:
                        staged_target, staged_admission_failure = self._resolve_private_staged_soak_target(target)
                        if staged_target is not None:
                            kwargs = dict(kwargs)
                            kwargs['staged_target'] = staged_target
                    visibility_failure = None if staged_admission_failure is not None else self._pre_action_visibility_failure(tool_name, target, kwargs)
                    if staged_admission_failure is not None:
                        success = False
                        frames = 0
                        error_type = staged_admission_failure
                        error_message = staged_admission_failure
                        raw_execution = {'status': 'failed', 'action': tool_name, 'action_executed': False, 'error_type': error_type, 'error_message': error_message}
                    elif visibility_failure is not None:
                        success = False
                        frames = 0
                        error_type = str(visibility_failure.get('error_type') or 'target_not_visible')
                        error_message = str(visibility_failure.get('error_message') or '{} failed visibility gate.'.format(tool_name))
                        state_delta = {'visibility_gate': visibility_failure}
                        raw_execution = {'status': 'failed', 'action': tool_name, 'target': target, **visibility_failure}
                    else:
                        held_before_execution = getattr(getattr(self.core, 'ego_state', None), 'held_object_name', None)
                        result = self.executor.execute_action(tool_name, target=target, **kwargs)
                        success = result.success
                        frames = result.frames
                        error_type = result.error_type
                        error_message = result.error_message
                        state_delta = to_jsonable(result.action_result.state_delta or {})
                        visual_mutations = self._apply_visual_action_result(result.action_result)
                        if tool_name == 'soak_under' and kwargs.get('staged_target'):
                            state_delta = self._sanitize_staged_wash_state_delta(state_delta)
                            result.action_result.state_delta = copy.deepcopy(state_delta)
                        if visual_mutations and (not (tool_name == 'soak_under' and kwargs.get('staged_target'))):
                            state_delta = dict(state_delta or {})
                            state_delta['visual_mutations'] = to_jsonable(visual_mutations)
                        if success and tool_name == 'pick_up':
                            mutation_failure = self._pickup_visual_mutation_failure(visual_mutations)
                            if mutation_failure is not None:
                                state_delta = dict(state_delta or {})
                                state_delta['held_pose_visual_sync'] = to_jsonable(mutation_failure)
                                success = False
                                error_type = str(mutation_failure.get('error_type') or 'held_pose_visual_sync_failed')
                                error_message = str(mutation_failure.get('error_message') or 'Renderer did not synchronize the picked object to held pose.')
                                self._mark_pickup_failed_after_visual_postcondition(result.action_result, error_type=error_type, diagnostic=mutation_failure)
                                state_delta = to_jsonable(result.action_result.state_delta or state_delta)
                                if visual_mutations and 'visual_mutations' not in state_delta:
                                    state_delta['visual_mutations'] = to_jsonable(visual_mutations)
                        if success and tool_name == 'pick_up' and self.require_rendered_held_pose:
                            held_check = self._validate_rendered_held_pose_after_pickup(result.action_result)
                            state_delta = dict(state_delta or {})
                            state_delta['held_pose_visual_sync'] = to_jsonable(held_check)
                            if not bool(held_check.get('success')):
                                success = False
                                error_type = str(held_check.get('error_type') or 'held_pose_visual_sync_failed')
                                error_message = str(held_check.get('error_message') or 'Renderer did not synchronize the picked object to held pose.')
                                self._mark_pickup_failed_after_visual_postcondition(result.action_result, error_type=error_type, diagnostic=held_check)
                                state_delta = to_jsonable(result.action_result.state_delta or state_delta)
                                if visual_mutations and 'visual_mutations' not in state_delta:
                                    state_delta['visual_mutations'] = to_jsonable(visual_mutations)
                        if success and tool_name in {'put_down', 'heat', 'cool'} and held_before_execution:
                            self._record_private_inside_parent_placement(str(held_before_execution), target, tool_name=tool_name, state_delta=state_delta)
                        if success and tool_name == 'put_down' and held_before_execution:
                            self._record_staged_object(str(held_before_execution), target)
                        if success and tool_name == 'pick_up':
                            held_after_execution = getattr(getattr(self.core, 'ego_state', None), 'held_object_name', None)
                            if held_after_execution:
                                self._staged_object_ledger.pop(str(held_after_execution), None)
                                self.invalidate_private_parent_placement(str(held_after_execution))
                        if tool_name == 'navigate_to':
                            state_delta, raw_execution, low_level_actions = self._standard_navigate_to_raw_execution(arguments, target, result.action_result, success=success, error_type=error_type, error_message=error_message, state_delta=state_delta, navigation_belief_before=navigation_belief_before)
                        if tool_name == 'look_for':
                            self.look_for_steps += 1
                            raw_execution = self._look_for_raw_execution(arguments, target, result.action_result)
                            success = raw_execution.get('status') == 'success'
                            error_type = raw_execution.get('error_type')
                            error_message = raw_execution.get('error_message')
                            low_level_actions = self._low_level_actions_from_action_result(result.action_result, source='look_for', target_name=target, success_override=success)
                        elif raw_execution is None:
                            raw_execution = {'status': 'success' if success else 'failed', 'action': getattr(result.action_result, 'action', tool_name), 'tool_name': tool_name, 'target': target, 'error_type': error_type, 'error_message': error_message, 'state_delta': state_delta}
                feedback = '{} succeeded.'.format(tool_name) if success else '{} failed: {}'.format(tool_name, error_message)
        if stream_interrupt_record is None and self._streaming_enabled_for_tool(tool_name) and (getattr(self.stream_controller, 'mode', None) == 'passive_expert_capture') and (self._stream_frame_index == stream_frame_index_before_tool):
            stream_interrupt_record = self._process_streaming_tool_frames(tool_name=tool_name, arguments=arguments, low_level_actions=low_level_actions, raw_execution=raw_execution)
        if stream_interrupt_record is not None:
            success = False
            error_type = 'stream_interrupted'
            error_message = 'Tool execution interrupted by VLM stream decision.'
            feedback = '{} interrupted by VLM stream decision.'.format(tool_name)
            frames = int(stream_interrupt_record.get('low_level_index', frames - 1) or 0) + 1
            low_level_actions = low_level_actions[:frames] if low_level_actions else low_level_actions
            raw_execution = dict(raw_execution or {})
            raw_execution.update({'status': 'stream_interrupted', 'interrupted': True, 'interrupt_frame_index': stream_interrupt_record.get('frame_index'), 'interrupt_timestamp_s': stream_interrupt_record.get('timestamp_s'), 'interrupt_tool_call': stream_interrupt_record.get('tool_call'), 'stream_decision': stream_interrupt_record})
        if tool_name in {'look_for', 'navigate_to'} and isinstance(raw_execution, Mapping):
            raw_execution = {**dict(raw_execution), **self._target_interaction_readiness(raw_execution)}
            raw_execution.pop('_private_interaction_target_id', None)
        binding_failed = any((record.get('status') != 'success' for record in self._current_tool_binding_records))
        if not success and tool_name != 'answer' and (stream_interrupt_record is None) and (not binding_failed) and (not harness_directed):
            self.invalid_tool_calls += 1
        if success and tool_name in {'navigate_to', 'navigate_to_room'}:
            self._local_interaction_context_active = bool(tool_name == 'navigate_to' and isinstance(raw_execution, Mapping) and (raw_execution.get('interaction_ready') is True))
        if not ((tool_name == 'navigate_to_room' or room_navigation_args is not None) and success):
            self._refresh_current_room()
        self.total_frames += frames
        self.total_low_level_steps += frames
        goal_result = self.check_goal(model_stopped=self.model_stopped)
        done, done_reason = self._done_status(goal_result)
        transition = {'tool_name': tool_name, 'arguments': arguments, 'reasoning': reasoning, 'success': success, 'feedback': feedback, 'frames': frames, 'num_low_level_steps': frames, 'total_low_level_steps': self.total_low_level_steps, 'env_feedback': {'error_type': error_type, 'error_message': error_message}, 'high_level_step': self.high_level_step, 'task_action_step': self.task_action_step, 'agent_pose': self._agent_pose(), 'current_room': self.current_room, 'visited_rooms': sorted(self.visited_rooms), 'held_objects': self._held_objects(), 'held_object_handles': self._public_held_object_handles(), 'state_delta': state_delta, 'goal_result': goal_result, 'done': done, 'done_reason': done_reason, 'observation': self._observation(), 'raw_execution': raw_execution, 'binding': dict(self._current_tool_binding_records[-1]) if self._current_tool_binding_records else None, 'bindings': [dict(record) for record in self._current_tool_binding_records], 'low_level_actions': low_level_actions, 'episode_metrics': self._episode_metrics()}
        preparation_result = self._public_preparation_result(tool_name, raw_execution)
        if preparation_result is not None:
            if binding_failed:
                preparation_result = {**preparation_result, 'status': 'binding_failed', 'interaction_ready': False}
            transition['preparation_result'] = preparation_result
        if stream_interrupt_record is not None:
            transition.update({'status': 'stream_interrupted', 'interrupted': True, 'invalid_tool_call': False, 'interrupt_frame_index': stream_interrupt_record.get('frame_index'), 'interrupt_timestamp_s': stream_interrupt_record.get('timestamp_s'), 'interrupt_tool_call': stream_interrupt_record.get('tool_call'), 'pending_tool_call': stream_interrupt_record.get('tool_call'), 'stream_interrupt': stream_interrupt_record})
        keyframe_record = self._emit_tool_result_stream_keyframe(tool_name=tool_name, arguments=arguments, success=success, feedback=feedback, state_delta=state_delta, raw_execution=raw_execution, goal_result=goal_result, stream_interrupt_record=stream_interrupt_record, post_action_observation=transition['observation'] if tool_name == 'scan' else None)
        if keyframe_record is not None:
            transition['stream_keyframe'] = keyframe_record
            pending = keyframe_record.get('tool_call')
            if pending:
                transition['post_tool_interrupt'] = True
                transition['pending_tool_call'] = pending
        self.tool_history.append(transition)
        return transition

    def sync_public_binding_inputs(self, safe_human_messages: Sequence[Mapping[str, Any]], effective_facts: Sequence[Mapping[str, Any]], scene_observations: Sequence[Mapping[str, Any]]=()) -> None:
        self._public_binding_human_messages = tuple((copy.deepcopy(dict(item)) for item in safe_human_messages))
        self._public_binding_effective_facts = tuple((copy.deepcopy(dict(item)) for item in effective_facts))
        self._public_binding_scene_observations = tuple((copy.deepcopy(dict(item)) for item in scene_observations))

    def sync_private_processed_binding_roles(self, roles: Sequence[str]) -> None:
        self._private_processed_binding_roles = tuple(sorted({str(role) for role in roles if isinstance(role, str) and role}, key=str.encode))

    def last_private_binding_roles(self) -> tuple[str, ...]:
        return tuple(sorted(self._current_tool_binding_roles, key=str.encode))

    @staticmethod
    def _public_preparation_result(tool_name: str, raw_execution: Mapping[str, Any] | None) -> dict[str, Any] | None:
        raw = raw_execution if isinstance(raw_execution, Mapping) else {}
        if tool_name == 'look_for':
            if raw.get('status') == 'success' and raw.get('object_visible') is True:
                status = 'found'
            elif raw.get('orientation_success') is False:
                status = 'motion_failed'
            else:
                status = 'not_found'
            return {'tool_name': 'look_for', 'status': status, 'interaction_ready': bool(raw.get('interaction_ready')), 'frame_id_or_null': None}
        if tool_name in {'navigate_to', 'navigate_to_room'}:
            ready = tool_name == 'navigate_to' and raw.get('interaction_ready') is True
            status = 'navigation_ready' if raw.get('status') == 'success' and ready else 'navigation_not_ready' if raw.get('status') == 'success' else 'navigation_failed'
            return {'tool_name': 'navigate_to', 'status': status, 'interaction_ready': ready, 'frame_id_or_null': None}
        if tool_name == 'scan':
            return {'tool_name': 'scan', 'status': 'scan_completed' if raw.get('status') == 'success' else 'motion_failed', 'interaction_ready': False, 'frame_id_or_null': None}
        return None

    def _confirmation_target_for_object(self, object_name: str) -> dict[str, Any] | None:
        if self.core is None:
            return None
        obj = self.core.objects.get(str(object_name))
        center = getattr(obj, 'center', None) if obj is not None else None
        if center is None:
            return None
        try:
            return {'position': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])}}
        except (TypeError, ValueError, IndexError):
            return None

    def last_private_bound_target(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._last_confirmation_bound_target)

    def capture_confirmation(self, intent: ConfirmationIntent, private_bound_target: Mapping[str, Any] | None=None, *, video_fps: float | None=None) -> Mapping[str, Any]:
        if self.core is None:
            raise BehaviorClosedLoopToolEnvError('capture_confirmation requires an active episode')
        start = self.core.ego_state.pose
        start_pose = EgoPose(x=float(start.x), y=float(start.y), z=float(start.z), yaw=float(start.yaw), pitch=float(start.pitch))
        target_pose = EgoPose(x=start_pose.x, y=start_pose.y, z=start_pose.z, yaw=start_pose.yaw, pitch=0.0)
        if intent.view_kind == 'bound_target':
            position = private_bound_target.get('position') if isinstance(private_bound_target, Mapping) else None
            if isinstance(position, Mapping):
                target_pose = EgoPose(**self._pose_looking_at_center(start_pose, (float(position.get('x', start_pose.x)), float(position.get('y', start_pose.y)), float(position.get('z', start_pose.z)))))
        poses = self._interpolate_pose_turn(start_pose, target_pose)
        low_level_actions: list[dict[str, Any]] = []
        video_rate = float(video_fps) if video_fps else None
        for index, pose in enumerate(poses, start=1):
            self._set_agent_pose_from_mapping(self._pose_to_dict(pose))
            if video_rate is not None and video_rate > 0:
                low_level_actions.append({'action': 'Look', 'requested_primitive': 'confirmation_camera', 'executed_primitive': 'confirmation_camera', 'order': index, 'video_fps': video_rate, 'pose_after': self._pose_to_dict(pose)})
        if not poses:
            self._set_agent_pose_from_mapping(self._pose_to_dict(target_pose))
        rgb = self._render_head_rgb_at_pose(self._pose_to_dict(target_pose))
        if rgb is None:
            rgb = self._observation().get('head_rgb')
        restore_poses = self._interpolate_pose_turn(target_pose, start_pose)
        restore_start = len(low_level_actions)
        for index, pose in enumerate(restore_poses, start=1):
            self._set_agent_pose_from_mapping(self._pose_to_dict(pose))
            if video_rate is not None and video_rate > 0:
                low_level_actions.append({'action': 'Look', 'requested_primitive': 'confirmation_camera_restore', 'executed_primitive': 'confirmation_camera_restore', 'order': restore_start + index, 'video_fps': video_rate, 'pose_after': self._pose_to_dict(pose), 'evidence_eligible': False})
        self._set_agent_pose_from_mapping(self._pose_to_dict(start_pose))
        restored_rgb = self._render_head_rgb_at_pose(self._pose_to_dict(start_pose))
        return {'observation': {'head_rgb': rgb, 'observation_pose': self._pose_to_dict(target_pose)}, 'restored_observation': {'head_rgb': restored_rgb, 'observation_pose': self._pose_to_dict(start_pose)}, 'low_level_actions': low_level_actions, 'camera': {'schema_version': 'confirmation-camera.v1', 'status': 'success', 'intent': intent.public_record(), 'interpolation': 'video_substeps' if video_rate else 'logical_keyframe', 'pose': self._pose_to_dict(target_pose), 'video_fps': video_rate, 'restore_status': 'success', 'restored_pose': self._pose_to_dict(start_pose), 'restore_frame_evidence_eligible': False}}

    def _observation(self) -> dict:
        observation_pose = self._prepare_prompt_observation_pose()
        self._sync_renderer_for_observation_pose(observation_pose)
        visual_sensor_settle = self._settle_pending_visual_sensor(observation_pose)
        head_rgb = None
        visible_object_ids: list[str] | None = None
        segmentation_evidence: Mapping[str, Any] | None = None
        capture_rgb_and_segmentation = getattr(self.renderer, 'capture_observation_at_pose', None) if self.capture_rendered_observation_evidence and self.renderer is not None and (self.core is not None) else None
        if callable(capture_rgb_and_segmentation):
            core_object_names = [str(name) for name in self.core.objects]
            query_object_names: set[str] = set(core_object_names)
            for name in core_object_names:
                query_object_names.update(self._visibility_equivalent_names(name))
            try:
                captured = capture_rgb_and_segmentation(observation_pose, object_names=sorted(query_object_names), min_pixels=self.action_visibility_min_pixels, modality=self.rendered_visibility_modality)
            except Exception as exc:
                self._last_render_visibility_status = 'rendered_visibility_error'
                self._last_render_visibility_error = str(exc)
                raise BehaviorClosedLoopToolEnvError('Required RGB/segmentation observation capture failed.') from exc
            if not isinstance(captured, Mapping) or not isinstance(captured.get('visibility'), Mapping):
                raise BehaviorClosedLoopToolEnvError('Required RGB/segmentation observation capture returned an invalid payload.')
            head_rgb = captured.get('head_rgb')
            segmentation_evidence = dict(captured['visibility'])
            visible_raw = segmentation_evidence.get('visible_object_ids')
            if isinstance(visible_raw, str) or not isinstance(visible_raw, Iterable):
                raise BehaviorClosedLoopToolEnvError('Required RGB/segmentation observation capture has invalid visible_object_ids.')
            core_object_name_set = set(core_object_names)
            normalized_visible: set[str] = set()
            for visible_name in visible_raw:
                visible_text = str(visible_name)
                if visible_text in core_object_name_set:
                    normalized_visible.add(visible_text)
                    continue
                for core_name in core_object_names:
                    if self._target_visibility_matches(core_name, [visible_text]):
                        normalized_visible.add(core_name)
            visible_object_ids = sorted(normalized_visible)
            self._last_visibility_mode = str(segmentation_evidence.get('mode') or 'rendered_segmentation')
            self._last_render_visibility_status = 'rendered_segmentation_available'
            self._last_render_visibility_error = None
        elif self.renderer is not None and self.core is not None:
            try:
                head_rgb = self.renderer.render_at_pose(observation_pose)
            except Exception as exc:
                if self.capture_rendered_observation_evidence:
                    raise BehaviorClosedLoopToolEnvError('Required RGB observation capture failed.') from exc
                head_rgb = None
        if visible_object_ids is None:
            visible_object_ids = self._observation_visible_object_names_at_pose(observation_pose)
            evidence_query = getattr(self.renderer, 'last_visibility_evidence', None)
            if callable(evidence_query) and self._last_visibility_mode == 'rendered_segmentation' and (self._last_render_visibility_status == 'rendered_segmentation_available'):
                candidate = evidence_query()
                if isinstance(candidate, Mapping):
                    segmentation_evidence = candidate
        discoverability = self.visual_discoverability_report()
        self._last_observation_privileged_evidence = {'observation_pose': self._pose_to_dict(observation_pose), 'visible_object_ids': list(visible_object_ids), 'visibility_source': discoverability.get('visibility_source'), 'visual_sensor_settle': visual_sensor_settle, 'segmentation_status': 'available' if discoverability.get('rendered_segmentation_available') else 'unknown', 'symbolic_fov_fallback': bool(discoverability.get('symbolic_fov_fallback')), 'segmentation_evidence': to_jsonable(segmentation_evidence) if segmentation_evidence else None}
        self._observe_visible_objects(visible_object_ids, source='observation', pose=observation_pose)
        object_catalog = self._object_catalog(visible_object_ids=visible_object_ids)
        visible_object_id_set = set(visible_object_ids)
        visible_object_catalog = [item for item in object_catalog if str(item.get('name') or item.get('objectId')) in visible_object_id_set]
        observation = {'task_id': self.current_task_spec['id'], 'instruction': self.current_task_spec.get('instruction', ''), 'head_rgb': head_rgb, 'agent_pose': self._agent_pose(), 'observation_pose': self._pose_to_dict(observation_pose), 'current_room': self.current_room, 'visited_rooms': sorted(self.visited_rooms), 'room_catalog': self._public_room_catalog(), 'visible_issues': self._public_visible_issues(), 'discovered_issue_ids': self._public_discovered_issue_ids(), 'held_objects': self._held_objects(), 'held_object_handles': self._public_held_object_handles(), 'visible_object_ids': visible_object_ids, 'visible_object_catalog': visible_object_catalog, 'visibility_mode': self._last_visibility_mode, 'visual_discoverability': self.visual_discoverability_report(), 'agent_belief': self._public_agent_belief(), 'object_catalog': object_catalog}
        if self.expose_role_bindings_to_policy:
            observation['visible_object_roles'] = self._roles_for_object_names(visible_object_ids)
        return observation

    def get_prompt_context(self, observation: Optional[Mapping[str, Any]]=None) -> dict:
        context = dict(observation or self._observation())
        visible_object_ids = list(context.get('visible_object_ids') or [])
        safe_catalog = self._prompt_safe_object_catalog(visible_object_ids)
        context['object_catalog'] = safe_catalog
        context['visible_object_catalog'] = safe_catalog
        context['tool_specs'] = self.get_tool_specs(model_visible_only=True)
        context['tool_contract'] = {'allowed_model_tools': list(MODEL_VISIBLE_TOOLS)}
        context['tool_history'] = prompt_safe_tool_history(self.tool_history)
        context['tool_target_handles'] = build_prompt_target_handles(role_map=self.role_map if self.expose_role_bindings_to_policy else {}, visible_object_ids=visible_object_ids, tracker=self._catalog_tracker)
        context['room_target_handles'] = self._room_target_handles()
        context['visible_issues'] = self._public_visible_issues()
        context['discovered_issue_ids'] = self._public_discovered_issue_ids()
        self._sync_public_human_state()
        context['human_messages'] = list(self.human_messages)
        context['human_answers'] = list(self.human_answers)
        scheduler_notices = list(getattr(self, '_scheduler_notices', []) or [])
        if scheduler_notices:
            context['scheduler_notices'] = scheduler_notices
            context['disturbance_notice'] = '\n'.join((str(item) for item in scheduler_notices))
            self._scheduler_notices = []
        return context

    def _streaming_enabled_for_tool(self, tool_name: str) -> bool:
        controller = self.stream_controller
        if controller is None:
            return False
        stream_config = getattr(controller, 'config', None)
        interruptible = set(getattr(stream_config, 'interruptible_tools', ()) or ())
        return str(tool_name) in interruptible

    def _stream_tool_result_keyframe_enabled(self) -> bool:
        controller = self.stream_controller
        if controller is None:
            return False
        stream_config = getattr(controller, 'config', None)
        keyframes = getattr(stream_config, 'keyframes', None)
        return keyframes is None or bool(getattr(keyframes, 'tool_result', True))

    def _stream_render_candidate_fps(self) -> float:
        controller = self.stream_controller
        stream_config = getattr(controller, 'config', None)
        candidate_fps = getattr(controller, 'render_candidate_fps', None)
        if candidate_fps is None:
            candidate_fps = getattr(stream_config, 'input_fps', 5.0)
        return float(candidate_fps)

    def _should_render_stream_frame(self, timestamp_s: float, *, is_keyframe: bool) -> bool:
        predicate = getattr(self.stream_controller, 'should_render_frame', None)
        if not callable(predicate):
            return True
        return bool(predicate(timestamp_s, is_keyframe=is_keyframe))

    def _settle_skipped_stream_frame(self, pose: Mapping[str, Any]) -> None:
        render_count = int(getattr(self.stream_controller, 'skipped_frame_settle_renders', 0) or 0)
        settle = getattr(self.renderer, 'advance_temporal_history_at_pose', None)
        if render_count <= 0 or not callable(settle):
            return
        settle(pose, renders=render_count)

    def _run_streaming_pose_checkpoints(self, *, tool_name: str, arguments: Mapping[str, Any], trace_id: str, poses: Sequence[Any], prompt_metadata: Optional[Mapping[str, Any]]=None, privileged_metadata: Optional[Mapping[str, Any]]=None, scan_view_seconds: float | None=None) -> dict[str, Any]:
        controller = self.stream_controller
        pose_dicts = [self._pose_to_dict(pose) for pose in poses]
        if controller is None or not pose_dicts:
            return {'interrupted': False, 'executed_poses': []}
        stream_config = getattr(controller, 'config', None)
        timing = getattr(stream_config, 'logical_timing', None)
        timeline = build_logical_timeline_from_poses(pose_dicts, navigation_speed_mps=float(getattr(timing, 'navigation_speed_mps', 0.4)), turn_speed_radps=float(getattr(timing, 'turn_speed_radps', 0.8)), scan_view_seconds=scan_view_seconds)
        sampled = resample_timeline_by_fps(timeline, self._stream_render_candidate_fps())
        base_time_s = float(self._stream_logical_time_s)
        executed_poses: list[Any] = []
        defer_final_sample_call = bool(sampled) and self._stream_tool_result_keyframe_enabled()
        for local_index, item in enumerate(sampled):
            pose = dict(item.pose)
            self._set_agent_pose_from_mapping(pose)
            if hasattr(self.core, '_update_held_pose_anchor'):
                try:
                    self.core._update_held_pose_anchor(self.core.ego_state.pose)
                except Exception:
                    pass
            pose_for_trace = self.core.ego_state.pose.copy()
            executed_poses.append(pose_for_trace)
            if hasattr(self.core, '_emit_frame_pose'):
                self.core._emit_frame_pose(trace_id, len(executed_poses) - 1, pose_for_trace)
            timestamp_s = base_time_s + float(item.timestamp_s)
            if defer_final_sample_call and local_index == len(sampled) - 1:
                continue
            if not self._should_render_stream_frame(timestamp_s, is_keyframe=False):
                self._settle_skipped_stream_frame(pose)
                continue
            frame = self._build_stream_frame_observation(tool_name=tool_name, arguments=arguments, source='fps_sample', timestamp_s=timestamp_s, is_keyframe=False, keyframe_reason=None, prompt_metadata={'tool_progress': {'sample_index': local_index, 'sample_count': len(sampled), 'logical_elapsed_s': float(item.timestamp_s)}, **dict(prompt_metadata or {})}, privileged_metadata={'planned_pose_count': len(pose_dicts), 'sampled_pose': pose, **dict(privileged_metadata or {})})
            decision = controller.on_frame(frame)
            if decision.interrupted:
                self._stream_logical_time_s = max(self._stream_logical_time_s, timestamp_s)
                return {'interrupted': True, 'interrupt_record': self._stream_interrupt_record(frame, decision, local_index), 'executed_poses': executed_poses, 'last_timestamp_s': timestamp_s}
        if sampled:
            self._stream_logical_time_s = max(self._stream_logical_time_s, base_time_s + float(sampled[-1].timestamp_s))
        return {'interrupted': False, 'executed_poses': executed_poses, 'last_timestamp_s': self._stream_logical_time_s}

    def _run_streaming_scan_endpoints(self, *, arguments: Mapping[str, Any], trace_id: str, start_pose: Any, endpoints: Sequence[Any], prompt_metadata: Optional[Mapping[str, Any]]=None) -> dict[str, Any]:
        controller = self.stream_controller
        endpoint_dicts = [self._pose_to_dict(pose) for pose in endpoints]
        if controller is None or not endpoint_dicts:
            return {'interrupted': False, 'executed_poses': []}
        stream_config = getattr(controller, 'config', None)
        timing = getattr(stream_config, 'logical_timing', None)
        timeline = build_logical_timeline_from_poses([self._pose_to_dict(start_pose), *endpoint_dicts], navigation_speed_mps=float(getattr(timing, 'navigation_speed_mps', 0.4)), turn_speed_radps=float(getattr(timing, 'turn_speed_radps', 0.8)), scan_view_seconds=float(getattr(timing, 'scan_view_seconds', 0.0) or 0.0))[1:]
        base_time_s = float(self._stream_logical_time_s)
        executed_poses: list[Any] = []
        defer_final_call = self._stream_tool_result_keyframe_enabled()
        for view_index, (pose, item) in enumerate(zip(endpoint_dicts, timeline)):
            self._set_agent_pose_from_mapping(pose)
            if hasattr(self.core, '_update_held_pose_anchor'):
                try:
                    self.core._update_held_pose_anchor(self.core.ego_state.pose)
                except Exception:
                    pass
            pose_for_trace = self.core.ego_state.pose.copy()
            executed_poses.append(pose_for_trace)
            if hasattr(self.core, '_emit_frame_pose'):
                self.core._emit_frame_pose(trace_id, view_index, pose_for_trace)
            timestamp_s = base_time_s + float(item.timestamp_s)
            if defer_final_call and view_index == len(endpoint_dicts) - 1:
                continue
            if not self._should_render_stream_frame(timestamp_s, is_keyframe=False):
                continue
            frame = self._build_stream_frame_observation(tool_name='scan', arguments=arguments, source='fps_sample', timestamp_s=timestamp_s, is_keyframe=False, keyframe_reason=None, prompt_metadata={'tool_progress': {'sample_index': view_index, 'sample_count': len(endpoint_dicts), 'logical_elapsed_s': float(item.timestamp_s), 'view_index': view_index}, **dict(prompt_metadata or {})}, privileged_metadata={'planned_pose_count': len(endpoint_dicts), 'sampled_pose': pose, 'view_index': view_index})
            decision = controller.on_frame(frame)
            if decision.interrupted:
                self._stream_logical_time_s = max(self._stream_logical_time_s, timestamp_s)
                return {'interrupted': True, 'interrupt_record': self._stream_interrupt_record(frame, decision, view_index), 'executed_poses': executed_poses, 'last_timestamp_s': timestamp_s}
        if timeline:
            self._stream_logical_time_s = max(self._stream_logical_time_s, base_time_s + float(timeline[-1].timestamp_s))
        return {'interrupted': False, 'executed_poses': executed_poses, 'last_timestamp_s': self._stream_logical_time_s}

    def emit_stream_event_keyframe(self, *, source: str, keyframe_reason: str, active_tool_name: str, active_tool_args: Optional[Mapping[str, Any]]=None, prompt_metadata: Optional[Mapping[str, Any]]=None, privileged_metadata: Optional[Mapping[str, Any]]=None) -> Optional[dict]:
        controller = self.stream_controller
        if controller is None or self.current_task_spec is None or self.core is None:
            return None
        stream_config = getattr(controller, 'config', None)
        keyframes = getattr(stream_config, 'keyframes', None)
        if keyframes is not None:
            if source == 'reset' and (not bool(getattr(keyframes, 'reset', True))):
                return None
            if source == 'scheduler_event' and (not bool(getattr(keyframes, 'scheduler_event', True))):
                return None
            if source == 'episode_end' and (not bool(getattr(keyframes, 'episode_end', True))):
                return None
            if source == 'interrupt' and (not bool(getattr(keyframes, 'interrupt', True))):
                return None
        timestamp_s = float(self._stream_logical_time_s)
        if not self._should_render_stream_frame(timestamp_s, is_keyframe=True):
            return None
        frame = self._build_stream_frame_observation(tool_name=active_tool_name, arguments=dict(active_tool_args or {}), source=source, timestamp_s=timestamp_s, is_keyframe=True, keyframe_reason=keyframe_reason, prompt_metadata=prompt_metadata, privileged_metadata=privileged_metadata)
        decision = controller.on_frame(frame)
        record = stream_frame_log_record(frame, decision, vlm_called=decision.metadata.get('decision_source') != 'passive_capture', latency_seconds=0.0)
        if decision.interrupted:
            record['tool_call'] = {'tool_name': decision.tool_call.tool_name, 'arguments': dict(decision.tool_call.arguments), 'reasoning': decision.tool_call.reasoning}
        return record

    def _process_streaming_tool_frames(self, *, tool_name: str, arguments: Mapping[str, Any], low_level_actions: Sequence[Mapping[str, Any]], raw_execution: Optional[Mapping[str, Any]]) -> Optional[dict]:
        controller = self.stream_controller
        if controller is None or not low_level_actions:
            return None
        stream_config = getattr(controller, 'config', None)
        interruptible = set(getattr(stream_config, 'interruptible_tools', ()) or ())
        if tool_name not in interruptible:
            return None
        poses = [item.get('agent_pose') for item in low_level_actions if isinstance(item, Mapping) and item.get('agent_pose') is not None]
        if not poses:
            return None
        timing = getattr(stream_config, 'logical_timing', None)
        timeline = build_logical_timeline_from_poses(poses, navigation_speed_mps=float(getattr(timing, 'navigation_speed_mps', 0.4)), turn_speed_radps=float(getattr(timing, 'turn_speed_radps', 0.8)), scan_view_seconds=None)
        sampled = resample_timeline_by_fps(timeline, self._stream_render_candidate_fps())
        if not sampled:
            return None
        base_timestamp_s = float(self._stream_logical_time_s)
        final_timestamp_s = base_timestamp_s + float(sampled[-1].timestamp_s)
        final_pose = self._agent_pose()
        try:
            for local_index, item in enumerate(sampled[:-1]):
                pose = dict(item.pose)
                self._set_agent_pose_from_mapping(pose)
                timestamp_s = base_timestamp_s + float(item.timestamp_s)
                if not self._should_render_stream_frame(timestamp_s, is_keyframe=False):
                    self._settle_skipped_stream_frame(pose)
                    continue
                frame = self._build_stream_frame_observation(tool_name=tool_name, arguments=arguments, source='fps_sample', timestamp_s=timestamp_s, is_keyframe=False, keyframe_reason=None, prompt_metadata={'tool_progress': {'sample_index': local_index, 'sample_count': len(sampled), 'raw_execution_status': (raw_execution or {}).get('status')}}, privileged_metadata={'low_level_action': dict(low_level_actions[min(local_index, len(low_level_actions) - 1)])})
                decision = controller.on_frame(frame)
                if decision.interrupted:
                    self._stream_logical_time_s = max(self._stream_logical_time_s, timestamp_s)
                    record = self._stream_interrupt_record(frame, decision, local_index)
                    return record
        finally:
            self._set_agent_pose_from_mapping(final_pose)
        self._stream_logical_time_s = max(self._stream_logical_time_s, final_timestamp_s)
        return None

    def _emit_tool_result_stream_keyframe(self, *, tool_name: str, arguments: Mapping[str, Any], success: bool, feedback: str, state_delta: Any, raw_execution: Optional[Mapping[str, Any]], goal_result: Mapping[str, Any], stream_interrupt_record: Optional[Mapping[str, Any]], post_action_observation: Optional[Mapping[str, Any]]=None) -> Optional[dict]:
        controller = self.stream_controller
        if controller is None:
            return None
        stream_config = getattr(controller, 'config', None)
        keyframes = getattr(stream_config, 'keyframes', None)
        if stream_interrupt_record is None:
            if keyframes is not None and (not bool(getattr(keyframes, 'tool_result', True))):
                return None
            source = 'tool_result'
            reason = 'tool_result'
        else:
            if keyframes is not None and (not bool(getattr(keyframes, 'interrupt', True))):
                return None
            source = 'interrupt'
            reason = 'interrupt'
        timing = getattr(stream_config, 'logical_timing', None)
        if stream_interrupt_record is None:
            if tool_name in {'navigate_to', 'navigate_to_room', 'scan', 'look_for'}:
                timestamp_s = self._stream_logical_time_s
            else:
                timestamp_s = self._stream_logical_time_s + float(getattr(timing, 'atomic_action_seconds', 0.6))
        else:
            timestamp_s = float(stream_interrupt_record.get('timestamp_s') or self._stream_logical_time_s)
        self._stream_logical_time_s = max(self._stream_logical_time_s, timestamp_s)
        if not self._should_render_stream_frame(timestamp_s, is_keyframe=True):
            return None
        frame = self._build_stream_frame_observation(tool_name=tool_name, arguments=arguments, source=source, timestamp_s=timestamp_s, is_keyframe=True, keyframe_reason=reason, prompt_metadata={'merged_stream_sources': ['fps_sample', source] if stream_interrupt_record is None and tool_name in {'navigate_to', 'navigate_to_room', 'scan', 'look_for'} else [source], 'tool_result': {'tool_success': bool(success), 'tool_feedback': feedback, 'goal_progress': goal_result.get('task_progress'), 'goal_success': goal_result.get('task_success')}, 'state_delta': state_delta or {}}, privileged_metadata={'raw_execution': dict(raw_execution or {})}, reusable_observation=post_action_observation)
        decision = controller.on_frame(frame)
        record = stream_frame_log_record(frame, decision, vlm_called=decision.metadata.get('decision_source') != 'passive_capture', latency_seconds=0.0)
        if decision.interrupted:
            record['tool_call'] = {'tool_name': decision.tool_call.tool_name, 'arguments': dict(decision.tool_call.arguments), 'reasoning': decision.tool_call.reasoning}
        return record

    def _build_stream_frame_observation(self, *, tool_name: str, arguments: Mapping[str, Any], source: str, timestamp_s: float, is_keyframe: bool, keyframe_reason: Optional[str], prompt_metadata: Optional[Mapping[str, Any]]=None, privileged_metadata: Optional[Mapping[str, Any]]=None, reusable_observation: Optional[Mapping[str, Any]]=None) -> StreamFrameObservation:
        pose = self._stream_observation_pose(source=source, keyframe_reason=keyframe_reason)
        self._sync_renderer_for_observation_pose(pose)
        with runtime_phase('stream_capture', tool_name=tool_name):
            reused = self._reusable_stream_observation(reusable_observation, pose)
            if reused is None:
                visible_ids = self._observation_visible_object_names_at_pose(pose)
                head_rgb = self._render_head_rgb_at_pose(pose)
            else:
                head_rgb, visible_ids = reused
        self._observe_visible_objects(visible_ids, source='stream_{}'.format(source), pose=pose)
        full_catalog = self._object_catalog(visible_object_ids=visible_ids)
        visible_id_set = set((str(item) for item in visible_ids))
        visible_full_catalog = [item for item in full_catalog if str(item.get('name') or item.get('objectId')) in visible_id_set]
        prompt_catalog = self._prompt_safe_object_catalog(visible_ids)
        privileged = {}
        if self._log_privileged_metadata():
            privileged = {'visible_object_ids': list(visible_ids), 'visible_object_catalog': visible_full_catalog, 'held_objects': self._held_objects(), 'current_room': self.current_room, 'visual_discoverability': self.visual_discoverability_report(), 'visibility_source': self.visual_discoverability_report().get('visibility_source'), 'segmentation_status': 'available' if self.visual_discoverability_report().get('rendered_segmentation_available') else 'unknown', 'symbolic_fov_fallback': bool(self.visual_discoverability_report().get('symbolic_fov_fallback'))}
            privileged.update(self._stream_target_evidence(tool_name, arguments, visible_ids))
            privileged.update(dict(privileged_metadata or {}))
        prompt = {'held_objects': self._held_objects(), 'current_room': self.current_room, 'tool_target_handles': build_prompt_target_handles(role_map=self.role_map if self.expose_role_bindings_to_policy else {}, visible_object_ids=visible_ids, tracker=self._catalog_tracker), 'room_target_handles': self._room_target_handles(), 'human_messages': list(self.human_messages), 'visible_issues': self._public_visible_issues(), 'metadata_mode': self.metadata_mode}
        if self.expose_role_bindings_to_policy:
            prompt['visible_object_roles'] = self._roles_for_object_names(visible_ids)
        if self._metadata_exposes('raw_object_ids'):
            prompt['visible_object_ids'] = list(visible_ids)
        prompt.update(dict(prompt_metadata or {}))
        frame = StreamFrameObservation(stream_id=self._stream_id, task_id=str(self.current_task_spec.get('id') if self.current_task_spec else ''), high_level_step=int(self.high_level_step), active_tool_name=str(tool_name), active_tool_args=dict(arguments), frame_index=int(self._stream_frame_index), timestamp_s=float(timestamp_s), source=str(source), is_keyframe=bool(is_keyframe), keyframe_reason=keyframe_reason, agent_pose=pose, head_rgb=head_rgb, visible_object_catalog=prompt_catalog, prompt_metadata=prompt, privileged_metadata=privileged)
        self._stream_frame_index += 1
        return frame

    def _reusable_stream_observation(self, observation: Optional[Mapping[str, Any]], pose: Mapping[str, Any]) -> tuple[Any, List[str]] | None:
        if not isinstance(observation, Mapping) or observation.get('head_rgb') is None:
            return None
        visible = observation.get('visible_object_ids')
        observation_pose = observation.get('observation_pose')
        if isinstance(visible, str) or not isinstance(visible, Iterable) or (not isinstance(observation_pose, Mapping)):
            return None
        expected = self._pose_to_dict(pose)
        actual = self._pose_to_dict(observation_pose)
        if any((not math.isclose(actual[name], expected[name], abs_tol=1e-09) for name in expected)):
            return None
        return (observation['head_rgb'], sorted((str(item) for item in visible)))

    def _stream_target_evidence(self, tool_name: str, arguments: Mapping[str, Any], visible_object_ids: Sequence[str]) -> dict[str, Any]:
        target_ref = arguments.get('target')
        if target_ref is None or self.core is None:
            return {}
        reference_key = self._binding_reference_key(target_ref)
        records = [self._last_resolved_targets.get('any'), self._last_resolved_targets.get('spatial_focus')]
        binding = next((record for record in records if isinstance(record, Mapping) and self._binding_reference_key(record.get('raw_reference')) == reference_key and (str(record.get('tool_name') or '') == str(tool_name))), None)
        if binding is None:
            binding = next((record for record in records if isinstance(record, Mapping) and self._binding_reference_key(record.get('raw_reference')) == reference_key), None)
        object_id = str(binding.get('selected_object') or '') if binding is not None else ''
        obj = self.core.objects.get(object_id) if object_id else None
        if obj is None:
            return {}
        pose = self.core.ego_state.pose
        distance = math.hypot(float(pose.x) - float(obj.center[0]), float(pose.y) - float(obj.center[1]))
        minimum, maximum = self.core.config.navigation.interaction_distance
        return {'target_object_ids': [object_id], 'target_distance_m': distance, 'interaction_ready': float(minimum) <= distance <= float(maximum), 'target_visible': object_id in {str(item) for item in visible_object_ids}}

    def _target_interaction_readiness(self, raw_execution: Mapping[str, Any]) -> dict[str, Any]:
        target_name = str(raw_execution.get('_private_interaction_target_id') or raw_execution.get('object_id') or '')
        if raw_execution.get('status') != 'success' or self.core is None or (not target_name) or (target_name not in self.core.objects):
            return {'target_distance_m': None, 'interaction_ready': False}
        target_visible = bool(raw_execution.get('object_visible') or raw_execution.get('target_visible_after_navigation'))
        if not target_visible:
            target_visible = target_name in {str(item) for item in raw_execution.get('visible_object_ids', ()) or ()}
        pose = self.core.ego_state.pose
        distance = self._object_xy_distance(tuple(pose.xy), self.core.objects[target_name])
        minimum, maximum = self.core.config.navigation.interaction_distance
        ready = bool(target_visible and distance is not None and (float(minimum) <= float(distance) <= float(maximum)))
        return {'target_distance_m': distance, 'interaction_ready': ready}

    def _stream_interrupt_record(self, frame: StreamFrameObservation, decision: StreamDecision, low_level_index: int) -> dict:
        tool_call = decision.tool_call
        assert tool_call is not None
        correlation = decision.metadata.get('trajectory_correlation')
        if not isinstance(correlation, Mapping):
            correlation = {}
        return {'source': 'stream_interrupt', 'stream_id': frame.stream_id, 'stream_obs_id': correlation.get('stream_obs_id'), 'frame_id': correlation.get('frame_id'), 'frame_index': frame.frame_index, 'low_level_index': int(low_level_index), 'timestamp_s': frame.timestamp_s, 'tool_call': {'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments), 'reasoning': tool_call.reasoning}, 'reasoning': decision.reasoning, 'raw_response': decision.raw_response, 'trajectory_correlation': dict(correlation)}

    def _render_head_rgb_at_pose(self, pose: Mapping[str, Any]) -> Any | None:
        if self.renderer is None:
            return None
        self._sync_renderer_for_observation_pose(pose)
        try:
            return self.renderer.render_at_pose(pose)
        except Exception:
            return None

    def _set_agent_pose_from_mapping(self, pose: Mapping[str, Any]) -> None:
        if self.core is None:
            return
        current = self.core.ego_state.pose
        try:
            current.x = float(pose.get('x', current.x))
            current.y = float(pose.get('y', current.y))
            current.z = float(pose.get('z', current.z))
            current.yaw = float(pose.get('yaw', current.yaw))
            current.pitch = float(pose.get('pitch', current.pitch))
        except Exception:
            return

    def _level_scan_pose(self, pose: Any) -> EgoPose:
        pose_dict = self._pose_to_dict(pose)
        return EgoPose(x=float(pose_dict.get('x', 0.0)), y=float(pose_dict.get('y', 0.0)), z=float(pose_dict.get('z', 1.55)), yaw=float(pose_dict.get('yaw', 0.0)), pitch=-math.pi / 6)

    def _effective_scan_num_views(self, requested_num_views: int) -> int:
        value = int(requested_num_views)
        return value if 1 <= value <= 8 else 0

    def _stream_observation_pose(self, *, source: str, keyframe_reason: Optional[str]) -> dict:
        if self._reset_observation_pose is not None and int(self.high_level_step) == 0 and (source in {'reset', 'scheduler_event'}) and (keyframe_reason in {'reset', 'episode_start'}):
            return self._pose_to_dict(self._reset_observation_pose)
        return self._agent_pose()

    def _prepare_initial_observation_pose(self) -> dict:
        if self.core is None:
            return {}
        mode = os.environ.get('EB_BEHAVIOR_INITIAL_VIEW_MODE', 'rgb_search')
        if mode not in {'rgb_search', 'task_pose'}:
            raise ValueError('unsupported EB_BEHAVIOR_INITIAL_VIEW_MODE: ' + mode)
        pose = self.core.ego_state.pose if mode == 'task_pose' else self._select_task_agnostic_rgb_observation_pose(self.core.ego_state.pose)
        self._reset_observation_pose = self._pose_to_dict(pose)
        return dict(self._reset_observation_pose)

    def _sync_renderer_for_observation_pose(self, pose: Mapping[str, Any]) -> None:
        if self.renderer is None or self.core is None or (not hasattr(self.renderer, 'sync_from_core')):
            return
        original_pose = self.core.ego_state.pose.copy()
        original_held_anchor = getattr(self.core, '_held_pose_anchor', None)
        try:
            original_held_anchor = original_held_anchor.copy() if original_held_anchor is not None else None
        except Exception:
            pass
        try:
            self._set_agent_pose_from_mapping(pose)
            if hasattr(self.core, '_update_held_pose_anchor'):
                try:
                    self.core._update_held_pose_anchor(self.core.ego_state.pose)
                except Exception:
                    pass
            self.renderer.sync_from_core(self.core)
        except Exception:
            pass
        finally:
            self.core.ego_state.pose = original_pose
            try:
                self.core._held_pose_anchor = original_held_anchor
            except Exception:
                pass

    def reset_pose_diagnostics(self) -> dict[str, Any]:
        source = self.current_task_spec.get('source') if isinstance(self.current_task_spec, Mapping) else {}
        occupancy = self.core.occupancy if self.core is not None else None
        metadata = dict(getattr(occupancy, 'metadata', {}) or {}) if occupancy is not None else {}
        placement_report = dict(getattr(getattr(self, 'scene_result', None), 'placement_report', {}) or {})
        return {'source_start_pose': dict(source.get('start_pose') or {}) if isinstance(source, Mapping) else {}, 'initial_view_mode': os.environ.get('EB_BEHAVIOR_INITIAL_VIEW_MODE', 'rgb_search'), 'scene_layout_mode': self.scene_layout_mode, 'load_doors': bool(self.load_doors), 'allow_task_relevant_start_pose_repair': bool(self.allow_task_relevant_start_pose_repair), 'allow_occupied_start_pose_repair': bool(self.allow_occupied_start_pose_repair), 'scene_build_scene_layout_mode': placement_report.get('scene_layout_mode'), 'source_json_full_layout': placement_report.get('source_json_full_layout') or {}, 'start_pose_room_repair': dict(metadata.get('start_pose_room_repair') or {}), 'start_pose_repair': dict(metadata.get('start_pose_repair') or {}), 'occupancy_component_count': int(metadata.get('component_count') or 0), 'renderer_pose_readback': dict(self._last_renderer_pose_readback or {}), 'final_reset_pose': self._agent_pose() if self.core is not None else {}, 'reset_observation_pose': dict(self._reset_observation_pose or {}), 'current_room': self.current_room}

    def _sync_core_from_renderer_runtime(self) -> dict[str, Any]:
        if self.renderer is None or self.core is None or (not hasattr(self.renderer, 'object_pose_catalog')):
            return {'enabled': False, 'reason': 'renderer_pose_catalog_unavailable'}
        try:
            catalog = dict(self.renderer.object_pose_catalog() or {})
        except Exception as exc:
            return {'enabled': False, 'reason': 'renderer_pose_catalog_error', 'error': repr(exc)}
        if not catalog:
            return {'enabled': False, 'reason': 'renderer_pose_catalog_empty'}
        by_safe_name = {str(name): dict(record) for name, record in catalog.items() if isinstance(record, Mapping)}
        updated: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for name, obj in getattr(self.core, 'objects', {}).items():
            candidates = _renderer_pose_match_candidates(str(name), obj)
            record = next((by_safe_name[candidate] for candidate in candidates if candidate in by_safe_name), None)
            if record is None:
                states = getattr(obj, 'states', {}) or {}
                if isinstance(states, Mapping) and states.get('scene_layout_source') == 'source_json_full':
                    missing.append({'name': str(name), 'candidates': candidates[:4]})
                continue
            before = {'center': list(getattr(obj, 'center', ())), 'extent': list(getattr(obj, 'extent', ())), 'root_position': list(getattr(obj, 'root_position', ()) or [])}
            if _update_aabb_object_from_runtime_pose(obj, record):
                updated.append({'name': str(name), 'renderer_name': str(record.get('name') or ''), 'before': before, 'after': {'center': list(getattr(obj, 'center', ())), 'extent': list(getattr(obj, 'extent', ())), 'root_position': list(getattr(obj, 'root_position', ()) or []), 'root_yaw': getattr(obj, 'root_yaw', None)}})
        return {'enabled': True, 'renderer_object_count': len(by_safe_name), 'updated_object_count': len(updated), 'missing_source_json_object_count': len(missing), 'updated_objects': updated[:200], 'missing_source_json_objects': missing[:200]}

    def _select_task_agnostic_rgb_observation_pose(self, pose: Any) -> dict:
        base_pose = self._pose_to_dict(pose)
        if self.renderer is None:
            return base_pose
        candidates = self._rgb_observation_pose_candidates(base_pose)
        scored: list[tuple[int, int, int, float, int, int, dict]] = []
        for index, candidate in enumerate(candidates):
            rgb = self._render_head_rgb_at_pose(candidate)
            visual_score = self._rgb_observation_pose_score(rgb)
            useful_visible, actionable_visible = self._generic_initial_view_visibility_score(candidate)
            base_preference = 1 if index == 0 else 0
            visual_bucket = int(float(visual_score) // 8.0)
            scored.append((int(visual_bucket), int(actionable_visible), int(useful_visible), float(visual_score), int(base_preference), -int(index), candidate))
        if not scored:
            return base_pose
        return self._pose_to_dict(max(scored, key=lambda item: item[:6])[6])

    def _generic_initial_view_visibility_score(self, pose: Any) -> tuple[int, int]:
        if self.core is None:
            return (0, 0)
        try:
            visible = list(self._prompt_visible_object_names_at_pose(pose))
        except Exception:
            return (0, 0)
        useful = 0
        actionable = 0
        for name in visible:
            obj = self.core.objects.get(str(name))
            if obj is None:
                continue
            category = str(getattr(obj, 'category', '') or '').strip().lower()
            if category in _OBSERVATION_STRUCTURAL_CATEGORIES:
                continue
            useful += 1
            abilities = set((str(item) for item in getattr(obj, 'abilities', ()) or ()))
            if abilities & {'pickupable', 'openable', 'receptacle', 'support_surface', 'container', 'toggleable'}:
                actionable += 1
        return (int(useful), int(actionable))

    def _rgb_observation_pose_candidates(self, base_pose: Mapping[str, Any]) -> list[dict]:
        pose = self._pose_to_dict(base_pose)
        base_yaw = float(pose.get('yaw', 0.0))
        base_pitch = float(pose.get('pitch', 0.0))
        candidates: list[dict] = []
        seen: set[tuple[int, int]] = set()
        for yaw_delta in _RGB_OBSERVATION_YAW_DELTAS:
            yaw = _normalize_angle(base_yaw + float(yaw_delta))
            for pitch in (base_pitch, *_RGB_OBSERVATION_PITCHES):
                candidate = dict(pose)
                candidate['yaw'] = yaw
                candidate['pitch'] = max(-0.85, min(0.45, float(pitch)))
                key = (round(float(candidate.get('yaw', 0.0)) * 1000), round(float(candidate.get('pitch', 0.0)) * 1000))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    def _rgb_observation_pose_score(self, rgb: Any) -> float:
        if rgb is None:
            return 0.0
        try:
            array = np.asarray(rgb)
            if array.ndim != 3 or array.shape[0] <= 1 or array.shape[1] <= 1:
                return 0.0
            if array.shape[-1] > 3:
                array = array[:, :, :3]
            arr = array.astype(np.float32, copy=False)
            information = self._image_information_score(arr)
            luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
            center = luminance[max(0, luminance.shape[0] // 4):max(1, luminance.shape[0] * 3 // 4), max(0, luminance.shape[1] // 4):max(1, luminance.shape[1] * 3 // 4)]
            center_std = float(np.std(center)) if center.size else 0.0
            step_y = max(1, int(arr.shape[0]) // 64)
            step_x = max(1, int(arr.shape[1]) // 64)
            sample = arr[::step_y, ::step_x, :]
            unique_ratio = float(len(np.unique(sample.reshape(-1, sample.shape[-1]).astype(np.uint8), axis=0))) / float(max(1, sample.shape[0] * sample.shape[1]))
            return float(information) + 0.2 * center_std + 120.0 * unique_ratio
        except Exception:
            return self._image_information_score(rgb)

    def _prepare_prompt_observation_pose(self) -> dict:
        if self.core is None:
            return {}
        base_pose = self._pose_to_dict(self.core.ego_state.pose)
        if self._reset_observation_pose is not None and int(self.high_level_step) == 0:
            return self._pose_to_dict(self._reset_observation_pose)
        return base_pose

    def _focused_observation_candidates(self, base_pose: Mapping[str, Any], visible_object_ids: Sequence[str]) -> list[tuple[dict, list[str], str | None]]:
        if self.core is None:
            return []
        candidates: list[tuple[dict, list[str], str | None]] = []
        for object_id in visible_object_ids:
            obj = self.core.objects.get(str(object_id))
            if obj is None or not self._is_prompt_focusable_object(obj):
                continue
            try:
                focused = pose_looking_at_object(EgoPose(**self._pose_to_dict(base_pose)), obj)
            except Exception:
                focused = None
            if focused is None:
                continue
            for focused_pose in self._prompt_pitch_variants(self._pose_to_dict(focused)):
                focused_visible = list(self._prompt_visible_object_names_at_pose(focused_pose))
                if str(object_id) not in set((str(name) for name in focused_visible)):
                    focused_visible.append(str(object_id))
                candidates.append((focused_pose, sorted(set((str(name) for name in focused_visible))), str(object_id)))
        return candidates

    def _prompt_pitch_variants(self, pose: Mapping[str, Any]) -> list[dict]:
        pose_dict = self._pose_to_dict(pose)
        try:
            base_pitch = float(pose_dict.get('pitch', 0.0))
        except Exception:
            base_pitch = 0.0
        variants: list[dict] = []
        seen: set[tuple[int, int]] = set()
        for delta in (0.0, -0.2, -0.35, 0.2):
            variant = dict(pose_dict)
            variant['pitch'] = max(-0.85, min(0.45, base_pitch + float(delta)))
            key = (round(float(variant.get('yaw', 0.0)) * 1000), round(float(variant.get('pitch', 0.0)) * 1000))
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant)
        return variants

    def _is_prompt_focusable_object(self, obj: Any) -> bool:
        category = str(getattr(obj, 'category', '') or '').strip().lower()
        if category in _OBSERVATION_STRUCTURAL_CATEGORIES:
            return False
        return True

    def _rendered_prompt_visibility_score(self, pose: Any) -> tuple[int, int]:
        if self.renderer is None or self.core is None or (not self.use_rendered_visibility) or self._strict_rendered_visibility_gate_enabled():
            return (0, 0)
        visibility_query = getattr(self.renderer, 'render_visibility_at_pose', None)
        if not callable(visibility_query):
            return (0, 0)
        object_names = [str(name) for name in self.core.objects.keys()]
        try:
            payload = visibility_query(pose, object_names=object_names, min_pixels=1)
        except TypeError:
            try:
                payload = visibility_query(pose)
            except Exception:
                return (0, 0)
        except Exception:
            return (0, 0)
        if not isinstance(payload, Mapping):
            return (0, 0)
        visible_ids = [str(name) for name in payload.get('visible_object_ids') or []]
        pixel_counts = payload.get('pixel_counts_by_label') or {}
        raw_to_safe = {raw_name: self._render_safe_name_for_object(raw_name) for raw_name in object_names}
        visible_non_structural = 0
        total_pixels = 0
        for raw_name in visible_ids:
            obj = self.core.objects.get(raw_name)
            if obj is None or not self._is_prompt_focusable_object(obj):
                continue
            visible_non_structural += 1
            safe_name = raw_to_safe.get(raw_name) or _render_safe_name(raw_name)
            total_pixels += self._render_pixel_count_for_object(raw_name, safe_name, pixel_counts)
        return (int(total_pixels), int(visible_non_structural))

    def _render_safe_name_for_object(self, object_id: str) -> str:
        raw_mapping = getattr(self.renderer, '_raw_name_to_safe_name', None)
        if isinstance(raw_mapping, Mapping):
            value = raw_mapping.get(str(object_id))
            if value:
                return str(value)
        return _render_safe_name(object_id)

    def _render_pixel_count_for_object(self, raw_name: str, safe_name: str, pixel_counts: Mapping[str, Any]) -> int:
        best = 0
        raw_text = str(raw_name)
        safe_text = str(safe_name)
        for label, count in pixel_counts.items():
            label_text = str(label)
            if label_text == raw_text or label_text == safe_text or raw_text in label_text or (safe_text in label_text):
                try:
                    best = max(best, int(count))
                except Exception:
                    continue
        return int(best)

    def _image_information_score(self, rgb: Any) -> float:
        if rgb is None:
            return 0.0
        try:
            array = np.asarray(rgb)
            if array.ndim != 3 or array.shape[0] <= 1 or array.shape[1] <= 1:
                return 0.0
            if array.shape[-1] > 3:
                array = array[:, :, :3]
            arr = array.astype(np.float32, copy=False)
            luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
            gradient = float(np.mean(np.abs(np.diff(luminance, axis=0)))) + float(np.mean(np.abs(np.diff(luminance, axis=1))))
            saturation = float(np.mean(np.max(arr, axis=2) - np.min(arr, axis=2)))
            return float(np.std(luminance)) + 0.5 * gradient + 0.05 * saturation
        except Exception:
            return 0.0

    def _done_status(self, goal_result: Mapping[str, Any]) -> tuple[bool, Optional[str]]:
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
        if float(goal_result.get('task_success', 0.0) or 0.0) == 1.0 and (not bool(goal_result.get('require_model_stop', False))):
            return (True, 'task_success')
        return (False, None)

    def _episode_metrics(self) -> dict:
        return {'high_level_steps': self.high_level_step, 'task_action_steps': self.task_action_step, 'total_low_level_steps': self.total_low_level_steps, 'total_frames': self.total_frames, 'invalid_tool_calls': self.invalid_tool_calls, 'model_stopped': self.model_stopped, 'scan_steps': self.scan_steps, 'look_for_steps': self.look_for_steps, 'answer_steps': self.answer_steps, 'seen_object_navigation': self.seen_object_navigation_steps, 'stale_object_navigation': self.stale_object_navigation_steps, 'unseen_object_exploration': self.unseen_object_exploration_steps, 'target_missing_at_last_seen': self.target_missing_at_last_seen_count, 'binding_metrics': self.binding_metrics(), 'navigation_metrics': {'seen_object_navigation': self.seen_object_navigation_steps, 'stale_object_navigation': self.stale_object_navigation_steps, 'unseen_object_exploration': self.unseen_object_exploration_steps, 'target_missing_at_last_seen': self.target_missing_at_last_seen_count}, 'visual_discoverability': self.visual_discoverability_report()}

    def visual_discoverability_report(self) -> dict:
        mode = str(self._last_visibility_mode or 'symbolic_fov')
        render_status = str(self._last_render_visibility_status or ('renderer_absent' if self.renderer is None else 'not_queried'))
        rendered = render_status == 'rendered_segmentation_available'
        symbolic_fallback = mode == 'symbolic_fov'
        visibility_source = 'behavior_rendered_segmentation' if rendered else 'behavior_symbolic_fov' if symbolic_fallback else 'unavailable'
        report = {'mode': mode, 'visibility_source': visibility_source, 'renderer_configured': self.renderer is not None, 'rendered_segmentation_available': rendered, 'rendered_visibility_status': render_status, 'symbolic_fov_fallback': symbolic_fallback, 'symbolic_fov': {'horizontal_degrees': _SYMBOLIC_HORIZONTAL_FOV_DEGREES, 'vertical_degrees': _SYMBOLIC_VERTICAL_FOV_DEGREES, 'max_distance': _SYMBOLIC_VISIBILITY_MAX_DISTANCE}, 'visual_stream_latency_available': False, 'visual_latency_status': 'unavailable_not_ai2_thor_stream', 'ai2_thor_controller_metadata': 'not_available', 'procthor_house_registry': 'not_required', 'non_goal': 'ai2_thor_controller_metadata_and_procthor_house_registry'}
        if self._last_render_visibility_error:
            report['rendered_visibility_error'] = self._last_render_visibility_error
        return report

    def get_tool_specs(self, model_visible_only: bool=True) -> List[dict]:
        names = MODEL_VISIBLE_TOOLS if model_visible_only else MODEL_VISIBLE_TOOLS + DEFERRED_TOOLS
        specs = []
        for name in names:
            spec = {'name': name, 'description': _TOOL_DESCRIPTIONS.get(name, name)}
            if name == 'scan':
                spec['parameters'] = _json_schema({'num_views': {'type': 'integer', 'description': 'Number of directions sampled around the current position. Use 4–8 for room inspection; 1 refreshes the current direction without added coverage.', 'minimum': 1, 'maximum': 8}}, required=[])
            elif name == 'navigate_to_room':
                spec['parameters'] = _json_schema({'room': {'type': 'string', 'description': 'Target room instance id, such as kitchen_0.'}}, required=['room'])
            elif name == 'answer':
                spec['parameters'] = _json_schema({'question_id': {'type': 'string', 'description': 'Identifier of the pending human QA.'}, 'value': {'type': 'string', 'description': 'Answer value to record.'}}, required=['question_id', 'value'])
            elif name == 'fill':
                spec['parameters'] = _json_schema({'target': _role_schema('Container role to fill.'), 'source': _role_schema('Fluid source role.')}, required=['target', 'source'])
            elif name in ('stop', 'drop'):
                spec['parameters'] = _json_schema({}, required=[])
            else:
                spec['parameters'] = _json_schema({'target': _role_schema('Target object or receptacle role.')}, required=['target'])
            specs.append(spec)
        return specs

    def _clip_occupancy_bounds_to_traversability(self, bounds: Any, occupancy_kwargs: Mapping[str, Any]) -> tuple[float, float, float, float]:
        try:
            min_x, min_y, max_x, max_y = [float(v) for v in bounds]
        except Exception:
            min_x, min_y, max_x, max_y = (-0.5, -0.5, 10.5, 6.5)
        if occupancy_kwargs.get('traversability_map') is None:
            return (min_x, min_y, max_x, max_y)
        try:
            resolution = float(occupancy_kwargs.get('traversability_map_resolution') or 0.0)
            size = int(occupancy_kwargs.get('traversability_map_size') or 0)
        except Exception:
            return (min_x, min_y, max_x, max_y)
        if resolution <= 0.0 or size <= 0:
            return (min_x, min_y, max_x, max_y)
        half_extent = float(size) * float(resolution) / 2.0
        clipped = (max(min_x, -half_extent), max(min_y, -half_extent), min(max_x, half_extent), min(max_y, half_extent))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            return (-half_extent, -half_extent, half_extent, half_extent)
        return clipped

    def _objects_with_navigation_overrides(self, aabb_objects: Sequence[AABBObject]) -> list[AABBObject]:
        objects = [obj for obj in aabb_objects if self.load_doors or not is_door_like(getattr(obj, 'category', ''), getattr(obj, 'name', ''))]
        by_name = {obj.name: obj for obj in objects}
        for extra in self.scene_navigation_objects:
            if not self.load_doors and is_door_like(getattr(extra, 'category', ''), getattr(extra, 'name', '')):
                continue
            if extra.name in by_name:
                existing = by_name[extra.name]
                existing_states = dict(getattr(existing, 'states', {}) or {})
                existing_states.update(dict(getattr(extra, 'states', {}) or {}))
                existing.states = existing_states
                existing.abilities = tuple(sorted(set(existing.abilities) | set(getattr(extra, 'abilities', ()) or ())))
                continue
            objects.append(extra)
            by_name[extra.name] = extra
        for obj in objects:
            faces = self._interaction_faces_for_name(obj.name)
            if not faces:
                continue
            states = dict(getattr(obj, 'states', {}) or {})
            states['interaction_faces'] = faces
            obj.states = states
        return objects

    def _repair_start_pose_after_occupancy(self) -> None:
        if self.core is None or self.core.occupancy is None:
            return
        occupancy = self.core.occupancy
        room_repair = occupancy.metadata.get('start_pose_room_repair')
        if isinstance(room_repair, Mapping) and bool(room_repair.get('success')):
            room_repair['occupancy_component_repair_skipped'] = True
            room_repair['occupancy_component_repair_skip_reason'] = 'task_relevant_room_reset_repair'
            return
        pose = self.core.ego_state.pose
        start_cell = occupancy.world_to_cell(pose.xy)
        target_components = self._primary_target_interaction_components(occupancy)
        start_is_free = occupancy.is_cell_free(start_cell)
        start_component = occupancy.component_at(pose.xy) if start_is_free else -1
        if start_is_free and (not target_components or int(start_component) in target_components):
            return
        nearest = self._nearest_free_cell(occupancy, pose.xy, allowed_components=target_components)
        used_target_component = True
        if nearest is None:
            nearest = self._nearest_free_cell(occupancy, pose.xy)
            used_target_component = False
        if nearest is None:
            occupancy.metadata['start_pose_repair'] = {'attempted': True, 'success': False, 'reason': 'no_eroded_free_cells', 'start_pose': pose.as_dict(), 'start_cell': [int(start_cell[0]), int(start_cell[1])]}
            return
        repaired_cell, repaired_xy, distance = nearest
        before_pose = pose.copy()
        pose.x = float(repaired_xy[0])
        pose.y = float(repaired_xy[1])
        pose.yaw = self._yaw_towards_primary_target(repaired_xy, fallback=pose.yaw)
        reason = 'start_occupied_after_occupancy'
        if start_is_free and target_components and (int(start_component) not in target_components):
            reason = 'start_not_connected_to_primary_target'
        repair = {'attempted': True, 'success': True, 'reason': reason, 'distance_m': float(distance), 'start_pose': before_pose.as_dict(), 'start_cell': [int(start_cell[0]), int(start_cell[1])], 'start_component': int(start_component), 'repaired_pose': pose.as_dict(), 'repaired_cell': [int(repaired_cell[0]), int(repaired_cell[1])], 'repaired_component': int(occupancy.component_at(repaired_xy)), 'preferred_target_components': sorted((int(component) for component in target_components)), 'used_target_component': used_target_component}
        occupancy.metadata['start_pose_repair'] = repair
        try:
            self.core.trace.emit('reset_occupancy', 'start_pose_repaired', **repair)
        except Exception:
            pass

    def _repair_start_pose_to_task_relevant_room(self) -> None:
        if self.core is None or self.core.occupancy is None or (not self.room_catalog):
            return
        source = self.current_task_spec.get('source') if isinstance(self.current_task_spec, Mapping) else {}
        start_pose = source.get('start_pose') if isinstance(source, Mapping) else None
        if not isinstance(start_pose, Mapping):
            return
        if not self._start_pose_looks_like_profile_placeholder(start_pose):
            return
        preferred_room_id = self._preferred_start_room_id_for_current_task()
        current_room_id = str(start_pose.get('room_id') or self._current_room_for_pose(self.core.ego_state.pose) or '')
        if not preferred_room_id or preferred_room_id == current_room_id:
            return
        room = next((item for item in self.room_catalog if str(item.get('room_id')) == preferred_room_id), None)
        if room is None:
            return
        before_pose = self.core.ego_state.pose.copy()
        target_xy, record, error = self._reset_room_repair_goal_xy(room)
        if target_xy is None:
            self.core.occupancy.metadata['start_pose_room_repair'] = {'attempted': True, 'success': False, 'reason': 'task_relevant_room_unreachable', 'start_pose': before_pose.as_dict(), 'start_room_id': current_room_id, 'preferred_room_id': preferred_room_id, 'room_navigation_record': dict(record or {}), 'error': error}
            return
        pose = self.core.ego_state.pose
        pose.x = float(target_xy[0])
        pose.y = float(target_xy[1])
        pose.yaw = self._yaw_towards_primary_target(target_xy, fallback=float(pose.yaw))
        repair = {'attempted': True, 'success': True, 'reason': 'profile_placeholder_start_in_non_task_room', 'start_pose': before_pose.as_dict(), 'start_room_id': current_room_id, 'preferred_room_id': preferred_room_id, 'repaired_pose': pose.as_dict(), 'room_navigation_record': dict(record or {})}
        self.core.occupancy.metadata['start_pose_room_repair'] = repair
        try:
            self.core.trace.emit('reset_occupancy', 'start_pose_room_repaired', **repair)
        except Exception:
            pass

    def _reset_room_repair_goal_xy(self, room: Mapping[str, Any]) -> tuple[tuple[float, float] | None, dict[str, Any], Optional[str]]:
        layout_xy, layout_record, _layout_error = self._room_center_xy_from_layout(room)
        if layout_xy is not None:
            record = dict(layout_record or {})
            record['source'] = 'floor_trav_room_center_reset_repair'
            record['selected_goal_xy'] = [float(layout_xy[0]), float(layout_xy[1])]
            occupancy = self.core.occupancy if self.core is not None else None
            if occupancy is not None:
                try:
                    record['core_occupancy_free_at_layout_xy'] = bool(occupancy.is_free(layout_xy))
                    record['core_occupancy_component_at_layout_xy'] = int(occupancy.component_at(layout_xy))
                except Exception:
                    pass
                nearest = self._nearest_free_cell(occupancy, layout_xy)
                if nearest is not None:
                    cell, nearest_xy, distance = nearest
                    record['nearest_core_free_cell'] = [int(cell[0]), int(cell[1])]
                    record['nearest_core_free_xy'] = [float(nearest_xy[0]), float(nearest_xy[1])]
                    record['nearest_core_free_distance_m'] = float(distance)
                    if float(distance) <= 1.0:
                        record['selected_goal_xy'] = [float(nearest_xy[0]), float(nearest_xy[1])]
                        record['selected_goal_cell'] = [int(cell[0]), int(cell[1])]
                        return (nearest_xy, record, None)
            return (layout_xy, record, None)
        return self._room_navigation_goal_xy(room)

    def _room_center_xy_from_layout(self, room: Mapping[str, Any]) -> tuple[tuple[float, float] | None, dict[str, Any], Optional[str]]:
        layout = self._room_layout_maps_for_task()
        if layout is None:
            return (None, {'reachable_status': 'room_layout_unavailable'}, self._room_layout_error)
        room_id = str(room.get('room_id') or '')
        room_mask = layout.room_mask(room_id)
        resolved_room_id = room_id
        if room_mask is None:
            room_type = str(room.get('room_type') or '')
            instances = list(layout.room_sem_name_to_ins_name.get(room_type) or [])
            if len(instances) == 1:
                resolved_room_id = instances[0]
                room_mask = layout.room_mask(resolved_room_id)
        if room_mask is None:
            return (None, {'reachable_status': 'room_mask_missing', 'source': 'floor_insseg_floor_semseg_floor_trav', 'room_id': room_id, 'available_room_ids_sample': sorted(layout.room_ins_name_to_ins_id)[:20]}, "Room mask is unavailable for '{}'.".format(room_id))
        free_mask = room_mask & layout.traversable_mask()
        rows, cols = np.where(free_mask)
        if len(rows) == 0:
            return (None, {'reachable_status': 'room_traversability_empty', 'source': 'floor_insseg_floor_semseg_floor_trav', 'room_id': room_id, 'resolved_room_id': resolved_room_id}, "Room '{}' has no traversable pixels.".format(room_id))
        center_row = float(np.mean(rows))
        center_col = float(np.mean(cols))
        distances = (rows.astype(np.float64) - center_row) ** 2 + (cols.astype(np.float64) - center_col) ** 2
        best_index = int(np.argmin(distances))
        row = int(rows[best_index])
        col = int(cols[best_index])
        xy = layout.map_to_world((row, col))
        return (xy, {'reachable_status': 'validated_from_room_mask_and_traversability', 'source': 'floor_insseg_floor_semseg_floor_trav', 'room_id': room_id, 'resolved_room_id': resolved_room_id, 'selected_goal_xy': [float(xy[0]), float(xy[1])], 'selected_goal_map_rc': [row, col], 'room_traversable_pixels': int(len(rows)), 'distance_from_room_centroid_m': float(float(distances[best_index]) ** 0.5 * layout.map_resolution), 'map_resolution': float(layout.map_resolution), 'map_size': int(layout.map_size), 'source_paths': dict(layout.source_paths)}, None)

    def _start_pose_looks_like_profile_placeholder(self, start_pose: Mapping[str, Any]) -> bool:
        selection_source = str(start_pose.get('selection_source') or '')
        pose_source = str(start_pose.get('pose_source') or start_pose.get('source') or '')
        if 'placeholder' in selection_source or 'placeholder' in pose_source:
            return True
        if start_pose.get('room_id') and str(start_pose.get('selection_source') or '').startswith('task_relevant_room'):
            return True
        try:
            start_xy = (float(start_pose.get('x')), float(start_pose.get('y')))
        except Exception:
            return False
        for room in self.room_catalog:
            pose = room.get('representative_pose')
            if not isinstance(pose, Mapping):
                continue
            room_source = str(pose.get('source') or '')
            if 'placeholder' not in room_source and str(room.get('reachable_status') or '') != 'profile_fallback':
                continue
            room_xy = self._xy_from_pose_position(pose.get('position'))
            if room_xy is not None and math.dist(start_xy, room_xy) <= 0.05:
                return True
        return False

    def _preferred_start_room_id_for_current_task(self) -> str | None:
        if not self.current_task_spec or not self.room_catalog:
            return None
        room_ids = {str(room.get('room_id')) for room in self.room_catalog if room.get('room_id')}
        if not room_ids:
            return None
        roles = self.current_task_spec.get('roles') if isinstance(self.current_task_spec, Mapping) else {}
        goal = self.current_task_spec.get('goal') if isinstance(self.current_task_spec, Mapping) else {}
        predicates = goal.get('predicates') if isinstance(goal, Mapping) else []
        scores: Counter[str] = Counter()
        if isinstance(predicates, Sequence):
            for predicate in predicates:
                if not isinstance(predicate, Mapping):
                    continue
                args = predicate.get('args')
                if not isinstance(args, Mapping):
                    continue
                for key in ('object', 'receptacle'):
                    role_name = args.get(key)
                    role = roles.get(str(role_name)) if isinstance(roles, Mapping) and role_name is not None else None
                    room_id = str((role or {}).get('room_id') or '')
                    if room_id in room_ids:
                        scores[room_id] += 2 if key == 'receptacle' else 1
        if isinstance(roles, Mapping):
            for role in roles.values():
                if not isinstance(role, Mapping):
                    continue
                room_id = str(role.get('room_id') or '')
                if room_id in room_ids and (not bool(role.get('requires_spawn'))):
                    scores[room_id] += 1
        if not scores:
            return None
        return min(scores, key=lambda room_id: (-int(scores[room_id]), room_id))

    def _nearest_free_cell(self, occupancy: Any, xy: tuple[float, float], allowed_components: set[int] | None=None) -> tuple[tuple[int, int], tuple[float, float], float] | None:
        best: tuple[float, int, int, tuple[float, float]] | None = None
        for row in range(occupancy.height):
            for col in range(occupancy.width):
                if not occupancy.is_cell_free((row, col)):
                    continue
                if allowed_components and int(occupancy.component_labels[row, col]) not in allowed_components:
                    continue
                cell_xy = occupancy.cell_to_world((row, col))
                distance = math.dist(xy, cell_xy)
                candidate = (distance, row, col, cell_xy)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return None
        distance, row, col, cell_xy = best
        return ((int(row), int(col)), cell_xy, float(distance))

    def _primary_target_interaction_components(self, occupancy: Any) -> set[int]:
        if self.core is None:
            return set()
        components: set[int] = set()
        for obj in self._primary_target_objects():
            target = NavigationTarget.from_aabb_object(obj)
            candidates = sample_interaction_candidates(target, self.core.config.navigation.candidate_count, self.core.config.navigation.interaction_distance)
            for xy in candidates:
                cell = occupancy.world_to_cell(xy)
                if not occupancy.is_cell_free(cell):
                    continue
                component = occupancy.component_at(xy)
                if component >= 0:
                    components.add(int(component))
            if not components:
                nearest = self._nearest_free_cell(occupancy, (float(obj.center[0]), float(obj.center[1])))
                if nearest is not None:
                    _, nearest_xy, _ = nearest
                    component = occupancy.component_at(nearest_xy)
                    if component >= 0:
                        components.add(int(component))
        return components

    def _primary_target_object(self) -> AABBObject | None:
        objects = self._primary_target_objects()
        return objects[0] if objects else None

    def _primary_target_objects(self) -> list[AABBObject]:
        if self.core is None:
            return []
        role_names: list[str] = []
        family = self.current_task_spec.get('family_spec') if isinstance(self.current_task_spec, Mapping) else {}
        human = family.get('human') if isinstance(family, Mapping) else {}
        primary = human.get('primary_task') if isinstance(human, Mapping) else {}
        if isinstance(primary, Mapping):
            for field_name in ('object_role', 'receptacle_role'):
                role = primary.get(field_name)
                if role is not None:
                    role_names.append(str(role))
        role_names.extend(['target_object', 'primary_object', 'target_0', 'target_1', 'target_2', 'target_receptacle', 'primary_receptacle', 'storage_0', 'storage_1', 'target_container', 'container', 'source'])
        objects: list[AABBObject] = []
        seen: set[str] = set()
        for role_name in role_names:
            object_name = self.role_map.get(role_name)
            if not object_name or object_name in seen:
                continue
            obj = self.core.objects.get(object_name)
            if obj is not None:
                objects.append(obj)
                seen.add(object_name)
        return objects

    def _yaw_towards_primary_target(self, xy: tuple[float, float], fallback: float) -> float:
        obj = self._primary_target_object()
        if obj is None:
            return fallback
        dx = float(obj.center[0]) - float(xy[0])
        dy = float(obj.center[1]) - float(xy[1])
        if abs(dx) < 1e-09 and abs(dy) < 1e-09:
            return fallback
        return math.atan2(dy, dx)

    def _interaction_faces_for_name(self, object_name: str) -> list[dict[str, Any]]:
        if object_name in self.interaction_faces_by_name:
            return [dict(face) for face in self.interaction_faces_by_name[object_name]]
        safe_name = object_name.replace('.', '_').replace('-', '_')
        if safe_name in self.interaction_faces_by_name:
            return [dict(face) for face in self.interaction_faces_by_name[safe_name]]
        return []

    def _apply_visual_action_result(self, action_result: Any) -> list[dict[str, Any]]:
        if self.renderer is None or not hasattr(self.renderer, 'apply_action_result'):
            return []
        try:
            mutations = list(self.renderer.apply_action_result(action_result) or [])
        except Exception as exc:
            return [{'success': False, 'reason': 'apply_action_result_error', 'action': getattr(action_result, 'action', None), 'error': repr(exc)}]
        mutations.extend(self._apply_source_object_pose_visual_sync(action_result))
        self._mark_visual_sensor_settle(mutations, reason='action_visual_state')
        for mutation in mutations:
            self._update_core_object_pose_from_visual_mutation(mutation)
        return mutations

    def _mark_visual_sensor_settle(self, mutations: Sequence[Mapping[str, Any]], *, reason: str) -> None:
        relevant = [dict(mutation) for mutation in mutations or () if isinstance(mutation, Mapping) and str(mutation.get('state') or '') in {'toggled_on', 'liquid_stream', 'dirty'}]
        if not relevant:
            return
        self._pending_visual_sensor_settle = True
        self._last_visual_sensor_settle = {'required': True, 'status': 'pending', 'reason': str(reason), 'mutation_count': len(relevant), 'states': sorted({str(item.get('state') or '') for item in relevant})}

    def _settle_pending_visual_sensor(self, pose: Mapping[str, Any]) -> dict[str, Any]:
        if not self._pending_visual_sensor_settle:
            return dict(self._last_visual_sensor_settle)
        self._pending_visual_sensor_settle = False
        diagnostic = dict(self._last_visual_sensor_settle)
        renderer = self.renderer
        render = getattr(renderer, 'render_at_pose', None) if renderer is not None else None
        if not callable(render):
            diagnostic['status'] = 'renderer_unavailable'
            self._last_visual_sensor_settle = diagnostic
            return dict(diagnostic)
        try:
            render(pose)
            self._sync_renderer_for_observation_pose(pose)
            diagnostic['resynced_before_capture'] = True
        except Exception as exc:
            diagnostic['status'] = 'discard_render_failed'
            diagnostic['error'] = repr(exc)
        else:
            diagnostic['status'] = 'settled'
            diagnostic['discarded_sensor_render_count'] = 1
        self._last_visual_sensor_settle = diagnostic
        return dict(diagnostic)

    def _apply_source_object_pose_visual_sync(self, action_result: Any) -> list[dict[str, Any]]:
        if self.renderer is None or self.core is None:
            return []
        state_delta = getattr(action_result, 'state_delta', None) or {}
        if not isinstance(state_delta, Mapping):
            return []
        action = getattr(action_result, 'action', None)
        diagnostics: list[dict[str, Any]] = []
        if action == 'PICK' and hasattr(self.renderer, 'set_held_pose'):
            held_delta = state_delta.get('held_object') if isinstance(state_delta.get('held_object'), Mapping) else {}
            held_pose = held_delta.get('held_pose') if isinstance(held_delta, Mapping) else None
            object_name = _held_pose_object_name(held_pose)
            render_object_name, resolution = self._render_object_name_for_core_name(object_name, operation='held')
            if held_pose is not None and render_object_name and (str(render_object_name) != str(object_name)):
                held_payload = _held_pose_with_object_name(held_pose, str(render_object_name))
                try:
                    try:
                        diagnostic = dict(self.renderer.set_held_pose(held_payload, verify_readback=True, ego_pose=getattr(action_result, 'end_pose', None)))
                    except TypeError:
                        diagnostic = dict(self.renderer.set_held_pose(held_payload, verify_readback=True))
                    diagnostic['action'] = 'PICK'
                    diagnostic['mutation'] = 'source_object_held_pose'
                    diagnostic['core_object_name'] = object_name
                    diagnostic['object_resolution'] = to_jsonable(resolution)
                    diagnostics.append(diagnostic)
                except Exception as exc:
                    diagnostics.append({'success': False, 'reason': 'source_object_held_pose_error', 'action': 'PICK', 'core_object_name': object_name, 'render_object_name': render_object_name, 'object_resolution': to_jsonable(resolution), 'error': repr(exc)})
        if action in {'PLACE', 'RELEASE'} and hasattr(self.renderer, 'set_object_center'):
            pose_delta = state_delta.get('object_pose') if isinstance(state_delta.get('object_pose'), Mapping) else {}
            target = str(pose_delta.get('target') or '')
            render_object_name, resolution = self._render_object_name_for_core_name(target, operation='pose')
            center = pose_delta.get('after_center')
            if center is not None and render_object_name and (str(render_object_name) != target):
                try:
                    diagnostic = dict(self.renderer.set_object_center(str(render_object_name), center, verify_readback=True))
                    diagnostic['action'] = str(action)
                    diagnostic['mutation'] = 'source_object_pose'
                    diagnostic['core_object_name'] = target
                    diagnostic['object_resolution'] = to_jsonable(resolution)
                    diagnostics.append(diagnostic)
                except Exception as exc:
                    diagnostics.append({'success': False, 'reason': 'source_object_pose_error', 'action': str(action), 'core_object_name': target, 'render_object_name': render_object_name, 'object_resolution': to_jsonable(resolution), 'error': repr(exc)})
        return diagnostics

    def _source_object_id_for_core_name(self, name: str | None) -> str | None:
        if not name or self.core is None:
            return None
        obj = getattr(self.core, 'objects', {}).get(str(name))
        if obj is None:
            return None
        states = getattr(obj, 'states', {}) or {}
        if isinstance(states, Mapping):
            for key in ('source_scene_object_id', 'source_object_id'):
                value = states.get(key)
                if not value:
                    continue
                text = str(value)
                if text.startswith('spawnable:'):
                    continue
                return text
        return None

    def _render_object_name_for_core_name(self, name: str | None, *, operation: str='pose') -> tuple[str | None, dict[str, Any]]:
        core_name = None if name is None else str(name)
        operation = str(operation or 'pose')
        diagnostic: dict[str, Any] = {'core_object_name': core_name, 'render_object_name': core_name, 'resolution': 'identity', 'operation': operation}
        if not core_name:
            diagnostic['resolution'] = 'missing_core_object_name'
            return (None, diagnostic)
        renderer = self.renderer
        if renderer is None:
            diagnostic['resolution'] = 'renderer_absent'
            return (core_name, diagnostic)
        public_resolver = getattr(renderer, 'resolve_render_object_name', None)
        if callable(public_resolver):
            try:
                resolved = dict(public_resolver(core_name, operation=operation))
            except Exception as exc:
                diagnostic['renderer_resolve_error'] = repr(exc)
            else:
                diagnostic['renderer_resolution'] = to_jsonable(resolved)
                if bool(resolved.get('success')):
                    render_name = str(resolved.get('resolved_name') or core_name)
                    diagnostic['render_object_name'] = render_name
                    if render_name != core_name or resolved.get('resolution') != 'identity':
                        diagnostic['resolution'] = str(resolved.get('resolution') or 'renderer_resolved')
                    return (render_name, diagnostic)
                reason = str(resolved.get('reason') or '')
                if reason in {'aliased_object', 'spawn_logical_id_not_renderable'} or str(resolved.get('resolution') or '') == 'renderer_alias_not_syncable':
                    diagnostic['resolution'] = str(resolved.get('resolution') or 'renderer_resolution_not_syncable')
                    diagnostic['reason'] = reason or 'renderer_resolution_not_syncable'
                    return (core_name, diagnostic)
        resolver = getattr(renderer, '_action_render_name_for_object', None)
        if callable(resolver):
            try:
                render_name, aliased_from = resolver(core_name, operation=operation)
            except TypeError:
                try:
                    render_name, aliased_from = resolver(core_name)
                except Exception as exc:
                    diagnostic['renderer_alias_error'] = repr(exc)
                else:
                    if render_name:
                        render_name = str(render_name)
                        diagnostic['render_object_name'] = render_name
                        if aliased_from is not None or render_name != core_name:
                            diagnostic['resolution'] = 'renderer_alias'
                            diagnostic['aliased_from_object_name'] = str(aliased_from or core_name)
                        return (render_name, diagnostic)
            except Exception as exc:
                diagnostic['renderer_alias_error'] = repr(exc)
            else:
                if render_name:
                    render_name = str(render_name)
                    diagnostic['render_object_name'] = render_name
                    if aliased_from is not None or render_name != core_name:
                        diagnostic['resolution'] = 'renderer_alias'
                        diagnostic['aliased_from_object_name'] = str(aliased_from or core_name)
                    return (render_name, diagnostic)
        safe_name = _render_safe_name(core_name)
        aliases = dict(getattr(renderer, '_object_aliases', {}) or {})
        alias_target = aliases.get(safe_name)
        if alias_target:
            alias_metadata = dict((getattr(renderer, '_render_alias_metadata', {}) or {}).get(safe_name) or {})
            alias_chain_info: dict[str, Any] = {'alias_chain': [safe_name, str(alias_target)], 'resolved_safe_name': str(alias_target), 'alias_metadata_chain': [alias_metadata], 'alias_cycle': False}
            chain_resolver = getattr(renderer, '_resolve_alias_chain', None)
            if callable(chain_resolver):
                try:
                    alias_chain_info = dict(chain_resolver(safe_name))
                except Exception as exc:
                    diagnostic['renderer_alias_chain_error'] = repr(exc)
            denial = None
            denial_resolver = getattr(renderer, '_alias_chain_denial', None)
            if callable(denial_resolver):
                try:
                    denial = denial_resolver(alias_chain_info, operation)
                except Exception as exc:
                    diagnostic['renderer_alias_denial_error'] = repr(exc)
            if denial is None:
                alias_kind = str(alias_metadata.get('alias_kind') or 'source_scene_object')
                allowed_key = {'held': 'held_sync_allowed', 'pose': 'pose_sync_allowed', 'visual_state': 'visual_state_sync_allowed'}.get(operation, 'pose_sync_allowed')
                allowed_default = alias_kind != 'derived_parent' or operation == 'visual_state'
                if not bool(alias_metadata.get(allowed_key, allowed_default)):
                    denial = {'alias': safe_name, 'target': str(alias_target), 'alias_kind': alias_kind, 'metadata': alias_metadata}
            if denial is not None:
                diagnostic['resolution'] = 'renderer_alias_not_syncable'
                diagnostic['reason'] = 'aliased_object'
                diagnostic['alias_kind'] = str(denial.get('alias_kind') or alias_metadata.get('alias_kind') or '')
                diagnostic['target_safe_name'] = str(denial.get('target') or alias_target)
                diagnostic['alias_chain'] = to_jsonable(alias_chain_info.get('alias_chain') or [])
                diagnostic['resolved_target_safe_name'] = str(alias_chain_info.get('resolved_safe_name') or alias_target)
                diagnostic['alias_metadata'] = to_jsonable(alias_metadata)
                diagnostic['denied_alias'] = to_jsonable(denial)
                return (core_name, diagnostic)
            if bool(alias_chain_info.get('alias_cycle')):
                diagnostic['resolution'] = 'renderer_alias_cycle'
                diagnostic['reason'] = 'alias_cycle'
                diagnostic['target_safe_name'] = str(alias_target)
                diagnostic['alias_chain'] = to_jsonable(alias_chain_info.get('alias_chain') or [])
                return (core_name, diagnostic)
            render_name = str(alias_chain_info.get('resolved_safe_name') or alias_target)
            diagnostic['render_object_name'] = render_name
            diagnostic['resolution'] = 'renderer_alias'
            diagnostic['aliased_from_object_name'] = core_name
            diagnostic['target_safe_name'] = str(alias_target)
            diagnostic['resolved_target_safe_name'] = render_name
            diagnostic['alias_chain'] = to_jsonable(alias_chain_info.get('alias_chain') or [safe_name, str(alias_target)])
            if alias_metadata:
                diagnostic['alias_metadata'] = to_jsonable(alias_metadata)
            return (render_name, diagnostic)
        source_object_id = self._source_object_id_for_core_name(core_name)
        if source_object_id and self._renderer_knows_object_name(source_object_id):
            diagnostic['render_object_name'] = source_object_id
            diagnostic['resolution'] = 'source_object_id'
            diagnostic['source_object_id'] = source_object_id
            return (source_object_id, diagnostic)
        if source_object_id:
            diagnostic['source_object_id'] = source_object_id
            diagnostic['source_object_id_resolution'] = 'not_known_to_renderer'
        return (core_name, diagnostic)

    def _renderer_knows_object_name(self, name: str | None) -> bool:
        if not name or self.renderer is None:
            return False
        text = str(name)
        safe_name = _render_safe_name(text)
        object_by_name = getattr(self.renderer, '_object_by_name', None)
        if callable(object_by_name):
            for candidate in (text, safe_name):
                try:
                    if object_by_name(candidate) is not None:
                        return True
                except Exception:
                    continue
        aliases = dict(getattr(self.renderer, '_object_aliases', {}) or {})
        alias_keys = {str(key) for key in aliases.keys()}
        alias_values = {str(value) for value in aliases.values()}
        if safe_name in alias_keys or safe_name in alias_values:
            return True
        raw_names = dict(getattr(self.renderer, '_raw_name_to_safe_name', {}) or {})
        return text in raw_names or safe_name in {str(value) for value in raw_names.values()}

    def _update_core_object_pose_from_visual_mutation(self, mutation: Mapping[str, Any]) -> None:
        if self.core is None or not isinstance(mutation, Mapping) or (not bool(mutation.get('success'))):
            return
        name = mutation.get('core_object_name') or mutation.get('object_name')
        if not name or str(name) not in getattr(self.core, 'objects', {}):
            return
        obj = self.core.objects[str(name)]
        _update_core_object_from_pose_mutation(obj, mutation)

    def _pickup_visual_mutation_failure(self, visual_mutations: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        for mutation in visual_mutations or ():
            if not isinstance(mutation, Mapping):
                continue
            if mutation.get('action') != 'PICK' or mutation.get('mutation') != 'held_pose':
                continue
            held_pose = mutation.get('held_pose') if isinstance(mutation.get('held_pose'), Mapping) else {}
            target_name = str(mutation.get('object_name') or held_pose.get('object_name') or '')
            gripper_visual = mutation.get('gripper_visual')
            if isinstance(gripper_visual, Mapping) and (not bool(gripper_visual.get('success', True))):
                reason = str(gripper_visual.get('reason') or 'gripper_visual_sync_failed')
                message_target = target_name or '<unknown>'
                return {'success': False, 'error_type': 'gripper_visual_sync_failed', 'error_message': "Renderer did not synchronize gripper visual for '{}': {}.".format(message_target, reason), 'sync': to_jsonable(dict(mutation)), 'visibility': {}, 'source': 'apply_action_result_visual_mutation'}
            if bool(mutation.get('success')):
                continue
            reason = str(mutation.get('reason') or 'set_held_pose_failed')
            message_target = target_name or '<unknown>'
            return {'success': False, 'error_type': 'held_pose_visual_sync_failed', 'error_message': "Renderer did not move held object '{}': {}.".format(message_target, reason), 'sync': to_jsonable(dict(mutation)), 'visibility': {}, 'source': 'apply_action_result_visual_mutation'}
        return None

    def _pre_action_visibility_failure(self, tool_name: str, target_name: Optional[str], kwargs: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        mode = str(self.action_visibility_mode or 'disabled')
        if mode == 'disabled' or tool_name not in _VISIBILITY_GATED_TOOLS:
            return None
        if self.core is None:
            return {'error_type': 'visibility_gate_unavailable', 'error_message': 'Action visibility gate cannot run before reset.', 'tool_name': tool_name, 'visibility_mode': mode, 'required_visible_targets': [], 'missing_visible_targets': []}
        required_targets = self._visibility_required_targets(tool_name, target_name, kwargs)
        required_targets = [str(name) for name in required_targets if name]
        if tool_name == 'soak_under' and mode in {'rendered_required', 'line_of_sight_required'} and (getattr(getattr(self.core, 'ego_state', None), 'held_object_name', None) is None):
            return {'error_type': 'staged_wash_requires_held_object', 'error_message': 'Strict visibility mode requires holding the object before soak_under.', 'tool_name': tool_name, 'visibility_mode': mode, 'required_visible_targets': required_targets, 'missing_visible_targets': [], 'visible_object_ids': [], 'rendered_visibility_status': self._last_render_visibility_status}
        if not required_targets:
            return None
        query = self._action_visibility_query(mode)
        if not bool(query.get('available')):
            error_type = str(query.get('error_type') or 'rendered_visibility_unavailable')
            return {'error_type': error_type, 'error_message': str(query.get('error_message') or 'Rendered visibility is unavailable.'), 'tool_name': tool_name, 'visibility_mode': str(query.get('visibility_mode') or mode), 'required_visible_targets': required_targets, 'missing_visible_targets': required_targets, 'visible_object_ids': list(query.get('visible_object_ids') or []), 'rendered_visibility_status': str(query.get('rendered_visibility_status') or self._last_render_visibility_status), 'rendered_visibility_error': query.get('rendered_visibility_error'), 'line_of_sight_status': query.get('line_of_sight_status'), 'line_of_sight_error': query.get('line_of_sight_error'), 'visibility_diagnostics': to_jsonable(query.get('visibility_diagnostics') or {}), 'min_pixels': int(self.action_visibility_min_pixels)}
        visible_ids = [str(name) for name in query.get('visible_object_ids') or []]
        missing = [name for name in required_targets if not (self._target_visible_by_nonspatial_alias(name, visible_ids) if tool_name == 'pick_up' and name in self._inside_parent_ledger else self._target_visible_or_equivalent(name, visible_ids))]
        if not missing:
            return None
        if tool_name == 'pick_up' and len(missing) == 1 and (self._private_inside_parent_for_interaction(missing[0], visible_ids) is not None):
            return None
        matches = {name: self._target_visibility_matches(name, visible_ids) for name in required_targets}
        missing_diagnostics = {name: self._visibility_diagnostics_for_target(name, query.get('visibility_diagnostics') or {}) for name in missing}
        error_type = 'target_not_visible'
        if str(query.get('visibility_mode') or mode) == 'symbolic_line_of_sight':
            blocked_reasons = {str(record.get('reason') or '') for records in missing_diagnostics.values() for record in records.values() if isinstance(record, Mapping)}
            if any((reason.startswith('line_of_sight_') for reason in blocked_reasons)):
                error_type = 'target_occluded_by_occupancy'
        return {'error_type': error_type, 'error_message': 'Action target is not visible or is occluded: {}.'.format(', '.join(missing)), 'tool_name': tool_name, 'visibility_mode': str(query.get('visibility_mode') or mode), 'required_visible_targets': required_targets, 'missing_visible_targets': missing, 'visible_object_ids': visible_ids, 'visibility_matches': matches, 'visibility_diagnostics': to_jsonable(missing_diagnostics), 'rendered_visibility_status': str(query.get('rendered_visibility_status') or self._last_render_visibility_status), 'rendered_visibility_error': query.get('rendered_visibility_error'), 'line_of_sight_status': query.get('line_of_sight_status'), 'line_of_sight_error': query.get('line_of_sight_error'), 'min_pixels': int(self.action_visibility_min_pixels)}

    def _visibility_required_targets(self, tool_name: str, target_name: Optional[str], kwargs: Mapping[str, Any]) -> list[str]:
        if tool_name == 'fill':
            source = kwargs.get('source') if isinstance(kwargs, Mapping) else None
            return [str(source)] if source else []
        if tool_name == 'drop':
            return []
        return [str(target_name)] if target_name else []

    def _action_visibility_query(self, mode: str) -> dict[str, Any]:
        pose = self.core.ego_state.pose if self.core is not None else None
        if mode == 'line_of_sight_required':
            return self._line_of_sight_visibility_query(pose)
        if mode in {'rendered_required', 'rendered_or_symbolic'}:
            visible = self._rendered_visible_object_names_at_pose(pose, min_pixels=self.action_visibility_min_pixels)
            if visible is not None:
                return {'available': True, 'visibility_mode': self._last_visibility_mode, 'visible_object_ids': list(visible), 'rendered_visibility_status': self._last_render_visibility_status, 'rendered_visibility_error': self._last_render_visibility_error}
            if mode == 'rendered_required':
                status = str(self._last_render_visibility_status or 'rendered_visibility_unavailable')
                return {'available': False, 'visibility_mode': 'rendered_required', 'visible_object_ids': [], 'rendered_visibility_status': status, 'rendered_visibility_error': self._last_render_visibility_error, 'error_type': self._rendered_visibility_error_type(status), 'error_message': 'Rendered visibility is required but unavailable: {}.'.format(status)}
        visible = self._symbolic_visible_object_names_at_pose(pose)
        self._last_visibility_mode = 'symbolic_fov'
        return {'available': True, 'visibility_mode': 'symbolic_fov', 'visible_object_ids': list(visible), 'rendered_visibility_status': self._last_render_visibility_status, 'rendered_visibility_error': self._last_render_visibility_error}

    def _line_of_sight_visibility_query(self, pose: Any) -> dict[str, Any]:
        occupancy = self._ensure_visibility_occupancy()
        if occupancy is None:
            return {'available': False, 'visibility_mode': 'symbolic_line_of_sight', 'visible_object_ids': [], 'rendered_visibility_status': self._last_render_visibility_status, 'rendered_visibility_error': self._last_render_visibility_error, 'line_of_sight_status': 'occupancy_unavailable', 'line_of_sight_error': 'Occupancy map is unavailable.', 'error_type': 'line_of_sight_visibility_unavailable', 'error_message': 'Line-of-sight visibility is required but occupancy is unavailable.', 'visibility_diagnostics': {}}
        pose_dict = self._pose_to_dict(pose)
        visible: list[str] = []
        diagnostics: dict[str, dict[str, Any]] = {}
        for name, obj in self.core.objects.items():
            fov_visible = self._object_visible_from_pose(obj, pose_dict)
            line = self._object_line_of_sight_from_pose(obj, pose_dict, occupancy) if fov_visible else {'clear': False, 'reason': 'outside_symbolic_fov'}
            record = dict(line)
            record['fov_visible'] = bool(fov_visible)
            diagnostics[str(name)] = record
            if fov_visible and bool(line.get('clear')):
                visible.append(str(name))
        self._last_visibility_mode = 'symbolic_line_of_sight'
        return {'available': True, 'visibility_mode': 'symbolic_line_of_sight', 'visible_object_ids': sorted(visible), 'rendered_visibility_status': self._last_render_visibility_status, 'rendered_visibility_error': self._last_render_visibility_error, 'line_of_sight_status': 'available', 'line_of_sight_error': None, 'visibility_diagnostics': diagnostics}

    def _ensure_visibility_occupancy(self) -> Any:
        if self.core is None:
            return None
        occupancy = getattr(self.core, 'occupancy', None)
        if occupancy is not None:
            return occupancy
        try:
            self.core.build_occupancy(trace_id='line_of_sight_visibility_occupancy')
        except Exception:
            return None
        return getattr(self.core, 'occupancy', None)

    def _object_line_of_sight_from_pose(self, obj: Any, pose_dict: Mapping[str, Any], occupancy: Any) -> dict[str, Any]:
        target_points = self._line_of_sight_target_points(obj, pose_dict)
        results: list[dict[str, Any]] = []
        for point in target_points:
            result = self._object_line_of_sight_to_point_from_pose(obj, pose_dict, occupancy, point['xyz'], target_point_label=str(point['label']))
            results.append(result)
            if bool(result.get('clear')):
                result['target_point_candidates'] = [str(item['label']) for item in target_points]
                result['selected_target_point_label'] = str(point['label'])
                result['target_point_results'] = self._compact_line_of_sight_point_results(results)
                return result
        if not results:
            return {'clear': False, 'reason': 'line_of_sight_invalid_geometry'}
        best_failure = min(results, key=lambda item: (len(item.get('blocked_cells') or []), len(item.get('out_of_bounds_cells') or []), int(item.get('checked_cells') or 0)))
        payload = dict(best_failure)
        payload['target_point_candidates'] = [str(item['label']) for item in target_points]
        payload['selected_target_point_label'] = best_failure.get('target_point_label')
        payload['target_point_results'] = self._compact_line_of_sight_point_results(results)
        return payload

    def _line_of_sight_target_points(self, obj: Any, pose_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
        try:
            cx, cy, cz = (float(obj.center[0]), float(obj.center[1]), float(obj.center[2]))
            ex, ey, ez = (max(0.0, float(obj.extent[0])), max(0.0, float(obj.extent[1])), max(0.0, float(obj.extent[2])))
        except Exception:
            return []
        z_top = cz + 0.5 * ez
        z_top_visible = z_top - min(0.03, ez * 0.15)
        points: list[tuple[str, tuple[float, float, float]]] = [('center', (cx, cy, self._line_of_sight_target_height_m(obj))), ('top_center', (cx, cy, z_top_visible))]
        try:
            sx, sy = (float(pose_dict['x']), float(pose_dict['y']))
            vx, vy = (sx - cx, sy - cy)
            norm = math.hypot(vx, vy)
            if norm > 1e-06:
                ux, uy = (vx / norm, vy / norm)
                front_x = cx + ux * 0.5 * ex
                front_y = cy + uy * 0.5 * ey
                points.extend([('front_center', (front_x, front_y, self._line_of_sight_target_height_m(obj))), ('front_top', (front_x, front_y, z_top_visible))])
        except Exception:
            pass
        try:
            for face in NavigationTarget.from_aabb_object(obj).interaction_faces:
                points.append(('interaction_face_{}'.format(face.source), tuple((float(v) for v in face.center))))
        except Exception:
            pass
        out: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int]] = set()
        for label, xyz in points:
            key = (round(float(xyz[0]) * 1000), round(float(xyz[1]) * 1000), round(float(xyz[2]) * 1000))
            if key in seen:
                continue
            seen.add(key)
            out.append({'label': label, 'xyz': [float(xyz[0]), float(xyz[1]), float(xyz[2])]})
        return out

    def _compact_line_of_sight_point_results(self, results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for result in results:
            compact.append({'target_point_label': result.get('target_point_label'), 'clear': bool(result.get('clear')), 'reason': result.get('reason'), 'blocked_objects': list(result.get('blocked_objects') or [])[:4], 'blocked_cells': list(result.get('blocked_cells') or [])[:3], 'out_of_bounds_cells': list(result.get('out_of_bounds_cells') or [])[:3]})
        return compact

    def _object_line_of_sight_to_point_from_pose(self, obj: Any, pose_dict: Mapping[str, Any], occupancy: Any, target_point: Sequence[float], *, target_point_label: str) -> dict[str, Any]:
        try:
            start_xy = (float(pose_dict['x']), float(pose_dict['y']))
            target_xy = (float(target_point[0]), float(target_point[1]))
        except Exception as exc:
            return {'clear': False, 'reason': 'line_of_sight_invalid_geometry', 'error': repr(exc)}
        try:
            start_cell = occupancy.world_to_cell(start_xy)
            target_cell = occupancy.world_to_cell(target_xy)
            cells = cells_on_segment(start_cell, target_cell)
        except Exception as exc:
            return {'clear': False, 'reason': 'line_of_sight_cell_trace_error', 'error': repr(exc)}
        target_clearance_m = self._line_of_sight_target_clearance_m(obj)
        start_height_m = self._line_of_sight_pose_height_m(pose_dict)
        try:
            target_height_m = float(target_point[2])
        except Exception:
            target_height_m = self._line_of_sight_target_height_m(obj)
        checked_cells = 0
        blocked_cells: list[list[int]] = []
        blocked_cell_world_xy: list[list[float]] = []
        blocked_objects: list[dict[str, Any]] = []
        height_exempted_cells: list[dict[str, Any]] = []
        height_exempted_objects: list[dict[str, Any]] = []
        support_surface_exempted_objects: list[dict[str, Any]] = []
        out_of_bounds_cells: list[list[int]] = []
        for cell in cells:
            clearance_region = self._line_of_sight_cell_clearance_region(cell, occupancy, start_xy=start_xy, target_xy=target_xy, target_clearance_m=target_clearance_m)
            if not occupancy.in_bounds_cell(cell):
                checked_cells += 1
                out_of_bounds_cells.append([int(cell[0]), int(cell[1])])
                if len(out_of_bounds_cells) >= 3:
                    break
                continue
            structural_blocked = self._occupancy_cell_blocked_by_traversability(occupancy, cell)
            if occupancy.is_cell_free(cell, eroded=False) and (not structural_blocked):
                if clearance_region:
                    continue
                checked_cells += 1
                continue
            checked_cells += 1
            cell_world_xy = self._occupancy_cell_world_xy(occupancy, cell)
            cell_blockers = self._line_of_sight_blockers_for_cell(occupancy, cell)
            if clearance_region == 'start' and self._line_of_sight_start_clearance_exempts_blockers(cell_blockers):
                exempted = [dict(blocker, height_exempt_reason='start_clearance') for blocker in cell_blockers]
                height_exempted_objects.extend(exempted)
                if len(height_exempted_cells) < 12:
                    height_exempted_cells.append({'cell': [int(cell[0]), int(cell[1])], 'world_xy': cell_world_xy, 'line_height_m': None, 'height_exempted_objects': exempted, 'support_surface_exempted_objects': []})
                continue
            if clearance_region == 'target' and self._line_of_sight_target_cell_exempts_blockers(cell_blockers, obj):
                exempted = [dict(blocker, height_exempt_reason='target_object_footprint') for blocker in cell_blockers]
                height_exempted_objects.extend(exempted)
                if len(height_exempted_cells) < 12:
                    height_exempted_cells.append({'cell': [int(cell[0]), int(cell[1])], 'world_xy': cell_world_xy, 'line_height_m': None, 'height_exempted_objects': exempted, 'support_surface_exempted_objects': []})
                continue
            height_filter = self._height_filter_line_of_sight_blockers(cell_blockers, cell_world_xy, start_xy=start_xy, target_xy=target_xy, start_height_m=start_height_m, target_height_m=target_height_m, target_obj=obj)
            blocking_objects = list(height_filter.get('blocking_objects') or [])
            exempted_objects = list(height_filter.get('height_exempted_objects') or [])
            support_exempted = list(height_filter.get('support_surface_exempted_objects') or [])
            if blocking_objects or not exempted_objects:
                blocked_cells.append([int(cell[0]), int(cell[1])])
                blocked_cell_world_xy.append(cell_world_xy)
                blocked_objects.extend(blocking_objects or cell_blockers)
            elif len(height_exempted_cells) < 12:
                height_exempted_cells.append({'cell': [int(cell[0]), int(cell[1])], 'world_xy': cell_world_xy, 'line_height_m': height_filter.get('line_height_m'), 'height_exempted_objects': exempted_objects, 'support_surface_exempted_objects': support_exempted})
            height_exempted_objects.extend(exempted_objects)
            support_surface_exempted_objects.extend(support_exempted)
            if len(blocked_cells) >= 3:
                break
        payload: dict[str, Any] = {'clear': not blocked_cells and (not out_of_bounds_cells), 'reason': 'line_of_sight_clear', 'start_cell': [int(start_cell[0]), int(start_cell[1])], 'target_cell': [int(target_cell[0]), int(target_cell[1])], 'cell_count': int(len(cells)), 'checked_cells': int(checked_cells), 'target_clearance_m': float(target_clearance_m), 'start_height_m': float(start_height_m), 'target_height_m': float(target_height_m), 'target_point_label': str(target_point_label), 'target_point_xyz': [float(target_point[0]), float(target_point[1]), float(target_height_m)]}
        deduped_exempted = self._dedupe_line_of_sight_blockers(height_exempted_objects)
        if deduped_exempted:
            payload['height_exempted_objects'] = deduped_exempted
        if height_exempted_cells:
            payload['height_exempted_cells'] = height_exempted_cells
        deduped_support_exempted = self._dedupe_line_of_sight_blockers(support_surface_exempted_objects)
        if deduped_support_exempted:
            payload['support_surface_exempted_objects'] = deduped_support_exempted
        if blocked_cells:
            payload.update({'clear': False, 'reason': 'line_of_sight_blocked', 'blocked_cells': blocked_cells, 'blocked_cell_world_xy': blocked_cell_world_xy, 'blocked_objects': self._dedupe_line_of_sight_blockers(blocked_objects)})
        elif out_of_bounds_cells:
            payload.update({'clear': False, 'reason': 'line_of_sight_out_of_bounds', 'out_of_bounds_cells': out_of_bounds_cells})
        return payload

    def _occupancy_cell_world_xy(self, occupancy: Any, cell: Sequence[int]) -> list[float]:
        try:
            xy = occupancy.cell_to_world(cell)
            return [float(xy[0]), float(xy[1])]
        except Exception:
            return []

    def _line_of_sight_blockers_for_cell(self, occupancy: Any, cell: Sequence[int]) -> list[dict[str, Any]]:
        try:
            row, col = (int(cell[0]), int(cell[1]))
        except Exception:
            return []
        metadata = getattr(occupancy, 'metadata', {}) or {}
        blockers: list[dict[str, Any]] = []
        for entry in metadata.get('rasterized_objects') or []:
            if not isinstance(entry, Mapping):
                continue
            bounds = entry.get('cell_bounds')
            if not isinstance(bounds, Sequence) or len(bounds) != 4:
                continue
            try:
                r0, c0, r1, c1 = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
            except Exception:
                continue
            if r0 <= row <= r1 and c0 <= col <= c1:
                blockers.append({'name': str(entry.get('name') or ''), 'category': str(entry.get('category') or ''), 'cell_bounds': [r0, c0, r1, c1]})
        if self._occupancy_cell_blocked_by_traversability(occupancy, cell, require_explicit=bool(blockers)):
            blockers.append({'name': 'scene_traversability', 'category': 'wall_or_static_obstacle'})
        return blockers

    def _dedupe_line_of_sight_blockers(self, blockers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for blocker in blockers:
            name = str(blocker.get('name') or '')
            category = str(blocker.get('category') or '')
            key = (name, category)
            if key in seen:
                continue
            seen.add(key)
            record = {'name': name, 'category': category}
            if blocker.get('cell_bounds') is not None:
                record['cell_bounds'] = list(blocker.get('cell_bounds') or [])
            if blocker.get('z_min_m') is not None:
                record['z_min_m'] = float(blocker.get('z_min_m') or 0.0)
            if blocker.get('z_max_m') is not None:
                record['z_max_m'] = float(blocker.get('z_max_m') or 0.0)
            if blocker.get('line_height_m') is not None:
                record['line_height_m'] = float(blocker.get('line_height_m') or 0.0)
            if blocker.get('height_margin_m') is not None:
                record['height_margin_m'] = float(blocker.get('height_margin_m') or 0.0)
            if blocker.get('height_exempt_reason') is not None:
                record['height_exempt_reason'] = str(blocker.get('height_exempt_reason') or '')
            deduped.append(record)
        return deduped

    def _height_filter_line_of_sight_blockers(self, blockers: Sequence[Mapping[str, Any]], cell_world_xy: Sequence[float], *, start_xy: Sequence[float], target_xy: Sequence[float], start_height_m: float, target_height_m: float, target_obj: Any | None=None) -> dict[str, Any]:
        line_height_m = self._line_of_sight_height_at_xy(cell_world_xy, start_xy=start_xy, target_xy=target_xy, start_height_m=start_height_m, target_height_m=target_height_m)
        blocking_objects: list[dict[str, Any]] = []
        height_exempted_objects: list[dict[str, Any]] = []
        support_surface_exempted_objects: list[dict[str, Any]] = []
        for blocker in blockers:
            blocker_record = self._line_of_sight_blocker_geometry(blocker)
            enriched = dict(blocker)
            enriched.update(blocker_record)
            enriched['line_height_m'] = float(line_height_m)
            if target_obj is not None and self._line_of_sight_blocker_matches_object(enriched, target_obj):
                enriched['height_exempt_reason'] = 'target_object_footprint'
                height_exempted_objects.append(enriched)
            elif target_obj is not None and self._line_of_sight_support_surface_exempt(enriched, target_obj, line_height_m=line_height_m, target_height_m=target_height_m):
                enriched['height_exempt_reason'] = 'target_support_surface'
                enriched['support_surface_exempted'] = True
                support_surface_exempted_objects.append(enriched)
                height_exempted_objects.append(enriched)
            else:
                vertical_exempt_reason = self._line_of_sight_blocker_vertical_exempt_reason(enriched, line_height_m)
                if vertical_exempt_reason:
                    enriched['height_exempt_reason'] = vertical_exempt_reason
                    try:
                        if vertical_exempt_reason == 'blocker_below_sightline':
                            enriched['height_margin_m'] = float(line_height_m) - float(enriched['z_max_m'])
                        else:
                            enriched['height_margin_m'] = float(enriched['z_min_m']) - float(line_height_m)
                    except Exception:
                        pass
                    height_exempted_objects.append(enriched)
                else:
                    blocking_objects.append(enriched)
        return {'line_height_m': float(line_height_m), 'blocking_objects': blocking_objects, 'height_exempted_objects': height_exempted_objects, 'support_surface_exempted_objects': support_surface_exempted_objects}

    def _line_of_sight_support_surface_exempt(self, blocker: Mapping[str, Any], target_obj: Any, *, line_height_m: float, target_height_m: float) -> bool:
        category = str(blocker.get('category') or '').lower()
        if category not in _NAVIGATION_SUPPORT_CATEGORIES:
            return False
        blocker_obj = self._object_from_line_of_sight_blocker(blocker)
        if blocker_obj is None:
            return False
        try:
            target_bottom = float(target_obj.center[2]) - 0.5 * float(target_obj.extent[2])
            support_top = float(blocker_obj.center[2]) + 0.5 * float(blocker_obj.extent[2])
        except Exception:
            return False
        if target_bottom < support_top - 0.08:
            return False
        if abs(target_bottom - support_top) > _NAVIGATION_SUPPORT_VERTICAL_TOLERANCE_M:
            return False
        if self._xy_aabb_distance(target_obj, blocker_obj) > _NAVIGATION_SUPPORT_HORIZONTAL_TOLERANCE_M:
            return False
        if float(target_height_m) + 0.08 < support_top:
            return False
        return float(line_height_m) + 0.05 >= support_top

    def _line_of_sight_pose_height_m(self, pose_dict: Mapping[str, Any]) -> float:
        try:
            return float(pose_dict.get('z', _LINE_OF_SIGHT_CAMERA_HEIGHT_FALLBACK_M))
        except Exception:
            return _LINE_OF_SIGHT_CAMERA_HEIGHT_FALLBACK_M

    def _line_of_sight_target_height_m(self, obj: Any) -> float:
        try:
            center = getattr(obj, 'center', (0.0, 0.0, 0.0))
            extent = getattr(obj, 'extent', (0.0, 0.0, 0.0))
            center_z = float(center[2])
            extent_z = max(0.0, float(extent[2]))
        except Exception:
            return 0.0
        category = str(getattr(obj, 'category', '') or '').lower()
        abilities = {str(name).lower() for name in getattr(obj, 'abilities', ()) or ()}
        if category in {'sink', 'sink_basin', 'faucet'} or 'water_source' in abilities:
            return center_z + min(0.2, extent_z * 0.25)
        return center_z

    def _line_of_sight_height_at_xy(self, cell_world_xy: Sequence[float], *, start_xy: Sequence[float], target_xy: Sequence[float], start_height_m: float, target_height_m: float) -> float:
        try:
            total = math.hypot(float(target_xy[0]) - float(start_xy[0]), float(target_xy[1]) - float(start_xy[1]))
            if total <= 1e-06:
                return float(target_height_m)
            current = math.hypot(float(cell_world_xy[0]) - float(start_xy[0]), float(cell_world_xy[1]) - float(start_xy[1]))
            t = max(0.0, min(1.0, current / total))
            return float(start_height_m) + (float(target_height_m) - float(start_height_m)) * t
        except Exception:
            return min(float(start_height_m), float(target_height_m))

    def _line_of_sight_blocker_geometry(self, blocker: Mapping[str, Any]) -> dict[str, Any]:
        obj = self._object_from_line_of_sight_blocker(blocker)
        if obj is None:
            return {}
        try:
            center = getattr(obj, 'center', (0.0, 0.0, 0.0))
            extent = getattr(obj, 'extent', (0.0, 0.0, 0.0))
            center_z = float(center[2])
            extent_z = max(0.0, float(extent[2]))
            return {'z_min_m': center_z - 0.5 * extent_z, 'z_max_m': center_z + 0.5 * extent_z, 'height_m': extent_z}
        except Exception:
            return {}

    def _object_from_line_of_sight_blocker(self, blocker: Mapping[str, Any]) -> Any | None:
        name = str(blocker.get('name') or '')
        if self.core is None:
            return None
        objects = getattr(self.core, 'objects', {}) or {}
        obj = objects.get(name)
        if obj is not None:
            return obj
        safe_name = _render_safe_name(name)
        for candidate_name, candidate in objects.items():
            if _render_safe_name(str(candidate_name)) == safe_name:
                return candidate
        return None

    def _line_of_sight_blocker_height_exempt(self, blocker: Mapping[str, Any], line_height_m: float) -> bool:
        return self._line_of_sight_blocker_vertical_exempt_reason(blocker, line_height_m) is not None

    def _line_of_sight_blocker_vertical_exempt_reason(self, blocker: Mapping[str, Any], line_height_m: float) -> str | None:
        if self._line_of_sight_blocker_is_structural(blocker):
            return None
        if blocker.get('z_min_m') is None or blocker.get('z_max_m') is None:
            return None
        try:
            z_min = float(blocker['z_min_m'])
            z_max = float(blocker['z_max_m'])
        except Exception:
            return None
        line_height = float(line_height_m)
        if z_min - _LINE_OF_SIGHT_HEIGHT_CLEARANCE_M > line_height:
            return 'blocker_above_sightline'
        if z_max + _LINE_OF_SIGHT_HEIGHT_CLEARANCE_M < line_height:
            return 'blocker_below_sightline'
        return None

    def _line_of_sight_blocker_matches_object(self, blocker: Mapping[str, Any], obj: Any) -> bool:
        blocker_obj = self._object_from_line_of_sight_blocker(blocker)
        if blocker_obj is obj:
            return True
        blocker_name = str(blocker.get('name') or '')
        obj_name = str(getattr(obj, 'name', '') or '')
        if blocker_name and obj_name and (_render_safe_name(blocker_name) == _render_safe_name(obj_name)):
            return True
        if blocker_name and obj_name:
            equivalents = self._visibility_equivalent_names(obj_name)
            normalized = {_render_safe_name(name) for name in equivalents}
            if blocker_name in equivalents or _render_safe_name(blocker_name) in normalized:
                return True
        return False

    def _line_of_sight_blocker_is_structural(self, blocker: Mapping[str, Any]) -> bool:
        name = str(blocker.get('name') or '').lower()
        category = str(blocker.get('category') or '').lower()
        if not name or name == 'scene_traversability':
            return True
        if category in _LINE_OF_SIGHT_STRUCTURAL_BLOCKER_CATEGORIES:
            return True
        return any((keyword in category or keyword in name for keyword in _LINE_OF_SIGHT_STRUCTURAL_BLOCKER_KEYWORDS))

    def _line_of_sight_start_clearance_exempts_blockers(self, blockers: Sequence[Mapping[str, Any]]) -> bool:
        if not blockers:
            return True
        return not any((self._line_of_sight_blocker_is_structural(blocker) for blocker in blockers))

    def _line_of_sight_target_cell_exempts_blockers(self, blockers: Sequence[Mapping[str, Any]], target_obj: Any) -> bool:
        if not blockers:
            return False
        has_target = any((self._line_of_sight_blocker_matches_object(blocker, target_obj) for blocker in blockers))
        if not has_target:
            return False
        for blocker in blockers:
            if self._line_of_sight_blocker_matches_object(blocker, target_obj):
                continue
            if str(blocker.get('name') or '') == 'scene_traversability':
                continue
            return False
        return True

    def _occupancy_cell_blocked_by_traversability(self, occupancy: Any, cell: Sequence[int], *, require_explicit: bool=False) -> bool:
        try:
            line_of_sight_free = getattr(occupancy, 'line_of_sight_traversability_free', None)
            if line_of_sight_free is not None and occupancy.in_bounds_cell(cell):
                row, col = (int(cell[0]), int(cell[1]))
                return not bool(line_of_sight_free[row, col])
            traversability_free = getattr(occupancy, 'traversability_free', None)
            if traversability_free is not None and occupancy.in_bounds_cell(cell):
                row, col = (int(cell[0]), int(cell[1]))
                return not bool(traversability_free[row, col])
            if require_explicit:
                return False
            return bool(occupancy.in_bounds_cell(cell)) and (not bool(occupancy.is_cell_free(cell, eroded=False)))
        except Exception:
            return False

    def _line_of_sight_cell_clearance_region(self, cell: Sequence[int], occupancy: Any, *, start_xy: Sequence[float], target_xy: Sequence[float], target_clearance_m: float) -> str:
        if not occupancy.in_bounds_cell(cell):
            return ''
        cell_xy = occupancy.cell_to_world(cell)
        if math.hypot(float(cell_xy[0]) - float(start_xy[0]), float(cell_xy[1]) - float(start_xy[1])) <= _LINE_OF_SIGHT_START_CLEARANCE_M + float(getattr(occupancy, 'resolution', 0.1)):
            return 'start'
        if math.hypot(float(cell_xy[0]) - float(target_xy[0]), float(cell_xy[1]) - float(target_xy[1])) <= float(target_clearance_m) + float(getattr(occupancy, 'resolution', 0.1)):
            return 'target'
        return ''

    def _line_of_sight_cell_is_exempt(self, cell: Sequence[int], occupancy: Any, *, start_xy: Sequence[float], target_xy: Sequence[float], target_clearance_m: float) -> bool:
        return bool(self._line_of_sight_cell_clearance_region(cell, occupancy, start_xy=start_xy, target_xy=target_xy, target_clearance_m=target_clearance_m))

    def _line_of_sight_target_clearance_m(self, obj: Any) -> float:
        try:
            extent = getattr(obj, 'extent', (0.0, 0.0, 0.0))
            radius = 0.5 * math.hypot(float(extent[0]), float(extent[1]))
        except Exception:
            radius = 0.0
        abilities = set((str(name) for name in getattr(obj, 'abilities', ()) or ()))
        category = str(getattr(obj, 'category', '') or '')
        base = _LINE_OF_SIGHT_MIN_TARGET_CLEARANCE_M
        if category in {'sink', 'faucet', 'countertop', 'breakfast_table'} or abilities.intersection({'water_source', 'support_surface'}):
            base = 0.42
        return min(_LINE_OF_SIGHT_MAX_TARGET_CLEARANCE_M, max(base, radius + 0.12))

    def _visibility_diagnostics_for_target(self, target_name: str, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
        equivalents = self._visibility_equivalent_names(target_name)
        normalized_equivalents = {_render_safe_name(name) for name in equivalents}
        records: dict[str, Any] = {}
        for name, record in diagnostics.items():
            name_text = str(name)
            if name_text in equivalents or _render_safe_name(name_text) in normalized_equivalents:
                records[name_text] = record
        return records

    def _rendered_visibility_error_type(self, status: str) -> str:
        status = str(status or '')
        if status.endswith('_error') or status == 'rendered_visibility_error':
            return 'rendered_visibility_error'
        if status.endswith('_invalid') or status == 'rendered_visibility_invalid':
            return 'rendered_visibility_invalid'
        return 'rendered_visibility_unavailable'

    def _target_visible_or_equivalent(self, target_name: str, visible_ids: Sequence[str]) -> bool:
        return bool(self._target_visibility_matches(target_name, visible_ids))

    def _target_visible_by_nonspatial_alias(self, target_name: str, visible_ids: Sequence[str]) -> bool:
        target = str(target_name)
        equivalents = {target, _render_safe_name(target)}
        source_object_id = self._source_object_id_for_core_name(target)
        if source_object_id:
            equivalents.update({source_object_id, _render_safe_name(source_object_id)})
        equivalents.update(self._renderer_alias_equivalents(target))
        normalized = {_render_safe_name(name) for name in equivalents}
        return any((str(visible) in equivalents or _render_safe_name(str(visible)) in normalized for visible in visible_ids))

    def _target_visibility_matches(self, target_name: str, visible_ids: Sequence[str]) -> list[str]:
        equivalents = self._visibility_equivalent_names(target_name)
        normalized_equivalents = {_render_safe_name(name) for name in equivalents}
        matches: set[str] = set()
        for visible in visible_ids:
            visible_text = str(visible)
            if visible_text in equivalents or _render_safe_name(visible_text) in normalized_equivalents:
                matches.add(visible_text)
        return sorted(matches)

    def _visibility_equivalent_names(self, target_name: str) -> set[str]:
        target = str(target_name)
        equivalents = {target, _render_safe_name(target)}
        source_object_id = self._source_object_id_for_core_name(target)
        if source_object_id:
            equivalents.add(source_object_id)
            equivalents.add(_render_safe_name(source_object_id))
            if '::' in source_object_id:
                parent_object_id = source_object_id.split('::', 1)[0]
                equivalents.add(parent_object_id)
                equivalents.add(_render_safe_name(parent_object_id))
        equivalents.update(self._renderer_alias_equivalents(target))
        equivalents.update(self._role_source_equivalent_names(target))
        if self.core is None:
            return equivalents
        target_obj = self.core.objects.get(target)
        if target_obj is None:
            return equivalents
        target_center = getattr(target_obj, 'center', None)
        for name, obj in self.core.objects.items():
            if str(name) == target:
                continue
            if self._centers_close(target_center, getattr(obj, 'center', None)):
                equivalents.add(str(name))
                equivalents.add(_render_safe_name(str(name)))
        return equivalents

    def _renderer_alias_equivalents(self, target_name: str) -> set[str]:
        renderer = self.renderer
        if renderer is None:
            return set()
        safe_target = _render_safe_name(target_name)
        safe_equivalents = {safe_target}
        aliases = dict(getattr(renderer, '_object_aliases', {}) or {})
        changed = True
        while changed:
            changed = False
            for alias_safe, target_safe in aliases.items():
                alias_safe = str(alias_safe)
                target_safe = str(target_safe)
                if alias_safe in safe_equivalents and target_safe not in safe_equivalents:
                    safe_equivalents.add(target_safe)
                    changed = True
                if target_safe in safe_equivalents and alias_safe not in safe_equivalents:
                    safe_equivalents.add(alias_safe)
                    changed = True
        equivalents = set(safe_equivalents)
        for raw_name, safe_name in dict(getattr(renderer, '_raw_name_to_safe_name', {}) or {}).items():
            if str(safe_name) in safe_equivalents:
                equivalents.add(str(raw_name))
                equivalents.add(_render_safe_name(str(raw_name)))
        return equivalents

    def _role_source_equivalent_names(self, target_name: str) -> set[str]:
        if not isinstance(self.current_task_spec, Mapping):
            return set()
        roles = self.current_task_spec.get('roles') or {}
        target_roles = []
        for role_name, role in roles.items():
            if not isinstance(role, Mapping):
                continue
            if str(role.get('bddl_inst') or '') == str(target_name):
                target_roles.append((str(role_name), role))
        if not target_roles:
            return set()
        equivalents: set[str] = set()
        target_object_ids = {str(role.get('object_id') or '') for _role_name, role in target_roles if str(role.get('object_id') or '')}
        target_centers = [role.get('center') for _role_name, role in target_roles]
        for role in roles.values():
            if not isinstance(role, Mapping):
                continue
            bddl_inst = str(role.get('bddl_inst') or '')
            object_id = str(role.get('object_id') or '')
            if not bddl_inst:
                continue
            if object_id and any((object_id == target_id or object_id.startswith(target_id + '::') or target_id.startswith(object_id + '::') for target_id in target_object_ids)):
                equivalents.add(bddl_inst)
                equivalents.add(_render_safe_name(bddl_inst))
                equivalents.add(object_id)
                equivalents.add(_render_safe_name(object_id))
            elif any((self._centers_close(center, role.get('center')) for center in target_centers)):
                equivalents.add(bddl_inst)
                equivalents.add(_render_safe_name(bddl_inst))
                if object_id:
                    equivalents.add(object_id)
                    equivalents.add(_render_safe_name(object_id))
        return equivalents

    def _centers_close(self, a: Any, b: Any) -> bool:
        try:
            if a is None or b is None:
                return False
            distance = math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2)
            return distance <= _VISIBILITY_EQUIVALENT_CENTER_TOLERANCE_M
        except Exception:
            return False

    def _validate_rendered_held_pose_after_pickup(self, action_result: ActionResult) -> dict[str, Any]:
        diagnostic: dict[str, Any] = {'success': False, 'error_type': 'held_pose_visual_sync_failed', 'error_message': 'Renderer did not synchronize the picked object to held pose.', 'sync': {}, 'visibility': {}, 'tolerance_m': float(self.held_pose_center_tolerance_m), 'min_pixels': int(self.held_pose_min_pixels)}
        if self.renderer is None:
            diagnostic['error_type'] = 'rendered_held_pose_unavailable'
            diagnostic['error_message'] = 'Rendered held pose is required but renderer is absent.'
            diagnostic['sync'] = {'success': False, 'reason': 'renderer_absent'}
            return diagnostic
        held_pose = self._held_pose_from_action_result(action_result)
        target_name = self._held_pose_field(held_pose, 'object_name', None)
        if not target_name:
            diagnostic['sync'] = {'success': False, 'reason': 'held_pose_missing_object_name'}
            return diagnostic
        set_held_pose = getattr(self.renderer, 'set_held_pose', None)
        if not callable(set_held_pose):
            diagnostic['error_type'] = 'rendered_held_pose_unavailable'
            diagnostic['error_message'] = 'Renderer does not expose set_held_pose().'
            diagnostic['sync'] = {'success': False, 'reason': 'renderer_set_held_pose_absent'}
            return diagnostic
        render_target_name, resolution = self._render_object_name_for_core_name(str(target_name), operation='held')
        diagnostic['object_resolution'] = to_jsonable(resolution)
        held_pose_for_renderer = held_pose
        if render_target_name and str(render_target_name) != str(target_name):
            held_pose_for_renderer = _held_pose_with_object_name(held_pose, str(render_target_name))
        pose = getattr(action_result, 'end_pose', None) or (self.core.ego_state.pose if self.core is not None else None)
        try:
            sync = set_held_pose(held_pose_for_renderer, ego_pose=pose)
        except Exception as exc:
            if isinstance(exc, TypeError):
                try:
                    sync = set_held_pose(held_pose_for_renderer)
                except Exception as fallback_exc:
                    sync = {'success': False, 'reason': 'set_held_pose_error', 'error': repr(fallback_exc)}
            else:
                sync = {'success': False, 'reason': 'set_held_pose_error', 'error': repr(exc)}
        if not isinstance(sync, Mapping):
            sync = {'success': False, 'reason': 'set_held_pose_returned_invalid_diagnostic', 'raw': repr(sync)}
        sync = dict(sync)
        sync.setdefault('core_object_name', str(target_name))
        if render_target_name:
            sync.setdefault('render_object_name', str(render_target_name))
        if held_pose_for_renderer is not held_pose:
            sync.setdefault('original_held_pose', to_jsonable(held_pose))
        diagnostic['sync'] = to_jsonable(sync)
        if not bool(sync.get('success')):
            reason = str(sync.get('reason') or 'set_held_pose_failed')
            diagnostic['error_message'] = "Renderer did not move held object '{}': {}.".format(target_name, reason)
            return diagnostic
        center_error = sync.get('center_error_m')
        try:
            center_error_f = float(center_error)
        except Exception:
            center_error_f = float('inf')
        diagnostic['center_error_m'] = center_error_f
        if center_error_f > self.held_pose_center_tolerance_m:
            diagnostic['sync']['reason'] = 'center_error_exceeds_tolerance'
            diagnostic['error_message'] = "Renderer held pose center error for '{}' exceeds tolerance: {:.3f}m > {:.3f}m.".format(target_name, center_error_f, self.held_pose_center_tolerance_m)
            return diagnostic
        visibility_held_pose = sync.get('held_pose') if isinstance(sync.get('held_pose'), Mapping) else held_pose
        visibility = self._held_pose_visibility_after_pickup(str(target_name), visibility_held_pose, pose)
        diagnostic['visibility'] = visibility
        if not bool(visibility.get('available')):
            status = str(visibility.get('rendered_visibility_status') or self._last_render_visibility_status)
            diagnostic['error_type'] = str(visibility.get('error_type') or self._rendered_visibility_error_type(status))
            diagnostic['error_message'] = str(visibility.get('error_message') or 'Rendered visibility is required after pickup but unavailable: {}.'.format(status))
            return diagnostic
        if not bool(visibility.get('matches')):
            diagnostic['error_type'] = 'held_object_not_visible_after_pickup'
            diagnostic['error_message'] = "Held object '{}' is not visible after pickup.".format(target_name)
            return diagnostic
        diagnostic['success'] = True
        diagnostic['error_type'] = None
        diagnostic['error_message'] = None
        return diagnostic

    def _held_pose_visibility_after_pickup(self, target_name: str, held_pose: Any, pose: Any) -> dict[str, Any]:
        if self.use_rendered_visibility:
            visible = self._rendered_visible_object_names_at_pose(pose, min_pixels=self.held_pose_min_pixels)
            visibility = {'available': visible is not None, 'visible_object_ids': list(visible or []), 'visibility_mode': self._last_visibility_mode, 'rendered_visibility_status': self._last_render_visibility_status, 'rendered_visibility_error': self._last_render_visibility_error, 'min_pixels': int(self.held_pose_min_pixels)}
            visibility['matches'] = self._target_visibility_matches(str(target_name), visibility['visible_object_ids'])
            if visible is not None:
                return visibility
        if str(self.action_visibility_mode or '') != 'line_of_sight_required':
            status = str(self._last_render_visibility_status or 'rendered_visibility_unavailable')
            return {'available': False, 'matches': False, 'visible_object_ids': [], 'visibility_mode': self._last_visibility_mode, 'rendered_visibility_status': status, 'rendered_visibility_error': self._last_render_visibility_error, 'min_pixels': int(self.held_pose_min_pixels), 'error_type': self._rendered_visibility_error_type(status), 'error_message': 'Rendered visibility is required after pickup but unavailable: {}.'.format(status)}
        center = self._held_pose_field(held_pose, 'center', None)
        held_visible = self._held_pose_center_visible_from_pose(center, pose)
        return {'available': True, 'matches': bool(held_visible.get('visible')), 'visible_object_ids': [str(target_name)] if bool(held_visible.get('visible')) else [], 'visibility_mode': 'symbolic_held_pose_fov', 'rendered_visibility_status': self._last_render_visibility_status, 'rendered_visibility_error': self._last_render_visibility_error, 'line_of_sight_status': 'held_pose_symbolic_fallback', 'min_pixels': int(self.held_pose_min_pixels), 'held_pose_visibility': held_visible}

    def _held_pose_center_visible_from_pose(self, center: Any, pose: Any) -> dict[str, Any]:
        try:
            center_xyz = (float(center[0]), float(center[1]), float(center[2]))
        except Exception as exc:
            return {'visible': False, 'reason': 'held_pose_center_invalid', 'error': repr(exc)}
        pose_dict = self._pose_to_dict(pose)
        dx = center_xyz[0] - pose_dict['x']
        dy = center_xyz[1] - pose_dict['y']
        dz = center_xyz[2] - pose_dict['z']
        horizontal = math.hypot(dx, dy)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > _SYMBOLIC_VISIBILITY_MAX_DISTANCE:
            return {'visible': False, 'reason': 'held_pose_too_far', 'distance_m': float(distance)}
        if horizontal <= 1e-06:
            yaw_error = 0.0
        else:
            yaw_error = abs(_normalize_angle(math.atan2(dy, dx) - pose_dict['yaw']))
        target_pitch = math.atan2(dz, horizontal) if horizontal > 1e-06 else 0.0
        pitch_error = abs(target_pitch - pose_dict['pitch'])
        visible = yaw_error <= math.radians(_SYMBOLIC_HORIZONTAL_FOV_DEGREES * 0.5) and pitch_error <= math.radians(_SYMBOLIC_VERTICAL_FOV_DEGREES * 0.5)
        return {'visible': bool(visible), 'reason': 'held_pose_in_symbolic_fov' if visible else 'held_pose_outside_symbolic_fov', 'distance_m': float(distance), 'yaw_error_rad': float(yaw_error), 'pitch_error_rad': float(pitch_error)}

    def _mark_pickup_failed_after_visual_postcondition(self, action_result: ActionResult, *, error_type: str, diagnostic: Mapping[str, Any]) -> None:
        target = str((getattr(action_result, 'args', {}) or {}).get('target') or '')
        before = ((getattr(action_result, 'state_delta', None) or {}).get('held_object') or {}).get('before') or {}
        previous_held = before.get('held_object_name')
        previous_visual = before.get('held_visual_name')
        if self.core is not None and getattr(self.core, 'ego_state', None) is not None:
            self.core.ego_state.held_object_name = previous_held
            self.core.ego_state.held_visual_name = previous_visual
            try:
                self.core._held_pose_anchor = None
            except Exception:
                pass
            try:
                self.core.build_occupancy(trace_id='visual_postcondition_failed_pickup_rollback')
            except Exception:
                pass
        renderer_rollback = self._rollback_renderer_after_failed_pickup(target)
        action_result.success = False
        action_result.postcondition = CheckResult.fail(error_type, target=target, rendered_held_pose=to_jsonable(dict(diagnostic)), rolled_back_held_object_name=previous_held, rolled_back_held_visual_name=previous_visual, renderer_rollback=to_jsonable(renderer_rollback))
        state_delta = dict(action_result.state_delta or {})
        held_delta = dict(state_delta.get('held_object') or {})
        held_delta['rolled_back_after_visual_postcondition'] = {'held_object_name': previous_held, 'held_visual_name': previous_visual}
        state_delta['held_object'] = held_delta
        state_delta['held_pose_visual_sync'] = to_jsonable(dict(diagnostic))
        state_delta['renderer_rollback_after_visual_postcondition'] = to_jsonable(renderer_rollback)
        action_result.state_delta = state_delta

    def _rollback_renderer_after_failed_pickup(self, target_name: str) -> dict[str, Any]:
        if not target_name:
            return {'success': False, 'reason': 'target_missing'}
        if self.renderer is None or not hasattr(self.renderer, 'set_object_center'):
            return {'success': False, 'reason': 'renderer_set_object_center_unavailable', 'target': target_name}
        if self.core is None or target_name not in getattr(self.core, 'objects', {}):
            return {'success': False, 'reason': 'core_object_not_found', 'target': target_name}
        obj = self.core.objects[target_name]
        try:
            center = getattr(obj, 'center')
            render_target_name, resolution = self._render_object_name_for_core_name(str(target_name), operation='pose')
            diagnostic = self.renderer.set_object_center(str(render_target_name or target_name), center)
            if isinstance(diagnostic, Mapping):
                payload = dict(diagnostic)
            else:
                payload = {'success': False, 'reason': 'invalid_renderer_rollback_diagnostic', 'raw': repr(diagnostic)}
            payload['target'] = str(target_name)
            if render_target_name:
                payload['render_target'] = str(render_target_name)
            payload['object_resolution'] = to_jsonable(resolution)
            return payload
        except Exception as exc:
            return {'success': False, 'reason': 'renderer_rollback_error', 'target': str(target_name), 'error': repr(exc)}

    def _held_pose_from_action_result(self, action_result: ActionResult) -> Any:
        state_delta = getattr(action_result, 'state_delta', None) or {}
        held_delta = state_delta.get('held_object') if isinstance(state_delta, Mapping) else None
        if isinstance(held_delta, Mapping):
            return held_delta.get('held_pose')
        return None

    def _held_pose_field(self, held_pose: Any, key: str, default: Any=None) -> Any:
        if isinstance(held_pose, Mapping):
            return held_pose.get(key, default)
        return getattr(held_pose, key, default)

    def _resolve_target(self, arguments: Mapping[str, Any], *, tool_name: str='', argument_name: str='target') -> Optional[str]:
        target_ref = arguments.get(argument_name)
        if target_ref is None and argument_name == 'target':
            target_ref = arguments.get('object_ref')
        if target_ref is None:
            return None
        result = self.object_resolver.resolve(target_ref, env=self, tool_name=tool_name, high_level_step=self.high_level_step, argument_name=argument_name, defer_log=True)
        if result.success:
            if result.binding_public_record is not None:
                self._current_tool_binding_records.append(dict(result.binding_public_record))
            if result.binding_role:
                self._current_tool_binding_roles.add(str(result.binding_role))
            self.object_resolver.log(result, env=self, tool_name=tool_name, high_level_step=self.high_level_step, argument_name=argument_name)
            self._remember_resolved_target(tool_name, target_ref, result)
            return result.selected_object
        if result.binding_public_record is not None:
            self.object_resolver.log(result, env=self, tool_name=tool_name, high_level_step=self.high_level_step, argument_name=argument_name)
            self._current_tool_binding_records.append(dict(result.binding_public_record))
            return None
        selected = self._resolve_navigation_multi_match(result, target_ref=target_ref, tool_name=tool_name)
        if selected is None:
            selected = self._resolve_physical_multi_match(result, target_ref=target_ref, tool_name=tool_name, argument_name=argument_name)
        if selected is not None:
            self.object_resolver.log(selected, env=self, tool_name=tool_name, high_level_step=self.high_level_step, argument_name=argument_name)
            self._remember_resolved_target(tool_name, target_ref, selected)
            return selected.selected_object
        self.object_resolver.log(result, env=self, tool_name=tool_name, high_level_step=self.high_level_step, argument_name=argument_name)
        target_text = str(target_ref)
        if self.object_resolver.config.allow_role_handles and target_text in self.role_map:
            return self.role_map[target_text]
        if self.object_resolver.config.allow_visible_aliases:
            resolved = self._catalog_tracker.resolve(target_text)
            if resolved != target_text:
                return resolved
        if self.object_resolver.config.allow_raw_object_ids:
            return target_text
        return None

    def prebind_synthetic_operation(self, tool_name: str, arguments: Mapping[str, Any]) -> ObjectBindingResult:
        target_ref = arguments.get('target', arguments.get('object_ref'))
        self._current_tool_binding_roles = set()
        self._current_tool_binding_records = []
        self._last_confirmation_bound_target = None
        result = self.object_resolver.resolve(target_ref, env=self, tool_name=str(tool_name), high_level_step=self.high_level_step, argument_name='target', defer_log=True)
        if not result.success:
            return result
        if result.binding_public_record is not None:
            self._current_tool_binding_records.append(dict(result.binding_public_record))
        if result.binding_role:
            self._current_tool_binding_roles.add(str(result.binding_role))
        self.object_resolver.log(result, env=self, tool_name=str(tool_name), high_level_step=self.high_level_step, argument_name='target')
        self._remember_resolved_target(str(tool_name), target_ref, result)
        return result

    def enable_locked_recovery_admission_bypass(self, tool_call: Mapping[str, Any]) -> None:
        tool_name = str(tool_call.get('tool_name') or '')
        arguments = tool_call.get('arguments')
        if not tool_name or not isinstance(arguments, Mapping):
            raise BehaviorClosedLoopToolEnvError('Locked recovery bypass requires a public ToolCall.')
        self._locked_recovery_admission_bypass = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments))}

    def suppress_completed_tool_call(self, tool_call: Mapping[str, Any]) -> None:
        tool_name = str(tool_call.get('tool_name') or '')
        arguments = tool_call.get('arguments')
        if not tool_name or not isinstance(arguments, Mapping):
            raise BehaviorClosedLoopToolEnvError('Completed-action suppression requires a public ToolCall.')
        self._suppressed_completed_tool_call = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments))}

    def reject_public_precondition_tool_call(self, tool_call: Mapping[str, Any], message: str) -> None:
        tool_name = str(tool_call.get('tool_name') or '')
        arguments = tool_call.get('arguments')
        if not tool_name or not isinstance(arguments, Mapping):
            raise BehaviorClosedLoopToolEnvError('Public-precondition rejection requires a public ToolCall.')
        self._public_precondition_rejection = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments)), 'message': str(message)}

    def mark_harness_directed_tool_call(self, tool_call: Mapping[str, Any]) -> None:
        tool_name = str(tool_call.get('tool_name') or '')
        arguments = tool_call.get('arguments')
        if not tool_name or not isinstance(arguments, Mapping):
            raise BehaviorClosedLoopToolEnvError('Harness-directed recovery requires a public ToolCall.')
        self._harness_directed_tool_call = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments))}

    def _remember_resolved_target(self, tool_name: str, target_ref: Any, result: ObjectBindingResult) -> None:
        if not result.success:
            return
        target_text = self._binding_reference_key(target_ref)
        if not target_text:
            return
        record = {'tool_name': str(tool_name), 'raw_reference': target_text, 'selected_object': str(result.selected_object), 'high_level_step': int(self.high_level_step), 'reason': str(result.reason), 'source': result.source, 'canonical_public_ref': (result.binding_public_record or {}).get('canonical_public_ref') or target_text, 'binding_role': result.binding_role}
        self._last_resolved_targets['any'] = dict(record)
        self._last_resolved_targets['spatial_focus'] = dict(record)

    def _binding_reference_key(self, target_ref: Any) -> str:
        return ' '.join(str(target_ref or '').strip().lower().replace('_', ' ').split())

    def _resolve_navigation_multi_match(self, result: ObjectBindingResult, *, target_ref: Any, tool_name: str) -> Optional[ObjectBindingResult]:
        if tool_name not in {'navigate_to', 'look_for'}:
            return None
        if result.status != 'ambiguous' or result.reason != 'ambiguous_description_match':
            return None
        candidates = [dict(item) for item in result.candidates if item.get('object_id') in getattr(self.core, 'objects', {}) and float(item.get('confidence', 0.0) or 0.0) > 0.0]
        if not candidates:
            return None
        ranked = self._rank_navigation_binding_candidates(candidates, tool_name=tool_name)
        if not ranked:
            return None
        if tool_name == 'look_for' and (not bool(ranked[0].get('visible_now'))):
            return None
        best = ranked[0]
        selected_object = str(best.get('object_id') or '')
        if not selected_object:
            return None
        selected_candidates = []
        for item in ranked[:8]:
            out = dict(item)
            out['selected_for_navigation'] = str(out.get('object_id')) == selected_object
            selected_candidates.append(out)
        return ObjectBindingResult(result.raw_reference, selected_object, 'success', float(best.get('confidence', result.confidence) or result.confidence), 'navigation_multi_match_selected', str(best.get('source') or result.source or 'description'), selected_candidates)

    def _resolve_physical_multi_match(self, result: ObjectBindingResult, *, target_ref: Any, tool_name: str, argument_name: str) -> Optional[ObjectBindingResult]:
        if tool_name in {'navigate_to', 'look_for', 'navigate_to_room', 'scan', 'answer', 'stop', 'drop'}:
            return None
        if result.status != 'ambiguous' or result.reason != 'ambiguous_description_match':
            return None
        candidates = [dict(item) for item in result.candidates if item.get('object_id') in getattr(self.core, 'objects', {}) and float(item.get('confidence', 0.0) or 0.0) > 0.0]
        if not candidates:
            return None
        inherited = self._physical_inherited_binding_candidate(candidates, target_ref=target_ref, tool_name=tool_name, argument_name=argument_name)
        if inherited is not None:
            return inherited
        ranked = [item for item in self._rank_physical_binding_candidates(candidates, tool_name=tool_name, argument_name=argument_name) if float(item.get('physical_affordance_score', 0.0) or 0.0) > 0.0]
        if not ranked:
            return None
        best = ranked[0]
        selected_object = str(best.get('object_id') or '')
        if not selected_object:
            return None
        selected_candidates = []
        for item in ranked[:8]:
            out = dict(item)
            out['selected_for_physical_interaction'] = str(out.get('object_id')) == selected_object
            selected_candidates.append(out)
        return ObjectBindingResult(result.raw_reference, selected_object, 'success', float(best.get('confidence', result.confidence) or result.confidence), 'physical_multi_match_selected', str(best.get('source') or result.source or 'description'), selected_candidates)

    def _physical_inherited_binding_candidate(self, candidates: Sequence[Mapping[str, Any]], *, target_ref: Any, tool_name: str, argument_name: str) -> Optional[ObjectBindingResult]:
        focus = dict(self._last_resolved_targets.get('spatial_focus') or {})
        if not focus:
            return None
        focus_step = int(focus.get('high_level_step') or -999999)
        current_step = int(self.high_level_step)
        if focus_step not in {current_step - 1, current_step}:
            return None
        if self._binding_reference_key(focus.get('raw_reference')) != self._binding_reference_key(target_ref):
            return None
        selected_object = str(focus.get('selected_object') or '')
        if not selected_object:
            return None
        matching = [dict(item) for item in candidates if str(item.get('object_id') or '') == selected_object]
        if not matching:
            return None
        ranked = self._rank_physical_binding_candidates(matching, tool_name=tool_name, argument_name=argument_name)
        if not ranked or float(ranked[0].get('physical_affordance_score', 0.0) or 0.0) <= 0.0:
            return None
        best = ranked[0]
        best['selected_for_physical_interaction'] = True
        best['inherited_from_tool'] = focus.get('tool_name')
        best['inherited_from_step'] = focus.get('high_level_step')
        return ObjectBindingResult(str(target_ref), selected_object, 'success', float(best.get('confidence', 1.0) or 1.0), 'physical_multi_match_inherited_focus', str(best.get('source') or 'physical_context'), [best])

    def _rank_physical_binding_candidates(self, candidates: Sequence[Mapping[str, Any]], *, tool_name: str, argument_name: str) -> list[dict[str, Any]]:
        visible_ids = set(self._visible_object_names_at_pose(self.core.ego_state.pose)) if self.core is not None else set()
        agent_xy = tuple(getattr(self.core.ego_state.pose, 'xy', (0.0, 0.0))) if self.core is not None else (0.0, 0.0)
        focus = dict(self._last_resolved_targets.get('spatial_focus') or {})
        focus_object = str(focus.get('selected_object') or '')
        ranked: list[dict[str, Any]] = []
        for candidate in candidates:
            object_id = str(candidate.get('object_id') or '')
            obj = getattr(self.core, 'objects', {}).get(object_id) if self.core is not None else None
            if obj is None:
                continue
            item = dict(candidate)
            item['visible_now'] = object_id in visible_ids or bool(candidate.get('visible'))
            item['physical_affordance_score'] = self._physical_affordance_score(tool_name, obj, argument_name=argument_name)
            item['distance_to_agent_m'] = self._object_xy_distance(agent_xy, obj)
            item['same_as_recent_focus'] = bool(focus_object and focus_object == object_id)
            ranked.append(item)
        return sorted(ranked, key=lambda item: (-float(item.get('confidence', 0.0) or 0.0), -float(item.get('physical_affordance_score', 0.0) or 0.0), 0 if item.get('same_as_recent_focus') else 1, 0 if item.get('visible_now') else 1, float(item.get('distance_to_agent_m') if item.get('distance_to_agent_m') is not None else 1000000000.0), str(item.get('object_id') or '')))

    def _physical_affordance_score(self, tool_name: str, obj: Any, *, argument_name: str='target') -> float:
        abilities = {str(item) for item in getattr(obj, 'abilities', ()) or ()}
        states = getattr(obj, 'states', {}) if isinstance(getattr(obj, 'states', {}), Mapping) else {}
        category = str(getattr(obj, 'category', '') or '')

        def has_any(names: Sequence[str]) -> bool:
            return bool(abilities & {str(item) for item in names})
        if tool_name == 'pick_up':
            return 1.0 if has_any(PICK_ABILITY_NAMES) else 0.0
        if tool_name == 'fill' and argument_name == 'source':
            return 1.0 if has_any(FLUID_SOURCE_ABILITY_NAMES) or category in {'faucet', 'sink', 'furniture_sink', 'water_source'} else 0.0
        if tool_name in {'put_down', 'heat', 'cool'}:
            if tool_name == 'heat' and (has_any(HEATSOURCE_ABILITY_NAMES) or category in {'stove', 'oven', 'microwave', 'grill', 'toaster_oven', 'range'}):
                return 1.0
            if tool_name == 'cool' and (has_any(COLDSOURCE_ABILITY_NAMES) or category in {'fridge', 'freezer', 'refrigerator'}):
                return 1.0
            if tool_name == 'put_down' and (has_any(PLACE_ON_SURFACE_ABILITY_NAMES) or has_any(PLACE_INSIDE_CONTAINER_ABILITY_NAMES)):
                return 1.0
            return 0.0
        if tool_name in {'open', 'close'}:
            return 1.0 if OPEN_ABILITY_NAME in abilities or 'open' in states else 0.0
        if tool_name in {'toggle_on', 'toggle_off'}:
            return 1.0 if TOGGLE_ABILITY_NAME in abilities or 'toggled_on' in states else 0.0
        if tool_name == 'cut':
            return 1.0 if has_any(CUT_TARGET_ABILITY_NAMES) or 'sliced' in states else 0.0
        if tool_name == 'wipe':
            return 1.0 if has_any(WIPE_TARGET_ABILITY_NAMES) or 'dirty' in states else 0.0
        if tool_name == 'soak_under':
            return 1.0 if has_any(FLUID_SOURCE_ABILITY_NAMES) or category in {'faucet', 'sink', 'furniture_sink', 'water_source'} else 0.0
        if tool_name == 'fill':
            return 1.0 if has_any(FILLABLE_ABILITY_NAMES) else 0.0
        if tool_name == 'pour':
            return 1.0 if has_any(POURABLE_ABILITY_NAMES) or has_any(PLACE_INSIDE_CONTAINER_ABILITY_NAMES) else 0.0
        return 0.5

    def _object_xy_distance(self, agent_xy: Sequence[float], obj: Any) -> Optional[float]:
        try:
            return float(math.dist((float(agent_xy[0]), float(agent_xy[1])), tuple(getattr(obj, 'xy_center'))))
        except Exception:
            try:
                center = getattr(obj, 'center')
                return float(math.dist((float(agent_xy[0]), float(agent_xy[1])), (float(center[0]), float(center[1]))))
            except Exception:
                return None

    def _rank_navigation_binding_candidates(self, candidates: Sequence[Mapping[str, Any]], *, tool_name: str) -> list[dict[str, Any]]:
        visible_ids = set(self._visible_object_names_at_pose(self.core.ego_state.pose)) if self.core is not None else set()
        ranked: list[dict[str, Any]] = []
        for candidate in candidates:
            object_id = str(candidate.get('object_id') or '')
            obj = getattr(self.core, 'objects', {}).get(object_id) if self.core is not None else None
            if obj is None:
                continue
            nav = self._navigation_candidate_record(object_id) if tool_name == 'navigate_to' else {}
            item = dict(candidate)
            item['visible_now'] = object_id in visible_ids or bool(candidate.get('visible'))
            item['navigation_plan'] = nav
            item['navigation_reachable'] = bool(nav.get('reachable')) if nav else True
            item['navigation_path_length'] = nav.get('path_length_m')
            item['navigation_goal_distance'] = nav.get('goal_distance_m')
            ranked.append(item)
        return sorted(ranked, key=lambda item: (-float(item.get('confidence', 0.0) or 0.0), 0 if item.get('navigation_reachable') else 1, 0 if item.get('visible_now') else 1, float(item.get('navigation_path_length') if item.get('navigation_path_length') is not None else 1000000000.0), float(item.get('navigation_goal_distance') if item.get('navigation_goal_distance') is not None else 1000000000.0), str(item.get('object_id') or '')))

    def _navigation_candidate_record(self, object_id: str) -> dict[str, Any]:
        if self.core is None or object_id not in self.core.objects:
            return {'reachable': False, 'reason': 'target_unavailable'}
        try:
            if self.core.occupancy is None:
                self.core.build_occupancy(trace_id='navigation_multi_match_occupancy')
            target = NavigationTarget.from_aabb_object(self.core.objects[object_id])
            plan = plan_navigation(self.core.ego_state.pose.copy(), target, self.core.occupancy, self.core.config.navigation)
        except Exception as exc:
            return {'reachable': False, 'reason': 'navigation_plan_error', 'error': repr(exc)}
        path = list(plan.path_xy or [])
        path_length = 0.0
        for before, after in zip(path, path[1:]):
            path_length += math.dist(before, after)
        goal_distance = None
        if plan.selected_goal_xy is not None:
            try:
                goal_distance = math.dist(self.core.ego_state.pose.xy, plan.selected_goal_xy)
            except Exception:
                goal_distance = None
        return {'reachable': bool(plan.success), 'path_length_m': float(path_length) if path else 0.0 if plan.success else None, 'goal_distance_m': float(goal_distance) if goal_distance is not None else None, 'selected_goal_xy': list(plan.selected_goal_xy) if plan.selected_goal_xy is not None else None, 'rejected_candidates': len(plan.rejected_candidates), 'failure_reasons': list(plan.check.reasons or []) if not plan.success else []}

    def _extra_kwargs(self, tool_name: str, arguments: Mapping[str, Any]) -> tuple[dict, Optional[str]]:
        if tool_name == 'fill':
            source_ref = arguments.get('source')
            if source_ref is None:
                return ({}, "fill requires a 'source' argument.")
            previous_count = len(self._current_tool_binding_records)
            source = self._resolve_target({'source': source_ref}, tool_name=tool_name, argument_name='source')
            self._ensure_fill_binding_record(argument_name='source', public_ref=source_ref, resolved=source, previous_count=previous_count)
            return ({'source': source}, None) if source is not None else ({}, None)
        return ({}, None)

    def _ensure_fill_binding_record(self, *, argument_name: str, public_ref: Any, resolved: Optional[str], previous_count: int) -> None:
        if len(self._current_tool_binding_records) > int(previous_count):
            return
        self._current_tool_binding_records.append({'status': 'success' if resolved is not None else 'unresolved', 'reason': 'public_reference_resolved' if resolved is not None else 'public_reference_unresolved', 'source': 'behavior_public_binding', 'match_kind': 'public_reference' if resolved is not None else None, 'canonical_public_ref': str(public_ref) if resolved is not None and public_ref is not None else None, 'navigation_class': None, 'argument_name': str(argument_name)})

    def _record_staged_object(self, object_name: str, receptacle_name: Optional[str]) -> None:
        if not object_name or not receptacle_name:
            return
        self._staged_object_ledger[str(object_name)] = {'receptacle': str(receptacle_name), 'staged_at_step': int(self.high_level_step)}

    def _record_private_inside_parent_placement(self, object_name: str, parent_name: Optional[str], *, tool_name: str, state_delta: Mapping[str, Any] | None) -> None:
        if self.core is None or not object_name or (not parent_name):
            return
        obj = self.core.objects.get(str(object_name))
        parent = self.core.objects.get(str(parent_name))
        if obj is None or parent is None or (not _is_inside(obj, parent)):
            self.invalidate_private_parent_placement(object_name)
            return
        delta = dict(state_delta or {})
        placement = delta.get('object_pose') if tool_name == 'put_down' else (delta.get('place') or {}).get('object_pose')
        if not isinstance(placement, Mapping) or placement.get('placement_relation') != 'inside':
            self.invalidate_private_parent_placement(object_name)
            return
        if str(placement.get('target') or '') != str(object_name):
            return
        if str(placement.get('place_target') or '') != str(parent_name):
            return
        self._inside_parent_ledger[str(object_name)] = {'parent': str(parent_name), 'object_center': tuple((float(value) for value in obj.center)), 'parent_center': tuple((float(value) for value in parent.center)), 'source_tool': str(tool_name), 'recorded_at_step': int(self.high_level_step)}

    def invalidate_private_parent_placement(self, object_name: str) -> None:
        self._inside_parent_ledger.pop(str(object_name), None)

    def _private_inside_parent_for_navigation(self, object_name: Optional[str]) -> Optional[str]:
        record = self._valid_private_inside_parent_record(object_name)
        return str(record['parent']) if record is not None else None

    def _private_inside_parent_for_interaction(self, object_name: Optional[str], visible_ids: Sequence[str]) -> Optional[str]:
        record = self._valid_private_inside_parent_record(object_name)
        if record is None or self.core is None:
            return None
        parent_name = str(record['parent'])
        rendered_ids = list(visible_ids)
        if self._last_visibility_mode != 'rendered_segmentation':
            rendered = self._rendered_visible_object_names_at_pose(self.core.ego_state.pose, min_pixels=self.action_visibility_min_pixels, force=True)
            if rendered is None:
                return None
            rendered_ids = list(rendered)
        if not self._target_visible_by_nonspatial_alias(parent_name, rendered_ids):
            return None
        parent = self.core.objects[parent_name]
        distance = self._object_xy_distance(tuple(self.core.ego_state.pose.xy), parent)
        minimum, maximum = self.core.config.navigation.interaction_distance
        if distance is None or not float(minimum) <= float(distance) <= float(maximum):
            return None
        return parent_name

    def _valid_private_inside_parent_record(self, object_name: Optional[str]) -> Optional[dict[str, Any]]:
        if self.core is None or not object_name:
            return None
        record = self._inside_parent_ledger.get(str(object_name))
        if not isinstance(record, Mapping):
            return None
        parent_name = str(record.get('parent') or '')
        obj = self.core.objects.get(str(object_name))
        parent = self.core.objects.get(parent_name)
        if obj is None or parent is None or actions.get_open_state(parent) is not True:
            self.invalidate_private_parent_placement(str(object_name))
            return None
        if not _is_inside(obj, parent):
            self.invalidate_private_parent_placement(str(object_name))
            return None
        if not self._centers_match(obj.center, record.get('object_center')):
            self.invalidate_private_parent_placement(str(object_name))
            return None
        if not self._centers_match(parent.center, record.get('parent_center')):
            self.invalidate_private_parent_placement(str(object_name))
            return None
        return dict(record)

    @staticmethod
    def _centers_match(current: Sequence[float], recorded: Any, *, tolerance: float=0.03) -> bool:
        if not isinstance(recorded, (list, tuple)) or len(recorded) != 3:
            return False
        try:
            return math.dist(tuple((float(value) for value in current)), tuple((float(value) for value in recorded))) <= tolerance
        except Exception:
            return False

    def _resolve_private_staged_soak_target(self, source_name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if self.core is None or not source_name:
            return (None, 'staged_target_unresolved')
        source = self.core.objects.get(str(source_name))
        if source is None:
            return (None, 'staged_target_unresolved')
        eligible: list[str] = []
        for object_name in self._staged_object_ledger:
            obj = self.core.objects.get(str(object_name))
            if obj is None or actions.get_dirty_state(obj) is not True:
                continue
            if not bool(getattr(obj, 'visible', True)):
                continue
            if self._object_distance_between(obj, source) > 1.6:
                continue
            eligible.append(str(object_name))
        if len(eligible) == 1:
            return (eligible[0], None)
        if len(eligible) > 1:
            return (None, 'staged_target_ambiguous')
        return (None, 'staged_target_unresolved')

    @staticmethod
    def _object_distance_between(first: Any, second: Any) -> float:
        try:
            first_center = getattr(first, 'center')
            second_center = getattr(second, 'center')
            return math.dist((float(first_center[0]), float(first_center[1])), (float(second_center[0]), float(second_center[1])))
        except Exception:
            return float('inf')

    @staticmethod
    def _sanitize_staged_wash_state_delta(state_delta: Any) -> dict[str, Any]:
        if not isinstance(state_delta, Mapping):
            return {}
        sanitized = copy.deepcopy(dict(state_delta))
        for key in ('object_state', 'cleaning'):
            record = sanitized.get(key)
            if isinstance(record, Mapping):
                record = dict(record)
                record.pop('target', None)
                record.pop('source', None)
                sanitized[key] = record
        return sanitized

    def binding_metrics(self) -> dict[str, Any]:
        records = list(getattr(self.object_resolver, 'records', []) or [])
        receptacle_records = [item for item in records if item.get('binding_target_kind') == 'receptacle']
        receptacle_successes = sum((1 for item in receptacle_records if item.get('status') == 'success'))
        return {'binding_attempts': len(records), 'failures_by_reason': {reason: sum((r.get('reason') == reason and r.get('status') != 'success' for r in records)) for reason in sorted({r.get('reason', 'unknown') for r in records if r.get('status') != 'success'})}, 'binding_successes': sum((1 for item in records if item.get('status') == 'success')), 'binding_ambiguous': sum((1 for item in records if item.get('status') == 'ambiguous')), 'binding_unresolved': sum((1 for item in records if item.get('status') == 'unresolved')), 'binding_conflict': sum((1 for item in records if item.get('status') == 'conflict')), 'binding_rejected': sum((1 for item in records if item.get('status') == 'rejected')), 'binding_not_interaction_ready': sum((1 for item in records if item.get('status') == 'not_interaction_ready')), 'receptacle_binding_attempts': len(receptacle_records), 'receptacle_binding_successes': receptacle_successes, 'receptacle_binding_failures': len(receptacle_records) - receptacle_successes}

    def _belief_navigation_enabled(self) -> bool:
        primary = self._mid_execution_primary_task()
        if not isinstance(primary, Mapping):
            return False
        if bool(primary.get('belief_navigation') or primary.get('belief_navigation_enabled')):
            return True
        return str(primary.get('type') or '') == 'object_relocated'

    def _belief_navigation_enabled_for_target(self, arguments: Mapping[str, Any], target_name: Optional[str]) -> bool:
        if not self._belief_navigation_enabled():
            return False
        if not target_name or self.core is None or target_name not in self.core.objects:
            return False
        primary = self._mid_execution_primary_task()
        object_role = str((primary or {}).get('object_role') or 'target_object')
        target_ref = arguments.get('target')
        if target_ref is not None and str(target_ref) == object_role:
            return True
        return self.role_map.get(object_role) == str(target_name)

    def _mid_execution_primary_task(self) -> Mapping[str, Any]:
        spec = self.current_task_spec if isinstance(self.current_task_spec, Mapping) else {}
        family = spec.get('family_spec') if isinstance(spec.get('family_spec'), Mapping) else {}
        mid = family.get('mid_execution') if isinstance(family.get('mid_execution'), Mapping) else {}
        primary = mid.get('primary_task') if isinstance(mid.get('primary_task'), Mapping) else {}
        return primary

    def _observe_visible_objects(self, visible_object_ids: Sequence[str], *, source: str, pose: Any) -> list[dict[str, Any]]:
        if self.core is None:
            return []
        return self._belief_tracker.observe_visible(visible_object_ids, self.core.objects, step=int(self.high_level_step), source=str(source), pose=self._pose_to_dict(pose))

    def _observe_visible_objects_across_poses(self, poses: Sequence[Any], *, source: str) -> tuple[list[dict[str, Any]], list[str]]:
        updates: list[dict[str, Any]] = []
        all_visible_ids: set[str] = set()
        for pose in poses:
            visible_ids = self._visible_object_names_at_pose(pose)
            all_visible_ids.update((str(item) for item in visible_ids))
            updates.extend(self._observe_visible_objects(visible_ids, source=source, pose=pose))
        return (updates, sorted(all_visible_ids))

    def _public_agent_belief(self) -> dict[str, Any]:
        role_beliefs: dict[str, Any] = {}
        object_ids: set[str] = set()
        for role_name, object_id in sorted(self.role_map.items()):
            object_ids.add(str(object_id))
            role_beliefs[str(role_name)] = self._belief_tracker.ensure(str(object_id)).as_dict()
        known_records = {object_id: record for object_id, record in self._belief_tracker.snapshot().items() if record.get('status') != 'unknown'}
        return {'enabled': self._belief_navigation_enabled(), 'role_beliefs': role_beliefs, 'known_object_beliefs': known_records, 'navigation_metrics': {'seen_object_navigation': self.seen_object_navigation_steps, 'stale_object_navigation': self.stale_object_navigation_steps, 'unseen_object_exploration': self.unseen_object_exploration_steps, 'target_missing_at_last_seen': self.target_missing_at_last_seen_count}}

    def _navigate_to_object_with_belief(self, arguments: Mapping[str, Any], target_name: str) -> dict:
        from scoper.benchmark.common import evaluation_navigation
        if evaluation_navigation.enabled(self):
            previous = evaluation_navigation.consume_previous_position(self, target_name)
            if previous is not None:
                return self._navigate_to_last_seen_object(arguments, target_name, self._belief_tracker.ensure(target_name), previous)
            return self._navigate_to_visible_object(arguments, target_name, self._navigation_belief_before(target_name))
        record = self._belief_tracker.ensure(target_name)
        last_seen_center = self._belief_tracker.valid_last_seen_center(target_name)
        if last_seen_center is None:
            return self._navigate_to_object_exploration(arguments, target_name, record)
        return self._navigate_to_last_seen_object(arguments, target_name, record, last_seen_center)

    def _navigation_belief_before(self, target_name: Optional[str]) -> dict[str, Any]:
        if not target_name:
            return {'belief_status_before': 'unknown', 'last_seen_center_before': None, 'last_seen_step_before': None, 'last_seen_source_before': None, 'navigation_mode': 'unseen_object_exploration'}
        record = self._belief_tracker.ensure(str(target_name))
        last_seen_center = self._belief_tracker.valid_last_seen_center(str(target_name))
        return {'belief_status_before': str(record.status), 'last_seen_center_before': [float(value) for value in last_seen_center] if last_seen_center is not None else None, 'last_seen_step_before': record.last_seen_step, 'last_seen_source_before': record.last_seen_source, 'navigation_mode': 'seen_object_navigation' if last_seen_center is not None else 'unseen_object_exploration'}

    def _navigate_to_streaming(self, arguments: Mapping[str, Any], target_name: Optional[str], navigation_belief_before: Mapping[str, Any] | None) -> dict:
        trace_id = 'step_{:04d}_navigate_to'.format(self.high_level_step)
        self.core.trace.emit(trace_id, 'action_requested', action='NAVIGATE_TO', target=target_name)
        start_pose = self.core.ego_state.pose.copy()
        if not target_name or target_name not in self.core.objects:
            pre = CheckResult.fail('target_not_found', target=target_name)
            action_result = ActionResult(action='NAVIGATE_TO', args={'target': target_name}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.core.ego_state.pose.copy())
            self.core.action_results.append(action_result)
            self.core.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            state_delta, raw, _low = self._standard_navigate_to_raw_execution(arguments, target_name, action_result, success=False, error_type='target_not_found', error_message="Target '{}' was not found.".format(arguments.get('target')), state_delta={}, navigation_belief_before=navigation_belief_before)
            return raw
        if self.core.occupancy is None:
            self.core.build_occupancy(trace_id='{}_occupancy'.format(trace_id))
        target_obj = self.core.objects[target_name]
        target = NavigationTarget.from_aabb_object(target_obj)
        self.core.trace.emit(trace_id, 'precondition_check_started', checks=['occupancy_path_planning'])
        plan = plan_navigation(start_pose, target, self.core.occupancy, self.core.config.navigation)
        self.core.trace.emit(trace_id, 'precondition_check_finished', check=plan.check)
        self.core.trace.emit(trace_id, 'navigation_planned', success=plan.success, check=plan.check, selected_goal_xy=plan.selected_goal_xy, rejected_candidates=plan.rejected_candidates)
        if not plan.success:
            action_result = ActionResult(action='NAVIGATE_TO', args={'target': target_name}, success=False, precondition=plan.check, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.core.ego_state.pose.copy())
            self.core.action_results.append(action_result)
            self.core.trace.emit(trace_id, 'action_failed', reasons=plan.check.reasons, metrics=plan.check.metrics)
            error_type, error_message = self._action_error(action_result)
            _state_delta, raw, _low = self._standard_navigate_to_raw_execution(arguments, target_name, action_result, success=False, error_type=error_type, error_message=error_message, state_delta={}, navigation_belief_before=navigation_belief_before)
            return raw
        stream = self._run_streaming_pose_checkpoints(tool_name='navigate_to', arguments=arguments, trace_id=trace_id, poses=[start_pose, *list(plan.frame_poses)], prompt_metadata={'navigation_target': arguments.get('target'), 'selected_goal_xy': list(plan.selected_goal_xy) if plan.selected_goal_xy is not None else None}, privileged_metadata={'path_xy': list(plan.path_xy), 'target_object_id': target_name})
        interrupted = bool(stream.get('interrupted'))
        post = CheckResult.fail('stream_interrupted') if interrupted else CheckResult.pass_(end_cell=self.core.occupancy.world_to_cell(self.core.ego_state.pose.xy))
        state_delta = {'ego_pose': {'before': start_pose, 'after': self.core.ego_state.pose.copy()}}
        action_result = ActionResult(action='NAVIGATE_TO', args={'target': target_name}, success=not interrupted, precondition=plan.check, postcondition=post, trace_id=trace_id, frames=len(stream.get('executed_poses') or []), start_pose=start_pose, end_pose=self.core.ego_state.pose.copy(), state_delta=state_delta)
        self.core.action_results.append(action_result)
        if interrupted:
            self.core.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        else:
            self.core.trace.emit(trace_id, 'action_succeeded', frames=action_result.frames, end_pose=action_result.end_pose)
        error_type, error_message = self._action_error(action_result)
        state_delta, raw, low_level_actions = self._standard_navigate_to_raw_execution(arguments, target_name, action_result, success=bool(action_result.success), error_type=error_type, error_message=error_message, state_delta=to_jsonable(action_result.state_delta or {}), navigation_belief_before=navigation_belief_before)
        raw['frame_count'] = int(action_result.frames)
        raw['low_level_actions'] = low_level_actions
        raw['state_delta'] = state_delta
        if interrupted:
            raw['_stream_interrupt_record'] = stream.get('interrupt_record')
        return raw

    def _standard_navigate_to_raw_execution(self, arguments: Mapping[str, Any], target_name: Optional[str], action_result: Any, *, success: bool, error_type: Optional[str], error_message: Optional[str], state_delta: Mapping[str, Any] | None, navigation_belief_before: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        belief_before = dict(navigation_belief_before or self._navigation_belief_before(target_name))
        navigation_mode = str(belief_before.get('navigation_mode') or 'unseen_object_exploration')
        target_id = str(target_name) if target_name is not None else None
        observation_pose = None
        if success and target_id and (self.core is not None) and (target_id in self.core.objects):
            try:
                observation_pose = pose_looking_at_object(self.core.ego_state.pose, self.core.objects[target_id])
                self._set_agent_orientation(self._pose_to_dict(observation_pose))
            except Exception:
                observation_pose = None
        visible_ids = self._visible_object_names_at_pose(self.core.ego_state.pose) if success and self.core is not None else []
        target_visible = bool(target_id and target_id in set(visible_ids))
        belief_updates = self._observe_visible_objects(visible_ids, source='navigate_to', pose=self.core.ego_state.pose) if success else []
        if success:
            if navigation_mode == 'seen_object_navigation':
                self.seen_object_navigation_steps += 1
            else:
                self.unseen_object_exploration_steps += 1
        nav_delta = {'target': target_id, 'target_ref': arguments.get('target'), 'navigation_mode': navigation_mode, 'belief_status_before': belief_before.get('belief_status_before'), 'last_seen_center_before': belief_before.get('last_seen_center_before'), 'last_seen_step_before': belief_before.get('last_seen_step_before'), 'last_seen_source_before': belief_before.get('last_seen_source_before'), 'target_visible_after_navigation': bool(target_visible), 'target_missing_at_last_seen': False, 'visible_object_ids': visible_ids, 'visible_object_roles': self._roles_for_object_names(visible_ids), 'belief_updates': belief_updates, 'navigation_target_source': 'standard_object_navigation', 'true_state_used_internally': True, 'agent_visible_disclosure': 'observation_only'}
        updated_state_delta = dict(state_delta or {})
        if success:
            ego_pose_delta = updated_state_delta.get('ego_pose') if isinstance(updated_state_delta.get('ego_pose'), Mapping) else {}
            updated_state_delta['ego_pose'] = {'before': ego_pose_delta.get('before'), 'after': self._agent_pose()}
        updated_state_delta['navigate_to'] = nav_delta
        if observation_pose is not None:
            nav_delta['target_observation_pose'] = self._pose_to_dict(observation_pose)
        low_level_actions = self._low_level_actions_from_object_navigation(action_result, target_name=target_id or str(arguments.get('target') or ''), target_ref=arguments.get('target'), navigation_mode=navigation_mode, success_override=success)
        message = 'navigate_to succeeded.' if success else error_message or 'navigate_to failed.'
        raw = {'status': 'success' if success else 'failed', 'action': 'navigate_to', 'target': arguments.get('target'), 'object_id': target_id, 'navigation_mode': navigation_mode, 'belief_status_before': belief_before.get('belief_status_before'), 'last_seen_center_before': belief_before.get('last_seen_center_before'), 'frame_count': int(getattr(action_result, 'frames', 0) or 0), 'final_pose': self._agent_pose(), 'visible_object_ids': visible_ids, 'visible_object_roles': self._roles_for_object_names(visible_ids), 'target_visible_after_navigation': bool(target_visible), 'target_missing_at_last_seen': False, 'message': message, 'state_delta': updated_state_delta, 'low_level_actions': low_level_actions}
        if error_type:
            raw['error_type'] = error_type
            raw['error_message'] = error_message
        return (updated_state_delta, raw, low_level_actions)

    def _strict_navigation_visibility_enabled(self) -> bool:
        return str(self.action_visibility_mode or '') in {'line_of_sight_required', 'rendered_required'}

    def _navigate_to_visible_object(self, arguments: Mapping[str, Any], target_name: Optional[str], navigation_belief_before: Mapping[str, Any] | None) -> dict[str, Any]:
        target_id = str(target_name) if target_name is not None else ''
        trace_id = 'step_{:04d}_navigate_to_visible'.format(self.high_level_step)
        if self.core is None or not target_id or target_id not in getattr(self.core, 'objects', {}):
            return self._failed_visible_navigation_raw(arguments, target_id, navigation_belief_before, error_type='target_not_found', error_message='navigate_to target is unavailable.', selection={'status': 'target_unavailable'})
        selection = self._select_visible_navigation_goal(target_id)
        if not bool(selection.get('success')):
            return self._failed_visible_navigation_raw(arguments, target_id, navigation_belief_before, error_type=str(selection.get('error_type') or 'navigation_target_not_visible'), error_message=str(selection.get('error_message') or "No reachable navigation pose has line-of-sight to '{}'.".format(target_id)), selection=selection)
        goal_xy = tuple((float(value) for value in selection['selected_goal_xy']))
        facing_xy = tuple((float(value) for value in selection['target_facing_xy']))
        action_result = self.core.navigate_to_pose(goal_xy, facing_xy=facing_xy, trace_id=trace_id, action_name='NAVIGATE_TO_VISIBLE', metadata={'target': target_id, 'target_ref': arguments.get('target'), 'visibility_mode': self.action_visibility_mode, 'goal_selection': to_jsonable(selection)})
        state_delta = to_jsonable(action_result.state_delta or {})
        error_type, error_message = self._action_error(action_result)
        success = bool(action_result.success)
        postcheck = {}
        observation_pose = None
        if success:
            selected_pose = selection.get('target_observation_pose') if isinstance(selection.get('target_observation_pose'), Mapping) else None
            if selected_pose is not None:
                observation_pose = dict(self._pose_to_dict(selected_pose))
                current_pose = self._agent_pose()
                observation_pose['x'] = current_pose['x']
                observation_pose['y'] = current_pose['y']
                observation_pose['z'] = current_pose['z']
            else:
                observation_pose = self._pose_looking_at_center(self.core.ego_state.pose, self.core.objects[target_id].center)
            self._set_agent_orientation(observation_pose)
            postcheck = self._navigation_target_visibility_check(target_id, observation_pose)
            if not bool(postcheck.get('visible')):
                success = False
                error_type = str(postcheck.get('error_type') or 'navigation_target_not_visible_after_arrival')
                error_message = str(postcheck.get('error_message') or "navigate_to reached a pose where '{}' is still not visible.".format(target_id))
            else:
                visible_after_arrival = self._visible_object_names_at_pose(self.core.ego_state.pose)
                target_visible_after_arrival = self._target_visible_or_equivalent(target_id, visible_after_arrival)
                arrival_readiness = self._target_interaction_readiness({'status': 'success', 'object_id': target_id, 'visible_object_ids': visible_after_arrival, 'target_visible_after_navigation': target_visible_after_arrival})
                if arrival_readiness.get('interaction_ready') is not True:
                    success = False
                    error_type = 'navigation_pose_rejected'
                    error_message = 'navigate_to did not reach a usable pose for the target.'
        visible_ids = self._visible_object_names_at_pose(self.core.ego_state.pose) if success else []
        target_visible = bool(target_id and self._target_visible_or_equivalent(target_id, visible_ids))
        belief_updates = self._observe_visible_objects(visible_ids, source='navigate_to', pose=self.core.ego_state.pose) if success and target_visible else []
        belief_before = dict(navigation_belief_before or self._navigation_belief_before(target_id))
        navigation_mode = str(belief_before.get('navigation_mode') or 'visible_object_navigation')
        if success:
            if navigation_mode == 'seen_object_navigation':
                self.seen_object_navigation_steps += 1
            else:
                self.unseen_object_exploration_steps += 1
        nav_delta = {'target': target_id, 'target_ref': arguments.get('target'), 'navigation_mode': navigation_mode, 'belief_status_before': belief_before.get('belief_status_before'), 'last_seen_center_before': belief_before.get('last_seen_center_before'), 'last_seen_step_before': belief_before.get('last_seen_step_before'), 'last_seen_source_before': belief_before.get('last_seen_source_before'), 'target_visible_after_navigation': bool(target_visible), 'target_missing_at_last_seen': False, 'visible_object_ids': visible_ids, 'visible_object_roles': self._roles_for_object_names(visible_ids), 'belief_updates': belief_updates, 'navigation_target_source': 'visibility_aware_object_navigation', 'true_state_used_internally': True, 'agent_visible_disclosure': 'observation_only', 'target_observation_pose': observation_pose, 'visibility_goal_selection': to_jsonable(selection), 'arrival_visibility_check': to_jsonable(postcheck)}
        state_delta = dict(state_delta or {})
        ego_pose_delta = state_delta.get('ego_pose') if isinstance(state_delta.get('ego_pose'), Mapping) else {}
        state_delta['ego_pose'] = {'before': ego_pose_delta.get('before'), 'after': self._agent_pose()}
        state_delta['navigate_to'] = nav_delta
        low_level_actions = self._low_level_actions_from_object_navigation(action_result, target_name=target_id, target_ref=arguments.get('target'), navigation_mode=navigation_mode, success_override=success)
        message = 'navigate_to reached a visible pose for {}.'.format(arguments.get('target') or target_id) if success else error_message or 'navigate_to failed.'
        raw = {'status': 'success' if success else 'failed', 'action': 'navigate_to', 'target': arguments.get('target'), 'object_id': target_id, 'navigation_mode': navigation_mode, 'belief_status_before': belief_before.get('belief_status_before'), 'last_seen_center_before': belief_before.get('last_seen_center_before'), 'frame_count': int(getattr(action_result, 'frames', 0) or 0), 'final_pose': self._agent_pose(), 'visible_object_ids': visible_ids, 'visible_object_roles': self._roles_for_object_names(visible_ids), 'target_visible_after_navigation': bool(target_visible), 'target_missing_at_last_seen': False, 'message': message, 'state_delta': state_delta, 'low_level_actions': low_level_actions, 'visibility_goal_selection': to_jsonable(selection), 'arrival_visibility_check': to_jsonable(postcheck)}
        if error_type:
            raw['error_type'] = error_type
            raw['error_message'] = error_message
        return raw

    def _retarget_inside_parent_navigation(self, raw_execution: Mapping[str, Any], *, target_name: Optional[str], parent_name: str) -> dict[str, Any]:
        raw = dict(raw_execution)
        if not target_name:
            return raw
        rendered = self._rendered_visible_object_names_at_pose(self.core.ego_state.pose, min_pixels=self.action_visibility_min_pixels, force=True) if self.core is not None else None
        visible_ids = [str(name) for name in (rendered if rendered is not None else raw.get('visible_object_ids') or [])]
        target_visible = self._target_visible_by_nonspatial_alias(str(target_name), visible_ids)
        raw['object_id'] = str(target_name)
        raw['target_visible_after_navigation'] = bool(target_visible)
        raw['visible_object_ids'] = visible_ids
        raw['visible_object_roles'] = self._roles_for_object_names(visible_ids)
        raw['navigation_mode'] = 'contained_object_parent_navigation'
        raw['_private_interaction_target_id'] = str(parent_name)
        raw['message'] = 'navigate_to reached the known container holding the requested object.' if raw.get('status') == 'success' else raw.get('message')
        state_delta = dict(raw.get('state_delta') or {})
        navigate_delta = dict(state_delta.get('navigate_to') or {})
        navigate_delta.update({'target': str(target_name), 'navigation_mode': 'contained_object_parent_navigation', 'target_visible_after_navigation': bool(target_visible), 'navigation_target_source': 'episode_placed_parent_container', 'navigation_anchor': str(parent_name)})
        state_delta['navigate_to'] = navigate_delta
        raw['state_delta'] = state_delta
        return raw

    def _failed_visible_navigation_raw(self, arguments: Mapping[str, Any], target_name: str, navigation_belief_before: Mapping[str, Any] | None, *, error_type: str, error_message: str, selection: Mapping[str, Any]) -> dict[str, Any]:
        belief_before = dict(navigation_belief_before or self._navigation_belief_before(target_name))
        navigation_mode = str(belief_before.get('navigation_mode') or 'visible_object_navigation')
        nav_delta = {'target': target_name, 'target_ref': arguments.get('target'), 'navigation_mode': navigation_mode, 'belief_status_before': belief_before.get('belief_status_before'), 'target_visible_after_navigation': False, 'target_missing_at_last_seen': False, 'visible_object_ids': [], 'visible_object_roles': [], 'belief_updates': [], 'navigation_target_source': 'visibility_aware_object_navigation', 'visibility_goal_selection': to_jsonable(selection), 'arrival_visibility_check': {}}
        state_delta = {'navigate_to': nav_delta}
        return {'status': 'failed', 'action': 'navigate_to', 'target': arguments.get('target'), 'object_id': target_name, 'navigation_mode': navigation_mode, 'frame_count': 0, 'final_pose': self._agent_pose(), 'visible_object_ids': [], 'visible_object_roles': [], 'target_visible_after_navigation': False, 'target_missing_at_last_seen': False, 'message': error_message, 'error_type': error_type, 'error_message': error_message, 'state_delta': state_delta, 'low_level_actions': [], 'visibility_goal_selection': to_jsonable(selection), 'arrival_visibility_check': {}}

    def _navigation_anchor_object_for_target(self, target_name: str, target_obj: Any) -> tuple[Any, dict[str, Any]]:
        default_anchor = {'source': 'target_object', 'target_name': target_name, 'anchor_name': target_name, 'anchor_category': getattr(target_obj, 'category', None), 'small_pickupable_target': self._is_small_pickupable_navigation_target(target_obj)}
        if self.core is None or not self._is_small_pickupable_navigation_target(target_obj):
            return (target_obj, default_anchor)
        support = self._infer_navigation_support_surface(target_name, target_obj)
        if support is None:
            return (target_obj, {**default_anchor, 'support_surface_found': False})
        support_obj, support_record = support
        return (support_obj, {**default_anchor, 'source': 'support_surface', 'anchor_name': support_obj.name, 'anchor_category': support_obj.category, 'support_surface_found': True, 'support_score': support_record.get('score'), 'support_vertical_gap_m': support_record.get('vertical_gap_m'), 'support_xy_distance_m': support_record.get('xy_distance_m')})

    def _is_small_pickupable_navigation_target(self, obj: Any) -> bool:
        abilities = {str(item) for item in getattr(obj, 'abilities', ()) or ()}
        if 'pickupable' not in abilities:
            return False
        try:
            footprint = max(float(obj.extent[0]), float(obj.extent[1]))
        except Exception:
            return False
        return footprint <= _NAVIGATION_SMALL_OBJECT_MAX_FOOTPRINT_M

    def _infer_navigation_support_surface(self, target_name: str, target_obj: Any) -> tuple[Any, dict[str, Any]] | None:
        if self.core is None:
            return None
        candidates: list[tuple[float, Any, dict[str, Any]]] = []
        try:
            target_bottom = float(target_obj.center[2]) - 0.5 * float(target_obj.extent[2])
        except Exception:
            return None
        for name, obj in self.core.objects.items():
            if str(name) == str(target_name):
                continue
            if not self._is_navigation_support_surface_candidate(obj):
                continue
            try:
                support_top = float(obj.center[2]) + 0.5 * float(obj.extent[2])
            except Exception:
                continue
            floor_height = float(getattr(getattr(self.core.config, 'occupancy', None), 'floor_height', 0.0))
            if support_top <= floor_height + _NAVIGATION_GROUND_SUPPORT_MAX_HEIGHT_M:
                continue
            vertical_gap = abs(target_bottom - support_top)
            if target_bottom < support_top - 0.08:
                continue
            if vertical_gap > _NAVIGATION_SUPPORT_VERTICAL_TOLERANCE_M:
                continue
            xy_distance = self._xy_aabb_distance(target_obj, obj)
            if xy_distance > _NAVIGATION_SUPPORT_HORIZONTAL_TOLERANCE_M:
                continue
            area = max(float(obj.extent[0]) * float(obj.extent[1]), 0.0)
            score = vertical_gap + xy_distance + 0.002 * area
            record = {'score': float(score), 'vertical_gap_m': float(vertical_gap), 'xy_distance_m': float(xy_distance)}
            candidates.append((float(score), obj, record))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return (candidates[0][1], candidates[0][2])

    def _is_navigation_support_surface_candidate(self, obj: Any) -> bool:
        category = str(getattr(obj, 'category', '') or '').lower()
        if category in _NAVIGATION_SUPPORT_CATEGORIES:
            return True
        abilities = {str(item) for item in getattr(obj, 'abilities', ()) or ()}
        if 'pickupable' in abilities:
            return False
        try:
            footprint = max(float(obj.extent[0]), float(obj.extent[1]))
        except Exception:
            footprint = 0.0
        if footprint < _NAVIGATION_SMALL_OBJECT_MAX_FOOTPRINT_M:
            return False
        return bool({'support_surface', 'receptacle', 'openable'} & abilities)

    def _xy_aabb_distance(self, a: Any, b: Any) -> float:
        ax0 = float(a.center[0]) - 0.5 * float(a.extent[0])
        ax1 = float(a.center[0]) + 0.5 * float(a.extent[0])
        ay0 = float(a.center[1]) - 0.5 * float(a.extent[1])
        ay1 = float(a.center[1]) + 0.5 * float(a.extent[1])
        bx0 = float(b.center[0]) - 0.5 * float(b.extent[0])
        bx1 = float(b.center[0]) + 0.5 * float(b.extent[0])
        by0 = float(b.center[1]) - 0.5 * float(b.extent[1])
        by1 = float(b.center[1]) + 0.5 * float(b.extent[1])
        dx = max(bx0 - ax1, ax0 - bx1, 0.0)
        dy = max(by0 - ay1, ay0 - by1, 0.0)
        return float(math.hypot(dx, dy))

    def _project_navigation_goal_to_component(self, goal_xy: Sequence[float], occupancy: Any, *, start_component: int, max_radius_m: float) -> dict[str, Any] | None:
        if int(start_component) < 0:
            return None
        try:
            origin_cell = occupancy.world_to_cell(goal_xy)
            resolution = max(float(getattr(occupancy, 'resolution', 0.1)), 1e-06)
        except Exception:
            return None
        max_cells = max(1, int(math.ceil(float(max_radius_m) / resolution)))
        best: tuple[float, tuple[int, int], tuple[float, float]] | None = None
        for radius in range(1, max_cells + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    cell = (int(origin_cell[0]) + dr, int(origin_cell[1]) + dc)
                    if not occupancy.in_bounds_cell(cell):
                        continue
                    if not occupancy.is_cell_free(cell, eroded=True):
                        continue
                    xy = occupancy.cell_to_world(cell)
                    distance = math.dist((float(goal_xy[0]), float(goal_xy[1])), xy)
                    if distance > float(max_radius_m):
                        continue
                    if int(occupancy.component_at(xy)) != int(start_component):
                        continue
                    if best is None or distance < best[0]:
                        best = (float(distance), cell, (float(xy[0]), float(xy[1])))
            if best is not None:
                break
        if best is None:
            return None
        return {'distance_m': float(best[0]), 'cell': [int(best[1][0]), int(best[1][1])], 'goal_xy': [float(best[2][0]), float(best[2][1])]}

    def _adaptive_navigation_candidates(self, target: NavigationTarget, *, base_candidates: Sequence[Sequence[float]], base_interaction_distance: Sequence[float], base_candidate_count: int) -> list[tuple[float, float]]:
        try:
            min_dist = float(base_interaction_distance[0])
            max_dist = float(base_interaction_distance[1])
        except Exception:
            min_dist, max_dist = (0.1, 1.8)
        expanded_band = (max(0.05, min_dist), max(max_dist, 2.4))
        expanded_count = max(int(base_candidate_count) * 2, 384)
        candidates = sample_interaction_candidates(target, expanded_count, expanded_band)
        seen = {(round(float(candidate[0]), 4), round(float(candidate[1]), 4)) for candidate in base_candidates}
        out: list[tuple[float, float]] = []
        for candidate in candidates:
            key = (round(float(candidate[0]), 4), round(float(candidate[1]), 4))
            if key in seen:
                continue
            seen.add(key)
            out.append((float(candidate[0]), float(candidate[1])))
        return out

    def _select_visible_navigation_goal(self, target_name: str) -> dict[str, Any]:
        if self.core is None or target_name not in getattr(self.core, 'objects', {}):
            return {'success': False, 'error_type': 'target_not_found', 'target': target_name}
        occupancy = self._ensure_visibility_occupancy()
        if occupancy is None:
            return {'success': False, 'error_type': 'line_of_sight_visibility_unavailable', 'error_message': 'Cannot select a visible navigation pose without occupancy.', 'target': target_name}
        target_obj = self.core.objects[target_name]
        navigation_obj, navigation_anchor = self._navigation_anchor_object_for_target(target_name, target_obj)
        target = NavigationTarget.from_aabb_object(navigation_obj)
        config = self.core.config.navigation
        candidates = sample_interaction_candidates(target, config.candidate_count, config.interaction_distance)
        start_pose = self.core.ego_state.pose.copy()
        start_component = occupancy.component_at(start_pose.xy)
        rejected: list[dict[str, Any]] = []
        candidate_records: list[dict[str, Any]] = []
        visible_records: list[dict[str, Any]] = []
        reachable_count = 0
        los_clear_count = 0
        for goal_xy in candidates:
            record = self._visible_navigation_candidate_record(target_name, target, target_obj, goal_xy, occupancy, start_pose=start_pose, start_component=start_component, navigation_anchor=navigation_anchor, adaptive_stage=1, projection_max_radius_m=1.0)
            candidate_records.append(record)
            if not bool(record.get('reachable')):
                if len(rejected) < _NAVIGATION_VISIBILITY_REJECT_LOG_LIMIT:
                    rejected.append(record)
                continue
            reachable_count += 1
            if not bool(record.get('line_of_sight_clear')):
                if len(rejected) < _NAVIGATION_VISIBILITY_REJECT_LOG_LIMIT:
                    rejected.append(record)
                continue
            los_clear_count += 1
            self._visible_navigation_candidate_score(record, config=config, start_pose=start_pose)
            visible_records.append(record)
        adaptive_stage_triggered = False
        if not visible_records and self.allow_navigation_candidate_fallbacks:
            adaptive_stage_triggered = True
            adaptive_candidates = self._adaptive_navigation_candidates(target, base_candidates=candidates, base_interaction_distance=config.interaction_distance, base_candidate_count=config.candidate_count)
            for goal_xy in adaptive_candidates:
                record = self._visible_navigation_candidate_record(target_name, target, target_obj, goal_xy, occupancy, start_pose=start_pose, start_component=start_component, navigation_anchor=navigation_anchor, adaptive_stage=2, projection_max_radius_m=1.8)
                candidate_records.append(record)
                if not bool(record.get('reachable')):
                    if len(rejected) < _NAVIGATION_VISIBILITY_REJECT_LOG_LIMIT:
                        rejected.append(record)
                    continue
                reachable_count += 1
                if not bool(record.get('line_of_sight_clear')):
                    if len(rejected) < _NAVIGATION_VISIBILITY_REJECT_LOG_LIMIT:
                        rejected.append(record)
                    continue
                los_clear_count += 1
                self._visible_navigation_candidate_score(record, config=config, start_pose=start_pose)
                record['adaptive_recovered'] = True
                visible_records.append(record)
        eligible_visible_records = self._eligible_navigation_candidate_records(target_name, visible_records)
        selected_record = self._select_visible_navigation_candidate(eligible_visible_records, config=config, start_pose=start_pose)
        filter_summary = self._visible_navigation_candidate_filter_summary(candidate_records, candidate_count=len(candidate_records), logged_rejected_count=len(rejected), start_component=start_component, interaction_distance=config.interaction_distance, optimal_observation_distance=config.optimal_observation_distance, selected_record=selected_record)
        filter_summary['adaptive_stage_triggered'] = bool(adaptive_stage_triggered)
        filter_summary['adaptive_stage_2_candidate_count'] = int(sum((1 for record in candidate_records if int(record.get('adaptive_stage') or 1) == 2)))
        filter_summary['previously_failed_candidate_count'] = int(len(visible_records) - len(eligible_visible_records))
        if selected_record is None:
            if visible_records and (not eligible_visible_records):
                error_type = 'navigation_candidates_exhausted'
            elif reachable_count <= 0:
                error_type = 'target_not_reachable'
            else:
                error_type = 'navigation_target_not_visible'
            return {'success': False, 'error_type': error_type, 'error_message': "No reachable navigation pose has line-of-sight to '{}'.".format(target_name) if reachable_count > 0 else "No reachable navigation pose is available for '{}'.".format(target_name), 'target': target_name, 'candidate_count': len(candidate_records), 'reachable_candidate_count': int(reachable_count), 'line_of_sight_clear_candidate_count': int(los_clear_count), 'rejected_candidates': to_jsonable(rejected), 'rejected_candidates_truncated': int(filter_summary['rejected_detail_truncated_count']), 'candidate_filter_summary': to_jsonable(filter_summary), 'navigation_anchor': to_jsonable(navigation_anchor)}
        selected = dict(selected_record)
        selected.update({'success': True, 'target': target_name, 'candidate_count': len(candidate_records), 'reachable_candidate_count': int(reachable_count), 'line_of_sight_clear_candidate_count': int(los_clear_count), 'rejected_candidates': to_jsonable(rejected), 'rejected_candidates_truncated': int(filter_summary['rejected_detail_truncated_count']), 'candidate_filter_summary': to_jsonable(filter_summary), 'navigation_anchor': to_jsonable(navigation_anchor)})
        return selected

    def _validate_default_pickup_route(self, target_name: str, *, stable_visibility_checks: int=2, start_from_last_seen_pose: bool=False) -> dict[str, Any]:
        if self.core is None or target_name not in getattr(self.core, 'objects', {}):
            return {'success': False, 'error_type': 'target_not_found', 'target': target_name}
        target_obj = self.core.objects[target_name]
        original_pose = self.core.ego_state.pose.copy()
        navigation_start = original_pose.copy()
        navigation_start_source = 'current_pose'
        if start_from_last_seen_pose:
            try:
                belief_record = self._belief_tracker.get(target_name)
            except Exception:
                belief_record = None
            last_seen_pose = getattr(belief_record, 'last_seen_pose', None)
            if isinstance(last_seen_pose, Mapping):
                navigation_start = EgoPose(**self._pose_to_dict(last_seen_pose))
                navigation_start_source = 'last_seen_pose'
        try:
            self.core.ego_state.pose = navigation_start.copy()
            selection = self._select_visible_navigation_goal(target_name)
        finally:
            self.core.ego_state.pose = original_pose
        if not bool(selection.get('success')):
            failed = dict(selection)
            failed['navigation_start_source'] = navigation_start_source
            return failed
        selected_goal_xy = selection['selected_goal_xy']
        observation_pose = dict(selection.get('target_observation_pose') or self._pose_looking_at_center(EgoPose(x=float(selected_goal_xy[0]), y=float(selected_goal_xy[1]), z=float(original_pose.z), yaw=float(original_pose.yaw), pitch=float(original_pose.pitch)), target_obj.center))
        report = {'success': False, 'target': target_name, 'navigation_start_source': navigation_start_source, 'selected_goal_xy': [float(selected_goal_xy[0]), float(selected_goal_xy[1])], 'navigation_anchor': to_jsonable(selection.get('navigation_anchor')), 'target_observation_pose': observation_pose, 'stable_visibility_required': max(1, int(stable_visibility_checks)), 'stable_visibility_passes': 0}
        if self.renderer is None:
            report.update({'success': True, 'validation_mode': 'symbolic_navigation_without_renderer', 'rendered_visible': None})
            return report
        visibility_results: list[bool] = []
        for _ in range(report['stable_visibility_required']):
            visible = self._rendered_visible_object_names_at_pose(observation_pose, force=True)
            if visible is None:
                report.update({'error_type': 'rendered_visibility_unavailable', 'visibility_results': visibility_results})
                return report
            matched = self._target_visible_by_nonspatial_alias(target_name, visible)
            visibility_results.append(bool(matched))
            if not matched:
                report.update({'error_type': 'target_not_stably_visible', 'visibility_results': visibility_results, 'stable_visibility_passes': sum(visibility_results)})
                return report
        report.update({'success': True, 'validation_mode': 'default_navigation_endpoint_rendered', 'rendered_visible': True, 'visibility_results': visibility_results, 'stable_visibility_passes': sum(visibility_results)})
        return report

    @staticmethod
    def _navigation_candidate_identity(record: Mapping[str, Any]) -> tuple[float, float] | None:
        goal = record.get('selected_goal_xy') or record.get('goal_xy')
        if not isinstance(goal, Sequence) or isinstance(goal, (str, bytes)) or len(goal) < 2:
            return None
        try:
            return (round(float(goal[0]), 4), round(float(goal[1]), 4))
        except (TypeError, ValueError):
            return None

    def _eligible_navigation_candidate_records(self, target_name: str, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        del target_name
        return list(records)

    def _update_failed_navigation_candidate_ledger(self, raw_execution: Mapping[str, Any]) -> None:
        del raw_execution

    def _select_visible_navigation_candidate(self, records: Sequence[dict[str, Any]], *, config: NavigationConfig, start_pose: EgoPose) -> dict[str, Any] | None:
        if not records:
            return None
        best: tuple[tuple[float, ...], dict[str, Any]] | None = None
        for record in records:
            raw_key = record.get('score_sort_key')
            if isinstance(raw_key, (list, tuple)):
                try:
                    score_key = tuple((float(value) for value in raw_key))
                except Exception:
                    score_key = self._visible_navigation_candidate_score(record, config=config, start_pose=start_pose)
            else:
                score_key = self._visible_navigation_candidate_score(record, config=config, start_pose=start_pose)
            if best is None or score_key < best[0]:
                best = (score_key, record)
        return best[1] if best is not None else None

    def _visible_navigation_candidate_score(self, record: dict[str, Any], *, config: NavigationConfig, start_pose: EgoPose) -> tuple[float, ...]:
        optimal_distance = max(float(config.optimal_observation_distance), 0.0)
        distance_to_target_center = float(record.get('distance_to_target_center_m') or math.inf)
        observation_distance_error = abs(distance_to_target_center - optimal_distance)
        path_length_m = float(record.get('path_length_m') or 0.0)
        target_yaw = record.get('target_yaw')
        yaw_delta = abs(_normalize_angle(float(target_yaw) - float(start_pose.yaw))) if target_yaw is not None else math.inf
        try:
            interaction_distance = float(record.get('interaction_distance_m'))
            interaction_min = float(config.interaction_distance[0])
            interaction_max = float(config.interaction_distance[1])
            interaction_margin = min(interaction_distance - interaction_min, interaction_max - interaction_distance)
            interaction_ready_penalty = 0.0 if interaction_margin >= 0.0 else 1.0
        except (TypeError, ValueError, IndexError):
            interaction_margin = -math.inf
            interaction_ready_penalty = 1.0
        score_key = (interaction_ready_penalty, -float(interaction_margin), observation_distance_error, path_length_m, yaw_delta)
        score_payload = {'navigation_score_policy': 'interaction_ready_margin_then_observation_distance', 'optimal_observation_distance_m': float(optimal_distance), 'observation_distance_error_m': float(observation_distance_error), 'interaction_distance_margin_m': float(interaction_margin), 'yaw_delta_from_start_rad': float(yaw_delta), 'score_sort_key': [float(value) for value in score_key], 'score': float(observation_distance_error)}
        record.update(score_payload)
        return score_key

    def _visible_navigation_candidate_record(self, target_name: str, navigation_target: NavigationTarget, target_obj: Any, goal_xy: Sequence[float], occupancy: Any, *, start_pose: EgoPose, start_component: int, navigation_anchor: Mapping[str, Any], adaptive_stage: int=1, projection_max_radius_m: float=1.0) -> dict[str, Any]:
        goal = (float(goal_xy[0]), float(goal_xy[1]))
        interaction_facing_xy = navigation_target.facing_xy_for_goal(goal)
        facing_xy = (float(target_obj.center[0]), float(target_obj.center[1]))
        observation_pose = self._pose_looking_at_center({'x': goal[0], 'y': goal[1], 'z': start_pose.z, 'yaw': start_pose.yaw, 'pitch': start_pose.pitch}, target_obj.center)
        record: dict[str, Any] = {'goal_xy': [float(goal[0]), float(goal[1])], 'target_facing_xy': [float(facing_xy[0]), float(facing_xy[1])], 'interaction_facing_xy': [float(interaction_facing_xy[0]), float(interaction_facing_xy[1])], 'navigation_anchor': dict(navigation_anchor), 'navigation_anchor_name': navigation_anchor.get('anchor_name'), 'navigation_anchor_source': navigation_anchor.get('source'), 'adaptive_stage': int(adaptive_stage), 'distance_to_target_center_m': float(math.dist(goal, (float(target_obj.center[0]), float(target_obj.center[1])))), 'distance_to_navigation_anchor_center_m': float(math.dist(goal, navigation_target.xy_center)), 'distance_to_interaction_facing_m': float(math.dist(goal, interaction_facing_xy)), 'target_observation_pose': observation_pose, 'goal_free': False, 'same_component': False, 'path_planned': False, 'reachable': False, 'fov_visible': False, 'line_of_sight_clear': False}
        if not occupancy.is_free(goal):
            record.update({'reason': 'goal_occupied', 'failed_filter': 'goal_free'})
            return record
        record['goal_free'] = True
        goal_component = occupancy.component_at(goal)
        record['goal_component'] = int(goal_component)
        record['component_projection_applied'] = False
        record['component_projection_success'] = False
        if self.allow_navigation_candidate_fallbacks and start_component != -1 and (goal_component != -1) and (start_component != goal_component):
            projection = self._project_navigation_goal_to_component(goal, occupancy, start_component=start_component, max_radius_m=float(projection_max_radius_m))
            if projection is not None:
                original_goal = goal
                goal = (float(projection['goal_xy'][0]), float(projection['goal_xy'][1]))
                interaction_facing_xy = navigation_target.facing_xy_for_goal(goal)
                observation_pose = self._pose_looking_at_center({'x': goal[0], 'y': goal[1], 'z': start_pose.z, 'yaw': start_pose.yaw, 'pitch': start_pose.pitch}, target_obj.center)
                record.update({'goal_xy': [float(goal[0]), float(goal[1])], 'projected_from_goal_xy': [float(original_goal[0]), float(original_goal[1])], 'component_projection_applied': True, 'component_projection_distance_m': float(projection['distance_m']), 'component_projection_max_radius_m': float(projection_max_radius_m), 'component_projection_cell': projection.get('cell'), 'goal_component': int(start_component), 'interaction_facing_xy': [float(interaction_facing_xy[0]), float(interaction_facing_xy[1])], 'distance_to_target_center_m': float(math.dist(goal, (float(target_obj.center[0]), float(target_obj.center[1])))), 'distance_to_navigation_anchor_center_m': float(math.dist(goal, navigation_target.xy_center)), 'distance_to_interaction_facing_m': float(math.dist(goal, interaction_facing_xy)), 'target_observation_pose': observation_pose})
            else:
                record.update({'reason': 'target_not_reachable', 'failed_filter': 'same_component', 'start_component': int(start_component), 'component_projection_attempted': True, 'component_projection_success': False})
                return record
        if start_component != -1 and goal_component != -1 and (start_component != goal_component) and (not bool(record.get('component_projection_applied'))):
            record.update({'reason': 'target_not_reachable', 'failed_filter': 'same_component', 'start_component': int(start_component)})
            return record
        record['same_component'] = True
        interaction_distance = self._object_xy_distance(goal, target_obj)
        interaction_min = float(self.core.config.navigation.interaction_distance[0])
        interaction_max = float(self.core.config.navigation.interaction_distance[1])
        record.update({'interaction_distance_m': interaction_distance, 'interaction_distance_in_range': bool(interaction_distance is not None and interaction_min <= float(interaction_distance) <= interaction_max)})
        plan = plan_navigation_to_pose(start_pose, goal, occupancy, self.core.config.navigation, facing_xy=facing_xy)
        if not bool(plan.success):
            record.update({'reason': plan.check.reasons[0] if plan.check.reasons else 'target_not_reachable', 'failed_filter': 'path_planning', 'failure_reasons': list(plan.check.reasons or []), 'metrics': dict(plan.check.metrics or {})})
            return record
        record['path_planned'] = True
        visibility = self._navigation_target_visibility_check(target_name, observation_pose)
        fov_visible = bool(visibility.get('fov_visible'))
        visible = bool(visibility.get('visible'))
        failed_filter = None
        if not visible:
            failed_filter = 'line_of_sight' if fov_visible else 'symbolic_fov'
        path = list(plan.path_xy or [])
        path_length = 0.0
        for before, after in zip(path, path[1:]):
            path_length += math.dist(before, after)
        record.update({'reachable': True, 'fov_visible': fov_visible, 'line_of_sight_clear': visible, 'reason': 'line_of_sight_clear' if visible else visibility.get('reason'), 'failed_filter': failed_filter, 'visibility': to_jsonable(visibility), 'path_length_m': float(path_length), 'path_waypoints': len(path), 'target_yaw': plan.target_yaw, 'selected_goal_xy': [float(goal[0]), float(goal[1])]})
        return record

    def _visible_navigation_candidate_filter_summary(self, records: Sequence[Mapping[str, Any]], *, candidate_count: int, logged_rejected_count: int, start_component: int, interaction_distance: Sequence[float], optimal_observation_distance: float, selected_record: Mapping[str, Any] | None) -> dict[str, Any]:
        reason_counts: Counter[str] = Counter()
        failed_filter_counts: Counter[str] = Counter()
        visibility_reason_counts: Counter[str] = Counter()
        visibility_error_counts: Counter[str] = Counter()
        blocked_object_counts: Counter[str] = Counter()
        blocked_category_counts: Counter[str] = Counter()
        adaptive_stage_counts: Counter[str] = Counter()
        stage_counts: Counter[str] = Counter({'sampled': len(records)})
        sampled_interaction_distances: list[float] = []
        reachable_interaction_distances: list[float] = []
        visible_interaction_distances: list[float] = []
        reachable_path_lengths: list[float] = []
        visible_path_lengths: list[float] = []
        height_exempted_candidate_count = 0
        component_projection_candidate_count = 0
        for record in records:
            adaptive_stage_counts[str(record.get('adaptive_stage') or 1)] += 1
            if bool(record.get('component_projection_applied')):
                component_projection_candidate_count += 1
            try:
                sampled_interaction_distances.append(float(record.get('distance_to_interaction_facing_m')))
            except Exception:
                pass
            if bool(record.get('goal_free')):
                stage_counts['goal_free'] += 1
            if bool(record.get('same_component')):
                stage_counts['same_component'] += 1
            if bool(record.get('path_planned')):
                stage_counts['path_planned'] += 1
            if bool(record.get('reachable')):
                stage_counts['reachable'] += 1
                try:
                    reachable_path_lengths.append(float(record.get('path_length_m')))
                except Exception:
                    pass
                try:
                    reachable_interaction_distances.append(float(record.get('distance_to_interaction_facing_m')))
                except Exception:
                    pass
            if bool(record.get('fov_visible')):
                stage_counts['fov_visible'] += 1
            if bool(record.get('line_of_sight_clear')):
                stage_counts['line_of_sight_clear'] += 1
                try:
                    visible_path_lengths.append(float(record.get('path_length_m')))
                except Exception:
                    pass
                try:
                    visible_interaction_distances.append(float(record.get('distance_to_interaction_facing_m')))
                except Exception:
                    pass
            reason = str(record.get('reason') or 'unknown')
            if reason and reason != 'line_of_sight_clear':
                reason_counts[reason] += 1
            failed_filter = str(record.get('failed_filter') or '')
            if failed_filter:
                failed_filter_counts[failed_filter] += 1
            visibility = record.get('visibility')
            if isinstance(visibility, Mapping):
                visibility_reason = str(visibility.get('reason') or '')
                if visibility_reason and visibility_reason != 'line_of_sight_clear':
                    visibility_reason_counts[visibility_reason] += 1
                error_type = str(visibility.get('error_type') or '')
                if error_type:
                    visibility_error_counts[error_type] += 1
                line = visibility.get('line_of_sight')
                if isinstance(line, Mapping):
                    if line.get('height_exempted_objects') or line.get('height_exempted_cells'):
                        height_exempted_candidate_count += 1
                    for blocker in line.get('blocked_objects') or []:
                        if not isinstance(blocker, Mapping):
                            continue
                        name = str(blocker.get('name') or 'unknown')
                        category = str(blocker.get('category') or 'unknown')
                        blocked_object_counts[name] += 1
                        blocked_category_counts[category] += 1

        def _top(counter: Counter[str], limit: int=12) -> dict[str, int]:
            return {str(key): int(value) for key, value in counter.most_common(limit)}

        def _path_stats(values: Sequence[float]) -> dict[str, float | int | None]:
            if not values:
                return {'count': 0, 'min': None, 'max': None}
            return {'count': int(len(values)), 'min': float(min(values)), 'max': float(max(values))}
        filtered_out_count = max(0, int(candidate_count) - int(stage_counts.get('line_of_sight_clear', 0)))
        selected = None
        if selected_record is not None:
            selected = {'goal_xy': selected_record.get('goal_xy'), 'navigation_anchor': selected_record.get('navigation_anchor'), 'navigation_anchor_name': selected_record.get('navigation_anchor_name'), 'navigation_anchor_source': selected_record.get('navigation_anchor_source'), 'adaptive_stage': selected_record.get('adaptive_stage'), 'adaptive_recovered': bool(selected_record.get('adaptive_recovered')), 'distance_to_target_center_m': selected_record.get('distance_to_target_center_m'), 'distance_to_navigation_anchor_center_m': selected_record.get('distance_to_navigation_anchor_center_m'), 'distance_to_interaction_facing_m': selected_record.get('distance_to_interaction_facing_m'), 'interaction_distance_m': selected_record.get('interaction_distance_m'), 'interaction_distance_in_range': bool(selected_record.get('interaction_distance_in_range')), 'interaction_distance_margin_m': selected_record.get('interaction_distance_margin_m'), 'optimal_observation_distance_m': selected_record.get('optimal_observation_distance_m'), 'observation_distance_error_m': selected_record.get('observation_distance_error_m'), 'path_length_m': selected_record.get('path_length_m'), 'yaw_delta_from_start_rad': selected_record.get('yaw_delta_from_start_rad'), 'score': selected_record.get('score'), 'score_sort_key': selected_record.get('score_sort_key'), 'navigation_score_policy': selected_record.get('navigation_score_policy'), 'target_yaw': selected_record.get('target_yaw'), 'reason': selected_record.get('reason')}
        return {'candidate_count': int(candidate_count), 'logged_rejected_candidate_count': int(logged_rejected_count), 'filtered_out_candidate_count': int(filtered_out_count), 'valid_visible_candidate_count': int(stage_counts.get('line_of_sight_clear', 0)), 'rejected_detail_truncated_count': max(0, int(filtered_out_count) - int(logged_rejected_count)), 'start_component': int(start_component), 'interaction_distance': [float(value) for value in interaction_distance], 'optimal_observation_distance_m': float(optimal_observation_distance), 'navigation_score_policy': 'interaction_ready_margin_then_observation_distance', 'filter_order': ['goal_free', 'same_component', 'path_planned', 'reachable', 'fov_visible', 'line_of_sight_clear'], 'stage_counts': {str(key): int(stage_counts.get(key, 0)) for key in ('sampled', 'goal_free', 'same_component', 'path_planned', 'reachable', 'fov_visible', 'line_of_sight_clear')}, 'adaptive_stage_counts': {str(key): int(value) for key, value in sorted(adaptive_stage_counts.items())}, 'failed_filter_counts': _top(failed_filter_counts), 'reject_reason_counts': _top(reason_counts), 'visibility_reason_counts': _top(visibility_reason_counts), 'visibility_error_type_counts': _top(visibility_error_counts), 'blocked_object_counts': _top(blocked_object_counts), 'blocked_category_counts': _top(blocked_category_counts), 'height_exempted_candidate_count': int(height_exempted_candidate_count), 'component_projection_candidate_count': int(component_projection_candidate_count), 'sampled_interaction_distance_m': _path_stats(sampled_interaction_distances), 'reachable_interaction_distance_m': _path_stats(reachable_interaction_distances), 'visible_interaction_distance_m': _path_stats(visible_interaction_distances), 'reachable_path_length_m': _path_stats(reachable_path_lengths), 'visible_path_length_m': _path_stats(visible_path_lengths), 'selected': selected}

    def _navigation_target_visibility_check(self, target_name: str, pose: Any) -> dict[str, Any]:
        if self.core is None or target_name not in getattr(self.core, 'objects', {}):
            return {'visible': False, 'reason': 'target_not_found', 'error_type': 'target_not_found'}
        obj = self.core.objects[target_name]
        pose_dict = self._pose_to_dict(pose)
        fov_visible = self._object_visible_from_pose(obj, pose_dict)
        occupancy = self._ensure_visibility_occupancy()
        if occupancy is None:
            line = {'clear': False, 'reason': 'occupancy_unavailable', 'error_type': 'line_of_sight_visibility_unavailable', 'error_message': 'Line-of-sight visibility is required but occupancy is unavailable.'}
        elif fov_visible:
            line = self._object_line_of_sight_from_pose(obj, pose_dict, occupancy)
        else:
            line = {'clear': False, 'reason': 'outside_symbolic_fov'}
        visible = bool(fov_visible and line.get('clear'))
        reason = 'line_of_sight_clear' if visible else str(line.get('reason') or 'target_not_visible')
        if visible:
            error_type = None
        elif reason == 'occupancy_unavailable':
            error_type = 'line_of_sight_visibility_unavailable'
        elif reason.startswith('line_of_sight_'):
            error_type = 'target_occluded_by_occupancy'
        else:
            error_type = 'target_not_visible'
        payload = {'visible': visible, 'reason': reason, 'error_type': error_type, 'error_message': None if visible else "Navigation target '{}' is not visible from the selected pose: {}.".format(target_name, reason), 'target': target_name, 'visibility_mode': 'symbolic_line_of_sight', 'fov_visible': bool(fov_visible), 'line_of_sight': to_jsonable(line), 'pose': pose_dict}
        return payload

    def _navigate_to_last_seen_object(self, arguments: Mapping[str, Any], target_name: str, record: Any, last_seen_center: Sequence[float]) -> dict:
        belief_status_before = str(record.status)
        trace_id = 'step_{:04d}_navigate_to_last_seen'.format(self.high_level_step)
        from scoper.benchmark.common import evaluation_navigation
        if evaluation_navigation.enabled(self):
            if self.core.occupancy is None:
                self.core.build_occupancy(trace_id=trace_id)
            occupancy = self.core.occupancy
            component = occupancy.component_at(self.core.ego_state.pose.xy)
            nearest = self._nearest_free_cell(occupancy, (float(last_seen_center[0]), float(last_seen_center[1])), allowed_components={int(component)} if component >= 0 else None)
            goal_selection = {'success': nearest is not None, 'selected_goal_xy': nearest[1] if nearest is not None else None, 'target_facing_xy': list(last_seen_center[:2]), 'target_observation_point': list(last_seen_center), 'source': 'relocation_previous_position'}
        else:
            goal_selection = self._select_visible_navigation_goal(target_name)
        if not bool(goal_selection.get('success')):
            return self._failed_raw_belief_navigation(str(goal_selection.get('error_type') or 'last_seen_not_reachable'), str(goal_selection.get('error_message') or "No visible reachable pose for '{}'.".format(arguments.get('target'))), target_name=target_name, navigation_mode='seen_object_navigation', belief_record=record.as_dict(), goal_record=goal_selection)
        goal_xy = tuple((float(value) for value in goal_selection['selected_goal_xy']))
        goal_record = goal_selection
        action_result = self.core.navigate_to_pose(goal_xy, facing_xy=tuple((float(value) for value in goal_selection['target_facing_xy'])), trace_id=trace_id, action_name='NAVIGATE_TO_LAST_SEEN', metadata={'target': target_name, 'target_ref': arguments.get('target'), 'belief_status': belief_status_before, 'last_seen_center': [float(value) for value in last_seen_center]})
        state_delta = to_jsonable(action_result.state_delta or {})
        error_type, error_message = self._action_error(action_result)
        success = bool(action_result.success)
        final_pose = self._agent_pose()
        last_seen_observation_pose = None
        if success:
            last_seen_observation_pose = self._pose_looking_at_center(self.core.ego_state.pose, goal_selection.get('target_observation_point') or last_seen_center)
            self._set_agent_orientation(last_seen_observation_pose)
            final_pose = self._agent_pose()
        visible_ids = self._visible_object_names_at_pose(self.core.ego_state.pose) if success else []
        target_visible = target_name in set(visible_ids)
        target_absent_from_last_seen = success and (not target_visible)
        navigation_mode = 'seen_object_navigation'
        message = 'navigate_to reached the last-seen location for {}.'.format(arguments.get('target') or target_name)
        belief_updates: list[dict[str, Any]] = []
        if success and target_visible:
            self.seen_object_navigation_steps += 1
            belief_updates = self._observe_visible_objects(visible_ids, source='navigate_to', pose=self.core.ego_state.pose)
        elif success and (not target_absent_from_last_seen):
            self.seen_object_navigation_steps += 1
        elif target_absent_from_last_seen:
            navigation_mode = 'stale_object_navigation'
            self.stale_object_navigation_steps += 1
            self.target_missing_at_last_seen_count += 1
            before = record.as_dict()
            updated = self._belief_tracker.mark_missing_at_last_seen(target_name, step=int(self.high_level_step), source='navigate_to', pose=final_pose)
            belief_updates = [{'object_id': target_name, 'before': before, 'after': updated.as_dict()}]
            error_type = None
            error_message = None
            message = 'navigate_to reached the last-seen location for {}, but the target is missing there. Use scan or look_for to search before trying again.'.format(arguments.get('target') or target_name)
        nav_delta = {'target': target_name, 'target_ref': arguments.get('target'), 'navigation_mode': navigation_mode, 'belief_status_before': belief_status_before, 'last_seen_center': [float(value) for value in last_seen_center], 'selected_goal_xy': [float(goal_xy[0]), float(goal_xy[1])], 'last_seen_observation_pose': last_seen_observation_pose, 'target_visible_after_navigation': bool(target_visible), 'target_missing_at_last_seen': bool(target_absent_from_last_seen), 'visible_object_ids': visible_ids, 'visible_object_roles': self._roles_for_object_names(visible_ids), 'belief_updates': belief_updates, 'goal_selection': goal_record}
        state_delta = dict(state_delta or {})
        state_delta['navigate_to'] = nav_delta
        low_level_actions = self._low_level_actions_from_object_navigation(action_result, target_name=target_name, target_ref=arguments.get('target'), navigation_mode=navigation_mode, success_override=success)
        raw = {'status': 'success' if success else 'failed', 'action': 'navigate_to', 'target': arguments.get('target'), 'object_id': target_name, 'navigation_mode': navigation_mode, 'belief_status_before': belief_status_before, 'frame_count': int(action_result.frames), 'final_pose': final_pose, 'visible_object_ids': visible_ids, 'visible_object_roles': self._roles_for_object_names(visible_ids), 'target_visible_after_navigation': bool(target_visible), 'target_missing_at_last_seen': bool(target_absent_from_last_seen), 'message': message if success else error_message or 'navigate_to failed.', 'state_delta': state_delta, 'low_level_actions': low_level_actions}
        if error_type:
            raw['error_type'] = error_type
            raw['error_message'] = error_message
        return raw

    def _navigate_to_object_exploration(self, arguments: Mapping[str, Any], target_name: str, record: Any) -> dict:
        belief_status_before = str(record.status)
        trace_id = 'step_{:04d}_navigate_to_exploration'.format(self.high_level_step)
        action_result = self.core.navigate_to(target_name, trace_id=trace_id)
        confirm_result = None
        if bool(action_result.success):
            confirm_result = self.core.look_for(target_name, trace_id='{}_confirm_visible'.format(trace_id))
        state_delta = to_jsonable(action_result.state_delta or {})
        error_type, error_message = self._action_error(action_result)
        success = bool(action_result.success)
        visible_ids = self._visible_object_names_at_pose(self.core.ego_state.pose) if success else []
        target_visible = target_name in set(visible_ids)
        belief_updates = self._observe_visible_objects(visible_ids, source='navigate_to_exploration', pose=self.core.ego_state.pose) if success else []
        if success:
            self.unseen_object_exploration_steps += 1
        navigation_mode = 'unseen_object_exploration'
        nav_delta = {'target': target_name, 'target_ref': arguments.get('target'), 'navigation_mode': navigation_mode, 'belief_status_before': belief_status_before, 'target_visible_after_navigation': bool(target_visible), 'target_missing_at_last_seen': False, 'visible_object_ids': visible_ids, 'visible_object_roles': self._roles_for_object_names(visible_ids), 'belief_updates': belief_updates, 'true_state_used_internally': True, 'agent_visible_disclosure': 'observation_only'}
        state_delta = dict(state_delta or {})
        if confirm_result is not None:
            state_delta['exploration_confirm_observation'] = to_jsonable(confirm_result.state_delta or {})
        state_delta['navigate_to'] = nav_delta
        low_level_actions = self._low_level_actions_from_object_navigation(action_result, target_name=target_name, target_ref=arguments.get('target'), navigation_mode=navigation_mode, success_override=success)
        if confirm_result is not None:
            low_level_actions.extend(self._low_level_actions_from_action_result(confirm_result, source='look_for', target_name=target_name, success_override=target_visible))
        message = 'navigate_to searched for {} and reached a visible location.'.format(arguments.get('target') or target_name) if success else error_message or 'navigate_to failed.'
        raw = {'status': 'success' if success else 'failed', 'action': 'navigate_to', 'target': arguments.get('target'), 'object_id': target_name, 'navigation_mode': navigation_mode, 'belief_status_before': belief_status_before, 'frame_count': int(action_result.frames) + int(getattr(confirm_result, 'frames', 0) or 0), 'final_pose': self._agent_pose(), 'visible_object_ids': visible_ids, 'visible_object_roles': self._roles_for_object_names(visible_ids), 'target_visible_after_navigation': bool(target_visible), 'target_missing_at_last_seen': False, 'message': message, 'state_delta': state_delta, 'low_level_actions': low_level_actions}
        if error_type:
            raw['error_type'] = error_type
            raw['error_message'] = error_message
        return raw

    def _last_seen_navigation_goal(self, target_name: str, last_seen_center: Sequence[float]) -> tuple[tuple[float, float] | None, dict[str, Any]]:
        if self.core is None:
            return (None, {'status': 'core_unavailable'})
        if self.core.occupancy is None:
            self.core.build_occupancy('belief_navigation_occupancy')
        obj = self.core.objects.get(target_name)
        if obj is None or self.core.occupancy is None:
            return (None, {'status': 'target_unavailable'})
        states = dict(getattr(obj, 'states', {}) or {})
        states.pop('interaction_faces', None)
        proxy = AABBObject(str(target_name), str(getattr(obj, 'category', 'object') or 'object'), (float(last_seen_center[0]), float(last_seen_center[1]), float(last_seen_center[2])), tuple((float(value) for value in getattr(obj, 'extent', (0.2, 0.2, 0.2)))), visible=bool(getattr(obj, 'visible', True)), abilities=tuple(getattr(obj, 'abilities', ()) or ()), states=states)
        plan = plan_navigation(self.core.ego_state.pose.copy(), NavigationTarget.from_aabb_object(proxy), self.core.occupancy, self.core.config.navigation)
        if plan.success and plan.selected_goal_xy is not None:
            return (plan.selected_goal_xy, {'status': 'interaction_candidate', 'selected_goal_xy': [float(plan.selected_goal_xy[0]), float(plan.selected_goal_xy[1])], 'rejected_candidates': len(plan.rejected_candidates), 'check_metrics': to_jsonable(plan.check.metrics)})
        allowed_components = None
        start_component = self.core.occupancy.component_at(self.core.ego_state.pose.xy)
        if int(start_component) >= 0:
            allowed_components = {int(start_component)}
        nearest = self._nearest_free_cell(self.core.occupancy, (float(last_seen_center[0]), float(last_seen_center[1])), allowed_components=allowed_components)
        if nearest is None:
            nearest = self._nearest_free_cell(self.core.occupancy, (float(last_seen_center[0]), float(last_seen_center[1])))
        if nearest is None:
            return (None, {'status': 'no_free_cell_near_last_seen', 'plan_reasons': list(plan.check.reasons), 'plan_metrics': to_jsonable(plan.check.metrics)})
        cell, xy, distance = nearest
        return (xy, {'status': 'nearest_free_cell', 'selected_goal_xy': [float(xy[0]), float(xy[1])], 'selected_goal_cell': [int(cell[0]), int(cell[1])], 'distance_from_last_seen_m': float(distance), 'plan_reasons': list(plan.check.reasons), 'plan_metrics': to_jsonable(plan.check.metrics)})

    def _failed_raw_belief_navigation(self, error_type: str, error_message: str, *, target_name: str, navigation_mode: str, belief_record: Mapping[str, Any], goal_record: Mapping[str, Any]) -> dict:
        return {'status': 'failed', 'action': 'navigate_to', 'object_id': target_name, 'navigation_mode': navigation_mode, 'belief_status_before': belief_record.get('status'), 'frame_count': 0, 'final_pose': self._agent_pose() if self.core is not None else None, 'error_type': error_type, 'error_message': error_message, 'message': error_message, 'state_delta': {'navigate_to': {'target': target_name, 'navigation_mode': navigation_mode, 'belief_status_before': belief_record.get('status'), 'goal_selection': dict(goal_record or {})}}, 'low_level_actions': []}

    def _navigate_to_room_streaming(self, arguments: Mapping[str, Any]) -> dict:
        room_ref = arguments.get('room')
        if room_ref is None:
            message = "navigate_to_room requires a 'room' argument."
            return self._failed_raw_room_navigation('missing_argument', message)
        room, error_type, error_message = self._resolve_room_ref(str(room_ref))
        if room is None:
            return self._failed_raw_room_navigation(error_type or 'room_not_found', error_message or 'Room not found.')
        target_xy, repair_record, repair_error = self._room_navigation_goal_xy(room)
        if target_xy is None:
            message = repair_error or "No reachable pose could be selected for room '{}'.".format(room['room_id'])
            return self._failed_raw_room_navigation('room_not_reachable', message, room=room)
        trace_id = 'step_{:04d}_navigate_to_room'.format(self.high_level_step)
        representative_xy = self._room_representative_xy(room)
        self.core.trace.emit(trace_id, 'action_requested', action='NAVIGATE_TO_ROOM', goal_xy=target_xy, metadata={'room_id': room['room_id'], 'room_type': room.get('room_type'), 'reachable_status': repair_record.get('reachable_status')})
        start_pose = self.core.ego_state.pose.copy()
        if self.core.occupancy is None:
            self.core.build_occupancy(trace_id='{}_occupancy'.format(trace_id))
        self.core.trace.emit(trace_id, 'precondition_check_started', checks=['occupancy_path_planning'])
        plan = plan_navigation_to_pose(start_pose, target_xy, self.core.occupancy, self.core.config.navigation, facing_xy=representative_xy)
        self.core.trace.emit(trace_id, 'precondition_check_finished', check=plan.check)
        self.core.trace.emit(trace_id, 'navigation_planned', success=plan.success, check=plan.check, selected_goal_xy=plan.selected_goal_xy, rejected_candidates=plan.rejected_candidates, target_type='pose')
        if not plan.success:
            action_result = ActionResult(action='NAVIGATE_TO_ROOM', args={'goal_xy': target_xy, 'facing_xy': representative_xy, 'metadata': {'room_id': room['room_id']}}, success=False, precondition=plan.check, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.core.ego_state.pose.copy())
            self.core.action_results.append(action_result)
            self.core.trace.emit(trace_id, 'action_failed', reasons=plan.check.reasons, metrics=plan.check.metrics)
            raw = self._failed_raw_room_navigation('room_not_reachable', '; '.join((str(reason) for reason in plan.check.reasons)) or 'Room navigation failed.', room=room)
            raw['low_level_actions'] = []
            raw['state_delta'] = {}
            return raw
        before_room = self.current_room
        stream = self._run_streaming_pose_checkpoints(tool_name='navigate_to_room', arguments=arguments, trace_id=trace_id, poses=[start_pose, *list(plan.frame_poses)], prompt_metadata={'room_id': room.get('room_id'), 'room_type': room.get('room_type'), 'selected_goal_xy': [float(target_xy[0]), float(target_xy[1])]}, privileged_metadata={'repair': repair_record, 'path_xy': list(plan.path_xy)})
        interrupted = bool(stream.get('interrupted'))
        selected_pose = self._pose_to_dict(self.core.ego_state.pose)
        after_room = room['room_id'] if not interrupted else self._current_room_for_pose(self.core.ego_state.pose)
        if not interrupted:
            self._record_room_visit(str(room['room_id']))
            after_room = room['room_id']
        room_delta = {'room_id': room['room_id'], 'room_type': room.get('room_type'), 'before_room': before_room, 'after_room': after_room, 'selected_goal_xy': [float(target_xy[0]), float(target_xy[1])], 'selected_goal_pose': selected_pose, 'representative_pose': room.get('representative_pose'), 'reachable_status': repair_record.get('reachable_status'), 'room_component': repair_record.get('room_component'), 'repair': repair_record}
        post = CheckResult.fail('stream_interrupted') if interrupted else CheckResult.pass_(end_cell=self.core.occupancy.world_to_cell(self.core.ego_state.pose.xy), selected_goal_xy=plan.selected_goal_xy, target_type='pose')
        state_delta = {'ego_pose': {'before': start_pose, 'after': self.core.ego_state.pose.copy()}, 'navigate_to_pose': {'selected_goal_xy': plan.selected_goal_xy, 'target_yaw': plan.target_yaw, 'path_xy': list(plan.path_xy), 'metadata': {'room_id': room['room_id'], 'room_type': room.get('room_type'), 'reachable_status': repair_record.get('reachable_status')}}, 'navigate_to_room': room_delta}
        action_result = ActionResult(action='NAVIGATE_TO_ROOM', args={'goal_xy': target_xy, 'facing_xy': representative_xy, 'metadata': {'room_id': room['room_id']}}, success=not interrupted, precondition=plan.check, postcondition=post, trace_id=trace_id, frames=len(stream.get('executed_poses') or []), start_pose=start_pose, end_pose=self.core.ego_state.pose.copy(), state_delta=state_delta)
        self.core.action_results.append(action_result)
        if interrupted:
            self.core.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        else:
            self.core.trace.emit(trace_id, 'action_succeeded', frames=action_result.frames, end_pose=action_result.end_pose)
        raw = {'status': 'success' if action_result.success else 'failed', 'action': 'navigate_to_room', 'room': arguments.get('room'), 'room_id': room['room_id'], 'room_type': room.get('room_type'), 'frame_count': int(action_result.frames), 'final_pose': self._agent_pose(), 'current_room': self.current_room, 'visited_rooms': sorted(self.visited_rooms), 'selected_goal_xy': room_delta['selected_goal_xy'], 'selected_goal_pose': selected_pose, 'reachable_status': repair_record.get('reachable_status'), 'state_delta': to_jsonable(state_delta), 'low_level_actions': self._low_level_actions_from_pose_navigation(action_result, source='navigate_to_room', room=room, success_override=bool(action_result.success))}
        err_type, err_message = self._action_error(action_result)
        if err_type:
            raw['error_type'] = err_type
            raw['error_message'] = err_message
            raw['message'] = err_message or 'Room navigation failed.'
        if interrupted:
            raw['_stream_interrupt_record'] = stream.get('interrupt_record')
        return raw

    def _navigate_to_room(self, arguments: Mapping[str, Any]) -> dict:
        room_ref = arguments.get('room')
        if room_ref is None:
            message = "navigate_to_room requires a 'room' argument."
            return self._failed_raw_room_navigation('missing_argument', message)
        room, error_type, error_message = self._resolve_room_ref(str(room_ref))
        if room is None:
            return self._failed_raw_room_navigation(error_type or 'room_not_found', error_message or 'Room not found.')
        target_xy, repair_record, repair_error = self._room_navigation_goal_xy(room)
        if target_xy is None:
            message = repair_error or "No reachable pose could be selected for room '{}'.".format(room['room_id'])
            return self._failed_raw_room_navigation('room_not_reachable', message, room=room)
        trace_id = 'step_{:04d}_navigate_to_room'.format(self.high_level_step)
        representative_xy = self._room_representative_xy(room)
        action_result = self.core.navigate_to_pose(target_xy, facing_xy=representative_xy, trace_id=trace_id, action_name='NAVIGATE_TO_ROOM', metadata={'room_id': room['room_id'], 'room_type': room.get('room_type'), 'reachable_status': repair_record.get('reachable_status')})
        state_delta = to_jsonable(action_result.state_delta or {})
        error_type, error_message = self._action_error(action_result)
        success = bool(action_result.success)
        selected_pose = self._pose_to_dict(action_result.end_pose or self.core.ego_state.pose)
        before_room = self.current_room
        after_room = room['room_id'] if success else self._current_room_for_pose(self.core.ego_state.pose)
        if success:
            self._record_room_visit(str(room['room_id']))
        room_delta = {'room_id': room['room_id'], 'room_type': room.get('room_type'), 'before_room': before_room, 'after_room': after_room, 'selected_goal_xy': [float(target_xy[0]), float(target_xy[1])], 'selected_goal_pose': selected_pose, 'representative_pose': room.get('representative_pose'), 'reachable_status': repair_record.get('reachable_status'), 'room_component': repair_record.get('room_component'), 'repair': repair_record}
        state_delta = dict(state_delta or {})
        state_delta['navigate_to_room'] = room_delta
        low_level_actions = self._low_level_actions_from_pose_navigation(action_result, source='navigate_to_room', room=room, success_override=success)
        raw = {'status': 'success' if success else 'failed', 'action': 'navigate_to_room', 'room': arguments.get('room'), 'room_id': room['room_id'], 'room_type': room.get('room_type'), 'frame_count': int(action_result.frames), 'final_pose': self._agent_pose(), 'current_room': self.current_room, 'visited_rooms': sorted(self.visited_rooms), 'selected_goal_xy': room_delta['selected_goal_xy'], 'selected_goal_pose': selected_pose, 'reachable_status': repair_record.get('reachable_status'), 'state_delta': state_delta, 'low_level_actions': low_level_actions}
        if error_type:
            raw['error_type'] = error_type
            raw['error_message'] = error_message
            raw['message'] = error_message or 'Room navigation failed.'
        return raw

    def _failed_raw_room_navigation(self, error_type: str, error_message: str, *, room: Mapping[str, Any] | None=None) -> dict:
        raw = {'status': 'failed', 'action': 'navigate_to_room', 'frame_count': 0, 'final_pose': self._agent_pose() if self.core is not None else None, 'current_room': self.current_room, 'visited_rooms': sorted(self.visited_rooms), 'error_type': error_type, 'error_message': error_message, 'message': error_message}
        if room is not None:
            raw['room_id'] = room.get('room_id')
            raw['room_type'] = room.get('room_type')
        return raw

    def _build_room_catalog(self, task_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
        source = task_spec.get('source') if isinstance(task_spec.get('source'), Mapping) else {}
        raw_rooms = source.get('room_catalog') or source.get('rooms') or task_spec.get('room_catalog') or []
        rooms: list[dict[str, Any]] = []
        seen: set[str] = set()
        if isinstance(raw_rooms, Mapping):
            raw_rooms = raw_rooms.values()
        if not isinstance(raw_rooms, Iterable) or isinstance(raw_rooms, (str, bytes)):
            raw_rooms = []
        for raw in raw_rooms:
            room = self._normalize_room_record(raw)
            if room is None or room['room_id'] in seen:
                continue
            seen.add(room['room_id'])
            rooms.append(room)
        return sorted(rooms, key=lambda item: str(item.get('room_id') or ''))

    def _normalize_room_record(self, raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, str):
            raw = {'room_id': raw}
        if not isinstance(raw, Mapping):
            return None
        room_id = str(raw.get('room_id') or raw.get('id') or raw.get('room') or '')
        if not room_id:
            return None
        room_type = str(raw.get('room_type') or raw.get('type') or room_id.rsplit('_', 1)[0])
        pose = raw.get('representative_pose') or raw.get('representative_reachable_pose')
        normalized_pose = self._normalize_room_pose(pose)
        target_xy = self._room_xy_from_raw(raw.get('target_xy') or raw.get('goal_xy'))
        if target_xy is None and normalized_pose is not None:
            target_xy = self._xy_from_pose_position(normalized_pose.get('position'))
        return {'room_id': room_id, 'room_type': room_type, 'object_count': int(raw.get('object_count') or 0), 'source': str(raw.get('source') or raw.get('room_instance_source') or 'profile'), 'geometry_status': str(raw.get('geometry_status') or 'unknown'), 'reachable_status': str(raw.get('reachable_status') or 'profile_fallback'), 'representative_pose': normalized_pose, 'target_xy': [float(target_xy[0]), float(target_xy[1])] if target_xy is not None else None}

    def _normalize_room_pose(self, pose: Any) -> dict[str, Any] | None:
        if not isinstance(pose, Mapping):
            return None
        position = pose.get('position') if isinstance(pose.get('position'), Mapping) else pose
        if not isinstance(position, Mapping):
            return None
        try:
            x = float(position.get('x'))
            y = float(position.get('y', 0.0))
            z = float(position.get('z', position.get('y', 0.0)))
        except Exception:
            return None
        rotation = pose.get('rotation') if isinstance(pose.get('rotation'), Mapping) else {}
        return {'position': {'x': x, 'y': y, 'z': z}, 'rotation': {'x': float(rotation.get('x', 0.0) or 0.0), 'y': float(rotation.get('y', 0.0) or 0.0), 'z': float(rotation.get('z', 0.0) or 0.0)}, 'source': str(pose.get('source') or 'room_record')}

    def _room_xy_from_raw(self, value: Any) -> tuple[float, float] | None:
        if isinstance(value, Mapping):
            try:
                return (float(value.get('x')), float(value.get('y', value.get('z'))))
            except Exception:
                return None
        if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))) and (len(value) >= 2):
            try:
                return (float(value[0]), float(value[1]))
            except Exception:
                return None
        return None

    def _xy_from_pose_position(self, position: Any) -> tuple[float, float] | None:
        if not isinstance(position, Mapping):
            return None
        try:
            return (float(position.get('x')), float(position.get('z', position.get('y'))))
        except Exception:
            return None

    def _public_room_catalog(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for room in self.room_catalog:
            item = {'room_id': room.get('room_id'), 'room_type': room.get('room_type'), 'object_count': room.get('object_count'), 'source': room.get('source'), 'geometry_status': room.get('geometry_status'), 'reachable_status': room.get('reachable_status'), 'representative_pose': room.get('representative_pose')}
            out.append(item)
        return out

    def _room_target_handles(self) -> dict[str, Any]:
        room_ids = [str(room['room_id']) for room in self.room_catalog]
        by_type: dict[str, list[str]] = {}
        for room in self.room_catalog:
            by_type.setdefault(str(room.get('room_type') or 'unknown'), []).append(str(room['room_id']))
        return {'room_ids': room_ids, 'room_types': sorted(by_type), 'unique_room_type_aliases': sorted((room_type for room_type, ids in by_type.items() if len(ids) == 1))}

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
        return {'type': str(scope.get('type') or ''), 'label': str(scope.get('label') or ''), 'room_ids': sorted(resolved, key=lambda item: item.encode('utf-8'))}

    def _active_task_spec(self) -> Mapping[str, Any]:
        active = getattr(self, '_composite_active_task_spec', None)
        if isinstance(active, Mapping):
            return active
        return self.current_task_spec or {}

    def _resolve_room_ref(self, room_ref: str) -> tuple[dict[str, Any] | None, Optional[str], Optional[str]]:
        from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding, match_public_rooms, public_room_records
        if task_uses_evidence_binding(self.current_task_spec):
            matches = match_public_rooms(room_ref, public_room_records(self.room_catalog))
            if len(matches) == 1:
                return (matches[0], None, None)
            if len(matches) > 1:
                return (None, 'ambiguous_room_type', 'Room reference is ambiguous; use a specific public room label.')
            return (None, 'room_not_found', 'Room reference has no public match; use the room name provided in the instruction.')
        if not self.room_catalog:
            return (None, 'no_room_catalog', 'No room catalog is available for this task.')
        room_text = str(room_ref)
        for room in self.room_catalog:
            if room_text == str(room.get('room_id')):
                return (room, None, None)
        by_type = [room for room in self.room_catalog if room_text == str(room.get('room_type'))]
        if len(by_type) == 1:
            return (by_type[0], None, None)
        if len(by_type) > 1:
            return (None, 'ambiguous_room_type', "Room type '{}' has multiple instances; use one of: {}.".format(room_text, ', '.join((str(room['room_id']) for room in by_type))))
        return (None, 'room_not_found', "Room '{}' was not found. Valid rooms: {}.".format(room_text, ', '.join((str(room['room_id']) for room in self.room_catalog))))

    def _room_navigation_goal_xy(self, room: Mapping[str, Any]) -> tuple[tuple[float, float] | None, dict[str, Any], Optional[str]]:
        if self.core is None or self.core.occupancy is None:
            return (None, {'reachable_status': 'occupancy_unavailable'}, 'Occupancy map is not available.')
        from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding
        issue_pose = None if task_uses_evidence_binding(self.current_task_spec) else self._unseen_issue_observation_pose(room)
        if issue_pose is not None:
            return issue_pose
        layout_result = self._room_navigation_goal_xy_from_layout(room)
        if layout_result is not None:
            layout_xy, layout_record, layout_error = layout_result
            if layout_xy is not None:
                return (layout_xy, layout_record, None)
        else:
            layout_record = None
        target_xy = self._room_representative_xy(room)
        if target_xy is None:
            return (None, {'reachable_status': 'room_pose_invalid'}, "Room '{}' has no representative pose.".format(room.get('room_id')))
        occupancy = self.core.occupancy
        start_component = occupancy.component_at(self.core.ego_state.pose.xy)
        nearest = None
        if occupancy.is_free(target_xy):
            component = occupancy.component_at(target_xy)
            if start_component == -1 or component == -1 or start_component == component:
                nearest = (occupancy.world_to_cell(target_xy), target_xy, 0.0)
        if nearest is None:
            allowed = {int(start_component)} if int(start_component) >= 0 else None
            nearest = self._nearest_free_cell(occupancy, target_xy, allowed_components=allowed)
        if nearest is None:
            nearest = self._nearest_free_cell(occupancy, target_xy)
        if nearest is None:
            return (None, {'reachable_status': 'no_free_cell_near_room', 'representative_xy': [float(target_xy[0]), float(target_xy[1])]}, "No free occupancy cell is available near room '{}'.".format(room.get('room_id')))
        repaired_cell, repaired_xy, distance = nearest
        component = occupancy.component_at(repaired_xy)
        status = 'profile_fallback_exact' if distance <= 1e-06 else 'profile_fallback_repaired'
        layout_attempt = layout_record if isinstance(layout_record, Mapping) else None
        return (repaired_xy, {'reachable_status': status, 'representative_xy': [float(target_xy[0]), float(target_xy[1])], 'selected_goal_xy': [float(repaired_xy[0]), float(repaired_xy[1])], 'repair_distance_m': float(distance), 'selected_goal_cell': [int(repaired_cell[0]), int(repaired_cell[1])], 'room_component': int(component), 'start_component': int(start_component), 'source': 'profile_room_representative_pose', 'layout_attempt': dict(layout_attempt) if layout_attempt is not None else None}, None)

    def _unseen_issue_observation_pose(self, room: Mapping[str, Any]):
        family = self.current_task_spec.get('family_spec') if isinstance(self.current_task_spec, Mapping) else {}
        observed = family.get('observation_conditioned') if isinstance(family, Mapping) else {}
        payloads = observed.get('latent_payloads') if isinstance(observed, Mapping) else None
        if not isinstance(payloads, list):
            return None
        room_id = str(room.get('room_id') or '')
        candidates = []
        for payload in payloads:
            if not isinstance(payload, Mapping) or str(payload.get('room_id') or '') != room_id:
                continue
            issue_id = str(payload.get('latent_id') or payload.get('issue_id') or '')
            if issue_id in self._discovered_issue_ids:
                continue
            role = payload.get('object_role')
            name = self.role_map.get(str(role)) if role else None
            obj = self.core.objects.get(name) if name else None
            center = getattr(obj, 'center', None) if obj is not None else None
            if center is not None and len(center) >= 2:
                candidates.append((float(center[0]), float(center[1])))
        if not candidates:
            return None
        occupancy = self.core.occupancy
        start_component = occupancy.component_at(self.core.ego_state.pose.xy)
        best = None
        for target in candidates:
            for radius in (0.8, 1.1, 1.4):
                for angle in range(0, 360, 30):
                    radians = math.radians(angle)
                    point = (target[0] + radius * math.cos(radians), target[1] + radius * math.sin(radians))
                    nearest = self._nearest_free_cell(occupancy, point, allowed_components={int(start_component)} if int(start_component) >= 0 else None)
                    if nearest is None or math.dist(nearest[1], target) < 0.8 or math.dist(nearest[1], target) > 2.0:
                        continue
                    score = math.dist(self.core.ego_state.pose.xy, nearest[1])
                    if best is None or score < best[0]:
                        best = (score, nearest[1], nearest[0])
        if best is None:
            return None
        return (best[1], {'reachable_status': 'issue_observation_ring', 'room_component': int(start_component), 'selected_goal_cell': [int(best[2][0]), int(best[2][1])]}, None)

    def _room_navigation_goal_xy_from_layout(self, room: Mapping[str, Any]) -> tuple[tuple[float, float] | None, dict[str, Any], Optional[str]] | None:
        layout = self._room_layout_maps_for_task()
        if layout is None:
            return None
        occupancy = self.core.occupancy
        room_id = str(room.get('room_id') or '')
        room_mask = layout.room_mask(room_id)
        resolved_room_id = room_id
        if room_mask is None:
            room_type = str(room.get('room_type') or '')
            instances = list(layout.room_sem_name_to_ins_name.get(room_type) or [])
            if len(instances) == 1:
                resolved_room_id = instances[0]
                room_mask = layout.room_mask(resolved_room_id)
        if room_mask is None:
            record = {'reachable_status': 'room_mask_missing', 'source': 'floor_insseg_floor_semseg_floor_trav', 'room_id': room_id, 'available_room_ids_sample': sorted(layout.room_ins_name_to_ins_id)[:20]}
            return (None, record, "Room mask is unavailable for '{}'.".format(room_id))
        free_mask = room_mask & layout.traversable_mask()
        rows, cols = np.where(free_mask)
        if len(rows) == 0:
            record = {'reachable_status': 'room_traversability_empty', 'source': 'floor_insseg_floor_semseg_floor_trav', 'room_id': room_id, 'resolved_room_id': resolved_room_id}
            return (None, record, "Room '{}' has no traversable pixels.".format(room_id))
        start_component = occupancy.component_at(self.core.ego_state.pose.xy)
        room_center_row = float(np.mean(rows))
        room_center_col = float(np.mean(cols))
        best: tuple[float, tuple[float, float], tuple[int, int], float, int] | None = None
        considered = 0
        occupancy_free_candidates = 0
        connected_candidates = 0
        step = max(1, int(len(rows) / 600))
        for row, col in zip(rows[::step], cols[::step]):
            xy = layout.map_to_world((int(row), int(col)))
            considered += 1
            cell = occupancy.world_to_cell(xy)
            if not occupancy.is_cell_free(cell):
                continue
            occupancy_free_candidates += 1
            component = occupancy.component_at(xy)
            if int(start_component) >= 0 and int(component) >= 0 and (int(component) != int(start_component)):
                continue
            connected_candidates += 1
            center_distance_px = ((float(row) - room_center_row) ** 2 + (float(col) - room_center_col) ** 2) ** 0.5
            agent_distance = math.dist(self.core.ego_state.pose.xy, xy)
            score = center_distance_px * layout.map_resolution + 0.05 * agent_distance
            candidate = (score, xy, (int(row), int(col)), center_distance_px, int(component))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            record = {'reachable_status': 'room_not_reachable_from_occupancy', 'source': 'floor_insseg_floor_semseg_floor_trav', 'room_id': room_id, 'resolved_room_id': resolved_room_id, 'start_component': int(start_component), 'room_traversable_pixels': int(len(rows)), 'sampled_candidates': int(considered), 'occupancy_free_candidates': int(occupancy_free_candidates), 'connected_candidates': int(connected_candidates)}
            return (None, record, "Room '{}' has no reachable occupancy candidate.".format(room_id))
        _, xy, row_col, center_distance_px, component = best
        return (xy, {'reachable_status': 'validated_from_room_mask_and_traversability', 'source': 'floor_insseg_floor_semseg_floor_trav', 'room_id': room_id, 'resolved_room_id': resolved_room_id, 'selected_goal_xy': [float(xy[0]), float(xy[1])], 'selected_goal_map_rc': [int(row_col[0]), int(row_col[1])], 'room_component': int(component), 'start_component': int(start_component), 'room_traversable_pixels': int(len(rows)), 'sampled_candidates': int(considered), 'occupancy_free_candidates': int(occupancy_free_candidates), 'connected_candidates': int(connected_candidates), 'distance_from_room_centroid_m': float(center_distance_px * layout.map_resolution), 'map_resolution': float(layout.map_resolution), 'map_size': int(layout.map_size), 'source_paths': dict(layout.source_paths)}, None)

    def _room_layout_maps_for_task(self) -> Optional[RoomLayoutMaps]:
        if self._room_layout_maps is not None:
            return self._room_layout_maps
        if self._room_layout_error is not None:
            return None
        source = self.current_task_spec.get('source') if isinstance(self.current_task_spec, Mapping) else {}
        scene_model = str((source or {}).get('scene_model') or '')
        if not scene_model:
            self._room_layout_error = 'missing_scene_model'
            return None
        try:
            self._room_layout_maps = load_behavior_room_layout_maps(scene_model, assets_root=(source or {}).get('assets_root'), floor=int((source or {}).get('floor', 0) or 0))
        except Exception as exc:
            self._room_layout_error = str(exc)
            return None
        return self._room_layout_maps

    def _room_representative_xy(self, room: Mapping[str, Any]) -> tuple[float, float] | None:
        target_xy = self._room_xy_from_raw(room.get('target_xy'))
        if target_xy is not None:
            return target_xy
        pose = room.get('representative_pose')
        if isinstance(pose, Mapping):
            return self._xy_from_pose_position(pose.get('position'))
        return None

    def _refresh_current_room(self) -> None:
        if self.core is None:
            return
        room_id = self._current_room_for_pose(self.core.ego_state.pose)
        if room_id:
            self._record_room_visit(room_id)
        else:
            self.current_room = None

    def _record_room_visit(self, room_id: str) -> None:
        self.current_room = str(room_id)
        self.visited_rooms.add(str(room_id))
        self.room_visit_counts[str(room_id)] = int(self.room_visit_counts.get(str(room_id), 0)) + 1

    def _current_room_for_pose(self, pose: Any) -> Optional[str]:
        if not self.room_catalog:
            return None
        pose_dict = self._pose_to_dict(pose)
        xy = (float(pose_dict['x']), float(pose_dict['y']))
        candidates: list[tuple[float, str]] = []
        for room in self.room_catalog:
            target_xy = self._room_representative_xy(room)
            if target_xy is None:
                continue
            candidates.append((math.dist(xy, target_xy), str(room['room_id'])))
        if not candidates:
            return None
        return sorted(candidates)[0][1]

    def _private_room_id_for_xy(self, xy: Sequence[float]) -> Optional[str]:
        layout = self._room_layout_maps_for_task()
        if layout is not None:
            row, col = layout.world_to_map(xy)
            if 0 <= row < layout.map_size and 0 <= col < layout.map_size:
                instance_id = int(layout.room_ins_map[row, col])
                if instance_id > 0:
                    for room_id, candidate_id in layout.room_ins_name_to_ins_id.items():
                        if int(candidate_id) == instance_id:
                            return str(room_id)
        if not self.room_catalog:
            return None
        proxy_pose = {'x': float(xy[0]), 'y': float(xy[1]), 'z': float(getattr(self.core.ego_state.pose, 'z', 1.55)) if self.core is not None else 1.55, 'yaw': 0.0, 'pitch': 0.0}
        return self._current_room_for_pose(proxy_pose)

    def _scan_in_place_streaming(self, arguments: Mapping[str, Any]) -> dict:
        try:
            requested_num_views = int(arguments.get('num_views', 4))
        except Exception:
            requested_num_views = 0
        num_views = self._effective_scan_num_views(requested_num_views)
        trace_id = 'step_{:04d}_scan'.format(self.high_level_step)
        scan_room = self.current_room or self._current_room_for_pose(self.core.ego_state.pose)
        self.room_scan_count += 1
        start_pose = self.core.ego_state.pose.copy()
        scan_start_pose = self._level_scan_pose(start_pose)
        self.core.trace.emit(trace_id, 'action_requested', action='SCAN', num_views=requested_num_views, effective_num_views=num_views)
        if num_views == 0:
            pre = CheckResult.fail('invalid_scan_num_views', num_views=requested_num_views)
            action_result = ActionResult(action='SCAN', args={'num_views': requested_num_views}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.core.ego_state.pose.copy())
            self.core.action_results.append(action_result)
            self.core.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return {'status': 'failed', 'action': 'scan', 'num_views': requested_num_views, 'requested_num_views': requested_num_views, 'effective_num_views': 0, 'num_views_completed': 0, 'current_room': scan_room, 'room_instance': scan_room, 'room_type': self._room_type(scan_room), 'room_scan_id': 'room_scan_{:04d}'.format(self.room_scan_count), 'visible_issues': [], 'discovered_issue_ids': [], 'frame_count': 0, 'final_pose': self._agent_pose(), 'belief_updates': [], 'state_delta': {}, 'low_level_actions': [], 'error_type': 'invalid_scan_num_views', 'error_message': 'invalid_scan_num_views', 'message': 'Scan failed.'}
        pre = CheckResult.pass_(num_views=num_views, requested_num_views=requested_num_views)
        yaw_step = 2.0 * math.pi / float(num_views)
        planned_poses: list[Any] = []
        for view_index in range(num_views):
            planned_poses.append(EgoPose(x=scan_start_pose.x, y=scan_start_pose.y, z=scan_start_pose.z, yaw=normalize_angle(scan_start_pose.yaw + yaw_step * float(view_index + 1)), pitch=scan_start_pose.pitch))
        stream = self._run_streaming_scan_endpoints(arguments=arguments, trace_id=trace_id, start_pose=scan_start_pose, endpoints=planned_poses, prompt_metadata={'scan': {'current_room': scan_room, 'room_type': self._room_type(scan_room), 'num_views': num_views}})
        interrupted = bool(stream.get('interrupted'))
        executed_poses = list(stream.get('executed_poses') or [])
        executed_view_indices = list(range(len(executed_poses)))
        view_poses = executed_poses
        num_views_completed = len(executed_poses)
        if interrupted:
            belief_updates, scan_visible_ids, discovered_issues = ([], [], [])
        else:
            belief_updates, scan_visible_ids = self._observe_visible_objects_across_poses(executed_poses, source='scan')
            discovered_issues = self._discover_observation_conditioned_issues(scan_room, scan_visible_ids)
        public_discovered_issues = [] if self._observation_conditioned_visual_only() else discovered_issues
        scan_state = {'requested_num_views': requested_num_views, 'num_views': num_views, 'effective_num_views': num_views, 'num_views_completed': num_views_completed, 'degrees_per_view': math.degrees(yaw_step), 'frame_poses': executed_poses, 'frame_view_indices': executed_view_indices, 'view_poses': view_poses, 'current_room': scan_room, 'room_instance': scan_room, 'room_type': self._room_type(scan_room), 'room_scan_id': 'room_scan_{:04d}'.format(self.room_scan_count), 'visible_issues': public_discovered_issues, 'discovered_issue_ids': [issue['issue_id'] for issue in public_discovered_issues]}
        if belief_updates:
            scan_state['belief_updates'] = belief_updates
        state_delta = {'ego_pose': {'before': start_pose, 'after': self.core.ego_state.pose.copy()}, 'scan': scan_state}
        if public_discovered_issues:
            state_delta['observation_conditioned_discovery'] = {'current_room': scan_room, 'room_scan_id': scan_state.get('room_scan_id'), 'visible_issues': public_discovered_issues}
        post = CheckResult.fail('stream_interrupted') if interrupted else CheckResult.pass_(num_views=num_views, num_views_completed=num_views, degrees_per_view=math.degrees(yaw_step))
        action_result = ActionResult(action='SCAN', args={'num_views': requested_num_views, 'effective_num_views': num_views}, success=not interrupted, precondition=pre, postcondition=post, trace_id=trace_id, frames=len(executed_poses), start_pose=start_pose, end_pose=self.core.ego_state.pose.copy(), state_delta=state_delta)
        self.core.action_results.append(action_result)
        if interrupted:
            self.core.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        else:
            self.core.trace.emit(trace_id, 'action_succeeded', frames=action_result.frames, end_pose=action_result.end_pose)
        low_level_actions = self._low_level_actions_from_action_result(action_result, source='scan', target_name=None, success_override=bool(action_result.success))
        raw = {'status': 'success' if action_result.success else 'failed', 'action': 'scan', 'num_views': requested_num_views, 'requested_num_views': requested_num_views, 'effective_num_views': num_views, 'num_views_completed': int(num_views_completed), 'current_room': scan_room, 'room_instance': scan_room, 'room_type': self._room_type(scan_room), 'room_scan_id': scan_state.get('room_scan_id'), 'visible_issues': public_discovered_issues, 'discovered_issue_ids': [issue['issue_id'] for issue in public_discovered_issues], 'frame_count': int(action_result.frames), 'final_pose': self._agent_pose(), 'belief_updates': belief_updates, 'state_delta': to_jsonable(state_delta), 'low_level_actions': low_level_actions}
        err_type, err_message = self._action_error(action_result)
        if err_type:
            raw['error_type'] = err_type
            raw['error_message'] = err_message
            raw['message'] = err_message or 'Scan failed.'
        if interrupted:
            raw['_stream_interrupt_record'] = stream.get('interrupt_record')
        return raw

    def _scan_in_place(self, arguments: Mapping[str, Any]) -> dict:
        try:
            requested_num_views = int(arguments.get('num_views', 4))
        except Exception:
            requested_num_views = 0
        num_views = self._effective_scan_num_views(requested_num_views)
        trace_id = 'step_{:04d}_scan'.format(self.high_level_step)
        scan_room = self.current_room or self._current_room_for_pose(self.core.ego_state.pose)
        self.room_scan_count += 1
        if num_views > 0:
            self._set_agent_pose_from_mapping(self._pose_to_dict(self._level_scan_pose(self.core.ego_state.pose)))
        action_result = self.core.scan_in_place(num_views=num_views, trace_id=trace_id)
        state_delta = to_jsonable(action_result.state_delta or {})
        success = bool(action_result.success)
        state_delta = dict(state_delta or {})
        scan_state = dict(state_delta.get('scan') if isinstance(state_delta.get('scan'), Mapping) else {})
        scan_poses = list(scan_state.get('frame_poses') or scan_state.get('view_poses') or [])
        scan_observer = getattr(self, 'scan_rgb_observer', None)
        if success and callable(scan_observer):
            if self.renderer is None:
                raise BehaviorClosedLoopToolEnvError('scan RGB requires a renderer')
            for scan_pose in scan_poses[:8]:
                pose = self._pose_to_dict(scan_pose)
                self._sync_renderer_for_observation_pose(pose)
                self.renderer.render_at_pose(pose)
                scan_rgb = self.renderer.render_at_pose(pose)
                from scoper.benchmark.common.binding_observation_ledger import register_binding_frame
                if getattr(self, '_binding_delivery_enforced', False):
                    register_binding_frame(self, scan_rgb, self._observation_visible_object_names_at_pose(pose))
                scan_observer(scan_rgb)
            self._sync_renderer_for_observation_pose(self._agent_pose())
        if success:
            belief_updates, scan_visible_ids = self._observe_visible_objects_across_poses(scan_poses, source='scan')
            discovered_issues = self._discover_observation_conditioned_issues(scan_room, scan_visible_ids)
        else:
            belief_updates, discovered_issues = ([], [])
        public_discovered_issues = [] if self._observation_conditioned_visual_only() else discovered_issues
        scan_state.update({'requested_num_views': requested_num_views, 'effective_num_views': num_views, 'current_room': scan_room, 'room_instance': scan_room, 'room_type': self._room_type(scan_room), 'room_scan_id': 'room_scan_{:04d}'.format(self.room_scan_count), 'visible_issues': public_discovered_issues, 'discovered_issue_ids': [issue['issue_id'] for issue in public_discovered_issues]})
        state_delta['scan'] = scan_state
        if belief_updates:
            scan_state['belief_updates'] = belief_updates
            state_delta['scan'] = scan_state
        if public_discovered_issues:
            state_delta['observation_conditioned_discovery'] = {'current_room': scan_room, 'room_scan_id': scan_state.get('room_scan_id'), 'visible_issues': public_discovered_issues}
        action_result.state_delta = state_delta
        low_level_actions = self._low_level_actions_from_action_result(action_result, source='scan', target_name=None)
        error_type, error_message = self._action_error(action_result)
        raw = {'status': 'success' if success else 'failed', 'action': 'scan', 'num_views': requested_num_views, 'requested_num_views': requested_num_views, 'effective_num_views': num_views, 'num_views_completed': int(scan_state.get('num_views_completed') or (num_views if success else 0)), 'current_room': scan_room, 'room_instance': scan_room, 'room_type': self._room_type(scan_room), 'room_scan_id': scan_state.get('room_scan_id'), 'visible_issues': public_discovered_issues, 'discovered_issue_ids': [issue['issue_id'] for issue in public_discovered_issues], 'frame_count': int(action_result.frames), 'final_pose': self._agent_pose(), 'belief_updates': belief_updates, 'state_delta': state_delta, 'low_level_actions': low_level_actions}
        if error_type:
            raw['error_type'] = error_type
            raw['error_message'] = error_message
            raw['message'] = error_message or 'Scan failed.'
        return raw

    def _room_type(self, room_id: Optional[str]) -> Optional[str]:
        if not room_id:
            return None
        for room in self.room_catalog:
            if str(room.get('room_id')) == str(room_id):
                return str(room.get('room_type') or '')
        return None

    def _observation_conditioned_payloads(self) -> list[dict[str, Any]]:
        family = self.current_task_spec.get('family_spec') if isinstance(self.current_task_spec, Mapping) else {}
        obs = family.get('observation_conditioned') if isinstance(family, Mapping) else None
        if not isinstance(obs, Mapping):
            return []
        return [dict(item) for item in obs.get('latent_payloads') or [] if isinstance(item, Mapping)]

    def observation_conditioned_visible_issues(self) -> list[dict[str, Any]]:
        return [dict(issue) for issue in self._visible_issues]

    def _discover_observation_conditioned_issues(self, room_id: Optional[str], visible_object_ids: Sequence[str]=()) -> list[dict[str, Any]]:
        if not room_id:
            return []
        family = self.current_task_spec.get('family_spec') if isinstance(self.current_task_spec, Mapping) else {}
        obs = family.get('observation_conditioned') if isinstance(family, Mapping) else None
        require_visual_evidence = isinstance(obs, Mapping) and bool(obs.get('require_visual_evidence', False))
        visible_id_set = {str(item) for item in visible_object_ids}
        discovered: list[dict[str, Any]] = []
        for payload in self._observation_conditioned_payloads():
            if str(payload.get('room_id') or '') != str(room_id):
                continue
            issue_id = str(payload.get('latent_id') or payload.get('issue_id') or '')
            if not issue_id:
                continue
            if issue_id in self._discovered_issue_ids:
                continue
            object_role = payload.get('object_role')
            object_name = self.role_map.get(str(object_role)) if object_role else None
            if require_visual_evidence and (not object_name or object_name not in visible_id_set):
                continue
            issue = {'issue_id': issue_id, 'room_id': str(room_id), 'issue_type': str(payload.get('issue_type') or 'unknown'), 'object_role': payload.get('object_role'), 'object_handle': payload.get('object_role'), 'suggested_affordances': list(payload.get('suggested_affordances') or []), 'public_after': 'scan'}
            if payload.get('repair_plan_template'):
                issue['repair_plan_template'] = payload.get('repair_plan_template')
            discovered.append(issue)
            self._visible_issues.append(issue)
            self._discovered_issue_ids.add(issue_id)
        return discovered

    def _public_visible_issues(self) -> list[dict[str, Any]]:
        if self._observation_conditioned_visual_only():
            return []
        return [dict(issue) for issue in self._visible_issues]

    def _public_discovered_issue_ids(self) -> list[str]:
        return [] if self._observation_conditioned_visual_only() else sorted(self._discovered_issue_ids)

    def _observation_conditioned_visual_only(self) -> bool:
        family = self.current_task_spec.get('family_spec') if isinstance(self.current_task_spec, Mapping) else {}
        obs = family.get('observation_conditioned') if isinstance(family, Mapping) else None
        return isinstance(obs, Mapping) and obs.get('public_discovery_mode') == 'visual_only'

    def _look_for_streaming(self, arguments: Mapping[str, Any], target_name: Optional[str]) -> dict:
        trace_id = 'step_{:04d}_look_for'.format(self.high_level_step)
        self.core.trace.emit(trace_id, 'action_requested', action='LOOK_FOR', target=target_name)
        start_pose = self.core.ego_state.pose.copy()
        if not target_name or target_name not in self.core.objects:
            pre = CheckResult.fail('target_not_found', target=target_name)
            action_result = ActionResult(action='LOOK_FOR', args={'target': target_name}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.core.ego_state.pose.copy())
            self.core.action_results.append(action_result)
            self.core.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            raw = self._look_for_raw_execution(arguments, target_name, action_result)
            raw['state_delta'] = {}
            raw['low_level_actions'] = []
            return raw
        self.core.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists'])
        pre = self.core.make_precondition_checker().check_target_exists(target_name)
        self.core.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        if not pre.allowed:
            action_result = ActionResult(action='LOOK_FOR', args={'target': target_name}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.core.ego_state.pose.copy())
            self.core.action_results.append(action_result)
            self.core.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            raw = self._look_for_raw_execution(arguments, target_name, action_result)
            raw['state_delta'] = {}
            raw['low_level_actions'] = []
            return raw
        target_obj = self.core.objects[target_name]
        end_pose = pose_looking_at_object(start_pose, target_obj)
        frame_poses = self._interpolate_pose_turn(start_pose, end_pose)
        stream = self._run_streaming_pose_checkpoints(tool_name='look_for', arguments=arguments, trace_id=trace_id, poses=[start_pose, *frame_poses], prompt_metadata={'look_for': {'target': arguments.get('target')}}, privileged_metadata={'target_object_id': target_name, 'target_position': tuple(target_obj.center)})
        interrupted = bool(stream.get('interrupted'))
        executed_poses = list(stream.get('executed_poses') or [])
        post = CheckResult.fail('stream_interrupted') if interrupted else self.core.make_precondition_checker().check_facing(target_name, threshold_degrees=1.0)
        success = bool(post.allowed) and (not interrupted)
        if not success and (not interrupted) and ('postcondition_failed' not in post.reasons):
            post = CheckResult.fail(['postcondition_failed', *post.reasons], **post.metrics)
        state_delta = {'ego_pose': {'before': start_pose, 'after': self.core.ego_state.pose.copy()}, 'look_for': {'target': target_name, 'frame_poses': [pose.copy() for pose in executed_poses], 'target_position': tuple(target_obj.center)}}
        action_result = ActionResult(action='LOOK_FOR', args={'target': target_name}, success=success, precondition=pre, postcondition=post, trace_id=trace_id, frames=len(executed_poses), start_pose=start_pose, end_pose=self.core.ego_state.pose.copy(), state_delta=state_delta)
        self.core.action_results.append(action_result)
        if success:
            self.core.trace.emit(trace_id, 'action_succeeded', frames=action_result.frames, end_pose=action_result.end_pose)
        else:
            self.core.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        raw = self._look_for_raw_execution(arguments, target_name, action_result)
        raw['state_delta'] = to_jsonable(state_delta)
        raw['frame_count'] = int(action_result.frames)
        raw['low_level_actions'] = self._low_level_actions_from_action_result(action_result, source='look_for', target_name=target_name, success_override=success)
        if interrupted:
            raw['_stream_interrupt_record'] = stream.get('interrupt_record')
        return raw

    def _look_for_raw_execution(self, arguments: Mapping[str, Any], target_name: Optional[str], action_result: Any) -> dict:
        error_type, error_message = self._action_error(action_result)
        visible_ids = self._visible_object_names_at_pose(self.core.ego_state.pose)
        post_metrics = action_result.postcondition.metrics if action_result.postcondition is not None else {}
        orientation_success = bool(action_result.success)
        object_visible = bool(target_name and target_name in visible_ids)
        success = orientation_success and object_visible
        belief_updates = self._observe_visible_objects(visible_ids, source='look_for', pose=self.core.ego_state.pose) if orientation_success else []
        if orientation_success and (not object_visible):
            error_type = 'target_not_visible'
            error_message = "Target '{}' is not visible after look_for.".format(target_name)
        raw = {'status': 'success' if success else 'failed', 'action': 'look_for', 'target': arguments.get('target'), 'object_id': target_name, 'orientation_success': orientation_success, 'object_visible': object_visible, 'visible_object_ids': visible_ids, 'visible_object_roles': self._roles_for_object_names(visible_ids), 'belief_updates': belief_updates, 'final_pose': self._agent_pose(), 'desired_yaw': post_metrics.get('target_yaw'), 'desired_horizon': self._agent_pose().get('pitch'), 'frame_count': int(action_result.frames)}
        if target_name in self.core.objects:
            raw['target_position'] = list(self.core.objects[target_name].center)
        if error_type:
            raw['error_type'] = error_type
            raw['error_message'] = error_message
        return raw

    def _low_level_actions_from_action_result(self, action_result: Any, *, source: str, target_name: Optional[str], success_override: Optional[bool]=None) -> List[dict]:
        state_delta = to_jsonable(action_result.state_delta or {})
        if source == 'scan':
            source_state = state_delta.get('scan') if isinstance(state_delta.get('scan'), Mapping) else {}
            poses = list(source_state.get('frame_poses') or [])
            view_indices = list(source_state.get('frame_view_indices') or [])
            degrees = source_state.get('degrees_per_view')
            executed_action = 'RotateRight'
        else:
            source_state = state_delta.get(source) if isinstance(state_delta.get(source), Mapping) else {}
            poses = list(source_state.get('frame_poses') or [])
            view_indices = []
            degrees = None
            executed_action = 'LookFor'
        records: List[dict] = []
        for index, pose in enumerate(poses):
            pose_dict = self._pose_to_dict(pose)
            visible_ids = self._visible_object_names_at_pose(pose_dict)
            record = {'action': source, 'executed_action': executed_action, 'label': source if target_name is None else '{} {}'.format(source, target_name), 'success': bool(action_result.success if success_override is None else success_override), 'agent_pose': pose_dict, 'frame_index': index, 'source': source, 'metadata_visible_object_ids': visible_ids, 'metadata_visible_roles': self._roles_for_object_names(visible_ids), 'visual_discoverability': self.visual_discoverability_report()}
            if target_name is not None:
                record['object_id'] = target_name
                record['target_visible'] = target_name in visible_ids
            if index < len(view_indices):
                record['view_index'] = int(view_indices[index])
            if degrees is not None:
                record['degrees'] = float(degrees)
            if source == 'scan':
                record['current_room'] = source_state.get('current_room')
                record['room_instance'] = source_state.get('room_instance')
                record['room_type'] = source_state.get('room_type')
                record['room_scan_id'] = source_state.get('room_scan_id')
            records.append(record)
        return records

    def _low_level_actions_from_pose_navigation(self, action_result: Any, *, source: str, room: Mapping[str, Any], success_override: Optional[bool]=None) -> List[dict]:
        poses = self._frame_poses_from_trace(action_result.trace_id)
        records: List[dict] = []
        for index, pose in enumerate(poses):
            pose_dict = self._pose_to_dict(pose)
            visible_ids = self._visible_object_names_at_pose(pose_dict)
            records.append({'action': source, 'executed_action': 'MoveAhead', 'label': '{} {}'.format(source, room.get('room_id')), 'success': bool(action_result.success if success_override is None else success_override), 'agent_pose': pose_dict, 'frame_index': index, 'source': source, 'room_id': room.get('room_id'), 'room_type': room.get('room_type'), 'current_room': room.get('room_id'), 'metadata_visible_object_ids': visible_ids, 'metadata_visible_roles': self._roles_for_object_names(visible_ids), 'visual_discoverability': self.visual_discoverability_report()})
        return records

    def _low_level_actions_from_object_navigation(self, action_result: Any, *, target_name: str, target_ref: Any, navigation_mode: str, success_override: Optional[bool]=None) -> List[dict]:
        poses = self._frame_poses_from_trace(action_result.trace_id)
        records: List[dict] = []
        for index, pose in enumerate(poses):
            pose_dict = self._pose_to_dict(pose)
            visible_ids = self._visible_object_names_at_pose(pose_dict)
            records.append({'action': 'navigate_to', 'executed_action': 'MoveAhead', 'label': 'navigate_to {}'.format(target_ref or target_name), 'success': bool(action_result.success if success_override is None else success_override), 'agent_pose': pose_dict, 'frame_index': index, 'source': 'navigate_to', 'object_id': target_name, 'target': target_ref, 'navigation_mode': navigation_mode, 'target_visible': target_name in set(visible_ids), 'metadata_visible_object_ids': visible_ids, 'metadata_visible_roles': self._roles_for_object_names(visible_ids), 'visual_discoverability': self.visual_discoverability_report()})
        return records

    def _frame_poses_from_trace(self, trace_id: str) -> list[Any]:
        events = list(getattr(getattr(self.core, 'trace', None), 'events', []) or [])
        poses: list[Any] = []
        for event in events:
            if getattr(event, 'trace_id', None) != trace_id:
                continue
            if getattr(event, 'event', None) != 'frame_pose':
                continue
            payload = getattr(event, 'payload', {}) or {}
            pose = payload.get('pose') if isinstance(payload, Mapping) else None
            if pose is not None:
                poses.append(pose)
        return poses

    def _action_error(self, action_result: Any) -> tuple[Optional[str], Optional[str]]:
        if bool(action_result.success):
            return (None, None)
        reasons = list(getattr(action_result.precondition, 'reasons', []) or [])
        postcondition = getattr(action_result, 'postcondition', None)
        if not reasons and postcondition is not None:
            reasons = list(getattr(postcondition, 'reasons', []) or [])
        if not reasons:
            return ('action_failed', 'Action did not succeed.')
        return (str(reasons[0]), '; '.join((str(reason) for reason in reasons)))

    def _interpolate_pose_turn(self, start_pose: Any, end_pose: Any) -> list[EgoPose]:
        yaw_delta = normalize_angle(float(end_pose.yaw) - float(start_pose.yaw))
        pitch_delta = float(end_pose.pitch) - float(start_pose.pitch)
        turn_spacing = float(getattr(self.core.config.navigation, 'turn_spacing_degrees', 10.0) or 10.0)
        step_radians = math.radians(max(turn_spacing, 0.001))
        min_steps = 2 if abs(yaw_delta) > 1e-09 or abs(pitch_delta) > 1e-09 else 1
        steps = max(min_steps, int(math.ceil(abs(yaw_delta) / step_radians)), int(math.ceil(abs(pitch_delta) / step_radians)))
        poses: list[EgoPose] = []
        for step in range(1, steps + 1):
            t = float(step) / float(steps)
            poses.append(EgoPose(x=float(start_pose.x), y=float(start_pose.y), z=float(start_pose.z), yaw=normalize_angle(float(start_pose.yaw) + yaw_delta * t), pitch=float(start_pose.pitch) + pitch_delta * t))
        return poses

    def _answer_human_question(self, arguments: Mapping[str, Any]) -> tuple[bool, str, Optional[str], Optional[str], Optional[dict]]:
        question_id = arguments.get('question_id')
        value = arguments.get('value')
        if not isinstance(question_id, str) or not question_id:
            msg = "answer requires a non-empty 'question_id'."
            return (False, 'answer failed: {}'.format(msg), 'missing_argument', msg, None)
        if value is None:
            msg = "answer requires a 'value'."
            return (False, 'answer failed: {}'.format(msg), 'missing_argument', msg, None)
        state = self._ensure_human_qa_state()
        pending_by_id = state.setdefault('pending', {})
        pending = pending_by_id.get(question_id)
        if pending is None:
            for message in self.human_messages:
                if message.get('type') == 'qa' and message.get('id') == question_id and (not message.get('answered')):
                    pending = message
                    break
        if pending is None:
            msg = "No pending human QA with id '{}'.".format(question_id)
            return (False, 'answer failed: {}'.format(msg), 'unknown_question', msg, None)
        answer = {'question_id': question_id, 'value': str(value)}
        state.setdefault('answers', {})[question_id] = answer
        state.setdefault('answered_steps', {})[question_id] = int(self.high_level_step)
        pending_by_id.pop(question_id, None)
        self._answered_human_question_ids.add(question_id)
        if isinstance(pending, dict):
            pending['answered'] = True
            pending['answer'] = str(value)
        self._sync_public_human_state()
        return (True, 'Answer recorded for {}.'.format(question_id), None, None, {'human_answer': answer, 'question': pending})

    def _empty_human_qa_state(self) -> dict[str, Any]:
        return {'pending': {}, 'answers': {}, 'inserts': {}, 'evaluator_keys': {}, 'posted_steps': {}, 'checkpoint_steps': {}, 'answered_steps': {}, 'qa_overlay_resolver_failures': [], 'unanswered': []}

    def _ensure_human_qa_state(self) -> dict[str, Any]:
        if not isinstance(getattr(self, '_human_qa_state', None), dict):
            self._human_qa_state = self._empty_human_qa_state()
        for key in ('pending', 'answers', 'inserts', 'evaluator_keys', 'posted_steps', 'checkpoint_steps'):
            if not isinstance(self._human_qa_state.get(key), dict):
                self._human_qa_state[key] = {}
        if not isinstance(self._human_qa_state.get('unanswered'), list):
            self._human_qa_state['unanswered'] = []
        return self._human_qa_state

    def _qa_overlay_rejected_stop_transition(self, scheduler_events: Sequence[Mapping[str, Any]]) -> dict:
        self.high_level_step += 1
        goal_result = self.check_goal(model_stopped=False)
        return {'tool_name': 'stop', 'arguments': {}, 'reasoning': '', 'success': False, 'feedback': 'A situated question is pending; answer it before requesting stop again.', 'frames': 0, 'num_low_level_steps': 0, 'total_low_level_steps': self.total_low_level_steps, 'env_feedback': {'error_type': 'qa_pending', 'error_message': None}, 'high_level_step': self.high_level_step, 'task_action_step': self.task_action_step, 'agent_pose': self._agent_pose(), 'current_room': self.current_room, 'visited_rooms': sorted(self.visited_rooms), 'held_objects': self._held_objects(), 'held_object_handles': self._public_held_object_handles(), 'state_delta': None, 'goal_result': goal_result, 'done': False, 'done_reason': None, 'observation': self._observation(), 'raw_execution': {'action': 'stop', 'action_executed': False, 'status': 'qa_pending'}, 'binding': None, 'bindings': [], 'low_level_actions': [], 'episode_metrics': self._episode_metrics(), 'scheduler_intercept': [{'name': str(event.get('name') or 'qa_overlay_event'), 'phase': 'intercept', 'payload': dict(event.get('payload') or {})} for event in scheduler_events]}

    def _sync_public_human_state(self) -> None:
        state = self._ensure_human_qa_state()
        posted_steps = state.get('posted_steps') or {}
        messages: list[tuple[int, str, dict]] = []
        for collection_name in ('pending', 'inserts'):
            collection = state.get(collection_name) or {}
            for item_id, item in collection.items():
                entry = dict(item) if isinstance(item, Mapping) else {'id': item_id}
                entry.setdefault('id', item_id)
                entry.setdefault('type', 'qa' if collection_name == 'pending' else 'insert')
                messages.append((int(posted_steps.get(item_id, 10 ** 9)), str(item_id), entry))
        if messages or state.get('answers'):
            messages.sort(key=lambda row: (row[0], row[1]))
            self.human_messages = [entry for _, _, entry in messages]
            answers = []
            for question_id, answer in (state.get('answers') or {}).items():
                entry = dict(answer) if isinstance(answer, Mapping) else {'value': answer}
                entry.setdefault('question_id', question_id)
                answers.append(entry)
            self.human_answers = answers

    def _human_events_from_task(self, task_spec: Mapping[str, Any]) -> List[dict]:
        family = task_spec.get('family_spec')
        if not isinstance(family, Mapping):
            return []
        human = family.get('human')
        if not isinstance(human, Mapping):
            return []
        events = human.get('human_events')
        if not isinstance(events, list):
            events = human.get('human_events_template')
        return [dict(event) for event in events or [] if isinstance(event, Mapping)]

    def _activate_human_events(self, transition: Mapping[str, Any]) -> tuple[List[dict], List[dict]]:
        posted: List[dict] = []
        scheduler_events: List[dict] = []
        if not transition.get('success'):
            return (posted, scheduler_events)
        for event in self._human_event_templates:
            event_id = event.get('id')
            if not isinstance(event_id, str) or event_id in self._activated_human_event_ids:
                continue
            if not self._human_event_due(event, transition):
                continue
            message = self._public_human_message(event)
            if message:
                self.human_messages.append(message)
                posted.append(dict(message))
                scheduler_events.append(self._human_scheduler_event(event, message))
            self._activated_human_event_ids.add(event_id)
        return (posted, scheduler_events)

    def _activate_step_human_events(self) -> tuple[List[dict], List[dict]]:
        posted: List[dict] = []
        scheduler_events: List[dict] = []
        for event in self._human_event_templates:
            event_id = event.get('id')
            if not isinstance(event_id, str) or event_id in self._activated_human_event_ids:
                continue
            activation = event.get('activation')
            if not isinstance(activation, Mapping):
                continue
            if activation.get('type') != 'high_level_step':
                continue
            after_step = int(activation.get('after_step', 0) or 0)
            if self.high_level_step <= after_step:
                continue
            if not self._visible_roles_satisfied(event.get('requires_visible_roles') or []):
                continue
            message = self._public_human_message(event)
            if message:
                self.human_messages.append(message)
                posted.append(dict(message))
                scheduler_events.append(self._human_scheduler_event(event, message))
            self._activated_human_event_ids.add(event_id)
        return (posted, scheduler_events)

    def _human_event_due(self, event: Mapping[str, Any], transition: Mapping[str, Any]) -> bool:
        activation = event.get('activation')
        if not isinstance(activation, Mapping):
            return False
        if activation.get('type') != 'milestone':
            return False
        if activation.get('milestone') != 'after_successful_tool_on_role':
            return False
        if activation.get('tool') and transition.get('tool_name') != activation.get('tool'):
            return False
        role = activation.get('role')
        if role is not None and str((transition.get('arguments') or {}).get('target')) != str(role):
            return False
        return self._visible_roles_satisfied(event.get('requires_visible_roles') or [])

    def _visible_roles_satisfied(self, role_names: Sequence[str]) -> bool:
        if not role_names:
            return True
        visible_names = set(self._visible_object_names_at_pose(self.core.ego_state.pose))
        for role in role_names:
            obj_name = self.role_map.get(str(role), str(role))
            if obj_name not in visible_names:
                return False
        return True

    def _public_human_message(self, event: Mapping[str, Any]) -> Optional[dict]:
        event_type = event.get('type')
        event_id = event.get('id')
        if not isinstance(event_id, str) or not event_id:
            return None
        if event_type == 'qa':
            question = event.get('question')
            if question is None:
                question = event.get('question_template')
            return {'type': 'qa', 'id': event_id, 'question_type': event.get('question_type'), 'question': self._format_human_template(str(question or '')), 'answer_set': list(event.get('answer_set') or []), 'requires_visible_roles': list(event.get('requires_visible_roles') or []), 'answered': False}
        if event_type == 'insert':
            instruction = event.get('instruction')
            if instruction is None:
                instruction = event.get('instruction_template')
            return {'type': 'insert', 'id': event_id, 'instruction': self._format_human_template(str(instruction or '')), 'priority': event.get('priority'), 'goal_predicates': list(event.get('goal_predicates') or [])}
        return None

    def _human_scheduler_event(self, event: Mapping[str, Any], message: Mapping[str, Any]) -> dict:
        event_type = str(event.get('type') or '')
        activation = event.get('activation') if isinstance(event.get('activation'), Mapping) else {}
        payload = {'id': event.get('id'), 'after_step': int(self.high_level_step), 'activation': dict(activation or {}), 'requires_visible_roles': list(event.get('requires_visible_roles') or [])}
        if event_type == 'qa':
            payload.update({'question': message.get('question'), 'question_type': event.get('question_type'), 'answer_set': list(event.get('answer_set') or []), 'evaluator_answer': event.get('evaluator_answer')})
            return {'name': 'human_qa_posted', 'phase': 'post_step', 'payload': payload}
        if event_type == 'insert':
            payload.update({'instruction': message.get('instruction'), 'priority': event.get('priority'), 'goal_predicates': list(event.get('goal_predicates') or [])})
            return {'name': 'human_insert_posted', 'phase': 'post_step', 'payload': payload}
        return {'name': 'human_event_posted', 'phase': 'post_step', 'payload': payload}

    def _format_human_template(self, template: str) -> str:
        if not template:
            return ''
        values: dict[str, str] = {}
        for role_name, role in (self.current_task_spec.get('roles') or {}).items():
            if not isinstance(role, Mapping):
                continue
            object_type = str(role.get('object_type') or role.get('category') or role_name)
            values['{}_type_lower'.format(role_name)] = object_type.replace('_', ' ').lower()
        try:
            return template.format(**values)
        except Exception:
            return template

    def _agent_pose(self) -> dict:
        return self._pose_to_dict(self.core.ego_state.pose)

    def _held_objects(self) -> List[str]:
        held = self.core.ego_state.held_object_name
        return [held] if held else []

    def _public_held_object_handles(self) -> List[str]:
        held = self.core.ego_state.held_object_name
        if not held:
            return []
        obj = self.core.objects.get(held)
        category = getattr(obj, 'category', None) if obj is not None else None
        return [self._catalog_tracker.alias_for(str(held), str(category or 'object'))]

    def _object_catalog(self, visible_object_ids: Optional[Sequence[str]]=None) -> List[dict]:
        visible_from_current = set((str(item) for item in visible_object_ids)) if visible_object_ids is not None else set(self._visible_object_names_at_pose(self.core.ego_state.pose))
        catalog = []
        for name, obj in self.core.objects.items():
            catalog.append({'name': name, 'objectId': name, 'alias': name, 'category': obj.category, 'objectType': obj.category, 'aabb_center': list(obj.center), 'aabb_extent': list(obj.extent), 'root_position': list(obj.root_position) if getattr(obj, 'root_position', None) is not None else None, 'root_orientation_xyzw': list(obj.root_orientation_xyzw) if getattr(obj, 'root_orientation_xyzw', None) is not None else None, 'root_yaw': getattr(obj, 'root_yaw', None), 'pose_source': getattr(obj, 'pose_source', None), 'position': {'x': obj.center[0], 'y': obj.center[1], 'z': obj.center[2]}, 'abilities': list(obj.abilities), 'states': dict(obj.states or {}), 'visible': True, 'visible_from_current_pose': str(name) in visible_from_current})
        return catalog

    def _metadata_exposes(self, name: str) -> bool:
        if name == 'visible_catalog':
            return bool(getattr(self.metadata_config, 'expose_visible_catalog', True))
        if self.metadata_mode == 'privileged':
            return True
        attr = 'expose_{}'.format(name)
        return bool(getattr(self.metadata_config, attr, False))

    def _log_privileged_metadata(self) -> bool:
        return bool(getattr(self.metadata_config, 'log_privileged_metadata', True))

    def _prompt_safe_object_catalog(self, visible_object_ids: Sequence[str]) -> List[dict]:
        if self.core is None:
            return []
        if not self._metadata_exposes('visible_catalog'):
            return []
        return build_prompt_safe_catalog(objects=self.core.objects, pose=self._agent_pose(), visible_object_ids=visible_object_ids, tracker=self._catalog_tracker, include_raw_ids=self._metadata_exposes('raw_object_ids'), include_states=self._metadata_exposes('states'), include_bbox=self._metadata_exposes('bbox'))

    def _pose_to_dict(self, pose: Any) -> dict:
        if isinstance(pose, Mapping):
            return {'x': float(pose.get('x', 0.0)), 'y': float(pose.get('y', 0.0)), 'z': float(pose.get('z', 1.55)), 'yaw': float(pose.get('yaw', 0.0)), 'pitch': float(pose.get('pitch', 0.0))}
        return {'x': float(getattr(pose, 'x', 0.0)), 'y': float(getattr(pose, 'y', 0.0)), 'z': float(getattr(pose, 'z', 1.55)), 'yaw': float(getattr(pose, 'yaw', 0.0)), 'pitch': float(getattr(pose, 'pitch', 0.0))}

    def _pose_looking_at_center(self, pose: Any, center: Sequence[float]) -> dict:
        pose_dict = self._pose_to_dict(pose)
        dx = float(center[0]) - pose_dict['x']
        dy = float(center[1]) - pose_dict['y']
        dz = float(center[2]) - pose_dict['z']
        horizontal = math.hypot(dx, dy)
        if horizontal > 1e-09:
            pose_dict['yaw'] = math.atan2(dy, dx)
            pose_dict['pitch'] = math.atan2(dz, horizontal)
        return pose_dict

    def _set_agent_orientation(self, pose: Mapping[str, Any]) -> None:
        if self.core is None:
            return
        try:
            self.core.ego_state.pose.yaw = float(pose.get('yaw', self.core.ego_state.pose.yaw))
            self.core.ego_state.pose.pitch = float(pose.get('pitch', self.core.ego_state.pose.pitch))
        except Exception:
            return

    def _visible_object_names_at_pose(self, pose: Any) -> List[str]:
        if self.core is None:
            return []
        if self._strict_rendered_visibility_gate_enabled():
            self._last_visibility_mode = 'symbolic_fov'
            return self._symbolic_visible_object_names_at_pose(pose)
        rendered_visible = self._rendered_visible_object_names_at_pose(pose)
        if rendered_visible is not None:
            return rendered_visible
        self._last_visibility_mode = 'symbolic_fov'
        return self._symbolic_visible_object_names_at_pose(pose)

    def _prompt_visible_object_names_at_pose(self, pose: Any) -> List[str]:
        if self._strict_rendered_visibility_gate_enabled():
            self._last_visibility_mode = 'symbolic_fov'
            return self._symbolic_visible_object_names_at_pose(pose)
        return self._visible_object_names_at_pose(pose)

    def _observation_visible_object_names_at_pose(self, pose: Any) -> List[str]:
        if not self.capture_rendered_observation_evidence and self.allow_symbolic_public_observation_evidence:
            return self._prompt_visible_object_names_at_pose(pose)
        if not self.capture_rendered_observation_evidence:
            self._last_visibility_mode = 'unobservable'
            self._last_render_visibility_status = 'segmentation_not_requested_by_execution_profile'
            self._last_render_visibility_error = None
            return []
        rendered = self._rendered_visible_object_names_at_pose(pose, min_pixels=self.action_visibility_min_pixels, force=True)
        if rendered is None:
            raise BehaviorClosedLoopToolEnvError('Rendered observation evidence is required but unavailable: {}.'.format(self._last_render_visibility_status or 'unknown'))
        return rendered

    def _strict_rendered_visibility_gate_enabled(self) -> bool:
        return str(self.action_visibility_mode or '') == 'rendered_required' or bool(self.require_rendered_held_pose)

    def _symbolic_visible_object_names_at_pose(self, pose: Any) -> List[str]:
        if self.core is None:
            return []
        pose_dict = self._pose_to_dict(pose)
        visible = [name for name, obj in self.core.objects.items() if self._object_visible_from_pose(obj, pose_dict)]
        return sorted(visible)

    def _rendered_visible_object_names_at_pose(self, pose: Any, *, min_pixels: int=_RENDER_VISIBILITY_MIN_PIXELS, force: bool=False) -> Optional[List[str]]:
        if self.core is None:
            self._last_render_visibility_status = 'core_absent'
            self._last_render_visibility_error = None
            return None
        if self.renderer is None:
            self._last_render_visibility_status = 'renderer_absent'
            self._last_render_visibility_error = None
            return None
        if not self.use_rendered_visibility and (not force):
            self._last_render_visibility_status = 'disabled_by_config'
            self._last_render_visibility_error = None
            return None
        self._sync_renderer_for_observation_pose(self._pose_to_dict(pose))
        core_object_names = [str(name) for name in self.core.objects.keys()]
        query_object_names: set[str] = set(core_object_names)
        for name in core_object_names:
            query_object_names.update(self._visibility_equivalent_names(name))
        object_names = sorted(query_object_names)
        core_object_name_set = set(core_object_names)
        visible: Any = None
        mode = 'rendered_segmentation'
        query = getattr(self.renderer, 'visible_object_names_at_pose', None)
        if callable(query):
            try:
                visible = query(pose, object_names=object_names, min_pixels=max(1, int(min_pixels)), modality=self.rendered_visibility_modality)
            except TypeError as exc:
                try:
                    visible = query(pose, object_names=object_names, min_pixels=max(1, int(min_pixels)))
                except TypeError:
                    try:
                        visible = query(pose)
                    except Exception as inner_exc:
                        self._last_render_visibility_status = 'rendered_visibility_error'
                        self._last_render_visibility_error = str(inner_exc or exc)
                        return None
                except Exception as inner_exc:
                    self._last_render_visibility_status = 'rendered_visibility_error'
                    self._last_render_visibility_error = str(inner_exc or exc)
                    return None
            except Exception as exc:
                self._last_render_visibility_status = 'rendered_visibility_error'
                self._last_render_visibility_error = str(exc)
                return None
        elif callable(getattr(self.renderer, 'render_visibility_at_pose', None)):
            visibility_query = self.renderer.render_visibility_at_pose
            try:
                payload = visibility_query(pose, object_names=object_names, min_pixels=max(1, int(min_pixels)), modality=self.rendered_visibility_modality)
            except TypeError as exc:
                try:
                    payload = visibility_query(pose)
                except Exception as inner_exc:
                    self._last_render_visibility_status = 'rendered_visibility_error'
                    self._last_render_visibility_error = str(inner_exc or exc)
                    return None
            except Exception as exc:
                self._last_render_visibility_status = 'rendered_visibility_error'
                self._last_render_visibility_error = str(exc)
                return None
            if isinstance(payload, Mapping):
                visible = payload.get('visible_object_ids')
                mode = str(payload.get('mode') or mode)
            else:
                visible = payload
        else:
            self._last_render_visibility_status = 'rendered_visibility_api_absent'
            self._last_render_visibility_error = None
            return None
        if isinstance(visible, Mapping):
            mode = str(visible.get('mode') or mode)
            visible = visible.get('visible_object_ids')
        if visible is None:
            self._last_render_visibility_status = 'rendered_visibility_invalid'
            self._last_render_visibility_error = 'Renderer visibility query returned no visible_object_ids.'
            return None
        if isinstance(visible, str) or not isinstance(visible, Iterable):
            self._last_render_visibility_status = 'rendered_visibility_invalid'
            self._last_render_visibility_error = 'Renderer visibility query returned a non-iterable payload.'
            return None
        visible_names = list(visible)
        self._last_visibility_mode = mode
        self._last_render_visibility_status = 'rendered_segmentation_available'
        self._last_render_visibility_error = None
        normalized_visible: set[str] = set()
        for visible_name in visible_names:
            visible_text = str(visible_name)
            if visible_text in core_object_name_set:
                normalized_visible.add(visible_text)
                continue
            for core_name in core_object_names:
                if self._target_visibility_matches(core_name, [visible_text]):
                    normalized_visible.add(core_name)
        return sorted(normalized_visible)

    def _object_visible_from_pose(self, obj: Any, pose: Any) -> bool:
        pose_dict = self._pose_to_dict(pose)
        dx = float(obj.center[0]) - pose_dict['x']
        dy = float(obj.center[1]) - pose_dict['y']
        dz = float(obj.center[2]) - pose_dict['z']
        horizontal = math.hypot(dx, dy)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > _SYMBOLIC_VISIBILITY_MAX_DISTANCE:
            return False
        if horizontal <= 1e-06:
            return True
        target_yaw = math.atan2(dy, dx)
        yaw_error = abs(_normalize_angle(target_yaw - pose_dict['yaw']))
        if yaw_error > math.radians(_SYMBOLIC_HORIZONTAL_FOV_DEGREES * 0.5):
            return False
        target_pitch = math.atan2(dz, horizontal)
        pitch_error = abs(target_pitch - pose_dict['pitch'])
        return pitch_error <= math.radians(_SYMBOLIC_VERTICAL_FOV_DEGREES * 0.5)

    def _roles_for_object_names(self, object_names: Sequence[str]) -> List[str]:
        inverse: dict[str, list[str]] = {}
        for role_name, object_name in self.role_map.items():
            inverse.setdefault(str(object_name), []).append(str(role_name))
        roles: list[str] = []
        for object_name in object_names:
            roles.extend(inverse.get(str(object_name), []))
        return sorted(roles)

def _normalize_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

def _render_safe_name(instance_name: str) -> str:
    return str(instance_name).replace('.', '_').replace('-', '_')

def _renderer_pose_match_candidates(name: str, obj: Any) -> list[str]:
    candidates = [_render_safe_name(name)]
    states = getattr(obj, 'states', {}) or {}
    if isinstance(states, Mapping):
        for key in ('source_scene_object_id', 'source_object_id', 'renderer_name', 'object_id'):
            value = states.get(key)
            if value and (not str(value).startswith('spawnable:')):
                candidates.append(_render_safe_name(str(value)))
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out

def _update_aabb_object_from_runtime_pose(obj: Any, record: Mapping[str, Any]) -> bool:
    try:
        center = tuple((float(v) for v in (record.get('aabb_center') or [])[:3]))
        extent = tuple((float(v) for v in (record.get('aabb_extent') or [])[:3]))
        root_position = tuple((float(v) for v in (record.get('root_position') or [])[:3]))
        root_orientation = tuple((float(v) for v in (record.get('root_orientation_xyzw') or [])[:4]))
    except Exception:
        return False
    if len(center) != 3 or len(extent) != 3 or len(root_position) != 3 or (len(root_orientation) != 4):
        return False
    obj.center = center
    obj.extent = extent
    obj.root_position = root_position
    obj.root_orientation_xyzw = root_orientation
    root_yaw = record.get('root_yaw')
    obj.root_yaw = None if root_yaw is None else float(root_yaw)
    obj.pose_source = 'renderer_readback'
    return True

def _update_core_object_from_pose_mutation(obj: Any, mutation: Mapping[str, Any]) -> None:
    center = mutation.get('actual_aabb_center') or mutation.get('actual_center')
    extent = mutation.get('actual_aabb_extent')
    root_position = mutation.get('actual_root_position')
    root_orientation = mutation.get('actual_root_orientation_xyzw')
    if center is not None:
        try:
            obj.center = tuple((float(v) for v in center[:3]))
        except Exception:
            pass
        else:
            if root_position is None:
                obj.root_position = None
                obj.root_orientation_xyzw = None
                obj.root_yaw = None
    if extent is not None:
        try:
            obj.extent = tuple((float(v) for v in extent[:3]))
        except Exception:
            pass
    if root_position is not None:
        try:
            obj.root_position = tuple((float(v) for v in root_position[:3]))
        except Exception:
            pass
    if root_orientation is not None:
        try:
            obj.root_orientation_xyzw = tuple((float(v) for v in root_orientation[:4]))
        except Exception:
            pass
    if mutation.get('actual_root_yaw') is not None:
        try:
            obj.root_yaw = float(mutation['actual_root_yaw'])
        except Exception:
            pass
    try:
        obj.pose_source = 'renderer_readback'
    except Exception:
        pass

def _held_pose_object_name(held_pose: Any) -> str | None:
    if held_pose is None:
        return None
    if isinstance(held_pose, Mapping):
        value = held_pose.get('object_name')
    else:
        value = getattr(held_pose, 'object_name', None)
    return None if value is None else str(value)

def _held_pose_with_object_name(held_pose: Any, object_name: str) -> dict[str, Any]:
    if isinstance(held_pose, Mapping):
        payload = dict(held_pose)
    else:
        payload = {key: getattr(held_pose, key) for key in ('object_name', 'visual_name', 'center', 'yaw', 'pitch', 'distance', 'lateral_offset', 'vertical_offset') if hasattr(held_pose, key)}
    payload['object_name'] = str(object_name)
    return payload

def _role_schema(description: str) -> dict:
    return {'type': 'string', 'description': description}

def _json_schema(properties: Mapping[str, Any], *, required: Sequence[str]) -> dict:
    return {'type': 'object', 'properties': {str(key): dict(value) for key, value in properties.items()}, 'required': [str(item) for item in required], 'additionalProperties': False}
__all__ = ['BehaviorClosedLoopToolEnv', 'BehaviorClosedLoopToolEnvError', 'MODEL_VISIBLE_TOOLS', 'SHARED_CORE_TOOLS', 'BEHAVIOR_EXTENDED_TOOLS', 'DEFERRED_TOOLS']
