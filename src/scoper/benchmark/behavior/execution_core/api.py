from __future__ import annotations
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any
from scoper.benchmark.behavior.execution_core.actions import add_container_substance, clear_container_substances, get_contained_substances, get_dirty_state, get_open_state, get_sliced_state, get_source_substance, get_substances, get_toggle_state, placement_relation_for_target, set_dirty_state, set_open_state, set_saturated_substance, set_sliced_state, set_toggle_state, supports_open_state
from scoper.benchmark.behavior.execution_core.constraints import EgoPreconditionChecker, InteractionPreconditionConfig, combine_checks, pose_looking_at_object
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose, EgoState, normalize_angle
from scoper.benchmark.behavior.execution_core.held_object import HeldObjectPose, held_pose_from_ego, held_visual_name
from scoper.benchmark.behavior.execution_core.navigation import NavigationConfig, NavigationTarget, plan_navigation, plan_navigation_to_pose
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject, OccupancyConfig, OccupancyMap
from scoper.benchmark.behavior.execution_core.trace import ActionResult, CheckResult, TraceLogger, to_jsonable, write_json
INSIDE_PLACEMENT_MARGIN_M = 0.03
INSIDE_PLACEMENT_CLEARANCE_M = 0.005
INTERIOR_VOLUME_EPSILON_M = 1e-06

@dataclass
class EgoSymbolicCoreConfig:
    occupancy: OccupancyConfig
    navigation: NavigationConfig

class EgoSymbolicCore:

    def __init__(self, objects: list[AABBObject], ego_state: EgoState, config: EgoSymbolicCoreConfig | None=None) -> None:
        self.objects = {obj.name: obj for obj in objects}
        self.ego_state = ego_state
        self.config = config or EgoSymbolicCoreConfig(OccupancyConfig(), NavigationConfig())
        self.trace = TraceLogger()
        self.occupancy: OccupancyMap | None = None
        self.action_results: list[ActionResult] = []
        self._held_pose_anchor: EgoPose | None = None

    def make_precondition_checker(self) -> EgoPreconditionChecker:
        return EgoPreconditionChecker(self.objects, self.ego_state, self.occupancy)

    def _interaction_precondition_config(self) -> InteractionPreconditionConfig:
        return InteractionPreconditionConfig(max_distance=max(self.config.navigation.interaction_distance[1], 2.0), facing_threshold_degrees=self.config.navigation.final_facing_threshold_degrees, require_facing=True, require_reachable=True, interaction_distance=self.config.navigation.interaction_distance, candidate_count=self.config.navigation.candidate_count)

    def build_occupancy(self, trace_id: str='occupancy_0000') -> OccupancyMap:
        self.trace.emit(trace_id, 'occupancy_generation_started', object_count=len(self.objects))
        ignored_names = set(self.config.occupancy.ignored_object_names)
        if self.ego_state.held_object_name:
            ignored_names.add(self.ego_state.held_object_name)
        occupancy_config = replace(self.config.occupancy, ignored_object_names=ignored_names)
        self.occupancy = OccupancyMap.from_aabb_objects(list(self.objects.values()), occupancy_config)
        self.trace.emit(trace_id, 'occupancy_generation_finished', source=self.occupancy.source, width=self.occupancy.width, height=self.occupancy.height, metadata=self.occupancy.metadata)
        return self.occupancy

    def current_held_pose(self) -> HeldObjectPose | None:
        if self.ego_state.held_object_name is None:
            return None
        anchor_pose = self._held_pose_anchor or self.ego_state.pose
        return held_pose_from_ego(anchor_pose, object_name=self.ego_state.held_object_name, visual_name=self.ego_state.held_visual_name)

    def _update_held_pose_anchor(self, pose: EgoPose) -> None:
        if self.ego_state.held_object_name is not None:
            self._held_pose_anchor = pose.copy()

    def _clear_held_pose_anchor(self) -> None:
        self._held_pose_anchor = None

    def _emit_frame_pose(self, trace_id: str, frame_index: int, pose: EgoPose) -> None:
        held_pose = self.current_held_pose()
        payload = {'frame_index': frame_index, 'pose': pose, 'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}
        if held_pose is not None:
            payload['held_pose'] = held_pose
        self.trace.emit(trace_id, 'frame_pose', **payload)
        if held_pose is not None:
            self.trace.emit(trace_id, 'held_object_pose', frame_index=frame_index, held_pose=held_pose)

    def navigate_to(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action='NAVIGATE_TO', target=target_name)
        start_pose = self.ego_state.pose.copy()
        self.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists'])
        target_exists = self.make_precondition_checker().check_target_exists(target_name)
        self.trace.emit(trace_id, 'precondition_check_finished', check=target_exists)
        if not target_exists.allowed:
            result = ActionResult(action='NAVIGATE_TO', args={'target': target_name}, success=False, precondition=target_exists, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=target_exists.reasons, metrics=target_exists.metrics)
            return result
        target_obj = self.objects[target_name]
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        assert self.occupancy is not None
        target = NavigationTarget.from_aabb_object(target_obj)
        self.trace.emit(trace_id, 'precondition_check_started', checks=['occupancy_path_planning'])
        plan = plan_navigation(start_pose, target, self.occupancy, self.config.navigation)
        self.trace.emit(trace_id, 'precondition_check_finished', check=plan.check)
        self.trace.emit(trace_id, 'navigation_planned', success=plan.success, check=plan.check, selected_goal_xy=plan.selected_goal_xy, rejected_candidates=plan.rejected_candidates)
        via_barrier = None
        if not plan.success:
            via_barrier = self._plan_navigation_via_openable_barrier(start_pose, target_obj, plan.check, trace_id)
        if not plan.success and via_barrier is None:
            result = ActionResult(action='NAVIGATE_TO', args={'target': target_name}, success=False, precondition=plan.check, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=plan.check.reasons, metrics=plan.check.metrics)
            return result
        opened_navigation_barriers: list[dict[str, Any]] = []
        frame_index = 0
        precondition_check = plan.check
        if via_barrier is not None:
            barrier_name = str(via_barrier['barrier_name'])
            barrier_plan = via_barrier['barrier_plan']
            target_plan = via_barrier['target_plan']
            for pose in barrier_plan.frame_poses:
                self.ego_state.pose = pose
                self._update_held_pose_anchor(pose)
                self._emit_frame_pose(trace_id, frame_index, pose)
                frame_index += 1
            barrier_obj = self.objects[barrier_name]
            before_open = get_open_state(barrier_obj)
            mutation = set_open_state(barrier_obj, True)
            after_open = get_open_state(barrier_obj)
            self.occupancy = self.build_occupancy(trace_id=f'{trace_id}_occupancy_after_open_{barrier_name}')
            opened_navigation_barriers.append({'name': barrier_name, 'category': barrier_obj.category, 'before': before_open, 'after': after_open, 'changed': mutation.changed})
            self.trace.emit(trace_id, 'navigation_opened_barrier', target=target_name, barrier=barrier_name, before=before_open, after=after_open)
            plan = target_plan
            precondition_metrics = dict(plan.check.metrics)
            precondition_metrics.update({'via_openable_barrier': True, 'opened_navigation_barriers': opened_navigation_barriers, 'initial_plan_reasons': list(via_barrier['initial_check'].reasons), 'barrier_plan_length': barrier_plan.check.metrics.get('path_length'), 'target_plan_length': target_plan.check.metrics.get('path_length')})
            precondition_check = CheckResult.pass_(**precondition_metrics)
            self.trace.emit(trace_id, 'navigation_via_openable_barrier_selected', target=target_name, barrier=barrier_name, check=precondition_check)
        for pose in plan.frame_poses:
            self.ego_state.pose = pose
            self._update_held_pose_anchor(pose)
            self._emit_frame_pose(trace_id, frame_index, pose)
            frame_index += 1
        post = CheckResult.pass_(end_cell=self.occupancy.world_to_cell(self.ego_state.pose.xy), opened_navigation_barriers=opened_navigation_barriers)
        state_delta = {'ego_pose': {'before': start_pose, 'after': self.ego_state.pose.copy()}}
        if opened_navigation_barriers:
            state_delta['navigation_opened_doors'] = opened_navigation_barriers
        result = ActionResult(action='NAVIGATE_TO', args={'target': target_name}, success=True, precondition=precondition_check, postcondition=post, trace_id=trace_id, frames=frame_index, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta=state_delta)
        self.action_results.append(result)
        self.trace.emit(trace_id, 'action_succeeded', frames=result.frames, end_pose=result.end_pose)
        return result

    def navigate_to_pose(self, goal_xy: tuple[float, float], *, yaw: float | None=None, facing_xy: tuple[float, float] | None=None, trace_id: str='step_0000', action_name: str='NAVIGATE_TO_POSE', metadata: dict[str, Any] | None=None) -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action=action_name, goal_xy=goal_xy, metadata=metadata or {})
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        assert self.occupancy is not None
        self.trace.emit(trace_id, 'precondition_check_started', checks=['occupancy_path_planning'])
        plan = plan_navigation_to_pose(start_pose, goal_xy, self.occupancy, self.config.navigation, final_yaw=yaw, facing_xy=facing_xy)
        self.trace.emit(trace_id, 'precondition_check_finished', check=plan.check)
        self.trace.emit(trace_id, 'navigation_planned', success=plan.success, check=plan.check, selected_goal_xy=plan.selected_goal_xy, rejected_candidates=plan.rejected_candidates, target_type='pose')
        if not plan.success:
            result = ActionResult(action=action_name, args={'goal_xy': goal_xy, 'yaw': yaw, 'facing_xy': facing_xy, 'metadata': metadata or {}}, success=False, precondition=plan.check, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=plan.check.reasons, metrics=plan.check.metrics)
            return result
        for frame_index, pose in enumerate(plan.frame_poses):
            self.ego_state.pose = pose
            self._update_held_pose_anchor(pose)
            self._emit_frame_pose(trace_id, frame_index, pose)
        post = CheckResult.pass_(end_cell=self.occupancy.world_to_cell(self.ego_state.pose.xy), selected_goal_xy=plan.selected_goal_xy, target_type='pose')
        state_delta = {'ego_pose': {'before': start_pose, 'after': self.ego_state.pose.copy()}, 'navigate_to_pose': {'selected_goal_xy': plan.selected_goal_xy, 'target_yaw': plan.target_yaw, 'path_xy': list(plan.path_xy), 'metadata': metadata or {}}}
        result = ActionResult(action=action_name, args={'goal_xy': goal_xy, 'yaw': yaw, 'facing_xy': facing_xy, 'metadata': metadata or {}}, success=True, precondition=plan.check, postcondition=post, trace_id=trace_id, frames=len(plan.frame_poses), start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta=state_delta)
        self.action_results.append(result)
        self.trace.emit(trace_id, 'action_succeeded', frames=result.frames, end_pose=result.end_pose)
        return result

    def _plan_navigation_via_openable_barrier(self, start_pose: EgoPose, target_obj: AABBObject, initial_check: CheckResult, trace_id: str) -> dict[str, Any] | None:
        if self.occupancy is None:
            return None
        initial_occupancy = self.occupancy
        target = NavigationTarget.from_aabb_object(target_obj)
        best: tuple[float, dict[str, Any]] | None = None
        rejected: list[dict[str, Any]] = []
        for barrier in self._closed_openable_navigation_barriers():
            if barrier.name == target_obj.name:
                continue
            barrier_target = NavigationTarget.from_aabb_object(barrier)
            barrier_plan = plan_navigation(start_pose, barrier_target, initial_occupancy, self.config.navigation)
            record = {'barrier': barrier.name, 'barrier_plan_success': barrier_plan.success}
            if not barrier_plan.success:
                record['barrier_plan_reasons'] = list(barrier_plan.check.reasons)
                rejected.append(record)
                continue
            opened_occupancy = self._occupancy_snapshot_with_open_state(barrier.name, True)
            if opened_occupancy is None:
                record['reason'] = 'could_not_build_opened_barrier_occupancy'
                rejected.append(record)
                continue
            barrier_end_pose = barrier_plan.frame_poses[-1].copy() if barrier_plan.frame_poses else start_pose.copy()
            target_plan = plan_navigation(barrier_end_pose, target, opened_occupancy, self.config.navigation)
            record['target_plan_success'] = target_plan.success
            if not target_plan.success:
                record['target_plan_reasons'] = list(target_plan.check.reasons)
                rejected.append(record)
                continue
            score = float(barrier_plan.check.metrics.get('path_length') or 0.0) + float(target_plan.check.metrics.get('path_length') or 0.0)
            candidate = {'barrier_name': barrier.name, 'barrier_plan': barrier_plan, 'target_plan': target_plan, 'opened_occupancy': opened_occupancy, 'initial_check': initial_check, 'rejected_barriers': rejected}
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is None:
            if rejected:
                self.trace.emit(trace_id, 'navigation_openable_barrier_rejected', rejected=rejected)
            return None
        best[1]['initial_check'] = initial_check
        return best[1]

    def _occupancy_snapshot_with_open_state(self, object_name: str, should_open: bool) -> OccupancyMap | None:
        obj = self.objects.get(object_name)
        if obj is None:
            return None
        before = get_open_state(obj)
        had_open_key = isinstance(getattr(obj, 'states', None), dict) and 'open' in obj.states
        try:
            set_open_state(obj, should_open)
            return self._build_occupancy_snapshot()
        except Exception:
            return None
        finally:
            try:
                if before is None:
                    if had_open_key:
                        obj.states['open'] = None
                    elif isinstance(getattr(obj, 'states', None), dict):
                        obj.states.pop('open', None)
                else:
                    set_open_state(obj, bool(before))
            except Exception:
                pass

    def _build_occupancy_snapshot(self) -> OccupancyMap:
        ignored_names = set(self.config.occupancy.ignored_object_names)
        if self.ego_state.held_object_name:
            ignored_names.add(self.ego_state.held_object_name)
        occupancy_config = replace(self.config.occupancy, ignored_object_names=ignored_names)
        return OccupancyMap.from_aabb_objects(list(self.objects.values()), occupancy_config)

    def _closed_openable_navigation_barriers(self) -> list[AABBObject]:
        barriers = []
        for obj in self.objects.values():
            if not _is_navigation_barrier_object(obj):
                continue
            if not supports_open_state(obj):
                continue
            if get_open_state(obj) is False:
                barriers.append(obj)
        return barriers

    def look_for(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        return self._orient_towards_target(target_name, trace_id=trace_id, action_name='LOOK_FOR')

    def look_at(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        return self._orient_towards_target(target_name, trace_id=trace_id, action_name='LOOK_AT')

    def _orient_towards_target(self, target_name: str, trace_id: str, action_name: str) -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action=action_name, target=target_name)
        start_pose = self.ego_state.pose.copy()
        self.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists'])
        pre = self.make_precondition_checker().check_target_exists(target_name)
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        if not pre.allowed:
            result = ActionResult(action=action_name, args={'target': target_name}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        target_obj = self.objects[target_name]
        end_pose = pose_looking_at_object(start_pose, target_obj)
        frame_poses = _interpolate_pose_turn(start_pose, end_pose, self.config.navigation.turn_spacing_degrees)
        for frame_index, pose in enumerate(frame_poses):
            self.ego_state.pose = pose
            self._emit_frame_pose(trace_id, frame_index, pose)
        post = self.make_precondition_checker().check_facing(target_name, threshold_degrees=1.0)
        success = post.allowed
        if not success and 'postcondition_failed' not in post.reasons:
            post = CheckResult.fail(['postcondition_failed', *post.reasons], **post.metrics)
        result = ActionResult(action=action_name, args={'target': target_name}, success=success, precondition=pre, postcondition=post, trace_id=trace_id, frames=len(frame_poses), start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'ego_pose': {'before': start_pose, 'after': self.ego_state.pose.copy()}, action_name.lower(): {'target': target_name, 'frame_poses': [pose.copy() for pose in frame_poses], 'target_position': tuple(target_obj.center)}})
        self.action_results.append(result)
        if success:
            self.trace.emit(trace_id, 'action_succeeded', frames=result.frames, end_pose=result.end_pose)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def scan_in_place(self, num_views: int=4, trace_id: str='step_0000') -> ActionResult:
        start_pose = self.ego_state.pose.copy()
        try:
            num_views = int(num_views)
        except Exception:
            num_views = 0
        self.trace.emit(trace_id, 'action_requested', action='SCAN', num_views=num_views)
        if not 1 <= num_views <= 8:
            pre = CheckResult.fail('invalid_scan_num_views', num_views=num_views)
            result = ActionResult(action='SCAN', args={'num_views': num_views}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        pre = CheckResult.pass_(num_views=num_views)
        frame_poses: list[EgoPose] = []
        frame_view_indices: list[int] = []
        view_poses: list[EgoPose] = []
        yaw_step = 2.0 * math.pi / float(num_views)
        for view_index in range(num_views):
            endpoint = EgoPose(x=start_pose.x, y=start_pose.y, z=start_pose.z, yaw=normalize_angle(start_pose.yaw + yaw_step * float(view_index + 1)), pitch=start_pose.pitch)
            self.ego_state.pose = endpoint
            frame_poses.append(endpoint.copy())
            frame_view_indices.append(view_index)
            view_poses.append(endpoint.copy())
            self._emit_frame_pose(trace_id, len(frame_poses) - 1, endpoint)
            self.trace.emit(trace_id, 'scan_view_completed', view_index=view_index, pose=self.ego_state.pose.copy())
        post = CheckResult.pass_(num_views=num_views, num_views_completed=num_views, degrees_per_view=math.degrees(yaw_step))
        result = ActionResult(action='SCAN', args={'num_views': num_views}, success=True, precondition=pre, postcondition=post, trace_id=trace_id, frames=len(frame_poses), start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'ego_pose': {'before': start_pose, 'after': self.ego_state.pose.copy()}, 'scan': {'num_views': num_views, 'num_views_completed': num_views, 'degrees_per_view': math.degrees(yaw_step), 'frame_poses': frame_poses, 'frame_view_indices': frame_view_indices, 'view_poses': view_poses}})
        self.action_results.append(result)
        self.trace.emit(trace_id, 'action_succeeded', frames=result.frames, end_pose=result.end_pose)
        return result

    def open_object(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        return self._set_open_state(target_name, True, trace_id)

    def close_object(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        return self._set_open_state(target_name, False, trace_id)

    def toggle_on(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        return self._set_toggle_state(target_name, True, trace_id)

    def toggle_off(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        return self._set_toggle_state(target_name, False, trace_id)

    def pick_object(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action='PICK', target=target_name)
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        checker = self.make_precondition_checker()
        precondition_config = InteractionPreconditionConfig(max_distance=max(self.config.navigation.interaction_distance[1], 2.0), facing_threshold_degrees=self.config.navigation.final_facing_threshold_degrees, require_facing=True, require_reachable=True, interaction_distance=self.config.navigation.interaction_distance, candidate_count=self.config.navigation.candidate_count)
        self.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists', 'distance', 'facing', 'reachability', 'pick_affordance', 'not_holding'])
        pre = combine_checks(checker.check_interaction(target_name, precondition_config), checker.check_pick_affordance(target_name), checker.check_not_holding())
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        if not pre.allowed:
            result = ActionResult(action='PICK', args={'target': target_name}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        before_held = {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}
        visual_name = held_visual_name(target_name)
        self.trace.emit(trace_id, 'held_object_mutation_attempted', target=target_name, before=before_held, requested_held_object_name=target_name, requested_held_visual_name=visual_name)
        self.ego_state.held_object_name = target_name
        self.ego_state.held_visual_name = visual_name
        self._update_held_pose_anchor(self.ego_state.pose)
        held_pose = self.current_held_pose()
        assert held_pose is not None
        self.trace.emit(trace_id, 'held_object_pose', frame_index=0, held_pose=held_pose)
        occupancy_after_pick = self.build_occupancy(trace_id=f'{trace_id}_occupancy_after_pick')
        ignored_names = set(occupancy_after_pick.metadata.get('ignored_object_names', []))
        rasterized_names = {entry['name'] for entry in occupancy_after_pick.metadata.get('rasterized_objects', [])}
        post = CheckResult.pass_(held_object_name=self.ego_state.held_object_name, held_visual_name=self.ego_state.held_visual_name, held_pose=held_pose, occupancy_ignored=target_name in ignored_names, target_rasterized_after_pick=target_name in rasterized_names)
        if self.ego_state.held_object_name != target_name or target_name not in ignored_names or target_name in rasterized_names:
            post = CheckResult.fail('postcondition_failed', held_object_name=self.ego_state.held_object_name, held_visual_name=self.ego_state.held_visual_name, held_pose=held_pose, occupancy_ignored=target_name in ignored_names, target_rasterized_after_pick=target_name in rasterized_names)
        self.trace.emit(trace_id, 'postcondition_check_finished', check=post)
        result = ActionResult(action='PICK', args={'target': target_name}, success=post.allowed, precondition=pre, postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'held_object': {'before': before_held, 'after': {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}, 'held_pose': held_pose}})
        self.action_results.append(result)
        if result.success:
            self.trace.emit(trace_id, 'action_succeeded', state_delta=result.state_delta)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def place(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action='PLACE', target=target_name)
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        checker = self.make_precondition_checker()
        self.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists', 'holding'])
        base_pre = combine_checks(checker.check_target_exists(target_name), checker.check_holding())
        self.trace.emit(trace_id, 'precondition_check_finished', check=base_pre)
        if not base_pre.allowed:
            result = ActionResult(action='PLACE', args={'target': target_name}, success=False, precondition=base_pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=base_pre.reasons, metrics=base_pre.metrics)
            return result
        precondition_config = InteractionPreconditionConfig(max_distance=max(self.config.navigation.interaction_distance[1], 2.0), facing_threshold_degrees=self.config.navigation.final_facing_threshold_degrees, require_facing=True, require_reachable=True, interaction_distance=self.config.navigation.interaction_distance, candidate_count=self.config.navigation.candidate_count)
        self.trace.emit(trace_id, 'precondition_check_started', checks=['distance', 'facing', 'reachability', 'place_affordance', 'container_open', 'target_not_held'])
        pre = combine_checks(checker.check_interaction(target_name, precondition_config), checker.check_place_affordance(target_name), checker.check_container_open_for_place(target_name), checker.check_target_not_held(target_name))
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        if not pre.allowed:
            result = ActionResult(action='PLACE', args={'target': target_name}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        held_name = self.ego_state.held_object_name
        assert held_name is not None
        held_obj = self.objects[held_name]
        target_obj = self.objects[target_name]
        placement_relation = pre.metrics.get('placement_relation') or placement_relation_for_target(target_obj)
        if placement_relation == 'on_top':
            placed_center = place_on_aabb_center(held_obj, target_obj, self.ego_state.pose)
        elif placement_relation == 'inside':
            placed_center = place_inside_aabb_center(held_obj, target_obj, self.ego_state.pose)
        else:
            post = CheckResult.fail('postcondition_failed', placed_object_name=held_name, place_target_name=target_name, placement_relation=placement_relation)
            result = ActionResult(action='PLACE', args={'target': target_name}, success=False, precondition=pre, postcondition=post, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
            return result
        before_held = {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}
        before_center = held_obj.center
        before_extent = held_obj.extent
        self.trace.emit(trace_id, 'object_pose_mutation_attempted', target=held_name, place_target=target_name, placement_relation=placement_relation, before_center=before_center, requested_center=placed_center)
        held_obj.center = placed_center
        self.trace.emit(trace_id, 'placed_object_pose', target=held_name, place_target=target_name, placement_relation=placement_relation, center=held_obj.center)
        self.ego_state.held_object_name = None
        self.ego_state.held_visual_name = None
        self._clear_held_pose_anchor()
        occupancy_after_place = self.build_occupancy(trace_id=f'{trace_id}_occupancy_after_place')
        ignored_names = set(occupancy_after_place.metadata.get('ignored_object_names', []))
        rasterized_names = {entry['name'] for entry in occupancy_after_place.metadata.get('rasterized_objects', [])}
        after_held = {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}
        target_rasterized = held_name in rasterized_names
        occupancy_ignored = held_name in ignored_names
        held_cleared = self.ego_state.held_object_name is None and self.ego_state.held_visual_name is None
        center_error = _point_distance(held_obj.center, placed_center)
        occupancy_config = self.config.occupancy
        band_min = occupancy_config.floor_height + occupancy_config.navigation_height_band[0]
        band_max = occupancy_config.floor_height + occupancy_config.navigation_height_band[1]
        outside_navigation_height = held_obj.center[2] - held_obj.extent[2] * 0.5 > band_max or held_obj.center[2] + held_obj.extent[2] * 0.5 < band_min
        post = CheckResult.pass_(placed_object_name=held_name, place_target_name=target_name, placement_relation=placement_relation, placed_center=held_obj.center, held_cleared=held_cleared, occupancy_ignored=occupancy_ignored, target_rasterized_after_place=target_rasterized, outside_navigation_height=outside_navigation_height, placement_center_error=center_error)
        if not held_cleared or occupancy_ignored or (not target_rasterized and (not outside_navigation_height)) or (center_error > 1e-09):
            post = CheckResult.fail('postcondition_failed', placed_object_name=held_name, place_target_name=target_name, placement_relation=placement_relation, placed_center=held_obj.center, held_cleared=held_cleared, occupancy_ignored=occupancy_ignored, target_rasterized_after_place=target_rasterized, placement_center_error=center_error)
        self.trace.emit(trace_id, 'postcondition_check_finished', check=post)
        result = ActionResult(action='PLACE', args={'target': target_name}, success=post.allowed, precondition=pre, postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'held_object': {'before': before_held, 'after': after_held}, 'object_pose': {'target': held_name, 'place_target': target_name, 'placement_relation': placement_relation, 'before_center': before_center, 'before_extent': before_extent, 'after_center': held_obj.center, 'after_extent': held_obj.extent}})
        self.action_results.append(result)
        if result.success:
            self.trace.emit(trace_id, 'action_succeeded', state_delta=result.state_delta)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def place_on(self, surface_name: str, trace_id: str='step_0000') -> ActionResult:
        return self.place(surface_name, trace_id=trace_id)

    def place_on_object(self, surface_name: str, trace_id: str='step_0000') -> ActionResult:
        return self.place(surface_name, trace_id=trace_id)

    def release(self, trace_id: str='step_0000') -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action='RELEASE')
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        checker = self.make_precondition_checker()
        self.trace.emit(trace_id, 'precondition_check_started', checks=['holding', 'release_pose'])
        holding_check = checker.check_holding()
        release_center = None
        if holding_check.allowed:
            held_name = self.ego_state.held_object_name
            assert held_name is not None
            release_center, release_pose_check = release_aabb_in_front(self.objects[held_name], self.ego_state.pose, self.occupancy, self.config.occupancy.floor_height)
        else:
            release_pose_check = CheckResult.pass_(release_pose_checked=False)
        pre = combine_checks(holding_check, release_pose_check)
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        if not pre.allowed:
            result = ActionResult(action='RELEASE', args={}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        held_name = self.ego_state.held_object_name
        assert held_name is not None
        assert release_center is not None
        held_obj = self.objects[held_name]
        before_held = {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}
        before_center = held_obj.center
        before_extent = held_obj.extent
        self.trace.emit(trace_id, 'object_pose_mutation_attempted', target=held_name, release_pose_type='in_front_of_ego_floor', before_center=before_center, requested_center=release_center)
        held_obj.center = release_center
        self.trace.emit(trace_id, 'released_object_pose', target=held_name, release_pose_type='in_front_of_ego_floor', center=held_obj.center)
        self.ego_state.held_object_name = None
        self.ego_state.held_visual_name = None
        self._clear_held_pose_anchor()
        occupancy_after_release = self.build_occupancy(trace_id=f'{trace_id}_occupancy_after_release')
        ignored_names = set(occupancy_after_release.metadata.get('ignored_object_names', []))
        rasterized_names = {entry['name'] for entry in occupancy_after_release.metadata.get('rasterized_objects', [])}
        after_held = {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}
        target_rasterized = held_name in rasterized_names
        occupancy_ignored = held_name in ignored_names
        held_cleared = self.ego_state.held_object_name is None and self.ego_state.held_visual_name is None
        center_error = _point_distance(held_obj.center, release_center)
        post = CheckResult.pass_(released_object_name=held_name, release_pose_type='in_front_of_ego_floor', released_center=held_obj.center, held_cleared=held_cleared, occupancy_ignored=occupancy_ignored, target_rasterized_after_release=target_rasterized, release_center_error=center_error)
        if not held_cleared or occupancy_ignored or (not target_rasterized) or (center_error > 1e-09):
            post = CheckResult.fail('postcondition_failed', released_object_name=held_name, release_pose_type='in_front_of_ego_floor', released_center=held_obj.center, held_cleared=held_cleared, occupancy_ignored=occupancy_ignored, target_rasterized_after_release=target_rasterized, release_center_error=center_error)
        self.trace.emit(trace_id, 'postcondition_check_finished', check=post)
        result = ActionResult(action='RELEASE', args={}, success=post.allowed, precondition=pre, postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'held_object': {'before': before_held, 'after': after_held}, 'object_pose': {'target': held_name, 'release_pose_type': 'in_front_of_ego_floor', 'before_center': before_center, 'before_extent': before_extent, 'after_center': held_obj.center, 'after_extent': held_obj.extent}})
        self.action_results.append(result)
        if result.success:
            self.trace.emit(trace_id, 'action_succeeded', state_delta=result.state_delta)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def drop(self, trace_id: str='step_0000') -> ActionResult:
        return self.release(trace_id=trace_id)

    def wipe(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        return self._set_tool_target_boolean_state(target_name=target_name, trace_id=trace_id, action='WIPE', state_name='dirty', requested_value=False, held_tool_check_name='wipe_tool_affordance', target_affordance_check_name='wipe_affordance', target_state_check_name='target_dirty', held_tool_check_fn=lambda checker: checker.check_held_wipe_tool_affordance(), target_affordance_check_fn=lambda checker: checker.check_wipe_affordance(target_name), target_state_check_fn=lambda checker: checker.check_target_dirty_for_wipe(target_name), get_state_fn=get_dirty_state, set_state_fn=set_dirty_state)

    def cut(self, target_name: str, trace_id: str='step_0000') -> ActionResult:
        return self._set_tool_target_boolean_state(target_name=target_name, trace_id=trace_id, action='CUT', state_name='sliced', requested_value=True, held_tool_check_name='cut_tool_affordance', target_affordance_check_name='cut_affordance', target_state_check_name='target_unsliced', held_tool_check_fn=lambda checker: checker.check_held_cut_tool_affordance(), target_affordance_check_fn=lambda checker: checker.check_cut_affordance(target_name), target_state_check_fn=lambda checker: checker.check_target_unsliced_for_cut(target_name), get_state_fn=get_sliced_state, set_state_fn=set_sliced_state)

    def soak(self, source_name: str, trace_id: str='step_0000') -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action='SOAK', source=source_name, soak_relation='under')
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        checker = self.make_precondition_checker()
        config = self._interaction_precondition_config()
        self.trace.emit(trace_id, 'precondition_check_started', checks=['source_exists', 'distance', 'facing', 'reachability', 'holding', 'soakable_affordance', 'fluid_source_affordance', 'fluid_source_active'])
        pre = combine_checks(checker.check_interaction(source_name, config), checker.check_holding(), checker.check_held_soakable_affordance(), checker.check_fluid_source_affordance(source_name), checker.check_fluid_source_active(source_name))
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        substance = pre.metrics.get('substance')
        if not pre.allowed:
            result = ActionResult(action='SOAK', args={'source': source_name, 'soak_relation': 'under', 'substance': substance}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        held_name = self.ego_state.held_object_name
        assert held_name is not None
        assert isinstance(substance, str)
        held_obj = self.objects[held_name]
        before_saturated = get_substances(held_obj, 'saturated')
        self.trace.emit(trace_id, 'object_state_mutation_attempted', target=held_name, source=source_name, state='saturated', substance=substance, before=before_saturated, requested=True)
        mutation = set_saturated_substance(held_obj, substance, True)
        after_saturated = get_substances(held_obj, 'saturated')
        post = CheckResult.pass_(soaked_object_name=held_name, source=source_name, soak_relation='under', substance=substance, saturated_substances=after_saturated, changed=mutation.changed, held_object_name=self.ego_state.held_object_name)
        if substance not in after_saturated or self.ego_state.held_object_name != held_name:
            post = CheckResult.fail('postcondition_failed', soaked_object_name=held_name, source=source_name, soak_relation='under', substance=substance, saturated_substances=after_saturated, changed=mutation.changed, held_object_name=self.ego_state.held_object_name)
        self.trace.emit(trace_id, 'postcondition_check_finished', check=post)
        result = ActionResult(action='SOAK', args={'source': source_name, 'soak_relation': 'under', 'substance': substance}, success=post.allowed, precondition=pre, postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'object_state': {'target': held_name, 'state': 'saturated', 'substance': substance, 'before': before_saturated, 'after': after_saturated, 'changed': mutation.changed}, 'held_object': {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}})
        self.action_results.append(result)
        if result.success:
            self.trace.emit(trace_id, 'action_succeeded', state_delta=result.state_delta)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def fill(self, container_name: str, source_name: str, trace_id: str='step_0000') -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action='FILL', target=container_name, source=source_name)
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        checker = self.make_precondition_checker()
        config = self._interaction_precondition_config()
        source_check = checker.check_fluid_source_affordance(source_name)
        substance = source_check.metrics.get('substance')
        self.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists', 'source_exists', 'distance', 'facing', 'reachability', 'target_held', 'fillable_affordance', 'container_open', 'fluid_source_affordance', 'fluid_source_active', 'container_missing_substance'])
        missing_substance_check = checker.check_container_missing_substance(container_name, substance) if isinstance(substance, str) else CheckResult.fail('missing_source_substance', source=source_name)
        pre = combine_checks(checker.check_interaction(source_name, config), checker.check_held_target(container_name), checker.check_fillable_affordance(container_name), checker.check_container_open_for_substance(container_name), source_check, checker.check_fluid_source_active(source_name), missing_substance_check)
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        if not pre.allowed:
            result = ActionResult(action='FILL', args={'target': container_name, 'source': source_name, 'substance': substance}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        assert isinstance(substance, str)
        target_obj = self.objects[container_name]
        before_contains = get_substances(target_obj, 'contains')
        before_filled = get_substances(target_obj, 'filled')
        self.trace.emit(trace_id, 'object_state_mutation_attempted', target=container_name, source=source_name, state='contains/filled', substance=substance, before={'contains': before_contains, 'filled': before_filled}, requested=True)
        mutations = add_container_substance(target_obj, substance)
        after_contains = get_substances(target_obj, 'contains')
        after_filled = get_substances(target_obj, 'filled')
        post = CheckResult.pass_(target=container_name, source=source_name, substance=substance, contains=after_contains, filled=after_filled, changed=any((mutation.changed for mutation in mutations)), held_object_name=self.ego_state.held_object_name)
        if substance not in after_contains or substance not in after_filled or self.ego_state.held_object_name != container_name:
            post = CheckResult.fail('postcondition_failed', target=container_name, source=source_name, substance=substance, contains=after_contains, filled=after_filled, changed=any((mutation.changed for mutation in mutations)), held_object_name=self.ego_state.held_object_name)
        self.trace.emit(trace_id, 'postcondition_check_finished', check=post)
        result = ActionResult(action='FILL', args={'target': container_name, 'source': source_name, 'substance': substance}, success=post.allowed, precondition=pre, postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'object_state': {'target': container_name, 'state': 'contains/filled', 'substance': substance, 'before': {'contains': before_contains, 'filled': before_filled}, 'after': {'contains': after_contains, 'filled': after_filled}, 'changed': any((mutation.changed for mutation in mutations))}, 'held_object': {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}})
        self.action_results.append(result)
        if result.success:
            self.trace.emit(trace_id, 'action_succeeded', state_delta=result.state_delta)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def pour(self, target_container_name: str, trace_id: str='step_0000') -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action='POUR', target=target_container_name, source=self.ego_state.held_object_name)
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        checker = self.make_precondition_checker()
        config = self._interaction_precondition_config()
        held_source_check = checker.check_held_pourable_source_affordance()
        substances = held_source_check.metrics.get('substances', [])
        duplicate_checks = [checker.check_container_missing_substance(target_container_name, substance) for substance in substances if isinstance(substance, str)] if isinstance(substances, list) else []
        self.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists', 'distance', 'facing', 'reachability', 'holding', 'pour_source_affordance', 'source_has_substance', 'fillable_affordance', 'container_open', 'target_not_held', 'target_missing_substance'])
        pre = combine_checks(checker.check_interaction(target_container_name, config), checker.check_holding(), held_source_check, checker.check_fillable_affordance(target_container_name), checker.check_container_open_for_substance(target_container_name), checker.check_target_not_held(target_container_name), *duplicate_checks)
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        source_name = self.ego_state.held_object_name
        if not pre.allowed:
            result = ActionResult(action='POUR', args={'target': target_container_name, 'source': source_name, 'substances': substances}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        assert source_name is not None
        assert isinstance(substances, list)
        source_obj = self.objects[source_name]
        target_obj = self.objects[target_container_name]
        before_source = {'contains': get_substances(source_obj, 'contains'), 'filled': get_substances(source_obj, 'filled')}
        before_target = {'contains': get_substances(target_obj, 'contains'), 'filled': get_substances(target_obj, 'filled')}
        self.trace.emit(trace_id, 'object_state_mutation_attempted', source=source_name, target=target_container_name, state='contains/filled', substances=substances, before={'source': before_source, 'target': before_target}, requested='transfer_all')
        add_mutations = []
        for substance in substances:
            add_mutations.extend(add_container_substance(target_obj, str(substance)))
        clear_mutation = clear_container_substances(source_obj)
        after_source = {'contains': get_substances(source_obj, 'contains'), 'filled': get_substances(source_obj, 'filled')}
        after_target = {'contains': get_substances(target_obj, 'contains'), 'filled': get_substances(target_obj, 'filled')}
        target_has_all = all((substance in after_target['contains'] and substance in after_target['filled'] for substance in substances))
        source_empty = not after_source['contains'] and (not after_source['filled'])
        post = CheckResult.pass_(source=source_name, target=target_container_name, substances=substances, target_contains=after_target['contains'], target_filled=after_target['filled'], source_contains=after_source['contains'], source_filled=after_source['filled'], source_depleted=source_empty, changed=any((mutation.changed for mutation in add_mutations)) or clear_mutation.changed, held_object_name=self.ego_state.held_object_name)
        if not target_has_all or not source_empty or self.ego_state.held_object_name != source_name:
            post = CheckResult.fail('postcondition_failed', source=source_name, target=target_container_name, substances=substances, target_contains=after_target['contains'], target_filled=after_target['filled'], source_contains=after_source['contains'], source_filled=after_source['filled'], source_depleted=source_empty, held_object_name=self.ego_state.held_object_name)
        self.trace.emit(trace_id, 'postcondition_check_finished', check=post)
        result = ActionResult(action='POUR', args={'target': target_container_name, 'source': source_name, 'substances': substances}, success=post.allowed, precondition=pre, postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'object_state': {'source': source_name, 'target': target_container_name, 'state': 'contains/filled', 'substances': substances, 'before': {'source': before_source, 'target': before_target}, 'after': {'source': after_source, 'target': after_target}, 'source_depleted': source_empty}, 'held_object': {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}})
        self.action_results.append(result)
        if result.success:
            self.trace.emit(trace_id, 'action_succeeded', state_delta=result.state_delta)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def empty(self, container_name: str, trace_id: str='step_0000') -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action='EMPTY', target=container_name)
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        checker = self.make_precondition_checker()
        config = self._interaction_precondition_config()
        target_is_held = self.ego_state.held_object_name == container_name
        interaction_check = CheckResult.pass_(target=container_name, target_is_held=True) if target_is_held else checker.check_interaction(container_name, config)
        self.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists', 'distance', 'facing', 'reachability', 'fillable_affordance', 'container_open', 'container_has_substance'])
        pre = combine_checks(interaction_check, checker.check_fillable_affordance(container_name), checker.check_container_open_for_substance(container_name), checker.check_container_has_substance(container_name))
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        if not pre.allowed:
            result = ActionResult(action='EMPTY', args={'target': container_name}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        target_obj = self.objects[container_name]
        before = {'contains': get_substances(target_obj, 'contains'), 'filled': get_substances(target_obj, 'filled')}
        self.trace.emit(trace_id, 'object_state_mutation_attempted', target=container_name, state='contains/filled', before=before, requested='clear')
        mutation = clear_container_substances(target_obj)
        after = {'contains': get_substances(target_obj, 'contains'), 'filled': get_substances(target_obj, 'filled')}
        emptied = not after['contains'] and (not after['filled'])
        post = CheckResult.pass_(target=container_name, removed_substances=mutation.removed_substances, contains=after['contains'], filled=after['filled'], emptied=emptied, changed=mutation.changed, held_object_name=self.ego_state.held_object_name)
        if not emptied:
            post = CheckResult.fail('postcondition_failed', target=container_name, removed_substances=mutation.removed_substances, contains=after['contains'], filled=after['filled'], emptied=emptied, changed=mutation.changed, held_object_name=self.ego_state.held_object_name)
        self.trace.emit(trace_id, 'postcondition_check_finished', check=post)
        result = ActionResult(action='EMPTY', args={'target': container_name}, success=post.allowed, precondition=pre, postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'object_state': {'target': container_name, 'state': 'contains/filled', 'before': before, 'after': after, 'removed_substances': mutation.removed_substances, 'changed': mutation.changed}, 'held_object': {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}})
        self.action_results.append(result)
        if result.success:
            self.trace.emit(trace_id, 'action_succeeded', state_delta=result.state_delta)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def _set_open_state(self, target_name: str, should_open: bool, trace_id: str) -> ActionResult:
        action = 'OPEN' if should_open else 'CLOSE'
        return self._set_boolean_object_state(target_name=target_name, requested_value=should_open, trace_id=trace_id, action=action, state_name='open', affordance_check_name='open_affordance', affordance_check_fn=lambda checker: checker.check_open_affordance(target_name), get_state_fn=get_open_state, set_state_fn=set_open_state)

    def _set_toggle_state(self, target_name: str, should_toggle_on: bool, trace_id: str) -> ActionResult:
        action = 'TOGGLE_ON' if should_toggle_on else 'TOGGLE_OFF'
        return self._set_boolean_object_state(target_name=target_name, requested_value=should_toggle_on, trace_id=trace_id, action=action, state_name='toggled_on', affordance_check_name='toggle_affordance', affordance_check_fn=lambda checker: checker.check_toggle_affordance(target_name), get_state_fn=get_toggle_state, set_state_fn=set_toggle_state)

    def _set_boolean_object_state(self, target_name: str, requested_value: bool, trace_id: str, action: str, state_name: str, affordance_check_name: str, affordance_check_fn, get_state_fn, set_state_fn) -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action=action, target=target_name)
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        checker = self.make_precondition_checker()
        precondition_config = InteractionPreconditionConfig(max_distance=max(self.config.navigation.interaction_distance[1], 2.0), facing_threshold_degrees=self.config.navigation.final_facing_threshold_degrees, require_facing=True, require_reachable=True, interaction_distance=self.config.navigation.interaction_distance, candidate_count=self.config.navigation.candidate_count)
        self.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists', 'distance', 'facing', 'reachability', affordance_check_name])
        interaction_check = checker.check_interaction(target_name, precondition_config)
        affordance_check = affordance_check_fn(checker)
        pre = combine_checks(interaction_check, affordance_check)
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        if not pre.allowed:
            result = ActionResult(action=action, args={'target': target_name}, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        target_obj = self.objects[target_name]
        before_state = get_state_fn(target_obj)
        self.trace.emit(trace_id, 'object_state_mutation_attempted', target=target_name, state=state_name, before=before_state, requested=requested_value)
        mutation = set_state_fn(target_obj, requested_value)
        after_state = get_state_fn(target_obj)
        if state_name == 'open':
            self.build_occupancy(trace_id=f'{trace_id}_occupancy_after_open_state')
        post = CheckResult.pass_(target=target_name, state=state_name, expected=requested_value, actual=after_state, changed=mutation.changed)
        if after_state != requested_value:
            post = CheckResult.fail('postcondition_failed', target=target_name, state=state_name, expected=requested_value, actual=after_state, changed=mutation.changed)
        self.trace.emit(trace_id, 'postcondition_check_finished', check=post)
        result = ActionResult(action=action, args={'target': target_name}, success=post.allowed, precondition=pre, postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'object_state': {'target': target_name, 'state': state_name, 'before': before_state, 'after': after_state, 'changed': mutation.changed}})
        self.action_results.append(result)
        if result.success:
            self.trace.emit(trace_id, 'action_succeeded', state_delta=result.state_delta)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def _set_tool_target_boolean_state(self, target_name: str, trace_id: str, action: str, state_name: str, requested_value: bool, held_tool_check_name: str, target_affordance_check_name: str, target_state_check_name: str, held_tool_check_fn, target_affordance_check_fn, target_state_check_fn, get_state_fn, set_state_fn) -> ActionResult:
        self.trace.emit(trace_id, 'action_requested', action=action, target=target_name, tool=self.ego_state.held_object_name)
        start_pose = self.ego_state.pose.copy()
        if self.occupancy is None:
            self.build_occupancy(trace_id=f'{trace_id}_occupancy')
        checker = self.make_precondition_checker()
        precondition_config = InteractionPreconditionConfig(max_distance=max(self.config.navigation.interaction_distance[1], 2.0), facing_threshold_degrees=self.config.navigation.final_facing_threshold_degrees, require_facing=True, require_reachable=True, interaction_distance=self.config.navigation.interaction_distance, candidate_count=self.config.navigation.candidate_count)
        self.trace.emit(trace_id, 'precondition_check_started', checks=['target_exists', 'distance', 'facing', 'reachability', 'holding', held_tool_check_name, target_affordance_check_name, target_state_check_name, 'target_not_held'])
        pre = combine_checks(checker.check_interaction(target_name, precondition_config), checker.check_holding(), held_tool_check_fn(checker), target_affordance_check_fn(checker), target_state_check_fn(checker), checker.check_target_not_held(target_name))
        self.trace.emit(trace_id, 'precondition_check_finished', check=pre)
        args = {'target': target_name, 'tool': self.ego_state.held_object_name}
        if not pre.allowed:
            result = ActionResult(action=action, args=args, success=False, precondition=pre, postcondition=None, trace_id=trace_id, start_pose=start_pose, end_pose=self.ego_state.pose.copy())
            self.action_results.append(result)
            self.trace.emit(trace_id, 'action_failed', reasons=pre.reasons, metrics=pre.metrics)
            return result
        tool_name = self.ego_state.held_object_name
        assert tool_name is not None
        target_obj = self.objects[target_name]
        before_state = get_state_fn(target_obj)
        self.trace.emit(trace_id, 'object_state_mutation_attempted', target=target_name, tool=tool_name, state=state_name, before=before_state, requested=requested_value)
        mutation = set_state_fn(target_obj, requested_value)
        after_state = get_state_fn(target_obj)
        post = CheckResult.pass_(target=target_name, tool=tool_name, state=state_name, expected=requested_value, actual=after_state, changed=mutation.changed, held_object_name=self.ego_state.held_object_name)
        if after_state != requested_value or self.ego_state.held_object_name != tool_name:
            post = CheckResult.fail('postcondition_failed', target=target_name, tool=tool_name, state=state_name, expected=requested_value, actual=after_state, changed=mutation.changed, held_object_name=self.ego_state.held_object_name)
        self.trace.emit(trace_id, 'postcondition_check_finished', check=post)
        result = ActionResult(action=action, args={'target': target_name, 'tool': tool_name}, success=post.allowed, precondition=pre, postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=self.ego_state.pose.copy(), state_delta={'object_state': {'target': target_name, 'state': state_name, 'before': before_state, 'after': after_state, 'changed': mutation.changed}, 'held_object': {'held_object_name': self.ego_state.held_object_name, 'held_visual_name': self.ego_state.held_visual_name}})
        self.action_results.append(result)
        if result.success:
            self.trace.emit(trace_id, 'action_succeeded', state_delta=result.state_delta)
        else:
            self.trace.emit(trace_id, 'action_failed', reasons=post.reasons, metrics=post.metrics)
        return result

    def summary(self) -> dict:
        return {'ego_state': self.ego_state.as_dict(), 'objects': [obj.__dict__ for obj in self.objects.values()], 'occupancy': None if self.occupancy is None else self.occupancy.metadata, 'actions': [to_jsonable(result) for result in self.action_results], 'trace': self.trace.as_list()}

    def write_summary(self, path: str | Path) -> None:
        write_json(path, self.summary())

def make_core(objects: list[AABBObject], start_pose: EgoPose, occupancy_config: OccupancyConfig | None=None, navigation_config: NavigationConfig | None=None) -> EgoSymbolicCore:
    return EgoSymbolicCore(objects=objects, ego_state=EgoState(pose=start_pose), config=EgoSymbolicCoreConfig(occupancy=occupancy_config or OccupancyConfig(), navigation=navigation_config or NavigationConfig()))

def place_on_aabb_center(held_obj: AABBObject, surface_obj: AABBObject, ego_pose: EgoPose) -> tuple[float, float, float]:
    usable_half_x = max(surface_obj.extent[0] * 0.5 - held_obj.extent[0] * 0.5, 0.0)
    usable_half_y = max(surface_obj.extent[1] * 0.5 - held_obj.extent[1] * 0.5, 0.0)
    x = surface_obj.center[0] + _clamp(ego_pose.x - surface_obj.center[0], -usable_half_x, usable_half_x)
    y = surface_obj.center[1] + _clamp(ego_pose.y - surface_obj.center[1], -usable_half_y, usable_half_y)
    surface_top = surface_obj.center[2] + surface_obj.extent[2] * 0.5
    z = surface_top + held_obj.extent[2] * 0.5
    return (x, y, z)

def place_inside_aabb_center(held_obj: AABBObject, container_obj: AABBObject, ego_pose: EgoPose) -> tuple[float, float, float]:
    _ = ego_pose
    states = container_obj.states if container_obj.states else {}
    interior_center = _xyz_state(states.get('container_interior_center'))
    interior_half = _xyz_state(states.get('container_interior_half_extent'))
    if interior_center is not None and interior_half is not None and _has_usable_interior_volume(interior_half):
        return _center_inside_volume(held_obj, interior_center, interior_half)
    if interior_center is not None:
        return (interior_center[0], interior_center[1], interior_center[2] + held_obj.extent[2] * 0.5 + INSIDE_PLACEMENT_CLEARANCE_M)
    container_half = (max(float(container_obj.extent[0]) * 0.5, 0.0), max(float(container_obj.extent[1]) * 0.5, 0.0), max(float(container_obj.extent[2]) * 0.5, 0.0))
    return _center_inside_volume(held_obj, container_obj.center, container_half)

def _center_inside_volume(held_obj: AABBObject, volume_center: tuple[float, float, float], volume_half_extent: tuple[float, float, float]) -> tuple[float, float, float]:
    held_half = (max(float(held_obj.extent[0]) * 0.5, 0.0), max(float(held_obj.extent[1]) * 0.5, 0.0), max(float(held_obj.extent[2]) * 0.5, 0.0))
    x = _centered_axis_in_volume(float(volume_center[0]), max(float(volume_half_extent[0]), 0.0), held_half[0], margin=INSIDE_PLACEMENT_MARGIN_M)
    y = _centered_axis_in_volume(float(volume_center[1]), max(float(volume_half_extent[1]), 0.0), held_half[1], margin=INSIDE_PLACEMENT_MARGIN_M)
    z = _resting_axis_in_volume(float(volume_center[2]), max(float(volume_half_extent[2]), 0.0), held_half[2], clearance=INSIDE_PLACEMENT_CLEARANCE_M)
    return (x, y, z)

def _centered_axis_in_volume(center: float, volume_half: float, object_half: float, *, margin: float) -> float:
    low = center - volume_half + object_half + margin
    high = center + volume_half - object_half - margin
    if low > high:
        return center
    return _clamp(center, low, high)

def _resting_axis_in_volume(center: float, volume_half: float, object_half: float, *, clearance: float) -> float:
    low = center - volume_half + object_half + clearance
    high = center + volume_half - object_half - clearance
    if low > high:
        return center
    return low

def _xyz_state(value: object) -> tuple[float, float, float] | None:
    try:
        if value is None or len(value) < 3:
            return None
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None

def _has_usable_interior_volume(half_extent: tuple[float, float, float]) -> bool:
    return max(half_extent) > INTERIOR_VOLUME_EPSILON_M and half_extent[2] > INTERIOR_VOLUME_EPSILON_M

def release_aabb_in_front(held_obj: AABBObject, ego_pose: EgoPose, occupancy: OccupancyMap | None, floor_height: float=0.0) -> tuple[tuple[float, float, float] | None, CheckResult]:
    if occupancy is None:
        return (None, CheckResult.fail('missing_occupancy', target=held_obj.name))
    forward = (math.cos(ego_pose.yaw), math.sin(ego_pose.yaw))
    right = (-forward[1], forward[0])
    candidates = [(0.75, 0.0), (0.6, 0.0), (0.9, 0.0), (0.75, 0.25), (0.75, -0.25), (0.55, 0.25), (0.55, -0.25), (1.1, 0.0), (0.9, 0.35), (0.9, -0.35), (0.25, 0.55), (0.25, -0.55), (0.0, 0.6), (0.0, -0.6), (-0.35, 0.0)]
    rejected = []
    for forward_distance, lateral_distance in candidates:
        xy = (ego_pose.x + forward[0] * forward_distance + right[0] * lateral_distance, ego_pose.y + forward[1] * forward_distance + right[1] * lateral_distance)
        footprint_free = _aabb_xy_footprint_free(xy, held_obj.extent, occupancy)
        if not footprint_free:
            rejected.append({'xy': xy, 'forward_distance': forward_distance, 'lateral_distance': lateral_distance, 'reason': 'release_footprint_occupied'})
            continue
        center = (xy[0], xy[1], floor_height + held_obj.extent[2] * 0.5)
        return (center, CheckResult.pass_(target=held_obj.name, release_pose_type='in_front_of_ego_floor', release_center=center, release_xy=xy, forward_distance=forward_distance, lateral_distance=lateral_distance, candidate_count=len(candidates), rejected_candidate_count=len(rejected)))
    return (None, CheckResult.fail('no_free_release_pose', target=held_obj.name, candidate_count=len(candidates), rejected_candidates=rejected))

def _aabb_xy_footprint_free(center_xy: tuple[float, float], extent: tuple[float, float, float], occupancy: OccupancyMap) -> bool:
    half_x = extent[0] * 0.5
    half_y = extent[1] * 0.5
    sample_points = [center_xy, (center_xy[0] - half_x, center_xy[1] - half_y), (center_xy[0] - half_x, center_xy[1] + half_y), (center_xy[0] + half_x, center_xy[1] - half_y), (center_xy[0] + half_x, center_xy[1] + half_y)]
    return all((occupancy.is_free(point, eroded=False) for point in sample_points))

def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)

def _interpolate_pose_turn(start_pose: EgoPose, end_pose: EgoPose, turn_spacing_degrees: float) -> list[EgoPose]:
    yaw_delta = normalize_angle(end_pose.yaw - start_pose.yaw)
    pitch_delta = end_pose.pitch - start_pose.pitch
    step_radians = math.radians(max(turn_spacing_degrees, 0.001))
    min_steps = 2 if abs(yaw_delta) > 1e-09 or abs(pitch_delta) > 1e-09 else 1
    steps = max(min_steps, int(math.ceil(abs(yaw_delta) / step_radians)), int(math.ceil(abs(pitch_delta) / step_radians)))
    poses: list[EgoPose] = []
    for step in range(1, steps + 1):
        t = step / steps
        poses.append(EgoPose(x=start_pose.x, y=start_pose.y, z=start_pose.z, yaw=normalize_angle(start_pose.yaw + yaw_delta * t), pitch=start_pose.pitch + pitch_delta * t))
    return poses

def _point_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

def _is_navigation_barrier_object(obj: AABBObject) -> bool:
    states = getattr(obj, 'states', {}) or {}
    if isinstance(states, dict):
        barrier = str(states.get('navigation_barrier') or '').lower()
        if barrier in {'door', 'openable_door', 'navigation_barrier'}:
            return True
        if bool(states.get('blocks_navigation_when_closed')):
            return True
    category = str(getattr(obj, 'category', '') or '').lower()
    name = str(getattr(obj, 'name', '') or '').lower()
    return 'door' in category or 'door' in name
