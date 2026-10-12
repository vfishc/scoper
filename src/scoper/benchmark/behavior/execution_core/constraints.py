from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from scoper.benchmark.behavior.execution_core.actions import get_contained_substances, get_dirty_state, get_open_state, get_sliced_state, get_source_substance, get_substances, get_toggle_state, placement_relation_for_target, supports_fillable_container, supports_fluid_source, supports_cut_target, supports_cut_tool, supports_open_state, supports_pick_action, supports_pourable_source, supports_soakable, supports_toggle_state, supports_wipe_target, supports_wipe_tool
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose, EgoState, normalize_angle, yaw_towards
from scoper.benchmark.behavior.execution_core.navigation import NavigationTarget, sample_interaction_candidates
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject, OccupancyMap
from scoper.benchmark.behavior.execution_core.trace import CheckResult

@dataclass(frozen=True)
class InteractionPreconditionConfig:
    max_distance: float = 2.0
    facing_threshold_degrees: float = 60.0
    require_facing: bool = True
    require_reachable: bool = True
    interaction_distance: tuple[float, float] = (0.1, 1.8)
    candidate_count: int = 192

class EgoPreconditionChecker:

    def __init__(self, objects: Mapping[str, AABBObject] | Sequence[AABBObject], ego_state: EgoState, occupancy: OccupancyMap | None=None) -> None:
        if isinstance(objects, Mapping):
            self.objects = dict(objects)
        else:
            self.objects = {obj.name: obj for obj in objects}
        self.ego_state = ego_state
        self.occupancy = occupancy

    def check_target_exists(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        return CheckResult.pass_(target=target_name, category=obj.category)

    def check_distance(self, target_name: str, max_distance: float) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        distance = distance_to_aabb_xy(self.ego_state.pose.xy, obj)
        metrics = {'target': target_name, 'distance_to_aabb': distance, 'max_distance': max_distance}
        if distance > max_distance:
            return CheckResult.fail('too_far', **metrics)
        return CheckResult.pass_(**metrics)

    def check_facing(self, target_name: str, threshold_degrees: float) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        target_point = _target_point_for_pose(self.ego_state.pose, obj)
        target_xy = (target_point[0], target_point[1])
        target_yaw = yaw_towards(self.ego_state.pose.xy, target_xy)
        yaw_error = abs(normalize_angle(target_yaw - self.ego_state.pose.yaw))
        threshold = math.radians(threshold_degrees)
        metrics = {'target': target_name, 'target_yaw': target_yaw, 'target_facing_xy': target_xy, 'ego_yaw': self.ego_state.pose.yaw, 'yaw_error_degrees': math.degrees(yaw_error), 'facing_threshold_degrees': threshold_degrees}
        if yaw_error > threshold:
            return CheckResult.fail('not_facing_target', **metrics)
        return CheckResult.pass_(**metrics)

    def check_reachable(self, target_name: str, interaction_distance: tuple[float, float]=(0.1, 1.2), candidate_count: int=192) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        if self.occupancy is None:
            return CheckResult.fail('missing_occupancy', target=target_name)
        start_cell = self.occupancy.world_to_cell(self.ego_state.pose.xy)
        if not self.occupancy.is_cell_free(start_cell):
            return CheckResult.fail('start_occupied', target=target_name, start_cell=start_cell)
        start_component = self.occupancy.component_at(self.ego_state.pose.xy)
        target = NavigationTarget.from_aabb_object(obj)
        candidates = sample_interaction_candidates(target, candidate_count, interaction_distance)
        free_candidates = 0
        same_component_candidates = 0
        nearest_component = -1
        nearest_distance = math.inf
        for candidate_xy in candidates:
            if not self.occupancy.is_free(candidate_xy):
                continue
            free_candidates += 1
            component = self.occupancy.component_at(candidate_xy)
            distance = math.dist(self.ego_state.pose.xy, candidate_xy)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_component = component
            if start_component != -1 and component == start_component:
                same_component_candidates += 1
        metrics = {'target': target_name, 'start_component': start_component, 'target_nearest_component': nearest_component, 'candidate_count': candidate_count, 'free_candidates': free_candidates, 'same_component_candidates': same_component_candidates}
        if same_component_candidates <= 0:
            return CheckResult.fail('target_not_reachable', **metrics)
        return CheckResult.pass_(**metrics)

    def check_interaction(self, target_name: str, config: InteractionPreconditionConfig | None=None) -> CheckResult:
        config = config or InteractionPreconditionConfig()
        target_check = self.check_target_exists(target_name)
        if not target_check.allowed:
            return target_check
        checks = [self.check_distance(target_name, config.max_distance)]
        if config.require_facing:
            checks.append(self.check_facing(target_name, config.facing_threshold_degrees))
        if config.require_reachable:
            checks.append(self.check_reachable(target_name, interaction_distance=config.interaction_distance, candidate_count=config.candidate_count))
        return combine_checks(*checks)

    def check_open_affordance(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        if not supports_open_state(obj):
            return CheckResult.fail('missing_affordance_open', target=target_name, category=obj.category)
        return CheckResult.pass_(target=target_name, category=obj.category)

    def check_toggle_affordance(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        if not supports_toggle_state(obj):
            return CheckResult.fail('missing_affordance_toggle', target=target_name, category=obj.category)
        return CheckResult.pass_(target=target_name, category=obj.category)

    def check_pick_affordance(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        if not supports_pick_action(obj):
            return CheckResult.fail('missing_affordance_pick', target=target_name, category=obj.category)
        return CheckResult.pass_(target=target_name, category=obj.category)

    def check_held_wipe_tool_affordance(self) -> CheckResult:
        held_object_name = self.ego_state.held_object_name
        if held_object_name is None:
            return CheckResult.fail('not_holding_object', held_object_name=None)
        obj = self.objects.get(held_object_name)
        if obj is None:
            return CheckResult.fail('held_object_not_found', held_object_name=held_object_name)
        if not supports_wipe_tool(obj):
            return CheckResult.fail('missing_affordance_wipe_tool', held_object_name=held_object_name, held_category=obj.category)
        return CheckResult.pass_(held_object_name=held_object_name, held_category=obj.category)

    def check_held_cut_tool_affordance(self) -> CheckResult:
        held_object_name = self.ego_state.held_object_name
        if held_object_name is None:
            return CheckResult.fail('not_holding_object', held_object_name=None)
        obj = self.objects.get(held_object_name)
        if obj is None:
            return CheckResult.fail('held_object_not_found', held_object_name=held_object_name)
        if not supports_cut_tool(obj):
            return CheckResult.fail('missing_affordance_cut_tool', held_object_name=held_object_name, held_category=obj.category)
        return CheckResult.pass_(held_object_name=held_object_name, held_category=obj.category)

    def check_held_soakable_affordance(self) -> CheckResult:
        held_object_name = self.ego_state.held_object_name
        if held_object_name is None:
            return CheckResult.fail('not_holding_object', held_object_name=None)
        obj = self.objects.get(held_object_name)
        if obj is None:
            return CheckResult.fail('held_object_not_found', held_object_name=held_object_name)
        if not supports_soakable(obj):
            return CheckResult.fail('missing_affordance_soak_object', held_object_name=held_object_name, held_category=obj.category)
        return CheckResult.pass_(held_object_name=held_object_name, held_category=obj.category)

    def check_held_pourable_source_affordance(self) -> CheckResult:
        held_object_name = self.ego_state.held_object_name
        if held_object_name is None:
            return CheckResult.fail('not_holding_object', held_object_name=None)
        obj = self.objects.get(held_object_name)
        if obj is None:
            return CheckResult.fail('held_object_not_found', held_object_name=held_object_name)
        if not supports_pourable_source(obj):
            return CheckResult.fail('missing_affordance_pour_source', held_object_name=held_object_name, held_category=obj.category)
        substances = get_contained_substances(obj)
        if not substances:
            return CheckResult.fail('source_empty', held_object_name=held_object_name, held_category=obj.category, substances=[])
        return CheckResult.pass_(held_object_name=held_object_name, held_category=obj.category, substances=substances)

    def check_held_target(self, target_name: str) -> CheckResult:
        held_object_name = self.ego_state.held_object_name
        if held_object_name != target_name:
            return CheckResult.fail('target_not_held', target=target_name, held_object_name=held_object_name)
        return CheckResult.pass_(target=target_name, held_object_name=held_object_name)

    def check_fluid_source_affordance(self, source_name: str) -> CheckResult:
        obj = self.objects.get(source_name)
        if obj is None:
            return CheckResult.fail('source_not_found', source=source_name)
        if not supports_fluid_source(obj):
            return CheckResult.fail('missing_affordance_fluid_source', source=source_name, category=obj.category)
        substance = get_source_substance(obj)
        if substance is None:
            return CheckResult.fail('missing_source_substance', source=source_name, category=obj.category)
        return CheckResult.pass_(source=source_name, source_category=obj.category, substance=substance)

    def check_fluid_source_active(self, source_name: str) -> CheckResult:
        obj = self.objects.get(source_name)
        if obj is None:
            return CheckResult.fail('source_not_found', source=source_name)
        toggled = None
        if supports_toggle_state(obj):
            toggled = get_toggle_state(obj)
            if toggled is False:
                return CheckResult.fail('fluid_source_off', source=source_name, toggled_on=toggled)
        return CheckResult.pass_(source=source_name, toggled_on=toggled)

    def check_fillable_affordance(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        if not supports_fillable_container(obj):
            return CheckResult.fail('missing_affordance_fillable', target=target_name, category=obj.category)
        return CheckResult.pass_(target=target_name, category=obj.category)

    def check_container_open_for_substance(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        open_state = get_open_state(obj)
        if open_state is False:
            return CheckResult.fail('container_closed', target=target_name, category=obj.category, open_state=open_state)
        return CheckResult.pass_(target=target_name, category=obj.category, open_state=open_state)

    def check_container_has_substance(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        substances = get_contained_substances(obj)
        if not substances:
            return CheckResult.fail('container_empty', target=target_name, category=obj.category, substances=[])
        return CheckResult.pass_(target=target_name, category=obj.category, substances=substances)

    def check_container_missing_substance(self, target_name: str, substance: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        contains = substance in set(get_substances(obj, 'contains'))
        filled = substance in set(get_substances(obj, 'filled'))
        if contains or filled:
            return CheckResult.fail('container_already_contains_substance', target=target_name, category=obj.category, substance=substance, contains=contains, filled=filled)
        return CheckResult.pass_(target=target_name, category=obj.category, substance=substance)

    def check_wipe_affordance(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        if not supports_wipe_target(obj):
            return CheckResult.fail('missing_affordance_wipe', target=target_name, category=obj.category)
        return CheckResult.pass_(target=target_name, category=obj.category)

    def check_target_dirty_for_wipe(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        dirty_state = get_dirty_state(obj)
        if dirty_state is None:
            return CheckResult.fail('missing_dirty_state', target=target_name, category=obj.category)
        if dirty_state is False:
            return CheckResult.fail('target_not_dirty', target=target_name, category=obj.category, dirty=dirty_state)
        return CheckResult.pass_(target=target_name, category=obj.category, dirty=dirty_state)

    def check_cut_affordance(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        if not supports_cut_target(obj):
            return CheckResult.fail('missing_affordance_cut', target=target_name, category=obj.category)
        return CheckResult.pass_(target=target_name, category=obj.category)

    def check_target_unsliced_for_cut(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        sliced_state = get_sliced_state(obj)
        if sliced_state is None:
            return CheckResult.fail('missing_sliced_state', target=target_name, category=obj.category)
        if sliced_state is True:
            return CheckResult.fail('target_already_cut', target=target_name, category=obj.category, sliced=sliced_state)
        return CheckResult.pass_(target=target_name, category=obj.category, sliced=sliced_state)

    def check_place_affordance(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        relation = placement_relation_for_target(obj)
        if relation is None:
            return CheckResult.fail('missing_affordance_place', target=target_name, category=obj.category)
        return CheckResult.pass_(target=target_name, category=obj.category, placement_relation=relation)

    def check_container_open_for_place(self, target_name: str) -> CheckResult:
        obj = self.objects.get(target_name)
        if obj is None:
            return CheckResult.fail('target_not_found', target=target_name)
        relation = placement_relation_for_target(obj)
        if relation != 'inside':
            return CheckResult.pass_(target=target_name, placement_relation=relation)
        open_state = get_open_state(obj)
        if open_state is False:
            return CheckResult.fail('container_closed', target=target_name, category=obj.category, placement_relation=relation, open_state=open_state)
        return CheckResult.pass_(target=target_name, category=obj.category, placement_relation=relation, open_state=open_state)

    def check_place_on_affordance(self, target_name: str) -> CheckResult:
        place_check = self.check_place_affordance(target_name)
        if not place_check.allowed:
            return CheckResult.fail('missing_affordance_place_on', **place_check.metrics)
        if place_check.metrics.get('placement_relation') != 'on_top':
            return CheckResult.fail('missing_affordance_place_on', **place_check.metrics)
        return place_check

    def check_holding(self) -> CheckResult:
        held_object_name = self.ego_state.held_object_name
        if held_object_name is None:
            return CheckResult.fail('not_holding_object', held_object_name=None)
        if held_object_name not in self.objects:
            return CheckResult.fail('held_object_not_found', held_object_name=held_object_name)
        return CheckResult.pass_(held_object_name=held_object_name, held_visual_name=self.ego_state.held_visual_name)

    def check_not_holding(self) -> CheckResult:
        held_object_name = self.ego_state.held_object_name
        if held_object_name is not None:
            return CheckResult.fail('already_holding_object', held_object_name=held_object_name)
        return CheckResult.pass_(held_object_name=None)

    def check_target_not_held(self, target_name: str) -> CheckResult:
        held_object_name = self.ego_state.held_object_name
        if held_object_name == target_name:
            return CheckResult.fail('cannot_place_held_object', target=target_name, held_object_name=held_object_name)
        return CheckResult.pass_(target=target_name, held_object_name=held_object_name)

def combine_checks(*checks: CheckResult) -> CheckResult:
    reasons: list[str] = []
    metrics = {}
    for check in checks:
        for reason in check.reasons:
            if reason not in reasons:
                reasons.append(reason)
        metrics.update(check.metrics)
    if reasons:
        return CheckResult.fail(reasons, **metrics)
    return CheckResult.pass_(**metrics)

def distance_to_aabb_xy(xy: tuple[float, float], obj: AABBObject) -> float:
    dx = max(abs(xy[0] - obj.center[0]) - obj.extent[0] * 0.5, 0.0)
    dy = max(abs(xy[1] - obj.center[1]) - obj.extent[1] * 0.5, 0.0)
    return math.hypot(dx, dy)

def pose_looking_at_object(pose: EgoPose, obj: AABBObject) -> EgoPose:
    target = _target_point_for_pose(pose, obj)
    dx = target[0] - pose.x
    dy = target[1] - pose.y
    dz = target[2] - pose.z
    horizontal_distance = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(dz, horizontal_distance) if horizontal_distance > 1e-09 else 0.0
    return EgoPose(x=pose.x, y=pose.y, z=pose.z, yaw=yaw, pitch=pitch)

def _target_point_for_pose(pose: EgoPose, obj: AABBObject) -> tuple[float, float, float]:
    target = NavigationTarget.from_aabb_object(obj)
    if not target.interaction_faces:
        return (float(obj.center[0]), float(obj.center[1]), float(obj.center[2]))
    facing_xy = target.facing_xy_for_goal(pose.xy)
    best_face = min(target.interaction_faces, key=lambda face: math.dist(face.xy_center, facing_xy))
    return (float(best_face.center[0]), float(best_face.center[1]), float(best_face.center[2]))
