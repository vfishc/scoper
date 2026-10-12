from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence
import numpy as np
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose, normalize_angle, yaw_towards
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject, OccupancyMap, cells_on_segment
from scoper.benchmark.behavior.execution_core.trace import CheckResult

@dataclass
class NavigationConfig:
    interaction_distance: tuple[float, float] = (0.1, 1.2)
    optimal_observation_distance: float = 0.3
    candidate_count: int = 192
    frame_spacing: float = 0.15
    turn_spacing_degrees: float = 12.0
    final_facing_threshold_degrees: float = 60.0
    simplify_path: bool = True

@dataclass(frozen=True)
class NavigationFace:
    center: tuple[float, float, float]
    normal_xy: tuple[float, float]
    source: str = 'unknown'

    @property
    def xy_center(self) -> tuple[float, float]:
        return (self.center[0], self.center[1])

@dataclass
class NavigationTarget:
    name: str
    category: str
    center: tuple[float, float, float]
    extent: tuple[float, float, float]
    interaction_faces: tuple[NavigationFace, ...] = ()

    @classmethod
    def from_aabb_object(cls, obj: AABBObject) -> 'NavigationTarget':
        return cls(name=obj.name, category=obj.category, center=obj.center, extent=obj.extent, interaction_faces=_interaction_faces_from_object(obj))

    @property
    def xy_center(self) -> tuple[float, float]:
        return (self.center[0], self.center[1])

    def facing_xy_for_goal(self, goal_xy: tuple[float, float]) -> tuple[float, float]:
        if not self.interaction_faces:
            return self.xy_center
        best = max(self.interaction_faces, key=lambda face: _face_alignment_score(face, goal_xy))
        return best.xy_center

@dataclass
class NavigationPlan:
    success: bool
    check: CheckResult
    path_xy: list[tuple[float, float]]
    frame_poses: list[EgoPose]
    target_yaw: float | None
    selected_goal_xy: tuple[float, float] | None
    rejected_candidates: list[dict]

def plan_navigation(start_pose: EgoPose, target: NavigationTarget, occupancy: OccupancyMap, config: NavigationConfig | None=None) -> NavigationPlan:
    config = config or NavigationConfig()
    rejected: list[dict] = []
    if not occupancy.is_free(start_pose.xy):
        check = CheckResult.fail('start_occupied', start_cell=occupancy.world_to_cell(start_pose.xy))
        return NavigationPlan(False, check, [], [], None, None, rejected)
    candidates = sample_interaction_candidates(target, config.candidate_count, config.interaction_distance)
    optimal_distance = max(float(config.optimal_observation_distance), 0.0)
    best: tuple[tuple[float, float, float], list[tuple[int, int]], tuple[float, float], float, tuple[float, float], float, float, float] | None = None
    start_cell = occupancy.world_to_cell(start_pose.xy)
    start_component = occupancy.component_at(start_pose.xy)
    for goal_xy in candidates:
        candidate_record = {'goal_xy': goal_xy}
        facing_xy = target.facing_xy_for_goal(goal_xy)
        distance_to_interaction_facing = math.dist(goal_xy, facing_xy)
        distance_to_target_center = math.dist(goal_xy, target.xy_center)
        observation_distance_error = abs(distance_to_target_center - optimal_distance)
        candidate_record['target_facing_xy'] = facing_xy
        candidate_record['distance_to_target'] = distance_to_target_center
        candidate_record['distance_to_target_center_m'] = distance_to_target_center
        candidate_record['distance_to_interaction_facing_m'] = distance_to_interaction_facing
        candidate_record['optimal_observation_distance_m'] = optimal_distance
        candidate_record['observation_distance_error_m'] = observation_distance_error
        if not occupancy.is_free(goal_xy):
            candidate_record['reason'] = 'goal_occupied'
            rejected.append(candidate_record)
            continue
        goal_component = occupancy.component_at(goal_xy)
        candidate_record['goal_component'] = goal_component
        if start_component != -1 and goal_component != -1 and (start_component != goal_component):
            candidate_record['reason'] = 'target_not_reachable'
            candidate_record['start_component'] = start_component
            rejected.append(candidate_record)
            continue
        goal_cell = occupancy.world_to_cell(goal_xy)
        path_cells = astar_grid(occupancy.eroded_free, start_cell, goal_cell)
        if path_cells is None:
            candidate_record['reason'] = 'target_not_reachable'
            rejected.append(candidate_record)
            continue
        path_xy = cells_to_world_path(path_cells, occupancy, start_pose.xy, goal_xy)
        validation = occupancy.validate_path(path_xy)
        if not validation.allowed:
            candidate_record['reason'] = validation.reasons[0] if validation.reasons else 'path_collides'
            candidate_record['metrics'] = validation.metrics
            rejected.append(candidate_record)
            continue
        length = path_length(path_xy)
        yaw = yaw_towards(goal_xy, facing_xy)
        yaw_delta = abs(normalize_angle(yaw - start_pose.yaw))
        score_key = (observation_distance_error, length, yaw_delta)
        candidate_record['navigation_score_policy'] = 'closest_to_target_center_observation_distance'
        candidate_record['score_sort_key'] = list(score_key)
        candidate_record['score'] = observation_distance_error
        if best is None or score_key < best[0]:
            best = (score_key, path_cells, goal_xy, yaw, facing_xy, distance_to_target_center, distance_to_interaction_facing, yaw_delta)
    if best is None:
        check = CheckResult.fail('target_not_reachable', rejected_candidates=len(rejected), start_component=start_component)
        return NavigationPlan(False, check, [], [], None, None, rejected)
    score_key, path_cells, goal_xy, final_yaw, facing_xy, distance_to_target_center, distance_to_interaction_facing, yaw_delta = best
    simplified_path_rejected = None
    if config.simplify_path:
        simplified_cells = simplify_cell_path(path_cells, occupancy)
        simplified_xy = cells_to_world_path(simplified_cells, occupancy, start_pose.xy, goal_xy)
        simplified_validation = occupancy.validate_path(simplified_xy)
        if simplified_validation.allowed:
            path_cells = simplified_cells
        else:
            simplified_path_rejected = {'reasons': list(simplified_validation.reasons), 'metrics': dict(simplified_validation.metrics), 'unsimplified_cells': len(path_cells), 'simplified_cells': len(simplified_cells)}
    path_xy = cells_to_world_path(path_cells, occupancy, start_pose.xy, goal_xy)
    frame_poses = interpolate_path(path_xy, start_pose, final_yaw, config.frame_spacing, config.turn_spacing_degrees)
    metrics = dict(rejected_candidates=len(rejected), path_waypoints=len(path_xy), frames=len(frame_poses), path_length=path_length(path_xy), target_facing_xy=facing_xy, interaction_face_sources=_interaction_face_sources(target), navigation_score_policy='closest_to_target_center_observation_distance', optimal_observation_distance_m=optimal_distance, selected_distance_to_target_center_m=distance_to_target_center, selected_distance_to_interaction_facing_m=distance_to_interaction_facing, selected_observation_distance_error_m=score_key[0], selected_yaw_delta_from_start_rad=yaw_delta, selected_score_sort_key=list(score_key))
    if simplified_path_rejected is not None:
        metrics['simplified_path_rejected'] = simplified_path_rejected
    check = CheckResult.pass_(**metrics)
    return NavigationPlan(True, check, path_xy, frame_poses, final_yaw, goal_xy, rejected)

def plan_navigation_to_pose(start_pose: EgoPose, goal_xy: Sequence[float], occupancy: OccupancyMap, config: NavigationConfig | None=None, *, final_yaw: float | None=None, facing_xy: Sequence[float] | None=None) -> NavigationPlan:
    config = config or NavigationConfig()
    rejected: list[dict[str, Any]] = []
    try:
        goal = (float(goal_xy[0]), float(goal_xy[1]))
    except Exception:
        check = CheckResult.fail('goal_pose_invalid', goal_xy=goal_xy)
        return NavigationPlan(False, check, [], [], None, None, rejected)
    if not occupancy.is_free(start_pose.xy):
        check = CheckResult.fail('start_occupied', start_cell=occupancy.world_to_cell(start_pose.xy))
        return NavigationPlan(False, check, [], [], None, None, rejected)
    if not occupancy.is_free(goal):
        check = CheckResult.fail('goal_occupied', goal_cell=occupancy.world_to_cell(goal), goal_xy=goal)
        return NavigationPlan(False, check, [], [], None, None, rejected)
    start_cell = occupancy.world_to_cell(start_pose.xy)
    goal_cell = occupancy.world_to_cell(goal)
    start_component = occupancy.component_at(start_pose.xy)
    goal_component = occupancy.component_at(goal)
    if start_component != -1 and goal_component != -1 and (start_component != goal_component):
        check = CheckResult.fail('target_not_reachable', start_component=start_component, goal_component=goal_component, goal_xy=goal)
        return NavigationPlan(False, check, [], [], None, None, rejected)
    path_cells = astar_grid(occupancy.eroded_free, start_cell, goal_cell)
    if path_cells is None:
        check = CheckResult.fail('target_not_reachable', start_component=start_component, goal_component=goal_component, goal_xy=goal)
        return NavigationPlan(False, check, [], [], None, None, rejected)
    simplified_path_rejected = None
    if config.simplify_path:
        simplified_cells = simplify_cell_path(path_cells, occupancy)
        simplified_xy = cells_to_world_path(simplified_cells, occupancy, start_pose.xy, goal)
        simplified_validation = occupancy.validate_path(simplified_xy)
        if simplified_validation.allowed:
            path_cells = simplified_cells
        else:
            simplified_path_rejected = {'reasons': list(simplified_validation.reasons), 'metrics': dict(simplified_validation.metrics), 'unsimplified_cells': len(path_cells), 'simplified_cells': len(simplified_cells)}
    path_xy = cells_to_world_path(path_cells, occupancy, start_pose.xy, goal)
    validation = occupancy.validate_path(path_xy)
    if not validation.allowed:
        return NavigationPlan(False, validation, [], [], None, None, rejected)
    resolved_facing_xy = None
    if facing_xy is not None:
        try:
            resolved_facing_xy = (float(facing_xy[0]), float(facing_xy[1]))
        except Exception:
            resolved_facing_xy = None
    if final_yaw is None and resolved_facing_xy is not None:
        final_yaw = yaw_towards(goal, resolved_facing_xy)
    if final_yaw is None and len(path_xy) >= 2 and (math.dist(path_xy[-2], path_xy[-1]) > 1e-06):
        final_yaw = yaw_towards(path_xy[-2], path_xy[-1])
    if final_yaw is None:
        final_yaw = start_pose.yaw
    final_yaw = normalize_angle(float(final_yaw))
    frame_poses = interpolate_path(path_xy, start_pose, final_yaw, config.frame_spacing, config.turn_spacing_degrees)
    metrics = dict(path_waypoints=len(path_xy), frames=len(frame_poses), path_length=path_length(path_xy), selected_goal_xy=goal, target_facing_xy=resolved_facing_xy, start_component=start_component, goal_component=goal_component, navigation_target_type='pose')
    if simplified_path_rejected is not None:
        metrics['simplified_path_rejected'] = simplified_path_rejected
    check = CheckResult.pass_(**metrics)
    return NavigationPlan(True, check, path_xy, frame_poses, final_yaw, goal, rejected)

def sample_interaction_candidates(target: NavigationTarget, count: int, distance_band: tuple[float, float]) -> list[tuple[float, float]]:
    count = max(int(count), 0)
    if count <= 0:
        return []
    if target.interaction_faces:
        return sample_interaction_face_candidates(target, count, distance_band)
    clearance_values = _interaction_distance_values(distance_band)
    footprint_radius = max(0.5 * max(float(target.extent[0]), float(target.extent[1])), 0.0)
    radii = _unique_floats([footprint_radius + clearance for clearance in clearance_values])
    if not radii:
        radii = (footprint_radius + 0.2,)
    samples_per_radius = max(1, int(math.ceil(count / len(radii))))
    candidates: list[tuple[float, float]] = []
    for angle_idx in range(samples_per_radius):
        for radius_idx, radius in enumerate(radii):
            if len(candidates) >= count:
                return candidates
            stagger = radius_idx / max(len(radii), 1) * 0.5
            angle = 2.0 * math.pi * (angle_idx + stagger) / samples_per_radius
            candidates.append((target.center[0] + math.cos(angle) * radius, target.center[1] + math.sin(angle) * radius))
    return candidates

def sample_interaction_face_candidates(target: NavigationTarget, count: int, distance_band: tuple[float, float]) -> list[tuple[float, float]]:
    count = max(int(count), 0)
    if count <= 0:
        return []
    distance_values = _interaction_distance_values(distance_band)
    lateral_limit = max(0.05, 0.5 * max(float(target.extent[0]), float(target.extent[1])))
    lateral_offsets = tuple((round(-lateral_limit + 2.0 * lateral_limit * index / 12.0, 4) for index in range(13)))
    candidates: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    faces = [face for face in target.interaction_faces if _unit2(face.normal_xy) is not None]
    if not faces:
        return candidates
    idx = 0
    max_attempts = max(count * 8, len(faces) * len(distance_values) * len(lateral_offsets))
    while len(candidates) < max(count, 0) and idx < max_attempts:
        face = faces[idx % len(faces)]
        normal = _unit2(face.normal_xy)
        assert normal is not None
        tangent = (-normal[1], normal[0])
        distance = distance_values[idx // len(faces) % len(distance_values)]
        offset = lateral_offsets[idx // (len(faces) * len(distance_values)) % len(lateral_offsets)]
        x = face.center[0] + normal[0] * distance + tangent[0] * offset
        y = face.center[1] + normal[1] * distance + tangent[1] * offset
        key = (round(float(x), 4), round(float(y), 4))
        if key not in seen:
            seen.add(key)
            candidates.append((float(x), float(y)))
        idx += 1
    return candidates

def cells_to_world_path(path_cells: Sequence[Sequence[int]], occupancy: OccupancyMap, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
    if len(path_cells) <= 1:
        return [start_xy, goal_xy]
    middle = [occupancy.cell_to_world(cell) for cell in path_cells[1:-1]]
    return [start_xy, *middle, goal_xy]

def astar_grid(free_grid: np.ndarray, start: Sequence[int], goal: Sequence[int]) -> list[tuple[int, int]] | None:
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    height, width = free_grid.shape
    if not _in_bounds(start, height, width) or not _in_bounds(goal, height, width):
        return None
    if not free_grid[start] or not free_grid[goal]:
        return None
    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (_heuristic(start, goal), 0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far = {start: 0.0}
    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct_path(came_from, current)
        if cost > cost_so_far[current]:
            continue
        for dr, dc, step_cost in neighbors:
            nxt = (current[0] + dr, current[1] + dc)
            if not _in_bounds(nxt, height, width) or not free_grid[nxt]:
                continue
            if dr != 0 and dc != 0 and (not free_grid[current[0] + dr, current[1]] or not free_grid[current[0], current[1] + dc]):
                continue
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + _heuristic(nxt, goal)
                heapq.heappush(open_heap, (priority, new_cost, nxt))
                came_from[nxt] = current
    return None

def simplify_cell_path(path: list[tuple[int, int]], occupancy: OccupancyMap) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    anchor_index = 0
    while anchor_index < len(path) - 1:
        next_index = len(path) - 1
        while next_index > anchor_index + 1:
            if _line_is_free(path[anchor_index], path[next_index], occupancy):
                break
            next_index -= 1
        simplified.append(path[next_index])
        anchor_index = next_index
    return simplified

def interpolate_path(path_xy: Sequence[Sequence[float]], start_pose: EgoPose, final_yaw: float, frame_spacing: float, turn_spacing_degrees: float=12.0) -> list[EgoPose]:
    if not path_xy:
        return []
    z = start_pose.z
    navigation_pitch = 0.0
    poses = [start_pose.copy()]
    for start_xy, end_xy in zip(path_xy[:-1], path_xy[1:]):
        sx, sy = (float(start_xy[0]), float(start_xy[1]))
        ex, ey = (float(end_xy[0]), float(end_xy[1]))
        segment_len = math.dist((sx, sy), (ex, ey))
        if segment_len <= 1e-06:
            continue
        steps = max(1, int(math.ceil(segment_len / frame_spacing)))
        segment_yaw = yaw_towards((sx, sy), (ex, ey))
        _append_turn_frames(poses, segment_yaw, z, navigation_pitch, turn_spacing_degrees)
        for step in range(1, steps + 1):
            t = step / steps
            x = sx + (ex - sx) * t
            y = sy + (ey - sy) * t
            poses.append(EgoPose(x=x, y=y, z=z, yaw=segment_yaw, pitch=navigation_pitch))
    _append_turn_frames(poses, final_yaw, z, navigation_pitch, turn_spacing_degrees)
    return poses

def path_length(path_xy: Sequence[Sequence[float]]) -> float:
    return sum((math.dist(a, b) for a, b in zip(path_xy[:-1], path_xy[1:])))

def _line_is_free(start: Sequence[int], end: Sequence[int], occupancy: OccupancyMap) -> bool:
    return all((occupancy.is_cell_free(cell) for cell in cells_on_segment(start, end)))

def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _in_bounds(cell: tuple[int, int], height: int, width: int) -> bool:
    return 0 <= cell[0] < height and 0 <= cell[1] < width

def _reconstruct_path(came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def _lerp_angle(a: float, b: float, t: float) -> float:
    return normalize_angle(a + normalize_angle(b - a) * t)

def _append_turn_frames(poses: list[EgoPose], target_yaw: float, z: float, target_pitch: float, turn_spacing_degrees: float) -> None:
    if not poses:
        return
    start = poses[-1]
    yaw_delta = normalize_angle(target_yaw - start.yaw)
    pitch_delta = target_pitch - start.pitch
    if abs(yaw_delta) < 1e-06 and abs(pitch_delta) < 1e-06:
        return
    step_radians = math.radians(max(turn_spacing_degrees, 0.001))
    steps = max(1, int(math.ceil(abs(yaw_delta) / step_radians)), int(math.ceil(abs(pitch_delta) / step_radians)))
    for step in range(1, steps + 1):
        t = step / steps
        yaw = _lerp_angle(start.yaw, target_yaw, t)
        pitch = start.pitch + pitch_delta * t
        poses.append(EgoPose(x=start.x, y=start.y, z=z, yaw=yaw, pitch=pitch))

def _interaction_faces_from_object(obj: AABBObject) -> tuple[NavigationFace, ...]:
    states = getattr(obj, 'states', {}) or {}
    raw_faces = states.get('interaction_faces') if isinstance(states, dict) else None
    faces: list[NavigationFace] = []
    if isinstance(raw_faces, Sequence) and (not isinstance(raw_faces, (str, bytes))):
        for raw_face in raw_faces:
            face = _coerce_face(raw_face, default_z=float(obj.center[2]))
            if face is not None:
                faces.append(face)
    if faces:
        return tuple(faces)
    if not _is_openable_like(obj):
        return ()
    return tuple(_aabb_fallback_faces(obj))

def _coerce_face(raw_face: Any, default_z: float) -> NavigationFace | None:
    if isinstance(raw_face, NavigationFace):
        return raw_face
    if not isinstance(raw_face, dict):
        return None
    center_raw = raw_face.get('center') or raw_face.get('center_xyz')
    normal_raw = raw_face.get('normal_xy') or raw_face.get('normal')
    if not isinstance(center_raw, Sequence) or isinstance(center_raw, (str, bytes)):
        return None
    if not isinstance(normal_raw, Sequence) or isinstance(normal_raw, (str, bytes)):
        return None
    try:
        if len(center_raw) >= 3:
            center = (float(center_raw[0]), float(center_raw[1]), float(center_raw[2]))
        elif len(center_raw) == 2:
            center = (float(center_raw[0]), float(center_raw[1]), float(default_z))
        else:
            return None
        normal = (float(normal_raw[0]), float(normal_raw[1]))
    except Exception:
        return None
    unit = _unit2(normal)
    if unit is None:
        return None
    return NavigationFace(center=center, normal_xy=unit, source=str(raw_face.get('source') or 'metadata'))

def _is_openable_like(obj: AABBObject) -> bool:
    states = getattr(obj, 'states', {}) or {}
    abilities = tuple(getattr(obj, 'abilities', ()) or ())
    if isinstance(states, dict) and 'open' in states:
        return True
    return 'openable' in abilities

def _is_door_like(obj: AABBObject) -> bool:
    states = getattr(obj, 'states', {}) or {}
    if isinstance(states, dict) and str(states.get('navigation_barrier') or '').lower() == 'door':
        return True
    category = str(getattr(obj, 'category', '') or '').lower()
    name = str(getattr(obj, 'name', '') or '').lower()
    return 'door' in category or 'door' in name

def _aabb_fallback_faces(obj: AABBObject) -> list[NavigationFace]:
    cx, cy, cz = (float(obj.center[0]), float(obj.center[1]), float(obj.center[2]))
    hx = max(float(obj.extent[0]) * 0.5, 0.0)
    hy = max(float(obj.extent[1]) * 0.5, 0.0)
    candidates = [NavigationFace((cx + hx, cy, cz), (1.0, 0.0), 'aabb_openable_face_fallback'), NavigationFace((cx - hx, cy, cz), (-1.0, 0.0), 'aabb_openable_face_fallback'), NavigationFace((cx, cy + hy, cz), (0.0, 1.0), 'aabb_openable_face_fallback'), NavigationFace((cx, cy - hy, cz), (0.0, -1.0), 'aabb_openable_face_fallback')]
    if not _is_door_like(obj):
        return candidates
    if hx <= hy:
        return candidates[:2]
    return candidates[2:]

def _unit2(value: Sequence[float]) -> tuple[float, float] | None:
    try:
        x, y = (float(value[0]), float(value[1]))
    except Exception:
        return None
    norm = math.hypot(x, y)
    if norm < 1e-09:
        return None
    return (x / norm, y / norm)

def _face_alignment_score(face: NavigationFace, goal_xy: tuple[float, float]) -> float:
    normal = _unit2(face.normal_xy)
    if normal is None:
        return -math.inf
    vector = (float(goal_xy[0]) - face.center[0], float(goal_xy[1]) - face.center[1])
    unit = _unit2(vector)
    if unit is None:
        return -math.inf
    distance = math.dist(goal_xy, face.xy_center)
    return unit[0] * normal[0] + unit[1] * normal[1] - 0.02 * distance

def _interaction_distance_values(distance_band: tuple[float, float]) -> tuple[float, ...]:
    try:
        min_dist = max(float(distance_band[0]), 0.05)
        max_dist = max(float(distance_band[1]), min_dist)
    except Exception:
        min_dist, max_dist = (0.1, 1.8)
    span = max_dist - min_dist
    absolute_values = (min_dist, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5, max_dist)
    fractional_values = tuple((min_dist + span * fraction for fraction in (0.0, 0.08, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0)))
    values = [min(max(value, min_dist), max_dist) for value in (*absolute_values, *fractional_values) if min_dist - 1e-06 <= value <= max_dist + 1e-06]
    return tuple(sorted(_unique_floats(values)))

def _unique_floats(values: Sequence[float]) -> tuple[float, ...]:
    out: list[float] = []
    for value in values:
        value = float(value)
        if not any((abs(value - existing) < 1e-06 for existing in out)):
            out.append(value)
    return tuple(out)

def _interaction_face_sources(target: NavigationTarget) -> list[str]:
    return sorted({str(face.source) for face in target.interaction_faces})
