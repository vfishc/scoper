from __future__ import annotations
import heapq
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
GridKey = Tuple[int, int]
XzBounds = Tuple[float, float, float, float]
DEFAULT_GRID_STEP = 0.25
DEFAULT_CAMERA_HEIGHT = 0.675
SURFACE_TARGET_Y_OFFSET = 0.55

def estimate_route_visibility(agent_initial_pose: Mapping[str, Any], primary_object_pos: Mapping[str, Any], primary_receptacle_pos: Mapping[str, Any], candidate_surface_pos: Mapping[str, Any], reachable_positions: Sequence[Mapping[str, Any]], large_objects_2d_bboxes: Sequence[Mapping[str, Any]], fov_h_deg: float=90.0, fov_v_deg: float=60.0, max_visible_distance: float=5.0, grid_step: float=DEFAULT_GRID_STEP) -> dict:
    start_pos = agent_initial_pose.get('position') or {}
    target_surface_pos = _surface_visibility_position(candidate_surface_pos)
    route_points = _estimate_route_astar(reachable_positions, start_pos, primary_object_pos, primary_receptacle_pos, grid_step=grid_step)
    if not route_points:
        return {'visible': False, 'first_visible_step_index': None, 'distance_to_route': -1}
    bounds = [parsed for parsed in (_object_xz_bounds(bbox) for bbox in large_objects_2d_bboxes) if parsed is not None]
    best_distance = min((_xz_distance(point, target_surface_pos) for point in route_points))
    for step_idx, point in enumerate(route_points):
        if step_idx + 1 < len(route_points):
            facing = _direction_xz(point, route_points[step_idx + 1])
        else:
            facing = _direction_xz(point, primary_receptacle_pos)
        if facing is None:
            rotation = agent_initial_pose.get('rotation') or {}
            yaw = float(rotation.get('y', agent_initial_pose.get('rotation_y', 0.0)))
            facing = _direction_from_yaw(yaw)
        dist = _xz_distance(point, target_surface_pos)
        if dist > max_visible_distance:
            continue
        direction_to_candidate = _direction_xz(point, target_surface_pos)
        if direction_to_candidate is None:
            continue
        angle_diff = _angle_between_xz(facing, direction_to_candidate)
        if abs(angle_diff) > fov_h_deg / 2.0:
            continue
        camera_y = float(point.get('y', start_pos.get('y', 0.0))) + DEFAULT_CAMERA_HEIGHT
        vertical_angle = _vertical_angle(camera_y, target_surface_pos, dist)
        if abs(vertical_angle) > fov_v_deg / 2.0:
            continue
        if _line_passes_through_any_bbox(point, target_surface_pos, bounds):
            continue
        return {'visible': True, 'first_visible_step_index': step_idx, 'distance_to_route': dist}
    return {'visible': False, 'first_visible_step_index': None, 'distance_to_route': best_distance}

def _estimate_route_astar(reachable: Sequence[Mapping[str, Any]], start_pos: Mapping[str, Any], waypoint_pos: Mapping[str, Any], end_pos: Mapping[str, Any], grid_step: float=DEFAULT_GRID_STEP) -> List[dict]:
    points_by_key = _points_by_key(reachable, grid_step)
    if not points_by_key:
        return []
    start = _nearest_key(start_pos, points_by_key)
    waypoint = _nearest_key(waypoint_pos, points_by_key)
    end = _nearest_key(end_pos, points_by_key)
    if start is None or waypoint is None or end is None:
        return []
    first = _astar_path(points_by_key, start, waypoint)
    second = _astar_path(points_by_key, waypoint, end)
    if not first or not second:
        return _interpolated_route(start_pos, waypoint_pos, end_pos, grid_step)
    keys = first + second[1:]
    return [dict(points_by_key[key]) for key in keys]

def _interpolated_route(start_pos: Mapping[str, Any], waypoint_pos: Mapping[str, Any], end_pos: Mapping[str, Any], grid_step: float=DEFAULT_GRID_STEP) -> List[dict]:
    first = _interpolate_segment(start_pos, waypoint_pos, grid_step)
    second = _interpolate_segment(waypoint_pos, end_pos, grid_step)
    if not first:
        return second
    if not second:
        return first
    return first + second[1:]

def _interpolate_segment(start_pos: Mapping[str, Any], end_pos: Mapping[str, Any], grid_step: float) -> List[dict]:
    if not _has_xz(start_pos) or not _has_xz(end_pos):
        return []
    distance = _xz_distance(start_pos, end_pos)
    steps = max(1, int(math.ceil(distance / max(grid_step, 1e-06))))
    y = float(start_pos.get('y', end_pos.get('y', 0.0)))
    points = []
    for index in range(steps + 1):
        alpha = index / float(steps)
        points.append({'x': float(start_pos['x']) + (float(end_pos['x']) - float(start_pos['x'])) * alpha, 'y': y, 'z': float(start_pos['z']) + (float(end_pos['z']) - float(start_pos['z'])) * alpha})
    return points

def _line_passes_through_any_bbox(p1: Mapping[str, Any], p2: Mapping[str, Any], bboxes: Sequence[Any], exclude_object_at: Optional[Mapping[str, Any]]=None) -> bool:
    if not _has_xz(p1) or not _has_xz(p2):
        return False
    parsed_bounds = [bbox if _is_bounds_tuple(bbox) else _object_xz_bounds(bbox) for bbox in bboxes]
    for bounds in parsed_bounds:
        if bounds is None:
            continue
        if exclude_object_at is not None and _point_in_bounds(exclude_object_at, bounds):
            continue
        if _segment_intersects_bounds(p1, p2, bounds):
            return True
    return False

def _surface_visibility_position(position: Mapping[str, Any]) -> dict:
    out = dict(position)
    out['y'] = float(out.get('y', 0.0)) + SURFACE_TARGET_Y_OFFSET
    return out

def _points_by_key(reachable: Sequence[Mapping[str, Any]], grid_step: float) -> Dict[GridKey, Mapping[str, Any]]:
    points: Dict[GridKey, Mapping[str, Any]] = {}
    for point in reachable:
        if _has_xz(point):
            points[_key_for_position(point, grid_step)] = point
    return points

def _key_for_position(position: Mapping[str, Any], grid_step: float) -> GridKey:
    return (int(round(float(position['x']) / grid_step)), int(round(float(position['z']) / grid_step)))

def _nearest_key(position: Mapping[str, Any], points_by_key: Mapping[GridKey, Mapping[str, Any]]) -> Optional[GridKey]:
    if not _has_xz(position) or not points_by_key:
        return None
    return min(points_by_key, key=lambda key: _squared_xz_distance(position, points_by_key[key]))

def _astar_path(points_by_key: Mapping[GridKey, Mapping[str, Any]], start: GridKey, goal: GridKey) -> List[GridKey]:
    frontier: List[Tuple[float, GridKey]] = [(0.0, start)]
    came_from: Dict[GridKey, Optional[GridKey]] = {start: None}
    cost_so_far: Dict[GridKey, float] = {start: 0.0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            return _reconstruct_path(came_from, current)
        for neighbor in _neighbors(current):
            if neighbor not in points_by_key:
                continue
            new_cost = cost_so_far[current] + _grid_cost(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + _grid_distance(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
    return []

def _neighbors(key: GridKey) -> Iterable[GridKey]:
    x, z = key
    yield (x + 1, z)
    yield (x - 1, z)
    yield (x, z + 1)
    yield (x, z - 1)

def _reconstruct_path(came_from: Mapping[GridKey, Optional[GridKey]], current: GridKey) -> List[GridKey]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def _grid_cost(a: GridKey, b: GridKey) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

def _grid_distance(a: GridKey, b: GridKey) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _direction_xz(a: Mapping[str, Any], b: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    if not _has_xz(a) or not _has_xz(b):
        return None
    dx = float(b['x']) - float(a['x'])
    dz = float(b['z']) - float(a['z'])
    norm = math.hypot(dx, dz)
    if norm <= 1e-09:
        return None
    return (dx / norm, dz / norm)

def _direction_from_yaw(yaw: float) -> Tuple[float, float]:
    radians = math.radians(yaw)
    return (math.sin(radians), math.cos(radians))

def _angle_between_xz(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    cross = a[0] * b[1] - a[1] * b[0]
    return math.degrees(math.atan2(cross, dot))

def _vertical_angle(camera_y: float, target_pos: Mapping[str, Any], xz_distance: float) -> float:
    target_y = float(target_pos.get('y', camera_y))
    return math.degrees(math.atan2(target_y - camera_y, max(1e-06, xz_distance)))

def _object_xz_bounds(obj: Any) -> Optional[XzBounds]:
    if _is_bounds_tuple(obj):
        return obj
    if not isinstance(obj, Mapping):
        return None
    for bbox_key in ('objectOrientedBoundingBox', 'axisAlignedBoundingBox'):
        bbox = obj.get(bbox_key)
        if isinstance(bbox, Mapping):
            bounds = _corner_points_xz_bounds(bbox.get('cornerPoints'))
            if bounds is not None:
                return bounds
            bounds = _center_size_xz_bounds(bbox.get('center'), bbox.get('size'))
            if bounds is not None:
                return bounds
    bounds = _corner_points_xz_bounds(obj.get('cornerPoints'))
    if bounds is not None:
        return bounds
    bounds = _center_size_xz_bounds(obj.get('center'), obj.get('size'))
    if bounds is not None:
        return bounds
    position = obj.get('position') or {}
    if _has_xz(position):
        x = float(position['x'])
        z = float(position['z'])
        return (x - 0.25, x + 0.25, z - 0.25, z + 0.25)
    return None

def _corner_points_xz_bounds(points: Any) -> Optional[XzBounds]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return None
    xs: List[float] = []
    zs: List[float] = []
    for point in points:
        x = _vector_component(point, 'x', 0)
        z = _vector_component(point, 'z', 2)
        if x is None or z is None:
            continue
        xs.append(x)
        zs.append(z)
    if not xs or not zs:
        return None
    return (min(xs), max(xs), min(zs), max(zs))

def _center_size_xz_bounds(center: Any, size: Any) -> Optional[XzBounds]:
    cx = _vector_component(center, 'x', 0)
    cz = _vector_component(center, 'z', 2)
    sx = _vector_component(size, 'x', 0)
    sz = _vector_component(size, 'z', 2)
    if cx is None or cz is None or sx is None or (sz is None):
        return None
    half_x = abs(sx) / 2.0
    half_z = abs(sz) / 2.0
    return (cx - half_x, cx + half_x, cz - half_z, cz + half_z)

def _segment_intersects_bounds(p1: Mapping[str, Any], p2: Mapping[str, Any], bounds: XzBounds) -> bool:
    if _point_in_bounds(p1, bounds) or _point_in_bounds(p2, bounds):
        return True
    min_x, max_x, min_z, max_z = bounds
    corners = [{'x': min_x, 'z': min_z}, {'x': max_x, 'z': min_z}, {'x': max_x, 'z': max_z}, {'x': min_x, 'z': max_z}]
    rect_edges = zip(corners, corners[1:] + corners[:1])
    return any((_segments_intersect(p1, p2, a, b) for a, b in rect_edges))

def _segments_intersect(a: Mapping[str, Any], b: Mapping[str, Any], c: Mapping[str, Any], d: Mapping[str, Any]) -> bool:
    ax, az = (float(a['x']), float(a['z']))
    bx, bz = (float(b['x']), float(b['z']))
    cx, cz = (float(c['x']), float(c['z']))
    dx, dz = (float(d['x']), float(d['z']))

    def orient(px: float, pz: float, qx: float, qz: float, rx: float, rz: float) -> float:
        return (qx - px) * (rz - pz) - (qz - pz) * (rx - px)
    o1 = orient(ax, az, bx, bz, cx, cz)
    o2 = orient(ax, az, bx, bz, dx, dz)
    o3 = orient(cx, cz, dx, dz, ax, az)
    o4 = orient(cx, cz, dx, dz, bx, bz)
    if o1 == 0 and _between(cx, ax, bx) and _between(cz, az, bz):
        return True
    if o2 == 0 and _between(dx, ax, bx) and _between(dz, az, bz):
        return True
    if o3 == 0 and _between(ax, cx, dx) and _between(az, cz, dz):
        return True
    if o4 == 0 and _between(bx, cx, dx) and _between(bz, cz, dz):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

def _between(value: float, low: float, high: float) -> bool:
    return min(low, high) - 1e-09 <= value <= max(low, high) + 1e-09

def _point_in_bounds(point: Mapping[str, Any], bounds: XzBounds) -> bool:
    if not _has_xz(point):
        return False
    min_x, max_x, min_z, max_z = bounds
    return min_x <= float(point['x']) <= max_x and min_z <= float(point['z']) <= max_z

def _squared_xz_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    dx = float(a.get('x', 0.0)) - float(b.get('x', 0.0))
    dz = float(a.get('z', 0.0)) - float(b.get('z', 0.0))
    return dx * dx + dz * dz

def _xz_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return math.sqrt(_squared_xz_distance(a, b))

def _has_xz(value: Mapping[str, Any]) -> bool:
    return 'x' in value and 'z' in value

def _vector_component(value: Any, key: str, index: int) -> Optional[float]:
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

def _is_bounds_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 4 and all((isinstance(item, (int, float)) for item in value))
__all__ = ['estimate_route_visibility', '_estimate_route_astar', '_line_passes_through_any_bbox']
