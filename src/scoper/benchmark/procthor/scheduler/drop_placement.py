import math
from collections import Counter
from typing import Any

def bounds(obj):
    box = obj.get('axisAlignedBoundingBox') or {}
    center, size = (box.get('center'), box.get('size'))
    if not center or not size:
        return None
    values = [float(center[axis]) for axis in ('x', 'y', 'z')]
    extents = [float(size[axis]) for axis in ('x', 'y', 'z')]
    if not all((math.isfinite(value) for value in values + extents)) or min(extents) < 0 or max(extents) <= 0:
        return None
    return (tuple((values[index] - extents[index] / 2 for index in range(3))), tuple((values[index] + extents[index] / 2 for index in range(3))))

def overlaps(first, second, margin=0.025):
    return all((first[0][axis] < second[1][axis] + margin and first[1][axis] > second[0][axis] - margin for axis in range(3)))

def inside_polygon(coordinate_x, coordinate_z, polygon):
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current['z'] > coordinate_z) != (previous['z'] > coordinate_z):
            cross = (previous['x'] - current['x']) * (coordinate_z - current['z']) / (previous['z'] - current['z']) + current['x']
            if coordinate_x < cross:
                inside = not inside
        previous = current
    return inside

def select_drop_position(env: Any, object_id: str, receptacle_id: str | None, *, search_radii=(0.2, 0.6, 1.0, 1.2), spacing=0.1):
    metadata = env.controller.last_event.metadata
    objects = {obj['objectId']: obj for obj in metadata.get('objects', [])}
    held = objects.get(object_id, {})
    held_bounds = bounds(held)
    target_bounds = bounds(objects.get(receptacle_id, {}))
    report = {'status': 'unavailable', 'reason': 'missing_geometry', 'candidates_checked': 0}
    if held_bounds is None or target_bounds is None or (not held.get('position')):
        return (None, report)
    rooms = (getattr(env, 'current_house', None) or {}).get('rooms', [])
    walls = {}
    for wall in (getattr(env, 'current_house', None) or {}).get('walls', []):
        polygon = wall.get('polygon') or []
        if polygon:
            walls[wall['id']] = (tuple((min((float(vertex[axis]) for vertex in polygon)) - 0.025 for axis in ('x', 'y', 'z'))), tuple((max((float(vertex[axis]) for vertex in polygon)) + 0.025 for axis in ('x', 'y', 'z'))))
    polygons = [room.get('floorPolygon') or room.get('floor_polygon') for room in rooms]
    polygons = [polygon for polygon in polygons if polygon and len(polygon) >= 3]
    if not polygons:
        report['reason'] = 'missing_floor_geometry'
        return (None, report)
    try:
        reachable = env.executor.reachable_positions()
    except Exception as exc:
        report.update(reason='reachable_query_failed', error=str(exc))
        return (None, report)
    agent = metadata.get('agent') or {}
    position = agent.get('position') or {}
    yaw = math.radians(float((agent.get('rotation') or {}).get('y', 0)))
    half = [(held_bounds[1][index] - held_bounds[0][index]) / 2 for index in range(3)]
    offset = [(held_bounds[1][index] + held_bounds[0][index]) / 2 - float(held['position'][axis]) for index, axis in enumerate(('x', 'y', 'z'))]
    options = []
    rejected = Counter()
    blocked_by = Counter()
    report['rejections'] = rejected
    report['blocked_by'] = blocked_by
    report['search_radii'] = list(search_radii)
    report['spacing'] = spacing
    report['wall_geometry_count'] = len(walls)
    report['missing_geometry'] = {other_id: {'position': other.get('position'), 'bbox': other.get('axisAlignedBoundingBox'), 'object_type': other.get('objectType')} for other_id, other in objects.items() if other_id != object_id and other.get('objectType') != 'Floor' and (bounds(other) is None) and (other_id not in walls)}
    reach_points = [(float(point['x']), float(point['z'])) for point in reachable]
    reach_cells = {(round(point[0] / 0.25), round(point[1] / 0.25)) for point in reach_points}
    points = {(round(point[0], 4), round(point[1], 4)) for point in reach_points}
    radius = max(search_radii)
    for index_x in range(math.ceil((target_bounds[1][0] - target_bounds[0][0] + 2 * radius) / spacing) + 1):
        for index_z in range(math.ceil((target_bounds[1][2] - target_bounds[0][2] + 2 * radius) / spacing) + 1):
            points.add((round(target_bounds[0][0] - radius + index_x * spacing, 4), round(target_bounds[0][2] - radius + index_z * spacing, 4)))
    report['candidates_generated'] = len(points)
    for coordinate_x, coordinate_z in sorted(points):
        point = {'x': coordinate_x, 'z': coordinate_z}
        dx, dz = (float(point['x']) - position['x'], float(point['z']) - position['z'])
        distance = math.hypot(dx, dz)
        if distance < max(0.45, math.hypot(half[0], half[2]) + 0.25):
            rejected['agent_clearance'] += 1
            continue
        forward = dx * math.sin(yaw) + dz * math.cos(yaw)
        target_distance = math.hypot(max(target_bounds[0][0] - point['x'], 0, point['x'] - target_bounds[1][0]), max(target_bounds[0][2] - point['z'], 0, point['z'] - target_bounds[1][2]))
        if target_distance > radius:
            continue
        stance = next((value for value in sorted(reach_points, key=lambda value: math.hypot(value[0] - coordinate_x, value[1] - coordinate_z)) if max(0.35, math.hypot(half[0], half[2]) + 0.25) <= math.hypot(value[0] - coordinate_x, value[1] - coordinate_z) <= 1.0 and all(((round((value[0] + (coordinate_x - value[0]) * fraction) / 0.25), round((value[1] + (coordinate_z - value[1]) * fraction) / 0.25)) in reach_cells for fraction in (0.25, 0.5, 0.75)))), None)
        if stance is None:
            rejected['no_recovery_approach'] += 1
            continue
        supported = False
        for polygon in polygons:
            if not all((inside_polygon(point['x'] + shift_x, point['z'] + shift_z, polygon) for shift_x in (-half[0] - 0.03, 0, half[0] + 0.03) for shift_z in (-half[2] - 0.03, 0, half[2] + 0.03))):
                continue
            supported = True
            floor_y = float(polygon[0].get('y', 0))
            center = (point['x'], floor_y + half[1] + 0.01, point['z'])
            proposed = (tuple((center[index] - half[index] for index in range(3))), tuple((center[index] + half[index] for index in range(3))))
            report['candidates_checked'] += 1
            blockers = []
            for other_id, other in objects.items():
                if other_id == object_id or other.get('objectType') == 'Floor':
                    continue
                other_bounds = walls.get(other_id) or bounds(other)
                if other_bounds is not None and overlaps(proposed, other_bounds):
                    blockers.append(other_id)
                elif other_bounds is None:
                    other_pos = other.get('position')
                    if other_pos is None or math.hypot(other_pos['x'] - center[0], other_pos['z'] - center[2]) < 2:
                        blockers.append(other_id)
            if blockers:
                rejected['collision_or_missing_geometry'] += 1
                blocked_by.update(blockers)
                continue
            root = {axis: center[index] - offset[index] for index, axis in enumerate(('x', 'y', 'z'))}
            layer = next((index for index, boundary in enumerate(search_radii) if target_distance <= boundary))
            options.append(((layer, distance + target_distance * 0.2 + (0.3 if forward < 0 else 0)), root, proposed, floor_y, stance))
        if not supported:
            rejected['unsupported_floor_footprint'] += 1
    if not options:
        report['reason'] = 'no_collision_free_floor_position'
        return (None, report)
    options.sort(key=lambda item: (item[0], item[1]['x'], item[1]['z']))
    _, selected, selected_bounds, floor_y, stance = options[0]
    report.update(status='passed', reason=None, bounds=selected_bounds, floor_y=floor_y, clearance=0.025, source='receptacle_grid_and_reachable_approach', rotation=held.get('rotation') or {}, recovery_stance={'x': stance[0], 'z': stance[1]}, valid_candidates=len(options))
    return (selected, report)
