from __future__ import annotations
from typing import Any, Mapping
DEFAULT_INTERACTION_DISTANCE_M = 1.5

def object_visible_from_metadata(metadata: Mapping[str, Any], object_id: str) -> bool | None:
    for item in metadata.get('objects', []) or []:
        if isinstance(item, Mapping) and item.get('objectId') == object_id:
            visible = item.get('visible')
            return visible if type(visible) is bool else None
    return None

def interaction_ready_from_metadata(metadata: Mapping[str, Any], object_id: str, *, max_distance_m: float=DEFAULT_INTERACTION_DISTANCE_M) -> bool | None:
    for item in metadata.get('objects', []) or []:
        if isinstance(item, Mapping) and item.get('objectId') == object_id:
            visible = item.get('visible')
            if type(visible) is not bool:
                return None
            distance = item.get('distance')
            if isinstance(distance, bool) or not isinstance(distance, (int, float)):
                return None
            return bool(visible) and float(distance) <= float(max_distance_m)
    return None

def room_id_for_position(position: Mapping[str, Any] | None, house: Mapping[str, Any] | None, *, metadata_room_id: Any=None) -> str | None:
    if isinstance(metadata_room_id, str) and metadata_room_id:
        return metadata_room_id
    if not isinstance(position, Mapping) or not isinstance(house, Mapping):
        return None
    for index, room in enumerate(house.get('rooms', []) or []):
        if not isinstance(room, Mapping):
            continue
        polygon = room.get('floorPolygon') or room.get('floor_polygon')
        if point_in_floor_polygon(position, polygon):
            return str(room.get('id') or room.get('roomId') or room.get('room_id') or 'room_{}'.format(index))
    return None

def point_in_floor_polygon(point: Mapping[str, Any], polygon: Any) -> bool:
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
    previous = len(vertices) - 1
    for index, (current_x, current_z) in enumerate(vertices):
        previous_x, previous_z = vertices[previous]
        if _point_on_segment(x, z, current_x, current_z, previous_x, previous_z):
            return True
        if (current_z > z) != (previous_z > z) and x < (previous_x - current_x) * (z - current_z) / (previous_z - current_z or 1e-12) + current_x:
            inside = not inside
        previous = index
    return inside

def _point_on_segment(x: float, z: float, first_x: float, first_z: float, second_x: float, second_z: float) -> bool:
    cross = (x - first_x) * (second_z - first_z) - (z - first_z) * (second_x - first_x)
    if abs(cross) > 1e-07:
        return False
    dot = (x - first_x) * (second_x - first_x) + (z - first_z) * (second_z - first_z)
    if dot < -1e-07:
        return False
    squared = (second_x - first_x) ** 2 + (second_z - first_z) ** 2
    return dot <= squared + 1e-07
__all__ = ['DEFAULT_INTERACTION_DISTANCE_M', 'interaction_ready_from_metadata', 'object_visible_from_metadata', 'point_in_floor_polygon', 'room_id_for_position']
