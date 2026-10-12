from __future__ import annotations
from collections.abc import Sequence
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject
CONTACT_CLEARANCE_M = 0.005

def aabb_top_z(obj: AABBObject) -> float:
    return float(obj.center[2] + obj.extent[2] * 0.5)

def aabb_bottom_z(obj: AABBObject) -> float:
    return float(obj.center[2] - obj.extent[2] * 0.5)

def root_position_for_contact_center(obj: AABBObject, aabb_offset_from_root: Sequence[float], center_xy: Sequence[float], support_top_z: float, *, clearance: float=CONTACT_CLEARANCE_M) -> tuple[float, float, float]:
    desired_center = (float(center_xy[0]), float(center_xy[1]), float(support_top_z) + obj.extent[2] * 0.5 + float(clearance))
    return (desired_center[0] - float(aabb_offset_from_root[0]), desired_center[1] - float(aabb_offset_from_root[1]), desired_center[2] - float(aabb_offset_from_root[2]))

def clamp_offset_to_support(support: AABBObject, obj: AABBObject, requested_offset_xy: Sequence[float], *, margin: float=0.05) -> tuple[float, float]:
    usable_half_x = max(support.extent[0] * 0.5 - obj.extent[0] * 0.5 - float(margin), 0.0)
    usable_half_y = max(support.extent[1] * 0.5 - obj.extent[1] * 0.5 - float(margin), 0.0)
    return (_clamp(float(requested_offset_xy[0]), -usable_half_x, usable_half_x), _clamp(float(requested_offset_xy[1]), -usable_half_y, usable_half_y))

def support_can_hold_offsets(support: AABBObject, objects_by_name: dict[str, AABBObject], requested_offsets_xy: dict[str, Sequence[float]], *, margin: float=0.05) -> bool:
    for name, offset_xy in requested_offsets_xy.items():
        obj = objects_by_name[name]
        clamped = clamp_offset_to_support(support, obj, offset_xy, margin=margin)
        if abs(clamped[0] - float(offset_xy[0])) > 1e-06 or abs(clamped[1] - float(offset_xy[1])) > 1e-06:
            return False
    return True

def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
