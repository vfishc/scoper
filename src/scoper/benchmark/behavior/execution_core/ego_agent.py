from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

@dataclass
class EgoPose:
    x: float
    y: float
    z: float = 1.55
    yaw: float = 0.0
    pitch: float = 0.0

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def copy(self) -> 'EgoPose':
        return EgoPose(x=self.x, y=self.y, z=self.z, yaw=self.yaw, pitch=self.pitch)

    def as_dict(self) -> dict:
        return {'x': self.x, 'y': self.y, 'z': self.z, 'yaw': self.yaw, 'pitch': self.pitch}

@dataclass
class EgoState:
    pose: EgoPose
    floor: int = 0
    held_object_name: str | None = None
    held_visual_name: str | None = None

    def as_dict(self) -> dict:
        return {'pose': self.pose.as_dict(), 'floor': self.floor, 'held_object_name': self.held_object_name, 'held_visual_name': self.held_visual_name}

def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def yaw_to_quat_xyzw(yaw: float, pitch: float=0.0) -> tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    x = -sp * sy
    y = sp * cy
    z = cp * sy
    w = cp * cy
    return (x, y, z, w)

def ego_camera_quat_xyzw(yaw: float, pitch: float=0.0) -> tuple[float, float, float, float]:
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    return look_at_quat_xyzw(forward)

def look_at_quat_xyzw(forward: Sequence[float]) -> tuple[float, float, float, float]:
    fx, fy, fz = _normalize3(forward)
    z_axis = (-fx, -fy, -fz)
    world_up = (0.0, 0.0, 1.0)
    x_axis = _cross3((fx, fy, fz), world_up)
    if _norm3(x_axis) < 1e-06:
        x_axis = (1.0, 0.0, 0.0)
    else:
        x_axis = _normalize3(x_axis)
    y_axis = _normalize3(_cross3(z_axis, x_axis))
    return _mat3_to_quat_xyzw(((x_axis[0], y_axis[0], z_axis[0]), (x_axis[1], y_axis[1], z_axis[1]), (x_axis[2], y_axis[2], z_axis[2])))

def yaw_towards(source_xy: tuple[float, float], target_xy: tuple[float, float]) -> float:
    return math.atan2(target_xy[1] - source_xy[1], target_xy[0] - source_xy[0])

def _cross3(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

def _norm3(value: Sequence[float]) -> float:
    return math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])

def _normalize3(value: Sequence[float]) -> tuple[float, float, float]:
    norm = _norm3(value)
    if norm < 1e-09:
        raise ValueError(f'Cannot normalize near-zero vector: {value}')
    return (value[0] / norm, value[1] / norm, value[2] / norm)

def _mat3_to_quat_xyzw(matrix: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float, float]:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)
