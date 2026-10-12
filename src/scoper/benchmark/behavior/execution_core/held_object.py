from __future__ import annotations
from dataclasses import dataclass
import math
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose
HELD_VISUAL_SUFFIX = '__held_visual'

@dataclass(frozen=True)
class HeldObjectConfig:
    distance: float = 0.38
    lateral_offset: float = 0.16
    vertical_offset: float = -0.12
    min_relative_z: float = -0.45
    max_relative_z: float = -0.05

@dataclass(frozen=True)
class HeldObjectPose:
    object_name: str
    visual_name: str
    center: tuple[float, float, float]
    yaw: float
    pitch: float
    distance: float
    lateral_offset: float
    vertical_offset: float

def held_visual_name(object_name: str) -> str:
    return f'{object_name}{HELD_VISUAL_SUFFIX}'

def held_pose_from_ego(ego_pose: EgoPose, object_name: str, visual_name: str | None=None, config: HeldObjectConfig | None=None) -> HeldObjectPose:
    config = config or HeldObjectConfig()
    visual_name = visual_name or held_visual_name(object_name)
    forward_xy = (math.cos(ego_pose.yaw), math.sin(ego_pose.yaw))
    right_xy = (-math.sin(ego_pose.yaw), math.cos(ego_pose.yaw))
    x = ego_pose.x + forward_xy[0] * config.distance + right_xy[0] * config.lateral_offset
    y = ego_pose.y + forward_xy[1] * config.distance + right_xy[1] * config.lateral_offset
    z = ego_pose.z + math.sin(ego_pose.pitch) * config.distance + config.vertical_offset
    z = max(ego_pose.z + config.min_relative_z, min(ego_pose.z + config.max_relative_z, z))
    return HeldObjectPose(object_name=object_name, visual_name=visual_name, center=(x, y, z), yaw=ego_pose.yaw, pitch=ego_pose.pitch, distance=config.distance, lateral_offset=config.lateral_offset, vertical_offset=config.vertical_offset)
