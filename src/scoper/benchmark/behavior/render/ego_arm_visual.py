from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
R1_EGO_FORWARD_GRIP_JOINT_POS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.radians(33.0), math.radians(-33.0), math.radians(162.0), math.radians(162.0), math.radians(-108.0), math.radians(-108.0), math.radians(34.0), math.radians(90.0), math.radians(73.0), math.radians(52.0), math.radians(-65.0), math.radians(-90.0), 0.035, 0.035, 0.035, 0.035)
RIGHT_ARM_LINKS = frozenset({'right_arm_link1', 'right_arm_link2', 'right_arm_link3', 'right_arm_link4', 'right_arm_link5', 'right_arm_link6', 'right_gripper_link1', 'right_gripper_link2', 'right_eef_link'})
RIGHT_FOREARM_LINKS = frozenset({'right_arm_link4', 'right_arm_link5', 'right_arm_link6', 'right_gripper_link1', 'right_gripper_link2', 'right_eef_link'})
SHOULDER_MOUNT_LINKS = frozenset({'torso_link4'})

@dataclass(frozen=True)
class EgoArmVisualConfig:
    enabled: bool = False
    model: str = 'r1'
    name: str = 'ego_r1_right_forearm_visual'
    relative_position: tuple[float, float, float] = (-0.25, -0.35, -1.33)
    yaw_offset: float = 0.16
    held_object_eef_offset: tuple[float, float, float] = (0.08, 0.0, 0.0)
    eef_link_name: str = 'right_eef_link'
    arm_visibility: str = 'forearm'
    keep_shoulder_mount: bool = False
    disable_collisions: bool = True
    visual_only: bool = True
    fixed_base: bool = True
    reset_joint_pos: tuple[float, ...] = R1_EGO_FORWARD_GRIP_JOINT_POS
    action_normalize: bool = False

    @classmethod
    def from_value(cls, value: Any) -> 'EgoArmVisualConfig':
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError('ego_arm_visual_config must be a mapping or EgoArmVisualConfig.')
        reset_joint_pos = value.get('reset_joint_pos', cls.reset_joint_pos)
        return cls(enabled=bool(value.get('enabled', True)), model=str(value.get('model', cls.model)), name=str(value.get('name', cls.name)), relative_position=_vector3(value.get('relative_position', cls.relative_position)), yaw_offset=float(value.get('yaw_offset', cls.yaw_offset)), held_object_eef_offset=_vector3(value.get('held_object_eef_offset', cls.held_object_eef_offset)), eef_link_name=str(value.get('eef_link_name', cls.eef_link_name)), arm_visibility=str(value.get('arm_visibility', cls.arm_visibility)), keep_shoulder_mount=bool(value.get('keep_shoulder_mount', cls.keep_shoulder_mount)), disable_collisions=bool(value.get('disable_collisions', cls.disable_collisions)), visual_only=bool(value.get('visual_only', cls.visual_only)), fixed_base=bool(value.get('fixed_base', cls.fixed_base)), reset_joint_pos=tuple((float(v) for v in reset_joint_pos)), action_normalize=bool(value.get('action_normalize', cls.action_normalize)))

    def robot_config(self) -> dict[str, Any]:
        return {'model': self.model, 'name': self.name, 'obs_modalities': [], 'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0, 1.0], 'visual_only': bool(self.visual_only), 'fixed_base': bool(self.fixed_base), 'reset_joint_pos': list(self.reset_joint_pos), 'action_normalize': bool(self.action_normalize)}

    def visible_link_names(self) -> set[str]:
        if self.arm_visibility == 'full':
            names = set(RIGHT_ARM_LINKS)
        else:
            names = set(RIGHT_FOREARM_LINKS)
        if self.keep_shoulder_mount:
            names.update(SHOULDER_MOUNT_LINKS)
        return names

class EgoArmVisualSynchronizer:

    def __init__(self, config: EgoArmVisualConfig):
        self.config = config
        self._robot: Any = None
        self._last_status: dict[str, Any] = {}

    @property
    def status(self) -> dict[str, Any]:
        return dict(self._last_status or {})

    def inject_env_config(self, env_config: dict[str, Any]) -> None:
        if not self.config.enabled:
            return
        robots = env_config.setdefault('robots', [])
        if not any((str(robot.get('name') or '') == self.config.name for robot in robots if isinstance(robot, Mapping))):
            robots.append(self.config.robot_config())

    def initialize(self, env: Any) -> dict[str, Any]:
        robot = self._find_robot(env)
        if robot is None:
            self._last_status = {'enabled': True, 'success': False, 'reason': 'robot_not_found', 'robot_name': self.config.name}
            return dict(self._last_status)
        self._robot = robot
        visibility = self._hide_unselected_links(robot)
        collision = self._disable_link_collisions(robot) if self.config.disable_collisions else {'disabled': False}
        self._last_status = {'enabled': True, 'success': True, 'reason': 'initialized', 'robot_name': self.config.name, 'robot_model': self.config.model, 'arm_visibility': self.config.arm_visibility, 'visibility': visibility, 'collision': collision}
        return dict(self._last_status)

    def sync_to_ego_pose(self, ego_pose: Any) -> dict[str, Any]:
        if not self.config.enabled:
            return {'enabled': False, 'success': True, 'reason': 'disabled'}
        if self._robot is None:
            self._last_status = {'enabled': True, 'success': False, 'reason': 'robot_not_initialized', 'robot_name': self.config.name}
            return dict(self._last_status)
        position = _relative_position(ego_pose, self.config.relative_position)
        yaw = float(_pose_field(ego_pose, 'yaw', 0.0)) + float(self.config.yaw_offset)
        orientation = _yaw_to_quat_xyzw(yaw)
        try:
            self._robot.set_position_orientation(position=position, orientation=orientation, frame='scene')
        except Exception as exc:
            self._last_status = {'enabled': True, 'success': False, 'reason': 'set_position_orientation_error', 'error': repr(exc), 'robot_name': self.config.name}
            return dict(self._last_status)
        held_anchor = self._held_anchor(yaw)
        self._last_status = {'enabled': True, 'success': True, 'reason': 'synced_to_ego_pose', 'robot_name': self.config.name, 'root_position': list(position), 'root_yaw': float(yaw), 'relative_position': list(self.config.relative_position), 'held_anchor': held_anchor}
        return dict(self._last_status)

    def _find_robot(self, env: Any) -> Any:
        for robot in getattr(env, 'robots', []) or []:
            if str(getattr(robot, 'name', '')) == self.config.name:
                return robot
        return None

    def _hide_unselected_links(self, robot: Any) -> dict[str, Any]:
        links = dict(getattr(robot, 'links', {}) or {})
        visible_names = self.config.visible_link_names()
        kept: list[str] = []
        hidden: list[str] = []
        for name, link in links.items():
            visible = str(name) in visible_names
            try:
                link.visible = visible
            except Exception:
                pass
            (kept if visible else hidden).append(str(name))
        return {'kept_links': sorted(kept), 'hidden_links': sorted(hidden), 'available_links': sorted((str(name) for name in links))}

    def _disable_link_collisions(self, robot: Any) -> dict[str, Any]:
        disabled: list[str] = []
        failed: list[dict[str, str]] = []
        for name, link in dict(getattr(robot, 'links', {}) or {}).items():
            disable = getattr(link, 'disable_collisions', None)
            if not callable(disable):
                continue
            try:
                disable()
                disabled.append(str(name))
            except Exception as exc:
                failed.append({'link': str(name), 'error': repr(exc)})
        return {'disabled': True, 'disabled_link_count': len(disabled), 'disabled_links': sorted(disabled), 'failed_links': failed}

    def _held_anchor(self, yaw: float) -> dict[str, Any] | None:
        link = getattr(self._robot, 'links', {}).get(self.config.eef_link_name) if self._robot is not None else None
        if link is None:
            return None
        try:
            position, _ = link.get_position_orientation(frame='scene')
            eef_position = _to_numpy(position).astype(float)[:3]
        except Exception:
            try:
                eef_position = _to_numpy(getattr(link, 'aabb_center')).astype(float)[:3]
            except Exception:
                return None
        offset = _rotate_xy(self.config.held_object_eef_offset, yaw)
        center = eef_position + np.asarray(offset, dtype=float)
        return {'center': center.astype(float).tolist(), 'yaw': float(yaw), 'pitch': 0.0, 'source': 'ego_arm_visual_eef', 'eef_link_name': self.config.eef_link_name, 'eef_position': eef_position.astype(float).tolist(), 'held_object_eef_offset': list(self.config.held_object_eef_offset)}

def ego_arm_visual_config_from_env(env: Mapping[str, str] | None=None) -> dict[str, Any] | None:
    values = os.environ if env is None else env
    raw_config = str(values.get('EB_BEHAVIOR_EGO_ARM_VISUAL_CONFIG', '') or '').strip()
    if raw_config:
        parsed = json.loads(raw_config)
        if not isinstance(parsed, Mapping):
            raise ValueError('EB_BEHAVIOR_EGO_ARM_VISUAL_CONFIG must be a JSON object.')
        config = {str(key): value for key, value in parsed.items()}
        config.setdefault('enabled', True)
        return config
    raw_enabled = str(values.get('EB_BEHAVIOR_EGO_ARM_VISUAL', '') or '').strip().lower()
    return {'enabled': raw_enabled not in {'0', 'false', 'no', 'off', 'none', 'disabled'}}

def _pose_field(pose: Any, key: str, default: Any=None) -> Any:
    if isinstance(pose, Mapping):
        return pose.get(key, default)
    return getattr(pose, key, default)

def _relative_position(ego_pose: Any, relative_position: Sequence[float]) -> list[float]:
    x = float(_pose_field(ego_pose, 'x', 0.0))
    y = float(_pose_field(ego_pose, 'y', 0.0))
    z = float(_pose_field(ego_pose, 'z', 1.55))
    yaw = float(_pose_field(ego_pose, 'yaw', 0.0))
    dx, dy, dz = _vector3(relative_position)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [x + c * dx - s * dy, y + s * dx + c * dy, z + dz]

def _yaw_to_quat_xyzw(yaw: float) -> list[float]:
    half = float(yaw) * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]

def _rotate_xy(offset: Sequence[float], yaw: float) -> tuple[float, float, float]:
    ox, oy, oz = _vector3(offset)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return (c * ox - s * oy, s * ox + c * oy, oz)

def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)

def _vector3(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError('Expected a 3D vector.')
    return (float(arr[0]), float(arr[1]), float(arr[2]))
__all__ = ['EgoArmVisualConfig', 'EgoArmVisualSynchronizer', 'ego_arm_visual_config_from_env']
