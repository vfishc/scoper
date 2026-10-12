from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class GripperVisualConfig:
    enabled: bool = False
    proxy_object_names: tuple[str, ...] = ()
    distance: float = 0.38
    lateral_offset: float = 0.16
    vertical_offset: float = -0.12
    min_relative_z: float = -0.45
    max_relative_z: float = -0.05
    yaw_offset: float = 0.0
    proxy_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    held_object_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    verify_readback: bool = False

    @classmethod
    def from_value(cls, value: Any) -> 'GripperVisualConfig':
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError('gripper_visual_config must be a mapping or GripperVisualConfig.')
        proxy_names = value.get('proxy_object_names', value.get('proxy_names', ()))
        if isinstance(proxy_names, str):
            proxy_names = (proxy_names,)
        return cls(enabled=bool(value.get('enabled', True)), proxy_object_names=tuple((str(name) for name in proxy_names or () if str(name))), distance=float(value.get('distance', cls.distance)), lateral_offset=float(value.get('lateral_offset', cls.lateral_offset)), vertical_offset=float(value.get('vertical_offset', cls.vertical_offset)), min_relative_z=float(value.get('min_relative_z', cls.min_relative_z)), max_relative_z=float(value.get('max_relative_z', cls.max_relative_z)), yaw_offset=float(value.get('yaw_offset', cls.yaw_offset)), proxy_center_offset=_vector3(value.get('proxy_center_offset', cls.proxy_center_offset)), held_object_center_offset=_vector3(value.get('held_object_center_offset', cls.held_object_center_offset)), verify_readback=bool(value.get('verify_readback', cls.verify_readback)))

@dataclass(frozen=True)
class GripperAnchor:
    center: tuple[float, float, float]
    yaw: float
    pitch: float

    def as_dict(self) -> dict[str, Any]:
        return {'center': list(self.center), 'yaw': float(self.yaw), 'pitch': float(self.pitch)}

    def center_with_offset(self, offset: Sequence[float]) -> tuple[float, float, float]:
        dx, dy, dz = _rotate_offset(offset, self.yaw)
        return (self.center[0] + dx, self.center[1] + dy, self.center[2] + dz)

class GripperVisualSynchronizer:

    def __init__(self, config: GripperVisualConfig):
        self.config = config

    def sync_from_core(self, renderer: Any, core: Any, *, held_pose: Any=None) -> dict[str, Any]:
        ego_pose = getattr(getattr(core, 'ego_state', None), 'pose', None)
        holding = bool(getattr(getattr(core, 'ego_state', None), 'held_object_name', None))
        if ego_pose is None and held_pose is None:
            return {'enabled': True, 'success': False, 'reason': 'pose_unavailable'}
        anchor = gripper_anchor_from_pose(ego_pose, self.config) if ego_pose is not None else anchor_from_held_pose(held_pose)
        return self._sync(renderer, anchor, held_pose=held_pose, state='holding' if holding else 'idle')

    def sync_from_held_pose(self, renderer: Any, held_pose: Any, *, state: str='holding') -> dict[str, Any]:
        if held_pose is None:
            return {'enabled': True, 'success': False, 'reason': 'held_pose_absent'}
        return self._sync(renderer, anchor_from_held_pose(held_pose), held_pose=held_pose, state=state)

    def sync_from_ego_pose(self, renderer: Any, ego_pose: Any, *, held_pose: Any=None, state: str='idle') -> dict[str, Any]:
        if ego_pose is None:
            return {'enabled': True, 'success': False, 'reason': 'pose_unavailable'}
        return self._sync(renderer, gripper_anchor_from_pose(ego_pose, self.config), held_pose=held_pose, state=state)

    def sync_from_anchor(self, renderer: Any, anchor: GripperAnchor, *, held_pose: Any=None, state: str='holding') -> dict[str, Any]:
        return self._sync(renderer, anchor, held_pose=held_pose, state=state)

    def _sync(self, renderer: Any, anchor: GripperAnchor, *, held_pose: Any=None, state: str) -> dict[str, Any]:
        held_pose_for_renderer = held_pose_at_gripper(held_pose, anchor, self.config) if held_pose is not None else None
        proxy_mutations = self._sync_proxy_objects(renderer, anchor)
        configured_proxies = list(self.config.proxy_object_names)
        proxy_success = all((bool(mutation.get('success')) for mutation in proxy_mutations))
        success = proxy_success if configured_proxies else True
        reason = 'synced' if success else 'proxy_sync_failed'
        if not configured_proxies:
            reason = 'no_proxy_objects_configured'
        return {'enabled': True, 'success': bool(success), 'reason': reason, 'state': str(state), 'anchor': anchor.as_dict(), 'proxy_object_names': configured_proxies, 'proxy_mutations': proxy_mutations, 'held_pose_for_renderer': held_pose_for_renderer}

    def _sync_proxy_objects(self, renderer: Any, anchor: GripperAnchor) -> list[dict[str, Any]]:
        set_object_center = getattr(renderer, 'set_object_center', None)
        if not callable(set_object_center):
            return [{'success': False, 'reason': 'renderer_set_object_center_absent', 'object_name': name} for name in self.config.proxy_object_names]
        center = anchor.center_with_offset(self.config.proxy_center_offset)
        mutations: list[dict[str, Any]] = []
        for name in self.config.proxy_object_names:
            try:
                mutation = dict(set_object_center(str(name), center, yaw=anchor.yaw, verify_readback=bool(self.config.verify_readback), operation='pose'))
            except Exception as exc:
                mutation = {'success': False, 'reason': 'proxy_sync_error', 'object_name': str(name), 'error': repr(exc)}
            mutation['mutation'] = 'gripper_proxy_pose'
            mutation['gripper_anchor'] = anchor.as_dict()
            mutations.append(mutation)
        return mutations

def gripper_visual_config_from_env(env: Mapping[str, str] | None=None) -> dict[str, Any] | None:
    values = os.environ if env is None else env
    raw_config = str(values.get('EB_BEHAVIOR_GRIPPER_VISUAL_CONFIG', '') or '').strip()
    if raw_config:
        parsed = json.loads(raw_config)
        if not isinstance(parsed, Mapping):
            raise ValueError('EB_BEHAVIOR_GRIPPER_VISUAL_CONFIG must be a JSON object.')
        config = {str(key): value for key, value in parsed.items()}
        config.setdefault('enabled', True)
        return config
    raw_enabled = str(values.get('EB_BEHAVIOR_GRIPPER_VISUAL', '') or '').strip().lower()
    if raw_enabled in {'1', 'true', 'yes', 'on'}:
        return {'enabled': True}
    return None

def gripper_anchor_from_pose(ego_pose: Any, config: GripperVisualConfig) -> GripperAnchor:
    x = float(_pose_field(ego_pose, 'x', 0.0))
    y = float(_pose_field(ego_pose, 'y', 0.0))
    z = float(_pose_field(ego_pose, 'z', 1.55))
    yaw = float(_pose_field(ego_pose, 'yaw', 0.0)) + float(config.yaw_offset)
    pitch = float(_pose_field(ego_pose, 'pitch', 0.0))
    forward_xy = (math.cos(yaw), math.sin(yaw))
    right_xy = (-math.sin(yaw), math.cos(yaw))
    center_x = x + forward_xy[0] * config.distance + right_xy[0] * config.lateral_offset
    center_y = y + forward_xy[1] * config.distance + right_xy[1] * config.lateral_offset
    center_z = z + math.sin(pitch) * config.distance + config.vertical_offset
    center_z = max(z + config.min_relative_z, min(z + config.max_relative_z, center_z))
    return GripperAnchor(center=(center_x, center_y, center_z), yaw=yaw, pitch=pitch)

def anchor_from_held_pose(held_pose: Any) -> GripperAnchor:
    center = _vector3(_held_pose_field(held_pose, 'center', (0.0, 0.0, 0.0)))
    return GripperAnchor(center=center, yaw=float(_held_pose_field(held_pose, 'yaw', 0.0)), pitch=float(_held_pose_field(held_pose, 'pitch', 0.0)))

def held_pose_at_gripper(held_pose: Any, anchor: GripperAnchor, config: GripperVisualConfig) -> dict[str, Any]:
    payload = _held_pose_to_dict(held_pose)
    payload['center'] = list(anchor.center_with_offset(config.held_object_center_offset))
    payload['yaw'] = float(anchor.yaw)
    payload['pitch'] = float(anchor.pitch)
    payload['gripper_anchor'] = anchor.as_dict()
    return payload

def _held_pose_to_dict(held_pose: Any) -> dict[str, Any]:
    if isinstance(held_pose, Mapping):
        return {str(key): _jsonable_value(value) for key, value in held_pose.items()}
    keys = ('object_name', 'visual_name', 'center', 'yaw', 'pitch', 'distance', 'lateral_offset', 'vertical_offset')
    return {key: _jsonable_value(getattr(held_pose, key)) for key in keys if hasattr(held_pose, key)}

def _jsonable_value(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value

def _pose_field(pose: Any, key: str, default: Any=None) -> Any:
    if isinstance(pose, Mapping):
        return pose.get(key, default)
    return getattr(pose, key, default)

def _held_pose_field(held_pose: Any, key: str, default: Any=None) -> Any:
    if isinstance(held_pose, Mapping):
        return held_pose.get(key, default)
    return getattr(held_pose, key, default)

def _vector3(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError('Expected a 3D vector.')
    return (float(arr[0]), float(arr[1]), float(arr[2]))

def _rotate_offset(offset: Sequence[float], yaw: float) -> tuple[float, float, float]:
    ox, oy, oz = _vector3(offset)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return (c * ox - s * oy, s * ox + c * oy, oz)
__all__ = ['GripperAnchor', 'GripperVisualConfig', 'GripperVisualSynchronizer', 'anchor_from_held_pose', 'gripper_anchor_from_pose', 'gripper_visual_config_from_env', 'held_pose_at_gripper']
