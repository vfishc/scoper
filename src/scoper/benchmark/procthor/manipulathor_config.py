from __future__ import annotations
from typing import Any, Mapping, Optional
_TRUTHY_STRINGS = {'1', 'true', 'yes', 'on'}
_FALSY_STRINGS = {'0', 'false', 'no', 'off'}
_VISUAL_ATTACH_PASSTHROUGH_KEYS = ('manipulathor_visual_attach_offset', 'manipulathor_visual_arm_hand_sphere_radius', 'manipulathor_visual_arm_base_y', 'manipulathor_visual_arm_wrist_position', 'manipulathor_visual_arm_move_speed', 'manipulathor_visual_arm_fixed_delta_time', 'manipulathor_visual_arm_restore_object_poses')
DEFAULT_MANIPULATHOR_VISUAL_ATTACH_AGENT_RADIUS = 0.3

def manipulathor_env_kwargs(config: Optional[Any]=None, *, manipulathor_arm: Optional[bool]=None, manipulathor_visual_attach: Optional[bool]=None, agent_radius: Optional[float]=None, closed_loop: bool=True) -> dict:
    visual_attach = _visual_attach_flag(config, explicit=manipulathor_visual_attach)
    arm_enabled = _coalesce_bool(manipulathor_arm, _config_bool(config, 'manipulathor_arm', None), default=False) or visual_attach
    kwargs = {}
    if arm_enabled:
        kwargs['manipulathor_arm'] = True
    resolved_agent_radius = _coalesce_float(agent_radius, _config_float(config, 'agent_radius', None))
    if resolved_agent_radius is None and visual_attach:
        resolved_agent_radius = DEFAULT_MANIPULATHOR_VISUAL_ATTACH_AGENT_RADIUS
    if resolved_agent_radius is not None:
        kwargs['agent_radius'] = resolved_agent_radius
    if closed_loop and visual_attach:
        kwargs['manipulathor_visual_attach_pickup'] = True
        for key in _VISUAL_ATTACH_PASSTHROUGH_KEYS:
            value = _config_get(config, key, None)
            if value is not None:
                kwargs[key] = value
    return kwargs

def manipulathor_visual_attach_enabled(config: Optional[Any]=None) -> bool:
    return _visual_attach_flag(config)

def _visual_attach_flag(config: Optional[Any], *, explicit: Optional[bool]=None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return bool(_config_bool(config, 'manipulathor_visual_attach', False) or _config_bool(config, 'manipulathor_visual_attach_pickup', False))

def _coalesce_bool(*values: Optional[bool], default: bool=False) -> bool:
    for value in values:
        if value is not None:
            return bool(value)
    return bool(default)

def _coalesce_float(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return float(value)
    return None

def _config_bool(config: Optional[Any], key: str, default: Optional[bool]) -> Optional[bool]:
    value = _config_get(config, key, None)
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUTHY_STRINGS:
            return True
        if normalized in _FALSY_STRINGS:
            return False
    return bool(value)

def _config_float(config: Optional[Any], key: str, default: Optional[float]) -> Optional[float]:
    value = _config_get(config, key, None)
    if value is None:
        return default
    return float(value)

def _config_get(config: Optional[Any], key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    get = getattr(config, 'get', None)
    if callable(get):
        try:
            return get(key, default)
        except TypeError:
            value = get(key)
            return default if value is None else value
    return getattr(config, key, default)
__all__ = ['DEFAULT_MANIPULATHOR_VISUAL_ATTACH_AGENT_RADIUS', 'manipulathor_env_kwargs', 'manipulathor_visual_attach_enabled']
