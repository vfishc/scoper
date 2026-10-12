from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

@dataclass
class EgoCameraConfig:
    name: str = 'ego_camera'
    height: float = 1.55
    image_width: int = 512
    image_height: int = 512
    modalities: tuple[str, ...] = ('rgb', 'depth_linear', 'seg_semantic', 'seg_instance')
    horizontal_aperture: float = 55.0
    clipping_range: tuple[float, float] = (0.01, 100.0)

@dataclass
class EgoEnvConfig:
    scene_model: str = 'Rs_int'
    scene_file: str | None = None
    scene_source: str = 'OG'
    load_object_categories: list[str] | None = None
    not_load_object_categories: list[str] | None = None
    objects: list[dict[str, Any]] = field(default_factory=list)
    trav_map_with_objects: bool = True
    trav_map_resolution: float = 0.1
    default_erosion_radius: float = 0.0
    action_frequency: int = 30
    physics_frequency: int = 120
    rendering_frequency: int = 30
    viewer_width: int = 512
    viewer_height: int = 512
    camera: EgoCameraConfig = field(default_factory=EgoCameraConfig)

def build_robot_free_env_config(config: EgoEnvConfig | None=None) -> dict[str, Any]:
    config = config or EgoEnvConfig()
    camera = config.camera
    env_config: dict[str, Any] = {'env': {'action_frequency': config.action_frequency, 'physics_frequency': config.physics_frequency, 'rendering_frequency': config.rendering_frequency, 'automatic_reset': False, 'external_sensors': [{'sensor_type': 'VisionSensor', 'name': camera.name, 'relative_prim_path': f'/{camera.name}', 'modalities': list(camera.modalities), 'position': [0.0, 0.0, camera.height], 'orientation': [0.0, 0.0, 0.0, 1.0], 'pose_frame': 'scene', 'include_in_obs': True, 'sensor_kwargs': {'image_width': camera.image_width, 'image_height': camera.image_height, 'horizontal_aperture': camera.horizontal_aperture, 'clipping_range': list(camera.clipping_range)}}]}, 'render': {'viewer_width': config.viewer_width, 'viewer_height': config.viewer_height}, 'scene': {'type': 'InteractiveTraversableScene', 'scene_model': config.scene_model, 'scene_file': config.scene_file, 'scene_source': config.scene_source, 'include_robots': False, 'trav_map_with_objects': config.trav_map_with_objects, 'trav_map_resolution': config.trav_map_resolution, 'default_erosion_radius': config.default_erosion_radius}, 'robots': [], 'objects': deepcopy(config.objects), 'task': {'type': 'DummyTask'}}
    if config.load_object_categories is not None:
        env_config['scene']['load_object_categories'] = list(config.load_object_categories)
    if config.not_load_object_categories is not None:
        env_config['scene']['not_load_object_categories'] = list(config.not_load_object_categories)
    return env_config

def validate_robot_free_env_config(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if config.get('robots') != []:
        reasons.append('robots_not_empty')
    if config.get('scene', {}).get('include_robots') is not False:
        reasons.append('scene_include_robots_not_false')
    external_sensors = config.get('env', {}).get('external_sensors')
    if not external_sensors:
        reasons.append('missing_external_sensors')
    else:
        ego_sensors = [sensor for sensor in external_sensors if sensor.get('name') == 'ego_camera']
        if not ego_sensors:
            reasons.append('missing_ego_camera')
        elif ego_sensors[0].get('sensor_type') != 'VisionSensor':
            reasons.append('ego_camera_not_vision_sensor')
    if config.get('task', {}).get('type') != 'DummyTask':
        reasons.append('task_not_dummy')
    return reasons

def make_omnigibson_environment(config: dict[str, Any]):
    import omnigibson as og
    return og.Environment(configs=_config_with_og_state_compat_scene_file(config))

def _config_with_og_state_compat_scene_file(config: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    scene = out.get('scene')
    if not isinstance(scene, dict):
        return out
    scene_file = scene.get('scene_file')
    if not scene_file:
        return out
    normalized = _og_state_compat_scene_file(Path(str(scene_file)))
    if normalized is not None:
        scene['scene_file'] = str(normalized)
    return out

def _og_state_compat_scene_file(scene_file: Path) -> Path | None:
    if not scene_file.exists():
        return None
    raw = scene_file.read_text(encoding='utf-8')
    try:
        scene = json.loads(raw)
    except json.JSONDecodeError:
        return None
    changed = _ensure_og_state_velocity_fields(scene)
    if not changed:
        return scene_file
    digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / 'eb_behavior_og_scene_state_compat'
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = cache_dir / '{}_{}.json'.format(scene_file.stem, digest)
    if not normalized.exists():
        normalized.write_text(json.dumps(scene, indent=2, sort_keys=False), encoding='utf-8')
    return normalized

def _ensure_og_state_velocity_fields(scene: dict[str, Any]) -> bool:
    registry = scene.get('state', {}).get('registry', {}).get('object_registry', {})
    if not isinstance(registry, dict):
        return False
    changed = False
    for state in registry.values():
        if not isinstance(state, dict):
            continue
        root = state.get('root_link')
        if not isinstance(root, dict):
            continue
        if 'lin_vel' not in root:
            root['lin_vel'] = [0.0, 0.0, 0.0]
            changed = True
        if 'ang_vel' not in root:
            root['ang_vel'] = [0.0, 0.0, 0.0]
            changed = True
    return changed
