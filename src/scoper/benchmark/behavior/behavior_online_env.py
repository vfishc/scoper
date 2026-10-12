from __future__ import annotations
import os
from pathlib import Path
from typing import Any, List, Mapping
from scoper.benchmark.behavior.execution_core.env import EgoEnvConfig, build_robot_free_env_config, make_omnigibson_environment, validate_robot_free_env_config
from scoper.benchmark.behavior.scene.doors import door_not_load_categories

def make_env_config(task_spec: Mapping[str, Any], *, scene_layout_mode: str='task_roles', load_doors: bool=False) -> dict[str, Any]:
    scene_model = task_spec['source']['scene_model']
    scene_file = None
    if scene_layout_mode == 'source_json_full':
        source_json = (task_spec.get('source') or {}).get('source_json')
        if source_json:
            from scoper.resources import resolve_resource
            path = resolve_resource(str(source_json))
            scene_file = str(path)
    cfg = EgoEnvConfig(scene_model=scene_model, scene_file=scene_file, not_load_object_categories=None if load_doors else door_not_load_categories())
    return build_robot_free_env_config(cfg)

def validate_env_config(env_config: dict[str, Any]) -> List[str]:
    return validate_robot_free_env_config(env_config)

class BehaviorOnlineEnv:

    def __init__(self, task_spec: Mapping[str, Any]):
        self.task_spec = task_spec
        self.env_config = make_env_config(task_spec)
        reasons = validate_env_config(self.env_config)
        if reasons:
            raise ValueError('Invalid env config: {}'.format(reasons))
        self._og = None

    def reset(self) -> dict[str, Any]:
        if os.environ.get('EB_BEHAVIOR_GPU') != '1':
            raise RuntimeError('route-A reset requires GPU; set EB_BEHAVIOR_GPU=1.')
        self._og = make_omnigibson_environment(self.env_config)
        obs, _ = self._og.reset()
        rgb = obs['external']['ego_camera']['rgb']
        return {'head_rgb': rgb[..., :3]}
__all__ = ['make_env_config', 'validate_env_config', 'BehaviorOnlineEnv']
