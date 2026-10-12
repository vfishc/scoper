from __future__ import annotations
from collections.abc import Callable, MutableMapping
from enum import Enum
from typing import Any
from scoper.benchmark.behavior.execution_core.trace import ActionResult
from scoper.benchmark.behavior.execution_core.visual_state_sync import REALISTIC_WATER_COLOR, OmniGibsonVisualStateSynchronizer, VisualStateSyncConfig

class WaterFlowMode(str, Enum):
    PARTICLES = 'particles'
    CONTINUOUS_MESH = 'continuous_mesh'
    CYLINDER = 'cylinder'
_STREAM_COLOR: tuple[float, float, float] = (0.18, 0.78, 1.0)

def make_water_flow_config(mode: WaterFlowMode=WaterFlowMode.CONTINUOUS_MESH, *, water_visual_steps: int=36, water_shutdown_steps: int=6, water_shutdown_render_steps: int=50, stream_length: float=0.42, stream_radius: float=0.025, stream_color: tuple[float, float, float]=_STREAM_COLOR, stream_opacity: float=0.85, stream_segments: int=28, stream_radial_segments: int=16, stream_wobble: float=0.004, stream_taper: float=0.28, stream_droplets: int=5, particle_color: tuple[float, float, float]=REALISTIC_WATER_COLOR) -> VisualStateSyncConfig:
    enable_particles = True
    enable_stream = mode in (WaterFlowMode.CONTINUOUS_MESH, WaterFlowMode.CYLINDER)
    style = 'continuous_mesh' if mode == WaterFlowMode.CONTINUOUS_MESH else 'cylinder'
    return VisualStateSyncConfig(water_visual_steps=water_visual_steps, water_shutdown_steps=water_shutdown_steps, water_shutdown_render_steps=water_shutdown_render_steps, enable_particle_source_visuals=enable_particles, enable_particle_source_stream_visuals=enable_stream, particle_source_stream_style=style, particle_source_stream_length=stream_length, particle_source_stream_radius=stream_radius, particle_source_stream_color=stream_color, particle_source_stream_opacity=stream_opacity, particle_source_stream_segments=stream_segments, particle_source_stream_radial_segments=stream_radial_segments, particle_source_stream_wobble=stream_wobble, particle_source_stream_taper=stream_taper, particle_source_stream_droplets=stream_droplets, liquid_particle_color=particle_color)

class WaterFlowController:

    def __init__(self, *, og_module: Any, env: Any, og_object_by_name: MutableMapping[str, Any], mode: WaterFlowMode=WaterFlowMode.CONTINUOUS_MESH, config: VisualStateSyncConfig | None=None, before_fire_step: Callable[[], None] | None=None, on_object_added: Callable[[str, Any], None] | None=None) -> None:
        self.mode = mode
        self.config = config or make_water_flow_config(mode)
        self.synchronizer = OmniGibsonVisualStateSynchronizer(og_module=og_module, env=env, og_object_by_name=og_object_by_name, config=self.config, before_fire_step=before_fire_step, on_object_added=on_object_added)

    def on_toggle(self, result: ActionResult) -> list[dict[str, Any]]:
        return self.synchronizer.apply_action_result(result)

    def apply_action_result(self, result: ActionResult) -> list[dict[str, Any]]:
        return self.synchronizer.apply_action_result(result)

    @property
    def visual_mutations(self) -> list[dict[str, Any]]:
        return self.synchronizer.visual_mutations
