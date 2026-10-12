from __future__ import annotations
from collections.abc import Callable, MutableMapping
from typing import Any
from scoper.benchmark.behavior.execution_core.trace import ActionResult
from scoper.benchmark.behavior.execution_core.visual_state_sync import OmniGibsonVisualStateSynchronizer, VisualStateSyncConfig

class FireController:

    def __init__(self, *, og_module: Any, env: Any, og_object_by_name: MutableMapping[str, Any], fire_visual_steps: int=18, fire_shutdown_steps: int=24, fire_shutdown_render_steps: int=6, deepen_fire_color: bool=True, fire_force_clear_on_shutdown: bool=True, before_fire_step: Callable[[], None] | None=None, on_object_added: Callable[[str, Any], None] | None=None) -> None:
        config = VisualStateSyncConfig(fire_visual_steps=fire_visual_steps, fire_shutdown_steps=fire_shutdown_steps, fire_shutdown_render_steps=fire_shutdown_render_steps, deepen_fire_color=deepen_fire_color, fire_force_clear_on_shutdown=fire_force_clear_on_shutdown, enable_particle_source_visuals=False, enable_particle_source_stream_visuals=False)
        self.synchronizer = OmniGibsonVisualStateSynchronizer(og_module=og_module, env=env, og_object_by_name=og_object_by_name, config=config, before_fire_step=before_fire_step, on_object_added=on_object_added)

    def on_toggle(self, result: ActionResult) -> list[dict[str, Any]]:
        return self.synchronizer.apply_action_result(result)

    def apply_action_result(self, result: ActionResult) -> list[dict[str, Any]]:
        return self.synchronizer.apply_action_result(result)

    @property
    def visual_mutations(self) -> list[dict[str, Any]]:
        return self.synchronizer.visual_mutations
