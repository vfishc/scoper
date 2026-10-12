from __future__ import annotations
from dataclasses import replace
from typing import Any, Mapping, Optional
from scoper.benchmark.behavior.scene.placement_symbolic import SceneInitResult
from scoper.benchmark.behavior.scene import placement_symbolic

def build_scene(task_spec: Mapping[str, Any], mode: str='symbolic', rng_seed: int=0, adapter: Optional[Any]=None, scene_layout_mode: str='task_roles', load_doors: bool=False):
    if mode == 'symbolic':
        result = placement_symbolic.build(task_spec, rng_seed=rng_seed)
        if scene_layout_mode == 'task_roles':
            return result
        if scene_layout_mode == 'source_json_full':
            from scoper.benchmark.behavior.scene.spawn_collision_repair import repair_task_spawn_collisions_after_merge
            from scoper.benchmark.behavior.scene.variant_layout import merge_source_json_layout
            merged = merge_source_json_layout(task_spec, result.aabb_objects, result.role_map, load_doors=load_doors)
            repaired_objects, repair_report = repair_task_spawn_collisions_after_merge(merged.objects, role_map=result.role_map, placements=(result.placement_report or {}).get('placements') or {})
            report = dict(result.placement_report or {})
            report['scene_layout_mode'] = scene_layout_mode
            report['source_json_full_layout'] = merged.report
            report['post_merge_collision_repairs'] = repair_report
            return replace(result, aabb_objects=repaired_objects, placement_report=report)
        raise ValueError("Unsupported scene_layout_mode {!r}; expected 'task_roles' or 'source_json_full'.".format(scene_layout_mode))
    if mode == 'render':
        from scoper.benchmark.behavior.scene import placement_render
        return placement_render.build(task_spec, adapter=adapter, rng_seed=rng_seed, scene_layout_mode=scene_layout_mode, load_doors=load_doors)
    raise ValueError("build_scene supports mode in {{'symbolic','render'}}, got {!r}".format(mode))
__all__ = ['build_scene', 'SceneInitResult']
