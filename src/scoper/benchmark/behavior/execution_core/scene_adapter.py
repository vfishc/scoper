from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject

@dataclass
class SceneAdapterReport:
    object_count: int
    converted_count: int
    skipped: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {'object_count': self.object_count, 'converted_count': self.converted_count, 'skipped': self.skipped}

def scene_objects_to_aabbs(objects: Iterable[Any], *, include_categories: set[str] | None=None, exclude_categories: set[str] | None=None) -> tuple[list[AABBObject], SceneAdapterReport]:
    converted: list[AABBObject] = []
    skipped: list[dict[str, Any]] = []
    object_count = 0
    for obj in objects:
        object_count += 1
        name = str(getattr(obj, 'name', f'object_{object_count:04d}'))
        category = str(getattr(obj, 'category', 'unknown'))
        if include_categories is not None and category not in include_categories:
            skipped.append({'name': name, 'category': category, 'reason': 'category_not_included'})
            continue
        if exclude_categories is not None and category in exclude_categories:
            skipped.append({'name': name, 'category': category, 'reason': 'category_excluded'})
            continue
        try:
            center = _triple(getattr(obj, 'aabb_center'))
            extent = _triple(getattr(obj, 'aabb_extent'))
        except Exception as exc:
            skipped.append({'name': name, 'category': category, 'reason': 'missing_aabb', 'error': repr(exc)})
            continue
        visible = bool(getattr(obj, 'visible', True))
        raw_abilities = getattr(obj, 'abilities', ())
        if isinstance(raw_abilities, dict):
            abilities = tuple((str(ability) for ability in raw_abilities))
        else:
            abilities = tuple((str(ability) for ability in raw_abilities or ()))
        converted.append(AABBObject(name=name, category=category, center=center, extent=extent, visible=visible, abilities=abilities))
    return (converted, SceneAdapterReport(object_count=object_count, converted_count=len(converted), skipped=skipped))

def scene_to_aabbs(scene: Any, **kwargs: Any) -> tuple[list[AABBObject], SceneAdapterReport]:
    if hasattr(scene, 'objects'):
        objects = scene.objects
    elif hasattr(scene, 'object_registry'):
        objects = scene.object_registry('name')
        if isinstance(objects, dict):
            objects = objects.values()
    else:
        raise ValueError('Scene does not expose objects or object_registry')
    return scene_objects_to_aabbs(objects, **kwargs)

def _triple(value: Any) -> tuple[float, float, float]:
    if callable(value):
        value = value()
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if len(value) != 3:
        raise ValueError(f'Expected length-3 value, got {value!r}')
    return (float(value[0]), float(value[1]), float(value[2]))
