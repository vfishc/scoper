from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from scoper.benchmark.behavior.scene_geometry.asset_catalog import AssetCatalog
from scoper.benchmark.behavior.scene_geometry.rules import AugmentationRules
from scoper.benchmark.behavior.scene_geometry.taxonomy_index import TaxonomyIndex
ROLE_NAMES = ('structure', 'fixture', 'large_floor', 'support_surface', 'small_prop', 'wall_object', 'light', 'outdoor', 'unsafe')

@dataclass(frozen=True)
class CategoryClassification:
    category: str
    role: str
    reason: str
    model_count: int
    avg_largest_xy: float | None
    avg_height: float | None
    synset: str | None
    object_type: str | None
    abilities: dict[str, bool]

    @property
    def spawnable_large(self) -> bool:
        return self.role in {'large_floor', 'support_surface'}

    @property
    def spawnable_small(self) -> bool:
        return self.role == 'small_prop'

class CategoryClassifier:

    def __init__(self, catalog: AssetCatalog, taxonomy: TaxonomyIndex, rules: AugmentationRules):
        self.catalog = catalog
        self.taxonomy = taxonomy
        self.rules = rules
        self._classifications: dict[str, CategoryClassification] | None = None

    def classify_all(self) -> dict[str, CategoryClassification]:
        if self._classifications is not None:
            return self._classifications
        stats = self.catalog.category_stats()
        out: dict[str, CategoryClassification] = {}
        for category in self.catalog.categories:
            stat = stats.get(category, {})
            tax = self.taxonomy.get(category)
            role, reason = self._classify_category(category, stat, tax.abilities if tax else {}, tax.object_type if tax else None)
            out[category] = CategoryClassification(category=category, role=role, reason=reason, model_count=int(stat.get('model_count') or 0), avg_largest_xy=_maybe_float(stat.get('avg_largest_xy')), avg_height=_maybe_float(stat.get('avg_height')), synset=tax.synset if tax else None, object_type=tax.object_type if tax else None, abilities=dict(tax.abilities) if tax else {})
        self._classifications = out
        return out

    def get(self, category: str) -> CategoryClassification | None:
        return self.classify_all().get(str(category))

    def categories_for_role(self, role: str) -> list[str]:
        return sorted((category for category, item in self.classify_all().items() if item.role == role))

    def summary(self) -> dict[str, Any]:
        rows = self.classify_all()
        by_role: dict[str, int] = {}
        for item in rows.values():
            by_role[item.role] = by_role.get(item.role, 0) + 1
        return {'category_count': len(rows), 'roles': dict(sorted(by_role.items())), 'categories': {category: {'role': item.role, 'reason': item.reason, 'model_count': item.model_count, 'avg_largest_xy': item.avg_largest_xy, 'avg_height': item.avg_height, 'synset': item.synset, 'object_type': item.object_type, 'abilities': item.abilities} for category, item in sorted(rows.items())}}

    def _classify_category(self, category: str, stat: dict[str, Any], abilities: dict[str, bool], object_type: str | None) -> tuple[str, str]:
        overrides = self.rules.manual_overrides
        override_order = ('structure', 'fixture', 'support_surface', 'large_floor', 'small_prop', 'wall_object', 'light', 'outdoor', 'unsafe')
        for role in override_order:
            if category in overrides.get(role, set()):
                return (role, f'manual_{role}')
        lowered = category.lower()
        if object_type and object_type not in {'rigidBody', 'cloth'}:
            return ('unsafe', f'object_type_{object_type}')
        if any((token in lowered for token in ('wall', 'floor', 'ceiling', 'roof'))):
            return ('structure', 'name_structure')
        if any((token in lowered for token in ('door', 'window', 'sink', 'toilet', 'stove', 'oven', 'shower'))):
            return ('fixture', 'name_fixture')
        if any((token in lowered for token in ('light', 'lamp', 'chandelier', 'lantern'))) and 'floor_lamp' not in lowered and ('table_lamp' not in lowered):
            return ('light', 'name_light')
        if any((token in lowered for token in ('picture', 'painting', 'mirror', 'clock', 'dispenser', 'rack', 'socket', 'alarm', 'map', 'poster', 'banner', 'sign'))):
            return ('wall_object', 'name_wall_object')
        if any((token in lowered for token in ('lawn', 'tree', 'bush', 'fence', 'pool', 'driveway', 'garden'))):
            return ('outdoor', 'name_outdoor')
        if abilities.get('particleSource') or abilities.get('particleSink'):
            return ('fixture', 'particle_fixture')
        avg_xy = float(stat.get('avg_largest_xy') or 0.0)
        avg_h = float(stat.get('avg_height') or 0.0)
        avg_volume = float(stat.get('avg_volume') or 0.0)
        if any((token in lowered for token in ('table', 'desk', 'countertop', 'shelf', 'bookcase', 'cabinet', 'stand'))):
            return ('support_surface', 'name_support_surface')
        if any((token in lowered for token in ('bed', 'sofa', 'chair', 'bench', 'ottoman', 'locker', 'wardrobe', 'fridge', 'plant', 'trash_can', 'carpet'))):
            return ('large_floor', 'name_large_floor')
        if avg_xy <= 0.55 and avg_h <= 0.65 and (avg_volume <= 0.08):
            return ('small_prop', 'size_small')
        if avg_xy <= 0.9 and avg_h <= 1.4:
            return ('large_floor', 'size_floor_candidate')
        return ('unsafe', 'unclassified_large_or_unstable')

def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
__all__ = ['CategoryClassification', 'CategoryClassifier', 'ROLE_NAMES']
