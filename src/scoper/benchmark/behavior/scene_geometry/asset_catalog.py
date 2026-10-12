from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from scoper.benchmark.behavior.scene_geometry.paths import assets_root as default_assets_root

@dataclass(frozen=True)
class ModelAsset:
    category: str
    model: str
    metadata_path: Path
    bbox_size: tuple[float, float, float]
    base_link_offset: tuple[float, float, float]
    metadata: dict[str, Any]

    @property
    def largest_xy(self) -> float:
        return max(float(self.bbox_size[0]), float(self.bbox_size[1]))

    @property
    def min_xy(self) -> float:
        return min(float(self.bbox_size[0]), float(self.bbox_size[1]))

    @property
    def height(self) -> float:
        return float(self.bbox_size[2])

    @property
    def volume(self) -> float:
        return float(self.bbox_size[0]) * float(self.bbox_size[1]) * float(self.bbox_size[2])

    @property
    def joint_count(self) -> int:
        joints = self.metadata.get('openable_joint_ids')
        openable_count = len(joints) if isinstance(joints, list) else 0
        link_count = _non_base_link_count(self.metadata.get('meta_links'))
        bbox_link_count = _non_base_link_count(self.metadata.get('link_bounding_boxes'))
        return max(openable_count, link_count, bbox_link_count)

class AssetCatalog:

    def __init__(self, models_by_category: dict[str, list[ModelAsset]], *, root: Path):
        self.models_by_category = {key: tuple(value) for key, value in models_by_category.items()}
        self.root = root

    @classmethod
    def build(cls, assets_root: str | Path | None=None) -> 'AssetCatalog':
        root = Path(assets_root) if assets_root is not None else default_assets_root()
        object_root = root / 'objects'
        models: dict[str, list[ModelAsset]] = {}
        for metadata_path in sorted(object_root.glob('*/*/misc/metadata.json')):
            category = metadata_path.parents[2].name
            model = metadata_path.parents[1].name
            try:
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            except Exception:
                continue
            bbox = _float_tuple(metadata.get('bbox_size'), 3)
            if bbox is None:
                continue
            offset = _float_tuple(metadata.get('base_link_offset'), 3) or (0.0, 0.0, 0.0)
            models.setdefault(category, []).append(ModelAsset(category=category, model=model, metadata_path=metadata_path, bbox_size=bbox, base_link_offset=offset, metadata=metadata))
        return cls(models, root=root)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted(self.models_by_category))

    @property
    def model_count(self) -> int:
        return sum((len(models) for models in self.models_by_category.values()))

    def has_category(self, category: str) -> bool:
        return str(category) in self.models_by_category

    def has_model(self, category: str, model: str) -> bool:
        return any((asset.model == str(model) for asset in self.models_by_category.get(str(category), ())))

    def get_models(self, category: str) -> tuple[ModelAsset, ...]:
        return self.models_by_category.get(str(category), ())

    def get_model(self, category: str, model: str) -> ModelAsset | None:
        for asset in self.get_models(category):
            if asset.model == str(model):
                return asset
        return None

    def iter_models(self, categories: Iterable[str] | None=None) -> Iterable[ModelAsset]:
        if categories is None:
            keys = self.categories
        else:
            keys = tuple((str(category) for category in categories))
        for category in keys:
            yield from self.get_models(category)

    def category_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for category, models in self.models_by_category.items():
            if not models:
                continue
            largest_xy = [asset.largest_xy for asset in models]
            heights = [asset.height for asset in models]
            volumes = [asset.volume for asset in models]
            stats[category] = {'model_count': len(models), 'avg_largest_xy': sum(largest_xy) / len(largest_xy), 'avg_height': sum(heights) / len(heights), 'avg_volume': sum(volumes) / len(volumes), 'min_largest_xy': min(largest_xy), 'max_largest_xy': max(largest_xy)}
        return stats

    def summary(self) -> dict[str, Any]:
        return {'assets_root': str(self.root), 'category_count': len(self.models_by_category), 'model_count': self.model_count, 'categories': self.category_stats()}

def scaled_bbox(asset: ModelAsset, scale: tuple[float, float, float] | list[float] | None=None) -> tuple[float, float, float]:
    if scale is None:
        scale = (1.0, 1.0, 1.0)
    return (float(asset.bbox_size[0]) * float(scale[0]), float(asset.bbox_size[1]) * float(scale[1]), float(asset.bbox_size[2]) * float(scale[2]))

def scaled_base_link_offset(asset: ModelAsset, scale: tuple[float, float, float] | list[float] | None=None) -> tuple[float, float, float]:
    if scale is None:
        scale = (1.0, 1.0, 1.0)
    return (float(asset.base_link_offset[0]) * float(scale[0]), float(asset.base_link_offset[1]) * float(scale[1]), float(asset.base_link_offset[2]) * float(scale[2]))

def root_z_for_floor(asset: ModelAsset, scale: tuple[float, float, float] | list[float] | None=None) -> float:
    return root_z_for_bbox_bottom(asset, 0.0, scale)

def root_z_for_bbox_bottom(asset: ModelAsset, bottom_z: float, scale: tuple[float, float, float] | list[float] | None=None) -> float:
    bbox = scaled_bbox(asset, scale)
    offset = scaled_base_link_offset(asset, scale)
    return float(bottom_z) + bbox[2] / 2.0 - offset[2]

def bbox_bottom_z(root_z: float, asset: ModelAsset, scale: tuple[float, float, float] | list[float] | None=None) -> float:
    bbox = scaled_bbox(asset, scale)
    offset = scaled_base_link_offset(asset, scale)
    return float(root_z) + offset[2] - bbox[2] / 2.0

def bbox_top_z(root_z: float, asset: ModelAsset, scale: tuple[float, float, float] | list[float] | None=None) -> float:
    bbox = scaled_bbox(asset, scale)
    offset = scaled_base_link_offset(asset, scale)
    return float(root_z) + offset[2] + bbox[2] / 2.0

def _float_tuple(value: Any, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        return tuple((float(v) for v in value))
    except (TypeError, ValueError):
        return None

def _non_base_link_count(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return sum((1 for key in value if str(key) != 'base_link'))
__all__ = ['AssetCatalog', 'ModelAsset', 'bbox_bottom_z', 'bbox_top_z', 'root_z_for_bbox_bottom', 'root_z_for_floor', 'scaled_base_link_offset', 'scaled_bbox']
