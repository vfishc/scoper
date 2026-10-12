"""3D AABB helpers for tabletop and surface object occupancy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from scoper.benchmark.behavior.scene_geometry.asset_catalog import AssetCatalog, ModelAsset, scaled_base_link_offset, scaled_bbox
from scoper.benchmark.behavior.scene_geometry.category_classifier import CategoryClassifier
from scoper.benchmark.behavior.scene_geometry.geometry import footprint_extent_for_yaw
from scoper.benchmark.behavior.scene_geometry.scene_io import SceneObjectRecord, parse_scene_objects, quat_to_yaw


@dataclass(frozen=True)
class ObjectAABB:
    name: str
    category: str
    room: str | None
    center_xyz: tuple[float, float, float]
    extent_xyz: tuple[float, float, float]
    yaw: float = 0.0
    source: str = "existing"
    owner: str | None = None
    parent: str | None = None
    task_relevant: bool = False

    @property
    def volume(self) -> float:
        return max(float(self.extent_xyz[0]) * float(self.extent_xyz[1]) * float(self.extent_xyz[2]), 0.0)

    def aabb(self, *, clearance: float = 0.0) -> tuple[float, float, float, float, float, float]:
        clearance = max(float(clearance), 0.0)
        ex = float(self.extent_xyz[0]) * 0.5 + clearance
        ey = float(self.extent_xyz[1]) * 0.5 + clearance
        ez = float(self.extent_xyz[2]) * 0.5 + clearance
        cx, cy, cz = self.center_xyz
        return (
            float(cx) - ex,
            float(cy) - ey,
            float(cz) - ez,
            float(cx) + ex,
            float(cy) + ey,
            float(cz) + ez,
        )


def object_aabb_from_record(
    record: SceneObjectRecord,
    asset: ModelAsset,
    *,
    source: str = "existing",
    owner: str | None = None,
    parent: str | None = None,
    task_relevant: bool = False,
) -> ObjectAABB:
    return object_aabb_from_pose(
        name=record.name,
        category=record.category,
        room=record.primary_room,
        pos=record.pos,
        yaw=quat_to_yaw(record.ori),
        asset=asset,
        scale=record.scale,
        source=source,
        owner=owner or record.name,
        parent=parent,
        task_relevant=task_relevant,
    )


def object_aabb_from_pose(
    *,
    name: str,
    category: str,
    room: str | None,
    pos: tuple[float, float, float],
    yaw: float,
    asset: ModelAsset,
    scale: tuple[float, float, float] | list[float] | None = None,
    source: str = "candidate",
    owner: str | None = None,
    parent: str | None = None,
    task_relevant: bool = False,
) -> ObjectAABB:
    bbox = scaled_bbox(asset, scale)
    offset = scaled_base_link_offset(asset, scale)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    center_x = float(pos[0]) + float(offset[0]) * c - float(offset[1]) * s
    center_y = float(pos[1]) + float(offset[0]) * s + float(offset[1]) * c
    center_z = float(pos[2]) + float(offset[2])
    extent_xy = footprint_extent_for_yaw((bbox[0], bbox[1]), yaw)
    return ObjectAABB(
        name=str(name),
        category=str(category),
        room=str(room) if room is not None else None,
        center_xyz=(center_x, center_y, center_z),
        extent_xyz=(float(extent_xy[0]), float(extent_xy[1]), float(bbox[2])),
        yaw=float(yaw),
        source=str(source),
        owner=str(owner) if owner is not None else str(name),
        parent=str(parent) if parent is not None else None,
        task_relevant=bool(task_relevant),
    )


def small_object_aabbs_from_scene(
    scene: dict,
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    *,
    include_sources: Iterable[str] | None = None,
) -> list[ObjectAABB]:
    allowed_sources = {str(item) for item in include_sources} if include_sources is not None else None
    out: list[ObjectAABB] = []
    for record in parse_scene_objects(scene).values():
        role = getattr(classifier.get(record.category), "role", None)
        if role != "small_prop":
            continue
        asset = catalog.get_model(record.category, record.model)
        if asset is None:
            continue
        source = "existing"
        if allowed_sources is not None and source not in allowed_sources:
            continue
        out.append(object_aabb_from_record(record, asset, source=source))
    return out


def aabb3d_overlap_ratio(a: ObjectAABB, b: ObjectAABB, *, clearance: float = 0.0) -> float:
    ax1, ay1, az1, ax2, ay2, az2 = a.aabb(clearance=clearance)
    bx1, by1, bz1, bx2, by2, bz2 = b.aabb(clearance=clearance)
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    iz = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter = ix * iy * iz
    if inter <= 0.0:
        return 0.0
    denom = max(min(a.volume, b.volume), 1e-8)
    return inter / denom


def aabb3d_overlaps(
    a: ObjectAABB,
    b: ObjectAABB,
    *,
    clearance: float = 0.0,
    max_overlap_ratio: float = 0.0,
) -> bool:
    return aabb3d_overlap_ratio(a, b, clearance=clearance) > float(max_overlap_ratio)


def xy_center_distance(a: ObjectAABB, b: ObjectAABB) -> float:
    return math.hypot(float(a.center_xyz[0]) - float(b.center_xyz[0]), float(a.center_xyz[1]) - float(b.center_xyz[1]))


__all__ = [
    "ObjectAABB",
    "aabb3d_overlap_ratio",
    "aabb3d_overlaps",
    "object_aabb_from_pose",
    "object_aabb_from_record",
    "small_object_aabbs_from_scene",
    "xy_center_distance",
]
