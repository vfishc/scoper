"""Occupancy debug artifacts for generated BEHAVIOR scene variants."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scoper.benchmark.behavior.scene.room_layout_maps import load_behavior_room_layout_maps

from scoper.benchmark.behavior.scene_geometry.asset_catalog import AssetCatalog, scaled_bbox
from scoper.benchmark.behavior.scene_geometry.category_classifier import CategoryClassifier
from scoper.benchmark.behavior.scene_geometry.geometry import footprint_extent_for_yaw
from scoper.benchmark.behavior.scene_geometry.scene_io import load_scene_json, parse_scene_objects
from scoper.benchmark.behavior.scene_geometry.scene_io import quat_to_yaw


def write_scene_occupancy_artifacts(
    *,
    scene_model: str,
    scene_file: str | Path,
    output_dir: str | Path,
    prefix: str,
    catalog: AssetCatalog,
    classifier: CategoryClassifier | None = None,
    assets_root: str | Path | None = None,
    resolution: float = 0.10,
    agent_radius: float = 0.20,
    safety_margin: float = 0.05,
    rooms: list[str] | tuple[str, ...] | set[str] | None = None,
    crop_to_rooms: bool = False,
) -> dict[str, Any]:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from scoper.benchmark.behavior.execution_core.occupancy import AABBObject, OccupancyConfig, OccupancyMap

    scene = load_scene_json(scene_file)
    records = parse_scene_objects(scene)
    selected_rooms = {str(room_id) for room_id in rooms or [] if str(room_id)}
    objects: list[AABBObject] = []
    skipped: list[dict[str, str]] = []
    ignored_categories = _ignored_occupancy_categories(classifier)
    for record in records.values():
        if selected_rooms and not (set(record.rooms) & selected_rooms):
            continue
        asset = catalog.get_model(record.category, record.model)
        if asset is None:
            skipped.append({"name": record.name, "reason": "missing_asset", "category": record.category, "model": record.model})
            continue
        bbox = scaled_bbox(asset, record.scale)
        extent_xy = footprint_extent_for_yaw((bbox[0], bbox[1]), quat_to_yaw(record.ori))
        objects.append(
            AABBObject(
                name=record.name,
                category=record.category,
                center=(float(record.pos[0]), float(record.pos[1]), float(record.pos[2])),
                extent=(float(extent_xy[0]), float(extent_xy[1]), float(bbox[2])),
                visible=True,
                states=_navigation_states_for_category(record.category),
            )
        )

    layout = load_behavior_room_layout_maps(str(scene_model), assets_root=assets_root, floor=0)
    indoor_mask = np.asarray(layout.room_ins_map) > 0
    traversability_map = np.where(indoor_mask, layout.trav_map, 0).astype(np.asarray(layout.trav_map).dtype, copy=False)
    bounds = _bounds_from_rooms(layout, selected_rooms) if crop_to_rooms and selected_rooms else _bounds_from_layout(layout)
    occupancy = OccupancyMap.from_aabb_objects(
        objects,
        OccupancyConfig(
            resolution=float(resolution),
            agent_radius=float(agent_radius),
            safety_margin=float(safety_margin),
            bounds=bounds,
            ignored_categories=ignored_categories,
            traversability_map=traversability_map,
            traversability_map_resolution=float(layout.map_resolution),
            traversability_map_size=int(layout.map_size),
            traversability_floor=int(layout.floor),
            traversability_source="scoper.benchmark.behavior.room_layout_maps.floor_trav&floor_insseg",
        ),
    )
    artifacts = occupancy.write_debug_artifacts(output_dir, prefix=prefix)
    metadata_path = Path(artifacts["metadata_json"])
    metadata = dict(occupancy.metadata)
    metadata.update(
        {
            "scene_model": str(scene_model),
            "scene_file": str(scene_file),
            "object_count": len(objects),
            "skipped_objects": skipped,
            "room_layout_source_paths": dict(layout.source_paths),
            "indoor_mask": {
                "enabled": True,
                "source": dict(layout.source_paths).get("floor_insseg"),
                "free_pixels_before_indoor_mask": int(np.count_nonzero(np.asarray(layout.trav_map) > 0)),
                "free_pixels_after_indoor_mask": int(np.count_nonzero(traversability_map > 0)),
            },
            "rooms": sorted(selected_rooms),
            "crop_to_rooms": bool(crop_to_rooms),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "status": "passed",
        "artifacts": artifacts,
        "object_count": len(objects),
        "skipped_count": len(skipped),
        "grid_shape": [int(occupancy.height), int(occupancy.width)],
        "component_count": int(occupancy.metadata.get("component_count", 0)),
        "free_cells": int(np.count_nonzero(occupancy.eroded_free)),
        "occupied_cells": int(np.count_nonzero(occupancy.occupied)),
    }


def _bounds_from_layout(layout: Any) -> tuple[float, float, float, float]:
    half_extent = float(layout.map_size) * float(layout.map_resolution) / 2.0
    return (-half_extent, -half_extent, half_extent, half_extent)


def _bounds_from_rooms(layout: Any, rooms: set[str], *, margin_m: float = 0.8) -> tuple[float, float, float, float]:
    masks = []
    for room_id in rooms:
        room_mask = layout.room_mask(room_id)
        if room_mask is not None:
            masks.append(np.asarray(room_mask, dtype=bool))
    if not masks:
        return _bounds_from_layout(layout)
    mask = np.zeros_like(np.asarray(layout.room_ins_map), dtype=bool)
    for room_mask in masks:
        mask |= room_mask
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return _bounds_from_layout(layout)
    min_xy = layout.map_to_world((int(rows.min()), int(cols.min())))
    max_xy = layout.map_to_world((int(rows.max()), int(cols.max())))
    x0, x1 = sorted((float(min_xy[0]), float(max_xy[0])))
    y0, y1 = sorted((float(min_xy[1]), float(max_xy[1])))
    return (x0 - float(margin_m), y0 - float(margin_m), x1 + float(margin_m), y1 + float(margin_m))


def _ignored_occupancy_categories(classifier: CategoryClassifier | None) -> set[str]:
    ignored = {
        "ceilings",
        "ceiling",
        "floors",
        "floor",
        "carpet",
        "rug",
        "mat",
    }
    if classifier is None:
        return ignored
    for category in getattr(classifier, "categories_for_role", lambda role: [])("small_prop"):
        ignored.add(str(category))
    return ignored


def _navigation_states_for_category(category: str) -> dict[str, Any]:
    text = str(category).lower()
    if "door" in text:
        return {
            "navigation_barrier": "door",
            "blocks_navigation_when_closed": True,
            "open": False,
        }
    return {}


__all__ = ["write_scene_occupancy_artifacts"]
