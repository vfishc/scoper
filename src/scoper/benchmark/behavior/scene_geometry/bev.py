"""BEV visualization and labels for BEHAVIOR scene variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scoper.benchmark.behavior.scene.room_layout_maps import load_behavior_room_layout_maps

from scoper.benchmark.behavior.scene_geometry.asset_catalog import AssetCatalog, scaled_bbox
from scoper.benchmark.behavior.scene_geometry.category_classifier import CategoryClassifier
from scoper.benchmark.behavior.scene_geometry.geometry import footprint_extent_for_yaw, footprint_rows_cols
from scoper.benchmark.behavior.scene_geometry.paths import assets_root as default_assets_root
from scoper.benchmark.behavior.scene_geometry.scene_io import load_scene_json, parse_scene_objects, quat_to_yaw, room_type_from_instance


ROLE_COLORS = {
    "structure": (85, 85, 85),
    "fixture": (130, 120, 110),
    "large_floor": (40, 118, 190),
    "support_surface": (45, 155, 105),
    "small_prop": (222, 143, 35),
    "wall_object": (155, 75, 175),
    "light": (245, 210, 70),
    "outdoor": (90, 150, 80),
    "unsafe": (170, 60, 60),
}


def render_bev(
    *,
    scene_model: str,
    scene_file: str | Path,
    output_png: str | Path,
    labels_json: str | Path | None = None,
    catalog: AssetCatalog | None = None,
    classifier: CategoryClassifier | None = None,
    assets_root: str | Path | None = None,
    max_labels: int = 180,
    rooms: list[str] | tuple[str, ...] | set[str] | None = None,
    crop_to_rooms: bool = False,
    hide_roles: list[str] | tuple[str, ...] | set[str] | None = None,
    hide_categories: list[str] | tuple[str, ...] | set[str] | None = None,
    modified_only: bool = False,
) -> dict[str, Any]:
    assets = Path(assets_root) if assets_root is not None else default_assets_root()
    scene = load_scene_json(scene_file)
    layout = load_behavior_room_layout_maps(str(scene_model), assets_root=assets, floor=0)
    catalog = catalog or AssetCatalog.build(assets)
    classifier = classifier

    selected_rooms = {str(room_id) for room_id in rooms or [] if str(room_id)}
    hidden_roles = {str(role) for role in hide_roles or [] if str(role)}
    hidden_categories = {str(category) for category in hide_categories or [] if str(category)}
    room = np.asarray(layout.room_ins_map)
    trav = np.asarray(layout.trav_map) > 0
    h, w = room.shape
    view_slice = _room_view_slice(layout, selected_rooms) if crop_to_rooms and selected_rooms else (slice(0, h), slice(0, w))
    row_offset = int(view_slice[0].start or 0)
    col_offset = int(view_slice[1].start or 0)
    out_h = int((view_slice[0].stop or h) - row_offset)
    out_w = int((view_slice[1].stop or w) - col_offset)
    image = Image.new("RGB", (out_w, out_h), (246, 246, 241))
    pixels = image.load()
    room_view = room[view_slice]
    trav_view = trav[view_slice]
    selected_mask = _selected_room_mask(layout, selected_rooms)[view_slice] if selected_rooms else room_view > 0
    for row in range(out_h):
        for col in range(out_w):
            if room_view[row, col] > 0 and selected_mask[row, col]:
                pixels[col, row] = (228, 232, 224)
            if trav_view[row, col] and selected_mask[row, col]:
                pixels[col, row] = (214, 226, 214)
    draw = ImageDraw.Draw(image, "RGBA")
    labels = []
    records = parse_scene_objects(scene)
    operation_by_name: dict[str, str] = {}
    aug_meta = (scene.get("metadata") or {}).get("scoper.benchmark.behavior.scene_geometry") or {}
    operation_log = aug_meta.get("operation_log") or []
    if isinstance(operation_log, list):
        for op in operation_log:
            if isinstance(op, dict) and op.get("name"):
                raw_op = str(op.get("op") or "modified")
                if raw_op == "add_large":
                    operation_by_name[str(op["name"])] = "added"
                elif raw_op in {"rewrite_large", "relocate_large"}:
                    operation_by_name[str(op["name"])] = "rewritten" if raw_op == "rewrite_large" else "relocated"
                elif raw_op == "model_swap":
                    operation_by_name.setdefault(str(op["name"]), "model_swap")
                elif raw_op == "add_small":
                    operation_by_name[str(op["name"])] = "added"
    for name, record in sorted(records.items()):
        record_rooms = set(record.rooms)
        if selected_rooms and not (record_rooms & selected_rooms):
            continue
        asset = catalog.get_model(record.category, record.model)
        if asset is None:
            continue
        role = getattr(classifier.get(record.category), "role", "unknown") if classifier is not None else "unknown"
        operation = operation_by_name.get(name, "kept")
        if "_aug_" in name and operation == "kept":
            operation = "added"
        if modified_only and operation == "kept":
            continue
        if role in hidden_roles and operation == "kept":
            continue
        if record.category in hidden_categories and operation == "kept":
            continue
        bbox = scaled_bbox(asset, record.scale)
        yaw = quat_to_yaw(record.ori)
        extent = footprint_extent_for_yaw((bbox[0], bbox[1]), yaw)
        rows, cols = footprint_rows_cols(layout, (record.pos[0], record.pos[1]), extent)
        if rows.size == 0 or cols.size == 0:
            continue
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        if row_max < row_offset or col_max < col_offset or row_min >= row_offset + out_h or col_min >= col_offset + out_w:
            continue
        row_min, row_max = max(row_offset, row_min), min(row_offset + out_h - 1, row_max)
        col_min, col_max = max(col_offset, col_min), min(col_offset + out_w - 1, col_max)
        draw_row_min = row_min - row_offset
        draw_row_max = row_max - row_offset
        draw_col_min = col_min - col_offset
        draw_col_max = col_max - col_offset
        color = ROLE_COLORS.get(role, (80, 80, 80))
        if operation == "added":
            outline = (255, 30, 30, 255)
        elif operation in {"rewritten", "relocated"}:
            outline = (25, 105, 255, 255)
        elif operation == "model_swap":
            outline = (130, 60, 210, 230)
        else:
            outline = (20, 20, 20, 180)
        fill = (*color, 100 if role != "small_prop" else 150)
        draw.rectangle((draw_col_min, draw_row_min, draw_col_max, draw_row_max), fill=fill, outline=outline, width=3 if operation != "kept" else 1)
        center_row, center_col = layout.world_to_map((record.pos[0], record.pos[1]))
        label = f"{record.category}"
        if len(labels) < max_labels and role in {"large_floor", "support_surface", "small_prop"}:
            draw.text((int(center_col) - col_offset + 3, int(center_row) - row_offset + 3), label[:24], fill=(20, 20, 20, 230), font=_font())
        labels.append(
            {
                "name": name,
                "category": record.category,
                "model": record.model,
                "role": role,
                "rooms": list(record.rooms),
                "room_type": room_type_from_instance(record.rooms[0]) if record.rooms else None,
                "pos": list(record.pos),
                "bbox_size": list(bbox),
                "footprint_extent_xy": list(extent),
                "generated": operation == "added",
                "operation": operation,
                "bev_rect": [col_min, row_min, col_max, row_max],
                "view_rect": [draw_col_min, draw_row_min, draw_col_max, draw_row_max],
            }
        )
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    manifest = {
        "scene_model": str(scene_model),
        "scene_file": str(scene_file),
        "bev_png": str(output_png),
        "label_count": len(labels),
        "rooms": sorted(selected_rooms),
        "crop_to_rooms": bool(crop_to_rooms),
        "view_slice": [row_offset, int(view_slice[0].stop or h), col_offset, int(view_slice[1].stop or w)],
        "hidden_roles": sorted(hidden_roles),
        "hidden_categories": sorted(hidden_categories),
        "modified_only": bool(modified_only),
        "labels": labels,
    }
    if labels_json is not None:
        out = Path(labels_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 10)
    except Exception:
        return ImageFont.load_default()


def _selected_room_mask(layout: Any, rooms: set[str]) -> np.ndarray:
    mask = np.zeros_like(np.asarray(layout.room_ins_map), dtype=bool)
    for room_id in rooms:
        room_mask = layout.room_mask(room_id)
        if room_mask is not None:
            mask |= np.asarray(room_mask, dtype=bool)
    return mask


def _room_view_slice(layout: Any, rooms: set[str], *, margin_px: int = 80) -> tuple[slice, slice]:
    mask = _selected_room_mask(layout, rooms)
    rows, cols = np.where(mask)
    h, w = mask.shape
    if rows.size == 0 or cols.size == 0:
        return (slice(0, h), slice(0, w))
    r0 = max(0, int(rows.min()) - int(margin_px))
    r1 = min(h, int(rows.max()) + int(margin_px) + 1)
    c0 = max(0, int(cols.min()) - int(margin_px))
    c1 = min(w, int(cols.max()) + int(margin_px) + 1)
    return (slice(r0, r1), slice(c0, c1))


__all__ = ["render_bev"]
