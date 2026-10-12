"""Layout quality audit for generated BEHAVIOR scene variants."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scoper.benchmark.behavior.scene.room_layout_maps import load_behavior_room_layout_maps

from scoper.benchmark.behavior.scene_geometry.asset_catalog import AssetCatalog, bbox_bottom_z, bbox_top_z, scaled_bbox
from scoper.benchmark.behavior.scene_geometry.category_classifier import CategoryClassifier
from scoper.benchmark.behavior.scene_geometry.geometry import aabb_overlap_ratio, existing_footprints, footprint_extent_for_yaw, footprint_rows_cols
from scoper.benchmark.behavior.scene_geometry.geometry import protected_zones_from_scene, room_free_space_metrics
from scoper.benchmark.behavior.scene_geometry.paths import assets_root as default_assets_root
from scoper.benchmark.behavior.scene_geometry.rules import AugmentationRules, load_rules
from scoper.benchmark.behavior.scene_geometry.scene_io import load_scene_json, parse_scene_objects, quat_to_yaw, room_type_from_instance
from scoper.benchmark.behavior.scene_geometry.surface_occupancy import ObjectAABB, aabb3d_overlap_ratio, object_aabb_from_record
from scoper.benchmark.behavior.scene_geometry.taxonomy_index import TaxonomyIndex


FLOOR_LAYER_CATEGORIES = {"carpet", "rug", "mat"}
SUPPORT_STACK_PAIRS = {
    ("bottom_cabinet", "countertop"),
    ("bottom_cabinet_no_top", "countertop"),
    ("bottom_cabinet", "top_cabinet"),
    ("bottom_cabinet_no_top", "top_cabinet"),
}
ASSOCIATED_OVERLAP_PAIRS = {
    ("breakfast_table", "straight_chair"),
    ("breakfast_table", "swivel_chair"),
    ("breakfast_table", "armchair"),
    ("coffee_table", "sofa"),
    ("coffee_table", "armchair"),
    ("desk", "straight_chair"),
    ("desk", "swivel_chair"),
}
EDGE_ANCHOR_CATEGORIES = {
    "bed",
    "bookcase",
    "bottom_cabinet",
    "bottom_cabinet_no_top",
    "console_table",
    "desk",
    "floor_lamp",
    "locker",
    "nightstand",
    "public_trash_can",
    "shelf",
    "sofa",
    "standing_loudspeaker",
    "standing_tv",
    "trash_can",
    "wardrobe",
}
MIDDLE_ANCHOR_CATEGORIES = {"breakfast_table", "carpet", "coffee_table", "ottoman", "pedestal_table", "taboret"}


def audit_layout_quality(
    *,
    scene_model: str,
    scene_file: str | Path,
    source_scene_file: str | Path | None = None,
    catalog: AssetCatalog | None = None,
    classifier: CategoryClassifier | None = None,
    rules: AugmentationRules | None = None,
    assets_root: str | Path | None = None,
) -> dict[str, Any]:
    assets = Path(assets_root) if assets_root is not None else default_assets_root()
    scene_path = Path(scene_file)
    scene = load_scene_json(scene_path)
    source_scene = load_scene_json(source_scene_file) if source_scene_file is not None and Path(source_scene_file).exists() else None
    catalog = catalog or AssetCatalog.build(assets)
    rules = rules or load_rules()
    classifier = classifier or CategoryClassifier(catalog, TaxonomyIndex.build(), rules)
    layout = load_behavior_room_layout_maps(str(scene_model), assets_root=assets, floor=0)

    records = parse_scene_objects(scene)
    source_records = parse_scene_objects(source_scene) if source_scene is not None else {}
    operation_by_name = _operation_by_name(scene)
    parent_by_name = _parent_by_name(scene)
    source_overlap_keys = _source_overlap_keys(source_scene, catalog, classifier) if source_scene is not None else set()
    source_pair_keys = _pair_keys(source_records)

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    stats = {
        "checked_objects": len(records),
        "generated_or_modified_objects": sum(1 for name in records if _is_augmented(name, operation_by_name, source_records)),
        "rewritten_objects": sum(1 for op in operation_by_name.values() if op == "rewritten"),
        "added_objects": sum(1 for op in operation_by_name.values() if op == "added"),
        "model_swap_objects": sum(1 for op in operation_by_name.values() if op == "model_swap"),
    }

    _check_object_room_and_bounds(records, source_records, operation_by_name, layout, catalog, classifier, rules, issues, warnings)
    _check_density(records, source_records, operation_by_name, layout, catalog, classifier, rules, issues, warnings)
    source_clearance_keys = _clearance_blockage_keys(source_scene, catalog, classifier, rules) if source_scene is not None else set()
    _check_usability_clearances(
        scene,
        records,
        source_records,
        operation_by_name,
        source_clearance_keys,
        layout,
        catalog,
        classifier,
        rules,
        issues,
        warnings,
    )
    _check_overlaps(
        records,
        source_records,
        source_overlap_keys,
        operation_by_name,
        catalog,
        classifier,
        issues,
        warnings,
    )
    _check_small_overlaps(records, source_records, operation_by_name, catalog, classifier, rules, issues, warnings)
    _check_small_object_support(records, source_pair_keys, operation_by_name, parent_by_name, catalog, classifier, rules, issues, warnings)
    _check_vertical_support(records, source_records, operation_by_name, parent_by_name, catalog, classifier, issues, warnings)
    _check_anchor_semantics(records, source_records, operation_by_name, layout, catalog, classifier, warnings)

    issue_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for row in issues:
        issue_counts[str(row["kind"])] = issue_counts.get(str(row["kind"]), 0) + 1
    for row in warnings:
        warning_counts[str(row["kind"])] = warning_counts.get(str(row["kind"]), 0) + 1
    return {
        "scene_model": str(scene_model),
        "scene_file": str(scene_path),
        "source_scene_file": str(source_scene_file) if source_scene_file is not None else None,
        "status": "passed" if not issues else "failed",
        "stats": stats,
        "issue_counts": dict(sorted(issue_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "issues": issues,
        "warnings": warnings[:300],
    }


def write_layout_quality_report(path: str | Path, report: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _check_object_room_and_bounds(
    records: dict[str, Any],
    source_records: dict[str, Any],
    operation_by_name: dict[str, str],
    layout: object,
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    rules: AugmentationRules,
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    for name, record in records.items():
        asset = catalog.get_model(record.category, record.model)
        if asset is None or not record.primary_room:
            continue
        operation = operation_by_name.get(name, "kept")
        augmented = _is_augmented(name, operation_by_name, source_records)
        placement_changed = operation in {"added", "rewritten", "relocated"} or name not in source_records or "_aug_" in name
        role = getattr(classifier.get(record.category), "role", "unknown")
        room_type = room_type_from_instance(record.primary_room)
        if placement_changed and role in {"large_floor", "support_surface", "small_prop"}:
            if not _category_allowed(record.category, room_type, role, rules):
                issues.append(
                    _issue(
                        "room_category_violation",
                        name=name,
                        category=record.category,
                        room=record.primary_room,
                        operation=operation,
                    )
                )
        bbox = scaled_bbox(asset, record.scale)
        extent = footprint_extent_for_yaw((bbox[0], bbox[1]), quat_to_yaw(record.ori))
        rows, cols = footprint_rows_cols(layout, (record.pos[0], record.pos[1]), extent)
        if rows.size == 0 or cols.size == 0:
            continue
        room = layout.room_mask(record.primary_room)
        if room is None:
            continue
        in_bounds = rows.min() >= 0 and cols.min() >= 0 and rows.max() < room.shape[0] and cols.max() < room.shape[1]
        if not in_bounds:
            target = issues if augmented else warnings
            target.append(_issue("footprint_out_of_map", name=name, category=record.category, room=record.primary_room, operation=operation))
            continue
        room_patch = np.asarray(room, dtype=bool)[np.ix_(rows, cols)]
        room_fraction = float(room_patch.mean()) if room_patch.size else 0.0
        if placement_changed and role in {"large_floor", "support_surface"} and room_fraction < 0.98:
            issues.append(
                _issue(
                    "augmented_large_outside_room",
                    name=name,
                    category=record.category,
                    room=record.primary_room,
                    operation=operation,
                    room_fraction=round(room_fraction, 4),
                )
            )
        elif role in {"large_floor", "support_surface"} and room_fraction < 0.90:
            warnings.append(
                _issue(
                    "kept_large_outside_room",
                    name=name,
                    category=record.category,
                    room=record.primary_room,
                    operation=operation,
                    room_fraction=round(room_fraction, 4),
                )
            )


def _check_density(
    records: dict[str, Any],
    source_records: dict[str, Any],
    operation_by_name: dict[str, str],
    layout: object,
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    rules: AugmentationRules,
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    max_ratio = float(rules.placement_policy.get("default_space_budget", rules.placement_policy.get("max_occupied_area_ratio", 0.45)))
    min_free = float(rules.placement_policy.get("min_traversable_free_ratio", 0.35))
    min_connected = float(rules.placement_policy.get("min_connected_free_ratio", 0.28))
    touched_rooms = _augmented_rooms(records, source_records, operation_by_name)
    source_ratios = _room_large_occupied_ratios(source_records, layout, catalog, classifier) if source_records else {}
    final_ratios = _room_large_occupied_ratios(records, layout, catalog, classifier)
    for room_id, ratio in sorted(final_ratios.items()):
        source_ratio = source_ratios.get(room_id)
        introduced = _ratio_violation_introduced(
            room_id=room_id,
            touched_rooms=touched_rooms,
            final_value=ratio,
            source_value=source_ratio,
            limit=max_ratio,
            higher_is_worse=True,
        )
        row = _issue(
            "room_space_budget_exceeded",
            room=room_id,
            occupied_ratio=round(ratio, 4),
            source_occupied_ratio=round(float(source_ratio), 4) if source_ratio is not None else None,
            limit=max_ratio,
            introduced_by_augmentation=introduced,
        )
        if ratio > max_ratio * 1.20 and introduced:
            issues.append(row)
        elif ratio > max_ratio:
            row["kind"] = "room_space_budget_near_limit" if introduced else "inherited_room_space_budget_exceeded"
            warnings.append(row)

    light_scene = _light_scene_from_records(records)
    occupied = existing_footprints(light_scene, catalog, classifier, include_roles={"large_floor", "support_surface", "fixture"})
    occupied.extend(protected_zones_from_scene(light_scene, catalog, classifier, policy=rules.placement_policy))
    source_metrics_by_room: dict[str, Any] = {}
    if source_records:
        source_light_scene = _light_scene_from_records(source_records)
        source_occupied = existing_footprints(source_light_scene, catalog, classifier, include_roles={"large_floor", "support_surface", "fixture"})
        source_occupied.extend(protected_zones_from_scene(source_light_scene, catalog, classifier, policy=rules.placement_policy))
        for room_id in sorted(layout.room_ins_name_to_ins_id):
            source_metrics_by_room[room_id] = room_free_space_metrics(
                layout,
                room_id,
                source_occupied,
                clearance_m=float(rules.placement_policy.get("usability_clearance_m", rules.placement_policy.get("large_rewrite_clearance_m", 0.05))),
            )
    for room_id in sorted(layout.room_ins_name_to_ins_id):
        metrics = room_free_space_metrics(
            layout,
            room_id,
            occupied,
            clearance_m=float(rules.placement_policy.get("usability_clearance_m", rules.placement_policy.get("large_rewrite_clearance_m", 0.05))),
        )
        if metrics.traversable_cells <= 0:
            continue
        if metrics.free_ratio < min_free:
            source_metrics = source_metrics_by_room.get(room_id)
            source_value = source_metrics.free_ratio if source_metrics is not None else None
            introduced = _ratio_violation_introduced(
                room_id=room_id,
                touched_rooms=touched_rooms,
                final_value=metrics.free_ratio,
                source_value=source_value,
                limit=min_free,
                higher_is_worse=False,
            )
            row = _issue(
                "free_space_ratio_too_low" if introduced else "inherited_free_space_ratio_too_low",
                room=room_id,
                free_ratio=round(metrics.free_ratio, 4),
                source_free_ratio=round(float(source_value), 4) if source_value is not None else None,
                limit=min_free,
                introduced_by_augmentation=introduced,
            )
            (issues if introduced else warnings).append(row)
        if metrics.connected_ratio < min_connected:
            source_metrics = source_metrics_by_room.get(room_id)
            source_value = source_metrics.connected_ratio if source_metrics is not None else None
            introduced = _ratio_violation_introduced(
                room_id=room_id,
                touched_rooms=touched_rooms,
                final_value=metrics.connected_ratio,
                source_value=source_value,
                limit=min_connected,
                higher_is_worse=False,
            )
            row = _issue(
                "free_space_disconnected" if introduced else "inherited_free_space_disconnected",
                room=room_id,
                connected_ratio=round(metrics.connected_ratio, 4),
                source_connected_ratio=round(float(source_value), 4) if source_value is not None else None,
                limit=min_connected,
                introduced_by_augmentation=introduced,
            )
            (issues if introduced else warnings).append(row)


def _check_usability_clearances(
    scene: dict[str, Any],
    records: dict[str, Any],
    source_records: dict[str, Any],
    operation_by_name: dict[str, str],
    source_clearance_keys: set[tuple[str, str, str]],
    layout: object,
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    rules: AugmentationRules,
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    del layout
    zones = protected_zones_from_scene(scene, catalog, classifier, policy=rules.placement_policy)
    object_fps = existing_footprints(scene, catalog, classifier, include_roles={"large_floor", "support_surface", "fixture"})
    for zone in zones:
        for fp in object_fps:
            if fp.room and zone.room and fp.room != zone.room:
                continue
            if zone.owner and zone.owner == fp.name:
                continue
            ratio = aabb_overlap_ratio(zone, fp)
            if ratio <= 0.0:
                continue
            fp_aug = _is_augmented(fp.name, operation_by_name, source_records)
            zone_aug = bool(zone.owner and _is_augmented(zone.owner, operation_by_name, source_records))
            kind = "door_clearance_blocked" if zone.source == "door_clearance" else "openable_clearance_blocked"
            key = (str(zone.owner or zone.name), str(fp.name), str(zone.source))
            introduced = key not in source_clearance_keys and bool(fp_aug or zone_aug)
            row = _issue(
                kind,
                zone=zone.name,
                owner=zone.owner,
                blocker=fp.name,
                blocker_category=fp.category,
                room=zone.room,
                ratio=round(float(ratio), 4),
                introduced_by_augmentation=introduced,
            )
            if introduced:
                issues.append(row)
            else:
                warnings.append(row)
            break


def _check_overlaps(
    records: dict[str, Any],
    source_records: dict[str, Any],
    source_overlap_keys: set[tuple[str, str]],
    operation_by_name: dict[str, str],
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    scene = _light_scene_from_records(records)
    fps = existing_footprints(scene, catalog, classifier, include_roles={"large_floor", "support_surface", "fixture"})
    for i, left in enumerate(fps):
        for right in fps[i + 1 :]:
            if left.room and right.room and left.room != right.room:
                continue
            ratio = aabb_overlap_ratio(left, right)
            if ratio <= 0.18:
                continue
            key = _pair_key(left.name, right.name)
            left_aug = _is_augmented(left.name, operation_by_name, source_records)
            right_aug = _is_augmented(right.name, operation_by_name, source_records)
            introduced = key not in source_overlap_keys and (left_aug or right_aug)
            benign = _is_benign_overlap(left.category, right.category)
            row = _issue(
                "introduced_large_overlap" if introduced else "inherited_large_overlap",
                a=left.name,
                b=right.name,
                categories=[left.category, right.category],
                ratio=round(float(ratio), 4),
                room=left.room,
                introduced_by_augmentation=introduced,
                benign_pair=benign,
            )
            if introduced and not benign and ratio > 0.25:
                issues.append(row)
            else:
                warnings.append(row)


def _check_small_overlaps(
    records: dict[str, Any],
    source_records: dict[str, Any],
    operation_by_name: dict[str, str],
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    rules: AugmentationRules,
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    threshold = max(0.0, float(rules.placement_policy.get("small_quality_max_overlap_ratio", rules.placement_policy.get("small_max_overlap_ratio", 0.0))))
    source_keys = _small_overlap_keys(source_records, catalog, classifier, threshold)
    aabbs = _small_aabbs_from_records(records, catalog, classifier)
    for i, left in enumerate(aabbs):
        for right in aabbs[i + 1 :]:
            if left.room and right.room and left.room != right.room:
                continue
            ratio = aabb3d_overlap_ratio(left, right)
            if ratio <= threshold:
                continue
            key = _pair_key(left.name, right.name)
            left_aug = _is_augmented(left.name, operation_by_name, source_records)
            right_aug = _is_augmented(right.name, operation_by_name, source_records)
            introduced = key not in source_keys and (left_aug or right_aug)
            row = _issue(
                "introduced_small_overlap" if introduced else "inherited_small_overlap",
                a=left.name,
                b=right.name,
                categories=[left.category, right.category],
                ratio=round(float(ratio), 4),
                room=left.room,
                introduced_by_augmentation=introduced,
            )
            if introduced:
                issues.append(row)
            else:
                warnings.append(row)


def _check_small_object_support(
    records: dict[str, Any],
    source_pair_keys: set[tuple[str, str]],
    operation_by_name: dict[str, str],
    parent_by_name: dict[str, str],
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    rules: AugmentationRules,
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    supports = set(str(item) for item in rules.receptacle_policy.get("support_categories") or [])
    children_by_support = rules.receptacle_policy.get("children_by_support") or {}
    support_records = [record for record in records.values() if record.category in supports]
    for name, record in records.items():
        role = getattr(classifier.get(record.category), "role", "unknown")
        if role != "small_prop" or operation_by_name.get(name) != "added":
            continue
        explicit_parent = records.get(parent_by_name.get(name, ""))
        parent = explicit_parent if explicit_parent is not None else _nearest_support(record, support_records, catalog)
        if parent is None:
            issues.append(_issue("added_small_without_support", name=name, category=record.category, operation=operation_by_name.get(name, "kept")))
            continue
        allowed = set(str(item) for item in children_by_support.get(parent.category) or rules.receptacle_policy.get("fallback_children") or [])
        pair = _pair_key(name, parent.name)
        if record.category not in allowed and pair not in source_pair_keys:
            warnings.append(
                _issue(
                    "small_support_policy_mismatch",
                    name=name,
                    category=record.category,
                    parent=parent.name,
                    parent_category=parent.category,
                    operation=operation_by_name.get(name, "kept"),
                )
            )


def _check_vertical_support(
    records: dict[str, Any],
    source_records: dict[str, Any],
    operation_by_name: dict[str, str],
    parent_by_name: dict[str, str],
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    floor_tol = 0.05
    support_tol = 0.05
    for name, record in records.items():
        asset = catalog.get_model(record.category, record.model)
        if asset is None:
            continue
        operation = operation_by_name.get(name, "kept")
        role = getattr(classifier.get(record.category), "role", "unknown")
        bottom_z = bbox_bottom_z(record.pos[2], asset, record.scale)
        if operation in {"added", "rewritten", "relocated"} and role in {"large_floor", "support_surface"}:
            if bottom_z > floor_tol:
                issues.append(
                    _issue(
                        "augmented_object_floating_above_floor",
                        name=name,
                        category=record.category,
                        operation=operation,
                        bottom_z=round(float(bottom_z), 4),
                        expected_bottom_z=0.0,
                    )
                )
            elif bottom_z < -floor_tol:
                issues.append(
                    _issue(
                        "augmented_object_sunk_below_floor",
                        name=name,
                        category=record.category,
                        operation=operation,
                        bottom_z=round(float(bottom_z), 4),
                        expected_bottom_z=0.0,
                    )
                )
        elif operation == "model_swap" and name in source_records and role in {"large_floor", "support_surface", "small_prop"}:
            old_asset = catalog.get_model(source_records[name].category, source_records[name].model)
            if old_asset is not None:
                old_bottom_z = bbox_bottom_z(source_records[name].pos[2], old_asset, source_records[name].scale)
                delta = bottom_z - old_bottom_z
                if abs(delta) > floor_tol:
                    issues.append(
                        _issue(
                            "model_swap_bottom_z_changed",
                            name=name,
                            category=record.category,
                            operation=operation,
                            bottom_z=round(float(bottom_z), 4),
                            source_bottom_z=round(float(old_bottom_z), 4),
                            delta_z=round(float(delta), 4),
                        )
                    )

        if operation == "added" and role == "small_prop":
            parent = records.get(parent_by_name.get(name, ""))
            if parent is None:
                continue
            parent_asset = catalog.get_model(parent.category, parent.model)
            if parent_asset is None:
                continue
            support_z = bbox_top_z(parent.pos[2], parent_asset, parent.scale)
            clearance = bottom_z - support_z
            if clearance > support_tol:
                issues.append(
                    _issue(
                        "added_small_floating_above_support",
                        name=name,
                        category=record.category,
                        parent=parent.name,
                        parent_category=parent.category,
                        bottom_z=round(float(bottom_z), 4),
                        support_z=round(float(support_z), 4),
                        clearance=round(float(clearance), 4),
                    )
                )
            elif clearance < -support_tol:
                issues.append(
                    _issue(
                        "added_small_sunk_into_support",
                        name=name,
                        category=record.category,
                        parent=parent.name,
                        parent_category=parent.category,
                        bottom_z=round(float(bottom_z), 4),
                        support_z=round(float(support_z), 4),
                        clearance=round(float(clearance), 4),
                    )
                )
            elif abs(clearance) > 0.025:
                warnings.append(
                    _issue(
                        "added_small_support_clearance_near_limit",
                        name=name,
                        category=record.category,
                        parent=parent.name,
                        clearance=round(float(clearance), 4),
                    )
                )


def _check_anchor_semantics(
    records: dict[str, Any],
    source_records: dict[str, Any],
    operation_by_name: dict[str, str],
    layout: object,
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    warnings: list[dict[str, Any]],
) -> None:
    for name, record in records.items():
        if not _is_augmented(name, operation_by_name, source_records) or not record.primary_room:
            continue
        role = getattr(classifier.get(record.category), "role", "unknown")
        if role not in {"large_floor", "support_surface"}:
            continue
        room = layout.room_mask(record.primary_room)
        asset = catalog.get_model(record.category, record.model)
        if room is None or asset is None:
            continue
        rows, cols = np.where(np.asarray(room, dtype=bool))
        if rows.size == 0:
            continue
        row, col = layout.world_to_map((record.pos[0], record.pos[1]))
        distance_px = float(min(row - rows.min(), rows.max() - row, col - cols.min(), cols.max() - col))
        distance_m = distance_px * float(layout.map_resolution)
        if record.category in EDGE_ANCHOR_CATEGORIES and distance_m > 1.25:
            warnings.append(
                _issue(
                    "edge_anchor_far_from_wall",
                    name=name,
                    category=record.category,
                    room=record.primary_room,
                    operation=operation_by_name.get(name, "kept"),
                    distance_to_room_bbox_m=round(distance_m, 3),
                )
            )
        if record.category in MIDDLE_ANCHOR_CATEGORIES and distance_m < 0.25:
            warnings.append(
                _issue(
                    "middle_anchor_too_close_to_wall",
                    name=name,
                    category=record.category,
                    room=record.primary_room,
                    operation=operation_by_name.get(name, "kept"),
                    distance_to_room_bbox_m=round(distance_m, 3),
                )
            )


def _source_overlap_keys(source_scene: dict[str, Any] | None, catalog: AssetCatalog, classifier: CategoryClassifier) -> set[tuple[str, str]]:
    if source_scene is None:
        return set()
    fps = existing_footprints(source_scene, catalog, classifier, include_roles={"large_floor", "support_surface", "fixture"})
    out = set()
    for i, left in enumerate(fps):
        for right in fps[i + 1 :]:
            if left.room and right.room and left.room != right.room:
                continue
            if aabb_overlap_ratio(left, right) > 0.18:
                out.add(_pair_key(left.name, right.name))
    return out


def _small_overlap_keys(
    records: dict[str, Any],
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    threshold: float,
) -> set[tuple[str, str]]:
    aabbs = _small_aabbs_from_records(records, catalog, classifier)
    out = set()
    for i, left in enumerate(aabbs):
        for right in aabbs[i + 1 :]:
            if left.room and right.room and left.room != right.room:
                continue
            if aabb3d_overlap_ratio(left, right) > threshold:
                out.add(_pair_key(left.name, right.name))
    return out


def _small_aabbs_from_records(
    records: dict[str, Any],
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
) -> list[ObjectAABB]:
    out: list[ObjectAABB] = []
    for record in records.values():
        role = getattr(classifier.get(record.category), "role", "unknown")
        if role != "small_prop":
            continue
        asset = catalog.get_model(record.category, record.model)
        if asset is None:
            continue
        out.append(object_aabb_from_record(record, asset))
    return out


def _clearance_blockage_keys(
    scene: dict[str, Any] | None,
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
    rules: AugmentationRules,
) -> set[tuple[str, str, str]]:
    if scene is None:
        return set()
    zones = protected_zones_from_scene(scene, catalog, classifier, policy=rules.placement_policy)
    object_fps = existing_footprints(scene, catalog, classifier, include_roles={"large_floor", "support_surface", "fixture"})
    out: set[tuple[str, str, str]] = set()
    for zone in zones:
        for fp in object_fps:
            if fp.room and zone.room and fp.room != zone.room:
                continue
            if zone.owner and zone.owner == fp.name:
                continue
            if aabb_overlap_ratio(zone, fp) > 0.0:
                out.add((str(zone.owner or zone.name), str(fp.name), str(zone.source)))
    return out


def _room_large_occupied_ratios(
    records: dict[str, Any],
    layout: object,
    catalog: AssetCatalog,
    classifier: CategoryClassifier,
) -> dict[str, float]:
    area_by_room: dict[str, float] = {}
    occupied_by_room: dict[str, float] = {}
    resolution = float(layout.map_resolution)
    for room_id in layout.room_ins_name_to_ins_id:
        room = layout.room_mask(room_id)
        if room is not None:
            area_by_room[room_id] = float(np.asarray(room, dtype=bool).sum()) * resolution * resolution
    for record in records.values():
        if not record.primary_room:
            continue
        role = getattr(classifier.get(record.category), "role", "unknown")
        if role not in {"large_floor", "support_surface"} or record.category in FLOOR_LAYER_CATEGORIES:
            continue
        asset = catalog.get_model(record.category, record.model)
        if asset is None:
            continue
        bbox = scaled_bbox(asset, record.scale)
        occupied_by_room[record.primary_room] = occupied_by_room.get(record.primary_room, 0.0) + float(bbox[0]) * float(bbox[1])
    out: dict[str, float] = {}
    for room_id, occupied_area in occupied_by_room.items():
        room_area = area_by_room.get(room_id, 0.0)
        if room_area > 0:
            out[room_id] = occupied_area / room_area
    return out


def _augmented_rooms(
    records: dict[str, Any],
    source_records: dict[str, Any],
    operation_by_name: dict[str, str],
) -> set[str]:
    rooms: set[str] = set()
    for name, record in records.items():
        if record.primary_room and _is_augmented(name, operation_by_name, source_records):
            rooms.add(str(record.primary_room))
    return rooms


def _ratio_violation_introduced(
    *,
    room_id: str,
    touched_rooms: set[str],
    final_value: float,
    source_value: float | None,
    limit: float,
    higher_is_worse: bool,
) -> bool:
    if room_id not in touched_rooms:
        return False
    epsilon = 0.01
    if source_value is None:
        return True
    if higher_is_worse:
        source_bad = source_value > limit
        return final_value > limit and (not source_bad or final_value > source_value + epsilon)
    source_bad = source_value < limit
    return final_value < limit and (not source_bad or final_value < source_value - epsilon)


def _pair_keys(records: dict[str, Any]) -> set[tuple[str, str]]:
    names = sorted(records)
    return {_pair_key(left, right) for i, left in enumerate(names) for right in names[i + 1 :]}


def _nearest_support(record: Any, supports: list[Any], catalog: AssetCatalog) -> Any | None:
    best = None
    best_dist = float("inf")
    for support in supports:
        if record.primary_room and support.primary_room and record.primary_room != support.primary_room:
            continue
        asset = catalog.get_model(support.category, support.model)
        if asset is None:
            continue
        bbox = scaled_bbox(asset, support.scale)
        dx = abs(float(record.pos[0]) - float(support.pos[0])) - bbox[0] / 2.0
        dy = abs(float(record.pos[1]) - float(support.pos[1])) - bbox[1] / 2.0
        dz = abs(float(record.pos[2]) - float(support.pos[2]))
        dist = max(0.0, dx) + max(0.0, dy) + max(0.0, dz - 1.5)
        if dist < best_dist:
            best = support
            best_dist = dist
    return best if best_dist <= 0.35 else None


def _operation_by_name(scene: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    aug_meta = (scene.get("metadata") or {}).get("scoper.benchmark.behavior.scene_geometry") or {}
    operation_log = aug_meta.get("operation_log") or []
    if not isinstance(operation_log, list):
        return out
    for op in operation_log:
        if not isinstance(op, dict) or not op.get("name"):
            continue
        raw = str(op.get("op") or "")
        name = str(op["name"])
        if raw == "add_large" or raw == "add_small":
            out[name] = "added"
        elif raw == "rewrite_large":
            out[name] = "rewritten"
        elif raw == "relocate_large":
            out[name] = "relocated"
        elif raw == "model_swap":
            out.setdefault(name, "model_swap")
        elif raw == "remove_large":
            out[name] = "removed"
    return out


def _parent_by_name(scene: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    aug_meta = (scene.get("metadata") or {}).get("scoper.benchmark.behavior.scene_geometry") or {}
    operation_log = aug_meta.get("operation_log") or []
    if not isinstance(operation_log, list):
        return out
    for op in operation_log:
        if isinstance(op, dict) and op.get("name") and op.get("parent"):
            out[str(op["name"])] = str(op["parent"])
    return out


def _is_augmented(name: str, operation_by_name: dict[str, str], source_records: dict[str, Any]) -> bool:
    return name in operation_by_name or name not in source_records or "_aug_" in name


def _is_benign_overlap(left: str, right: str) -> bool:
    pair = {str(left), str(right)}
    if pair & FLOOR_LAYER_CATEGORIES:
        return True
    ordered = (str(left), str(right))
    reverse = (str(right), str(left))
    return ordered in SUPPORT_STACK_PAIRS or reverse in SUPPORT_STACK_PAIRS or ordered in ASSOCIATED_OVERLAP_PAIRS or reverse in ASSOCIATED_OVERLAP_PAIRS


def _category_allowed(category: str, room_type: str, role: str, rules: AugmentationRules) -> bool:
    if role in {"large_floor", "support_surface"}:
        mapping = rules.placement_policy.get("allowed_large_by_room") or {}
        if room_type in mapping:
            return category in {str(item) for item in mapping.get(room_type) or []}
    if role == "small_prop":
        mapping = rules.receptacle_policy.get("allowed_small_by_room") or {}
        if room_type in mapping:
            return category in {str(item) for item in mapping.get(room_type) or []}
    return True


def _pair_key(left: str, right: str) -> tuple[str, str]:
    a, b = sorted((str(left), str(right)))
    return a, b


def _light_scene_from_records(records: dict[str, Any]) -> dict[str, Any]:
    scene = {
        "objects_info": {"init_info": {}},
        "state": {"registry": {"object_registry": {}}},
    }
    for name, record in records.items():
        scene["objects_info"]["init_info"][name] = {
            "args": {
                "name": name,
                "category": record.category,
                "model": record.model,
                "scale": list(record.scale),
                "in_rooms": list(record.rooms),
            }
        }
        scene["state"]["registry"]["object_registry"][name] = {"root_link": {"pos": list(record.pos), "ori": list(record.ori)}}
    return scene


def _issue(kind: str, **kwargs: Any) -> dict[str, Any]:
    row = {"kind": kind}
    row.update(kwargs)
    return row


__all__ = ["audit_layout_quality", "write_layout_quality_report"]
