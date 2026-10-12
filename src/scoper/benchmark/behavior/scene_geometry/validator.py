"""Validation for generated BEHAVIOR scene variants."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from scoper.benchmark.behavior.scene.room_layout_maps import load_behavior_room_layout_maps

from scoper.benchmark.behavior.scene_geometry.asset_catalog import AssetCatalog
from scoper.benchmark.behavior.scene_geometry.category_classifier import CategoryClassifier
from scoper.benchmark.behavior.scene_geometry.geometry import aabb_overlap_ratio, existing_footprints
from scoper.benchmark.behavior.scene_geometry.paths import assets_root as default_assets_root
from scoper.benchmark.behavior.scene_geometry.rules import AugmentationRules, load_rules
from scoper.benchmark.behavior.scene_geometry.scene_io import load_scene_json, parse_scene_objects, room_type_from_instance
from scoper.benchmark.behavior.scene_geometry.surface_occupancy import aabb3d_overlap_ratio, small_object_aabbs_from_scene


def validate_scene_variant(
    *,
    scene_model: str,
    scene_file: str | Path,
    catalog: AssetCatalog | None = None,
    classifier: CategoryClassifier | None = None,
    rules: AugmentationRules | None = None,
    source_scene_file: str | Path | None = None,
    assets_root: str | Path | None = None,
) -> dict[str, Any]:
    assets = Path(assets_root) if assets_root is not None else default_assets_root()
    scene = load_scene_json(scene_file)
    catalog = catalog or AssetCatalog.build(assets)
    rules = rules or load_rules()
    records = parse_scene_objects(scene)
    errors: list[str] = []
    warnings: list[str] = []
    aug_meta = (scene.get("metadata") or {}).get("scoper.benchmark.behavior.scene_geometry") or {}
    op_log = aug_meta.get("operation_log") if isinstance(aug_meta, dict) else None
    rewritten_names: set[str] = set()
    relocated_names: set[str] = set()
    generated_names: set[str] = set()
    removed_names: set[str] = set()
    discarded_names: set[str] = set()
    model_changed_names: set[str] = set()
    large_displacements: list[float] = []
    op_counts: dict[str, int] = {}
    if isinstance(op_log, list):
        for op in op_log:
            if not isinstance(op, dict):
                continue
            op_name = str(op.get("op") or "")
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
            name = str(op.get("name") or "")
            if op_name in {"add_large", "add_small"} and name:
                generated_names.add(name)
            if op_name == "rewrite_large" and name:
                rewritten_names.add(name)
                relocated_names.add(name)
                if op.get("old_name") and op.get("old_name") != name:
                    removed_names.add(str(op["old_name"]))
                if op.get("removed_name"):
                    removed_names.add(str(op["removed_name"]))
            if op_name == "relocate_large" and name:
                relocated_names.add(name)
            if op_name == "remove_large" and name:
                removed_names.add(name)
                discarded_names.add(name)
            if op_name == "room_relayout":
                continue
            if op_name in {"rewrite_large", "model_swap"} and name and op.get("old_model") != op.get("model"):
                model_changed_names.add(name)
            old_pos = op.get("old_pos")
            new_pos = op.get("new_pos")
            if op_name in {"rewrite_large", "relocate_large"} and _is_vec(old_pos, 3) and _is_vec(new_pos, 3):
                large_displacements.append(_xy_distance(old_pos, new_pos))
    init_keys = set((scene.get("objects_info", {}).get("init_info", {}) or {}).keys())
    reg_keys = set((scene.get("state", {}).get("registry", {}).get("object_registry", {}) or {}).keys())
    if init_keys != reg_keys:
        errors.append(f"init_registry_key_mismatch init_only={len(init_keys-reg_keys)} registry_only={len(reg_keys-init_keys)}")
    layout = None
    try:
        layout = load_behavior_room_layout_maps(str(scene_model), assets_root=assets, floor=0)
    except Exception as exc:
        errors.append(f"layout_load_failed:{exc}")

    role_counts: dict[str, int] = {}
    room_counts: dict[str, dict[str, int]] = {}
    operation_counts: dict[str, int] = {}
    for name, record in records.items():
        if record.name != name:
            errors.append(f"name_key_mismatch:{name}")
        asset = catalog.get_model(record.category, record.model)
        if asset is None:
            errors.append(f"missing_asset:{name}:{record.category}/{record.model}")
        if len(record.pos) != 3 or len(record.ori) != 4:
            errors.append(f"bad_pose:{name}")
        state = (((scene.get("state") or {}).get("registry") or {}).get("object_registry") or {}).get(name) or {}
        if asset is not None:
            joint_pos = state.get("joint_pos") or []
            joint_vel = state.get("joint_vel") or []
            if name in generated_names and asset.joint_count:
                errors.append(f"generated_articulated_asset_without_joint_defaults:{name}:{record.category}/{record.model}:{asset.joint_count}")
            if name in model_changed_names:
                old_model = _old_model_for_name(op_log, name)
                old_asset = catalog.get_model(record.category, old_model) if old_model else None
                if old_asset is not None and old_asset.joint_count != asset.joint_count:
                    errors.append(
                        f"model_changed_joint_count_mismatch:{name}:{record.category}:{old_model}({old_asset.joint_count})->{record.model}({asset.joint_count})"
                    )
                if len(joint_pos) not in {0, asset.joint_count} or len(joint_vel) not in {0, asset.joint_count}:
                    errors.append(f"model_changed_bad_joint_state_len:{name}:{record.category}/{record.model}:{len(joint_pos)}/{len(joint_vel)}:{asset.joint_count}")
        if layout is not None:
            for room in record.rooms:
                if room not in layout.room_ins_name_to_ins_id:
                    warnings.append(f"unknown_room:{name}:{room}")
        role = getattr(classifier.get(record.category), "role", "unknown") if classifier is not None else "unknown"
        role_counts[role] = role_counts.get(role, 0) + 1
        for room in record.rooms or ("?",):
            row = room_counts.setdefault(room, {"object_count": 0, "large_floor": 0, "support_surface": 0, "small_prop": 0})
            row["object_count"] += 1
            if role in row:
                row[role] += 1

        is_generated_or_rewritten = "_aug_" in name or name in generated_names or name in rewritten_names or name in relocated_names
        if is_generated_or_rewritten:
            for room in record.rooms:
                room_type = room_type_from_instance(room)
                if not _category_allowed_in_room(record.category, room_type, role, rules):
                    errors.append(f"large_layout_room_category_violation:{name}:{record.category}:{room_type}")
        operation = "kept"
        if name in rewritten_names:
            operation = "rewritten"
        elif name in relocated_names:
            operation = "relocated"
        elif "_aug_" in name:
            operation = "generated"
        operation_counts[operation] = operation_counts.get(operation, 0) + 1

    if source_scene_file is not None:
        source = load_scene_json(source_scene_file)
        source_counts = _category_counts(source)
        variant_counts = _category_counts(scene)
        for category in rules.placement_policy.get("structural_integrity_categories") or []:
            source_count = int(source_counts.get(str(category), 0))
            variant_count = int(variant_counts.get(str(category), 0))
            if variant_count < source_count:
                errors.append(f"structural_category_count_decreased:{category}:{source_count}->{variant_count}")

    overlaps = []
    generated_overlaps = []
    rewritten_overlaps = []
    small_overlaps = []
    generated_small_overlaps = []
    if classifier is not None:
        fps = existing_footprints(
            scene,
            catalog,
            classifier,
            include_roles={"large_floor", "support_surface", "fixture"},
        )
        for i, left in enumerate(fps):
            for right in fps[i + 1 :]:
                if left.room and right.room and left.room != right.room:
                    continue
                ratio = aabb_overlap_ratio(left, right)
                if ratio > 0.18:
                    row = {"a": left.name, "b": right.name, "ratio": ratio, "room": left.room}
                    overlaps.append(row)
                    if "_aug_" in left.name or "_aug_" in right.name:
                        generated_overlaps.append(row)
                    if left.name in rewritten_names or right.name in rewritten_names:
                        rewritten_overlaps.append(row)
        if generated_overlaps:
            warnings.append(f"generated_large_overlap_count:{len(generated_overlaps)}")
        if rewritten_overlaps:
            warnings.append(f"rewritten_large_overlap_count:{len(rewritten_overlaps)}")
        small_aabbs = small_object_aabbs_from_scene(scene, catalog, classifier)
        for i, left in enumerate(small_aabbs):
            for right in small_aabbs[i + 1 :]:
                if left.room and right.room and left.room != right.room:
                    continue
                ratio = aabb3d_overlap_ratio(left, right)
                if ratio <= 0.0:
                    continue
                row = {"a": left.name, "b": right.name, "ratio": ratio, "room": left.room}
                small_overlaps.append(row)
                if left.name in generated_names or right.name in generated_names:
                    generated_small_overlaps.append(row)
        if generated_small_overlaps:
            warnings.append(f"generated_small_overlap_count:{len(generated_small_overlaps)}")

    large_layout_delta = {
        "added_large": op_counts.get("add_large", 0),
        "removed_large": len(removed_names),
        "discarded_large": len(discarded_names),
        "rewritten_large": op_counts.get("rewrite_large", 0),
        "relocated_large": op_counts.get("rewrite_large", 0) + op_counts.get("relocate_large", 0),
        "room_relayout_rooms": op_counts.get("room_relayout", 0),
        "placed_groups": int(aug_meta.get("operations", {}).get("placed_groups", 0)) if isinstance(aug_meta.get("operations"), dict) else 0,
        "failed_groups": int(aug_meta.get("operations", {}).get("failed_groups", 0)) if isinstance(aug_meta.get("operations"), dict) else 0,
        "group_objects_relocated": int(aug_meta.get("operations", {}).get("group_objects_relocated", 0)) if isinstance(aug_meta.get("operations"), dict) else 0,
        "category_swaps": int(aug_meta.get("operations", {}).get("category_swaps", 0)) if isinstance(aug_meta.get("operations"), dict) else 0,
        "avg_large_displacement_m": round(sum(large_displacements) / len(large_displacements), 4) if large_displacements else 0.0,
        "room_category_violations": sum(1 for item in errors if item.startswith("large_layout_room_category_violation:")),
    }

    return {
        "scene_model": str(scene_model),
        "scene_file": str(scene_file),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings[:200],
        "object_count": len(records),
        "role_counts": dict(sorted(role_counts.items())),
        "room_counts": room_counts,
        "large_overlaps": overlaps[:100],
        "generated_large_overlaps": generated_overlaps[:100],
        "rewritten_large_overlaps": rewritten_overlaps[:100],
        "small_overlaps": small_overlaps[:100],
        "generated_small_overlaps": generated_small_overlaps[:100],
        "operation_counts": dict(sorted(operation_counts.items())),
        "operation_log_counts": dict(sorted(op_counts.items())),
        "large_layout_delta": large_layout_delta,
        "metadata_scene_augmentation": (scene.get("metadata") or {}).get("scoper.benchmark.behavior.scene_geometry"),
        "source_scene_file": str(source_scene_file) if source_scene_file is not None else None,
    }


def _category_allowed_in_room(category: str, room_type: str, role: str, rules: AugmentationRules) -> bool:
    if role in {"large_floor", "support_surface"}:
        mapping = rules.placement_policy.get("allowed_large_by_room") or {}
        if room_type in mapping:
            return category in {str(item) for item in mapping.get(room_type) or []}
    if role == "small_prop":
        mapping = rules.receptacle_policy.get("allowed_small_by_room") or {}
        if room_type in mapping:
            return category in {str(item) for item in mapping.get(room_type) or []}
    return True


def _category_counts(scene: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in parse_scene_objects(scene).values():
        counts[record.category] = counts.get(record.category, 0) + 1
    return counts


def _is_vec(value: Any, length: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == length and all(isinstance(item, (int, float)) for item in value)


def _xy_distance(left: Any, right: Any) -> float:
    return ((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2) ** 0.5


def _old_model_for_name(op_log: Any, name: str) -> str | None:
    if not isinstance(op_log, list):
        return None
    for op in op_log:
        if isinstance(op, dict) and op.get("name") == name and op.get("old_model"):
            return str(op["old_model"])
    return None


def write_validation_report(path: str | Path, report: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def og_load_reset_smoke(*, scene_model: str, scene_file: str | Path, width: int = 320, height: int = 240) -> dict[str, Any]:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from scoper.benchmark.behavior.execution_core.env import EgoCameraConfig, EgoEnvConfig, build_robot_free_env_config, make_omnigibson_environment

    config = build_robot_free_env_config(
        EgoEnvConfig(
            scene_model=str(scene_model),
            scene_file=str(scene_file),
            load_object_categories=None,
            viewer_width=int(width),
            viewer_height=int(height),
            camera=EgoCameraConfig(image_width=int(width), image_height=int(height), modalities=("rgb",)),
        )
    )
    env = None
    try:
        env = make_omnigibson_environment(config)
        obs, _ = env.reset()
        count = len(getattr(getattr(env, "scene", None), "objects", []) or [])
        return {"status": "passed", "loaded_object_count": count, "obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else []}
    except Exception as exc:
        return {"status": "failed", "error": repr(exc)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


__all__ = ["og_load_reset_smoke", "validate_scene_variant", "write_validation_report"]
