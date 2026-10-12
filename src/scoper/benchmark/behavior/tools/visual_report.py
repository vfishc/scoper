"""Stage-0 visual migration dashboard for scoper.benchmark.behavior.

This command is intentionally read-only with respect to benchmark behavior: it scans the
current repository, the eb_procthor target templates, and any previous stage outputs, then
writes a unified manifest + HTML entry point under outputs/eb_behavior_visual_migration.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from scoper.benchmark.behavior.task_generation.validator_outputs import (
    family_coverage,
    split_records,
    stage4_validation_records,
    stage5_validation_records,
    stage6_validation_records,
    summarize_validation_records,
)
from scoper.benchmark.behavior.task_generation.template_activation import TEMPLATE_DISABLED_FAILURE_CODE
from scoper.benchmark.behavior.paths import data_dir, repo_root, resolve_repo_path


CAPABILITY_COLUMNS = (
    "template_schema",
    "scene_profile",
    "scene_match",
    "role_binding",
    "task_generation",
    "init_reset",
    "scene_mutation",
    "oracle_plan",
    "tool_execution",
    "render_video",
    "goal_check",
    "metrics",
    "vlm_ready",
    "visual_gate",
)

STATUS_ORDER = ("missing", "partial", "implemented", "visually_verified", "blocked")

from scoper.benchmark.procthor.paths import data_dir as procthor_data_dir

DEFAULT_PROCTHOR_TEMPLATE_DIR = procthor_data_dir() / "task_templates"
TASK_TEMPLATE_SCHEMA_VERSION = "stage1.0"

# Conservative mapping from target eb_procthor templates to currently available
# scoper.benchmark.behavior YAML templates. Exact schema parity still belongs to Stage 1.
TARGET_TO_EB_TEMPLATE = {
    "static_place_001": "static_place_001",
    "hidden_precondition_001": "hidden_precondition_001",
    "hidden_precondition_place_001": "hidden_precondition_001",
    "heat_then_place_001": "prepare_heat_place_001",
    "put_away_food_to_cold_storage_001": "put_away_by_category_001",
    "put_away_dishware_to_storage_001": "put_away_by_category_001",
    "put_away_cleaning_supplies_to_storage_001": "put_away_by_category_001",
}


@dataclass
class TemplateRecord:
    path: Path
    filename: str
    template_id: str
    category: str
    description: str = ""
    roles: list[str] = field(default_factory=list)
    goal_predicates: list[str] = field(default_factory=list)
    source: str = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatrixRow:
    template: TemplateRecord
    matched_eb_template_id: str | None
    statuses: dict[str, str]
    notes: list[str] = field(default_factory=list)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate scoper.benchmark.behavior visual migration dashboards.")
    parser.add_argument(
        "--stage",
        default="status",
        choices=("status", "template_catalog", "scene_profile_match", "task_generation", "task_generation_reset", "reset_mutation", "stage6_execution", "stage7_goal_metrics", "stage8_oracle_teachers", "stage9_render_video"),
        help="Report stage to generate.",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/eb_behavior_visual_migration",
        help="Output root, absolute or relative to the repository root.",
    )
    parser.add_argument("--run-id", default=None, help="Stable run id. Defaults to timestamp_stage0_status.")
    parser.add_argument(
        "--procthor-template-dir",
        default=str(DEFAULT_PROCTHOR_TEMPLATE_DIR),
        help="eb_procthor task template directory.",
    )
    parser.add_argument(
        "--scene-models",
        default="Rs_int,Beechwood_0_int,Wainscott_0_int,house_single_floor",
        help="Comma-separated BEHAVIOR scene models for Stage 2/3 scene profiling and matching.",
    )
    parser.add_argument(
        "--min-pass-ratio",
        type=float,
        default=0.75,
        help="Minimum Stage 5 reset/mutation pass ratio for bulk acceptance.",
    )
    parser.add_argument(
        "--min-execution-pass-ratio",
        type=float,
        default=0.50,
        help="Minimum Stage 6 execution pass ratio among attempted executable tasks.",
    )
    parser.add_argument(
        "--max-high-level-steps",
        type=int,
        default=100,
        help="Maximum high-level public tool calls per Stage 6 episode.",
    )
    parser.add_argument(
        "--skip-reachability",
        action="store_true",
        help="Skip Stage 5 symbolic navigation reachability probes.",
    )
    parser.add_argument(
        "--stage6-run",
        default="outputs/eb_behavior_visual_migration/stage6_execution/stage6_execution_acceptance_final",
        help="Stage 6 output directory consumed by stage7_goal_metrics.",
    )
    parser.add_argument("--render", action="store_true", help="For Stage 9, attempt real RGB rendering.")
    parser.add_argument(
        "--require-render",
        action="store_true",
        help="For Stage 9, fail the report if real RGB rendering cannot complete.",
    )
    parser.add_argument("--video-fps", type=int, default=10, help="For Stage 9, rendered MP4 FPS.")
    parser.add_argument("--renders-per-frame", type=int, default=2, help="For Stage 9, OmniGibson render steps per captured frame.")
    parser.add_argument("--render-nav-max-frames", type=int, default=0, help="For Stage 9, max rendered navigation frames per tool; 0 keeps all symbolic poses.")
    parser.add_argument("--navigation-speed-mps", type=float, default=0.4, help="For Stage 9, resample navigation frames to approximate this ego linear speed.")
    parser.add_argument("--navigation-turn-speed-radps", type=float, default=0.8, help="For Stage 9, resample in-place turns to approximate this ego angular speed.")
    parser.add_argument("--all-templates", action="store_true", help="For Stage 9, run one selected task per generated template.")
    args = parser.parse_args(argv)

    if args.stage == "stage9_render_video":
        from scoper.benchmark.behavior.tools.stage9_render_video_report import build_stage9_render_video_report

        report = build_stage9_render_video_report(
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
            procthor_template_dir=Path(args.procthor_template_dir),
            scene_models=[item.strip() for item in args.scene_models.split(",") if item.strip()],
            max_high_level_steps=args.max_high_level_steps,
            render=args.render,
            require_render=args.require_render,
            video_fps=args.video_fps,
            renders_per_frame=args.renders_per_frame,
            render_nav_max_frames=args.render_nav_max_frames,
            navigation_speed_mps=args.navigation_speed_mps,
            navigation_turn_speed_radps=args.navigation_turn_speed_radps,
            all_templates=args.all_templates,
        )
        label = "Stage-9 render/video report"
    elif args.stage == "stage8_oracle_teachers":
        from scoper.benchmark.behavior.tools.stage8_oracle_teacher_report import build_stage8_oracle_teacher_report

        report = build_stage8_oracle_teacher_report(
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
            procthor_template_dir=Path(args.procthor_template_dir),
            scene_models=[item.strip() for item in args.scene_models.split(",") if item.strip()],
            min_stage5_pass_ratio=args.min_pass_ratio,
            min_execution_pass_ratio=args.min_execution_pass_ratio,
            run_reachability=not args.skip_reachability,
            max_high_level_steps=args.max_high_level_steps,
        )
        label = "Stage-8 oracle teacher report"
    elif args.stage == "stage7_goal_metrics":
        from scoper.benchmark.behavior.tools.stage7_goal_metrics_report import build_stage7_goal_metrics_report

        report = build_stage7_goal_metrics_report(
            stage6_run=resolve_repo_path(args.stage6_run),
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
        )
        label = "Stage-7 goal/metrics report"
    elif args.stage == "stage6_execution":
        report = build_stage6_execution_report(
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
            procthor_template_dir=Path(args.procthor_template_dir),
            scene_models=[item.strip() for item in args.scene_models.split(",") if item.strip()],
            min_stage5_pass_ratio=args.min_pass_ratio,
            min_execution_pass_ratio=args.min_execution_pass_ratio,
            run_reachability=not args.skip_reachability,
            max_high_level_steps=args.max_high_level_steps,
        )
        label = "Stage-6 execution report"
    elif args.stage == "reset_mutation":
        report = build_stage5_reset_mutation_report(
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
            procthor_template_dir=Path(args.procthor_template_dir),
            scene_models=[item.strip() for item in args.scene_models.split(",") if item.strip()],
            min_pass_ratio=args.min_pass_ratio,
            run_reachability=not args.skip_reachability,
        )
        label = "Stage-5 reset/mutation report"
    elif args.stage == "task_generation_reset":
        report = build_stage4_5_task_generation_reset_report(
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
            procthor_template_dir=Path(args.procthor_template_dir),
            scene_models=[item.strip() for item in args.scene_models.split(",") if item.strip()],
        )
        label = "Stage-4/5 task generation/reset report"
    elif args.stage == "task_generation":
        report = build_stage4_task_generation_report(
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
            procthor_template_dir=Path(args.procthor_template_dir),
            scene_models=[item.strip() for item in args.scene_models.split(",") if item.strip()],
        )
        label = "Stage-4 task generation report"
    elif args.stage == "scene_profile_match":
        report = build_stage2_3_scene_profile_match_report(
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
            procthor_template_dir=Path(args.procthor_template_dir),
            scene_models=[item.strip() for item in args.scene_models.split(",") if item.strip()],
        )
        label = "Stage-2/3 scene profile/match report"
    elif args.stage == "template_catalog":
        report = build_stage1_template_catalog_report(
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
            procthor_template_dir=Path(args.procthor_template_dir),
        )
        label = "Stage-1 template catalog report"
    else:
        report = build_stage0_report(
            out_root=resolve_repo_path(args.out_root),
            run_id=args.run_id,
            procthor_template_dir=Path(args.procthor_template_dir),
        )
        label = "Stage-0 visual report"
    print("{} generated".format(label))
    print("  run_id: {}".format(report["run_id"]))
    print("  output: {}".format(report["output_dir"]))
    print("  index:  {}".format(report["artifacts"]["index"]))


def build_stage0_report(
    *,
    out_root: Path,
    run_id: str | None = None,
    procthor_template_dir: Path = DEFAULT_PROCTHOR_TEMPLATE_DIR,
) -> dict[str, Any]:
    """Generate the Stage-0 dashboard and return its manifest dict."""

    run_id = run_id or "{}_stage0_status".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage0" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    procthor_templates = load_template_records(procthor_template_dir, source="eb_procthor")
    eb_templates = load_template_records(data_dir() / "task_templates", source="scoper.benchmark.behavior")
    repo_status = collect_repo_status()
    matrix = build_capability_matrix(procthor_templates, eb_templates, repo_status)

    template_scope_path = output_dir / "template_scope.html"
    capability_matrix_path = output_dir / "capability_matrix.html"
    repo_status_path = output_dir / "repo_status.html"
    index_path = output_dir / "index.html"
    manifest_path = output_dir / "manifest.json"

    write_template_scope(template_scope_path, procthor_templates, eb_templates, matrix)
    write_capability_matrix(capability_matrix_path, matrix)
    write_repo_status(repo_status_path, repo_status)

    summary = build_summary(procthor_templates, eb_templates, matrix, repo_status)
    manifest = {
        "run_id": run_id,
        "stage": "stage0_visual_status",
        "status": "passed",
        "visual_gate": "generated_pending_human_review",
        "template_target_count": len(procthor_templates),
        "eb_behavior_template_count": len(eb_templates),
        "status_counts": summary["status_counts"],
        "artifacts": {
            "index": "index.html",
            "capability_matrix": "capability_matrix.html",
            "template_scope": "template_scope.html",
            "repo_status": "repo_status.html",
        },
        "failure_codes": [],
        "notes": [
            "Stage 0 is a repository visibility gate; per-template visual verification is deferred.",
            "Previous stage outputs are linked as existing evidence but not treated as visually_verified.",
        ],
    }

    write_index(index_path, summary, manifest)
    write_json(manifest_path, manifest)
    write_json(output_dir / "logs" / "stage0_summary.json", summary)
    write_json(output_dir / "logs" / "capability_matrix.json", serialize_matrix(matrix))

    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {
        key: str(output_dir / value) for key, value in manifest["artifacts"].items()
    }
    return manifest


def build_stage1_template_catalog_report(
    *,
    out_root: Path,
    run_id: str | None = None,
    procthor_template_dir: Path = DEFAULT_PROCTHOR_TEMPLATE_DIR,
) -> dict[str, Any]:
    """Generate the Stage-1 template catalog/schema parity report."""

    run_id = run_id or "{}_stage1_template_catalog".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage1_template_catalog" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    procthor_templates = load_template_records(procthor_template_dir, source="eb_procthor")
    eb_templates = load_template_records(data_dir() / "task_templates", source="scoper.benchmark.behavior")
    repo_status = collect_repo_status()
    matrix = build_capability_matrix(procthor_templates, eb_templates, repo_status)
    entries = build_template_catalog_entries(procthor_templates, eb_templates, matrix)
    family_overview = build_template_family_overview(entries)
    dependency_graph = build_template_dependency_graph(entries)
    template_schema = build_task_template_schema(entries)
    summary = build_stage1_summary(entries, eb_templates, family_overview)

    write_template_catalog(output_dir / "template_catalog.html", entries)
    write_template_family_overview(output_dir / "template_family_overview.html", family_overview)
    write_template_dependency_graph(output_dir / "template_dependency_graph.html", dependency_graph)
    write_task_template_schema(output_dir / "task_template_schema.html", template_schema)

    manifest = {
        "run_id": run_id,
        "stage": "stage1_template_catalog",
        "status": "passed",
        "visual_gate": "generated_pending_human_review",
        "schema_version": template_schema["schema_version"],
        "schema_top_level_field_count": len(template_schema["top_level_fields"]),
        "template_target_count": len(entries),
        "eb_behavior_template_count": len(eb_templates),
        "mapped_target_count": summary["mapped_target_count"],
        "family_count": len(family_overview),
        "artifacts": {
            "index": "index.html",
            "task_template_schema": "task_template_schema.html",
            "template_catalog": "template_catalog.html",
            "template_family_overview": "template_family_overview.html",
            "template_dependency_graph": "template_dependency_graph.html",
            "task_template_schema_json": "logs/task_template_schema.json",
            "template_catalog_json": "logs/template_catalog.json",
            "template_family_overview_json": "logs/template_family_overview.json",
            "template_dependency_graph_json": "logs/template_dependency_graph.json",
        },
        "failure_codes": [],
        "notes": [
            "Stage 1 catalogs target template dependencies; it does not claim runtime task generation parity.",
            "Manual symbolic smoke remains a fallback only; scene-profile-driven binding starts in Stage 2/3/4.",
        ],
    }

    write_stage1_index(output_dir / "index.html", summary, manifest)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "logs" / "stage1_summary.json", summary)
    write_json(output_dir / "logs" / "task_template_schema.json", template_schema)
    write_json(output_dir / "logs" / "template_catalog.json", entries)
    write_json(output_dir / "logs" / "template_family_overview.json", family_overview)
    write_json(output_dir / "logs" / "template_dependency_graph.json", dependency_graph)

    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {
        key: str(output_dir / value) for key, value in manifest["artifacts"].items()
    }
    return manifest


def build_stage2_3_scene_profile_match_report(
    *,
    out_root: Path,
    run_id: str | None = None,
    procthor_template_dir: Path = DEFAULT_PROCTHOR_TEMPLATE_DIR,
    scene_models: list[str] | None = None,
) -> dict[str, Any]:
    """Generate the Stage 2/3 BEHAVIOR SceneProfile + no-reset matching report."""

    from scoper.benchmark.behavior.scene.profiler import profile_scenes
    from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates, match_scene_to_templates

    scene_models = scene_models or ["Rs_int", "Beechwood_0_int", "Wainscott_0_int", "house_single_floor"]
    run_id = run_id or "{}_stage2_3_scene_profile_match".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage2_3_scene_profile_match" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    profiles = profile_scenes(tuple(scene_models))
    templates = load_procthor_templates(procthor_template_dir)
    match_records = []
    for profile in profiles:
        match_records.extend(match_scene_to_templates(profile, templates, max_bindings_per_template=4))

    summary = build_stage2_3_summary(profiles, templates, match_records)
    write_stage2_3_index(output_dir / "index.html", summary)
    write_scene_profiles_html(output_dir / "scene_profiles.html", profiles)
    write_scene_template_matrix_html(output_dir / "scene_template_matrix.html", match_records)

    write_json(output_dir / "logs" / "scene_profiles.json", profiles)
    write_json(output_dir / "logs" / "scene_template_matches.json", match_records)
    write_json(output_dir / "logs" / "stage2_3_summary.json", summary)

    manifest = {
        "run_id": run_id,
        "stage": "stage2_3_scene_profile_match",
        "status": "passed" if summary["profile_validation_passed"] and summary["template_target_count"] > 0 else "failed",
        "visual_gate": "generated_pending_human_review",
        "scene_count": summary["scene_count"],
        "template_target_count": summary["template_target_count"],
        "scene_template_pair_count": summary["scene_template_pair_count"],
        "matched_pair_count": summary["matched_pair_count"],
        "rejected_pair_count": summary["rejected_pair_count"],
        "match_status_counts": summary["match_status_counts"],
        "matched_profile_only_pair_count": summary["matched_profile_only_pair_count"],
        "matched_requires_spawn_pair_count": summary["matched_requires_spawn_pair_count"],
        "matched_requires_runtime_validation_pair_count": summary[
            "matched_requires_runtime_validation_pair_count"
        ],
        "artifacts": {
            "index": "index.html",
            "scene_profiles": "scene_profiles.html",
            "scene_template_matrix": "scene_template_matrix.html",
            "scene_profiles_json": "logs/scene_profiles.json",
            "scene_template_matches_json": "logs/scene_template_matches.json",
            "summary_json": "logs/stage2_3_summary.json",
        },
        "failure_codes": [],
        "notes": [
            "Stage 2 profiles are file-based BEHAVIOR SceneProfiles with explicit unknowns.",
            "Stage 3 matching is no-reset; spawn, visibility, reachable-map, and relocation are recorded as runtime requirements.",
            "Matched does not mean reset/render/oracle validation has passed.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {key: str(output_dir / value) for key, value in manifest["artifacts"].items()}
    return manifest


def build_stage2_3_summary(
    profiles: list[Mapping[str, Any]],
    templates: list[Mapping[str, Any]],
    match_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    runtime_counter = Counter()
    for record in match_records:
        runtime_counter.update(record.get("runtime_capabilities_required") or [])
    status_counter = Counter(str(record.get("status") or "unknown") for record in match_records)
    rejection_counter = Counter(
        reason
        for record in match_records
        for reason in record.get("rejection_reasons") or []
    )
    matched_by_scene = Counter(
        record["scene_model"]
        for record in match_records
        if is_profile_match_record(record)
    )
    spawn_binding_count = sum(
        1
        for record in match_records
        for binding in record.get("bindings") or []
        if "spawn_object" in set(binding.get("required_init_mutations") or [])
    )
    spawn_required_pair_count = sum(
        1
        for record in match_records
        if is_profile_match_record(record)
        and (
            bool(record.get("requires_spawn"))
            or "spawn_object" in set(record.get("required_init_mutations") or [])
            or "spawn_placement" in set(record.get("runtime_capabilities_required") or [])
        )
    )
    runtime_validation_pair_count = sum(
        1
        for record in match_records
        if is_profile_match_record(record) and bool(record.get("requires_runtime_validation"))
    )
    profile_validation_passed = all(
        profile.get("source") == "behavior"
        and profile.get("house_id")
        and isinstance(profile.get("affordance_index"), Mapping)
        and isinstance(profile.get("object_type_index"), Mapping)
        and isinstance(profile.get("spawnable_catalog"), list)
        for profile in profiles
    )
    matched_pair_count = sum(1 for record in match_records if is_profile_match_record(record))
    rejected_pair_count = len(match_records) - matched_pair_count
    return {
        "profile_validation_passed": profile_validation_passed,
        "scene_count": len(profiles),
        "scene_models": [str(profile.get("scene_model")) for profile in profiles],
        "template_target_count": len(templates),
        "scene_template_pair_count": len(match_records),
        "matched_pair_count": matched_pair_count,
        "rejected_pair_count": rejected_pair_count,
        "match_status_counts": dict(sorted(status_counter.items())),
        "matched_profile_only_pair_count": int(status_counter.get("matched_profile_only", 0)),
        "matched_requires_spawn_pair_count": int(status_counter.get("matched_requires_spawn", 0)),
        "matched_requires_runtime_validation_pair_count": int(
            status_counter.get("matched_requires_runtime_validation", 0)
        ),
        "spawn_required_pair_count": spawn_required_pair_count,
        "runtime_validation_pair_count": runtime_validation_pair_count,
        "matched_templates_by_scene": dict(sorted(matched_by_scene.items())),
        "runtime_capability_counts": dict(sorted(runtime_counter.items())),
        "rejection_reason_counts": dict(sorted(rejection_counter.items())),
        "spawn_binding_count": spawn_binding_count,
        "profile_object_counts": {
            str(profile.get("scene_model")): int(profile.get("object_count") or 0)
            for profile in profiles
        },
        "profile_spawnable_counts": {
            str(profile.get("scene_model")): len(profile.get("spawnable_catalog") or [])
            for profile in profiles
        },
        "reachable_statuses": {
            str(profile.get("scene_model")): str((profile.get("reachable") or {}).get("status"))
            for profile in profiles
        },
    }


def build_stage4_task_generation_report(
    *,
    out_root: Path,
    run_id: str | None = None,
    procthor_template_dir: Path = DEFAULT_PROCTHOR_TEMPLATE_DIR,
    scene_models: list[str] | None = None,
) -> dict[str, Any]:
    """Generate profile-driven Stage 4 task specs without reset/oracle execution."""

    from scoper.benchmark.behavior.scene.profiler import profile_scenes
    from scoper.benchmark.behavior.task_generation.profile_task_generator import generate_stage4_tasks
    from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates, match_scene_to_templates
    from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec

    scene_models = scene_models or ["Rs_int", "Beechwood_0_int", "Wainscott_0_int", "house_single_floor"]
    run_id = run_id or "{}_stage4_task_generation".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage4_task_generation" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    profiles = profile_scenes(tuple(scene_models))
    templates = load_procthor_templates(procthor_template_dir)
    match_records = []
    for profile in profiles:
        match_records.extend(match_scene_to_templates(profile, templates, max_bindings_per_template=2))
    generation = generate_stage4_tasks(
        profiles=profiles,
        templates=templates,
        match_records=match_records,
        max_tasks_per_template_scene=1,
        seed=0,
    )
    structure_results = []
    for task in generation["tasks"]:
        result = {
            "task_id": task.get("id"),
            "structure_ok": False,
            "failure_code": None,
            "failure_reason": None,
        }
        try:
            validate_behavior_task_spec(task)
            result["structure_ok"] = True
        except Exception as exc:
            result["failure_code"] = "TASK_SPEC_VALIDATION_FAILED"
            result["failure_reason"] = str(exc)
        structure_results.append(result)

    summary = build_stage4_summary(generation, structure_results)
    validation_records = stage4_validation_records(generation, structure_results)
    validation_summary = summarize_validation_records(validation_records)
    coverage = family_coverage(generation["tasks"])
    summary["validator_records"] = {
        "accepted_total": validation_summary["accepted_total"],
        "rejected_total": validation_summary["rejected_total"],
        "buckets": validation_summary["buckets"],
    }
    summary["category_summary"] = validation_summary["by_category"]
    summary["family_coverage"] = coverage

    write_stage4_index(output_dir / "index.html", summary)
    write_stage4_task_gallery(output_dir / "task_generation_gallery.html", generation["tasks"], structure_results)
    write_stage4_rejected_gallery(output_dir / "rejected_task_gallery.html", generation["rejected"])

    write_json(output_dir / "logs" / "task_bank.json", generation["tasks"])
    write_json(output_dir / "logs" / "rejected_tasks.json", generation["rejected"])
    write_json(output_dir / "logs" / "structure_validation.json", structure_results)
    write_json(output_dir / "logs" / "stage4_summary.json", summary)
    write_json(output_dir / "logs" / "category_summary.json", validation_summary)
    write_json(output_dir / "logs" / "family_coverage.json", coverage)
    write_jsonl(output_dir / "logs" / "accepted_tasks.jsonl", split_records(validation_records, "accepted"))
    write_jsonl(output_dir / "logs" / "rejected_tasks.jsonl", split_records(validation_records, "rejected"))

    blocking_rejections = blocking_stage4_rejection_codes(summary.get("rejection_codes") or {})
    passed = (
        summary["generated_total"] > 0
        and summary["structure_passed"] == summary["generated_total"]
        and not blocking_rejections
    )
    manifest = {
        "run_id": run_id,
        "stage": "stage4_task_generation",
        "status": "passed" if passed else "failed",
        "visual_gate": "generated_pending_human_review",
        "generated_total": summary["generated_total"],
        "rejected_total": summary["rejected_total"],
        "structure_passed": summary["structure_passed"],
        "artifacts": {
            "index": "index.html",
            "task_generation_gallery": "task_generation_gallery.html",
            "rejected_task_gallery": "rejected_task_gallery.html",
            "task_bank_json": "logs/task_bank.json",
            "rejected_tasks_json": "logs/rejected_tasks.json",
            "accepted_tasks_jsonl": "logs/accepted_tasks.jsonl",
            "rejected_tasks_jsonl": "logs/rejected_tasks.jsonl",
            "category_summary_json": "logs/category_summary.json",
            "family_coverage_json": "logs/family_coverage.json",
            "structure_validation_json": "logs/structure_validation.json",
            "summary_json": "logs/stage4_summary.json",
        },
        "blocking_rejection_codes": blocking_rejections,
        "failure_codes": [] if passed else ["STAGE4_VALIDATION_FAILED"],
        "notes": [
            "Stage 4 materializes task specs from Stage 3 match records.",
            "This report does not execute reset, mutation, oracle, or rendering validation.",
            "Unsupported goal predicates are rejected with stable failure codes instead of being silently skipped.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {key: str(output_dir / value) for key, value in manifest["artifacts"].items()}
    return manifest


def blocking_stage4_rejection_codes(rejection_codes: Mapping[str, Any]) -> list[str]:
    allowed = {
        "NO_STAGE3_MATCH_BINDING",
        "TEMPLATE_NOT_IN_STAGE4_SCOPE",
        "TEMPLATE_NOT_IN_STAGE4_5_GOLD_SCOPE",
        TEMPLATE_DISABLED_FAILURE_CODE,
    }
    return sorted(str(code) for code in rejection_codes if str(code) not in allowed)


def build_stage4_5_task_generation_reset_report(
    *,
    out_root: Path,
    run_id: str | None = None,
    procthor_template_dir: Path = DEFAULT_PROCTHOR_TEMPLATE_DIR,
    scene_models: list[str] | None = None,
) -> dict[str, Any]:
    """Generate Stage 4 task specs and Stage 5 symbolic reset/oracle validation."""

    from scoper.benchmark.behavior.scene.profiler import profile_scenes
    from scoper.benchmark.behavior.task_generation.profile_task_generator import (
        STAGE4_5_GOLD_TEMPLATE_IDS,
        generate_stage4_tasks,
        validate_reset_and_oracle,
    )
    from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates, match_scene_to_templates

    scene_models = scene_models or ["Rs_int", "Beechwood_0_int", "Wainscott_0_int", "house_single_floor"]
    run_id = run_id or "{}_stage4_5_task_generation_reset".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage4_5_task_generation_reset" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    profiles = profile_scenes(tuple(scene_models))
    templates = load_procthor_templates(procthor_template_dir)
    match_records = []
    for profile in profiles:
        match_records.extend(match_scene_to_templates(profile, templates, max_bindings_per_template=2))
    generation = generate_stage4_tasks(
        profiles=profiles,
        templates=templates,
        match_records=match_records,
        template_ids=STAGE4_5_GOLD_TEMPLATE_IDS,
        out_of_scope_failure_code="TEMPLATE_NOT_IN_STAGE4_5_GOLD_SCOPE",
        max_tasks_per_template_scene=1,
        seed=0,
    )
    validation_results = [
        validate_reset_and_oracle(task, rng_seed=0, check_oracle=True)
        for task in generation["tasks"]
    ]
    summary = build_stage4_5_summary(generation, validation_results)

    write_stage4_5_index(output_dir / "index.html", summary)
    write_stage4_task_gallery(output_dir / "task_generation_gallery.html", generation["tasks"], validation_results)
    write_stage4_rejected_gallery(output_dir / "rejected_task_gallery.html", generation["rejected"])
    write_stage5_reset_validation(output_dir / "reset_validation.html", validation_results)

    write_json(output_dir / "logs" / "task_bank.json", generation["tasks"])
    write_json(output_dir / "logs" / "rejected_tasks.json", generation["rejected"])
    write_json(output_dir / "logs" / "reset_validation.json", validation_results)
    write_json(output_dir / "logs" / "stage4_5_summary.json", summary)

    passed = (
        summary["generated_total"] > 0
        and summary["structure_passed"] == summary["generated_total"]
        and summary["reset_passed"] == summary["generated_total"]
        and summary["init_passed"] == summary["generated_total"]
        and summary["oracle_passed"] == summary["generated_total"]
    )
    manifest = {
        "run_id": run_id,
        "stage": "stage4_5_task_generation_reset",
        "status": "passed" if passed else "failed",
        "visual_gate": "generated_pending_human_review",
        "generated_total": summary["generated_total"],
        "rejected_total": summary["rejected_total"],
        "oracle_passed": summary["oracle_passed"],
        "artifacts": {
            "index": "index.html",
            "task_generation_gallery": "task_generation_gallery.html",
            "rejected_task_gallery": "rejected_task_gallery.html",
            "reset_validation": "reset_validation.html",
            "task_bank_json": "logs/task_bank.json",
            "rejected_tasks_json": "logs/rejected_tasks.json",
            "reset_validation_json": "logs/reset_validation.json",
            "summary_json": "logs/stage4_5_summary.json",
        },
        "failure_codes": [] if passed else ["STAGE4_5_VALIDATION_FAILED"],
        "notes": [
            "Stage 4/5 uses a narrow gold subset of profile-driven templates.",
            "Reset validation is symbolic and does not yet launch OmniGibson.",
            "Templates outside the gold subset are rejected with TEMPLATE_NOT_IN_STAGE4_5_GOLD_SCOPE, not silently ignored.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {key: str(output_dir / value) for key, value in manifest["artifacts"].items()}
    return manifest


def build_stage5_reset_mutation_report(
    *,
    out_root: Path,
    run_id: str | None = None,
    procthor_template_dir: Path = DEFAULT_PROCTHOR_TEMPLATE_DIR,
    scene_models: list[str] | None = None,
    min_pass_ratio: float = 0.75,
    run_reachability: bool = True,
) -> dict[str, Any]:
    """Generate Stage 5 reset/mutation validation artifacts for the full Stage 4 task bank."""

    from scoper.benchmark.behavior.scene.profiler import profile_scenes
    from scoper.benchmark.behavior.task_generation.profile_task_generator import generate_stage4_tasks
    from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates, match_scene_to_templates
    from scoper.benchmark.behavior.task_generation.stage5_validator import summarize_stage5_results, validate_stage5_reset_mutation

    scene_models = scene_models or ["Rs_int", "Beechwood_0_int", "Wainscott_0_int", "house_single_floor"]
    run_id = run_id or "{}_stage5_reset_mutation".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage5_reset_mutation" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    profiles = profile_scenes(tuple(scene_models))
    templates = load_procthor_templates(procthor_template_dir)
    match_records = []
    for profile in profiles:
        match_records.extend(match_scene_to_templates(profile, templates, max_bindings_per_template=2))
    generation = generate_stage4_tasks(
        profiles=profiles,
        templates=templates,
        match_records=match_records,
        max_tasks_per_template_scene=1,
        seed=0,
    )
    validation_results = [
        validate_stage5_reset_mutation(task, rng_seed=0, run_reachability=run_reachability)
        for task in generation["tasks"]
    ]
    summary = summarize_stage5_results(generation, validation_results)
    validation_records = stage5_validation_records(generation, validation_results)
    validation_summary = summarize_validation_records(validation_records)
    summary["validator_records"] = {
        "accepted_total": validation_summary["accepted_total"],
        "rejected_total": validation_summary["rejected_total"],
        "buckets": validation_summary["buckets"],
    }
    summary["category_summary"] = validation_summary["by_category"]

    write_stage5_reset_mutation_index(output_dir / "index.html", summary)
    write_stage4_task_gallery(output_dir / "task_generation_gallery.html", generation["tasks"], validation_results)
    write_stage4_rejected_gallery(output_dir / "rejected_task_gallery.html", generation["rejected"])
    write_stage5_validation_table(output_dir / "stage5_validation.html", validation_results)
    write_stage5_reset_before_after(output_dir / "reset_before_after.html", validation_results)
    write_stage5_reset_before_after_png(output_dir / "reset_before_after.png", validation_results)
    write_stage5_state_predicate_check(output_dir / "state_predicate_check.html", validation_results)
    write_stage5_mutation_diff(output_dir / "mutation_diff.html", validation_results)
    write_stage5_mutation_replay(output_dir / "mutation_replay.html", validation_results)

    write_json(output_dir / "logs" / "task_bank.json", generation["tasks"])
    write_json(output_dir / "logs" / "rejected_tasks.json", generation["rejected"])
    write_json(output_dir / "logs" / "stage5_validation.json", validation_results)
    write_json(output_dir / "logs" / "stage5_summary.json", summary)
    write_json(output_dir / "logs" / "category_summary.json", validation_summary)
    write_jsonl(output_dir / "logs" / "accepted_tasks.jsonl", split_records(validation_records, "accepted"))
    write_jsonl(output_dir / "logs" / "rejected_tasks.jsonl", split_records(validation_records, "rejected"))

    pass_ratio = float(summary.get("reset_mutation_pass_ratio") or 0.0)
    failure_codes_stable = all(code for code in summary.get("failure_codes", {}))
    passed = summary["validated_total"] > 0 and pass_ratio >= float(min_pass_ratio) and failure_codes_stable
    manifest = {
        "run_id": run_id,
        "stage": "stage5_reset_mutation",
        "status": "passed" if passed else "failed",
        "visual_gate": "reset_mutation_validated_pending_human_review" if passed else "reset_mutation_failed",
        "generated_total": summary["generated_total"],
        "validated_total": summary["validated_total"],
        "reset_mutation_passed": summary["reset_mutation_passed"],
        "reset_mutation_pass_ratio": pass_ratio,
        "mutation_runtime_complete": summary["mutation_runtime_complete"],
        "mutation_runtime_complete_ratio": summary["mutation_runtime_complete_ratio"],
        "min_pass_ratio": float(min_pass_ratio),
        "run_reachability": bool(run_reachability),
        "deferred_codes": summary["deferred_codes"],
        "failure_codes": sorted(summary["failure_codes"]),
        "artifacts": {
            "index": "index.html",
            "task_generation_gallery": "task_generation_gallery.html",
            "rejected_task_gallery": "rejected_task_gallery.html",
            "stage5_validation": "stage5_validation.html",
            "reset_before_after": "reset_before_after.html",
            "reset_before_after_png": "reset_before_after.png",
            "state_predicate_check": "state_predicate_check.html",
            "mutation_diff": "mutation_diff.html",
            "mutation_replay": "mutation_replay.html",
            "task_bank_json": "logs/task_bank.json",
            "rejected_tasks_json": "logs/rejected_tasks.json",
            "accepted_tasks_jsonl": "logs/accepted_tasks.jsonl",
            "rejected_tasks_jsonl": "logs/rejected_tasks.jsonl",
            "category_summary_json": "logs/category_summary.json",
            "stage5_validation_json": "logs/stage5_validation.json",
            "summary_json": "logs/stage5_summary.json",
        },
        "notes": [
            "Stage 5 validates symbolic reset, deterministic placement, init predicates, coarse geometry/reachability, and supported scheduler dry-runs.",
            "Bulk acceptance is pass-ratio based; failed candidates remain visible with stable failure codes and are not admitted to oracle.",
            "Unsupported scheduler modes remain in deferred buckets with stable codes rather than being silently treated as executable.",
            "This is not OmniGibson RGB/physics validation; visual human review still uses the generated symbolic artifacts.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {key: str(output_dir / value) for key, value in manifest["artifacts"].items()}
    return manifest


def build_stage6_execution_report(
    *,
    out_root: Path,
    run_id: str | None = None,
    procthor_template_dir: Path = DEFAULT_PROCTHOR_TEMPLATE_DIR,
    scene_models: list[str] | None = None,
    min_stage5_pass_ratio: float = 0.75,
    min_execution_pass_ratio: float = 0.50,
    run_reachability: bool = False,
    max_high_level_steps: int = 100,
) -> dict[str, Any]:
    """Generate Stage 6 lightweight teacher/oracle public-tool execution artifacts."""

    from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv, MODEL_VISIBLE_TOOLS
    from scoper.benchmark.behavior.scene.profiler import profile_scenes
    from scoper.benchmark.behavior.task_generation.profile_task_generator import generate_stage4_tasks
    from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates, match_scene_to_templates
    from scoper.benchmark.behavior.task_generation.stage5_validator import summarize_stage5_results, validate_stage5_reset_mutation
    from scoper.benchmark.behavior.task_generation.stage6_executor import (
        build_tool_micro_cases,
        summarize_stage6_results,
        summarize_tool_micro_cases,
        validate_stage6_execution,
    )

    scene_models = scene_models or ["Rs_int"]
    run_id = run_id or "{}_stage6_execution".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage6_execution" / run_id
    logs_dir = output_dir / "logs"
    traces_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    traces_dir.mkdir(exist_ok=True)

    profiles = profile_scenes(tuple(scene_models))
    templates = load_procthor_templates(procthor_template_dir)
    match_records = []
    for profile in profiles:
        match_records.extend(match_scene_to_templates(profile, templates, max_bindings_per_template=2))
    generation = generate_stage4_tasks(
        profiles=profiles,
        templates=templates,
        match_records=match_records,
        max_tasks_per_template_scene=1,
        seed=0,
    )
    stage5_results = [
        validate_stage5_reset_mutation(task, rng_seed=0, run_reachability=run_reachability)
        for task in generation["tasks"]
    ]
    stage5_by_task = {item.get("task_id"): item for item in stage5_results}
    stage6_results = [
        validate_stage6_execution(
            task,
            stage5_result=stage5_by_task.get(task.get("id")),
            rng_seed=0,
            max_high_level_steps=max_high_level_steps,
            trace_dir=traces_dir,
        )
        for task in generation["tasks"]
    ]
    stage5_summary = summarize_stage5_results(generation, stage5_results)
    stage6_summary = summarize_stage6_results(generation, stage5_results, stage6_results)
    tool_micro_cases = build_tool_micro_cases(rng_seed=0)
    tool_micro_summary = summarize_tool_micro_cases(tool_micro_cases)
    validation_records = stage6_validation_records(generation, stage6_results)
    validation_summary = summarize_validation_records(validation_records)
    stage6_summary["validator_records"] = {
        "accepted_total": validation_summary["accepted_total"],
        "rejected_total": validation_summary["rejected_total"],
        "buckets": validation_summary["buckets"],
    }
    stage6_summary["category_summary"] = validation_summary["by_category"]

    tool_specs = BehaviorClosedLoopToolEnv().get_tool_specs(model_visible_only=True)
    write_stage6_execution_index(output_dir / "index.html", stage5_summary, stage6_summary, tool_micro_summary)
    write_stage6_teacher_readiness(output_dir / "teacher_readiness.html", stage6_results)
    write_stage6_execution_table(output_dir / "stage6_execution.html", stage6_results)
    write_stage6_tool_trace(output_dir / "tool_trace.html", stage6_results)
    write_stage6_predicate_progress(output_dir / "predicate_progress.html", stage6_results)
    write_stage6_metrics_dashboard(output_dir / "metrics_dashboard.html", stage6_summary, stage6_results)
    write_stage6_failure_breakdown(output_dir / "failure_breakdown.html", stage6_summary, stage6_results)
    write_stage6_tool_catalog(output_dir / "tool_catalog.html", tool_specs)
    write_stage6_tool_micro_cases(output_dir / "tool_micro_cases.html", tool_micro_cases, tool_micro_summary)
    write_stage5_validation_table(output_dir / "stage5_validation.html", stage5_results)

    write_json(logs_dir / "task_bank.json", generation["tasks"])
    write_json(logs_dir / "stage5_validation.json", stage5_results)
    write_json(logs_dir / "stage5_summary.json", stage5_summary)
    write_json(logs_dir / "stage6_execution.json", stage6_results)
    write_json(logs_dir / "stage6_summary.json", stage6_summary)
    write_json(logs_dir / "tool_micro_cases.json", tool_micro_cases)
    write_json(logs_dir / "tool_micro_summary.json", tool_micro_summary)
    write_json(logs_dir / "category_summary.json", validation_summary)
    write_jsonl(logs_dir / "accepted_tasks.jsonl", split_records(validation_records, "accepted"))
    write_jsonl(logs_dir / "rejected_tasks.jsonl", split_records(validation_records, "rejected"))

    stage5_pass_ratio = float(stage5_summary.get("reset_mutation_pass_ratio") or 0.0)
    execution_pass_ratio = float(stage6_summary.get("execution_pass_ratio") or 0.0)
    passed = (
        stage5_pass_ratio >= float(min_stage5_pass_ratio)
        and stage6_summary.get("execution_attempted", 0) > 0
        and execution_pass_ratio >= float(min_execution_pass_ratio)
        and bool(tool_micro_summary.get("all_cases_match_expected"))
    )
    manifest = {
        "run_id": run_id,
        "stage": "stage6_execution",
        "status": "passed" if passed else "failed",
        "visual_gate": "execution_validated_pending_human_review" if passed else "execution_validation_failed",
        "scene_models": scene_models,
        "generated_total": stage6_summary["generated_total"],
        "stage5_admitted": stage6_summary["stage5_admitted"],
        "teacher_ready": stage6_summary["teacher_ready"],
        "strict_teacher_ready": stage6_summary["strict_teacher_ready"],
        "predicate_fallback_ready": stage6_summary["predicate_fallback_ready"],
        "execution_attempted": stage6_summary["execution_attempted"],
        "execution_passed": stage6_summary["execution_passed"],
        "execution_passed_runtime_complete": stage6_summary["execution_passed_runtime_complete"],
        "execution_passed_with_deferred_runtime": stage6_summary["execution_passed_with_deferred_runtime"],
        "execution_pass_ratio": execution_pass_ratio,
        "execution_runtime_complete_ratio": stage6_summary["execution_runtime_complete_ratio"],
        "min_stage5_pass_ratio": float(min_stage5_pass_ratio),
        "min_execution_pass_ratio": float(min_execution_pass_ratio),
        "run_reachability": bool(run_reachability),
        "max_high_level_steps": int(max_high_level_steps),
        "stage6_buckets": stage6_summary["stage6_buckets"],
        "failure_codes": sorted(stage6_summary["failure_codes"]),
        "deferred_codes": stage6_summary["deferred_codes"],
        "tool_micro_cases_match_expected": bool(tool_micro_summary.get("all_cases_match_expected")),
        "model_visible_tools": list(MODEL_VISIBLE_TOOLS),
        "artifacts": {
            "index": "index.html",
            "teacher_readiness": "teacher_readiness.html",
            "stage6_execution": "stage6_execution.html",
            "tool_trace": "tool_trace.html",
            "predicate_progress": "predicate_progress.html",
            "metrics_dashboard": "metrics_dashboard.html",
            "failure_breakdown": "failure_breakdown.html",
            "tool_catalog": "tool_catalog.html",
            "tool_micro_cases": "tool_micro_cases.html",
            "stage5_validation": "stage5_validation.html",
            "task_bank_json": "logs/task_bank.json",
            "accepted_tasks_jsonl": "logs/accepted_tasks.jsonl",
            "rejected_tasks_jsonl": "logs/rejected_tasks.jsonl",
            "category_summary_json": "logs/category_summary.json",
            "stage6_execution_json": "logs/stage6_execution.json",
            "summary_json": "logs/stage6_summary.json",
            "tool_micro_cases_json": "logs/tool_micro_cases.json",
        },
        "notes": [
            "Stage 6 is symbolic lightweight oracle/tool execution validation, not full family-oracle parity.",
            "Family teachers are used when registered; otherwise supported goal predicates use OracleBrain as predicate fallback.",
            "All execution attempts run through BehaviorClosedLoopToolEnv public tools and EpisodeRunner.",
            "Deferred teacher/scheduler/predicate cases remain visible with stable codes.",
            "execution_passed includes deferred-runtime successes; execution_passed_runtime_complete excludes them.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {key: str(output_dir / value) for key, value in manifest["artifacts"].items()}
    return manifest


def build_stage4_summary(
    generation: Mapping[str, Any],
    structure_results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    generated = generation.get("tasks") or []
    rejected = generation.get("rejected") or []
    runtime_requirements = Counter()
    init_mutations = Counter()
    readiness_debt = Counter()
    family_specs = Counter()
    for task in generated:
        gen = task.get("generation") or {}
        runtime_requirements.update(gen.get("runtime_capabilities_required") or [])
        init_mutations.update(gen.get("required_init_mutations") or [])
        readiness_debt.update(gen.get("stage4_readiness_debt") or [])
        family_spec = task.get("family_spec")
        if isinstance(family_spec, Mapping):
            family_specs.update([str(family_spec.get("family") or "unknown")])
    return {
        "generated_total": len(generated),
        "rejected_total": len(rejected),
        "generated_by_template": (generation.get("summary") or {}).get("generated_by_template", {}),
        "generated_by_scene": (generation.get("summary") or {}).get("generated_by_scene", {}),
        "generated_by_stage3_status": (generation.get("summary") or {}).get("generated_by_stage3_status", {}),
        "generated_by_readiness": (generation.get("summary") or {}).get("generated_by_readiness", {}),
        "rejection_codes": (generation.get("summary") or {}).get("rejection_codes", {}),
        "structure_passed": sum(1 for result in structure_results if result.get("structure_ok")),
        "structure_failure_codes": dict(sorted(Counter(result.get("failure_code") for result in structure_results if result.get("failure_code")).items())),
        "runtime_requirements": dict(sorted(runtime_requirements.items())),
        "init_mutations": dict(sorted(init_mutations.items())),
        "readiness_debt": dict(sorted(readiness_debt.items())),
        "generated_with_family_spec": sum(1 for task in generated if isinstance(task.get("family_spec"), Mapping)),
        "generated_with_disturbance_spec": sum(1 for task in generated if task.get("disturbance_spec")),
        "generated_with_failure_spec": sum(1 for task in generated if task.get("failure_spec")),
        "generated_with_unsupported_scheduler_specs": sum(
            1 for task in generated if (task.get("generation") or {}).get("unsupported_scheduler_specs")
        ),
        "generated_by_family": dict(sorted(family_specs.items())),
    }


def build_stage4_5_summary(
    generation: Mapping[str, Any],
    validation_results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    generated = generation.get("tasks") or []
    rejected = generation.get("rejected") or []
    return {
        "generated_total": len(generated),
        "rejected_total": len(rejected),
        "generated_by_template": (generation.get("summary") or {}).get("generated_by_template", {}),
        "generated_by_scene": (generation.get("summary") or {}).get("generated_by_scene", {}),
        "rejection_codes": (generation.get("summary") or {}).get("rejection_codes", {}),
        "structure_passed": sum(1 for result in validation_results if result.get("structure_ok")),
        "reset_passed": sum(1 for result in validation_results if result.get("reset_ok")),
        "init_passed": sum(1 for result in validation_results if result.get("init_ok")),
        "initial_goal_false_passed": sum(1 for result in validation_results if result.get("initial_goal_false")),
        "oracle_passed": sum(1 for result in validation_results if result.get("oracle_ok")),
        "validation_failure_codes": dict(sorted(Counter(result.get("failure_code") for result in validation_results if result.get("failure_code")).items())),
    }


def is_profile_match_record(record: Mapping[str, Any]) -> bool:
    if "profile_match" in record:
        return bool(record.get("profile_match"))
    return str(record.get("status")) == "matched"


def load_template_records(template_dir: Path, *, source: str) -> list[TemplateRecord]:
    records: list[TemplateRecord] = []
    if not template_dir.exists():
        return records
    for path in sorted(template_dir.iterdir()):
        if path.suffix not in (".json", ".yaml", ".yml"):
            continue
        data = load_structured_file(path)
        if not isinstance(data, Mapping):
            continue
        template_id = str(data.get("template_id") or path.stem)
        category = str(data.get("category") or infer_category_from_filename(path.name))
        records.append(
            TemplateRecord(
                path=path,
                filename=path.name,
                template_id=template_id,
                category=category,
                description=str(data.get("description") or ""),
                roles=extract_roles(data),
                goal_predicates=extract_goal_predicates(data),
                source=source,
                raw=dict(data),
            )
        )
    return records


def load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def extract_roles(data: Mapping[str, Any]) -> list[str]:
    if isinstance(data.get("roles"), Mapping):
        return sorted(str(k) for k in data["roles"].keys())
    if isinstance(data.get("requires"), Mapping):
        return sorted(str(k) for k in data["requires"].keys())
    return []


def extract_goal_predicates(data: Mapping[str, Any]) -> list[str]:
    predicates = data.get("goal_predicates") or []
    out: list[str] = []
    for predicate in predicates:
        if isinstance(predicate, Mapping):
            out.append(str(predicate.get("type") or predicate.get("predicate") or "unknown"))
    return out


def infer_category_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if stem.startswith(("clean_then_place", "cool_then_place", "heat_then_place", "slice_then_place")):
        return "prepare_then_place"
    if stem.startswith("failure_recovery"):
        return "verify_after_action"
    if stem.startswith("put_away"):
        return "put_away"
    if stem.startswith("recover_"):
        return "recover"
    if stem.startswith("mid_execution") or stem.startswith("semantic_"):
        return "mid_execution_state_disturbance"
    return stem


def collect_repo_status() -> dict[str, Any]:
    root = repo_root()
    eb_root = root / "scoper.benchmark.behavior"
    outputs = root / "outputs"

    modules = {
        "task_generation": (eb_root / "task_generation" / "generator.py").exists(),
        "role_binder": (eb_root / "task_generation" / "role_binder.py").exists(),
        "tool_env": (eb_root / "behavior_closed_loop_tool_env.py").exists(),
        "oracle": (eb_root / "brains" / "oracle.py").exists(),
        "goal_checker": (eb_root / "goal_checker.py").exists(),
        "metrics": (eb_root / "metrics.py").exists(),
        "vlm": (eb_root / "brains" / "vlm").exists(),
        "video": (eb_root / "render" / "video.py").exists(),
        "symbolic_scene": (eb_root / "scene" / "placement_symbolic.py").exists(),
        "render_scene": (eb_root / "scene" / "placement_render.py").exists(),
        "scheduler": (eb_root / "scheduler").exists(),
    }
    stage_outputs = []
    for path in sorted(outputs.glob("eb_behavior_stage*")) if outputs.exists() else []:
        if not path.is_dir():
            continue
        summary = path / "summary.json"
        stage_outputs.append(
            {
                "name": path.name,
                "path": str(path.relative_to(root)),
                "has_summary": summary.exists(),
                "has_trace": bool(list(path.glob("trace*.jsonl"))),
                "has_video": bool(list(path.rglob("*.mp4"))),
            }
        )
    viz_dir = outputs / "eb_behavior_viz"
    return {
        "repo_root": str(root),
        "modules": modules,
        "tests": sorted(str(p.relative_to(root)) for p in (eb_root / "tests").glob("test_*.py")),
        "task_sets": sorted(str(p.relative_to(root)) for p in (data_dir() / "task_sets").glob("*.json")),
        "stage_outputs": stage_outputs,
        "legacy_visualizations": {
            "path": str(viz_dir.relative_to(root)) if viz_dir.exists() else None,
            "png_count": len(list(viz_dir.glob("*.png"))) if viz_dir.exists() else 0,
            "has_index": (viz_dir / "index.md").exists(),
        },
    }


def build_capability_matrix(
    procthor_templates: list[TemplateRecord],
    eb_templates: list[TemplateRecord],
    repo_status: Mapping[str, Any],
) -> list[MatrixRow]:
    eb_by_id = {t.template_id: t for t in eb_templates}
    implemented_modules = repo_status["modules"]
    rows: list[MatrixRow] = []
    for template in procthor_templates:
        match_id = find_matching_eb_template(template, eb_by_id)
        covered = match_id is not None
        row = {
            "template_schema": "implemented" if match_id in eb_by_id else "missing",
            "scene_profile": "missing",
            "scene_match": "missing",
            "role_binding": "partial" if covered and implemented_modules.get("role_binder") else "missing",
            "task_generation": "partial" if covered and implemented_modules.get("task_generation") else "missing",
            "init_reset": "partial" if covered and implemented_modules.get("symbolic_scene") else "missing",
            "scene_mutation": "partial" if mutation_like_template(template) and implemented_modules.get("scheduler") else "missing",
            "oracle_plan": "partial" if covered and implemented_modules.get("oracle") else "missing",
            "tool_execution": "partial" if covered and implemented_modules.get("tool_env") else "missing",
            "render_video": "partial" if covered and implemented_modules.get("video") and has_any_video_output(repo_status) else "missing",
            "goal_check": "partial" if covered and implemented_modules.get("goal_checker") else "missing",
            "metrics": "partial" if covered and implemented_modules.get("metrics") else "missing",
            "vlm_ready": "partial" if covered and implemented_modules.get("vlm") else "missing",
            "visual_gate": "missing",
        }
        notes = []
        if match_id:
            notes.append("mapped to scoper.benchmark.behavior template {}".format(match_id))
        if template.template_id not in TARGET_TO_EB_TEMPLATE and not match_id:
            notes.append("no Stage-0 schema mapping yet")
        if row["render_video"] == "partial":
            notes.append("legacy video artifact exists; unified Stage-0/0.5 gate still pending")
        rows.append(MatrixRow(template=template, matched_eb_template_id=match_id, statuses=row, notes=notes))
    return rows


def find_matching_eb_template(template: TemplateRecord, eb_by_id: Mapping[str, TemplateRecord]) -> str | None:
    if template.template_id in eb_by_id:
        return template.template_id
    mapped = TARGET_TO_EB_TEMPLATE.get(template.template_id)
    if mapped in eb_by_id:
        return mapped
    # Some eb_procthor template_ids in the JSON differ slightly from filenames. Use the
    # filename stem as a conservative fallback for the few known migrated smoke families.
    mapped = TARGET_TO_EB_TEMPLATE.get(template.path.stem)
    if mapped in eb_by_id:
        return mapped
    return None


def mutation_like_template(template: TemplateRecord) -> bool:
    text = " ".join([template.filename, template.template_id, template.category]).lower()
    return any(token in text for token in ("disturbance", "recovery", "recover", "mid_execution", "semantic"))


def has_any_video_output(repo_status: Mapping[str, Any]) -> bool:
    return any(stage.get("has_video") for stage in repo_status.get("stage_outputs", []))


def build_summary(
    procthor_templates: list[TemplateRecord],
    eb_templates: list[TemplateRecord],
    matrix: list[MatrixRow],
    repo_status: Mapping[str, Any],
) -> dict[str, Any]:
    family_counts = Counter(t.category for t in procthor_templates)
    status_counts: dict[str, dict[str, int]] = {}
    for column in CAPABILITY_COLUMNS:
        status_counts[column] = dict(Counter(row.statuses[column] for row in matrix))
    covered_rows = [row for row in matrix if row.matched_eb_template_id]
    return {
        "template_target_count": len(procthor_templates),
        "eb_behavior_template_count": len(eb_templates),
        "target_family_counts": dict(sorted(family_counts.items())),
        "covered_target_count": len(covered_rows),
        "covered_target_ids": [row.template.template_id for row in covered_rows],
        "status_counts": status_counts,
        "module_count": sum(1 for ok in repo_status["modules"].values() if ok),
        "test_count": len(repo_status["tests"]),
        "stage_output_count": len(repo_status["stage_outputs"]),
        "legacy_visualization_png_count": repo_status["legacy_visualizations"]["png_count"],
    }


def serialize_matrix(matrix: list[MatrixRow]) -> list[dict[str, Any]]:
    return [
        {
            "template_id": row.template.template_id,
            "filename": row.template.filename,
            "category": row.template.category,
            "matched_eb_template_id": row.matched_eb_template_id,
            "statuses": row.statuses,
            "notes": row.notes,
        }
        for row in matrix
    ]


def build_template_catalog_entries(
    procthor_templates: list[TemplateRecord],
    eb_templates: list[TemplateRecord],
    matrix: list[MatrixRow],
) -> list[dict[str, Any]]:
    eb_by_id = {template.template_id: template for template in eb_templates}
    match_by_id = {row.template.template_id: row.matched_eb_template_id for row in matrix}
    entries = []
    for template in procthor_templates:
        raw = template.raw
        matched_id = match_by_id.get(template.template_id)
        matched_template = eb_by_id.get(matched_id or "")
        initial_predicates = predicate_summaries(raw.get("initial_state_constraints") or raw.get("init_constraints") or [])
        goal_predicates = predicate_summaries(raw.get("goal_predicates") or [])
        required_scene_features = infer_scene_features(raw)
        required_object_states = infer_object_state_dependencies(raw, initial_predicates, goal_predicates)
        required_tools = infer_required_tools(raw, goal_predicates)
        scheduler_mutation = infer_scheduler_mutation(raw, template)
        vlm_observation = infer_vlm_observation(raw, template)
        entries.append(
            {
                "template_id": template.template_id,
                "filename": template.filename,
                "category": template.category,
                "template_version": str(raw.get("template_version") or ""),
                "description": template.description,
                "roles": template.roles,
                "role_requirements": compact_role_requirements(raw),
                "initial_predicates": initial_predicates,
                "goal_predicates": goal_predicates,
                "required_scene_features": required_scene_features,
                "required_object_states": required_object_states,
                "required_tools": required_tools,
                "oracle_teacher": infer_oracle_teacher(raw, template),
                "scheduler_mutation": scheduler_mutation,
                "metrics_requirements": infer_metrics_requirements(raw, scheduler_mutation, vlm_observation),
                "vlm_observation": vlm_observation,
                "expected_events": list(raw.get("expected_events") or []),
                "expected_brain_behaviors": list(raw.get("expected_brain_behaviors") or []),
                "goal_requires_model_stop": bool((raw.get("goal_options") or {}).get("require_model_stop", False)),
                "matched_eb_template_id": matched_id,
                "matched_eb_template_file": matched_template.filename if matched_template else None,
                "migration_status": "mapped_current_yaml" if matched_template else "cataloged_target_only",
                "schema_status": "target_json_cataloged",
                "notes": template_catalog_notes(template, matched_template, required_scene_features, scheduler_mutation, vlm_observation),
            }
        )
    return entries


def predicate_summaries(predicates: Any) -> list[str]:
    out = []
    if not isinstance(predicates, list):
        return out
    for predicate in predicates:
        if not isinstance(predicate, Mapping):
            continue
        ptype = str(predicate.get("type") or predicate.get("predicate") or "unknown")
        object_role = predicate.get("object_role")
        receptacle_role = predicate.get("receptacle_role")
        receptacle_roles = predicate.get("receptacle_roles")
        if object_role and receptacle_role:
            out.append("{}:{}->{}".format(ptype, object_role, receptacle_role))
        elif object_role and isinstance(receptacle_roles, list):
            out.append("{}:{}->[{}]".format(ptype, object_role, ",".join(str(v) for v in receptacle_roles)))
        elif object_role:
            out.append("{}:{}".format(ptype, object_role))
        else:
            args = predicate.get("args")
            if isinstance(args, Mapping):
                out.append("{}:{}".format(ptype, ",".join("{}={}".format(k, v) for k, v in sorted(args.items()))))
            elif isinstance(args, list):
                out.append("{}:{}".format(ptype, ",".join(str(v) for v in args)))
            else:
                out.append(ptype)
    return sorted(dict.fromkeys(out))


def compact_role_requirements(raw: Mapping[str, Any]) -> dict[str, list[str]]:
    requires = raw.get("requires")
    if not isinstance(requires, Mapping):
        requires = raw.get("roles")
    if not isinstance(requires, Mapping):
        return {}
    result = {}
    for role, requirement in requires.items():
        if not isinstance(requirement, Mapping):
            result[str(role)] = []
            continue
        bits = []
        for key in sorted(requirement):
            value = requirement[key]
            if isinstance(value, (str, int, float, bool)):
                bits.append("{}={}".format(key, value))
            elif isinstance(value, list):
                bits.append("{}=[{}]".format(key, ",".join(str(v) for v in value)))
            elif isinstance(value, Mapping) and key == "relocate_if":
                modes = value.get("modes") or []
                fallback = value.get("fallback_to")
                target = value.get("to_receptacle_role")
                label = "relocate_if=modes:[{}]".format(",".join(str(v) for v in modes))
                if fallback:
                    label += ",fallback:{}".format(fallback)
                if target:
                    label += ",target:{}".format(target)
                bits.append(label)
        result[str(role)] = bits
    return result


def infer_scene_features(raw: Mapping[str, Any]) -> list[str]:
    features = set()
    requires = raw.get("requires") if isinstance(raw.get("requires"), Mapping) else {}
    for role, requirement in requires.items():
        if not isinstance(requirement, Mapping):
            continue
        taxonomy_group = requirement.get("taxonomy_group")
        if taxonomy_group:
            features.add("taxonomy_group:{}".format(taxonomy_group))
        for field in ("object_type_allowlist", "object_type_blocklist"):
            for value in requirement.get(field) or []:
                features.add("{}:{}".format(field, value))
        for field in (
            "must_be_pickupable",
            "must_be_receptacle",
            "must_be_openable",
            "must_be_toggleable",
            "must_be_sliceable",
            "must_not_be_pickupable",
            "must_have_parent_receptacle",
            "must_not_have_parent_receptacle",
            "must_not_have_closed_parent_receptacle",
            "must_be_visible_from_start",
            "must_not_be_visible_from_start",
            "optional",
            "relocate_target_only",
        ):
            if requirement.get(field):
                features.add("{}:{}".format(role, field))
        for field in (
            "same_room_as",
            "different_room_from",
            "same_object_type_as",
            "near_role",
            "parent_of_role_base",
        ):
            if requirement.get(field):
                features.add("spatial:{}:{}={}".format(role, field, requirement[field]))
        relocate_if = requirement.get("relocate_if")
        if isinstance(relocate_if, Mapping):
            modes = [str(mode) for mode in (relocate_if.get("modes") or [])]
            fallback = relocate_if.get("fallback_to")
            if fallback:
                modes.append(str(fallback))
            for mode in modes:
                features.add("relocate:{}:{}".format(role, mode))
    for constraint in raw.get("initial_state_constraints") or []:
        if isinstance(constraint, Mapping):
            ctype = str(constraint.get("type") or "")
            if ctype in {"object_in_receptacle", "object_not_in_receptacle"}:
                features.add("initial_spatial:{}".format(ctype))
    if raw.get("disturbance_spec"):
        features.add("scheduler_disturbance_required")
    if infer_vlm_observation(raw, TemplateRecord(Path(), "", "", "")) == "required":
        features.add("visual_observation_required")
    return sorted(features)


def infer_object_state_dependencies(
    raw: Mapping[str, Any],
    initial_predicates: list[str],
    goal_predicates: list[str],
) -> list[str]:
    states = set()
    state_tokens = (
        "open",
        "closed",
        "toggle",
        "toggled",
        "dirty",
        "clean",
        "hot",
        "cold",
        "cool",
        "cooked",
        "sliced",
        "filled",
        "contains",
        "held",
    )
    for summary in [*initial_predicates, *goal_predicates]:
        lowered = summary.lower()
        for token in state_tokens:
            if token in lowered:
                states.add(token)
    requires = raw.get("requires") if isinstance(raw.get("requires"), Mapping) else {}
    for role, requirement in requires.items():
        if not isinstance(requirement, Mapping):
            continue
        for field, state in (
            ("must_be_openable", "openable_state"),
            ("must_be_toggleable", "toggle_state"),
            ("must_be_sliceable", "slice_state"),
        ):
            if requirement.get(field):
                states.add("{}:{}".format(role, state))
    return sorted(states)


def infer_required_tools(raw: Mapping[str, Any], goal_predicates: list[str]) -> list[str]:
    tools = set()
    text = " ".join(
        [
            *goal_predicates,
            *[str(v) for v in raw.get("expected_events") or []],
            *[str(v) for v in raw.get("expected_brain_behaviors") or []],
            str(raw.get("category") or ""),
            str(raw.get("template_id") or ""),
        ]
    ).lower()
    if any(token in text for token in ("object_in_receptacle", "object_on_receptacle", "place", "put_away", "store")):
        tools.update({"navigate_to", "pick_up", "put_down"})
    if "open" in text:
        tools.add("open")
    if "closed" in text or "close" in text or "reclosed" in text:
        tools.add("close")
    if "toggle" in text or "faucet" in text:
        tools.update({"toggle_on", "toggle_off"})
    if "clean" in text or "wash" in text or "wipe" in text:
        tools.update({"wipe", "soak_under"})
    if "heat" in text or "cook" in text or "hot" in text:
        tools.add("heat")
    if "cool" in text or "cold" in text:
        tools.add("cool")
    if "slice" in text or "cut" in text:
        tools.add("cut")
    if any(token in text for token in ("visual", "discover", "explore", "look", "qa")):
        tools.add("scan")
    return sorted(tools)


def infer_scheduler_mutation(raw: Mapping[str, Any], template: TemplateRecord) -> list[str]:
    mutations = set()
    disturbance_spec = raw.get("disturbance_spec")
    if isinstance(disturbance_spec, Mapping):
        for disturbance in disturbance_spec.get("disturbances") or []:
            if isinstance(disturbance, Mapping):
                mutations.add(str(disturbance.get("type") or "disturbance"))
    text = " ".join([template.filename, template.template_id, template.category]).lower()
    if "mid_execution" in text:
        mutations.add("mid_execution_disturbance")
    if "recover" in text or "recovery" in text:
        mutations.add("recovery_setup")
    if "dynamic_state" in text or "semantic_re" in text or "disturbance" in text:
        mutations.add("state_or_semantic_mutation")
    return sorted(mutations)


def infer_vlm_observation(raw: Mapping[str, Any], template: TemplateRecord) -> str:
    text = " ".join(
        [
            template.filename,
            template.template_id,
            template.category,
            str(raw.get("description") or ""),
            *[str(v) for v in raw.get("expected_brain_behaviors") or []],
        ]
    ).lower()
    if any(token in text for token in ("visual", "discover", "explore", "human_qa", "incident", "cross_room")):
        return "required"
    if any(token in text for token in ("visible_from_start", "scan", "look")):
        return "helpful"
    return "not_required_for_template_logic"


def infer_oracle_teacher(raw: Mapping[str, Any], template: TemplateRecord) -> str:
    category = template.category
    if category == "static_simple":
        return "static_simple_teacher"
    if category in {"prepare_then_place", "prepare_multi_stage"}:
        return "prepare_teacher_family"
    if category == "verify_after_action":
        return "failure_recovery_teacher"
    if category == "put_away_by_category":
        return "category_policy_teacher"
    if category == "tidy_by_policy":
        return "tidy_policy_teacher"
    if category in {"recover_and_store_incident_object", "recover_multi_incident"}:
        return "incident_recovery_teacher"
    if category == "human_interactive_execution":
        return "human_interaction_teacher"
    if category == "mid_execution_state_disturbance":
        return "disturbance_recovery_teacher"
    if category in {"explore_then_act", "put_away_visual_judgment", "put_away_cross_room"}:
        return "exploration_or_visual_judgment_teacher"
    if raw.get("disturbance_spec"):
        return "disturbance_recovery_teacher"
    return "{}_teacher_pending".format(category or "unknown")


def infer_metrics_requirements(
    raw: Mapping[str, Any],
    scheduler_mutation: list[str],
    vlm_observation: str,
) -> list[str]:
    metrics = {"predicate_progress", "task_success", "tool_success_rate"}
    if (raw.get("goal_options") or {}).get("require_model_stop"):
        metrics.add("model_stop_success")
    if scheduler_mutation:
        metrics.update({"disturbance_handled", "recovery_count"})
    if vlm_observation == "required":
        metrics.add("visual_discovery_success")
    if raw.get("category") in {"put_away_by_category", "tidy_by_policy"}:
        metrics.add("policy_violation_count")
    return sorted(metrics)


def template_catalog_notes(
    template: TemplateRecord,
    matched_template: TemplateRecord | None,
    required_scene_features: list[str],
    scheduler_mutation: list[str],
    vlm_observation: str,
) -> list[str]:
    notes = []
    if matched_template:
        notes.append("current YAML mapping: {}".format(matched_template.template_id))
    else:
        notes.append("no current scoper.benchmark.behavior YAML mapping")
    if required_scene_features:
        notes.append("requires BEHAVIOR scene profile before generation")
    if scheduler_mutation:
        notes.append("requires scheduler/mutation support")
    if vlm_observation == "required":
        notes.append("requires visual observation path")
    return notes


def build_template_family_overview(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_family[entry["category"]].append(entry)
    rows = []
    for family in sorted(by_family):
        family_entries = by_family[family]
        tool_counts = Counter(tool for entry in family_entries for tool in entry["required_tools"])
        scene_feature_counts = Counter(feature for entry in family_entries for feature in entry["required_scene_features"])
        rows.append(
            {
                "category": family,
                "template_count": len(family_entries),
                "mapped_count": sum(1 for entry in family_entries if entry["matched_eb_template_id"]),
                "vlm_required_count": sum(1 for entry in family_entries if entry["vlm_observation"] == "required"),
                "mutation_required_count": sum(1 for entry in family_entries if entry["scheduler_mutation"]),
                "required_tools": sorted(tool_counts),
                "top_scene_features": [feature for feature, _ in scene_feature_counts.most_common(12)],
                "template_ids": [entry["template_id"] for entry in family_entries],
            }
        )
    return rows


def build_template_dependency_graph(entries: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = []
    edges = []
    seen_nodes = set()

    def add_node(node_id: str, node_type: str, label: str) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append({"id": node_id, "type": node_type, "label": label})

    for entry in entries:
        template_node = "template:{}".format(entry["template_id"])
        add_node(template_node, "template", entry["template_id"])
        for dependency_type, values in (
            ("scene_feature", entry["required_scene_features"]),
            ("tool", entry["required_tools"]),
            ("object_state", entry["required_object_states"]),
            ("mutation", entry["scheduler_mutation"]),
            ("metric", entry["metrics_requirements"]),
        ):
            for value in values:
                dep_node = "{}:{}".format(dependency_type, value)
                add_node(dep_node, dependency_type, value)
                edges.append({"source": template_node, "target": dep_node, "type": "requires"})
        oracle_node = "oracle:{}".format(entry["oracle_teacher"])
        add_node(oracle_node, "oracle", entry["oracle_teacher"])
        edges.append({"source": template_node, "target": oracle_node, "type": "requires"})
        if entry["matched_eb_template_id"]:
            eb_node = "eb_template:{}".format(entry["matched_eb_template_id"])
            add_node(eb_node, "eb_template", entry["matched_eb_template_id"])
            edges.append({"source": template_node, "target": eb_node, "type": "mapped_to"})
    return {"nodes": nodes, "edges": edges}


def build_stage1_summary(
    entries: list[dict[str, Any]],
    eb_templates: list[TemplateRecord],
    family_overview: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "template_target_count": len(entries),
        "eb_behavior_template_count": len(eb_templates),
        "mapped_target_count": sum(1 for entry in entries if entry["matched_eb_template_id"]),
        "family_count": len(family_overview),
        "families": family_overview,
        "vlm_required_count": sum(1 for entry in entries if entry["vlm_observation"] == "required"),
        "mutation_required_count": sum(1 for entry in entries if entry["scheduler_mutation"]),
        "templates_with_scene_features": sum(1 for entry in entries if entry["required_scene_features"]),
        "templates_with_tool_requirements": sum(1 for entry in entries if entry["required_tools"]),
        "all_target_template_ids": [entry["template_id"] for entry in entries],
    }


def build_task_template_schema(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the Stage-1 scoper.benchmark.behavior task-template schema contract.

    This is the migration-level template schema, not the fully materialized runtime
    task_spec schema validated by scoper.benchmark.behavior.task_spec.
    """

    observed_goal_predicates = sorted(
        {
            predicate.split(":", 1)[0]
            for entry in entries
            for predicate in entry["goal_predicates"]
        }
    )
    observed_initial_predicates = sorted(
        {
            predicate.split(":", 1)[0]
            for entry in entries
            for predicate in entry["initial_predicates"]
        }
    )
    return {
        "schema_name": "eb_behavior_task_template",
        "schema_version": TASK_TEMPLATE_SCHEMA_VERSION,
        "schema_scope": "migration_catalog",
        "top_level_fields": [
            {
                "name": "template_id",
                "required": True,
                "type": "string",
                "purpose": "Stable template identifier used in task source metadata.",
            },
            {
                "name": "template_version",
                "required": True,
                "type": "string",
                "purpose": "Template schema/content revision.",
            },
            {
                "name": "category",
                "required": True,
                "type": "string",
                "purpose": "Template family used for oracle, metrics, and report grouping.",
            },
            {
                "name": "description",
                "required": False,
                "type": "string",
                "purpose": "Human-readable task-family summary.",
            },
            {
                "name": "family_spec_template",
                "required": False,
                "type": "mapping",
                "purpose": "Structured family plan for prepare, put-away, recovery, or disturbance templates.",
            },
            {
                "name": "requires",
                "required": True,
                "type": "mapping[role_name, role_requirement]",
                "purpose": "Scene-profile role binding constraints.",
            },
            {
                "name": "initial_state_constraints",
                "required": False,
                "type": "list[predicate]",
                "purpose": "State and spatial constraints to enforce at reset.",
            },
            {
                "name": "goal_predicates",
                "required": True,
                "type": "list[predicate]",
                "purpose": "Goal predicates checked by the task oracle/metrics path.",
            },
            {
                "name": "goal_options",
                "required": False,
                "type": "mapping",
                "purpose": "Goal evaluation options such as require_model_stop.",
            },
            {
                "name": "instruction_templates",
                "required": True,
                "type": "list[string]",
                "purpose": "Natural-language instruction templates filled from role bindings.",
            },
            {
                "name": "expected_events",
                "required": False,
                "type": "list[string]",
                "purpose": "Event tags expected during successful execution.",
            },
            {
                "name": "expected_brain_behaviors",
                "required": False,
                "type": "list[string]",
                "purpose": "High-level policy behaviors expected from the agent.",
            },
            {
                "name": "disturbance_spec",
                "required": False,
                "type": "mapping",
                "purpose": "Scheduler mutations injected during or before execution.",
            },
        ],
        "role_requirement_fields": [
            "taxonomy_group",
            "object_type_allowlist",
            "object_type_blocklist",
            "must_be_pickupable",
            "must_be_receptacle",
            "must_be_openable",
            "must_be_toggleable",
            "must_be_sliceable",
            "must_not_be_pickupable",
            "must_have_parent_receptacle",
            "must_not_have_parent_receptacle",
            "must_not_have_closed_parent_receptacle",
            "must_be_visible_from_start",
            "must_not_be_visible_from_start",
            "different_from",
            "same_room_as",
            "different_room_from",
            "same_object_type_as",
            "near_role",
            "parent_of_role_base",
            "relocate_if",
            "relocate_target_only",
            "optional",
        ],
        "predicate_shapes": [
            {
                "name": "unary_object_state",
                "fields": ["type", "object_role"],
                "examples": ["object_clean", "object_closed", "object_hot"],
            },
            {
                "name": "binary_object_receptacle",
                "fields": ["type", "object_role", "receptacle_role"],
                "examples": ["object_in_receptacle", "object_not_in_receptacle"],
            },
            {
                "name": "multi_receptacle_policy",
                "fields": ["type", "object_role", "receptacle_roles", "include_if"],
                "examples": ["object_in_any_receptacle", "object_not_in_any_receptacle"],
            },
        ],
        "dependency_columns": [
            "required_scene_features",
            "required_object_states",
            "required_tools",
            "oracle_teacher",
            "scheduler_mutation",
            "metrics_requirements",
            "vlm_observation",
        ],
        "migration_status_values": [
            "mapped_current_yaml",
            "cataloged_target_only",
        ],
        "observed_categories": sorted({entry["category"] for entry in entries}),
        "observed_goal_predicates": observed_goal_predicates,
        "observed_initial_predicates": observed_initial_predicates,
        "observed_tools": sorted({tool for entry in entries for tool in entry["required_tools"]}),
        "observed_metrics": sorted({metric for entry in entries for metric in entry["metrics_requirements"]}),
        "notes": [
            "Stage 1 defines the migration template contract used to catalog eb_procthor families.",
            "Runtime task_spec validation remains in scoper.benchmark.behavior.task_spec and is intentionally separate.",
            "Scene binding and reset enforcement are deferred to Stage 2/3/4.",
        ],
    }


def write_index(path: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    cards = [
        ("Target templates", summary["template_target_count"]),
        ("Current scoper.benchmark.behavior templates", summary["eb_behavior_template_count"]),
        ("Mapped target templates", summary["covered_target_count"]),
        ("Detected tests", summary["test_count"]),
        ("Previous stage output dirs", summary["stage_output_count"]),
        ("Legacy top-down PNGs", summary["legacy_visualization_png_count"]),
    ]
    body = [
        html_header("scoper.benchmark.behavior Stage 0 Migration Dashboard"),
        "<main>",
        "<h1>scoper.benchmark.behavior Stage 0 Migration Dashboard</h1>",
        "<p class='muted'>Run <code>{}</code>. Visual gate: <strong>{}</strong>.</p>".format(
            escape(str(manifest["run_id"])), escape(str(manifest["visual_gate"]))
        ),
        "<section class='cards'>",
    ]
    for label, value in cards:
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(value, escape(label)))
    body.extend(
        [
            "</section>",
            "<section>",
            "<h2>Artifacts</h2>",
            "<ul>",
            "<li><a href='capability_matrix.html'>Capability matrix</a></li>",
            "<li><a href='template_scope.html'>Template scope</a></li>",
            "<li><a href='repo_status.html'>Repository status</a></li>",
            "<li><a href='manifest.json'>Manifest JSON</a></li>",
            "<li><a href='logs/capability_matrix.json'>Capability matrix JSON</a></li>",
            "</ul>",
            "</section>",
            "<section>",
            "<h2>Stage 0 Notes</h2>",
            "<p>This dashboard is a visibility baseline. It intentionally marks per-template visual gates as <code>missing</code> until Stage 0.5 or later creates task-level evidence in the new artifact format.</p>",
            "</section>",
            "</main>",
            html_footer(),
        ]
    )
    path.write_text("\n".join(body), encoding="utf-8")


def write_template_scope(
    path: Path,
    procthor_templates: list[TemplateRecord],
    eb_templates: list[TemplateRecord],
    matrix: list[MatrixRow],
) -> None:
    mapped_by_target = {row.template.template_id: row.matched_eb_template_id for row in matrix}
    body = [
        html_header("Template Scope"),
        "<main>",
        "<h1>Template Scope</h1>",
        "<h2>eb_procthor Target Templates ({})</h2>".format(len(procthor_templates)),
        "<table><thead><tr><th>Template</th><th>File</th><th>Category</th><th>Roles</th><th>Goal Predicates</th><th>scoper.benchmark.behavior Mapping</th></tr></thead><tbody>",
    ]
    for t in procthor_templates:
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(t.template_id),
                escape(t.filename),
                escape(t.category),
                escape(", ".join(t.roles) or "-"),
                escape(", ".join(t.goal_predicates) or "-"),
                escape(mapped_by_target.get(t.template_id) or "-"),
            )
        )
    body.extend(
        [
            "</tbody></table>",
            "<h2>Current scoper.benchmark.behavior Templates ({})</h2>".format(len(eb_templates)),
            "<table><thead><tr><th>Template</th><th>File</th><th>Category</th><th>Roles</th><th>Goal Predicates</th></tr></thead><tbody>",
        ]
    )
    for t in eb_templates:
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(t.template_id),
                escape(t.filename),
                escape(t.category),
                escape(", ".join(t.roles) or "-"),
                escape(", ".join(t.goal_predicates) or "-"),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_capability_matrix(path: Path, matrix: list[MatrixRow]) -> None:
    body = [
        html_header("Capability Matrix"),
        "<main>",
        "<h1>Capability Matrix</h1>",
        "<p class='muted'>Statuses are conservative Stage-0 scanner results. No target row is marked <code>visually_verified</code> by this stage.</p>",
        "<table class='matrix'><thead><tr><th>Template</th><th>Category</th><th>Mapping</th>",
    ]
    for column in CAPABILITY_COLUMNS:
        body.append("<th>{}</th>".format(escape(column)))
    body.append("<th>Notes</th></tr></thead><tbody>")
    for row in matrix:
        body.append(
            "<tr><td><code>{}</code><div class='muted'>{}</div></td><td>{}</td><td>{}</td>".format(
                escape(row.template.template_id),
                escape(row.template.filename),
                escape(row.template.category),
                escape(row.matched_eb_template_id or "-"),
            )
        )
        for column in CAPABILITY_COLUMNS:
            status = row.statuses[column]
            body.append("<td class='status {}'>{}</td>".format(status, escape(status)))
        body.append("<td>{}</td></tr>".format(escape("; ".join(row.notes) or "-")))
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_repo_status(path: Path, repo_status: Mapping[str, Any]) -> None:
    body = [
        html_header("Repository Status"),
        "<main>",
        "<h1>Repository Status</h1>",
        "<h2>Modules</h2>",
        "<table><thead><tr><th>Capability</th><th>Detected</th></tr></thead><tbody>",
    ]
    for name, detected in sorted(repo_status["modules"].items()):
        body.append("<tr><td>{}</td><td class='status {}'>{}</td></tr>".format(
            escape(name), "implemented" if detected else "missing", "yes" if detected else "no"
        ))
    body.extend(
        [
            "</tbody></table>",
            "<h2>Task Sets</h2>",
            html_list(repo_status["task_sets"]),
            "<h2>Tests</h2>",
            html_list(repo_status["tests"]),
            "<h2>Previous Stage Outputs</h2>",
            "<table><thead><tr><th>Directory</th><th>Summary</th><th>Trace</th><th>Video</th></tr></thead><tbody>",
        ]
    )
    for stage in repo_status["stage_outputs"]:
        body.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(stage["path"]),
                "yes" if stage["has_summary"] else "no",
                "yes" if stage["has_trace"] else "no",
                "yes" if stage["has_video"] else "no",
            )
        )
    legacy = repo_status["legacy_visualizations"]
    body.extend(
        [
            "</tbody></table>",
            "<h2>Legacy Visualizations</h2>",
            "<p>Path: <code>{}</code>, PNG count: {}, index.md: {}</p>".format(
                escape(str(legacy["path"] or "-")),
                int(legacy["png_count"]),
                "yes" if legacy["has_index"] else "no",
            ),
            "</main>",
            html_footer(),
        ]
    )
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage1_index(path: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    cards = [
        ("Target templates", summary["template_target_count"]),
        ("Current scoper.benchmark.behavior templates", summary["eb_behavior_template_count"]),
        ("Mapped target templates", summary["mapped_target_count"]),
        ("Template families", summary["family_count"]),
        ("VLM-required templates", summary["vlm_required_count"]),
        ("Mutation templates", summary["mutation_required_count"]),
        ("Schema fields", manifest["schema_top_level_field_count"]),
    ]
    body = [
        html_header("scoper.benchmark.behavior Stage 1 Template Catalog"),
        "<main>",
        "<h1>scoper.benchmark.behavior Stage 1 Template Catalog</h1>",
        "<p class='muted'>Run <code>{}</code>. Visual gate: <strong>{}</strong>.</p>".format(
            escape(str(manifest["run_id"])), escape(str(manifest["visual_gate"]))
        ),
        "<section class='cards'>",
    ]
    for label, value in cards:
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(value, escape(label)))
    body.extend(
        [
            "</section>",
            "<section>",
            "<h2>Artifacts</h2>",
            "<ul>",
            "<li><a href='task_template_schema.html'>Task template schema</a></li>",
            "<li><a href='template_catalog.html'>Template catalog</a></li>",
            "<li><a href='template_family_overview.html'>Template family overview</a></li>",
            "<li><a href='template_dependency_graph.html'>Template dependency graph</a></li>",
            "<li><a href='logs/task_template_schema.json'>Task template schema JSON</a></li>",
            "<li><a href='logs/template_catalog.json'>Template catalog JSON</a></li>",
            "<li><a href='manifest.json'>Manifest JSON</a></li>",
            "</ul>",
            "</section>",
            "<section>",
            "<h2>Stage 1 Scope</h2>",
            "<p>This stage records template roles, predicates, scene affordance dependencies, tool dependencies, oracle teacher families, metrics, and migration mappings. It intentionally does not mark task generation or scene binding as complete.</p>",
            "</section>",
            "</main>",
            html_footer(),
        ]
    )
    path.write_text("\n".join(body), encoding="utf-8")


def write_task_template_schema(path: Path, schema: Mapping[str, Any]) -> None:
    body = [
        html_header("Task Template Schema"),
        "<main>",
        "<h1>Task Template Schema</h1>",
        "<p class='muted'><code>{}</code> / <code>{}</code>. This is the Stage-1 migration template schema, not the materialized runtime task_spec.</p>".format(
            escape(str(schema["schema_name"])),
            escape(str(schema["schema_version"])),
        ),
        "<h2>Top-level Fields</h2>",
        "<table><thead><tr><th>Field</th><th>Required</th><th>Type</th><th>Purpose</th></tr></thead><tbody>",
    ]
    for field in schema["top_level_fields"]:
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(field["name"])),
                "yes" if field["required"] else "no",
                escape(str(field["type"])),
                escape(str(field["purpose"])),
            )
        )
    body.extend(
        [
            "</tbody></table>",
            "<h2>Role Requirement Fields</h2>",
            html_list([str(field) for field in schema["role_requirement_fields"]]),
            "<h2>Predicate Shapes</h2>",
            "<table><thead><tr><th>Shape</th><th>Fields</th><th>Examples</th></tr></thead><tbody>",
        ]
    )
    for shape in schema["predicate_shapes"]:
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(shape["name"])),
                escape(", ".join(shape["fields"])),
                escape(", ".join(shape["examples"])),
            )
        )
    body.extend(
        [
            "</tbody></table>",
            "<h2>Dependency Columns</h2>",
            html_list([str(column) for column in schema["dependency_columns"]]),
            "<h2>Observed Categories</h2>",
            html_list([str(category) for category in schema["observed_categories"]]),
            "<h2>Observed Goal Predicates</h2>",
            html_list([str(predicate) for predicate in schema["observed_goal_predicates"]]),
            "<h2>Observed Tools</h2>",
            html_list([str(tool) for tool in schema["observed_tools"]]),
            "<h2>Notes</h2>",
            html_list([str(note) for note in schema["notes"]]),
            "</main>",
            html_footer(),
        ]
    )
    path.write_text("\n".join(body), encoding="utf-8")


def write_template_catalog(path: Path, entries: list[dict[str, Any]]) -> None:
    body = [
        html_header("Template Catalog"),
        "<main>",
        "<h1>Template Catalog</h1>",
        "<p class='muted'>One card per eb_procthor target template. Dependencies are inferred conservatively from template fields.</p>",
        "<section class='template-grid'>",
    ]
    for entry in entries:
        body.append(
            "<article class='template-card'>"
            "<h2><code>{}</code></h2>"
            "<p class='muted'>{} / {}</p>"
            "<p>{}</p>"
            "<table>"
            "<tr><th>category</th><td>{}</td></tr>"
            "<tr><th>roles</th><td>{}</td></tr>"
            "<tr><th>initial predicates</th><td>{}</td></tr>"
            "<tr><th>goal predicates</th><td>{}</td></tr>"
            "<tr><th>scene features</th><td>{}</td></tr>"
            "<tr><th>tools</th><td>{}</td></tr>"
            "<tr><th>object states</th><td>{}</td></tr>"
            "<tr><th>oracle teacher</th><td>{}</td></tr>"
            "<tr><th>metrics</th><td>{}</td></tr>"
            "<tr><th>VLM</th><td>{}</td></tr>"
            "<tr><th>migration</th><td>{}</td></tr>"
            "<tr><th>notes</th><td>{}</td></tr>"
            "</table>"
            "</article>".format(
                escape(entry["template_id"]),
                escape(entry["filename"]),
                escape(entry["template_version"] or "-"),
                escape(entry["description"] or "-"),
                escape(entry["category"]),
                escape(format_entry_roles(entry) or "-"),
                escape("; ".join(entry["initial_predicates"]) or "-"),
                escape("; ".join(entry["goal_predicates"]) or "-"),
                escape("; ".join(entry["required_scene_features"]) or "-"),
                escape(", ".join(entry["required_tools"]) or "-"),
                escape(", ".join(entry["required_object_states"]) or "-"),
                escape(entry["oracle_teacher"]),
                escape(", ".join(entry["metrics_requirements"]) or "-"),
                escape(entry["vlm_observation"]),
                escape(entry["migration_status"] + (": " + entry["matched_eb_template_id"] if entry["matched_eb_template_id"] else "")),
                escape("; ".join(entry["notes"]) or "-"),
            )
        )
    body.extend(["</section>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def format_entry_roles(entry: Mapping[str, Any]) -> str:
    roles = entry.get("roles") or []
    if roles and isinstance(roles[0], Mapping):
        return ", ".join(str(role.get("role") or "") for role in roles)
    return ", ".join(str(role) for role in roles)


def write_template_family_overview(path: Path, families: list[dict[str, Any]]) -> None:
    body = [
        html_header("Template Family Overview"),
        "<main>",
        "<h1>Template Family Overview</h1>",
        "<table><thead><tr><th>Family</th><th>Templates</th><th>Mapped</th><th>Tools</th><th>Top scene features</th><th>Mutation</th><th>VLM required</th></tr></thead><tbody>",
    ]
    for family in families:
        body.append(
            "<tr><td><code>{}</code></td><td>{}<div class='muted'>{}</div></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(family["category"]),
                int(family["template_count"]),
                escape(", ".join(family["template_ids"])),
                int(family["mapped_count"]),
                escape(", ".join(family["required_tools"]) or "-"),
                escape("; ".join(family["top_scene_features"]) or "-"),
                int(family["mutation_required_count"]),
                int(family["vlm_required_count"]),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_template_dependency_graph(path: Path, graph: Mapping[str, Any]) -> None:
    by_template: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges", []):
        source = str(edge.get("source") or "")
        if not source.startswith("template:"):
            continue
        by_template[source.split(":", 1)[1]].append(edge)
    label_by_id = {node["id"]: node["label"] for node in graph.get("nodes", [])}
    type_by_id = {node["id"]: node["type"] for node in graph.get("nodes", [])}
    body = [
        html_header("Template Dependency Graph"),
        "<main>",
        "<h1>Template Dependency Graph</h1>",
        "<p class='muted'>Tabular graph view: each row is a template and each dependency edge inferred from the source template.</p>",
        "<table><thead><tr><th>Template</th><th>Dependencies</th></tr></thead><tbody>",
    ]
    for template_id in sorted(by_template):
        deps = []
        for edge in by_template[template_id]:
            target = edge["target"]
            deps.append("{}: {}".format(type_by_id.get(target, "dependency"), label_by_id.get(target, target)))
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(template_id),
                escape("; ".join(sorted(deps)) or "-"),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage2_3_index(path: Path, summary: Mapping[str, Any]) -> None:
    cards = [
        ("Scene profiles", summary["scene_count"]),
        ("Target templates", summary["template_target_count"]),
        ("Scene-template pairs", summary["scene_template_pair_count"]),
        ("Matched pairs", summary["matched_pair_count"]),
        ("Rejected pairs", summary["rejected_pair_count"]),
        ("Profile-only", summary.get("matched_profile_only_pair_count", 0)),
        ("Needs spawn", summary.get("matched_requires_spawn_pair_count", 0)),
        ("Needs runtime validation", summary.get("matched_requires_runtime_validation_pair_count", 0)),
        ("Spawn bindings", summary["spawn_binding_count"]),
    ]
    body = [
        html_header("scoper.benchmark.behavior Stage 2/3 Scene Profile Match"),
        "<main>",
        "<h1>scoper.benchmark.behavior Stage 2/3 Scene Profile Match</h1>",
        "<p class='muted'>No-reset matching report. Runtime requirements are preserved instead of being silently treated as validated.</p>",
        "<section class='cards'>",
    ]
    for label, value in cards:
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(value, escape(label)))
    body.extend(
        [
            "</section>",
            "<section>",
            "<h2>Artifacts</h2>",
            "<ul>",
            "<li><a href='scene_profiles.html'>Scene profiles</a></li>",
            "<li><a href='scene_template_matrix.html'>Scene-template matrix</a></li>",
            "<li><a href='logs/scene_profiles.json'>Scene profiles JSON</a></li>",
            "<li><a href='logs/scene_template_matches.json'>Scene-template matches JSON</a></li>",
            "<li><a href='logs/stage2_3_summary.json'>Summary JSON</a></li>",
            "<li><a href='manifest.json'>Manifest JSON</a></li>",
            "</ul>",
            "</section>",
            "<section>",
            "<h2>Match Status</h2>",
            "<p class='muted'><code>matched_profile_only</code> means the binding is supported by the file profile alone. <code>matched_requires_spawn</code> needs reset-time asset spawning. <code>matched_requires_runtime_validation</code> needs runtime checks such as state override, visibility, reachable map, or relocation.</p>",
            "<table><thead><tr><th>Status</th><th>Pair count</th></tr></thead><tbody>",
        ]
    )
    for status, count in summary.get("match_status_counts", {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(status), int(count)))
    body.extend(
        [
            "</tbody></table>",
            "<h2>Runtime Requirements</h2>",
            "<table><thead><tr><th>Capability</th><th>Pair count</th></tr></thead><tbody>",
        ]
    )
    for capability, count in summary["runtime_capability_counts"].items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(capability), int(count)))
    body.extend(
        [
            "</tbody></table>",
            "<h2>Reachability Status</h2>",
            "<table><thead><tr><th>Scene</th><th>Status</th></tr></thead><tbody>",
        ]
    )
    for scene, status in summary["reachable_statuses"].items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(scene), escape(status)))
    body.extend(["</tbody></table>", "</section>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_scene_profiles_html(path: Path, profiles: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("BEHAVIOR Scene Profiles"),
        "<main>",
        "<h1>BEHAVIOR Scene Profiles</h1>",
        "<table><thead><tr><th>Scene</th><th>Objects</th><th>Rooms</th><th>Spawnable Catalog</th><th>Key Types</th><th>Reachable</th></tr></thead><tbody>",
    ]
    key_types = ("Fridge", "Microwave", "SinkBasin", "Faucet", "CounterTop", "Cabinet", "Shelf")
    for profile in profiles:
        type_index = profile.get("object_type_index") or {}
        type_bits = ["{}:{}".format(key, len(type_index.get(key) or [])) for key in key_types if type_index.get(key)]
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(profile.get("scene_model"))),
                int(profile.get("object_count") or 0),
                int(profile.get("room_count") or 0),
                len(profile.get("spawnable_catalog") or []),
                escape(", ".join(type_bits) or "-"),
                escape(str((profile.get("reachable") or {}).get("status"))),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_scene_template_matrix_html(path: Path, records: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Scene-Template Match Matrix"),
        "<main>",
        "<h1>Scene-Template Match Matrix</h1>",
        "<p class='muted'>Profile matches are split by validation debt: profile-only, spawn-required, or runtime-validation-required.</p>",
        "<table><thead><tr><th>Scene</th><th>Template</th><th>Category</th><th>Status</th><th>Binding Count</th><th>Candidate Counts</th><th>Init Mutations</th><th>Runtime Requirements</th><th>Rejection</th></tr></thead><tbody>",
    ]
    for record in sorted(records, key=lambda item: (str(item.get("scene_model")), str(item.get("template_id")))):
        status = str(record.get("status"))
        candidates = ", ".join(
            "{}={}".format(role, count)
            for role, count in sorted((record.get("candidate_counts_by_role") or {}).items())
        )
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code><div class='muted'>{}</div></td><td>{}</td><td class='status {}'>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(record.get("scene_model"))),
                escape(str(record.get("template_id"))),
                escape(str(record.get("template_file") or "")),
                escape(str(record.get("category"))),
                status_css_class(status),
                escape(status),
                int(record.get("binding_count") or 0),
                escape(candidates or "-"),
                escape(", ".join(record.get("required_init_mutations") or []) or "-"),
                escape(", ".join(record.get("runtime_capabilities_required") or []) or "-"),
                escape(", ".join(record.get("rejection_reasons") or []) or "-"),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage4_index(path: Path, summary: Mapping[str, Any]) -> None:
    cards = [
        ("Generated tasks", summary["generated_total"]),
        ("Rejected candidates", summary["rejected_total"]),
        ("Structure passed", summary["structure_passed"]),
        ("Family specs", summary.get("generated_with_family_spec", 0)),
        ("Disturbances", summary.get("generated_with_disturbance_spec", 0)),
        ("Failures", summary.get("generated_with_failure_spec", 0)),
    ]
    body = [
        html_header("scoper.benchmark.behavior Stage 4 Task Generation"),
        "<main>",
        "<h1>scoper.benchmark.behavior Stage 4 Task Generation</h1>",
        "<p class='muted'>Profile-driven task spec materialization from Stage 3 match records. Reset, mutation, oracle, and rendering validation are deferred to later stages.</p>",
        "<section class='cards'>",
    ]
    for label, value in cards:
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(value, escape(label)))
    body.extend(
        [
            "</section>",
            "<section><h2>Artifacts</h2><ul>",
            "<li><a href='task_generation_gallery.html'>Task generation gallery</a></li>",
            "<li><a href='rejected_task_gallery.html'>Rejected task gallery</a></li>",
            "<li><a href='logs/task_bank.json'>Task bank JSON</a></li>",
            "<li><a href='logs/structure_validation.json'>Structure validation JSON</a></li>",
            "<li><a href='manifest.json'>Manifest JSON</a></li>",
            "</ul></section>",
            "<section><h2>Generated By Readiness</h2>",
            "<table><thead><tr><th>Readiness</th><th>Tasks</th></tr></thead><tbody>",
        ]
    )
    for readiness, count in summary["generated_by_readiness"].items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(readiness)), int(count)))
    body.extend(
        [
            "</tbody></table>",
            "<h2>Readiness Debt</h2>",
            "<table><thead><tr><th>Debt</th><th>Tasks</th></tr></thead><tbody>",
        ]
    )
    for debt, count in summary.get("readiness_debt", {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(debt)), int(count)))
    body.extend(
        [
            "</tbody></table>",
            "<h2>Generated By Family</h2>",
            "<table><thead><tr><th>Family</th><th>Tasks</th></tr></thead><tbody>",
        ]
    )
    for family, count in summary.get("generated_by_family", {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(family)), int(count)))
    body.extend(["</tbody></table>", "<h2>Generated By Template</h2>", "<table><thead><tr><th>Template</th><th>Tasks</th></tr></thead><tbody>"])
    for template_id, count in summary["generated_by_template"].items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(template_id)), int(count)))
    body.extend(["</tbody></table>", "<h2>Rejection Codes</h2>", "<table><thead><tr><th>Code</th><th>Count</th></tr></thead><tbody>"])
    for code, count in summary["rejection_codes"].items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(code)), int(count)))
    body.extend(["</tbody></table>", "</section>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage4_5_index(path: Path, summary: Mapping[str, Any]) -> None:
    cards = [
        ("Generated tasks", summary["generated_total"]),
        ("Rejected candidates", summary["rejected_total"]),
        ("Structure passed", summary["structure_passed"]),
        ("Reset passed", summary["reset_passed"]),
        ("Init passed", summary["init_passed"]),
        ("Oracle passed", summary["oracle_passed"]),
    ]
    body = [
        html_header("scoper.benchmark.behavior Stage 4/5 Task Generation Reset"),
        "<main>",
        "<h1>scoper.benchmark.behavior Stage 4/5 Task Generation Reset</h1>",
        "<p class='muted'>Profile-driven gold task generation plus symbolic reset/oracle validation.</p>",
        "<section class='cards'>",
    ]
    for label, value in cards:
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(value, escape(label)))
    body.extend(
        [
            "</section>",
            "<section><h2>Artifacts</h2><ul>",
            "<li><a href='task_generation_gallery.html'>Task generation gallery</a></li>",
            "<li><a href='rejected_task_gallery.html'>Rejected task gallery</a></li>",
            "<li><a href='reset_validation.html'>Reset validation</a></li>",
            "<li><a href='logs/task_bank.json'>Task bank JSON</a></li>",
            "<li><a href='logs/reset_validation.json'>Reset validation JSON</a></li>",
            "<li><a href='manifest.json'>Manifest JSON</a></li>",
            "</ul></section>",
            "<section><h2>Generated By Template</h2>",
            "<table><thead><tr><th>Template</th><th>Tasks</th></tr></thead><tbody>",
        ]
    )
    for template_id, count in summary["generated_by_template"].items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(template_id)), int(count)))
    body.extend(["</tbody></table>", "<h2>Rejection Codes</h2>", "<table><thead><tr><th>Code</th><th>Count</th></tr></thead><tbody>"])
    for code, count in summary["rejection_codes"].items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(code)), int(count)))
    body.extend(["</tbody></table>", "</section>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage4_task_gallery(
    path: Path,
    tasks: list[Mapping[str, Any]],
    validation_results: list[Mapping[str, Any]],
) -> None:
    result_by_id = {result.get("task_id"): result for result in validation_results}
    body = [
        html_header("Task Generation Gallery"),
        "<main>",
        "<h1>Task Generation Gallery</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Scene</th><th>Instruction</th><th>Readiness</th><th>Debt</th><th>Family</th><th>Schedulers</th><th>Roles</th><th>Init</th><th>Goal</th><th>Validator</th></tr></thead><tbody>",
    ]
    for task in tasks:
        result = result_by_id.get(task.get("id"), {})
        generation = task.get("generation") or {}
        family_spec = task.get("family_spec") or {}
        schedulers = []
        if task.get("disturbance_spec"):
            schedulers.append("disturbance:{}".format(len(task.get("disturbance_spec") or [])))
        if task.get("failure_spec"):
            schedulers.append("failure:{}".format(len(task.get("failure_spec") or [])))
        unsupported_scheduler_specs = generation.get("unsupported_scheduler_specs") or []
        if unsupported_scheduler_specs:
            schedulers.append("unsupported:{}".format(len(unsupported_scheduler_specs)))
        roles = "; ".join(
            "{}={} ({})".format(role, data.get("object_type"), data.get("source_kind"))
            for role, data in sorted((task.get("roles") or {}).items())
        )
        init = "; ".join("{}({})".format(item.get("predicate"), ",".join(str(v) for v in item.get("args", []))) for item in task.get("init", [])) or "-"
        goal = "; ".join("{}({})".format(pred.get("type"), ",".join("{}={}".format(k, v) for k, v in (pred.get("args") or {}).items())) for pred in task.get("goal", {}).get("predicates", []))
        if "reset_ok" in result:
            if "stage5_reset_mutation_ok" in result:
                ok = bool(result.get("stage5_reset_mutation_ok"))
                label = str(result.get("stage5_bucket") or ("passed" if ok else "failed"))
            else:
                ok = all(result.get(key) for key in ("structure_ok", "reset_ok", "init_ok", "initial_goal_false", "oracle_ok"))
                label = "passed" if ok else str(result.get("failure_code") or "failed")
        else:
            ok = bool(result.get("structure_ok", True))
            label = "structure_passed" if ok else str(result.get("failure_code") or "failed")
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class='status {}'>{}</td></tr>".format(
                escape(str(task.get("id"))),
                escape(str((task.get("template") or {}).get("template_id"))),
                escape(str((task.get("source") or {}).get("scene_model"))),
                escape(str(task.get("instruction"))),
                escape(str(generation.get("stage4_readiness") or "-")),
                escape(", ".join(str(item) for item in generation.get("stage4_readiness_debt") or []) or "-"),
                escape(str(family_spec.get("family") or "-")),
                escape(", ".join(schedulers) or "-"),
                escape(roles),
                escape(init),
                escape(goal),
                "implemented" if ok else "missing",
                escape(label),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage4_rejected_gallery(path: Path, rejected: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Rejected Task Gallery"),
        "<main>",
        "<h1>Rejected Task Gallery</h1>",
        "<p class='muted'>Rejected here means this Stage 4 materializer could not safely emit a valid task spec for the candidate, or the candidate was outside an explicitly requested subset.</p>",
        "<table><thead><tr><th>Scene</th><th>Template</th><th>Category</th><th>Failure Code</th><th>Extra</th></tr></thead><tbody>",
    ]
    for item in rejected:
        extra = {k: v for k, v in item.items() if k not in {"scene_model", "template_id", "category", "failure_code", "stage3_status"}}
        body.append(
            "<tr><td>{}</td><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(item.get("scene_model"))),
                escape(str(item.get("template_id"))),
                escape(str(item.get("category"))),
                escape(str(item.get("failure_code"))),
                escape(json.dumps(extra, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage5_reset_validation(path: Path, validation_results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Reset Validation"),
        "<main>",
        "<h1>Reset Validation</h1>",
        "<table><thead><tr><th>Task</th><th>Structure</th><th>Reset</th><th>Init</th><th>Goal Initially False</th><th>Oracle</th><th>Failure</th></tr></thead><tbody>",
    ]
    for result in validation_results:
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                yes_no(result.get("structure_ok")),
                yes_no(result.get("reset_ok")),
                yes_no(result.get("init_ok")),
                yes_no(result.get("initial_goal_false")),
                yes_no(result.get("oracle_ok")),
                escape(str(result.get("failure_code") or "-")),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage5_reset_mutation_index(path: Path, summary: Mapping[str, Any]) -> None:
    cards = [
        ("Generated tasks", summary["generated_total"]),
        ("Validated tasks", summary["validated_total"]),
        ("Reset/mutation passed", summary["reset_mutation_passed"]),
        ("Pass ratio", "{:.1%}".format(float(summary.get("reset_mutation_pass_ratio") or 0.0))),
        ("Deterministic reset", summary["deterministic_reset_passed"]),
        ("Init passed", summary["init_passed"]),
        ("Reachability passed", summary.get("reachability_passed", 0)),
        ("Reachability skipped", summary.get("reachability_skipped", 0)),
        ("Mutation replay passed", summary["mutation_replay_passed"]),
        ("Mutation runtime complete", summary.get("mutation_runtime_complete", 0)),
    ]
    body = [
        html_header("scoper.benchmark.behavior Stage 5 Reset Mutation"),
        "<main>",
        "<h1>scoper.benchmark.behavior Stage 5 Reset Mutation</h1>",
        "<p class='muted'>Symbolic reset/mutation validation for the full profile-generated task bank.</p>",
        "<section class='cards'>",
    ]
    for label, value in cards:
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(value, escape(label)))
    body.extend(
        [
            "</section>",
            "<section><h2>Artifacts</h2><ul>",
            "<li><a href='stage5_validation.html'>Stage 5 validation</a></li>",
            "<li><a href='reset_before_after.html'>Reset before/after</a> (<a href='reset_before_after.png'>PNG contact sheet</a>)</li>",
            "<li><a href='state_predicate_check.html'>State predicate check</a></li>",
            "<li><a href='mutation_diff.html'>Mutation diff</a></li>",
            "<li><a href='mutation_replay.html'>Mutation replay</a></li>",
            "<li><a href='task_generation_gallery.html'>Task generation gallery</a></li>",
            "<li><a href='logs/stage5_summary.json'>Summary JSON</a></li>",
            "<li><a href='manifest.json'>Manifest JSON</a></li>",
            "</ul></section>",
            "<section><h2>Stage 5 Buckets</h2><table><thead><tr><th>Bucket</th><th>Tasks</th></tr></thead><tbody>",
        ]
    )
    for bucket, count in summary.get("stage5_buckets", {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(bucket)), int(count)))
    body.extend(["</tbody></table>", "<h2>Deferred Codes</h2>", "<table><thead><tr><th>Code</th><th>Tasks</th></tr></thead><tbody>"])
    for code, count in summary.get("deferred_codes", {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(code)), int(count)))
    body.extend(["</tbody></table>", "<h2>Failure Codes</h2>", "<table><thead><tr><th>Code</th><th>Tasks</th></tr></thead><tbody>"])
    for code, count in summary.get("failure_codes", {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(code)), int(count)))
    if not summary.get("failure_codes"):
        body.append("<tr><td><code>-</code></td><td>0</td></tr>")
    body.extend(["</tbody></table>", "</section>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage5_validation_table(path: Path, validation_results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 5 Validation"),
        "<main>",
        "<h1>Stage 5 Validation</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Scene</th><th>Bucket</th><th>Reset</th><th>Deterministic</th><th>Init</th><th>Goal Initially False</th><th>Geometry</th><th>Reachability</th><th>Mutation Spec</th><th>Mutation Replay</th><th>Runtime Complete</th><th>Oracle Gate</th><th>Deferred</th><th>Failure</th></tr></thead><tbody>",
    ]
    for result in validation_results:
        ok = bool(result.get("stage5_reset_mutation_ok"))
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td class='status {}'>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(str(result.get("scene_model"))),
                "implemented" if ok else "partial",
                escape(str(result.get("stage5_bucket"))),
                yes_no(result.get("reset_ok")),
                yes_no(result.get("deterministic_reset")),
                yes_no(result.get("init_ok")),
                yes_no(result.get("initial_goal_false")),
                yes_no(result.get("geometry_ok")),
                yes_no(result.get("reachability_ok")),
                yes_no(result.get("mutation_spec_ok")),
                yes_no(result.get("mutation_replay_ok")),
                yes_no(result.get("mutation_runtime_complete")),
                escape(str(result.get("oracle_gate") or "-")),
                escape(", ".join(result.get("deferred_codes") or []) or "-"),
                escape(str(result.get("failure_code") or "-")),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage5_reset_before_after(path: Path, validation_results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Reset Before After"),
        "<main>",
        "<h1>Reset Before After</h1>",
        "<table><thead><tr><th>Task</th><th>Applied States</th><th>Applied Spatial Relations</th><th>After Reset Objects</th></tr></thead><tbody>",
    ]
    for result in validation_results:
        reset = result.get("reset_before_after") or {}
        placement = reset.get("placement_report") or result.get("placement_report") or {}
        after_objects = ((reset.get("after_reset") or {}).get("objects") or {})
        object_summary = "; ".join(
            "{}@{}".format(name, data.get("center"))
            for name, data in list(sorted(after_objects.items()))[:8]
        )
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(json.dumps(placement.get("applied_states") or [], sort_keys=True)),
                escape(json.dumps(placement.get("applied_spatial_relations") or [], sort_keys=True)),
                escape(object_summary),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage5_state_predicate_check(path: Path, validation_results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("State Predicate Check"),
        "<main>",
        "<h1>State Predicate Check</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Checks</th><th>Deferred Assertions</th></tr></thead><tbody>",
    ]
    for result in validation_results:
        checks = result.get("state_predicate_checks") or []
        assertions = result.get("deferred_assertion_checks") or []
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(json.dumps(checks, sort_keys=True)),
                escape(json.dumps(assertions, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage5_mutation_diff(path: Path, validation_results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Mutation Diff"),
        "<main>",
        "<h1>Mutation Diff</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Deferred</th><th>Diff</th></tr></thead><tbody>",
    ]
    for result in validation_results:
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(", ".join(result.get("deferred_codes") or []) or "-"),
                escape(json.dumps(result.get("mutation_diff") or [], sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage5_mutation_replay(path: Path, validation_results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Mutation Replay"),
        "<main>",
        "<h1>Mutation Replay</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Replay</th></tr></thead><tbody>",
    ]
    for result in validation_results:
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(json.dumps(result.get("mutation_replay") or [], sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage5_reset_before_after_png(path: Path, validation_results: list[Mapping[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    rows = max(1, min(12, len(validation_results)))
    cell_w, cell_h = 360, 220
    img = Image.new("RGB", (cell_w * 2, cell_h * rows), "white")
    draw = ImageDraw.Draw(img)
    for row, result in enumerate(validation_results[:rows]):
        y0 = row * cell_h
        draw.rectangle([0, y0, cell_w * 2 - 1, y0 + cell_h - 1], outline=(215, 221, 229))
        draw.text((8, y0 + 6), str(result.get("template_id"))[:44], fill=(23, 32, 42))
        reset = result.get("reset_before_after") or {}
        before = reset.get("before_role_spec") or {}
        after = ((reset.get("after_reset") or {}).get("objects") or {})
        bounds = _plot_bounds(before, after)
        _draw_stage5_points(draw, before, bounds, 0, y0 + 26, cell_w, cell_h - 28, "before")
        _draw_stage5_points(draw, after, bounds, cell_w, y0 + 26, cell_w, cell_h - 28, "after")
    img.save(path)


def _plot_bounds(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[float, float, float, float]:
    xs = []
    ys = []
    for data in list(before.values()) + list(after.values()):
        center = data.get("center")
        if isinstance(center, list) and len(center) >= 2:
            xs.append(float(center[0]))
            ys.append(float(center[1]))
    if not xs or not ys:
        return (-1.0, -1.0, 1.0, 1.0)
    return (min(xs) - 0.5, min(ys) - 0.5, max(xs) + 0.5, max(ys) + 0.5)


def _draw_stage5_points(draw: Any, data: Mapping[str, Any], bounds: tuple[float, float, float, float], x0: int, y0: int, w: int, h: int, label: str) -> None:
    min_x, min_y, max_x, max_y = bounds
    draw.text((x0 + 8, y0 + 4), label, fill=(93, 109, 126))
    draw.rectangle([x0 + 8, y0 + 22, x0 + w - 8, y0 + h - 8], outline=(215, 221, 229))
    colors = [(20, 94, 168), (190, 80, 40), (50, 130, 80), (120, 80, 170), (150, 110, 30)]
    for idx, (name, item) in enumerate(sorted(data.items())):
        center = item.get("center")
        if not isinstance(center, list) or len(center) < 2:
            continue
        px = x0 + 12 + int((float(center[0]) - min_x) / max(max_x - min_x, 1e-6) * (w - 28))
        py = y0 + 26 + int((float(center[1]) - min_y) / max(max_y - min_y, 1e-6) * (h - 40))
        color = colors[idx % len(colors)]
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
        draw.text((px + 6, py - 5), str(name)[:18], fill=color)


def write_stage6_execution_index(
    path: Path,
    stage5_summary: Mapping[str, Any],
    stage6_summary: Mapping[str, Any],
    tool_micro_summary: Mapping[str, Any],
) -> None:
    cards = [
        ("Generated", stage6_summary["generated_total"]),
        ("Stage 5 admitted", stage6_summary["stage5_admitted"]),
        ("Teacher ready", stage6_summary["teacher_ready"]),
        ("Strict teachers", stage6_summary.get("strict_teacher_ready", 0)),
        ("Predicate fallback", stage6_summary.get("predicate_fallback_ready", 0)),
        ("Execution attempted", stage6_summary["execution_attempted"]),
        ("Execution passed", stage6_summary["execution_passed"]),
        ("Runtime complete", stage6_summary.get("execution_passed_runtime_complete", 0)),
        ("Execution pass ratio", "{:.1%}".format(float(stage6_summary.get("execution_pass_ratio") or 0.0))),
        ("Runtime complete ratio", "{:.1%}".format(float(stage6_summary.get("execution_runtime_complete_ratio") or 0.0))),
        ("Tool micro cases", tool_micro_summary.get("case_count", 0)),
        ("Tool cases match", yes_no(tool_micro_summary.get("all_cases_match_expected"))),
    ]
    body = [
        html_header("scoper.benchmark.behavior Stage 6 Execution"),
        "<main>",
        "<h1>scoper.benchmark.behavior Stage 6 Execution</h1>",
        "<p class='muted'>Lightweight symbolic oracle/teacher execution through the public tool API.</p>",
        "<section class='cards'>",
    ]
    for label, value in cards:
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(value, escape(label)))
    body.extend(
        [
            "</section>",
            "<section><h2>Artifacts</h2><ul>",
            "<li><a href='teacher_readiness.html'>Teacher readiness</a></li>",
            "<li><a href='stage6_execution.html'>Stage 6 execution table</a></li>",
            "<li><a href='tool_trace.html'>Tool trace</a></li>",
            "<li><a href='predicate_progress.html'>Predicate progress</a></li>",
            "<li><a href='metrics_dashboard.html'>Metrics dashboard</a></li>",
            "<li><a href='failure_breakdown.html'>Failure breakdown</a></li>",
            "<li><a href='tool_catalog.html'>Tool catalog</a></li>",
            "<li><a href='tool_micro_cases.html'>Tool micro cases</a></li>",
            "<li><a href='stage5_validation.html'>Stage 5 validation reused by Stage 6</a></li>",
            "<li><a href='logs/stage6_summary.json'>Summary JSON</a></li>",
            "<li><a href='manifest.json'>Manifest JSON</a></li>",
            "</ul></section>",
            "<section><h2>Stage 6 Buckets</h2><table><thead><tr><th>Bucket</th><th>Tasks</th></tr></thead><tbody>",
        ]
    )
    for bucket, count in stage6_summary.get("stage6_buckets", {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(bucket)), int(count)))
    body.extend(["</tbody></table>", "<h2>Stage 5 Carryover</h2>", "<table><tbody>"])
    body.append("<tr><th>Stage 5 reset/mutation pass ratio</th><td>{:.1%}</td></tr>".format(float(stage5_summary.get("reset_mutation_pass_ratio") or 0.0)))
    body.append("<tr><th>Stage 5 deferred codes</th><td>{}</td></tr>".format(escape(json.dumps(stage5_summary.get("deferred_codes") or {}, sort_keys=True))))
    body.extend(["</tbody></table>", "</section>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage6_teacher_readiness(path: Path, results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 6 Teacher Readiness"),
        "<main>",
        "<h1>Stage 6 Teacher Readiness</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Category</th><th>Stage 5</th><th>Teacher Ready</th><th>Source</th><th>Type</th><th>Deferred</th><th>Failure</th></tr></thead><tbody>",
    ]
    for result in results:
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(str(result.get("category"))),
                yes_no(result.get("stage5_admitted")),
                yes_no(result.get("teacher_ready")),
                escape(str(result.get("teacher_source") or "-")),
                escape(str(result.get("teacher_type") or "-")),
                escape(", ".join(result.get("deferred_codes") or []) or "-"),
                escape(str(result.get("failure_code") or "-")),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage6_execution_table(path: Path, results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 6 Execution"),
        "<main>",
        "<h1>Stage 6 Execution</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Scene</th><th>Bucket</th><th>Teacher</th><th>Attempted</th><th>Goal</th><th>Steps</th><th>Failure</th><th>Trace</th></tr></thead><tbody>",
    ]
    for result in results:
        ok = bool(result.get("execution_ok"))
        trace = result.get("trace_path")
        trace_link = "<a href='{}'>trace</a>".format(escape(_relative_path_for_html(path.parent, trace))) if trace else "-"
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td class='status {}'>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(str(result.get("scene_model"))),
                "visually_verified" if ok else ("blocked" if result.get("stage6_bucket") == "teacher_deferred" else "partial"),
                escape(str(result.get("stage6_bucket"))),
                escape(str(result.get("teacher_source") or "-")),
                yes_no(result.get("execution_attempted")),
                yes_no(result.get("goal_success")),
                int(result.get("transition_count") or 0),
                escape(str(result.get("failure_code") or "-")),
                trace_link,
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage6_tool_trace(path: Path, results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 6 Tool Trace"),
        "<main>",
        "<h1>Stage 6 Tool Trace</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Tool Calls</th></tr></thead><tbody>",
    ]
    for result in results:
        calls = result.get("tool_calls") or []
        compact = [
            {
                "step": item.get("step"),
                "tool": item.get("tool_name"),
                "success": item.get("success"),
                "error": item.get("error_type"),
                "progress": item.get("task_progress"),
            }
            for item in calls
        ]
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(json.dumps(compact, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage6_predicate_progress(path: Path, results: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 6 Predicate Progress"),
        "<main>",
        "<h1>Stage 6 Predicate Progress</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Progress</th></tr></thead><tbody>",
    ]
    for result in results:
        progress = [
            {
                "step": item.get("step"),
                "tool": item.get("tool_name"),
                "task_success": item.get("task_success"),
                "task_progress": item.get("task_progress"),
                "satisfied": item.get("satisfied"),
                "total": item.get("total"),
            }
            for item in result.get("predicate_progress") or []
        ]
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(json.dumps(progress, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage6_metrics_dashboard(
    path: Path,
    summary: Mapping[str, Any],
    results: list[Mapping[str, Any]],
) -> None:
    body = [
        html_header("Stage 6 Metrics"),
        "<main>",
        "<h1>Stage 6 Metrics</h1>",
        "<section class='cards'>",
    ]
    for label, key in (
        ("Attempted", "execution_attempted"),
        ("Passed", "execution_passed"),
        ("Failed", "execution_failed"),
        ("Pass ratio", "execution_pass_ratio"),
        ("Overall Stage5 ratio", "overall_stage5_pass_ratio"),
    ):
        value = summary.get(key)
        if isinstance(value, float):
            value = "{:.1%}".format(value)
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(value, escape(label)))
    body.extend(["</section>", "<section><h2>Per Task Metrics</h2><table><thead><tr><th>Task</th><th>Template</th><th>Status</th><th>Metrics</th></tr></thead><tbody>"])
    for result in results:
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(str(result.get("stage6_bucket"))),
                escape(json.dumps(result.get("metrics") or {}, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</section>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage6_failure_breakdown(
    path: Path,
    summary: Mapping[str, Any],
    results: list[Mapping[str, Any]],
) -> None:
    body = [
        html_header("Stage 6 Failure Breakdown"),
        "<main>",
        "<h1>Stage 6 Failure Breakdown</h1>",
        "<h2>Failure Codes</h2><table><thead><tr><th>Code</th><th>Tasks</th></tr></thead><tbody>",
    ]
    for code, count in summary.get("failure_codes", {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(code)), int(count)))
    if not summary.get("failure_codes"):
        body.append("<tr><td><code>-</code></td><td>0</td></tr>")
    body.extend(["</tbody></table>", "<h2>Failures</h2><table><thead><tr><th>Task</th><th>Template</th><th>Bucket</th><th>Code</th><th>Reason</th></tr></thead><tbody>"])
    for result in results:
        if not result.get("failure_code"):
            continue
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(str(result.get("stage6_bucket"))),
                escape(str(result.get("failure_code"))),
                escape(str(result.get("failure_reason") or "-")),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage6_tool_catalog(path: Path, tool_specs: list[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 6 Tool Catalog"),
        "<main>",
        "<h1>Stage 6 Tool Catalog</h1>",
        "<table><thead><tr><th>Tool</th><th>Description</th><th>Parameters</th></tr></thead><tbody>",
    ]
    for spec in tool_specs:
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(spec.get("name"))),
                escape(str(spec.get("description") or "")),
                escape(json.dumps(spec.get("parameters") or {}, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_stage6_tool_micro_cases(
    path: Path,
    cases: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    body = [
        html_header("Stage 6 Tool Micro Cases"),
        "<main>",
        "<h1>Stage 6 Tool Micro Cases</h1>",
        "<p class='muted'>Synthetic symbolic success/failure examples for public tool transition schema.</p>",
        "<h2>Coverage</h2><table><thead><tr><th>Tool</th><th>Cases</th><th>Success Examples</th><th>Failure Examples</th><th>All Expected</th></tr></thead><tbody>",
    ]
    for tool, item in (summary.get("by_tool") or {}).items():
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(tool)),
                int(item.get("cases") or 0),
                int(item.get("success_examples") or 0),
                int(item.get("failure_examples") or 0),
                yes_no(item.get("all_match_expected")),
            )
        )
    body.extend(["</tbody></table>", "<h2>Cases</h2><table><thead><tr><th>Case</th><th>Tool</th><th>Expected</th><th>Success</th><th>Error</th><th>State Delta</th></tr></thead><tbody>"])
    for case in cases:
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(case.get("case_id"))),
                escape(str(case.get("tool"))),
                yes_no(case.get("expected_success")),
                yes_no(case.get("success")),
                escape(str(case.get("error_type") or "-")),
                escape(json.dumps(case.get("state_delta") or {}, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def _relative_path_for_html(base_dir: Path, target: Any) -> str:
    if not target:
        return ""
    try:
        return str(Path(target).resolve().relative_to(base_dir.resolve()))
    except Exception:
        return str(target)


def yes_no(value: Any) -> str:
    if value is None:
        return "-"
    return "yes" if bool(value) else "no"


def status_css_class(status: str) -> str:
    if status == "matched_profile_only":
        return "visually_verified"
    if status == "matched_requires_spawn":
        return "partial"
    if status == "matched_requires_runtime_validation":
        return "blocked"
    return "missing"


def html_header(title: str) -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f8fa;
  --text: #17202a;
  --muted: #5d6d7e;
  --line: #d7dde5;
  --card: #ffffff;
  --missing: #f6d2d2;
  --partial: #ffe5b4;
  --implemented: #cfe8d5;
  --verified: #b7d7f2;
  --blocked: #ded7ef;
}}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width: 1320px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin-top: 28px; font-size: 20px; }}
a {{ color: #145ea8; }}
code {{ background: #eef2f6; border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px; }}
table {{ border-collapse: collapse; width: 100%; background: var(--card); border: 1px solid var(--line); }}
th, td {{ border: 1px solid var(--line); padding: 8px 10px; vertical-align: top; text-align: left; }}
th {{ background: #eef2f6; position: sticky; top: 0; z-index: 1; }}
.matrix th {{ white-space: nowrap; }}
.matrix td {{ min-width: 92px; }}
.muted {{ color: var(--muted); font-size: 12px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
.cards article {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
.template-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }}
.template-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
.template-card h2 {{ margin-top: 0; }}
.template-card table {{ font-size: 12px; }}
.template-card th {{ width: 140px; position: static; }}
.num {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
.status {{ font-weight: 650; text-align: center; }}
.status.missing {{ background: var(--missing); }}
.status.partial {{ background: var(--partial); }}
.status.implemented {{ background: var(--implemented); }}
.status.visually_verified {{ background: var(--verified); }}
.status.blocked {{ background: var(--blocked); }}
</style>
</head>
<body>""".format(escape(title))


def html_footer() -> str:
    return "</body>\n</html>"


def html_list(items: list[str]) -> str:
    if not items:
        return "<p class='muted'>None detected.</p>"
    return "<ul>{}</ul>".format("".join("<li><code>{}</code></li>".format(escape(item)) for item in items))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
