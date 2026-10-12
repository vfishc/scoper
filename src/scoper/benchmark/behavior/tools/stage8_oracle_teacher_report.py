"""Stage 8 oracle/teacher family parity report.

Stage 8 reuses the Stage 5 symbolic admission gate and Stage 6 public-tool
execution loop, but applies the stricter EB-ProcTHOR teacher contract:

- family categories must route to registered family teachers;
- legacy categories route to the legacy OracleBrain;
- predicate fallback is treated as an early-stage convenience and must be zero;
- every attempted oracle trace is written as a gold-trace candidate for VLM parity.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv, MODEL_VISIBLE_TOOLS
from scoper.benchmark.behavior.brains.teachers.dispatcher import (
    FAMILY_TEACHER_CATEGORIES,
    LEGACY_TEACHER_CATEGORIES,
    TeacherDispatcherError,
    default_dispatcher,
)
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
from scoper.benchmark.behavior.tools.visual_report import (
    DEFAULT_PROCTHOR_TEMPLATE_DIR,
    html_footer,
    html_header,
    write_json,
    write_stage5_validation_table,
    write_stage6_execution_table,
    write_stage6_failure_breakdown,
    write_stage6_metrics_dashboard,
    write_stage6_predicate_progress,
    write_stage6_teacher_readiness,
    write_stage6_tool_catalog,
    write_stage6_tool_micro_cases,
    write_stage6_tool_trace,
    yes_no,
)


REQUIRED_TEACHER_BY_CATEGORY = {
    "static_simple": "legacy:OracleBrain",
    "hidden_precondition": "legacy:OracleBrain",
    "completion_timing": "legacy:OracleBrain",
    "dynamic_state_change": "legacy:OracleBrain",
    "disturbance_recovery": "legacy:OracleBrain",
    "mid_execution_discovery": "legacy:OracleBrain",
    "prepare_then_place": "family:PrepareTeacher",
    "prepare_multi_stage": "family:PrepareTeacher",
    "put_away_by_category": "family:PutAwayTeacher",
    "put_away_visual_judgment": "family:PutAwayTeacher",
    "put_away_cross_room": "family:PutAwayTeacher",
    "verify_after_action": "family:VerifyAfterActionTeacher",
    "mid_execution_state_disturbance": "family:MidExecutionDisturbanceTeacher",
    "recover_and_store_incident_object": "family:RecoverTeacher",
    "recover_multi_incident": "family:RecoverTeacher",
    "tidy_by_policy": "family:TidyTeacher",
    "human_interactive_execution": "family:HumanInteractionTeacher",
    "explore_then_act": "family:ExploreTeacher",
    "composite": "family:CompositeTeacher",
}


def build_stage8_oracle_teacher_report(
    *,
    out_root: Path,
    run_id: str | None = None,
    procthor_template_dir: Path = DEFAULT_PROCTHOR_TEMPLATE_DIR,
    scene_models: list[str] | None = None,
    min_stage5_pass_ratio: float = 0.90,
    min_execution_pass_ratio: float = 0.75,
    min_strict_teacher_ratio: float = 0.90,
    run_reachability: bool = False,
    max_high_level_steps: int = 120,
) -> dict[str, Any]:
    """Generate Stage 8 oracle-teacher execution artifacts and return a manifest."""

    scene_models = scene_models or ["Rs_int"]
    run_id = run_id or "{}_stage8_oracle_teachers".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage8_oracle_teachers" / run_id
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

    route_matrix = build_oracle_route_matrix(generation["tasks"])
    stage5_results = [
        validate_stage5_reset_mutation(task, rng_seed=0, run_reachability=run_reachability)
        for task in generation["tasks"]
    ]
    stage5_by_task = {item.get("task_id"): item for item in stage5_results}
    stage8_results = [
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
    execution_summary = summarize_stage6_results(generation, stage5_results, stage8_results)
    tool_micro_cases = build_tool_micro_cases(rng_seed=0)
    tool_micro_summary = summarize_tool_micro_cases(tool_micro_cases)
    coverage = build_oracle_teacher_coverage(generation["tasks"], route_matrix, stage5_results, stage8_results)
    route_summary = summarize_route_matrix(route_matrix)
    stage8_summary = build_stage8_summary(
        stage5_summary=stage5_summary,
        execution_summary=execution_summary,
        route_summary=route_summary,
        coverage=coverage,
        tool_micro_summary=tool_micro_summary,
        min_stage5_pass_ratio=min_stage5_pass_ratio,
        min_execution_pass_ratio=min_execution_pass_ratio,
        min_strict_teacher_ratio=min_strict_teacher_ratio,
    )

    tool_specs = BehaviorClosedLoopToolEnv().get_tool_specs(model_visible_only=True)
    family_gold_tasks = [
        task for task, result in zip(generation["tasks"], stage8_results) if result.get("execution_ok")
    ]

    write_stage8_index(output_dir / "index.html", stage8_summary)
    write_oracle_teacher_coverage_html(output_dir / "oracle_teacher_coverage.html", coverage)
    write_family_route_matrix_html(output_dir / "family_route_matrix.html", route_matrix, route_summary)
    write_oracle_family_gallery(output_dir / "oracle_family_gallery.html", coverage)
    write_execution_timeline(output_dir / "execution_timeline.html", stage8_results)
    write_oracle_plan_vs_execution(output_dir / "oracle_plan_vs_execution.html", generation["tasks"], stage8_results)
    write_failure_recovery_replay(output_dir / "failure_recovery_replay.html", generation["tasks"], stage8_results)
    write_scheduler_events(output_dir / "scheduler_events.html", stage8_results)
    write_symbolic_replay(output_dir / "symbolic_replay.html", stage8_results)
    write_stage6_teacher_readiness(output_dir / "teacher_readiness.html", stage8_results)
    write_stage6_execution_table(output_dir / "stage8_execution.html", stage8_results)
    write_stage6_tool_trace(output_dir / "tool_trace.html", stage8_results)
    write_stage6_predicate_progress(output_dir / "predicate_progress.html", stage8_results)
    write_stage6_metrics_dashboard(output_dir / "metrics_dashboard.html", execution_summary, stage8_results)
    write_stage6_failure_breakdown(output_dir / "failure_breakdown.html", execution_summary, stage8_results)
    write_stage6_tool_catalog(output_dir / "tool_catalog.html", tool_specs)
    write_stage6_tool_micro_cases(output_dir / "tool_micro_cases.html", tool_micro_cases, tool_micro_summary)
    write_stage5_validation_table(output_dir / "stage5_validation.html", stage5_results)

    write_json(logs_dir / "task_bank.json", generation["tasks"])
    write_json(logs_dir / "family_gold_tasks.json", family_gold_tasks)
    write_json(logs_dir / "stage5_validation.json", stage5_results)
    write_json(logs_dir / "stage5_summary.json", stage5_summary)
    write_json(logs_dir / "stage8_execution.json", stage8_results)
    write_json(logs_dir / "stage8_summary.json", stage8_summary)
    write_json(logs_dir / "oracle_teacher_coverage.json", coverage)
    write_json(logs_dir / "family_route_matrix.json", route_matrix)
    write_json(logs_dir / "family_route_summary.json", route_summary)
    write_json(logs_dir / "tool_micro_cases.json", tool_micro_cases)
    write_json(logs_dir / "tool_micro_summary.json", tool_micro_summary)

    passed = bool(stage8_summary.get("stage8_acceptance_passed"))
    manifest = {
        "run_id": run_id,
        "stage": "stage8_oracle_teachers",
        "status": "passed" if passed else "failed",
        "visual_gate": "oracle_teacher_gold_traces_pending_human_review" if passed else "oracle_teacher_alignment_failed",
        "scene_models": list(scene_models),
        "generated_total": stage8_summary["generated_total"],
        "stage5_admitted": stage8_summary["stage5_admitted"],
        "teacher_ready": stage8_summary["teacher_ready"],
        "strict_teacher_ready": stage8_summary["strict_teacher_ready"],
        "predicate_fallback_ready": stage8_summary["predicate_fallback_ready"],
        "execution_attempted": stage8_summary["execution_attempted"],
        "execution_passed": stage8_summary["execution_passed"],
        "execution_pass_ratio": stage8_summary["execution_pass_ratio"],
        "gold_trace_count": stage8_summary["gold_trace_count"],
        "min_stage5_pass_ratio": float(min_stage5_pass_ratio),
        "min_execution_pass_ratio": float(min_execution_pass_ratio),
        "min_strict_teacher_ratio": float(min_strict_teacher_ratio),
        "run_reachability": bool(run_reachability),
        "max_high_level_steps": int(max_high_level_steps),
        "stage8_buckets": stage8_summary["stage8_buckets"],
        "failure_codes": sorted(stage8_summary["failure_codes"]),
        "deferred_codes": stage8_summary["deferred_codes"],
        "tool_micro_cases_match_expected": bool(tool_micro_summary.get("all_cases_match_expected")),
        "model_visible_tools": list(MODEL_VISIBLE_TOOLS),
        "artifacts": {
            "index": "index.html",
            "oracle_teacher_coverage": "oracle_teacher_coverage.html",
            "family_route_matrix": "family_route_matrix.html",
            "oracle_family_gallery": "oracle_family_gallery.html",
            "execution_timeline": "execution_timeline.html",
            "oracle_plan_vs_execution": "oracle_plan_vs_execution.html",
            "predicate_progress": "predicate_progress.html",
            "tool_trace": "tool_trace.html",
            "metrics_dashboard": "metrics_dashboard.html",
            "failure_breakdown": "failure_breakdown.html",
            "failure_recovery_replay": "failure_recovery_replay.html",
            "scheduler_events": "scheduler_events.html",
            "symbolic_replay": "symbolic_replay.html",
            "teacher_readiness": "teacher_readiness.html",
            "stage8_execution": "stage8_execution.html",
            "tool_catalog": "tool_catalog.html",
            "tool_micro_cases": "tool_micro_cases.html",
            "stage5_validation": "stage5_validation.html",
            "task_bank_json": "logs/task_bank.json",
            "family_gold_tasks_json": "logs/family_gold_tasks.json",
            "stage8_execution_json": "logs/stage8_execution.json",
            "summary_json": "logs/stage8_summary.json",
            "oracle_teacher_coverage_json": "logs/oracle_teacher_coverage.json",
            "family_route_matrix_json": "logs/family_route_matrix.json",
            "tool_micro_cases_json": "logs/tool_micro_cases.json",
        },
        "notes": [
            "Stage 8 validates strict oracle teacher routing and symbolic gold traces.",
            "Real RGB video remains Stage 9; this report provides symbolic replay and JSONL traces.",
            "Predicate fallback must remain zero for Stage 8 acceptance.",
            "Execution uses EpisodeRunner and BehaviorClosedLoopToolEnv public tools only.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {key: str(output_dir / value) for key, value in manifest["artifacts"].items()}
    return manifest


def build_oracle_route_matrix(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dispatcher = default_dispatcher()
    rows = []
    for task in tasks:
        category = str(task.get("category") or "")
        try:
            brain, source = dispatcher.select(task)
            teacher_type = type(brain).__name__
            failure = None
        except TeacherDispatcherError as exc:
            source = None
            teacher_type = None
            failure = str(exc)
        if source == "predicate_fallback":
            route_status = "predicate_fallback"
        elif source and source.startswith("family:"):
            route_status = "family_ready"
        elif source and source.startswith("legacy:"):
            route_status = "legacy_ready"
        else:
            route_status = "teacher_deferred"
        rows.append(
            {
                "task_id": task.get("id"),
                "template_id": (task.get("template") or {}).get("template_id"),
                "category": category,
                "required_teacher": REQUIRED_TEACHER_BY_CATEGORY.get(category, "{}:teacher".format(category or "unknown")),
                "route_source": source,
                "teacher_type": teacher_type,
                "route_status": route_status,
                "failure_reason": failure,
                "is_family_category": category in FAMILY_TEACHER_CATEGORIES,
                "is_legacy_category": category in LEGACY_TEACHER_CATEGORIES,
            }
        )
    return rows


def summarize_route_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row.get("category")) for row in rows}):
        items = [row for row in rows if row.get("category") == category]
        by_category[category] = {
            "tasks": len(items),
            "required_teacher": items[0].get("required_teacher") if items else None,
            "route_sources": dict(sorted(Counter(str(item.get("route_source") or "-") for item in items).items())),
            "route_statuses": dict(sorted(Counter(str(item.get("route_status") or "-") for item in items).items())),
            "teacher_types": dict(sorted(Counter(str(item.get("teacher_type") or "-") for item in items).items())),
            "templates": sorted({str(item.get("template_id")) for item in items}),
        }
    predicate_fallback = sum(1 for row in rows if row.get("route_status") == "predicate_fallback")
    strict_ready = sum(1 for row in rows if row.get("route_status") in {"family_ready", "legacy_ready"})
    family_predicate_fallback = sum(
        1 for row in rows if row.get("is_family_category") and row.get("route_status") == "predicate_fallback"
    )
    return {
        "total": len(rows),
        "strict_ready": strict_ready,
        "teacher_deferred": sum(1 for row in rows if row.get("route_status") == "teacher_deferred"),
        "predicate_fallback": predicate_fallback,
        "family_predicate_fallback": family_predicate_fallback,
        "route_statuses": dict(sorted(Counter(str(row.get("route_status") or "-") for row in rows).items())),
        "by_category": by_category,
    }


def build_oracle_teacher_coverage(
    tasks: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    stage5_results: Sequence[Mapping[str, Any]],
    execution_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    route_by_task = {row.get("task_id"): row for row in routes}
    stage5_by_task = {row.get("task_id"): row for row in stage5_results}
    execution_by_task = {row.get("task_id"): row for row in execution_results}
    coverage = []
    for task in tasks:
        task_id = task.get("id")
        route = route_by_task.get(task_id, {})
        stage5 = stage5_by_task.get(task_id, {})
        execution = execution_by_task.get(task_id, {})
        tool_calls = execution.get("tool_calls") or []
        scheduler_events = execution.get("scheduler_events") or []
        coverage.append(
            {
                "task_id": task_id,
                "template_id": (task.get("template") or {}).get("template_id"),
                "category": task.get("category"),
                "scene_model": (task.get("source") or {}).get("scene_model"),
                "required_teacher": route.get("required_teacher"),
                "route_source": route.get("route_source"),
                "teacher_type": route.get("teacher_type"),
                "route_status": route.get("route_status"),
                "stage5_admitted": bool(stage5.get("stage5_reset_mutation_ok")),
                "stage5_bucket": stage5.get("stage5_bucket"),
                "execution_bucket": execution.get("stage6_bucket"),
                "execution_attempted": bool(execution.get("execution_attempted")),
                "execution_ok": bool(execution.get("execution_ok")),
                "goal_success": bool(execution.get("goal_success")),
                "trace_path": execution.get("trace_path"),
                "transition_count": int(execution.get("transition_count") or 0),
                "tools_used": sorted({str(item.get("tool_name")) for item in tool_calls if item.get("tool_name")}),
                "scheduler_event_names": sorted({str(item.get("name")) for item in scheduler_events if item.get("name")}),
                "has_failure_spec": bool(task.get("failure_spec")),
                "has_disturbance_spec": bool(task.get("disturbance_spec")),
                "has_human_events": bool(((task.get("family_spec") or {}).get("human") or {}).get("human_events_template")),
                "failure_code": execution.get("failure_code"),
                "failure_reason": execution.get("failure_reason"),
            }
        )
    return coverage


def build_stage8_summary(
    *,
    stage5_summary: Mapping[str, Any],
    execution_summary: Mapping[str, Any],
    route_summary: Mapping[str, Any],
    coverage: Sequence[Mapping[str, Any]],
    tool_micro_summary: Mapping[str, Any],
    min_stage5_pass_ratio: float,
    min_execution_pass_ratio: float,
    min_strict_teacher_ratio: float,
) -> dict[str, Any]:
    generated_total = int(execution_summary.get("generated_total") or 0)
    strict_route_ratio = float(route_summary.get("strict_ready", 0) / generated_total) if generated_total else 0.0
    pass_ratio = float(execution_summary.get("execution_pass_ratio") or 0.0)
    stage5_pass_ratio = float(stage5_summary.get("reset_mutation_pass_ratio") or 0.0)
    gold_trace_count = sum(1 for item in coverage if item.get("execution_ok") and item.get("trace_path"))
    family_counts = defaultdict(lambda: Counter({"tasks": 0, "attempted": 0, "passed": 0}))
    for item in coverage:
        counter = family_counts[str(item.get("category"))]
        counter["tasks"] += 1
        if item.get("execution_attempted"):
            counter["attempted"] += 1
        if item.get("execution_ok"):
            counter["passed"] += 1
    acceptance_checks = {
        "stage5_pass_ratio": stage5_pass_ratio >= float(min_stage5_pass_ratio),
        "strict_teacher_ratio": strict_route_ratio >= float(min_strict_teacher_ratio),
        "predicate_fallback_zero": int(execution_summary.get("predicate_fallback_ready") or 0) == 0
        and int(route_summary.get("predicate_fallback") or 0) == 0,
        "family_predicate_fallback_zero": int(route_summary.get("family_predicate_fallback") or 0) == 0,
        "execution_pass_ratio": pass_ratio >= float(min_execution_pass_ratio),
        "execution_attempted": int(execution_summary.get("execution_attempted") or 0) > 0,
        "tool_micro_cases": bool(tool_micro_summary.get("all_cases_match_expected")),
        "gold_traces": gold_trace_count > 0,
    }
    summary = dict(execution_summary)
    summary.update(
        {
            "stage": "stage8_oracle_teachers",
            "stage5_pass_ratio": stage5_pass_ratio,
            "strict_route_ready": int(route_summary.get("strict_ready") or 0),
            "strict_route_ratio": strict_route_ratio,
            "route_statuses": route_summary.get("route_statuses") or {},
            "route_predicate_fallback": int(route_summary.get("predicate_fallback") or 0),
            "family_predicate_fallback": int(route_summary.get("family_predicate_fallback") or 0),
            "gold_trace_count": gold_trace_count,
            "stage8_buckets": execution_summary.get("stage6_buckets") or {},
            "family_execution": {
                category: dict(counter)
                for category, counter in sorted(family_counts.items())
            },
            "tool_micro_summary": tool_micro_summary,
            "stage8_acceptance_checks": acceptance_checks,
            "stage8_acceptance_passed": all(acceptance_checks.values()),
        }
    )
    return summary


def write_stage8_index(path: Path, summary: Mapping[str, Any]) -> None:
    cards = [
        ("Generated", summary.get("generated_total")),
        ("Stage 5 admitted", summary.get("stage5_admitted")),
        ("Strict route ready", summary.get("strict_route_ready")),
        ("Predicate fallback", summary.get("predicate_fallback_ready")),
        ("Attempted", summary.get("execution_attempted")),
        ("Passed", summary.get("execution_passed")),
        ("Pass ratio", "{:.1%}".format(float(summary.get("execution_pass_ratio") or 0.0))),
        ("Gold traces", summary.get("gold_trace_count")),
        ("Stage 8 pass", yes_no(summary.get("stage8_acceptance_passed"))),
    ]
    body = [
        html_header("scoper.benchmark.behavior Stage 8 Oracle Teachers"),
        "<main>",
        "<h1>scoper.benchmark.behavior Stage 8 Oracle Teachers</h1>",
        "<p class='muted'>Strict family/legacy teacher routing with public-tool oracle gold traces. RGB video remains Stage 9.</p>",
        "<section class='cards'>",
    ]
    for label, value in cards:
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(escape(str(value)), escape(label)))
    body.extend(
        [
            "</section>",
            "<section><h2>Artifacts</h2><ul>",
            "<li><a href='oracle_teacher_coverage.html'>Oracle teacher coverage</a></li>",
            "<li><a href='family_route_matrix.html'>Family route matrix</a></li>",
            "<li><a href='oracle_family_gallery.html'>Oracle family gallery</a></li>",
            "<li><a href='execution_timeline.html'>Execution timeline</a></li>",
            "<li><a href='oracle_plan_vs_execution.html'>Oracle plan vs execution</a></li>",
            "<li><a href='predicate_progress.html'>Predicate progress</a></li>",
            "<li><a href='tool_trace.html'>Tool trace</a></li>",
            "<li><a href='failure_recovery_replay.html'>Failure/disturbance replay</a></li>",
            "<li><a href='scheduler_events.html'>Scheduler events</a></li>",
            "<li><a href='symbolic_replay.html'>Symbolic replay</a></li>",
            "<li><a href='logs/stage8_summary.json'>Summary JSON</a></li>",
            "<li><a href='manifest.json'>Manifest JSON</a></li>",
            "</ul></section>",
            "<section><h2>Acceptance Checks</h2><table><thead><tr><th>Check</th><th>Passed</th></tr></thead><tbody>",
        ]
    )
    for check, passed in (summary.get("stage8_acceptance_checks") or {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(check)), yes_no(passed)))
    body.extend(["</tbody></table>", "<h2>Buckets</h2><table><thead><tr><th>Bucket</th><th>Tasks</th></tr></thead><tbody>"])
    for bucket, count in (summary.get("stage8_buckets") or {}).items():
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(bucket)), int(count)))
    body.extend(["</tbody></table>", "</section>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_oracle_teacher_coverage_html(path: Path, coverage: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Oracle Teacher Coverage"),
        "<main>",
        "<h1>Oracle Teacher Coverage</h1>",
        "<table><thead><tr><th>Template</th><th>Category</th><th>Scene</th><th>Required</th><th>Route</th><th>Stage5</th><th>Execution</th><th>Tools</th><th>Scheduler</th><th>Failure</th></tr></thead><tbody>",
    ]
    for item in coverage:
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td><code>{}</code></td><td><code>{}</code><div class='muted'>{}</div></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(item.get("template_id"))),
                escape(str(item.get("category"))),
                escape(str(item.get("scene_model"))),
                escape(str(item.get("required_teacher"))),
                escape(str(item.get("route_source") or "-")),
                escape(str(item.get("teacher_type") or "-")),
                yes_no(item.get("stage5_admitted")),
                "{} / {}".format(yes_no(item.get("execution_ok")), escape(str(item.get("execution_bucket")))),
                escape(", ".join(item.get("tools_used") or []) or "-"),
                escape(", ".join(item.get("scheduler_event_names") or []) or "-"),
                escape(str(item.get("failure_code") or "-")),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_family_route_matrix_html(
    path: Path,
    route_matrix: Sequence[Mapping[str, Any]],
    route_summary: Mapping[str, Any],
) -> None:
    body = [
        html_header("Family Route Matrix"),
        "<main>",
        "<h1>Family Route Matrix</h1>",
        "<h2>By Category</h2><table><thead><tr><th>Category</th><th>Tasks</th><th>Required</th><th>Routes</th><th>Statuses</th><th>Teacher Types</th><th>Templates</th></tr></thead><tbody>",
    ]
    for category, item in (route_summary.get("by_category") or {}).items():
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(category)),
                int(item.get("tasks") or 0),
                escape(str(item.get("required_teacher") or "-")),
                escape(json.dumps(item.get("route_sources") or {}, sort_keys=True)),
                escape(json.dumps(item.get("route_statuses") or {}, sort_keys=True)),
                escape(json.dumps(item.get("teacher_types") or {}, sort_keys=True)),
                escape(", ".join(item.get("templates") or [])),
            )
        )
    body.extend(["</tbody></table>", "<h2>Per Task Routes</h2><table><thead><tr><th>Task</th><th>Template</th><th>Category</th><th>Status</th><th>Source</th><th>Failure</th></tr></thead><tbody>"])
    for row in route_matrix:
        status_class = "visually_verified" if row.get("route_status") in {"family_ready", "legacy_ready"} else "blocked"
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td class='status {}'>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(row.get("task_id"))),
                escape(str(row.get("template_id"))),
                escape(str(row.get("category"))),
                status_class,
                escape(str(row.get("route_status"))),
                escape(str(row.get("route_source") or "-")),
                escape(str(row.get("failure_reason") or "-")),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_oracle_family_gallery(path: Path, coverage: Sequence[Mapping[str, Any]]) -> None:
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in coverage:
        by_category[str(item.get("category"))].append(item)
    body = [html_header("Oracle Family Gallery"), "<main>", "<h1>Oracle Family Gallery</h1>", "<section class='template-grid'>"]
    for category in sorted(by_category):
        items = by_category[category]
        passed = sum(1 for item in items if item.get("execution_ok"))
        attempted = sum(1 for item in items if item.get("execution_attempted"))
        tools = sorted({tool for item in items for tool in item.get("tools_used") or []})
        body.append(
            "<article class='template-card'><h2><code>{}</code></h2><p class='muted'>{}/{} passed, {} attempted</p><table><tr><th>required teacher</th><td><code>{}</code></td></tr><tr><th>route sources</th><td>{}</td></tr><tr><th>tools</th><td>{}</td></tr><tr><th>templates</th><td>{}</td></tr></table></article>".format(
                escape(category),
                int(passed),
                len(items),
                int(attempted),
                escape(str(items[0].get("required_teacher") or "-")),
                escape(json.dumps(dict(sorted(Counter(str(item.get("route_source") or "-") for item in items).items())), sort_keys=True)),
                escape(", ".join(tools) or "-"),
                escape(", ".join(sorted({str(item.get("template_id")) for item in items}))),
            )
        )
    body.extend(["</section>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_execution_timeline(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Stage 8 Execution Timeline"),
        "<main>",
        "<h1>Stage 8 Execution Timeline</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Teacher</th><th>Timeline</th></tr></thead><tbody>",
    ]
    for result in results:
        timeline = [
            {
                "step": call.get("step"),
                "tool": call.get("tool_name"),
                "args": call.get("arguments"),
                "success": call.get("success"),
                "error": call.get("error_type"),
                "progress": call.get("task_progress"),
            }
            for call in result.get("tool_calls") or []
        ]
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(str(result.get("teacher_source") or "-")),
                escape(json.dumps(timeline, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_oracle_plan_vs_execution(
    path: Path,
    tasks: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> None:
    task_by_id = {task.get("id"): task for task in tasks}
    body = [
        html_header("Oracle Plan vs Execution"),
        "<main>",
        "<h1>Oracle Plan vs Execution</h1>",
        "<table><thead><tr><th>Task</th><th>Expected Skeleton</th><th>Actual Tools</th><th>Status</th></tr></thead><tbody>",
    ]
    for result in results:
        task = task_by_id.get(result.get("task_id")) or {}
        expected = (task.get("oracle_teacher_hint") or {}).get("expected_plan_skeleton") or []
        actual = [call.get("tool_name") for call in result.get("tool_calls") or []]
        body.append(
            "<tr><td><code>{}</code><div class='muted'>{}</div></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("teacher_source") or "-")),
                escape(json.dumps(expected, sort_keys=True)),
                escape(json.dumps(actual)),
                yes_no(result.get("execution_ok")),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_failure_recovery_replay(
    path: Path,
    tasks: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> None:
    task_by_id = {task.get("id"): task for task in tasks}
    body = [
        html_header("Failure and Disturbance Replay"),
        "<main>",
        "<h1>Failure and Disturbance Replay</h1>",
        "<table><thead><tr><th>Task</th><th>Specs</th><th>Scheduler Events</th><th>Recovery Trace</th><th>Final</th></tr></thead><tbody>",
    ]
    for result in results:
        task = task_by_id.get(result.get("task_id")) or {}
        if not (task.get("failure_spec") or task.get("disturbance_spec") or result.get("scheduler_events")):
            continue
        specs = {
            "failure_spec": task.get("failure_spec") or [],
            "disturbance_spec": task.get("disturbance_spec") or [],
        }
        trace = [
            {
                "step": call.get("step"),
                "tool": call.get("tool_name"),
                "success": call.get("success"),
                "error": call.get("error_type"),
            }
            for call in result.get("tool_calls") or []
        ]
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(json.dumps(specs, sort_keys=True)),
                escape(json.dumps(result.get("scheduler_events") or [], sort_keys=True)),
                escape(json.dumps(trace, sort_keys=True)),
                yes_no(result.get("execution_ok")),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_scheduler_events(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Scheduler Events"),
        "<main>",
        "<h1>Scheduler Events</h1>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Events</th></tr></thead><tbody>",
    ]
    for result in results:
        events = result.get("scheduler_events") or []
        if not events:
            continue
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(json.dumps(events, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_symbolic_replay(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    body = [
        html_header("Symbolic Replay"),
        "<main>",
        "<h1>Symbolic Replay</h1>",
        "<p class='muted'>Compact top-down symbolic replay evidence from transition poses. Real RGB/MP4 is Stage 9.</p>",
        "<table><thead><tr><th>Task</th><th>Template</th><th>Pose Trace</th></tr></thead><tbody>",
    ]
    for result in results:
        pose_trace = [
            {
                "step": call.get("step"),
                "tool": call.get("tool_name"),
                "success": call.get("success"),
                "pose": (call.get("env_feedback") or {}).get("agent_pose") or call.get("agent_pose"),
                "frames": call.get("frames"),
            }
            for call in result.get("tool_calls") or []
        ]
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(result.get("task_id"))),
                escape(str(result.get("template_id"))),
                escape(json.dumps(pose_trace, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


__all__ = ["build_stage8_oracle_teacher_report"]
