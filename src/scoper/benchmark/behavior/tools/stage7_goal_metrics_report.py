"""Stage 7 goal / metrics report builder.

Stage 7 consumes a Stage 6 execution corpus. It does not rerun teachers or
oracle execution; it enriches public artifacts with predicate explanations,
stable failure reports, aggregate metrics, and consistency checks.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

from scoper.benchmark.behavior.failure_analysis import build_failure_report
from scoper.benchmark.behavior.goal_predicates import format_goal_predicate, normalize_goal_predicate
from scoper.benchmark.behavior.metrics import (
    aggregate_metrics,
    build_metrics_consistency_records,
    compute_episode_metrics,
)
from scoper.benchmark.behavior.paths import resolve_repo_path


PREDICATE_CONTRACT_ROWS = [
    ("object_on_receptacle", "implemented", "symbolic_aabb_ontop", "stage7_explained"),
    ("object_in_receptacle", "implemented", "symbolic_aabb_inside", "stage7_explained"),
    ("object_not_in_receptacle", "implemented", "negated_symbolic_aabb_inside", "stage7_explained"),
    ("object_in_any_receptacle", "implemented", "first_class_symbolic_any_inside", "stage7_explained"),
    ("object_not_in_any_receptacle", "implemented", "negated_symbolic_any_inside", "stage7_explained"),
    ("object_held", "implemented", "symbolic_inventory", "stage7_explained"),
    ("object_not_held", "implemented", "negated_symbolic_inventory", "stage7_explained"),
    ("object_open", "implemented", "symbolic_open_state", "stage7_explained"),
    ("object_closed", "implemented", "negated_symbolic_open_state", "stage7_explained"),
    ("object_toggled_on", "implemented", "symbolic_toggle_state", "stage7_explained"),
    ("object_toggled_off", "implemented", "negated_symbolic_toggle_state", "stage7_explained"),
    ("object_clean", "implemented", "negated_symbolic_dirty_state", "stage7_explained"),
    ("object_dirty", "implemented", "symbolic_dirty_state", "stage7_explained"),
    ("object_sliced", "implemented", "symbolic_sliced_state", "stage7_explained"),
    ("object_hot", "partial", "approximated_as_cooked", "stage7_explained"),
    ("object_cooked", "implemented", "symbolic_cooked_state", "stage7_explained"),
    ("object_cool", "implemented", "symbolic_cooled_state", "stage7_explained"),
    ("object_filled", "implemented", "symbolic_filled_substance", "stage7_explained"),
    ("object_soaked", "implemented", "symbolic_saturated_substance", "stage7_explained"),
    ("agent_near", "implemented", "symbolic_xy_distance", "stage7_explained"),
]


def build_stage7_goal_metrics_report(
    *,
    stage6_run: Path,
    out_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Generate Stage 7 artifacts from a Stage 6 execution output directory."""

    stage6_run = resolve_repo_path(str(stage6_run))
    run_id = run_id or "{}_stage7_goal_metrics".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(out_root) / "stage7_goal_metrics" / run_id
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    corpus = load_stage6_corpus(stage6_run)
    enriched_records = build_enriched_records(corpus)
    goal_explanations = [
        {
            "task_id": record.get("task_id"),
            "template_id": record.get("template_id"),
            "category": record.get("category"),
            "stage6_bucket": record.get("stage6_bucket"),
            "execution_attempted": bool(record.get("execution_attempted")),
            "execution_ok": bool(record.get("execution_ok")),
            "task_success": (record.get("final_goal") or {}).get("task_success"),
            "task_progress": (record.get("final_goal") or {}).get("task_progress"),
            "predicate_records": (record.get("final_goal") or {}).get("predicate_records") or [],
        }
        for record in enriched_records
    ]
    failure_reports = build_stage7_failure_reports(enriched_records)
    for record in enriched_records:
        report = failure_reports.get(record.get("task_id"))
        if report:
            record["stage7_failure_code"] = report.get("failure_code")
            record["stage7_failure_reason"] = report.get("failure_reason")
            record["stage7_failure_details"] = report.get("failure_details")

    metrics_aggregate = aggregate_metrics(enriched_records)
    consistency = build_metrics_consistency_records(enriched_records)
    predicate_contract = build_predicate_contract_matrix(corpus["tasks"], enriched_records)
    family_metrics = build_family_metrics_records(enriched_records)
    scheduler_metrics = build_scheduler_metrics_records(enriched_records)
    summary = build_stage7_summary(
        corpus=corpus,
        records=enriched_records,
        goal_explanations=goal_explanations,
        failure_reports=failure_reports,
        consistency=consistency,
        predicate_contract=predicate_contract,
        family_metrics=family_metrics,
    )

    write_stage7_index(output_dir / "index.html", summary)
    write_predicate_contract_matrix(output_dir / "predicate_contract_matrix.html", predicate_contract)
    write_goal_explanation(output_dir / "goal_explanation.html", goal_explanations)
    write_predicate_progress(output_dir / "predicate_progress.html", enriched_records)
    write_metrics_dashboard(output_dir / "metrics_dashboard.html", metrics_aggregate, enriched_records)
    write_metrics_consistency(output_dir / "metrics_consistency.html", consistency)
    write_failure_breakdown(output_dir / "failure_breakdown.html", failure_reports)
    write_family_metrics(output_dir / "family_metrics.html", family_metrics)
    write_scheduler_metrics(output_dir / "scheduler_metrics.html", scheduler_metrics)

    write_json(logs_dir / "stage7_summary.json", summary)
    write_json(logs_dir / "predicate_contract_matrix.json", predicate_contract)
    write_json(logs_dir / "goal_explanations.json", goal_explanations)
    write_json(logs_dir / "metrics_aggregate.json", metrics_aggregate)
    write_json(logs_dir / "metrics_consistency.json", consistency)
    write_json(logs_dir / "failure_reports.json", list(failure_reports.values()))
    write_json(logs_dir / "family_metrics.json", family_metrics)
    write_json(logs_dir / "scheduler_metrics.json", scheduler_metrics)
    write_json(logs_dir / "stage7_records.json", enriched_records)

    passed = (
        summary["num_tasks"] > 0
        and summary["num_goal_explanations"] == summary["num_tasks"]
        and summary["metrics_consistency_status"] == "passed"
        and summary["failure_report_status"] == "passed"
    )
    manifest = {
        "run_id": run_id,
        "stage": "stage7_goal_metrics",
        "status": "passed" if passed else "failed",
        "source_stage6_run": str(stage6_run),
        "num_tasks": summary["num_tasks"],
        "num_execution_attempted": summary["num_execution_attempted"],
        "num_execution_passed": summary["num_execution_passed"],
        "num_goal_explanations": summary["num_goal_explanations"],
        "num_failure_reports": summary["num_failure_reports"],
        "predicate_contract_status": summary["predicate_contract_status"],
        "metrics_consistency_status": summary["metrics_consistency_status"],
        "failure_report_status": summary["failure_report_status"],
        "family_metrics_status": summary["family_metrics_status"],
        "visual_gate": "goal_metrics_pending_human_review" if passed else "goal_metrics_validation_failed",
        "artifacts": {
            "index": "index.html",
            "predicate_contract_matrix": "predicate_contract_matrix.html",
            "goal_explanation": "goal_explanation.html",
            "predicate_progress": "predicate_progress.html",
            "metrics_dashboard": "metrics_dashboard.html",
            "metrics_consistency": "metrics_consistency.html",
            "failure_breakdown": "failure_breakdown.html",
            "family_metrics": "family_metrics.html",
            "scheduler_metrics": "scheduler_metrics.html",
            "summary_json": "logs/stage7_summary.json",
            "goal_explanations_json": "logs/goal_explanations.json",
            "metrics_aggregate_json": "logs/metrics_aggregate.json",
            "failure_reports_json": "logs/failure_reports.json",
        },
        "notes": [
            "Stage 7 consumes Stage 6 execution artifacts and does not rerun oracle execution.",
            "Old Stage 6 final_goal entries without rich predicate records are enriched from task_bank goal args.",
            "Rendered/video validation remains the later Stage 9 path.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    manifest["output_dir"] = str(output_dir)
    manifest["artifacts"] = {key: str(output_dir / value) for key, value in manifest["artifacts"].items()}
    return manifest


def load_stage6_corpus(stage6_run: Path) -> dict[str, Any]:
    logs = Path(stage6_run) / "logs"
    return {
        "stage6_run": str(stage6_run),
        "manifest": read_json(Path(stage6_run) / "manifest.json", default={}),
        "tasks": read_json(logs / "task_bank.json", default=[]),
        "stage6_results": read_json(logs / "stage6_execution.json", default=[]),
        "stage6_summary": read_json(logs / "stage6_summary.json", default={}),
    }


def build_enriched_records(corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = list(corpus.get("tasks") or [])
    results = list(corpus.get("stage6_results") or [])
    by_task = {task.get("id"): task for task in tasks if isinstance(task, Mapping)}
    by_result = {result.get("task_id"): result for result in results if isinstance(result, Mapping)}
    records: list[dict[str, Any]] = []

    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = task.get("id")
        result = dict(by_result.get(task_id) or {})
        result.setdefault("task_id", task_id)
        result.setdefault("template_id", (task.get("template") or {}).get("template_id") or (task.get("source") or {}).get("template_id"))
        result.setdefault("category", task.get("category"))
        result.setdefault("scene_model", (task.get("source") or {}).get("scene_model"))
        result["task_spec"] = task
        result["scheduler_events"] = list(result.get("scheduler_events") or _scheduler_events_from_trace(result.get("trace_path")))

        final_goal = dict(result.get("final_goal") or {})
        final_goal["predicate_records"] = _merge_predicate_records(task, final_goal)
        if "predicates" not in final_goal or not final_goal.get("predicates"):
            final_goal["predicates"] = [
                {k: v for k, v in item.items() if k in {"type", "satisfied", "args", "expected", "actual", "reason"}}
                for item in final_goal["predicate_records"]
            ]
        result["final_goal"] = final_goal

        transitions = list(result.get("tool_calls") or [])
        old_metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
        done_reason = old_metrics.get("done_reason") if isinstance(old_metrics, Mapping) else None
        if transitions and done_reason is None:
            done_reason = transitions[-1].get("done_reason")
        result["metrics"] = compute_episode_metrics(
            transitions=transitions,
            final_goal=final_goal,
            done_reason=done_reason,
            model_stopped=bool(final_goal.get("model_stopped", old_metrics.get("model_stopped", False) if isinstance(old_metrics, Mapping) else False)),
            elapsed_seconds=old_metrics.get("elapsed_seconds") if isinstance(old_metrics, Mapping) else None,
            scheduler_events=result["scheduler_events"],
            task_spec=task,
        )
        records.append(result)

    # Keep any extra Stage 6 result that did not match task_bank visible.
    for result in results:
        if result.get("task_id") in by_task:
            continue
        item = dict(result)
        item.setdefault("task_spec", {})
        item["final_goal"] = dict(item.get("final_goal") or {})
        item["final_goal"]["predicate_records"] = _merge_predicate_records({}, item["final_goal"])
        item["metrics"] = compute_episode_metrics(
            transitions=list(item.get("tool_calls") or []),
            final_goal=item["final_goal"],
            done_reason=(item.get("metrics") or {}).get("done_reason") if isinstance(item.get("metrics"), Mapping) else None,
            model_stopped=bool(item["final_goal"].get("model_stopped", False)),
            scheduler_events=list(item.get("scheduler_events") or []),
            task_spec=None,
        )
        records.append(item)
    return records


def build_stage7_failure_reports(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = record.get("task_id")
        if record.get("execution_ok") is True:
            continue
        if not record.get("execution_attempted"):
            code = str(record.get("failure_code") or "deferred").lower()
            reports[str(task_id)] = {
                "task_id": task_id,
                "template_id": record.get("template_id"),
                "category": record.get("category"),
                "stage6_bucket": record.get("stage6_bucket"),
                "failure_code": code,
                "failure_reason": str(record.get("failure_reason") or "Execution was deferred before Stage 6 attempted public tools."),
                "failure_details": {
                    "deferred_codes": list(record.get("deferred_codes") or []),
                    "unsupported_goal_predicates": list(record.get("unsupported_goal_predicates") or []),
                    "task_id": task_id,
                    "category": record.get("category"),
                },
            }
            continue
        report = build_failure_report(
            transitions=list(record.get("tool_calls") or []),
            final_goal=record.get("final_goal") or {},
            failure_type=record.get("failure_type"),
            failure_reason=record.get("failure_reason"),
            scheduler_events=list(record.get("scheduler_events") or []),
            task_spec=record.get("task_spec") if isinstance(record.get("task_spec"), Mapping) else None,
            model_stopped=bool((record.get("final_goal") or {}).get("model_stopped", False)),
        )
        reports[str(task_id)] = {
            "task_id": task_id,
            "template_id": record.get("template_id"),
            "category": record.get("category"),
            "stage6_bucket": record.get("stage6_bucket"),
            **report,
        }
    return reports


def build_predicate_contract_matrix(
    tasks: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    used = Counter()
    unsatisfied = Counter()
    for task in tasks:
        for predicate in ((task.get("goal") or {}).get("predicates") or []):
            used[str(predicate.get("type"))] += 1
    for record in records:
        for predicate in ((record.get("final_goal") or {}).get("predicate_records") or []):
            if predicate.get("satisfied") is False:
                unsatisfied[str(predicate.get("type"))] += 1
    rows = []
    for ptype, status, evaluator, visual_status in PREDICATE_CONTRACT_ROWS:
        rows.append(
            {
                "predicate": ptype,
                "status": status,
                "evaluator": evaluator,
                "visual_status": visual_status,
                "used_in_stage6_tasks": int(used.get(ptype, 0)),
                "unsatisfied_final_count": int(unsatisfied.get(ptype, 0)),
            }
        )
    return rows


def build_family_metrics_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
        family_metrics = metrics.get("family_metrics") if isinstance(metrics.get("family_metrics"), Mapping) else None
        task_spec = record.get("task_spec") if isinstance(record.get("task_spec"), Mapping) else {}
        family_spec = task_spec.get("family_spec") if isinstance(task_spec.get("family_spec"), Mapping) else {}
        rows.append(
            {
                "task_id": record.get("task_id"),
                "template_id": record.get("template_id"),
                "category": record.get("category"),
                "family": (family_spec or {}).get("family") or record.get("category"),
                "execution_ok": bool(record.get("execution_ok")),
                "family_metrics": family_metrics or {"status": "absent", "reason": "no_family_spec_or_not_applicable"},
            }
        )
    return rows


def build_scheduler_metrics_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
        rows.append(
            {
                "task_id": record.get("task_id"),
                "template_id": record.get("template_id"),
                "category": record.get("category"),
                "stage6_bucket": record.get("stage6_bucket"),
                "num_disturbance_events": int(metrics.get("num_disturbance_events", 0) or 0),
                "num_failure_injections": int(metrics.get("num_failure_injections", 0) or 0),
                "disturbance_counts": metrics.get("disturbance_counts") or {},
                "failure_counts": metrics.get("failure_counts") or {},
                "scheduler_events": list(record.get("scheduler_events") or []),
            }
        )
    return rows


def build_stage7_summary(
    *,
    corpus: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    goal_explanations: Sequence[Mapping[str, Any]],
    failure_reports: Mapping[str, Mapping[str, Any]],
    consistency: Sequence[Mapping[str, Any]],
    predicate_contract: Sequence[Mapping[str, Any]],
    family_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failed_consistency = [item for item in consistency if item.get("status") != "passed"]
    expected_reports = [
        record for record in records
        if record.get("execution_ok") is not True
    ]
    missing_reports = [
        record.get("task_id") for record in expected_reports
        if not failure_reports.get(str(record.get("task_id"))) or not failure_reports[str(record.get("task_id"))].get("failure_code")
    ]
    family_statuses = Counter(
        str((item.get("family_metrics") or {}).get("status") or "absent")
        for item in family_metrics
    )
    implemented_or_structural = sum(family_statuses.get(key, 0) for key in ("implemented", "structural"))
    family_status = "implemented" if implemented_or_structural == len(family_metrics) else ("partial" if implemented_or_structural else "deferred")
    return {
        "stage": "stage7_goal_metrics",
        "source_stage6_run": corpus.get("stage6_run"),
        "num_tasks": len(records),
        "num_execution_attempted": sum(1 for record in records if record.get("execution_attempted")),
        "num_execution_passed": sum(1 for record in records if record.get("execution_ok")),
        "num_goal_explanations": len(goal_explanations),
        "num_failure_reports": len(failure_reports),
        "expected_failure_reports": len(expected_reports),
        "missing_failure_report_task_ids": missing_reports,
        "failure_report_status": "passed" if not missing_reports else "failed",
        "predicate_contract_status": "passed" if predicate_contract else "failed",
        "metrics_consistency_status": "passed" if not failed_consistency else "failed",
        "metrics_consistency_failures": len(failed_consistency),
        "family_metrics_status": family_status,
        "family_metric_status_counts": dict(sorted(family_statuses.items())),
        "stage6_summary": corpus.get("stage6_summary") or {},
        "stage7_failure_code_counts": dict(sorted(Counter(report.get("failure_code") for report in failure_reports.values()).items())),
    }


def _merge_predicate_records(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any]) -> list[dict[str, Any]]:
    goal_specs = list(((task_spec.get("goal") or {}).get("predicates") or [])) if isinstance(task_spec, Mapping) else []
    final_predicates = list(final_goal.get("predicate_records") or final_goal.get("predicates") or [])
    if not final_predicates and goal_specs:
        final_predicates = [{"type": spec.get("type"), "satisfied": None} for spec in goal_specs]
    used: set[int] = set()
    records = []
    for predicate in final_predicates:
        if not isinstance(predicate, Mapping):
            continue
        spec = _matching_goal_spec(predicate, goal_specs, used)
        merged = dict(spec or {})
        merged.update(predicate)
        if "args" not in merged and isinstance(spec, Mapping) and "args" in spec:
            merged["args"] = spec["args"]
        record = normalize_goal_predicate(merged)
        if "satisfied" not in record:
            record["satisfied"] = predicate.get("satisfied")
        if not record.get("expected"):
            record["expected"] = format_goal_predicate(record)
        if not record.get("actual"):
            if record.get("satisfied") is True:
                record["actual"] = "Stage 6 final goal reported predicate satisfied."
            elif record.get("satisfied") is False:
                record["actual"] = "Stage 6 final goal reported predicate unsatisfied."
            else:
                record["actual"] = "Predicate was not evaluated in Stage 6 execution."
        if not record.get("reason"):
            if record.get("satisfied") is True:
                record["reason"] = "predicate satisfied"
            elif record.get("satisfied") is False:
                record["reason"] = "predicate unsatisfied in Stage 6 final goal"
            else:
                record["reason"] = "execution deferred before final goal evaluation"
        record["explanation_source"] = "goal_checker" if predicate.get("actual") or predicate.get("reason") else "stage7_task_spec_merge"
        records.append(record)
    return records


def _matching_goal_spec(predicate: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], used: set[int]) -> Mapping[str, Any] | None:
    ptype = predicate.get("type")
    for idx, spec in enumerate(specs):
        if idx in used:
            continue
        if spec.get("type") == ptype:
            used.add(idx)
            return spec
    for idx, spec in enumerate(specs):
        if idx not in used:
            used.add(idx)
            return spec
    return None


def _scheduler_events_from_trace(trace_path: Any) -> list[dict[str, Any]]:
    if not trace_path:
        return []
    path = Path(str(trace_path))
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record_type") == "step":
                    events.extend([dict(ev) for ev in record.get("scheduler_events") or [] if isinstance(ev, Mapping)])
                elif record.get("record_type") == "summary":
                    events.extend([dict(ev) for ev in record.get("scheduler_events") or [] if isinstance(ev, Mapping)])
    except Exception:
        return events
    return events


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_stage7_index(path: Path, summary: Mapping[str, Any]) -> None:
    cards = [
        ("Tasks", summary["num_tasks"]),
        ("Attempted", summary["num_execution_attempted"]),
        ("Passed", summary["num_execution_passed"]),
        ("Goal explanations", summary["num_goal_explanations"]),
        ("Failure reports", summary["num_failure_reports"]),
        ("Metrics consistency", summary["metrics_consistency_status"]),
        ("Family metrics", summary["family_metrics_status"]),
    ]
    body = [html_header("scoper.benchmark.behavior Stage 7 Goal Metrics"), "<main>", "<h1>scoper.benchmark.behavior Stage 7 Goal Metrics</h1>"]
    body.append("<p class='muted'>Judge layer report over Stage 6 execution artifacts.</p>")
    body.append("<section class='cards'>")
    for label, value in cards:
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(escape(str(value)), escape(label)))
    body.extend(
        [
            "</section>",
            "<section><h2>Artifacts</h2><ul>",
            "<li><a href='predicate_contract_matrix.html'>Predicate contract matrix</a></li>",
            "<li><a href='goal_explanation.html'>Goal explanation</a></li>",
            "<li><a href='predicate_progress.html'>Predicate progress</a></li>",
            "<li><a href='metrics_dashboard.html'>Metrics dashboard</a></li>",
            "<li><a href='metrics_consistency.html'>Metrics consistency</a></li>",
            "<li><a href='failure_breakdown.html'>Failure breakdown</a></li>",
            "<li><a href='family_metrics.html'>Family metrics</a></li>",
            "<li><a href='scheduler_metrics.html'>Scheduler metrics</a></li>",
            "<li><a href='logs/stage7_summary.json'>Summary JSON</a></li>",
            "<li><a href='manifest.json'>Manifest JSON</a></li>",
            "</ul></section>",
            "</main>",
            html_footer(),
        ]
    )
    path.write_text("\n".join(body), encoding="utf-8")


def write_predicate_contract_matrix(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    body = [html_header("Predicate Contract Matrix"), "<main>", "<h1>Predicate Contract Matrix</h1>"]
    body.append("<table><thead><tr><th>Predicate</th><th>Status</th><th>Evaluator</th><th>Visual Status</th><th>Used</th><th>Unsatisfied Final</th></tr></thead><tbody>")
    for row in rows:
        body.append(
            "<tr><td><code>{}</code></td><td class='status {}'>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(row.get("predicate"))),
                _status_class(row.get("status")),
                escape(str(row.get("status"))),
                escape(str(row.get("evaluator"))),
                escape(str(row.get("visual_status"))),
                int(row.get("used_in_stage6_tasks") or 0),
                int(row.get("unsatisfied_final_count") or 0),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_goal_explanation(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    body = [html_header("Goal Explanation"), "<main>", "<h1>Goal Explanation</h1>"]
    body.append("<table><thead><tr><th>Task</th><th>Template</th><th>Bucket</th><th>Progress</th><th>Predicate Records</th></tr></thead><tbody>")
    for row in rows:
        preds = [
            {
                "type": p.get("type"),
                "satisfied": p.get("satisfied"),
                "expected": p.get("expected"),
                "actual": p.get("actual"),
                "reason": p.get("reason"),
            }
            for p in row.get("predicate_records") or []
        ]
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(row.get("task_id"))),
                escape(str(row.get("template_id"))),
                escape(str(row.get("stage6_bucket"))),
                escape(str(row.get("task_progress"))),
                escape(json.dumps(preds, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_predicate_progress(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    body = [html_header("Predicate Progress"), "<main>", "<h1>Predicate Progress</h1>"]
    body.append("<table><thead><tr><th>Task</th><th>Template</th><th>Progress</th></tr></thead><tbody>")
    for record in records:
        progress = [
            {
                "step": item.get("step"),
                "tool": item.get("tool_name"),
                "task_progress": item.get("task_progress"),
                "satisfied": item.get("satisfied"),
                "total": item.get("total"),
            }
            for item in record.get("predicate_progress") or []
        ]
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>".format(
                escape(str(record.get("task_id"))),
                escape(str(record.get("template_id"))),
                escape(json.dumps(progress, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_metrics_dashboard(path: Path, aggregate: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    body = [html_header("Metrics Dashboard"), "<main>", "<h1>Metrics Dashboard</h1>", "<section class='cards'>"]
    for label, key in (
        ("Episodes", "num_episodes"),
        ("Success", "num_success"),
        ("Mean success", "mean_task_success"),
        ("Mean progress", "mean_task_progress"),
        ("Low-level steps", "total_low_level_steps"),
    ):
        value = aggregate.get(key)
        if isinstance(value, float):
            value = "{:.3f}".format(value)
        body.append("<article><div class='num'>{}</div><div>{}</div></article>".format(escape(str(value)), escape(label)))
    body.append("</section><h2>Per Task</h2><table><thead><tr><th>Task</th><th>Template</th><th>Status</th><th>Metrics</th></tr></thead><tbody>")
    for record in records:
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(record.get("task_id"))),
                escape(str(record.get("template_id"))),
                escape(str(record.get("stage6_bucket"))),
                escape(json.dumps(record.get("metrics") or {}, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_metrics_consistency(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    body = [html_header("Metrics Consistency"), "<main>", "<h1>Metrics Consistency</h1>"]
    body.append("<table><thead><tr><th>Task</th><th>Status</th><th>Mismatches</th><th>Steps</th><th>Tool Calls</th><th>Invalid</th><th>Failed Calls</th></tr></thead><tbody>")
    for row in rows:
        body.append(
            "<tr><td><code>{}</code></td><td class='status {}'>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(row.get("task_id"))),
                "visually_verified" if row.get("status") == "passed" else "blocked",
                escape(str(row.get("status"))),
                escape(", ".join(row.get("mismatches") or []) or "-"),
                escape(str(row.get("high_level_steps"))),
                escape(str(row.get("tool_calls"))),
                escape(str(row.get("invalid_tool_calls"))),
                escape(str(row.get("failed_tool_calls"))),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_failure_breakdown(path: Path, reports: Mapping[str, Mapping[str, Any]]) -> None:
    counts = Counter(report.get("failure_code") for report in reports.values())
    body = [html_header("Failure Breakdown"), "<main>", "<h1>Failure Breakdown</h1>"]
    body.append("<h2>Codes</h2><table><thead><tr><th>Code</th><th>Tasks</th></tr></thead><tbody>")
    for code, count in sorted(counts.items()):
        body.append("<tr><td><code>{}</code></td><td>{}</td></tr>".format(escape(str(code)), int(count)))
    body.append("</tbody></table><h2>Reports</h2><table><thead><tr><th>Task</th><th>Template</th><th>Bucket</th><th>Code</th><th>Reason</th><th>Details</th></tr></thead><tbody>")
    for report in reports.values():
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(str(report.get("task_id"))),
                escape(str(report.get("template_id"))),
                escape(str(report.get("stage6_bucket"))),
                escape(str(report.get("failure_code"))),
                escape(str(report.get("failure_reason") or "-")),
                escape(json.dumps(report.get("failure_details") or {}, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_family_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    body = [html_header("Family Metrics"), "<main>", "<h1>Family Metrics</h1>"]
    body.append("<table><thead><tr><th>Task</th><th>Template</th><th>Family</th><th>Execution</th><th>Metrics</th></tr></thead><tbody>")
    for row in rows:
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(row.get("task_id"))),
                escape(str(row.get("template_id"))),
                escape(str(row.get("family"))),
                "yes" if row.get("execution_ok") else "no",
                escape(json.dumps(row.get("family_metrics") or {}, sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def write_scheduler_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    body = [html_header("Scheduler Metrics"), "<main>", "<h1>Scheduler Metrics</h1>"]
    body.append("<table><thead><tr><th>Task</th><th>Template</th><th>Disturbance</th><th>Failure Injection</th><th>Counts</th><th>Events</th></tr></thead><tbody>")
    for row in rows:
        body.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(row.get("task_id"))),
                escape(str(row.get("template_id"))),
                int(row.get("num_disturbance_events") or 0),
                int(row.get("num_failure_injections") or 0),
                escape(json.dumps({"disturbance": row.get("disturbance_counts"), "failure": row.get("failure_counts")}, sort_keys=True)),
                escape(json.dumps(row.get("scheduler_events") or [], sort_keys=True)),
            )
        )
    body.extend(["</tbody></table>", "</main>", html_footer()])
    path.write_text("\n".join(body), encoding="utf-8")


def _status_class(status: Any) -> str:
    if status == "implemented":
        return "implemented"
    if status == "partial":
        return "partial"
    return "missing"


def html_header(title: str) -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{}</title>
<style>
:root {{ --bg: #f7f8fa; --text: #17202a; --muted: #5d6d7e; --line: #d7dde5; --card: #ffffff; --partial: #ffe5b4; --implemented: #cfe8d5; --verified: #b7d7f2; --blocked: #ded7ef; --missing: #f6d2d2; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width: 1320px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin-top: 28px; font-size: 20px; }}
a {{ color: #145ea8; }}
code {{ background: #eef2f6; border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px; }}
table {{ border-collapse: collapse; width: 100%; background: var(--card); border: 1px solid var(--line); }}
th, td {{ border: 1px solid var(--line); padding: 8px 10px; vertical-align: top; text-align: left; }}
th {{ background: #eef2f6; position: sticky; top: 0; z-index: 1; }}
.muted {{ color: var(--muted); font-size: 12px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
.cards article {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
.num {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
.status {{ font-weight: 650; text-align: center; }}
.status.partial {{ background: var(--partial); }}
.status.implemented {{ background: var(--implemented); }}
.status.visually_verified {{ background: var(--verified); }}
.status.blocked {{ background: var(--blocked); }}
.status.missing {{ background: var(--missing); }}
</style>
</head>
<body>""".format(escape(title))


def html_footer() -> str:
    return "</body>\n</html>"


__all__ = [
    "build_stage7_goal_metrics_report",
    "load_stage6_corpus",
    "build_enriched_records",
    "build_stage7_failure_reports",
]
