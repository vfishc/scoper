"""Goal-grounded QA candidates and evidence-reviewed attachment to saved episodes."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .qa_overlay_runtime import QAOverlayRuntime
from .situated_qa import normalize_task_for_qa, validate_qa_overlay
from .trace_conditioned_qa import attach_trace_conditioned_qa, project_oracle_trace_for_qa


STATE_PHRASES = {
    "object_clean": "is clean", "object_dirty": "is dirty",
    "object_open": "is open", "object_closed": "is closed",
    "object_toggled_on": "is turned on", "object_toggled_off": "is turned off",
    "object_hot": "is hot", "object_cool": "is cool", "object_sliced": "is sliced",
}
RELATION_TYPES = {"object_in_receptacle", "object_on_receptacle", "object_in_any_receptacle"}


def _role(predicate: Mapping[str, Any], name: str) -> str | None:
    args = predicate.get("args") or {}
    return predicate.get(name + "_role") or predicate.get(name) or args.get(name)


def _clause(task: Mapping[str, Any], index: int) -> tuple[str, str, set[str]]:
    predicates = (task.get("goal") or {}).get("predicates") or []
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(predicates):
        raise ValueError("invalid goal clause index")
    predicate = predicates[index]
    roles = task.get("roles") or {}
    normalized = normalize_task_for_qa(task)
    required_roles = set()

    def label(role: str | None) -> str:
        metadata = roles.get(role) or {}
        if not role or not any(metadata.get(field) for field in ("object_type", "category", "synset")):
            raise ValueError("goal role has no public object label")
        required_roles.add(role)
        return "the " + normalized["roles"][role]["label"]

    object_label = label(_role(predicate, "object"))
    predicate_type = predicate.get("type")
    if predicate_type in STATE_PHRASES:
        sentence = object_label + " " + STATE_PHRASES[predicate_type]
        kind = "target_state_choice"
    elif predicate_type in RELATION_TYPES:
        if predicate_type == "object_in_any_receptacle":
            sentence = object_label + " is in an appropriate storage receptacle"
        else:
            relation = "inside" if predicate_type == "object_in_receptacle" else "on"
            if predicate_type == "object_in_receptacle" and task.get("backend") == "procthor":
                relation = "in or on"
            sentence = object_label + " is " + relation + " " + label(_role(predicate, "receptacle"))
        kind = "target_relation_choice"
    else:
        raise ValueError("unsupported generic QA predicate: {}".format(predicate_type))
    return kind, sentence, required_roles


def build_generic_qa_question(
    task: Mapping[str, Any], *, question_type: str, clause_indices: Sequence[int],
    activation: Mapping[str, Any], question_id: str = "generic_qa",
) -> dict[str, Any]:
    if not 1 <= len(clause_indices) <= 2 or len(set(clause_indices)) != len(clause_indices):
        raise ValueError("generic QA requires one or two unique goal clauses")
    clauses = [_clause(task, index) for index in clause_indices]
    if question_type != "mid_execution_progress_choice":
        if len(clauses) != 1 or question_type != clauses[0][0]:
            raise ValueError("generic QA type does not match its goal predicate")
        question = "Is the following statement true now: {}?".format(clauses[0][1])
        answers = ["yes", "no"]
        mapping = {"0": "yes", "1": "no"}
    elif len(clauses) == 1:
        question = "Has this requirement been completed so far: {}?".format(clauses[0][1])
        answers = ["completed", "not completed"]
        mapping = {"0": answers[0], "1": answers[1]}
    else:
        question = "Which requirements are complete so far? 1: {}. 2: {}.".format(clauses[0][1], clauses[1][1])
        answers = ["both", "only requirement 1", "only requirement 2", "neither"]
        mapping = {"00": answers[0], "01": answers[1], "10": answers[2], "11": answers[3]}
    normalized = normalize_task_for_qa(task)
    result = {
        "id": question_id, "question_type": question_type, "question": question,
        "answer_set": answers, "activation": copy.deepcopy(dict(activation)),
        "resolver": {"type": question_type,
                     "clause_ids": [normalized["goal_clauses"][index]["id"] for index in clause_indices],
                     "answer_by_incomplete_mask": mapping},
    }
    return validate_qa_overlay(
        {"schema_version": "qa_overlay.v1", "questions": [result]}, role_names=list(task.get("roles") or {})
    )["questions"][0]


def attach_reviewed_generic_qa(
    task: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]], *, seed: int = 0,
) -> dict[str, Any]:
    if not 1 <= len(requests) <= 2:
        raise ValueError("single-task attachment requires one or two reviewed questions")
    if ((task.get("family_spec") or {}).get("composite") or {}).get("subtasks"):
        raise ValueError("generic attachment currently supports single tasks")
    predicate_count = len((task.get("goal") or {}).get("predicates") or [])
    meta = next((record for record in records if record.get("record_type") == "meta"), {})
    if meta.get("task_id") != task.get("id"):
        raise ValueError("trace task identity differs")
    if meta.get("instruction") != task.get("instruction"):
        raise ValueError("trace instruction differs from base task")
    goals = [((meta.get("reset") or {}).get("initial_goal") or {})]
    steps = [record for record in records if record.get("record_type") == "step"]
    if not steps or not predicate_count:
        raise ValueError("missing step records or task predicates")
    for index, record in enumerate(steps, start=1):
        transition = record.get("transition") or {}
        if transition.get("high_level_step") != index or not transition.get("tool_name"):
            raise ValueError("missing or out-of-order action evidence")
    if not any(record.get("record_type") == "summary" for record in records):
        raise ValueError("episode trace has no completion summary")
    goals.extend((record.get("transition") or {}).get("goal_result") or {} for record in steps)
    for goal in goals:
        values = goal.get("predicate_records") or goal.get("predicates") or []
        if len(values) != predicate_count or any(not isinstance(value.get("satisfied"), bool) for value in values):
            raise ValueError("incomplete per-step predicate evidence")
        if [value.get("type") for value in values] != [value.get("type") for value in task["goal"]["predicates"]]:
            raise ValueError("trace predicate order or types differ from base task")
    public, evaluator = project_oracle_trace_for_qa(task, records)
    candidates = []
    evidence_manifest = []
    for request in requests:
        if request.get("visually_reviewed") is not True:
            raise ValueError("generic attachment requires reviewed public image evidence")
        question = build_generic_qa_question(
            task, question_type=request["question_type"], clause_indices=request["clause_indices"],
            activation=request["activation"],
        )
        matching = [event for event in public if QAOverlayRuntime._activation_matches(
            question["activation"], trigger=event["trigger"], task_action_step=event["task_action_step"],
            transition=event["transition"], insert_id=event["insert_id"], handoff_subtask_id=event["handoff_subtask_id"],
        )]
        if not matching:
            raise ValueError("reviewed question activation did not occur")
        event = matching[0]
        latest_step = event["total_step"] - int(event["trigger"] in {"task_step", "before_stop"})
        required_roles = set().union(*[_clause(task, index)[2] for index in request["clause_indices"]])
        observed_roles = set()
        evidence_rows = []
        for evidence in request.get("evidence") or []:
            step = evidence.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= latest_step:
                raise ValueError("image evidence occurs after the question or lacks a valid step")
            path = Path(evidence["frame_path"])
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError("reviewed public image is missing")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if evidence.get("sha256") != digest:
                raise ValueError("reviewed public image hash differs")
            observed_roles.update(evidence.get("roles") or [])
            evidence_rows.append(dict(evidence))
        if not required_roles.issubset(observed_roles):
            raise ValueError("missing reviewed visual evidence for referenced roles")
        candidates.append(question)
        evidence_manifest.append({"question_type": question["question_type"], "evidence": evidence_rows})
    result = attach_trace_conditioned_qa(
        task, public, evaluator, seed=seed, min_questions=len(candidates), max_questions=len(candidates),
        question_candidates=candidates,
    )
    result["reviewed_evidence"] = evidence_manifest
    return result
