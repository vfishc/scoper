"""Opt-in, append-only QA bank extension anchored in existing action references.

This prepares new inputs; it does not fabricate visual-review evidence or scores.
Old overlays remain unchanged. Answers are resolved by the live runtime.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping

from .generic_qa import _clause, build_generic_qa_question
from .situated_qa import normalize_task_for_qa, validate_qa_overlay

GENERIC_TYPES = ("target_state_choice", "target_relation_choice", "mid_execution_progress_choice")
# Temperature requires more than merely naming a heat/cold-source action.
VISUAL_STATE_PREDICATES = {"object_clean", "object_dirty", "object_open", "object_closed",
                           "object_toggled_on", "object_toggled_off", "object_sliced"}
INTERACTION_TOOLS = {"pick_up", "put_down", "open", "close", "toggle_on", "toggle_off",
                     "slice", "cut", "wipe", "soak_under", "drop", "heat", "cool"}


def _words(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _named(label, instruction):
    # Deliberately conservative: do not disclose latent goal objects in questions.
    label = _words(label)
    return bool(label and re.search(r"\b" + re.escape(label) + r"(?:s|es)?\b", _words(instruction)))


def append_generic_questions(task: Mapping, reference: Mapping):
    original = copy.deepcopy(dict(task))
    report = {"task_id": task["id"], "added": [], "skipped": []}
    children = ((task.get("family_spec") or {}).get("composite") or {}).get("subtasks") or []
    if children:
        report["skipped"].append("composite_scope_preserved")
        return original, report
    if reference.get("status") != "ready" or reference.get("task_id") != task["id"]:
        raise ValueError("missing or mismatched ready action reference")
    old = validate_qa_overlay(task["qa_overlay"], role_names=list(task.get("roles") or {}))
    normalized = normalize_task_for_qa(task)
    roles = task.get("roles") or {}
    reverse = {}
    for name, role in roles.items():
        for field in ("object_id", "bddl_inst"):
            if role.get(field): reverse.setdefault(str(role[field]), set()).add(name)
    nodes = reference.get("nodes") or []
    seen, anchors = set(), []
    for i, node in enumerate(nodes):
        target_roles = reverse.get(str(node.get("target")), set())
        subject_roles = reverse.get(str(node.get("subject")), set())
        seen.update(target_roles | subject_roles)
        if node.get("tool") in INTERACTION_TOOLS and len(target_roles) == 1:
            role = next(iter(target_roles))
            # Runtime activation is first matching tool+role; do not use a later
            # repeat to justify an earlier trigger.
            if any(a[1] == node["tool"] and a[2] == role for a in anchors): continue
            anchors.append((i, node["tool"], role, set(seen), target_roles | subject_roles))
    clauses = []
    for i, pred in enumerate((task.get("goal") or {}).get("predicates") or []):
        try:
            kind, _, required = _clause(task, i)
        except ValueError:
            continue
        if kind == "target_state_choice" and pred.get("type") not in VISUAL_STATE_PREDICATES:
            continue
        if not all(_named(normalized["roles"][r]["label"], task["instruction"]) for r in required):
            continue
        clauses.append((i, kind, required))
    existing_types = {q["question_type"] for q in old["questions"]}
    used_ids = {q["id"] for q in old["questions"]}
    additions = []
    for kind in GENERIC_TYPES:
        if kind in existing_types:
            report["skipped"].append(kind + ":already_present")
            continue
        choices = []
        for anchor in anchors:
            available = [c for c in clauses if c[2] <= anchor[3] and c[2] & anchor[4]]
            if kind != "mid_execution_progress_choice":
                available = [c for c in available if c[1] == kind]
            if not available: continue
            # Prefer a state/relation pair for progress, while all referents must
            # already occur in the reference prefix. Never use elapsed tool steps.
            selected = available[:1]
            if kind == "mid_execution_progress_choice":
                other = next((c for c in clauses if c[0] != selected[0][0]
                              and c[2] <= anchor[3] and c[1] != selected[0][1]), None)
                if other: selected.append(other)
            choices.append((anchor, selected))
        if not choices:
            report["skipped"].append(kind + ":no_publicly_named_action_anchor")
            continue
        if kind == "mid_execution_progress_choice":
            choices.sort(key=lambda item: (-len(item[1]), item[0][0]))
        anchor, selected = choices[0]
        qid = "qa_added_" + kind
        while qid in used_ids: qid += "_new"
        activation = {"type": "after_tool_result", "tool_name": anchor[1], "target_role_or_null": anchor[2]}
        q = build_generic_qa_question(task, question_type=kind,
            clause_indices=[c[0] for c in selected], activation=activation, question_id=qid)
        additions.append(q);used_ids.add(qid)
        report["added"].append({"question_id": qid, "question_type": kind,
            "clause_indices": [c[0] for c in selected], "reference_node_index": anchor[0],
            "activation": activation, "reference_trace_sha256": reference.get("trace_sha256"),
            "evidence_status": "public_name_and_reference_action_checked; not_image_reviewed"})
    if additions:
        overlay = {"schema_version": "qa_overlay.v2", "questions": [*old["questions"], *additions]}
        original["qa_overlay"] = validate_qa_overlay(overlay, role_names=list(roles))
        assert original["qa_overlay"]["questions"][:len(old["questions"])] == old["questions"]
    return original, report
