from __future__ import annotations
import copy
import random
from typing import Any, Mapping, Sequence
from scoper.benchmark.common.canonical import sha256_json, validate_json_value
from scoper.benchmark.common.qa_overlay_runtime import QAOverlayRuntime
from scoper.benchmark.common.situated_qa import QA_OVERLAY_SCHEMA_VERSION, build_qa_overlay_candidates, compute_qa_clause_incomplete_state, compute_qa_overlay_metrics, normalize_task_for_qa, resolve_qa_overlay_answer, validate_qa_overlay
QA_TRACE_ATTACHMENT_SCHEMA_VERSION = 'qa-trace-attachment.v1'
QA_TRACE_EVENT_TRIGGERS = frozenset({'after_human_insert', 'after_tool_result', 'before_stop', 'subtask_handoff', 'task_step'})
_QUESTION_PRIORITY = {'insert_obligation_choice': 0, 'active_issue_choice': 1, 'remaining_requirement_choice': 2, 'post_action_outcome_choice': 3, 'composite_progress_choice': 4, 'target_state_choice': 5, 'target_relation_choice': 6, 'mid_execution_progress_choice': 7}

def attach_trace_conditioned_qa(task_spec: Mapping[str, Any], public_trace: Sequence[Mapping[str, Any]], evaluator_trace: Sequence[Mapping[str, Any]], *, seed: int, min_questions: int | None=None, max_questions: int | None=None, question_candidates: Sequence[Mapping[str, Any]] | None=None) -> dict[str, Any]:
    if 'qa_overlay' in task_spec:
        raise ValueError('trace-conditioned QA requires a QA-free physical task')
    public_events, evaluator_states = validate_qa_attachment_trace(public_trace, evaluator_trace)
    normalized_task = normalize_task_for_qa(task_spec)
    subtask_ids = [str(item['id']) for item in normalized_task.get('subtasks') or []]
    minimum, maximum = _question_limits(subtask_ids, min_questions, max_questions)
    candidates = _expanded_candidates(normalized_task) if question_candidates is None else copy.deepcopy(list(question_candidates))
    bank_policy = min_questions == 2 and max_questions == 3
    if bank_policy and question_candidates is None:
        candidates.extend(_bank_action_candidates(task_spec, normalized_task))
    eligible, candidate_report = _eligible_candidates(candidates, public_events, evaluator_states)
    selected = _select_candidates(eligible, subtask_ids=subtask_ids, limit=maximum, seed=int(seed), distinct_actions=bank_policy)
    base_hash = sha256_json(task_spec)
    report: dict[str, Any] = {'candidate_count': len(candidates), 'candidates': candidate_report, 'required_question_count': minimum, 'schema_version': QA_TRACE_ATTACHMENT_SCHEMA_VERSION, 'seed': int(seed), 'selected_question_count': len(selected), 'subtask_ids': subtask_ids, 'task_sha256': base_hash}
    flexible_coverage = min_questions == 2 and max_questions == 3
    if len(selected) < minimum or not _selection_shape_valid(selected, subtask_ids, allow_partial=flexible_coverage):
        report['status'] = 'qa_attachment_blocked'
        report['failure_reason'] = _selection_failure_reason(selected, subtask_ids, minimum)
        return _blocked_result(report, base_hash)
    questions = []
    for index, item in enumerate(selected):
        question = copy.deepcopy(item['question'])
        question['id'] = 'qa_overlay_{:02d}'.format(index)
        questions.append(question)
    overlay = validate_qa_overlay({'questions': questions, 'schema_version': QA_OVERLAY_SCHEMA_VERSION}, role_names=list((normalized_task.get('roles') or {}).keys()), subtask_ids=subtask_ids)
    replay = replay_qa_overlay(overlay, public_events, evaluator_states)
    if replay['status'] != 'passed':
        report['status'] = 'qa_attachment_blocked'
        report['failure_reason'] = 'scheduler_only_replay_failed'
        report['replay_sha256'] = sha256_json(replay)
        result = _blocked_result(report, base_hash)
        result['replay_record'] = replay
        return result
    attached_task = copy.deepcopy(dict(task_spec))
    attached_task['qa_overlay'] = overlay
    report.update({'attached_task_sha256': sha256_json(attached_task), 'overlay_sha256': sha256_json(overlay), 'replay_sha256': sha256_json(replay), 'selected': [{'activation_event_index': int(item['event_index']), 'owner_subtask_id': item['question'].get('owner_subtask_id'), 'question_type': item['question']['question_type']} for item in selected], 'status': 'attached'})
    result = {'attached_task': attached_task, 'base_task_sha256': base_hash, 'qa_overlay': overlay, 'replay_record': replay, 'schema_version': QA_TRACE_ATTACHMENT_SCHEMA_VERSION, 'selection_report': report, 'status': 'attached'}
    validate_json_value(result, json_pointer='/qa_trace_attachment')
    return result

def validate_qa_attachment_trace(public_trace: Sequence[Mapping[str, Any]], evaluator_trace: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(public_trace) != len(evaluator_trace):
        raise ValueError('public and evaluator QA traces must be aligned')
    public_events: list[dict[str, Any]] = []
    evaluator_states: list[dict[str, Any]] = []
    allowed_public = {'active_subtask_id', 'handoff_subtask_id', 'insert_id', 'task_action_step', 'total_step', 'transition', 'trigger'}
    allowed_private = {'active_issue_ids', 'active_subtask_id', 'clause_incomplete', 'composite_complete', 'last_tool_outcome', 'posted_insert_ids'}
    for index, (raw_public, raw_private) in enumerate(zip(public_trace, evaluator_trace)):
        if not isinstance(raw_public, Mapping) or not isinstance(raw_private, Mapping):
            raise ValueError('QA trace event {} is not an object'.format(index))
        if set(raw_public) - allowed_public:
            raise ValueError('public QA trace event contains unsupported fields')
        if set(raw_private) - allowed_private:
            raise ValueError('evaluator QA trace state contains unsupported fields')
        trigger = raw_public.get('trigger')
        if trigger not in QA_TRACE_EVENT_TRIGGERS:
            raise ValueError('public QA trace trigger is invalid')
        public_event = {'active_subtask_id': raw_public.get('active_subtask_id'), 'handoff_subtask_id': raw_public.get('handoff_subtask_id'), 'insert_id': raw_public.get('insert_id'), 'task_action_step': int(raw_public.get('task_action_step', 0)), 'total_step': int(raw_public.get('total_step', index + 1)), 'transition': copy.deepcopy(raw_public.get('transition')), 'trigger': trigger}
        if public_event['task_action_step'] < 0 or public_event['total_step'] < 0:
            raise ValueError('QA trace step values must be non-negative')
        for key in ('active_subtask_id', 'handoff_subtask_id', 'insert_id'):
            value = public_event[key]
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError('public QA trace {} is invalid'.format(key))
        transition = public_event['transition']
        if transition is not None and (not isinstance(transition, Mapping)):
            raise ValueError('public QA trace transition is invalid')
        public_events.append(public_event)
        evaluator_states.append(copy.deepcopy(dict(raw_private)))
    validate_json_value(public_events, json_pointer='/public_qa_trace')
    validate_json_value(evaluator_states, json_pointer='/evaluator_qa_trace')
    return (public_events, evaluator_states)

def replay_qa_overlay(overlay: Mapping[str, Any], public_trace: Sequence[Mapping[str, Any]], evaluator_trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    public_events, evaluator_states = validate_qa_attachment_trace(public_trace, evaluator_trace)
    runtime = QAOverlayRuntime(overlay)
    qa_state: dict[str, Any] = {}
    runtime.reset(qa_state)
    posted_ids: list[str] = []
    replay_events: list[dict[str, Any]] = []
    for event_index, (event, evaluator_state) in enumerate(zip(public_events, evaluator_states)):
        emitted = runtime.consider(qa_state, trigger=str(event['trigger']), total_step=int(event['total_step']), task_action_step=int(event['task_action_step']), active_subtask_id=event.get('active_subtask_id'), resolve_state=lambda question, state=evaluator_state: state, transition=event.get('transition'), insert_id=event.get('insert_id'), handoff_subtask_id=event.get('handoff_subtask_id'))
        while emitted:
            replay_events.extend(_public_replay_events(emitted, event_index))
            posted = [item for item in emitted if item.get('name') == 'human_qa_posted']
            if not posted:
                break
            question_id = str(posted[0]['payload']['id'])
            expected = qa_state.get('evaluator_keys', {}).get(question_id)
            if expected is None:
                break
            posted_ids.append(question_id)
            answer_step = int(event['total_step']) + 1
            qa_state.setdefault('answers', {})[question_id] = {'value': expected}
            qa_state.setdefault('answered_steps', {})[question_id] = answer_step
            qa_state.setdefault('pending', {}).pop(question_id, None)
            emitted = runtime.post_next(qa_state, total_step=answer_step, resolve_state=lambda question, state=evaluator_state: state)
    metrics = compute_qa_overlay_metrics(overlay, posted_ids=posted_ids, answers=qa_state.get('answers', {}), evaluator_keys=qa_state.get('evaluator_keys', {}), resolver_failure_ids=qa_state.get('qa_overlay_resolver_failures', []), posted_steps=qa_state.get('posted_steps', {}), answered_steps=qa_state.get('answered_steps', {}))
    result = {'answers': copy.deepcopy(qa_state.get('answers', {})), 'evaluator_keys': copy.deepcopy(qa_state.get('evaluator_keys', {})), 'events': replay_events, 'metrics': metrics, 'overlay_sha256': sha256_json(overlay), 'schema_version': 'qa-scheduler-replay.v1', 'status': 'passed' if metrics['qa_success'] == 1 and runtime.complete else 'failed'}
    validate_json_value(result, json_pointer='/qa_scheduler_replay')
    return result

def project_oracle_trace_for_qa(task_spec: Mapping[str, Any], trace_records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    meta = next((record for record in trace_records if isinstance(record, Mapping) and record.get('record_type') == 'meta'), {})
    prior_goal = dict((meta.get('reset') or {}).get('initial_goal') or {})
    step_records = [record for record in trace_records if isinstance(record, Mapping) and record.get('record_type') == 'step']
    subtask_ids = [str(item.get('subtask_id')) for item in ((task_spec.get('family_spec') or {}).get('composite') or {}).get('subtasks') or [] if isinstance(item, Mapping) and item.get('subtask_id')]
    current_active = subtask_ids[0] if subtask_ids else None
    posted_insert_ids: list[str] = []
    public_events: list[dict[str, Any]] = []
    evaluator_states: list[dict[str, Any]] = []

    def append_event(trigger: str, *, total_step: int, task_action_step: int, active_subtask_id: str | None, goal: Mapping[str, Any], transition: Mapping[str, Any] | None=None, insert_id: str | None=None, handoff_subtask_id: str | None=None) -> None:
        public_transition = None
        if isinstance(transition, Mapping):
            public_transition = {'arguments': copy.deepcopy(dict(transition.get('arguments') or {})), 'tool_name': str(transition.get('tool_name') or '')}
        public_events.append({'active_subtask_id': active_subtask_id, 'handoff_subtask_id': handoff_subtask_id, 'insert_id': insert_id, 'task_action_step': int(task_action_step), 'total_step': int(total_step), 'transition': public_transition, 'trigger': trigger})
        evaluator_states.append(_evaluator_state_from_goal(task_spec, goal, transition=transition, active_subtask_id=handoff_subtask_id if trigger == 'subtask_handoff' else active_subtask_id, posted_insert_ids=posted_insert_ids))
    for ordinal, record in enumerate(step_records, start=1):
        transition = record.get('transition') if isinstance(record.get('transition'), Mapping) else {}
        total_step = int(transition.get('high_level_step') or record.get('high_level_step') or ordinal)
        task_action_step = int(transition.get('task_action_step') or total_step)
        next_active = _active_subtask_from_goal(task_spec, prior_goal, subtask_ids, current_active)
        if current_active is not None and next_active is not None and (next_active != current_active):
            append_event('subtask_handoff', total_step=total_step, task_action_step=task_action_step, active_subtask_id=current_active, handoff_subtask_id=next_active, goal=prior_goal)
        current_active = next_active
        scheduler_events = [item for item in record.get('scheduler_events') or [] if isinstance(item, Mapping)]
        for event in scheduler_events:
            if event.get('name') == 'human_insert_posted' and event.get('phase') == 'pre_step':
                insert_id = str((event.get('payload') or {}).get('id') or '')
                if insert_id:
                    posted_insert_ids.append(insert_id)
                    append_event('after_human_insert', total_step=total_step, task_action_step=task_action_step, active_subtask_id=current_active, insert_id=insert_id, goal=prior_goal)
        append_event('task_step', total_step=total_step, task_action_step=task_action_step, active_subtask_id=current_active, goal=prior_goal)
        if transition.get('tool_name') == 'stop':
            append_event('before_stop', total_step=total_step, task_action_step=task_action_step, active_subtask_id=current_active, goal=prior_goal)
        current_goal = transition.get('parent_goal_result') if subtask_ids else None
        if not isinstance(current_goal, Mapping) or not list(current_goal.get('predicate_records') or current_goal.get('predicates') or []):
            current_goal = transition.get('goal_result')
        if not isinstance(current_goal, Mapping):
            current_goal = prior_goal
        append_event('after_tool_result', total_step=total_step, task_action_step=task_action_step, active_subtask_id=current_active, goal=current_goal, transition=transition)
        for event in scheduler_events:
            if event.get('name') == 'human_insert_posted' and event.get('phase') != 'pre_step':
                insert_id = str((event.get('payload') or {}).get('id') or '')
                if insert_id:
                    posted_insert_ids.append(insert_id)
                    append_event('after_human_insert', total_step=total_step, task_action_step=task_action_step, active_subtask_id=current_active, insert_id=insert_id, goal=current_goal, transition=transition)
        prior_goal = dict(current_goal)
    return validate_qa_attachment_trace(public_events, evaluator_states)

def _evaluator_state_from_goal(task_spec: Mapping[str, Any], goal: Mapping[str, Any], *, transition: Mapping[str, Any] | None, active_subtask_id: str | None, posted_insert_ids: Sequence[str]) -> dict[str, Any]:
    goal_specs = list((task_spec.get('goal') or {}).get('predicates') or [])
    records = list(goal.get('predicate_records') or goal.get('predicates') or [])
    raw_execution = transition.get('raw_execution') if isinstance(transition, Mapping) else None
    raw_status = raw_execution.get('status') if isinstance(raw_execution, Mapping) else None
    success = transition.get('success') if isinstance(transition, Mapping) else None
    outcome = 'ambiguous' if raw_status in {'ambiguous', 'uncertain', 'verify_required'} or not isinstance(success, bool) else 'success' if success else 'failure'
    return {'active_issue_ids': _active_issue_ids(task_spec, goal_specs, records), 'active_subtask_id': active_subtask_id, 'clause_incomplete': compute_qa_clause_incomplete_state(goal_specs, records), 'composite_complete': bool(active_subtask_id is None and _goal_complete(goal)), 'last_tool_outcome': outcome, 'posted_insert_ids': list(dict.fromkeys((str(item) for item in posted_insert_ids)))}

def _active_subtask_from_goal(task_spec: Mapping[str, Any], goal: Mapping[str, Any], subtask_ids: Sequence[str], fallback: str | None) -> str | None:
    if not subtask_ids:
        return None
    goal_specs = list((task_spec.get('goal') or {}).get('predicates') or [])
    records = list(goal.get('predicate_records') or goal.get('predicates') or [])
    if not records:
        return fallback
    if fallback is None:
        return None
    start_index = subtask_ids.index(fallback) if fallback in subtask_ids else 0
    for subtask_id in subtask_ids[start_index:]:
        statuses = [records[index].get('satisfied') for index, spec in enumerate(goal_specs) if isinstance(spec, Mapping) and spec.get('owner_subtask_id') == subtask_id and (index < len(records)) and isinstance(records[index], Mapping)]
        if statuses and (not all((value is True for value in statuses))):
            return subtask_id
    return None

def _active_issue_ids(task_spec: Mapping[str, Any], goal_specs: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> list[str]:
    family = task_spec.get('family_spec') if isinstance(task_spec.get('family_spec'), Mapping) else {}
    tidy = family.get('tidy') if isinstance(family.get('tidy'), Mapping) else {}
    recover = family.get('recover') if isinstance(family.get('recover'), Mapping) else {}
    raw_issues = list(tidy.get('violations') or [])
    if isinstance(recover.get('incident'), Mapping):
        raw_issues.append(recover['incident'])
    raw_issues.extend(recover.get('incidents') or [])
    active: list[str] = []
    for index, issue in enumerate(raw_issues):
        if not isinstance(issue, Mapping):
            continue
        issue_id = str(issue.get('id') or issue.get('violation_id') or 'issue_{}'.format(index))
        role = issue.get('object_role') or issue.get('target_role')
        matched = [item_index for item_index, spec in enumerate(goal_specs) if isinstance(spec, Mapping) and (role in spec.values() or (isinstance(spec.get('args'), Mapping) and role in spec['args'].values()))]
        if matched and any((item_index < len(records) and isinstance(records[item_index], Mapping) and (records[item_index].get('satisfied') is False) for item_index in matched)):
            active.append(issue_id)
    return sorted(set(active))

def _goal_complete(goal: Mapping[str, Any]) -> bool:
    if goal.get('predicate_success') == 1.0:
        return True
    records = list(goal.get('predicate_records') or goal.get('predicates') or [])
    return bool(records) and all((isinstance(record, Mapping) and record.get('satisfied') is True for record in records))

def _expanded_candidates(normalized_task: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = build_qa_overlay_candidates(normalized_task)
    subtasks = list(normalized_task.get('subtasks') or [])
    progress = [item for item in candidates if item.get('question_type') == 'composite_progress_choice']
    candidates = [item for item in candidates if item.get('question_type') != 'composite_progress_choice']
    if progress:
        for subtask in subtasks[1:]:
            candidate = copy.deepcopy(progress[0])
            candidate['activation'] = {'subtask_id': str(subtask['id']), 'type': 'subtask_handoff'}
            candidates.append(candidate)
    return candidates

def _eligible_candidates(candidates: Sequence[Mapping[str, Any]], public_events: Sequence[Mapping[str, Any]], evaluator_states: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for candidate_index, raw_question in enumerate(candidates):
        question = copy.deepcopy(dict(raw_question))
        question['id'] = 'qa_candidate_{:03d}'.format(candidate_index)
        activation_index = None
        answer = None
        for event_index, (event, state) in enumerate(zip(public_events, evaluator_states)):
            owner = question.get('owner_subtask_id')
            if isinstance(owner, str) and owner != event.get('active_subtask_id'):
                continue
            if not QAOverlayRuntime._activation_matches(question.get('activation') or {}, trigger=str(event['trigger']), task_action_step=int(event['task_action_step']), transition=event.get('transition'), insert_id=event.get('insert_id'), handoff_subtask_id=event.get('handoff_subtask_id')):
                continue
            activation_index = event_index
            answer = resolve_qa_overlay_answer(question, state)
            break
        is_eligible = activation_index is not None and answer is not None
        row = {'activation_event_index': activation_index, 'candidate_index': candidate_index, 'eligible': is_eligible, 'owner_subtask_id': question.get('owner_subtask_id'), 'question_type': question.get('question_type'), 'reason': None if is_eligible else 'activation_missing' if activation_index is None else 'first_activation_not_uniquely_resolvable'}
        report.append(row)
        if is_eligible:
            eligible.append({'answer': answer, 'candidate_index': candidate_index, 'event_index': activation_index, 'question': question})
    return (eligible, report)

def _bank_action_candidates(task_spec: Mapping[str, Any], normalized: Mapping[str, Any]) -> list[dict[str, Any]]:
    if normalized.get('subtasks'):
        return []
    roles = normalized.get('roles') or {}
    actions = []
    for predicate in (task_spec.get('goal') or {}).get('predicates') or task_spec.get('goal_predicates') or []:
        if predicate.get('type') == 'object_in_any_receptacle':
            for role in predicate.get('receptacle_roles') or []:
                if role in roles:
                    actions.append({'tool_name': 'put_down', 'target_role_or_null': role, 'target_label': roles[role]['label']})
    actions.append({'tool_name': 'scan', 'target_role_or_null': None, 'target_label': 'the scene'})
    for tool in ('pick_up', 'navigate_to'):
        for role, details in roles.items():
            actions.append({'tool_name': tool, 'target_role_or_null': role, 'target_label': details['label']})
    candidates = build_qa_overlay_candidates({'action_candidates': actions})
    labels = {'put_down': 'placing the object at', 'pick_up': 'picking up', 'navigate_to': 'navigating to', 'scan': 'scanning'}
    for question in candidates:
        activation = question['activation']
        role = activation.get('target_role_or_null')
        label = roles[role]['label'] if role in roles else 'the scene'
        question['question'] = 'What was the outcome of {} {}?'.format(labels[activation['tool_name']], label)
    return candidates

def _select_candidates(eligible: Sequence[Mapping[str, Any]], *, subtask_ids: Sequence[str], limit: int, seed: int, distinct_actions: bool=False) -> list[dict[str, Any]]:
    ranked = [copy.deepcopy(dict(item)) for item in eligible]
    random.Random(seed).shuffle(ranked)
    ranked.sort(key=lambda item: (_QUESTION_PRIORITY[str(item['question']['question_type'])], int(item['event_index']) if subtask_ids and item['question']['question_type'] == 'post_action_outcome_choice' else 0))
    if subtask_ids:
        selected = []
        for subtask_id in subtask_ids:
            match = next((item for item in ranked if item['question'].get('owner_subtask_id') == subtask_id), None)
            if match is not None:
                selected.append(match)
        if len(subtask_ids) == 2:
            progress = next((item for item in ranked if item['question'].get('question_type') == 'composite_progress_choice' and item['question'].get('owner_subtask_id') is None), None)
            if progress is not None:
                selected.append(progress)
        return selected[:limit]
    selected = []
    used_types: set[Any] = set()
    for item in ranked:
        question_type = str(item['question']['question_type'])
        signature: Any = question_type
        if distinct_actions and question_type == 'post_action_outcome_choice':
            signature = (question_type, item['question']['activation'].get('tool_name'))
        if signature in used_types:
            continue
        selected.append(item)
        used_types.add(signature)
        if len(selected) == limit:
            break
    return selected

def _question_limits(subtask_ids: Sequence[str], minimum: int | None, maximum: int | None) -> tuple[int, int]:
    if minimum == 2 and maximum == 3:
        if subtask_ids and len(subtask_ids) not in {2, 3}:
            raise ValueError('trace-conditioned composite QA requires two or three children')
        return (2, 3)
    if subtask_ids:
        required = 3
        if len(subtask_ids) not in {2, 3}:
            raise ValueError('trace-conditioned composite QA requires two or three children')
        if minimum not in (None, required) or maximum not in (None, required):
            raise ValueError('composite trace-conditioned QA requires exactly three questions')
        return (required, required)
    low = 1 if minimum is None else int(minimum)
    high = 2 if maximum is None else int(maximum)
    if not 1 <= low <= high <= 2:
        raise ValueError('atomic trace-conditioned QA requires one or two questions')
    return (low, high)

def _selection_shape_valid(selected: Sequence[Mapping[str, Any]], subtask_ids: Sequence[str], *, allow_partial: bool=False) -> bool:
    if not subtask_ids:
        return bool(selected)
    owners = [item['question'].get('owner_subtask_id') for item in selected]
    if allow_partial:
        covered = [owner for owner in owners if owner is not None]
        return len(set(covered)) >= 2 and len(covered) == len(set(covered))
    if any((owners.count(subtask_id) != 1 for subtask_id in subtask_ids)):
        return False
    if len(subtask_ids) == 2:
        return any((item['question'].get('question_type') == 'composite_progress_choice' and item['question'].get('owner_subtask_id') is None for item in selected))
    return len(selected) == 3

def _selection_failure_reason(selected: Sequence[Mapping[str, Any]], subtask_ids: Sequence[str], minimum: int) -> str:
    if len(selected) < minimum:
        return 'insufficient_trace_conditioned_questions'
    if subtask_ids:
        return 'composite_question_coverage_incomplete'
    return 'qa_selection_invalid'

def _blocked_result(report: Mapping[str, Any], base_hash: str) -> dict[str, Any]:
    result = {'attached_task': None, 'base_task_sha256': base_hash, 'qa_overlay': None, 'replay_record': None, 'schema_version': QA_TRACE_ATTACHMENT_SCHEMA_VERSION, 'selection_report': copy.deepcopy(dict(report)), 'status': 'qa_attachment_blocked'}
    validate_json_value(result, json_pointer='/qa_trace_attachment')
    return result

def _public_replay_events(events: Sequence[Mapping[str, Any]], event_index: int) -> list[dict[str, Any]]:
    return [{'event_index': int(event_index), 'name': str(event.get('name') or 'qa_overlay_event'), 'question_id': (event.get('payload') or {}).get('id')} for event in events]
__all__ = ['QA_TRACE_ATTACHMENT_SCHEMA_VERSION', 'QA_TRACE_EVENT_TRIGGERS', 'attach_trace_conditioned_qa', 'project_oracle_trace_for_qa', 'replay_qa_overlay', 'validate_qa_attachment_trace']
