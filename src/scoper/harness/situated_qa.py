from __future__ import annotations
import copy
import itertools
import random
from typing import Any, Mapping, Sequence
from scoper.harness.canonical import sha256_json, validate_json_value
QA_OVERLAY_SCHEMA_VERSION = 'qa_overlay.v1'
QA_OVERLAY_EXTENDED_SCHEMA_VERSION = 'qa_overlay.v2'
QA_OVERLAY_QUESTION_TYPES = frozenset({'active_issue_choice', 'composite_progress_choice', 'insert_obligation_choice', 'post_action_outcome_choice', 'remaining_requirement_choice', 'target_state_choice', 'target_relation_choice', 'mid_execution_progress_choice'})
QA_OVERLAY_ACTIVATION_TYPES = frozenset({'after_human_insert', 'after_tool_result', 'before_stop', 'subtask_handoff', 'task_step'})

def validate_qa_overlay(value: Any, *, role_names: Sequence[str]=(), subtask_ids: Sequence[str]=()) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {'schema_version', 'questions'}:
        raise ValueError('qa_overlay must contain schema_version and questions')
    schema_version = value.get('schema_version')
    if schema_version not in {QA_OVERLAY_SCHEMA_VERSION, QA_OVERLAY_EXTENDED_SCHEMA_VERSION}:
        raise ValueError('qa_overlay schema_version is invalid')
    questions = value.get('questions')
    maximum = 3 if schema_version == QA_OVERLAY_SCHEMA_VERSION else 6
    if not isinstance(questions, list) or not 1 <= len(questions) <= maximum:
        raise ValueError(f'qa_overlay questions must contain one to {maximum} entries')
    known_roles = set(role_names)
    known_subtasks = set(subtask_ids)
    seen_ids: set[str] = set()
    normalized_questions: list[dict[str, Any]] = []
    owner_counts: dict[str, int] = {}
    for index, raw in enumerate(questions):
        if not isinstance(raw, Mapping):
            raise ValueError('qa_overlay question {} is not an object'.format(index))
        required = {'id', 'question_type', 'question', 'answer_set', 'activation', 'resolver'}
        optional = {'owner_subtask_id'}
        if set(raw) != required and set(raw) != required | optional:
            raise ValueError('qa_overlay question fields are invalid')
        question_id = raw.get('id')
        if not isinstance(question_id, str) or not question_id or question_id in seen_ids:
            raise ValueError('qa_overlay question id is invalid or duplicated')
        seen_ids.add(question_id)
        question_type = raw.get('question_type')
        if question_type not in QA_OVERLAY_QUESTION_TYPES:
            raise ValueError('qa_overlay question_type is invalid')
        question = raw.get('question')
        if not isinstance(question, str) or not question.strip():
            raise ValueError('qa_overlay question text is invalid')
        answers = raw.get('answer_set')
        if not isinstance(answers, list) or not 2 <= len(answers) <= 4 or any((not isinstance(item, str) or not item.strip() for item in answers)) or (len(set(answers)) != len(answers)):
            raise ValueError('qa_overlay answer_set must contain two to four unique strings')
        owner = raw.get('owner_subtask_id')
        if owner is not None:
            if not isinstance(owner, str) or not owner:
                raise ValueError('qa_overlay owner_subtask_id is invalid')
            if known_subtasks and owner not in known_subtasks:
                raise ValueError('qa_overlay owner_subtask_id is unknown')
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
            if owner_counts[owner] > 1:
                raise ValueError('qa_overlay permits at most one question per child')
        normalized: dict[str, Any] = {'activation': _validate_activation(raw['activation'], known_roles, known_subtasks), 'answer_set': list(answers), 'id': question_id, 'question': question, 'question_type': question_type, 'resolver': _validate_resolver(raw['resolver'], question_type, answers)}
        if owner is not None:
            normalized['owner_subtask_id'] = owner
        normalized_questions.append(normalized)
    result = {'questions': normalized_questions, 'schema_version': schema_version}
    validate_json_value(result, json_pointer='/qa_overlay')
    return result

def _validate_activation(value: Any, known_roles: set[str], known_subtasks: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError('qa_overlay activation is invalid')
    activation_type = value.get('type')
    allowed_by_type = {'after_tool_result': {'type', 'tool_name', 'target_role_or_null'}, 'task_step': {'type', 'after_step'}, 'after_human_insert': {'type', 'insert_id'}, 'subtask_handoff': {'type', 'subtask_id'}, 'before_stop': {'type'}}
    if activation_type not in QA_OVERLAY_ACTIVATION_TYPES or set(value) != allowed_by_type[activation_type]:
        raise ValueError('qa_overlay activation fields are invalid')
    result = copy.deepcopy(dict(value))
    if activation_type == 'after_tool_result':
        if not isinstance(result.get('tool_name'), str) or not result['tool_name']:
            raise ValueError('after_tool_result requires tool_name')
        role = result.get('target_role_or_null')
        if role is not None and (not isinstance(role, str) or (known_roles and role not in known_roles)):
            raise ValueError('after_tool_result target role is invalid')
    elif activation_type == 'task_step':
        step = result.get('after_step')
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('task_step requires non-negative after_step')
    elif activation_type == 'after_human_insert':
        if not isinstance(result.get('insert_id'), str) or not result['insert_id']:
            raise ValueError('after_human_insert requires insert_id')
    elif activation_type == 'subtask_handoff':
        subtask_id = result.get('subtask_id')
        if not isinstance(subtask_id, str) or not subtask_id or (known_subtasks and subtask_id not in known_subtasks):
            raise ValueError('subtask_handoff requires known subtask_id')
    return result

def _validate_resolver(value: Any, question_type: str, answers: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get('type') != question_type:
        raise ValueError('qa_overlay resolver must match question_type')
    result = copy.deepcopy(dict(value))
    expected_keys: set[str]
    resolver_answers: list[Any]
    if question_type == 'post_action_outcome_choice':
        expected_keys = {'type', 'answer_by_outcome'}
        mapping = result.get('answer_by_outcome')
        if not isinstance(mapping, Mapping) or set(mapping) != {'success', 'failure', 'ambiguous'}:
            raise ValueError('post-action resolver outcomes are invalid')
        resolver_answers = list(mapping.values())
    elif question_type in {'remaining_requirement_choice', 'target_state_choice', 'target_relation_choice', 'mid_execution_progress_choice'}:
        expected_keys = {'type', 'clause_ids', 'answer_by_incomplete_mask'}
        clause_ids = result.get('clause_ids')
        if not isinstance(clause_ids, list) or not 1 <= len(clause_ids) <= 2 or any((not isinstance(item, str) or not item for item in clause_ids)) or (len(set(clause_ids)) != len(clause_ids)):
            raise ValueError('remaining-requirement clause_ids are invalid')
        mapping = result.get('answer_by_incomplete_mask')
        masks = {''.join(bits) for bits in itertools.product('01', repeat=len(clause_ids))}
        if not isinstance(mapping, Mapping) or set(mapping) != masks:
            raise ValueError('remaining-requirement masks are incomplete')
        resolver_answers = list(mapping.values())
    elif question_type == 'active_issue_choice':
        expected_keys = {'type', 'issue_ids', 'answer_by_active_issue_id', 'none_answer'}
        issue_ids = result.get('issue_ids')
        mapping = result.get('answer_by_active_issue_id')
        if not isinstance(issue_ids, list) or not 1 <= len(issue_ids) <= 3 or any((not isinstance(item, str) or not item for item in issue_ids)) or (len(set(issue_ids)) != len(issue_ids)) or (not isinstance(mapping, Mapping)) or (set(mapping) != set(issue_ids)):
            raise ValueError('active-issue resolver is invalid')
        resolver_answers = list(mapping.values()) + [result.get('none_answer')]
    elif question_type == 'composite_progress_choice':
        expected_keys = {'type', 'subtask_ids', 'answer_by_active_subtask_id', 'all_complete_answer'}
        subtask_ids = result.get('subtask_ids')
        mapping = result.get('answer_by_active_subtask_id')
        if not isinstance(subtask_ids, list) or not 2 <= len(subtask_ids) <= 3 or any((not isinstance(item, str) or not item for item in subtask_ids)) or (len(set(subtask_ids)) != len(subtask_ids)) or (not isinstance(mapping, Mapping)) or (set(mapping) != set(subtask_ids)):
            raise ValueError('composite-progress resolver is invalid')
        resolver_answers = list(mapping.values()) + [result.get('all_complete_answer')]
    else:
        expected_keys = {'type', 'insert_id', 'answer'}
        if not isinstance(result.get('insert_id'), str) or not result['insert_id']:
            raise ValueError('insert-obligation resolver insert_id is invalid')
        resolver_answers = [result.get('answer')]
    if set(result) != expected_keys:
        raise ValueError('qa_overlay resolver fields are invalid')
    if any((not isinstance(item, str) or item not in answers for item in resolver_answers)):
        raise ValueError('qa_overlay resolver answer is outside answer_set')
    return result

def normalize_task_for_qa(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    roles = task_spec.get('roles') if isinstance(task_spec.get('roles'), Mapping) else {}
    normalized_roles = {str(name): {'label': _role_label(str(name), role), 'room_id_or_null': role.get('room_id') if isinstance(role, Mapping) else None} for name, role in sorted(roles.items())}
    raw_goal = task_spec.get('goal') if isinstance(task_spec.get('goal'), Mapping) else {}
    predicates = list(raw_goal.get('predicates') or task_spec.get('goal_predicates') or [])
    clauses: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for index, predicate in enumerate(predicates):
        if not isinstance(predicate, Mapping):
            continue
        clause_id = str(predicate.get('id') or 'goal_clause_{}'.format(index))
        owner = predicate.get('owner_subtask_id')
        clauses.append({'id': clause_id, 'label': _predicate_label(predicate, normalized_roles), 'owner_subtask_id_or_null': owner if isinstance(owner, str) else None})
        action = _action_candidate(predicate, normalized_roles)
        if action is not None:
            action['clause_id'] = clause_id
            action['owner_subtask_id_or_null'] = owner if isinstance(owner, str) else None
            actions.append(action)
    family = task_spec.get('family_spec') if isinstance(task_spec.get('family_spec'), Mapping) else {}
    human = family.get('human') if isinstance(family.get('human'), Mapping) else {}
    inserts: list[dict[str, str]] = []
    for event in human.get('human_events') or human.get('human_events_template') or []:
        if isinstance(event, Mapping) and event.get('type') == 'insert':
            text = event.get('instruction', event.get('instruction_template'))
            if isinstance(text, str) and text:
                inserts.append({'id': str(event.get('id') or 'insert'), 'label': text})
    issues = _normalized_issues(family, normalized_roles)
    composite = family.get('composite') if isinstance(family.get('composite'), Mapping) else {}
    subtasks: list[dict[str, str]] = []
    for subtask in composite.get('subtasks') or []:
        if not isinstance(subtask, Mapping):
            continue
        subtask_id = subtask.get('subtask_id')
        if isinstance(subtask_id, str) and subtask_id:
            subtasks.append({'id': subtask_id, 'label': str(subtask.get('instruction_fragment') or subtask.get('category') or subtask_id)})
    normalized = {'action_candidates': actions, 'goal_clauses': clauses, 'human_inserts': inserts, 'instruction': str(task_spec.get('instruction') or ''), 'issues': issues, 'roles': normalized_roles, 'requires_model_stop': bool(raw_goal.get('require_model_stop', False)), 'subtasks': subtasks, 'task_id': str(task_spec.get('id') or task_spec.get('task_id') or 'task')}
    validate_json_value(normalized, json_pointer='/normalized_task_for_qa')
    return normalized

def build_qa_overlay_spec(normalized_task: Mapping[str, Any], *, max_questions: int | None=None, seed: int=0) -> dict[str, Any]:
    subtasks = list(normalized_task.get('subtasks') or [])
    limit = int(max_questions if max_questions is not None else 3 if subtasks else 2)
    limit = max(1, min(limit, 3 if subtasks else 2))
    candidates = _qa_candidates(normalized_task)
    random.Random(int(seed)).shuffle(candidates)
    chosen: list[dict[str, Any]] = []
    used_owners: set[str] = set()
    for candidate in candidates:
        owner = candidate.get('owner_subtask_id')
        if isinstance(owner, str) and owner in used_owners:
            continue
        chosen.append(candidate)
        if isinstance(owner, str):
            used_owners.add(owner)
        if len(chosen) >= limit:
            break
    if not chosen:
        raise ValueError('task has no supported situated QA candidates')
    questions: list[dict[str, Any]] = []
    for index, candidate in enumerate(chosen):
        question = copy.deepcopy(candidate)
        question['id'] = 'qa_overlay_{:02d}'.format(index)
        questions.append(question)
    return validate_qa_overlay({'questions': questions, 'schema_version': QA_OVERLAY_SCHEMA_VERSION}, role_names=list((normalized_task.get('roles') or {}).keys()), subtask_ids=[str(item['id']) for item in subtasks])

def build_qa_overlay_candidates(normalized_task: Mapping[str, Any]) -> list[dict[str, Any]]:
    return copy.deepcopy(_qa_candidates(normalized_task))

def _qa_candidates(normalized_task: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    actions = list(normalized_task.get('action_candidates') or [])
    clauses = list(normalized_task.get('goal_clauses') or [])
    inserts = list(normalized_task.get('human_inserts') or [])
    issues = list(normalized_task.get('issues') or [])
    subtasks = list(normalized_task.get('subtasks') or [])
    for action in actions:
        target = action.get('target_label') or 'the target'
        answers = ['the action succeeded', 'the action failed', 'the result is uncertain']
        candidate = {'activation': {'target_role_or_null': action.get('target_role_or_null'), 'tool_name': action['tool_name'], 'type': 'after_tool_result'}, 'answer_set': answers, 'question': 'What was the outcome of the attempted action on {}?'.format(target), 'question_type': 'post_action_outcome_choice', 'resolver': {'answer_by_outcome': {'ambiguous': answers[2], 'failure': answers[1], 'success': answers[0]}, 'type': 'post_action_outcome_choice'}}
        owner = action.get('owner_subtask_id_or_null')
        if isinstance(owner, str):
            candidate['owner_subtask_id'] = owner
        candidates.append(candidate)
    if clauses and normalized_task.get('requires_model_stop') is True:
        selected = clauses[:2]
        labels = [str(item['label']) for item in selected]
        if len(selected) == 1:
            answers = [labels[0] + ' remains', 'that requirement is complete']
            status_answers = {'0': answers[1], '1': answers[0]}
        else:
            answers = ['requirement 1 ({}) remains'.format(labels[0]), 'requirement 2 ({}) remains'.format(labels[1]), 'both remain', 'neither remains']
            status_answers = {'00': answers[3], '01': answers[1], '10': answers[0], '11': answers[2]}
        candidates.append({'activation': {'type': 'before_stop'}, 'answer_set': answers, 'question': 'Which listed task requirement is still incomplete?', 'question_type': 'remaining_requirement_choice', 'resolver': {'answer_by_incomplete_mask': status_answers, 'clause_ids': [str(item['id']) for item in selected], 'type': 'remaining_requirement_choice'}})
    if issues:
        selected_issues = []
        seen_issue_labels: set[str] = set()
        for issue in issues:
            label = str(issue['label'])
            if label in seen_issue_labels:
                continue
            seen_issue_labels.add(label)
            selected_issues.append(issue)
            if len(selected_issues) == 3:
                break
        answers = [str(item['label']) for item in selected_issues] + ['no listed issue is active']
        candidates.append({'activation': {'after_step': 1, 'type': 'task_step'}, 'answer_set': answers, 'question': 'Which listed issue is currently active?', 'question_type': 'active_issue_choice', 'resolver': {'answer_by_active_issue_id': {str(item['id']): str(item['label']) for item in selected_issues}, 'issue_ids': [str(item['id']) for item in selected_issues], 'none_answer': answers[-1], 'type': 'active_issue_choice'}})
    if len(subtasks) >= 2:
        selected_subtasks = subtasks[:3]
        answers = ['subtask {}: {}'.format(index + 1, item['label']) for index, item in enumerate(selected_subtasks)] + ['all subtasks are complete']
        candidates.append({'activation': {'subtask_id': str(selected_subtasks[0]['id']), 'type': 'subtask_handoff'}, 'answer_set': answers, 'question': 'Which subtask is active after this handoff?', 'question_type': 'composite_progress_choice', 'resolver': {'all_complete_answer': answers[-1], 'answer_by_active_subtask_id': {str(item['id']): answers[index] for index, item in enumerate(selected_subtasks)}, 'subtask_ids': [str(item['id']) for item in selected_subtasks], 'type': 'composite_progress_choice'}})
    for insert in inserts:
        distractors = [str(item['label']) for item in clauses[:2]]
        answers = list(dict.fromkeys([str(insert['label']), *distractors]))[:4]
        if len(answers) < 2:
            continue
        candidates.append({'activation': {'insert_id': str(insert['id']), 'type': 'after_human_insert'}, 'answer_set': answers, 'question': 'Which additional obligation was just requested?', 'question_type': 'insert_obligation_choice', 'resolver': {'answer': str(insert['label']), 'insert_id': str(insert['id']), 'type': 'insert_obligation_choice'}})
    return candidates

def rewrite_qa_overlay_roles(overlay: Mapping[str, Any], role_mapping: Mapping[str, str], *, subtask_id: str | None=None) -> dict[str, Any]:
    result = copy.deepcopy(dict(overlay))
    for question in result.get('questions') or []:
        if not isinstance(question, dict):
            continue
        question['activation'] = _rewrite_role_fields(question.get('activation'), role_mapping)
        question['resolver'] = _rewrite_role_fields(question.get('resolver'), role_mapping)
        if subtask_id:
            question['id'] = '{}__{}'.format(subtask_id, question.get('id') or 'qa')
            question['owner_subtask_id'] = subtask_id
            activation = question.get('activation')
            if isinstance(activation, dict) and activation.get('type') == 'subtask_handoff':
                activation['subtask_id'] = subtask_id
    return result

def merge_qa_overlays(overlays: Sequence[Mapping[str, Any]], *, parent_overlay: Mapping[str, Any] | None=None, max_questions: int=3) -> dict[str, Any] | None:
    questions: list[dict[str, Any]] = []
    for overlay in [*overlays, *([parent_overlay] if parent_overlay is not None else [])]:
        if not isinstance(overlay, Mapping):
            continue
        questions.extend(copy.deepcopy(list(overlay.get('questions') or [])))
    if not questions:
        return None
    questions.sort(key=lambda item: (str(item.get('owner_subtask_id') or '~'), str(item.get('id') or '')))
    return validate_qa_overlay({'questions': questions[:max(1, min(int(max_questions), 3))], 'schema_version': QA_OVERLAY_SCHEMA_VERSION})

def _rewrite_role_fields(value: Any, role_mapping: Mapping[str, str], key: str='') -> Any:
    if isinstance(value, Mapping):
        return {str(item_key): _rewrite_role_fields(item, role_mapping, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        if key.endswith('_roles') or key.endswith('_role_ids'):
            return [role_mapping.get(item, item) if isinstance(item, str) else item for item in value]
        return [_rewrite_role_fields(item, role_mapping) for item in value]
    if isinstance(value, str) and (key.endswith('_role') or key.endswith('_role_or_null')):
        return role_mapping.get(value, value)
    return value

def resolve_qa_overlay_answer(question: Mapping[str, Any], state: Mapping[str, Any]) -> str | None:
    resolver = question.get('resolver') if isinstance(question.get('resolver'), Mapping) else {}
    question_type = question.get('question_type')
    answer: Any = None
    if question_type == 'post_action_outcome_choice':
        answer = (resolver.get('answer_by_outcome') or {}).get(state.get('last_tool_outcome'))
    elif question_type in {'remaining_requirement_choice', 'target_state_choice', 'target_relation_choice', 'mid_execution_progress_choice'}:
        statuses = state.get('clause_incomplete')
        if isinstance(statuses, Mapping):
            clause_ids = list(resolver.get('clause_ids') or [])
            if all((statuses.get(item) in (True, False) for item in clause_ids)):
                mask = ''.join(('1' if statuses[item] else '0' for item in clause_ids))
                answer = (resolver.get('answer_by_incomplete_mask') or {}).get(mask)
    elif question_type == 'active_issue_choice':
        active = [str(item) for item in state.get('active_issue_ids') or []]
        configured = set(resolver.get('issue_ids') or [])
        active = sorted(set(active).intersection(configured))
        if not active:
            answer = resolver.get('none_answer')
        elif len(active) == 1:
            answer = (resolver.get('answer_by_active_issue_id') or {}).get(active[0])
    elif question_type == 'composite_progress_choice':
        active = state.get('active_subtask_id')
        if active is None and state.get('composite_complete') is True:
            answer = resolver.get('all_complete_answer')
        elif isinstance(active, str):
            answer = (resolver.get('answer_by_active_subtask_id') or {}).get(active)
    elif question_type == 'insert_obligation_choice':
        if resolver.get('insert_id') in set(state.get('posted_insert_ids') or []):
            answer = resolver.get('answer')
    return answer if answer in question.get('answer_set', ()) else None

def compute_qa_clause_incomplete_state(goal_specs: Sequence[Mapping[str, Any]], predicate_records: Sequence[Mapping[str, Any]], *, owner_subtask_id: str | None=None) -> dict[str, bool]:
    paired = [(spec, predicate_records[index]) for index, spec in enumerate(goal_specs) if isinstance(spec, Mapping) and index < len(predicate_records) and isinstance(predicate_records[index], Mapping) and (owner_subtask_id is None or spec.get('owner_subtask_id') == owner_subtask_id)]
    result: dict[str, bool] = {}
    for local_index, (spec, record) in enumerate(paired):
        clause_id = spec.get('id') or 'goal_clause_{}'.format(local_index)
        satisfied = record.get('satisfied')
        if not isinstance(clause_id, str) or not clause_id or satisfied not in (True, False):
            continue
        if clause_id in result:
            result.pop(clause_id, None)
            continue
        result[clause_id] = not satisfied
    return result

def compute_qa_overlay_metrics(overlay: Mapping[str, Any] | None, *, posted_ids: Sequence[str], answers: Mapping[str, Any], evaluator_keys: Mapping[str, Any], resolver_failure_ids: Sequence[str]=(), posted_steps: Mapping[str, int] | None=None, answered_steps: Mapping[str, int] | None=None) -> dict[str, Any]:
    questions = list((overlay or {}).get('questions') or [])
    configured = [str(item.get('id')) for item in questions]
    posted = set((str(item) for item in posted_ids)).intersection(configured)
    failures = set((str(item) for item in resolver_failure_ids)).intersection(configured)
    answered: set[str] = set()
    correct: set[str] = set()
    records: list[dict[str, Any]] = []
    latencies: list[int] = []
    for question in questions:
        question_id = str(question.get('id'))
        raw_answer = answers.get(question_id)
        value = raw_answer.get('value') if isinstance(raw_answer, Mapping) else raw_answer
        expected = evaluator_keys.get(question_id)
        if value is not None:
            answered.add(question_id)
        if value is not None and expected is not None and (_normalized_answer(value) == _normalized_answer(expected)):
            correct.add(question_id)
        if posted_steps and answered_steps and (question_id in posted_steps) and (question_id in answered_steps):
            latencies.append(max(0, int(answered_steps[question_id]) - int(posted_steps[question_id])))
        records.append({'answered': question_id in answered, 'correct': question_id in correct, 'expected_answer': expected, 'id': question_id, 'posted': question_id in posted, 'resolver_failed': question_id in failures, 'selected_answer': value})
    total = len(configured)
    qa_success = total > 0 and len(posted) == total and (len(correct) == total) and (not failures)
    return {'num_qa_answered': len(answered), 'num_qa_correct': len(correct), 'num_qa_posted': len(posted), 'num_qa_resolver_failures': len(failures), 'num_qa_total': total, 'per_question': records, 'qa_accuracy': len(correct) / float(total) if total else 1.0, 'qa_response_latency_tool_steps': sum(latencies) / float(len(latencies)) if latencies else -1, 'qa_success': int(qa_success)}

def qa_overlay_sha256(overlay: Mapping[str, Any], *, role_names: Sequence[str]=(), subtask_ids: Sequence[str]=()) -> str:
    return sha256_json(validate_qa_overlay(overlay, role_names=role_names, subtask_ids=subtask_ids))

def _normalized_answer(value: Any) -> str:
    return str(value).strip().casefold()

def _role_label(name: str, role: Any) -> str:
    if isinstance(role, Mapping):
        value = role.get('object_type') or role.get('category') or role.get('synset')
        if isinstance(value, str) and value:
            return value.split('.', 1)[0].replace('_', ' ').lower()
    return name.replace('_', ' ')

def _predicate_label(predicate: Mapping[str, Any], roles: Mapping[str, Mapping[str, Any]]) -> str:
    predicate_type = str(predicate.get('type') or 'task requirement').replace('_', ' ')
    args = predicate.get('args') if isinstance(predicate.get('args'), Mapping) else {}
    object_role = predicate.get('object') or predicate.get('object_role') or args.get('object')
    receptacle_role = predicate.get('receptacle') or predicate.get('receptacle_role') or args.get('receptacle')
    object_label = (roles.get(str(object_role)) or {}).get('label', object_role)
    receptacle_label = (roles.get(str(receptacle_role)) or {}).get('label', receptacle_role)
    return ' '.join((str(item) for item in (object_label, predicate_type, receptacle_label) if item))

def _action_candidate(predicate: Mapping[str, Any], roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    tool_by_type = {'object_closed': 'close', 'object_clean': 'wash', 'object_cool': 'cool', 'object_hot': 'heat', 'object_in_receptacle': 'put_down', 'object_on_receptacle': 'put_down', 'object_open': 'open', 'object_sliced': 'slice', 'object_toggled_off': 'toggle_off', 'object_toggled_on': 'toggle_on'}
    tool_name = tool_by_type.get(predicate.get('type'))
    if tool_name is None:
        return None
    if tool_name == 'put_down':
        args = predicate.get('args') if isinstance(predicate.get('args'), Mapping) else {}
        target_role = predicate.get('receptacle') or predicate.get('receptacle_role') or args.get('receptacle')
    else:
        args = predicate.get('args') if isinstance(predicate.get('args'), Mapping) else {}
        target_role = predicate.get('object') or predicate.get('object_role') or args.get('object')
    if not isinstance(target_role, str):
        return None
    return {'target_label': (roles.get(target_role) or {}).get('label', target_role), 'target_role_or_null': target_role, 'tool_name': tool_name}

def _normalized_issues(family: Mapping[str, Any], roles: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    raw_issues: list[Any] = []
    tidy = family.get('tidy') if isinstance(family.get('tidy'), Mapping) else {}
    raw_issues.extend(tidy.get('violations') or [])
    recover = family.get('recover') if isinstance(family.get('recover'), Mapping) else {}
    if isinstance(recover.get('incident'), Mapping):
        raw_issues.append(recover['incident'])
    raw_issues.extend(recover.get('incidents') or [])
    result: list[dict[str, str]] = []
    for index, raw in enumerate(raw_issues):
        if not isinstance(raw, Mapping):
            continue
        issue_id = str(raw.get('id') or raw.get('violation_id') or 'issue_{}'.format(index))
        role = raw.get('object_role') or raw.get('target_role')
        role_label = (roles.get(str(role)) or {}).get('label') if role is not None else None
        issue_type = str(raw.get('type') or raw.get('violation_type') or 'task issue').replace('_', ' ')
        label = '{} ({})'.format(issue_type, role_label) if role_label else issue_type
        result.append({'id': issue_id, 'label': label})
    return result
__all__ = ['QA_OVERLAY_ACTIVATION_TYPES', 'QA_OVERLAY_QUESTION_TYPES', 'QA_OVERLAY_SCHEMA_VERSION', 'build_qa_overlay_spec', 'build_qa_overlay_candidates', 'compute_qa_overlay_metrics', 'compute_qa_clause_incomplete_state', 'merge_qa_overlays', 'normalize_task_for_qa', 'qa_overlay_sha256', 'resolve_qa_overlay_answer', 'rewrite_qa_overlay_roles', 'validate_qa_overlay']
