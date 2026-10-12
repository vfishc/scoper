from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Iterable, List, Mapping, Optional, Sequence
from scoper.benchmark.common import compute_qa_overlay_metrics
_PHYSICAL_TOOL_NAMES = {'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'slice'}

def compute_episode_metrics(*, transitions: Sequence[Mapping[str, Any]], final_goal: Mapping[str, Any], done_reason: Optional[str], model_stopped: bool, elapsed_seconds: Optional[float]=None, scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None, task_spec: Optional[Mapping[str, Any]]=None) -> dict:
    high_level_steps = len(transitions)
    task_action_steps = max([int(transition.get('task_action_step', 0) or 0) for transition in transitions] or [0])
    invalid_tool_calls = sum((1 for t in transitions if not t.get('success', False) and str(t.get('tool_name') or '') != 'answer' and ((t.get('raw_execution') or {}).get('status') != 'qa_pending')))
    total_low_level_steps = sum((int(t.get('num_low_level_steps', 0)) for t in transitions))
    tool_call_counts: Counter = Counter()
    for t in transitions:
        name = t.get('tool_name')
        if name:
            tool_call_counts[str(name)] += 1
    disturbance_counts: Counter = Counter()
    failure_counts: Counter = Counter()
    num_disturbance_events = 0
    num_failure_injections = 0
    for ev in scheduler_events or []:
        name = str(ev.get('name', ''))
        if name in {'close_object', 'open_object', 'toggle_on', 'toggle_off', 'restore_initial_state', 'dirty_object', 'relocate_object'}:
            num_disturbance_events += 1
            disturbance_counts[name] += 1
        elif name.startswith('failure_') and name != 'failure_armed':
            payload = ev.get('payload', {})
            if isinstance(payload, Mapping) and payload.get('injection_applied') is False:
                continue
            num_failure_injections += 1
            tool = payload.get('tool') or 'any'
            failure_counts['{}:{}'.format(name, tool)] += 1
    metrics = {'task_success': float(final_goal.get('task_success', 0.0)), 'task_progress': float(final_goal.get('task_progress', 0.0)), 'predicate_success': float(final_goal.get('predicate_success', 0.0)), 'completion_candidate': bool(final_goal.get('completion_candidate', False)), 'require_model_stop': bool(final_goal.get('require_model_stop', False)), 'model_stopped': bool(model_stopped), 'done_reason': done_reason, 'high_level_steps': high_level_steps, 'task_action_steps': task_action_steps, 'invalid_tool_calls': invalid_tool_calls, 'total_low_level_steps': total_low_level_steps, 'tool_call_counts': dict(tool_call_counts), 'elapsed_seconds': elapsed_seconds, 'num_disturbance_events': num_disturbance_events, 'num_failure_injections': num_failure_injections, 'disturbance_counts': dict(disturbance_counts), 'failure_counts': dict(failure_counts)}
    if task_spec is not None:
        family_metrics = compute_family_metrics(task_spec, final_goal, transitions, scheduler_events=scheduler_events)
        metrics['family_metrics'] = family_metrics
        metrics.update(family_metrics)
        from scoper.benchmark.common.evaluation_reporting import native_qa_questions
        native_questions = native_qa_questions(task_spec)
        if native_questions:
            metrics['native_qa'] = _qa_overlay_episode_metrics({'questions': native_questions}, transitions, scheduler_events or [], final_goal)
        if isinstance(task_spec.get('qa_overlay'), Mapping):
            metrics.update(_qa_overlay_episode_metrics(task_spec['qa_overlay'], transitions, scheduler_events or [], final_goal))
    return metrics

def _qa_overlay_episode_metrics(overlay: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], scheduler_events: Sequence[Mapping[str, Any]], final_goal: Mapping[str, Any]) -> dict[str, Any]:
    configured = {str(question.get('id')) for question in overlay.get('questions') or []}
    posted_ids: list[str] = []
    evaluator_keys: dict[str, Any] = {}
    posted_steps: dict[str, int] = {}
    resolver_failures: list[str] = []
    for event in scheduler_events:
        payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
        question_id = str(payload.get('id') or '')
        if event.get('name') == 'human_qa_posted' and question_id in configured:
            posted_ids.append(question_id)
            evaluator_keys[question_id] = payload.get('evaluator_answer')
            posted_steps[question_id] = int(payload.get('step', payload.get('after_step', 0)) or 0)
        elif event.get('name') == 'qa_overlay_resolver_failed' and question_id in configured:
            resolver_failures.append(question_id)
    answers: dict[str, Any] = {}
    answered_steps: dict[str, int] = {}
    for index, transition in enumerate(transitions, start=1):
        if transition.get('tool_name') != 'answer' or not transition.get('success'):
            continue
        arguments = transition.get('arguments') if isinstance(transition.get('arguments'), Mapping) else {}
        question_id = str(arguments.get('question_id') or '')
        if question_id in configured:
            answers[question_id] = arguments.get('value')
            answered_steps[question_id] = int(transition.get('high_level_step', index) or index)
    qa_metrics = compute_qa_overlay_metrics(overlay, posted_ids=posted_ids, answers=answers, evaluator_keys=evaluator_keys, resolver_failure_ids=resolver_failures, posted_steps=posted_steps, answered_steps=answered_steps)
    base_success = float(final_goal.get('task_success', 0.0))
    qa_metrics['base_task_success'] = base_success
    qa_metrics['combined_success'] = int(base_success == 1.0 and qa_metrics['qa_success'] == 1)
    return qa_metrics

def compute_family_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]], scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None) -> dict:
    category = task_spec.get('category')
    if category in {'prepare_then_place', 'prepare_multi_stage'} and 'family_spec' in task_spec:
        return compute_prepare_metrics(task_spec, final_goal)
    if category in {'put_away_by_category', 'put_away_visual_judgment', 'put_away_cross_room'} and 'family_spec' in task_spec:
        return compute_put_away_metrics(task_spec, final_goal)
    if category == 'tidy_by_policy' and 'family_spec' in task_spec:
        return compute_tidy_metrics(task_spec, final_goal, tool_calls)
    if category in {'recover_and_store_incident_object', 'recover_multi_incident'} and 'family_spec' in task_spec:
        return compute_recover_metrics(task_spec, final_goal, tool_calls, _low_level_records_from_transitions(tool_calls))
    if category == 'human_interactive_execution' and 'family_spec' in task_spec:
        return compute_human_metrics(task_spec, final_goal, tool_calls, scheduler_events=scheduler_events)
    if category == 'mid_execution_state_disturbance' and 'family_spec' in task_spec:
        return compute_mid_execution_disturbance_metrics(task_spec, final_goal, tool_calls, scheduler_events=scheduler_events)
    if category == 'explore_then_act' and 'family_spec' in task_spec:
        return compute_explore_metrics(task_spec, final_goal, tool_calls)
    if category in {'observation_conditioned_room_inspection', 'observation_conditioned_task'} and 'family_spec' in task_spec:
        return compute_observation_conditioned_metrics(task_spec, final_goal, tool_calls)
    if category == 'composite' and 'family_spec' in task_spec:
        return compute_composite_metrics(task_spec, final_goal, tool_calls)
    return {}

def compute_mid_execution_disturbance_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]], scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None) -> dict:
    mid = task_spec['family_spec']['mid_execution']
    primary = mid['primary_task']
    disturbance = mid['disturbance']
    disturbance_type = disturbance['type']
    target_role = disturbance['target_role']
    primary_success = int(float(final_goal.get('predicate_success', 0.0)) >= 1.0)
    event_step = _first_scheduler_event_step(scheduler_events, disturbance_type, target_role)
    recovery_index = _first_mid_execution_recovery_index(primary, disturbance, tool_calls, event_step)
    event_occurred = event_step is not None
    recovery_success = int(event_occurred and primary_success and (recovery_index is not None))
    return {'primary_task_success': primary_success, 'disturbance_type': disturbance_type, 'disturbance_event_occurred': int(event_occurred), 'disturbance_detected': int(event_occurred and recovery_index is not None), 'recovery_success': recovery_success, 'recovery_latency_tool_steps': max(0, recovery_index + 1 - event_step) if event_step is not None and recovery_index is not None else -1, 'combined_success': int(bool(primary_success) and bool(recovery_success))}

def compute_human_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]], scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None, env_qa_state: Optional[Mapping[str, Any]]=None) -> dict:
    human = task_spec['family_spec']['human']
    events = list(human.get('human_events') or [])
    qa_events = [event for event in events if event.get('type') == 'qa']
    insert_events = [event for event in events if event.get('type') == 'insert']
    predicate_records = _predicate_records(task_spec, final_goal)
    runtime_keys = _human_runtime_answer_keys(scheduler_events, env_qa_state)
    answered = 0
    correct = 0
    latencies: List[int] = []
    posted_steps = _human_posted_steps(scheduler_events, env_qa_state)
    for event in qa_events:
        question_id = event.get('id')
        answer_call = _first_answer_tool_call(tool_calls, str(question_id))
        if answer_call is None:
            continue
        answered += 1
        expected = runtime_keys.get(str(question_id)) or event.get('evaluator_answer')
        if _normalize_answer(_answer_value(answer_call)) == _normalize_answer(expected):
            correct += 1
        posted_step = posted_steps.get(str(question_id))
        answer_step = _transition_high_level_step(answer_call, tool_calls)
        if posted_step is not None and answer_step is not None:
            latencies.append(max(0, int(answer_step) - int(posted_step)))
    qa_total = len(qa_events)
    qa_accuracy = correct / float(qa_total) if qa_total else 1.0
    qa_posted = len({str(event.get('payload', {}).get('id')) for event in scheduler_events or [] if event.get('name') == 'human_qa_posted' and str(event.get('payload', {}).get('id')) in {str(q.get('id')) for q in qa_events}})
    qa_trigger_rate = qa_posted / float(qa_total) if qa_total else 1.0
    completed_inserts = 0
    for event in insert_events:
        if _human_insert_completed(event, predicate_records):
            completed_inserts += 1
    insert_success = int(completed_inserts == len(insert_events)) if insert_events else 1
    primary_success = _human_primary_success(task_spec, final_goal)
    return {'primary_task_success': int(primary_success), 'num_qa_events': qa_total, 'num_qa_posted': qa_posted, 'num_qa_unposted': max(0, qa_total - qa_posted), 'num_qa_answered': answered, 'num_qa_correct': correct, 'qa_trigger_rate': float(qa_trigger_rate), 'qa_accuracy': float(qa_accuracy), 'qa_success_rate': float(qa_accuracy), 'qa_response_latency_tool_steps': sum(latencies) / float(len(latencies)) if latencies else -1, 'num_inserts': len(insert_events), 'num_inserts_completed': completed_inserts, 'insert_task_success': insert_success, 'interaction_success': int(primary_success == 1 and qa_accuracy == 1.0 and (insert_success == 1))}

def compute_prepare_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any]) -> dict:
    prepare = task_spec['family_spec']['prepare']
    predicate_records = _predicate_records(task_spec, final_goal)
    target_role = str(prepare.get('target_role') or 'target_object')
    stages = list(prepare.get('stages') or [])
    if not stages:
        return {'prepare_success': int(float(final_goal.get('predicate_success', 0.0)) >= 1.0)}
    stage_successes: List[int] = []
    metrics: dict = {'num_stages': len(stages)}
    for index, stage in enumerate(stages):
        stage_type = str(stage.get('type') or '')
        success = _prepare_stage_success(stage_type, target_role, stage, prepare, predicate_records)
        stage_successes.append(success)
        metrics['stage_{}_{}_success'.format(index, stage_type or 'unknown')] = success
        metrics['stage_{}_success'.format(index)] = success
    metrics['all_stages_success'] = int(all(stage_successes))
    metrics['overall_success'] = int(float(final_goal.get('predicate_success', 0.0)) >= 1.0)
    return metrics

def compute_put_away_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any]) -> dict:
    put_away = task_spec['family_spec']['put_away']
    target_roles = list(put_away.get('target_roles') or [])
    distractor_roles = list(put_away.get('protected_distractor_roles') or [])
    predicate_records = _predicate_records(task_spec, final_goal)
    completed_targets = {pred.get('object_role') for pred in predicate_records if pred.get('type') in {'object_in_receptacle', 'object_in_any_receptacle'} and pred.get('object_role') in target_roles and bool(pred.get('satisfied', False))}
    distractor_violations = {pred.get('object_role') for pred in predicate_records if pred.get('type') in {'object_not_in_receptacle', 'object_not_in_any_receptacle'} and pred.get('object_role') in distractor_roles and (not bool(pred.get('satisfied', False)))}
    num_targets = len(target_roles)
    return {'num_targets': num_targets, 'num_targets_completed': len(completed_targets), 'target_completion_rate': len(completed_targets) / float(num_targets) if num_targets else 0.0, 'num_distractors': len(distractor_roles), 'num_distractor_violations': len(distractor_violations), 'no_over_action_success': 1 if not distractor_violations else 0}

def compute_tidy_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]]) -> dict:
    violations = list(task_spec['family_spec']['tidy'].get('violations') or [])
    predicate_records = _predicate_records(task_spec, final_goal)
    fixed_count = 0
    for violation in violations:
        if _tidy_violation_fixed(violation, predicate_records):
            fixed_count += 1
    num_violations = len(violations)
    return {'num_violations': num_violations, 'num_violations_fixed': fixed_count, 'violation_success_rate': fixed_count / float(num_violations) if num_violations else 0.0, 'faucet_urgency_success': _compute_faucet_urgency_success(violations, tool_calls)}

def compute_recover_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]], low_level_records: Optional[Sequence[Mapping[str, Any]]]=None) -> dict:
    recover = task_spec['family_spec']['recover']
    primary = recover.get('primary_task')
    incidents = list(recover.get('incidents') or [])
    if not incidents and 'incident' in recover:
        incidents = [recover['incident']]
    primary_success = _compute_recover_primary_success(task_spec, final_goal, primary) if isinstance(primary, Mapping) else 1
    incident_successes = [_compute_recover_incident_success(task_spec, final_goal, incident, tool_calls) for incident in incidents if isinstance(incident, Mapping)]
    incident_success = int(all(incident_successes)) if incident_successes else 1
    metrics = {'primary_task_success': int(primary_success), 'incident_obligation_success': int(incident_success), 'combined_success': int(bool(primary_success) and bool(incident_success)), 'incident_count': len(incident_successes), 'incident_type': incidents[0].get('type') if len(incidents) == 1 else [i.get('type') for i in incidents], 'incident_urgency': incidents[0].get('urgency') if len(incidents) == 1 else [i.get('urgency') for i in incidents], 'incident_response_latency_tool_steps': _compute_incident_response_latency(task_spec, tool_calls, incidents[0] if incidents else {}), 'incident_detection_latency_stream_steps': _compute_incident_detection_latency(task_spec, tool_calls, incidents[0] if incidents else {}, low_level_records)}
    for index, success in enumerate(incident_successes):
        metrics['incident_{}_success'.format(index)] = int(success)
    metrics['faucet_urgency_success'] = _recover_faucet_urgency_success(incidents, tool_calls)
    return metrics

def compute_explore_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]]) -> dict:
    explore = task_spec['family_spec']['explore']
    discovery = explore.get('discovery_spec') or {}
    post_task = explore.get('post_discovery_task') or {}
    scan_tools = {'scan', 'look_for'}
    total_scans = sum((1 for call in tool_calls if _tool_name(call) in scan_tools))
    discovery_step = _first_tool_index(tool_calls, scan_tools)
    post_success = int(float(final_goal.get('predicate_success', 0.0)) >= 1.0)
    return {'target_role': discovery.get('target_role') or 'target_object', 'post_discovery_task_type': post_task.get('task_type'), 'exploration_steps': int(discovery_step + 1) if discovery_step is not None else 0, 'discovery_step': discovery_step if discovery_step is not None else -1, 'total_scans': total_scans, 'post_discovery_success': post_success, 'overall_success': int(float(final_goal.get('task_success', 0.0)) >= 1.0)}

def compute_observation_conditioned_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]]) -> dict:
    obs = task_spec['family_spec']['observation_conditioned']
    payloads = [p for p in obs.get('latent_payloads') or [] if isinstance(p, Mapping)]
    predicate_records = _predicate_records(task_spec, final_goal)
    scan_indices = [index for index, call in enumerate(tool_calls) if _tool_name(call) == 'scan']
    first_scan = scan_indices[0] if scan_indices else None
    repaired = 0
    per_issue_success = []
    repair_latencies: List[int] = []
    for payload in payloads:
        predicates = list(payload.get('goal_predicates') or [])
        success = int(bool(predicates) and all((_matching_predicate_satisfied(predicate_records, pred) for pred in predicates)))
        per_issue_success.append(success)
        if success:
            repaired += 1
        repair_index = _first_observation_conditioned_repair_index(payload, tool_calls)
        if first_scan is not None and repair_index is not None:
            repair_latencies.append(max(0, int(repair_index) - int(first_scan)))
    total = len(payloads)
    return {'latent_issue_count': total, 'visible_issue_count': total if scan_indices else 0, 'scan_count': len(scan_indices), 'first_scan_step': first_scan + 1 if first_scan is not None else -1, 'repaired_issue_count': repaired, 'issue_repair_success_rate': repaired / float(total) if total else 1.0, 'observation_conditioned_success': int(bool(scan_indices) and repaired == total and (float(final_goal.get('predicate_success', 0.0)) >= 1.0)), 'mean_repair_latency_tool_steps': sum(repair_latencies) / float(len(repair_latencies)) if repair_latencies else -1, 'per_issue_success': per_issue_success}

def compute_composite_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]]) -> dict:
    composite = task_spec['family_spec']['composite']
    subtasks = list(composite.get('subtasks') or [])
    predicate_records = _predicate_records(task_spec, final_goal)
    by_owner = _group_composite_predicates(predicate_records, subtasks)
    per_subtask_success = {owner: bool(records) and all((bool(rec.get('satisfied', False)) for rec in records)) for owner, records in by_owner.items()}
    total_steps = sum((1 for call in tool_calls if _tool_name(call) != 'stop'))
    completion_steps = _composite_completion_steps(task_spec, tool_calls, subtasks)
    metrics = {'overall_success': int(float(final_goal.get('predicate_success', 0.0)) >= 1.0), 'all_subtasks_success': int(all((per_subtask_success.get(str(subtask.get('subtask_id')), False) for subtask in subtasks))), 'total_steps': total_steps, 'steps_per_subtask': [], 'predicates_satisfied': sum((1 for record in predicate_records if record.get('satisfied'))), 'predicates_total': len(predicate_records)}
    previous_completion = 0
    for index, subtask in enumerate(subtasks):
        subtask_id = str(subtask.get('subtask_id') or 'subtask_{}'.format(index))
        completion = completion_steps.get(subtask_id)
        if completion is None:
            step_count = max(0, total_steps - previous_completion)
        else:
            step_count = max(0, int(completion) - previous_completion)
            previous_completion = int(completion)
        metrics['steps_per_subtask'].append(step_count)
        metrics['subtask_{}_id'.format(index)] = subtask_id
        metrics['subtask_{}_success'.format(index)] = int(per_subtask_success.get(subtask_id, False))
        metrics['subtask_{}_steps'.format(index)] = step_count
    return metrics

def _tidy_violation_fixed(violation: Mapping[str, Any], predicate_records: Sequence[Mapping[str, Any]]) -> bool:
    object_role = violation['object_role']
    v_type = violation['type']
    if v_type == 'faucet_on':
        return _predicate_satisfied(predicate_records, 'object_toggled_off', object_role)
    if v_type == 'open_container':
        return _predicate_satisfied(predicate_records, 'object_closed', object_role)
    if v_type == 'dirty_dish':
        return _predicate_satisfied(predicate_records, 'object_clean', object_role) and (_predicate_satisfied(predicate_records, 'object_in_receptacle', object_role) or _predicate_satisfied(predicate_records, 'object_in_any_receptacle', object_role))
    return False

def _prepare_stage_success(stage_type: str, target_role: str, stage: Mapping[str, Any], prepare: Mapping[str, Any], predicate_records: Sequence[Mapping[str, Any]]) -> int:
    if stage_type == 'wash':
        return int(_predicate_satisfied(predicate_records, 'object_clean', target_role))
    if stage_type == 'heat':
        return int(_predicate_satisfied(predicate_records, 'object_hot', target_role))
    if stage_type == 'cool':
        return int(_predicate_satisfied(predicate_records, 'object_cool', target_role))
    if stage_type == 'slice':
        return int(_predicate_satisfied(predicate_records, 'object_sliced', target_role))
    if stage_type in {'store', 'place'}:
        receptacle_role = stage.get('receptacle_role') or prepare.get('final_receptacle_role') or 'target_receptacle'
        return int(_predicate_satisfied_in_receptacle(predicate_records, target_role, str(receptacle_role)))
    return 0

def _predicate_satisfied_in_receptacle(predicate_records: Sequence[Mapping[str, Any]], object_role: str, receptacle_role: str) -> bool:
    for pred in predicate_records:
        if pred.get('object_role') != object_role:
            continue
        if pred.get('type') == 'object_in_receptacle':
            if pred.get('receptacle_role') == receptacle_role and bool(pred.get('satisfied', False)):
                return True
        if pred.get('type') == 'object_in_any_receptacle':
            receptacles = set(pred.get('receptacle_roles') or [])
            if receptacle_role in receptacles and bool(pred.get('satisfied', False)):
                return True
    return False

def _compute_recover_primary_success(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], primary: Mapping[str, Any]) -> int:
    object_role = primary['object_role']
    receptacle_role = primary['receptacle_role']
    predicate_records = _predicate_records(task_spec, final_goal)
    relevant = [pred for pred in predicate_records if pred.get('object_role') == object_role or pred.get('object_role') == receptacle_role or pred.get('receptacle_role') == receptacle_role]
    if not relevant:
        return 0
    return int(all((bool(pred.get('satisfied', False)) for pred in relevant)))

def _compute_recover_incident_success(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], incident: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]]) -> int:
    predicate_records = _predicate_records(task_spec, final_goal)
    incident_roles = _incident_roles(incident)
    relevant = [pred for pred in predicate_records if pred.get('object_role') in incident_roles or pred.get('receptacle_role') in incident_roles]
    final_ok = bool(relevant) and all((bool(pred.get('satisfied', False)) for pred in relevant))
    if incident.get('urgency') == 'immediate':
        final_ok = final_ok and _check_immediate_urgency(tool_calls, incident)
    return int(final_ok)

def _incident_roles(incident: Mapping[str, Any]) -> set:
    roles = {_incident_object_role(incident)}
    support_roles = incident.get('support_roles') or {}
    if isinstance(support_roles, Mapping):
        roles.update((str(role) for role in support_roles.values() if role))
    for key in ('sink_role', 'faucet_role', 'storage_role', 'sink', 'faucet'):
        value = incident.get(key)
        if isinstance(value, str) and value:
            roles.add(value)
    return roles

def _check_immediate_urgency(tool_calls: Sequence[Mapping[str, Any]], incident: Mapping[str, Any]) -> bool:
    incident_role = _incident_object_role(incident)
    saw_navigation = False
    safe_put_down_used = False
    for call in tool_calls:
        tool_name = _tool_name(call)
        if tool_name == 'navigate_to' and (not saw_navigation):
            saw_navigation = True
            continue
        if tool_name not in _PHYSICAL_TOOL_NAMES:
            continue
        if not saw_navigation:
            return False
        target = _tool_target(call)
        if tool_name == 'put_down' and (not safe_put_down_used) and (target != incident_role):
            safe_put_down_used = True
            continue
        return target == incident_role
    return False

def _compute_incident_response_latency(task_spec: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]], incident: Mapping[str, Any]) -> int:
    first_visible = _first_visible_tool_step(task_spec)
    first_corrective = _first_corrective_tool_index(tool_calls, incident)
    if first_corrective is None:
        return -1
    return max(0, first_corrective - first_visible)

def _compute_incident_detection_latency(task_spec: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]], incident: Mapping[str, Any], low_level_records: Optional[Sequence[Mapping[str, Any]]]) -> int:
    if not low_level_records:
        return -1
    incident_role = _incident_object_role(incident)
    incident_object_id = (task_spec.get('roles', {}).get(incident_role) or {}).get('object_id')
    if not incident_object_id:
        return -1
    first_visible_stream = None
    first_visible_high_step = None
    for index, record in enumerate(low_level_records):
        visible_ids = record.get('metadata_visible_object_ids') or []
        if incident_object_id in visible_ids:
            first_visible_stream = index
            first_visible_high_step = int(record.get('high_level_step', 0) or 0)
            break
    if first_visible_stream is None:
        return -1
    first_corrective = _first_corrective_tool_index(tool_calls, incident)
    if first_corrective is None:
        return -1
    corrective_high_step = first_corrective + 1
    visible_to_corrective = [record for record in low_level_records[first_visible_stream:] if int(record.get('high_level_step', 0) or 0) <= corrective_high_step]
    if first_visible_high_step and corrective_high_step < first_visible_high_step:
        return -1
    return max(0, len(visible_to_corrective) - 1)

def _first_visible_tool_step(task_spec: Mapping[str, Any]) -> int:
    metadata = task_spec.get('metadata') or {}
    visual = metadata.get('visual_discoverability') or {}
    value = visual.get('first_visible_tool_step')
    if isinstance(value, int):
        return max(0, value - 1)
    return 0

def _first_corrective_tool_index(tool_calls: Sequence[Mapping[str, Any]], incident: Mapping[str, Any]) -> Optional[int]:
    incident_role = _incident_object_role(incident)
    for index, call in enumerate(tool_calls):
        if _tool_target(call) == incident_role and _tool_name(call) in _PHYSICAL_TOOL_NAMES | {'navigate_to'}:
            return index
    return None

def _incident_object_role(incident: Mapping[str, Any]) -> str:
    role = incident.get('object_role') or incident.get('target_role') or incident.get('target')
    return str(role) if isinstance(role, str) and role else ''

def _first_tool_index(tool_calls: Sequence[Mapping[str, Any]], tool_names: set[str]) -> Optional[int]:
    for index, call in enumerate(tool_calls):
        if _tool_name(call) in tool_names:
            return index
    return None

def _group_composite_predicates(predicate_records: Sequence[Mapping[str, Any]], subtasks: Sequence[Mapping[str, Any]]) -> dict:
    by_owner: dict = defaultdict(list)
    fallback_owner = [str(subtask.get('subtask_id') or 'subtask_{}'.format(index)) for index, subtask in enumerate(subtasks)]
    for index, record in enumerate(predicate_records):
        owner = record.get('owner_subtask_id')
        if not isinstance(owner, str) or not owner:
            owner = fallback_owner[min(index, len(fallback_owner) - 1)] if fallback_owner else 'unknown'
        by_owner[owner].append(record)
    return by_owner

def _composite_completion_steps(task_spec: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]], subtasks: Sequence[Mapping[str, Any]]) -> dict:
    owners = [str(subtask.get('subtask_id') or 'subtask_{}'.format(index)) for index, subtask in enumerate(subtasks)]
    completion_steps = {}
    for step_index, transition in enumerate(tool_calls, start=1):
        step_goal = transition.get('parent_goal_result') or transition.get('goal_result')
        if not isinstance(step_goal, Mapping):
            continue
        step_records = _predicate_records(task_spec, step_goal)
        grouped = _group_composite_predicates(step_records, subtasks)
        for owner in owners:
            if owner in completion_steps:
                continue
            records = grouped.get(owner) or []
            if records and all((bool(record.get('satisfied', False)) for record in records)):
                completion_steps[owner] = step_index
    return completion_steps

def _tool_name(call: Mapping[str, Any]) -> str:
    return str(call.get('tool_name') or call.get('name') or '')

def _tool_target(call: Mapping[str, Any]) -> Optional[str]:
    args = call.get('arguments') or call.get('args') or {}
    if isinstance(args, Mapping):
        target = args.get('target')
        if target is not None:
            return str(target)
    return None

def _first_scheduler_event_step(scheduler_events: Optional[Sequence[Mapping[str, Any]]], event_name: str, target_role: Optional[str]) -> Optional[int]:
    for event in scheduler_events or []:
        if event.get('name') != event_name:
            continue
        payload = event.get('payload') or {}
        if target_role and payload.get('target_role') not in {None, target_role}:
            continue
        step = payload.get('step')
        if isinstance(step, int):
            return step
    return None

def _first_mid_execution_recovery_index(primary: Mapping[str, Any], disturbance: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]], event_step: Optional[int]) -> Optional[int]:
    disturbance_type = disturbance['type']
    target_role = disturbance['target_role']
    start_index = max(0, int(event_step or 1) - 1)
    for index, call in enumerate(tool_calls[start_index:], start=start_index):
        tool_name = _tool_name(call)
        target = _tool_target(call)
        if disturbance_type == 'dirty_object':
            faucet_role = primary.get('faucet_role') or 'faucet'
            if tool_name == 'toggle_on' and target == faucet_role:
                return index
        elif disturbance_type == 'relocate_object':
            if tool_name in {'scan', 'look_for', 'navigate_to'} and target in {target_role, None}:
                return index
        elif disturbance_type == 'close_object':
            if tool_name == 'open' and target == target_role:
                return index
    return None

def _first_observation_conditioned_repair_index(payload: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]]) -> Optional[int]:
    template = payload.get('repair_plan_template')
    object_role = payload.get('object_role')
    support = payload.get('support_roles') if isinstance(payload.get('support_roles'), Mapping) else {}
    for index, call in enumerate(tool_calls):
        tool_name = _tool_name(call)
        target = _tool_target(call)
        if template == 'toggle_off' and tool_name == 'toggle_off' and (target == object_role):
            return index
        if template == 'close' and tool_name == 'close' and (target == object_role):
            return index
        if template == 'wash_then_place':
            faucet_role = support.get('faucet') or 'faucet'
            if tool_name == 'toggle_on' and target == faucet_role:
                return index
    return None

def _first_answer_tool_call(tool_calls: Sequence[Mapping[str, Any]], question_id: str) -> Optional[Mapping[str, Any]]:
    for call in tool_calls:
        if _tool_name(call) != 'answer':
            continue
        args = call.get('arguments') or call.get('args') or {}
        if isinstance(args, Mapping) and str(args.get('question_id')) == question_id:
            return call
    return None

def _answer_value(call: Mapping[str, Any]) -> Any:
    args = call.get('arguments') or call.get('args') or {}
    if isinstance(args, Mapping):
        return args.get('value')
    return None

def _normalize_answer(value: Any) -> str:
    return str(value or '').strip().lower()

def _transition_high_level_step(call: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]]) -> Optional[int]:
    value = call.get('high_level_step')
    if isinstance(value, int):
        return value
    for index, candidate in enumerate(tool_calls, start=1):
        if candidate is call:
            return index
    return None

def _human_runtime_answer_keys(scheduler_events: Optional[Sequence[Mapping[str, Any]]], env_qa_state: Optional[Mapping[str, Any]]) -> dict:
    keys = {}
    if isinstance(env_qa_state, Mapping):
        evaluator_keys = env_qa_state.get('evaluator_keys') or {}
        if isinstance(evaluator_keys, Mapping):
            keys.update({str(k): v for k, v in evaluator_keys.items()})
    for event in scheduler_events or []:
        if event.get('name') != 'human_qa_posted':
            continue
        payload = event.get('payload') or {}
        question_id = payload.get('id')
        evaluator_answer = payload.get('evaluator_answer')
        if question_id is not None and evaluator_answer is not None:
            keys[str(question_id)] = evaluator_answer
    return keys

def _human_posted_steps(scheduler_events: Optional[Sequence[Mapping[str, Any]]], env_qa_state: Optional[Mapping[str, Any]]) -> dict:
    steps = {}
    if isinstance(env_qa_state, Mapping):
        posted = env_qa_state.get('posted_steps') or {}
        if isinstance(posted, Mapping):
            for key, value in posted.items():
                if isinstance(value, int):
                    steps[str(key)] = value
    for event in scheduler_events or []:
        if event.get('name') not in {'human_qa_posted', 'human_insert_posted'}:
            continue
        payload = event.get('payload') or {}
        event_id = payload.get('id')
        after_step = payload.get('after_step')
        if event_id is not None and isinstance(after_step, int):
            steps[str(event_id)] = after_step
    return steps

def _human_insert_completed(event: Mapping[str, Any], predicate_records: Sequence[Mapping[str, Any]]) -> bool:
    predicates = list(event.get('goal_predicates') or [])
    if not predicates:
        return False
    return all((_matching_predicate_satisfied(predicate_records, predicate) for predicate in predicates))

def _matching_predicate_satisfied(predicate_records: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]) -> bool:
    for record in predicate_records:
        if _predicate_matches(record, expected) and bool(record.get('satisfied', False)):
            return True
    return False

def _predicate_matches(candidate: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key in ('type', 'object_role', 'receptacle_role', 'receptacle_roles'):
        if key in expected and candidate.get(key) != expected.get(key):
            return False
    return True

def _human_primary_success(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any]) -> int:
    primary = task_spec['family_spec']['human']['primary_task']
    primary_type = primary.get('type')
    predicate_records = _predicate_records(task_spec, final_goal)
    if primary_type == 'simple_place':
        object_role = primary.get('object_role')
        receptacle_role = primary.get('receptacle_role')
        placed = any((record.get('type') == 'object_in_receptacle' and record.get('object_role') == object_role and (record.get('receptacle_role') == receptacle_role) and bool(record.get('satisfied', False)) for record in predicate_records))
        relevant_closed = [record for record in predicate_records if record.get('type') == 'object_closed' and record.get('object_role') == receptacle_role]
        closed_ok = all((bool(record.get('satisfied', False)) for record in relevant_closed))
        return int(bool(placed) and closed_ok)
    if primary_type == 'put_away_by_category':
        sub_spec = dict(task_spec)
        sub_spec['family_spec'] = {'put_away': primary['put_away_ref']}
        metrics = compute_put_away_metrics(sub_spec, final_goal)
        return int(float(metrics.get('target_completion_rate', 0.0)) == 1.0 and int(metrics.get('no_over_action_success', 0)) == 1)
    return 0

def _low_level_records_from_transitions(transitions: Sequence[Mapping[str, Any]]) -> List[dict]:
    records: List[dict] = []
    for index, transition in enumerate(transitions, start=1):
        for record in transition.get('low_level_actions') or []:
            if isinstance(record, Mapping):
                out = dict(record)
                out.setdefault('high_level_step', index)
                records.append(out)
    return records

def _predicate_satisfied(predicate_records: Sequence[Mapping[str, Any]], predicate_type: str, object_role: str) -> bool:
    return any((pred.get('type') == predicate_type and pred.get('object_role') == object_role and bool(pred.get('satisfied', False)) for pred in predicate_records))

def _recover_faucet_urgency_success(incidents: Sequence[Mapping[str, Any]], tool_calls: Sequence[Mapping[str, Any]]) -> int:
    faucet_roles = {_incident_object_role(incident) for incident in incidents if incident.get('type') == 'faucet_on'}
    dirty_roles = {_incident_object_role(incident) for incident in incidents if incident.get('type') in {'dirty_dish', 'dirty_dish_on_floor_or_surface'}}
    if not faucet_roles or not dirty_roles:
        return 1
    first_faucet = None
    first_dirty = None
    for index, call in enumerate(tool_calls):
        tool_name = _tool_name(call)
        target = _tool_target(call)
        if first_faucet is None and tool_name == 'toggle_off' and (target in faucet_roles):
            first_faucet = index
        if first_dirty is None and target in dirty_roles and (tool_name in _PHYSICAL_TOOL_NAMES):
            first_dirty = index
        if first_faucet is not None and first_dirty is not None:
            break
    if first_faucet is None or first_dirty is None:
        return 0
    return 1 if first_faucet < first_dirty else 0

def _compute_faucet_urgency_success(violations: Sequence[Mapping[str, Any]], tool_calls: Sequence[Mapping[str, Any]]) -> int:
    faucet_roles = {violation['object_role'] for violation in violations if violation.get('type') == 'faucet_on' or violation.get('urgency') == 'immediate'}
    if not faucet_roles:
        return 1
    physical_tools = {'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'slice'}
    for call in tool_calls:
        tool_name = call.get('tool_name') or call.get('name')
        if tool_name not in physical_tools:
            continue
        target = (call.get('arguments') or {}).get('target')
        return 1 if target in faucet_roles else 0
    return 0

def _predicate_records(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any]) -> List[dict]:
    goal_predicates = list(task_spec.get('goal', {}).get('predicates') or [])
    result_predicates = list(final_goal.get('predicates') or final_goal.get('predicate_results') or [])
    records: List[dict] = []
    for index, predicate in enumerate(goal_predicates):
        record = dict(predicate)
        if index < len(result_predicates):
            result = result_predicates[index]
            if isinstance(result, Mapping):
                record['satisfied'] = bool(result.get('satisfied', False))
        records.append(record)
    return records

def aggregate_metrics(episode_records: Iterable[Mapping[str, Any]]) -> dict:
    records = list(episode_records)
    n = len(records)
    if n == 0:
        return {'num_episodes': 0}

    def _metrics(r: Mapping[str, Any]) -> Mapping[str, Any]:
        return r.get('metrics', r)

    def _mean(field: str, subset: Sequence[Mapping[str, Any]]) -> float:
        if not subset:
            return 0.0
        return sum((float(_metrics(r).get(field, 0.0)) for r in subset)) / len(subset)

    def _sum(field: str, subset: Sequence[Mapping[str, Any]]) -> int:
        return sum((int(_metrics(r).get(field, 0)) for r in subset))

    def _block(subset: Sequence[Mapping[str, Any]]) -> dict:
        injected = [r for r in subset if int(_metrics(r).get('num_disturbance_events', 0)) + int(_metrics(r).get('num_failure_injections', 0)) > 0]
        recovered = [r for r in injected if float(_metrics(r).get('task_success', 0.0)) == 1.0]
        failure_code_counts = Counter((str(r.get('failure_code') or 'unknown') for r in subset if r.get('status') == 'failed' or float(_metrics(r).get('task_success', 0.0)) != 1.0))
        return {'num_episodes': len(subset), 'num_success': sum((1 for r in subset if float(_metrics(r).get('task_success', 0.0)) == 1.0)), 'mean_task_success': _mean('task_success', subset), 'mean_task_progress': _mean('task_progress', subset), 'mean_predicate_success': _mean('predicate_success', subset), 'mean_high_level_steps': _mean('high_level_steps', subset), 'mean_invalid_tool_calls': _mean('invalid_tool_calls', subset), 'total_low_level_steps': _sum('total_low_level_steps', subset), 'mean_num_disturbance_events': _mean('num_disturbance_events', subset), 'mean_num_failure_injections': _mean('num_failure_injections', subset), 'num_episodes_with_injection': len(injected), 'recovery_rate': float(len(recovered)) / float(len(injected)) if injected else 0.0, 'failure_code_counts': dict(sorted(failure_code_counts.items()))}
    by_category: dict = defaultdict(list)
    for r in records:
        category = r.get('category') or _metrics(r).get('category') or 'unknown'
        by_category[str(category)].append(r)
    return {**_block(records), 'by_category': {cat: _block(items) for cat, items in by_category.items()}}
__all__ = ['aggregate_metrics', 'compute_composite_metrics', 'compute_episode_metrics', 'compute_explore_metrics', 'compute_family_metrics', 'compute_human_metrics', 'compute_observation_conditioned_metrics', 'compute_prepare_metrics', 'compute_put_away_metrics', 'compute_recover_metrics', 'compute_tidy_metrics']
