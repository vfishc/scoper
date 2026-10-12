from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Mapping, Optional, Sequence
from scoper.benchmark.common import compute_qa_overlay_metrics
_PHYSICAL_TOOL_NAMES = {'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'cut'}
DISTURBANCE_EVENT_NAMES = frozenset({'close_object', 'open_object', 'toggle_on', 'toggle_off', 'dirty_object', 'relocate_object', 'restore_initial_state'})
FAMILY_ALIAS_KEYS = frozenset({'family', 'status', 'primary_task_success', 'num_qa_events', 'num_qa_posted', 'num_qa_unposted', 'num_qa_answered', 'num_qa_correct', 'qa_trigger_rate', 'qa_accuracy', 'qa_success_rate', 'qa_response_latency_tool_steps', 'num_inserts', 'num_inserts_completed', 'insert_task_success', 'interaction_success', 'disturbance_type', 'disturbance_detected', 'recovery_success', 'recovery_latency_tool_steps', 'combined_success', 'failure_injected', 'incident_response_latency_tool_steps', 'incident_detection_latency_stream_steps', 'incident_obligation_success'})

def compute_episode_metrics(*, transitions: Sequence[Mapping[str, Any]], final_goal: Mapping[str, Any], done_reason: Optional[str]=None, model_stopped: bool=False, elapsed_seconds: Optional[float]=None, scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None, task_spec: Optional[Mapping[str, Any]]=None) -> dict[str, Any]:
    transitions = list(transitions or [])
    scheduler_events = list(scheduler_events or [])
    high_level_steps = len(transitions)
    task_action_steps = max([int(transition.get('task_action_step', 0) or 0) for transition in transitions] or [0])
    invalid_tool_calls = sum((1 for t in transitions if t.get('success') is False and str(t.get('tool_name') or '') != 'answer' and ((t.get('raw_execution') or {}).get('status') != 'qa_pending')))
    total_low_level_steps = sum((int(t.get('num_low_level_steps', t.get('frames', 0)) or 0) for t in transitions))
    total_frames = sum((int(t.get('frames', t.get('num_low_level_steps', 0)) or 0) for t in transitions))
    tool_call_counts: Counter[str] = Counter()
    for t in transitions:
        name = t.get('tool_name')
        if name:
            tool_call_counts[str(name)] += 1
    scheduler_counts = scheduler_event_counts(scheduler_events)
    metrics = {'task_success': float(final_goal.get('task_success', 0.0) or 0.0), 'task_progress': float(final_goal.get('task_progress', 0.0) or 0.0), 'predicate_success': float(final_goal.get('predicate_success', 0.0) or 0.0), 'completion_candidate': bool(final_goal.get('completion_candidate', False)), 'require_model_stop': bool(final_goal.get('require_model_stop', False)), 'model_stopped': bool(model_stopped), 'done_reason': done_reason, 'high_level_steps': high_level_steps, 'task_action_steps': task_action_steps, 'invalid_tool_calls': invalid_tool_calls, 'total_low_level_steps': total_low_level_steps, 'total_frames': total_frames, 'tool_call_counts': dict(tool_call_counts), 'elapsed_seconds': elapsed_seconds, 'num_disturbance_events': scheduler_counts['num_disturbance_events'], 'num_failure_injections': scheduler_counts['num_failure_injections'], 'disturbance_counts': scheduler_counts['disturbance_counts'], 'failure_counts': scheduler_counts['failure_counts']}
    if task_spec is not None:
        family_metrics = compute_family_metrics(task_spec=task_spec, final_goal=final_goal, transitions=transitions, scheduler_events=scheduler_events)
        if family_metrics:
            metrics['family_metrics'] = family_metrics
            collisions = _copy_family_metric_aliases(metrics, family_metrics)
            if collisions:
                metrics['family_metric_alias_collisions'] = collisions
        from scoper.benchmark.common.evaluation_reporting import native_qa_questions
        native_questions = native_qa_questions(task_spec)
        if native_questions:
            metrics['native_qa'] = _qa_overlay_episode_metrics({'questions': native_questions}, transitions, scheduler_events, final_goal)
        if isinstance(task_spec.get('qa_overlay'), Mapping):
            metrics.update(_qa_overlay_episode_metrics(task_spec['qa_overlay'], transitions, scheduler_events, final_goal))
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

def scheduler_event_counts(scheduler_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    disturbance_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    for ev in scheduler_events or []:
        name = str(ev.get('name') or '')
        if name in DISTURBANCE_EVENT_NAMES:
            disturbance_counts[name] += 1
        elif name.startswith('failure_') and name != 'failure_armed':
            payload = ev.get('payload') if isinstance(ev.get('payload'), Mapping) else {}
            if payload.get('injection_applied') is False:
                continue
            tool = payload.get('tool') or 'any'
            failure_counts['{}:{}'.format(name, tool)] += 1
    return {'num_disturbance_events': sum(disturbance_counts.values()), 'num_failure_injections': sum(failure_counts.values()), 'disturbance_counts': dict(sorted(disturbance_counts.items())), 'failure_counts': dict(sorted(failure_counts.items()))}

def compute_family_metrics(*, task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None) -> dict[str, Any]:
    family_spec = task_spec.get('family_spec')
    if not isinstance(family_spec, Mapping):
        return {}
    family = str(family_spec.get('family') or task_spec.get('category') or '')
    if family in {'prepare_then_place', 'prepare_multi_stage'}:
        return compute_prepare_family_metrics(task_spec, final_goal)
    if family in {'put_away_by_category', 'put_away_visual_judgment', 'put_away_cross_room'}:
        return compute_put_away_family_metrics(task_spec, final_goal)
    if family == 'tidy_by_policy':
        return compute_tidy_metrics(task_spec, final_goal, transitions)
    if family in {'recover_and_store_incident_object', 'recover_multi_incident'}:
        return compute_recover_metrics(task_spec, final_goal, transitions, low_level_records=_low_level_records_from_transitions(transitions))
    if family == 'human_interactive_execution':
        return compute_human_interaction_metrics(task_spec, final_goal, transitions, scheduler_events=scheduler_events or [])
    if family == 'mid_execution_state_disturbance':
        return compute_mid_execution_disturbance_metrics(task_spec, final_goal, transitions, scheduler_events=scheduler_events or [])
    if family in {'verify_after_action', 'disturbance_recovery'}:
        return compute_scheduler_family_metrics(final_goal, scheduler_events or [], transitions=transitions)
    if family == 'explore_then_act':
        return compute_explore_metrics(task_spec, final_goal, transitions)
    if family in {'observation_conditioned_room_inspection', 'observation_conditioned_task', 'observation_conditioned_tidy'}:
        return compute_observation_conditioned_metrics(task_spec, final_goal, transitions)
    if family == 'composite':
        return compute_composite_metrics(task_spec, final_goal, transitions, scheduler_events=scheduler_events or [])
    return {'family': family, 'status': 'deferred', 'reason': 'family_metric_not_implemented_stage7'}

def compute_prepare_family_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any]) -> dict[str, Any]:
    family_spec = task_spec.get('family_spec') or {}
    prepare = family_spec.get('prepare') if isinstance(family_spec, Mapping) else {}
    stages = list((prepare or {}).get('stages') or [])
    predicate_records = _predicate_records(final_goal)
    stage_records = []
    for stage in stages:
        stype = str(stage.get('type') or '')
        stage_records.append({'type': stype, 'success': _prepare_stage_success(stype, predicate_records)})
    all_stages_success = bool(stage_records) and all((item['success'] for item in stage_records))
    return {'family': 'prepare_then_place', 'status': 'implemented', 'num_stages': len(stages), 'stage_records': stage_records, 'all_stages_success': bool(all_stages_success), 'overall_success': bool(all_stages_success and float(final_goal.get('task_success', 0.0) or 0.0) >= 1.0)}

def compute_put_away_family_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any]) -> dict[str, Any]:
    predicate_records = _predicate_records(final_goal)
    placement = [p for p in predicate_records if str(p.get('type')) in {'object_in_receptacle', 'object_on_receptacle'}]
    distractor_protection = [p for p in predicate_records if str(p.get('type')).startswith('object_not_')]
    target_completed = sum((1 for p in placement if p.get('satisfied') is True))
    distractor_ok = all((p.get('satisfied') is True for p in distractor_protection)) if distractor_protection else True
    return {'family': str((task_spec.get('family_spec') or {}).get('family') or task_spec.get('category')), 'status': 'structural', 'target_predicates': len(placement), 'target_completed': target_completed, 'target_completion_rate': float(target_completed / len(placement)) if placement else 0.0, 'distractor_protection_predicates': len(distractor_protection), 'distractor_protection_success': bool(distractor_ok), 'overall_success': bool(float(final_goal.get('task_success', 0.0) or 0.0) >= 1.0 and distractor_ok)}

def compute_tidy_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tidy = (task_spec.get('family_spec') or {}).get('tidy') or {}
    violations = [item for item in tidy.get('violations') or [] if isinstance(item, Mapping)]
    predicate_records = _predicate_records(final_goal)
    fixed_count = sum((1 for violation in violations if _tidy_violation_fixed(violation, predicate_records)))
    num_violations = len(violations)
    return {'family': 'tidy_by_policy', 'status': 'implemented', 'num_violations': num_violations, 'num_violations_fixed': fixed_count, 'violation_success_rate': fixed_count / float(num_violations) if num_violations else 0.0, 'faucet_urgency_success': _compute_faucet_urgency_success(violations, transitions)}

def compute_recover_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], low_level_records: Optional[Sequence[Mapping[str, Any]]]=None) -> dict[str, Any]:
    recover = (task_spec.get('family_spec') or {}).get('recover') or {}
    primary = recover.get('primary_task')
    incidents = [item for item in recover.get('incidents') or [] if isinstance(item, Mapping)]
    if not incidents and isinstance(recover.get('incident'), Mapping):
        incidents = [recover['incident']]
    primary_success = _compute_recover_primary_success(task_spec, final_goal, primary) if isinstance(primary, Mapping) else 1
    incident_successes = [_compute_recover_incident_success(task_spec, final_goal, incident, transitions) for incident in incidents]
    incident_success = int(all(incident_successes)) if incident_successes else 1
    metrics: dict[str, Any] = {'family': str((task_spec.get('family_spec') or {}).get('family') or task_spec.get('category') or 'recover'), 'status': 'implemented', 'primary_task_success': int(primary_success), 'incident_obligation_success': int(incident_success), 'combined_success': int(bool(primary_success) and bool(incident_success)), 'incident_count': len(incident_successes), 'incident_type': incidents[0].get('type') if len(incidents) == 1 else [item.get('type') for item in incidents], 'incident_urgency': incidents[0].get('urgency') if len(incidents) == 1 else [item.get('urgency') for item in incidents], 'incident_response_latency_tool_steps': _compute_incident_response_latency(task_spec, transitions, incidents[0] if incidents else {}), 'incident_detection_latency_stream_steps': _compute_incident_detection_latency(task_spec, transitions, incidents[0] if incidents else {}, low_level_records), 'faucet_urgency_success': _recover_faucet_urgency_success(incidents, transitions)}
    for index, success in enumerate(incident_successes):
        metrics['incident_{}_success'.format(index)] = int(success)
    return metrics

def compute_mid_execution_disturbance_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None) -> dict[str, Any]:
    scheduler_events = list(scheduler_events or [])
    counts = scheduler_event_counts(scheduler_events)
    family_spec = task_spec.get('family_spec') if isinstance(task_spec.get('family_spec'), Mapping) else {}
    mid = family_spec.get('mid_execution') if isinstance(family_spec.get('mid_execution'), Mapping) else {}
    primary = mid.get('primary_task') if isinstance(mid.get('primary_task'), Mapping) else {}
    disturbance = mid.get('disturbance') if isinstance(mid.get('disturbance'), Mapping) else {}
    if not disturbance:
        disturbance = _first_disturbance_spec(task_spec) or _first_disturbance_event_spec(scheduler_events)
    disturbance_type = str(disturbance.get('type') or '')
    target_role = disturbance.get('target_role')
    primary_success = int(float(final_goal.get('predicate_success', final_goal.get('task_success', 0.0)) or 0.0) >= 1.0)
    event_step = _first_scheduler_event_step(scheduler_events, disturbance_type, target_role) if disturbance_type else _first_injection_step(scheduler_events)
    recovery_index = _first_mid_execution_recovery_index(primary, disturbance, transitions, event_step) if disturbance_type and target_role else None
    fallback_recovery_step = _first_recovery_step(transitions, event_step)
    if recovery_index is not None and event_step is not None:
        latency = max(0, int(recovery_index) + 1 - int(event_step))
    elif fallback_recovery_step is not None and event_step is not None:
        latency = max(0, int(fallback_recovery_step) - int(event_step))
    else:
        latency = None
    event_detected = counts['num_disturbance_events'] > 0
    recovery_action_detected = recovery_index is not None
    recovery_success = bool(event_step is not None and primary_success and (recovery_action_detected or fallback_recovery_step is not None))
    return {'family': 'mid_execution_state_disturbance', 'status': 'implemented', 'primary_task_success': int(primary_success), 'disturbance_type': disturbance_type or None, 'target_role': target_role, 'disturbance_event_occurred': bool(event_step is not None), 'disturbance_detected': bool(event_step is not None and (event_detected or recovery_action_detected)), 'recovery_action_detected': bool(recovery_action_detected), 'failure_injected': counts['num_failure_injections'] > 0, 'recovery_success': bool(recovery_success), 'recovery_latency_tool_steps': latency, 'combined_success': bool(primary_success and event_step is not None and (event_detected or recovery_action_detected))}

def compute_explore_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    explore = (task_spec.get('family_spec') or {}).get('explore') or {}
    discovery = explore.get('discovery_spec') if isinstance(explore.get('discovery_spec'), Mapping) else {}
    post_task = explore.get('post_discovery_task') if isinstance(explore.get('post_discovery_task'), Mapping) else {}
    scan_tools = {'scan', 'look_for'}
    total_scans = sum((1 for call in transitions if _tool_name(call) in scan_tools))
    discovery_step = _first_tool_index(transitions, scan_tools)
    return {'family': 'explore_then_act', 'status': 'implemented', 'target_role': discovery.get('target_role') or 'target_object', 'post_discovery_task_type': post_task.get('task_type'), 'exploration_steps': int(discovery_step + 1) if discovery_step is not None else 0, 'discovery_step': discovery_step if discovery_step is not None else -1, 'total_scans': total_scans, 'post_discovery_success': int(float(final_goal.get('predicate_success', 0.0) or 0.0) >= 1.0), 'overall_success': int(float(final_goal.get('task_success', 0.0) or 0.0) >= 1.0)}

def compute_observation_conditioned_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    obs = (task_spec.get('family_spec') or {}).get('observation_conditioned') or {}
    payloads = [item for item in obs.get('latent_payloads') or [] if isinstance(item, Mapping)]
    room_ids = [str(room) for room in obs.get('rooms') or [] if room]
    discovered_issue_ids = set()
    discovery_steps: dict[str, int] = {}
    scan_steps_by_room: dict[str, int] = {}
    room_entry_steps: dict[str, int] = {}
    for index, transition in enumerate(transitions):
        tool_name = _tool_name(transition)
        state_delta = transition.get('state_delta') if isinstance(transition.get('state_delta'), Mapping) else {}
        raw = transition.get('raw_execution') if isinstance(transition.get('raw_execution'), Mapping) else {}
        if tool_name == 'navigate_to_room':
            room_delta = state_delta.get('navigate_to_room') if isinstance(state_delta.get('navigate_to_room'), Mapping) else {}
            room_id = room_delta.get('room_id') or raw.get('room_id') or ((transition.get('arguments') or {}).get('room') if isinstance(transition.get('arguments'), Mapping) else None)
            if room_id and room_id not in room_entry_steps:
                room_entry_steps[str(room_id)] = index
        if tool_name == 'scan':
            scan = state_delta.get('scan') if isinstance(state_delta.get('scan'), Mapping) else {}
            room_id = scan.get('current_room') or scan.get('room_instance') or raw.get('current_room')
            if room_id and room_id not in scan_steps_by_room:
                scan_steps_by_room[str(room_id)] = index
            for issue_id in scan.get('discovered_issue_ids') or raw.get('discovered_issue_ids') or []:
                issue_id = str(issue_id)
                discovered_issue_ids.add(issue_id)
                discovery_steps.setdefault(issue_id, index)
    repair_violations = []
    payload_records = []
    for payload in payloads:
        issue_id = str(payload.get('latent_id') or payload.get('issue_id') or '')
        room_id = str(payload.get('room_id') or '')
        roles = _payload_roles(payload)
        first_repair = _first_transition_targeting_roles(transitions, roles, start_tools=_PHYSICAL_TOOL_NAMES | {'soak_under', 'wipe', 'fill', 'heat', 'cool'})
        discovery_step = discovery_steps.get(issue_id)
        repair_after_scan = first_repair is None or (discovery_step is not None and first_repair > discovery_step)
        entered_before_scan = room_id in room_entry_steps and room_id in scan_steps_by_room and (room_entry_steps[room_id] < scan_steps_by_room[room_id])
        payload_record = {'issue_id': issue_id, 'room_id': room_id, 'issue_type': payload.get('issue_type'), 'discovered': issue_id in discovered_issue_ids, 'discovery_step': discovery_step if discovery_step is not None else -1, 'first_repair_step': first_repair if first_repair is not None else -1, 'repair_after_scan': bool(repair_after_scan), 'entered_before_scan': bool(entered_before_scan)}
        payload_records.append(payload_record)
        if not repair_after_scan:
            repair_violations.append(payload_record)
    discovered_count = sum((1 for record in payload_records if record['discovered']))
    entered_rooms = sum((1 for room in room_ids if room in room_entry_steps))
    scanned_rooms = sum((1 for room in room_ids if room in scan_steps_by_room))
    return {'family': str((task_spec.get('family_spec') or {}).get('family') or 'observation_conditioned_room_inspection'), 'status': 'implemented', 'num_rooms': len(room_ids), 'entered_rooms': entered_rooms, 'scanned_rooms': scanned_rooms, 'num_latent_payloads': len(payloads), 'num_discovered': discovered_count, 'discovery_rate': float(discovered_count / len(payloads)) if payloads else 0.0, 'payload_records': payload_records, 'repair_before_discovery_violations': repair_violations, 'discovery_compliance': bool(payloads and discovered_count == len(payloads) and (not repair_violations)), 'room_navigation_success': bool(room_ids and entered_rooms == len(room_ids)), 'overall_success': int(float(final_goal.get('task_success', 0.0) or 0.0) >= 1.0)}

def compute_composite_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None) -> dict[str, Any]:
    composite = (task_spec.get('family_spec') or {}).get('composite') or {}
    subtasks = [item for item in composite.get('subtasks') or [] if isinstance(item, Mapping)]
    predicate_records = _predicate_records(final_goal)
    by_owner = _group_composite_predicates(predicate_records, subtasks)
    per_subtask_success = {owner: bool(records) and all((bool(record.get('satisfied', False)) for record in records)) for owner, records in by_owner.items()}
    total_steps = sum((1 for call in transitions if _tool_name(call) != 'stop'))
    completion_steps = _composite_completion_steps(task_spec, transitions, subtasks)
    scheduler_events = list(scheduler_events or [])
    metrics: dict[str, Any] = {'family': 'composite', 'status': 'implemented', 'overall_success': int(float(final_goal.get('predicate_success', 0.0) or 0.0) >= 1.0), 'all_subtasks_success': int(all((per_subtask_success.get(str(subtask.get('subtask_id') or 'subtask_{}'.format(index)), False) for index, subtask in enumerate(subtasks)))), 'total_steps': total_steps, 'steps_per_subtask': [], 'subtask_records': [], 'predicates_satisfied': sum((1 for record in predicate_records if record.get('satisfied'))), 'predicates_total': len(predicate_records)}
    previous_completion = 0
    complex_successes = []
    scheduler_scoped_successes = []
    for index, subtask in enumerate(subtasks):
        subtask_id = str(subtask.get('subtask_id') or 'subtask_{}'.format(index))
        window_start = previous_completion + 1
        completion = completion_steps.get(subtask_id)
        if completion is None:
            step_count = max(0, total_steps - previous_completion)
            window_end = total_steps
        else:
            step_count = max(0, int(completion) - previous_completion)
            window_end = int(completion)
            previous_completion = int(completion)
        metrics['steps_per_subtask'].append(step_count)
        metrics['subtask_{}_id'.format(index)] = subtask_id
        metrics['subtask_{}_success'.format(index)] = int(per_subtask_success.get(subtask_id, False))
        metrics['subtask_{}_steps'.format(index)] = step_count
        subtask_events = _scheduler_events_for_subtask(scheduler_events, subtask_id, window_start, window_end)
        subtask_counts = scheduler_event_counts(subtask_events)
        failure_injected = subtask_counts['num_failure_injections'] > 0
        disturbance_detected = subtask_counts['num_disturbance_events'] > 0
        injection_step = _first_injection_step(subtask_events)
        recovery_step = _first_composite_subtask_recovery_step(transitions, injection_step, subtasks, subtask_id, end_step=window_end)
        recovery_latency = int(recovery_step) - int(injection_step) if injection_step is not None and recovery_step is not None else None
        subtask_success = bool(per_subtask_success.get(subtask_id, False))
        category = str(subtask.get('category') or subtask.get('family') or '')
        complex_subtask = bool(category in {'verify_after_action', 'disturbance_recovery', 'mid_execution_state_disturbance', 'recover_and_store_incident_object', 'recover_multi_incident', 'observation_conditioned_room_inspection', 'observation_conditioned_task'} or failure_injected or disturbance_detected or subtask.get('failure_spec') or subtask.get('disturbance_spec'))
        scheduler_scoped = bool(failure_injected or disturbance_detected or subtask.get('failure_spec') or subtask.get('disturbance_spec'))
        recovery_success = bool(subtask_success and (not scheduler_scoped or failure_injected or disturbance_detected))
        if complex_subtask:
            complex_successes.append(subtask_success)
        if scheduler_scoped:
            scheduler_scoped_successes.append(recovery_success)
        record = {'subtask_id': subtask_id, 'category': category, 'subtask_success': int(subtask_success), 'step_count': step_count, 'window_start': window_start, 'window_end': window_end, 'failure_injected': bool(failure_injected), 'disturbance_detected': bool(disturbance_detected), 'recovery_success': bool(recovery_success), 'recovery_latency_tool_steps': recovery_latency, 'scheduler_event_count': len(subtask_events)}
        metrics['subtask_records'].append(record)
        metrics['subtask_{}_failure_injected'.format(index)] = int(failure_injected)
        metrics['subtask_{}_disturbance_detected'.format(index)] = int(disturbance_detected)
        metrics['subtask_{}_recovery_success'.format(index)] = int(recovery_success)
    metrics['complex_subtasks_success'] = int(all(complex_successes)) if complex_successes else 1
    metrics['scheduler_scoped_success'] = int(all(scheduler_scoped_successes)) if scheduler_scoped_successes else 1
    return metrics

def compute_human_interaction_metrics(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None) -> dict[str, Any]:
    human = (task_spec.get('family_spec') or {}).get('human') or {}
    events = [event for event in _human_events(human) if isinstance(event, Mapping)]
    qa_events = [event for event in events if event.get('type') == 'qa']
    insert_events = [event for event in events if event.get('type') == 'insert']
    posted_steps = _human_posted_steps(transitions, scheduler_events or [])
    runtime_answers = _human_runtime_answer_keys(scheduler_events or [])
    qa_event_ids = {str(event.get('id')) for event in qa_events if event.get('id') is not None}
    posted_qa_ids = {event_id for event_id, record in posted_steps.items() if record.get('type') == 'qa' and event_id in qa_event_ids}
    answer_calls = _answer_tool_calls_by_question(transitions)
    correct_ids = set()
    answered_ids = set()
    latencies = []
    for event in qa_events:
        event_id = str(event.get('id') or '')
        if not event_id:
            continue
        answer_call = answer_calls.get(event_id)
        if answer_call is None:
            continue
        answered_ids.add(event_id)
        expected = _expected_human_answer(event, runtime_answers)
        if _normalize_answer(_answer_value(answer_call)) == _normalize_answer(expected):
            correct_ids.add(event_id)
        posted = posted_steps.get(event_id)
        answer_step = _transition_step(answer_call, transitions)
        if posted is not None and answer_step is not None:
            latencies.append(max(0, int(answer_step) - int(posted.get('step', 0))))
    qa_total = len(qa_events)
    answered_count = len(answered_ids)
    qa_accuracy = len(correct_ids) / float(qa_total) if qa_total else 1.0
    qa_trigger_rate = len(posted_qa_ids) / float(qa_total) if qa_total else 1.0
    completed_inserts = sum((1 for event in insert_events if _human_insert_completed(final_goal, event)))
    insert_success = int(completed_inserts == len(insert_events)) if insert_events else 1
    primary_task_success = _human_primary_task_success(task_spec, final_goal)
    interaction_success = int(primary_task_success >= 1.0 and len(correct_ids) == qa_total and (insert_success == 1))
    return {'family': 'human_interactive_execution', 'status': 'implemented', 'primary_task_success': float(primary_task_success), 'num_qa_events': qa_total, 'num_qa_posted': len(posted_qa_ids), 'num_qa_unposted': max(0, qa_total - len(posted_qa_ids)), 'num_qa_answered': answered_count, 'num_qa_correct': len(correct_ids), 'qa_trigger_rate': float(qa_trigger_rate), 'qa_accuracy': float(qa_accuracy), 'qa_success_rate': float(qa_accuracy), 'qa_response_latency_tool_steps': sum(latencies) / float(len(latencies)) if latencies else -1, 'num_inserts': len(insert_events), 'num_inserts_completed': completed_inserts, 'insert_task_success': insert_success, 'interaction_success': interaction_success}

def compute_scheduler_family_metrics(final_goal: Mapping[str, Any], scheduler_events: Sequence[Mapping[str, Any]], transitions: Optional[Sequence[Mapping[str, Any]]]=None) -> dict[str, Any]:
    counts = scheduler_event_counts(scheduler_events)
    if not scheduler_events:
        return {'family': 'scheduler_runtime', 'status': 'deferred', 'reason': 'missing_scheduler_events'}
    injection_step = _first_injection_step(scheduler_events)
    recovery_step = _first_recovery_step(transitions or [], injection_step)
    latency = int(recovery_step) - int(injection_step) if injection_step is not None and recovery_step is not None else None
    task_success = bool(float(final_goal.get('task_success', 0.0) or 0.0) >= 1.0)
    disturbance_detected = counts['num_disturbance_events'] > 0
    failure_injected = counts['num_failure_injections'] > 0
    return {'family': 'scheduler_runtime', 'status': 'implemented', 'disturbance_detected': disturbance_detected, 'failure_injected': failure_injected, 'recovery_success': task_success, 'recovery_latency_tool_steps': latency, 'combined_success': bool(task_success and (disturbance_detected or failure_injected))}

def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records or [])
    metrics_list = [_record_metrics(record) for record in records]
    success_values = [_float_metric(m, 'task_success') for m in metrics_list]
    progress_values = [_float_metric(m, 'task_progress') for m in metrics_list]
    predicate_values = [_float_metric(m, 'predicate_success') for m in metrics_list]
    categories: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    failure_codes: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    num_injection = 0
    recovery_successes = 0
    recovery_candidates = 0
    recovery_latencies = []
    qa_accuracies = []
    for record, metrics in zip(records, metrics_list):
        category = str(record.get('category') or (record.get('task_spec') or {}).get('category') or '-')
        categories[category].append(record)
        code = record.get('stage7_failure_code') or record.get('failure_code')
        if code:
            failure_codes[str(code)] += 1
        for tool, count in (metrics.get('tool_call_counts') or {}).items():
            tool_counts[str(tool)] += int(count or 0)
        injections = int(metrics.get('num_disturbance_events', 0) or 0) + int(metrics.get('num_failure_injections', 0) or 0)
        num_injection += 1 if injections > 0 else 0
        if injections > 0:
            recovery_candidates += 1
            if _float_metric(metrics, 'task_success') >= 1.0:
                recovery_successes += 1
        family_metrics = _family_metrics(metrics)
        latency = family_metrics.get('recovery_latency_tool_steps')
        if isinstance(latency, (int, float)) and latency >= 0:
            recovery_latencies.append(float(latency))
        if 'qa_accuracy' in family_metrics:
            qa_accuracies.append(_float_metric(family_metrics, 'qa_accuracy'))
    return {'num_episodes': len(records), 'num_success': sum((1 for value in success_values if value >= 1.0)), 'mean_task_success': _mean(success_values), 'mean_task_progress': _mean(progress_values), 'mean_predicate_success': _mean(predicate_values), 'mean_high_level_steps': _mean([_float_metric(m, 'high_level_steps') for m in metrics_list]), 'mean_invalid_tool_calls': _mean([_float_metric(m, 'invalid_tool_calls') for m in metrics_list]), 'total_low_level_steps': sum((int(m.get('total_low_level_steps', m.get('total_frames', 0)) or 0) for m in metrics_list)), 'total_frames': sum((int(m.get('total_frames', m.get('total_low_level_steps', 0)) or 0) for m in metrics_list)), 'mean_num_disturbance_events': _mean([_float_metric(m, 'num_disturbance_events') for m in metrics_list]), 'mean_num_failure_injections': _mean([_float_metric(m, 'num_failure_injections') for m in metrics_list]), 'num_episodes_with_injection': num_injection, 'recovery_rate': float(recovery_successes / recovery_candidates) if recovery_candidates else None, 'mean_recovery_latency': _mean(recovery_latencies) if recovery_latencies else None, 'mean_qa_accuracy': _mean(qa_accuracies) if qa_accuracies else None, 'failure_code_counts': dict(sorted(failure_codes.items())), 'tool_call_counts': dict(sorted(tool_counts.items())), 'by_category': {category: _aggregate_category(items) for category, items in sorted(categories.items())}}

def build_metrics_consistency_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for record in records:
        metrics = _record_metrics(record)
        tool_calls = list(record.get('tool_calls') or record.get('transitions') or [])
        final_goal = record.get('final_goal') or {}
        mismatches = []
        if record.get('execution_attempted') and int(metrics.get('high_level_steps', 0) or 0) != len(tool_calls):
            mismatches.append('high_level_steps_vs_tool_calls')
        failed = sum((1 for call in tool_calls if call.get('success') is False))
        if int(metrics.get('invalid_tool_calls', 0) or 0) != failed:
            mismatches.append('invalid_tool_calls_vs_failed_calls')
        for key in ('task_success', 'task_progress', 'predicate_success'):
            if key in final_goal and abs(_float_metric(metrics, key) - float(final_goal.get(key) or 0.0)) > 1e-06:
                mismatches.append('{}_vs_final_goal'.format(key))
        low = int(metrics.get('total_low_level_steps', metrics.get('total_frames', 0)) or 0)
        frames = int(metrics.get('total_frames', low) or 0)
        if low != frames:
            mismatches.append('total_low_level_steps_vs_total_frames')
        checks.append({'task_id': record.get('task_id') or record.get('id'), 'template_id': record.get('template_id'), 'category': record.get('category'), 'status': 'passed' if not mismatches else 'failed', 'mismatches': mismatches, 'high_level_steps': metrics.get('high_level_steps'), 'tool_calls': len(tool_calls), 'invalid_tool_calls': metrics.get('invalid_tool_calls'), 'failed_tool_calls': failed})
    return checks

def _record_metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get('metrics') if isinstance(record.get('metrics'), Mapping) else {}

def _family_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    family = dict(metrics.get('family_metrics') if isinstance(metrics.get('family_metrics'), Mapping) else {})
    for key in FAMILY_ALIAS_KEYS:
        if key in metrics and key not in family:
            family[key] = metrics[key]
    return family

def _copy_family_metric_aliases(metrics: dict[str, Any], family_metrics: Mapping[str, Any]) -> list[str]:
    collisions = []
    for key, value in family_metrics.items():
        if key in metrics:
            collisions.append(str(key))
            continue
        metrics[str(key)] = value
    return sorted(collisions)

def _aggregate_category(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [_record_metrics(record) for record in records]
    family_metrics = [_family_metrics(item) for item in metrics]
    recovery_latencies = [float(item['recovery_latency_tool_steps']) for item in family_metrics if isinstance(item.get('recovery_latency_tool_steps'), (int, float)) and item.get('recovery_latency_tool_steps') >= 0]
    qa_accuracies = [_float_metric(item, 'qa_accuracy') for item in family_metrics if 'qa_accuracy' in item]
    return {'num_episodes': len(records), 'num_success': sum((1 for m in metrics if _float_metric(m, 'task_success') >= 1.0)), 'mean_task_success': _mean([_float_metric(m, 'task_success') for m in metrics]), 'mean_task_progress': _mean([_float_metric(m, 'task_progress') for m in metrics]), 'mean_high_level_steps': _mean([_float_metric(m, 'high_level_steps') for m in metrics]), 'mean_recovery_latency': _mean(recovery_latencies) if recovery_latencies else None, 'mean_qa_accuracy': _mean(qa_accuracies) if qa_accuracies else None}

def _predicate_records(final_goal: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(final_goal.get('predicate_records') or final_goal.get('predicates') or [])

def _prepare_stage_success(stage_type: str, predicate_records: Sequence[Mapping[str, Any]]) -> bool:
    stage_type = str(stage_type or '')
    if stage_type == 'wash':
        candidates = {'object_clean', 'object_soaked', 'object_filled'}
    elif stage_type == 'heat':
        candidates = {'object_hot', 'object_cooked'}
    elif stage_type == 'cool':
        candidates = {'object_cool'}
    elif stage_type == 'slice':
        candidates = {'object_sliced'}
    elif stage_type in {'store', 'place'}:
        candidates = {'object_in_receptacle', 'object_on_receptacle'}
    else:
        return False
    matched = [p for p in predicate_records if str(p.get('type')) in candidates]
    return bool(matched) and all((p.get('satisfied') is True for p in matched))

def _tidy_violation_fixed(violation: Mapping[str, Any], predicate_records: Sequence[Mapping[str, Any]]) -> bool:
    object_role = str(violation.get('object_role') or '')
    violation_type = str(violation.get('type') or '')
    if violation_type == 'faucet_on':
        return _predicate_satisfied(predicate_records, 'object_toggled_off', object_role)
    if violation_type == 'open_container':
        return _predicate_satisfied(predicate_records, 'object_closed', object_role)
    if violation_type in {'dirty_dish', 'dirty_dish_on_floor_or_surface'}:
        return _predicate_satisfied(predicate_records, 'object_clean', object_role) and (_predicate_satisfied(predicate_records, 'object_in_receptacle', object_role) or _predicate_satisfied(predicate_records, 'object_on_receptacle', object_role) or _predicate_satisfied(predicate_records, 'object_in_any_receptacle', object_role))
    return False

def _compute_recover_primary_success(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], primary: Mapping[str, Any]) -> int:
    object_role = primary.get('object_role')
    receptacle_role = primary.get('receptacle_role')
    predicate_records = _predicate_records(final_goal)
    relevant = [predicate for predicate in predicate_records if predicate.get('object_role') == object_role or predicate.get('object_role') == receptacle_role or predicate.get('receptacle_role') == receptacle_role]
    if not relevant:
        return int(float(final_goal.get('task_success', 0.0) or 0.0) >= 1.0)
    return int(all((bool(predicate.get('satisfied', False)) for predicate in relevant)))

def _compute_recover_incident_success(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any], incident: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]]) -> int:
    predicate_records = _predicate_records(final_goal)
    incident_roles = _incident_roles(incident)
    relevant = [predicate for predicate in predicate_records if predicate.get('object_role') in incident_roles or predicate.get('receptacle_role') in incident_roles]
    final_ok = bool(relevant) and all((bool(predicate.get('satisfied', False)) for predicate in relevant))
    if incident.get('urgency') == 'immediate':
        final_ok = final_ok and _check_immediate_urgency(transitions, incident)
    return int(final_ok)

def _incident_roles(incident: Mapping[str, Any]) -> set[str]:
    roles = {_incident_object_role(incident)}
    support_roles = incident.get('support_roles') or {}
    if isinstance(support_roles, Mapping):
        roles.update((str(role) for role in support_roles.values() if role))
    for key in ('sink_role', 'faucet_role', 'storage_role', 'sink', 'faucet'):
        value = incident.get(key)
        if isinstance(value, str) and value:
            roles.add(value)
    return {role for role in roles if role}

def _check_immediate_urgency(transitions: Sequence[Mapping[str, Any]], incident: Mapping[str, Any]) -> bool:
    incident_role = _incident_object_role(incident)
    saw_navigation = False
    safe_put_down_used = False
    for call in transitions:
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

def _compute_incident_response_latency(task_spec: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], incident: Mapping[str, Any]) -> int:
    first_visible = _first_visible_tool_step(task_spec)
    first_corrective = _first_corrective_tool_index(transitions, incident)
    if first_corrective is None:
        return -1
    return max(0, first_corrective - first_visible)

def _compute_incident_detection_latency(task_spec: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], incident: Mapping[str, Any], low_level_records: Optional[Sequence[Mapping[str, Any]]]) -> int:
    if not low_level_records:
        return -1
    incident_role = _incident_object_role(incident)
    incident_object_id = ((task_spec.get('roles') or {}).get(incident_role) or {}).get('object_id')
    if not incident_object_id:
        incident_object_id = ((task_spec.get('roles') or {}).get(incident_role) or {}).get('bddl_inst')
    if not incident_object_id:
        return -1
    first_visible_stream = None
    first_visible_high_step = None
    for index, record in enumerate(low_level_records):
        visible_ids = record.get('metadata_visible_object_ids') or record.get('visible_object_ids') or []
        if incident_object_id in visible_ids:
            first_visible_stream = index
            first_visible_high_step = int(record.get('high_level_step', 0) or 0)
            break
    if first_visible_stream is None:
        return -1
    first_corrective = _first_corrective_tool_index(transitions, incident)
    if first_corrective is None:
        return -1
    corrective_high_step = first_corrective + 1
    if first_visible_high_step and corrective_high_step < first_visible_high_step:
        return -1
    visible_to_corrective = [record for record in low_level_records[first_visible_stream:] if int(record.get('high_level_step', 0) or 0) <= corrective_high_step]
    return max(0, len(visible_to_corrective) - 1)

def _first_visible_tool_step(task_spec: Mapping[str, Any]) -> int:
    metadata = task_spec.get('metadata') or {}
    visual = metadata.get('visual_discoverability') or {}
    value = visual.get('first_visible_tool_step')
    if isinstance(value, int):
        return max(0, value - 1)
    return 0

def _first_corrective_tool_index(transitions: Sequence[Mapping[str, Any]], incident: Mapping[str, Any]) -> Optional[int]:
    incident_role = _incident_object_role(incident)
    for index, call in enumerate(transitions):
        if _tool_target(call) == incident_role and _tool_name(call) in _PHYSICAL_TOOL_NAMES | {'navigate_to'}:
            return index
    return None

def _incident_object_role(incident: Mapping[str, Any]) -> str:
    role = incident.get('object_role') or incident.get('target_role') or incident.get('target')
    return str(role) if isinstance(role, str) and role else ''

def _first_disturbance_spec(task_spec: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    spec = task_spec.get('disturbance_spec')
    if isinstance(spec, list):
        return next((item for item in spec if isinstance(item, Mapping)), None)
    if isinstance(spec, Mapping):
        disturbances = spec.get('disturbances')
        if isinstance(disturbances, list):
            return next((item for item in disturbances if isinstance(item, Mapping)), None)
        if spec.get('type') is not None:
            return spec
    return None

def _first_disturbance_event_spec(scheduler_events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for event in scheduler_events or []:
        name = str(event.get('name') or '')
        if name not in DISTURBANCE_EVENT_NAMES:
            continue
        payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
        return {'type': name, 'target_role': payload.get('target_role')}
    return {}

def _first_scheduler_event_step(scheduler_events: Sequence[Mapping[str, Any]], event_name: str, target_role: Any) -> Optional[int]:
    for event in scheduler_events or []:
        if event.get('name') != event_name:
            continue
        payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
        if target_role and payload.get('target_role') not in {None, target_role}:
            continue
        for key in ('step', 'high_level_step', 'after_step'):
            value = payload.get(key)
            if isinstance(value, int):
                return value
    return None

def _first_mid_execution_recovery_index(primary: Mapping[str, Any], disturbance: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], event_step: Optional[int]) -> Optional[int]:
    disturbance_type = str(disturbance.get('type') or '')
    target_role = disturbance.get('target_role')
    start_index = max(0, int(event_step or 1) - 1)
    for index, call in enumerate(transitions[start_index:], start=start_index):
        tool_name = _tool_name(call)
        target = _tool_target(call)
        if disturbance_type == 'dirty_object':
            faucet_role = primary.get('faucet_role') or disturbance.get('faucet_role') or 'faucet'
            if tool_name in {'toggle_on', 'soak_under', 'wipe'} and target in {faucet_role, target_role}:
                return index
        elif disturbance_type == 'relocate_object':
            if tool_name in {'scan', 'look_for', 'navigate_to'} and target in {target_role, None}:
                return index
        elif disturbance_type in {'close_object', 'restore_initial_state'}:
            if tool_name == 'open' and target == target_role:
                return index
        elif disturbance_type == 'open_object':
            if tool_name == 'close' and target == target_role:
                return index
        elif disturbance_type == 'toggle_off':
            if tool_name == 'toggle_on' and target == target_role:
                return index
        elif disturbance_type == 'toggle_on':
            if tool_name == 'toggle_off' and target == target_role:
                return index
    return None

def _tool_name(call: Mapping[str, Any]) -> str:
    return str(call.get('tool_name') or call.get('name') or '')

def _tool_target(call: Mapping[str, Any]) -> Optional[str]:
    args = call.get('arguments') or call.get('args') or {}
    if isinstance(args, Mapping):
        target = args.get('target')
        if target is not None:
            return str(target)
    return None

def _first_tool_index(transitions: Sequence[Mapping[str, Any]], tool_names: set[str]) -> Optional[int]:
    for index, call in enumerate(transitions):
        if _tool_name(call) in tool_names:
            return index
    return None

def _payload_roles(payload: Mapping[str, Any]) -> set[str]:
    roles = set()
    object_role = payload.get('object_role')
    if isinstance(object_role, str) and object_role:
        roles.add(object_role)
    support_roles = payload.get('support_roles')
    if isinstance(support_roles, Mapping):
        roles.update((str(role) for role in support_roles.values() if role))
    return roles

def _first_transition_targeting_roles(transitions: Sequence[Mapping[str, Any]], roles: set[str], *, start_tools: set[str]) -> Optional[int]:
    if not roles:
        return None
    for index, call in enumerate(transitions):
        if _tool_name(call) not in start_tools:
            continue
        target = _tool_target(call)
        if target in roles:
            return index
    return None

def _group_composite_predicates(predicate_records: Sequence[Mapping[str, Any]], subtasks: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_owner: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    fallback_owner = [str(subtask.get('subtask_id') or 'subtask_{}'.format(index)) for index, subtask in enumerate(subtasks)]
    for index, record in enumerate(predicate_records):
        owner = record.get('owner_subtask_id')
        if not isinstance(owner, str) or not owner:
            owner = fallback_owner[min(index, len(fallback_owner) - 1)] if fallback_owner else 'unknown'
        by_owner[owner].append(record)
    return dict(by_owner)

def _scheduler_events_for_subtask(scheduler_events: Sequence[Mapping[str, Any]], subtask_id: str, window_start: int, window_end: int) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for event in scheduler_events or []:
        payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
        owner = payload.get('owner_subtask_id')
        activation = payload.get('activation') if isinstance(payload.get('activation'), Mapping) else {}
        if owner is None:
            owner = activation.get('subtask_id') or activation.get('owner_subtask_id')
        if owner is not None:
            if str(owner) == str(subtask_id):
                out.append(event)
            continue
        step = _scheduler_event_step(event)
        if step is not None and int(window_start) <= int(step) <= int(window_end):
            out.append(event)
    return out

def _scheduler_event_step(event: Mapping[str, Any]) -> Optional[int]:
    payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
    for key in ('step', 'high_level_step', 'after_step'):
        value = payload.get(key)
        if isinstance(value, int):
            return int(value)
    return None

def _composite_completion_steps(task_spec: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]], subtasks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    owners = [str(subtask.get('subtask_id') or 'subtask_{}'.format(index)) for index, subtask in enumerate(subtasks)]
    completion_steps: dict[str, int] = {}
    for step_index, transition in enumerate(transitions, start=1):
        step_goal = transition.get('goal_result')
        if not isinstance(step_goal, Mapping):
            continue
        grouped = _group_composite_predicates(_predicate_records(step_goal), subtasks)
        for owner in owners:
            if owner in completion_steps:
                continue
            records = grouped.get(owner) or []
            if records and all((bool(record.get('satisfied', False)) for record in records)):
                completion_steps[owner] = step_index
    return completion_steps

def _low_level_records_from_transitions(transitions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, transition in enumerate(transitions, start=1):
        for record in transition.get('low_level_actions') or []:
            if isinstance(record, Mapping):
                out = dict(record)
                out.setdefault('high_level_step', index)
                records.append(out)
    return records

def _predicate_satisfied(predicate_records: Sequence[Mapping[str, Any]], predicate_type: str, object_role: str) -> bool:
    return any((predicate.get('type') == predicate_type and predicate.get('object_role') == object_role and bool(predicate.get('satisfied', False)) for predicate in predicate_records))

def _compute_faucet_urgency_success(violations: Sequence[Mapping[str, Any]], transitions: Sequence[Mapping[str, Any]]) -> int:
    faucet_roles = {str(violation.get('object_role')) for violation in violations if violation.get('object_role') is not None and (violation.get('type') == 'faucet_on' or violation.get('urgency') == 'immediate')}
    if not faucet_roles:
        return 1
    for call in transitions:
        tool_name = _tool_name(call)
        if tool_name not in _PHYSICAL_TOOL_NAMES:
            continue
        return 1 if _tool_target(call) in faucet_roles else 0
    return 0

def _recover_faucet_urgency_success(incidents: Sequence[Mapping[str, Any]], transitions: Sequence[Mapping[str, Any]]) -> int:
    faucet_roles = {_incident_object_role(incident) for incident in incidents if incident.get('type') == 'faucet_on'}
    dirty_roles = {_incident_object_role(incident) for incident in incidents if incident.get('type') in {'dirty_dish', 'dirty_dish_on_floor_or_surface'}}
    faucet_roles.discard('')
    dirty_roles.discard('')
    if not faucet_roles or not dirty_roles:
        return 1
    first_faucet = None
    first_dirty = None
    for index, call in enumerate(transitions):
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

def _human_events(human: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = human.get('human_events')
    if not isinstance(events, list):
        events = human.get('human_events_template')
    return [event for event in events or [] if isinstance(event, Mapping)]

def _human_posted_steps(transitions: Sequence[Mapping[str, Any]], scheduler_events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    posted: dict[str, dict[str, Any]] = {}
    for transition in transitions:
        step = _transition_step(transition, transitions)
        for event in transition.get('human_events_posted') or []:
            if not isinstance(event, Mapping):
                continue
            event_id = event.get('id')
            if event_id is None:
                continue
            posted.setdefault(str(event_id), {'type': str(event.get('type') or ''), 'step': int(step or 0)})
    for event in scheduler_events or []:
        name = event.get('name')
        if name not in {'human_qa_posted', 'human_insert_posted'}:
            continue
        payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
        event_id = payload.get('id')
        if event_id is None:
            continue
        posted[str(event_id)] = {'type': 'qa' if name == 'human_qa_posted' else 'insert', 'step': int(payload.get('after_step', 0) or 0)}
    return posted

def _answer_tool_calls_by_question(transitions: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    calls = {}
    for transition in transitions:
        if str(transition.get('tool_name') or '') != 'answer':
            continue
        args = transition.get('arguments') if isinstance(transition.get('arguments'), Mapping) else {}
        question_id = args.get('question_id')
        if question_id is not None and str(question_id) not in calls:
            calls[str(question_id)] = transition
    return calls

def _answer_value(transition: Mapping[str, Any]) -> Any:
    args = transition.get('arguments') if isinstance(transition.get('arguments'), Mapping) else {}
    return args.get('value')

def _expected_human_answer(event: Mapping[str, Any], runtime_answers: Optional[Mapping[str, Any]]=None) -> Any:
    event_id = event.get('id')
    if runtime_answers is not None and event_id is not None and (str(event_id) in runtime_answers):
        return runtime_answers[str(event_id)]
    if event.get('evaluator_answer') is not None:
        return event.get('evaluator_answer')
    return None

def _normalize_answer(value: Any) -> str:
    return str(value or '').strip().lower()

def _human_insert_completed(final_goal: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    predicates = [p for p in event.get('goal_predicates') or [] if isinstance(p, Mapping)]
    if not predicates:
        return False
    records = _predicate_records(final_goal)
    return all((_matching_predicate_satisfied(records, predicate) for predicate in predicates))

def _matching_predicate_satisfied(predicate_records: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]) -> bool:
    for record in predicate_records:
        if _predicate_matches(record, expected) and record.get('satisfied') is True:
            return True
    return False

def _predicate_matches(candidate: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if candidate.get('type') != expected.get('type'):
        return False
    for key in ('object', 'receptacle'):
        expected_value = _predicate_arg(expected, key)
        if expected_value is None:
            continue
        candidate_key = 'object_role' if key == 'object' else 'receptacle_role'
        if str(candidate.get(candidate_key)) != str(expected_value):
            return False
    expected_receptacles = _predicate_arg(expected, 'receptacle_roles')
    if expected_receptacles is not None:
        if list(candidate.get('receptacle_roles') or []) != list(expected_receptacles or []):
            return False
    return True

def _predicate_arg(predicate: Mapping[str, Any], key: str) -> Any:
    args = predicate.get('args') if isinstance(predicate.get('args'), Mapping) else {}
    if key == 'object':
        return predicate.get('object_role') or predicate.get('role') or args.get('object')
    if key == 'receptacle':
        return predicate.get('receptacle_role') or args.get('receptacle')
    if key == 'receptacle_roles':
        return predicate.get('receptacle_roles') or args.get('receptacle_roles')
    return args.get(key)

def _human_primary_task_success(task_spec: Mapping[str, Any], final_goal: Mapping[str, Any]) -> float:
    primary = ((task_spec.get('family_spec') or {}).get('human') or {}).get('primary_task') or {}
    if not isinstance(primary, Mapping):
        return float(final_goal.get('task_success', 0.0) or 0.0)
    if primary.get('type') == 'simple_place':
        records = _predicate_records(final_goal)
        expected = {'type': 'object_in_receptacle', 'object_role': primary.get('object_role'), 'receptacle_role': primary.get('receptacle_role')}
        placed = _matching_predicate_satisfied(records, expected)
        expected['type'] = 'object_on_receptacle'
        placed = placed or _matching_predicate_satisfied(records, expected)
        relevant_closed = [record for record in records if record.get('type') == 'object_closed' and str(record.get('object_role')) == str(primary.get('receptacle_role'))]
        closed_ok = all((record.get('satisfied') is True for record in relevant_closed))
        return 1.0 if placed and closed_ok else 0.0
    return float(final_goal.get('task_success', 0.0) or 0.0)

def _human_runtime_answer_keys(scheduler_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys: dict[str, Any] = {}
    for event in scheduler_events or []:
        if event.get('name') != 'human_qa_posted':
            continue
        payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
        question_id = payload.get('id')
        evaluator_answer = payload.get('evaluator_answer')
        if question_id is not None and evaluator_answer is not None:
            keys[str(question_id)] = evaluator_answer
    return keys

def _first_injection_step(scheduler_events: Sequence[Mapping[str, Any]]) -> Optional[int]:
    for event in scheduler_events or []:
        name = str(event.get('name') or '')
        if name not in DISTURBANCE_EVENT_NAMES and (not (name.startswith('failure_') and name != 'failure_armed')):
            continue
        payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
        for key in ('step', 'high_level_step', 'after_step'):
            value = payload.get(key)
            if isinstance(value, int):
                return value
    return None

def _first_recovery_step(transitions: Sequence[Mapping[str, Any]], disturbance_step: Optional[int], *, end_step: Optional[int]=None) -> Optional[int]:
    if disturbance_step is None:
        return None
    for transition in transitions or []:
        step = _transition_step(transition, transitions)
        if step is None or int(step) < int(disturbance_step):
            continue
        if end_step is not None and int(step) > int(end_step):
            continue
        goal = transition.get('goal_result') if isinstance(transition.get('goal_result'), Mapping) else {}
        if float(goal.get('task_success', 0.0) or 0.0) >= 1.0:
            return int(step)
    return None

def _first_composite_subtask_recovery_step(transitions: Sequence[Mapping[str, Any]], disturbance_step: Optional[int], subtasks: Sequence[Mapping[str, Any]], subtask_id: str, *, end_step: Optional[int]=None) -> Optional[int]:
    if disturbance_step is None:
        return None
    for transition in transitions or []:
        step = _transition_step(transition, transitions)
        if step is None or int(step) < int(disturbance_step):
            continue
        if end_step is not None and int(step) > int(end_step):
            continue
        goal = transition.get('goal_result') if isinstance(transition.get('goal_result'), Mapping) else {}
        grouped = _group_composite_predicates(_predicate_records(goal), subtasks)
        records = grouped.get(str(subtask_id)) or []
        if records and all((bool(record.get('satisfied', False)) for record in records)):
            return int(step)
    return None

def _transition_step(transition: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]]) -> Optional[int]:
    value = transition.get('high_level_step')
    if isinstance(value, int):
        return value
    for index, candidate in enumerate(transitions, start=1):
        if candidate is transition:
            return index
    return None

def _float_metric(metrics: Mapping[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0

def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0
__all__ = ['compute_episode_metrics', 'scheduler_event_counts', 'compute_family_metrics', 'compute_human_interaction_metrics', 'compute_mid_execution_disturbance_metrics', 'compute_recover_metrics', 'compute_tidy_metrics', 'compute_explore_metrics', 'compute_observation_conditioned_metrics', 'compute_composite_metrics', 'aggregate_metrics', 'build_metrics_consistency_records']
