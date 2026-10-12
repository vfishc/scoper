from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv, MODEL_VISIBLE_TOOLS
from scoper.benchmark.behavior.brains.teachers.dispatcher import TeacherDispatcherError, default_dispatcher
from scoper.benchmark.behavior.episode_runner import EpisodeRunner, EpisodeResult
from scoper.benchmark.behavior.scheduler import build_scheduler_from_task_spec
from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec

def validate_stage6_execution(task_spec: Mapping[str, Any], *, stage5_result: Mapping[str, Any] | None=None, rng_seed: int=0, max_high_level_steps: int=100, trace_dir: Path | None=None, scene_layout_mode: str='task_roles', load_doors: bool=False) -> dict[str, Any]:
    result: dict[str, Any] = {'task_id': task_spec.get('id'), 'template_id': (task_spec.get('template') or {}).get('template_id'), 'category': task_spec.get('category'), 'scene_model': (task_spec.get('source') or {}).get('scene_model'), 'stage5_admitted': None, 'structure_ok': False, 'teacher_ready': False, 'teacher_source': None, 'teacher_type': None, 'scheduler_ready': True, 'execution_attempted': False, 'execution_ok': False, 'goal_success': False, 'stage6_bucket': 'not_started', 'failure_code': None, 'failure_reason': None, 'deferred_codes': [], 'unsupported_goal_predicates': [], 'episode_status': None, 'final_goal': {}, 'metrics': {}, 'transition_count': 0, 'tool_calls': [], 'predicate_progress': [], 'trace_path': None, 'scene_layout_mode': str(scene_layout_mode or 'task_roles'), 'load_doors': bool(load_doors)}
    if stage5_result is not None:
        admitted = bool(stage5_result.get('stage5_reset_mutation_ok'))
        result['stage5_admitted'] = admitted
        result['deferred_codes'] = sorted(set(stage5_result.get('deferred_codes') or []))
        if not admitted:
            result['stage6_bucket'] = 'stage5_not_admitted'
            result['failure_code'] = 'STAGE5_NOT_ADMITTED'
            result['failure_reason'] = str(stage5_result.get('failure_code') or 'Stage 5 reset/mutation gate failed.')
            return result
    try:
        validate_behavior_task_spec(task_spec)
        result['structure_ok'] = True
    except Exception as exc:
        result['stage6_bucket'] = 'structure_failed'
        result['failure_code'] = 'TASK_SPEC_VALIDATION_FAILED'
        result['failure_reason'] = str(exc)
        return result
    unsupported_scheduler = list((task_spec.get('generation') or {}).get('unsupported_scheduler_specs') or [])
    if unsupported_scheduler:
        result['scheduler_ready'] = False
        result['deferred_codes'] = sorted(set(result['deferred_codes']) | {'UNSUPPORTED_SCHEDULER_SPEC'})
    dispatcher = default_dispatcher()
    try:
        brain, teacher_source = dispatcher.select(task_spec)
        result['teacher_ready'] = True
        result['teacher_source'] = teacher_source
        result['teacher_type'] = type(brain).__name__
    except TeacherDispatcherError as exc:
        result['stage6_bucket'] = 'teacher_deferred'
        result['failure_code'] = 'TEACHER_DEFERRED'
        result['failure_reason'] = str(exc)
        result['deferred_codes'] = sorted(set(result['deferred_codes']) | {'TEACHER_DEFERRED'})
        result['unsupported_goal_predicates'] = _unsupported_goal_predicates_from_message(str(exc))
        return result
    trace_path = None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / '{}.jsonl'.format(_safe_filename(str(task_spec.get('id') or 'task')))
        result['trace_path'] = str(trace_path)
    try:
        env = BehaviorClosedLoopToolEnv(max_high_level_steps=max_high_level_steps, scene_layout_mode=scene_layout_mode, load_doors=load_doors)
        scheduler = build_scheduler_from_task_spec(task_spec)
    except Exception as exc:
        result['scheduler_ready'] = False
        result['stage6_bucket'] = 'execution_failed'
        result['failure_code'] = 'SCHEDULER_CONFIG_INVALID'
        result['failure_reason'] = str(exc)
        return result
    try:
        episode = EpisodeRunner(env, max_high_level_steps=max_high_level_steps).run(task_spec, brain=brain, scheduler=scheduler, trace_path=trace_path, rng_seed=rng_seed)
    except Exception as exc:
        result['execution_attempted'] = True
        result['stage6_bucket'] = 'execution_failed'
        result['failure_code'] = 'STAGE6_EXECUTION_EXCEPTION'
        result['failure_reason'] = str(exc)
        return result
    result.update(_episode_result_fields(episode))
    result['execution_attempted'] = True
    result['execution_ok'] = episode.status == 'passed'
    result['goal_success'] = float((episode.final_goal or {}).get('task_success', 0.0)) == 1.0
    if result['execution_ok'] and result['deferred_codes']:
        result['stage6_bucket'] = 'execution_passed_with_deferred_runtime'
    elif result['execution_ok']:
        result['stage6_bucket'] = 'execution_passed'
    else:
        result['stage6_bucket'] = 'execution_failed'
        result['failure_code'] = episode.failure_code or _execution_failure_code(episode)
        result['failure_reason'] = episode.failure_reason or 'Final task_success is not 1.0.'
        if episode.failure_details:
            result['failure_details'] = episode.failure_details
    return result

def summarize_stage6_results(generation: Mapping[str, Any], stage5_results: Sequence[Mapping[str, Any]], stage6_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    generated = generation.get('tasks') or []
    stage5_passed = sum((1 for item in stage5_results if item.get('stage5_reset_mutation_ok')))
    attempted = sum((1 for item in stage6_results if item.get('execution_attempted')))
    passed = sum((1 for item in stage6_results if item.get('execution_ok')))
    teacher_ready = sum((1 for item in stage6_results if item.get('teacher_ready')))
    strict_teacher_ready = sum((1 for item in stage6_results if str(item.get('teacher_source') or '').startswith(('family:', 'legacy:'))))
    predicate_fallback_ready = sum((1 for item in stage6_results if item.get('teacher_source') == 'predicate_fallback'))
    execution_passed_runtime_complete = sum((1 for item in stage6_results if item.get('stage6_bucket') == 'execution_passed'))
    execution_passed_with_deferred_runtime = sum((1 for item in stage6_results if item.get('stage6_bucket') == 'execution_passed_with_deferred_runtime'))
    deferred = sum((1 for item in stage6_results if item.get('stage6_bucket') in {'teacher_deferred', 'stage5_not_admitted'}))
    return {'generated_total': len(generated), 'stage5_validated_total': len(stage5_results), 'stage5_admitted': stage5_passed, 'stage6_validated_total': len(stage6_results), 'teacher_ready': teacher_ready, 'strict_teacher_ready': strict_teacher_ready, 'predicate_fallback_ready': predicate_fallback_ready, 'execution_attempted': attempted, 'execution_passed': passed, 'execution_passed_runtime_complete': execution_passed_runtime_complete, 'execution_passed_with_deferred_runtime': execution_passed_with_deferred_runtime, 'execution_failed': sum((1 for item in stage6_results if item.get('stage6_bucket') == 'execution_failed')), 'deferred_total': deferred, 'execution_pass_ratio': float(passed / attempted) if attempted else 0.0, 'execution_runtime_complete_ratio': float(execution_passed_runtime_complete / attempted) if attempted else 0.0, 'overall_stage5_pass_ratio': float(passed / stage5_passed) if stage5_passed else 0.0, 'overall_stage5_runtime_complete_ratio': float(execution_passed_runtime_complete / stage5_passed) if stage5_passed else 0.0, 'stage6_buckets': dict(sorted(Counter((item.get('stage6_bucket') for item in stage6_results)).items())), 'teacher_sources': dict(sorted(Counter((item.get('teacher_source') or '-' for item in stage6_results)).items())), 'teacher_types': dict(sorted(Counter((item.get('teacher_type') or '-' for item in stage6_results)).items())), 'failure_codes': dict(sorted(Counter((item.get('failure_code') for item in stage6_results if item.get('failure_code'))).items())), 'deferred_codes': dict(sorted(Counter((code for item in stage6_results for code in item.get('deferred_codes') or [])).items())), 'executed_by_template': dict(sorted(Counter((item.get('template_id') for item in stage6_results if item.get('execution_attempted'))).items())), 'passed_by_template': dict(sorted(Counter((item.get('template_id') for item in stage6_results if item.get('execution_ok'))).items())), 'failed_by_template': dict(sorted(Counter((item.get('template_id') for item in stage6_results if item.get('failure_code'))).items()))}

def build_tool_micro_cases(rng_seed: int=0) -> list[dict[str, Any]]:
    cases: list[tuple[str, str, list[tuple[str, dict[str, Any]]], str, dict[str, Any], bool | None]] = [('navigate_to_success', 'navigate_to', [], 'navigate_to', {'target': 'target_object'}, True), ('navigate_to_failure', 'navigate_to', [], 'navigate_to', {'target': 'missing_role'}, False), ('navigate_to_room_success', 'navigate_to_room', [], 'navigate_to_room', {'room': 'kitchen_0'}, True), ('navigate_to_room_failure', 'navigate_to_room', [], 'navigate_to_room', {'room': 'garage_0'}, False), ('pick_up_success', 'pick_up', [('navigate_to', {'target': 'target_object'})], 'pick_up', {'target': 'target_object'}, True), ('pick_up_failure', 'pick_up', [('navigate_to', {'target': 'target_receptacle'})], 'pick_up', {'target': 'target_receptacle'}, False), ('put_down_success', 'put_down', [('navigate_to', {'target': 'target_object'}), ('pick_up', {'target': 'target_object'}), ('navigate_to', {'target': 'target_receptacle'})], 'put_down', {'target': 'target_receptacle'}, True), ('put_down_failure', 'put_down', [('navigate_to', {'target': 'target_receptacle'})], 'put_down', {'target': 'target_receptacle'}, False), ('drop_success', 'drop', [('navigate_to', {'target': 'target_object'}), ('pick_up', {'target': 'target_object'})], 'drop', {}, True), ('drop_failure', 'drop', [], 'drop', {}, False), ('open_success', 'open', [('navigate_to', {'target': 'fridge'})], 'open', {'target': 'fridge'}, True), ('open_failure', 'open', [('navigate_to', {'target': 'target_receptacle'})], 'open', {'target': 'target_receptacle'}, False), ('close_success', 'close', [('navigate_to', {'target': 'fridge'}), ('open', {'target': 'fridge'})], 'close', {'target': 'fridge'}, True), ('close_failure', 'close', [('navigate_to', {'target': 'target_receptacle'})], 'close', {'target': 'target_receptacle'}, False), ('toggle_on_success', 'toggle_on', [('navigate_to', {'target': 'faucet'})], 'toggle_on', {'target': 'faucet'}, True), ('toggle_on_failure', 'toggle_on', [('navigate_to', {'target': 'target_receptacle'})], 'toggle_on', {'target': 'target_receptacle'}, False), ('toggle_off_success', 'toggle_off', [('navigate_to', {'target': 'faucet'}), ('toggle_on', {'target': 'faucet'})], 'toggle_off', {'target': 'faucet'}, True), ('toggle_off_failure', 'toggle_off', [('navigate_to', {'target': 'target_receptacle'})], 'toggle_off', {'target': 'target_receptacle'}, False), ('scan_success', 'scan', [], 'scan', {}, True), ('look_for_success', 'look_for', [], 'look_for', {'target': 'target_object'}, True), ('look_for_failure', 'look_for', [], 'look_for', {'target': 'missing_role'}, False), ('answer_success', 'answer', [('look_for', {'target': 'target_object'})], 'answer', {'question_id': 'micro_qa', 'value': 'apple'}, True), ('answer_failure', 'answer', [], 'answer', {'question_id': 'missing_qa', 'value': 'apple'}, False), ('stop_success', 'stop', [], 'stop', {}, True), ('wipe_success', 'wipe', [('navigate_to', {'target': 'rag'}), ('pick_up', {'target': 'rag'}), ('navigate_to', {'target': 'target_object'})], 'wipe', {'target': 'target_object'}, True), ('wipe_failure', 'wipe', [('navigate_to', {'target': 'target_object'})], 'wipe', {'target': 'target_object'}, False), ('cut_success', 'cut', [('navigate_to', {'target': 'knife'}), ('pick_up', {'target': 'knife'}), ('navigate_to', {'target': 'target_object'})], 'cut', {'target': 'target_object'}, True), ('cut_failure', 'cut', [('navigate_to', {'target': 'target_object'})], 'cut', {'target': 'target_object'}, False), ('soak_under_success', 'soak_under', [('navigate_to', {'target': 'cup'}), ('pick_up', {'target': 'cup'}), ('navigate_to', {'target': 'faucet'}), ('toggle_on', {'target': 'faucet'})], 'soak_under', {'target': 'faucet'}, True), ('soak_under_failure', 'soak_under', [('navigate_to', {'target': 'faucet'}), ('toggle_on', {'target': 'faucet'})], 'soak_under', {'target': 'faucet'}, False), ('fill_success', 'fill', [('navigate_to', {'target': 'empty_cup'}), ('pick_up', {'target': 'empty_cup'}), ('navigate_to', {'target': 'faucet'}), ('toggle_on', {'target': 'faucet'})], 'fill', {'target': 'empty_cup', 'source': 'faucet'}, True), ('fill_failure', 'fill', [], 'fill', {'target': 'empty_cup'}, False), ('pour_success', 'pour', [('navigate_to', {'target': 'cup'}), ('pick_up', {'target': 'cup'}), ('navigate_to', {'target': 'bowl'})], 'pour', {'target': 'bowl'}, True), ('pour_failure', 'pour', [('navigate_to', {'target': 'bowl'})], 'pour', {'target': 'bowl'}, False), ('heat_success', 'heat', [('navigate_to', {'target': 'target_object'}), ('pick_up', {'target': 'target_object'}), ('navigate_to', {'target': 'microwave'}), ('open', {'target': 'microwave'})], 'heat', {'target': 'microwave'}, True), ('heat_failure', 'heat', [('navigate_to', {'target': 'microwave'})], 'heat', {'target': 'microwave'}, False), ('cool_success', 'cool', [('navigate_to', {'target': 'target_object'}), ('pick_up', {'target': 'target_object'}), ('navigate_to', {'target': 'fridge'}), ('open', {'target': 'fridge'})], 'cool', {'target': 'fridge'}, True), ('cool_failure', 'cool', [('navigate_to', {'target': 'fridge'})], 'cool', {'target': 'fridge'}, False)]
    results = []
    for case_id, tool_group, setup, tool_name, arguments, expected in cases:
        env = BehaviorClosedLoopToolEnv(max_high_level_steps=40)
        task_spec = _micro_tool_task_spec()
        env.reset_to_task(task_spec, rng_seed=rng_seed)
        scheduler = build_scheduler_from_task_spec(task_spec)
        scheduler.on_episode_start(env, task_spec)
        setup_records = []
        for setup_tool, setup_args in setup:
            setup_records.append(_micro_scheduler_call(env, scheduler, setup_tool, setup_args, reasoning='stage6 micro setup'))
        transition = _micro_scheduler_call(env, scheduler, tool_name, arguments, reasoning='stage6 micro case')
        success = bool(transition.get('success'))
        results.append({'case_id': case_id, 'tool': tool_group, 'tool_name': tool_name, 'arguments': dict(arguments), 'expected_success': expected, 'success': success, 'matches_expected': expected is None or success is bool(expected), 'error_type': (transition.get('env_feedback') or {}).get('error_type'), 'error_message': (transition.get('env_feedback') or {}).get('error_message'), 'state_delta': transition.get('state_delta'), 'setup_success': all((bool(item.get('success')) for item in setup_records)), 'setup': [_transition_summary(item) for item in setup_records], 'transition': _transition_summary(transition)})
    return results

def _micro_scheduler_call(env: BehaviorClosedLoopToolEnv, scheduler: Any, tool_name: str, arguments: Mapping[str, Any], *, reasoning: str) -> dict[str, Any]:
    next_step = int(env.high_level_step) + 1
    scheduler.pre_step(env, next_step, env.tool_history[-1] if env.tool_history else None)
    transition = env.call_tool(tool_name, arguments, reasoning=reasoning)
    scheduler.post_step(env, env.high_level_step, transition)
    return transition

def summarize_tool_micro_cases(micro_cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_tool: dict[str, dict[str, Any]] = {}
    for tool in MODEL_VISIBLE_TOOLS:
        cases = [case for case in micro_cases if case.get('tool') == tool]
        by_tool[tool] = {'cases': len(cases), 'success_examples': sum((1 for case in cases if case.get('success') is True)), 'failure_examples': sum((1 for case in cases if case.get('success') is False)), 'matches_expected': sum((1 for case in cases if case.get('matches_expected'))), 'all_match_expected': bool(cases) and all((case.get('matches_expected') for case in cases))}
    return {'tool_count': len(MODEL_VISIBLE_TOOLS), 'case_count': len(micro_cases), 'all_cases_match_expected': all((case.get('matches_expected') for case in micro_cases)), 'by_tool': by_tool}

def _episode_result_fields(episode: EpisodeResult) -> dict[str, Any]:
    return {'episode_status': episode.status, 'final_goal': episode.final_goal, 'metrics': episode.metrics, 'transition_count': len(episode.transitions), 'tool_calls': [_transition_summary(item) for item in episode.transitions], 'predicate_progress': _predicate_progress(episode.transitions), 'trace_path': episode.trace_path, 'scheduler_events': episode.scheduler_events, 'failure_code': episode.failure_code, 'failure_details': episode.failure_details}

def _transition_summary(transition: Mapping[str, Any]) -> dict[str, Any]:
    feedback = transition.get('env_feedback') or {}
    goal = transition.get('goal_result') or {}
    return {'step': transition.get('high_level_step'), 'tool_name': transition.get('tool_name'), 'arguments': transition.get('arguments') or {}, 'success': bool(transition.get('success')), 'error_type': feedback.get('error_type'), 'error_message': feedback.get('error_message'), 'frames': transition.get('frames'), 'num_low_level_steps': transition.get('num_low_level_steps', transition.get('frames')), 'done': bool(transition.get('done')), 'done_reason': transition.get('done_reason'), 'task_success': goal.get('task_success'), 'task_progress': goal.get('task_progress'), 'env_feedback': dict(feedback), 'agent_pose': transition.get('agent_pose') or {}, 'held_objects': list(transition.get('held_objects') or []), 'state_delta_keys': sorted((transition.get('state_delta') or {}).keys()) if isinstance(transition.get('state_delta'), Mapping) else []}

def _predicate_progress(transitions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    progress = []
    for transition in transitions:
        goal = transition.get('goal_result') or {}
        predicates = goal.get('predicates') or []
        progress.append({'step': transition.get('high_level_step'), 'tool_name': transition.get('tool_name'), 'task_success': goal.get('task_success'), 'task_progress': goal.get('task_progress'), 'satisfied': sum((1 for item in predicates if item.get('satisfied'))), 'total': len(predicates), 'predicates': predicates})
    return progress

def _execution_failure_code(episode: EpisodeResult) -> str:
    if episode.failure_type == 'BrainError':
        return 'BRAIN_ERROR'
    for transition in episode.transitions:
        if not transition.get('success'):
            error_type = (transition.get('env_feedback') or {}).get('error_type') or 'unknown'
            return 'TOOL_EXECUTION_FAILED:{}'.format(error_type)
    if episode.failure_type:
        return 'EPISODE_FAILED:{}'.format(episode.failure_type)
    return 'GOAL_NOT_SATISFIED'

def _unsupported_goal_predicates_from_message(message: str) -> list[str]:
    marker = 'unsupported goal predicates:'
    if marker not in message:
        return []
    tail = message.split(marker, 1)[1].strip()
    if not tail or tail == '-':
        return []
    return [item.strip() for item in tail.split(',') if item.strip()]

def _safe_filename(value: str) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '_', value)[:180]

def _micro_tool_task_spec() -> dict[str, Any]:
    return {'id': 'stage6_tool_micro_case', 'version': 'stage6.0', 'backend': 'behavior', 'category': 'stage6_tool_micro_case', 'source': {'type': 'template', 'scene_model': 'stage6_symbolic_micro_scene', 'occupancy_bounds': [-1.0, -1.0, 8.5, 7.5], 'start_pose': {'x': 0.5, 'y': 0.5, 'z': 1.55, 'yaw': 0.0, 'pitch': 0.0}, 'room_catalog': [{'room_id': 'kitchen_0', 'room_type': 'kitchen', 'object_count': 8, 'source': 'stage6_micro_case', 'geometry_status': 'synthetic_profile', 'reachable_status': 'profile_fallback', 'representative_reachable_pose': {'position': {'x': 1.5, 'y': 0.0, 'z': 1.5}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'source': 'stage6_micro_case'}}, {'room_id': 'dining_room_0', 'room_type': 'dining_room', 'object_count': 2, 'source': 'stage6_micro_case', 'geometry_status': 'synthetic_profile', 'reachable_status': 'profile_fallback', 'representative_reachable_pose': {'position': {'x': 6.6, 'y': 0.0, 'z': 5.7}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'source': 'stage6_micro_case'}}]}, 'instruction': 'Stage 6 public tool micro-case scene.', 'family_spec': {'family': 'human_interactive_execution', 'human': {'human_events_template': [{'id': 'micro_qa', 'type': 'qa', 'question_type': 'object_identity', 'question_template': 'What is the target object?', 'answer_set': ['apple'], 'requires_visible_roles': ['target_object'], 'activation': {'type': 'milestone', 'milestone': 'after_successful_tool_on_role', 'tool': 'look_for', 'role': 'target_object'}}]}}, 'roles': {'floor': {'synset': 'floor.n.01', 'bddl_inst': 'floor.n.01_1', 'category': 'floors', 'center': [3.5, 3.5, -0.02], 'extent': [8.0, 8.0, 0.04]}, 'target_object': {'synset': 'apple.n.01', 'bddl_inst': 'apple.n.01_1', 'category': 'apple', 'abilities': ['pickupable', 'wipeable', 'cuttable', 'soakable'], 'center': [1.5, 1.5, 0.06], 'extent': [0.12, 0.12, 0.12], 'init_states': {'dirty': True, 'sliced': False, 'cooked': False, 'cooled': False, 'saturated': {}}}, 'target_receptacle': {'synset': 'breakfast_table.n.01', 'bddl_inst': 'breakfast_table.n.01_1', 'category': 'breakfast_table', 'abilities': ['support_surface'], 'center': [6.6, 5.7, 0.45], 'extent': [0.7, 0.7, 0.9]}, 'fridge': {'synset': 'electric_refrigerator.n.01', 'bddl_inst': 'electric_refrigerator.n.01_1', 'category': 'fridge', 'abilities': ['openable', 'container', 'coldSource'], 'center': [5.0, 1.5, 0.9], 'extent': [0.8, 0.8, 1.8], 'init_states': {'open': False}}, 'microwave': {'synset': 'microwave.n.02', 'bddl_inst': 'microwave.n.02_1', 'category': 'microwave', 'abilities': ['openable', 'container', 'heatSource', 'toggleable'], 'center': [6.6, 1.5, 0.7], 'extent': [0.6, 0.5, 0.5], 'init_states': {'open': False, 'toggled_on': False}}, 'faucet': {'synset': 'faucet.n.01', 'bddl_inst': 'faucet.n.01_1', 'category': 'faucet', 'abilities': ['toggleable', 'water_source'], 'center': [1.5, 4.2, 0.7], 'extent': [0.16, 0.16, 0.2], 'init_states': {'toggled_on': False, 'source_substance': 'water'}}, 'rag': {'synset': 'rag.n.01', 'bddl_inst': 'rag.n.01_1', 'category': 'rag', 'abilities': ['pickupable', 'cleaning_tool'], 'center': [2.4, 4.2, 0.04], 'extent': [0.16, 0.12, 0.08]}, 'knife': {'synset': 'knife.n.01', 'bddl_inst': 'knife.n.01_1', 'category': 'knife', 'abilities': ['pickupable', 'cutting_tool'], 'center': [3.2, 4.2, 0.03], 'extent': [0.28, 0.05, 0.04]}, 'cup': {'synset': 'cup.n.01', 'bddl_inst': 'cup.n.01_1', 'category': 'cup', 'abilities': ['pickupable', 'fillable', 'pourable', 'container', 'soakable'], 'center': [4.1, 4.2, 0.08], 'extent': [0.16, 0.16, 0.16], 'init_states': {'filled': {'water': True}, 'contains': {'water': True}, 'open': True, 'saturated': {}}}, 'empty_cup': {'synset': 'cup.n.01', 'bddl_inst': 'cup.n.01_2', 'category': 'cup', 'abilities': ['pickupable', 'fillable', 'pourable', 'container', 'soakable'], 'center': [4.1, 5.2, 0.08], 'extent': [0.16, 0.16, 0.16], 'init_states': {'filled': {}, 'contains': {}, 'open': True, 'saturated': {}}}, 'bowl': {'synset': 'bowl.n.01', 'bddl_inst': 'bowl.n.01_1', 'category': 'bowl', 'abilities': ['fillable', 'container'], 'center': [5.0, 4.2, 0.08], 'extent': [0.22, 0.22, 0.12], 'init_states': {'filled': {}, 'contains': {}, 'open': True}}}, 'init': [], 'goal': {'type': 'predicates', 'require_model_stop': False, 'bddl_goal': None, 'predicates': [{'type': 'object_on_receptacle', 'args': {'object': 'target_object', 'receptacle': 'target_receptacle'}}]}}
__all__ = ['validate_stage6_execution', 'summarize_stage6_results', 'build_tool_micro_cases', 'summarize_tool_micro_cases']
