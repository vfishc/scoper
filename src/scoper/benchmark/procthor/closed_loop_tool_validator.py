from __future__ import annotations
import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple
from scoper.benchmark.procthor.closed_loop_tool_env import ProcTHORClosedLoopToolEnv
from scoper.benchmark.procthor.failure_analysis import classify_failure_code_from_record
from scoper.benchmark.procthor.manipulathor_config import manipulathor_env_kwargs
from scoper.benchmark.procthor.online_task_loader import load_task_bank, load_task_set, resolve_task_set
from scoper.benchmark.procthor.online_task_spec import validate_online_task_spec
from scoper.benchmark.procthor.paths import resolve_repo_path
from scoper.benchmark.procthor.split_builder import build_task_sets, write_task_sets
CLOSED_LOOP_VALIDATION_VERSION = '0.1'
CLOSED_LOOP_VALIDATION_CHECK_KEY = 'closed_loop_tool_execution'
ALLOWED_FORCE_NAVIGATION_ACTIONS = {'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft', 'RotateRight'}
ALLOWED_FORCE_OPEN_CONTAINER_ACTIONS = {'OpenObject'}
ALLOWED_FORCE_ARM_OPEN_CLOSE_ACTIONS = {'OpenObject', 'CloseObject'}
ALLOWED_FORCE_PUT_OBJECT_ACTIONS = {'DropHandObject', 'PutObject'}
ALLOWED_FORCE_PICKUP_OPEN_CONTAINER_ACTIONS = {'PickupObject'}

class ClosedLoopToolValidationError(ValueError):
    pass

def run_closed_loop_tool_validation(task_bank_path: str, output_dir: str, task_set_path: Optional[str]=None, task_ids: Optional[Sequence[str]]=None, task_categories: Optional[Sequence[str]]=None, first_per_category: bool=False, max_tasks: Optional[int]=None, width: int=320, height: int=240, cloud_rendering: bool=True, gpu_device: Optional[int]=0, manipulathor_arm: bool=False, manipulathor_visual_attach: bool=True, agent_radius: Optional[float]=None, env_factory: Callable[..., ProcTHORClosedLoopToolEnv]=ProcTHORClosedLoopToolEnv, validated_output_path: Optional[str]=None, rejected_output_path: Optional[str]=None, task_sets_output_dir: Optional[str]=None, task_sets_allow_missing_categories: bool=False, stream: bool=False) -> Tuple[List[dict], dict]:
    task_bank = load_task_bank(str(resolve_repo_path(task_bank_path)))
    if task_set_path:
        task_set = load_task_set(str(resolve_repo_path(task_set_path)), task_bank=task_bank)
        tasks = resolve_task_set(task_bank, task_set)
    else:
        task_set = None
        tasks = _select_tasks(task_bank, task_ids=task_ids, first_per_category=first_per_category)
    if task_categories:
        wanted_categories = set(task_categories)
        tasks = [task for task in tasks if task.get('category') in wanted_categories]
    if max_tasks is not None:
        tasks = tasks[:int(max_tasks)]
    if not tasks:
        raise ClosedLoopToolValidationError('No tasks selected for closed-loop validation.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    env_kwargs = {'width': width, 'height': height, 'cloud_rendering': cloud_rendering, 'gpu_device': gpu_device}
    env_kwargs.update(manipulathor_env_kwargs(manipulathor_arm=manipulathor_arm, manipulathor_visual_attach=manipulathor_visual_attach, agent_radius=agent_radius))
    env = env_factory(**env_kwargs)
    try:
        for index, task in enumerate(tasks, start=1):
            if stream:
                print('[validator] episode {}/{} task={} category={}'.format(index, len(tasks), task['id'], task['category']), flush=True)
            records.append(validate_closed_loop_tool_task(task, env=env, episode_index=index, output_dir=output, stream=stream))
    finally:
        env.close()
    summary = summarize_closed_loop_tool_records(records, task_bank_size=len(task_bank), task_set=task_set)
    _write_jsonl(records, output / 'episode_results.jsonl')
    _write_json(summary, output / 'summary.json')
    if validated_output_path or rejected_output_path or task_sets_output_dir:
        validated_tasks, rejected_tasks = build_tool_validated_task_banks(tasks, records)
        if validated_output_path:
            _write_task_jsonl(validated_tasks, Path(validated_output_path))
            summary['validated_output'] = str(validated_output_path)
            summary['num_tool_validated'] = len(validated_tasks)
        if rejected_output_path:
            _write_task_jsonl(rejected_tasks, Path(rejected_output_path))
            summary['rejected_output'] = str(rejected_output_path)
            summary['num_tool_rejected'] = len(rejected_tasks)
        if task_sets_output_dir:
            task_sets = build_task_sets(validated_tasks, require_all_categories=not task_sets_allow_missing_categories and _has_all_allowed_categories(validated_tasks))
            task_set_summary = write_task_sets(task_sets, str(task_sets_output_dir))
            summary['task_sets_output_dir'] = str(task_sets_output_dir)
            summary['task_sets_summary'] = task_set_summary
        _write_json(summary, output / 'summary.json')
    return (records, summary)

def validate_closed_loop_tool_task(task_spec: Mapping[str, Any], env: ProcTHORClosedLoopToolEnv, episode_index: int=1, output_dir: Optional[Path]=None, stream: bool=False) -> dict:
    from scoper.benchmark.procthor.brains.teachers import get_dispatcher
    from scoper.benchmark.procthor.episode_runner import EpisodeRunner
    from scoper.benchmark.procthor.hooks import ExecutionTimelineHook, HookBus
    validate_online_task_spec(task_spec)
    runner = EpisodeRunner(env)
    hook_list = []
    if output_dir is not None:
        episode_name = 'episode_{:03d}_{}'.format(episode_index, _safe_label(task_spec['id']))
        hook_list.append(ExecutionTimelineHook(output_dir=Path(output_dir), episode_name=episode_name))
    if stream:
        hook_list.append(ConsoleProgressHook(episode_index=episode_index))
    hooks = HookBus(hook_list) if hook_list else None
    result = runner.run(task_spec, brain=get_dispatcher().build(task_spec), hooks=hooks)
    tool_calls = [_json_safe(t) for t in result.transitions]
    contains_teleport_full = any((ll.get('action') == 'TeleportFull' for call in tool_calls for ll in call.get('low_level_actions', [])))
    contains_force_action = any((ll.get('forceAction') for call in tool_calls for ll in call.get('low_level_actions', [])))
    force_navigation_fallback_count = sum((1 for call in tool_calls for ll in call.get('low_level_actions', []) if _is_allowed_force_navigation_fallback(ll)))
    force_open_container_fallback_count = sum((1 for call in tool_calls for ll in call.get('low_level_actions', []) if _is_allowed_force_open_container_fallback(ll)))
    force_arm_open_close_collision_fallback_count = sum((1 for call in tool_calls for ll in call.get('low_level_actions', []) if _is_allowed_force_arm_open_close_collision_fallback(ll)))
    force_put_object_fallback_count = sum((1 for call in tool_calls for ll in call.get('low_level_actions', []) if _is_allowed_force_put_object_fallback(ll)))
    force_pickup_open_container_fallback_count = sum((1 for call in tool_calls for ll in call.get('low_level_actions', []) if _is_allowed_force_pickup_open_container_fallback(ll)))
    contains_forbidden_force_action = any((ll.get('forceAction') and (not _is_allowed_force_fallback(ll)) for call in tool_calls for ll in call.get('low_level_actions', [])))
    timeline_manifest = (result.hook_artifacts.get('execution_timeline') if isinstance(result.hook_artifacts, Mapping) else {}) or {}
    record = {'episode_index': episode_index, 'task_id': task_spec['id'], 'category': task_spec['category'], 'house_id': task_spec['source']['house_id'], 'instruction': task_spec['instruction'], 'status': result.status, 'failure_reason': result.failure_reason, 'failure_type': result.failure_type, 'failure_code': result.failure_code, 'failure_details': _json_safe(result.failure_details), 'tool_calls': tool_calls, 'scheduler_events': _json_safe(result.scheduler_events), 'hook_events': _json_safe(result.hook_events), 'hook_artifacts': _json_safe(result.hook_artifacts), 'init_mutations': _json_safe(result.init_mutations), 'scene_mutations': _json_safe(result.scene_mutations), 'timeline': timeline_manifest.get('timeline'), 'timeline_json': timeline_manifest.get('timeline_json'), 'metrics': _json_safe(result.metrics), 'contains_teleport_full': contains_teleport_full, 'contains_force_action': contains_force_action, 'contains_forbidden_force_action': contains_forbidden_force_action, 'force_navigation_fallback_count': force_navigation_fallback_count, 'force_open_container_fallback_count': force_open_container_fallback_count, 'force_arm_open_close_collision_fallback_count': force_arm_open_close_collision_fallback_count, 'force_put_object_fallback_count': force_put_object_fallback_count, 'force_pickup_open_container_fallback_count': force_pickup_open_container_fallback_count, 'final_goal': result.final_goal, 'env_state_snapshot': {'qa_state': _json_safe(getattr(env, '_human_qa_state', {}))}, 'model_stopped': bool(getattr(env, 'model_stopped', False)), 'episode_elapsed_seconds': result.elapsed_seconds}
    return _record_with_human_interaction_status(task_spec, record)

def _is_allowed_force_navigation_fallback(low_level: Mapping[str, Any]) -> bool:
    return bool(low_level.get('forceAction')) and bool(low_level.get('force_navigation_fallback')) and (low_level.get('action') in ALLOWED_FORCE_NAVIGATION_ACTIONS) and (bool(low_level.get('held_object_id')) or bool(low_level.get('logical_blocking_object_id')))

def _is_allowed_force_open_container_fallback(low_level: Mapping[str, Any]) -> bool:
    return bool(low_level.get('forceAction')) and bool(low_level.get('force_open_container_fallback')) and (low_level.get('action') in ALLOWED_FORCE_OPEN_CONTAINER_ACTIONS) and bool(low_level.get('held_object_id')) and bool(low_level.get('object_id'))

def _is_allowed_force_put_object_fallback(low_level: Mapping[str, Any]) -> bool:
    return bool(low_level.get('forceAction')) and bool(low_level.get('force_put_object_fallback')) and (low_level.get('action') in ALLOWED_FORCE_PUT_OBJECT_ACTIONS) and bool(low_level.get('held_object_id')) and bool(low_level.get('object_id'))

def _is_allowed_force_arm_open_close_collision_fallback(low_level: Mapping[str, Any]) -> bool:
    action = low_level.get('action')
    desired = low_level.get('desired_is_open')
    error = low_level.get('pre_force_error_message')
    normalized_error = error.casefold().replace('-', '_').replace(' ', '_') if isinstance(error, str) else ''
    return bool(low_level.get('forceAction')) and bool(low_level.get('force_arm_open_close_collision_fallback')) and (action in ALLOWED_FORCE_ARM_OPEN_CLOSE_ACTIONS) and bool(low_level.get('object_id')) and (low_level.get('arm_mode') in {'official', 'visual_attach'}) and (type(desired) is bool) and (desired is (action == 'OpenObject')) and any((token in normalized_error for token in ('robot_arm', 'robot_wrist', 'robot_shoulder', 'robot_hand', 'gripper')))

def _is_allowed_force_pickup_open_container_fallback(low_level: Mapping[str, Any]) -> bool:
    return bool(low_level.get('forceAction')) and bool(low_level.get('force_pickup_open_container_fallback')) and (low_level.get('action') in ALLOWED_FORCE_PICKUP_OPEN_CONTAINER_ACTIONS) and bool(low_level.get('object_id')) and bool(low_level.get('parent_container_id'))

def _is_allowed_force_fallback(low_level: Mapping[str, Any]) -> bool:
    return _is_allowed_force_navigation_fallback(low_level) or _is_allowed_force_open_container_fallback(low_level) or _is_allowed_force_arm_open_close_collision_fallback(low_level) or _is_allowed_force_put_object_fallback(low_level) or _is_allowed_force_pickup_open_container_fallback(low_level)

def classify_failure_code(record: Mapping[str, Any]) -> Optional[str]:
    return classify_failure_code_from_record(record)

def build_tool_validated_task_banks(task_specs: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> Tuple[List[dict], List[dict]]:
    records_by_id = {record['task_id']: record for record in records}
    validated: List[dict] = []
    rejected: List[dict] = []
    for task_spec in task_specs:
        task_id = task_spec['id']
        record = records_by_id.get(task_id)
        if record is None:
            raise ClosedLoopToolValidationError("Missing closed-loop tool record for task '{}'.".format(task_id))
        record = _record_with_human_interaction_status(task_spec, record)
        task = _task_with_tool_validation(task_spec, record)
        if record.get('status') == 'passed':
            validated.append(task)
        else:
            rejected.append(task)
    return (sorted(validated, key=lambda task: task['id']), sorted(rejected, key=lambda task: task['id']))

def _record_with_human_interaction_status(task_spec: Mapping[str, Any], record: Mapping[str, Any]) -> dict:
    out = dict(record)
    if task_spec.get('category') != 'human_interactive_execution':
        return out
    if out.get('status') != 'passed':
        return out
    metrics = out.get('metrics') or {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    if int(metrics.get('interaction_success', 0)) == 1:
        return out
    failure_details = dict(out.get('failure_details') or {})
    failure_details['human_metrics'] = _json_safe(metrics)
    out['status'] = 'failed'
    out['failure_type'] = 'HumanInteractionValidationError'
    out['failure_reason'] = 'Human interaction QA/insert success is not 1.0.'
    out['failure_code'] = 'human_interaction_failed'
    out['failure_details'] = failure_details
    return out

class ConsoleProgressHook:
    name = 'console_progress'

    def __init__(self, episode_index: int) -> None:
        self.episode_index = int(episode_index)

    def on_episode_start(self, env, task_spec, reset_observation):
        print('[episode {:03d}] start {} :: {}'.format(self.episode_index, task_spec['id'], task_spec.get('instruction', '')), flush=True)
        return None

    def before_tool(self, env, tool_call, high_level_step):
        print('[episode {:03d} step {:02d}] -> {} {}'.format(self.episode_index, high_level_step, tool_call.tool_name, dict(tool_call.arguments or {})), flush=True)
        return None

    def after_tool(self, env, tool_call, transition, high_level_step):
        feedback = str(transition.get('env_feedback') or '')
        if len(feedback) > 140:
            feedback = feedback[:137] + '...'
        print('[episode {:03d} step {:02d}] <- {} success={} done={} feedback={}'.format(self.episode_index, high_level_step, transition.get('tool_name'), bool(transition.get('success', False)), bool(transition.get('done', False)), feedback), flush=True)
        return None

    def on_scheduler_events(self, env, events, phase, high_level_step):
        for event in events or []:
            payload = dict(event.get('payload') or {})
            compact = {key: payload[key] for key in ('id', 'question', 'instruction', 'answer_set', 'evaluator_answer', 'after_step') if key in payload}
            print('[episode {:03d} step {:02d}] scheduler:{} phase={} payload={}'.format(self.episode_index, high_level_step, event.get('name'), phase, compact), flush=True)
        return None

    def on_episode_end(self, env, final_goal, status):
        print('[episode {:03d}] end status={} task_success={}'.format(self.episode_index, status, final_goal.get('task_success')), flush=True)
        return None

def summarize_closed_loop_tool_records(records: Sequence[Mapping[str, Any]], task_bank_size: Optional[int]=None, task_set: Optional[Mapping[str, Any]]=None) -> dict:
    category_counts = Counter((record['category'] for record in records))
    passed_category_counts = Counter((record['category'] for record in records if record['status'] == 'passed'))
    failed_category_counts = Counter((record['category'] for record in records if record['status'] == 'failed'))
    failure_reasons = Counter((record.get('failure_reason') or 'unknown' for record in records if record['status'] == 'failed'))
    failure_codes = Counter((record.get('failure_code') or 'unknown' for record in records if record['status'] == 'failed'))
    tool_counts = Counter()
    failed_tool_counts = Counter()
    low_level_counts = Counter()
    for record in records:
        for call in record.get('tool_calls', []):
            tool_counts[call.get('tool_name') or 'unknown'] += 1
            if not call.get('success'):
                failed_tool_counts[call.get('tool_name') or 'unknown'] += 1
            for low_level in call.get('low_level_actions', []):
                low_level_counts[low_level.get('action') or 'unknown'] += 1
    return {'task_set': task_set.get('name') if task_set else None, 'task_bank_size': task_bank_size, 'num_tasks': len(records), 'num_passed': sum((1 for record in records if record['status'] == 'passed')), 'num_failed': sum((1 for record in records if record['status'] == 'failed')), 'category_counts': dict(sorted(category_counts.items())), 'passed_category_counts': dict(sorted(passed_category_counts.items())), 'failed_category_counts': dict(sorted(failed_category_counts.items())), 'failure_reasons': dict(sorted(failure_reasons.items())), 'failure_codes': dict(sorted(failure_codes.items())), 'tool_counts': dict(sorted(tool_counts.items())), 'failed_tool_counts': dict(sorted(failed_tool_counts.items())), 'low_level_action_counts': dict(sorted(low_level_counts.items())), 'num_records_with_teleport_full': sum((1 for record in records if record.get('contains_teleport_full'))), 'num_records_with_force_action': sum((1 for record in records if record.get('contains_force_action'))), 'num_records_with_forbidden_force_action': sum((1 for record in records if record.get('contains_forbidden_force_action'))), 'force_navigation_fallback_count': sum((int(record.get('force_navigation_fallback_count', 0)) for record in records)), 'force_open_container_fallback_count': sum((int(record.get('force_open_container_fallback_count', 0)) for record in records)), 'force_arm_open_close_collision_fallback_count': sum((int(record.get('force_arm_open_close_collision_fallback_count', 0)) for record in records)), 'force_put_object_fallback_count': sum((int(record.get('force_put_object_fallback_count', 0)) for record in records)), 'force_pickup_open_container_fallback_count': sum((int(record.get('force_pickup_open_container_fallback_count', 0)) for record in records))}

def _has_all_allowed_categories(task_specs: Sequence[Mapping[str, Any]]) -> bool:
    from scoper.benchmark.procthor.online_task_spec import ALLOWED_CATEGORIES
    present = {task.get('category') for task in task_specs}
    return set(ALLOWED_CATEGORIES).issubset(present)

def main(argv: Optional[Sequence[str]]=None) -> None:
    parser = argparse.ArgumentParser(description='Validate EB-ProcTHOR tasks through the closed-loop tool API.')
    parser.add_argument('--task-bank', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--task-set', default=None)
    parser.add_argument('--task-id', action='append', dest='task_ids', default=None)
    parser.add_argument('--task-categories', nargs='+', default=None)
    parser.add_argument('--first-per-category', action='store_true')
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--no-cloud-rendering', action='store_true')
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--allow-failures', action='store_true')
    parser.add_argument('--manipulathor-arm', action='store_true', help="Use AI2-THOR's official ManipulaTHOR arm agent mode.")
    parser.add_argument('--manipulathor-visual-attach', action=argparse.BooleanOptionalAction, default=True, help='Show the ManipulaTHOR arm and simulate pickup/put-down visually without physical arm pickup.')
    parser.add_argument('--agent-radius', type=float, default=None, help='Optional AI2-THOR SetAgentRadius override in meters. Visual-attach mode defaults to 0.3.')
    parser.add_argument('--validated-output', default=None, help='Write tool-validated tasks to this jsonl path.')
    parser.add_argument('--rejected-output', default=None, help='Write tool-rejected tasks (with failure metadata) to this jsonl path.')
    parser.add_argument('--task-sets-output-dir', default=None, help='Build category-balanced task sets from the validated bank.')
    parser.add_argument('--task-sets-allow-missing-categories', action='store_true', help='Permit task-set construction even when some categories are empty.')
    parser.add_argument('--stream', action='store_true', help='Print a synchronous per-step console trace while validating.')
    args = parser.parse_args(argv)
    _, summary = run_closed_loop_tool_validation(task_bank_path=args.task_bank, task_set_path=args.task_set, output_dir=args.output_dir, task_ids=args.task_ids, task_categories=args.task_categories, first_per_category=args.first_per_category, max_tasks=args.max_tasks, width=args.width, height=args.height, cloud_rendering=not args.no_cloud_rendering, gpu_device=args.gpu_device, manipulathor_arm=args.manipulathor_arm, manipulathor_visual_attach=args.manipulathor_visual_attach, agent_radius=args.agent_radius, validated_output_path=args.validated_output, rejected_output_path=args.rejected_output, task_sets_output_dir=args.task_sets_output_dir, task_sets_allow_missing_categories=args.task_sets_allow_missing_categories, stream=args.stream)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if summary['num_failed'] and (not args.allow_failures):
        raise SystemExit(1)

def _task_with_tool_validation(task_spec: Mapping[str, Any], record: Mapping[str, Any]) -> dict:
    task = copy.deepcopy(dict(task_spec))
    passed = record.get('status') == 'passed'
    checks = dict(task.get('validation', {}).get('checks') or {})
    checks[CLOSED_LOOP_VALIDATION_CHECK_KEY] = _task_tool_check(record)
    task['validation'] = {'status': 'validated' if passed else 'rejected', 'checks': checks, 'failure_reason': None if passed else record.get('failure_reason')}
    tags = set(task.get('tags') or [])
    tags.discard('execution_rejected')
    if passed:
        tags.add('tool_validated')
        tags.discard('tool_rejected')
    else:
        tags.add('tool_rejected')
        tags.discard('tool_validated')
    if record.get('failure_code'):
        tags.add(str(record['failure_code']))
    task['tags'] = sorted(tags)
    validate_online_task_spec(task)
    return task

def _task_tool_check(record: Mapping[str, Any]) -> dict:
    tool_calls = list(record.get('tool_calls') or [])
    tool_call_counts: Counter = Counter()
    for call in tool_calls:
        name = call.get('tool_name')
        if name:
            tool_call_counts[str(name)] += 1
    return {'version': CLOSED_LOOP_VALIDATION_VERSION, 'status': record.get('status'), 'failure_code': record.get('failure_code'), 'failure_reason': record.get('failure_reason'), 'failure_details': _json_safe(record.get('failure_details') or {}), 'final_task_success': _goal_value(record.get('final_goal'), 'task_success'), 'final_predicate_success': _goal_value(record.get('final_goal'), 'predicate_success'), 'high_level_steps': len(tool_calls), 'tool_call_counts': dict(tool_call_counts), 'contains_teleport_full': bool(record.get('contains_teleport_full')), 'contains_force_action': bool(record.get('contains_force_action')), 'contains_forbidden_force_action': bool(record.get('contains_forbidden_force_action')), 'force_navigation_fallback_count': int(record.get('force_navigation_fallback_count', 0)), 'force_open_container_fallback_count': int(record.get('force_open_container_fallback_count', 0)), 'force_arm_open_close_collision_fallback_count': int(record.get('force_arm_open_close_collision_fallback_count', 0)), 'force_put_object_fallback_count': int(record.get('force_put_object_fallback_count', 0)), 'force_pickup_open_container_fallback_count': int(record.get('force_pickup_open_container_fallback_count', 0)), 'model_stopped': bool(record.get('model_stopped', False)), 'init_mutations': _json_safe(record.get('init_mutations') or []), 'scene_mutations': _json_safe(record.get('scene_mutations') or []), 'timeline': record.get('timeline'), 'timeline_json': record.get('timeline_json')}

def _goal_value(goal_result: Any, key: str) -> Optional[float]:
    if isinstance(goal_result, Mapping) and key in goal_result:
        return float(goal_result[key])
    return None

def _select_tasks(task_bank: Mapping[str, Mapping[str, Any]], task_ids: Optional[Sequence[str]], first_per_category: bool) -> List[Mapping[str, Any]]:
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in task_bank]
        if missing:
            raise ClosedLoopToolValidationError('Unknown task ids: {}'.format(', '.join(missing)))
        tasks = [task_bank[task_id] for task_id in task_ids]
    else:
        tasks = list(task_bank.values())
    if first_per_category:
        selected = []
        seen = set()
        for task in sorted(tasks, key=lambda item: (item['category'], item['id'])):
            if task['category'] in seen:
                continue
            seen.add(task['category'])
            selected.append(task)
        tasks = selected
    return tasks

def _safe_label(value: str) -> str:
    label = re.sub('[^A-Za-z0-9_.-]+', '_', str(value).strip())
    return label.strip('_') or 'episode'

def _write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')

def _write_task_jsonl(tasks: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for task in tasks:
            validate_online_task_spec(task)
            json.dump(task, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')

def _write_json(record: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != 'head_rgb'}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
if __name__ == '__main__':
    main()
__all__ = ['CLOSED_LOOP_VALIDATION_CHECK_KEY', 'CLOSED_LOOP_VALIDATION_VERSION', 'ClosedLoopToolValidationError', 'build_tool_validated_task_banks', 'classify_failure_code', 'run_closed_loop_tool_validation', 'summarize_closed_loop_tool_records', 'validate_closed_loop_tool_task']
