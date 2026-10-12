from __future__ import annotations
import argparse
import copy
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from scoper.benchmark.procthor.online_env import ProcTHOROnlineEnv
from scoper.benchmark.procthor.online_task_loader import load_task_bank, load_task_set, resolve_task_set
from scoper.benchmark.procthor.online_task_spec import validate_online_task_spec
from scoper.benchmark.procthor.paths import resolve_repo_path
from scoper.benchmark.procthor.split_builder import build_task_sets, write_task_sets

class ScriptedOracleValidationError(ValueError):
    pass
EXECUTION_VALIDATION_VERSION = '0.1'
EXECUTION_VALIDATION_CHECK_KEY = 'scripted_oracle_execution'

def build_scripted_plan(task_spec: Mapping[str, Any]) -> List[dict]:
    validate_online_task_spec(task_spec)
    plan: List[dict] = []
    opened_roles = set()
    for predicate in task_spec['goal']['predicates']:
        predicate_type = predicate.get('type')
        if predicate_type == 'object_in_receptacle':
            object_role = _required_role_name(predicate, 'object_role')
            receptacle_role = _required_role_name(predicate, 'receptacle_role')
            source_role = _source_container_role(task_spec, object_role)
            if source_role and source_role not in opened_roles:
                plan.append(_ai2thor_step(action='OpenObject', role=source_role, object_id=_role_object_id(task_spec, source_role), label='open {}'.format(source_role), purpose='release_source_container'))
                opened_roles.add(source_role)
            plan.append(_conditional_open_step(role=receptacle_role, object_id=_role_object_id(task_spec, receptacle_role), label='ensure {} open'.format(receptacle_role), purpose='allow_put_into_openable_receptacle'))
            plan.append(_ai2thor_step(action='PickupObject', role=object_role, object_id=_role_object_id(task_spec, object_role), label='pickup {}'.format(object_role), purpose='hold_object_before_place'))
            plan.append(_ai2thor_step(action='PutObject', role=receptacle_role, object_id=_role_object_id(task_spec, receptacle_role), label='put {} into {}'.format(object_role, receptacle_role), purpose='satisfy_object_in_receptacle', uses_inventory_role=object_role))
        elif predicate_type in {'object_toggled_on', 'object_toggled_off'}:
            object_role = _required_role_name(predicate, 'object_role')
            action = 'ToggleObjectOn' if predicate_type == 'object_toggled_on' else 'ToggleObjectOff'
            plan.append(_ai2thor_step(action=action, role=object_role, object_id=_role_object_id(task_spec, object_role), label='{} {}'.format(action, object_role), purpose='satisfy_{}'.format(predicate_type)))
        elif predicate_type in {'object_open', 'object_closed'}:
            object_role = _required_role_name(predicate, 'object_role')
            action = 'OpenObject' if predicate_type == 'object_open' else 'CloseObject'
            plan.append(_ai2thor_step(action=action, role=object_role, object_id=_role_object_id(task_spec, object_role), label='{} {}'.format(action, object_role), purpose='satisfy_{}'.format(predicate_type)))
        elif predicate_type == 'object_not_in_receptacle':
            plan.append({'kind': 'noop', 'label': 'verify object_not_in_receptacle', 'purpose': 'predicate_is_expected_to_hold_without_action', 'predicate': dict(predicate)})
        else:
            raise ScriptedOracleValidationError('Unsupported predicate for scripted oracle: {}'.format(predicate_type))
    if bool(task_spec['goal'].get('require_model_stop', False)):
        plan.append({'kind': 'model', 'action': 'Stop', 'label': 'stop', 'purpose': 'satisfy_require_model_stop'})
    if not plan:
        raise ScriptedOracleValidationError('Scripted oracle produced an empty plan.')
    return plan

def validate_scripted_oracle_task(task_spec: Mapping[str, Any], env: Any, episode_index: int=1, capture_dir: Optional[Path]=None) -> dict:
    validate_online_task_spec(task_spec)
    start_time = time.time()
    record = _base_record(task_spec, episode_index)
    snapshots: List[dict] = []
    action_results: List[dict] = []
    model_stopped = False
    try:
        obs = env.reset_to_task(task_spec)
        record['reset_success'] = True
        record['initial_observation_available'] = bool(obs.get('head_rgb') is not None)
        initial_goal = env.check_goal(model_stopped=False)
        record['goal_checker_runs'] = True
        record['initial_goal'] = initial_goal
        if float(initial_goal.get('predicate_success', 0.0)) != 0.0:
            raise ScriptedOracleValidationError('Goal predicates are already true at reset.')
        snapshots.append(_capture_snapshot(env=env, task_spec=task_spec, capture_dir=capture_dir, episode_index=episode_index, step_index=0, label='reset', goal_result=initial_goal))
        plan = build_scripted_plan(task_spec)
        record['plan'] = plan
        for step_index, step in enumerate(plan, start=1):
            action_result, model_stopped = _execute_plan_step(env=env, task_spec=task_spec, step=step, step_index=step_index, model_stopped=model_stopped)
            action_results.append(action_result)
            snapshots.append(_capture_snapshot(env=env, task_spec=task_spec, capture_dir=capture_dir, episode_index=episode_index, step_index=step_index, label=_safe_label(step.get('label') or step.get('action') or 'step'), goal_result=action_result.get('goal_after')))
            if not action_result['success']:
                raise ScriptedOracleValidationError('Scripted action {} failed: {}'.format(action_result['label'], action_result.get('error_message') or 'unknown error'))
        final_goal = env.check_goal(model_stopped=model_stopped)
        record['actions'] = action_results
        record['final_goal'] = final_goal
        record['model_stopped'] = model_stopped
        record['final_task_success'] = final_goal['task_success']
        record['final_predicate_success'] = final_goal['predicate_success']
        if float(final_goal.get('task_success', 0.0)) != 1.0:
            raise ScriptedOracleValidationError('Scripted plan finished but task_success is not 1.0.')
        record['status'] = 'passed'
    except Exception as exc:
        record['status'] = 'failed'
        record['failure_type'] = type(exc).__name__
        record['failure_reason'] = str(exc)
    finally:
        record['actions'] = action_results
        record['failure_code'] = classify_failure_code(record)
        record['snapshots'] = [snapshot for snapshot in snapshots if snapshot]
        record['episode_elapsed_seconds'] = time.time() - start_time
    return record

def run_scripted_oracle_validation(task_bank_path: str, output_dir: str, task_set_path: Optional[str]=None, task_ids: Optional[Sequence[str]]=None, categories: Optional[Sequence[str]]=None, first_per_category: bool=False, max_tasks: Optional[int]=None, width: int=320, height: int=240, cloud_rendering: bool=True, gpu_device: Optional[int]=0, capture_images: bool=False, validated_output_path: Optional[str]=None, rejected_output_path: Optional[str]=None, task_sets_output_dir: Optional[str]=None, task_sets_allow_missing_categories: bool=False, env_factory: Callable[..., Any]=ProcTHOROnlineEnv) -> Tuple[List[dict], dict]:
    task_bank = load_task_bank(str(resolve_repo_path(task_bank_path)))
    tasks = _select_tasks(task_bank=task_bank, task_set_path=task_set_path, task_ids=task_ids, categories=categories, first_per_category=first_per_category, max_tasks=max_tasks)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture_dir = output / 'images' if capture_images else None
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    env = env_factory(width=width, height=height, cloud_rendering=cloud_rendering, gpu_device=gpu_device)
    try:
        for episode_index, task in enumerate(tasks, start=1):
            records.append(validate_scripted_oracle_task(task, env=env, episode_index=episode_index, capture_dir=capture_dir))
    finally:
        env.close()
    summary = summarize_scripted_oracle(records, task_bank_size=len(task_bank))
    _write_jsonl(records, output / 'episode_results.jsonl')
    _write_json(summary, output / 'summary.json')
    if validated_output_path or rejected_output_path:
        validated_tasks, rejected_tasks = build_execution_task_banks(tasks, records)
        if validated_output_path:
            _write_task_jsonl(validated_tasks, Path(validated_output_path))
            summary['validated_output'] = str(validated_output_path)
            summary['num_execution_validated'] = len(validated_tasks)
        if rejected_output_path:
            _write_task_jsonl(rejected_tasks, Path(rejected_output_path))
            summary['rejected_output'] = str(rejected_output_path)
            summary['num_execution_rejected'] = len(rejected_tasks)
        if task_sets_output_dir:
            task_sets = build_task_sets(validated_tasks, require_all_categories=not task_sets_allow_missing_categories)
            task_set_summary = write_task_sets(task_sets, task_sets_output_dir)
            summary['task_sets_output_dir'] = str(task_sets_output_dir)
            summary['task_sets_summary'] = task_set_summary
        _write_json(summary, output / 'summary.json')
    if capture_images:
        manifest = _write_visual_outputs(records, output)
        summary['visual_manifest'] = str(Path('visual_manifest.json'))
        summary['overview_contact_sheet'] = manifest.get('overview_contact_sheet')
        _write_json(summary, output / 'summary.json')
    return (records, summary)

def classify_failure_code(record: Mapping[str, Any]) -> Optional[str]:
    if record.get('status') == 'passed':
        return None
    failure_reason = str(record.get('failure_reason') or '')
    failure_type = str(record.get('failure_type') or '')
    actions = list(record.get('actions') or [])
    failed_actions = [action for action in actions if isinstance(action, Mapping) and (not action.get('success', False))]
    failed_action_names = {str(action.get('action') or action.get('kind') or '') for action in failed_actions}
    if 'No valid positions to place object found' in failure_reason:
        return 'no_valid_put_position'
    if 'Goal predicates are already true at reset' in failure_reason:
        return 'goal_true_at_reset'
    if 'Unsupported predicate' in failure_reason:
        return 'unsupported_predicate'
    if 'reset_to_task' in failure_reason or not record.get('reset_success'):
        return 'reset_failed'
    if failed_actions:
        if 'PutObject' in failed_action_names:
            return 'put_object_failed'
        if 'PickupObject' in failed_action_names:
            return 'pickup_object_failed'
        if 'OpenObject' in failed_action_names:
            return 'open_object_failed'
        if 'ToggleObjectOn' in failed_action_names or 'ToggleObjectOff' in failed_action_names:
            return 'toggle_object_failed'
        return 'scripted_action_failed'
    if 'task_success is not 1.0' in failure_reason:
        return 'final_goal_not_success'
    if failure_type:
        return 'scripted_oracle_error'
    return 'unknown_failure'

def build_execution_task_banks(task_specs: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> Tuple[List[dict], List[dict]]:
    records_by_id = {record['task_id']: record for record in records}
    validated = []
    rejected = []
    for task_spec in task_specs:
        task_id = task_spec['id']
        record = records_by_id.get(task_id)
        if record is None:
            raise ScriptedOracleValidationError("Missing scripted oracle record for task '{}'.".format(task_id))
        task = _task_with_execution_validation(task_spec, record)
        if record.get('status') == 'passed':
            validated.append(task)
        else:
            rejected.append(task)
    return (sorted(validated, key=lambda task: task['id']), sorted(rejected, key=lambda task: task['id']))

def summarize_scripted_oracle(records: Sequence[Mapping[str, Any]], task_bank_size: Optional[int]=None) -> dict:
    category_counts = Counter((record['category'] for record in records))
    passed_category_counts = Counter((record['category'] for record in records if record['status'] == 'passed'))
    failed_category_counts = Counter((record['category'] for record in records if record['status'] == 'failed'))
    failure_reasons = Counter((record.get('failure_reason') or 'unknown' for record in records if record['status'] == 'failed'))
    action_counts = Counter()
    action_failure_counts = Counter()
    failure_code_counts = Counter()
    require_model_stop_count = 0
    for record in records:
        if record.get('require_model_stop'):
            require_model_stop_count += 1
        if record['status'] == 'failed':
            failure_code_counts[record.get('failure_code') or 'unknown_failure'] += 1
        for action in record.get('actions', []):
            action_counts[action.get('action') or action.get('kind') or 'unknown'] += 1
            if not action.get('success', False):
                action_failure_counts[action.get('action') or action.get('kind') or 'unknown'] += 1
    return {'task_bank_size': task_bank_size, 'num_tasks': len(records), 'num_passed': sum((1 for record in records if record['status'] == 'passed')), 'num_failed': sum((1 for record in records if record['status'] == 'failed')), 'num_reset_success': sum((1 for record in records if record.get('reset_success'))), 'num_goal_checker_runs': sum((1 for record in records if record.get('goal_checker_runs'))), 'num_require_model_stop': require_model_stop_count, 'category_counts': dict(sorted(category_counts.items())), 'passed_category_counts': dict(sorted(passed_category_counts.items())), 'failed_category_counts': dict(sorted(failed_category_counts.items())), 'action_counts': dict(sorted(action_counts.items())), 'action_failure_counts': dict(sorted(action_failure_counts.items())), 'failure_code_counts': dict(sorted(failure_code_counts.items())), 'failure_reasons': dict(sorted(failure_reasons.items()))}

def main(argv: Optional[Sequence[str]]=None) -> None:
    parser = argparse.ArgumentParser(description='Run scripted oracle validation for EB-ProcTHOR tasks.')
    parser.add_argument('--task-bank', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--task-set', default=None)
    parser.add_argument('--task-id', action='append', dest='task_ids', default=None)
    parser.add_argument('--category', action='append', dest='categories', default=None)
    parser.add_argument('--first-per-category', action='store_true')
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--no-cloud-rendering', action='store_true')
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--capture-images', action='store_true')
    parser.add_argument('--validated-output', default=None, help='Optional JSONL path for execution-validated task specs.')
    parser.add_argument('--rejected-output', default=None, help='Optional JSONL path for execution-rejected task specs.')
    parser.add_argument('--task-sets-output-dir', default=None, help='Optional directory for task sets built from execution-validated tasks.')
    parser.add_argument('--task-sets-allow-missing-categories', action='store_true', help='Allow task set construction if the selected execution-valid bank misses a category.')
    parser.add_argument('--allow-failures', action='store_true', help='Write reports and exit 0 even when one or more tasks fail.')
    args = parser.parse_args(argv)
    _, summary = run_scripted_oracle_validation(task_bank_path=args.task_bank, task_set_path=args.task_set, output_dir=args.output_dir, task_ids=args.task_ids, categories=args.categories, first_per_category=args.first_per_category, max_tasks=args.max_tasks, width=args.width, height=args.height, cloud_rendering=not args.no_cloud_rendering, gpu_device=args.gpu_device, capture_images=args.capture_images, validated_output_path=args.validated_output, rejected_output_path=args.rejected_output, task_sets_output_dir=args.task_sets_output_dir, task_sets_allow_missing_categories=args.task_sets_allow_missing_categories)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if summary['num_failed'] and (not args.allow_failures):
        raise SystemExit(1)

def _select_tasks(task_bank: Mapping[str, Mapping[str, Any]], task_set_path: Optional[str], task_ids: Optional[Sequence[str]], categories: Optional[Sequence[str]], first_per_category: bool, max_tasks: Optional[int]) -> List[Mapping[str, Any]]:
    if task_set_path:
        task_set = load_task_set(str(resolve_repo_path(task_set_path)), task_bank=task_bank)
        tasks = resolve_task_set(task_bank, task_set)
    else:
        tasks = list(task_bank.values())
    if task_ids:
        requested = list(task_ids)
        missing = [task_id for task_id in requested if task_id not in task_bank]
        if missing:
            raise ScriptedOracleValidationError('Unknown task ids: {}'.format(', '.join(missing)))
        selected_ids = set(requested)
        tasks = [task for task in tasks if task['id'] in selected_ids]
    if categories:
        selected_categories = set(categories)
        tasks = [task for task in tasks if task['category'] in selected_categories]
    if first_per_category:
        seen = set()
        filtered = []
        for task in sorted(tasks, key=lambda item: (item['category'], item['id'])):
            if task['category'] in seen:
                continue
            seen.add(task['category'])
            filtered.append(task)
        tasks = filtered
    if max_tasks is not None:
        tasks = tasks[:int(max_tasks)]
    if not tasks:
        raise ScriptedOracleValidationError('No tasks selected for scripted validation.')
    return tasks

def _execute_plan_step(env: Any, task_spec: Mapping[str, Any], step: Mapping[str, Any], step_index: int, model_stopped: bool) -> Tuple[dict, bool]:
    kind = step.get('kind')
    if kind == 'model':
        model_stopped = step.get('action') == 'Stop' or model_stopped
        goal_after = env.check_goal(model_stopped=model_stopped)
        return ({'step_index': step_index, 'kind': kind, 'action': step.get('action'), 'label': step.get('label'), 'success': True, 'error_message': '', 'goal_after': goal_after}, model_stopped)
    if kind == 'noop':
        goal_after = env.check_goal(model_stopped=model_stopped)
        return ({'step_index': step_index, 'kind': kind, 'action': 'NoOp', 'label': step.get('label'), 'success': True, 'error_message': '', 'goal_after': goal_after}, model_stopped)
    if kind == 'conditional_open':
        object_id = step['object_id']
        obj = _metadata_object(env, object_id)
        if bool(obj.get('openable', False)) and obj.get('isOpen') is False:
            event = _step_raw(env, action='OpenObject', objectId=object_id, forceAction=True)
            success = bool(event.metadata.get('lastActionSuccess', True))
            error_message = event.metadata.get('errorMessage') or ''
            executed_action = 'OpenObject'
        else:
            success = True
            error_message = ''
            executed_action = 'EnsureOpenObject'
        goal_after = env.check_goal(model_stopped=model_stopped)
        return ({'step_index': step_index, 'kind': kind, 'action': executed_action, 'label': step.get('label'), 'role': step.get('role'), 'object_id': object_id, 'success': success, 'error_message': error_message, 'goal_after': goal_after}, model_stopped)
    if kind != 'ai2thor':
        raise ScriptedOracleValidationError('Unknown plan step kind: {}'.format(kind))
    kwargs = dict(step['kwargs'])
    event = _step_raw(env, **kwargs)
    success = bool(event.metadata.get('lastActionSuccess', True))
    error_message = event.metadata.get('errorMessage') or ''
    goal_after = env.check_goal(model_stopped=model_stopped)
    return ({'step_index': step_index, 'kind': kind, 'action': step.get('action'), 'label': step.get('label'), 'role': step.get('role'), 'object_id': step.get('object_id'), 'success': success, 'error_message': error_message, 'goal_after': goal_after}, model_stopped)

def _step_raw(env: Any, **kwargs: Any) -> Any:
    if getattr(env, 'controller', None) is None:
        raise ScriptedOracleValidationError('Environment controller is not initialized.')
    return env.controller.step(**kwargs)

def _ai2thor_step(action: str, role: str, object_id: str, label: str, purpose: str, uses_inventory_role: Optional[str]=None) -> dict:
    step = {'kind': 'ai2thor', 'action': action, 'role': role, 'object_id': object_id, 'label': label, 'purpose': purpose, 'kwargs': {'action': action, 'objectId': object_id, 'forceAction': True}}
    if uses_inventory_role:
        step['uses_inventory_role'] = uses_inventory_role
    return step

def _conditional_open_step(role: str, object_id: str, label: str, purpose: str) -> dict:
    return {'kind': 'conditional_open', 'action': 'EnsureOpenObject', 'role': role, 'object_id': object_id, 'label': label, 'purpose': purpose}

def _metadata_object(env: Any, object_id: str) -> Mapping[str, Any]:
    controller = getattr(env, 'controller', None)
    event = getattr(controller, 'last_event', None)
    metadata = getattr(event, 'metadata', None)
    if metadata is None:
        metadata = getattr(controller, 'metadata', None)
    if not isinstance(metadata, Mapping):
        raise ScriptedOracleValidationError('Environment metadata is not available.')
    for obj in metadata.get('objects', []):
        if isinstance(obj, Mapping) and obj.get('objectId') == object_id:
            return obj
    raise ScriptedOracleValidationError("Object '{}' not found in environment metadata.".format(object_id))

def _source_container_role(task_spec: Mapping[str, Any], object_role: str) -> Optional[str]:
    if object_role != 'target_object':
        return None
    role = task_spec.get('roles', {}).get('source_container')
    if role is None:
        return None
    return 'source_container'

def _required_role_name(predicate: Mapping[str, Any], field: str) -> str:
    role_name = predicate.get(field)
    if not isinstance(role_name, str) or not role_name:
        raise ScriptedOracleValidationError("Predicate '{}' missing role field '{}'.".format(predicate.get('type'), field))
    return role_name

def _role_object_id(task_spec: Mapping[str, Any], role_name: str) -> str:
    role = task_spec['roles'].get(role_name)
    if role is None:
        raise ScriptedOracleValidationError("Unknown role '{}'.".format(role_name))
    return role['object_id']

def _base_record(task_spec: Mapping[str, Any], episode_index: int) -> dict:
    return {'episode_index': episode_index, 'task_id': task_spec['id'], 'category': task_spec['category'], 'house_id': task_spec['source']['house_id'], 'instruction': task_spec['instruction'], 'roles': dict(task_spec['roles']), 'goal': dict(task_spec['goal']), 'require_model_stop': bool(task_spec['goal'].get('require_model_stop', False)), 'reset_success': False, 'initial_observation_available': False, 'goal_checker_runs': False, 'status': 'failed', 'failure_reason': None, 'failure_code': None, 'actions': [], 'snapshots': []}

def _task_with_execution_validation(task_spec: Mapping[str, Any], record: Mapping[str, Any]) -> dict:
    task = copy.deepcopy(dict(task_spec))
    passed = record.get('status') == 'passed'
    checks = dict(task.get('validation', {}).get('checks') or {})
    checks[EXECUTION_VALIDATION_CHECK_KEY] = _task_execution_check(record)
    task['validation'] = {'status': 'validated' if passed else 'rejected', 'checks': checks, 'failure_reason': None if passed else record.get('failure_reason')}
    tags = set(task.get('tags') or [])
    tags.add('execution_validated' if passed else 'execution_rejected')
    if record.get('failure_code'):
        tags.add(str(record['failure_code']))
    task['tags'] = sorted(tags)
    validate_online_task_spec(task)
    return task

def _task_execution_check(record: Mapping[str, Any]) -> dict:
    action_names = [action.get('action') or action.get('kind') or 'unknown' for action in record.get('actions', [])]
    return {'version': EXECUTION_VALIDATION_VERSION, 'status': record.get('status'), 'failure_code': record.get('failure_code'), 'failure_reason': record.get('failure_reason'), 'initial_task_success': _goal_value(record.get('initial_goal'), 'task_success'), 'final_task_success': _goal_value(record.get('final_goal'), 'task_success'), 'final_predicate_success': _goal_value(record.get('final_goal'), 'predicate_success'), 'num_actions': len(action_names), 'actions': action_names, 'model_stopped': bool(record.get('model_stopped', False))}

def _goal_value(goal_result: Any, key: str) -> Optional[float]:
    if isinstance(goal_result, Mapping) and key in goal_result:
        return float(goal_result[key])
    return None

def _capture_snapshot(env: Any, task_spec: Mapping[str, Any], capture_dir: Optional[Path], episode_index: int, step_index: int, label: str, goal_result: Optional[Mapping[str, Any]]) -> Optional[dict]:
    if capture_dir is None:
        return None
    controller = getattr(env, 'controller', None)
    event = getattr(controller, 'last_event', None)
    frame = getattr(event, 'frame', None)
    if frame is None:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ScriptedOracleValidationError('Pillow is required for image capture.') from exc
    image = Image.fromarray(frame).convert('RGB')
    label_height = 46
    canvas = Image.new('RGB', (image.width, image.height + label_height), (245, 246, 248))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    progress = None
    success = None
    if isinstance(goal_result, Mapping):
        progress = goal_result.get('task_progress')
        success = goal_result.get('task_success')
    draw.text((8, 5), '{} | {} | {}'.format(task_spec['category'], task_spec['id'], label), fill=(18, 22, 28), font=font)
    draw.text((8, 25), 'progress={} success={}'.format(progress, success), fill=(80, 88, 96), font=small)
    filename = '{:03d}_{}_step{:02d}_{}.png'.format(episode_index, _safe_label(task_spec['id']), step_index, _safe_label(label))
    path = capture_dir / filename
    canvas.save(path)
    return {'step_index': step_index, 'label': label, 'path': str(path)}

def _write_visual_outputs(records: Sequence[Mapping[str, Any]], output_dir: Path) -> dict:
    manifest = {'records': [{'task_id': record['task_id'], 'category': record['category'], 'status': record['status'], 'snapshots': record.get('snapshots', [])} for record in records]}
    overview = _write_contact_sheet(records, output_dir)
    if overview:
        manifest['overview_contact_sheet'] = overview
    _write_visual_report(records, output_dir, overview)
    _write_json(manifest, output_dir / 'visual_manifest.json')
    return manifest

def _write_contact_sheet(records: Sequence[Mapping[str, Any]], output_dir: Path) -> Optional[str]:
    rows = [record for record in records if record.get('snapshots')]
    if not rows:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ScriptedOracleValidationError('Pillow is required for contact sheets.') from exc
    thumb_w, thumb_h = (220, 165)
    label_h = 50
    margin = 12
    gap = 8
    max_cols = min(max((len(row.get('snapshots', [])) for row in rows)), 5)
    width = margin * 2 + max_cols * thumb_w + (max_cols - 1) * gap
    height = margin * 2 + len(rows) * (thumb_h + label_h) + (len(rows) - 1) * gap
    sheet = Image.new('RGB', (width, height), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    for row_index, record in enumerate(rows):
        y = margin + row_index * (thumb_h + label_h + gap)
        title = '{} | {} | {}'.format(record['category'], record['task_id'], record['status'])
        draw.text((margin, y), title, fill=(20, 24, 28), font=font)
        draw.text((margin, y + 18), record['instruction'][:110], fill=(80, 88, 96), font=small)
        snapshots = record.get('snapshots', [])[:max_cols]
        for col_index, snapshot in enumerate(snapshots):
            x = margin + col_index * (thumb_w + gap)
            image = Image.open(snapshot['path']).convert('RGB')
            image.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            tile = Image.new('RGB', (thumb_w, thumb_h), (232, 235, 238))
            tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
            sheet.paste(tile, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + thumb_w - 1, y + label_h + thumb_h - 1), outline=(96, 104, 112), width=1)
    overview_path = output_dir / 'overview_contact_sheet.png'
    sheet.save(overview_path)
    return str(overview_path)

def _write_visual_report(records: Sequence[Mapping[str, Any]], output_dir: Path, overview_path: Optional[str]) -> None:
    lines = ['# EB-ProcTHOR Scripted Oracle Visual Report', '', 'This report is generated by `scripted_oracle_validator.py`.', '']
    if overview_path:
        lines.extend(['## Overview', '', '![overview](overview_contact_sheet.png)', ''])
    lines.append('## Episodes')
    lines.append('')
    for record in records:
        lines.append('### {} / {} / {}'.format(record['category'], record['task_id'], record['status']))
        lines.append('')
        lines.append(record['instruction'])
        lines.append('')
        for snapshot in record.get('snapshots', []):
            rel = Path(snapshot['path']).relative_to(output_dir)
            lines.append('`{}`'.format(snapshot['label']))
            lines.append('')
            lines.append('![{}]({})'.format(snapshot['label'], rel.as_posix()))
            lines.append('')
    (output_dir / 'visual_report.md').write_text('\n'.join(lines), encoding='utf-8')

def _write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> None:
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
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')

def _safe_label(value: str) -> str:
    label = re.sub('[^A-Za-z0-9_.-]+', '_', value.strip())
    return label.strip('_') or 'item'
if __name__ == '__main__':
    main()
__all__ = ['ScriptedOracleValidationError', 'build_scripted_plan', 'build_execution_task_banks', 'classify_failure_code', 'run_scripted_oracle_validation', 'summarize_scripted_oracle', 'validate_scripted_oracle_task']
