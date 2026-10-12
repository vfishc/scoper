from __future__ import annotations
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from scoper.benchmark.procthor.online_env import ProcTHOROnlineEnv, ProcTHOROnlineEnvError
from scoper.benchmark.procthor.online_task_loader import load_task_bank
from scoper.benchmark.procthor.online_task_spec import validate_online_task_spec
from scoper.benchmark.procthor.predicate_goal_checker import validate_goal_predicates
from scoper.benchmark.procthor.task_generator import load_scene_profiles

class TaskValidationError(ValueError):
    pass

def validate_tasks(task_specs: Sequence[Mapping[str, Any]], scene_profiles: Sequence[Mapping[str, Any]], run_reset: bool=True, width: int=320, height: int=240, cloud_rendering: bool=False, gpu_device: Optional[int]=None) -> Tuple[List[dict], List[dict], dict]:
    profiles_by_house = {profile['house_id']: profile for profile in scene_profiles}
    validated = []
    rejected = []
    env = ProcTHOROnlineEnv(width=width, height=height, cloud_rendering=cloud_rendering, gpu_device=gpu_device) if run_reset else None
    try:
        for task_spec in task_specs:
            task_result = validate_one_task(task_spec, profiles_by_house=profiles_by_house, env=env, run_reset=run_reset)
            if task_result['validation']['status'] == 'validated':
                validated.append(task_result)
            else:
                rejected.append(task_result)
    finally:
        if env is not None:
            env.close()
    return (validated, rejected, summarize_validation(validated, rejected))

def validate_one_task(task_spec: Mapping[str, Any], profiles_by_house: Mapping[str, Mapping[str, Any]], env: Optional[ProcTHOROnlineEnv]=None, run_reset: bool=True) -> dict:
    task = copy.deepcopy(dict(task_spec))
    checks: Dict[str, Any] = {}
    try:
        validate_online_task_spec(task)
        checks['schema_valid'] = True
        _validate_against_profile(task, profiles_by_house, checks)
        validate_goal_predicates(task)
        checks['goal_predicates_valid'] = True
        if run_reset:
            if env is None:
                raise TaskValidationError('run_reset requires a ProcTHOROnlineEnv.')
            obs = env.reset_to_task(task)
            checks['reset_success'] = True
            checks['initial_observation_available'] = bool(obs.get('head_rgb') is not None)
            goal_result = env.check_goal()
            checks['goal_checker_runs'] = True
            checks['initial_goal_result'] = goal_result
            if goal_result['predicate_success']:
                raise TaskValidationError('Goal is already satisfied at reset.')
            checks['goal_false_at_start'] = True
        _set_validation(task, 'validated', checks, None)
    except Exception as exc:
        checks.setdefault('failure_type', type(exc).__name__)
        _set_validation(task, 'rejected', checks, str(exc))
    return task

def summarize_validation(validated: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> dict:
    rejection_reasons = Counter((task['validation'].get('failure_reason') or 'unknown' for task in rejected))
    category_counts = Counter((task['category'] for task in validated))
    rejected_category_counts = Counter((task['category'] for task in rejected))
    init_override_counts = Counter()
    require_model_stop_count = 0
    initial_task_success_counts = Counter()
    initial_task_progress_counts = Counter()
    for task in validated:
        for override in task.get('init', {}).get('object_state_overrides', []):
            init_override_counts[override.get('type') or 'unknown'] += 1
        if bool(task.get('goal', {}).get('require_model_stop', False)):
            require_model_stop_count += 1
        initial_goal = task.get('validation', {}).get('checks', {}).get('initial_goal_result')
        if isinstance(initial_goal, Mapping):
            initial_task_success_counts[str(initial_goal.get('task_success'))] += 1
            initial_task_progress_counts[str(initial_goal.get('task_progress'))] += 1
    return {'num_validated': len(validated), 'num_rejected': len(rejected), 'validated_category_counts': dict(sorted(category_counts.items())), 'rejected_category_counts': dict(sorted(rejected_category_counts.items())), 'validated_init_override_counts': dict(sorted(init_override_counts.items())), 'validated_require_model_stop_count': require_model_stop_count, 'initial_task_success_counts': dict(sorted(initial_task_success_counts.items())), 'initial_task_progress_counts': dict(sorted(initial_task_progress_counts.items())), 'rejection_reasons': dict(sorted(rejection_reasons.items()))}

def write_task_jsonl(tasks: Iterable[Mapping[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        for task in tasks:
            validate_online_task_spec(task)
            json.dump(task, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')

def main(argv: Optional[Sequence[str]]=None) -> None:
    parser = argparse.ArgumentParser(description='Validate ProcTHOR online tasks.')
    parser.add_argument('--task-bank', required=True)
    parser.add_argument('--profiles', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rejected-output', required=True)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--cloud-rendering', action='store_true')
    parser.add_argument('--gpu-device', type=int, default=None, help='Optional CUDA GPU index to pass to AI2-THOR. With CloudRendering, AI2-THOR uses vulkaninfo to map this to a Vulkan device index.')
    parser.add_argument('--no-reset', action='store_true', help='Run schema/profile/predicate validation without AI2-THOR reset.')
    args = parser.parse_args(argv)
    task_specs = list(load_task_bank(args.task_bank).values())
    profiles = load_scene_profiles(args.profiles)
    validated, rejected, summary = validate_tasks(task_specs, profiles, run_reset=not args.no_reset, width=args.width, height=args.height, cloud_rendering=args.cloud_rendering, gpu_device=args.gpu_device)
    write_task_jsonl(validated, args.output)
    write_task_jsonl(rejected, args.rejected_output)
    summary['task_bank'] = args.task_bank
    summary['output'] = args.output
    summary['rejected_output'] = args.rejected_output
    summary_path = str(Path(args.output).with_suffix(Path(args.output).suffix + '.summary.json'))
    Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if rejected:
        raise SystemExit(1)

def _validate_against_profile(task_spec: Mapping[str, Any], profiles_by_house: Mapping[str, Mapping[str, Any]], checks: Dict[str, Any]) -> None:
    profile = profiles_by_house.get(task_spec['source']['house_id'])
    if profile is None:
        raise TaskValidationError("Missing SceneProfile for house_id '{}'.".format(task_spec['source']['house_id']))
    checks['profile_available'] = True
    if profile.get('reachable', {}).get('num_positions', 0) <= 0:
        raise TaskValidationError('SceneProfile has no reachable positions.')
    checks['reachable_positions_available'] = True
    profile_objects = {obj['object_id']: obj for obj in profile.get('objects', []) if isinstance(obj.get('object_id'), str)}
    for role_name, role in task_spec['roles'].items():
        profile_obj = profile_objects.get(role['object_id'])
        if profile_obj is None:
            raise TaskValidationError("Role '{}' object id missing in SceneProfile: {}.".format(role_name, role['object_id']))
        if profile_obj.get('object_type') != role['object_type']:
            raise TaskValidationError("Role '{}' object type mismatch: task={}, profile={}.".format(role_name, role['object_type'], profile_obj.get('object_type')))
    checks['roles_exist_in_profile'] = True
    checks['role_object_types_match_profile'] = True
    _validate_init_overrides_against_profile(task_spec, profile_objects)
    checks['init_overrides_valid'] = True

def _validate_init_overrides_against_profile(task_spec: Mapping[str, Any], profile_objects: Mapping[str, Mapping[str, Any]]) -> None:
    for index, override in enumerate(task_spec['init'].get('object_state_overrides', [])):
        object_role = override.get('object_role')
        role = task_spec['roles'].get(object_role)
        if role is None:
            raise TaskValidationError("Init override {} references unknown role '{}'.".format(index, object_role))
        profile_obj = profile_objects.get(role['object_id'])
        if profile_obj is None:
            raise TaskValidationError('Init override {} object missing in SceneProfile: {}.'.format(index, role['object_id']))
        override_type = override.get('type')
        if override_type == 'set_open_state':
            if not bool(profile_obj.get('openable', False)):
                raise TaskValidationError("Init override {} targets non-openable object '{}'.".format(index, role['object_id']))
        elif override_type == 'set_toggle_state':
            if not bool(profile_obj.get('toggleable', False)):
                raise TaskValidationError("Init override {} targets non-toggleable object '{}'.".format(index, role['object_id']))
        elif override_type == 'set_dirty_state':
            pass
        else:
            raise TaskValidationError("Unsupported init override type '{}'.".format(override_type))

def _set_validation(task: Dict[str, Any], status: str, checks: Mapping[str, Any], failure_reason: Optional[str]) -> None:
    task['validation'] = {'status': status, 'checks': dict(checks), 'failure_reason': failure_reason}
if __name__ == '__main__':
    main()
__all__ = ['TaskValidationError', 'summarize_validation', 'validate_one_task', 'validate_tasks', 'write_task_jsonl']
