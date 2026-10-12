from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from scoper.benchmark.procthor.online_task_loader import load_task_bank
from scoper.benchmark.procthor.online_task_spec import ALLOWED_CATEGORIES, TaskSpecValidationError, validate_task_set
TASK_SET_VERSION = '0.1'
DEFAULT_TASK_SET_NAMES = ('smoke', 'mvp', 'train', 'val', 'test')

class SplitBuilderError(ValueError):
    pass

def build_task_sets(task_specs: Sequence[Mapping[str, Any]], smoke_per_category: int=2, mvp_per_category: int=50, seed: int=0, require_all_categories: bool=True, train_ratio: float=0.67, val_ratio: float=0.17) -> Dict[str, dict]:
    tasks = _validated_tasks(task_specs)
    if not tasks:
        raise SplitBuilderError('No validated tasks were provided.')
    tasks_by_id = {task['id']: task for task in tasks}
    by_category = _tasks_by_category(tasks)
    categories = sorted(ALLOWED_CATEGORIES)
    if require_all_categories:
        missing = [category for category in categories if category not in by_category]
        if missing:
            raise SplitBuilderError('Missing task categories required for task sets: {}.'.format(', '.join(missing)))
    smoke_ids = _select_per_category(by_category, smoke_per_category, categories)
    mvp_ids = _select_per_category(by_category, mvp_per_category, categories)
    train_houses, val_houses, test_houses = split_house_ids(sorted({task['source']['house_id'] for task in tasks}), seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    split_policy = {'type': 'house_disjoint_split', 'seed': seed, 'train_ratio': train_ratio, 'val_ratio': val_ratio, 'test_ratio': max(0.0, 1.0 - train_ratio - val_ratio)}
    return {'smoke': _task_set('smoke', smoke_ids, tasks_by_id, description='Small category-balanced smoke set.', split_policy={'type': 'category_balanced', 'per_category': smoke_per_category}), 'mvp': _task_set('mvp', mvp_ids, tasks_by_id, description='Larger category-balanced MVP set.', split_policy={'type': 'category_balanced', 'per_category': mvp_per_category}), 'train': _task_set('train', _ids_for_houses(tasks, train_houses), tasks_by_id, description='House-disjoint train split.', split_policy=dict(split_policy, split='train', house_ids=train_houses)), 'val': _task_set('val', _ids_for_houses(tasks, val_houses), tasks_by_id, description='House-disjoint validation split.', split_policy=dict(split_policy, split='val', house_ids=val_houses)), 'test': _task_set('test', _ids_for_houses(tasks, test_houses), tasks_by_id, description='House-disjoint test split.', split_policy=dict(split_policy, split='test', house_ids=test_houses))}

def split_house_ids(house_ids: Sequence[str], seed: int=0, train_ratio: float=0.67, val_ratio: float=0.17) -> Tuple[List[str], List[str], List[str]]:
    if not house_ids:
        return ([], [], [])
    if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise SplitBuilderError('Expected ratios with 0 < train, 0 <= val, train + val < 1.')
    shuffled = list(house_ids)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        return (sorted(shuffled), [], [])
    num_houses = len(shuffled)
    num_val = max(1, int(round(num_houses * val_ratio)))
    num_test = max(1, num_houses - int(round(num_houses * (train_ratio + val_ratio))))
    num_train = max(1, num_houses - num_val - num_test)
    if num_train + num_val + num_test > num_houses:
        num_train = max(1, num_houses - num_val - num_test)
    train = sorted(shuffled[:num_train])
    val = sorted(shuffled[num_train:num_train + num_val])
    test = sorted(shuffled[num_train + num_val:])
    return (train, val, test)

def write_task_sets(task_sets: Mapping[str, Mapping[str, Any]], output_dir: str) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in DEFAULT_TASK_SET_NAMES:
        task_set = task_sets[name]
        validate_task_set(task_set)
        path = output / '{}.json'.format(name)
        path.write_text(json.dumps(task_set, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
        summary[name] = task_set['summary']
    summary_path = output / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    return summary

def main(argv: Optional[Sequence[str]]=None) -> None:
    parser = argparse.ArgumentParser(description='Build EB-ProcTHOR task sets.')
    parser.add_argument('--task-bank', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke-per-category', type=int, default=2)
    parser.add_argument('--mvp-per-category', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-ratio', type=float, default=0.67)
    parser.add_argument('--val-ratio', type=float, default=0.17)
    parser.add_argument('--allow-missing-categories', action='store_true', help='Build task sets even if the bank does not cover all v1 categories.')
    args = parser.parse_args(argv)
    task_bank = load_task_bank(args.task_bank)
    task_sets = build_task_sets(list(task_bank.values()), smoke_per_category=args.smoke_per_category, mvp_per_category=args.mvp_per_category, seed=args.seed, require_all_categories=not args.allow_missing_categories, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    summary = write_task_sets(task_sets, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

def _validated_tasks(task_specs: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    tasks = [task for task in task_specs if task.get('validation', {}).get('status') == 'validated']
    return sorted(tasks, key=lambda task: task['id'])

def _tasks_by_category(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    by_category: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_category[task['category']].append(task)
    for category, category_tasks in by_category.items():
        by_category[category] = sorted(category_tasks, key=lambda task: (task['source']['house_id'], task['roles'].get('target_object', {}).get('object_type', ''), task['id']))
    return by_category

def _select_per_category(by_category: Mapping[str, Sequence[Mapping[str, Any]]], per_category: int, categories: Sequence[str]) -> List[str]:
    if per_category <= 0:
        return []
    selected = []
    for category in categories:
        for task in by_category.get(category, [])[:per_category]:
            selected.append(task['id'])
    return selected

def _ids_for_houses(tasks: Sequence[Mapping[str, Any]], house_ids: Sequence[str]) -> List[str]:
    houses = set(house_ids)
    return [task['id'] for task in sorted(tasks, key=lambda task: (task['source']['house_id'], task['id'])) if task['source']['house_id'] in houses]

def _task_set(name: str, task_ids: Sequence[str], tasks_by_id: Mapping[str, Mapping[str, Any]], description: str, split_policy: Mapping[str, Any]) -> dict:
    unique_ids = []
    seen = set()
    for task_id in task_ids:
        if task_id not in tasks_by_id:
            raise TaskSpecValidationError("Task set '{}' references missing id '{}'.".format(name, task_id))
        if task_id not in seen:
            unique_ids.append(task_id)
            seen.add(task_id)
    task_set = {'name': name, 'version': TASK_SET_VERSION, 'description': description, 'task_ids': unique_ids, 'split_policy': dict(split_policy), 'summary': _summary(unique_ids, tasks_by_id)}
    validate_task_set(task_set, task_ids=tasks_by_id.keys())
    return task_set

def _summary(task_ids: Sequence[str], tasks_by_id: Mapping[str, Mapping[str, Any]]) -> dict:
    tasks = [tasks_by_id[task_id] for task_id in task_ids]
    category_counts = Counter((task['category'] for task in tasks))
    house_counts = Counter((task['source']['house_id'] for task in tasks))
    return {'num_tasks': len(tasks), 'category_counts': dict(sorted(category_counts.items())), 'house_counts': dict(sorted(house_counts.items())), 'house_ids': sorted(house_counts.keys())}
if __name__ == '__main__':
    main()
__all__ = ['DEFAULT_TASK_SET_NAMES', 'SplitBuilderError', 'TASK_SET_VERSION', 'build_task_sets', 'split_house_ids', 'write_task_sets']
