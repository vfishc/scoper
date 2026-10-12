from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
PRIOR_DATASET_NAME = 'procthor-10k'
PRIOR_URI_PREFIX = 'prior://procthor-10k/'
DEFAULT_SPLITS = ('train', 'val', 'test')
EXPECTED_SPLIT_COUNTS = {'train': 10000, 'val': 1000, 'test': 1000}
_DATASET_CACHE: Dict[Tuple[str, Optional[str], Optional[str]], Any] = {}

class ProcTHORDatasetAssetError(ValueError):
    pass

def is_prior_house_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith(PRIOR_URI_PREFIX)

def make_prior_house_uri(split: str, index: int) -> str:
    split = _validate_split(split)
    index = int(index)
    if index < 0:
        raise ProcTHORDatasetAssetError('House index must be non-negative.')
    return '{}{}/{}'.format(PRIOR_URI_PREFIX, split, index)

def parse_prior_house_uri(uri: str) -> Tuple[str, int]:
    if not is_prior_house_uri(uri):
        raise ProcTHORDatasetAssetError('Not a ProcTHOR prior URI: {}'.format(uri))
    suffix = uri[len(PRIOR_URI_PREFIX):]
    parts = suffix.split('/')
    if len(parts) != 2:
        raise ProcTHORDatasetAssetError('Prior URI must have shape prior://procthor-10k/{split}/{index}: {}'.format(uri))
    split = _validate_split(parts[0])
    try:
        index = int(parts[1])
    except ValueError as exc:
        raise ProcTHORDatasetAssetError('Prior URI index must be an integer: {}'.format(uri)) from exc
    if index < 0:
        raise ProcTHORDatasetAssetError('Prior URI index must be non-negative.')
    return (split, index)

def load_prior_dataset(dataset_name: str=PRIOR_DATASET_NAME, revision: Optional[str]=None, prior_home: Optional[str]=None, offline: bool=False) -> Any:
    resolved_prior_home = prior_home or os.environ.get('EB_PROCTHOR_PRIOR_HOME')
    resolved_offline = bool(offline) or _env_flag('EB_PROCTHOR_PRIOR_OFFLINE')
    cache_key = (dataset_name, revision, str(resolved_prior_home) if resolved_prior_home else None, resolved_offline)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]
    try:
        import prior
    except ImportError as exc:
        raise ProcTHORDatasetAssetError("The 'prior' package is required to load ProcTHOR-10K assets.") from exc
    if resolved_prior_home:
        prior.BASE_DIR = str(Path(resolved_prior_home) / '.prior')
        prior.DATASET_DIR = str(Path(prior.BASE_DIR) / 'datasets')
    kwargs = {}
    if revision:
        kwargs['revision'] = revision
    if resolved_offline:
        kwargs['offline'] = True
    dataset = prior.load_dataset(dataset_name, **kwargs)
    _DATASET_CACHE[cache_key] = dataset
    return dataset

def get_prior_split(dataset: Any, split: str) -> Sequence[Mapping[str, Any]]:
    split = _validate_split(split)
    if hasattr(dataset, split):
        return getattr(dataset, split)
    try:
        return dataset[split]
    except Exception as exc:
        raise ProcTHORDatasetAssetError("Prior dataset does not expose split '{}'. Available object: {}.".format(split, type(dataset).__name__)) from exc

def load_prior_house(uri: str, dataset_name: str=PRIOR_DATASET_NAME, revision: Optional[str]=None, prior_home: Optional[str]=None) -> dict:
    split, index = parse_prior_house_uri(uri)
    dataset = load_prior_dataset(dataset_name=dataset_name, revision=revision, prior_home=prior_home)
    records = get_prior_split(dataset, split)
    if index >= len(records):
        raise ProcTHORDatasetAssetError("Prior URI index {} is out of range for split '{}' with {} houses.".format(index, split, len(records)))
    house = dict(records[index])
    validate_prior_house_record(house, split=split, index=index)
    return house

def validate_prior_house_record(house: Mapping[str, Any], split: str='unknown', index: Optional[int]=None) -> None:
    context = '{}[{}]'.format(split, index) if index is not None else split
    if not isinstance(house, Mapping):
        raise ProcTHORDatasetAssetError('{} house must be a mapping.'.format(context))
    rooms = house.get('rooms')
    if not isinstance(rooms, list) or not rooms:
        raise ProcTHORDatasetAssetError('{} house has no rooms list.'.format(context))
    if not any((isinstance(room, Mapping) and room.get('roomType') for room in rooms)):
        raise ProcTHORDatasetAssetError('{} house rooms do not expose roomType.'.format(context))
    if 'objects' in house and (not isinstance(house['objects'], list)):
        raise ProcTHORDatasetAssetError('{} house objects field is not a list.'.format(context))

def build_prior_house_registry(splits: Sequence[str]=DEFAULT_SPLITS, limit_per_split: Optional[int]=None, stride: int=1, dataset_name: str=PRIOR_DATASET_NAME, revision: Optional[str]=None, prior_home: Optional[str]=None, validate_records: bool=True) -> Tuple[List[dict], dict]:
    if stride <= 0:
        raise ProcTHORDatasetAssetError('stride must be > 0.')
    dataset = load_prior_dataset(dataset_name=dataset_name, revision=revision, prior_home=prior_home)
    entries: List[dict] = []
    split_counts = {}
    selected_counts = {}
    room_type_counts: Dict[str, int] = {}
    for split in splits:
        split = _validate_split(split)
        records = get_prior_split(dataset, split)
        split_counts[split] = len(records)
        selected = range(0, len(records), stride)
        if limit_per_split is not None:
            selected = list(selected)[:int(limit_per_split)]
        selected_counts[split] = len(selected)
        for index in selected:
            house = records[index]
            if validate_records:
                validate_prior_house_record(house, split=split, index=index)
                for room in house.get('rooms', []):
                    room_type = room.get('roomType')
                    if room_type:
                        room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
            entries.append({'house_id': '{}_{:06d}'.format(split, index), 'split': split, 'source': 'procthor', 'house_path': make_prior_house_uri(split, index), 'metadata': {'dataset': dataset_name, 'prior_uri': make_prior_house_uri(split, index), 'split_index': index}})
    summary = {'dataset': dataset_name, 'revision': revision, 'splits': list(splits), 'split_counts': split_counts, 'selected_counts': selected_counts, 'num_entries': len(entries), 'limit_per_split': limit_per_split, 'stride': stride, 'expected_split_counts': {split: EXPECTED_SPLIT_COUNTS.get(split) for split in splits}, 'room_type_counts': dict(sorted(room_type_counts.items()))}
    return (entries, summary)

def write_registry_jsonl(entries: Iterable[Mapping[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')

def verify_prior_dataset_assets(splits: Sequence[str]=DEFAULT_SPLITS, dataset_name: str=PRIOR_DATASET_NAME, revision: Optional[str]=None, prior_home: Optional[str]=None, full_scan: bool=True) -> dict:
    dataset = load_prior_dataset(dataset_name=dataset_name, revision=revision, prior_home=prior_home)
    failures = []
    split_counts = {}
    checked_counts = {}
    for split in splits:
        split = _validate_split(split)
        records = get_prior_split(dataset, split)
        split_counts[split] = len(records)
        indices = range(len(records)) if full_scan else range(min(5, len(records)))
        checked = 0
        for index in indices:
            try:
                validate_prior_house_record(records[index], split=split, index=index)
            except Exception as exc:
                failures.append({'split': split, 'index': index, 'error_type': type(exc).__name__, 'error': str(exc)})
            checked += 1
        checked_counts[split] = checked
    return {'dataset': dataset_name, 'revision': revision, 'splits': list(splits), 'split_counts': split_counts, 'checked_counts': checked_counts, 'expected_split_counts': {split: EXPECTED_SPLIT_COUNTS.get(split) for split in splits}, 'num_failures': len(failures), 'failures': failures[:50], 'full_scan': bool(full_scan)}

def main(argv: Optional[Sequence[str]]=None) -> None:
    parser = argparse.ArgumentParser(description='Manage ProcTHOR-10K prior assets.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    verify = subparsers.add_parser('verify', help='Verify prior dataset structure.')
    _add_common_args(verify)
    verify.add_argument('--quick', action='store_true', help='Check only first 5 houses per split.')
    verify.add_argument('--output', default=None, help='Optional JSON summary output.')
    registry = subparsers.add_parser('build-registry', help='Build prior:// house registry.')
    _add_common_args(registry)
    registry.add_argument('--output', required=True)
    registry.add_argument('--limit-per-split', type=int, default=None)
    registry.add_argument('--stride', type=int, default=1)
    registry.add_argument('--no-validate-records', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'verify':
        summary = verify_prior_dataset_assets(splits=args.splits, dataset_name=args.dataset_name, revision=args.revision, prior_home=args.prior_home, full_scan=not args.quick)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        if summary['num_failures']:
            raise SystemExit(1)
    elif args.command == 'build-registry':
        entries, summary = build_prior_house_registry(splits=args.splits, limit_per_split=args.limit_per_split, stride=args.stride, dataset_name=args.dataset_name, revision=args.revision, prior_home=args.prior_home, validate_records=not args.no_validate_records)
        write_registry_jsonl(entries, args.output)
        summary['output'] = args.output
        summary_path = str(Path(args.output).with_suffix(Path(args.output).suffix + '.summary.json'))
        Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--dataset-name', default=PRIOR_DATASET_NAME)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--prior-home', default=None)
    parser.add_argument('--splits', nargs='+', default=list(DEFAULT_SPLITS))

def _validate_split(split: str) -> str:
    split = str(split)
    if split not in DEFAULT_SPLITS:
        raise ProcTHORDatasetAssetError("Unsupported ProcTHOR split '{}'. Expected one of: {}.".format(split, ', '.join(DEFAULT_SPLITS)))
    return split

def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}
if __name__ == '__main__':
    main()
__all__ = ['DEFAULT_SPLITS', 'EXPECTED_SPLIT_COUNTS', 'PRIOR_DATASET_NAME', 'PRIOR_URI_PREFIX', 'ProcTHORDatasetAssetError', 'build_prior_house_registry', 'is_prior_house_uri', 'load_prior_dataset', 'load_prior_house', 'make_prior_house_uri', 'parse_prior_house_uri', 'verify_prior_dataset_assets', 'write_registry_jsonl']
