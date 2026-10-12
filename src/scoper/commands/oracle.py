"""Run one bank episode through the packaged simulator and oracle.

External simulator binaries/assets must already be installed. This command
never downloads assets and refuses to overwrite an existing output directory.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['behavior', 'procthor'], required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('choose a new output directory')
    raw = [json.loads(l) for l in args.bank.read_text().splitlines()] if args.bank.suffix=='.jsonl' else json.loads(args.bank.read_text())
    tasks = raw if isinstance(raw,list) else raw['tasks']
    task = next((t for t in tasks if t['id']==args.task_id), None)
    if task is None:
        parser.error('task ID is absent from the bank')
    args.output.mkdir(parents=True)
    from importlib import import_module
    workflow = import_module('scoper.workflows.' + args.backend)
    result = workflow.oracle(args, task)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str))
    passed = workflow.oracle_passed(args, result)
    if not passed:
        raise SystemExit('Oracle or real rendering did not pass; see result.json')


if __name__=='__main__':
    main()
