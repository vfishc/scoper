"""Validate a portable task bank without starting either simulator."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from scoper.benchmark.common.binding_compiler import load_binding_sidecar
from scoper.benchmark.common.binding import validate_task_binding_spec
from scoper.benchmark.common.situated_qa import validate_qa_overlay
from scoper.benchmark.common.evaluation_binding_policy import public_reference_rules
from scoper.benchmark.common.action_coverage import load_references


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank_directory',type=Path)
    args=parser.parse_args(argv)
    bank=args.bank_directory/'tasks_with_qa.json'
    digest=hashlib.sha256(bank.read_bytes()).hexdigest()
    tasks=json.loads(bank.read_text())['tasks']
    specs=load_binding_sidecar(json.loads((args.bank_directory/'binding.json').read_text()),bank_sha256=digest)
    references=load_references(args.bank_directory/'oracle_reference.json',digest)
    ids={t['id'] for t in tasks}
    assert len(ids)==len(tasks) and ids==set(specs)==set(references)
    rules=public_reference_rules().lower()
    qa_count=0
    for task in tasks:
        validate_task_binding_spec(task,specs[task['id']])
        children=task.get('family_spec',{}).get('composite',{}).get('subtasks',[])
        overlay=validate_qa_overlay(task['qa_overlay'],role_names=list(task['roles']),subtask_ids=[c['subtask_id'] for c in children])
        qa_count+=len(overlay['questions'])
        for role in specs[task['id']].roles.values():
            assert any(re.search(r'(?<![a-z])'+re.escape(term.lower())+r'(?![a-z])',rules) for term in role.public_terms),'uncovered public role'
        assert references[task['id']]['status']=='ready'
    print(json.dumps(dict(status='passed',tasks=len(tasks),overlay_qa=qa_count,bank_sha256=digest,
                         checks=['binding_identity','action_references','qa_schemas','public_vocabulary'],simulator_run=False),indent=2))


if __name__=='__main__':main()
