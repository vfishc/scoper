"""Generate candidates from an official layout using packaged matchers.

This is a bounded example, not an automatic full-bank scheduler. BEHAVIOR
performs symbolic admission; ProcTHOR profiles the actual loaded Unity scene.
New candidates require oracle execution before they are accepted into a bank.
"""
import argparse
import json
from pathlib import Path

from scoper.resources import package_root


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['behavior', 'procthor'], required=True)
    p.add_argument('--layout', required=True, help='BEHAVIOR scene model or ProcTHOR house URI')
    p.add_argument('--template', default='static_simple.json', help='Packaged template filename')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=2)
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error('choose a new output directory')
    if not 1 <= a.limit <= 8:
        p.error('this bounded example supports one to eight candidates')
    a.output.mkdir(parents=True)
    data = package_root()/'benchmark/procthor/data'
    template_path = data/'task_templates'/a.template
    if not template_path.is_file() or template_path.parent != data/'task_templates':
        p.error('choose a packaged template filename')

    def save(name, value):
        (a.output/name).write_text(json.dumps(value, indent=2, default=str))

    from importlib import import_module
    workflow = import_module('scoper.workflows.' + a.backend)
    profile, generated, tasks, admitted = workflow.generate(a, save, template_path)
    save('profile.json', profile)
    save('generation.json', generated)
    save('tasks.json', {'tasks': tasks})
    save('symbolically_admitted.json', {'tasks': admitted})
    report = {'backend':a.backend,'candidate_count':len(tasks),'symbolically_admitted':len(admitted),
              'real_oracle_admission':'not_run','seed':a.seed,'template':a.template}
    save('summary.json',report)
    print(json.dumps(report))
    if not tasks:
        raise SystemExit('No candidate matched this template/layout; see reports')


if __name__ == '__main__':
    main()
