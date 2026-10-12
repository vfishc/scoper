import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('backend,count', [('procthor',450),('behavior',420)])
def test_bank_custom_template_references_are_packaged(backend, count):
    root=Path(__file__).resolve().parents[1]
    tasks=json.loads((root/'task_bank'/backend/'tasks_with_qa.json').read_text())['tasks']
    assert len(tasks)==count
    templates=root/'src/scoper/benchmark/procthor/data/task_templates'
    for task in tasks:
        template=task.get('template',{}).get('template_file')
        if template:
            assert (templates/template).is_file(),template
        source=task['source']
        if backend=='behavior':
            assert source['source_json'].startswith('datasets/behavior-1k-assets/scenes/')
        else:
            assert source['house_path'].startswith('prior://procthor-10k/')


def test_human_insert_template_is_loadable():
    from scoper.resources import package_root
    from scoper.benchmark.procthor.task_templates import load_task_templates
    path=package_root()/'benchmark/procthor/data/task_templates/human_qa_and_insert_simple_place_001.json'
    assert load_task_templates(str(path))[0]['template_id']=='human_qa_and_insert_simple_place_001'
