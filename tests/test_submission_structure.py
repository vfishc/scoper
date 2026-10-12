import importlib
import json
from pathlib import Path
import pytest
from scoper.resources import package_root
from scoper.benchmark.common.container_inspection.teacher import is_dishware, same_contents
from scoper.benchmark.common.container_inspection.definition import TEMPLATE, CATEGORY

@pytest.mark.parametrize('backend', ['behavior', 'procthor'])
def test_updated_template_teacher_dispatch(backend):
    module = importlib.import_module(f'scoper.benchmark.{backend}.brains.teachers.dispatcher')
    dispatcher = module.TeacherDispatcher()
    task = {'category': CATEGORY, 'template': {'template_id': TEMPLATE}}
    teacher = dispatcher.build(task)
    assert teacher.__class__.__name__ == 'InspectTeacher'

@pytest.mark.parametrize('name,expected', [('Cup', True), ('coffee_cup', True), ('Bowl', True), ('Plate', True), ('Mug', True), ('Knife', False), ('Apple', False)])
def test_inspection_dishware(name, expected):
    assert is_dishware(name) is expected

def test_inspection_contents_guard():
    original = [{'id': 'cup', 'type': 'Cup', 'position': [0, 0, 0]}]
    assert same_contents(original, [{'id': 'cup', 'type': 'Cup', 'position': [0.001, 0, 0]}])
    assert not same_contents(original, [])
    assert not same_contents(original, [{'id': 'cup', 'type': 'Cup', 'position': [1, 0, 0]}])

@pytest.mark.parametrize('backend', ['behavior', 'procthor'])
def test_prompt_resources(backend):
    mod = importlib.import_module(f'scoper.agents.{backend}.prompt_builder')
    assert mod.DEFAULT_TEMPLATE_DIR.is_dir()
    assert any(mod.DEFAULT_TEMPLATE_DIR.iterdir())

def test_matching_resources():
    from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates
    from scoper.benchmark.behavior.scene.taxonomy import DEFAULT_PROCTHOR_TAXONOMY
    from scoper.benchmark.behavior.scene_geometry.rules import load_rules
    assert load_procthor_templates()
    assert DEFAULT_PROCTHOR_TAXONOMY.is_file()
    assert load_rules().placement_policy

@pytest.mark.parametrize('backend', ['behavior', 'procthor'])
def test_updated_examples_present(backend):
    root = Path(__file__).resolve().parents[1] / 'tests/fixtures/tasks' / backend
    tasks = [json.loads(p.read_text()) for p in root.glob('*.json')]
    updated = [t for t in tasks if t.get('template', {}).get('template_id') == TEMPLATE]
    assert len(tasks) == 10
    assert {t['conditional_storage_policy']['expected_branch'] for t in updated} == {'dishware', 'no_dishware'}
