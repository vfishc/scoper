import importlib.util
from pathlib import Path
import json
import pytest
from scoper.harness import RGBFrame, JSONBackbone, SCOPERRuntime
from scoper.harness.evaluation import score_episode, aggregate, DirectRuntime
from scoper.harness.evaluation_policy import step_budget
from scoper.harness.prompt_contracts import hardqa_prompt_contract_descriptor_for_harness_version
from scoper.harness.binding import TaskBindingSpec, validate_task_binding_spec
from scoper.harness.situated_qa import validate_qa_overlay
from scoper.harness.backends import ProcTHORAdapter, BEHAVIORAdapter
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('smoke', ROOT / 'tests/fixtures/scripted_smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

@pytest.mark.parametrize('backend', ['procthor', 'behavior'])
@pytest.mark.parametrize('harness', [False, True])
@pytest.mark.parametrize('json_transport', [False, True])
def test_two_backends_and_two_transports(backend, harness, json_transport):
    record = smoke.run(backend, harness, json_transport)
    assert record['backend'] == backend
    assert record['model_calls'] >= record['high_level_steps']

@pytest.mark.parametrize('children,budget', [(1, 24), (2, 36), (3, 48)])
def test_budget(children, budget):
    task = {} if children == 1 else {'family_spec': {'composite': {'subtasks': [{}] * children}}}
    assert step_budget(task) == budget

def test_weighted_metrics_and_unanswered_questions():
    a = score_episode(stopped=True, goal_satisfied=[True, True], coverage_matched=1, coverage_total=1, qa_configured=10, qa_triggered=2, qa_correct=1)
    b = score_episode(stopped=False, goal_satisfied=[True], coverage_matched=0, coverage_total=1, qa_configured=2, qa_triggered=2, qa_correct=0)
    r = aggregate([a, b])
    assert r['task_success_rate'] == 0.5
    assert r['qa_trigger_rate'] == 4 / 12 and r['qa_accuracy'] == 1 / 4
    assert r['task_progress'] == 1 and r['coverage'] == 0.5

def test_history_and_frame_limits():
    backbone = JSONBackbone(lambda *a: {}, 'a' * 64)
    direct = DirectRuntime(backbone, smoke.TOOLS, [])
    harness = SCOPERRuntime(backbone, smoke.TOOLS, [])
    assert direct.frames.maxlen == 32 and direct.history.maxlen == 48 and (harness.history.maxlen == 48)
    descriptor = hardqa_prompt_contract_descriptor_for_harness_version('scoper-1')
    assert descriptor['policy_history_limit'] == 48
    assert descriptor['context_limits']['history'] == 48
    assert descriptor['context_limits']['policy_frames'] == 4
    assert descriptor['context_limits']['observation_frames'] == 8

@pytest.mark.parametrize('adapter_class', [ProcTHORAdapter, BEHAVIORAdapter])
def test_stop_freezes_goals_and_disallows_physical_actions(adapter_class):
    state = [True]
    a = adapter_class(execute=lambda c: None, expected_effect=lambda c, m: None, goal_predicates=lambda: state, submit_answer=lambda x: None, poll_public_messages=lambda: (), scoring_counts=lambda: dict(coverage_matched=0, coverage_total=0, qa_configured=0, qa_triggered=0, qa_correct=0))
    a.stop()
    state[0] = False
    assert a.finalize()['task_success'] == 1
    with pytest.raises(RuntimeError):
        a.execute({'tool_name': 'scan', 'arguments': {}})

@pytest.mark.parametrize('backend', ['procthor', 'behavior'])
def test_task_subset_and_updated_template(backend):
    tasks = [json.loads(p.read_text()) for p in sorted((ROOT / 'tests/fixtures/tasks' / backend).glob('*.json'))]
    assert len(tasks) == 10
    new = [t for t in tasks if 'conditional_storage_policy' in t]
    assert len(new) == 4
    assert sorted((t['conditional_storage_policy']['expected_branch'] for t in new)) == ['dishware', 'dishware', 'no_dishware', 'no_dishware']
    for task in tasks:
        assert 'human_qa_simple_place_001' not in json.dumps(task)
        assert task['goal']['require_model_stop'] is True
        validate_qa_overlay(task['qa_overlay'])
        if 'task_binding_spec' in task:
            validate_task_binding_spec(task, TaskBindingSpec.from_dict(task['task_binding_spec']))
        assert not any((x in json.dumps(task) for x in ['/home/', '/mnt/', '/data/']))

@pytest.mark.parametrize('adapter_class', [ProcTHORAdapter, BEHAVIORAdapter])
def test_navigation_requires_verification_spec(adapter_class):
    a = adapter_class(execute=lambda c: None, expected_effect=lambda c, m: None, goal_predicates=lambda: [True], submit_answer=lambda x: None, poll_public_messages=lambda: (), scoring_counts=lambda: {})
    assert a.expected_effect({'tool_name': 'scan'}, {}) is None
    with pytest.raises(ValueError, match='navigation and manipulation'):
        a.expected_effect({'tool_name': 'navigate_to'}, {})

def test_handoff_is_evaluator_owned():
    from scoper.harness.evaluation import evaluate_episode
    with pytest.raises(ValueError, match='handoff belongs'):
        evaluate_episode({}, None, None, [{'name': 'handoff'}], [], None)
