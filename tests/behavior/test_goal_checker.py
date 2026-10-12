from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.bddl_adapter import BDDLAdapter
from scoper.benchmark.behavior.goal_checker import BehaviorGoalChecker
_EXPECTED_COMPAT_FIELDS = {'task_success', 'task_progress', 'predicate_success', 'completion_candidate', 'require_model_stop', 'model_stopped', 'predicates'}

def _place_bowl_on_table(core):
    bowl = core.objects['bowl.n.01_1']
    table = core.objects['breakfast_table.n.01_1']
    bowl.center = (table.center[0], table.center[1], table.center[2] + table.extent[2] * 0.5 + bowl.extent[2] * 0.5)

def test_initial_zero(core, smoke_task_spec, role_map):
    result = BehaviorGoalChecker(BDDLAdapter(), core, role_map).check(smoke_task_spec)
    assert result['task_success'] == 0.0
    assert result['task_progress'] == 0.0
    assert _EXPECTED_COMPAT_FIELDS.issubset(set(result))
    assert 'predicate_records' in result

def test_progress_partial(core, smoke_task_spec, role_map):
    actions.set_open_state(core.objects['electric_refrigerator.n.01_1'], True)
    result = BehaviorGoalChecker(BDDLAdapter(), core, role_map).check(smoke_task_spec)
    assert result['task_progress'] == 0.5
    assert result['task_success'] == 0.0
    assert result['predicate_success'] == 0.0

def test_after_mutation_one(core, smoke_task_spec, role_map):
    _place_bowl_on_table(core)
    actions.set_open_state(core.objects['electric_refrigerator.n.01_1'], True)
    result = BehaviorGoalChecker(BDDLAdapter(), core, role_map).check(smoke_task_spec)
    assert result['task_success'] == 1.0
    assert result['task_progress'] == 1.0
    assert result['predicate_success'] == 1.0

def test_require_model_stop(core, smoke_task_spec, role_map):
    smoke_task_spec['goal']['require_model_stop'] = True
    _place_bowl_on_table(core)
    actions.set_open_state(core.objects['electric_refrigerator.n.01_1'], True)
    checker = BehaviorGoalChecker(BDDLAdapter(), core, role_map)
    not_stopped = checker.check(smoke_task_spec, model_stopped=False)
    assert not_stopped['predicate_success'] == 1.0
    assert not_stopped['task_success'] == 0.0
    stopped = checker.check(smoke_task_spec, model_stopped=True)
    assert stopped['task_success'] == 1.0

def test_eb_private_predicates(core, smoke_task_spec, role_map):
    smoke_task_spec['goal']['predicates'] = [{'type': 'object_sliced', 'args': {'object': 'target_object'}}, {'type': 'agent_near', 'args': {'object': 'fridge'}}]
    checker = BehaviorGoalChecker(BDDLAdapter(), core, role_map)
    initial = checker.check(smoke_task_spec)
    assert initial['task_progress'] == 0.0
    actions.set_sliced_state(core.objects['bowl.n.01_1'], True)
    fridge = core.objects['electric_refrigerator.n.01_1']
    core.ego_state.pose.x = fridge.center[0]
    core.ego_state.pose.y = fridge.center[1]
    final = checker.check(smoke_task_spec)
    assert final['task_success'] == 1.0
    assert final['task_progress'] == 1.0

def test_mixed_bddl_and_eb_private(core, smoke_task_spec, role_map):
    smoke_task_spec['goal']['predicates'] = [{'type': 'object_open', 'args': {'object': 'fridge'}}, {'type': 'object_sliced', 'args': {'object': 'target_object'}}]
    actions.set_open_state(core.objects['electric_refrigerator.n.01_1'], True)
    result = BehaviorGoalChecker(BDDLAdapter(), core, role_map).check(smoke_task_spec)
    assert result['task_progress'] == 0.5
    types = {p['type']: p['satisfied'] for p in result['predicates']}
    assert types['object_open'] is True
    assert types['object_sliced'] is False
