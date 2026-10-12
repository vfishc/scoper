import copy
import pytest
from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.benchmark.behavior.brains.teachers.prepare import PrepareTeacher
from scoper.benchmark.behavior.episode_runner import EpisodeRunner
from scoper.benchmark.behavior.task_spec import TaskSpecValidationError, validate_behavior_task_spec
from tests.behavior.conftest import stage5_task_by_id

def test_family_spec_validate_accepts_prepare():
    spec = stage5_task_by_id('prepare_heat_store')
    validate_behavior_task_spec(spec)

def test_family_spec_validate_rejects_bad_stage():
    spec = copy.deepcopy(stage5_task_by_id('prepare_heat_store'))
    spec['family_spec']['prepare']['stages'][0]['type'] = 'teleport'
    with pytest.raises(TaskSpecValidationError):
        validate_behavior_task_spec(spec)

def test_family_spec_validate_rejects_empty_stages():
    spec = copy.deepcopy(stage5_task_by_id('prepare_heat_store'))
    spec['family_spec']['prepare']['stages'] = []
    with pytest.raises(TaskSpecValidationError):
        validate_behavior_task_spec(spec)

def test_explicit_store_stage_opens_closed_openable_container():
    spec = copy.deepcopy(stage5_task_by_id('prepare_heat_store'))
    spec['roles']['target_receptacle'].update({'synset': 'cabinet.n.01', 'bddl_inst': 'cabinet.n.01_1', 'category': 'bottom_cabinet_no_top', 'abilities': ['container', 'openable', 'receptacle'], 'init_states': {'open': False, 'contains': {}, 'filled': {}}})
    spec['goal']['predicates'][-1] = {'type': 'object_in_receptacle', 'args': {'object': 'target_object', 'receptacle': 'target_receptacle'}}
    env = BehaviorClosedLoopToolEnv()
    env.reset_to_task(spec, rng_seed=0)
    brain = PrepareTeacher()
    brain.on_episode_start(env, spec)
    seq = [(call.tool_name, call.arguments.get('target')) for call in brain._queue]
    assert ('open', 'target_receptacle') in seq
    assert seq.index(('open', 'target_receptacle')) < seq.index(('put_down', 'target_receptacle'))

def test_heat_then_place():
    spec = stage5_task_by_id('prepare_heat_store')
    env = BehaviorClosedLoopToolEnv()
    result = EpisodeRunner(env, max_high_level_steps=50).run(spec, brain=PrepareTeacher(), rng_seed=0)
    assert result.final_goal['task_success'] == 1.0
    sat = {p['type']: p['satisfied'] for p in result.final_goal['predicates']}
    assert sat['object_cooked'] is True
    assert sat['object_on_receptacle'] is True
    assert actions.get_cooked_state(env.core.objects['apple.n.01_1']) is True

def test_slice_then_store():
    spec = copy.deepcopy(stage5_task_by_id('prepare_heat_store'))
    spec['id'] = 'behavior_stage5_prepare_slice_store_test'
    spec['roles']['tool'] = {'synset': 'carving_knife.n.01', 'bddl_inst': 'carving_knife.n.01_1', 'category': 'carving_knife', 'abilities': ['pickupable', 'cutting_tool'], 'extent': [0.2, 0.04, 0.04]}
    spec['roles']['target_object']['abilities'] = ['pickupable', 'sliceable']
    spec['roles']['target_object']['init_states'] = {'sliced': False}
    spec['init'] = [{'predicate': 'sliced', 'args': ['target_object']}]
    spec['family_spec']['prepare']['stages'] = [{'type': 'slice', 'tool': 'tool'}, {'type': 'store', 'receptacle': 'target_receptacle'}]
    spec['goal']['predicates'] = [{'type': 'object_sliced', 'args': {'object': 'target_object', 'tool': 'tool'}}, {'type': 'object_on_receptacle', 'args': {'object': 'target_object', 'receptacle': 'target_receptacle'}}]
    validate_behavior_task_spec(spec)
    env = BehaviorClosedLoopToolEnv()
    result = EpisodeRunner(env, max_high_level_steps=50).run(spec, brain=PrepareTeacher(), rng_seed=0)
    assert result.final_goal['task_success'] == 1.0
