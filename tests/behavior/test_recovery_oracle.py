from __future__ import annotations
import copy
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.benchmark.behavior.brains import Brain, RecoveryOracleBrain
from scoper.benchmark.behavior.episode_runner import EpisodeRunner
from scoper.benchmark.behavior.scheduler import FailureInjectionScheduler
from tests.behavior.conftest import build_stage2_task_spec, stage5_task_by_id

def _run(spec, scheduler=None, max_steps=50):
    env = BehaviorClosedLoopToolEnv()
    result = EpisodeRunner(env, max_high_level_steps=max_steps).run(spec, brain=RecoveryOracleBrain(), scheduler=scheduler, rng_seed=0)
    return (env, result)

def _inside_spec():
    spec = copy.deepcopy(build_stage2_task_spec())
    spec['roles']['target_receptacle'] = spec['roles']['fridge']
    spec['roles']['target_receptacle']['abilities'] = ['openable', 'container', 'fillable']
    spec['roles']['target_receptacle']['init_states'] = {'open': True}
    del spec['roles']['fridge']
    spec['init'] = [{'predicate': 'open', 'args': ['target_receptacle']}]
    spec['goal']['predicates'] = [{'type': 'object_in_receptacle', 'args': {'object': 'target_object', 'receptacle': 'target_receptacle'}}]
    return spec

def test_recovery_oracle_satisfies_brain_protocol():
    assert isinstance(RecoveryOracleBrain(), Brain)

def test_recovery_oracle_solves_placement_without_disturbance():
    env, result = _run(build_stage2_task_spec())
    assert result.status == 'passed'
    assert result.final_goal['task_success'] == 1.0
    assert env.core.ego_state.held_object_name is None

def test_recovery_oracle_recovers_from_grasp_slip():
    scheduler = FailureInjectionScheduler([{'mode': 'grasp_slip', 'after_step': 1, 'once': True}])
    _, result = _run(build_stage2_task_spec(), scheduler=scheduler)
    assert result.status == 'passed'
    assert any((t['tool_name'] == 'pick_up' and (not t['success']) for t in result.transitions))
    assert sum((1 for t in result.transitions if t['tool_name'] == 'pick_up')) >= 2

def test_recovery_oracle_recovers_from_drop_beside():
    scheduler = FailureInjectionScheduler([{'mode': 'drop_beside', 'after_step': 0, 'once': True}])
    _, result = _run(build_stage2_task_spec(), scheduler=scheduler)
    assert result.status == 'passed'
    assert any((t['tool_name'] == 'put_down' and (not t['success']) for t in result.transitions))
    assert sum((1 for t in result.transitions if t['tool_name'] == 'pick_up')) >= 2
    assert sum((1 for t in result.transitions if t['tool_name'] == 'put_down')) >= 2

def test_recovery_oracle_recovers_from_partial_open():
    spec = _inside_spec()
    spec['roles']['target_receptacle']['init_states'] = {'open': False}
    spec['init'] = [{'predicate': 'closed', 'args': ['target_receptacle']}]
    scheduler = FailureInjectionScheduler([{'mode': 'partial_open', 'after_step': 0, 'once': True}])
    _, result = _run(spec, scheduler=scheduler)
    assert result.status == 'passed'
    open_calls = [t for t in result.transitions if t['tool_name'] == 'open']
    assert any((not t['success'] for t in open_calls))
    assert any((t['success'] for t in open_calls))

def test_recovery_oracle_recovers_from_nav_wrong_target():
    scheduler = FailureInjectionScheduler([{'mode': 'nav_wrong_target', 'after_step': 0, 'once': True}])
    _, result = _run(build_stage2_task_spec(), scheduler=scheduler)
    assert result.status == 'passed'
    assert any((t['tool_name'] == 'navigate_to' and (not t['success']) for t in result.transitions))
    assert sum((1 for t in result.transitions if t['tool_name'] == 'navigate_to')) >= 2

def test_recovery_oracle_stops_under_persistent_block():
    scheduler = FailureInjectionScheduler([{'mode': 'block', 'after_step': 0, 'once': False}])
    _, result = _run(build_stage2_task_spec(), scheduler=scheduler, max_steps=10)
    assert result.status == 'failed'
    assert len(result.transitions) <= 11
    assert result.transitions[2]['tool_name'] == 'stop'
    assert all((t['tool_name'] == 'stop' for t in result.transitions[2:]))

def test_recovery_oracle_solves_clean_with_wipe_tool():
    env, result = _run(stage5_task_by_id('clean'))
    assert result.status == 'passed'
    assert result.final_goal['task_success'] == 1.0
    assert [t['tool_name'] for t in result.transitions if t['tool_name'] != 'stop'] == ['navigate_to', 'pick_up', 'navigate_to', 'wipe']
    assert env.core.ego_state.held_object_name == env.role_map['tool']

def test_recovery_oracle_solves_clean_with_faucet_when_no_wipe_tool():
    spec = copy.deepcopy(stage5_task_by_id('soaked'))
    spec['id'] = 'behavior_stage5_clean_with_faucet_000001'
    spec['instruction'] = 'Wash the dirty bowl under the faucet.'
    spec['roles']['target_object']['synset'] = 'bowl.n.01'
    spec['roles']['target_object']['bddl_inst'] = 'bowl.n.01_1'
    spec['roles']['target_object']['category'] = 'bowl'
    spec['roles']['target_object']['abilities'] = ['pickupable', 'soakable']
    spec['roles']['target_object']['init_states'] = {'dirty': True, 'saturated': {}}
    spec['roles']['faucet'] = spec['roles'].pop('source')
    spec['roles']['faucet']['synset'] = 'faucet.n.01'
    spec['roles']['faucet']['bddl_inst'] = 'faucet.n.01_1'
    spec['roles']['faucet']['category'] = 'faucet'
    spec['roles']['faucet']['abilities'] = ['water_source', 'toggleable']
    spec['roles']['faucet']['init_states'] = {'toggled_on': False}
    spec['init'] = [{'predicate': 'dirty', 'args': ['target_object']}, {'predicate': 'toggled_off', 'args': ['faucet']}]
    spec['family_spec'] = {'recover': {'incident': {'type': 'dirty_dish', 'object_role': 'target_object', 'support_roles': {'faucet': 'faucet'}}}}
    spec['goal']['predicates'] = [{'type': 'object_clean', 'args': {'object': 'target_object'}}]
    spec['goal']['require_model_stop'] = True
    env, result = _run(spec)
    assert result.status == 'passed'
    assert result.final_goal['task_success'] == 1.0
    tools = [t['tool_name'] for t in result.transitions if t['tool_name'] != 'stop']
    assert tools == ['navigate_to', 'pick_up', 'navigate_to', 'toggle_on', 'soak_under', 'toggle_off']
    assert not any((t['arguments'].get('target') == 'rag' for t in result.transitions if isinstance(t.get('arguments'), dict)))
    assert env.core.objects[env.role_map['target_object']].states['dirty'] is False
    assert env.core.objects[env.role_map['faucet']].states['toggled_on'] is False

def test_recovery_oracle_solves_filled_and_soaked():
    for task_id in ('filled', 'soaked'):
        _, result = _run(stage5_task_by_id(task_id))
        assert result.status == 'passed'
        assert result.final_goal['task_success'] == 1.0
