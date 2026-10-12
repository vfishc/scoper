import pytest
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.benchmark.behavior.brains.base import BrainError, BrainInput
from scoper.benchmark.behavior.brains.oracle import OracleBrain
from tests.behavior.conftest import build_stage2_task_spec

def _reset_env(spec):
    env = BehaviorClosedLoopToolEnv()
    env.reset_to_task(spec)
    return env

def test_plan_place_sequence():
    spec = build_stage2_task_spec()
    env = _reset_env(spec)
    brain = OracleBrain()
    brain.on_episode_start(env, spec)
    seq = [(c.tool_name, c.arguments.get('target')) for c in brain._queue]
    assert ('navigate_to', 'target_object') in seq
    assert ('pick_up', 'target_object') in seq
    assert ('put_down', 'target_receptacle') in seq
    assert ('open', 'fridge') in seq
    assert seq.index(('pick_up', 'target_object')) < seq.index(('put_down', 'target_receptacle'))

def test_oracle_emits_stop_when_queue_empty():
    spec = build_stage2_task_spec()
    env = _reset_env(spec)
    brain = OracleBrain()
    brain.on_episode_start(env, spec)
    brain._queue = []
    call = brain.act(BrainInput(observation={}, prompt_context={}, last_transition=None, high_level_step=1))
    assert call.tool_name == 'stop'

def test_unsupported_predicate_raises():
    spec = build_stage2_task_spec()
    spec['goal']['predicates'] = [{'type': 'object_dirty', 'args': {'object': 'target_object'}}]
    env = _reset_env(spec)
    with pytest.raises(BrainError):
        OracleBrain().on_episode_start(env, spec)

def test_open_skipped_when_already_open():
    spec = build_stage2_task_spec()
    spec['roles']['target_receptacle']['abilities'] = ['support_surface', 'openable']
    spec['roles']['target_receptacle']['init_states'] = {'open': True}
    env = _reset_env(spec)
    brain = OracleBrain()
    brain.on_episode_start(env, spec)
    seq = [(c.tool_name, c.arguments.get('target')) for c in brain._queue]
    assert ('open', 'target_receptacle') not in seq
