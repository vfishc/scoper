import pytest
from scoper.benchmark.behavior.execution_core.api import make_core
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose
from scoper.benchmark.behavior.execution_core.navigation import NavigationConfig
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject, OccupancyConfig
from scoper.benchmark.behavior.behavior_executor import ACTION_ROUTING, BehaviorExecutor, BehaviorExecutorError

def _core():
    objs = [AABBObject('floors_1', 'floors', (5.0, 3.0, -0.02), (10.0, 6.0, 0.04)), AABBObject('breakfast_table.n.01_1', 'breakfast_table', (6.5, 4.5, 0.45), (0.7, 0.7, 0.9), abilities=('support_surface',)), AABBObject('mug.n.01_1', 'mug', (5.1, 1.1, 0.25), (0.16, 0.16, 0.18), abilities=('pickupable',), states={})]
    core = make_core(objs, EgoPose(2.0, 2.5, yaw=0.0), OccupancyConfig(resolution=0.1, agent_radius=0.2, safety_margin=0.05, bounds=(-0.5, -0.5, 10.5, 6.5)), NavigationConfig(frame_spacing=0.2, candidate_count=96))
    core.build_occupancy('occ')
    return core

def test_action_routing_known_and_deferred():
    for tool in ('navigate_to', 'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'wipe', 'cut', 'soak_under', 'fill', 'pour', 'heat'):
        assert tool in ACTION_ROUTING
    assert 'soak_inside' not in ACTION_ROUTING

def test_unknown_tool_raises():
    executor = BehaviorExecutor(_core())
    with pytest.raises(BehaviorExecutorError):
        executor.execute_action('teleport', target='mug.n.01_1')

def test_navigate_pick_place():
    core = _core()
    executor = BehaviorExecutor(core)
    r1 = executor.execute_action('navigate_to', target='mug.n.01_1')
    assert r1.success and r1.frames > 0
    r2 = executor.execute_action('pick_up', target='mug.n.01_1')
    assert r2.success
    assert core.ego_state.held_object_name == 'mug.n.01_1'
    r3 = executor.execute_action('navigate_to', target='breakfast_table.n.01_1')
    assert r3.success
    r4 = executor.execute_action('put_down', target='breakfast_table.n.01_1')
    assert r4.success
    assert core.ego_state.held_object_name is None

def test_failed_action_reports_error_type():
    core = _core()
    executor = BehaviorExecutor(core)
    result = executor.execute_action('pick_up', target='breakfast_table.n.01_1')
    assert result.success is False
    assert result.error_type is not None

@pytest.mark.parametrize('surface_top', [1.0, 2.2])
def test_place_above_navigation_height_is_not_a_failed_release(surface_top):
    core = _core()
    surface = core.objects['breakfast_table.n.01_1']
    surface.center = (surface.center[0], surface.center[1], surface_top / 2)
    surface.extent = (0.7, 0.7, surface_top)
    executor = BehaviorExecutor(core)
    assert executor.execute_action('navigate_to', target='mug.n.01_1').success
    assert executor.execute_action('pick_up', target='mug.n.01_1').success
    assert executor.execute_action('navigate_to', target=surface.name).success
    result = executor.execute_action('put_down', target=surface.name)
    assert result.success
    assert core.ego_state.held_object_name is None
    assert core.objects['mug.n.01_1'].center[2] == pytest.approx(surface_top + 0.09)
    rasterized = {row['name'] for row in core.occupancy.metadata['rasterized_objects']}
    assert ('mug.n.01_1' in rasterized) is (surface_top < 2.0)

def test_stream_callback_fires_once_per_frame():
    core = _core()
    seen = []
    executor = BehaviorExecutor(core, stream_callback=lambda pose, i: seen.append(i))
    result = executor.execute_action('navigate_to', target='mug.n.01_1')
    assert len(seen) == result.frames
    assert result.frames > 0

def test_soak_under_cleans_only_explicit_staged_dirty_object_near_source():
    objs = [AABBObject('floors_1', 'floors', (2.5, 2.5, -0.02), (5.0, 5.0, 0.04)), AABBObject('faucet.n.01_1', 'faucet', (2.0, 2.0, 0.7), (0.16, 0.16, 0.2), abilities=('toggleable', 'water_source'), states={'toggled_on': True, 'source_substance': 'water'}), AABBObject('bowl.n.01_1', 'bowl', (2.1, 2.0, 0.2), (0.22, 0.22, 0.12), abilities=('pickupable', 'soakable'), states={'dirty': True, 'saturated': {}}), AABBObject('plate.n.01_1', 'plate', (2.05, 2.0, 0.2), (0.22, 0.22, 0.06), abilities=('pickupable', 'soakable'), states={'dirty': True, 'saturated': {}})]
    core = make_core(objs, EgoPose(0.8, 0.8, yaw=0.0), OccupancyConfig(resolution=0.1, agent_radius=0.2, safety_margin=0.05, bounds=(0.0, 0.0, 5.0, 5.0)), NavigationConfig(frame_spacing=0.2, candidate_count=96))
    executor = BehaviorExecutor(core)
    nav = executor.execute_action('navigate_to', target='faucet.n.01_1')
    assert nav.success
    result = executor.execute_action('soak_under', target='faucet.n.01_1', staged_target='bowl.n.01_1')
    assert result.success is True
    assert core.ego_state.held_object_name is None
    assert core.objects['bowl.n.01_1'].states['dirty'] is False
    assert core.objects['plate.n.01_1'].states['dirty'] is True
    assert result.action_result.state_delta['cleaning']['mode'] == 'staged_sink_wash'

def test_soak_under_readiness_exception_fails_closed_without_state_change(monkeypatch):
    objs = [AABBObject('floors_1', 'floors', (2.5, 2.5, -0.02), (5.0, 5.0, 0.04)), AABBObject('faucet.n.01_1', 'faucet', (2.0, 2.0, 0.7), (0.16, 0.16, 0.2), abilities=('toggleable', 'water_source'), states={'toggled_on': True, 'source_substance': 'water'}), AABBObject('bowl.n.01_1', 'bowl', (2.1, 2.0, 0.2), (0.22, 0.22, 0.12), abilities=('pickupable', 'soakable'), states={'dirty': True, 'saturated': {}})]
    core = make_core(objs, EgoPose(1.8, 1.8, yaw=0.0), OccupancyConfig(resolution=0.1, agent_radius=0.2, safety_margin=0.05, bounds=(0.0, 0.0, 5.0, 5.0)), NavigationConfig(frame_spacing=0.2, candidate_count=96))

    class FailingChecker:

        def check_interaction(self, *_args, **_kwargs):
            raise RuntimeError('private geometry failure')
    monkeypatch.setattr(core, 'make_precondition_checker', lambda: FailingChecker())
    executor = BehaviorExecutor(core)
    result = executor.execute_action('soak_under', target='faucet.n.01_1', staged_target='bowl.n.01_1')
    assert result.success is False
    assert result.error_type == 'staged_source_readiness_check_failed'
    assert core.objects['bowl.n.01_1'].states['dirty'] is True
    assert core.objects['bowl.n.01_1'].states['saturated'] == {}

def test_soak_under_without_explicit_staged_target_does_not_guess_nearest_object():
    core = _core()
    core.objects['mug.n.01_1'].states.update({'dirty': True, 'saturated': {}})
    core.objects['mug.n.01_1'].center = (2.1, 2.0, 0.2)
    core.objects['faucet.n.01_1'] = AABBObject('faucet.n.01_1', 'faucet', (2.0, 2.0, 0.7), (0.16, 0.16, 0.2), abilities=('toggleable', 'water_source'), states={'toggled_on': True, 'source_substance': 'water'})
    executor = BehaviorExecutor(core)
    result = executor.execute_action('soak_under', target='faucet.n.01_1')
    assert result.success is False
    assert result.error_type == 'staged_target_unresolved'
    assert core.objects['mug.n.01_1'].states['dirty'] is True

def test_soak_under_held_object_path_is_unchanged():
    objs = [AABBObject('floors_1', 'floors', (2.5, 2.5, -0.02), (5.0, 5.0, 0.04)), AABBObject('faucet.n.01_1', 'faucet', (3.5, 3.5, 0.7), (0.16, 0.16, 0.2), abilities=('toggleable', 'water_source'), states={'toggled_on': True, 'source_substance': 'water'}), AABBObject('bowl.n.01_1', 'bowl', (1.4, 1.4, 0.2), (0.22, 0.22, 0.12), abilities=('pickupable', 'soakable'), states={'dirty': True, 'saturated': {}})]
    core = make_core(objs, EgoPose(0.8, 0.8, yaw=0.0), OccupancyConfig(resolution=0.1, agent_radius=0.2, safety_margin=0.05, bounds=(0.0, 0.0, 5.0, 5.0)), NavigationConfig(frame_spacing=0.2, candidate_count=96))
    executor = BehaviorExecutor(core)
    assert executor.execute_action('navigate_to', target='bowl.n.01_1').success
    assert executor.execute_action('pick_up', target='bowl.n.01_1').success
    assert executor.execute_action('navigate_to', target='faucet.n.01_1').success
    result = executor.execute_action('soak_under', target='faucet.n.01_1')
    assert result.success is True
    assert core.ego_state.held_object_name == 'bowl.n.01_1'
    assert core.objects['bowl.n.01_1'].states['dirty'] is False
