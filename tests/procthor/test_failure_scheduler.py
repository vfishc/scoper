from __future__ import annotations
import unittest
from types import SimpleNamespace
from scoper.benchmark.procthor.brains.base import ToolCall
from scoper.benchmark.procthor.brains.scripted import ScriptedBrain
from scoper.benchmark.procthor.episode_runner import EpisodeRunner
from scoper.benchmark.procthor.scheduler import FailureConfig, FailureScheduler
from tests.procthor.fake_env import RichFakeToolEnv

def _task(category: str='static_simple'):
    return {'id': 'procthor_static_simple_test001', 'category': category, 'instruction': 'test', 'source': {'house_id': 'fake_house'}, 'roles': {'target_object': {'object_id': 'Mug|1'}, 'target_receptacle': {'object_id': 'Sink|1'}}, 'goal': {'predicates': [], 'require_model_stop': False}}

def _objects():
    return [{'objectId': 'Mug|1', 'objectType': 'Mug', 'visible': True}, {'objectId': 'Mug|2', 'objectType': 'Mug', 'visible': True}, {'objectId': 'Sink|1', 'objectType': 'Sink', 'visible': True, 'openable': False}, {'objectId': 'Fridge|1', 'objectType': 'Fridge', 'openable': True, 'isOpen': False, 'visible': True}]

def _make_env(success_after=2):
    return RichFakeToolEnv(objects=_objects(), roles={'target_object': 'Mug|1', 'target_receptacle': 'Sink|1', 'source_container': 'Fridge|1'}, success_after=success_after)

class FailureConfigValidationTests(unittest.TestCase):

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            FailureConfig(tool='navigate_to', mode='bad_mode')

    def test_unknown_tool(self):
        with self.assertRaises(ValueError):
            FailureConfig(tool='unknown_tool', mode='block')

    def test_wrong_target_requires_specific_tool(self):
        with self.assertRaises(ValueError):
            FailureConfig(tool='any', mode='wrong_target')

    def test_nav_modes_require_navigate_to(self):
        with self.assertRaisesRegex(ValueError, 'nav_blocked_partway'):
            FailureConfig(tool='pick_up', mode='nav_blocked_partway')
        with self.assertRaisesRegex(ValueError, 'nav_wrong_target'):
            FailureConfig(tool='put_down', mode='nav_wrong_target')
        FailureConfig(tool='navigate_to', mode='nav_blocked_partway')
        FailureConfig(tool='navigate_to', mode='nav_wrong_target')

class BlockModeTests(unittest.TestCase):

    def test_target_role_gate_compares_prebound_object_id(self):
        env = _make_env(success_after=99)
        env.prebind_synthetic_operation = lambda _tool, _args: SimpleNamespace(object_id='Mug|1', binding_public_record={'status': 'success'})
        result = EpisodeRunner(env, require_passed_status=False).run(_task(), brain=ScriptedBrain([ToolCall('navigate_to', {'target': 'mug_alias'}), ToolCall('stop', {})]), scheduler=FailureScheduler([{'tool': 'navigate_to', 'mode': 'block', 'target_role': 'target_object'}], seed=0))
        self.assertFalse(result.transitions[0]['success'])

    def test_block_navigate_to_does_not_advance_env(self):
        env = _make_env()
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'navigate_to', 'mode': 'block', 'probability': 1.0}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        self.assertEqual(env.high_level_step, 1)
        self.assertFalse(result.transitions[0]['success'])
        self.assertEqual(result.transitions[0]['raw_execution']['injected_failure'], 'block')
        self.assertEqual(result.transitions[0]['tool_name'], 'navigate_to')

    def test_probability_zero_never_intercepts(self):
        env = _make_env(success_after=1)
        runner = EpisodeRunner(env)
        brain = ScriptedBrain([ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'any', 'mode': 'block', 'probability': 0.0}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        self.assertEqual(result.status, 'passed')

    def test_after_step_gate(self):
        env = _make_env(success_after=99)
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'navigate_to', 'mode': 'block', 'after_step': 2}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        self.assertTrue(result.transitions[0]['success'])
        self.assertFalse(result.transitions[1]['success'])
        self.assertEqual(result.transitions[1]['raw_execution'].get('injected_failure'), 'block')

    def test_target_role_gate(self):
        env = _make_env(success_after=99)
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('navigate_to', {'target': 'target_receptacle'}), ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'navigate_to', 'mode': 'block', 'target_role': 'target_object'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        self.assertTrue(result.transitions[0]['success'])
        self.assertFalse(result.transitions[1]['success'])
        self.assertEqual(result.transitions[1]['raw_execution'].get('injected_failure'), 'block')

class ReportFailModeTests(unittest.TestCase):

    def test_report_fail_runs_underlying_action(self):
        env = _make_env(success_after=99)
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'navigate_to', 'mode': 'report_fail'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        self.assertEqual(env.high_level_step, 2)
        self.assertFalse(result.transitions[0]['success'])
        self.assertIn('reported failure', result.transitions[0]['env_feedback'])

class DelayedSuccessModeTests(unittest.TestCase):

    def test_delayed_notice_appears_at_next_pre_step(self):
        env = _make_env(success_after=99)
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'navigate_to', 'mode': 'delayed_success', 'max_count': 1}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        self.assertFalse(result.transitions[0]['success'])
        self.assertTrue(result.transitions[1]['success'])

class PartialExecutionTests(unittest.TestCase):

    def test_open_partial_sets_intermediate_openness(self):
        env = _make_env()
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('open', {'target': 'source_container'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'open', 'mode': 'partial_execution'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        self.assertFalse(result.transitions[0]['success'])
        self.assertIn('openness=', result.transitions[0]['env_feedback'])
        forced = [s for s in env.controller.step_history if s.get('action') == 'OpenObject' and 'openness' in s]
        self.assertTrue(forced, 'expected at least one OpenObject step with openness')

    def test_toggle_partial_falls_back_to_report_fail(self):
        env = _make_env(success_after=99)
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('toggle_on', {'target': 'source_container'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'toggle_on', 'mode': 'partial_execution'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        self.assertEqual(env.high_level_step, 2)
        self.assertFalse(result.transitions[0]['success'])

class WrongTargetTests(unittest.TestCase):

    def test_wrong_target_swaps_to_same_type(self):
        env = _make_env()
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'navigate_to', 'mode': 'wrong_target'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        first = env.tool_history[0]
        self.assertEqual(first['tool_name'], 'navigate_to')
        self.assertEqual(first['arguments']['target'], 'target_object')
        event = next((ev for ev in result.scheduler_events if ev['name'] == 'failure_wrong_target'))
        self.assertEqual(event['payload']['swapped_target'], 'Mug|2')

    def test_nav_wrong_target_has_distinct_event_name(self):
        env = _make_env()
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'navigate_to', 'mode': 'nav_wrong_target'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        first = env.tool_history[0]
        self.assertEqual(first['arguments']['target'], 'target_object')
        self.assertEqual([ev['name'] for ev in result.scheduler_events if ev['name'] == 'failure_nav_wrong_target'], ['failure_nav_wrong_target'])
        event = next((ev for ev in result.scheduler_events if ev['name'] == 'failure_nav_wrong_target'))
        self.assertEqual(event['payload']['swapped_target'], 'Mug|2')

class NavBlockedPartwayTests(unittest.TestCase):

    def test_wrong_target_uses_internal_navigation_and_requires_displacement(self):
        env = _make_env()
        calls = []

        def navigate(object_id, arguments, reasoning=''):
            calls.append((object_id, dict(arguments)))
            env.controller.last_event.metadata['agent']['position']['x'] = 1.0
            return {'success': True}
        env._call_scheduler_navigation = navigate
        scheduler = FailureScheduler([{'tool': 'navigate_to', 'mode': 'nav_wrong_target'}], seed=0)
        scheduler.on_episode_start(env, _task())
        result = scheduler.intercept_tool_call(env, ToolCall('navigate_to', {'target': 'target_object'}), 1, lambda: {})
        assert calls == [('Mug|2', {'target': 'target_object'})]
        payload = result['scheduler_intercept'][0].payload
        assert payload['injection_applied'] is True
        assert payload['displacement_m'] == 1.0
        scheduler.on_episode_start(env, _task())
        result = scheduler.intercept_tool_call(env, ToolCall('navigate_to', {'target': 'target_object'}), 1, lambda: {})
        assert result['scheduler_intercept'][0].payload['injection_applied'] is False

    def test_nav_blocked_partway_reports_failure_event(self):
        env = _make_env()
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('navigate_to', {'target': 'target_object'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'navigate_to', 'mode': 'nav_blocked_partway'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        self.assertFalse(result.transitions[0]['success'])
        self.assertIn('stopped before reaching', result.transitions[0]['env_feedback'])
        events = [ev for ev in result.scheduler_events if ev['name'] == 'failure_nav_blocked_partway']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['payload']['tool'], 'navigate_to')

class RealisticFailureModeValidationTests(unittest.TestCase):

    def test_grasp_slip_requires_pick_up(self):
        with self.assertRaisesRegex(ValueError, 'grasp_slip'):
            FailureConfig(tool='put_down', mode='grasp_slip')
        with self.assertRaisesRegex(ValueError, 'grasp_slip'):
            FailureConfig(tool='any', mode='grasp_slip')
        FailureConfig(tool='pick_up', mode='grasp_slip')

    def test_drop_beside_requires_put_down(self):
        with self.assertRaisesRegex(ValueError, 'drop_beside'):
            FailureConfig(tool='pick_up', mode='drop_beside')
        FailureConfig(tool='put_down', mode='drop_beside')

    def test_partial_open_requires_open(self):
        with self.assertRaisesRegex(ValueError, 'partial_open'):
            FailureConfig(tool='close', mode='partial_open')
        FailureConfig(tool='open', mode='partial_open')

    def test_partial_close_requires_close(self):
        with self.assertRaisesRegex(ValueError, 'partial_close'):
            FailureConfig(tool='open', mode='partial_close')
        FailureConfig(tool='close', mode='partial_close')

class GraspSlipTests(unittest.TestCase):

    def test_underlying_pick_failure_does_not_consume_injection(self):
        env = _make_env()
        env.reset_to_task(_task())
        sched = FailureScheduler([{'tool': 'pick_up', 'mode': 'grasp_slip'}], seed=0)
        sched.on_episode_start(env, _task())
        call = ToolCall('pick_up', {'target': 'target_object'})
        transition = sched.intercept_tool_call(env, call, 1, lambda: {'success': False, 'env_feedback': 'normal pick failure'})
        self.assertFalse(transition['scheduler_intercept'][0].payload['injection_applied'])
        self.assertEqual(sched._fire_counts[0], 0)

    def test_grasp_slip_drops_object_on_floor(self):
        env = _make_env()
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('pick_up', {'target': 'target_object'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'pick_up', 'mode': 'grasp_slip'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        tr = result.transitions[0]
        self.assertFalse(tr['success'])
        self.assertIn('slipped', tr['env_feedback'])
        drop_calls = [s for s in env.controller.step_history if s.get('action') == 'DropHandObject']
        self.assertEqual(len(drop_calls), 1)
        sched_events = [ev for ev in result.scheduler_events if ev.get('name') == 'failure_grasp_slip']
        self.assertEqual(len(sched_events), 1)
        self.assertTrue(sched_events[0]['payload']['underlying_success'])

    def test_grasp_slip_releases_visual_attachment(self):
        env = _make_env()
        env._visual_attached_object_id = 'Mug|1'
        calls = []

        def teleport_visual_object(**kwargs):
            calls.append(dict(kwargs))
            return {'success': True}

        def clear_visual_attachment_state(*, update_metadata):
            self.assertTrue(update_metadata)
            env._visual_attached_object_id = None
        sticky = []
        env._teleport_visual_object = teleport_visual_object
        env._clear_visual_attachment_state = clear_visual_attachment_state
        env._set_sticky_object_state = lambda object_id, **values: sticky.append((object_id, values))
        original_step = env.controller.step

        def fail_physical_drop(**kwargs):
            event = original_step(**kwargs)
            if kwargs.get('action') == 'DropHandObject':
                event.metadata['lastActionSuccess'] = False
            return event
        env.controller.step = fail_physical_drop
        runner = EpisodeRunner(env, require_passed_status=False)
        result = runner.run(_task(), brain=ScriptedBrain([ToolCall('pick_up', {'target': 'target_object'}), ToolCall('stop', {})]), scheduler=FailureScheduler([{'tool': 'pick_up', 'mode': 'grasp_slip'}], seed=0))
        event = next((row for row in result.scheduler_events if row.get('name') == 'failure_grasp_slip'))
        self.assertTrue(event['payload']['drop_success'])
        self.assertIsNone(env._visual_attached_object_id)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['record_action'], 'VisualFailureDrop')
        self.assertEqual(sticky, [('Mug|1', {'parentReceptacles': []})])

class DropBesideTests(unittest.TestCase):

    def setUp(self):
        self.env = _make_env()
        self.env.current_house = {'rooms': [{'floorPolygon': [{'x': -3, 'y': 0, 'z': -3}, {'x': 3, 'y': 0, 'z': -3}, {'x': 3, 'y': 0, 'z': 3}, {'x': -3, 'y': 0, 'z': 3}]}]}
        for index, obj in enumerate(self.env.controller._objects.values()):
            center = {'x': 0 if index == 0 else 2, 'y': 1, 'z': 1 if index == 0 else 0.75}
            if obj['objectId'] == 'Sink|1':
                center = {'x': 0, 'y': 0.7, 'z': 1.5}
            obj.update(position=center, rotation={'x': 0, 'y': 0, 'z': 0}, axisAlignedBoundingBox={'center': center, 'size': {'x': 0.2, 'y': 0.2, 'z': 0.2}})
        self.env.controller._sync_event_objects()
        self.env.executor.reachable_positions = lambda: [{'x': index_x * 0.25, 'y': 0.9, 'z': index_z * 0.25} for index_x in range(-8, 9) for index_z in range(-8, 9)]

    def test_no_safe_drop_does_not_release_object(self):
        env = self.env
        env.controller.last_event.metadata['inventoryObjects'] = [{'objectId': 'Mug|1'}]
        env.executor.reachable_positions = lambda: []
        result = EpisodeRunner(env, require_passed_status=False).run(_task(), brain=ScriptedBrain([ToolCall('put_down', {'target': 'target_receptacle'})]), scheduler=FailureScheduler([{'tool': 'put_down', 'mode': 'drop_beside'}], seed=0))
        event = next((row for row in result.scheduler_events if row.get('name') == 'failure_drop_beside'))
        self.assertFalse(event['payload']['injection_applied'])
        self.assertFalse(any((row.get('action') == 'DropHandObject' for row in env.controller.step_history)))

    def test_drop_avoids_obstacle_and_respects_object_bottom(self):
        from scoper.benchmark.procthor.scheduler.drop_placement import bounds, overlaps, select_drop_position
        env = self.env
        obstacle = env.controller._objects['Mug|2']
        obstacle['axisAlignedBoundingBox'] = {'center': {'x': 0, 'y': 0.1, 'z': 0.75}, 'size': {'x': 0.3, 'y': 0.2, 'z': 0.3}}
        env.controller._sync_event_objects()
        position, report = select_drop_position(env, 'Mug|1', 'Sink|1')
        self.assertIsNotNone(position)
        self.assertFalse(overlaps(report['bounds'], bounds(obstacle)))
        self.assertAlmostEqual(position['y'], 0.11)
        self.assertEqual(report['status'], 'passed')

    def test_drop_can_select_behind_agent_beyond_old_radius(self):
        from scoper.benchmark.procthor.scheduler.drop_placement import select_drop_position
        env = self.env
        env.current_house = {'rooms': [{'floorPolygon': [{'x': -1, 'y': 0, 'z': -2.5}, {'x': 1, 'y': 0, 'z': -2.5}, {'x': 1, 'y': 0, 'z': -1.5}, {'x': -1, 'y': 0, 'z': -1.5}]}]}
        target = env.controller._objects['Sink|1']
        target['axisAlignedBoundingBox']['center'] = {'x': 0, 'y': 0.7, 'z': -2}
        env.controller._sync_event_objects()
        position, report = select_drop_position(env, 'Mug|1', 'Sink|1')
        self.assertIsNotNone(position)
        self.assertLess(position['z'], -1.5)
        self.assertIn('recovery_stance', report)

    def test_wall_polygon_takes_precedence_over_aggregate_bounds(self):
        from scoper.benchmark.procthor.scheduler.drop_placement import select_drop_position
        env = self.env
        env.current_house['walls'] = [{'id': 'wall', 'polygon': [{'x': 2.9, 'y': 0, 'z': -3}, {'x': 2.9, 'y': 0, 'z': 3}, {'x': 2.9, 'y': 3, 'z': 3}, {'x': 2.9, 'y': 3, 'z': -3}]}]
        env.controller._objects['wall'] = {'objectId': 'wall', 'objectType': 'Wall', 'axisAlignedBoundingBox': {'center': {'x': 0, 'y': 1, 'z': 0}, 'size': {'x': 6, 'y': 3, 'z': 6}}}
        env.controller._sync_event_objects()
        position, report = select_drop_position(env, 'Mug|1', 'Sink|1')
        self.assertIsNotNone(position)
        self.assertEqual(report['status'], 'passed')

    def test_drop_beside_releases_visual_attachment_without_native_drop(self):
        env = self.env
        env.controller.last_event.metadata['inventoryObjects'] = [{'objectId': 'Mug|1'}]
        env._visual_attached_object_id = 'Mug|1'
        calls = []
        sticky = []

        def teleport_visual_object(**kwargs):
            calls.append(kwargs)
            return {'success': True}

        def clear_visual_attachment_state(*, update_metadata):
            self.assertTrue(update_metadata)
            env._visual_attached_object_id = None
            env.controller.last_event.metadata['inventoryObjects'] = []
        env._teleport_visual_object = teleport_visual_object
        env._clear_visual_attachment_state = clear_visual_attachment_state
        env._set_sticky_object_state = lambda object_id, **values: sticky.append((object_id, values))
        result = EpisodeRunner(env, require_passed_status=False).run(_task(), brain=ScriptedBrain([ToolCall('put_down', {'target': 'target_receptacle'}), ToolCall('stop', {})]), scheduler=FailureScheduler([{'tool': 'put_down', 'mode': 'drop_beside'}], seed=0))
        event = next((row for row in result.scheduler_events if row.get('name') == 'failure_drop_beside'))
        self.assertTrue(event['payload']['injection_applied'])
        self.assertIsNone(env._visual_attached_object_id)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sticky, [('Mug|1', {'parentReceptacles': []})])
        self.assertFalse(any((row.get('action') == 'DropHandObject' for row in env.controller.step_history)))

    def test_drop_beside_drops_held_object_near_receptacle(self):
        env = self.env
        env.controller.last_event.metadata['inventoryObjects'] = [{'objectId': 'Mug|1'}]
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('put_down', {'target': 'target_receptacle'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'put_down', 'mode': 'drop_beside'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        tr = result.transitions[0]
        self.assertFalse(tr['success'])
        self.assertIn('landed on the floor', tr['env_feedback'])
        drop_calls = [s for s in env.controller.step_history if s.get('action') == 'DropHandObject']
        self.assertEqual(len(drop_calls), 1)
        teleport_calls = [s for s in env.controller.step_history if s.get('action') == 'TeleportObject']
        self.assertEqual(len(teleport_calls), 1)
        self.assertEqual(teleport_calls[0]['objectId'], 'Mug|1')
        sched_events = [ev for ev in result.scheduler_events if ev.get('name') == 'failure_drop_beside']
        self.assertEqual(len(sched_events), 1)

class PartialOpenTests(unittest.TestCase):

    def test_partial_open_sets_openness_to_0_05(self):
        env = _make_env()
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('open', {'target': 'source_container'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'open', 'mode': 'partial_open'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        tr = result.transitions[0]
        self.assertFalse(tr['success'])
        self.assertIn('jammed', tr['env_feedback'])
        open_calls = [s for s in env.controller.step_history if s.get('action') == 'OpenObject' and s.get('openness') == 0.05]
        self.assertEqual(len(open_calls), 1)

class PartialCloseTests(unittest.TestCase):

    def test_partial_close_sets_openness_to_0_6(self):
        env = _make_env()
        runner = EpisodeRunner(env, require_passed_status=False)
        brain = ScriptedBrain([ToolCall('close', {'target': 'source_container'}), ToolCall('stop', {})])
        sched = FailureScheduler([{'tool': 'close', 'mode': 'partial_close'}], seed=0)
        result = runner.run(_task(), brain=brain, scheduler=sched)
        tr = result.transitions[0]
        self.assertFalse(tr['success'])
        self.assertIn('jammed', tr['env_feedback'])
        open_calls = [s for s in env.controller.step_history if s.get('action') == 'OpenObject' and s.get('openness') == 0.6]
        self.assertEqual(len(open_calls), 1)
if __name__ == '__main__':
    unittest.main()
