import copy
import json
import math
import os
import tempfile
import unittest
from scoper.benchmark.common import ConfirmationIntent, compile_task_binding_spec
from scoper.benchmark.procthor.closed_loop_tool_env import ProcTHORClosedLoopToolEnv, ProcTHORClosedLoopToolEnvError, get_tool_specs
from scoper.benchmark.procthor.closed_loop_tool_validator import run_closed_loop_tool_validation
from scoper.benchmark.procthor.nl_target_resolver import ResolvedTarget

def _obj(object_id, object_type=None, **kwargs):
    data = {'objectId': object_id, 'objectType': object_type or object_id.split('|', 1)[0], 'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'visible': False, 'pickupable': False, 'receptacle': False, 'openable': False, 'toggleable': False, 'sliceable': False, 'parentReceptacles': [], 'distance': 10.0}
    data.update(kwargs)
    return data

def _metadata():
    return {'lastActionSuccess': True, 'errorMessage': '', 'objects': [_obj('Mug|0', 'Mug', pickupable=True, dirtyable=True, isDirty=False, position={'x': 1.75, 'y': 0.9, 'z': 0.0}), _obj('Sink|0', 'SinkBasin', receptacle=True, receptacleObjectIds=[], position={'x': 2.0, 'y': 0.9, 'z': 0.0}), _obj('Faucet|0', 'Faucet', toggleable=True, isToggled=False, position={'x': 2.0, 'y': 0.9, 'z': 0.25})], 'inventoryObjects': [], 'arm': {'heldObjects': [], 'pickupableObjects': [], 'handSphereCenter': {'x': 0.4, 'y': 0.9, 'z': 0.0}, 'handSphereRadius': 0.35}, 'agent': {'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'rotation': {'y': 0.0}, 'cameraHorizon': 0.0, 'isStanding': True}}

def _task_spec(require_stop=False):
    return {'id': 'procthor_static_simple_000001', 'version': '0.1', 'backend': 'procthor', 'category': 'static_simple', 'source': {'house_id': 'synthetic_house', 'house_path': '/tmp/synthetic_house.json', 'split': 'train'}, 'template': {'template_id': 'static_simple_place_001', 'template_version': '0.1'}, 'instruction': 'Put the mug in the sink.', 'roles': {'target_object': {'object_id': 'Mug|0', 'object_type': 'Mug'}, 'target_receptacle': {'object_id': 'Sink|0', 'object_type': 'SinkBasin'}, 'sink_basin': {'object_id': 'Sink|0', 'object_type': 'SinkBasin'}, 'faucet': {'object_id': 'Faucet|0', 'object_type': 'Faucet'}}, 'init': {'agent_pose': {'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'rotation': {'y': 0.0}, 'horizon': 0.0}, 'object_state_overrides': []}, 'goal': {'type': 'predicate', 'predicates': [{'type': 'object_in_receptacle', 'object_role': 'target_object', 'receptacle_role': 'target_receptacle'}], 'require_model_stop': require_stop}, 'online_requirements': {'expected_events': [], 'expected_brain_behaviors': []}, 'validation': {'status': 'validated', 'checks': {}, 'failure_reason': None}, 'tags': ['static_simple']}

class _FakeEvent:

    def __init__(self, metadata):
        self.metadata = metadata
        self.frame = [['rgb']]

class _FakeController:

    def __init__(self, metadata):
        self.metadata = copy.deepcopy(metadata)
        self.actions = []
        self.last_event = _FakeEvent(self.metadata)
        self.reachable = [{'x': x / 4.0, 'y': 0.9, 'z': 0.0} for x in range(0, 13)]
        self._update_visibility()

    def step(self, **kwargs):
        self.actions.append(dict(kwargs))
        action = kwargs['action']
        self.metadata['lastActionSuccess'] = True
        self.metadata['errorMessage'] = ''
        if action == 'Pass':
            pass
        elif action == 'TeleportFull':
            self.metadata['agent'] = {'position': dict(kwargs['position']), 'rotation': dict(kwargs['rotation']), 'cameraHorizon': kwargs['horizon'], 'isStanding': bool(kwargs.get('standing', True))}
        elif action == 'GetReachablePositions':
            self.metadata['actionReturn'] = copy.deepcopy(self.reachable)
        elif action == 'GetInteractablePoses':
            self.metadata['actionReturn'] = self._interactable_poses(kwargs['objectId'])
        elif action == 'RotateRight':
            self._rotate(float(kwargs.get('degrees', 90)))
        elif action == 'RotateLeft':
            self._rotate(-float(kwargs.get('degrees', 90)))
        elif action == 'MoveAhead':
            self._move(float(kwargs.get('moveMagnitude', 0.25)))
        elif action == 'PickupObject':
            if 'objectIdCandidates' in kwargs:
                self._pickup_arm(kwargs.get('objectIdCandidates') or [])
            else:
                self._pickup(kwargs['objectId'])
        elif action == 'PutObject':
            self._put(kwargs['objectId'])
        elif action == 'DropHandObject':
            self._drop_hand_object()
        elif action == 'SetObjectPoses':
            self._set_object_poses(kwargs.get('objectPoses') or [])
        elif action == 'SetHandSphereRadius':
            self._set_hand_sphere_radius(float(kwargs.get('radius', 0.35)))
        elif action == 'SetAgentRadius':
            self.metadata['agentRadius'] = float(kwargs.get('agentRadius'))
        elif action == 'MoveArm':
            self._move_arm(kwargs.get('position') or {})
        elif action == 'TeleportObject':
            obj = self._object(kwargs['objectId'])
            obj['position'] = dict(kwargs['position'])
            obj['rotation'] = dict(kwargs.get('rotation') or obj.get('rotation', {}))
            obj['parentReceptacles'] = []
        elif action == 'ToggleObjectOn' or action == 'ToggleObjectOff':
            obj = self._object(kwargs['objectId'])
            obj['isToggled'] = action == 'ToggleObjectOn'
        elif action == 'OpenObject' or action == 'CloseObject':
            obj = self._object(kwargs['objectId'])
            obj['isOpen'] = action == 'OpenObject'
        elif action == 'CleanObject':
            obj = self._object(kwargs['objectId'])
            obj['isDirty'] = False
        elif action == 'FillObjectWithLiquid':
            obj = self._object(kwargs['objectId'])
            obj['isFilledWithLiquid'] = True
            obj['fillLiquid'] = kwargs.get('fillLiquid', 'water')
        elif action == 'LookUp' or action == 'LookDown':
            direction = 1.0 if action == 'LookDown' else -1.0
            self.metadata['agent']['cameraHorizon'] = max(-30.0, min(60.0, float(self.metadata['agent'].get('cameraHorizon', 0.0)) + direction * float(kwargs.get('degrees', 30.0))))
        elif action == 'Stand':
            self.metadata['agent']['isStanding'] = True
        elif action == 'Crouch':
            self.metadata['agent']['isStanding'] = False
        else:
            self._fail('unsupported action')
        self._update_visibility()
        self.last_event = _FakeEvent(self.metadata)
        return self.last_event

    def stop(self):
        pass

    def _interactable_poses(self, object_id):
        obj = self._object(object_id)
        obj_pos = obj['position']
        poses = []
        for point in self.reachable:
            distance = math.hypot(point['x'] - obj_pos['x'], point['z'] - obj_pos['z'])
            if distance <= 1.45:
                yaw = round(math.degrees(math.atan2(obj_pos['x'] - point['x'], obj_pos['z'] - point['z'])) / 90.0) * 90 % 360
                poses.append({'x': point['x'], 'y': point['y'], 'z': point['z'], 'rotation': yaw, 'horizon': 0.0, 'standing': True})
        return poses

    def _rotate(self, degrees):
        rotation = self.metadata['agent']['rotation']
        rotation['y'] = (float(rotation.get('y', 0.0)) + degrees) % 360

    def _move(self, magnitude):
        yaw = round(float(self.metadata['agent']['rotation']['y']) / 90.0) * 90 % 360
        dx, dz = {0: (0.0, magnitude), 90: (magnitude, 0.0), 180: (0.0, -magnitude), 270: (-magnitude, 0.0)}[yaw]
        pos = self.metadata['agent']['position']
        new_pos = {'x': round(pos['x'] + dx, 2), 'y': pos['y'], 'z': round(pos['z'] + dz, 2)}
        if not any((abs(item['x'] - new_pos['x']) < 1e-06 for item in self.reachable)):
            self._fail('move target is not reachable')
            return
        self.metadata['agent']['position'] = new_pos

    def _pickup(self, object_id):
        obj = self._object(object_id)
        if not obj.get('visible'):
            self._fail('object is not visible')
            return
        self.metadata['inventoryObjects'] = [{'objectId': obj['objectId'], 'objectType': obj['objectType']}]
        for parent_id in obj.get('parentReceptacles') or []:
            parent = self._object(parent_id)
            if 'receptacleObjectIds' in parent:
                parent['receptacleObjectIds'] = [item for item in parent.get('receptacleObjectIds', []) if item != object_id]
        obj['parentReceptacles'] = []

    def _pickup_arm(self, object_id_candidates):
        pickupable = set(self.metadata.get('arm', {}).get('pickupableObjects') or [])
        for object_id in object_id_candidates:
            obj = self._object(object_id)
            if object_id not in pickupable and (not obj.get('visible')):
                continue
            self.metadata.setdefault('arm', {})['heldObjects'] = [object_id]
            obj['parentReceptacles'] = []
            return
        self._fail('No objects (specified by objectId) were valid to be picked up by the arm')

    def _put(self, receptacle_id):
        receptacle = self._object(receptacle_id)
        if not receptacle.get('visible'):
            self._fail('receptacle is not visible')
            return
        if not self.metadata['inventoryObjects']:
            self._fail('inventory is empty')
            return
        held_id = self.metadata['inventoryObjects'][0]['objectId']
        self._object(held_id)['parentReceptacles'] = [receptacle_id]
        receptacle.setdefault('receptacleObjectIds', [])
        if held_id not in receptacle['receptacleObjectIds']:
            receptacle['receptacleObjectIds'].append(held_id)
        self.metadata['inventoryObjects'] = []

    def _drop_hand_object(self):
        if not self.metadata['inventoryObjects']:
            arm = self.metadata.get('arm') or {}
            if not arm.get('heldObjects'):
                self._fail('inventory is empty')
                return
            arm['heldObjects'] = []
            return
        self.metadata['inventoryObjects'] = []

    def _set_hand_sphere_radius(self, radius):
        arm = self.metadata.setdefault('arm', {})
        arm['handSphereRadius'] = radius
        agent = self.metadata['agent']
        position = agent['position']
        yaw = math.radians(float(agent['rotation'].get('y', 0.0)))
        arm['handSphereCenter'] = {'x': position['x'] + math.sin(yaw) * 0.4, 'y': position['y'], 'z': position['z'] + math.cos(yaw) * 0.4}

    def _move_arm(self, position):
        if not {'x', 'y', 'z'}.issubset(position):
            self._fail('MoveArm requires x, y, z')
            return
        self.metadata.setdefault('arm', {})['handSphereCenter'] = {'x': float(position['x']), 'y': float(position['y']), 'z': float(position['z'])}

    def _set_object_poses(self, object_poses):
        for pose in object_poses:
            object_name = pose.get('objectName')
            for obj in self.metadata['objects']:
                if object_name in {obj.get('name'), obj.get('objectId')}:
                    obj['position'] = dict(pose.get('position') or obj['position'])
                    obj['rotation'] = dict(pose.get('rotation') or obj.get('rotation', {}))
                    break

    def _update_visibility(self):
        agent_pos = self.metadata['agent']['position']
        yaw = round(float(self.metadata['agent']['rotation'].get('y', 0.0)) / 90.0) * 90 % 360
        for obj in self.metadata['objects']:
            obj_pos = obj['position']
            dx = obj_pos['x'] - agent_pos['x']
            dz = obj_pos['z'] - agent_pos['z']
            distance = math.hypot(dx, dz)
            desired_yaw = round(math.degrees(math.atan2(dx, dz)) / 90.0) * 90 % 360
            obj['distance'] = distance
            obj['visible'] = distance <= 1.45 and desired_yaw == yaw
        arm = self.metadata.get('arm')
        if isinstance(arm, dict):
            center = arm.get('handSphereCenter') or {}
            radius = float(arm.get('handSphereRadius', 0.0))
            pickupable = []
            for obj in self.metadata['objects']:
                if not obj.get('pickupable'):
                    continue
                obj_pos = obj['position']
                distance_to_hand = math.sqrt((obj_pos['x'] - center.get('x', 0.0)) ** 2 + (obj_pos['y'] - center.get('y', 0.0)) ** 2 + (obj_pos['z'] - center.get('z', 0.0)) ** 2)
                if distance_to_hand <= radius + 0.05:
                    pickupable.append(obj['objectId'])
            arm['pickupableObjects'] = pickupable

    def _fail(self, message):
        self.metadata['lastActionSuccess'] = False
        self.metadata['errorMessage'] = message

    def _object(self, object_id):
        for obj in self.metadata['objects']:
            if obj['objectId'] == object_id:
                return obj
        raise KeyError(object_id)

class _FakeProcTHORToolEnv(ProcTHORClosedLoopToolEnv):

    def __init__(self, **kwargs):
        kwargs.setdefault('width', 64)
        kwargs.setdefault('height', 64)
        super().__init__(**kwargs)
        self.fake_controller = _FakeController(_metadata())

    def _load_house(self, house_path):
        return {'house_path': house_path, 'rooms': [{'id': 'room_0', 'roomType': 'Kitchen', 'floorPolygon': [{'x': -0.5, 'z': -0.5}, {'x': 3.5, 'z': -0.5}, {'x': 3.5, 'z': 0.5}, {'x': -0.5, 'z': 0.5}]}]}

    def _create_controller(self, house):
        return self.fake_controller

class ProcTHORClosedLoopToolEnvTest(unittest.TestCase):

    def test_scheduler_navigation_override_is_consumed_and_public_ids_remain_rejected(self):
        env = _FakeProcTHORToolEnv()
        task = _task_spec()
        task['task_binding_spec'] = compile_task_binding_spec(task).to_dict()
        env.reset_to_task(task)
        result = env._call_scheduler_navigation('Mug|0', {'target': 'the mug'})
        self.assertTrue(result['success'])
        self.assertIsNone(env._scheduler_navigation_target)
        rejected = env.call_tool('navigate_to', {'target': 'Mug|0'})
        self.assertFalse(rejected['success'])
        self.assertEqual(rejected['binding']['reason'], 'private_reference_forbidden')

    def test_v15_public_scope_and_catalog_hide_internal_room_handles(self):
        env = object.__new__(ProcTHORClosedLoopToolEnv)
        env._rooms_for_navigation = [{'room_id': 'room|2', 'room_type': 'kitchen', 'label': 'kitchen'}, {'room_id': 'room|5', 'room_type': 'bathroom', 'label': 'bathroom'}]
        env.current_task_spec = {'family_spec': {'scope': {'type': 'single_room', 'label': 'kitchen', 'room_ids': ['room|2']}}}
        env._composite_active_task_spec = None
        env._current_room_id = 'room|5'
        assert env.public_task_scope() == {'type': 'single_room', 'label': 'kitchen', 'room_ids': ['kitchen']}
        assert env._public_room_id(env._current_room_id) == 'bathroom'
        assert env.public_room_catalog() == [{'room_id': 'kitchen', 'label': 'kitchen', 'room_type': 'kitchen'}, {'room_id': 'bathroom', 'label': 'bathroom', 'room_type': 'bathroom'}]

    def test_confirmation_camera_uses_exact_look_geometry(self):
        target = {'position': {'x': 1.0, 'y': 0.9, 'z': 1.0}}
        intent = ConfirmationIntent('open', 'bound_target', 'drawer')
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        result = env.capture_confirmation(intent, target)
        expected_yaw = 45.0
        expected_horizon = -math.degrees(math.atan2(0.9 - (0.9 + 0.675), math.sqrt(2.0)))
        camera = result['camera']
        self.assertAlmostEqual(camera['requested_yaw'], expected_yaw)
        self.assertAlmostEqual(camera['requested_horizon'], expected_horizon)
        self.assertAlmostEqual(camera['actual_yaw'], expected_yaw)
        self.assertAlmostEqual(camera['actual_horizon'], round(expected_horizon, 1))
        self.assertTrue(camera['used_bound_coordinate'])
        self.assertEqual([record['action'] for record in result['low_level_actions']], ['RotateRight', 'LookDown', 'RotateLeft', 'LookUp'])

    def test_confirmation_video_and_capture_off_end_at_same_pose(self):
        target = {'position': {'x': 1.0, 'y': 0.9, 'z': 1.0}}
        intent = ConfirmationIntent('open', 'bound_target', 'drawer')
        direct_env = _FakeProcTHORToolEnv()
        video_env = _FakeProcTHORToolEnv()
        direct_env.reset_to_task(_task_spec())
        video_env.reset_to_task(_task_spec())
        direct = direct_env.capture_confirmation(intent, target)
        video = video_env.capture_confirmation(intent, target, video_fps=30.0)
        self.assertAlmostEqual(direct['camera']['actual_yaw'], video['camera']['actual_yaw'])
        self.assertAlmostEqual(direct['camera']['actual_horizon'], video['camera']['actual_horizon'])

    def test_synthetic_prebind_preserves_public_binding_and_private_camera_target(self):
        env = _FakeProcTHORToolEnv()
        fridge = _obj('Fridge|0', 'Fridge', openable=True, isOpen=False, position={'x': 1.5, 'y': 0.95, 'z': 0.0})
        env.fake_controller.metadata['objects'].append(fridge)
        env.fake_controller._update_visibility()
        task = _task_spec()
        task['roles']['source_container'] = {'object_id': 'Fridge|0', 'object_type': 'Fridge'}
        task['task_binding_spec'] = compile_task_binding_spec(task).to_dict()
        env.reset_to_task(task)
        navigation = env.call_tool('navigate_to', {'target': 'fridge'})
        self.assertTrue(navigation['success'])
        open_count = sum((action.get('action') == 'OpenObject' for action in env.fake_controller.actions))
        resolved = env.prebind_synthetic_operation('open', {'target': 'fridge'})
        self.assertEqual(resolved.object_id, 'Fridge|0')
        self.assertEqual(resolved.binding_public_record['status'], 'success')
        self.assertEqual(env.last_private_bound_target(), {'position': {'x': 1.5, 'y': 0.95, 'z': 0.0}})
        self.assertEqual(sum((action.get('action') == 'OpenObject' for action in env.fake_controller.actions)), open_count)

    def test_smooth_move_execution_disables_snap_to_grid_by_default(self):
        env = ProcTHORClosedLoopToolEnv(move_execution_step=1.0 / 60.0)
        self.assertFalse(env.snap_to_grid)

    def test_intercepted_transition_is_recorded_once_in_requested_history(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        transition = {'tool_name': 'open', 'arguments': {'target': 'fridge'}, 'high_level_step': 2, 'success': False, 'binding': {'status': 'success', 'canonical_public_ref': 'fridge'}}
        self.assertTrue(env.record_intercepted_transition(transition))
        self.assertFalse(env.record_intercepted_transition(transition))
        self.assertEqual([(item['tool_name'], item['arguments']) for item in env.tool_history], [('open', {'target': 'fridge'})])

    def test_public_placed_open_container_navigation_is_opt_in_and_fail_closed(self):
        task = _task_spec()
        task['task_binding_spec'] = compile_task_binding_spec(task).to_dict()
        resolved = ResolvedTarget(target_ref='target_object', resolved_via='role', object_id='Mug|0', binding_role='target_object')
        calls = []

        def navigate(object_id, **kwargs):
            calls.append(object_id)
            return {'status': 'success', 'object_id': object_id, 'interaction_ready': True}
        disabled = _FakeProcTHORToolEnv()
        disabled.reset_to_task(task)
        disabled.executor.navigate_to_object_id = navigate
        disabled._public_placed_container_by_object['Mug|0'] = 'Fridge|0'
        disabled._public_open_container_ids.add('Fridge|0')
        disabled._execute_resolved_navigation(resolved)
        self.assertEqual(calls.pop(), 'Mug|0')
        enabled = _FakeProcTHORToolEnv(allow_public_placed_open_container_navigation=True)
        enabled.reset_to_task(task)
        enabled.executor.navigate_to_object_id = navigate
        enabled._apply_state_transitions('put_down', {'_held_object_id': 'Mug|0', '_resolved_object_id': 'Fridge|0'})
        enabled._apply_state_transitions('open', {'_resolved_object_id': 'Fridge|0'})
        result = enabled._execute_resolved_navigation(resolved)
        self.assertEqual(calls.pop(), 'Fridge|0')
        self.assertEqual(result['object_id'], 'Mug|0')
        self.assertEqual(result['resolved_via'], 'public_placed_open_container')
        self.assertTrue(result['child_visibility_postcondition_skipped'])
        enabled._set_sticky_object_state('Mug|0', parentReceptacles=[])
        enabled._execute_resolved_navigation(resolved)
        self.assertEqual(calls.pop(), 'Mug|0')

    def test_snap_to_grid_can_be_explicitly_overridden(self):
        env = ProcTHORClosedLoopToolEnv(move_execution_step=1.0 / 60.0, snap_to_grid=True)
        self.assertTrue(env.snap_to_grid)

    def test_binding_metrics_count_receptacle_capable_targets(self):
        env = _FakeProcTHORToolEnv()
        task = _task_spec()
        task['roles'] = {'target_object': task['roles']['target_object'], 'target_receptacle': task['roles']['target_receptacle']}
        task['task_binding_spec'] = compile_task_binding_spec(task).to_dict()
        env.reset_to_task(task)
        with self.assertRaises(ProcTHORClosedLoopToolEnvError):
            env._resolve_operation_target({'target': 'sink'}, tool_name='open')
        env._resolve_navigation_target({'target': 'mug'})
        env._resolve_navigation_target({'target': 'sink'})
        metrics = env._episode_metrics()['binding_metrics']
        self.assertEqual(metrics['attempts'], 3)
        self.assertEqual(metrics['successes'], 2)
        self.assertEqual(metrics['receptacle_binding_attempts'], 2)
        self.assertEqual(metrics['receptacle_binding_successes'], 1)
        self.assertEqual(metrics['receptacle_binding_failures'], 1)

    def test_context_exposes_model_visible_tools_and_role_handles(self):
        env = _FakeProcTHORToolEnv()
        obs = env.reset_to_task(_task_spec())
        context = obs['prompt_context']
        tool_names = [tool['name'] for tool in context['tool_specs']]
        self.assertIn('navigate_to', tool_names)
        self.assertIn('scan', tool_names)
        self.assertIn('answer', tool_names)
        self.assertIn('pick_up', tool_names)
        self.assertIn('target_object', context['tool_target_handles']['role_handles'])
        self.assertIn('object_catalog', context)
        for entry in context['object_catalog']:
            self.assertNotIn('isDirty', entry)
            self.assertNotIn('parentReceptacles', entry)
            self.assertNotIn('visible', entry)

    def test_composite_active_task_overrides_prompt_observation_and_goal(self):
        env = _FakeProcTHORToolEnv()
        parent = _task_spec()
        env.reset_to_task(parent)
        child = copy.deepcopy(parent)
        child['id'] = 'parent::child'
        child['instruction'] = 'Clean the mug.'
        child['goal'] = {'type': 'predicate', 'predicates': [{'type': 'object_clean', 'object_role': 'target_object'}], 'require_model_stop': False}
        env._composite_active_task_spec = child
        self.assertEqual(env.check_goal()['predicate_success'], 1.0)
        self.assertEqual(env._observation()['task_id'], 'parent::child')
        self.assertEqual(env.get_prompt_context()['task_id'], 'parent::child')
        self.assertEqual(env._done_status({'task_success': 1.0, 'require_model_stop': False}), (False, None))
        self.assertEqual(env.current_task_spec['id'], parent['id'])

    def test_dirty_override_uses_sticky_fallback_when_simulator_actions_fail(self):
        env = _FakeProcTHORToolEnv()
        task = _task_spec()
        task['init']['object_state_overrides'] = [{'type': 'set_dirty_state', 'object_role': 'target_object', 'is_dirty': True}]
        env.reset_to_task(task)
        self.assertTrue(env._objects_by_id()['Mug|0']['isDirty'])

    def test_dirty_override_uses_sticky_fallback_when_dirtyable_flag_missing(self):
        env = _FakeProcTHORToolEnv()
        task = _task_spec()
        task['init']['object_state_overrides'] = [{'type': 'set_dirty_state', 'object_role': 'target_object', 'is_dirty': True}]
        env.fake_controller._object('Mug|0')['dirtyable'] = False
        env.reset_to_task(task)
        self.assertTrue(env._objects_by_id()['Mug|0']['isDirty'])

    def test_faucet_clean_clears_sticky_dirty_state(self):
        env = _FakeProcTHORToolEnv()
        task = _task_spec()
        task['init']['object_state_overrides'] = [{'type': 'set_dirty_state', 'object_role': 'target_object', 'is_dirty': True}]
        env.reset_to_task(task)
        env.controller._object('Mug|0')['parentReceptacles'] = ['Sink|0']
        env.controller._object('Sink|0')['receptacleObjectIds'] = ['Mug|0']
        events = env._transition_on_faucet_toggle('Faucet|0')
        self.assertEqual(events[0]['name'], 'faucet_clean')
        self.assertFalse(env._objects_by_id()['Mug|0']['isDirty'])

    def test_faucet_clean_uses_sticky_parent_relation(self):
        env = _FakeProcTHORToolEnv()
        task = _task_spec()
        task['init']['object_state_overrides'] = [{'type': 'set_dirty_state', 'object_role': 'target_object', 'is_dirty': True}]
        env.reset_to_task(task)
        env._set_sticky_object_state('Mug|0', parentReceptacles=['Sink|0'])
        env.controller._object('Sink|0')['receptacleObjectIds'] = []
        events = env._transition_on_faucet_toggle('Faucet|0')
        self.assertEqual(events[0]['name'], 'faucet_clean')
        self.assertFalse(env._objects_by_id()['Mug|0']['isDirty'])

    def test_faucet_clean_does_not_fill_without_filled_goal(self):
        env = _FakeProcTHORToolEnv()
        task = _task_spec()
        env.reset_to_task(task)
        mug = env.controller._object('Mug|0')
        mug['parentReceptacles'] = ['Sink|0']
        mug['dirtyable'] = True
        mug['isDirty'] = True
        mug['canFillWithLiquid'] = True
        mug['isFilledWithLiquid'] = False
        env.controller._object('Sink|0')['receptacleObjectIds'] = ['Mug|0']
        events = env._transition_on_faucet_toggle('Faucet|0')
        self.assertEqual([event['name'] for event in events], ['faucet_clean'])
        self.assertFalse(env._objects_by_id()['Mug|0'].get('isFilledWithLiquid', False))
        self.assertNotIn('FillObjectWithLiquid', [action['action'] for action in env.controller.actions])

    def test_faucet_fill_requires_filled_goal_target(self):
        env = _FakeProcTHORToolEnv()
        task = _task_spec()
        task['goal']['predicates'] = [{'type': 'object_filled', 'object_role': 'target_object', 'liquid': 'water'}]
        env.reset_to_task(task)
        mug = env.controller._object('Mug|0')
        mug['parentReceptacles'] = ['Sink|0']
        mug['canFillWithLiquid'] = True
        mug['isFilledWithLiquid'] = False
        env.controller._object('Sink|0')['receptacleObjectIds'] = ['Mug|0']
        events = env._transition_on_faucet_toggle('Faucet|0')
        self.assertEqual([event['name'] for event in events], ['faucet_fill'])
        self.assertTrue(env._objects_by_id()['Mug|0']['isFilledWithLiquid'])
        self.assertEqual(env._objects_by_id()['Mug|0']['fillLiquid'], 'water')

    def test_navigation_then_pickup_then_put_completes_task(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        nav = env.call_tool('navigate_to', {'target': 'target_object'})
        pickup = env.call_tool('pick_up', {'target': 'target_object'})
        nav_receptacle = env.call_tool('navigate_to', {'target': 'target_receptacle'})
        put = env.call_tool('put_down', {'target': 'target_receptacle'})
        self.assertTrue(nav['success'])
        self.assertTrue(pickup['success'])
        self.assertTrue(nav_receptacle['success'])
        self.assertTrue(put['success'])
        self.assertTrue(put['done'])
        self.assertEqual(put['done_reason'], 'task_success')
        self.assertEqual(put['goal_result']['task_success'], 1.0)
        low_level_actions = [action['action'] for action in nav['low_level_actions']]
        self.assertIn('MoveAhead', low_level_actions)
        self.assertNotIn('TeleportFull', low_level_actions)

    def test_pickup_and_put_update_sticky_parent_relation(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        env.call_tool('navigate_to', {'target': 'target_object'})
        pickup = env.call_tool('pick_up', {'target': 'target_object'})
        env.call_tool('navigate_to', {'target': 'target_receptacle'})
        put = env.call_tool('put_down', {'target': 'target_receptacle'})
        self.assertEqual(pickup['state_transitions'][0]['name'], 'sticky_parent_clear')
        self.assertIn('sticky_parent_set', [event['name'] for event in put['state_transitions']])
        self.assertEqual(env._metadata_with_sticky_object_states()['objects'][0]['parentReceptacles'], ['Sink|0'])

    def test_put_down_logical_fallback_handles_no_valid_positions(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        original_put = env.controller._put
        original_step = env.controller.step

        def _put_without_valid_positions(_receptacle_id):
            env.controller._fail('No valid positions to place object found')

        def _step_with_force_drop_required(**kwargs):
            if kwargs.get('action') != 'DropHandObject':
                return original_step(**kwargs)
            env.controller.actions.append(dict(kwargs))
            if kwargs.get('forceAction'):
                env.controller.metadata['inventoryObjects'] = []
                env.controller.metadata['lastActionSuccess'] = True
                env.controller.metadata['errorMessage'] = ''
            else:
                env.controller._fail('held object collides with agent')
            env.controller.last_event = _FakeEvent(env.controller.metadata)
            return env.controller.last_event
        env.controller._put = _put_without_valid_positions
        env.controller.step = _step_with_force_drop_required
        env.call_tool('navigate_to', {'target': 'target_object'})
        env.call_tool('pick_up', {'target': 'target_object'})
        env.call_tool('navigate_to', {'target': 'target_receptacle'})
        put = env.call_tool('put_down', {'target': 'target_receptacle'})
        env.controller._put = original_put
        env.controller.step = original_step
        self.assertTrue(put['success'])
        self.assertEqual(put['goal_result']['task_success'], 1.0)
        self.assertTrue(put['raw_execution']['logical_put_down_fallback'])
        self.assertIn('DropHandObject', [action['action'] for action in put['low_level_actions']])
        self.assertIn('SetObjectPoses', [action['action'] for action in put['low_level_actions']])
        self.assertTrue(any((action.get('forceAction') and action.get('force_put_object_fallback') and action.get('force_put_object_fallback_applied') for action in put['low_level_actions'])))
        self.assertEqual(env.controller.metadata['inventoryObjects'], [])

    def test_put_down_logical_fallback_requires_visible_near_target(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        env.controller.metadata['inventoryObjects'] = [{'objectId': 'Mug|0'}]
        invisible_failure = {'status': 'failed', 'error_message': 'No valid positions to place object found', 'attempts': [{'visible_before_action': False, 'pose': {'distance': 0.5}}]}
        self.assertIsNone(env._try_logical_put_down_fallback(target_ref='target_receptacle', receptacle_object_id='Sink|0', held_object_id='Mug|0', failed_execution=invisible_failure))
        self.assertEqual(invisible_failure['logical_put_down_fallback_rejected']['reason'], 'target_not_visible_for_put_down_fallback')
        far_failure = {'status': 'failed', 'error_message': 'No valid positions to place object found', 'attempts': [{'visible_before_action': True, 'pose': {'distance': 3.0}}]}
        self.assertIsNone(env._try_logical_put_down_fallback(target_ref='target_receptacle', receptacle_object_id='Sink|0', held_object_id='Mug|0', failed_execution=far_failure))
        self.assertEqual(far_failure['logical_put_down_fallback_rejected']['reason'], 'target_too_far_for_put_down_fallback')

    def test_scan_rotates_eight_views_and_marks_low_level_source(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        start_yaw = env.controller.metadata['agent']['rotation']['y']
        result = env.call_tool('scan')
        self.assertTrue(result['success'])
        self.assertEqual(result['episode_metrics']['scan_steps'], 1)
        self.assertEqual(len(result['low_level_actions']), 9)
        self.assertTrue(all((action.get('source') == 'scan' for action in result['low_level_actions'])))
        self.assertEqual(env.controller.metadata['agent']['rotation']['y'], start_yaw)

    def test_answer_consumes_pending_qa_without_invalidating_missing_pending(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        env._human_qa_state['pending']['qa_0'] = {'question': 'Ready?'}
        answer = env.call_tool('answer', {'question_id': 'qa_0', 'value': 'yes'})
        missing = env.call_tool('answer', {'question_id': 'qa_missing', 'value': 'no'})
        self.assertTrue(answer['success'])
        self.assertEqual(env._human_qa_state['answers']['qa_0']['value'], 'yes')
        self.assertFalse(missing['success'])
        self.assertEqual(missing['episode_metrics']['invalid_tool_calls'], 0)
        self.assertEqual(missing['episode_metrics']['answer_steps'], 2)

    def test_pending_qa_delays_auto_completion_until_answer(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        env.call_tool('navigate_to', {'target': 'target_object'})
        env.call_tool('pick_up', {'target': 'target_object'})
        env.call_tool('navigate_to', {'target': 'target_receptacle'})
        env._human_qa_state['pending']['qa_0'] = {'question': 'Did placement work?'}
        put = env.call_tool('put_down', {'target': 'target_receptacle'})
        answer = env.call_tool('answer', {'question_id': 'qa_0', 'value': 'yes'})
        self.assertEqual(put['goal_result']['task_success'], 1.0)
        self.assertFalse(put['done'])
        self.assertTrue(answer['done'])
        self.assertEqual(answer['done_reason'], 'task_success')
        self.assertEqual(env.task_action_step, 4)
        self.assertEqual(env.high_level_step, 5)

    def test_harness_directed_failure_does_not_consume_invalid_call_budget(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        call = {'tool_name': 'unknown_recovery_tool', 'arguments': {}}
        env.mark_harness_directed_tool_call(call)
        directed = env.call_tool(call['tool_name'], call['arguments'])
        ordinary = env.call_tool(call['tool_name'], call['arguments'])
        self.assertFalse(directed['success'])
        self.assertEqual(directed['episode_metrics']['invalid_tool_calls'], 0)
        self.assertFalse(ordinary['success'])
        self.assertEqual(ordinary['episode_metrics']['invalid_tool_calls'], 1)

    def test_public_precondition_rejection_is_nonfatal_and_non_actuating(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        call = {'tool_name': 'pick_up', 'arguments': {'target': 'mug'}}
        env.reject_public_precondition_tool_call(call, 'already holding an object')
        result = env.call_tool(call['tool_name'], call['arguments'])
        self.assertFalse(result['success'])
        self.assertEqual(result['binding']['reason'], 'public_precondition_failed')
        self.assertFalse(result['raw_execution']['action_executed'])
        self.assertEqual(result['num_low_level_steps'], 0)
        self.assertEqual(result['episode_metrics']['invalid_tool_calls'], 0)

    def test_completed_action_suppression_does_not_execute_physical_tool(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        call = {'tool_name': 'open', 'arguments': {'target': 'cabinet'}}
        env.suppress_completed_tool_call(call)
        result = env.call_tool(call['tool_name'], call['arguments'])
        self.assertTrue(result['success'])
        self.assertFalse(result['raw_execution']['action_executed'])
        self.assertEqual(result['num_low_level_steps'], 0)

    def test_prompt_history_hides_raw_execution_and_goal_result(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        env.call_tool('scan')
        context = env.get_prompt_context()
        self.assertIn('tool_history', context)
        self.assertNotIn('raw_execution', context['tool_history'][0])
        self.assertNotIn('goal_result', context['tool_history'][0])

    def test_rejects_stateful_navigation_target(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        result = env.call_tool('navigate_to', {'target': 'the dirty mug'})
        self.assertFalse(result['success'])
        self.assertIn('spatial-layer', result['env_feedback'])

    def test_navigate_to_room_label_uses_position_navigation(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        result = env.call_tool('navigate_to', {'target': 'kitchen'})
        self.assertTrue(result['success'])
        self.assertEqual(result['raw_execution']['resolved_via'], 'room_label')
        self.assertFalse(result['raw_execution']['object_visibility_guaranteed'])
        self.assertEqual(env.get_prompt_context()['room_catalog'], [{'label': 'kitchen', 'room_type': 'kitchen'}])

    def test_historical_alias_navigation_uses_last_known_position(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        env.call_tool('navigate_to', {'target': 'target_object'})
        env.controller._object('Mug|0')['visible'] = False
        env._refresh_agent_catalog_view()
        result = env.call_tool('navigate_to', {'target': 'Mug'})
        self.assertTrue(result['success'])
        self.assertEqual(result['raw_execution']['resolved_via'], 'last_known_position')
        self.assertEqual(result['raw_execution']['object_id'], 'Mug|0')
        self.assertTrue(result['raw_execution']['object_visible'])
        self.assertTrue(result['raw_execution']['interaction_ready'])

    def test_historical_alias_does_not_follow_private_current_position(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        env.call_tool('navigate_to', {'target': 'target_object'})
        mug = env.controller._object('Mug|0')
        historical_position = dict(mug['position'])
        mug['position'] = {'x': 10.0, 'y': 0.1, 'z': 0.0}
        mug['visible'] = False
        env._refresh_agent_catalog_view()
        result = env.call_tool('navigate_to', {'target': 'Mug'})
        self.assertTrue(result['success'])
        self.assertEqual(result['raw_execution']['resolved_via'], 'last_known_position')
        self.assertEqual(result['raw_execution']['target_position'], historical_position)
        self.assertFalse(result['raw_execution']['object_visible'])
        self.assertFalse(result['raw_execution']['interaction_ready'])

    def test_verify_after_action_masks_feedback(self):
        env = _FakeProcTHORToolEnv()
        task = _task_spec()
        task['category'] = 'verify_after_action'
        task['family_spec'] = {'family': 'verify_after_action', 'scope': {'type': 'single_room', 'label': 'kitchen', 'room_ids': ['room_0']}, 'verify': {'verification_task': {'type': 'place_object', 'object_role': 'target_object', 'receptacle_role': 'target_receptacle'}, 'execution_anomaly': {'mode': 'drop_beside', 'tool': 'put_down', 'target_role': 'target_receptacle', 'max_count': 1}}}
        env.reset_to_task(task)
        result = env.call_tool('navigate_to', {'target': 'the dirty mug'})
        self.assertEqual(result['env_feedback'], 'Execution finished. Check the scene to verify the result.')

    def test_operation_without_navigation_fails(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec())
        result = env.call_tool('pick_up', {'target': 'target_object'})
        self.assertFalse(result['success'])
        self.assertIn('not visible', result['env_feedback'])
        self.assertEqual(result['goal_result']['task_success'], 0.0)

    def test_manipulathor_arm_pickup_uses_arm_api_and_move_arm(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True)
        env.reset_to_task(_task_spec())
        env.call_tool('navigate_to', {'target': 'target_object'})
        pickup = env.call_tool('pick_up', {'target': 'target_object'})
        self.assertTrue(pickup['success'])
        self.assertEqual(env.controller.metadata['arm']['heldObjects'], ['Mug|0'])
        self.assertEqual(pickup['observation']['held_objects'], [{'objectId': 'Mug|0', 'objectType': 'Mug'}])
        pickup_actions = [action for action in env.controller.actions if action.get('action') == 'PickupObject']
        self.assertTrue(pickup_actions)
        self.assertNotIn('objectId', pickup_actions[-1])
        self.assertEqual(pickup_actions[-1]['objectIdCandidates'], ['Mug|0'])
        action_names = [action['action'] for action in pickup['low_level_actions']]
        self.assertIn('SetHandSphereRadius', action_names)
        self.assertIn('MoveArm', action_names)
        self.assertNotIn('TeleportObject', action_names)
        self.assertNotIn('TeleportFull', action_names)

    def test_manipulathor_visual_attach_pickup_skips_physical_arm_checks(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True, manipulathor_visual_attach_pickup=True)
        env.reset_to_task(_task_spec())
        pickup = env.call_tool('pick_up', {'target': 'target_object'})
        self.assertTrue(pickup['success'])
        self.assertEqual(env.controller.metadata['arm']['heldObjects'], ['Mug|0'])
        self.assertEqual(pickup['observation']['held_objects'], [{'objectId': 'Mug|0', 'objectType': 'Mug'}])
        action_names = [action['action'] for action in env.controller.actions]
        self.assertIn('TeleportObject', action_names)
        self.assertNotIn('PickupObject', action_names)
        self.assertIn('MoveArm', action_names)
        self.assertIn('SetHandSphereRadius', action_names)
        self.assertEqual([action['action'] for action in pickup['low_level_actions']], ['VisualAttachObject'])
        self.assertTrue(pickup['raw_execution']['visual_attach_pickup'])
        self.assertTrue(pickup['raw_execution']['navigation_precondition']['skipped'])
        radius_actions = [action for action in env.controller.actions if action.get('action') == 'SetAgentRadius']
        self.assertEqual(radius_actions[0]['agentRadius'], 0.3)

    def test_visual_attach_offset_uses_object_bbox_size(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True, manipulathor_visual_attach_pickup=True)
        env.reset_to_task(_task_spec())
        mug = env.controller._object('Mug|0')
        mug['axisAlignedBoundingBox'] = {'size': {'x': 0.2, 'y': 0.2, 'z': 0.2}}
        pickup = env.call_tool('pick_up', {'target': 'target_object'})
        offset = pickup['raw_execution']['sync']['visual_attach_offset']
        self.assertAlmostEqual(offset['x'], 0.0)
        self.assertAlmostEqual(offset['y'], 0.0)
        self.assertAlmostEqual(offset['z'], 0.0)
        geometry = pickup['raw_execution']['sync']['private_visual_attach_geometry']
        self.assertAlmostEqual(geometry['bbox_height'], 0.2)
        self.assertAlmostEqual(geometry['anchor_contact_gap_m'], -0.1)
        center_offset = geometry['object_center_offset_from_hand']
        self.assertAlmostEqual(center_offset['x'], 0.0)
        self.assertAlmostEqual(center_offset['y'], 0.0)
        self.assertAlmostEqual(center_offset['z'], 0.0)

    def test_local_navigation_context_allows_open_then_put_without_renavigation(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True, manipulathor_visual_attach_pickup=True)
        task = _task_spec()
        task['instruction'] = 'Put the mug in the drawer.'
        task['roles']['target_receptacle'] = {'object_id': 'Drawer|0', 'object_type': 'Drawer'}
        task['goal']['predicates'][0]['receptacle_role'] = 'target_receptacle'
        task['task_binding_spec'] = compile_task_binding_spec(task).to_dict()
        env.fake_controller.metadata['objects'].append(_obj('Drawer|0', 'Drawer', receptacle=True, receptacleObjectIds=[], openable=True, isOpen=False, position={'x': 2.0, 'y': 0.9, 'z': 0.0}))
        env.fake_controller._update_visibility()
        env.reset_to_task(task)
        direct_pickup = env.call_tool('pick_up', {'target': 'mug'})
        navigate_mug = env.call_tool('navigate_to', {'target': 'mug'})
        pickup = env.call_tool('pick_up', {'target': 'mug'})
        navigate_drawer = env.call_tool('navigate_to', {'target': 'drawer'})
        opened = env.call_tool('open', {'target': 'drawer'})
        placed = env.call_tool('put_down', {'target': 'drawer'})
        self.assertFalse(direct_pickup['success'])
        self.assertEqual(direct_pickup['binding']['reason'], 'navigation_context_required')
        self.assertTrue(navigate_mug['success'])
        self.assertTrue(pickup['success'])
        self.assertTrue(navigate_drawer['success'])
        self.assertTrue(opened['success'])
        self.assertTrue(placed['success'])
        self.assertEqual(opened['binding']['canonical_public_ref'], placed['binding']['canonical_public_ref'])
        self.assertEqual(env.controller._object('Mug|0')['parentReceptacles'], ['Drawer|0'])

    def test_openable_container_pose_sync_keeps_placed_child_attached(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True, manipulathor_visual_attach_pickup=True)
        task = _task_spec()
        task['instruction'] = 'Put the mug in the drawer.'
        task['roles']['target_receptacle'] = {'object_id': 'Drawer|0', 'object_type': 'Drawer'}
        task['goal']['predicates'][0]['receptacle_role'] = 'target_receptacle'
        task['task_binding_spec'] = compile_task_binding_spec(task).to_dict()
        env.fake_controller.metadata['objects'].append(_obj('Drawer|0', 'Drawer', receptacle=True, receptacleObjectIds=[], openable=True, isOpen=False, position={'x': 2.0, 'y': 0.9, 'z': 0.0}))
        env.fake_controller._update_visibility()
        env.reset_to_task(task)
        env.call_tool('navigate_to', {'target': 'mug'})
        pickup = env.call_tool('pick_up', {'target': 'mug'})
        env.call_tool('navigate_to', {'target': 'drawer'})
        opened = env.call_tool('open', {'target': 'drawer'})
        placed = env.call_tool('put_down', {'target': 'drawer'})
        self.assertTrue(pickup['success'])
        self.assertTrue(opened['success'])
        self.assertTrue(placed['success'])
        mug = env.controller._object('Mug|0')
        placed_position = dict(mug['position'])
        drawer = env.controller._object('Drawer|0')
        drawer['position'] = {'x': 2.0, 'y': 0.9, 'z': 0.5}
        env.controller._update_visibility()
        closed = env.call_tool('close', {'target': 'drawer'})
        self.assertTrue(closed['success'])
        self.assertNotEqual(mug['position'], placed_position)
        self.assertEqual(mug['position']['z'], 0.5)
        self.assertEqual(mug['parentReceptacles'], ['Drawer|0'])
        self.assertTrue(any((event.get('name') == 'container_pose_sync' for event in closed['state_transitions'])))
        drawer['position'] = {'x': 2.0, 'y': 0.9, 'z': 0.0}
        env.controller._update_visibility()
        reopened = env.call_tool('open', {'target': 'drawer'})
        self.assertTrue(reopened['success'])
        self.assertEqual(mug['position']['x'], 2.0)
        self.assertEqual(mug['parentReceptacles'], ['Drawer|0'])

    def test_visual_attach_offset_keeps_large_objects_centered_on_hand(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True, manipulathor_visual_attach_pickup=True)
        env.reset_to_task(_task_spec())
        mug = env.controller._object('Mug|0')
        mug['axisAlignedBoundingBox'] = {'size': {'x': 1.2, 'y': 0.8, 'z': 0.2}}
        pickup = env.call_tool('pick_up', {'target': 'target_object'})
        offset = pickup['raw_execution']['sync']['visual_attach_offset']
        self.assertAlmostEqual(offset['x'], 0.0)
        self.assertAlmostEqual(offset['y'], 0.0)
        self.assertAlmostEqual(offset['z'], 0.0)

    def test_visual_attach_offset_can_use_bbox_corner_points(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True, manipulathor_visual_attach_pickup=True)
        env.reset_to_task(_task_spec())
        mug = env.controller._object('Mug|0')
        mug['axisAlignedBoundingBox'] = {'cornerPoints': [{'x': 0.0, 'y': 0.0, 'z': 0.0}, {'x': 0.4, 'y': 0.0, 'z': 0.0}, {'x': 0.0, 'y': 0.3, 'z': 0.0}, {'x': 0.0, 'y': 0.0, 'z': 0.6}, {'x': 0.4, 'y': 0.3, 'z': 0.6}]}
        pickup = env.call_tool('pick_up', {'target': 'target_object'})
        offset = pickup['raw_execution']['sync']['visual_attach_offset']
        self.assertAlmostEqual(offset['x'], 0.0)
        self.assertAlmostEqual(offset['y'], 0.0)
        self.assertAlmostEqual(offset['z'], 0.0)

    def test_visual_attached_object_follows_ego_after_low_level_move(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True, manipulathor_visual_attach_pickup=True)
        env.reset_to_task(_task_spec())
        pickup = env.call_tool('pick_up', {'target': 'target_object'})
        self.assertTrue(pickup['success'])
        env._pending_low_level_records = []
        initial_position = dict(env.controller._object('Mug|0')['position'])
        move_event = env.controller.step(action='MoveAhead', moveMagnitude=0.25)
        record = env._record_low_level_action({'action': 'MoveAhead', 'executed_action': 'MoveAhead', 'label': 'navigate with held object', 'success': True, 'agent_pose': {'position': dict(env.controller.metadata['agent']['position']), 'rotation': dict(env.controller.metadata['agent']['rotation']), 'cameraHorizon': 0.0, 'isStanding': True}}, move_event)
        moved_position = env.controller._object('Mug|0')['position']
        self.assertGreater(moved_position['z'], initial_position['z'])
        self.assertAlmostEqual(moved_position['z'] - initial_position['z'], 0.25)
        self.assertTrue(record['visual_attach_sync']['success'])
        self.assertEqual(env.controller.actions[-1]['action'], 'TeleportObject')
        self.assertEqual([item['action'] for item in env._pending_low_level_records], ['MoveAhead'])

    def test_manipulathor_visual_arm_pose_restores_pickupable_object_drift(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True, manipulathor_visual_attach_pickup=True)
        env.reset_to_task(_task_spec())
        mug = env.controller._object('Mug|0')
        original_position = dict(mug['position'])
        original_step = env.controller.step

        def drifting_step(**kwargs):
            event = original_step(**kwargs)
            if kwargs.get('action') == 'MoveArm':
                mug['position'] = {'x': 99.0, 'y': 99.0, 'z': 99.0}
            return event
        env.controller.step = drifting_step
        records = env._apply_visual_arm_pose(label='test visual arm pose restore', append_record=True)
        self.assertEqual(mug['position'], original_position)
        action_names = [action['action'] for action in env.controller.actions]
        self.assertIn('MoveArm', action_names)
        self.assertIn('SetObjectPoses', action_names)
        move_arm = [a for a in env.controller.actions if a['action'] == 'MoveArm'][-1]
        self.assertEqual(move_arm['position'], {'x': -0.22, 'y': 0.48, 'z': 0.2})
        self.assertEqual(move_arm['speed'], 100000.0)
        self.assertEqual(move_arm['fixedDeltaTime'], 0.0001)
        self.assertIn('VisualArmPoseRestore', [record['action'] for record in records])

    def test_manipulathor_visual_attach_put_down_updates_goal_without_physics(self):
        env = _FakeProcTHORToolEnv(manipulathor_arm=True, manipulathor_visual_attach_pickup=True)
        env.reset_to_task(_task_spec())
        pickup = env.call_tool('pick_up', {'target': 'target_object'})
        put = env.call_tool('put_down', {'target': 'target_receptacle'})
        self.assertTrue(pickup['success'])
        self.assertTrue(put['success'])
        self.assertEqual(env.controller.metadata['arm']['heldObjects'], [])
        self.assertEqual(put['observation']['held_objects'], [])
        self.assertEqual(env._metadata_with_sticky_object_states()['objects'][0]['parentReceptacles'], ['Sink|0'])
        self.assertEqual(put['goal_result']['task_success'], 1.0)
        self.assertEqual([action['action'] for action in put['low_level_actions']], ['VisualPutObject'])
        self.assertTrue(put['raw_execution']['visual_attach_put_down'])

    def test_stop_finishes_completion_timing_episode(self):
        env = _FakeProcTHORToolEnv()
        env.reset_to_task(_task_spec(require_stop=True))
        env.call_tool('navigate_to', {'target': 'target_object'})
        env.call_tool('pick_up', {'target': 'target_object'})
        env.call_tool('navigate_to', {'target': 'target_receptacle'})
        put = env.call_tool('put_down', {'target': 'target_receptacle'})
        stop = env.call_tool('stop')
        self.assertFalse(put['done'])
        self.assertTrue(stop['done'])
        self.assertEqual(stop['goal_result']['task_success'], 1.0)

    def test_get_tool_specs_can_include_evaluator_helpers(self):
        names = [tool['name'] for tool in get_tool_specs(model_visible_only=False)]
        self.assertIn('check_goal', names)
        self.assertIn('get_context', names)

    def test_slice_rebinds_public_role_to_deterministic_sliced_child(self):
        env = ProcTHORClosedLoopToolEnv.__new__(ProcTHORClosedLoopToolEnv)
        env.role_object_ids = {'target_object': 'Apple|1', 'other': 'Bowl|1'}
        objects = {'Apple|1|AppleSliced_1': _obj('Apple|1|AppleSliced_1', 'AppleSliced', pickupable=True), 'Apple|1|AppleSliced_0': _obj('Apple|1|AppleSliced_0', 'AppleSliced', pickupable=True), 'Bowl|1': _obj('Bowl|1', 'Bowl', receptacle=True)}
        env._objects_by_id = lambda: objects
        env._rebind_roles_to_sliced_child('Apple|1')
        self.assertEqual(env.role_object_ids, {'target_object': 'Apple|1|AppleSliced_0', 'other': 'Bowl|1'})

    def test_batch_validator_runs_through_public_tool_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = os.path.join(tmpdir, 'task_bank.jsonl')
            output_dir = os.path.join(tmpdir, 'closed_loop')
            with open(bank_path, 'w', encoding='utf-8') as f:
                json.dump(_task_spec(), f, ensure_ascii=False, sort_keys=True)
                f.write('\n')
            records, summary = run_closed_loop_tool_validation(task_bank_path=bank_path, output_dir=output_dir, env_factory=_FakeProcTHORToolEnv)
        self.assertEqual(records[0]['status'], 'passed')
        self.assertEqual(summary['num_passed'], 1)
        self.assertEqual(summary['num_records_with_teleport_full'], 0)
        self.assertIn('navigate_to', summary['tool_counts'])
if __name__ == '__main__':
    unittest.main()
