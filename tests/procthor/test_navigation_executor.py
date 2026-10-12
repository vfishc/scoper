import copy
import math
import unittest
from scoper.benchmark.procthor.navigation_executor import InteractionPose, ProcTHORLowLevelExecutionError, ProcTHORLowLevelExecutor, _astar_path, _path_num_turns, _points_by_key
from scoper.benchmark.procthor.spatial_state import interaction_ready_from_metadata, object_visible_from_metadata

def _obj(object_id, object_type=None, **kwargs):
    data = {'objectId': object_id, 'objectType': object_type or object_id.split('|', 1)[0], 'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'visible': False, 'pickupable': False, 'receptacle': False, 'openable': False, 'toggleable': False, 'sliceable': False, 'parentReceptacles': [], 'distance': 10.0}
    data.update(kwargs)
    return data

def _task_spec():
    return {'id': 'procthor_static_simple_000001', 'version': '0.1', 'backend': 'procthor', 'category': 'static_simple', 'source': {'house_id': 'synthetic_house', 'house_path': '/tmp/synthetic_house.json', 'split': 'train'}, 'template': {'template_id': 'static_simple_place_001', 'template_version': '0.1'}, 'instruction': 'Put the mug in the sink.', 'roles': {'target_object': {'object_id': 'Mug|0', 'object_type': 'Mug'}, 'target_receptacle': {'object_id': 'Sink|0', 'object_type': 'Sink'}, 'stateful_object': {'object_id': 'Faucet|0', 'object_type': 'Faucet'}}, 'init': {'agent_pose': {'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'rotation': {'y': 0.0}, 'horizon': 0.0}, 'object_state_overrides': []}, 'goal': {'type': 'predicate', 'predicates': [{'type': 'object_in_receptacle', 'object_role': 'target_object', 'receptacle_role': 'target_receptacle'}], 'require_model_stop': False}, 'online_requirements': {'expected_events': [], 'expected_brain_behaviors': []}, 'validation': {'status': 'validated', 'checks': {}, 'failure_reason': None}, 'tags': ['static_simple']}

def _metadata():
    return {'lastActionSuccess': True, 'errorMessage': '', 'objects': [_obj('Mug|0', 'Mug', pickupable=True, position={'x': 1.75, 'y': 0.9, 'z': 0.0}), _obj('Sink|0', 'Sink', receptacle=True, position={'x': 2.0, 'y': 0.9, 'z': 0.0}), _obj('Faucet|0', 'Faucet', toggleable=True, isToggled=False, position={'x': 1.0, 'y': 0.9, 'z': 1.0})], 'inventoryObjects': [], 'agent': {'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'rotation': {'y': 0.0}, 'cameraHorizon': 0.0}}

class _FakeEvent:

    def __init__(self, metadata):
        self.metadata = metadata
        self.frame = None

class _FakeController:

    def __init__(self, metadata):
        self.metadata = copy.deepcopy(metadata)
        self.actions = []
        self.last_event = _FakeEvent(self.metadata)
        self.reachable = [{'x': x / 4.0, 'y': 0.9, 'z': z / 4.0} for x in range(0, 9) for z in range(0, 5)]
        self._update_visibility()

    def step(self, **kwargs):
        self.actions.append(dict(kwargs))
        action = kwargs['action']
        self.metadata['lastActionSuccess'] = True
        self.metadata['errorMessage'] = ''
        if action == 'GetReachablePositions':
            self.metadata['actionReturn'] = copy.deepcopy(self.reachable)
        elif action == 'RotateRight':
            self._rotate(float(kwargs.get('degrees', 90)))
        elif action == 'RotateLeft':
            self._rotate(-float(kwargs.get('degrees', 90)))
        elif action == 'LookDown':
            self._look(float(kwargs.get('degrees', 30)))
        elif action == 'LookUp':
            self._look(-float(kwargs.get('degrees', 30)))
        elif action == 'MoveAhead':
            self._move_action(action, float(kwargs.get('moveMagnitude', 0.25)))
        elif action == 'MoveBack':
            self._move_action(action, float(kwargs.get('moveMagnitude', 0.25)))
        elif action == 'MoveLeft':
            self._move_action(action, float(kwargs.get('moveMagnitude', 0.25)))
        elif action == 'MoveRight':
            self._move_action(action, float(kwargs.get('moveMagnitude', 0.25)))
        elif action == 'PickupObject':
            self._pickup(kwargs['objectId'])
        elif action == 'PutObject':
            self._put(kwargs['objectId'])
        elif action == 'OpenObject':
            self._open_close(kwargs['objectId'], True)
        elif action == 'CloseObject':
            self._open_close(kwargs['objectId'], False)
        elif action == 'ToggleObjectOn':
            self._toggle(kwargs['objectId'], True)
        elif action == 'ToggleObjectOff':
            self._toggle(kwargs['objectId'], False)
        else:
            self._fail('unsupported action')
        self._update_visibility()
        self.last_event = _FakeEvent(self.metadata)
        return self.last_event

    def stop(self):
        pass

    def _rotate(self, degrees):
        rotation = self.metadata['agent']['rotation']
        rotation['y'] = (float(rotation.get('y', 0.0)) + degrees) % 360

    def _look(self, degrees):
        self.metadata['agent']['cameraHorizon'] = max(-30.0, min(60.0, float(self.metadata['agent'].get('cameraHorizon', 0.0)) + degrees))

    def _move_action(self, action, magnitude):
        try:
            self._move(magnitude, action)
        except TypeError:
            self._move(magnitude)

    def _move(self, magnitude, action='MoveAhead'):
        yaw = round(float(self.metadata['agent']['rotation']['y']) / 90.0) * 90 % 360
        if action == 'MoveBack':
            yaw = (yaw + 180) % 360
        elif action == 'MoveLeft':
            yaw = (yaw - 90) % 360
        elif action == 'MoveRight':
            yaw = (yaw + 90) % 360
        dx, dz = {0: (0.0, magnitude), 90: (magnitude, 0.0), 180: (0.0, -magnitude), 270: (-magnitude, 0.0)}[yaw]
        pos = self.metadata['agent']['position']
        new_pos = {'x': round(pos['x'] + dx, 2), 'y': pos['y'], 'z': round(pos['z'] + dz, 2)}
        if not any((abs(item['x'] - new_pos['x']) < 1e-06 and abs(item['z'] - new_pos['z']) < 1e-06 for item in self.reachable)):
            self._fail('move target is not reachable')
            return
        self.metadata['agent']['position'] = new_pos

    def _pickup(self, object_id):
        obj = self._object(object_id)
        if not obj.get('visible'):
            self._fail('object is not visible')
            return
        self.metadata['inventoryObjects'] = [{'objectId': obj['objectId'], 'objectType': obj['objectType']}]
        obj['parentReceptacles'] = []

    def _open_close(self, object_id, is_open):
        obj = self._object(object_id)
        if not obj.get('openable'):
            self._fail('object is not openable')
            return
        obj['isOpen'] = bool(is_open)

    def _put(self, receptacle_id):
        receptacle = self._object(receptacle_id)
        if not receptacle.get('visible'):
            self._fail('receptacle is not visible')
            return
        if not self.metadata.get('inventoryObjects'):
            self._fail('inventory is empty')
            return
        held_id = self.metadata['inventoryObjects'][0]['objectId']
        self._object(held_id)['parentReceptacles'] = [receptacle_id]
        self.metadata['inventoryObjects'] = []

    def _toggle(self, object_id, value):
        obj = self._object(object_id)
        if not obj.get('visible'):
            self._fail('object is not visible')
            return
        obj['isToggled'] = value

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

    def _fail(self, message):
        self.metadata['lastActionSuccess'] = False
        self.metadata['errorMessage'] = message

    def _object(self, object_id):
        for obj in self.metadata['objects']:
            if obj['objectId'] == object_id:
                return obj
        raise KeyError(object_id)

class _FakeEnv:

    def __init__(self):
        self.controller = _FakeController(_metadata())
        self.role_object_ids = {role: data['object_id'] for role, data in _task_spec()['roles'].items()}

class _VisualAttachFakeEnv(_FakeEnv):

    def __init__(self):
        super().__init__()
        self.manipulathor_visual_attach_pickup = True

def _arm_open_close_collision_case(*, action='OpenObject', visual_attach=False, error_message='robot_wrist collider blocked the interaction', openable=True, forced_post_state=None, enabled=True, budget=8):
    env = _FakeEnv()
    env.manipulathor_arm = True
    env.manipulathor_visual_attach_pickup = bool(visual_attach)
    env.total_low_level_steps = 0
    env._pending_low_level_records = []
    env.max_total_low_level_steps = 100
    sink = env.controller._object('Sink|0')
    sink['openable'] = bool(openable)
    sink['receptacle'] = True
    sink['isOpen'] = action == 'CloseObject'
    original_step = env.controller.step

    def _step(**kwargs):
        if kwargs.get('action') in {'OpenObject', 'CloseObject'}:
            env.controller.actions.append(dict(kwargs))
            if not kwargs.get('forceAction'):
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = error_message
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            if forced_post_state is None:
                sink['isOpen'] = kwargs['action'] == 'OpenObject'
            else:
                sink['isOpen'] = forced_post_state
            env.controller.metadata['lastActionSuccess'] = True
            env.controller.metadata['errorMessage'] = ''
            env.controller.last_event = _FakeEvent(env.controller.metadata)
            return env.controller.last_event
        return original_step(**kwargs)
    env.controller.step = _step
    executor = ProcTHORLowLevelExecutor(env, _task_spec(), allow_force_arm_open_close_collision_fallback=enabled, max_force_arm_open_close_collision_fallbacks_per_tool=budget)
    return (env, executor)

class ProcTHORNavigationExecutorTest(unittest.TestCase):

    def test_metadata_readiness_requires_visibility_and_interaction_distance(self):
        metadata = _metadata()
        mug = metadata['objects'][0]
        mug['visible'] = True
        mug['distance'] = 1.51
        self.assertTrue(object_visible_from_metadata(metadata, 'Mug|0'))
        self.assertFalse(interaction_ready_from_metadata(metadata, 'Mug|0'))
        mug['distance'] = 1.5
        self.assertTrue(interaction_ready_from_metadata(metadata, 'Mug|0'))

    def test_navigation_does_not_report_success_for_visible_but_unready_target(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())

        def _single_candidate(_object_id, *, exclude=None, **_kwargs):
            if exclude:
                return []
            return [InteractionPose(key=(0, 0), position={'x': 0.0, 'y': 0.9, 'z': 0.0}, rotation_y=0, horizon=0, standing=True, distance=1.0, path_length=0, score=0.0, target_object_id='Mug|0')]
        executor.rank_interaction_poses = _single_candidate
        executor.is_object_visible = lambda _object_id: True
        executor.is_object_interaction_ready = lambda _object_id: False
        result = executor.navigate_to_object_id('Mug|0', allow_failure=True)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['attempts'][0]['interaction_ready'])
        self.assertIn('outside the interaction radius', result['attempts'][0]['error_message'])

    def test_navigation_is_idempotent_when_target_is_already_interaction_ready(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor.is_object_interaction_ready = lambda _object_id: True
        result = executor.navigate_to_object_id('Mug|0', target_ref='mug')
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['interaction_ready'])
        self.assertEqual(result['interaction_pose_source'], 'current_pose')
        self.assertEqual(result['attempts'], [])
        self.assertEqual(env.controller.actions, [])

    def test_pickup_navigates_before_interaction(self):
        env = _FakeEnv()
        stream = []
        executor = ProcTHORLowLevelExecutor(env, _task_spec(), stream_callback=lambda record, _event: stream.append(dict(record)))
        result = executor.execute_role_action('PickupObject', 'target_object', label='pickup target_object')
        self.assertEqual(result['status'], 'success')
        actions = [action['action'] for action in stream]
        self.assertIn('MoveAhead', actions)
        self.assertEqual(actions[-1], 'PickupObject')
        self.assertIn('metadata_visible_object_ids', stream[-1])
        self.assertEqual([action for action in stream if action['action'] == 'MoveAhead'][0]['source'], 'navigation_stream')
        self.assertNotIn('TeleportFull', actions)
        self.assertEqual(env.controller.metadata['inventoryObjects'][0]['objectId'], 'Mug|0')

    def test_manipulathor_pickup_payload_uses_object_id_candidates(self):
        env = _FakeEnv()
        env.manipulathor_arm = True
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        payloads = executor._execution_payloads({'action': 'PickupObject', 'objectId': 'Mug|0'})
        self.assertEqual(payloads, [{'action': 'PickupObject', 'objectIdCandidates': ['Mug|0']}])

    def test_viewpoint_payloads_are_normalized_to_tenth_degree(self):
        executor = ProcTHORLowLevelExecutor(_FakeEnv(), _task_spec())
        self.assertEqual(executor._execution_payloads({'action': 'LookDown', 'degrees': 17.298137664794922}), [{'action': 'LookDown', 'degrees': 17.3}])
        self.assertEqual(executor._execution_payloads({'action': 'RotateRight', 'degrees': 44.444}), [{'action': 'RotateRight', 'degrees': 44.4}])

    def test_put_navigates_to_receptacle_and_places_held_object(self):
        env = _FakeEnv()
        env.controller.metadata['inventoryObjects'] = [{'objectId': 'Mug|0', 'objectType': 'Mug'}]
        stream = []
        executor = ProcTHORLowLevelExecutor(env, _task_spec(), stream_callback=lambda record, _event: stream.append(dict(record)))
        result = executor.execute_role_action('PutObject', 'target_receptacle', label='put target_object into target_receptacle')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(env.controller._object('Mug|0')['parentReceptacles'], ['Sink|0'])
        self.assertEqual(stream[-1]['action'], 'PutObject')

    def test_force_put_fallback_uses_explicit_held_and_receptacle_payload(self):
        env = _FakeEnv()
        env.controller.metadata['agent']['position'] = {'x': 1.0, 'y': 0.9, 'z': 0.0}
        env.controller.metadata['agent']['rotation'] = {'y': 90.0}
        env.controller._update_visibility()
        env.controller.metadata['inventoryObjects'] = [{'objectId': 'Mug|0', 'objectType': 'Mug'}]
        original_step = env.controller.step

        def _step_with_explicit_put_required(**kwargs):
            if kwargs.get('action') != 'PutObject':
                return original_step(**kwargs)
            env.controller.actions.append(dict(kwargs))
            if kwargs.get('forceAction') and kwargs.get('receptacleObjectId'):
                receptacle_id = kwargs['receptacleObjectId']
                env.controller._object('Mug|0')['parentReceptacles'] = [receptacle_id]
                env.controller.metadata['inventoryObjects'] = []
                env.controller.metadata['lastActionSuccess'] = True
                env.controller.metadata['errorMessage'] = ''
            else:
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'No valid positions to place object found'
            env.controller.last_event = _FakeEvent(env.controller.metadata)
            return env.controller.last_event
        env.controller.step = _step_with_explicit_put_required
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        result = executor.interact_with_object_id('PutObject', 'Sink|0', target_ref='target_receptacle', allow_failure=True)
        self.assertEqual(result['status'], 'success')
        force_put_actions = [action for action in env.controller.actions if action.get('action') == 'PutObject' and action.get('forceAction')]
        self.assertTrue(force_put_actions)
        self.assertEqual(force_put_actions[-1]['objectId'], 'Mug|0')
        self.assertEqual(force_put_actions[-1]['receptacleObjectId'], 'Sink|0')
        self.assertTrue(force_put_actions[-1]['placeStationary'])
        self.assertEqual(executor.action_records[-1]['object_id'], 'Sink|0')
        self.assertEqual(executor.action_records[-1]['held_object_id'], 'Mug|0')
        self.assertEqual(executor.action_records[-1]['put_object_payload_style'], 'explicit_receptacle')

    def test_force_put_fallback_tries_parent_receptacle_for_child_surface(self):
        env = _FakeEnv()
        env.controller.metadata['agent']['position'] = {'x': 1.0, 'y': 0.9, 'z': 0.0}
        env.controller.metadata['agent']['rotation'] = {'y': 0.0}
        env.controller.metadata['objects'].extend([_obj('Cabinet|0', 'Cabinet', receptacle=True, position={'x': 1.0, 'y': 0.9, 'z': 0.0}, visible=True), _obj('Cabinet|0___0', 'Shelf', receptacle=True, position={'x': 1.0, 'y': 1.2, 'z': 0.0}, visible=True)])
        env.controller._update_visibility()
        env.controller.metadata['inventoryObjects'] = [{'objectId': 'Mug|0', 'objectType': 'Mug'}]
        original_step = env.controller.step

        def _step_parent_receptacle_required(**kwargs):
            if kwargs.get('action') != 'PutObject':
                return original_step(**kwargs)
            env.controller.actions.append(dict(kwargs))
            effective_receptacle = kwargs.get('receptacleObjectId') or kwargs.get('objectId')
            if kwargs.get('forceAction') and effective_receptacle == 'Cabinet|0':
                env.controller._object('Mug|0')['parentReceptacles'] = ['Cabinet|0']
                env.controller.metadata['inventoryObjects'] = []
                env.controller.metadata['lastActionSuccess'] = True
                env.controller.metadata['errorMessage'] = ''
            else:
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'No valid positions to place object found'
            env.controller.last_event = _FakeEvent(env.controller.metadata)
            return env.controller.last_event
        env.controller.step = _step_parent_receptacle_required
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        result = executor.interact_with_object_id('PutObject', 'Cabinet|0___0', target_ref='target_receptacle', allow_failure=True)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(executor.action_records[-1]['requested_receptacle_object_id'], 'Cabinet|0___0')
        self.assertEqual(executor.action_records[-1]['effective_receptacle_object_id'], 'Cabinet|0')
        self.assertEqual(executor.action_records[-1]['receptacle_fallback'], 'parent_object')

    def test_rejects_force_action(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        with self.assertRaisesRegex(ProcTHORLowLevelExecutionError, 'forceAction'):
            executor._step_physical({'action': 'PickupObject', 'objectId': 'Mug|0', 'forceAction': True}, label='forced pickup')

    def test_force_navigation_fallback_only_for_held_collision(self):
        env = _FakeEnv()
        env.controller.metadata['inventoryObjects'] = [{'objectId': 'Mug|0', 'objectType': 'Mug'}]
        original_step = env.controller.step

        def _step_with_held_collision(**kwargs):
            if kwargs.get('action') == 'RotateLeft' and (not kwargs.get('forceAction')):
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'a held item: Mug|0 with something if agent rotates Left 90 degrees'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_held_collision
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor._force_navigation_fallback_active = True
        event = executor._step_physical({'action': 'RotateLeft'}, label='fallback rotate', allow_failure=True)
        self.assertTrue(event.metadata['lastActionSuccess'])
        self.assertEqual(env.controller.metadata['agent']['rotation']['y'], 270.0)
        self.assertTrue(executor.action_records[-1]['forceAction'])
        self.assertTrue(executor.action_records[-1]['force_navigation_fallback'])
        self.assertEqual(executor.action_records[-1]['held_object_id'], 'Mug|0')

    def test_force_navigation_fallback_allows_logical_put_blocker(self):
        env = _FakeEnv()
        env._logical_put_down_object_ids = {'Mug|0'}
        original_step = env.controller.step

        def _step_with_logical_blocker(**kwargs):
            if kwargs.get('action') == 'MoveAhead' and (not kwargs.get('forceAction')):
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'Mug|0 is blocking Agent 0 from moving by (0.0000, 0.0000, 0.2500).'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_logical_blocker
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor._force_navigation_fallback_active = True
        event = executor._step_physical({'action': 'MoveAhead', 'moveMagnitude': 0.25}, label='fallback move', allow_failure=True)
        self.assertTrue(event.metadata['lastActionSuccess'])
        self.assertTrue(executor.action_records[-1]['forceAction'])
        self.assertTrue(executor.action_records[-1]['force_navigation_fallback'])
        self.assertEqual(executor.action_records[-1]['logical_blocking_object_id'], 'Mug|0')

    def test_force_navigation_fallback_rejects_openable_blocker(self):
        env = _FakeEnv()
        sink = env.controller._object('Sink|0')
        sink['openable'] = True
        sink['isOpen'] = True
        original_step = env.controller.step

        def _step_with_openable_blocker(**kwargs):
            if kwargs.get('action') == 'MoveAhead' and (not kwargs.get('forceAction')) and env.controller._object('Sink|0').get('isOpen'):
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'Sink|0 is blocking Agent 0 from moving by (0.0000, 0.0000, 0.2500).'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_openable_blocker
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor._force_navigation_fallback_active = True
        event = executor._step_physical({'action': 'MoveAhead', 'moveMagnitude': 0.25}, label='fallback move', allow_failure=True)
        self.assertFalse(event.metadata['lastActionSuccess'])
        self.assertNotIn('forceAction', executor.action_records[-1])
        self.assertTrue(sink['isOpen'])
        self.assertFalse(any((action.get('action') == 'CloseObject' and action.get('forceAction') for action in env.controller.actions)))

    def test_visual_attach_navigation_force_fallback_ignores_generic_collision(self):
        env = _VisualAttachFakeEnv()
        original_step = env.controller.step

        def _step_with_generic_collision(**kwargs):
            if kwargs.get('action') == 'MoveAhead' and (not kwargs.get('forceAction')):
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'Agent is blocked by geometry.'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_generic_collision
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor._force_navigation_fallback_active = True
        event = executor._step_physical({'action': 'MoveAhead', 'moveMagnitude': 0.25}, label='visual fallback move', allow_failure=True)
        self.assertTrue(event.metadata['lastActionSuccess'])
        force_moves = [action for action in env.controller.actions if action.get('action') == 'MoveAhead' and action.get('forceAction')]
        self.assertTrue(force_moves)
        self.assertTrue(executor.action_records[-1]['force_navigation_fallback'])

    def test_visual_attach_look_for_collision_forces_viewpoint_once(self):
        env = _VisualAttachFakeEnv()
        original_step = env.controller.step

        def _step_with_rotation_collision(**kwargs):
            if kwargs.get('action') == 'RotateRight' and (not kwargs.get('forceAction')):
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'Agent rotation is blocked by collision.'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_rotation_collision
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        event = executor._step_physical({'action': 'RotateRight', 'degrees': 45.0}, label='look rotation', source='look_for', allow_failure=True)
        self.assertTrue(event.metadata['lastActionSuccess'])
        rotations = [action for action in env.controller.actions if action.get('action') == 'RotateRight']
        self.assertEqual(len(rotations), 2)
        self.assertNotIn('forceAction', rotations[0])
        self.assertTrue(rotations[1]['forceAction'])
        self.assertEqual(env.controller.metadata['agent']['rotation']['y'], 45.0)
        self.assertTrue(executor.action_records[-1]['force_viewpoint_fallback'])
        self.assertTrue(executor.action_records[-1]['force_viewpoint_fallback_applied'])
        self.assertEqual(env._force_viewpoint_fallbacks_total, 1)

    def test_real_static_collision_uses_viewpoint_fallback_during_navigation_window(self):
        env = _VisualAttachFakeEnv()
        original_step = env.controller.step

        def _step_with_static_collision(**kwargs):
            if kwargs.get('action') == 'RotateRight' and (not kwargs.get('forceAction')):
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = "Collided with static sim object: 'Fridge|6|1', could not reach target: '(0.0, 0.4, 0.0, 0.9)'."
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_static_collision
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor._force_navigation_fallback_active = True
        event = executor._step_physical({'action': 'RotateRight', 'degrees': 42.22}, label='look toward fridge', source='look_for', allow_failure=True)
        self.assertTrue(event.metadata['lastActionSuccess'])
        self.assertTrue(executor.action_records[-1]['force_viewpoint_fallback'])
        self.assertNotIn('force_navigation_fallback', executor.action_records[-1])
        self.assertEqual(env.controller.actions[-1]['degrees'], 42.2)

    def test_visual_attach_viewpoint_force_requires_requested_pose(self):
        env = _VisualAttachFakeEnv()

        def _step_without_rotation(**kwargs):
            env.controller.actions.append(dict(kwargs))
            env.controller.metadata['lastActionSuccess'] = bool(kwargs.get('forceAction'))
            env.controller.metadata['errorMessage'] = '' if kwargs.get('forceAction') else 'rotation collision'
            env.controller.last_event = _FakeEvent(env.controller.metadata)
            return env.controller.last_event
        env.controller.step = _step_without_rotation
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        event = executor._step_physical({'action': 'RotateLeft', 'degrees': 30.0}, label='look rotation mismatch', source='look_for', allow_failure=True)
        self.assertFalse(event.metadata['lastActionSuccess'])
        self.assertIn('did not reach', event.metadata['errorMessage'])
        self.assertTrue(executor.action_records[-1]['force_viewpoint_fallback'])
        self.assertFalse(executor.action_records[-1]['force_viewpoint_fallback_applied'])

    def test_visual_attach_viewpoint_non_collision_does_not_force(self):
        env = _VisualAttachFakeEnv()

        def _step_with_non_collision_error(**kwargs):
            env.controller.actions.append(dict(kwargs))
            env.controller.metadata['lastActionSuccess'] = False
            env.controller.metadata['errorMessage'] = 'Invalid degrees argument.'
            env.controller.last_event = _FakeEvent(env.controller.metadata)
            return env.controller.last_event
        env.controller.step = _step_with_non_collision_error
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        event = executor._step_physical({'action': 'LookDown', 'degrees': 15.0}, label='invalid look', source='scan', allow_failure=True)
        self.assertFalse(event.metadata['lastActionSuccess'])
        self.assertEqual(len(env.controller.actions), 1)
        self.assertNotIn('forceAction', env.controller.actions[0])
        self.assertNotIn('force_viewpoint_fallback', executor.action_records[-1])

    def test_scan_collision_uses_one_real_clearance_move_then_retries_rotation(self):
        env = _FakeEnv()
        original_step = env.controller.step

        def _step_with_blocked_rotation(**kwargs):
            if kwargs.get('action') == 'RotateRight' and env.controller.metadata['agent']['position']['x'] == 0.0:
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'Agent rotation is blocked by collision.'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_blocked_rotation
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        result = executor.scan_in_place(num_views=2)
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['clearance_used'])
        self.assertEqual([item['action'] for item in result['clearance_attempts']], ['MoveBack', 'MoveLeft', 'MoveRight'])
        self.assertEqual([item['success'] for item in result['clearance_attempts']], [False, False, True])
        self.assertFalse(any((action.get('forceAction') for action in env.controller.actions)))

    def test_scan_frames_are_tilted_without_look_interpolation(self):
        for initial_horizon in [-45.0, 0.0, 45.0, 60.0]:
            with self.subTest(initial_horizon=initial_horizon):
                env = _FakeEnv()
                env.controller.metadata['agent']['cameraHorizon'] = initial_horizon
                executor = ProcTHORLowLevelExecutor(env, _task_spec(), look_execution_degrees=5.0)
                result = executor.scan_in_place(num_views=4)
                self.assertEqual(result['status'], 'success')
                self.assertEqual(result['num_views_completed'], 4)
                self.assertEqual(result['final_pose']['cameraHorizon'], 45.0)
                self.assertTrue(all((record['agent_pose']['cameraHorizon'] == 45.0 for record in executor.action_records)))
                look_actions = [action for action in env.controller.actions if action['action'] in {'LookDown', 'LookUp'}]
                self.assertEqual(len(look_actions), int(initial_horizon != 45.0))

    def test_scan_clearance_is_bounded_when_no_translation_succeeds(self):
        env = _FakeEnv()
        env.controller.reachable = [copy.deepcopy(env.controller.metadata['agent']['position'])]
        original_step = env.controller.step

        def _step_with_rotation_collision(**kwargs):
            if kwargs.get('action') == 'RotateRight':
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'Agent rotation is blocked by collision.'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_rotation_collision
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        result = executor.scan_in_place(num_views=4)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['num_views_completed'], 0)
        self.assertEqual(len(result['clearance_attempts']), 4)
        self.assertFalse(any((item['success'] for item in result['clearance_attempts'])))
        self.assertEqual(sum((action.get('action') == 'RotateRight' for action in env.controller.actions)), 1)

    def test_look_for_reports_motion_failure_separately_from_not_found(self):
        env = _VisualAttachFakeEnv()
        original_step = env.controller.step

        def _step_with_yaw_error(**kwargs):
            if kwargs.get('action') == 'RotateRight':
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'Invalid rotation request.'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_yaw_error
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        motion_failed = executor.look_for_object('Mug|0')
        self.assertEqual(motion_failed['status'], 'failed')
        self.assertTrue(motion_failed['motion_failed'])
        env = _VisualAttachFakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        not_found = executor.look_for_object('Mug|0')
        self.assertEqual(not_found['status'], 'failed')
        self.assertFalse(not_found['motion_failed'])
        self.assertFalse(not_found['object_visible'])

    def test_visual_attach_navigation_teleports_one_waypoint_then_continues(self):
        env = _VisualAttachFakeEnv()
        env.controller.metadata['agent']['cameraHorizon'] = 0.0
        original_step = env.controller.step

        def _teleport_aware_step(**kwargs):
            if kwargs.get('action') == 'TeleportFull':
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['agent'] = {'position': dict(kwargs['position']), 'rotation': dict(kwargs['rotation']), 'cameraHorizon': kwargs['horizon']}
                env.controller.metadata['lastActionSuccess'] = True
                env.controller.metadata['errorMessage'] = ''
                env.controller._update_visibility()
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            if kwargs.get('action') in {'RotateLeft', 'RotateRight'}:
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'opened drawer blocks rotation'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _teleport_aware_step
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        pose = InteractionPose(key=(3, 0), position={'x': 0.75, 'y': 0.9, 'z': 0.0}, rotation_y=90, horizon=0, standing=True, distance=1.0, path_length=3, score=0.0)
        result = executor.navigate_to_pose(pose, label='navigate around drawer')
        self.assertEqual(result['status'], 'success')
        teleport_actions = [action for action in env.controller.actions if action.get('action') == 'TeleportFull']
        self.assertEqual(len(teleport_actions), 1)
        self.assertEqual(teleport_actions[0]['position']['x'], 0.25)
        self.assertNotEqual(teleport_actions[0]['position'], pose.position)
        self.assertEqual(result['final_pose']['position']['x'], 0.75)
        self.assertFalse(result['waypoint_fallbacks'][0]['destination_is_waypoint'])
        self.assertEqual(result['waypoint_fallbacks'][0]['to_waypoint'], [1, 0])
        self.assertTrue(any((action.get('action') == 'MoveAhead' and (not action.get('forceAction')) and (action is not env.controller.actions[0]) for action in env.controller.actions)))

    def test_visual_attach_waypoint_may_equal_adjacent_destination(self):
        env = _VisualAttachFakeEnv()
        original_step = env.controller.step
        failed_moves = 0

        def _teleport_aware_step(**kwargs):
            nonlocal failed_moves
            if kwargs.get('action') == 'TeleportFull':
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['agent'] = {'position': dict(kwargs['position']), 'rotation': dict(kwargs['rotation']), 'cameraHorizon': kwargs['horizon']}
                env.controller.metadata['lastActionSuccess'] = True
                env.controller.metadata['errorMessage'] = ''
                env.controller._update_visibility()
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            if kwargs.get('action') == 'MoveAhead' and failed_moves < 2:
                failed_moves += 1
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'movement blocked'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _teleport_aware_step
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        pose = InteractionPose(key=(1, 0), position={'x': 0.25, 'y': 0.9, 'z': 0.0}, rotation_y=90, horizon=0, standing=True, distance=1.0, path_length=1, score=0.0)
        result = executor.navigate_to_pose(pose, label='adjacent destination')
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['waypoint_fallbacks'][0]['destination_is_waypoint'])
        self.assertEqual(result['waypoint_fallbacks'][0]['to_waypoint'], [1, 0])

    def test_failed_waypoint_teleport_blocks_edge_and_replans(self):
        env = _VisualAttachFakeEnv()
        original_step = env.controller.step
        failed_moves = 0

        def _step(**kwargs):
            nonlocal failed_moves
            if kwargs.get('action') == 'TeleportFull':
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'waypoint is obstructed'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            if kwargs.get('action') == 'MoveAhead' and failed_moves < 2:
                failed_moves += 1
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'edge is obstructed'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        pose = InteractionPose(key=(2, 0), position={'x': 0.5, 'y': 0.9, 'z': 0.0}, rotation_y=90, horizon=0, standing=True, distance=1.0, path_length=2, score=0.0)
        result = executor.navigate_to_pose(pose, label='replan after blocked edge')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['waypoint_fallbacks'][0]['status'], 'failed')
        self.assertIn(((0, 0), (1, 0)), executor.blocked_edges)
        teleport_actions = [action for action in env.controller.actions if action.get('action') == 'TeleportFull']
        self.assertEqual(len(teleport_actions), 1)
        self.assertEqual(teleport_actions[0]['position']['x'], 0.25)
        self.assertNotEqual(teleport_actions[0]['position'], pose.position)

    def test_no_path_does_not_trigger_waypoint_teleport(self):
        env = _VisualAttachFakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor.blocked_edges.update({((0, 0), (0, 1)), ((0, 0), (1, 0))})
        pose = InteractionPose(key=(2, 0), position={'x': 0.5, 'y': 0.9, 'z': 0.0}, rotation_y=90, horizon=0, standing=True, distance=1.0, path_length=2, score=0.0)
        with self.assertRaisesRegex(ProcTHORLowLevelExecutionError, 'No unblocked path'):
            executor.navigate_to_pose(pose, label='no route')
        self.assertNotIn('TeleportFull', [action.get('action') for action in env.controller.actions])

    def test_object_position_navigation_aims_at_public_position(self):
        env = _FakeEnv()
        mug = env.controller._object('Mug|0')
        mug['position'] = {'x': 1.75, 'y': 0.1, 'z': 0.0}
        env.controller.reachable = [point for point in env.controller.reachable if abs(point['x'] - 1.75) > 1e-06]
        env.controller._update_visibility()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        result = executor.navigate_to_position({'x': 1.75, 'y': 0.1, 'z': 0.0}, target_ref='mug', resolved_via='last_known_position', object_id='Mug|0')
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['object_visible'])
        self.assertTrue(result['interaction_ready'])
        self.assertEqual(result['final_pose']['rotation']['y'], 90.0)
        self.assertEqual(result['final_pose']['cameraHorizon'], 60.0)

    def test_last_known_position_rotation_collision_tries_nearby_pose(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        attempted_keys = []

        def _navigate(pose, **_kwargs):
            attempted_keys.append(pose.key)
            if len(attempted_keys) == 1:
                raise ProcTHORLowLevelExecutionError("Could not rotate to yaw 270: Collided with static sim object: 'Sink|0'.")
            return {'status': 'success', 'target_pose': pose.to_dict()}
        executor.navigate_to_pose = _navigate
        executor.is_object_visible = lambda _object_id: True
        executor.is_object_interaction_ready = lambda _object_id: True
        result = executor.navigate_to_position({'x': 1.75, 'y': 0.1, 'z': 0.0}, target_ref='mug', resolved_via='last_known_position', object_id='Mug|0')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(attempted_keys), 2)
        self.assertNotEqual(attempted_keys[0], attempted_keys[1])
        self.assertEqual([item['status'] for item in result['last_known_position_navigation_attempts']], ['failed', 'success'])

    def test_last_known_position_non_rotation_failure_does_not_fallback(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        attempted_keys = []

        def _navigate(pose, **_kwargs):
            attempted_keys.append(pose.key)
            raise ProcTHORLowLevelExecutionError('No unblocked path remains.')
        executor.navigate_to_pose = _navigate
        result = executor.navigate_to_position({'x': 1.75, 'y': 0.1, 'z': 0.0}, target_ref='mug', resolved_via='last_known_position', object_id='Mug|0', allow_failure=True)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(attempted_keys), 1)
        self.assertNotIn('last_known_position_navigation_attempts', result)

    def test_position_navigation_without_stale_object_does_not_fallback(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        attempted_keys = []

        def _navigate(pose, **_kwargs):
            attempted_keys.append(pose.key)
            raise ProcTHORLowLevelExecutionError("Could not rotate to yaw 270: Collided with static sim object: 'Sink|0'.")
        executor.navigate_to_pose = _navigate
        result = executor.navigate_to_position({'x': 1.75, 'y': 0.1, 'z': 0.0}, resolved_via='room_label', allow_failure=True)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(attempted_keys), 1)

    def test_last_known_position_fallback_is_bounded(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        attempted_keys = []

        def _navigate(pose, **_kwargs):
            attempted_keys.append(pose.key)
            raise ProcTHORLowLevelExecutionError("Could not rotate to yaw 270: Collided with static sim object: 'Sink|0'.")
        executor.navigate_to_pose = _navigate
        result = executor.navigate_to_position({'x': 1.75, 'y': 0.1, 'z': 0.0}, resolved_via='last_known_position', object_id='Mug|0', allow_failure=True)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(attempted_keys), 4)
        self.assertEqual(len(result['last_known_position_navigation_attempts']), 4)

    def test_open_openable_obstacle_does_not_add_aabb_blocked_edges(self):
        env = _FakeEnv()
        env.controller.metadata['objects'].append(_obj('Drawer|0', 'Drawer', openable=True, isOpen=True, axisAlignedBoundingBox={'center': {'x': 0.125, 'y': 0.5, 'z': 0.0}, 'size': {'x': 0.08, 'y': 0.2, 'z': 0.08}}))
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        points = {(0, 0): {'x': 0.0, 'z': 0.0}, (1, 0): {'x': 0.25, 'z': 0.0}, (0, 1): {'x': 0.0, 'z': 0.25}, (1, 1): {'x': 0.25, 'z': 0.25}}
        path = _astar_path(points, (0, 0), (1, 0), executor._planning_blocked_edges(points))
        self.assertEqual(path, [(0, 0), (1, 0)])

    def test_real_move_failure_still_blocks_astar_edge(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor.blocked_edges.add(((0, 0), (1, 0)))
        points = {(0, 0): {'x': 0.0, 'z': 0.0}, (1, 0): {'x': 0.25, 'z': 0.0}, (0, 1): {'x': 0.0, 'z': 0.25}, (1, 1): {'x': 0.25, 'z': 0.25}}
        path = _astar_path(points, (0, 0), (1, 0), executor._planning_blocked_edges(points))
        self.assertEqual(path, [(0, 0), (0, 1), (1, 1), (1, 0)])

    def test_held_navigation_clearance_prioritizes_backoff_without_view_change(self):
        env = _FakeEnv()
        env.controller.metadata['agent']['position'] = {'x': 1.0, 'y': 0.9, 'z': 0.5}
        env.controller.metadata['agent']['rotation'] = {'y': 0.0}
        env.controller.metadata['agent']['cameraHorizon'] = 30.0
        env.controller.metadata['inventoryObjects'] = [{'objectId': 'Mug|0', 'objectType': 'Mug'}]
        env.controller.metadata['objects'].append(_obj('CounterTop|clearance', 'CounterTop', receptacle=True, axisAlignedBoundingBox={'center': {'x': 1.0, 'y': 0.5, 'z': 0.5}, 'size': {'x': 0.2, 'y': 0.2, 'z': 0.2}}))
        env.controller._update_visibility()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        points = _points_by_key(env.controller.reachable, executor.step_size)
        clearance = executor._move_to_held_navigation_clearance(points, target_key=None, label='held clearance', source='test')
        self.assertIsNotNone(clearance)
        self.assertEqual(clearance['strategy'], 'backoff')
        self.assertEqual(env.controller.actions[0]['action'], 'MoveBack')
        self.assertEqual(env.controller.metadata['agent']['rotation']['y'], 0.0)
        self.assertEqual(env.controller.metadata['agent']['cameraHorizon'], 30.0)
        self.assertFalse(any((action['action'] in {'RotateLeft', 'RotateRight', 'LookUp', 'LookDown'} for action in env.controller.actions[:1])))

    def test_held_navigation_clearance_uses_lateral_translation_when_backoff_unavailable(self):
        env = _FakeEnv()
        env.controller.reachable = [{'x': 1.0, 'y': 0.9, 'z': 0.5}, {'x': 1.25, 'y': 0.9, 'z': 0.5}]
        env.controller.metadata['agent']['position'] = {'x': 1.0, 'y': 0.9, 'z': 0.5}
        env.controller.metadata['agent']['rotation'] = {'y': 0.0}
        env.controller.metadata['agent']['cameraHorizon'] = 30.0
        env.controller.metadata['inventoryObjects'] = [{'objectId': 'Mug|0', 'objectType': 'Mug'}]
        env.controller.metadata['objects'].append(_obj('CounterTop|clearance', 'CounterTop', receptacle=True, axisAlignedBoundingBox={'center': {'x': 1.0, 'y': 0.5, 'z': 0.5}, 'size': {'x': 0.2, 'y': 0.2, 'z': 0.2}}))
        env.controller._update_visibility()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        points = _points_by_key(env.controller.reachable, executor.step_size)
        clearance = executor._move_to_held_navigation_clearance(points, target_key=None, label='held clearance', source='test')
        self.assertIsNotNone(clearance)
        self.assertEqual(clearance['strategy'], 'lateral_clearance')
        self.assertEqual(env.controller.actions[0]['action'], 'MoveRight')
        self.assertEqual(env.controller.metadata['agent']['rotation']['y'], 0.0)
        self.assertEqual(env.controller.metadata['agent']['cameraHorizon'], 30.0)
        self.assertFalse(any((action['action'] in {'RotateLeft', 'RotateRight', 'LookUp', 'LookDown'} for action in env.controller.actions[:1])))

    def test_force_open_container_fallback_only_for_held_collision(self):
        env = _FakeEnv()
        env.controller.metadata['inventoryObjects'] = [{'objectId': 'Mug|0', 'objectType': 'Mug'}]
        sink = env.controller._object('Sink|0')
        sink['openable'] = True
        sink['receptacle'] = True
        sink['isOpen'] = False
        original_step = env.controller.step

        def _step_with_held_open_collision(**kwargs):
            if kwargs.get('action') == 'OpenObject':
                env.controller.actions.append(dict(kwargs))
                if not kwargs.get('forceAction'):
                    env.controller.metadata['lastActionSuccess'] = False
                    env.controller.metadata['errorMessage'] = 'a held item: Mug|0 with something if agent opens Sink|0'
                    env.controller.last_event = _FakeEvent(env.controller.metadata)
                    return env.controller.last_event
                sink['isOpen'] = True
                env.controller.metadata['lastActionSuccess'] = True
                env.controller.metadata['errorMessage'] = ''
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_held_open_collision
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        event = executor._step_physical({'action': 'OpenObject', 'objectId': 'Sink|0'}, label='open held-blocked sink', allow_failure=True)
        self.assertTrue(event.metadata['lastActionSuccess'])
        self.assertTrue(env.controller._object('Sink|0')['isOpen'])
        self.assertTrue(executor.action_records[-1]['forceAction'])
        self.assertTrue(executor.action_records[-1]['force_open_container_fallback'])
        self.assertEqual(executor.action_records[-1]['held_object_id'], 'Mug|0')

    def test_arm_open_collision_fallback_official_and_visual_empty_inventory(self):
        for visual_attach in (False, True):
            with self.subTest(visual_attach=visual_attach):
                env, executor = _arm_open_close_collision_case(visual_attach=visual_attach)
                event = executor._step_physical({'action': 'OpenObject', 'objectId': 'Sink|0'}, label='open arm-blocked sink', allow_failure=True)
                self.assertTrue(event.metadata['lastActionSuccess'])
                self.assertEqual(env.controller.metadata['inventoryObjects'], [])
                record = executor.action_records[-1]
                self.assertTrue(record['force_arm_open_close_collision_fallback_applied'])
                self.assertEqual(record['arm_mode'], 'visual_attach' if visual_attach else 'official')
                self.assertTrue(record['post_is_open'])

    def test_arm_close_collision_fallback_official_and_visual(self):
        for visual_attach in (False, True):
            with self.subTest(visual_attach=visual_attach):
                env, executor = _arm_open_close_collision_case(action='CloseObject', visual_attach=visual_attach)
                event = executor._step_physical({'action': 'CloseObject', 'objectId': 'Sink|0'}, label='close arm-blocked sink', allow_failure=True)
                self.assertTrue(event.metadata['lastActionSuccess'])
                record = executor.action_records[-1]
                self.assertFalse(record['desired_is_open'])
                self.assertFalse(record['post_is_open'])
                self.assertTrue(record['force_arm_open_close_collision_fallback_applied'])

    def test_arm_open_close_fallback_rejects_generic_and_non_openable(self):
        cases = ({'error_message': 'object collided with the wall'}, {'openable': False})
        for config in cases:
            with self.subTest(config=config):
                env, executor = _arm_open_close_collision_case(**config)
                event = executor._step_physical({'action': 'OpenObject', 'objectId': 'Sink|0'}, label='rejected fallback', allow_failure=True)
                self.assertFalse(event.metadata['lastActionSuccess'])
                self.assertEqual(len(env.controller.actions), 1)
                self.assertNotIn('force_arm_open_close_collision_fallback', executor.action_records[-1])

    def test_arm_open_close_fallback_rejects_flag_off_and_budget_exhausted(self):
        for config in ({'enabled': False}, {'budget': 0}):
            with self.subTest(config=config):
                env, executor = _arm_open_close_collision_case(**config)
                event = executor._step_physical({'action': 'OpenObject', 'objectId': 'Sink|0'}, label='disabled fallback', allow_failure=True)
                self.assertFalse(event.metadata['lastActionSuccess'])
                self.assertEqual(len(env.controller.actions), 1)

    def test_arm_open_close_forced_success_with_wrong_state_remains_failure(self):
        env, executor = _arm_open_close_collision_case(forced_post_state=False)
        event = executor._step_physical({'action': 'OpenObject', 'objectId': 'Sink|0'}, label='wrong post state', allow_failure=True)
        self.assertFalse(event.metadata['lastActionSuccess'])
        record = executor.action_records[-1]
        self.assertFalse(record['success'])
        self.assertFalse(record['force_arm_open_close_collision_fallback_applied'])
        self.assertFalse(record['post_is_open'])

    def test_moveahead_reports_successful_but_stationary_substep(self):
        env = _FakeEnv()
        env.controller._move = lambda _magnitude: None
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        with self.assertRaisesRegex(ProcTHORLowLevelExecutionError, 'no agent displacement'):
            executor._step_physical({'action': 'MoveAhead', 'moveMagnitude': 0.25}, label='stationary move')

    def test_rotate_to_yaw_tries_opposite_direction_after_collision(self):
        env = _FakeEnv()
        env.controller.metadata['agent']['rotation']['y'] = 180.0
        original_step = env.controller.step

        def _step_with_left_collision(**kwargs):
            if kwargs.get('action') == 'RotateLeft':
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'held item collision'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_with_left_collision
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        steps = executor._rotate_to_yaw(90.0, 'rotate around collision')
        self.assertEqual(steps, 4)
        self.assertEqual(env.controller.metadata['agent']['rotation']['y'], 90.0)
        self.assertEqual([action['action'] for action in env.controller.actions], ['RotateLeft', 'RotateRight', 'RotateRight', 'RotateRight'])

    def test_navigate_adjusts_final_view_when_target_visible_after_grid_navigation(self):
        env = _FakeEnv()
        original_step = env.controller.step

        def _step_failing_look_down(**kwargs):
            if kwargs.get('action') == 'LookDown':
                env.controller.actions.append(dict(kwargs))
                env.controller.metadata['lastActionSuccess'] = False
                env.controller.metadata['errorMessage'] = 'held item blocks look'
                env.controller.last_event = _FakeEvent(env.controller.metadata)
                return env.controller.last_event
            return original_step(**kwargs)
        env.controller.step = _step_failing_look_down
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor._get_interactable_poses = lambda _object_id: [{'x': 1.5, 'y': 0.9, 'z': 0.0, 'rotation': 90.0, 'horizon': 60.0, 'standing': True}]
        result = executor.navigate_to_object_id('Mug|0', target_ref='target_object')
        self.assertEqual(result['status'], 'success')
        self.assertIn('LookDown', [action['action'] for action in env.controller.actions])

    def test_object_navigation_replans_to_alternate_pose_after_final_rotation_collision(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        first = InteractionPose(key=(1, 0), position={'x': 0.25, 'y': 0.9, 'z': 0.0}, rotation_y=90.0, horizon=0.0, standing=True, distance=1.0, path_length=1, score=0.0)
        alternate = InteractionPose(key=(2, 0), position={'x': 0.5, 'y': 0.9, 'z': 0.0}, rotation_y=180.0, horizon=0.0, standing=True, distance=1.0, path_length=2, score=1.0)
        ranked_calls = 0
        navigated = []
        rotations = 0

        def _rank(*_args, **_kwargs):
            nonlocal ranked_calls
            ranked_calls += 1
            return [first] if ranked_calls == 1 else [alternate]

        def _navigate(pose, label):
            del label
            navigated.append(pose.key)
            return {'status': 'success', 'waypoint_fallbacks': []}

        def _rotate(_yaw, _label):
            nonlocal rotations
            rotations += 1
            if rotations == 1:
                raise ProcTHORLowLevelExecutionError('held object blocks final rotation')
            return 1
        executor.rank_interaction_poses = _rank
        executor._navigate_to_grid_key = _navigate
        executor._rotate_to_yaw = _rotate
        executor._set_standing = lambda *_args: 0
        executor._look_to_horizon = lambda *_args: 0
        executor.is_object_visible = lambda _object_id: True
        readiness_checks = 0

        def _ready(_object_id):
            nonlocal readiness_checks
            readiness_checks += 1
            return readiness_checks > 1
        executor.is_object_interaction_ready = _ready
        result = executor.navigate_to_object_id('Mug|0', target_ref='mug')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(navigated, [(1, 0), (2, 0)])
        self.assertEqual(result['attempts'][0]['error_message'], 'orientation adjustment failed')
        self.assertEqual(result['interaction_pose_source'], alternate.source)

    def test_ranking_prefers_standing_pose_before_crouch_pose(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec())
        executor._get_interactable_poses = lambda _object_id: [{'x': 0.5, 'y': 0.9, 'z': 0.0, 'rotation': 90.0, 'horizon': -30.0, 'standing': False}]
        poses = executor.rank_interaction_poses('Mug|0', prefer_standing=True)
        self.assertTrue(poses)
        self.assertTrue(poses[0].standing)

    def test_ranking_falls_back_to_simulator_pose_after_natural_probes(self):
        env = _FakeEnv()
        executor = ProcTHORLowLevelExecutor(env, _task_spec(), max_natural_pose_probes=2)
        executor._get_interactable_poses = lambda _object_id: [{'x': 0.5, 'y': 0.9, 'z': 0.0, 'rotation': 90.0, 'horizon': -30.0, 'standing': False}]
        first = executor.rank_interaction_poses('Mug|0', prefer_standing=True)[0]
        second = executor.rank_interaction_poses('Mug|0', exclude={tuple((first.key, first.rotation_y, first.horizon, first.standing))}, prefer_standing=True)[0]
        poses = executor.rank_interaction_poses('Mug|0', exclude={tuple((first.key, first.rotation_y, first.horizon, first.standing)), tuple((second.key, second.rotation_y, second.horizon, second.standing))}, prefer_standing=True)
        self.assertTrue(poses)
        self.assertEqual(poses[0].source, 'simulator_interactable_pose')

    def test_astar_turn_penalty_prefers_low_turn_path(self):
        points = {(x, z): {'x': x / 4.0, 'y': 0.9, 'z': z / 4.0} for x in range(4) for z in range(4)}
        path = _astar_path(points, (0, 0), (3, 3), turn_penalty=3.0)
        self.assertIsNotNone(path)
        self.assertLessEqual(_path_num_turns(path), 1)
if __name__ == '__main__':
    unittest.main()
