from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional
import numpy as np

class FakeMetadata:

    def __init__(self, objects: Optional[List[dict]]=None):
        checker = np.indices((64, 64)).sum(axis=0) % 2 * 255
        self.frame = np.repeat(checker[:, :, None], 3, axis=2).astype(np.uint8)
        self.metadata = {'agent': {'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'cameraHorizon': 0.0, 'isStanding': True}, 'objects': list(objects or []), 'lastActionSuccess': True, 'errorMessage': ''}

class FakeController:

    def __init__(self, objects: Optional[List[dict]]=None):
        self.last_event = FakeMetadata(objects)
        self._objects: Dict[str, dict] = {obj['objectId']: dict(obj) for obj in objects or []}
        self._sync_event_objects()
        self.step_history: List[dict] = []
        self.stopped = False

    def step(self, **kwargs):
        self.step_history.append(dict(kwargs))
        action = kwargs.get('action')
        object_id = kwargs.get('objectId')
        if action == 'OpenObject' and object_id in self._objects:
            obj = self._objects[object_id]
            if 'openness' in kwargs:
                obj['openness'] = float(kwargs['openness'])
                obj['isOpen'] = float(kwargs['openness']) > 0.0
            else:
                obj['isOpen'] = True
                obj['openness'] = 1.0
        elif action == 'CloseObject' and object_id in self._objects:
            self._objects[object_id]['isOpen'] = False
            self._objects[object_id]['openness'] = 0.0
        elif action == 'ToggleObjectOn' and object_id in self._objects:
            self._objects[object_id]['isToggled'] = True
        elif action == 'ToggleObjectOff' and object_id in self._objects:
            self._objects[object_id]['isToggled'] = False
        elif action == 'DirtyObject' and object_id in self._objects:
            self._objects[object_id]['isDirty'] = True
        elif action == 'CleanObject' and object_id in self._objects:
            self._objects[object_id]['isDirty'] = False
        elif action == 'SetObjectStates':
            states = kwargs.get('SetObjectStates') or {}
            object_type = states.get('objectType')
            for obj in self._objects.values():
                if object_type and obj.get('objectType') != object_type:
                    continue
                for key, value in states.items():
                    if key != 'objectType':
                        obj[key] = value
        elif action == 'PickupObject' and object_id in self._objects:
            self.last_event.metadata['inventoryObjects'] = [{'objectId': object_id}]
        elif action == 'DropHandObject':
            self.last_event.metadata['inventoryObjects'] = []
        elif action == 'TeleportObject' and object_id in self._objects:
            pos = kwargs.get('position') or {}
            self._objects[object_id]['position'] = dict(pos)
        elif action == 'GetSpawnCoordinatesAboveReceptacle' and object_id in self._objects:
            pos = self._objects[object_id].get('position') or {}
            self.last_event.metadata['actionReturn'] = [{'x': float(pos.get('x', 0.0)) + 0.1, 'y': float(pos.get('y', 0.9)) + 0.1, 'z': float(pos.get('z', 0.0)) + 0.1}]
        self._sync_event_objects()
        return self.last_event

    def set_visibility(self, object_id: str, visible: bool) -> None:
        if object_id in self._objects:
            self._objects[object_id]['visible'] = bool(visible)
        self._sync_event_objects()

    def stop(self):
        self.stopped = True

    def _sync_event_objects(self) -> None:
        self.last_event.metadata['objects'] = [dict(o) for o in self._objects.values()]

class FakeExecutor:

    def __init__(self):
        self.stream_callback = lambda action_record, event: dict(action_record)

class RichFakeToolEnv:

    def __init__(self, objects: List[dict], roles: Mapping[str, str], success_after: int=1):
        self.controller = FakeController(objects)
        self.executor = FakeExecutor()
        self.high_level_step = 0
        self.total_low_level_steps = 0
        self.invalid_tool_calls = 0
        self.model_stopped = False
        self.tool_history: List[dict] = []
        self.role_object_ids: dict = dict(roles)
        self._success_after = success_after
        self._calls = 0
        self.width = 320
        self.height = 240
        self.cloud_rendering = False
        self.max_high_level_steps = 20
        self.max_total_low_level_steps = 100
        self.max_invalid_tool_calls = 5
        self.max_nav_steps_per_tool = 50
        self.move_execution_step = None
        self.rotate_execution_degrees = None
        self.look_execution_degrees = None
        self.snap_to_grid = True

    def reset_to_task(self, task_spec):
        self.high_level_step = 0
        self._calls = 0
        self.model_stopped = False
        return {'head_rgb': None, 'prompt_context': {}}

    def _objects_by_id(self):
        return {obj['objectId']: obj for obj in self.controller.last_event.metadata['objects']}

    def check_goal(self, model_stopped=False):
        if self._calls >= self._success_after:
            success = 1.0
        else:
            success = 0.0
        return {'task_success': success, 'task_progress': success, 'predicate_success': success, 'completion_candidate': success == 1.0, 'require_model_stop': False, 'model_stopped': model_stopped}

    def get_prompt_context(self):
        return {'tools': ['navigate_to', 'stop']}

    def _observation(self):
        return {'head_rgb': None}

    def call_tool(self, tool_name, arguments=None, reasoning=''):
        self.high_level_step += 1
        self._calls += 1
        if tool_name == 'stop':
            self.model_stopped = True
        success = True
        goal = self.check_goal(model_stopped=self.model_stopped)
        done = float(goal['task_success']) == 1.0 or tool_name == 'stop'
        reason = 'task_success' if done and goal['task_success'] == 1.0 else 'model_stopped' if tool_name == 'stop' else None
        transition = {'tool_name': tool_name, 'arguments': dict(arguments or {}), 'reasoning': reasoning, 'high_level_step': self.high_level_step, 'success': success, 'env_feedback': 'ok', 'done': done, 'done_reason': reason, 'observation': {'head_rgb': None}, 'goal_result': goal, 'raw_execution': {'status': 'success'}, 'low_level_actions': [], 'num_low_level_steps': 0, 'episode_metrics': {}, 'ready_target': None, 'elapsed_seconds': 0.0}
        self.tool_history.append(transition)
        return transition
__all__ = ['FakeController', 'FakeExecutor', 'FakeMetadata', 'RichFakeToolEnv']
