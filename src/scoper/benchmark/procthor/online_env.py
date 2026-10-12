from __future__ import annotations
import json
import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from scoper.benchmark.common.object_catalog import ObjectCatalog, build_object_catalog
from scoper.benchmark.procthor.agent_catalog_filter import build_inventory_view
from scoper.benchmark.procthor.online_task_spec import validate_online_task_spec
from scoper.benchmark.procthor.paths import resolve_repo_path
from scoper.benchmark.procthor.predicate_goal_checker import check_goal
from scoper.benchmark.procthor.procthor_dataset_assets import is_prior_house_uri, load_prior_house
from scoper.benchmark.procthor.spatial_state import room_id_for_position

class ProcTHOROnlineEnvError(RuntimeError):
    pass

class ProcTHOROnlineEnv:

    def __init__(self, width: int=320, height: int=240, cloud_rendering: bool=False, gpu_device: Optional[int]=None, render_depth_image: bool=False, render_instance_segmentation: bool=False, snap_to_grid: Optional[bool]=None, agent_radius: Optional[float]=None, manipulathor_arm: bool=False, manipulathor_visibility_distance: float=1.5, manipulathor_field_of_view: float=75.0):
        self.width = width
        self.height = height
        self.cloud_rendering = cloud_rendering
        self.gpu_device = gpu_device
        self.render_depth_image = render_depth_image
        self.render_instance_segmentation = render_instance_segmentation
        self.snap_to_grid = snap_to_grid
        self.agent_radius = None if agent_radius is None else float(agent_radius)
        self.manipulathor_arm = bool(manipulathor_arm)
        self.manipulathor_visibility_distance = float(manipulathor_visibility_distance)
        self.manipulathor_field_of_view = float(manipulathor_field_of_view)
        self.controller = None
        self.current_task_spec: Optional[Mapping[str, Any]] = None
        self.current_house: Optional[Mapping[str, Any]] = None
        self.object_catalog: Optional[ObjectCatalog] = None
        self.role_object_ids: Dict[str, str] = {}
        self.scene_mutations: List[dict] = []
        self.init_mutations: List[dict] = []
        self._sticky_object_states: Dict[str, Dict[str, Any]] = {}
        self._trajectory_capture_enabled = False
        self._trajectory_audit_sink = None
        self._trajectory_episode_context = None
        self._trajectory_event_ordinal = 0

    def configure_trajectory_capture(self, episode_context: Any, sink: Any, execution_profile: str, allowed_agent_fallbacks: Sequence[str]=()) -> None:
        from scoper.benchmark.procthor.export.execution_audit import configure_trajectory_capture
        configure_trajectory_capture(self, episode_context=episode_context, execution_profile=execution_profile, sink=sink, allowed_agent_fallbacks=tuple(allowed_agent_fallbacks))

    def reset_to_task(self, task_spec: Mapping[str, Any]) -> dict:
        validate_online_task_spec(task_spec)
        runtime_task_spec = copy.deepcopy(dict(task_spec))
        house = self._load_house(runtime_task_spec['source']['house_path'])
        from scoper.benchmark.common.scene_exclusions import prune_procthor_house
        house = prune_procthor_house(house, runtime_task_spec)
        self.current_house = house
        self.close()
        self._sticky_object_states = {}
        self.controller = self._create_controller(house)
        from scoper.benchmark.procthor.export.execution_audit import install_audited_controller
        install_audited_controller(self)
        self._apply_agent_radius()
        self._step(action='Pass')
        self.current_task_spec = runtime_task_spec
        self.role_object_ids = self._resolve_role_object_ids(runtime_task_spec)
        self.scene_mutations = []
        self.init_mutations = []
        self._apply_object_state_overrides(runtime_task_spec['init'].get('object_state_overrides', []))
        self._apply_object_pose_overrides(runtime_task_spec['init'].get('object_pose_overrides', []))
        self._teleport_agent(runtime_task_spec['init']['agent_pose'])
        self._refresh_role_objects_exist(runtime_task_spec)
        self.object_catalog = build_object_catalog(self._metadata_with_sticky_object_states())
        return {'head_rgb': self.controller.last_event.frame, 'task_id': runtime_task_spec['id'], 'instruction': runtime_task_spec['instruction'], 'scene_mutations': [dict(mutation) for mutation in self.scene_mutations], 'init_mutations': [dict(mutation) for mutation in self.init_mutations]}

    def get_task_context(self) -> dict:
        if self.current_task_spec is None or self.object_catalog is None:
            raise ProcTHOROnlineEnvError('Call reset_to_task before get_task_context.')
        metadata = self._metadata()
        return {'task_id': self.current_task_spec['id'], 'instruction': self.current_task_spec['instruction'], 'object_catalog': self.object_catalog.to_prompt_context(), 'held_objects': [obj.get('objectType') for obj in build_inventory_view(metadata)]}

    def check_goal(self, model_stopped: bool=False) -> dict:
        if self.current_task_spec is None:
            raise ProcTHOROnlineEnvError('Call reset_to_task before check_goal.')
        return check_goal(self.current_task_spec, self._metadata_with_sticky_object_states(), model_stopped=model_stopped)

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def _load_house(self, house_path: str) -> dict:
        if is_prior_house_uri(house_path):
            return load_prior_house(house_path)
        resolved_house_path = resolve_repo_path(house_path)
        try:
            house = json.loads(resolved_house_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ProcTHOROnlineEnvError("Invalid ProcTHOR house JSON '{}': {}.".format(house_path, exc)) from exc
        if not isinstance(house, dict):
            raise ProcTHOROnlineEnvError("ProcTHOR house JSON '{}' must contain an object.".format(house_path))
        return house

    def _create_controller(self, house: Mapping[str, Any]) -> Any:
        from ai2thor.controller import Controller
        kwargs = {'scene': dict(house), 'width': self.width, 'height': self.height, 'renderDepthImage': self.render_depth_image, 'renderInstanceSegmentation': self.render_instance_segmentation}
        if self.cloud_rendering:
            from ai2thor.platform import CloudRendering
            kwargs['platform'] = CloudRendering
        if self.gpu_device is not None:
            kwargs['gpu_device'] = self.gpu_device
        if self.snap_to_grid is not None:
            kwargs['snapToGrid'] = bool(self.snap_to_grid)
        if self.manipulathor_arm:
            kwargs.update({'agentMode': 'arm', 'massThreshold': None, 'visibilityDistance': self.manipulathor_visibility_distance, 'fieldOfView': self.manipulathor_field_of_view})
        executable = os.environ.get('EB_PROCTHOR_LOCAL_EXECUTABLE')
        if executable:
            if not Path(executable).is_file() or not os.access(executable, os.X_OK):
                raise ProcTHOROnlineEnvError('Configured local Unity executable is unavailable')
            kwargs['local_executable_path'] = executable
            if self.cloud_rendering:

                class LocalCloudRenderingController(Controller):

                    def start(controller, *args, **start_kwargs):
                        controller._build.platform = CloudRendering
                        return super().start(*args, **start_kwargs)
                return LocalCloudRenderingController(**kwargs)
        return Controller(**kwargs)

    def _apply_agent_radius(self) -> None:
        if self.agent_radius is None:
            return
        self._step(action='SetAgentRadius', agentRadius=float(self.agent_radius))

    def _step(self, **kwargs: Any) -> Any:
        if self.controller is None:
            raise ProcTHOROnlineEnvError('Controller is not initialized.')
        event = self.controller.step(**kwargs)
        if not bool(event.metadata.get('lastActionSuccess', True)):
            raise ProcTHOROnlineEnvError("AI2-THOR action '{}' failed: {}".format(kwargs.get('action'), event.metadata.get('errorMessage', '')))
        return event

    def _metadata(self) -> Mapping[str, Any]:
        if self.controller is None:
            raise ProcTHOROnlineEnvError('Controller is not initialized.')
        return self.controller.last_event.metadata

    def _metadata_with_sticky_object_states(self) -> Mapping[str, Any]:
        metadata = self._metadata()
        sticky = getattr(self, '_sticky_object_states', None) or {}
        if not sticky:
            return metadata
        merged = dict(metadata)
        merged_objects = []
        for obj in metadata.get('objects', []):
            if not isinstance(obj, Mapping):
                merged_objects.append(obj)
                continue
            object_id = obj.get('objectId')
            if isinstance(object_id, str) and object_id in sticky:
                updated = dict(obj)
                updated.update(sticky[object_id])
                merged_objects.append(updated)
            else:
                merged_objects.append(obj)
        merged['objects'] = merged_objects
        return merged

    def _set_sticky_object_state(self, object_id: str, **state: Any) -> None:
        if not object_id:
            return
        from scoper.benchmark.procthor.export.execution_audit import emit_overlay_audit
        emit_overlay_audit(self, kind='sticky_write', details={'object_id': str(object_id), 'fields': sorted(state), 'values': copy.deepcopy(dict(state))})
        current = self._sticky_object_states.setdefault(str(object_id), {})
        current.update(dict(state))

    def _resolve_role_object_ids(self, task_spec: Mapping[str, Any]) -> Dict[str, str]:
        objects_by_id = self._objects_by_id()
        role_object_ids = {}
        for role_name, role in task_spec['roles'].items():
            object_id = role['object_id']
            obj = objects_by_id.get(object_id)
            if obj is None:
                raise ProcTHOROnlineEnvError("Role '{}' object id not found after reset: {}.".format(role_name, object_id))
            expected_type = role.get('object_type')
            actual_type = obj.get('objectType') or object_id.split('|', 1)[0]
            if expected_type and actual_type != expected_type:
                raise ProcTHOROnlineEnvError("Role '{}' expected object type '{}', got '{}'.".format(role_name, expected_type, actual_type))
            role_object_ids[role_name] = object_id
        return role_object_ids

    def _refresh_role_objects_exist(self, task_spec: Mapping[str, Any]) -> None:
        objects_by_id = self._objects_by_id()
        for role_name, object_id in self.role_object_ids.items():
            if object_id not in objects_by_id:
                raise ProcTHOROnlineEnvError("Role '{}' object disappeared during initialization: {}.".format(role_name, object_id))

    def _apply_object_state_overrides(self, overrides: Any) -> None:
        for index, override in enumerate(overrides):
            override_type = override.get('type')
            if override_type == 'set_open_state':
                self._apply_open_state_override(index, override)
            elif override_type == 'set_toggle_state':
                self._apply_toggle_state_override(index, override)
            elif override_type == 'set_dirty_state':
                self._apply_dirty_state_override(index, override)
            else:
                raise ProcTHOROnlineEnvError('Unsupported init override type at index {}: {}.'.format(index, override_type))

    def _apply_object_pose_overrides(self, overrides: Sequence[Mapping[str, Any]]) -> None:
        for index, override in enumerate(overrides):
            override_type = override.get('type')
            if override_type == 'relocate_object':
                self._apply_relocate_object(index, override)
            elif override_type == 'clear_receptacle_clutter':
                self._apply_clear_receptacle_clutter(index, override)
            else:
                raise ProcTHOROnlineEnvError('Unsupported pose override type at index {}: {}.'.format(index, override_type))

    def _apply_relocate_object(self, index: int, override: Mapping[str, Any]) -> None:
        object_role = override.get('object_role')
        if not isinstance(object_role, str) or not object_role:
            raise ProcTHOROnlineEnvError('Pose override {} must include object_role.'.format(index))
        object_id = self.role_object_ids.get(object_role)
        if object_id is None:
            raise ProcTHOROnlineEnvError("Pose override {} references unknown role '{}'.".format(index, object_role))
        parent_id = self._closed_openable_parent_of(object_id)
        target_id = self.role_object_ids.get(override.get('to_receptacle_role'))
        target_was_closed = False
        if override.get('require_inside_receptacle') and target_id:
            override = dict(override)
            override['_closed_target_bounds'] = dict(self._objects_by_id().get(target_id, {}).get('axisAlignedBoundingBox') or {})
        if override.get('open_target_during_relocate') and target_id:
            target = self._objects_by_id().get(target_id, {})
            target_was_closed = bool(target.get('openable')) and (not bool(target.get('isOpen')))
            if target_was_closed:
                self._step(action='OpenObject', objectId=target_id, forceAction=True)
        parent_was_open: Optional[bool] = None
        if parent_id and override.get('open_parent_during_relocate', True):
            parent_obj = self._objects_by_id().get(parent_id, {})
            parent_was_open = bool(parent_obj.get('isOpen', False))
            if not parent_was_open:
                self._step(action='OpenObject', objectId=parent_id, forceAction=True)
        try:
            if override.get('mode') == 'near_primary_object_floor':
                self._apply_floor_relocate_object(index, override, object_role=object_role, object_id=object_id, parent_id=parent_id, parent_was_open=parent_was_open)
            else:
                self._apply_receptacle_relocate_object(index, override, object_role=object_role, object_id=object_id, parent_id=parent_id, parent_was_open=parent_was_open)
        except Exception:
            self._restore_relocate_parent_state(parent_id, parent_was_open, override)
            raise
        finally:
            if target_was_closed:
                self._step(action='CloseObject', objectId=target_id, forceAction=True)

    def _restore_relocate_parent_state(self, parent_id: Optional[str], parent_was_open: Optional[bool], override: Mapping[str, Any]) -> None:
        if parent_id and parent_was_open is False and override.get('restore_parent_state_after_relocate', True):
            self._step(action='CloseObject', objectId=parent_id, forceAction=True)

    def _apply_receptacle_relocate_object(self, index: int, override: Mapping[str, Any], *, object_role: str, object_id: str, parent_id: Optional[str], parent_was_open: Optional[bool]) -> None:
        target_role = override.get('to_receptacle_role')
        if not isinstance(target_role, str) or not target_role:
            raise ProcTHOROnlineEnvError('Pose override {} must include to_receptacle_role.'.format(index))
        target_receptacle_id = self.role_object_ids.get(target_role)
        if target_receptacle_id is None:
            raise ProcTHOROnlineEnvError("Pose override {} references unknown target role '{}'.".format(index, target_role))
        spawn_event = self._step(action='GetSpawnCoordinatesAboveReceptacle', objectId=target_receptacle_id, anywhere=True)
        coordinates = spawn_event.metadata.get('actionReturn') or []
        if override.get('prefer_receptacle_center'):
            center = self._objects_by_id().get(target_receptacle_id, {}).get('position') or {}
            coordinates = sorted(coordinates, key=lambda p: sum(((float(p.get(k, 0)) - float(center.get(k, 0))) ** 2 for k in ('x', 'z'))))
        if override.get('collision_checked_relocation'):
            coordinates = [dict(point, y=float(point['y']) + lift) for point in coordinates for lift in (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.14, 0.18, 0.22)]
        if not coordinates:
            raise ProcTHOROnlineEnvError('Pose override {}: no spawn coordinates available on {} for relocating {}.'.format(index, target_receptacle_id, object_id))
        object_obj = self._objects_by_id().get(object_id, {})
        bounds = override.get('_closed_target_bounds') or {}
        source_bounds = object_obj.get('axisAlignedBoundingBox') or {}
        if override.get('require_inside_receptacle') and (not (bounds.get('center') and bounds.get('size') and source_bounds.get('size'))):
            raise ProcTHOROnlineEnvError('Interior placement requires closed-container and object bounds')
        if override.get('require_inside_receptacle'):
            from scoper.benchmark.procthor.interior_placement import interior_positions
            coordinates = interior_positions(bounds, object_obj) + coordinates
        before_snapshot = _object_state_snapshot(object_obj)
        object_rotation = dict(object_obj.get('rotation') or {'x': 0.0, 'y': 0.0, 'z': 0.0})
        chosen = None
        last_teleport_error = ''
        for coordinate in coordinates:
            if not isinstance(coordinate, Mapping):
                continue
            candidate = dict(coordinate)
            if override.get('require_inside_receptacle'):
                fits = all((abs(float(candidate[k]) + float(source_bounds['center'][k]) - float(object_obj['position'][k]) - float(bounds['center'][k])) + float(source_bounds['size'][k]) * 0.5 <= float(bounds['size'][k]) * 0.5 - 0.002 for k in ('x', 'y', 'z')))
                if not fits:
                    last_teleport_error = 'candidate outside closed container interior bounds'
                    continue
            teleport_options = {}
            if override.get('keep_object_kinematic'):
                teleport_options['forceKinematic'] = True
            event = self.controller.step(action='TeleportObject', objectId=object_id, position=candidate, rotation=object_rotation, forceAction=not bool(override.get('collision_checked_relocation')), **teleport_options)
            if bool(event.metadata.get('lastActionSuccess', False)):
                if override.get('require_inside_receptacle'):
                    placed = self._objects_by_id().get(object_id, {})
                    if target_receptacle_id not in (placed.get('parentReceptacles') or []):
                        last_teleport_error = 'interior candidate belongs to another receptacle'
                        continue
                chosen = candidate
                break
            last_teleport_error = event.metadata.get('errorMessage', '')
        if chosen is None:
            raise ProcTHOROnlineEnvError('Pose override {}: no stable spawn coordinate on {} for relocating {}. Last error: {}'.format(index, target_receptacle_id, object_id, last_teleport_error))
        self._restore_relocate_parent_state(parent_id, parent_was_open, override)
        self._step(action='Pass')
        after_snapshot = _object_state_snapshot(self._objects_by_id().get(object_id, {}))
        mutation_record = {'source': 'init.object_pose_overrides', 'override_index': index, 'type': 'relocate_object', 'object_role': object_role, 'object_id': object_id, 'mode': override.get('mode'), 'to_receptacle_role': target_role, 'to_receptacle_object_id': target_receptacle_id, 'from_position': before_snapshot.get('position'), 'position': chosen, 'before': before_snapshot, 'after': after_snapshot, 'applied': True, 'method': 'TeleportObject'}
        if parent_id:
            mutation_record['opened_parent_object_id'] = parent_id
            mutation_record['restored_parent_state'] = bool(parent_was_open is False and override.get('restore_parent_state_after_relocate', True))
        if 'rationale' in override:
            mutation_record['rationale'] = dict(override.get('rationale') or {})
        self.scene_mutations.append(mutation_record)
        self.init_mutations.append(dict(mutation_record))

    def _apply_floor_relocate_object(self, index: int, override: Mapping[str, Any], *, object_role: str, object_id: str, parent_id: Optional[str], parent_was_open: Optional[bool]) -> None:
        anchor_role = override.get('to_floor_anchor_role')
        if not isinstance(anchor_role, str) or not anchor_role:
            raise ProcTHOROnlineEnvError('Pose override {} must include to_floor_anchor_role.'.format(index))
        anchor_id = self.role_object_ids.get(anchor_role)
        if anchor_id is None:
            raise ProcTHOROnlineEnvError("Pose override {} references unknown floor anchor role '{}'.".format(index, anchor_role))
        candidates = override.get('floor_position_candidates')
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ProcTHOROnlineEnvError('Pose override {} must include floor_position_candidates.'.format(index))
        object_obj = self._objects_by_id().get(object_id, {})
        before_snapshot = _object_state_snapshot(object_obj)
        object_rotation = dict(object_obj.get('rotation') or {'x': 0.0, 'y': 0.0, 'z': 0.0})
        chosen = None
        chosen_index = None
        chosen_interactable_count = 0
        last_error = ''
        rationale = override.get('rationale')
        scope_room_ids, scope_conflict = _relocation_scope_room_ids(rationale)
        if scope_conflict:
            raise ProcTHOROnlineEnvError('Pose override {} has conflicting relocation room scopes.'.format(index))
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                continue
            if not all((axis in candidate for axis in ('x', 'y', 'z'))):
                continue
            position = {'x': float(candidate['x']), 'y': float(candidate['y']), 'z': float(candidate['z'])}
            candidate_room_id = room_id_for_position(position, self.current_house)
            if scope_room_ids and (candidate_room_id is None or candidate_room_id not in scope_room_ids):
                last_error = 'candidate is outside required room(s): {}'.format(scope_room_ids)
                continue
            event = self.controller.step(action='TeleportObject', objectId=object_id, position=position, rotation=object_rotation, forceAction=True)
            if not bool(event.metadata.get('lastActionSuccess', False)):
                last_error = event.metadata.get('errorMessage', '')
                continue
            self._step(action='Pass')
            interactable_event = self.controller.step(action='GetInteractablePoses', objectId=object_id)
            if not bool(interactable_event.metadata.get('lastActionSuccess', True)):
                last_error = interactable_event.metadata.get('errorMessage', '')
                continue
            interactable_poses = interactable_event.metadata.get('actionReturn') or []
            if not interactable_poses:
                last_error = 'GetInteractablePoses returned no poses'
                continue
            actual_position = self._objects_by_id().get(object_id, {}).get('position') or position
            actual_room_id = room_id_for_position(actual_position, self.current_house)
            if scope_room_ids and (actual_room_id is None or actual_room_id not in scope_room_ids):
                last_error = 'relocated object is outside required room(s): {}'.format(scope_room_ids)
                continue
            chosen = position
            chosen_index = candidate_index
            chosen_interactable_count = len(interactable_poses)
            break
        if chosen is None:
            raise ProcTHOROnlineEnvError('Pose override {}: no interactable floor coordinate near {} for relocating {}. Last error: {}'.format(index, anchor_id, object_id, last_error))
        self._restore_relocate_parent_state(parent_id, parent_was_open, override)
        after_snapshot = _object_state_snapshot(self._objects_by_id().get(object_id, {}))
        mutation_record = {'source': 'init.object_pose_overrides', 'override_index': index, 'type': 'relocate_object', 'object_role': object_role, 'object_id': object_id, 'mode': override.get('mode'), 'placement_kind': 'floor_near_anchor', 'to_floor_anchor_role': anchor_role, 'to_floor_anchor_object_id': anchor_id, 'from_position': before_snapshot.get('position'), 'position': chosen, 'floor_candidate_index': chosen_index, 'interactable_pose_count': chosen_interactable_count, 'room_id': room_id_for_position(chosen, self.current_house), 'before': before_snapshot, 'after': after_snapshot, 'applied': True, 'method': 'TeleportObject'}
        if parent_id:
            mutation_record['opened_parent_object_id'] = parent_id
            mutation_record['restored_parent_state'] = bool(parent_was_open is False and override.get('restore_parent_state_after_relocate', True))
        if 'rationale' in override:
            mutation_record['rationale'] = dict(override.get('rationale') or {})
        self.scene_mutations.append(mutation_record)
        self.init_mutations.append(dict(mutation_record))

    def _apply_clear_receptacle_clutter(self, index: int, override: Mapping[str, Any]) -> None:
        receptacle_role = override.get('receptacle_role')
        if not isinstance(receptacle_role, str) or not receptacle_role:
            raise ProcTHOROnlineEnvError('Pose override {} must include receptacle_role.'.format(index))
        receptacle_id = self.role_object_ids.get(receptacle_role)
        if receptacle_id is None:
            raise ProcTHOROnlineEnvError("Pose override {} references unknown receptacle role '{}'.".format(index, receptacle_role))
        to_receptacle_id = override.get('to_receptacle_object_id')
        if not isinstance(to_receptacle_id, str):
            to_receptacle_id = None
        object_ids = override.get('object_ids')
        if not isinstance(object_ids, list) or not object_ids:
            raise ProcTHOROnlineEnvError('Pose override {} must include object_ids.'.format(index))
        objects = self._objects_by_id()
        if receptacle_id not in objects:
            raise ProcTHOROnlineEnvError('Pose override {} receptacle object not found: {}.'.format(index, receptacle_id))
        for object_id in object_ids:
            if not isinstance(object_id, str) or not object_id:
                continue
            self._move_clutter_object_off_receptacle(index, override, object_id=object_id, receptacle_role=receptacle_role, receptacle_id=receptacle_id, to_receptacle_id=to_receptacle_id)

    def _move_clutter_object_off_receptacle(self, index: int, override: Mapping[str, Any], *, object_id: str, receptacle_role: str, receptacle_id: str, to_receptacle_id: Optional[str]) -> None:
        objects = self._objects_by_id()
        obj = objects.get(object_id)
        if obj is None:
            self._record_pose_init_mutation(index, override, object_id=object_id, receptacle_role=receptacle_role, receptacle_id=receptacle_id, to_receptacle_id=to_receptacle_id, before={}, after={}, applied=False, method='no_op', reason='object_missing')
            return
        before_snapshot = _object_state_snapshot(obj)
        parent_ids = set(obj.get('parentReceptacles') or [])
        if receptacle_id not in parent_ids:
            self._record_pose_init_mutation(index, override, object_id=object_id, receptacle_role=receptacle_role, receptacle_id=receptacle_id, to_receptacle_id=to_receptacle_id, before=before_snapshot, after=before_snapshot, applied=False, method='no_op', reason='object_not_in_receptacle')
            return
        if not bool(obj.get('pickupable', False)):
            raise ProcTHOROnlineEnvError('Pose override {} cannot clear non-pickupable object {} from {}.'.format(index, object_id, receptacle_id))
        event = self.controller.step(action='DisableObject', objectId=object_id)
        if not bool(event.metadata.get('lastActionSuccess', False)):
            raise ProcTHOROnlineEnvError('Pose override {}: could not remove clutter object {} from {}. Error: {}'.format(index, object_id, receptacle_id, event.metadata.get('errorMessage', '')))
        self._step(action='Pass')
        after_snapshot = dict(_object_state_snapshot(self._objects_by_id().get(object_id, {})))
        after_snapshot['disabled'] = True
        record = self._pose_init_mutation_record(index, override, object_id=object_id, receptacle_role=receptacle_role, receptacle_id=receptacle_id, to_receptacle_id=to_receptacle_id, before=before_snapshot, after=after_snapshot, applied=True, method='DisableObject')
        record['from_position'] = before_snapshot.get('position')
        record['removed_from_scene'] = True
        self.scene_mutations.append(dict(record))
        self.init_mutations.append(record)

    def _record_pose_init_mutation(self, index: int, override: Mapping[str, Any], *, object_id: str, receptacle_role: str, receptacle_id: str, to_receptacle_id: Optional[str], before: Mapping[str, Any], after: Mapping[str, Any], applied: bool, method: str, reason: Optional[str]=None, error_message: Optional[str]=None) -> None:
        record = self._pose_init_mutation_record(index, override, object_id=object_id, receptacle_role=receptacle_role, receptacle_id=receptacle_id, to_receptacle_id=to_receptacle_id, before=before, after=after, applied=applied, method=method, reason=reason, error_message=error_message)
        self.init_mutations.append(record)

    def _pose_init_mutation_record(self, index: int, override: Mapping[str, Any], *, object_id: str, receptacle_role: str, receptacle_id: str, to_receptacle_id: Optional[str], before: Mapping[str, Any], after: Mapping[str, Any], applied: bool, method: str, reason: Optional[str]=None, error_message: Optional[str]=None) -> dict:
        record = {'source': 'init.object_pose_overrides', 'override_index': index, 'type': override.get('type'), 'object_id': object_id, 'receptacle_role': receptacle_role, 'receptacle_object_id': receptacle_id, 'requested': {key: value for key, value in dict(override).items() if key not in {'rationale'}}, 'before': dict(before), 'after': dict(after), 'applied': bool(applied), 'method': method}
        if reason:
            record['reason'] = reason
        if error_message:
            record['error_message'] = error_message
        if to_receptacle_id:
            record['to_receptacle_object_id'] = to_receptacle_id
        if 'rationale' in override:
            record['rationale'] = dict(override.get('rationale') or {})
        return record

    def _closed_openable_parent_of(self, object_id: str) -> Optional[str]:
        objects = self._objects_by_id()
        obj = objects.get(object_id)
        if obj is None:
            return None
        for parent_id in obj.get('parentReceptacles') or []:
            parent = objects.get(parent_id, {})
            if parent.get('openable') and parent.get('isOpen') is False:
                return parent_id
        return None

    def _apply_open_state_override(self, index: int, override: Mapping[str, Any]) -> None:
        object_id = self._role_object_id(index, override)
        object_role = override.get('object_role')
        desired_open = _required_bool(override, 'is_open', index)
        obj = self._objects_by_id().get(object_id)
        if obj is None:
            raise ProcTHOROnlineEnvError('Init override {} object not found: {}.'.format(index, object_id))
        if not bool(obj.get('openable', False)):
            raise ProcTHOROnlineEnvError('Init override {} targets non-openable object: {}.'.format(index, object_id))
        before = _object_state_snapshot(obj)
        if obj.get('isOpen') is desired_open:
            self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=before, applied=False, method='no_op', reason='already_desired_state')
            return
        action = 'OpenObject' if desired_open else 'CloseObject'
        try:
            self._step(action=action, objectId=object_id, forceAction=True)
        except Exception as exc:
            self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=before, applied=False, method=action, error_message=str(exc))
            raise
        after = _object_state_snapshot(self._objects_by_id().get(object_id, {}))
        self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=after, applied=True, method=action)

    def _apply_toggle_state_override(self, index: int, override: Mapping[str, Any]) -> None:
        object_id = self._role_object_id(index, override)
        object_role = override.get('object_role')
        desired_toggled = _required_bool(override, 'is_toggled', index)
        obj = self._objects_by_id().get(object_id)
        if obj is None:
            raise ProcTHOROnlineEnvError('Init override {} object not found: {}.'.format(index, object_id))
        if not bool(obj.get('toggleable', False)):
            raise ProcTHOROnlineEnvError('Init override {} targets non-toggleable object: {}.'.format(index, object_id))
        before = _object_state_snapshot(obj)
        if obj.get('isToggled') is desired_toggled:
            self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=before, applied=False, method='no_op', reason='already_desired_state')
            return
        action = 'ToggleObjectOn' if desired_toggled else 'ToggleObjectOff'
        try:
            self._step(action=action, objectId=object_id, forceAction=True)
        except Exception as exc:
            self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=before, applied=False, method=action, error_message=str(exc))
            raise
        after = _object_state_snapshot(self._objects_by_id().get(object_id, {}))
        self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=after, applied=True, method=action)

    def _apply_dirty_state_override(self, index: int, override: Mapping[str, Any]) -> None:
        object_id = self._role_object_id(index, override)
        object_role = override.get('object_role')
        desired_dirty = _required_bool(override, 'is_dirty', index)
        obj = self._objects_by_id().get(object_id)
        if obj is None:
            raise ProcTHOROnlineEnvError('Init override {} object not found: {}.'.format(index, object_id))
        before = _object_state_snapshot(obj)
        if not bool(obj.get('dirtyable', False)):
            sticky_setter = getattr(self, '_set_sticky_object_state', None)
            if callable(sticky_setter):
                sticky_setter(object_id, isDirty=desired_dirty)
                after = _object_state_snapshot(self._objects_by_id().get(object_id, {}))
                self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=after, applied=True, method='sticky_state', reason='object_not_dirtyable')
                import logging
                logging.warning('Init override %d: object %s is not marked dirtyable; using sticky isDirty=%s fallback', index, object_id, desired_dirty)
                return
            raise ProcTHOROnlineEnvError('Init override {} targets non-dirtyable object: {}.'.format(index, object_id))
        if obj.get('isDirty') is desired_dirty:
            self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=before, applied=False, method='no_op', reason='already_desired_state')
            return
        primary_action = 'DirtyObject' if desired_dirty else 'CleanObject'
        event = self.controller.step(action=primary_action, objectId=object_id, forceAction=True)
        if event.metadata.get('lastActionSuccess', False):
            after = _object_state_snapshot(self._objects_by_id().get(object_id, {}))
            self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=after, applied=True, method=primary_action)
            return
        fallback_event = self.controller.step(action='SetObjectStates', SetObjectStates={'objectType': obj.get('objectType'), 'isDirty': desired_dirty})
        if fallback_event.metadata.get('lastActionSuccess', False):
            after = _object_state_snapshot(self._objects_by_id().get(object_id, {}))
            self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=after, applied=True, method='SetObjectStates', error_message=event.metadata.get('errorMessage', ''))
            return
        sticky_setter = getattr(self, '_set_sticky_object_state', None)
        if callable(sticky_setter):
            sticky_setter(object_id, isDirty=desired_dirty)
            after = _object_state_snapshot(self._objects_by_id().get(object_id, {}))
            self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=after, applied=True, method='sticky_state', error_message=event.metadata.get('errorMessage', ''))
            import logging
            logging.warning('Init override %d: %s on %s failed: %s; using sticky isDirty=%s fallback', index, primary_action, object_id, event.metadata.get('errorMessage', '')[:100], desired_dirty)
            return
        import logging
        self._record_init_mutation(index, override, object_role=object_role, object_id=object_id, before=before, after=before, applied=False, method='failed_continued', error_message=event.metadata.get('errorMessage', ''))
        logging.warning('Init override %d: %s on %s failed: %s (continuing anyway)', index, primary_action, object_id, event.metadata.get('errorMessage', '')[:100])

    def _teleport_agent(self, agent_pose: Mapping[str, Any]) -> None:
        position = dict(agent_pose.get('position') or {})
        rotation = dict(agent_pose.get('rotation') or {})
        if not {'x', 'y', 'z'}.issubset(position):
            raise ProcTHOROnlineEnvError('Agent pose position must contain x, y, z.')
        rotation = {'x': float(rotation.get('x', 0.0)), 'y': float(rotation.get('y', 0.0)), 'z': float(rotation.get('z', 0.0))}
        payload = {'action': 'TeleportFull', 'position': {'x': float(position['x']), 'y': float(position['y']), 'z': float(position['z'])}, 'rotation': rotation, 'horizon': float(agent_pose.get('horizon', 0.0)), 'standing': bool(agent_pose.get('standing', True))}
        if self.manipulathor_arm:
            payload['forceAction'] = True
        self._step(**payload)

    def _role_object_id(self, index: int, override: Mapping[str, Any]) -> str:
        role_name = override.get('object_role')
        if not isinstance(role_name, str) or not role_name:
            raise ProcTHOROnlineEnvError('Init override {} must include object_role.'.format(index))
        object_id = self.role_object_ids.get(role_name)
        if object_id is None:
            raise ProcTHOROnlineEnvError("Init override {} references unknown role '{}'.".format(index, role_name))
        return object_id

    def _objects_by_id(self) -> Dict[str, Mapping[str, Any]]:
        return {obj['objectId']: obj for obj in self._metadata_with_sticky_object_states().get('objects', []) if isinstance(obj.get('objectId'), str)}

    def _record_init_mutation(self, index: int, override: Mapping[str, Any], *, object_role: Any, object_id: str, before: Mapping[str, Any], after: Mapping[str, Any], applied: bool, method: str, reason: Optional[str]=None, error_message: Optional[str]=None) -> None:
        record = {'source': 'init.object_state_overrides', 'override_index': index, 'type': override.get('type'), 'object_role': object_role, 'object_id': object_id, 'requested': {key: value for key, value in dict(override).items() if key not in {'rationale'}}, 'before': dict(before), 'after': dict(after), 'applied': bool(applied), 'method': method}
        if reason:
            record['reason'] = reason
        if error_message:
            record['error_message'] = error_message
        if 'rationale' in override:
            record['rationale'] = dict(override.get('rationale') or {})
        self.init_mutations.append(record)

def _required_bool(mapping: Mapping[str, Any], field: str, override_index: int) -> bool:
    value = mapping.get(field)
    if not isinstance(value, bool):
        raise ProcTHOROnlineEnvError("Init override {} field '{}' must be a bool.".format(override_index, field))
    return value

def _relocation_scope_room_ids(rationale: Any) -> tuple[list[str], bool]:
    if not isinstance(rationale, Mapping):
        return ([], False)
    scopes: list[set[str]] = []

    def add_scope(value: Any) -> None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return
        scope = {item.strip() for item in value if isinstance(item, str) and item.strip()}
        if scope:
            scopes.append(scope)
    add_scope(rationale.get('scope_room_ids'))
    candidates = rationale.get('candidates')
    if isinstance(candidates, Sequence) and (not isinstance(candidates, (str, bytes))):
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                add_scope(candidate.get('scope_room_ids'))
    if not scopes:
        return ([], False)
    allowed = set(scopes[0])
    for scope in scopes[1:]:
        allowed.intersection_update(scope)
    return (sorted(allowed), not bool(allowed))

def _object_state_snapshot(obj: Mapping[str, Any]) -> dict:
    return {key: obj.get(key) for key in ('objectId', 'objectType', 'isOpen', 'openness', 'isToggled', 'isDirty', 'parentReceptacles', 'position') if key in obj}
__all__ = ['ProcTHOROnlineEnv', 'ProcTHOROnlineEnvError']
