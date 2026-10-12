import copy
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def initialization_lock():
    path = os.environ.get('EB_BEHAVIOR_SCENE_INIT_LOCK')
    if not path:
        yield
        return
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield

class FreshProcessRequired(BaseException):
    pass
SAFE_ACTIONS = frozenset({'navigate_to', 'navigate_to_room', 'scan', 'look_for', 'pick_up', 'put_down', 'drop', 'open', 'close', 'stop', 'answer', 'answer_question'})

def requires_fresh_after(transition):
    if transition.get('tool_name') in SAFE_ACTIONS:
        return False
    raw = transition.get('raw_execution') or {}
    rejected_before_execution = transition.get('success') is False and raw.get('action_executed') is False and (raw.get('status') == 'rejected') and (raw.get('error_type') == 'binding_rejected') and (not raw.get('state_delta'))
    return not rejected_before_execution

def config_signature(config):
    config = copy.deepcopy(config)
    for obj in config.get('objects', []):
        obj.pop('position', None)
        obj.pop('orientation', None)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

def renderer_signature(renderer):
    config = copy.deepcopy(renderer.env_config)
    config.pop('objects', None)
    config.pop('render_aliases', None)
    config.pop('render_alias_metadata', None)
    return config_signature({'configuration': config_signature(config), 'camera_settle_renders': getattr(renderer, 'camera_settle_renders', 0), 'initial_view_mode': os.environ.get('EB_BEHAVIOR_INITIAL_VIEW_MODE', 'rgb_search'), 'render_profile': renderer.render_profile, 'particles': renderer.particle_systems_enabled, 'native_sources': renderer.native_particle_source_visuals_enabled, 'source_streams': renderer.particle_source_stream_visuals_enabled})

def object_configs(config):
    entries = config.get('objects', [])
    result = {entry['name']: copy.deepcopy(entry) for entry in entries}
    if len(result) != len(entries):
        raise ValueError('duplicate_task_object_name')
    return result

def inventory_changes(previous, requested):
    changed = {name for name in previous.keys() & requested.keys() if config_signature({'objects': [previous[name]]}) != config_signature({'objects': [requested[name]]})}
    return (set(previous) - set(requested) | changed, set(requested) - set(previous) | changed)

def add_task_object(scene, config):
    from omnigibson.objects import REGISTERED_OBJECTS
    from omnigibson.utils.python_utils import create_class_from_registry_and_config
    config = copy.deepcopy(config)
    position = config.pop('position', None)
    orientation = config.pop('orientation', None)
    obj = create_class_from_registry_and_config(cls_name=config['type'], cls_registry=REGISTERED_OBJECTS, cfg=config, cls_type_descriptor='object')
    scene.add_object(obj)
    obj.set_position_orientation(position=position, orientation=orientation, frame='scene')

def reconcile_reset_state(clean, current, added):
    if clean.keys() != current.keys():
        raise ValueError('scene_inventory_changed')
    result = copy.deepcopy(clean)
    for scene_id, scene in current.items():
        before = clean[scene_id]['registry']['object_registry']
        after = scene['registry']['object_registry']
        objects = {}
        for name, state in after.items():
            if name in added:
                objects[name] = copy.deepcopy(state)
            else:
                if name not in before:
                    raise ValueError('unexpected_reset_object:' + name)
                objects[name] = copy.deepcopy(before[name])
        result[scene_id]['registry']['object_registry'] = objects
    return result

class EvaluationSceneSession:

    def __init__(self):
        self.owner = None
        self.force_fresh = False
        self.layout = None
        self.last_reset = {}

    def begin_episode(self, layout):
        if self.owner is not None and (self.force_fresh or layout != self.layout):
            raise FreshProcessRequired('unsafe_previous_episode_or_layout_changed')
        self.layout = layout

    def finish_episode(self, result, error=None):
        if result is None and error is not None:
            from scoper.benchmark.common.direct_response import DirectResponseParseError
            record = getattr(error, 'evaluation_failure_record', None)
            if isinstance(error, DirectResponseParseError) and isinstance(record, dict) and getattr(error, '_behavior_scene_reuse_receipt_complete', False):
                receipt = record.get('transitions')
                if isinstance(receipt, (list, tuple)) and record.get('environment_tool_calls') == len(receipt) and (record.get('high_level_steps') == len(receipt)) and all((isinstance(t, dict) and t.get('tool_name') for t in receipt)):
                    self.force_fresh = any((requires_fresh_after(t) for t in receipt))
                    return
        transitions = getattr(result, 'transitions', ()) if result is not None else ()
        self.force_fresh = result is None or any((requires_fresh_after(t) for t in transitions))

    def __enter__(self):
        from scoper.benchmark.behavior.render import ego_renderer
        self.original = ego_renderer.EgoRenderer
        session, original = (self, self.original)
        tokens = ('VisualLiquid', 'VisualWater', 'VisualDirty', 'VisualParticle', 'VisualAction')

        def visual_paths(renderer):
            return {str(prim.GetPath()) for prim in renderer._og.sim.stage.Traverse() if any((token in str(prim.GetPath()) for token in tokens))}

        class SessionRenderer(original):

            def reset(renderer):
                signature = renderer_signature(renderer)
                requested = object_configs(renderer.env_config)
                if session.owner is None:
                    with initialization_lock():
                        super().reset()
                    session.signature = signature
                    session.state = copy.deepcopy(renderer._og.sim.dump_state(serialized=False))
                    session.visuals = {obj.name: {'visible': bool(obj.visible), 'scale': obj.scale.clone()} for obj in renderer._env.scene.objects}
                    session.paths = visual_paths(renderer)
                    session.owner = renderer
                    session.configs = requested
                    session.house_objects = set(session.visuals) - set(requested)
                    session.last_reset = {'reused': False, 'signature': signature}
                    return renderer
                if session.force_fresh or signature != session.signature:
                    raise FreshProcessRequired('incompatible_renderer_signature')
                try:
                    owner = session.owner
                    owner._detach_visibility_annotator()
                    renderer._og, renderer._env = (owner._og, owner._env)
                    simulator = renderer._og.sim
                    extras = [obj for obj in renderer._env.scene.objects if obj.name not in session.visuals]
                    if extras:
                        simulator.stop()
                        for obj in extras:
                            renderer._env.scene.remove_object(obj)
                        simulator.play()
                    roots = []
                    with simulator.editing_usd():
                        for path in sorted(visual_paths(renderer) - session.paths, key=lambda p: p.count('/')):
                            if not any((path.startswith(root + '/') for root in roots)):
                                simulator.stage.RemovePrim(path)
                                roots.append(path)
                    if visual_paths(renderer) - session.paths:
                        raise RuntimeError('visual_cleanup_incomplete')
                    if {obj.name for obj in renderer._env.scene.objects} != set(session.visuals):
                        raise RuntimeError('baseline_object_inventory_changed')
                    simulator.update_handles()
                    removed, added = inventory_changes(session.configs, requested)
                    if set(requested) & session.house_objects:
                        raise RuntimeError('task_object_collides_with_house_inventory')
                    if removed or added:
                        simulator.stop()
                        for name in sorted(removed):
                            renderer._env.scene.remove_object(renderer._object_by_name(name))
                        for name in sorted(added):
                            add_task_object(renderer._env.scene, requested[name])
                        simulator.play()
                    expected = session.house_objects | set(requested)
                    if {obj.name for obj in renderer._env.scene.objects} != expected:
                        raise RuntimeError('reconciled_object_inventory_mismatch')
                    simulator.update_handles()
                    restored = reconcile_reset_state(session.state, simulator.dump_state(serialized=False), added)
                    simulator.load_state(restored, serialized=False)
                    for obj in renderer._env.scene.objects:
                        if obj.name in added:
                            continue
                        saved = session.visuals[obj.name]
                        obj.visible = saved['visible']
                        if not bool((obj.scale == saved['scale']).all()):
                            obj.scale = saved['scale']
                    for config in renderer.env_config.get('objects', []):
                        renderer._object_by_name(config['name']).set_position_orientation(position=config.get('position'), orientation=config.get('orientation'), frame='scene')
                    renderer._hidden_scene_structures = renderer._hide_scene_structures()
                    renderer._align_loaded_objects_to_desired_centers()
                    renderer._init_visual_state_sync()
                    renderer._init_ego_arm_visual()
                    renderer._container_interiors = renderer._capture_container_interiors()
                    for _ in range(4):
                        simulator.render()
                    renderer._camera = renderer._ego_camera()
                    session.state = copy.deepcopy(simulator.dump_state(serialized=False))
                    session.visuals = {obj.name: {'visible': bool(obj.visible), 'scale': obj.scale.clone()} for obj in renderer._env.scene.objects}
                    session.paths = visual_paths(renderer)
                    session.configs = requested
                    session.owner = renderer
                    session.last_reset = {'reused': True, 'signature': signature, 'removed_objects': len(extras) + len(removed), 'added_objects': len(added)}
                    return renderer
                except Exception as exc:
                    session.force_fresh = True
                    raise FreshProcessRequired('reset_failed:' + type(exc).__name__) from exc

            def close(renderer):
                renderer._detach_visibility_annotator()
        ego_renderer.EgoRenderer = SessionRenderer
        return self

    def __exit__(self, *exc):
        from scoper.benchmark.behavior.render import ego_renderer
        ego_renderer.EgoRenderer = self.original
        if self.owner is not None:
            self.original.close(self.owner)
        self.owner = None
