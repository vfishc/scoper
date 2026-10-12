from collections import OrderedDict
import hashlib

class BindingObservationLedger:

    def __init__(self, max_pending_frames=64):
        self.max_pending_frames = max_pending_frames
        self.pending = OrderedDict()
        self.delivered_ids = set()

    @staticmethod
    def frame_key(rgb):
        import numpy as np
        frame = np.asarray(rgb)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError('Binding RGB provenance requires HWC uint8')
        return (frame.shape, hashlib.sha256(frame.tobytes()).digest())

    def register(self, rgb, visible_ids):
        key = self.frame_key(rgb)
        visible_ids = frozenset((str(x) for x in visible_ids))
        if key in self.pending:
            visible_ids &= self.pending.pop(key)
        self.pending[key] = visible_ids
        while len(self.pending) > self.max_pending_frames:
            self.pending.popitem(last=False)

    def deliver(self, frames):
        keys = [self.frame_key(frame) for frame in frames]
        if any((key not in self.pending for key in keys)):
            raise ValueError('Selected model RGB has no registered Binding provenance')
        for key in keys:
            self.delivered_ids.update(self.pending[key])

def enable_binding_delivery(env, task):
    from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding
    env._binding_delivery_enforced = False
    env._binding_observation_ledger = None
    tracker = getattr(env, '_agent_catalog_tracker', None)
    if tracker is not None and hasattr(tracker, 'binding_delivered_ids'):
        del tracker.binding_delivered_ids
    roles = (task.get('task_binding_spec') or {}).get('roles', {})
    strict = any((role.get('source_policy') in {'observation', 'task_generic_witness'} and role.get('activation') != 'concrete_reference' for role in roles.values()))
    if task_uses_evidence_binding(task) and strict:
        env._binding_delivery_enforced = True
        env._binding_observation_ledger = BindingObservationLedger()
        if tracker is not None:
            tracker.binding_delivered_ids = env._binding_observation_ledger.delivered_ids

def register_binding_frame(env, rgb, visible_ids):
    ledger = getattr(env, '_binding_observation_ledger', None)
    if getattr(env, '_binding_delivery_enforced', False) and ledger is not None:
        ledger.register(rgb, visible_ids)

def deliver_binding_frames(env, frames, current_observation):
    if not getattr(env, '_binding_delivery_enforced', False):
        return
    if 'visible_object_ids' in current_observation:
        visible = current_observation['visible_object_ids']
    else:
        metadata = env._metadata()
        visible = [obj['objectId'] for obj in metadata.get('objects', []) if obj.get('visible')]
    register_binding_frame(env, current_observation['head_rgb'], visible)
    env._binding_observation_ledger.deliver(frames)
