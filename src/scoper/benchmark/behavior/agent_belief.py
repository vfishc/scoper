from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
BELIEF_STATUSES = frozenset({'unknown', 'known', 'stale', 'missing_at_last_seen'})

@dataclass
class AgentBeliefRecord:
    object_id: str
    status: str = 'unknown'
    last_seen_center: tuple[float, float, float] | None = None
    last_seen_step: int | None = None
    last_seen_source: str | None = None
    last_seen_pose: dict[str, float] | None = None
    missing_step: int | None = None
    missing_pose: dict[str, float] | None = None
    missing_source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {'object_id': self.object_id, 'status': self.status, 'last_seen_center': list(self.last_seen_center) if self.last_seen_center is not None else None, 'last_seen_step': self.last_seen_step, 'last_seen_source': self.last_seen_source, 'last_seen_pose': dict(self.last_seen_pose) if self.last_seen_pose is not None else None, 'missing_step': self.missing_step, 'missing_pose': dict(self.missing_pose) if self.missing_pose is not None else None, 'missing_source': self.missing_source}

class AgentBeliefTracker:

    def __init__(self) -> None:
        self._records: dict[str, AgentBeliefRecord] = {}

    def reset(self, objects: Mapping[str, Any] | Sequence[Any]) -> None:
        self._records = {}
        if isinstance(objects, Mapping):
            names = [str(name) for name in objects.keys()]
        else:
            names = [str(getattr(obj, 'name', '')) for obj in objects if getattr(obj, 'name', None)]
        for name in sorted(names):
            self._records[name] = AgentBeliefRecord(object_id=name)

    def ensure(self, object_id: str) -> AgentBeliefRecord:
        object_id = str(object_id)
        if object_id not in self._records:
            self._records[object_id] = AgentBeliefRecord(object_id=object_id)
        return self._records[object_id]

    def get(self, object_id: Any) -> AgentBeliefRecord | None:
        if object_id is None:
            return None
        return self._records.get(str(object_id))

    def observe_visible(self, visible_object_ids: Sequence[str], objects: Mapping[str, Any], *, step: int, source: str, pose: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        pose_dict = _pose_dict(pose)
        for object_id in sorted({str(item) for item in visible_object_ids}):
            obj = objects.get(object_id)
            if obj is None:
                continue
            center = _center_tuple(getattr(obj, 'center', None))
            if center is None:
                continue
            record = self.ensure(object_id)
            before = record.as_dict()
            record.status = 'known'
            record.last_seen_center = center
            record.last_seen_step = int(step)
            record.last_seen_source = str(source)
            record.last_seen_pose = dict(pose_dict) if pose_dict is not None else None
            record.missing_step = None
            record.missing_pose = None
            record.missing_source = None
            after = record.as_dict()
            if before != after:
                updates.append({'object_id': object_id, 'before': before, 'after': after})
        return updates

    def mark_stale(self, object_id: str, *, step: int, source: str) -> AgentBeliefRecord:
        record = self.ensure(object_id)
        if record.status == 'known':
            record.status = 'stale'
        record.missing_step = int(step)
        record.missing_source = str(source)
        return record

    def mark_missing_at_last_seen(self, object_id: str, *, step: int, source: str, pose: Mapping[str, Any] | None) -> AgentBeliefRecord:
        record = self.ensure(object_id)
        record.status = 'missing_at_last_seen'
        record.missing_step = int(step)
        record.missing_source = str(source)
        pose_dict = _pose_dict(pose)
        record.missing_pose = dict(pose_dict) if pose_dict is not None else None
        return record

    def valid_last_seen_center(self, object_id: str) -> tuple[float, float, float] | None:
        record = self.get(object_id)
        if record is None or record.last_seen_center is None:
            return None
        if record.status not in {'known', 'stale'}:
            return None
        return record.last_seen_center

    def snapshot(self, object_ids: Sequence[str] | None=None) -> dict[str, dict[str, Any]]:
        if object_ids is None:
            keys = sorted(self._records.keys())
        else:
            keys = sorted({str(item) for item in object_ids})
        return {key: self.ensure(key).as_dict() for key in keys}

def _center_tuple(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return None

def _pose_dict(value: Mapping[str, Any] | None) -> Optional[dict[str, float]]:
    if not isinstance(value, Mapping):
        return None
    try:
        return {'x': float(value.get('x', 0.0)), 'y': float(value.get('y', 0.0)), 'z': float(value.get('z', 1.55)), 'yaw': float(value.get('yaw', 0.0)), 'pitch': float(value.get('pitch', 0.0))}
    except Exception:
        return None
__all__ = ['AgentBeliefRecord', 'AgentBeliefTracker', 'BELIEF_STATUSES']
