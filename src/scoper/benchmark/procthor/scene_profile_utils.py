from __future__ import annotations
import re
from typing import Any, Mapping, Optional
_ROOM_OBJECT_ID_RE = re.compile('^(?:room\\|)?(\\d+)$')

def profile_object_room_id(obj: Mapping[str, Any]) -> Optional[str]:
    room_id = obj.get('room_id') or obj.get('roomId')
    if isinstance(room_id, str) and room_id:
        return room_id
    object_id = obj.get('object_id') or obj.get('objectId')
    if not isinstance(object_id, str) or not object_id:
        return None
    return room_id_from_object_id(object_id)

def room_id_from_object_id(object_id: str) -> Optional[str]:
    base = object_id.split('___', 1)[0]
    parts = base.split('|')
    if len(parts) < 3:
        return None
    for token in parts[1:]:
        match = _ROOM_OBJECT_ID_RE.match(token)
        if match:
            return 'room|{}'.format(match.group(1))
    return None
__all__ = ['profile_object_room_id', 'room_id_from_object_id']
