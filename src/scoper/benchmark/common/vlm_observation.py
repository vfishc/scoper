from collections import deque

class ScanObservationWindow:

    def __init__(self, max_frames: int=8):
        if not 1 <= max_frames <= 8:
            raise ValueError('scan window must contain between one and eight frames')
        self.max_frames = max_frames
        self._frames = deque(maxlen=max_frames)

    def reset(self):
        self._frames.clear()

    def append(self, rgb):
        import numpy as np
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError('scan evidence requires HWC RGB uint8')
        if not any((np.array_equal(frame, previous) for previous in self._frames)):
            self._frames.append(frame.copy())

    def consume(self, current_rgb, fallback_frames):
        import numpy as np
        if not self._frames:
            return list(fallback_frames)
        frames = list(self._frames)
        self.reset()
        frames = [frame for frame in frames if not np.array_equal(frame, current_rgb)]
        return frames[:self.max_frames - 1] + [np.array(current_rgb, copy=True)]
PRIVILEGED_EXPOSURE_FIELDS = frozenset({'visible_catalog', 'target_handles', 'room_handles', 'agent_pose', 'current_room', 'distance', 'states', 'bbox', 'raw_object_ids', 'held_objects', 'visible_issues', 'scheduler_notices', 'goal_progress', 'state_delta', 'structured_memory'})

def validate_standard_vlm_exposure(exposure):
    forbidden = sorted((name for name in PRIVILEGED_EXPOSURE_FIELDS if getattr(exposure, name, False)))
    if forbidden:
        raise ValueError('standard_vlm forbids privileged exposure: ' + ', '.join(forbidden))
