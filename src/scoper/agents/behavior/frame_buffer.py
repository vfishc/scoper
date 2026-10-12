from __future__ import annotations
from collections import deque
from typing import Deque, List
import numpy as np
from scoper.agents.behavior.config import VLMConfigError

class FrameBuffer:

    def __init__(self, mode: str, num_frames: int=1) -> None:
        self.mode = str(mode)
        if self.mode not in {'single_frame', 'multi_frame', 'streaming'}:
            raise VLMConfigError("Unsupported observation mode '{}'.".format(mode))
        self.num_frames = max(1, int(num_frames))
        capacity = self.num_frames if self.mode in {'multi_frame', 'streaming'} else 1
        self._frames: Deque[np.ndarray] = deque(maxlen=capacity)

    def reset(self) -> None:
        self._frames.clear()

    def update(self, rgb_array: np.ndarray) -> None:
        frame = np.asarray(rgb_array)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise VLMConfigError('head_rgb must be an HxWx3 or HxWx4 array.')
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        self._frames.append(np.ascontiguousarray(frame))

    def get_frames(self) -> List[np.ndarray]:
        return [np.array(frame, copy=True) for frame in self._frames]
__all__ = ['FrameBuffer']
