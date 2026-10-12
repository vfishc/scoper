from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import numpy as np
try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None
__all__ = ['to_numpy', 'jsonable', 'rgb_uint8', 'render_rgb', 'capture_rgb', 'set_camera_pose', 'frame_pixel_stats', 'warm_signal_over_baseline', 'write_video', 'write_high_quality_video', 'write_frames_video', 'write_frame_montage']

def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.array(value)

def jsonable(value: Any) -> Any:
    if hasattr(value, 'detach') or hasattr(value, 'cpu') or hasattr(value, 'numpy'):
        value = to_numpy(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, dict)):
        return value
    return value

def rgb_uint8(value: Any) -> np.ndarray:
    rgb = to_numpy(value)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32, copy=False)
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.5 and (float(finite.min()) >= -0.01):
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb

def render_rgb(og: Any, camera: Any, renders: int=1) -> np.ndarray:
    for _ in range(max(1, renders)):
        og.sim.render()
    obs, _ = camera.get_obs()
    return rgb_uint8(obs['rgb'])

def capture_rgb(og: Any, camera: Any, path: Path, renders: int=4, *, stats: bool=False) -> dict[str, Any]:
    rgb = render_rgb(og, camera, renders=renders)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        Image.fromarray(rgb).save(path)
    payload: dict[str, Any] = {'path': str(path), 'shape': list(rgb.shape)}
    if stats:
        payload.update(frame_pixel_stats(rgb))
    return payload

def set_camera_pose(camera: Any, pose: Any) -> None:
    from scoper.benchmark.behavior.execution_core.ego_agent import ego_camera_quat_xyzw
    camera.set_position_orientation(position=[pose.x, pose.y, pose.z], orientation=list(ego_camera_quat_xyzw(pose.yaw, pose.pitch)), frame='scene')

def frame_pixel_stats(rgb: np.ndarray) -> dict[str, Any]:
    cyan = (rgb[:, :, 1] > 120) & (rgb[:, :, 2] > 130) & (rgb[:, :, 0] < 120)
    blue = (rgb[:, :, 2] > 150) & (rgb[:, :, 0] < 120) & (rgb[:, :, 1] < 190)
    return {'min': int(rgb.min()), 'max': int(rgb.max()), 'variance': float(np.var(rgb.astype(np.float32))), 'cyan_pixels': int(cyan.sum()), 'blue_pixels': int(blue.sum()), 'mean_rgb': rgb.reshape(-1, 3).mean(axis=0).tolist(), 'max_rgb': rgb.reshape(-1, 3).max(axis=0).tolist()}

def warm_signal_over_baseline(frame_rgb: np.ndarray, before_rgb: np.ndarray, threshold: int=20) -> dict[str, Any]:
    diff = frame_rgb.astype(np.int32) - before_rgb.astype(np.int32)
    warm_gain = diff[:, :, 0] - diff[:, :, 2]
    mask = warm_gain > threshold
    return {'warm_pixels': int(mask.sum()), 'peak_warm_gain': int(warm_gain.max())}

def write_video(frames: list[np.ndarray], path: Path, fps: int, *, crf: int=18, preset: str='slow', pix_fmt: str='yuv420p') -> dict[str, Any]:
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=10, macro_block_size=1, output_params=['-crf', str(int(crf)), '-preset', str(preset), '-pix_fmt', str(pix_fmt)]) as writer:
        for frame in frames:
            writer.append_data(frame)
    first_path = path.with_suffix('.first.png')
    last_path = path.with_suffix('.last.png')
    if frames and Image is not None:
        Image.fromarray(frames[0]).save(first_path)
        Image.fromarray(frames[-1]).save(last_path)
    return {'path': str(path), 'fps': int(fps), 'frames': len(frames), 'crf': int(crf), 'preset': str(preset), 'pix_fmt': str(pix_fmt), 'first_frame': str(first_path) if frames else None, 'last_frame': str(last_path) if frames else None, 'first_stats': frame_pixel_stats(frames[0]) if frames else None, 'last_stats': frame_pixel_stats(frames[-1]) if frames else None}

def write_high_quality_video(frames: list[np.ndarray], mp4_path: Path, gif_path: Path | None=None, fps: int=30, quality: int=9) -> dict[str, Any]:
    import imageio.v2 as imageio
    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(mp4_path, fps=fps, codec='libx264', quality=quality, macro_block_size=1, ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    if gif_path is not None:
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=min(fps, 12))
    return {'path': str(mp4_path), 'gif': str(gif_path) if gif_path else None, 'fps': int(fps), 'frames': len(frames)}

def write_frames_video(frames: list[np.ndarray], mp4_path: Path, gif_path: Path | None=None, fps: int=30) -> dict[str, Any]:
    return write_high_quality_video(frames, mp4_path, gif_path, fps=fps, quality=8)

def write_frame_montage(frames: list[np.ndarray], out_path: Path, max_tiles: int=12) -> Path:
    if Image is None or ImageDraw is None:
        raise RuntimeError('Pillow is required for write_frame_montage')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError('write_frame_montage requires at least one frame')
    if len(frames) <= max_tiles:
        selected = list(enumerate(frames))
    else:
        idxs = np.linspace(0, len(frames) - 1, max_tiles).round().astype(int)
        selected = [(int(idx), frames[int(idx)]) for idx in idxs]
    tile_w, tile_h = (selected[0][1].shape[1], selected[0][1].shape[0])
    cols = min(4, len(selected))
    rows = int(math.ceil(len(selected) / cols))
    label_h = 18
    montage = Image.new('RGB', (cols * tile_w, rows * (tile_h + label_h)), color=(245, 245, 245))
    draw = ImageDraw.Draw(montage)
    for tile_index, (frame_index, frame) in enumerate(selected):
        row, col = divmod(tile_index, cols)
        x = col * tile_w
        y = row * (tile_h + label_h)
        montage.paste(Image.fromarray(frame), (x, y + label_h))
        draw.text((x + 4, y + 2), f'frame {frame_index}', fill=(0, 0, 0))
    montage.save(out_path)
    return out_path
