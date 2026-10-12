from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

class SymbolicVideoRecorder:

    def __init__(self, video_path: Path, frames_dir: Path, fps: int, save_frames: bool, burn_in_subtitles: bool=True) -> None:
        self.video_path = Path(video_path)
        self.frames_dir = Path(frames_dir)
        self.fps = int(fps)
        self.save_frames = bool(save_frames)
        self.burn_in_subtitles = bool(burn_in_subtitles)
        self.frame_records: List[dict] = []
        self._writer = None
        self._frame_count = 0
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame: Any, *, subtitle: str='', srt_text: Optional[str]=None, repeat: int=1, label: Optional[str]=None, agent_pose: Any=None) -> dict:
        if frame is None:
            return {}
        frame = _as_rgb_array(frame)
        subtitle = str(subtitle or '')
        srt_text = str(srt_text if srt_text is not None else subtitle)
        output_frame = _draw_subtitle(frame, subtitle) if self.burn_in_subtitles and subtitle else frame
        latest_record: dict = {}
        for _ in range(max(1, int(repeat))):
            self._ensure_writer()
            self._writer.append_data(output_frame)
            image_path = None
            if self.save_frames:
                image_path = self.frames_dir / 'frame_{:06d}.png'.format(self._frame_count)
                Image.fromarray(output_frame).save(image_path)
            latest_record = {'frame_index': self._frame_count, 'video_timestamp_seconds': self._frame_count / float(self.fps), 'image_path': str(image_path) if image_path is not None else None, 'label': label, 'subtitle': subtitle, 'srt_text': srt_text, 'agent_pose': agent_pose}
            self.frame_records.append(latest_record)
            self._frame_count += 1
        return latest_record

    def hold(self, frame: Any, *, seconds: float, subtitle: str='', srt_text: Optional[str]=None, label: Optional[str]=None, agent_pose: Any=None) -> None:
        frames = int(round(max(0.0, float(seconds)) * self.fps))
        if frames <= 0:
            return
        self.add_frame(frame, subtitle=subtitle, srt_text=srt_text, repeat=frames, label=label, agent_pose=agent_pose)

    def current_time_seconds(self) -> float:
        return self._frame_count / float(self.fps) if self.fps else 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> dict:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        manifest = {'video': str(self.video_path), 'frames_dir': str(self.frames_dir) if self.save_frames else None, 'fps': self.fps, 'frame_count': self._frame_count, 'duration_seconds': self._frame_count / float(self.fps) if self.fps else 0.0, 'burn_in_subtitles': self.burn_in_subtitles, 'frame_records': self.frame_records}
        manifest_path = self.video_path.with_suffix('.manifest.json')
        srt_path = self.video_path.with_suffix('.srt')
        _write_srt(self.frame_records, srt_path, fps=self.fps)
        _write_json(manifest, manifest_path)
        manifest['manifest_path'] = str(manifest_path)
        manifest['subtitle_path'] = str(srt_path)
        return manifest

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        import imageio.v2 as imageio
        self._writer = imageio.get_writer(str(self.video_path), fps=self.fps, codec='libx264', pixelformat='yuv420p', macro_block_size=1)

def _as_rgb_array(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return np.ascontiguousarray(array)

def _subtitle_font(width: int):
    size = max(14, min(28, width // 36))
    for font_path in ('DejaVuSans.ttf', 'LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def _wrap_subtitle(text: str, draw, font, max_width: int, max_lines: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ''
    for word in words:
        candidate = (current + ' ' + word).strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) >= max_lines and (current or len(words) > sum((len(l.split()) for l in lines))):
        lines[-1] = lines[-1].rstrip('.') + '...'
    return lines[:max_lines]

def _draw_subtitle(frame: np.ndarray, subtitle: str) -> np.ndarray:
    text = ' '.join(str(subtitle or '').split())
    if not text or Image is None:
        return frame
    image = Image.fromarray(frame).convert('RGBA')
    width, height = image.size
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _subtitle_font(width)
    margin = max(8, width // 50)
    max_text_width = max(1, width - 2 * margin)
    lines = _wrap_subtitle(text, draw, font, max_text_width, max_lines=3)
    if not lines:
        return frame
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    line_height = max(line_heights) + 4
    band_height = min(height // 3, line_height * len(lines) + 2 * margin)
    top = height - band_height
    draw.rectangle([0, top, width, height], fill=(0, 0, 0, 170))
    y = top + margin
    for line in lines:
        text_width = draw.textbbox((0, 0), line, font=font)[2]
        x = max(margin, (width - text_width) // 2)
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height
    return np.ascontiguousarray(np.array(Image.alpha_composite(image, overlay).convert('RGB')))

def _write_srt(frame_records: Sequence[Mapping[str, Any]], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments = _subtitle_segments(frame_records, fps=fps)
    lines: List[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append('{} --> {}'.format(_srt_timestamp(segment['start']), _srt_timestamp(segment['end'])))
        lines.append(segment['text'])
        lines.append('')
    output_path.write_text('\n'.join(lines), encoding='utf-8')

def _subtitle_segments(frame_records: Sequence[Mapping[str, Any]], fps: int) -> List[dict]:
    segments: List[dict] = []
    current_text = None
    current_start = 0.0
    last_time = 0.0
    frame_duration = 1.0 / float(fps or 30)
    for record in frame_records:
        text = str(record.get('srt_text') or record.get('subtitle') or '').strip()
        timestamp = float(record.get('video_timestamp_seconds', 0.0) or 0.0)
        if not text:
            if current_text:
                segments.append({'start': current_start, 'end': max(timestamp, current_start + frame_duration), 'text': current_text})
                current_text = None
            last_time = timestamp
            continue
        if text != current_text:
            if current_text:
                segments.append({'start': current_start, 'end': max(timestamp, current_start + frame_duration), 'text': current_text})
            current_text = text
            current_start = timestamp
        last_time = timestamp
    if current_text:
        segments.append({'start': current_start, 'end': max(last_time + frame_duration, current_start + frame_duration), 'text': current_text})
    return segments

def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return '{:02d}:{:02d}:{:02d},{:03d}'.format(h, m, s, ms)

def _write_json(record: Mapping[str, Any], output_path: Path) -> None:
    from scoper.benchmark.behavior.json_utils import json_safe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_safe(dict(record)), indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')

def reencode_frames_video(frames_dir: Path, mp4_path: Path, *, fps: int=30, pattern: str='frame_*.png') -> dict[str, Any]:
    import imageio.v2 as imageio
    frames_dir = Path(frames_dir)
    mp4_path = Path(mp4_path)
    frame_paths = sorted(frames_dir.glob(pattern))
    if not frame_paths:
        raise ValueError('No frame PNGs found in {}'.format(frames_dir))
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(mp4_path), fps=int(fps), codec='libx264', pixelformat='yuv420p', macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    reader = imageio.get_reader(str(mp4_path))
    try:
        frame_count = reader.count_frames()
    finally:
        reader.close()
    return {'path': str(mp4_path), 'frames_dir': str(frames_dir), 'frame_count': len(frame_paths), 'mp4_frame_count': frame_count, 'fps': int(fps)}
__all__ = ['SymbolicVideoRecorder', 'reencode_frames_video']
