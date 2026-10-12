from __future__ import annotations
import io
from typing import Any, Mapping, Sequence
import numpy as np
from PIL import Image
from scoper.benchmark.behavior.export.types import TrajectoryConfigError
QUALITY_THRESHOLDS = {'min_width': 64, 'min_height': 64, 'flat_dynamic_range_lt': 2.0, 'flat_luminance_std_lt': 1.0, 'black_mean_lte': 1.0, 'black_luminance_std_lt': 2.0, 'white_mean_gte': 254.0, 'white_luminance_std_lt': 2.0, 'low_information_luminance_std_lt': 3.0, 'low_information_gradient_lt': 1.0, 'possible_blur_score_lt': 20.0, 'stream_invalid_fraction_gt': 0.01}
HARD_REQUIRED_FRAME_KINDS = frozenset({'reset', 'decision_input', 'pre_action', 'post_action', 'intervention_before', 'intervention_after', 'intervention_first_visible', 'episode_final'})

def compute_rgb_quality(rgb: Any) -> Mapping[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        array = np.asarray(rgb)
    except Exception as exc:
        raise TrajectoryConfigError('rgb_invalid', f'RGB value cannot be converted to an array: {exc}') from exc
    shape = list(array.shape)
    dtype = str(array.dtype)
    if array.ndim != 3 or array.shape[-1:] != (3,):
        errors.append('invalid_rgb_shape')
    if array.dtype != np.uint8:
        errors.append('invalid_rgb_dtype')
    try:
        finite = bool(np.isfinite(array).all())
    except (TypeError, ValueError):
        finite = False
    if not finite:
        errors.append('non_finite_rgb')
    metrics = {'mean': None, 'dynamic_range': None, 'luminance_std': None, 'gradient': None, 'blur_score': None}
    if array.ndim == 3 and array.shape[2] == 3 and finite:
        numeric = array.astype(np.float64, copy=False)
        height, width = numeric.shape[:2]
        if height < QUALITY_THRESHOLDS['min_height'] or width < QUALITY_THRESHOLDS['min_width']:
            errors.append('rgb_resolution_too_small')
        luminance = 0.2126 * numeric[:, :, 0] + 0.7152 * numeric[:, :, 1] + 0.0722 * numeric[:, :, 2]
        dynamic_range = float(numeric.max() - numeric.min()) if numeric.size else 0.0
        luminance_std = float(np.std(luminance)) if luminance.size else 0.0
        gradient_vertical = float(np.mean(np.abs(np.diff(luminance, axis=0)))) if height > 1 else 0.0
        gradient_horizontal = float(np.mean(np.abs(np.diff(luminance, axis=1)))) if width > 1 else 0.0
        gradient = gradient_vertical + gradient_horizontal
        if height >= 3 and width >= 3:
            laplacian = -4.0 * luminance[1:-1, 1:-1] + luminance[:-2, 1:-1] + luminance[2:, 1:-1] + luminance[1:-1, :-2] + luminance[1:-1, 2:]
            blur_score = float(np.var(laplacian))
        else:
            blur_score = 0.0
        mean = float(np.mean(luminance)) if luminance.size else 0.0
        metrics.update({'mean': mean, 'dynamic_range': dynamic_range, 'luminance_std': luminance_std, 'gradient': gradient, 'blur_score': blur_score})
        if dynamic_range < 2.0 and luminance_std < 1.0:
            errors.append('flat_rgb')
        if mean <= 1.0 and luminance_std < 2.0:
            errors.append('near_black_rgb')
        if mean >= 254.0 and luminance_std < 2.0:
            errors.append('near_white_rgb')
        if luminance_std < 3.0 and gradient < 1.0:
            warnings.append('low_information_rgb')
        if blur_score < 20.0:
            warnings.append('possible_blur')
    return {'valid': not errors, 'shape': shape, 'dtype': dtype, 'metrics': metrics, 'errors': sorted(set(errors)), 'warnings': sorted(set(warnings)), 'thresholds': dict(QUALITY_THRESHOLDS)}

def decode_image_bytes(encoded: bytes) -> tuple[np.ndarray, Mapping[str, Any]]:
    if not isinstance(encoded, bytes) or not encoded:
        raise TrajectoryConfigError('encoded_image_invalid', 'encoded image must be non-empty bytes')
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            source_mode = image.mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            array = np.asarray(image, dtype=np.uint8)
            metadata = {'format': image.format, 'source_mode': source_mode}
    except Exception as exc:
        raise TrajectoryConfigError('encoded_image_invalid', f'cannot decode image bytes: {exc}') from exc
    return (array, metadata)

def summarize_frame_quality(frame_records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    hard_errors: list[str] = []
    warnings: list[str] = []
    stream_count = 0
    invalid_stream_count = 0
    previous_sha: str | None = None
    for record in frame_records:
        quality = record.get('quality') if isinstance(record, Mapping) else None
        if not isinstance(quality, Mapping):
            hard_errors.append('frame_quality_missing')
            continue
        frame_kind = record.get('frame_kind')
        valid = bool(quality.get('valid'))
        if frame_kind == 'stream_sample':
            stream_count += 1
            if not valid:
                invalid_stream_count += 1
        elif frame_kind in HARD_REQUIRED_FRAME_KINDS and (not valid):
            hard_errors.append(f"required_frame_invalid:{record.get('frame_id')}")
        warnings.extend((str(item) for item in quality.get('warnings', []) if isinstance(item, str)))
        sha = record.get('sha256')
        if isinstance(sha, str) and previous_sha == sha:
            warnings.append(f"exact_duplicate:{record.get('frame_id')}")
        previous_sha = sha if isinstance(sha, str) else None
    invalid_fraction = invalid_stream_count / stream_count if stream_count else 0.0
    stream_export_eligible = invalid_fraction <= QUALITY_THRESHOLDS['stream_invalid_fraction_gt']
    return {'quality_valid': not hard_errors, 'errors': sorted(set(hard_errors)), 'warnings': sorted(set(warnings)), 'stream_frame_count': stream_count, 'invalid_stream_frame_count': invalid_stream_count, 'invalid_stream_fraction': invalid_fraction, 'stream_export_eligible': stream_export_eligible, 'thresholds': dict(QUALITY_THRESHOLDS)}
__all__ = ['HARD_REQUIRED_FRAME_KINDS', 'QUALITY_THRESHOLDS', 'compute_rgb_quality', 'decode_image_bytes', 'summarize_frame_quality']
