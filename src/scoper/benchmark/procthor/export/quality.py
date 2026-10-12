from __future__ import annotations
import io
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from PIL import Image, UnidentifiedImageError
CRITICAL_FRAME_KINDS = frozenset({'reset', 'decision_input', 'pre_action', 'post_action', 'intervention_before', 'intervention_after', 'intervention_first_visible', 'verifier_evidence', 'model_input', 'episode_final'})
QUALITY_THRESHOLDS = {'minimum_height': 64, 'minimum_width': 64, 'blank_dynamic_range_lt': 2.0, 'blank_luminance_std_lt': 1.0, 'underexposed_mean_lte': 1.0, 'overexposed_mean_gte': 254.0, 'exposure_luminance_std_lt': 2.0, 'low_information_luminance_std_lt': 3.0, 'low_information_gradient_lt': 1.0, 'possible_blur_score_lt': 20.0, 'stream_invalid_ratio_gt': 0.01}
QUALITY_FLOAT_DECIMALS = 8

def analyze_rgb(rgb: Any) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(rgb, np.ndarray):
        return _invalid_result(shape=None, dtype=type(rgb).__name__, errors=['rgb_array_required'])
    shape = list(rgb.shape)
    dtype = str(rgb.dtype)
    if rgb.ndim != 3 or rgb.shape[2:] != (3,):
        errors.append('rgb_shape_invalid')
    if rgb.dtype != np.uint8:
        errors.append('rgb_dtype_invalid')
    if np.issubdtype(rgb.dtype, np.number) and (not np.isfinite(rgb).all()):
        errors.append('rgb_non_finite')
    if errors:
        return _invalid_result(shape=shape, dtype=dtype, errors=errors)
    height, width, _ = rgb.shape
    if height < QUALITY_THRESHOLDS['minimum_height'] or width < QUALITY_THRESHOLDS['minimum_width']:
        errors.append('rgb_too_small')
    if height == 0 or width == 0:
        return _invalid_result(shape=shape, dtype=dtype, errors=errors)
    values = rgb.astype(np.float64, copy=False)
    luminance = 0.2126 * values[:, :, 0] + 0.7152 * values[:, :, 1] + 0.0722 * values[:, :, 2]
    dynamic_range = float(int(rgb.max()) - int(rgb.min()))
    luminance_mean = _stable_metric(np.mean(luminance))
    luminance_std = _stable_metric(np.std(luminance))
    vertical_gradient = float(np.mean(np.abs(np.diff(luminance, axis=0)))) if height > 1 else 0.0
    horizontal_gradient = float(np.mean(np.abs(np.diff(luminance, axis=1)))) if width > 1 else 0.0
    gradient = _stable_metric(vertical_gradient + horizontal_gradient)
    if height >= 3 and width >= 3:
        laplacian = -4.0 * luminance[1:-1, 1:-1] + luminance[:-2, 1:-1] + luminance[2:, 1:-1] + luminance[1:-1, :-2] + luminance[1:-1, 2:]
        blur_score = _stable_metric(np.var(laplacian))
    else:
        blur_score = 0.0
    if dynamic_range < 2.0 and luminance_std < 1.0:
        errors.append('rgb_blank_dynamic_range')
    if luminance_mean <= 1.0 and luminance_std < 2.0:
        errors.append('rgb_underexposed')
    if luminance_mean >= 254.0 and luminance_std < 2.0:
        errors.append('rgb_overexposed')
    if luminance_std < 3.0 and gradient < 1.0:
        warnings.append('low_information_rgb')
    if blur_score < 20.0:
        warnings.append('possible_blur')
    return {'valid': not errors, 'shape': shape, 'dtype': dtype, 'dynamic_range': dynamic_range, 'luminance_mean': luminance_mean, 'luminance_std': luminance_std, 'gradient': gradient, 'blur_score': blur_score, 'errors': sorted(set(errors)), 'warnings': sorted(set(warnings)), 'thresholds': dict(QUALITY_THRESHOLDS)}

def analyze_encoded_image(encoded: bytes) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(encoded, bytes) or not encoded:
        return (_invalid_result(shape=None, dtype=None, errors=['image_bytes_invalid']), {'format': None, 'mode': None})
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            image_format = image.format
            mode = image.mode
            rgb = np.asarray(image)
    except (OSError, ValueError, UnidentifiedImageError):
        return (_invalid_result(shape=None, dtype=None, errors=['image_decode_failed']), {'format': None, 'mode': None})
    quality = analyze_rgb(rgb)
    return (quality, {'format': image_format, 'mode': mode})

def analyze_image_file(path: Path | str) -> tuple[dict[str, Any], Mapping[str, Any]]:
    try:
        encoded = Path(path).read_bytes()
    except OSError:
        return (_invalid_result(shape=None, dtype=None, errors=['image_read_failed']), {'format': None, 'mode': None})
    return analyze_encoded_image(encoded)

def is_critical_frame_kind(frame_kind: str) -> bool:
    return frame_kind in CRITICAL_FRAME_KINDS

def _invalid_result(*, shape: list[int] | None, dtype: str | None, errors: list[str]) -> dict[str, Any]:
    return {'valid': False, 'shape': shape, 'dtype': dtype, 'dynamic_range': None, 'luminance_mean': None, 'luminance_std': None, 'gradient': None, 'blur_score': None, 'errors': sorted(set(errors)), 'warnings': [], 'thresholds': dict(QUALITY_THRESHOLDS)}

def _stable_metric(value: Any) -> float:
    return float(round(float(value), QUALITY_FLOAT_DECIMALS))
__all__ = ['CRITICAL_FRAME_KINDS', 'QUALITY_THRESHOLDS', 'QUALITY_FLOAT_DECIMALS', 'analyze_encoded_image', 'analyze_image_file', 'analyze_rgb', 'is_critical_frame_kind']
