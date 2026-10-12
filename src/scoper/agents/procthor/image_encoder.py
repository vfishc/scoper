from __future__ import annotations
import base64
import io
from typing import Tuple
import numpy as np
from PIL import Image

class ImageEncoder:

    def __init__(self, max_resolution: Tuple[int, int]=(1024, 1024), format: str='png', jpeg_quality: int=90) -> None:
        self.max_resolution = (int(max_resolution[0]), int(max_resolution[1]))
        self.format = str(format or 'png').lower()
        self.jpeg_quality = int(jpeg_quality)

    def to_base64(self, rgb: np.ndarray) -> Tuple[str, str]:
        encoded, mime = self.encode_bytes(rgb)
        return (base64.b64encode(encoded).decode('ascii'), mime)

    def encode_bytes(self, rgb: np.ndarray) -> Tuple[bytes, str]:
        image = self.to_pil(rgb)
        buffer = io.BytesIO()
        if self.format == 'jpeg':
            image.save(buffer, format='JPEG', quality=self.jpeg_quality, subsampling=0, optimize=False, progressive=False)
            mime = 'image/jpeg'
        else:
            image.save(buffer, format='PNG', compress_level=6, optimize=False)
            mime = 'image/png'
        return (buffer.getvalue(), mime)

    def to_pil(self, rgb: np.ndarray) -> Image.Image:
        array = np.asarray(rgb)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[:, :, :3]
        image = Image.fromarray(np.ascontiguousarray(array)).convert('RGB')
        resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')
        image.thumbnail(self.max_resolution, resample)
        return image
__all__ = ['ImageEncoder']
