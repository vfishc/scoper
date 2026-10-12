from __future__ import annotations
import base64
import io
from typing import Any, Mapping, Tuple
import numpy as np
from PIL import Image

class ImageEncoder:

    def __init__(self, max_resolution: Tuple[int, int]=(1024, 1024), format: str='png', jpeg_quality: int=90) -> None:
        self.max_resolution = (int(max_resolution[0]), int(max_resolution[1]))
        self.format = str(format or 'png').lower()
        self.jpeg_quality = int(jpeg_quality)

    def to_base64(self, rgb: np.ndarray) -> Tuple[str, str]:
        encoded, mime = self.encode(rgb)
        return (base64.b64encode(encoded).decode('ascii'), mime)

    def encode(self, rgb: np.ndarray) -> Tuple[bytes, str]:
        image = self.to_pil(rgb)
        buffer = io.BytesIO()
        if self.format == 'jpeg':
            image.save(buffer, format='JPEG', quality=self.jpeg_quality)
            mime = 'image/jpeg'
        else:
            image.save(buffer, format='PNG')
            mime = 'image/png'
        return (buffer.getvalue(), mime)

    def encode_block(self, block: Mapping[str, Any]) -> Tuple[bytes, str]:
        encoded = block.get('_encoded_bytes')
        mime = block.get('_mime_type')
        if isinstance(encoded, bytes) and mime in {'image/png', 'image/jpeg'}:
            return (encoded, mime)
        return self.encode(block['rgb'])

    def block_to_base64(self, block: Mapping[str, Any]) -> Tuple[str, str]:
        encoded, mime = self.encode_block(block)
        return (base64.b64encode(encoded).decode('ascii'), mime)

    def block_to_pil(self, block: Mapping[str, Any]) -> Image.Image:
        encoded, _mime = self.encode_block(block)
        with Image.open(io.BytesIO(encoded)) as image:
            return image.convert('RGB').copy()

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
