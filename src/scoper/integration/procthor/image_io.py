from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Any
import numpy as np
from PIL import Image
from scoper.benchmark.procthor.export.ids import sha256_bytes
from scoper.integration.procthor.contract import MethodContractError

@dataclass(frozen=True)
class EncodedPublicImage:
    encoded: bytes
    mime_type: str
    width: int
    height: int
    media_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {'media_sha256': self.media_sha256, 'mime_type': self.mime_type, 'width': self.width, 'height': self.height}

def encode_public_rgb(value: Any) -> EncodedPublicImage:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3 or (value.shape[-1] != 3):
        raise MethodContractError('required_frame_missing', 'Method RGB must be HWC uint8')
    contiguous = np.ascontiguousarray(value)
    buffer = io.BytesIO()
    Image.fromarray(contiguous, mode='RGB').save(buffer, format='PNG', optimize=False, compress_level=6)
    encoded = buffer.getvalue()
    return EncodedPublicImage(encoded=encoded, mime_type='image/png', width=int(contiguous.shape[1]), height=int(contiguous.shape[0]), media_sha256=sha256_bytes(encoded))
__all__ = ['EncodedPublicImage', 'encode_public_rgb']
