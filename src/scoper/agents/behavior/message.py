from __future__ import annotations
from typing import Any, Dict, List, Literal, TypedDict, Union

class TextBlock(TypedDict):
    type: Literal['text']
    text: str

class ImageBlock(TypedDict):
    type: Literal['image']
    rgb: Any
ContentBlock = Union[TextBlock, ImageBlock]

class Message(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: List[ContentBlock]

def text_message(role: str, text: str) -> Dict[str, Any]:
    return {'role': role, 'content': [{'type': 'text', 'text': str(text)}]}
__all__ = ['ContentBlock', 'ImageBlock', 'Message', 'TextBlock', 'text_message']
