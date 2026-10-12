from __future__ import annotations
from collections.abc import Iterable
DOOR_NOT_LOAD_CATEGORIES = ('door', 'doors', 'sliding_door')

def is_door_like(category: object='', name: object='') -> bool:
    category_l = str(category or '').strip().lower()
    name_l = str(name or '').strip().lower()
    return 'door' in category_l or 'door' in name_l

def non_door_categories(categories: Iterable[object]) -> list[str]:
    return [str(category) for category in categories if not is_door_like(category)]

def door_not_load_categories() -> list[str]:
    return list(DOOR_NOT_LOAD_CATEGORIES)
__all__ = ['DOOR_NOT_LOAD_CATEGORIES', 'door_not_load_categories', 'is_door_like', 'non_door_categories']
