from __future__ import annotations
import json
import os
import re
from importlib.util import find_spec
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
DEFAULT_PROCTHOR_TAXONOMY = Path(__file__).resolve().parents[2] / 'procthor/data/object_taxonomy.json'
UNKNOWN_OBJECT_TYPE = 'Unknown'
_CATEGORY_EXACT: dict[str, str] = {'apple': 'Apple', 'bowl': 'Bowl', 'breakfast_table': 'DiningTable', 'coffee_table': 'CoffeeTable', 'countertop': 'CounterTop', 'electric_switch': 'LightSwitch', 'floor_lamp': 'FloorLamp', 'fridge': 'Fridge', 'furniture_sink': 'SinkBasin', 'microwave': 'Microwave', 'mug': 'Mug', 'pedestal_sink': 'SinkBasin', 'plate': 'Plate', 'public_trash_can': 'GarbageCan', 'shelf': 'Shelf', 'sofa': 'Sofa', 'spray_bottle': 'SprayBottle', 'table_lamp': 'DeskLamp', 'top_cabinet': 'Cabinet', 'bottom_cabinet': 'Cabinet', 'bottom_cabinet_no_top': 'Cabinet'}
_SYNSET_EXACT: dict[str, str] = {'apple.n.01': 'Apple', 'bowl.n.01': 'Bowl', 'bread.n.01': 'Bread', 'cabinet.n.01': 'Cabinet', 'cabinet.n.03': 'Cabinet', 'coffee_table.n.01': 'CoffeeTable', 'countertop.n.01': 'CounterTop', 'electric_refrigerator.n.01': 'Fridge', 'egg.n.02': 'Egg', 'lettuce.n.03': 'Lettuce', 'microwave.n.02': 'Microwave', 'mug.n.04': 'Mug', 'plate.n.04': 'Plate', 'potato.n.01': 'Potato', 'sink.n.01': 'SinkBasin', 'shelf.n.01': 'Shelf', 'spray_bottle.n.01': 'SprayBottle', 'tomato.n.01': 'Tomato'}

def normalize_behavior_object_type(*, category: str, synset: str, abilities: list[str] | tuple[str, ...] | set[str]=()) -> str:
    category_l = (category or '').lower()
    synset_l = (synset or '').lower()
    text = '{} {}'.format(category_l, synset_l)
    ability_set = set(abilities or [])
    from scoper.benchmark.behavior.scene.concrete_categories import CONCRETE_CATEGORY_TYPES
    if category_l in CONCRETE_CATEGORY_TYPES:
        return CONCRETE_CATEGORY_TYPES[category_l]
    if category_l in _CATEGORY_EXACT:
        return _CATEGORY_EXACT[category_l]
    if synset_l in _SYNSET_EXACT:
        return _SYNSET_EXACT[synset_l]
    if 'faucet' in text:
        return 'Faucet'
    if 'fridge' in text or 'refrigerator' in text or 'freezer' in text:
        return 'Fridge'
    if 'microwave' in text:
        return 'Microwave'
    if 'oven' in text:
        return 'Oven'
    if 'burner' in text or 'stove' in text:
        return 'StoveBurner'
    if 'sink' in text:
        return 'SinkBasin'
    if 'switch' in text:
        return 'LightSwitch'
    if 'lamp' in text or 'downlight' in text or 'wall_mounted_light' in text:
        return 'FloorLamp'
    if 'cabinet' in text or 'locker' in text or 'wardrobe' in text:
        return 'Cabinet'
    if 'drawer' in text:
        return 'Drawer'
    if 'shelf' in text or 'bookcase' in text:
        return 'Shelf'
    if 'counter' in text or category_l in {'bar', 'checkout_counter', 'commercial_kitchen_table'}:
        return 'CounterTop'
    if 'coffee_table' in text:
        return 'CoffeeTable'
    if 'side_table' in text or 'console_table' in text or 'nightstand' in text:
        return 'SideTable'
    if 'table' in text or 'desk' in text:
        return 'DiningTable'
    if 'trash' in text or 'garbage' in text:
        return 'GarbageCan'
    if 'sofa' in text:
        return 'Sofa'
    if 'ottoman' in text or 'footstool' in text:
        return 'Ottoman'
    if 'armchair' in text or 'chair' in text:
        return 'ArmChair'
    if 'tv_stand' in text or 'television_stand' in text or 'tv stand' in text:
        return 'TVStand'
    if 'knife' in text:
        return 'Knife'
    if 'spoon' in text:
        return 'Spoon'
    if 'fork' in text:
        return 'Fork'
    if 'spatula' in text:
        return 'Spatula'
    if 'ladle' in text:
        return 'Ladle'
    words = set(re.findall('[a-z]+', text))
    if words & {'cup', 'cups', 'teacup', 'teacups'}:
        return 'Cup'
    if 'mug' in text:
        return 'Mug'
    if 'plate' in words or 'plates' in words:
        return 'Plate'
    if 'bowl' in text:
        return 'Bowl'
    if 'pan' in words or 'skillet' in words:
        return 'Pan'
    if words & {'pot', 'pots', 'saucepan', 'saucepot', 'stockpot'}:
        return 'Pot'
    if 'soap' in text and 'dish' not in text:
        return 'SoapBottle'
    if 'spray' in text or 'atomizer' in text:
        return 'SprayBottle'
    if 'book' in text:
        return 'Book'
    if 'pencil' in text:
        return 'Pencil'
    if words & {'key', 'keys', 'keychain', 'keychains'}:
        return 'KeyChain'
    if 'tomato' in text:
        return 'Tomato'
    if 'potato' in text:
        return 'Potato'
    if 'lettuce' in text:
        return 'Lettuce'
    if 'bread' in text:
        return 'Bread'
    if words & {'egg', 'eggs'}:
        return 'Egg'
    if words & {'apple', 'apples'}:
        return 'Apple'
    if 'heatSource' in ability_set:
        return 'StoveBurner'
    return UNKNOWN_OBJECT_TYPE

@lru_cache(maxsize=1)
def load_behavior_task_taxonomy(path: str | Path | None=None) -> dict[str, Any]:
    configured = path or os.environ.get('EB_BEHAVIOR_PROCTHOR_TAXONOMY')
    taxonomy_path = Path(configured) if configured else DEFAULT_PROCTHOR_TAXONOMY
    if not configured and (not taxonomy_path.is_file()):
        spec = find_spec('embodiedbench')
        roots = () if spec is None else spec.submodule_search_locations or ()
        for root in roots:
            candidate = Path(root) / 'envs/eb_procthor/data/object_taxonomy.json'
            if candidate.is_file():
                taxonomy_path = candidate
                break
    data = json.loads(taxonomy_path.read_text(encoding='utf-8'))
    groups = {name: set(values) for name, values in (data.get('groups') or {}).items()}
    from scoper.benchmark.behavior.scene.concrete_categories import CONCRETE_GROUP_ADDITIONS
    for name, values in CONCRETE_GROUP_ADDITIONS.items():
        groups.setdefault(name, set()).update(values)
    groups.setdefault('heat_sources', set()).update({'Oven', 'StoveBurner'})
    groups.setdefault('toggleable_objects', set()).update({'Oven'})
    groups.setdefault('openable_containers', set()).update({'Oven'})
    groups.setdefault('place_receptacles', set()).update({'Oven'})
    groups.setdefault('lights', set()).update({'LightSwitch', 'FloorLamp', 'DeskLamp'})
    groups.setdefault('sink_basins', set()).add('SinkBasin')
    groups.setdefault('faucets', set()).add('Faucet')
    return {'version': data.get('version', '0.1') + '+behavior', 'groups': groups, 'excluded_object_types': set(data.get('excluded_object_types') or []), 'aliases': dict(data.get('aliases') or {})}

def normalize_task_object_type(object_type: str) -> str:
    taxonomy = load_behavior_task_taxonomy()
    return taxonomy['aliases'].get(object_type, object_type)

def object_type_in_group(object_type: str, group_name: str) -> bool:
    taxonomy = load_behavior_task_taxonomy()
    normalized = normalize_task_object_type(object_type)
    if normalized in taxonomy['excluded_object_types']:
        return False
    return normalized in taxonomy['groups'].get(group_name, set())

def taxonomy_groups_for_object_type(object_type: str) -> list[str]:
    normalized = normalize_task_object_type(object_type)
    taxonomy = load_behavior_task_taxonomy()
    return sorted((group_name for group_name, values in taxonomy['groups'].items() if normalized in values))
__all__ = ['DEFAULT_PROCTHOR_TAXONOMY', 'UNKNOWN_OBJECT_TYPE', 'load_behavior_task_taxonomy', 'normalize_behavior_object_type', 'normalize_task_object_type', 'object_type_in_group', 'taxonomy_groups_for_object_type']
