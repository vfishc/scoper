from __future__ import annotations
from dataclasses import dataclass
from typing import Any
OPEN_STATE_NAME = 'open'
OPEN_ABILITY_NAME = 'openable'
TOGGLE_STATE_NAME = 'toggled_on'
TOGGLE_ABILITY_NAME = 'toggleable'
DIRTY_STATE_NAME = 'dirty'
SLICED_STATE_NAME = 'sliced'
SATURATED_STATE_NAME = 'saturated'
COOKED_STATE_NAME = 'cooked'
COOLED_STATE_NAME = 'cooled'
CONTAINS_STATE_NAME = 'contains'
FILLED_STATE_NAME = 'filled'
SOURCE_SUBSTANCE_STATE_NAME = 'source_substance'
PICK_ABILITY_NAMES = ('pickupable', 'pickable')
PLACE_ON_SURFACE_ABILITY_NAMES = ('placeable_surface', 'support_surface', 'receptacle', 'surface')
PLACE_INSIDE_CONTAINER_ABILITY_NAMES = ('placeable_container', 'container', 'inside')
WIPE_TOOL_ABILITY_NAMES = ('cleaning_tool', 'wipe_tool', 'wiper', 'cloth', 'sponge', 'particleRemover', 'particle_remover')
WIPE_TARGET_ABILITY_NAMES = ('wipeable', 'cleanable', 'stainable', 'dustyable', 'grassyable', 'moldyable', 'rustable', 'tarnishable', 'disinfectable')
CUT_TOOL_ABILITY_NAMES = ('cutting_tool', 'slicer', 'knife')
CUT_TARGET_ABILITY_NAMES = ('cuttable', 'sliceable')
SOAKABLE_ABILITY_NAMES = ('soakable', 'saturatable', 'particleRemover', 'particle_remover', 'cleaning_tool', 'sponge', 'cloth', 'rag')
FLUID_SOURCE_ABILITY_NAMES = ('fluidSource', 'fluid_source', 'particleSource', 'particle_source', 'waterSource', 'water_source')
FILLABLE_ABILITY_NAMES = ('fillable', 'openfillable', 'container', 'placeable_container')
POURABLE_ABILITY_NAMES = ('pourable', 'fillable', 'container')
DEFAULT_SOURCE_SUBSTANCE = 'water'
HEATSOURCE_ABILITY_NAMES = ('heatSource', 'heatsource', 'heat_source')
HEATSOURCE_CATEGORIES = {'stove', 'oven', 'microwave', 'grill', 'toaster_oven', 'range'}
COLDSOURCE_ABILITY_NAMES = ('coldSource', 'coldsource', 'cold_source', 'cool_source')
COLDSOURCE_CATEGORIES = {'fridge', 'freezer', 'refrigerator'}
CONTAINER_CATEGORIES = {'basket', 'bin', 'bottom_cabinet', 'box', 'cabinet', 'drawer', 'freezer', 'fridge', 'shelf', 'sink', 'top_cabinet', 'trash_can'}
NON_PLACE_SURFACE_CATEGORIES = {*CONTAINER_CATEGORIES, 'ceilings', 'door', 'downlight', 'fire_alarm', 'fixed_window', 'floors', 'room_light', 'room_ventilator', 'square_light', 'walls', 'wall_nail', 'window_blind'}

@dataclass
class BooleanStateMutation:
    state_name: str
    before: bool | None
    after: bool
    changed: bool

@dataclass
class SubstanceStateMutation:
    state_name: str
    substance: str
    before: bool
    after: bool
    changed: bool

@dataclass
class SubstanceClearMutation:
    state_names: tuple[str, ...]
    before: dict[str, list[str]]
    after: dict[str, list[str]]
    removed_substances: list[str]
    changed: bool

def supports_open_state(obj: Any) -> bool:
    if getattr(obj, 'renderer_openable', True) is False:
        return False
    states = getattr(obj, 'states', None)
    abilities = tuple(getattr(obj, 'abilities', ()))
    if isinstance(states, dict) and OPEN_STATE_NAME in states:
        return True
    if OPEN_ABILITY_NAME in abilities:
        return True
    open_state = _og_open_state_class()
    return open_state is not None and isinstance(states, dict) and (open_state in states)

def get_open_state(obj: Any) -> bool | None:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        return None
    if OPEN_STATE_NAME in states:
        return bool(states[OPEN_STATE_NAME])
    open_state = _og_open_state_class()
    if open_state is not None and open_state in states:
        return bool(states[open_state].get_value())
    return None

def set_open_state(obj: Any, should_open: bool) -> BooleanStateMutation:
    states = getattr(obj, 'states', None)
    if not supports_open_state(obj) or not isinstance(states, dict):
        raise ValueError(f"Object {getattr(obj, 'name', obj)!r} does not support open state")
    before = get_open_state(obj)
    if OPEN_STATE_NAME in states or OPEN_ABILITY_NAME in tuple(getattr(obj, 'abilities', ())):
        states[OPEN_STATE_NAME] = bool(should_open)
    else:
        open_state = _og_open_state_class()
        assert open_state is not None
        states[open_state].set_value(bool(should_open))
    after = bool(get_open_state(obj))
    return BooleanStateMutation(state_name=OPEN_STATE_NAME, before=before, after=after, changed=before != after)

def supports_toggle_state(obj: Any) -> bool:
    states = getattr(obj, 'states', None)
    abilities = tuple(getattr(obj, 'abilities', ()))
    if isinstance(states, dict) and TOGGLE_STATE_NAME in states:
        return True
    if TOGGLE_ABILITY_NAME in abilities:
        return True
    toggle_state = _og_toggle_state_class()
    return toggle_state is not None and isinstance(states, dict) and (toggle_state in states)

def supports_pick_action(obj: Any) -> bool:
    abilities = tuple(getattr(obj, 'abilities', ()))
    return any((ability in abilities for ability in PICK_ABILITY_NAMES))

def supports_wipe_tool(obj: Any) -> bool:
    abilities = tuple(getattr(obj, 'abilities', ()))
    return any((ability in abilities for ability in WIPE_TOOL_ABILITY_NAMES))

def supports_wipe_target(obj: Any) -> bool:
    if not bool(getattr(obj, 'visible', True)):
        return False
    abilities = tuple(getattr(obj, 'abilities', ()))
    if any((ability in abilities for ability in WIPE_TARGET_ABILITY_NAMES)):
        return True
    return get_dirty_state(obj) is not None

def supports_cut_tool(obj: Any) -> bool:
    abilities = tuple(getattr(obj, 'abilities', ()))
    return any((ability in abilities for ability in CUT_TOOL_ABILITY_NAMES))

def supports_cut_target(obj: Any) -> bool:
    if not bool(getattr(obj, 'visible', True)):
        return False
    abilities = tuple(getattr(obj, 'abilities', ()))
    if any((ability in abilities for ability in CUT_TARGET_ABILITY_NAMES)):
        return True
    return get_sliced_state(obj) is not None

def supports_soakable(obj: Any) -> bool:
    abilities = tuple(getattr(obj, 'abilities', ()))
    return any((ability in abilities for ability in SOAKABLE_ABILITY_NAMES))

def supports_fluid_source(obj: Any) -> bool:
    if not bool(getattr(obj, 'visible', True)):
        return False
    abilities = tuple(getattr(obj, 'abilities', ()))
    if any((ability in abilities for ability in FLUID_SOURCE_ABILITY_NAMES)):
        return True
    if get_source_substance(obj) is not None:
        return True
    return getattr(obj, 'category', '') in {'commercial_kitchen_sink', 'drop_in_sink', 'faucet', 'furniture_sink', 'multi_station_furniture_sink', 'multi_station_tabletop_sink', 'multi_station_wall_mounted_sink', 'pedestal_sink', 'sink', 'shower', 'showerhead', 'tabletop_sink', 'wall_mounted_sink', 'water_source'}

def supports_fillable_container(obj: Any) -> bool:
    if not bool(getattr(obj, 'visible', True)):
        return False
    abilities = tuple(getattr(obj, 'abilities', ()))
    if any((ability in abilities for ability in FILLABLE_ABILITY_NAMES)):
        return True
    if get_substances(obj, CONTAINS_STATE_NAME) or get_substances(obj, FILLED_STATE_NAME):
        return True
    return getattr(obj, 'category', '') in CONTAINER_CATEGORIES

def supports_pourable_source(obj: Any) -> bool:
    abilities = tuple(getattr(obj, 'abilities', ()))
    if any((ability in abilities for ability in POURABLE_ABILITY_NAMES)):
        return True
    return supports_fillable_container(obj)

def placement_relation_for_target(obj: Any) -> str | None:
    if getattr(obj, 'renderer_openable', True) is False:
        return 'on_top'
    if supports_place_inside_container(obj):
        return 'inside'
    if supports_place_on_surface(obj):
        return 'on_top'
    return None

def supports_place_inside_container(obj: Any) -> bool:
    if not bool(getattr(obj, 'visible', True)):
        return False
    abilities = tuple(getattr(obj, 'abilities', ()))
    if any((ability in abilities for ability in PLACE_INSIDE_CONTAINER_ABILITY_NAMES)):
        return True
    return getattr(obj, 'category', '') in CONTAINER_CATEGORIES

def supports_place_on_surface(obj: Any) -> bool:
    if not bool(getattr(obj, 'visible', True)):
        return False
    abilities = tuple(getattr(obj, 'abilities', ()))
    category = getattr(obj, 'category', '')
    extent = tuple(getattr(obj, 'extent', (0.0, 0.0, 0.0)))
    if category in NON_PLACE_SURFACE_CATEGORIES or len(extent) != 3:
        return False
    if supports_pick_action(obj):
        return False
    if any((ability in abilities for ability in PLACE_ON_SURFACE_ABILITY_NAMES)):
        return True
    if any((ability in abilities for ability in HEATSOURCE_ABILITY_NAMES)):
        return True
    return extent[0] >= 0.1 and extent[1] >= 0.1 and (extent[2] > 0.0)

def get_toggle_state(obj: Any) -> bool | None:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        return None
    if TOGGLE_STATE_NAME in states:
        return bool(states[TOGGLE_STATE_NAME])
    toggle_state = _og_toggle_state_class()
    if toggle_state is not None and toggle_state in states:
        return bool(states[toggle_state].get_value())
    return None

def get_dirty_state(obj: Any) -> bool | None:
    state_class = _native_boolean_state_class(obj, DIRTY_STATE_NAME, _og_dirty_state_class)
    return _get_boolean_state(obj, DIRTY_STATE_NAME, state_class)

def set_dirty_state(obj: Any, should_dirty: bool) -> BooleanStateMutation:
    state_class = _native_boolean_state_class(obj, DIRTY_STATE_NAME, _og_dirty_state_class)
    return _set_boolean_state(obj, DIRTY_STATE_NAME, bool(should_dirty), state_class)

def get_sliced_state(obj: Any) -> bool | None:
    state_class = _native_boolean_state_class(obj, SLICED_STATE_NAME, _og_sliced_state_class)
    return _get_boolean_state(obj, SLICED_STATE_NAME, state_class)

def set_sliced_state(obj: Any, should_sliced: bool) -> BooleanStateMutation:
    state_class = _native_boolean_state_class(obj, SLICED_STATE_NAME, _og_sliced_state_class)
    return _set_boolean_state(obj, SLICED_STATE_NAME, bool(should_sliced), state_class)

def get_cooked_state(obj: Any) -> bool | None:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        return None
    if COOKED_STATE_NAME in states:
        return bool(states[COOKED_STATE_NAME])
    return None

def set_cooked_state(obj: Any, should_cook: bool) -> BooleanStateMutation:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        raise ValueError(f"Object {getattr(obj, 'name', obj)!r} does not expose mutable states")
    before = get_cooked_state(obj)
    states[COOKED_STATE_NAME] = bool(should_cook)
    after = bool(get_cooked_state(obj))
    return BooleanStateMutation(state_name=COOKED_STATE_NAME, before=before, after=after, changed=before != after)

def get_cooled_state(obj: Any) -> bool | None:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        return None
    if COOLED_STATE_NAME in states:
        return bool(states[COOLED_STATE_NAME])
    return None

def set_cooled_state(obj: Any, should_cool: bool) -> BooleanStateMutation:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        raise ValueError(f"Object {getattr(obj, 'name', obj)!r} does not expose mutable states")
    before = get_cooled_state(obj)
    states[COOLED_STATE_NAME] = bool(should_cool)
    after = bool(get_cooled_state(obj))
    return BooleanStateMutation(state_name=COOLED_STATE_NAME, before=before, after=after, changed=before != after)

def supports_heatsource(obj: Any) -> bool:
    abilities = tuple(getattr(obj, 'abilities', ()))
    if any((ability in abilities for ability in HEATSOURCE_ABILITY_NAMES)):
        return True
    return getattr(obj, 'category', '') in HEATSOURCE_CATEGORIES

def supports_coldsource(obj: Any) -> bool:
    abilities = tuple(getattr(obj, 'abilities', ()))
    if any((ability in abilities for ability in COLDSOURCE_ABILITY_NAMES)):
        return True
    return getattr(obj, 'category', '') in COLDSOURCE_CATEGORIES

def get_source_substance(obj: Any) -> str | None:
    states = getattr(obj, 'states', None)
    if isinstance(states, dict):
        source = states.get(SOURCE_SUBSTANCE_STATE_NAME)
        if isinstance(source, str) and source:
            return source
        if isinstance(source, (list, tuple, set)) and source:
            return str(sorted(source)[0])
        insource = states.get('insource')
        if isinstance(insource, dict):
            for substance, present in sorted(insource.items()):
                if present:
                    return str(substance)
        if isinstance(insource, str) and insource:
            return insource
    abilities = tuple(getattr(obj, 'abilities', ()))
    if 'water_source' in abilities:
        return DEFAULT_SOURCE_SUBSTANCE
    if getattr(obj, 'category', '') in {'faucet', 'sink', 'shower', 'showerhead'}:
        return DEFAULT_SOURCE_SUBSTANCE
    return None

def get_substances(obj: Any, state_name: str) -> list[str]:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict) or state_name not in states:
        return []
    value = states[state_name]
    if isinstance(value, dict):
        return sorted((str(substance) for substance, present in value.items() if bool(present)))
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return sorted((str(substance) for substance in value))
    if isinstance(value, bool):
        return [DEFAULT_SOURCE_SUBSTANCE] if value else []
    return []

def get_contained_substances(obj: Any) -> list[str]:
    return sorted(set(get_substances(obj, CONTAINS_STATE_NAME)) | set(get_substances(obj, FILLED_STATE_NAME)))

def has_substance(obj: Any, state_name: str, substance: str) -> bool:
    return substance in set(get_substances(obj, state_name))

def set_substance_state(obj: Any, state_name: str, substance: str, present: bool) -> SubstanceStateMutation:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        raise ValueError(f"Object {getattr(obj, 'name', obj)!r} does not expose mutable states")
    substance = str(substance)
    before = has_substance(obj, state_name, substance)
    value = states.get(state_name)
    if not isinstance(value, dict):
        value = {existing: True for existing in get_substances(obj, state_name)}
    value[substance] = bool(present)
    states[state_name] = value
    after = has_substance(obj, state_name, substance)
    return SubstanceStateMutation(state_name=state_name, substance=substance, before=before, after=after, changed=before != after)

def add_container_substance(obj: Any, substance: str) -> list[SubstanceStateMutation]:
    return [set_substance_state(obj, CONTAINS_STATE_NAME, substance, True), set_substance_state(obj, FILLED_STATE_NAME, substance, True)]

def set_saturated_substance(obj: Any, substance: str, present: bool=True) -> SubstanceStateMutation:
    return set_substance_state(obj, SATURATED_STATE_NAME, substance, present)

def clear_container_substances(obj: Any, state_names: tuple[str, ...]=(CONTAINS_STATE_NAME, FILLED_STATE_NAME)) -> SubstanceClearMutation:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        raise ValueError(f"Object {getattr(obj, 'name', obj)!r} does not expose mutable states")
    before = {state_name: get_substances(obj, state_name) for state_name in state_names}
    removed = sorted({substance for substances in before.values() for substance in substances})
    for state_name in state_names:
        states[state_name] = {}
    after = {state_name: get_substances(obj, state_name) for state_name in state_names}
    return SubstanceClearMutation(state_names=state_names, before=before, after=after, removed_substances=removed, changed=any((before[state_name] != after[state_name] for state_name in state_names)))

def set_toggle_state(obj: Any, should_toggle_on: bool) -> BooleanStateMutation:
    states = getattr(obj, 'states', None)
    if not supports_toggle_state(obj) or not isinstance(states, dict):
        raise ValueError(f"Object {getattr(obj, 'name', obj)!r} does not support toggle state")
    before = get_toggle_state(obj)
    if TOGGLE_STATE_NAME in states or TOGGLE_ABILITY_NAME in tuple(getattr(obj, 'abilities', ())):
        states[TOGGLE_STATE_NAME] = bool(should_toggle_on)
    else:
        toggle_state = _og_toggle_state_class()
        assert toggle_state is not None
        states[toggle_state].set_value(bool(should_toggle_on))
    after = bool(get_toggle_state(obj))
    return BooleanStateMutation(state_name=TOGGLE_STATE_NAME, before=before, after=after, changed=before != after)

def _native_boolean_state_class(obj: Any, state_name: str, resolver: Any) -> Any | None:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict) or state_name in states or all((isinstance(key, str) for key in states)):
        return None
    return resolver()

def _get_boolean_state(obj: Any, state_name: str, og_state_class: Any | None) -> bool | None:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        return None
    if state_name in states:
        return bool(states[state_name])
    if og_state_class is not None and og_state_class in states:
        return bool(states[og_state_class].get_value())
    return None

def _set_boolean_state(obj: Any, state_name: str, requested_value: bool, og_state_class: Any | None=None) -> BooleanStateMutation:
    states = getattr(obj, 'states', None)
    if not isinstance(states, dict):
        raise ValueError(f"Object {getattr(obj, 'name', obj)!r} does not expose mutable states")
    before = _get_boolean_state(obj, state_name, og_state_class)
    if state_name in states or og_state_class is None or og_state_class not in states:
        states[state_name] = bool(requested_value)
    else:
        states[og_state_class].set_value(bool(requested_value))
    after = bool(_get_boolean_state(obj, state_name, og_state_class))
    return BooleanStateMutation(state_name=state_name, before=before, after=after, changed=before != after)

def _og_open_state_class():
    try:
        from omnigibson import object_states
    except Exception:
        return None
    return getattr(object_states, 'Open', None)

def _og_toggle_state_class():
    try:
        from omnigibson import object_states
    except Exception:
        return None
    return getattr(object_states, 'ToggledOn', None)

def _og_dirty_state_class():
    try:
        from omnigibson import object_states
    except Exception:
        return None
    return getattr(object_states, 'Dirty', None)

def _og_sliced_state_class():
    try:
        from omnigibson import object_states
    except Exception:
        return None
    return getattr(object_states, 'Sliced', None)
