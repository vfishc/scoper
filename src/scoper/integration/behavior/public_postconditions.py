from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.common import PublicActionState, PublicPropertyValue, PublicRelationValue
from scoper.benchmark.behavior.goal_checker import _is_inside, _is_ontop

def capture_public_action_state(env: Any) -> Mapping[str, Any]:
    core = getattr(env, 'core', None)
    tracker = getattr(env, '_catalog_tracker', None)
    if core is None or tracker is None:
        return PublicActionState(True, (), (), (), (), ()).to_mapping()
    visible_names = set(env._visible_object_names_at_pose(core.ego_state.pose))
    aliases = {name: tracker.alias_for(name, str(getattr(core.objects.get(name), 'category', 'object'))) for name in visible_names if core.objects.get(name) is not None}
    held_name = getattr(core.ego_state, 'held_object_name', None)
    held_refs = ()
    if isinstance(held_name, str) and held_name:
        held_object = core.objects.get(held_name)
        held_refs = (tracker.alias_for(held_name, str(getattr(held_object, 'category', 'object'))),)
    properties = []
    for name, public_ref in aliases.items():
        states = getattr(core.objects[name], 'states', {}) or {}
        if not isinstance(states, Mapping):
            continue
        for state_name in ('open', 'toggled_on', 'sliced', 'dirty', 'heated', 'cooled', 'filled', 'saturated'):
            value = states.get(state_name)
            if isinstance(value, bool):
                properties.append(PublicPropertyValue(public_ref, state_name, value))
    relations = []
    for subject_name, subject_ref in aliases.items():
        subject = core.objects[subject_name]
        for target_name, target_ref in aliases.items():
            if subject_name == target_name:
                continue
            target = core.objects[target_name]
            if _is_inside(subject, target):
                relations.append(PublicRelationValue(subject_ref, 'inside', target_ref))
            elif _is_ontop(subject, target):
                relations.append(PublicRelationValue(subject_ref, 'on_top', target_ref))
    observed_refs = tuple(sorted(aliases.values(), key=lambda value: value.encode('utf-8')))
    state = PublicActionState(True, tuple(sorted(held_refs, key=lambda value: value.encode('utf-8'))), observed_refs, observed_refs, tuple(sorted(properties, key=lambda item: (item.public_ref.encode('utf-8'), item.property_name.encode('utf-8')))), tuple(sorted(set(relations), key=lambda item: (item.subject_public_ref.encode('utf-8'), item.relation.encode('utf-8'), item.object_public_ref.encode('utf-8')))))
    return state.to_mapping()

def build_public_postcondition_evidence(pre_state: Mapping[str, Any] | None, post_state: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if pre_state is None or post_state is None:
        return None
    return {'schema_version': 'method-public-postcondition-evidence.v1', 'pre': dict(pre_state), 'post': dict(post_state)}
__all__ = ['build_public_postcondition_evidence', 'capture_public_action_state']
