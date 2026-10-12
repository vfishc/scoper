from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence
from scoper.benchmark.common import BindingCandidate, BindingFocus, BindingRequest, PublicBindingState, TaskBindingSpec, derive_public_phase_role_order, resolve_binding, SHARED_VOCABULARY_VERSION, scene_observation_activates_binding_role
from scoper.benchmark.procthor.spatial_state import DEFAULT_INTERACTION_DISTANCE_M
STATE_WORDS = frozenset({'dirty', 'clean', 'open', 'opened', 'closed', 'running', 'on', 'off', 'toggled', 'heated', 'hot', 'warm', 'cooled', 'chilled', 'cold', 'frozen', 'sliced', 'cut', 'filled', 'full', 'empty', 'lit', 'wet', 'soaked'})
MEMORY_PHRASES = ('where it was', 'where i picked', 'where i got', 'where i found', 'where i saw', 'earlier', 'before', 'previous place', 'original place', 'originally', 'the place i')
FUNCTIONAL_PHRASES = ('should be kept cold', 'kept cold', 'place to wash', 'wash dishes', 'cleaning supplies', 'something to cut', 'something to heat', 'something to chill', 'food that', 'a place to')
COLLECTION_WORDS = frozenset({'all', 'both', 'every'})

@dataclass(frozen=True)
class ResolvedTarget:
    target_ref: str
    resolved_via: str
    object_id: Optional[str] = None
    position: Optional[dict] = None
    room_id: Optional[str] = None
    room_label: Optional[str] = None
    last_seen_step: Optional[int] = None
    last_known_position: Optional[dict] = None
    binding_public_record: Optional[dict] = None
    binding_role: Optional[str] = None
    navigation_class: Optional[str] = None

class NLTargetResolverError(ValueError):

    def __init__(self, message: str, *, binding_public_record: Mapping[str, Any] | None=None) -> None:
        super().__init__(message)
        self.binding_public_record = None if binding_public_record is None else dict(binding_public_record)

@dataclass(frozen=True)
class ObjectBindingConfig:
    allow_role_handles: bool = True
    allow_visible_aliases: bool = True
    allow_memory: bool = True
    allow_description_match: bool = True
    allow_raw_object_ids: bool = True
    allow_task_text_candidates: bool = False
    ambiguous_score_margin: float = 0.05
    benevolent_task_progress: bool = False
    vocabulary_version: str | None = None

def extract_tool_target(arguments: Mapping[str, Any]) -> str:
    raw_target = arguments.get('target') or arguments.get('object_ref') or arguments.get('object') or arguments.get('receptacle')
    if not isinstance(raw_target, str) or not raw_target.strip():
        raise NLTargetResolverError("Tool call requires a non-empty 'target' or 'object_ref' string.")
    return raw_target.strip()

def resolve_navigation_target(target: str, *, role_object_ids: Mapping[str, str], objects_by_id: Mapping[str, Mapping[str, Any]], tracker: Any, agent_position: Mapping[str, Any], rooms: Sequence[Mapping[str, Any]]=(), binding_config: Optional[ObjectBindingConfig]=None, task_spec: Optional[Mapping[str, Any]]=None, ready_target: Optional[Mapping[str, Any]]=None, publicly_processed_roles: Sequence[str]=(), active_public_event_roles: Sequence[str]=(), scene_observation_roles: Sequence[str]=(), local_interaction_context_active: bool=False, benevolent_role_order: Sequence[str]=()) -> ResolvedTarget:
    target = str(target or '').strip()
    if not target:
        raise NLTargetResolverError("Tool call requires a non-empty 'target' string.")
    binding_config = binding_config or ObjectBindingConfig()
    strict_vocabulary = binding_config.vocabulary_version == SHARED_VOCABULARY_VERSION
    task_binding_spec = _task_binding_spec(task_spec)
    if task_binding_spec is not None:
        from scoper.benchmark.common.evaluation_binding_policy import uses_evidence_binding
        if uses_evidence_binding(task_binding_spec):
            room_resolved = _resolve_room_target(target, rooms, require_unique=True)
            if room_resolved is not None:
                return room_resolved
        if binding_config.allow_role_handles and (not strict_vocabulary) and (target in role_object_ids):
            return ResolvedTarget(target_ref=target, resolved_via='privileged_teacher_role', object_id=role_object_ids[target])
        resolved = _resolve_task_binding(target, tool_name='navigate_to', spec=task_binding_spec, role_object_ids=role_object_ids, objects_by_id=objects_by_id, tracker=tracker, ready_target=ready_target, publicly_processed_roles=publicly_processed_roles, active_public_event_roles=active_public_event_roles, scene_observation_roles=scene_observation_roles, vocabulary_version=binding_config.vocabulary_version, local_interaction_context_active=local_interaction_context_active, task_instruction=str((task_spec or {}).get('instruction') or ''), public_phase_role_order=derive_public_phase_role_order(task_spec or {}), task_spec_for_context=task_spec, benevolent_role_order=benevolent_role_order)
        if resolved is not None:
            return resolved
        room_resolved = _resolve_room_target(target, rooms)
        if room_resolved is not None:
            return room_resolved
        raise NLTargetResolverError("navigate_to target '{}' has no public Binding match.".format(target))
    if binding_config.allow_role_handles and (not strict_vocabulary) and (target in role_object_ids):
        return ResolvedTarget(target_ref=target, resolved_via='role', object_id=role_object_ids[target])
    if binding_config.allow_raw_object_ids and (not strict_vocabulary) and (target in objects_by_id):
        return ResolvedTarget(target_ref=target, resolved_via='object_id', object_id=target)
    _reject_non_spatial_target(target)
    if tracker is not None:
        tracker_entry = tracker.resolve(target)
        if binding_config.allow_visible_aliases and tracker_entry is not None:
            object_id = _object_id_for_tracker_entry(tracker, tracker_entry)
            last_known = dict(tracker_entry.get('last_known_position') or {})
            current_obj = objects_by_id.get(object_id) if object_id else None
            if current_obj is not None and bool(current_obj.get('visible', False)):
                return ResolvedTarget(target_ref=target, resolved_via='alias', object_id=object_id)
            if last_known:
                return ResolvedTarget(target_ref=target, resolved_via='last_known_position', object_id=object_id, position=last_known, last_known_position=last_known, last_seen_step=tracker_entry.get('last_seen_step'))
        type_resolved = _resolve_type_target(target, tracker=tracker, agent_position=agent_position) if binding_config.allow_description_match else None
        if type_resolved is not None:
            return type_resolved
    if binding_config.allow_task_text_candidates:
        task_text_resolved = _resolve_task_text_target(target, task_spec=task_spec or {}, objects_by_id=objects_by_id, agent_position=agent_position)
        if task_text_resolved is not None:
            return task_text_resolved
    room_resolved = _resolve_room_target(target, rooms)
    if room_resolved is not None:
        return room_resolved
    raise NLTargetResolverError(_non_spatial_error(target))

def resolve_operation_target(target: str, *, role_object_ids: Mapping[str, str], objects_by_id: Mapping[str, Mapping[str, Any]], tracker: Any, agent_position: Mapping[str, Any], binding_config: Optional[ObjectBindingConfig]=None, task_spec: Optional[Mapping[str, Any]]=None, tool_name: str='operation', ready_target: Optional[Mapping[str, Any]]=None, publicly_processed_roles: Sequence[str]=(), active_public_event_roles: Sequence[str]=(), scene_observation_roles: Sequence[str]=(), local_interaction_context_active: bool=False, bypass_interaction_readiness: bool=False, benevolent_role_order: Sequence[str]=()) -> ResolvedTarget:
    binding_config = binding_config or ObjectBindingConfig()
    strict_vocabulary = binding_config.vocabulary_version == SHARED_VOCABULARY_VERSION
    task_binding_spec = _task_binding_spec(task_spec)
    if task_binding_spec is not None:
        if binding_config.allow_role_handles and (not strict_vocabulary) and (target in role_object_ids):
            return ResolvedTarget(target_ref=target, resolved_via='privileged_teacher_role', object_id=role_object_ids[target])
        resolved = _resolve_task_binding(target, tool_name=tool_name, spec=task_binding_spec, role_object_ids=role_object_ids, objects_by_id=objects_by_id, tracker=tracker, ready_target=ready_target, publicly_processed_roles=publicly_processed_roles, active_public_event_roles=active_public_event_roles, scene_observation_roles=scene_observation_roles, vocabulary_version=binding_config.vocabulary_version, local_interaction_context_active=local_interaction_context_active, bypass_interaction_readiness=bypass_interaction_readiness, task_instruction=str((task_spec or {}).get('instruction') or ''), public_phase_role_order=derive_public_phase_role_order(task_spec or {}), task_spec_for_context=task_spec, benevolent_role_order=benevolent_role_order)
        if resolved is None:
            raise NLTargetResolverError("Tool target '{}' has no public Binding match.".format(target))
        if resolved.object_id is None:
            raise NLTargetResolverError("Tool target '{}' resolved to a position, not an object. Operation tools require an object target.".format(target))
        return resolved
    resolved = resolve_navigation_target(target, role_object_ids=role_object_ids, objects_by_id=objects_by_id, tracker=tracker, agent_position=agent_position, rooms=(), binding_config=binding_config, task_spec=task_spec, ready_target=ready_target, publicly_processed_roles=publicly_processed_roles, active_public_event_roles=active_public_event_roles, scene_observation_roles=scene_observation_roles, local_interaction_context_active=local_interaction_context_active, benevolent_role_order=benevolent_role_order)
    if resolved.object_id is None:
        raise NLTargetResolverError("Tool target '{}' resolved to a position, not an object. Operation tools require an object target.".format(target))
    return resolved

def _task_binding_spec(task_spec: Optional[Mapping[str, Any]]) -> Optional[TaskBindingSpec]:
    if not isinstance(task_spec, Mapping):
        return None
    value = task_spec.get('task_binding_spec')
    if isinstance(value, Mapping):
        return TaskBindingSpec.from_dict(value)
    return None

def _resolve_task_binding(target: str, *, tool_name: str, spec: TaskBindingSpec, role_object_ids: Mapping[str, str], objects_by_id: Mapping[str, Mapping[str, Any]], tracker: Any, ready_target: Optional[Mapping[str, Any]], publicly_processed_roles: Sequence[str], active_public_event_roles: Sequence[str], scene_observation_roles: Sequence[str], local_interaction_context_active: bool, bypass_interaction_readiness: bool=False, task_instruction: str='', public_phase_role_order: Sequence[str]=(), benevolent_role_order: Sequence[str]=(), vocabulary_version: str | None=None, task_spec_for_context: Optional[Mapping[str, Any]]=None) -> Optional[ResolvedTarget]:
    candidates = _binding_candidates(spec, tool_name=tool_name, role_object_ids=role_object_ids, objects_by_id=objects_by_id, tracker=tracker, ready_target=ready_target, active_public_event_roles=active_public_event_roles, scene_observation_roles=scene_observation_roles)
    focus = None
    if isinstance(ready_target, Mapping) and ready_target.get('object_id'):
        focus = BindingFocus(private_id=str(ready_target['object_id']), canonical_public_ref=str(ready_target.get('canonical_public_ref') or ready_target.get('target') or 'object'), role=ready_target.get('binding_role'), interaction_ready=bool(ready_target.get('interaction_ready')))
    from scoper.benchmark.common.evaluation_binding_policy import public_destination_roles
    result = resolve_binding(BindingRequest(reference=target, tool_name=tool_name, spec=spec, candidates=tuple(candidates), task_instruction=task_instruction, public_phase_role_order=tuple(public_phase_role_order), recent_focus=focus, local_interaction_context_active=local_interaction_context_active, bypass_interaction_readiness=bypass_interaction_readiness, publicly_processed_roles=tuple(publicly_processed_roles), benevolent_role_order=tuple(benevolent_role_order), role_handles=tuple(role_object_ids), raw_object_ids=tuple(objects_by_id), vocabulary_version=vocabulary_version, public_destination_roles=public_destination_roles(task_spec_for_context or {}, focus.role if focus else None)))
    if not result.success:
        if result.status in {'ambiguous', 'conflict', 'not_interaction_ready', 'rejected'}:
            raise NLTargetResolverError("Binding {}: {} for target '{}'.".format(result.status, result.reason, target), binding_public_record=result.public_record())
        return None
    tracker_entry = getattr(tracker, 'entries_by_object_id', {}).get(str(result.private_id), {})
    resolved_via = str(result.source or 'task_binding')
    if result.navigation_class == 'stale_object_navigation':
        resolved_via = 'last_known_position'
        position = dict(tracker_entry.get('last_known_position') or {})
    else:
        obj = objects_by_id.get(str(result.private_id), {})
        position = dict(obj.get('position') or {})
    return ResolvedTarget(target_ref=str(result.canonical_public_ref), resolved_via=resolved_via, object_id=str(result.private_id), position=position or None, last_seen_step=tracker_entry.get('last_seen_step'), last_known_position=dict(tracker_entry.get('last_known_position') or {}) or None, binding_public_record=result.public_record(), binding_role=result.role, navigation_class=result.navigation_class)

def _binding_candidates(spec: TaskBindingSpec, *, tool_name: str, role_object_ids: Mapping[str, str], objects_by_id: Mapping[str, Mapping[str, Any]], tracker: Any, ready_target: Optional[Mapping[str, Any]], active_public_event_roles: Sequence[str], scene_observation_roles: Sequence[str]) -> List[BindingCandidate]:
    reverse_roles = {object_id: role_name for role_name, object_id in role_object_ids.items() if role_name in spec.roles}
    tracker_entries = getattr(tracker, 'entries_by_object_id', {}) if tracker is not None else {}
    active_event_roles = set(active_public_event_roles)
    scene_state = PublicBindingState(scene_observation_roles=tuple(scene_observation_roles))
    candidate_ids = set(tracker_entries) | set(reverse_roles)
    from scoper.benchmark.common.evaluation_binding_policy import unobserved_scene_binding_enabled
    allow_scene = unobserved_scene_binding_enabled(spec)
    if allow_scene:
        candidate_ids.update(objects_by_id)
    candidate_ids.intersection_update(objects_by_id)
    candidates: List[BindingCandidate] = []
    from scoper.benchmark.common.evaluation_binding_policy import candidate_role_pairs
    for object_id, role_name in candidate_role_pairs(spec, role_object_ids, candidate_ids):
        obj = objects_by_id.get(object_id, {})
        entry = tracker_entries.get(object_id, {})
        role_spec = spec.roles.get(role_name or '')
        visible = bool(obj.get('visible', False))
        scene_observed = False
        if role_spec is None:
            source = 'observation' if visible else 'memory' if entry else 'scene'
            activated = bool(entry) or allow_scene
            public_terms = ()
        else:
            source = role_spec.source_policy if role_spec.source_policy in {'observation', 'public_event'} else 'task'
            scene_observed = scene_observation_activates_binding_role(scene_state, role_name, tool_name)
            if role_spec.source_policy in {'observation', 'task_generic_witness'}:
                activated = bool(entry) or visible or scene_observed
            elif role_spec.source_policy == 'public_event':
                activated = role_name in active_event_roles
            else:
                activated = True
            public_terms = role_spec.public_terms
        object_type = _object_type_from_metadata(obj, object_id)
        delivered_ids = getattr(tracker, 'binding_delivered_ids', None)
        if delivered_ids is not None:
            scene_observed = False
            if role_spec is None or role_spec.source_policy in {'observation', 'task_generic_witness'}:
                activated = object_id in delivered_ids
        alias = str(entry.get('alias') or (public_terms[0] if public_terms else object_type))
        if allow_scene and role_spec is None:
            from scoper.benchmark.common.public_object_names import public_object_text
            alias = public_object_text(alias.replace('_', ' '))
        aliases = set(public_terms)
        if entry.get('alias'):
            aliases.add(str(entry['alias']))
        ready = False
        distance = obj.get('distance')
        from scoper.benchmark.common.evaluation_binding_policy import uses_evidence_binding, spec_requires_observation
        allow_unobserved = uses_evidence_binding(spec) and (not spec_requires_observation(spec))
        if (visible or allow_unobserved) and distance is not None:
            try:
                ready = float(distance) <= DEFAULT_INTERACTION_DISTANCE_M and _object_supports_tool(obj, tool_name)
            except (TypeError, ValueError):
                pass
        candidates.append(BindingCandidate(private_id=object_id, canonical_public_ref=alias, object_type=object_type, aliases=tuple(sorted(aliases, key=str.encode)), source=source, role=role_name, visible=visible, interaction_ready=ready, activated=activated, scene_observed=scene_observed, last_seen_step=entry.get('last_seen_step')))
    return candidates

def _object_supports_tool(obj: Mapping[str, Any], tool_name: str) -> bool:
    if tool_name in {'navigate_to', 'look_for'}:
        return False
    if tool_name == 'pick_up':
        return bool(obj.get('pickupable', False))
    if tool_name == 'put_down':
        return bool(obj.get('receptacle', False))
    if tool_name in {'open', 'close'}:
        return bool(obj.get('openable', False))
    if tool_name in {'toggle_on', 'toggle_off'}:
        return bool(obj.get('toggleable', False))
    if tool_name == 'slice':
        return bool(obj.get('sliceable', False))
    return True

def _resolve_task_text_target(target: str, *, task_spec: Mapping[str, Any], objects_by_id: Mapping[str, Mapping[str, Any]], agent_position: Mapping[str, Any]) -> Optional[ResolvedTarget]:
    instruction = str(task_spec.get('instruction') or '').lower()
    if not instruction:
        return None
    target_tokens = _tokens(target)
    if not target_tokens:
        return None
    role_candidates = _task_role_text_candidates(target_tokens, task_spec=task_spec, objects_by_id=objects_by_id)
    if role_candidates:
        role_candidates.sort(key=lambda item: (-float(item['score']), _squared_xz_distance(agent_position, item.get('position') or {}), str(item['object_id'])))
        best = role_candidates[0]
        return ResolvedTarget(target_ref=target, resolved_via='task_text', object_id=str(best['object_id']), position=dict(best.get('position') or {}))
    candidates = []
    for object_id, obj in sorted(objects_by_id.items()):
        object_type = _object_type_from_metadata(obj, object_id)
        object_tokens = _tokens(_readable_object_type(object_type))
        if not object_tokens:
            continue
        if not object_tokens.intersection(_tokens(instruction)):
            continue
        score = _description_score(target_tokens, object_tokens)
        if score <= 0:
            continue
        candidates.append({'object_id': object_id, 'score': score, 'position': dict(obj.get('position') or {}), 'object_type': object_type})
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-float(item['score']), _squared_xz_distance(agent_position, item.get('position') or {}), str(item['object_id'])))
    best = candidates[0]
    return ResolvedTarget(target_ref=target, resolved_via='task_text', object_id=str(best['object_id']), position=dict(best.get('position') or {}))

def _task_role_text_candidates(target_tokens: set[str], *, task_spec: Mapping[str, Any], objects_by_id: Mapping[str, Mapping[str, Any]]) -> List[dict]:
    instruction_tokens = _tokens(str(task_spec.get('instruction') or ''))
    roles = task_spec.get('roles') or {}
    if not isinstance(roles, Mapping):
        return []
    candidates = []
    for role in roles.values():
        if not isinstance(role, Mapping):
            continue
        object_id = role.get('object_id')
        if not isinstance(object_id, str) or object_id not in objects_by_id:
            continue
        object_type = str(role.get('object_type') or '')
        object_tokens = _tokens(_readable_object_type(object_type))
        if not object_tokens:
            continue
        if not object_tokens.intersection(instruction_tokens):
            continue
        score = _description_score(target_tokens, object_tokens)
        if score <= 0:
            continue
        candidates.append({'object_id': object_id, 'score': score, 'position': dict(objects_by_id[object_id].get('position') or {}), 'object_type': object_type})
    return candidates

def _resolve_type_target(target: str, *, tracker: Any, agent_position: Mapping[str, Any]) -> Optional[ResolvedTarget]:
    lowered = target.lower().strip()
    nearest = False
    type_name = target.strip()
    match = re.fullmatch('(?:the\\s+)?(?:nearest|closest)\\s+([A-Za-z][A-Za-z0-9_]*)', lowered)
    if match:
        nearest = True
        type_name = match.group(1)
    else:
        match = re.fullmatch('(?:a|an|the)\\s+([A-Za-z][A-Za-z0-9_]*)', lowered)
        if match:
            nearest = True
            type_name = match.group(1)
    candidates = tracker.entries_for_type(type_name)
    if not candidates:
        return None
    if len(candidates) == 1 and (not nearest):
        object_id = _object_id_for_tracker_entry(tracker, candidates[0])
        return ResolvedTarget(target_ref=candidates[0]['alias'], resolved_via='unique_type', object_id=object_id)
    best = min(candidates, key=lambda entry: _squared_xz_distance(agent_position, entry.get('last_known_position') or {}))
    object_id = _object_id_for_tracker_entry(tracker, best)
    return ResolvedTarget(target_ref=best['alias'], resolved_via='nearest_type', object_id=object_id, position=dict(best.get('last_known_position') or {}), last_known_position=dict(best.get('last_known_position') or {}), last_seen_step=best.get('last_seen_step'))

def _resolve_room_target(target: str, rooms: Sequence[Mapping[str, Any]], *, require_unique: bool=False) -> Optional[ResolvedTarget]:
    normalized = _normalize_room_label(target)
    if require_unique:
        from scoper.benchmark.common.evaluation_binding_policy import match_public_rooms
        public_matches = match_public_rooms(target, rooms)
        if len(public_matches) > 1:
            raise NLTargetResolverError('Room reference is ambiguous; use a specific public room label.')
        rooms = public_matches
        if rooms:
            normalized = _normalize_room_label(str(rooms[0].get('label') or rooms[0].get('room_type')))
    for room in rooms:
        room_id = str(room.get('room_id') or '')
        room_type = str(room.get('room_type') or '')
        labels = {_normalize_room_label(room_id), _normalize_room_label(room_type), _normalize_room_label(str(room.get('label') or ''))}
        if normalized not in labels:
            continue
        pose = room.get('representative_reachable_pose') or {}
        position = pose.get('position') if isinstance(pose, Mapping) else None
        if not isinstance(position, Mapping):
            raise NLTargetResolverError("Room target '{}' has no representative reachable pose.".format(target))
        return ResolvedTarget(target_ref=target, resolved_via='room_label', position=dict(position), room_id=room_id or None, room_label=room_type or room_id or target)
    return None

def _reject_non_spatial_target(target: str) -> None:
    lowered = target.lower()
    tokens = set(re.findall('[a-z]+', lowered))
    if tokens.intersection(STATE_WORDS):
        raise NLTargetResolverError(_non_spatial_error(target))
    if tokens.intersection(COLLECTION_WORDS):
        raise NLTargetResolverError(_non_spatial_error(target))
    if any((phrase in lowered for phrase in MEMORY_PHRASES)):
        raise NLTargetResolverError(_non_spatial_error(target))
    if any((phrase in lowered for phrase in FUNCTIONAL_PHRASES)):
        raise NLTargetResolverError(_non_spatial_error(target))

def _non_spatial_error(target: str) -> str:
    return "navigate_to target '{}' is not a spatial-layer reference. navigation only resolves objectIds, role handles, object aliases, type-unique matches, '<Type>' / 'the nearest <Type>' phrases, or room labels. Use object_catalog (in prompt_context) and your visual observations to resolve state/memory/category references yourself.".format(target)

def _object_id_for_tracker_entry(tracker: Any, tracker_entry: Mapping[str, Any]) -> Optional[str]:
    alias = tracker_entry.get('alias')
    if not alias:
        return None
    return getattr(tracker, 'alias_to_object_id', {}).get(alias)

def _normalize_room_label(value: str) -> str:
    value = value.strip().lower()
    if value.startswith('the '):
        value = value[4:]
    return re.sub('[^a-z0-9_]+', '', value)

def _tokens(value: str) -> set[str]:
    return {token for token in re.findall('[a-z0-9]+', str(value or '').lower()) if token not in {'a', 'an', 'the', 'to', 'of', 'on', 'in', 'into', 'onto'}}

def _description_score(reference_tokens: set[str], object_tokens: set[str]) -> float:
    if not reference_tokens or not object_tokens:
        return 0.0
    overlap = reference_tokens.intersection(object_tokens)
    if not overlap:
        return 0.0
    if reference_tokens == object_tokens:
        return 1.0
    return len(overlap) / max(len(reference_tokens), len(object_tokens))

def _object_type_from_metadata(obj: Mapping[str, Any], object_id: str) -> str:
    object_type = obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    return str(object_id).split('|', 1)[0]

def _readable_object_type(object_type: str) -> str:
    return re.sub('(?<!^)(?=[A-Z])', ' ', str(object_type or '')).replace('_', ' ').lower()

def _squared_xz_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return (float(a.get('x', 0.0)) - float(b.get('x', 0.0))) ** 2 + (float(a.get('z', 0.0)) - float(b.get('z', 0.0))) ** 2
__all__ = ['NLTargetResolverError', 'ObjectBindingConfig', 'ResolvedTarget', 'extract_tool_target', 'resolve_navigation_target', 'resolve_operation_target']
