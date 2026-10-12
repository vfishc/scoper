from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.common import BindingCandidate, BindingFocus, BindingRequest, TaskBindingSpec, derive_public_binding_state, derive_public_phase_role_order, derive_benevolent_role_order, reference_forms, resolve_binding, SHARED_VOCABULARY_VERSION, scene_observation_activates_binding_role
from scoper.agents.behavior.config import ObjectBindingConfig
from scoper.benchmark.behavior.json_utils import _json_safe

@dataclass
class ObjectBindingResult:
    raw_reference: str
    selected_object: str | None
    status: str
    confidence: float
    reason: str
    source: str | None
    candidates: list[dict[str, Any]]
    binding_public_record: dict[str, Any] | None = None
    binding_role: str | None = None

    @property
    def success(self) -> bool:
        return self.status == 'success' and self.selected_object is not None

    def as_dict(self) -> dict[str, Any]:
        return {'raw_reference': self.raw_reference, 'selected_object': self.selected_object, 'status': self.status, 'confidence': self.confidence, 'reason': self.reason, 'source': self.source, 'candidates': self.candidates, 'binding': self.binding_public_record}

class ObjectReferenceResolver:

    def __init__(self, config: ObjectBindingConfig, *, log_path: str | Path | None=None) -> None:
        self.config = config
        self.log_path = Path(log_path) if log_path is not None else None
        self._fh = None
        self._memory: dict[str, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.log_path.open('w', encoding='utf-8')

    def reset(self) -> None:
        self._memory.clear()
        self.records.clear()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def observe(self, env: Any) -> None:
        visible_ids = _visible_ids(env)
        for object_id in visible_ids:
            obj = getattr(env, 'core', None).objects.get(object_id) if getattr(env, 'core', None) is not None else None
            if obj is None:
                continue
            alias = getattr(getattr(env, '_catalog_tracker', None), 'alias_for', lambda oid, typ=None: oid)(object_id, getattr(obj, 'category', None))
            self._memory[str(object_id)] = _candidate_record(object_id=str(object_id), alias=str(alias), obj=obj, visible=True, source='visible')
        if self.config.allow_memory:
            self._observe_belief_memory(env, visible_ids)

    def resolve(self, reference: Any, *, env: Any, tool_name: str, high_level_step: int, argument_name: str='target', defer_log: bool=False) -> ObjectBindingResult:
        text = str(reference or '').strip()
        self.observe(env)
        role_map = dict(getattr(env, 'role_map', {}) or {})
        visible_ids = set(_visible_ids(env))
        tracker = getattr(env, '_catalog_tracker', None)

        def finish(result: ObjectBindingResult) -> ObjectBindingResult:
            if not defer_log:
                self.log(result, env=env, tool_name=tool_name, high_level_step=high_level_step, argument_name=argument_name)
            return result
        if not text:
            result = ObjectBindingResult(text, None, 'unresolved', 0.0, 'empty_reference', None, [])
            return finish(result)
        strict_vocabulary = self.config.vocabulary_version == SHARED_VOCABULARY_VERSION
        if self.config.allow_role_handles and (not strict_vocabulary) and (text in role_map):
            selected = str(role_map[text])
            result = ObjectBindingResult(text, selected, 'success', 1.0, 'matched_role_handle', 'role_handle', [_candidate_summary(selected, env=env, visible_ids=visible_ids, source='role_handle')])
            return finish(result)
        shared_unmatched: ObjectBindingResult | None = None
        task_binding_spec = _task_binding_spec(getattr(env, 'current_task_spec', None))
        if task_binding_spec is not None:
            shared_result = _resolve_shared_binding(text, env=env, tool_name=tool_name, spec=task_binding_spec, memory=self._memory, benevolent_task_progress=self.config.benevolent_task_progress, vocabulary_version=self.config.vocabulary_version)
            if shared_result.success:
                result = ObjectBindingResult(text, shared_result.private_id, 'success', 1.0, shared_result.reason, shared_result.source, [], shared_result.public_record(), shared_result.role)
            else:
                result = ObjectBindingResult(text, None, shared_result.status, 0.0, shared_result.reason, shared_result.source, [], shared_result.public_record())
            if shared_result.success or not (shared_result.status == 'unresolved' and shared_result.reason == 'no_public_match'):
                return finish(result)
            shared_unmatched = result
        public_scene_candidates: list[dict[str, Any]] | None = None
        if shared_unmatched is not None:
            public_scene_candidates = [item for item in self._candidate_records(env, visible_ids) if item.get('source') in {'visible', 'memory'}]
            public_scene_ids = {str(item.get('object_id')) for item in public_scene_candidates}
        else:
            public_scene_ids = set()
        if self.config.allow_visible_aliases and tracker is not None:
            resolved = str(tracker.resolve(text))
            if resolved != text and resolved in getattr(env.core, 'objects', {}) and (shared_unmatched is None or resolved in public_scene_ids):
                result = ObjectBindingResult(text, resolved, 'success', 1.0, 'matched_visible_alias', 'visible_alias', [_candidate_summary(resolved, env=env, visible_ids=visible_ids, source='visible_alias')])
                if shared_unmatched is not None:
                    result = _public_scene_binding_result(result, env=env, tool_name=tool_name, visible_ids=visible_ids)
                return finish(result)
        if self.config.allow_raw_object_ids and (not strict_vocabulary) and (text in getattr(env.core, 'objects', {})):
            result = ObjectBindingResult(text, text, 'success', 1.0, 'matched_raw_object_id', 'raw_object_id', [_candidate_summary(text, env=env, visible_ids=visible_ids, source='raw_object_id')])
            return finish(result)
        candidates = public_scene_candidates if public_scene_candidates is not None else self._candidate_records(env, visible_ids)
        multi_object = _multi_object_reference_reason(text, candidates)
        if multi_object is not None:
            result = ObjectBindingResult(text, None, 'ambiguous', 0.0, multi_object, None, candidates[:8])
            if shared_unmatched is not None:
                result = _public_scene_binding_result(result, env=env, tool_name=tool_name, visible_ids=visible_ids)
            return finish(result)
        scored = _score_candidates(text, candidates) if self.config.allow_description_match else []
        if scored:
            best = scored[0]
            second_score = scored[1]['score'] if len(scored) > 1 else -1.0
            ambiguous = second_score >= 0 and best['score'] - second_score <= self.config.ambiguous_score_margin
            if best['score'] <= 0:
                status = 'unresolved'
                selected = None
                reason = 'no_description_match'
            elif ambiguous:
                status = 'ambiguous'
                selected = None
                reason = 'ambiguous_description_match'
            else:
                status = 'success'
                selected = str(best['object_id'])
                reason = 'matched_description'
            result = ObjectBindingResult(text, selected, status, _confidence(best['score']), reason, best.get('source'), [_strip_score(item) for item in scored[:8]])
            if shared_unmatched is not None:
                if not result.success and result.status == 'unresolved':
                    return finish(shared_unmatched)
                result = _public_scene_binding_result(result, env=env, tool_name=tool_name, visible_ids=visible_ids)
            return finish(result)
        if shared_unmatched is not None:
            return finish(shared_unmatched)
        result = ObjectBindingResult(text, None, 'unresolved', 0.0, 'no_candidates', None, candidates[:8])
        return finish(result)

    def _candidate_records(self, env: Any, visible_ids: set[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        tracker = getattr(env, '_catalog_tracker', None)
        for object_id in sorted(visible_ids):
            obj = env.core.objects.get(object_id)
            if obj is None:
                continue
            alias = tracker.alias_for(object_id, getattr(obj, 'category', None)) if tracker is not None else object_id
            records.append(_candidate_record(object_id=object_id, alias=alias, obj=obj, visible=True, source='visible'))
        if self.config.allow_memory:
            self._observe_belief_memory(env, visible_ids)
            for object_id, record in sorted(self._memory.items()):
                if object_id not in visible_ids:
                    remembered = dict(record)
                    remembered['visible'] = False
                    remembered['source'] = remembered.get('source') or 'memory'
                    records.append(remembered)
        if self.config.allow_task_text_candidates:
            known_ids = {str(item.get('object_id')) for item in records}
            records.extend(self._task_text_candidate_records(env, known_ids))
        return records

    def _observe_belief_memory(self, env: Any, visible_ids: Sequence[str]) -> None:
        core = getattr(env, 'core', None)
        tracker = getattr(env, '_catalog_tracker', None)
        belief = getattr(env, '_belief_tracker', None)
        snapshot = getattr(belief, 'snapshot', None)
        if core is None or not callable(snapshot):
            return
        visible = {str(item) for item in visible_ids}
        try:
            records = snapshot()
        except Exception:
            return
        if not isinstance(records, Mapping):
            return
        for object_id, record in sorted(records.items()):
            object_id = str(object_id)
            if object_id in visible:
                continue
            if not isinstance(record, Mapping):
                continue
            if record.get('status') not in {'known', 'stale'}:
                continue
            if record.get('last_seen_center') is None:
                continue
            obj = core.objects.get(object_id)
            if obj is None:
                continue
            alias = tracker.alias_for(object_id, getattr(obj, 'category', None)) if tracker is not None else object_id
            memory = _candidate_record(object_id=object_id, alias=alias, obj=obj, visible=False, source='memory')
            memory['last_seen_source'] = record.get('last_seen_source')
            memory['last_seen_step'] = record.get('last_seen_step')
            self._memory[object_id] = memory

    def _task_text_candidate_records(self, env: Any, known_ids: set[str]) -> list[dict[str, Any]]:
        core = getattr(env, 'core', None)
        task_spec = getattr(env, 'current_task_spec', None)
        if core is None or not isinstance(task_spec, Mapping):
            return []
        instruction = str(task_spec.get('instruction') or '').lower()
        if not instruction:
            return []
        tracker = getattr(env, '_catalog_tracker', None)
        records: list[dict[str, Any]] = []
        for object_id, obj in sorted(getattr(core, 'objects', {}).items()):
            object_id = str(object_id)
            if object_id in known_ids:
                continue
            category = str(getattr(obj, 'category', '') or '')
            if not category:
                continue
            if not _category_mentioned_in_text(category, instruction):
                continue
            alias = tracker.alias_for(object_id, category) if tracker is not None else object_id
            record = _candidate_record(object_id=object_id, alias=alias, obj=obj, visible=False, source='task_text')
            records.append(record)
        return records

    def log(self, result: ObjectBindingResult, *, env: Any, tool_name: str, high_level_step: int, argument_name: str) -> None:
        record = result.as_dict()
        record['binding_target_kind'] = _binding_target_kind(result, env=env)
        record.update({'task_id': (getattr(env, 'current_task_spec', {}) or {}).get('id'), 'tool_name': tool_name, 'argument_name': argument_name, 'high_level_step': high_level_step, 'binding_config': dict(self.config.__dict__)})
        self.records.append(dict(record))
        if self._fh is None:
            return
        self._fh.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True))
        self._fh.write('\n')
        self._fh.flush()

def _visible_ids(env: Any) -> list[str]:
    if getattr(env, 'core', None) is None:
        return []
    try:
        return [str(item) for item in env._visible_object_names_at_pose(env.core.ego_state.pose)]
    except Exception:
        return []

def _task_binding_spec(task_spec: Any) -> TaskBindingSpec | None:
    if not isinstance(task_spec, Mapping):
        return None
    value = task_spec.get('task_binding_spec')
    if not isinstance(value, Mapping):
        return None
    return TaskBindingSpec.from_dict(value)
_RECEPTACLE_ABILITIES = frozenset({'container', 'receptacle', 'support_surface'})

def _binding_target_kind(result: ObjectBindingResult, *, env: Any) -> str:
    task_spec = getattr(env, 'current_task_spec', None)
    role_names: set[str] = set()
    if result.binding_role:
        role_names.add(str(result.binding_role))
    if not role_names:
        spec = _task_binding_spec(task_spec)
        query_forms = reference_forms(result.raw_reference)
        if spec is not None and query_forms:
            for role_name, role_spec in spec.roles.items():
                public_forms: set[str] = set()
                for term in (*role_spec.public_terms, *role_spec.generic_terms):
                    public_forms.update(reference_forms(term))
                if query_forms & public_forms:
                    role_names.add(str(role_name))
    if len(role_names) == 1 and _role_is_receptacle_capable(next(iter(role_names)), env=env):
        return 'receptacle'
    if result.selected_object and _object_is_receptacle_capable(str(result.selected_object), env=env):
        return 'receptacle'
    return 'object' if result.success else 'unknown'

def _role_is_receptacle_capable(role_name: str, *, env: Any) -> bool:
    task_spec = getattr(env, 'current_task_spec', None)
    roles = task_spec.get('roles') if isinstance(task_spec, Mapping) else None
    role = roles.get(role_name) if isinstance(roles, Mapping) else None
    abilities = set(role.get('abilities') or ()) if isinstance(role, Mapping) else set()
    if abilities & _RECEPTACLE_ABILITIES:
        return True
    role_map = getattr(env, 'role_map', {}) or {}
    object_id = role_map.get(role_name)
    if object_id is None and isinstance(role, Mapping):
        object_id = role.get('object_id') or role.get('bddl_inst')
    return bool(object_id and _object_is_receptacle_capable(str(object_id), env=env))

def _object_is_receptacle_capable(object_id: str, *, env: Any) -> bool:
    core = getattr(env, 'core', None)
    obj = getattr(core, 'objects', {}).get(object_id) if core is not None else None
    abilities = set(getattr(obj, 'abilities', ()) or ()) if obj is not None else set()
    return bool(abilities & _RECEPTACLE_ABILITIES)

def _resolve_shared_binding(reference: str, *, env: Any, tool_name: str, spec: TaskBindingSpec, memory: Mapping[str, Mapping[str, Any]], benevolent_task_progress: bool=False, vocabulary_version: str | None=None):
    visible_ids = set(_visible_ids(env))
    role_map = dict(getattr(env, 'role_map', {}) or {})
    core_objects = getattr(getattr(env, 'core', None), 'objects', {}) or {}
    tracker = getattr(env, '_catalog_tracker', None)
    object_to_alias = dict(getattr(tracker, '_object_to_alias', {}) or {})
    reverse_roles = {str(object_id): str(role_name) for role_name, object_id in role_map.items() if role_name in spec.roles}
    public_binding_state = _public_binding_state(env, spec)
    active_event_roles = set(public_binding_state.active_public_event_roles)
    candidate_ids = set(memory) | set(visible_ids) | set(reverse_roles)
    from scoper.benchmark.common.evaluation_binding_policy import unobserved_scene_binding_enabled
    allow_scene = unobserved_scene_binding_enabled(spec)
    if allow_scene:
        candidate_ids.update(core_objects)
    candidates = []
    from scoper.benchmark.common.evaluation_binding_policy import candidate_role_pairs
    for object_id, role_name in candidate_role_pairs(spec, role_map, candidate_ids):
        obj = core_objects.get(object_id)
        if obj is None:
            continue
        role_spec = spec.roles.get(role_name or '')
        visible = object_id in visible_ids
        remembered = memory.get(object_id) or {}
        scene_observed = False
        if role_spec is None:
            source = 'observation' if visible else 'memory' if remembered else 'scene'
            activated = visible or bool(remembered) or allow_scene
            public_terms = ()
        else:
            source = role_spec.source_policy if role_spec.source_policy in {'observation', 'public_event'} else 'task'
            scene_observed = scene_observation_activates_binding_role(public_binding_state, role_name, tool_name)
            if role_spec.source_policy in {'observation', 'task_generic_witness'}:
                activated = visible or bool(remembered) or scene_observed
            elif role_spec.source_policy == 'public_event':
                activated = role_name in active_event_roles
            else:
                activated = True
            public_terms = role_spec.public_terms
        object_type = str(getattr(obj, 'category', '') or 'object')
        if getattr(env, '_binding_delivery_enforced', False):
            scene_observed = False
            if role_spec is None or role_spec.source_policy in {'observation', 'task_generic_witness'}:
                ledger = getattr(env, '_binding_observation_ledger', None)
                activated = ledger is not None and object_id in ledger.delivered_ids
        canonical_ref = str(object_to_alias.get(object_id) or remembered.get('alias') or (public_terms[0] if public_terms else object_type))
        aliases = set(public_terms)
        if allow_scene and role_spec is None:
            from scoper.benchmark.common.public_object_names import public_object_text
            canonical_ref = public_object_text(canonical_ref.replace('_', ' '))
        if remembered.get('alias'):
            aliases.add(str(remembered['alias']))
        if object_to_alias.get(object_id):
            aliases.add(str(object_to_alias[object_id]))
        candidates.append(BindingCandidate(private_id=object_id, canonical_public_ref=canonical_ref, object_type=object_type, aliases=tuple(sorted(aliases, key=str.encode)), source=source, role=role_name, visible=visible, interaction_ready=_interaction_ready(env, obj, tool_name, visible), activated=activated, scene_observed=scene_observed, last_seen_step=remembered.get('last_seen_step')))
    focus = _recent_binding_focus(env, candidates)
    task_spec = getattr(env, 'current_task_spec', None) or {}
    benevolent_role_order = _benevolent_role_order(env, task_spec) if benevolent_task_progress else ()
    from scoper.benchmark.common.evaluation_binding_policy import public_destination_roles, uses_evidence_binding, public_placed_roles
    return resolve_binding(BindingRequest(reference=reference, tool_name=tool_name, spec=spec, candidates=tuple(candidates), task_instruction=str(task_spec.get('instruction') or ''), public_phase_role_order=derive_public_phase_role_order(task_spec), recent_focus=focus, local_interaction_context_active=bool(getattr(env, '_local_interaction_context_active', False)), bypass_interaction_readiness=bool(getattr(env, '_bypass_interaction_readiness_active', False)), publicly_processed_roles=tuple(public_placed_roles(task_spec, getattr(env, 'tool_history', ()), role_map) if uses_evidence_binding(spec) else getattr(env, '_private_processed_binding_roles', ()) or ()), benevolent_role_order=benevolent_role_order, role_handles=tuple(role_map), raw_object_ids=tuple((str(object_id) for object_id in core_objects)), vocabulary_version=vocabulary_version, public_destination_roles=public_destination_roles(task_spec, focus.role if focus else None)))

def _benevolent_role_order(env: Any, task_spec: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        result = env.check_goal(model_stopped=False)
        statuses = tuple((bool(item.get('satisfied')) for item in result.get('predicates', ()) if isinstance(item, Mapping)))
        held_name = getattr(getattr(getattr(env, 'core', None), 'ego_state', None), 'held_object_name', None)
        held_roles = tuple((role for role, object_id in (getattr(env, 'role_map', {}) or {}).items() if held_name is not None and str(object_id) == str(held_name)))
        return derive_benevolent_role_order(task_spec, predicate_satisfied=statuses, held_roles=held_roles)
    except (KeyError, TypeError, ValueError):
        return ()

def _interaction_ready(env: Any, obj: Any, tool_name: str, visible: bool) -> bool:
    if tool_name in {'navigate_to', 'look_for'}:
        return False
    from scoper.benchmark.common.evaluation_binding_policy import uses_evidence_binding, spec_requires_observation
    spec = _task_binding_spec(getattr(env, 'current_task_spec', None))
    allow_unobserved = spec is not None and uses_evidence_binding(spec) and (not spec_requires_observation(spec))
    if not visible and (not allow_unobserved):
        return False
    score = getattr(env, '_physical_affordance_score', lambda *args, **kwargs: 0.0)(tool_name, obj)
    if float(score or 0.0) <= 0.0:
        return False
    core = getattr(env, 'core', None)
    if core is None:
        return False
    pose = getattr(getattr(core, 'ego_state', None), 'pose', None)
    distance = getattr(env, '_object_xy_distance', lambda *args: None)(tuple(getattr(pose, 'xy', (0.0, 0.0))), obj)
    interval = getattr(getattr(core, 'config', None), 'navigation', None)
    interval = getattr(interval, 'interaction_distance', (0.0, 1.5))
    return distance is not None and float(interval[0]) <= float(distance) <= float(interval[1])

def _public_scene_binding_result(result: ObjectBindingResult, *, env: Any, tool_name: str, visible_ids: set[str]) -> ObjectBindingResult:
    selected = result.selected_object
    candidate = next((item for item in result.candidates if selected is not None and str(item.get('object_id')) == str(selected)), None)
    status = result.status
    reason = result.reason
    canonical_ref = None
    navigation_class = None
    source = result.source
    if result.success and candidate is not None:
        canonical_ref = str(candidate.get('alias') or result.raw_reference)
        source = 'public_evidence' if str(selected) in visible_ids else 'memory'
        if tool_name == 'navigate_to':
            navigation_class = 'seen_object_navigation' if str(selected) in visible_ids else 'stale_object_navigation'
        elif tool_name != 'look_for':
            local_context = bool(getattr(env, '_local_interaction_context_active', False))
            obj = getattr(getattr(env, 'core', None), 'objects', {}).get(selected)
            ready = bool(obj is not None and _interaction_ready(env, obj, tool_name, str(selected) in visible_ids))
            if not local_context:
                status = 'not_interaction_ready'
                reason = 'navigation_context_required'
                selected = None
            elif not ready:
                status = 'not_interaction_ready'
                reason = 'target_not_interaction_ready'
                selected = None
    binding_record = {'status': status, 'reason': reason, 'source': source, 'match_kind': 'public_alias' if result.reason == 'matched_visible_alias' else 'lexical', 'canonical_public_ref': canonical_ref, 'navigation_class': navigation_class}
    return ObjectBindingResult(raw_reference=result.raw_reference, selected_object=selected, status=status, confidence=result.confidence, reason=reason, source=source, candidates=result.candidates, binding_public_record=binding_record)

def _recent_binding_focus(env: Any, candidates: Sequence[BindingCandidate]) -> BindingFocus | None:
    record = dict((getattr(env, '_last_resolved_targets', {}) or {}).get('spatial_focus') or {})
    focus_step = int(record.get('high_level_step') or -999999)
    current_step = int(getattr(env, 'high_level_step', 0))
    if focus_step not in {current_step - 1, current_step}:
        return None
    object_id = str(record.get('selected_object') or '')
    matches = [candidate for candidate in candidates if candidate.private_id == object_id]
    if len(matches) != 1:
        return None
    candidate = matches[0]
    return BindingFocus(private_id=candidate.private_id, canonical_public_ref=str(record.get('canonical_public_ref') or candidate.canonical_public_ref), role=candidate.role, interaction_ready=candidate.interaction_ready, high_level_step=record.get('high_level_step'))

def _public_binding_state(env: Any, spec: TaskBindingSpec):
    messages = getattr(env, '_public_binding_human_messages', None)
    if messages is None:
        messages = tuple(getattr(env, 'human_messages', ()) or ())
    facts = tuple(getattr(env, '_public_binding_effective_facts', ()) or ())
    observations = tuple(getattr(env, '_public_binding_scene_observations', ()) or ())
    return derive_public_binding_state(spec, safe_human_messages=messages, effective_facts=facts, scene_observations=observations, task_instruction=str((getattr(env, 'current_task_spec', None) or {}).get('instruction') or ''))

def _candidate_record(*, object_id: str, alias: str, obj: Any, visible: bool, source: str) -> dict[str, Any]:
    return {'object_id': str(object_id), 'alias': str(alias), 'object_type': str(getattr(obj, 'category', '') or ''), 'visible': bool(visible), 'source': str(source)}

def _candidate_summary(object_id: str, *, env: Any, visible_ids: set[str], source: str) -> dict[str, Any]:
    obj = getattr(env.core, 'objects', {}).get(object_id)
    if obj is None:
        return {'object_id': object_id, 'source': source, 'visible': object_id in visible_ids}
    tracker = getattr(env, '_catalog_tracker', None)
    alias = tracker.alias_for(object_id, getattr(obj, 'category', None)) if tracker is not None else object_id
    return _candidate_record(object_id=object_id, alias=alias, obj=obj, visible=object_id in visible_ids, source=source)

def _score_candidates(reference: str, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ref_text = _normalize_text(reference)
    ref_tokens = _tokens(reference)
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        alias = str(candidate.get('alias') or '')
        object_type = str(candidate.get('object_type') or '')
        text = ' '.join([alias, object_type]).strip()
        cand_tokens = _tokens(text)
        score = 0.0
        if ref_text == _normalize_text(alias):
            score += 1.0
        if ref_text == _normalize_text(object_type):
            score += 0.8
        if object_type and _normalize_text(object_type) in ref_text:
            score += 0.6
        alias_score = _semantic_alias_score(ref_text, candidate)
        if alias_score:
            score += alias_score
        overlap = len(ref_tokens & cand_tokens)
        if overlap:
            score += min(0.5, overlap / max(len(ref_tokens), 1))
        item = dict(candidate)
        item['score'] = float(score)
        scored.append(item)
    return sorted(scored, key=lambda item: (-float(item['score']), str(item.get('object_id'))))

def _multi_object_reference_reason(reference: str, candidates: Sequence[Mapping[str, Any]]) -> str | None:
    text = str(reference or '').strip().lower()
    if not text:
        return None
    matched_types = _matched_candidate_types(text, candidates)
    has_disjunction = bool(re.search('\\b(or|and)\\b|/|,', text))
    if has_disjunction and len(matched_types) > 1:
        return 'multi_object_reference'
    relation_words = {'on', 'in', 'inside', 'under', 'above', 'with', 'near', 'next', 'beside', 'between'}
    tokens = _tokens(text)
    if tokens & relation_words and len(matched_types) > 1:
        return 'relational_multi_object_reference'
    return None

def _matched_candidate_types(reference: str, candidates: Sequence[Mapping[str, Any]]) -> set[str]:
    ref_tokens = _tokens(reference)
    ref_text = _normalize_text(reference)
    matched: set[str] = set()
    for candidate in candidates:
        object_type = str(candidate.get('object_type') or '').lower()
        alias = str(candidate.get('alias') or '').lower()
        type_tokens = _tokens(object_type)
        alias_tokens = _tokens(alias)
        if object_type and object_type in reference:
            matched.add(object_type)
            continue
        if _semantic_alias_score(ref_text, candidate) > 0:
            matched.add(object_type or alias)
            continue
        if type_tokens and ref_tokens & type_tokens:
            matched.add(object_type or alias)
            continue
        if alias and alias in reference:
            matched.add(object_type or alias)
            continue
        if alias_tokens and ref_tokens & alias_tokens:
            matched.add(object_type or alias)
    return {item for item in matched if item}

def _strip_score(item: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out['confidence'] = _confidence(out.pop('score', 0.0))
    return out

def _confidence(score: Any) -> float:
    return float(min(max(float(score), 0.0), 1.0))

def _tokens(text: str) -> set[str]:
    return {token for token in re.split('[^a-zA-Z0-9]+', str(text).lower()) if token}

def _normalize_text(text: Any) -> str:
    return ' '.join((token for token in re.split('[^a-zA-Z0-9]+', str(text).replace('_', ' ').lower()) if token))

def _semantic_alias_score(reference: str, candidate: Mapping[str, Any]) -> float:
    object_type = _normalize_text(candidate.get('object_type') or '')
    aliases = _category_aliases(object_type)
    if not aliases:
        return 0.0
    ref_tokens = _tokens(reference)
    best = 0.0
    for alias in aliases:
        alias_text = _normalize_text(alias)
        if not alias_text:
            continue
        alias_tokens = _tokens(alias_text)
        if reference == alias_text:
            best = max(best, 0.85)
        elif alias_text in reference or reference in alias_text:
            best = max(best, 0.75)
        elif alias_tokens and alias_tokens <= ref_tokens:
            best = max(best, 0.65)
        elif alias_tokens and ref_tokens & alias_tokens:
            best = max(best, min(0.45, len(ref_tokens & alias_tokens) / max(len(ref_tokens), 1)))
    return best

def _category_aliases(category: str) -> set[str]:
    key = _normalize_text(category).replace(' ', '_')
    aliases = {'apple': {'apple', 'food', 'fruit'}, 'bar_soap': {'bar soap', 'soap', 'soap bottle', 'cleaner', 'cleaning item', 'cleaning supply', 'cleaning supplies'}, 'bottom_cabinet': {'bottom cabinet', 'cabinet', 'storage cabinet', 'storage container', 'container'}, 'bowl': {'bowl', 'dish', 'dishware', 'dirty dish', 'dirty dishes', 'clean dish', 'clean dishware', 'dirty dishware'}, 'breakfast_table': {'breakfast table', 'table', 'dining table'}, 'carving_knife': {'carving knife', 'knife', 'cutting tool', 'cutting utensil'}, 'coffee_table': {'coffee table', 'table'}, 'countertop': {'countertop', 'counter', 'counter top', 'storage', 'cleaning storage'}, 'derived_faucet': {'faucet', 'tap', 'water source', 'running water'}, 'electric_refrigerator': {'electric refrigerator', 'refrigerator', 'fridge'}, 'fridge': {'electric refrigerator', 'refrigerator', 'fridge'}, 'furniture_sink': {'furniture sink', 'sink', 'sink basin', 'basin'}, 'mug': {'mug', 'cup'}, 'plate': {'plate', 'dish', 'dishware', 'dirty dish', 'clean dishware'}}
    return set(aliases.get(key, set()))

def _category_mentioned_in_text(category: str, text: str) -> bool:
    category_tokens = [token for token in re.split('[^a-zA-Z0-9]+', str(category).lower()) if token]
    text_tokens = _tokens(text)
    if not category_tokens:
        return False
    if all((token in text_tokens for token in category_tokens)):
        return True
    compact_category = ' '.join(category_tokens)
    compact_text = ' '.join(sorted(text_tokens))
    if compact_category in str(text).lower().replace('_', ' '):
        return True
    aliases = {'electric_refrigerator': {'fridge', 'refrigerator'}, 'breakfast_table': {'breakfast table', 'table'}, 'furniture_sink': {'sink'}, 'bottom_cabinet': {'cabinet'}, 'top_cabinet': {'cabinet'}}
    aliases.update({key: names for key, names in ((category, _category_aliases(category)),) if names})
    alias_text = str(text).lower().replace('_', ' ')
    for key, names in aliases.items():
        if category == key and any((name in alias_text for name in names)):
            return True
    if len(category_tokens) > 1 and category_tokens[-1] in text_tokens:
        generic_ok = category_tokens[-1] in {'table', 'cabinet', 'sink', 'fridge', 'refrigerator'}
        return generic_ok
    return compact_category in compact_text
__all__ = ['ObjectBindingResult', 'ObjectReferenceResolver']
