from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from scoper.benchmark.common.canonical import canonical_json_bytes, sha256_bytes, sha256_json
SHARED_VOCABULARY_SCHEMA_VERSION = 'method-shared-vocabulary.v1'
SHARED_VOCABULARY_VERSION = SHARED_VOCABULARY_SCHEMA_VERSION
PUBLIC_VOCABULARY_CONTRACT_VERSION = SHARED_VOCABULARY_SCHEMA_VERSION
OBSERVE_MEMORY_SCHEMA_VERSION = 'method-text-hardqa.v2'
OBSERVE_REQUEST_SCHEMA_VERSION = 'memory-proposer-request.v3'
OBSERVE_MAX_FACT_MUTATIONS = 4
OBSERVE_MAX_RETAINED_FACTS = 16

def normalize_public_term(value: Any) -> str:
    text = re.sub('(?<!^)(?=[A-Z])', ' ', str(value or ''))
    tokens = [token for token in re.split('[^a-z0-9]+', text.lower().replace('_', ' ')) if token and token not in {'a', 'an', 'the', 'this', 'that', 'one'}]
    if tokens:
        final = tokens[-1]
        if final.endswith('ies') and len(final) > 3:
            tokens[-1] = final[:-3] + 'y'
        elif final.endswith('s') and (not final.endswith('ss')):
            tokens[-1] = final[:-1]
    return ' '.join(tokens)

@dataclass(frozen=True)
class VocabularyFamily:
    canonical_family: str
    public_aliases: tuple[str, ...]
    raw_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.canonical_family or re.fullmatch('[a-z][a-z0-9_]*', self.canonical_family) is None:
            raise ValueError('canonical family must be a normalized identifier')
        aliases = tuple(self.public_aliases)
        if not aliases or aliases != tuple(sorted(set(aliases), key=str.encode)):
            raise ValueError('public aliases must be sorted and unique')
        if any((normalize_public_term(alias) != alias or not alias for alias in aliases)):
            raise ValueError('public aliases must be normalized non-empty text')
        raw_types = tuple(self.raw_types)
        if raw_types != tuple(sorted(set(raw_types), key=str.encode)):
            raise ValueError('raw types must be sorted and unique')

    @property
    def public_forms(self) -> tuple[str, ...]:
        display_name = self.canonical_family.replace('_', ' ')
        return tuple(sorted({display_name, *self.public_aliases}, key=str.encode))

    def to_public_dict(self) -> dict[str, Any]:
        return {'canonical_family': self.canonical_family, 'public_aliases': list(self.public_aliases)}

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_public_dict(), 'raw_types': list(self.raw_types)}

class SharedVocabularyRegistry:

    def __init__(self, families: Iterable[VocabularyFamily], *, forbidden_operation_terms: Iterable[str], query_only_terms: Iterable[str], backend_specific_exclusions: Mapping[str, Iterable[str]], version: str=SHARED_VOCABULARY_VERSION) -> None:
        normalized_families = tuple(sorted(tuple(families), key=lambda item: item.canonical_family.encode('utf-8')))
        if not normalized_families or len({item.canonical_family for item in normalized_families}) != len(normalized_families):
            raise ValueError('vocabulary families must be non-empty and unique')
        if not isinstance(version, str) or not version:
            raise ValueError('vocabulary version must be non-empty text')
        forbidden = _normalized_terms(forbidden_operation_terms)
        query_only = _normalized_terms(query_only_terms)
        if set(forbidden) & set(query_only):
            raise ValueError('forbidden and query-only terms must be disjoint')
        alias_to_family: dict[str, str] = {}
        for family in normalized_families:
            for form in family.public_forms:
                normalized_form = normalize_public_term(form)
                if normalized_form in alias_to_family and alias_to_family[normalized_form] != family.canonical_family:
                    raise ValueError(f'public vocabulary alias maps to multiple families: {form!r}')
                alias_to_family[normalized_form] = family.canonical_family
        if set(alias_to_family) & (set(forbidden) | set(query_only)):
            raise ValueError('broad vocabulary terms cannot be family aliases')
        exclusions: dict[str, tuple[str, ...]] = {}
        for backend, terms in backend_specific_exclusions.items():
            if not isinstance(backend, str) or not backend:
                raise ValueError('backend exclusion names must be non-empty text')
            exclusions[backend] = _normalized_terms(terms)
        self.version = version
        self.families = normalized_families
        self.forbidden_operation_terms = forbidden
        self.query_only_terms = query_only
        self.backend_specific_exclusions = {key: exclusions[key] for key in sorted(exclusions, key=str.encode)}
        self._alias_to_family = alias_to_family
        self._family_by_name = {item.canonical_family: item for item in normalized_families}

    @property
    def canonical_families(self) -> tuple[str, ...]:
        return tuple((item.canonical_family for item in self.families))

    @property
    def public_terms(self) -> tuple[str, ...]:
        return tuple(sorted({form for family in self.families for form in family.public_forms}, key=str.encode))

    @property
    def vocabulary_hash(self) -> str:
        return sha256_json(self.to_public_dict())

    @property
    def prompt_hash(self) -> str:
        return sha256_bytes(self.prompt_bytes())

    def family(self, value: Any) -> str | None:
        normalized = normalize_public_term(value)
        return self._alias_to_family.get(normalized)

    def aliases(self, family: Any) -> tuple[str, ...]:
        canonical = self.family(family) or normalize_public_term(family).replace(' ', '_')
        item = self._family_by_name.get(canonical)
        return () if item is None else item.public_forms

    def classify(self, value: Any) -> str:
        normalized = normalize_public_term(value)
        if not normalized:
            return 'unknown'
        if normalized in self.forbidden_operation_terms:
            return 'forbidden_operation'
        if normalized in self.query_only_terms:
            return 'query_only'
        if normalized in self._alias_to_family:
            return 'family'
        return 'unknown'

    def is_operation_target(self, value: Any) -> bool:
        return self.classify(value) == 'family'

    def is_query(self, value: Any) -> bool:
        return self.classify(value) in {'query_only', 'forbidden_operation'}

    def to_public_dict(self) -> dict[str, Any]:
        return {'schema_version': SHARED_VOCABULARY_SCHEMA_VERSION, 'version': self.version, 'families': [item.to_public_dict() for item in self.families], 'forbidden_operation_terms': list(self.forbidden_operation_terms), 'query_only_terms': list(self.query_only_terms), 'backend_specific_exclusions': {key: list(value) for key, value in self.backend_specific_exclusions.items()}}

    def to_dict(self) -> dict[str, Any]:
        value = self.to_public_dict()
        value['families'] = [item.to_dict() for item in self.families]
        return value

    def prompt_descriptor(self) -> dict[str, Any]:
        return {'schema_version': SHARED_VOCABULARY_SCHEMA_VERSION, 'version': self.version, 'vocabulary_sha256': self.vocabulary_hash, 'canonical_families': list(self.canonical_families), 'public_forms': {item.canonical_family: list(item.public_forms) for item in self.families}, 'forbidden_operation_terms': list(self.forbidden_operation_terms), 'query_only_terms': list(self.query_only_terms), 'unknown_term_policy': 'advisory_observation_only'}

    def prompt_bytes(self) -> bytes:
        return b'Use the following fixed public semantic vocabulary for concrete references. It is a global vocabulary, not a list of task targets. Choose the most specific visible family and use one of its listed public forms. Do not expose simulator IDs or backend type names. Broad and unknown terms may guide observation but are not interaction targets.\n' + canonical_json_bytes(self.prompt_descriptor()) + b'\n'

    def public_contract(self) -> dict[str, Any]:
        return {'schema_version': SHARED_VOCABULARY_SCHEMA_VERSION, 'vocabulary': self.prompt_descriptor(), 'prompt_sha256': self.prompt_hash, 'operation_target_rule': 'family_only; broad_or_unknown_rejected', 'unknown_observation_rule': 'advisory_observation_only'}

def _normalized_terms(values: Iterable[str]) -> tuple[str, ...]:
    normalized = {normalize_public_term(value) for value in values}
    normalized.discard('')
    return tuple(sorted(normalized, key=str.encode))

def _family(canonical: str, aliases: Iterable[str], raw_types: Iterable[str]) -> VocabularyFamily:
    normalized_aliases = _normalized_terms(aliases)
    return VocabularyFamily(canonical_family=str(canonical), public_aliases=normalized_aliases, raw_types=tuple(sorted(set(raw_types), key=str.encode)))
_FAMILY_GROUPS = ((('apple', ('apple',), ('Apple',)), ('bread', ('bread',), ('Bread',)), ('egg', ('egg',), ('Egg',)), ('lettuce', ('lettuce',), ('Lettuce',)), ('potato', ('potato',), ('Potato',)), ('tomato', ('tomato',), ('Tomato',))), (('plate', ('plate',), ('Plate',)), ('bowl', ('bowl',), ('Bowl',)), ('drinking_vessel', ('cup', 'mug'), ('Cup', 'Mug')), ('pan', ('pan', 'skillet'), ('Pan',)), ('pot', ('pot',), ('Pot',))), (('knife', ('knife',), ('Knife',)), ('fork', ('fork',), ('Fork',)), ('spoon', ('spoon',), ('Spoon',)), ('spatula', ('spatula',), ('Spatula',)), ('ladle', ('ladle',), ('Ladle',))), (('soap_bottle', ('soap bottle',), ('SoapBottle',)), ('spray_bottle', ('spray bottle',), ('SprayBottle',))), (('book', ('book',), ('Book',)), ('pencil', ('pencil',), ('Pencil',)), ('keychain', ('keychain', 'key chain'), ('KeyChain',))), (('cabinet', ('cabinet', 'cupboard'), ('Cabinet',)), ('drawer', ('drawer',), ('Drawer',)), ('shelf', ('shelf', 'shelving unit'), ('Shelf',)), ('fridge', ('fridge', 'refrigerator'), ('Fridge',)), ('microwave', ('microwave',), ('Microwave',)), ('sink_basin', ('sink', 'sink basin', 'basin'), ('Sink', 'SinkBasin')), ('faucet', ('faucet', 'tap'), ('Faucet',))), (('countertop', ('countertop', 'counter top'), ('CounterTop',)), ('dining_table', ('dining table',), ('DiningTable',)), ('coffee_table', ('coffee table',), ('CoffeeTable',)), ('side_table', ('side table',), ('SideTable',)), ('armchair', ('armchair',), ('ArmChair',)), ('sofa', ('sofa', 'couch'), ('Sofa',)), ('ottoman', ('ottoman',), ('Ottoman',)), ('tv_stand', ('tv stand', 'television stand'), ('TVStand',)), ('garbage_can', ('garbage can', 'trash can'), ('GarbageCan',))), (('desk_lamp', ('desk lamp', 'table lamp'), ('DeskLamp',)), ('floor_lamp', ('floor lamp',), ('FloorLamp',)), ('light_switch', ('light switch', 'switch'), ('LightSwitch',)), ('stove_burner', ('stove burner', 'burner'), ('StoveBurner',))))
SHARED_VOCABULARY_REGISTRY = SharedVocabularyRegistry((_family(canonical, aliases, raw_types) for group in _FAMILY_GROUPS for canonical, aliases, raw_types in group), forbidden_operation_terms=('dish', 'dishware', 'storage', 'storage container'), query_only_terms=('food', 'fruit', 'household issue', 'safety issue', 'table', 'container'), backend_specific_exclusions={'BEHAVIOR': ('oven',), 'ProcTHOR': ('butter knife', 'stove knob')})
PUBLIC_VOCABULARY_TERMS = SHARED_VOCABULARY_REGISTRY.public_terms
CANONICAL_PUBLIC_FAMILIES = SHARED_VOCABULARY_REGISTRY.canonical_families
FORBIDDEN_OPERATION_TERMS = SHARED_VOCABULARY_REGISTRY.forbidden_operation_terms
QUERY_ONLY_TERMS = SHARED_VOCABULARY_REGISTRY.query_only_terms
BACKEND_SPECIFIC_EXCLUSIONS = SHARED_VOCABULARY_REGISTRY.backend_specific_exclusions
SHARED_VOCABULARY_SHA256 = SHARED_VOCABULARY_REGISTRY.vocabulary_hash
PUBLIC_VOCABULARY_PROMPT_BYTES = SHARED_VOCABULARY_REGISTRY.prompt_bytes()
PUBLIC_VOCABULARY_PROMPT_SHA256 = SHARED_VOCABULARY_REGISTRY.prompt_hash

def resolve_public_family(value: Any) -> str | None:
    return SHARED_VOCABULARY_REGISTRY.family(value)

def classify_public_reference(value: Any) -> str:
    return SHARED_VOCABULARY_REGISTRY.classify(value)

def is_operation_target(value: Any) -> bool:
    return SHARED_VOCABULARY_REGISTRY.is_operation_target(value)

def shared_vocabulary_descriptor() -> Mapping[str, Any]:
    return SHARED_VOCABULARY_REGISTRY.public_contract()
__all__ = ['BACKEND_SPECIFIC_EXCLUSIONS', 'CANONICAL_PUBLIC_FAMILIES', 'FORBIDDEN_OPERATION_TERMS', 'OBSERVE_MAX_FACT_MUTATIONS', 'OBSERVE_MAX_RETAINED_FACTS', 'OBSERVE_MEMORY_SCHEMA_VERSION', 'OBSERVE_REQUEST_SCHEMA_VERSION', 'PUBLIC_VOCABULARY_CONTRACT_VERSION', 'PUBLIC_VOCABULARY_PROMPT_BYTES', 'PUBLIC_VOCABULARY_PROMPT_SHA256', 'PUBLIC_VOCABULARY_TERMS', 'QUERY_ONLY_TERMS', 'SHARED_VOCABULARY_REGISTRY', 'SHARED_VOCABULARY_SCHEMA_VERSION', 'SHARED_VOCABULARY_SHA256', 'SHARED_VOCABULARY_VERSION', 'SharedVocabularyRegistry', 'VocabularyFamily', 'classify_public_reference', 'is_operation_target', 'normalize_public_term', 'resolve_public_family', 'shared_vocabulary_descriptor']
