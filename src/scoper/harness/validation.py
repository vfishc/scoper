from __future__ import annotations
import math
import re
import hashlib
from typing import Any, Mapping, Sequence
from scoper.harness.contracts import MethodContractError
from scoper.harness.leakage import assert_no_method_leakage
SHA256_RE = re.compile('^[0-9a-f]{64}$')
_PRIVATE_SEMANTIC_TOKENS = ('exactpose', 'goaltruth', 'objectid', 'oracle', 'private', 'privileged', 'schedulerevent', 'schedulertruth')

def exact_mapping(value: Any, fields: set[str] | frozenset[str], *, pointer: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodContractError('method_schema_invalid', 'value must be an object', pointer=pointer)
    if set(value) != set(fields):
        raise MethodContractError('method_schema_invalid', 'object keys differ; expected={!r}, actual={!r}'.format(sorted(fields), sorted(value)), pointer=pointer)
    return value

def non_empty_string(value: Any, *, pointer: str) -> str:
    if not isinstance(value, str) or not value:
        raise MethodContractError('method_schema_invalid', 'value must be non-empty text', pointer=pointer)
    return value

def optional_string(value: Any, *, pointer: str) -> str | None:
    if value is None:
        return None
    return non_empty_string(value, pointer=pointer)

def public_semantic_name(value: Any, *, pointer: str) -> str:
    result = non_empty_string(value, pointer=pointer)
    normalized = re.sub('[^a-z0-9]', '', result.casefold())
    if any((token in normalized for token in _PRIVATE_SEMANTIC_TOKENS)):
        raise MethodContractError('method_private_key_forbidden', 'private-only semantic value is forbidden', pointer=pointer)
    return result

def integer(value: Any, *, pointer: str, minimum: int=0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MethodContractError('method_schema_invalid', 'value must be an integer >= {}'.format(minimum), pointer=pointer)
    return value

def finite_number(value: Any, *, pointer: str, minimum: float | None=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        raise MethodContractError('method_schema_invalid', 'value must be a finite number', pointer=pointer)
    result = float(value)
    if minimum is not None and result < minimum:
        raise MethodContractError('method_schema_invalid', 'value must be >= {}'.format(minimum), pointer=pointer)
    return result

def probability(value: Any, *, pointer: str) -> float:
    result = finite_number(value, pointer=pointer)
    if not 0.0 <= result <= 1.0:
        raise MethodContractError('method_schema_invalid', 'probability must be in [0, 1]', pointer=pointer)
    return result

def sha256(value: Any, *, pointer: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise MethodContractError('method_schema_invalid', 'value must be lowercase sha256', pointer=pointer)
    return value

def string_sequence(value: Any, *, pointer: str, non_empty: bool=False, unique: bool=True, sorted_utf8: bool=False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise MethodContractError('method_schema_invalid', 'value must be a string list', pointer=pointer)
    result = tuple(value)
    if non_empty and (not result):
        raise MethodContractError('method_schema_invalid', 'list must not be empty', pointer=pointer)
    if any((not isinstance(item, str) or not item for item in result)):
        raise MethodContractError('method_schema_invalid', 'list items must be non-empty text', pointer=pointer)
    if unique and len(set(result)) != len(result):
        raise MethodContractError('method_schema_invalid', 'list items must be unique', pointer=pointer)
    if sorted_utf8 and result != tuple(sorted(result, key=lambda item: item.encode('utf-8'))):
        raise MethodContractError('method_schema_invalid', 'list must be UTF-8 sorted', pointer=pointer)
    return result

def public_json(value: Any, *, pointer: str, include_future_action: bool=True) -> None:
    assert_no_method_leakage(value, root_pointer=pointer, include_geometry=True, include_future_action=include_future_action)

def validate_image_payload_bindings(declared_images: Mapping[str, Any], image_payloads: Mapping[str, bytes], *, actual_metadata: Mapping[str, Mapping[str, Any]] | None=None) -> None:
    if not isinstance(declared_images, Mapping) or not isinstance(image_payloads, Mapping):
        raise MethodContractError('method_media_invalid', 'image bindings must be mappings')
    declared_roles = tuple(declared_images)
    payload_roles = tuple(image_payloads)
    if set(declared_roles) != set(payload_roles):
        raise MethodContractError('method_event_correlation_invalid', 'image payload roles differ from request')
    canonical_declared = ordered_image_roles(declared_images)
    canonical_payload = ordered_image_roles(image_payloads)
    if canonical_declared != canonical_payload or payload_roles != canonical_payload:
        raise MethodContractError('method_event_correlation_invalid', 'image payload semantic order differs from request')
    if actual_metadata is not None and set(actual_metadata) != set(payload_roles):
        raise MethodContractError('method_event_correlation_invalid', 'encoded image roles/order differ from request')
    for role in canonical_declared:
        metadata = declared_images[role]
        if not isinstance(metadata, Mapping):
            raise MethodContractError('method_media_invalid', 'image metadata is invalid')
        digest = metadata.get('media_sha256')
        if sha256(digest, pointer=f'/images/{role}/media_sha256') != hashlib.sha256(image_payloads[role]).hexdigest():
            raise MethodContractError('method_media_invalid', f'image payload hash differs for {role}')
        if actual_metadata is not None and dict(actual_metadata[role]) != dict(metadata):
            raise MethodContractError('method_media_invalid', f'encoded image metadata differs for {role}')

def ordered_image_roles(images: Mapping[str, Any]) -> tuple[str, ...]:
    roles = tuple(images)
    if set(roles) == {'pre_rgb', 'post_rgb'}:
        return ('pre_rgb', 'post_rgb')
    if set(roles) == {'pre_rgb', 'immediate_post_rgb', 'confirmation_rgb'}:
        return ('pre_rgb', 'immediate_post_rgb', 'confirmation_rgb')
    if 'current_rgb' in roles and len(roles) == 1:
        return ('current_rgb',)
    if roles and all((role.startswith('window_rgb_') and role.removeprefix('window_rgb_').isdigit() for role in roles)):
        return tuple(sorted(roles, key=lambda role: int(role.removeprefix('window_rgb_'))))
    return roles

def public_scalar(value: Any, *, pointer: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        public_json(value, pointer=pointer)
        return value
    if isinstance(value, float) and math.isfinite(value):
        public_json(value, pointer=pointer)
        return value
    raise MethodContractError('method_schema_invalid', 'public fact object/value must be a finite JSON scalar', pointer=pointer)

def ensure_members(values: Sequence[str], allowed: Sequence[str], *, pointer: str) -> None:
    allowed_set = set(allowed)
    dangling = sorted(set(values) - allowed_set, key=lambda item: item.encode('utf-8'))
    if dangling:
        raise MethodContractError('method_public_reference_invalid', 'dangling public references: {!r}'.format(dangling), pointer=pointer)
__all__ = ['ensure_members', 'exact_mapping', 'finite_number', 'integer', 'non_empty_string', 'optional_string', 'probability', 'public_json', 'public_semantic_name', 'public_scalar', 'sha256', 'string_sequence', 'ordered_image_roles', 'validate_image_payload_bindings']
