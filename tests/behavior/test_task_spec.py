import pytest
from scoper.benchmark.behavior.task_spec import TaskSpecValidationError, validate_behavior_task_spec
from tests.behavior.conftest import build_smoke_task_spec

def test_valid_spec():
    validate_behavior_task_spec(build_smoke_task_spec())

def test_missing_top_level_field_rejected():
    spec = build_smoke_task_spec()
    del spec['goal']
    with pytest.raises(TaskSpecValidationError):
        validate_behavior_task_spec(spec)

def test_wrong_backend_rejected():
    spec = build_smoke_task_spec()
    spec['backend'] = 'procthor'
    with pytest.raises(TaskSpecValidationError):
        validate_behavior_task_spec(spec)

def test_unsupported_predicate_rejected():
    spec = build_smoke_task_spec()
    spec['goal']['predicates'].append({'type': 'object_frozen', 'args': {'object': 'target_object'}})
    with pytest.raises(TaskSpecValidationError):
        validate_behavior_task_spec(spec)

def test_unknown_role_reference_rejected():
    spec = build_smoke_task_spec()
    spec['goal']['predicates'][0]['args']['object'] = 'no_such_role'
    with pytest.raises(TaskSpecValidationError):
        validate_behavior_task_spec(spec)

def test_substance_literal_not_treated_as_role():
    spec = build_smoke_task_spec()
    spec['goal']['predicates'] = [{'type': 'object_filled', 'args': {'object': 'target_object', 'substance': 'water'}}]
    validate_behavior_task_spec(spec)

def test_empty_predicates_rejected():
    spec = build_smoke_task_spec()
    spec['goal']['predicates'] = []
    with pytest.raises(TaskSpecValidationError):
        validate_behavior_task_spec(spec)

def test_bad_source_type_rejected():
    spec = build_smoke_task_spec()
    spec['source']['type'] = 'not_a_real_type'
    with pytest.raises(TaskSpecValidationError):
        validate_behavior_task_spec(spec)
