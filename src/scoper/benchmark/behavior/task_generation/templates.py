from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import yaml
from scoper.benchmark.behavior.task_spec import SUPPORTED_GOAL_PREDICATES
REQUIRED_TEMPLATE_FIELDS = ('template_id', 'template_version', 'category', 'roles', 'goal_predicates', 'instruction_templates')

class TemplateValidationError(ValueError):
    pass

def load_template(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_template(data)
    return data

def load_templates(dir_path: str | Path) -> list[dict]:
    templates = []
    for p in sorted(Path(dir_path).glob('*.yaml')):
        templates.append(load_template(p))
    return templates

def validate_template(template: Any) -> None:
    if not isinstance(template, Mapping):
        raise TemplateValidationError('Template must be a mapping.')
    missing = [f for f in REQUIRED_TEMPLATE_FIELDS if f not in template]
    if missing:
        raise TemplateValidationError('Template missing fields: {}.'.format(missing))
    if not isinstance(template['roles'], Mapping) or not template['roles']:
        raise TemplateValidationError("Template 'roles' must be a non-empty mapping.")
    preds = template['goal_predicates']
    if not isinstance(preds, list) or not preds:
        raise TemplateValidationError("Template 'goal_predicates' must be a non-empty list.")
    for p in preds:
        ptype = p.get('type')
        if ptype not in SUPPORTED_GOAL_PREDICATES:
            raise TemplateValidationError("Unsupported goal predicate '{}' in template.".format(ptype))
__all__ = ['load_template', 'load_templates', 'validate_template', 'TemplateValidationError']
