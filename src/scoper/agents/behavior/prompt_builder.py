from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence
import yaml
from scoper.benchmark.behavior.brains.base import BrainInput
from scoper.agents.behavior.config import PromptExposureConfig
from scoper.agents.behavior.streaming import StreamFrameObservation
from scoper.agents.behavior.message import Message
from scoper.benchmark.behavior.paths import prompt_templates_dir
DEFAULT_TEMPLATE_DIR = prompt_templates_dir()

class PromptTemplateError(ValueError):
    pass

class PromptBuilder:

    def __init__(self, template_name: str, template_dir: Optional[Path]=None, exposure: PromptExposureConfig | None=None, eval_mode: str='agent', direct_multistep_v2: bool=False) -> None:
        self.template_name = str(template_name)
        self.template_dir = Path(template_dir or DEFAULT_TEMPLATE_DIR)
        self.template = self._load_template(self.template_name)
        self.exposure = exposure or PromptExposureConfig()
        self.eval_mode = str(eval_mode or 'agent')
        self.direct_multistep_v2 = bool(direct_multistep_v2)
        self.task_context: dict = {}

    def set_task_context(self, task_spec: Mapping[str, Any], tool_specs: Sequence[Mapping[str, Any]], prompt_context: Optional[Mapping[str, Any]]=None) -> None:
        from scoper.benchmark.common.evaluation_binding_policy import task_uses_evidence_binding
        self.evidence_binding = task_uses_evidence_binding(task_spec)
        from scoper.benchmark.common.evaluation_binding_policy import task_uses_refined_binding
        self.refined_binding = task_uses_refined_binding(task_spec)
        context = dict(prompt_context or {})
        self.task_context = {'task_id': None if self.evidence_binding else task_spec.get('id'), 'instruction': task_spec.get('instruction', ''), 'category': 'household task' if self.evidence_binding else task_spec.get('category', ''), 'tool_specs': self._tool_specs_for_prompt(tool_specs) if self.exposure.tool_specs else [], 'tool_contract': context.get('tool_contract') if self.exposure.tool_contract else {}}
        if self.eval_mode == 'standard_vlm' and self.exposure.tool_contract:
            from scoper.benchmark.common.direct_tool_guidance import public_tool_contract
            from scoper.benchmark.common.evaluation_policy import step_budget
            from scoper.benchmark.common.public_object_names import public_object_text
            self.task_context['tool_contract'] = public_tool_contract('behavior', self.task_context['tool_specs'])
            self.task_context['instruction'] = public_object_text(self.task_context['instruction'])
            self.task_context['step_budget'] = step_budget(task_spec)

    def build_system_message(self) -> Message:
        values = {'instruction': self.task_context.get('instruction', ''), 'category': self.task_context.get('category', ''), 'tool_specs_json': format_tool_specs(self.task_context.get('tool_specs') or []), 'tool_contract': format_json(self.task_context.get('tool_contract') or {}), 'standard_vlm_rules': self._standard_vlm_rules()}
        return _text_message('system', self._render('system', values))

    def build_user_message(self, brain_input: BrainInput, frames: Sequence[Any]) -> Message:
        observation = brain_input.observation or {}
        prompt_context = brain_input.prompt_context or {}
        if not self.exposure.visible_catalog:
            catalog_for_prompt = []
        elif 'visible_object_catalog' in prompt_context:
            catalog_for_prompt = visible_object_catalog(prompt_context)
        elif 'visible_object_catalog' in observation:
            catalog_for_prompt = visible_object_catalog(observation)
        elif 'object_catalog' in prompt_context:
            catalog_for_prompt = visible_object_catalog(prompt_context)
        else:
            catalog_for_prompt = visible_object_catalog(observation)
        values = {'high_level_step': brain_input.high_level_step, 'last_transition_summary': format_last_transition(brain_input.last_transition, exposure=self.exposure), 'object_catalog_summary': format_object_catalog(catalog_for_prompt or [], exposure=self.exposure), 'held_objects': format_json(observation.get('held_objects') or prompt_context.get('held_objects') or []) if self.exposure.held_objects else '', 'agent_pose': format_json(observation.get('agent_pose') or {}) if self.exposure.agent_pose else '', 'current_room': format_json(observation.get('current_room') or prompt_context.get('current_room') or None) if self.exposure.current_room else '', 'target_handles': format_json(prompt_context.get('tool_target_handles') or {}) if self.exposure.target_handles else '', 'room_handles': format_json(prompt_context.get('room_target_handles') or {}) if self.exposure.room_handles else '', 'human_messages': format_json(prompt_context.get('human_messages') or observation.get('human_messages') or []) if self.exposure.human_messages else '', 'visible_issues': format_json(prompt_context.get('visible_issues') or observation.get('visible_issues') or []) if self.exposure.visible_issues else '', 'scheduler_notices': format_json(prompt_context.get('scheduler_notices') or []) if self.exposure.scheduler_notices else '', 'tool_history': format_tool_history(prompt_context.get('tool_history') or [], exposure=self.exposure, history_limit=self._public_history_limit()), 'standard_vlm_turn_rules': self._standard_vlm_turn_rules(), 'prompt_context_json': format_prompt_context(prompt_context, exposure=self.exposure)}
        text = self._render('user_turn', values)
        if self.eval_mode == 'standard_vlm' and self.exposure.tool_history:
            history_text = values.get('tool_history', '')
            if history_text and history_text != '(empty)' and (history_text not in text):
                text += '\n\nRequested action history (not evidence of success):\n' + history_text
        if self.eval_mode == 'standard_vlm' and 'step_budget' in self.task_context:
            from scoper.benchmark.common.direct_tool_guidance import stop_budget_guidance
            maximum = brain_input.prompt_context.get('evaluation_step_budget', self.task_context['step_budget'])
            text += '\n\n' + stop_budget_guidance(brain_input.high_level_step, maximum)
            if getattr(self, 'refined_binding', False):
                from scoper.benchmark.common.direct_tool_guidance import qa_phase_guidance
                text += '\n\n' + qa_phase_guidance(prompt_context.get('tool_history') or [])
        content = [{'type': 'text', 'text': text}]
        image_blocks = self._image_blocks(frames)
        if str(self.template.get('image_position') or 'after_text') == 'before_text':
            content = image_blocks + content
        else:
            content.extend(image_blocks)
        return {'role': 'user', 'content': content}

    def build(self, brain_input: BrainInput, frames: Sequence[Any], conversation_history: Sequence[Message]) -> List[Message]:
        return [self.build_system_message(), *list(conversation_history), self.build_user_message(brain_input, frames)]

    def build_stream_message(self, frame: StreamFrameObservation) -> Message:
        values = {'instruction': self.task_context.get('instruction', ''), 'active_tool_name': frame.active_tool_name, 'active_tool_args': format_json(frame.active_tool_args) if self.exposure.target_handles else '[arguments hidden]', 'frame_index': frame.frame_index, 'timestamp_s': '{:.3f}'.format(float(frame.timestamp_s)), 'source': frame.source, 'is_keyframe': frame.is_keyframe, 'keyframe_reason': frame.keyframe_reason or '', 'object_catalog_summary': format_object_catalog(frame.visible_object_catalog or [], exposure=self.exposure) if self.exposure.visible_catalog else '(not exposed)', 'held_objects': format_json(frame.prompt_metadata.get('held_objects') or []) if self.exposure.held_objects else '', 'current_room': format_json(frame.prompt_metadata.get('current_room')) if self.exposure.current_room else '', 'target_handles': format_json(frame.prompt_metadata.get('tool_target_handles') or {}) if self.exposure.target_handles else '', 'room_handles': format_json(frame.prompt_metadata.get('room_target_handles') or {}) if self.exposure.room_handles else '', 'human_messages': format_json(frame.prompt_metadata.get('human_messages') or []) if self.exposure.human_messages else '', 'visible_issues': format_json(frame.prompt_metadata.get('visible_issues') or []) if self.exposure.visible_issues else '', 'tool_progress': format_json(frame.prompt_metadata.get('tool_progress') or {}), 'tool_result_summary': format_stream_tool_result(frame.prompt_metadata.get('tool_result') or {}, exposure=self.exposure, eval_mode=self.eval_mode) if self.exposure.tool_feedback else '', 'prompt_context_json': format_prompt_context(frame.prompt_metadata or {}, exposure=self.exposure), 'standard_vlm_turn_rules': self._standard_vlm_turn_rules()}
        text = self._render('stream_turn', values)
        content = [{'type': 'text', 'text': text}]
        if frame.head_rgb is not None:
            content.append({'type': 'image', 'rgb': frame.head_rgb})
        return {'role': 'user', 'content': content}

    def exposure_snapshot(self) -> dict:
        return {'eval_mode': self.eval_mode, 'exposure': dict(self.exposure.__dict__)}

    def _load_template(self, template_name: str) -> Mapping[str, Any]:
        path = self.template_dir / '{}.yaml'.format(template_name)
        if not path.exists():
            raise PromptTemplateError("Prompt template '{}' not found.".format(path))
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if not isinstance(data, Mapping):
            raise PromptTemplateError('Prompt template must be a mapping.')
        for key in ('system', 'user_turn'):
            if not isinstance(data.get(key), str) or not data.get(key):
                raise PromptTemplateError("Prompt template '{}' requires non-empty '{}'.".format(template_name, key))
        return data

    def _render(self, field: str, values: Mapping[str, Any]) -> str:
        template = str(self.template.get(field) or _default_template_field(field) or '')
        if self.eval_mode == 'standard_vlm':
            from scoper.benchmark.common.public_prompt_template import prepare_public_template
            template = prepare_public_template(template, self.exposure, hide_empty_human_messages=values.get('human_messages') in {'', '[]'})
        try:
            rendered = template.format(**values).strip()
        except KeyError as exc:
            raise PromptTemplateError("Prompt template references unknown key '{}'.".format(exc.args[0])) from exc
        if self.eval_mode == 'standard_vlm':
            human_messages = str(values.get('human_messages') or '')
            if self.exposure.human_messages and human_messages not in {'', '[]'} and ('Human messages:' not in rendered):
                rendered = '{}\n\nHuman messages:\n{}'.format(rendered, human_messages)
            if field == 'system' and self.task_context.get('tool_specs'):
                from scoper.benchmark.common.public_prompt_template import require_complete_tool_schemas
                require_complete_tool_schemas(rendered, values['tool_specs_json'])
        return rendered

    def _public_history_limit(self):
        if self.eval_mode == 'standard_vlm':
            from scoper.benchmark.common.direct_tool_guidance import public_history_limit
            return public_history_limit(self.task_context.get('step_budget', 48))
        return 32 if self.direct_multistep_v2 else 8

    def _image_blocks(self, frames: Sequence[Any]) -> List[dict]:
        label_format = str(self.template.get('image_label_format') or '')
        blocks: List[dict] = []
        for index, frame in enumerate(frames, start=1):
            if label_format:
                blocks.append({'type': 'text', 'text': label_format.format(index=index, total=len(frames))})
            blocks.append({'type': 'image', 'rgb': frame})
        return blocks

    def _tool_specs_for_prompt(self, tool_specs: Sequence[Mapping[str, Any]]) -> list[dict]:
        specs = [dict(spec) for spec in tool_specs]
        if self.eval_mode != 'standard_vlm':
            return specs
        converted: list[dict] = []
        for spec in specs:
            item = dict(spec)
            name = str(item.get('name') or '')
            item['description'] = _standard_vlm_tool_description(name, str(item.get('description') or ''))
            params = item.get('parameters')
            if isinstance(params, Mapping):
                params = dict(params)
                props = dict(params.get('properties') or {})
                required = list(params.get('required') or [])
                if 'target' in props:
                    props.pop('target', None)
                    props['object_ref'] = {'type': 'string', 'description': "Visual reference to exactly one object or receptacle in the image, e.g. 'mug', 'fridge', or a visible alias if one was shown. Do not include a second object or a relation phrase."}
                    required = ['object_ref' if value == 'target' else value for value in required]
                if 'room' in props:
                    props['room'] = {'type': 'string', 'description': "Natural-language room name or visible room label if one is known, such as 'kitchen'."}
                params['properties'] = props
                params['required'] = required
                item['parameters'] = params
            if getattr(self, 'evidence_binding', False):
                from scoper.benchmark.common.evaluation_binding_policy import reference_tool_schema
                item = reference_tool_schema(item)
            converted.append(item)
        return converted

    def _standard_vlm_rules(self) -> str:
        from scoper.benchmark.common.evaluation_policy import standard_system_rules
        if self.eval_mode != 'standard_vlm':
            return ''
        rules = standard_system_rules(self.direct_multistep_v2)
        if getattr(self, 'evidence_binding', False):
            from scoper.benchmark.common.evaluation_binding_policy import public_reference_rules
            rules += '\n' + public_reference_rules()
        if getattr(self, 'refined_binding', False):
            from scoper.benchmark.common.evaluation_binding_policy import QA_FINISH_RULES
            rules += '\n' + QA_FINISH_RULES
        return rules

    def _standard_vlm_turn_rules(self) -> str:
        from scoper.benchmark.common.evaluation_policy import standard_turn_rules
        return standard_turn_rules() if self.eval_mode == 'standard_vlm' else ''

    def _standard_vlm_dynamic_rules(self, tool_history) -> str:
        return ''

def _text_message(role: str, text: str) -> Message:
    return {'role': role, 'content': [{'type': 'text', 'text': str(text)}]}

def _standard_vlm_tool_description(name: str, fallback: str) -> str:
    from scoper.benchmark.common.direct_tool_guidance import tool_description
    return tool_description('behavior', name, fallback)

def _default_template_field(field: str) -> str:
    if field != 'stream_turn':
        return ''
    return 'You are watching the current tool execution stream.\n\nCurrent task: {instruction}\nActive tool: {active_tool_name}({active_tool_args})\nFrame: {frame_index}, timestamp={timestamp_s}\nSource: {source}, keyframe={is_keyframe}, reason={keyframe_reason}\nVisible objects:\n{object_catalog_summary}\nHeld objects: {held_objects}\nCurrent room: {current_room}\nAvailable role handles:\n{target_handles}\nAvailable room handles:\n{room_handles}\nHuman messages:\n{human_messages}\nVisible issues:\n{visible_issues}\nTool progress:\n{tool_progress}\nTool result:\n{tool_result_summary}\nAdditional prompt context:\n{prompt_context_json}\n\nReturn exactly one JSON object:\n{{"stream_decision": "continue", "reasoning": "..."}}\nor\n{{"stream_decision": "interrupt", "tool_name": "<tool>", "arguments": {{}}, "reasoning": "..."}}\n\nIf interrupting with an object tool, use role handles in arguments, not visible aliases.\nIf interrupting with navigate_to_room, use a room_id from the available room handles.'

def format_object_catalog(catalog: Sequence[Mapping[str, Any]], *, exposure: PromptExposureConfig | None=None) -> str:
    exposure = exposure or PromptExposureConfig()
    if not catalog:
        return '(none visible)'
    lines = []
    for item in catalog[:40]:
        affordances = [key for key in ('pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable') if item.get(key)]
        fields = ['- {}'.format(item.get('alias') or item.get('objectType') or 'object'), 'type={}'.format(item.get('objectType') or 'unknown')]
        if exposure.agent_pose:
            position = item.get('position') or {}
            fields.append('pos=({:.2f}, {:.2f}, {:.2f})'.format(float(position.get('x', 0.0)), float(position.get('y', 0.0)), float(position.get('z', 0.0))))
        if exposure.distance:
            distance = item.get('distance')
            fields.append('distance={}'.format('{:.2f}m'.format(float(distance)) if isinstance(distance, (int, float)) else 'unknown'))
        if exposure.current_room:
            fields.append('room={}'.format(item.get('room_id') or 'unknown'))
        fields.append('affordances=[{}]'.format(', '.join(affordances) or 'none'))
        if exposure.states and item.get('states') is not None:
            fields.append('states={}'.format(format_json(item.get('states'))))
        if exposure.bbox:
            bbox = {key: item.get(key) for key in ('aabb_center', 'aabb_extent') if item.get(key) is not None}
            if bbox:
                fields.append('bbox={}'.format(format_json(bbox)))
        if exposure.raw_object_ids:
            raw_id = item.get('objectId') or item.get('name')
            if raw_id:
                fields.append('id={}'.format(raw_id))
        lines.append('; '.join(fields))
    if len(catalog) > 40:
        lines.append('- ... {} more visible objects omitted'.format(len(catalog) - 40))
    return '\n'.join(lines)

def visible_object_catalog(source: Mapping[str, Any]) -> Optional[Sequence[Mapping[str, Any]]]:
    explicit = source.get('visible_object_catalog')
    if isinstance(explicit, Sequence) and (not isinstance(explicit, (str, bytes))):
        return explicit
    if 'object_catalog' not in source:
        return None
    catalog = source.get('object_catalog') or []
    if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes)):
        return []
    if any((isinstance(item, Mapping) and 'visible_from_current_pose' in item for item in catalog)):
        return [item for item in catalog if isinstance(item, Mapping) and bool(item.get('visible_from_current_pose'))]
    if any((isinstance(item, Mapping) and 'visible' in item for item in catalog)):
        return [item for item in catalog if isinstance(item, Mapping) and bool(item.get('visible'))]
    return catalog

def format_last_transition(last: Optional[Mapping[str, Any]], *, exposure: PromptExposureConfig | None=None) -> str:
    exposure = exposure or PromptExposureConfig()
    if not last:
        return 'No previous tool call.'
    if not exposure.tool_feedback:
        return 'A previous requested action has finished running.'
    if not exposure.target_handles and (not exposure.goal_progress) and (not exposure.state_delta):
        requested_args = _standard_vlm_requested_args(last.get('arguments') or {})
        return 'Previous requested action: {}({}).\n'.format(last.get('tool_name') or 'unknown', requested_args) + 'The previous requested action has finished running.\nOutcome is not provided; infer the current state from the image.'
    parts = ['Previous tool: {}({})'.format(last.get('tool_name') or 'unknown', format_json(last.get('arguments') or {}) if exposure.target_handles else '[arguments hidden]'), 'Success: {}'.format(bool(last.get('success', False)))]
    if exposure.tool_feedback:
        feedback = last.get('env_feedback')
        if feedback:
            parts.append('Feedback: {}'.format(feedback))
        if last.get('feedback'):
            parts.append('Tool message: {}'.format(last.get('feedback')))
    if last.get('done'):
        parts.append('Done reason: {}'.format(last.get('done_reason')))
    goal = last.get('goal_result') or {}
    if exposure.goal_progress and isinstance(goal, Mapping):
        parts.append('Goal: task_success={}, task_progress={}'.format(goal.get('task_success'), goal.get('task_progress')))
    state_transitions = last.get('state_transitions') or []
    if exposure.state_delta and state_transitions:
        parts.append('State changes: {}'.format(format_json(state_transitions)))
    return '\n'.join(parts)

def format_stream_tool_result(tool_result: Mapping[str, Any], *, exposure: PromptExposureConfig | None=None, eval_mode: str='agent') -> str:
    exposure = exposure or PromptExposureConfig()
    if not tool_result:
        return '{}'
    if str(eval_mode) == 'standard_vlm':
        return 'The requested action observation has finished running.\nOutcome is not provided; infer the current state from the image.'
    allowed = {'tool_feedback', 'tool_success'}
    if exposure.goal_progress:
        allowed.update({'goal_progress', 'goal_success'})
    filtered = {key: value for key, value in tool_result.items() if key in allowed}
    return format_json(filtered)

def format_tool_specs(specs: Sequence[Mapping[str, Any]]) -> str:
    compact = []
    for spec in specs:
        compact.append({'name': spec.get('name'), 'description': spec.get('description'), 'parameters': spec.get('parameters') or spec.get('args') or {}})
    return format_json(compact)

def format_tool_history(history: Sequence[Mapping[str, Any]], *, exposure: PromptExposureConfig | None=None, history_limit: int=8) -> str:
    exposure = exposure or PromptExposureConfig()
    if not exposure.tool_history or not history:
        return '(empty)'
    if not exposure.target_handles and (not exposure.goal_progress) and (not exposure.state_delta):
        rows = []
        for item in history[-history_limit:]:
            args = _standard_vlm_requested_args(item.get('arguments') or {})
            rows.append('- step {step}: requested {tool}({args}); outcome not provided'.format(step=item.get('high_level_step', '?'), tool=item.get('tool_name', 'unknown'), args=args))
        return '\n'.join(rows)
    rows = []
    for item in history[-history_limit:]:
        rows.append('- step {step}: {tool} success={success} feedback={feedback}'.format(step=item.get('high_level_step', '?'), tool=item.get('tool_name', 'unknown'), success=item.get('success'), feedback=item.get('feedback') or item.get('env_feedback', '')))
    return '\n'.join(rows)

def _standard_vlm_requested_args(arguments: Any) -> str:
    if not isinstance(arguments, Mapping) or not arguments:
        return ''
    safe: dict[str, Any] = {}
    for key, value in dict(arguments).items():
        key_text = str(key)
        if key_text == 'target':
            key_text = 'object_ref'
        if key_text == 'object_ref' and _looks_like_internal_handle(value):
            safe[key_text] = '[object reference hidden]'
        else:
            safe[key_text] = value
    return format_json(safe)

def _looks_like_internal_handle(value: Any) -> bool:
    text = str(value or '')
    return bool(text in {'target_object', 'target_receptacle'} or text.startswith('target_') or _looks_like_behavior_object_id(text))

def _looks_like_behavior_object_id(text: str) -> bool:
    parts = str(text or '').split('.n.')
    if len(parts) != 2:
        return False
    suffix = parts[1]
    return '_' in suffix and suffix.replace('_', '').isdigit()

def format_prompt_context(prompt_context: Mapping[str, Any], *, exposure: PromptExposureConfig | None=None) -> str:
    exposure = exposure or PromptExposureConfig()
    allowed = set()
    if exposure.structured_memory:
        allowed.update({'agent_belief', 'memory_summary'})
    if exposure.human_messages:
        allowed.update({'human_messages', 'human_answers'})
    if exposure.visible_issues:
        allowed.update({'visible_issues', 'discovered_issue_ids'})
    if exposure.scheduler_notices:
        allowed.update({'scheduler_notices', 'disturbance_notice'})
    if exposure.held_objects:
        allowed.add('held_objects')
    if exposure.current_room:
        allowed.update({'current_room', 'visited_rooms'})
    if exposure.target_handles:
        allowed.add('tool_target_handles')
    if exposure.room_handles:
        allowed.add('room_target_handles')
    filtered = {key: value for key, value in prompt_context.items() if key in allowed and key not in {'head_rgb', 'tool_specs', 'object_catalog', 'visible_object_catalog'}}
    return format_json(filtered)

def format_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

def text_only_message(message: Message, *, keep_images: bool=False) -> Message:
    content = []
    for block in message.get('content', []):
        if block.get('type') == 'text':
            content.append({'type': 'text', 'text': str(block.get('text', ''))})
        elif block.get('type') == 'image':
            if keep_images:
                content.append(block)
            else:
                content.append({'type': 'text', 'text': '[image omitted from history]'})
    return {'role': message['role'], 'content': content}
__all__ = ['PromptBuilder', 'PromptTemplateError', 'format_last_transition', 'format_object_catalog', 'format_tool_specs', 'visible_object_catalog', 'text_only_message']
