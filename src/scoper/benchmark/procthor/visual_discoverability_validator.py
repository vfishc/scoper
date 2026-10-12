from __future__ import annotations
import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence
from scoper.benchmark.procthor.manipulathor_config import manipulathor_env_kwargs

@dataclass(frozen=True)
class VisualDiscoverabilityResult:
    incident_first_visible_tool_step: Optional[int] = None
    incident_first_visible_stream_step: Optional[int] = None
    incident_visibility_source: Optional[str] = None
    discoverability_passed: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

def validate_incident_visual_discoverability(task_spec: Mapping[str, Any], env: Any, geometry_estimator_already_passed: Optional[bool]=None) -> VisualDiscoverabilityResult:
    if geometry_estimator_already_passed is False:
        return VisualDiscoverabilityResult(discoverability_passed=False, failure_reason='rejected_by_geometry_prefilter')
    discovery = _visual_discovery_route(task_spec)
    if discovery is None:
        return VisualDiscoverabilityResult(discoverability_passed=False, failure_reason='visual_discovery_incident_or_route_missing')
    incident, route_roles = discovery
    incident_role = incident['object_role']
    incident_object_id = task_spec['roles'][incident_role]['object_id']
    env.reset_to_task(task_spec)
    if _object_visible_in_event(env.controller.last_event.metadata, incident_object_id):
        return VisualDiscoverabilityResult(discoverability_passed=False, failure_reason='incident visible at initial reset')
    probes = []
    for index, route_role in enumerate(route_roles):
        probes.extend([('navigation_stream', lambda role=route_role: env.call_tool('navigate_to', {'target': role}, reasoning='visual discoverability probe: {}'.format(role))), ('scan_at_object' if index == 0 else 'scan_at_receptacle', lambda role=route_role: env.call_tool('scan', {}, reasoning='visual discoverability probe: scan at {}'.format(role)))])
    for visibility_source, probe in probes:
        transition = probe()
        hit = _scan_low_level_records_for_visibility(transition.get('low_level_actions') or [], incident_object_id, visibility_source)
        if hit is not None:
            return VisualDiscoverabilityResult(incident_first_visible_tool_step=int(transition.get('high_level_step', getattr(env, 'high_level_step', 0)) or 0), incident_first_visible_stream_step=hit['stream_step'], incident_visibility_source=visibility_source, discoverability_passed=True)
    return VisualDiscoverabilityResult(discoverability_passed=False, failure_reason='incident not visible across all probe phases')

def _scan_low_level_records_for_visibility(records: Sequence[Mapping[str, Any]], target_object_id: str, visibility_source: Optional[str]=None) -> Optional[dict]:
    for index, record in enumerate(records):
        record_source = record.get('source')
        if visibility_source == 'navigation_stream':
            if record_source != 'navigation_stream':
                continue
        elif visibility_source in {'scan_at_object', 'scan_at_receptacle', 'scope_scan'}:
            if record_source != 'scan':
                continue
        elif visibility_source is not None and record_source != visibility_source:
            continue
        visible_ids = record.get('metadata_visible_object_ids') or []
        if target_object_id in visible_ids:
            return {'stream_step': index, 'source': record_source}
    return None

def _object_visible_in_event(metadata: Mapping[str, Any], object_id: str) -> bool:
    for obj in metadata.get('objects', []) or []:
        if obj.get('objectId') == object_id and obj.get('visible') is True:
            return True
    return False

def validate_task_bank_visual_discoverability(tasks: Iterable[Mapping[str, Any]], env: Any) -> tuple[List[dict], List[dict], dict]:
    kept: List[dict] = []
    rejected: List[dict] = []
    reasons: Counter = Counter()
    for task in tasks:
        task_copy = dict(task)
        if not has_visual_discovery_incident(task_copy):
            kept.append(task_copy)
            continue
        result = validate_incident_visual_discoverability(task_copy, env, geometry_estimator_already_passed=_geometry_prefilter_passed(task_copy))
        if result.discoverability_passed:
            task_copy.setdefault('metadata', {})['visual_discoverability'] = {'passed': True, 'source': result.incident_visibility_source, 'first_visible_tool_step': result.incident_first_visible_tool_step, 'first_visible_stream_step': result.incident_first_visible_stream_step}
            kept.append(task_copy)
        else:
            task_copy.setdefault('metadata', {})['visual_discoverability'] = {'passed': False, 'failure_reason': result.failure_reason}
            rejected.append(task_copy)
            reasons[result.failure_reason or 'unknown'] += 1
    summary = {'num_input': len(kept) + len(rejected), 'num_passed': len(kept), 'num_rejected': len(rejected), 'rejection_reasons': dict(sorted(reasons.items()))}
    return (kept, rejected, summary)

def has_visual_discovery_incident(task_spec: Mapping[str, Any]) -> bool:
    return _visual_discovery_route(task_spec) is not None

def _visual_discovery_route(task_spec: Mapping[str, Any]) -> Optional[tuple[Mapping[str, Any], List[str]]]:
    family = task_spec.get('family_spec')
    recover = family.get('recover') if isinstance(family, Mapping) else None
    if not isinstance(recover, Mapping):
        return None
    incidents: List[Mapping[str, Any]] = []
    incident = recover.get('incident')
    if isinstance(incident, Mapping):
        incidents.append(incident)
    incidents.extend((item for item in recover.get('incidents') or [] if isinstance(item, Mapping)))
    visual_incident = next((item for item in incidents if isinstance(item.get('activation'), Mapping) and item['activation'].get('type') == 'visual_discovery'), None)
    if visual_incident is None or not isinstance(visual_incident.get('object_role'), str):
        return None
    route_roles: List[str] = []
    primary = recover.get('primary_task')
    if isinstance(primary, Mapping):
        for field in ('object_role', 'receptacle_role'):
            role = primary.get(field)
            if isinstance(role, str) and role and (role not in route_roles):
                route_roles.append(role)
    if not route_roles:
        for item in incidents:
            if item is visual_incident:
                continue
            role = item.get('object_role') or item.get('target_role')
            if isinstance(role, str) and role and (role not in route_roles):
                route_roles.append(role)
    role_names = set((task_spec.get('roles') or {}).keys())
    route_roles = [role for role in route_roles if role in role_names]
    if not route_roles:
        return None
    return (visual_incident, route_roles)

def _geometry_prefilter_passed(task_spec: Mapping[str, Any]) -> Optional[bool]:
    for override in task_spec.get('init', {}).get('object_pose_overrides', []) or []:
        if not isinstance(override, Mapping):
            continue
        if override.get('mode') != 'route_visible_surface':
            continue
        if 'geometry_estimator_already_passed' in override:
            return bool(override['geometry_estimator_already_passed'])
    return None

def _load_jsonl(path: str) -> List[dict]:
    records = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records

def _write_jsonl(records: Iterable[Mapping[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')

def main(argv: Optional[Sequence[str]]=None) -> None:
    parser = argparse.ArgumentParser(description='Validate recover incident visual discoverability.')
    parser.add_argument('--task-bank', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rejected-output', required=True)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--gpu-device', type=int, default=None)
    parser.add_argument('--cloud-rendering', action='store_true')
    parser.add_argument('--manipulathor-arm', action='store_true', help="Use AI2-THOR's official ManipulaTHOR arm agent mode.")
    parser.add_argument('--manipulathor-visual-attach', action='store_true', help='Show the ManipulaTHOR arm and use the visual-attach arm pose.')
    parser.add_argument('--agent-radius', type=float, default=None, help='Optional AI2-THOR SetAgentRadius override in meters. Visual-attach mode defaults to 0.3.')
    args = parser.parse_args(argv)
    from scoper.benchmark.procthor.closed_loop_tool_env import ProcTHORClosedLoopToolEnv
    env_kwargs = {'width': args.width, 'height': args.height, 'cloud_rendering': args.cloud_rendering, 'gpu_device': args.gpu_device}
    env_kwargs.update(manipulathor_env_kwargs(manipulathor_arm=args.manipulathor_arm, manipulathor_visual_attach=args.manipulathor_visual_attach, agent_radius=args.agent_radius))
    env = ProcTHORClosedLoopToolEnv(**env_kwargs)
    try:
        kept, rejected, summary = validate_task_bank_visual_discoverability(_load_jsonl(args.task_bank), env)
    finally:
        close = getattr(env, 'close', None)
        if callable(close):
            close()
    _write_jsonl(kept, args.output)
    _write_jsonl(rejected, args.rejected_output)
    summary_path = str(Path(args.output).with_suffix(Path(args.output).suffix + '.summary.json'))
    Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
if __name__ == '__main__':
    main()
__all__ = ['VisualDiscoverabilityResult', '_geometry_prefilter_passed', '_object_visible_in_event', '_scan_low_level_records_for_visibility', 'validate_incident_visual_discoverability', 'validate_task_bank_visual_discoverability']
