"""Public command entrypoint. Simulator imports occur after configuration."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import sys


COMMANDS = ('generate', 'oracle', 'evaluate', 'check-bank', 'summarize', 'doctor')


def configure(path):
    os.environ.setdefault('EB_PROCTHOR_PRIOR_OFFLINE', '1')
    if path is None:
        return {}
    import yaml
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text()) or {}
    allowed = {'resource_root', 'behavior_assets', 'procthor_prior_home',
               'procthor_executable', 'model', 'cache_root'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Unknown configuration keys; use configs/local.example.yaml')
    for key, value in config.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f'{key} must be a path string')
        expanded = os.path.expandvars(os.path.expanduser(value))
        if '$' in expanded:
            raise ValueError(f'Unresolved environment variable in {key}')
        p = Path(expanded)
        config[key] = str(p if p.is_absolute() else (path.parent/p).resolve())
    variables = {'resource_root':'SCOPER_RESOURCE_ROOT', 'behavior_assets':'SCOPER_BEHAVIOR_ASSETS',
                 'procthor_prior_home':'EB_PROCTHOR_PRIOR_HOME',
                 'procthor_executable':'EB_PROCTHOR_LOCAL_EXECUTABLE'}
    for key, env in variables.items():
        if key in config:
            os.environ[env] = config[key]
    os.environ.setdefault('EB_PROCTHOR_PRIOR_OFFLINE', '1')
    if 'cache_root' in config:
        root = Path(config['cache_root']); root.mkdir(parents=True, exist_ok=True)
        for key, directory in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCH_EXTENSIONS_DIR','torch_extensions')]:
            target=root/directory;target.mkdir(exist_ok=True);os.environ[key]=str(target)
    return config


def main(argv=None):
    parser=argparse.ArgumentParser(description='SCOPER task generation, oracle and direct-VLM evaluation')
    parser.add_argument('--config', type=Path, help='Paths YAML; relative paths resolve beside this file')
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    try:
        config=configure(args.config)
        if args.command in {'generate','oracle','check-bank'}:
            module=importlib.import_module('scoper.commands.'+args.command.replace('-','_'))
            return module.main(args.arguments)
        if args.command=='evaluate':
            return evaluate(args.arguments, config)
        if args.command=='summarize':
            return summarize(args.arguments)
        return doctor(args.arguments,config)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f'Error: {exc}\n')


def evaluate(argv, config):
    p=argparse.ArgumentParser(prog='scoper evaluate',description='Direct VLM only; no legacy Method adapter')
    p.add_argument('--backend',choices=['procthor','behavior'],required=True)
    p.add_argument('--bank-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--agent-config',type=Path)
    p.add_argument('--api-config',type=Path)
    p.add_argument('--task-id',action='append',default=[])
    p.add_argument('--limit',type=int,default=1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--max-steps',type=int,help='Override default 24/36/48; omit for normal evaluation')
    p.add_argument('--reuse-layout',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--dry-run',action='store_true',help='Print the resolved command without loading a model/simulator')
    a=p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError('Choose a new output directory')
    for name in ['tasks_with_qa.json','binding.json','oracle_reference.json']:
        if not (a.bank_dir/name).is_file():raise FileNotFoundError(a.bank_dir/name)
    if a.limit<1 or (a.max_steps is not None and a.max_steps<1):
        raise ValueError('limit and max-steps must be positive')
    import yaml
    if a.agent_config:
        document=yaml.safe_load(a.agent_config.read_text())
    else:
        from scoper.resources import package_root
        default=package_root()/'benchmark/procthor/configs/vlm_qwen35_9b_single_frame_standard.yaml'
        document=yaml.safe_load(default.read_text())
        model=a.model or config.get('model')
        if not model:raise ValueError('Provide --model, --agent-config, or model in the paths config')
        document['adapter']['model']=str(Path(model).resolve())
    document.setdefault('metadata',{})['log_privileged_metadata']=False
    document.setdefault('observation', {}).update(mode='multi_frame', num_frames=8)
    module=importlib.import_module('scoper.workflows.' + a.backend).EVALUATION_MODULE
    command=['--task-bank',str(a.bank_dir/'tasks_with_qa.json'),'--binding-sidecar',str(a.bank_dir/'binding.json'),
             '--oracle-reference',str(a.bank_dir/'oracle_reference.json'),'--agent-kind','vlm_api' if a.api_config else 'vlm_local',
             '--agent-config',str(a.output/'agent.yaml'),'--evaluation-protocol','standard_vlm',
             '--observation-mode','multi_frame','--output-jsonl',str(a.output/'results.jsonl'),
             '--diagnostic-output-jsonl',str(a.output/'diagnostics.jsonl'),
             '--run-id',a.output.name,'--seed',str(a.seed),'--limit',str(a.limit)]
    for task in a.task_id:command+=['--task-id',task]
    if a.max_steps is not None:command+=['--max-high-level-steps',str(a.max_steps)]
    if a.api_config:command+=['--api-config',str(a.api_config)]
    if a.backend=='procthor':command+=['--gpu-device',str(a.gpu),'--manipulathor-visual-attach']
    elif a.reuse_layout:command+=['--reuse-layout-session']
    if a.dry_run:
        print(json.dumps({'module':module,'arguments':command},indent=2));return 0
    a.output.mkdir(parents=True)
    (a.output/'agent.yaml').write_text(yaml.safe_dump(document))
    if a.backend=='behavior':
        os.environ.setdefault('EB_BEHAVIOR_GPU','1');os.environ.setdefault('OMNIGIBSON_HEADLESS','1')
    return importlib.import_module(module).main(command)


def summarize(argv):
    p=argparse.ArgumentParser(prog='scoper summarize')
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError(a.output)
    raw=json.loads(a.bank.read_text());tasks=raw if isinstance(raw,list) else raw['tasks']
    rows=[json.loads(x) for x in a.results.read_text().splitlines() if x.strip()]
    if len({(x.get('agent_id'),x.get('run_id')) for x in rows})>1:
        raise ValueError('Summarize one model/run at a time')
    from scoper.benchmark.common.evaluation_reporting import summarize_evaluation
    report=summarize_evaluation(rows,tasks);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(str(a.output));return 0


def doctor(argv,config):
    p=argparse.ArgumentParser(prog='scoper doctor',description='Check configured paths and installed modules without downloading or launching a simulator')
    p.add_argument('--backend',choices=['behavior','procthor'],required=True);a=p.parse_args(argv)
    from importlib.util import find_spec
    keys=['behavior_assets'] if a.backend=='behavior' else ['procthor_prior_home','procthor_executable']
    modules=['torch','transformers']+(['omnigibson','bddl','isaacsim'] if a.backend=='behavior' else ['ai2thor','prior'])
    checks={key:bool(config.get(key) and Path(config[key]).exists()) for key in keys}
    checks.update({name:find_spec(name) is not None for name in modules})
    print(json.dumps({'checks':checks,'gpu_rendering_tested':False},indent=2))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
