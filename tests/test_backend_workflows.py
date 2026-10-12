import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('backend', ['procthor', 'behavior'])
@pytest.mark.parametrize('command', ['generate', 'oracle', 'evaluate', 'doctor', 'check-bank', 'summarize'])
def test_backend_dispatch(backend, command, monkeypatch):
    from scoper import cli
    dispatch = Mock(return_value=0)
    monkeypatch.setattr(cli, 'main', dispatch)
    main = importlib.import_module('scoper.' + backend).main
    assert main(['--config', 'paths.yaml', command, '--help']) == 0
    expected = ['--config', 'paths.yaml', command]
    if command in {'generate', 'oracle', 'evaluate', 'doctor'}:
        expected += ['--backend', backend]
    dispatch.assert_called_once_with(expected + ['--help'])


@pytest.mark.parametrize('backend', ['procthor', 'behavior'])
def test_backend_cannot_be_overridden(backend):
    main = importlib.import_module('scoper.' + backend).main
    with pytest.raises(SystemExit) as exc:
        main(['evaluate', '--backend=other'])
    assert exc.value.code == 2


@pytest.mark.parametrize('backend', ['procthor', 'behavior'])
def test_entrypoint_dry_run_without_loading_simulators(backend, tmp_path):
    script = '''
import importlib,json,sys
backend,bank,out=sys.argv[1:]
importlib.import_module('scoper.'+backend).main(['evaluate','--bank-dir',bank,'--output',out,'--model','/unused/model','--dry-run'])
assert not any(m.split('.')[0] in {'omnigibson','isaacsim','ai2thor','torch'} for m in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', script, backend,
                             str(ROOT / 'task_bank' / backend), str(tmp_path / 'run')],
                            capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert backend in resolved['module']
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('backend', ['procthor', 'behavior'])
def test_oracle_routes_to_selected_workflow(backend, tmp_path, monkeypatch):
    from scoper.commands.oracle import main
    workflow = importlib.import_module('scoper.workflows.' + backend)
    run = Mock(return_value={'checked': backend})
    admission = Mock(return_value=True)
    monkeypatch.setattr(workflow, 'oracle', run)
    monkeypatch.setattr(workflow, 'oracle_passed', admission)
    bank = tmp_path / 'bank.json'
    task = {'id': 'test-task'}
    bank.write_text(json.dumps({'tasks': [task]}))
    main(['--backend', backend, '--bank', str(bank), '--task-id', 'test-task',
          '--output', str(tmp_path / 'out')])
    assert run.call_args.args[1] == task
    assert admission.call_args.args[1] == {'checked': backend}


@pytest.mark.parametrize('backend', ['procthor', 'behavior'])
def test_generation_routes_and_saves_admission(backend, tmp_path, monkeypatch):
    from scoper.commands.generate import main
    workflow = importlib.import_module('scoper.workflows.' + backend)
    tasks = [{'id': 'candidate'}]
    generated = {'tasks': tasks}
    admitted = tasks if backend == 'behavior' else []
    run = Mock(return_value=({'layout': 'test'}, generated, tasks, admitted))
    monkeypatch.setattr(workflow, 'generate', run)
    out = tmp_path / 'out'
    main(['--backend', backend, '--layout', 'test', '--output', str(out)])
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['candidate_count'] == 1
    assert summary['symbolically_admitted'] == len(admitted)
    assert summary['real_oracle_admission'] == 'not_run'


def test_oracle_admission_still_requires_rendering(tmp_path):
    from scoper.workflows import behavior, procthor
    args = SimpleNamespace(output=tmp_path)
    assert not procthor.oracle_passed(args, ({}, {'num_passed': 1, 'total_video_frames': 0}))
    assert not behavior.oracle_passed(args, {'task_success': 1, 'render_status': 'passed', 'video_quality_status': 'passed'})
    (tmp_path/'execution.manifest.json').write_text('{}')
    assert behavior.oracle_passed(args, {'task_success': 1, 'render_status': 'passed', 'video_quality_status': 'passed'})
