import json
from pathlib import Path
import pytest
from scoper import cli

ROOT=Path(__file__).resolve().parents[1]

def test_config_relative_paths(tmp_path,monkeypatch):
    monkeypatch.delenv('EB_PROCTHOR_PRIOR_OFFLINE',raising=False)
    monkeypatch.delenv('SCOPER_RESOURCE_ROOT',raising=False)
    p=tmp_path/'paths.yaml';p.write_text('resource_root: data\n')
    assert cli.configure(p)['resource_root']==str(tmp_path/'data')
    assert cli.os.environ['SCOPER_RESOURCE_ROOT']==str(tmp_path/'data')
    assert cli.os.environ['EB_PROCTHOR_PRIOR_OFFLINE']=='1'

@pytest.mark.parametrize('content',['unexpected: value','model: ${SCOPER_UNDEFINED_TEST_VALUE}','model: 4'])
def test_config_rejects_invalid(tmp_path,content):
    p=tmp_path/'paths.yaml';p.write_text(content)
    with pytest.raises(ValueError):cli.configure(p)

@pytest.mark.parametrize('backend',['procthor','behavior'])
def test_evaluation_dry_run(tmp_path,capsys,backend):
    out=tmp_path/'run'
    cli.evaluate(['--backend',backend,'--bank-dir',str(ROOT/'task_bank'/backend),
                  '--model',str(tmp_path/'model'),'--output',str(out),'--dry-run'],{})
    record=json.loads(capsys.readouterr().out)
    assert 'standard_vlm' in record['arguments']
    assert 'multi_frame' in record['arguments']
    assert '--max-high-level-steps' not in record['arguments']
    assert ('--reuse-layout-session' in record['arguments']) == (backend=='behavior')
    assert not out.exists()

def test_never_overwrites_output(tmp_path):
    with pytest.raises(FileExistsError):
        cli.evaluate(['--backend','procthor','--bank-dir',str(ROOT/'task_bank/procthor'),
                      '--output',str(tmp_path),'--model','unused'],{})

@pytest.mark.parametrize('command',['generate','oracle','check-bank','evaluate','summarize','doctor'])
def test_command_help(command):
    with pytest.raises(SystemExit) as e:cli.main([command,'--help'])
    assert e.value.code==0
