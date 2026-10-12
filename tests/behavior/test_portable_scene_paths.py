from pathlib import Path

def test_symbolic_and_renderer_share_external_scene_root(tmp_path,monkeypatch):
    from scoper.benchmark.behavior.scene.variant_layout import _resolve_source_json
    from scoper.benchmark.behavior.behavior_online_env import make_env_config
    monkeypatch.setenv('SCOPER_BEHAVIOR_ASSETS',str(tmp_path))
    source='datasets/behavior-1k-assets/scenes/example/json/example_best.json'
    expected=tmp_path/'scenes/example/json/example_best.json'
    task={'source':{'scene_model':'example','source_json':source}}
    assert _resolve_source_json(source)==expected
    assert Path(make_env_config(task,scene_layout_mode='source_json_full')['scene']['scene_file'])==expected
