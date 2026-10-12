"""Packaging checks; these deliberately do not claim simulator acceptance."""
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", [
    "scoper.benchmark.behavior.tools.run_lightweight_evaluation",
    "scoper.benchmark.behavior.tools.stage9_render_video_report",
    "scoper.benchmark.behavior.render.evaluation_session",
    "scoper.benchmark.procthor.lightweight_evaluation",
    "scoper.benchmark.procthor.oracle_tool_video_runner",
])
def test_real_entrypoints_are_in_package(name):
    from scoper.resources import package_root
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(package_root().resolve())


def test_renderer_template_default_uses_packaged_resources():
    from scoper.resources import package_root
    from scoper.benchmark.behavior.tools.visual_report import DEFAULT_PROCTHOR_TEMPLATE_DIR
    assert DEFAULT_PROCTHOR_TEMPLATE_DIR.is_relative_to(package_root())
    assert DEFAULT_PROCTHOR_TEMPLATE_DIR.is_dir()
    assert any(DEFAULT_PROCTHOR_TEMPLATE_DIR.glob("*.json"))
