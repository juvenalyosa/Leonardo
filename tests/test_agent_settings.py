"""Assistant settings: defaults and migration of the old tool-call limit."""
import json

from leonardo.agent import config


def test_default_tool_call_limit_fits_pipelines(tmp_path):
    assert config.load_settings(str(tmp_path / "missing.json"))["max_tool_calls"] == 300


def test_old_default_limit_is_migrated_but_user_choices_are_kept(tmp_path):
    path = tmp_path / "ai_assistant.json"
    path.write_text(json.dumps({"max_tool_calls": 60}))
    assert config.load_settings(str(path))["max_tool_calls"] == 300
    path.write_text(json.dumps({"max_tool_calls": 120}))
    assert config.load_settings(str(path))["max_tool_calls"] == 120
