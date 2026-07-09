import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import constants
import src.llm
import src.static_ana
import src.utils
from src.schemas import Evaluation, JudgementE
from src.start import _parse_arg, start


class TestDirectModeCLI:
    @patch("src.start.Path.is_file", return_value=True)
    @patch("sys.argv", ["script.py", "path/to/code.java", "--direct"])
    def test_parse_arg_direct_success(self, mock_is_file):
        args = _parse_arg()
        assert args.direct
        assert args.code_path == "path/to/code.java"
        assert not args.resume

    @patch("sys.argv", ["script.py", "--resume", "--direct"])
    def test_parse_arg_resume_and_direct_error(self):
        with pytest.raises(SystemExit):
            _parse_arg()

    @patch("sys.argv", ["script.py", "--direct"])
    def test_parse_arg_direct_without_path_error(self):
        with pytest.raises(SystemExit):
            _parse_arg()


class TestDirectModeStart:
    @patch("src.start._parse_arg")
    @patch("src.start._load")
    @patch("src.start.AppConfig")
    @patch("src.start.PipelineState")
    @patch("src.start.Container")
    def test_start_configures_is_direct_and_eval_only(
        self,
        mock_container,
        mock_state,
        mock_config,
        mock_load,
        mock_parse,
    ):
        args = MagicMock()
        args.force = False
        args.local = False
        args.resume = False
        args.eval = False
        args.direct = True
        args.code_path = "test.java"
        mock_parse.return_value = args

        mock_load.return_value = ("heuristics", "em", "jem", "rm", "jrm")

        start()

        mock_config.assert_called_once_with(
            is_eval_only=True,
            is_local=False,
            is_direct=True,
            code_path=Path("test.java"),
            start_stage="static_analysis",
            eval_model="em",
            j_eval_model="jem",
            ref_model="rm",
            j_ref_model="jrm",
            heuristic_data="heuristics",
        )


@pytest.fixture
def direct_integration_setup(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    heuristics_file = tmp_path / "heuristics.json"
    checkpoint_file = tmp_path / "checkpoint.json"
    log_file = tmp_path / "log.json"
    dummy_code = tmp_path / "DummyClass.java"

    config_file.write_text(
        json.dumps(
            {
                "local": {
                    "eval_model": "ollama/mistral",
                    "j_eval_model": "ollama/gemma",
                    "ref_model": "ollama/mistral",
                    "j_ref_model": "ollama/gemma",
                },
                "cloud": {
                    "eval_model": "mistral/mistral",
                    "j_eval_model": "gemini/gemini",
                    "ref_model": "mistral/mistral",
                    "j_ref_model": "gemini/gemini",
                },
            }
        )
    )

    heuristics_file.write_text(
        json.dumps(
            {
                "God Class": {"H1": "Is it a God Class?"},
                "Feature Envy": {"H1": "Feature Envy check?"},
            }
        )
    )

    dummy_code.write_text("public class DummyClass {}")

    monkeypatch.setattr(constants, "CONFIG_PATH", config_file)
    monkeypatch.setattr(constants, "HEURISTICS_PATH", heuristics_file)
    monkeypatch.setattr(constants, "CHECKPOINT_PATH", checkpoint_file)
    monkeypatch.setattr(constants, "LOG_PATH", log_file)
    monkeypatch.setattr(constants, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(src.utils, "CHECKPOINT_PATH", checkpoint_file)

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    return {
        "dummy_code": dummy_code,
        "results_file": data_dir / "results_DummyClass.json",
    }


def mock_direct_unified_call(prompt, model, schema):
    if schema == Evaluation:
        return {
            "heuristics": [
                {
                    "name": "Design Rule",
                    "evaluation": "It looks okay.",
                    "conclusion": "met",
                }
            ],
            "summary": "direct check done",
            "justification": "looks clean",
            "status": "rejected",
        }
    elif schema == JudgementE:
        return {
            "rubrics": {f"R{i}": "ok" for i in range(1, 7)},
            "critical_flaw": "None",
            "verdict": "approved",
            "feedback": "",
        }
    return {}


@patch("src.static_ana.static")
@patch("src.llm.unified_call", side_effect=mock_direct_unified_call)
def test_direct_mode_integration_flow(
    mock_call,
    mock_static,
    direct_integration_setup,
    monkeypatch,
):
    import runpy

    # Verify that integration execution argv runs in direct mode
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--direct", str(direct_integration_setup["dummy_code"])],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_path("main.py", run_name="__main__")

    assert exc.value.code == 0
    # verify static analysis was NOT called
    mock_static.assert_not_called()

    assert direct_integration_setup["results_file"].exists()

    with open(direct_integration_setup["results_file"], encoding="utf-8") as f:
        results = json.load(f)

    # Since we have "God Class" and "Feature Envy" in our mock heuristics file,
    # Direct Mode evaluates those two types directly using the file contents.
    assert len(results) == 2
    assert results[0]["smell_type"] in ("God Class", "Feature Envy")
    assert results[1]["smell_type"] in ("God Class", "Feature Envy")
    # All of them should only have evaluation (refactoring should be None/skipped)
    assert results[0]["evaluation"] is not None
    assert results[0]["proposal"] is None
    assert results[1]["evaluation"] is not None
    assert results[1]["proposal"] is None
