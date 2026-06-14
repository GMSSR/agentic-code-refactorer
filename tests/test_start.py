import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pydantic import BaseModel, ValidationError

from src.start import (
    CheckpointError,
    ConfigError,
    _load,
    _parse_arg,
    _resume,
    start,
)


# =====================================================================
# Fixtures & Helpers
# =====================================================================
@pytest.fixture
def mock_constants():
    """Mock the constants module imported as 'c' in your_module."""
    with patch("src.start.c") as mock_c:
        mock_c.CONFIG_PATH = MagicMock(spec=Path)
        mock_c.HEURISTICS_PATH = MagicMock(spec=Path)
        mock_c.CHECKPOINT_PATH = MagicMock(spec=Path)
        yield mock_c


@pytest.fixture
def dummy_validation_error():
    """Generates a real Pydantic ValidationError for testing."""

    class DummyModel(BaseModel):
        val: int

    with pytest.raises(ValidationError) as exc_info:
        DummyModel(val="not_an_int")  # type: ignore
    return exc_info.value


# =====================================================================
# Tests for _parse_arg
# =====================================================================
class TestParseArg:
    @patch("src.start.Path.is_file", return_value=True)
    @patch("sys.argv", ["script.py", "path/to/code.py"])
    def test_parse_arg_fresh_success(self, mock_is_file):
        args = _parse_arg()
        assert args.code_path == "path/to/code.py"
        assert not args.resume
        assert not args.force

    @patch("sys.argv", ["script.py", "--resume"])
    def test_parse_arg_resume_success(self):
        args = _parse_arg()
        assert args.resume
        assert args.code_path is None

    @patch("sys.argv", ["script.py"])
    def test_parse_arg_missing_both_errors(self):
        with pytest.raises(SystemExit):
            _parse_arg()

    @patch("sys.argv", ["script.py", "--resume", "path/to/code.py"])
    def test_parse_arg_resume_with_path_errors(self):
        with pytest.raises(SystemExit):
            _parse_arg()

    @patch("sys.argv", ["script.py", "--resume", "--force"])
    def test_parse_arg_resume_with_force_errors(self):
        with pytest.raises(SystemExit):
            _parse_arg()

    @patch("sys.argv", ["script.py", "--force"])
    def test_parse_arg_force_without_path_errors(self):
        with pytest.raises(SystemExit):
            _parse_arg()

    @patch("src.start.Path.is_file", return_value=False)
    @patch("sys.argv", ["script.py", "invalid/path.py"])
    def test_parse_arg_file_not_found_errors(self, mock_is_file):
        with pytest.raises(SystemExit):
            _parse_arg()


# =====================================================================
# Tests for _load
# =====================================================================
class TestLoad:
    @patch("src.start.Setting.model_validate")
    def test_load_local_success(self, mock_validate, mock_constants):
        # Setup mocks
        mock_constants.CONFIG_PATH.open.return_value.__enter__.return_value = mock_open(
            read_data='{"config": true}'
        )()
        mock_constants.HEURISTICS_PATH.open.return_value.__enter__.return_value = (
            mock_open(read_data='{"heuristic": true}')()
        )

        mock_setting = MagicMock()
        mock_setting.local.eval_model = "local_eval"
        mock_setting.local.j_eval_model = "local_j_eval"
        mock_setting.local.ref_model = "local_ref"
        mock_setting.local.j_ref_model = "local_j_ref"
        mock_validate.return_value = mock_setting

        heuristics, em, jem, rm, jrm = _load(local=True)

        assert heuristics == {"heuristic": True}
        assert em == "local_eval"
        assert jem == "local_j_eval"
        assert rm == "local_ref"
        assert jrm == "local_j_ref"

    @patch("src.start.Setting.model_validate")
    @patch.dict(os.environ, {"MISTRAL_API_KEY": "secret1", "GEMINI_API_KEY": "secret2"})
    def test_load_cloud_success(self, mock_validate, mock_constants):
        mock_constants.CONFIG_PATH.open.return_value.__enter__.return_value = mock_open(
            read_data="{}"
        )()
        mock_constants.HEURISTICS_PATH.open.return_value.__enter__.return_value = (
            mock_open(read_data="{}")()
        )

        mock_setting = MagicMock()
        mock_setting.cloud.eval_model = "cloud_eval"
        mock_setting.cloud.j_eval_model = "cloud_j_eval"
        mock_setting.cloud.ref_model = "cloud_ref"
        mock_setting.cloud.j_ref_model = "cloud_j_ref"
        mock_validate.return_value = mock_setting

        _, em, _, _, _ = _load(local=False)
        assert em == "cloud_eval"

    def test_load_config_file_not_found(self, mock_constants):
        mock_constants.CONFIG_PATH.open.side_effect = FileNotFoundError()
        with pytest.raises(ConfigError, match="Config file not found"):
            _load(local=True)

    def test_load_config_json_decode_error(self, mock_constants):
        mock_constants.CONFIG_PATH.open.return_value.__enter__.return_value = mock_open(
            read_data="{invalid json"
        )()
        with pytest.raises(ConfigError, match="Failed to decode JSON"):
            _load(local=True)

    def test_load_heuristics_file_not_found(self, mock_constants):
        mock_constants.CONFIG_PATH.open.return_value.__enter__.return_value = mock_open(
            read_data="{}"
        )()
        mock_constants.HEURISTICS_PATH.open.side_effect = FileNotFoundError()
        with pytest.raises(ConfigError, match="Heuristic File not found"):
            _load(local=True)

    @patch("src.start.Setting.model_validate")
    def test_load_validation_error(
        self, mock_validate, mock_constants, dummy_validation_error
    ):
        mock_constants.CONFIG_PATH.open.return_value.__enter__.return_value = mock_open(
            read_data="{}"
        )()
        mock_constants.HEURISTICS_PATH.open.return_value.__enter__.return_value = (
            mock_open(read_data="{}")()
        )
        mock_validate.side_effect = dummy_validation_error

        with pytest.raises(ConfigError, match="Configuration file failed validation"):
            _load(local=True)

    @patch("src.start.Setting.model_validate")
    @patch.dict(os.environ, {}, clear=True)
    def test_load_cloud_missing_api_keys(self, mock_validate, mock_constants):
        mock_constants.CONFIG_PATH.open.return_value.__enter__.return_value = mock_open(
            read_data="{}"
        )()
        mock_constants.HEURISTICS_PATH.open.return_value.__enter__.return_value = (
            mock_open(read_data="{}")()
        )
        mock_validate.return_value = MagicMock()

        with pytest.raises(ConfigError, match="Error getting the API Keys"):
            _load(local=False)


# =====================================================================
# Tests for _resume
# =====================================================================
class TestResume:
    def test_resume_success(self, mock_constants):
        mock_constants.CHECKPOINT_PATH.is_file.return_value = True
        checkpoint_mock_data = {
            "code_path": "src/main.py",
            "current_stage": "refactoring",
            "approved_eval": [{"id": 1}],
            "rejected_eval": [],
            "approved_proposal": [],
            "rejected_proposal": [],
            "output": [],
        }
        mock_constants.CHECKPOINT_PATH.open.return_value.__enter__.return_value = (
            mock_open(read_data=json.dumps(checkpoint_mock_data))()
        )

        code_path, stage, app_ev, _, _, _, _ = _resume()
        assert code_path == Path("src/main.py")
        assert stage == "refactoring"
        assert app_ev == [{"id": 1}]

    def test_resume_no_checkpoint_file(self, mock_constants):
        mock_constants.CHECKPOINT_PATH.is_file.return_value = False
        with pytest.raises(CheckpointError, match="No active checkpoint found"):
            _resume()

    def test_resume_corrupt_file_error(self, mock_constants):
        mock_constants.CHECKPOINT_PATH.is_file.return_value = True
        mock_constants.CHECKPOINT_PATH.open.side_effect = Exception("Read error")
        with pytest.raises(CheckpointError, match="Error reading checkpoint file"):
            _resume()

    def test_resume_missing_code_path(self, mock_constants):
        mock_constants.CHECKPOINT_PATH.is_file.return_value = True
        mock_constants.CHECKPOINT_PATH.open.return_value.__enter__.return_value = (
            mock_open(read_data=json.dumps({"current_stage": "static_analysis"}))()
        )
        with pytest.raises(CheckpointError, match="missing the 'code_path' key"):
            _resume()


# =====================================================================
# Tests for start
# =====================================================================
class TestStart:
    @patch("src.start._parse_arg")
    @patch("src.start._load")
    @patch("src.start.AppConfig")
    @patch("src.start.PipelineState")
    @patch("src.start.Container")
    def test_start_fresh_run(
        self,
        mock_container,
        mock_state,
        mock_config,
        mock_load,
        mock_parse,
        mock_constants,
    ):
        # Configure arguments
        args = MagicMock()
        args.force = False
        args.local = True
        args.resume = False
        args.eval = True
        args.code_path = "test.py"
        mock_parse.return_value = args

        # Configure load output
        mock_load.return_value = ("heuristics", "em", "jem", "rm", "jrm")

        start()

        # Check if right configuration properties were processed
        mock_config.assert_called_once_with(
            is_eval_only=True,
            is_local=True,
            code_path=Path("test.py"),
            start_stage="static_analysis",
            eval_model="em",
            j_eval_model="jem",
            ref_model="rm",
            j_ref_model="jrm",
            heuristic_data="heuristics",
        )
        mock_state.assert_called_once_with(
            approved_eval=[],
            rejected_eval=[],
            approved_proposal=[],
            rejected_proposal=[],
            output=[],
        )
        mock_container.assert_called_once()

    @patch("src.start._parse_arg")
    @patch("src.start._load")
    @patch("src.start._resume")
    @patch("src.start.AppConfig")
    @patch("src.start.PipelineState")
    @patch("src.start.Container")
    def test_start_resume_run(
        self,
        mock_container,
        mock_state,
        mock_config,
        mock_resume,
        mock_load,
        mock_parse,
        mock_constants,
    ):
        args = MagicMock()
        args.force = False
        args.local = False
        args.resume = True
        args.eval = False
        mock_parse.return_value = args

        mock_load.return_value = ("heuristics", "em", "jem", "rm", "jrm")
        mock_resume.return_value = (
            Path("resumed.py"),
            "stage2",
            [1],
            [2],
            [3],
            [4],
            [5],
        )

        start()

        mock_resume.assert_called_once()
        mock_config.assert_called_once_with(
            is_eval_only=False,
            is_local=False,
            code_path=Path("resumed.py"),
            start_stage="stage2",
            eval_model="em",
            j_eval_model="jem",
            ref_model="rm",
            j_ref_model="jrm",
            heuristic_data="heuristics",
        )
        mock_state.assert_called_once_with(
            approved_eval=[1],
            rejected_eval=[2],
            approved_proposal=[3],
            rejected_proposal=[4],
            output=[5],
        )

    @patch("src.start._parse_arg")
    @patch("src.start._load")
    def test_start_force_deletes_checkpoint(
        self, mock_load, mock_parse, mock_constants
    ):
        args = MagicMock()
        args.force = True
        args.resume = False
        args.code_path = "test.py"
        mock_parse.return_value = args

        mock_constants.CHECKPOINT_PATH.exists.return_value = True
        mock_load.return_value = ("heuristics", "em", "jem", "rm", "jrm")

        start()

        mock_constants.CHECKPOINT_PATH.unlink.assert_called_once_with(missing_ok=True)
