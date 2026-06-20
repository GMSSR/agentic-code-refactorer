import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from langchain_text_splitters import Language

from src.language import (
    _clang_snippet,
    _clang_tool,
    _clippy_parser,
    _clippy_tool,
    _eslint_tool,
    _find_binary,
    _find_project_root,
    _is_file,
    _issue_snippet,
    _pylint_tool,
    _sonar_tool,
    _swift_tool,
    get_langchain_enum,
    get_language,
    get_tool,
)
from src.schemas import Candidate, SmellCode

# ==========================================
# 1. Tests for get_language
# ==========================================


def test_get_language():
    result = get_language(Path("/mock/main.py"))

    assert result == "python"

    result = get_language(Path("/mock/main.c"))

    assert result == "c"

    result = get_language(Path("/mock/main.h"))

    assert result == "cpp"

    with pytest.raises(ValueError, match=r"Error: The file language couldn't be determined."):
        get_language(Path("/mock/main.R"))


# ==========================================
# 2. Tests for get_tool
# ==========================================


def test_get_langchain_enum():
    result = get_langchain_enum("python")

    assert result == Language.PYTHON

    result = get_langchain_enum("json")

    assert result is None


# ==========================================
# 3. Tests for get_langchain_enum
# ==========================================


@patch("src.language.get_language")
@patch("src.language._pylint_tool")
def test_get_tool(mock_pylint, mock_language):
    mock_pylint.return_value = []
    mock_language.return_value = "python"
    file = Path("/mock/main.py")
    result = get_tool(file)

    assert result == mock_pylint.return_value
    mock_language.assert_called_once_with(file)
    mock_pylint.assert_called_once_with(file)

    mock_language.return_value = "r"
    file = Path("/mock/main.R")
    with pytest.raises(ValueError, match=f"No static analysis tool assigned to {mock_language.return_value}."):
        get_tool(file)


# ==========================================
# 4. Tests for _clang_tool
# ==========================================


@patch("src.language._find_binary")
@patch("src.language.subprocess.run")
@patch("src.language._clang_snippet")
@patch("src.language._is_file")
def test__clang_tool(mock_file, mock_snippet, mock_subprocess, mock_binary, tmp_path):
    mock_sat = tmp_path / "data" / "sat.json"

    with patch("src.language.SAT_PATH", mock_sat):
        mock_file.return_value = False
        mock_binary.return_value = tmp_path / "bin" / "clang"
        file = tmp_path / "src" / "main.h"
        mock_binary.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.parent.mkdir(parents=True, exist_ok=True)
        file.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.write_text("old results")
        mock_snippet.return_value = (1, "Test")

        def mock_sub(*args, **kwargs):
            assert not mock_sat.exists()
            mock_sat.write_text(
                textwrap.dedent(
                    """
                    ---
                    MainSourceFile:  'src/main.cpp'
                    Diagnostics:
                      - DiagnosticName:  'modernize-use-nullptr'
                        DiagnosticMessage:
                          Message:         'use nullptr'
                          FilePath:        'src/main.cpp'
                          FileOffset:      42
                          Replacements:
                            - FilePath:        'src/main.cpp'
                              Offset:          42
                              Length:          4
                              ReplacementText: 'nullptr'
                    ...
                    """
                ).strip()
            )
            return subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"], returncode=0, stdout="main\n", stderr=""
            )

        mock_subprocess.side_effect = mock_sub
        result = _clang_tool(file)

        assert result == []

        mock_file.return_value = True
        result = _clang_tool(file)

        assert result == [
            Candidate(
                smell_type="modernize-use-nullptr",
                smell=SmellCode(
                    file_name="main.h",
                    line=1,
                    snippet="Test",
                    description="use nullptr",
                    context="",
                ),
            )
        ]

        mock_binary.return_value = None
        result = _clang_tool(file)
        assert result == []

        mock_binary.return_value = tmp_path / "bin" / "clang"


@patch("src.language._find_binary")
@patch("src.language.subprocess.run")
@patch("src.language._clang_snippet")
@patch("src.language._is_file")
def test__clang_tool_yaml(mock_file, mock_snippet, mock_subprocess, mock_binary, tmp_path):
    mock_sat = tmp_path / "data" / "sat.json"

    with patch("src.language.SAT_PATH", mock_sat):
        mock_file.return_value = False
        mock_binary.return_value = tmp_path / "bin" / "clang"
        file = tmp_path / "src" / "main.h"
        mock_binary.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.parent.mkdir(parents=True, exist_ok=True)
        file.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.write_text("old results")
        mock_snippet.return_value = (1, "Test")

        def mock_sub(*args, **kwargs):
            if mock_sat.exists():
                mock_sat.unlink()
            mock_sat.write_text(
                textwrap.dedent(
                    """
                    ---
                    MainSourceFile:  'src/main.cpp'
                    Diagnostics:
                      - DiagnosticName:  'modernize-use-nullptr'
                        DiagnosticMessage:
                          Message:         'use nullptr'
                          FilePath:        'src/main.cpp'
                          FileOffset:      42
                          Replacements:
                            - FilePath:        'src/main.cpp'
                              Offset:          42
                              Length:          4
                              ReplacementText: 'nullptr'
                    ...
                    """
                ).strip()
            )
            return subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"], returncode=1, stdout="main\n", stderr=""
            )

        mock_subprocess.side_effect = mock_sub
        result = _clang_tool(file)
        assert result == []


@patch("src.language._find_binary")
@patch("src.language.subprocess.run")
@patch("src.language._clang_snippet")
@patch("src.language._is_file")
def test__clang_tool_diag(mock_file, mock_snippet, mock_subprocess, mock_binary, tmp_path):
    mock_sat = tmp_path / "data" / "sat.json"

    with patch("src.language.SAT_PATH", mock_sat):
        mock_file.return_value = False
        mock_binary.return_value = tmp_path / "bin" / "clang"
        file = tmp_path / "src" / "main.h"
        mock_binary.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.parent.mkdir(parents=True, exist_ok=True)
        file.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.write_text("old results")
        mock_snippet.return_value = (1, "Test")

        def mock_sub(*args, **kwargs):
            mock_sat.write_text("MainSourceFile: 'src/main.cpp'")
            return subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"], returncode=0, stdout="main\n", stderr=""
            )

        mock_subprocess.side_effect = mock_sub
        result = _clang_tool(file)
        assert result == []


@patch("src.language._find_binary")
@patch("src.language.subprocess.run")
@patch("src.language._clang_snippet")
@patch("src.language._is_file")
def test__clang_tool_message(mock_file, mock_snippet, mock_subprocess, mock_binary, tmp_path):
    mock_sat = tmp_path / "data" / "sat.json"

    with patch("src.language.SAT_PATH", mock_sat):
        mock_file.return_value = False
        mock_binary.return_value = tmp_path / "bin" / "clang"
        file = tmp_path / "src" / "main.h"
        mock_binary.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.parent.mkdir(parents=True, exist_ok=True)
        file.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.write_text("old results")
        mock_snippet.return_value = (1, "Test")

        def mock_sub(*args, **kwargs):
            if mock_sat.exists():
                mock_sat.unlink()
            mock_sat.write_text(
                textwrap.dedent(
                    """
                    ---
                    MainSourceFile:  'src/main.cpp'
                    Diagnostics:
                      - DiagnosticName:  'modernize-use-nullptr'
                    ...
                    """
                ).strip()
            )
            return subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"], returncode=0, stdout="main\n", stderr=""
            )

        mock_subprocess.side_effect = mock_sub
        result = _clang_tool(file)
        assert result == []


@patch("src.language._find_binary")
@patch("src.language.subprocess.run")
@patch("src.language._clang_snippet")
@patch("src.language._is_file")
def test__clang_tool_offset(mock_file, mock_snippet, mock_subprocess, mock_binary, tmp_path):
    mock_sat = tmp_path / "data" / "sat.json"

    with patch("src.language.SAT_PATH", mock_sat):
        mock_file.return_value = False
        mock_binary.return_value = tmp_path / "bin" / "clang"
        file = tmp_path / "src" / "main.h"
        mock_binary.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.parent.mkdir(parents=True, exist_ok=True)
        file.parent.mkdir(parents=True, exist_ok=True)
        mock_sat.write_text("old results")
        mock_snippet.return_value = (1, "Test")

        def mock_sub(*args, **kwargs):
            if mock_sat.exists():
                mock_sat.unlink()
            mock_sat.write_text(
                textwrap.dedent(
                    """
                    ---
                    MainSourceFile:  'src/main.cpp'
                    Diagnostics:
                      - DiagnosticName:  'modernize-use-nullptr'
                        DiagnosticMessage:
                          Message:         'use nullptr'
                          FilePath:        'src/main.cpp'
                          FileOffset:      42
                          Replacements:
                            - FilePath:        'src/main.cpp'
                              Length:          4
                              ReplacementText: 'nullptr'
                    ...
                    """
                ).strip()
            )
            return subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"], returncode=0, stdout="main\n", stderr=""
            )

        mock_subprocess.side_effect = mock_sub
        result = _clang_tool(file)
        assert result == []


# ==========================================
# 5. Tests for _is_file
# ==========================================


def test__is_file(tmp_path):
    code_path = tmp_path / "main.py"
    diag_path = None
    results = _is_file(diag_path, code_path)
    assert not results
    diag_path = tmp_path / "hello.h"
    results = _is_file(diag_path, code_path)
    assert not results
    diag_path = tmp_path / "main.py"
    results = _is_file(diag_path, code_path)
    assert results


# ==========================================
# 6. Tests for _clang_snippet
# ==========================================


def test__clang_snippet(tmp_path):
    # Test valid snippet retrieval
    file_path = tmp_path / "test.cpp"
    file_path.write_bytes(b"line1\nline2\nline3")

    # Offset 0 is "line1"
    line, content = _clang_snippet(file_path, 0)
    assert line == 1
    assert content == "line1"

    # Offset 6 (after \n) is "line2"
    line, content = _clang_snippet(file_path, 6)
    assert line == 2
    assert content == "line2"

    # Test out of bounds offset
    line, content = _clang_snippet(file_path, 100)
    assert line == 0
    assert content == "[Offset out of bounds]"

    line, content = _clang_snippet(file_path, -1)
    assert line == 0
    assert content == "[Offset out of bounds]"

    # Test exception handling (directory path raises OSError on read)
    line, content = _clang_snippet(tmp_path, 0)
    assert line == 0
    assert content == "[Error reading file]"


# ==========================================
# 7. Tests for _clippy_tool
# ==========================================


@patch("src.language._sonar_tool")
@patch("src.language._clippy_parser")
@patch("src.language.subprocess.run")
@patch("src.language._find_project_root")
@patch("src.language._find_binary")
def test__clippy_tool(
    mock_find_binary,
    mock_find_root,
    mock_run,
    mock_parser,
    mock_sonar,
    tmp_path,
):
    code_path = tmp_path / "main.rs"

    # Scenario A: cargo or project_root not found -> fallback to sonar
    mock_find_binary.return_value = None
    mock_find_root.return_value = None
    _clippy_tool(code_path)
    mock_sonar.assert_called_once_with(code_path)

    # Reset mocks
    mock_sonar.reset_mock()
    mock_find_binary.return_value = Path("/mock/cargo")
    mock_find_root.return_value = tmp_path

    # Scenario B: normal execution path
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="line1\nline2",
        stderr="",
    )
    mock_parser.side_effect = lambda line, code_path, project_root: (
        Candidate(
            smell_type="test-type",
            smell=SmellCode(file_name="main.rs", line=1, snippet="", description=line),
        )
        if line == "line1"
        else None
    )

    result = _clippy_tool(code_path)
    assert len(result) == 1
    assert result[0].smell.description == "line1"

    # Scenario C: FileNotFoundError raised by subprocess.run
    mock_run.side_effect = FileNotFoundError()
    _clippy_tool(code_path)
    mock_sonar.assert_called_once_with(code_path)


# ==========================================
# 8. Tests for _clippy_parser
# ==========================================


@patch("src.language._is_file")
@patch("src.language._issue_snippet")
def test__clippy_parser(mock_issue_snippet, mock_is_file, tmp_path):

    project_root = tmp_path
    code_path = tmp_path / "src" / "main.rs"

    # Empty/Invalid JSON output
    assert _clippy_parser("", code_path, project_root) is None
    assert _clippy_parser("invalid-json", code_path, project_root) is None

    # Not compiler-message reason
    assert _clippy_parser('{"reason": "other"}', code_path, project_root) is None

    # Warning/Error compiler-message, no primary span
    msg_json = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "warning message",
            "spans": [],
        },
    }
    assert _clippy_parser(json.dumps(msg_json), code_path, project_root) is None

    # Valid compiler-message
    msg_json_valid = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "clippy warning",
            "code": {"code": "clippy::style_lint"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/main.rs",
                    "line_start": 5,
                    "line_end": 10,
                }
            ],
        },
    }
    mock_is_file.return_value = True
    mock_issue_snippet.return_value = "fn main() {}"

    result = _clippy_parser(json.dumps(msg_json_valid), code_path, project_root)
    assert result is not None
    assert result.smell_type == "clippy::style_lint"
    assert result.smell.file_name == str(project_root / "src/main.rs")
    assert result.smell.line == 5
    assert result.smell.snippet == "fn main() {}"
    assert result.smell.description == "clippy warning"


# ==========================================
# 9. Tests for _pylint_tool
# ==========================================


@patch("src.language._issue_snippet")
@patch("src.language.PylintRun")
def test__pylint_tool(mock_pylint_run, mock_issue_snippet, tmp_path):

    mock_sat = tmp_path / "sat.json"
    code_path = tmp_path / "main.py"

    pylint_output = [
        {
            "symbol": "unused-variable",
            "line": 12,
            "endLine": 12,
            "message": "Unused variable 'x'",
        }
    ]

    with patch("src.language.SAT_PATH", mock_sat):
        # We simulate PylintRun writing results to SAT_PATH
        def simulate_run(*args, **kwargs):
            mock_sat.write_text(json.dumps(pylint_output))

        mock_pylint_run.side_effect = simulate_run
        mock_issue_snippet.return_value = "x = 42"

        result = _pylint_tool(code_path)

        assert len(result) == 1
        assert result[0].smell_type == "unused-variable"
        assert result[0].smell.file_name == "main.py"
        assert result[0].smell.line == 12
        assert result[0].smell.snippet == "x = 42"
        assert result[0].smell.description == "Unused variable 'x'"


# ==========================================
# 10. Tests for _issue_snippet
# ==========================================


def test__issue_snippet(tmp_path):
    file_path = tmp_path / "code.py"
    file_path.write_text("line1\nline2\nline3\nline4\n")

    result = _issue_snippet(file_path, 2, 3)
    assert result == "line2\nline3\n"

    result = _issue_snippet(file_path, 1, 1)
    assert result == "line1\n"


# ==========================================
# 11. Tests for _eslint_tool
# ==========================================


@patch("src.language._sonar_tool")
@patch("src.language._is_file")
@patch("src.language._issue_snippet")
@patch("src.language.subprocess.run")
@patch("src.language._find_project_root")
@patch("src.language._find_binary")
def test__eslint_tool(
    mock_find_binary,
    mock_find_root,
    mock_run,
    mock_issue_snippet,
    mock_is_file,
    mock_sonar,
    tmp_path,
):

    mock_sat = tmp_path / "sat.json"
    code_path = tmp_path / "main.js"

    with patch("src.language.SAT_PATH", mock_sat):
        # Scenario A: eslint not found -> fallback to sonar
        mock_find_binary.return_value = None
        _eslint_tool(code_path)
        mock_sonar.assert_called_once_with(code_path)

        # Scenario B: success flow
        mock_sonar.reset_mock()
        mock_find_binary.return_value = Path("/mock/eslint")
        mock_find_root.return_value = tmp_path

        eslint_output = [
            {
                "filePath": str(code_path),
                "messages": [
                    {
                        "ruleId": "no-unused-vars",
                        "line": 4,
                        "endLine": 4,
                        "message": "unused var",
                    }
                ],
            }
        ]

        def simulate_run(*args, **kwargs):
            mock_sat.write_text(json.dumps(eslint_output))

        mock_run.side_effect = simulate_run
        mock_is_file.return_value = True
        mock_issue_snippet.return_value = "const x = 5;"

        result = _eslint_tool(code_path)
        assert len(result) == 1
        assert result[0].smell_type == "no-unused-vars"
        assert result[0].smell.line == 4
        assert result[0].smell.snippet == "const x = 5;"


# ==========================================
# 12. Tests for _swift_tool
# ==========================================


@patch("src.language._is_file")
@patch("src.language._issue_snippet")
@patch("src.language.subprocess.run")
@patch("src.language._find_binary")
def test__swift_tool(mock_find_binary, mock_run, mock_issue_snippet, mock_is_file, tmp_path):

    mock_sat = tmp_path / "sat.json"
    code_path = tmp_path / "main.swift"

    with patch("src.language.SAT_PATH", mock_sat):
        # Scenario A: swiftlint not found
        mock_find_binary.return_value = None
        result = _swift_tool(code_path)
        assert result == []

        # Scenario B: success flow
        mock_find_binary.return_value = Path("/mock/swiftlint")
        swiftlint_output = [
            {
                "file": str(code_path),
                "line": 10,
                "reason": "force cast violation",
                "rule_id": "force_cast",
            }
        ]

        def simulate_run(*args, **kwargs):
            mock_sat.write_text(json.dumps(swiftlint_output))

        mock_run.side_effect = simulate_run
        mock_is_file.return_value = True
        mock_issue_snippet.return_value = "let x = y as! Int"

        result = _swift_tool(code_path)
        assert len(result) == 1
        assert result[0].smell_type == "force_cast"
        assert result[0].smell.line == 10
        assert result[0].smell.snippet == "let x = y as! Int"


# ==========================================
# 13. Tests for _sonar_tool
# ==========================================


@patch("src.language.SonarQubeClient")
@patch("src.language._issue_snippet")
@patch("src.language.subprocess.run")
@patch("src.language.shutil.which")
def test__sonar_tool(mock_which, mock_run, mock_issue_snippet, mock_client_cls, tmp_path):

    code_path = tmp_path / "main.py"

    # Scenario A: missing SONAR_TOKEN
    with patch.dict(os.environ, {}, clear=True):
        result = _sonar_tool(code_path)
        assert result == []

    # Scenario B: pysonar executable not found
    with patch.dict(os.environ, {"SONAR_TOKEN": "mock-token"}):
        mock_which.return_value = None
        result = _sonar_tool(code_path)
        assert result == []

        # Scenario C: pysonar found but fails
        mock_which.return_value = "/usr/bin/pysonar"
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="pysonar", stderr="failed")
        result = _sonar_tool(code_path)
        assert result == []

        # Scenario D: success flow
        mock_run.side_effect = None
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        mock_issue = MagicMock()
        mock_issue.line = 15
        mock_issue.text_range = MagicMock(start_line=15, end_line=15)
        mock_issue.message = "sonar-smell"

        mock_client = MagicMock()
        mock_client.issues.search.return_value.issues = [mock_issue]
        mock_client_cls.return_value = mock_client

        mock_issue_snippet.return_value = "def bad_code():"

        result = _sonar_tool(code_path)
        assert len(result) == 1
        assert result[0].smell_type == "sonar-smell"
        assert result[0].smell.line == 15
        assert result[0].smell.snippet == "def bad_code():"


# ==========================================
# 14. Tests for _find_project_root
# ==========================================


def test__find_project_root(tmp_path):
    code_path = tmp_path / "src" / "main.py"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    venv = tmp_path / ".venv"
    venv.mkdir(parents=True, exist_ok=True)
    result = _find_project_root(code_path, ".venv")
    assert result == tmp_path
    result = _find_project_root(code_path, "node")
    assert result is None


# ==========================================
# 15. Tests for _find_binary
# ==========================================


@patch("src.language.os.access")
@patch("src.language.Path.home")
@patch("src.language._find_project_root")
@patch("src.language.shutil.which")
def test__find_binary(mock_which, mock_find_root, mock_home, mock_access, tmp_path):

    code_path = tmp_path / "main.js"

    # Scenario A: found in PATH
    mock_which.return_value = "/usr/bin/eslint"
    result = _find_binary("eslint", code_path)
    assert result == Path("/usr/bin/eslint")

    # Scenario B: found in local node_modules/.bin
    mock_which.return_value = None
    mock_find_root.return_value = tmp_path
    local_bin = tmp_path / "node_modules" / ".bin" / "eslint"

    # Mock Path methods to pretend local file exists and is executable
    with patch.object(Path, "is_file", return_value=True):
        mock_access.return_value = True
        result = _find_binary("eslint", code_path)
        assert result is not None
        assert result.resolve() == local_bin.resolve()

    # Scenario C: found in cargo home bin
    mock_find_root.return_value = None
    mock_home.return_value = tmp_path
    cargo_bin = tmp_path / ".cargo" / "bin" / "cargo"

    with patch.object(Path, "is_file", return_value=True):
        mock_access.return_value = True
        result = _find_binary("cargo", code_path)
        assert result is not None
        assert result.resolve() == cargo_bin.resolve()

    # Scenario D: not found anywhere
    with patch.object(Path, "is_file", return_value=False):
        result = _find_binary("eslint", code_path)
        assert result is None
