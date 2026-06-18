import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import yaml
from langchain_text_splitters import Language
from pylint.lint import Run as PylintRun
from sonarqube import SonarQubeClient

from constants import SAT_PATH
from src.schemas import Candidate, SmellCode


def get_language(file_path: Path) -> str:
    """
    Identifies if a file is a source code file belonging to a specific
    list of languages based on its extension.
    """
    EXTENSION_MAP: dict[str, str] = {
        # C / CPP
        ".c": "c",
        ".h": "cpp",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        # C# & VB.NET
        ".cs": "csharp",
        ".vb": "visualbasic",
        # Web Core
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        # JavaScript & TypeScript
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        # Go & Rust
        ".go": "go",
        ".rs": "rust",
        # JVM Languages
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".sc": "scala",
        # Scripting & Backend
        ".php": "php",
        ".py": "python",
        ".pyw": "python",
        ".rb": "ruby",
        # Apple Ecosystem
        ".swift": "swift",
        # Infrastructure & Data
        ".tf": "terraform",
        ".tfvars": "terraform",
        ".xml": "xml",
        ".xsd": "xml",
    }

    file_extension = file_path.suffix.lower()

    lang_string = EXTENSION_MAP.get(file_extension)
    if lang_string is None:
        raise ValueError("Error: The file language couldn't be determined.")
    return lang_string


def get_langchain_enum(lang_string: str) -> Language | None:
    """Maps generic language strings to LangChain Enums."""
    langchain_map = {
        "c": Language.C,
        "cpp": Language.CPP,
        "csharp": Language.CSHARP,
        "html": Language.HTML,
        "javascript": Language.JS,
        "typescript": Language.TS,
        "go": Language.GO,
        "rust": Language.RUST,
        "java": Language.JAVA,
        "kotlin": Language.KOTLIN,
        "scala": Language.SCALA,
        "php": Language.PHP,
        "python": Language.PYTHON,
        "ruby": Language.RUBY,
        "swift": Language.SWIFT,
    }
    return langchain_map.get(lang_string)


def get_tool(code_path: Path) -> list[Candidate]:
    """Maps language strings to a static analysis tool."""
    tool_map: dict[str, Callable] = {
        "c": _clang_tool,
        "cpp": _clang_tool,
        "rust": _clippy_tool,
        "python": _pylint_tool,
        "swift": _swift_tool,
        "javascript": _eslint_tool,
        "typescript": _eslint_tool,
        "csharp": _sonar_tool,
        "html": _sonar_tool,
        "go": _sonar_tool,
        "java": _sonar_tool,
        "kotlin": _sonar_tool,
        "scala": _sonar_tool,
        "php": _sonar_tool,
        "ruby": _sonar_tool,
        "visualbasic": _sonar_tool,
        "css": _sonar_tool,
        "terraform": _sonar_tool,
        "xml": _sonar_tool,
    }
    lang_string = get_language(code_path)
    static_analysis_tool = tool_map.get(lang_string)
    if static_analysis_tool is None:
        raise ValueError(f"No static analysis tool assigned to {lang_string}.")
    return static_analysis_tool(code_path)


def _clang_tool(code_path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    SAT_PATH.unlink(missing_ok=True)
    try:
        clang_path = find_binary(tool="clang-tidy", code_path=code_path)
        if not clang_path:
            raise FileNotFoundError
        code = subprocess.run(  # noqa: S603
            [clang_path, code_path, f"--export-fixes={SAT_PATH}"], capture_output=True, text=True, check=False
        )
        if code.returncode != 0 and not SAT_PATH.exists():
            print(f"Clang-tidy failed with exit code {code.returncode}")
            print(f"Standard Error:\n{code.stderr}")
            return candidates
    except FileNotFoundError:
        print("Error: 'clang-tidy' executable not found in PATH.")
        return candidates

    try:
        with SAT_PATH.open() as file:
            output = yaml.safe_load(file)
    except (
        yaml.YAMLError,
        FileNotFoundError,
    ):
        return candidates

    if not output or "Diagnostics" not in output:
        return candidates

    output = output["Diagnostics"]
    for issue in output:
        if "DiagnosticMessage" not in issue:
            continue
        diagnostic = issue.get("DiagnosticMessage")
        if not _is_file(diagnostic.get("FilePath"), code_path):
            continue
        offset = diagnostic.get("FileOffset")
        if offset is not None:
            line, snippet = _clang_snippet(code_path, offset)
        else:
            line, snippet = 0, ""

        smell = SmellCode(
            file_name=str(code_path.name),
            line=line,
            snippet=snippet,
            description=str(diagnostic.get("Message")),
        )

        candidates.append(Candidate(smell_type=issue.get("DiagnosticName"), smell=smell))
    return candidates


def _is_file(diag_path: Path | None, code_path: Path):
    """Checks if the issue belongs to the target file."""
    if not diag_path:
        return False

    diag_path = Path(diag_path)

    if diag_path.name != code_path.name:
        return False

    if diag_path.is_absolute():
        return diag_path.resolve() == code_path.resolve()

    return True


def _clang_snippet(file_path: Path, byte_offset) -> tuple[int, str]:
    try:
        with open(file_path, "rb") as f:
            content = f.read()

            if byte_offset < 0 or byte_offset > len(content):
                return 0, "[Offset out of bounds]"

            # Slice the file content up to the target offset
            prefix = content[:byte_offset]

            # Line number is the number of newlines before the offset + 1
            line_number = prefix.count(b"\n") + 1

            # Find the start of the current line
            line_start = prefix.rfind(b"\n")
            if line_start == -1:
                line_start = 0
            else:
                line_start += 1  # Move past the newline character

            # Find the end of the current line
            line_end = content.find(b"\n", byte_offset)
            if line_end == -1:
                line_end = len(content)

            # Extract and decode the line safely
            line_bytes = content[line_start:line_end]
            line_content = line_bytes.decode("utf-8", errors="replace").rstrip()

            return line_number, line_content

    except (
        OSError,
        ValueError,
    ):
        return 0, "[Error reading file]"


def _clippy_tool(code_path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    return candidates


def _pylint_tool(code_path: Path) -> list[Candidate]:
    SAT_PATH.unlink(missing_ok=True)
    PylintRun(
        [
            str(code_path),
            "--output-format=json",
            f"--output={SAT_PATH}",
            "--ignore=.venv,tests",
            "--disable=C",
        ],
        exit=False,
    )

    with SAT_PATH.open() as file:
        output = json.load(file)

    candidates: list[Candidate] = []
    for issue in output:
        smell = SmellCode(
            file_name=str(code_path.name),
            line=int(issue.get("line")),
            snippet=str(
                _issue_snippet(
                    file_path=code_path,
                    start_line=issue.get("line"),
                    end_line=issue.get("endLine"),
                )
            ),
            description=str(issue.get("message")),
        )

        candidates.append(Candidate(smell_type=issue.get("symbol"), smell=smell))
    return candidates


def _issue_snippet(file_path: Path, start_line: int, end_line: int) -> str:
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    start_idx = max(0, start_line - 1)
    return "".join(lines[start_idx:end_line])


def _eslint_tool(code_path: Path) -> list[Candidate]:
    SAT_PATH.unlink(missing_ok=True)
    try:
        eslint_path = find_binary(tool="eslint", code_path=code_path)
        project_root = find_project_root(code_path, marker="node_modules")
        if not eslint_path:
            return _sonar_tool(code_path)
        subprocess.run(  # noqa: S603
            [eslint_path, code_path, "--format=json", f"--output-file={SAT_PATH}"], cwd=project_root, check=False
        )
    except FileNotFoundError:
        print("Error: 'eslint' executable not found in PATH.")
        return _sonar_tool(code_path)

    with SAT_PATH.open() as file:
        output = json.load(file)

    candidates: list[Candidate] = []
    for file in output:
        if not _is_file(Path(file.get("filePath")), code_path):
            continue
        issues = file.get("messages")
        for issue in issues:
            smell = SmellCode(
                file_name=str(code_path.name),
                line=int(issue.get("line")),
                snippet=str(
                    _issue_snippet(
                        file_path=code_path,
                        start_line=issue.get("line"),
                        end_line=issue.get("endLine"),
                    )
                ),
                description=str(issue.get("message")),
            )

            candidates.append(Candidate(smell_type=issue.get("ruleId"), smell=smell))
    return candidates


def _swift_tool(code_path: Path) -> list[Candidate]:
    SAT_PATH.unlink(missing_ok=True)
    candidates: list[Candidate] = []
    try:
        swiftlint_path = find_binary(tool="swiftlint", code_path=code_path)
        if not swiftlint_path:
            print("Error: swiftlint not found")
            return candidates
        subprocess.run(  # noqa: S603
            [swiftlint_path, "lint", code_path, "--reporter", "json", "--output", f"{SAT_PATH}"], check=False
        )
    except FileNotFoundError:
        print("Error: 'swiftlint' executable not found in PATH.")
        return candidates

    try:
        with SAT_PATH.open() as file:
            output = json.load(file)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
    ):
        print("Error: output from swiftlint couldn't be read.")
        return candidates

    for issue in output:
        if issue.get("file") is None:
            continue
        if not _is_file(Path(issue.get("file")), code_path):
            continue
        smell = SmellCode(
            file_name=code_path.name,
            line=int(issue.get("line") or 0),
            snippet=str(
                _issue_snippet(
                    file_path=code_path,
                    start_line=(issue.get("line") or 0),
                    end_line=(issue.get("line") or 0),
                )
            ),
            description=str(issue.get("reason")),
        )

        candidates.append(Candidate(smell_type=issue.get("rule_id"), smell=smell))
    return candidates


def _sonar_tool(code_path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    file_path = code_path.as_posix()

    sonar_token = os.environ.get("SONAR_TOKEN")
    if not sonar_token:
        print("Error: 'SONAR_TOKEN' environment variable is missing.")
        return candidates

    pysonar_executable = shutil.which("pysonar")
    if pysonar_executable is None:
        return candidates
    try:
        subprocess.run(  # noqa: S603
            [
                pysonar_executable,
                f"-Dsonar.sources={file_path}",
                "--token",
                sonar_token,
                "-Dsonar.host.url=http://localhost:9000",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"pysonar failed with exit code {e.returncode}")
        print("Error output:\n", e.stderr)
        return candidates

    client = SonarQubeClient(base_url="http://localhost:9000", token=sonar_token)
    component_key = "SAT:" + str(code_path)
    output = client.issues.search(component_keys=[component_key], resolved=False)

    for issue in output.issues:
        smell = SmellCode(
            file_name=code_path.name,
            line=int(issue.line or 0),
            snippet=str(
                _issue_snippet(
                    file_path=code_path,
                    start_line=((issue.text_range.start_line if issue.text_range else None) or issue.line or 0),
                    end_line=((issue.text_range.end_line if issue.text_range else None) or issue.line or 0),
                )
            ),
            description=str(issue.message),
        )

        candidates.append(Candidate(smell_type=str(issue.message), smell=smell))
    return candidates


def find_project_root(code_path: Path, marker: str = "node_modules") -> Path | None:
    """
    Climbs up the directory tree starting from the file's location
    to find the root directory containing the specified marker.
    """
    for parent in code_path.resolve().parents:
        if (parent / marker).exists():
            return parent
    return None


def find_binary(tool: str, code_path: Path) -> Path | None:
    """
    Finds a binary by checking system PATH, climbing up to find a local
    node_modules folder relative to the file, or checking the user's Cargo binary directory.
    """
    system_path = shutil.which(tool)
    if system_path:
        return Path(system_path)

    project_root = find_project_root(code_path, marker="node_modules")
    if project_root:
        local_node_bin = project_root / "node_modules" / ".bin" / tool
        if local_node_bin.is_file() and os.access(local_node_bin, os.X_OK):
            return local_node_bin.resolve()

    cargo_bin = Path.home() / ".cargo" / "bin" / tool
    if cargo_bin.is_file() and os.access(cargo_bin, os.X_OK):
        return cargo_bin.resolve()

    return None
