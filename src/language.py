from pathlib import Path

from langchain_text_splitters import Language


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
