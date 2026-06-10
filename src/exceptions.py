from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from pydantic import BaseModel
from mistralai.client import Mistral
from google import genai

class StartupError(Exception):
    """Base exception for initialization errors."""
    pass

class ConfigError(StartupError):
    """Raised when configuration loading or validation fails."""
    pass

class CheckpointError(StartupError):
    """Raised when checkpoint loading or validation fails."""
    pass