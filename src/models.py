from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, NamedTuple

@dataclass(frozen=True)
class AppConfig:
    """Pure, immutable configuration and metadata."""
    is_eval_only: bool
    is_local: bool
    code_path: Path
    start_stage: str
    
    # Model string identifiers
    eval_model: str
    j_eval_model: str
    ref_model: str
    j_ref_model: str

    # Static loaded data
    heuristic_data: dict[str, Any]

@dataclass
class PipelineState:
    """The mutable runtime state tracking data flowing through the pipeline."""
    approved_eval: list[Any] = field(default_factory=list)
    rejected_eval: list[Any] = field(default_factory=list)
    approved_proposal: list[Any] = field(default_factory=list)
    rejected_proposal: list[Any] = field(default_factory=list)
    output: list[Any] = field(default_factory=list)

class Container(NamedTuple):
    config: AppConfig
    state: PipelineState