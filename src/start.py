import argparse
import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

import constants as c

from .exceptions import CheckpointError, ConfigError
from .models import AppConfig, Container, PipelineState
from .schemas import Setting


def _parse_arg() -> argparse.Namespace:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluation of code smell candidates and creation of refactoring proposals."
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from a checkpoint"
    )
    parser.add_argument(
        "code_path",
        nargs="?",
        help="Path to the file to be evaluated (required for fresh runs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass interactive prompts and force a fresh evaluation, wipes any existing checkpoint.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Skips the refactoring stage, only effetuating evaluation of the candidates code smells.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Runs the script locally without using cloud APIs",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Runs in Direct Mode, reading target file content directly and skipping local tools/databases.",
    )

    args = parser.parse_args()

    # Validates the arguments passed
    if not args.code_path and not args.resume:
        parser.error("Either --resume or a code-path need to be provided.")

    if args.resume and (args.code_path or args.force or args.direct):
        parser.error("Cannot use --resume with --code-path, --force, or --direct.")

    if args.force and not args.code_path:
        parser.error("Argument --force requires a code path to be given.")

    if args.direct and not args.code_path:
        parser.error("Argument --direct requires a code path to be given.")

    if args.code_path and not (Path(args.code_path).is_file() or Path(args.code_path).is_dir()):
        parser.error(
            "The provided code path does not exist"
        )

    return args


def _load(local: bool) -> tuple[dict[str, Any], str, str, str, str]:
    """Load configuration, and heuristics from disk. Initializes the API keys if running on cloud."""

    try:
        with c.CONFIG_PATH.open("r", encoding="utf-8") as file:
            config_data = json.load(file)
    except FileNotFoundError:
        raise ConfigError("Error: Config file not found.") from None
    except json.JSONDecodeError as e:
        raise ConfigError(f"Failed to decode JSON: {e.msg} at line {e.lineno}") from None
    except Exception as e:
        raise ConfigError(
            f"An unexpected error occurred while opening the config file: {e}"
        ) from None

    try:
        with c.HEURISTICS_PATH.open("r", encoding="utf-8") as file:
            heuristics = json.load(file)
    except FileNotFoundError:
        raise ConfigError("Error: Heuristic File not found.") from None
    except json.JSONDecodeError as e:
        raise ConfigError(f"Failed to decode JSON: {e.msg} at line {e.lineno}") from None
    except Exception as e:
        raise ConfigError(
            f"An unexpected error occurred while opening the heuristics file: {e}"
        ) from None

    try:
        validated_config = Setting.model_validate(config_data)
    except ValidationError as e:
        error_message = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        raise ConfigError(f"Configuration file failed validation: {error_message}") from None

    if local:
        eval_model = validated_config.local.eval_model
        j_eval_model = validated_config.local.j_eval_model
        ref_model = validated_config.local.ref_model
        j_ref_model = validated_config.local.j_ref_model
    else:
        eval_model = validated_config.cloud.eval_model
        j_eval_model = validated_config.cloud.j_eval_model
        ref_model = validated_config.cloud.ref_model
        j_ref_model = validated_config.cloud.j_ref_model

        is_mistral_used = any("mistral" in m or "codestral" in m for m in (eval_model, j_eval_model, ref_model, j_ref_model))
        is_gemini_used = any("gemini" in m for m in (eval_model, j_eval_model, ref_model, j_ref_model))

        if is_mistral_used and not os.getenv("MISTRAL_API_KEY"):
            raise ConfigError("Error getting the MISTRAL_API_KEY.")
        if is_gemini_used and not os.getenv("GEMINI_API_KEY"):
            raise ConfigError("Error getting the GEMINI_API_KEY.")

    return heuristics, eval_model, j_eval_model, ref_model, j_ref_model


def _resume() -> tuple[
    Path,
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Load the variables needed to resume from the saved checkpoint."""

    if c.CHECKPOINT_PATH.is_file():
        try:
            with c.CHECKPOINT_PATH.open("r", encoding="utf-8") as file:
                checkpoint_data = json.load(file)
            code_path = checkpoint_data.get("code_path")
            print(f"Checkpoint found. Resuming evaluation for: {code_path}")
        except Exception as e:
            raise CheckpointError(f"Error reading checkpoint file: {e}") from None
    else:
        raise CheckpointError("Error: No active checkpoint found.")

    if not code_path:
        raise CheckpointError("Error: Checkpoint file is missing the 'code_path' key.")
    code_path = Path(code_path)

    start_stage = checkpoint_data.get("current_stage", "static_analysis")

    approved_eval = checkpoint_data.get("approved_eval", [])
    rejected_eval = checkpoint_data.get("rejected_eval", [])
    approved_proposal = checkpoint_data.get("approved_proposal", [])
    rejected_proposal = checkpoint_data.get("rejected_proposal", [])
    output = checkpoint_data.get("output", [])

    return (
        code_path,
        start_stage,
        approved_eval,
        rejected_eval,
        approved_proposal,
        rejected_proposal,
        output,
    )


def start() -> Container:
    args = _parse_arg()
    if args.force and c.CHECKPOINT_PATH.exists():
        c.CHECKPOINT_PATH.unlink(missing_ok=True)

    heuristic_data, eval_model, j_eval_model, ref_model, j_ref_model = _load(args.local)

    if args.resume:
        (
            code_path,
            start_stage,
            approved_eval,
            rejected_eval,
            approved_proposal,
            rejected_proposal,
            output,
        ) = _resume()
    else:
        code_path = Path(args.code_path)
        start_stage = "static_analysis"
        approved_eval = []
        rejected_eval = []
        approved_proposal = []
        rejected_proposal = []
        output = []

    config = AppConfig(
        is_eval_only=args.eval or args.direct,
        is_local=args.local,
        is_direct=args.direct,
        code_path=code_path,
        start_stage=start_stage,
        eval_model=eval_model,
        j_eval_model=j_eval_model,
        ref_model=ref_model,
        j_ref_model=j_ref_model,
        heuristic_data=heuristic_data,
    )

    state = PipelineState(
        approved_eval=approved_eval,
        rejected_eval=rejected_eval,
        approved_proposal=approved_proposal,
        rejected_proposal=rejected_proposal,
        output=output,
    )

    container = Container(config=config, state=state)

    return container
