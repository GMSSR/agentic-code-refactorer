import json
import signal
import sys
from pathlib import Path
from threading import Event

from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import constants as c  # noqa: E402
from src.llm import unified_call  # noqa: E402
from src.prompt import eval_prompt, j_eval_prompt, j_ref_prompt, ref_prompt  # noqa: E402
from src.schemas import Evaluation, JudgementE, JudgementR, Refactor  # noqa: E402
from src.start import start  # noqa: E402
from src.static_ana import static  # noqa: E402
from src.utils import feedback_loop, save_checkpoint  # noqa: E402

if str(c.PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(c.PROJECT_DIR))

"""Flag for handling interruption, to be deprecated in favor of being more granularly in the future.
With the goal of allowing quicker handling.
"""
interrupted = Event()


def signal_handler(sig, frame):
    """To be deprecated"""
    if not interrupted.is_set():
        print("Exiting after the next checkpoint is saved, press 'ctrl + c' again to force exit before the checkpoint.")
        interrupted.set()
    else:
        print("Force Closing.")
        sys.exit(10)


def eval_func(old_eval):  # likely should be moved to another file
    smell_type, smell, heuristics, evaluation, judgment = old_eval
    evaluation = unified_call(
        prompt=eval_prompt(
            type_smell=smell_type,
            heuristics=heuristics,
            smell=smell,
            previous_eval=evaluation,
            feedback=judgment.get("feedback"),
        ),
        model=container.config.eval_model,
        schema=Evaluation,
    )
    return (smell_type, smell, heuristics, evaluation)


def judge_eval_func(evaluation):  # likely should be moved to another file
    smell_type, smell, heuristics, evaluation = evaluation
    judgment = unified_call(
        prompt=j_eval_prompt(type_smell=smell_type, heuristics=heuristics, smell=smell, evaluation=evaluation),
        model=container.config.j_eval_model,
        schema=JudgementE,
    )
    return (smell_type, smell, heuristics, evaluation, judgment)


def proposal_func(old_proposal):  # likely should be moved to another file
    smell_type, smell, heuristics, evaluation, proposal, judgment = old_proposal
    proposal = unified_call(
        prompt=ref_prompt(
            type_smell=smell_type,
            heuristics=heuristics,
            smell=smell,
            evaluation=evaluation,
            previous_proposal=proposal,
            feedback=judgment.get("feedback"),
        ),
        model=container.config.ref_model,
        schema=Refactor,
    )
    return (smell_type, smell, heuristics, evaluation, proposal)


def judge_proposal_func(refactor):  # likely should be moved to another file
    smell_type, smell, heuristics, evaluation, proposal = refactor
    judgment = unified_call(
        prompt=j_ref_prompt(
            type_smell=smell_type,
            heuristics=heuristics,
            smell=smell,
            evaluation=evaluation,
            ref=proposal,
        ),
        model=container.config.j_ref_model,
        schema=JudgementR,
    )
    return (smell_type, smell, heuristics, evaluation, proposal, judgment)


if __name__ == "__main__":  # needed due to using ProcessPoolExecutor on static_ana
    signal.signal(signal.SIGINT, signal_handler)

    container = start()

    approved_eval = container.state.approved_eval
    rejected_eval = container.state.rejected_eval
    approved_proposal = container.state.approved_proposal
    rejected_proposal = container.state.rejected_proposal

    file_pure_name = Path(container.config.code_path).stem
    RESULTS_PATH = c.SCRIPT_DIR / "data" / f"results_{file_pure_name}.json"
    print(f"Results for this run will be saved to: {RESULTS_PATH}\n")

    start_index = c.STAGES.index(container.config.start_stage) if container.config.start_stage in c.STAGES else 0

    if start_index <= c.STAGES.index("static_analysis"):
        skips = 0
        temp = []
        skipped_smells = []
        if container.config.is_direct:
            try:
                target_code_content = container.config.code_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Error reading target file: {e}", file=sys.stderr)
                sys.exit(1)

            code_smells = []
            for smell_type in container.config.heuristic_data.keys():
                if smell_type == "Default":
                    continue
                smell_code = {
                    "file_name": container.config.code_path.name,
                    "class_name": "N/A",
                    "method_name": "N/A",
                    "line": 1,
                    "snippet": "",
                    "description": f"Direct evaluation for {smell_type}",
                    "context": target_code_content,
                }
                code_smells.append((smell_type, smell_code))
        else:
            code_smells = static(code_path=container.config.code_path)

        for smell_type, smell in code_smells:
            heuristics = container.config.heuristic_data.get(smell_type, "N")
            if heuristics == "N":
                # skips += 1
                # skipped_smells.append([smell_type, smell])
                # continue
                heuristics = container.config.heuristic_data.get("Default")
            evaluation = unified_call(
                prompt=eval_prompt(type_smell=smell_type, heuristics=heuristics, smell=smell),
                model=container.config.eval_model,
                schema=Evaluation,
            )
            temp.append([smell_type, smell, heuristics, evaluation])

        for (
            smell_type,
            smell,
            heuristics,
            evaluation,
        ) in temp:  # This likely can be offloaded to feedback_loop, remember to change save_checkpoint to support
            judgment = unified_call(
                prompt=j_eval_prompt(
                    type_smell=smell_type,
                    heuristics=heuristics,
                    smell=smell,
                    evaluation=evaluation,
                ),
                model=container.config.j_eval_model,
                schema=JudgementE,
            )

            if judgment.get("verdict") == "approved":
                approved_eval.append([smell_type, smell, heuristics, evaluation])
            else:
                rejected_eval.append([smell_type, smell, heuristics, evaluation, judgment])

        print(f"The number of smells that were skipped due to missing heuristics was: {skips}\n")
        try:
            tmp_filename = c.LOG_PATH.with_name(f"{c.LOG_PATH.name}.tmp")
            with tmp_filename.open("w", encoding="utf-8") as f:
                json.dump(skipped_smells, f, indent=4, ensure_ascii=False)
            tmp_filename.replace(c.LOG_PATH)
        except Exception as e:
            print(f"Warning: Failed to save logs: {e}", file=sys.stderr)
        current_stage = "eval_loop"
        save_checkpoint(
            current_stage,
            approved_e=approved_eval,
            rejected_e=rejected_eval,
            approved_p=approved_proposal,
            rejected_p=rejected_proposal,
            output=container.state.output,
            code_path=container.config.code_path,
        )
        if interrupted.is_set():
            sys.exit(130)

    if start_index <= c.STAGES.index("eval_loop"):
        new_approved_eval, rejected_eval = feedback_loop(
            rejected_eval, process_func=eval_func, judge_func=judge_eval_func
        )

        approved_eval.extend(new_approved_eval)

        for smell_type, smell, __, __, __ in rejected_eval:
            container.state.output.append(
                {
                    "smell_type": smell_type,
                    "smell": smell,
                    "evaluation": None,
                    "proposal": None,
                }
            )

        temp = []

        for smell_type, smell, heuristics, evaluation in approved_eval:
            if evaluation.get("status") != "accepted":
                container.state.output.append(
                    {
                        "smell_type": smell_type,
                        "smell": smell,
                        "evaluation": evaluation,
                        "proposal": None,
                    }
                )
            else:
                temp.append([smell_type, smell, heuristics, evaluation])

        approved_eval = temp
        temp = []
        current_stage = "initial_refactoring"
        save_checkpoint(
            current_stage,
            approved_e=approved_eval,
            rejected_e=rejected_eval,
            approved_p=approved_proposal,
            rejected_p=rejected_proposal,
            output=container.state.output,
            code_path=container.config.code_path,
        )
        if interrupted.is_set():
            sys.exit(130)

    if start_index <= c.STAGES.index("initial_refactoring") and not container.config.is_eval_only:
        temp = []
        for smell_type, smell, heuristics, evaluation in approved_eval:
            proposal = unified_call(
                prompt=ref_prompt(
                    type_smell=smell_type,
                    heuristics=heuristics,
                    smell=smell,
                    evaluation=evaluation,
                ),
                model=container.config.ref_model,
                schema=Refactor,
            )
            temp.append([smell_type, smell, heuristics, evaluation, proposal])

        for smell_type, smell, heuristics, evaluation, proposal in temp:
            judgment = unified_call(
                prompt=j_ref_prompt(
                    type_smell=smell_type,
                    heuristics=heuristics,
                    smell=smell,
                    evaluation=evaluation,
                    ref=proposal,
                ),
                model=container.config.j_ref_model,
                schema=JudgementR,
            )  # This likely can be easily offloaded to feedback_loop, remember to change save_checkpoint to support

            if judgment.get("verdict") == "approved":
                approved_proposal.append([smell_type, smell, heuristics, evaluation, proposal])
            else:
                rejected_proposal.append([smell_type, smell, heuristics, evaluation, proposal, judgment])
        current_stage = "refactoring_loop"
        save_checkpoint(
            current_stage,
            approved_e=approved_eval,
            rejected_e=rejected_eval,
            approved_p=approved_proposal,
            rejected_p=rejected_proposal,
            output=container.state.output,
            code_path=container.config.code_path,
        )
        if interrupted.is_set():
            sys.exit(130)

    if start_index <= c.STAGES.index("refactoring_loop") and not container.config.is_eval_only:
        new_approved_proposal, rejected_proposal = feedback_loop(
            rejected_proposal,
            process_func=proposal_func,
            judge_func=judge_proposal_func,
        )

        approved_proposal.extend(new_approved_proposal)

        for (
            smell_type,
            smell,
            _heuristics,
            evaluation,
            _proposal,
            _judgment,
        ) in rejected_proposal:
            container.state.output.append(
                {
                    "smell_type": smell_type,
                    "smell": smell,
                    "evaluation": evaluation,
                    "proposal": None,
                }
            )

        for smell_type, smell, _heuristics, evaluation, proposal in approved_proposal:
            container.state.output.append(
                {
                    "smell_type": smell_type,
                    "smell": smell,
                    "evaluation": evaluation,
                    "proposal": proposal,
                }
            )
        current_stage = "saving"
        save_checkpoint(
            current_stage,
            approved_e=approved_eval,
            rejected_e=rejected_eval,
            approved_p=approved_proposal,
            rejected_p=rejected_proposal,
            output=container.state.output,
            code_path=container.config.code_path,
        )
        if interrupted.is_set():
            sys.exit(130)

    print(
        "The number of smells that weren't able to have an evaluation approved "
        f"within the maximum number of attempts is {len(rejected_eval)}."
    )
    if not container.config.is_eval_only:
        print(
            "The number of smells that weren't able to have a refactoring proposal approved "
            f"within the maximum number of attempts is {len(rejected_proposal)}."
        )

    if container.config.is_eval_only and container.config.start_stage != "saving":
        for smell_type, smell, _heuristics, evaluation in approved_eval:
            container.state.output.append(
                {
                    "smell_type": smell_type,
                    "smell": smell,
                    "evaluation": evaluation,
                    "proposal": None,
                }
            )

    try:
        tmp_filename = RESULTS_PATH.with_name(f"{RESULTS_PATH.name}.tmp")
        with tmp_filename.open("w", encoding="utf-8") as f:
            json.dump(container.state.output, f, indent=4, ensure_ascii=False)
        tmp_filename.replace(RESULTS_PATH)
        if c.CHECKPOINT_PATH.exists():
            c.CHECKPOINT_PATH.unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: Failed to save results: {e}", file=sys.stderr)
        sys.exit(12)

    sys.exit(0)
