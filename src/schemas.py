from typing import Literal

from pydantic import BaseModel, Field

# --- Confic Validation Schema ---

lllm_regex = r"^(ollama|mistral|gemini)/.+$"


class Models(BaseModel):
    eval_model: str = Field(pattern=lllm_regex)
    j_eval_model: str = Field(pattern=lllm_regex)
    ref_model: str = Field(pattern=lllm_regex)
    j_ref_model: str = Field(pattern=lllm_regex)


class Setting(BaseModel):
    """Used to validate config.json"""

    local: Models
    cloud: Models


# --- Static Analysis Schema ---


class SmellCode(BaseModel):
    file_name: str
    line: int
    snippet: str
    description: str
    context: str = ""


class Candidate(BaseModel):
    smell_type: str
    smell: SmellCode


# --- evaluation schema ---


class Heuristic(BaseModel):
    name: str = Field(description="The name of the specific design or code quality heuristic being evaluated.")
    evaluation: str = Field(
        description="A detailed analysis of whether the heuristic is met or unmet, explaining the step-by-step reasoning based on the code."  # noqa: E501
    )
    conclusion: Literal["met", "unmet"] = Field(
        description="The final determination for this specific heuristic: 'met' if the code adheres to it, or 'unmet' if it violates it."  # noqa: E501
    )


class Evaluation(BaseModel):
    heuristics: list[Heuristic] = Field(
        description="A list containing an evaluation object for each individual heuristic that applies to the code smell candidate."  # noqa: E501
    )
    summary: str = Field(
        description="A concise overview synthesizing the findings from all the individual heuristic evaluations."
    )
    justification: str = Field(
        description="The overarching rationale for the final decision. This must logically connect the individual heuristic conclusions to explain why the candidate was ultimately accepted or rejected."  # noqa: E501
    )
    status: Literal["accepted", "rejected"] = Field(
        description="The final verdict on the candidate. Use 'accepted' if the code smell is confirmed, or 'rejected' if it is determined not to be a code smell."  # noqa: E501
    )


# --- Evaluation Judgment Schema ---


class RubricE(BaseModel):
    R1: str = Field(
        description="Detailed critique evaluating whether the previous evaluator analyzed ALL applicable heuristics provided in the input."  # noqa: E501
    )
    R2: str = Field(
        description="Detailed critique evaluating whether the evaluator cited specific, concrete evidence directly from the affected code snippet."  # noqa: E501
    )
    R3: str = Field(
        description="Detailed critique checking if the evaluator's final classification is logically consistent with their individual heuristic verdicts."  # noqa: E501
    )
    R4: str = Field(
        description="Detailed critique on whether the evaluator actively investigated and ruled out the possibility of a false positive."  # noqa: E501
    )
    R5: str = Field(
        description="Detailed critique assessing whether the evaluator's confidence level is appropriately calibrated based on the available evidence."  # noqa: E501
    )
    R6: str = Field(
        description="Detailed critique on the overall logical coherence of the summary, ensuring it is entirely free of internal contradictions."  # noqa: E501
    )


class JudgementE(BaseModel):
    rubrics: RubricE = Field(
        description="A structured object containing the granular audits for each of the six rubric criteria (R1 through R6)."  # noqa: E501
    )
    critical_flaw: str = Field(
        description="Document any critical error, fatal omission, or major logical contradiction that completely invalidates the evaluation. If no major flaws exist, strictly enter 'None'."  # noqa: E501
    )
    verdict: Literal["approved", "rejected"] = Field(
        description="The final meta-verdict on the audit quality. Select 'approved' if the reasoning is entirely sound and trustworthy. Select 'rejected' if it contains significant errors or critical flaws."  # noqa: E501
    )
    feedback: str = Field(
        description="Provide clear, corrective feedback explaining exactly what went wrong, what was missed, and how to fix the evaluation. Required if the verdict is 'rejected'."  # noqa: E501
    )


# --- Refactoring Schema ---


class Refactor(BaseModel):
    thought: str = Field(
        description=(
            "A detailed, step-by-step reasoning block. Trace out the root cause "
            "of the code smell, explicitly plan your refactoring steps, and justify "
            "the selected implementation strategy before writing any code."
        )
    )
    technique_chosen: str = Field(
        description=(
            "The specific, formal name of the design pattern or refactoring technique "
            "applied to resolve the smell (e.g., 'Extract Class', 'Move Method', "
            "'Replace Conditional with Polymorphism')."
        )
    )
    proposed_code: str = Field(
        description=(
            "The complete, clean, and fully functional Python syntax replacement block. "
            "It must eliminate the smell while strictly preserving the original functional behavior."
        )
    )
    explanation: str = Field(
        description=(
            "A detailed walkthrough explaining exactly what structural aspects of the code "
            "changed and how this specific implementation successfully neutralizes the flagged heuristics."
        )
    )


# --- Reafactoring Judgement Schema ---


class RubricR(BaseModel):
    R1: str = Field(
        description="Heuristic coverage: Evaluate if the proposal fixes all the issues pointed out by the original heuristics."  # noqa: E501
    )
    R2: str = Field(
        description="Justification groundedness: Evaluate if the proposal provided clear technical reasoning for each structural change."  # noqa: E501
    )
    R3: str = Field(
        description="Code correctness: Evaluate if the refactored code is free of syntactical/structural flaws that would stop it from running."  # noqa: E501
    )
    R4: str = Field(
        description="Free of Bugs: Evaluate if the code completely preserves the original system logic without introducing edge-case regressions or bugs."  # noqa: E501
    )


class JudgementR(BaseModel):
    rubrics: RubricR = Field(
        description="An object containing targeted feedback on the quality of the refactoring and its explanations for each rubric point (R1 through R4)."  # noqa: E501
    )
    critical_flaw: str = Field(
        description="Clear description of any syntax issues, behavioral changes, or logic bugs found in the code. If the code is completely clean, write 'None'."  # noqa: E501
    )
    verdict: Literal["approved", "rejected"] = Field(
        description="Strict structural verdict: Select 'approved' if the implementation is safe to deploy, or 'rejected' if it fails any rubric point."  # noqa: E501
    )
    feedback: str = Field(
        description="Actionable instructions and remediation steps detailing what needs to be changed if the implementation was rejected."  # noqa: E501
    )
