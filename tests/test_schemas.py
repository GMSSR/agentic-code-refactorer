import pytest
from pydantic import ValidationError

from src.schemas import (
    Candidate,
    Evaluation,
    Heuristic,
    JudgementE,
    JudgementR,
    Models,
    Refactor,
    RubricE,
    RubricR,
    Setting,
    SmellCode,
)

# --- Fixtures for Valid Sub-Objects ---


@pytest.fixture
def valid_models_data():
    return {
        "eval_model": "ollama/llama3",
        "j_eval_model": "gemini/gemini-1.5-pro",
        "ref_model": "mistral/mistral-large",
        "j_ref_model": "ollama/codellama",
    }


@pytest.fixture
def valid_smell_code_data():
    return {
        "file_name": "auth.py",
        "class_name": "Authenticator",
        "method_name": "login",
        "line": 42,
        "snippet": "def login(self):\n    pass",
        "description": "Hardcoded credentials suspect.",
        "context": "Prior to refactoring security layer.",
    }


@pytest.fixture
def valid_heuristic_data():
    return {
        "name": "Single Responsibility Principle",
        "evaluation": "The method handles both logging and database connection, violating SRP.",
        "conclusion": "unmet",
    }


@pytest.fixture
def valid_rubric_e_data():
    return {f"R{i}": f"Critique for criteria R{i}" for i in range(1, 7)}


@pytest.fixture
def valid_rubric_r_data():
    return {f"R{i}": f"Critique for criteria R{i}" for i in range(1, 5)}


# --- Test Config Validation Schema (Models & Setting) ---


def test_models_valid(valid_models_data):
    """Ensure valid model formats pass regex validation."""
    model_instance = Models(**valid_models_data)
    assert model_instance.eval_model == "ollama/llama3"


@pytest.mark.parametrize(
    "invalid_model_string",
    [
        "openai/gpt-4",  # Invalid provider
        "ollama/",  # Missing name after slash
        "gemini",  # Missing slash and name
        "mistral /",  # White space before the slash
    ],
)
def test_models_invalid_pattern(valid_models_data, invalid_model_string):
    """Ensure invalid model naming formats trigger validation errors."""
    valid_models_data["eval_model"] = invalid_model_string
    with pytest.raises(ValidationError) as exc_info:
        Models(**valid_models_data)
    assert "eval_model" in str(exc_info.value)


def test_setting_valid(valid_models_data):
    """Ensure configuration setting aggregates local and cloud models properly."""
    setting_data = {"local": valid_models_data, "cloud": valid_models_data}
    setting = Setting(**setting_data)
    assert isinstance(setting.local, Models)
    assert isinstance(setting.cloud, Models)


# --- Test Static Analysis Schema (SmellCode & Candidate) ---


def test_smell_code_defaults(valid_smell_code_data):
    """Ensure context defaults to an empty string if omitted."""
    valid_smell_code_data.pop("context")
    smell = SmellCode(**valid_smell_code_data)
    assert smell.context == ""


def test_smell_code_invalid_types(valid_smell_code_data):
    """Ensure strict type tracking triggers a ValidationError for incorrect types."""
    valid_smell_code_data["line"] = "not-an-int"
    with pytest.raises(ValidationError):
        SmellCode(**valid_smell_code_data)


def test_candidate_valid(valid_smell_code_data):
    """Verify Candidate properly encompasses structural code smell metadata."""
    candidate_data = {"smell_type": "Long Method", "smell": valid_smell_code_data}
    candidate = Candidate(**candidate_data)
    assert candidate.smell.file_name == "auth.py"


# --- Test Evaluation Schema (Heuristic & Evaluation) ---


@pytest.mark.parametrize("conclusion", ["met", "unmet"])
def test_heuristic_valid_literal(valid_heuristic_data, conclusion):
    """Ensure 'met' and 'unmet' are accepted outcomes for Heuristics."""
    valid_heuristic_data["conclusion"] = conclusion
    heuristic = Heuristic(**valid_heuristic_data)
    assert heuristic.conclusion == conclusion


def test_heuristic_invalid_literal(valid_heuristic_data):
    """Ensure unlisted literal terms fail validation."""
    valid_heuristic_data["conclusion"] = "partially_met"  # Invalid Choice
    with pytest.raises(ValidationError):
        Heuristic(**valid_heuristic_data)


def test_evaluation_valid(valid_heuristic_data):
    """Verify evaluation structures accept sets of heuristics and a final status."""
    eval_data = {
        "heuristics": [valid_heuristic_data],
        "summary": "The class looks problematic.",
        "justification": "Too many structural exceptions encountered.",
        "status": "accepted",
    }
    evaluation = Evaluation(**eval_data)
    assert len(evaluation.heuristics) == 1
    assert evaluation.status == "accepted"


# --- Test Evaluation Judgment Schema (RubricE & JudgementE) ---


def test_rubric_e_valid(valid_rubric_e_data):
    """Check if all 6 evaluation criteria rubrics register correctly."""
    rubric = RubricE(**valid_rubric_e_data)
    assert rubric.R1 == "Critique for criteria R1"
    assert rubric.R6 == "Critique for criteria R6"


@pytest.mark.parametrize("verdict", ["approved", "rejected"])
def test_judgement_e_valid(valid_rubric_e_data, verdict):
    """Validate meta-verdict assertions for Evaluation judgements."""
    judgement_data = {
        "rubrics": valid_rubric_e_data,
        "critical_flaw": "None",
        "verdict": verdict,
        "feedback": "Good job or please re-evaluate.",
    }
    judgement = JudgementE(**judgement_data)
    assert judgement.verdict == verdict


# --- Test Refactoring Schema (Refactor) ---


def test_refactor_valid():
    """Validate refactoring proposals hold structured code strategy logs."""
    refactor_data = {
        "thought": "Let's split this class up into single tasks.",
        "technique_chosen": "Extract Class",
        "proposed_code": "class NewClass:\n    pass",
        "explanation": "Separated concern A from concern B.",
    }
    refactor = Refactor(**refactor_data)
    assert refactor.technique_chosen == "Extract Class"


# --- Test Refactoring Judgment Schema (RubricR & JudgementR) ---


def test_rubric_r_missing_field(valid_rubric_r_data):
    """Ensure missing evaluation entries fail validation."""
    valid_rubric_r_data.pop("R4")
    with pytest.raises(ValidationError):
        RubricR(**valid_rubric_r_data)


def test_judgement_r_valid(valid_rubric_r_data):
    """Verify overall structural feedback configuration maps properly for refactoring reviews."""
    judgement_data = {
        "rubrics": valid_rubric_r_data,
        "critical_flaw": "Syntax error on line 3 of proposed code.",
        "verdict": "rejected",
        "feedback": "Please correct the trailing indentation block.",
    }
    judgement = JudgementR(**judgement_data)
    assert judgement.verdict == "rejected"
    assert judgement.rubrics.R1 == "Critique for criteria R1"
