import json

import pytest

from src.prompt import (
    eval_prompt,
    j_eval_prompt,
    j_ref_prompt,
    ref_prompt,
)


@pytest.fixture
def mock_data():
    """Provides reusable mock inputs for testing prompt generation functions."""
    return {
        "type_smell": "Long Method",
        "heuristics": [
            {"id": "H1", "name": "Lines of Code", "threshold": " > 50 lines"}
        ],
        "smell": {
            "file_name": "analytics.py",
            "class_name": "DataProcessor",
            "method_name": "calculate_metrics",
            "description": "The method is over 150 lines long.",
            "context": "def calculate_metrics(self):\n    # 150 lines of code here\n    pass",
        },
        "evaluation": {
            "summary": "Method exceeds line limit thresholds significantly.",
            "heuristics": {
                "Lines of Code": {
                    "evaluation": "Found 150 lines.",
                    "conclusion": "met",
                }
            },
        },
        "feedback": {"error": "Missing line depth analysis"},
        "previous_eval": {"status": "rejected", "summary": "Bad summary"},
        "previous_proposal": {"proposed_code": "def short_method(): pass"},
    }


# ==========================================
# Tests for eval_prompt
# ==========================================


def test_eval_prompt_baseline(mock_data):
    """Verifies baseline prompt generation without previous feedback."""
    prompt = eval_prompt(
        type_smell=mock_data["type_smell"],
        heuristics=mock_data["heuristics"],
        smell=mock_data["smell"],
    )

    assert "You are an expert software engineer" in prompt
    assert f"Type: {mock_data['type_smell']}" in prompt
    assert (
        f"Location: {mock_data['smell']['file_name']} -- Class: {mock_data['smell']['class_name']}"
        in prompt
    )

    # Put json to work: Verify exact serialization of the heuristics constitution
    expected_json = json.dumps(mock_data["heuristics"], indent=4, ensure_ascii=False)
    assert expected_json in prompt
    assert "PREVIOUS ATTEMPT & FEEDBACK" not in prompt


def test_eval_prompt_with_feedback(mock_data):
    """Verifies that the conditional feedback block is cleanly appended."""
    prompt = eval_prompt(
        type_smell=mock_data["type_smell"],
        heuristics=mock_data["heuristics"],
        smell=mock_data["smell"],
        previous_eval=mock_data["previous_eval"],
        feedback=mock_data["feedback"],
    )

    assert "PREVIOUS ATTEMPT & FEEDBACK" in prompt

    # Verify the conditional JSON outputs are present
    assert (
        json.dumps(mock_data["previous_eval"], indent=4, ensure_ascii=False) in prompt
    )
    assert json.dumps(mock_data["feedback"], indent=4, ensure_ascii=False) in prompt


def test_eval_prompt_missing_smell_keys(mock_data):
    """Ensures default fallbacks run smoothly when the smell dictionary lacks keys."""
    prompt = eval_prompt(
        type_smell=mock_data["type_smell"],
        heuristics=mock_data["heuristics"],
        smell={},
    )

    assert "Location: Unknown File -- Class: N/A / Method: N/A" in prompt


# ==========================================
# Tests for j_eval_prompt
# ==========================================


def test_j_eval_prompt(mock_data):
    """Validates the auditor evaluation prompt structure and dictionary serialization."""
    prompt = j_eval_prompt(
        type_smell=mock_data["type_smell"],
        heuristics=mock_data["heuristics"],
        smell=mock_data["smell"],
        eval=mock_data["evaluation"],
    )

    assert "You are a senior software engineering auditor" in prompt
    assert "AUDIT RUBRIC" in prompt

    # Verify exact JSON dump of the evaluation payload under audit
    expected_eval_json = json.dumps(
        mock_data["evaluation"], indent=4, ensure_ascii=False
    )
    assert expected_eval_json in prompt


def test_j_eval_prompt_defaults(mock_data):
    """Ensures j_eval_prompt handles missing dictionary tags gracefully."""
    prompt = j_eval_prompt(
        type_smell=mock_data["type_smell"],
        heuristics=mock_data["heuristics"],
        smell={},
        eval=mock_data["evaluation"],
    )

    assert "Location: Unknown -- N/A / N/A" in prompt


# ==========================================
# Tests for ref_prompt
# ==========================================


def test_ref_prompt_baseline(mock_data):
    """Validates the baseline refactoring suggestion prompt layout."""
    prompt = ref_prompt(
        type_smell=mock_data["type_smell"],
        heuristics=mock_data["heuristics"],
        smell=mock_data["smell"],
        eval=mock_data["evaluation"],
    )

    assert "You are a senior software engineer specialized in clean code" in prompt
    assert "FEEDBACK ON PREVIOUS ATTEMPT" not in prompt


def test_ref_prompt_with_feedback(mock_data):
    """Confirms ref_prompt adds feedback notes when an existing attempt fails validation."""
    prompt = ref_prompt(
        type_smell=mock_data["type_smell"],
        heuristics=mock_data["heuristics"],
        smell=mock_data["smell"],
        eval=mock_data["evaluation"],
        previous_proposal=mock_data["previous_proposal"],
        feedback="Code block has missing syntax requirements.",
    )

    assert "FEEDBACK ON PREVIOUS ATTEMPT" in prompt
    assert (
        json.dumps(mock_data["previous_proposal"], indent=4, ensure_ascii=False)
        in prompt
    )
    assert "Code block has missing syntax requirements." in prompt


# ==========================================
# Tests for j_ref_prompt
# ==========================================


def test_j_ref_prompt(mock_data):
    """Validates prompt output context for refactoring audit tasks."""
    prompt = j_ref_prompt(
        type_smell=mock_data["type_smell"],
        heuristics=mock_data["heuristics"],
        smell=mock_data["smell"],
        eval=mock_data["evaluation"],
        ref=mock_data["previous_proposal"],
    )

    assert (
        "You are a senior software engineering auditor specialized in evaluating the quality of proposed source code refactorings."
        in prompt
    )
    assert (
        json.dumps(mock_data["previous_proposal"], indent=4, ensure_ascii=False)
        in prompt
    )
