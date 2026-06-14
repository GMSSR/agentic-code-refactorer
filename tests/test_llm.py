import re
from unittest.mock import MagicMock, patch

import litellm
import pytest
from pydantic import BaseModel, ValidationError

from src import llm


# pydantic schema to test
class FakeSchema(BaseModel):
    item: str
    quantity: int


# ==========================================
# 1. HAPPY PATH TEST
# ==========================================
@patch("src.llm.litellm.completion")
def test_unified_call_success(mock_completion):
    """Verifies successful LLM execution, schema validation, and config parameters."""
    # Build a nested mock structure mirroring litellm.ModelResponse
    mock_response = MagicMock(spec=litellm.ModelResponse)
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"item": "laptop", "quantity": 5}'))
    ]
    mock_completion.return_value = mock_response

    # Execute the function
    result = llm.unified_call(
        prompt="Order 5 laptops", model="gpt-4o", schema=FakeSchema
    )

    # Assertions
    assert result == {"item": "laptop", "quantity": 5}

    # Verify that your internal constants were passed accurately to litellm
    mock_completion.assert_called_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Order 5 laptops"}],
        response_format=FakeSchema,
        temperature=llm.TEMPERATURE,
        num_retries=3,
        max_tokens=llm.MAX_TOKENS,
    )


# ==========================================
# 2. UNHAPPY PATHS: TYPE & VALIDATION ERRORS
# ==========================================
@patch("src.llm.litellm.completion")
def test_unified_call_fails_on_streamed_response(mock_completion):
    """Verifies that an object that isn't a ModelResponse throws a TypeError."""
    # A standard MagicMock does NOT match litellm.ModelResponse spec
    mock_completion.return_value = MagicMock()

    with pytest.raises(TypeError, match=re.escape("Streamed response received.")):
        llm.unified_call("prompt", "gpt-4o", FakeSchema)


@patch("src.llm.litellm.completion")
def test_unified_call_fails_on_non_string_content(mock_completion):
    """Verifies that empty/None content fields trigger a ValueError."""
    mock_response = MagicMock(spec=litellm.ModelResponse)
    # Simulate an empty content field (e.g. content = None)
    mock_response.choices = [MagicMock(message=MagicMock(content=None))]
    mock_completion.return_value = mock_response

    with pytest.raises(ValueError, match="returned an empty or invalid response"):
        llm.unified_call("prompt", "gpt-4o", FakeSchema)


@patch("src.llm.litellm.completion")
def test_unified_call_fails_on_pydantic_schema_mismatch(mock_completion):
    """Verifies that valid JSON failing the Pydantic schema propagates the error."""
    mock_response = MagicMock(spec=litellm.ModelResponse)
    # The JSON structure is missing 'quantity' required by ToySchema
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"item": "bad_json"}'))
    ]
    mock_completion.return_value = mock_response

    # Pydantic validation errors propagate cleanly out of your function
    with pytest.raises(ValidationError):
        llm.unified_call("prompt", "gpt-4o", FakeSchema)


# ==========================================
# 3. EXCEPTION HANDLING & LOGGING TEST
# ==========================================
@patch("src.llm.litellm.completion")
def test_unified_call_logs_and_re_raises_api_exceptions(mock_completion, capsys):
    """Verifies that raw API exceptions are logged to terminal and re-raised."""
    # Simulate a sudden API Connection or Timeout failure
    mock_completion.side_effect = Exception("API rate limit exceeded")

    with pytest.raises(Exception, match="API rate limit exceeded"):
        llm.unified_call("prompt", "gpt-4o", FakeSchema)

    # Use capsys fixture to check what was printed by your try/except block
    captured = capsys.readouterr()
    assert "Pipeline failed for model gpt-4o: API rate limit exceeded" in captured.out


@patch("src.llm.litellm.completion")
def test_unified_call_fails_on_empty_choices(mock_completion):
    """Verifies that an empty choices list triggers an IndexError (propagated out)."""
    mock_response = MagicMock(spec=litellm.ModelResponse)
    mock_response.choices = []
    mock_completion.return_value = mock_response

    with pytest.raises(IndexError):
        llm.unified_call("prompt", "gpt-4o", FakeSchema)


@patch("src.llm.litellm.completion")
def test_unified_call_fails_on_malformed_json(mock_completion):
    """Verifies that malformed JSON returned by LLM propagates a ValidationError."""
    mock_response = MagicMock(spec=litellm.ModelResponse)
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"item": "laptop", "quantity": '))
    ]
    mock_completion.return_value = mock_response

    with pytest.raises(ValidationError):
        llm.unified_call("prompt", "gpt-4o", FakeSchema)
