import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils import feedback_loop, save_checkpoint

## ==========================================
## Tests for feedback_loop
## ==========================================


def test_feedback_loop_all_approved():
    """Test when all items are approved on the very first iteration."""
    items = ["item1", "item2"]

    # Fixed Ruff E731: Replaced lambda assignments with local def functions
    def process_func(item):
        return (item, "processed")

    def judge_func(item):
        return (*item, {"verdict": "approved"})

    with patch("src.utils.MAX_TENTATIVES", 3):
        approved, rejected = feedback_loop(items, process_func, judge_func)

    assert len(approved) == 2
    assert len(rejected) == 0
    assert approved[0] == ("item1", "processed")
    assert approved[1] == ("item2", "processed")


def test_feedback_loop_always_rejected():
    """Test when items are consistently rejected until MAX_TENTATIVES is reached."""
    items = ["item1"]

    def process_func(item):
        if isinstance(item, tuple):
            return item
        return (item, "processed")

    # Fixed Ruff E731: Replaced lambda assignment with local def function
    def judge_func(item):
        return (*item, {"verdict": "rejected"})

    # Setting MAX_TENTATIVES to 3 means it will run exactly 2 loops (i=1, i=2)
    with patch("src.utils.MAX_TENTATIVES", 3):
        approved, rejected = feedback_loop(items, process_func, judge_func)

    assert len(approved) == 0
    assert len(rejected) == 1
    assert rejected[0][-1] == {"verdict": "rejected"}


def test_feedback_loop_empty_input():
    """Test with an empty list of items."""
    process_func = MagicMock()
    judge_func = MagicMock()

    approved, rejected = feedback_loop([], process_func, judge_func)

    assert approved == []
    assert rejected == []
    process_func.assert_not_called()
    judge_func.assert_not_called()


## ==========================================
## Tests for save_checkpoint
## ==========================================


def test_save_checkpoint_success(tmp_path):
    """Test successful creation and replacement of the checkpoint file."""
    mock_checkpoint_path = tmp_path / "checkpoint.json"

    stage_name = "stage_1"
    approved_e = ["app_e1"]
    rejected_e = ["rej_e1"]
    approved_p = ["app_p1"]
    rejected_p = ["rej_p1"]
    output = "final_output"
    code_path = Path("/mock/path/to/code.py")

    with patch("src.utils.CHECKPOINT_PATH", mock_checkpoint_path):
        save_checkpoint(
            stage_name,
            approved_e,
            rejected_e,
            approved_p,
            rejected_p,
            output,
            code_path,
        )

    assert mock_checkpoint_path.exists()

    with open(mock_checkpoint_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["current_stage"] == stage_name
    assert data["code_path"] == str(code_path)
    assert data["approved_eval"] == approved_e
    assert data["rejected_proposal"] == rejected_p
    assert data["output"] == output


def test_save_checkpoint_exception_handling(capsys):
    """Test that exceptions during save are caught and logged to stderr."""
    mock_path = MagicMock()
    mock_path.with_name.side_effect = Exception("Disk full or permission denied")

    with patch("src.utils.CHECKPOINT_PATH", mock_path):
        save_checkpoint("stage_fail", [], [], [], [], "output", Path("code.py"))

    captured = capsys.readouterr()

    assert "Warning: Failed to save progress checkpoint" in captured.err
    assert "Disk full or permission denied" in captured.err


def test_feedback_loop_iterative_fixes():
    """Test feedback loop when items are rejected first but fixed and approved on retry."""
    items = ["item1"]

    def process_func(item):
        if isinstance(item, tuple):
            # Item is the rejected 3-tuple from previous iteration
            original_item = item[0]
            feedback = item[2]["feedback"]
            assert feedback == "please fix"
            return (original_item, "processed_v2")
        return (item, "processed_v1")

    def judge_func(item):
        if item[1] == "processed_v2":
            return (*item, {"verdict": "approved"})
        return (*item, {"verdict": "rejected", "feedback": "please fix"})

    with patch("src.utils.MAX_TENTATIVES", 3):
        approved, rejected = feedback_loop(items, process_func, judge_func)

    assert len(approved) == 1
    assert len(rejected) == 0
    assert approved[0] == ("item1", "processed_v2")


def test_feedback_loop_max_tentatives_bounds():
    """Test feedback loop when MAX_TENTATIVES is 1 (loop should not run)."""
    items = ["item1"]
    process_func = MagicMock()
    judge_func = MagicMock()

    with patch("src.utils.MAX_TENTATIVES", 1):
        approved, rejected = feedback_loop(items, process_func, judge_func)

    assert approved == []
    assert rejected == ["item1"]
    process_func.assert_not_called()
    judge_func.assert_not_called()
