import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch
import pytest
from src.metrics import load_oracle, compile_metrics

def test_load_oracle_missing():
    path = Path("non_existent_file.csv")
    oracle = load_oracle(path)
    assert oracle == {}

def test_load_oracle_valid(tmp_path):
    csv_file = tmp_path / "Oracle.csv"
    csv_content = (
        "Class_File,Class_Name,Code_Smell,Ratings,Average,Oracle\n"
        "AbstractWriteHolder.java,AbstractWriteHolder,Feature Envy,3.0,3.0,0\n"
        "AbortedTransactionException.java,AbortedTransactionException,Refused Bequest,3.0,3.0,1\n"
        "AndroidGraphics.java,AndroidGraphics,God Class,3.0,3.0,1\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")
    
    oracle = load_oracle(csv_file)
    # AndroidGraphics.java has smell_type matched as god class -> mapped to large class
    assert oracle[("abstractwriteholder.java", "feature envy")] == 0
    assert oracle[("abortedtransactionexception.java", "refused bequest")] == 1
    assert oracle[("androidgraphics.java", "large class")] == 1

def test_compile_metrics_calculation(tmp_path):
    csv_file = tmp_path / "Oracle.csv"
    csv_content = (
        "Class_File,Class_Name,Code_Smell,Ratings,Average,Oracle\n"
        "A.java,A,God Class,3.0,3.0,1\n"      # TP (prediction: accepted, oracle: 1)
        "B.java,B,Feature Envy,3.0,3.0,0\n"   # FP (prediction: accepted, oracle: 0)
        "C.java,C,Data Class,3.0,3.0,0\n"     # TN (prediction: rejected, oracle: 0)
        "D.java,D,Refused Bequest,3.0,3.0,1\n" # FN (prediction: rejected, oracle: 1)
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    results = [
        {"smell_type": "God Class", "smell": {"file_name": "A.java"}, "evaluation": {"status": "accepted"}},
        {"smell_type": "Feature Envy", "smell": {"file_name": "B.java"}, "evaluation": {"status": "accepted"}},
        {"smell_type": "Data Class", "smell": {"file_name": "C.java"}, "evaluation": {"status": "rejected"}},
        {"smell_type": "Refused Bequest", "smell": {"file_name": "D.java"}, "evaluation": {"status": "rejected"}},
        {"smell_type": "Feature Envy", "smell": {"file_name": "E.java"}, "evaluation": {"status": "accepted"}}, # Not in oracle, ignored
    ]

    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    report = compile_metrics(
        results=results,
        oracle_path=csv_file,
        inference_time=12.34,
        output_report_json=report_json,
        output_report_md=report_md
    )

    metrics = report["metrics"]
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["inference_time_seconds"] == 12.34
    assert metrics["matched_with_oracle"] == 4
    assert metrics["total_evaluated"] == 5

    assert report_json.exists()
    assert report_md.exists()

    # Read md content and make sure headers are present
    md_content = report_md.read_text(encoding="utf-8")
    assert "# Evaluation Metrics Report" in md_content
    assert "| **True Positives (TP)** | 1 |" in md_content
