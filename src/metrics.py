import csv
import json
from pathlib import Path
from typing import Any

def load_oracle(oracle_path: Path) -> dict[tuple[str, str], int]:
    """Loads Oracle.csv into a lookup dictionary of {(file_name.lower(), smell_type.lower()): oracle_value}."""
    oracle_dict = {}
    if not oracle_path.is_file():
        return oracle_dict
    
    # Mapping for heuristics names to Oracle names (e.g. God Class -> Large Class)
    smell_mapping = {
        "god class": "large class",
    }
    
    try:
        with oracle_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("Class_File", "").strip().lower()
                smell_type = row.get("Code_Smell", "").strip().lower()
                oracle_val = row.get("Oracle", "").strip()
                if file_name and smell_type and oracle_val != "":
                    # Map names if needed
                    smell_type = smell_mapping.get(smell_type, smell_type)
                    try:
                        oracle_dict[(file_name, smell_type)] = int(oracle_val)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Warning: Failed to load Oracle.csv: {e}")
    return oracle_dict

def compile_metrics(
    results: list[dict[str, Any]],
    oracle_path: Path,
    inference_time: float,
    output_report_json: Path,
    output_report_md: Path,
) -> dict[str, Any]:
    """Compares the aggregated results with the Oracle.csv ground truth, calculates metrics,

    and writes the report files.
    """
    oracle_dict = load_oracle(oracle_path)
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    matched_count = 0
    details = []

    smell_mapping = {
        "god class": "large class",
    }

    for item in results:
        smell_type = item.get("smell_type", "")
        smell = item.get("smell", {})
        file_name = smell.get("file_name", "")
        evaluation = item.get("evaluation", {})
        
        # Check verdict in evaluation status (or is accepted)
        # Note: if evaluation is None, prediction is rejected (0)
        status = evaluation.get("status") if evaluation else None
        prediction = 1 if status == "accepted" else 0
        
        # Match keys
        f_key = file_name.strip().lower()
        s_key = smell_type.strip().lower()
        s_key = smell_mapping.get(s_key, s_key)
        
        lookup_key = (f_key, s_key)
        if lookup_key in oracle_dict:
            oracle_val = oracle_dict[lookup_key]
            matched_count += 1
            
            classification = ""
            if prediction == 1 and oracle_val == 1:
                tp += 1
                classification = "TP"
            elif prediction == 1 and oracle_val == 0:
                fp += 1
                classification = "FP"
            elif prediction == 0 and oracle_val == 0:
                tn += 1
                classification = "TN"
            elif prediction == 0 and oracle_val == 1:
                fn += 1
                classification = "FN"
                
            details.append({
                "file_name": file_name,
                "smell_type": smell_type,
                "prediction": "accepted" if prediction == 1 else "rejected",
                "oracle": "present" if oracle_val == 1 else "absent",
                "classification": classification
            })

    # Calculations
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    report = {
        "metrics": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "inference_time_seconds": round(inference_time, 2),
            "total_evaluated": len(results),
            "matched_with_oracle": matched_count
        },
        "details": details
    }

    # Write JSON report
    try:
        with output_report_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save metrics JSON: {e}")

    # Write Markdown report
    try:
        md_lines = [
            "# Evaluation Metrics Report",
            "",
            "Metrics compiled against `data/Oracle.csv` ground truth data.",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **True Positives (TP)** | {tp} |",
            f"| **False Positives (FP)** | {fp} |",
            f"| **True Negatives (TN)** | {tn} |",
            f"| **False Negatives (FN)** | {fn} |",
            f"| **Precision** | {precision:.4f} |",
            f"| **Recall** | {recall:.4f} |",
            f"| **F1-Score** | {f1:.4f} |",
            f"| **Accuracy** | {accuracy:.4f} |",
            f"| **Inference Time** | {inference_time:.2f} seconds |",
            f"| **Total Processed Smells** | {len(results)} |",
            f"| **Smells Matched with Oracle** | {matched_count} |",
            "",
            "## Detailed Classifications",
            "",
            "| File Name | Code Smell | Prediction | Oracle | Classification |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]
        for detail in details:
            md_lines.append(
                f"| {detail['file_name']} | {detail['smell_type']} | {detail['prediction']} | {detail['oracle']} | **{detail['classification']}** |"
            )
            
        with output_report_md.open("w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
    except Exception as e:
        print(f"Warning: Failed to save metrics Markdown: {e}")

    return report
