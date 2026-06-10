import sys
import json
from constants import CHECKPOINT_PATH, MAX_TENTATIVES

def feedback_loop(items, process_func, judge_func):
    approved = []
    rejected = items
    i = 1

    while len(rejected) > 0 and i < MAX_TENTATIVES:
        # 1. Regenerate rejected items using feedback (returns 4-tuples or 5-tuples)
        temp = [process_func(item) for item in rejected]
        
        rejected = []
        # 2. Re-judge the newly generated items
        for item in temp:
            # judge_func returns the complete tuple: (*item, judgment_dict)
            full_judged_tuple = judge_func(item)
            judgment_dict = full_judged_tuple[-1] 
            
            if judgment_dict.get('verdict') == 'approved':
                approved.append(item)  # Stores the clean processed item
            else:
                # Keep the entire 5-tuple (or 6-tuple) for the next retry iteration
                rejected.append(full_judged_tuple) 
        i += 1

    return approved, rejected

def save_checkpoint(stage_name, approved_e, rejected_e, approved_p, rejected_p, output, code_path):
    cp_payload = {
        "code_path": str(code_path),
        "current_stage": stage_name,
        "approved_eval": approved_e,
        "rejected_eval": rejected_e,
        "approved_proposal": approved_p,
        "rejected_proposal": rejected_p,
        "output": output
    }
    try:
        tmp_cp = CHECKPOINT_PATH.with_name(f"{CHECKPOINT_PATH.name}.tmp")
        with tmp_cp.open("w", encoding="utf-8") as f:
            json.dump(cp_payload, f, indent=4, ensure_ascii=False)
        tmp_cp.replace(CHECKPOINT_PATH)
    except Exception as e:
        print(f"Warning: Failed to save progress checkpoint: {e}", file=sys.stderr)