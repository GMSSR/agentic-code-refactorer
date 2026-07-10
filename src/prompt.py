import json


def eval_prompt(type_smell: str, heuristics, smell, previous_eval=None, feedback=None) -> str:
    file_name = smell.get("file_name", "Unknown File")
    class_name = smell.get("class_name", "N/A")
    method_name = smell.get("method_name", "N/A")

    prompt = f"""ROLE
You are an expert software engineer specialized in code quality assessment. Your task is to critically evaluate whether a reported code smell is genuinely valid, using a structured set of heuristics as your evaluation criteria.

TASK 
You will receive:
  - The type of code smell reported
  - The affected code snippet and context
  - The class/method architecture location

Evaluate the smell step by step, analyzing each applicable heuristic below. Be critical: reject false positives. A smell is only valid if the heuristics clearly support its presence.

CONSTITUTION (Heuristics by smell type -- Junionello & de Mello, 2021)
{json.dumps(heuristics, indent=4, ensure_ascii=False)}

REPORTED SMELL
  Type: {type_smell}
  Location: {file_name} -- Class: {class_name} / Method: {method_name}
  Description: {smell.get("description", "")}
  Affected Code Context:
{smell.get("context", "")}
"""

    if previous_eval and feedback:
        prompt += f"""
PREVIOUS ATTEMPT & FEEDBACK
You previously attempted this evaluation, but it was rejected by an auditor. You must correct your course based on this info:
  - Previous Evaluator Attempt: {json.dumps(previous_eval, indent=4, ensure_ascii=False)}
  - Auditor Feedback: {json.dumps(feedback, indent=4, ensure_ascii=False)}
"""

    prompt += """
INSTRUCTIONS FOR YOUR STRUCTURED OBJECT
1. Populate the `heuristics` list: Provide a dedicated item for every single heuristic provided in the constitution. For each one, fill in its `name`, your explicit text `evaluation` (citing specific lines or patterns), and a strict `conclusion` choice ("met" or "unmet").
2. Keep details and explanations direct and concise (typically 2-3 sentences max per heuristic) to avoid output truncation.
3. Fill out the `summary` with a high-level technical overview of your findings.
4. Provide a clear `justification` detailing why the overall candidate is ultimately flagged or dismissed.
5. Set your final `status` value: Choose "accepted" if the code smell is genuinely present and valid, or "rejected" if it is a false positive.
"""
    return prompt


def j_eval_prompt(type_smell: str, heuristics, smell, evaluation) -> str:
    prompt = f"""ROLE
You are a senior software engineering auditor specialized in evaluating the quality of automated code smell assessments. Your task is NOT to re-evaluate the code smell itself, but to critically audit the reasoning and conclusion produced by a previous evaluator.

TASK
You will receive:
  - The type of code smell under analysis
  - The affected code snippet
  - The full output of a previous evaluator

Your job is to audit the evaluator's output by applying the rubric below. Be skeptical: identify flaws, inconsistencies, omissions, and unjustified conclusions.

AUDIT RUBRIC
  [R1 -- Heuristic coverage] Did the evaluator analyze ALL heuristics applicable?
  [R2 -- Justification groundedness] Did the evaluator cite specific evidence from the code?
  [R3 -- Verdict consistency] Is the final classification consistent with the heuristic verdicts?
  [R4 -- False positive awareness] Did the evaluator actively check if this is a false positive?
  [R5 -- Confidence calibration] Is the confidence level appropriate given the evidence?
  [R6 -- Logical coherence] Is the summary coherent and free of contradictions?

INPUTS
  Smell type: {type_smell}
  Location: {smell.get("file_name", "Unknown")} -- {smell.get("class_name", "N/A")} / {smell.get("method_name", "N/A")}
  Affected code: {smell.get("context", "")}
  Heuristics: {json.dumps(heuristics, indent=4, ensure_ascii=False)}

Evaluator output under audit: 
{json.dumps(evaluation, indent=4, ensure_ascii=False)} 

INSTRUCTIONS FOR YOUR STRUCTURED OBJECT
1. Fill out the `rubrics` object fields (R1 through R6) detailing your exact critique for each evaluation criterion.
2. If you find a critical error or inconsistency that invalidates the evaluation, document it inside `critical_flaw`. If no major issues exist, enter "None".
3. Set your final meta-`verdict`: Choose "approved" if the evaluator's reasoning is totally sound and trustworthy, or "rejected" if it contains significant errors.
4. Provide corrective `feedback` explaining what went wrong if you rejected the attempt.
"""
    return prompt


def ref_prompt(type_smell: str, heuristics, smell, evaluation, previous_proposal=None, feedback=None) -> str:
    prompt = f"""ROLE
You are a senior software engineer specialized in clean code and software refactoring. You will receive a validated code smell and its evaluation; you must propose a concrete refactoring to eliminate it while strictly preserving the original functional behavior.

VALIDATED SMELL
  Type: {type_smell}
  Location: {smell.get("file_name", "Unknown")} -- {smell.get("class_name", "N/A")} / {smell.get("method_name", "N/A")}
  Validation Summary: {evaluation.get("summary", "")}
  Affected Code: {smell.get("context", "")}

EVALUATION OF THE VALIDATED SMELL
  Heuristics Reference: {json.dumps(heuristics, indent=4, ensure_ascii=False)}
  Heuristics Analysis: {json.dumps(evaluation.get("heuristics", {}), indent=4, ensure_ascii=False)}
"""

    if previous_proposal and feedback:
        prompt += f"""
FEEDBACK ON PREVIOUS ATTEMPT
Your previous refactoring proposal was audited and flagged with issues. Use this feedback to deliver a corrected version:
  - Previous proposal: {json.dumps(previous_proposal, indent=4, ensure_ascii=False)}
  - Auditor Feedback: {feedback}
"""

    prompt += """
INSTRUCTIONS FOR YOUR STRUCTURED OBJECT
1. First, populate your `thought` key: Trace out the root cause of the smell, explicitly plan your steps, and choose the cleanest implementation strategy.
2. Inside `technique_chosen`, specify the formal design pattern or refactoring technique applied (e.g., "Extract Class", "Move Method").
3. Provide your complete, clean, functional Python syntax replacement block inside `proposed_code`.
4. Use `explanation` to walk through what structural aspects changed and why this successfully neutralizes the heuristics.
"""
    return prompt


def j_ref_prompt(type_smell: str, heuristics, smell, evaluation, ref) -> str:
    prompt = f"""ROLE
You are a senior software engineering auditor specialized in evaluating the quality of proposed source code refactorings. Your task is to critically audit the code correctness and suitability produced by a generator step.

TASK
You will receive the original code smell, the heuristic evaluation validations, and the full text of the proposed refactoring solution. Audit the proposal applying the rubric below.

AUDIT RUBRIC
  [R1 -- Heuristic coverage] Does the proposal fix all the issues pointed out by the original heuristics?
  [R2 -- Justification groundedness] Did the proposal provide clear technical reasoning for each structural change?
  [R3 -- Code correctness] Is the refactored code free of syntactical/structural flaws that would stop it from running?
  [R4 -- Free of Bugs] Does the code completely preserve the original system logic without introducing edge-case regressions or bugs?

INPUTS
  Smell type: {type_smell}
  Location: {smell.get("file_name", "Unknown")} -- {smell.get("class_name", "N/A")} / {smell.get("method_name", "N/A")}
  Validation Summary: {evaluation.get("summary", "")}
  Affected code: {smell.get("context", "")}
  Heuristics Evaluations: {json.dumps(evaluation.get("heuristics", {}), indent=4, ensure_ascii=False)}

Generator proposal under audit: 
{json.dumps(ref, indent=4, ensure_ascii=False)} 

INSTRUCTIONS FOR YOUR STRUCTURED OBJECT
1. Complete the fields inside the `rubrics` object (R1 through R4) with targeted feedback on the quality of the refactoring and its explanations.
2. If there are syntax issues, behavioral changes, or logic bugs, clearly describe them inside `critical_flaw`. Write "None" if the code is clean.
3. Select your strict structural `verdict`: "approved" if the implementation is safe to deploy, or "rejected" if it fails any rubric point.
4. Supply actionable instructions inside the `feedback` key if changes are required.
"""
    return prompt
