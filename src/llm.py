import time
from collections import deque
import litellm
from pydantic import BaseModel

from constants import MAX_TOKENS, TEMPERATURE


class RateTracker:
    def __init__(self):
        self.request_times = deque()
        self.token_history = deque()
        self.limit_rps = 1.0
        self.limit_tpm = 50_000
        self.last_request_time = 0.0

    def update_limits(self, model: str):
        import os
        is_paid = os.environ.get("MISTRAL_PAID", "").lower() == "true"

        if "mistral-large" in model:
            self.limit_rps = 2.0 if is_paid else 0.07
            self.limit_tpm = 2_000_000 if is_paid else 250_000
        elif "mistral-medium" in model:
            self.limit_rps = 2.0 if is_paid else 0.83
            self.limit_tpm = 2_000_000 if is_paid else 25_000
        elif "ministral-3b" in model:
            self.limit_rps = 12.50
            self.limit_tpm = 1_300_000
        elif "ministral-8b" in model:
            self.limit_rps = 3.13
            self.limit_tpm = 625_000
        elif "open-mistral-nemo" in model:
            self.limit_rps = 2.0 if is_paid else 0.50
            self.limit_tpm = 2_000_000 if is_paid else 500_000
        elif "mistral-small" in model:
            self.limit_rps = 5.0 if is_paid else 0.83
            self.limit_tpm = 2_000_000 if is_paid else 50_000
        elif "gemini" in model:
            self.limit_rps = 5.0
            self.limit_tpm = 1_000_000
        else:
            self.limit_rps = 2.0 if is_paid else 1.0
            self.limit_tpm = 1_000_000 if is_paid else 50_000

    def record_request(self, tokens: int):
        now = time.time()
        self.request_times.append(now)
        self.token_history.append((now, tokens))
        self.last_request_time = now
        self.clean_history(now)

    def clean_history(self, now: float):
        while self.request_times and now - self.request_times[0] > 1.0:
            self.request_times.popleft()
        while self.token_history and now - self.token_history[0][0] > 60.0:
            self.token_history.popleft()

    def get_status(self) -> tuple[float, int]:
        now = time.time()
        self.clean_history(now)
        rps = len(self.request_times)
        tpm = sum(tokens for _, tokens in self.token_history)
        return float(rps), tpm


tracker = RateTracker()


def unified_call(prompt: str, model: str, schema: type[BaseModel]) -> dict:
    max_attempts = 5
    base_delay = 5.0
    
    if not model.startswith("ollama/"):
        tracker.update_limits(model)
        
        # Enforce RPS gap
        min_interval = 1.0 / tracker.limit_rps
        elapsed = time.time() - tracker.last_request_time
        if elapsed < min_interval:
            sleep_needed = min_interval - elapsed
            time.sleep(sleep_needed)
            
        # Enforce TPM limit
        # Estimate typical token consumption: prompt chars / 4 + 1000 reserve
        estimated_input = len(prompt) // 4
        while True:
            _, current_tpm = tracker.get_status()
            if current_tpm + estimated_input + 1000 >= tracker.limit_tpm:
                print(f"[MISTRAL RATE LIMIT] Near TPM threshold ({current_tpm:,}/{tracker.limit_tpm:,}). Pausing for 2.0s...")
                time.sleep(2.0)
            else:
                break
                
        model_name = model.split("/")[-1] if "/" in model else model
        current_rps, current_tpm = tracker.get_status()
        print(f"[MISTRAL API CALL] Model: {model_name} | RPS: {current_rps:.1f}/{tracker.limit_rps:.1f} | TPM (60s): {current_tpm:,}/{tracker.limit_tpm:,}")

    for attempt in range(max_attempts):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": schema,
                "temperature": TEMPERATURE,
                "num_retries": 3,
                "max_tokens": MAX_TOKENS,
            }
            if model.startswith("ollama/"):
                kwargs["num_ctx"] = 8192

            response = litellm.completion(**kwargs)

            if not isinstance(response, litellm.ModelResponse):
                raise TypeError("Streamed response received.")
            raw_content = response.choices[0].message.content

            if not isinstance(raw_content, str):
                raise ValueError(f"Model {model} returned an empty or invalid response.")

            # Record token usage to tracker
            if not model.startswith("ollama/"):
                recorded_tokens = 2000
                if hasattr(response, "usage") and response.usage:
                    recorded_tokens = getattr(response.usage, "total_tokens", 2000)
                tracker.record_request(recorded_tokens)

            return schema.model_validate_json(raw_content).model_dump()

        except litellm.exceptions.RateLimitError as e:
            if attempt == max_attempts - 1:
                print(f"Pipeline failed for model {model}: Rate limit exceeded after {max_attempts} attempts.")
                raise e
            sleep_time = base_delay * (2 ** attempt)
            if "mistral-large" in model:
                sleep_time = max(sleep_time, 15.0)
            print(f"Rate limit hit for model {model}. Retrying in {sleep_time} seconds (attempt {attempt + 1}/{max_attempts})...")
            # Clear or wait in tracker
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Pipeline failed for model {model}: {e}")
            raise e
