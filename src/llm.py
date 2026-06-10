import litellm
from pydantic import BaseModel
from constants import TEMPERATURE, MAX_TOKENS

def unified_call(prompt: str, model: str, schema: type[BaseModel]) -> dict:
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
            temperature=TEMPERATURE,
            num_retries=3,
            max_tokens=MAX_TOKENS
        )
        
        if not isinstance(response, litellm.ModelResponse):
            raise TypeError("Streamed response received.")
        raw_content = response.choices[0].message.content

        if not isinstance(raw_content, str):
            raise ValueError(f"Model {model} returned an empty or invalid response.")

        return schema.model_validate_json(raw_content).model_dump()

    except Exception as e:
        print(f"Pipeline failed for model {model}: {e}")
        raise e