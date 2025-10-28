from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
import json

class CryptoSummary(BaseModel):
    summary: str
    sentiment: str

model = OpenAIChatModel("gpt-4o-mini")

agent = Agent(
    model,
    system_prompt=(
        "You are crypto analyst. "
        "Return ONLY JSON like this: "
        '{"summary": "...", "sentiment": "Bullish/Bearish/Neutral"}. '
        "Do not add any other words."
    ),
)

def run_with_retry(text: str, schema):
    """Ask the AI, and retry if the JSON is invalid."""
    max_retries = 2

    for attempt in range(1, max_retries + 1):
        print(f"\nAttemtpt {attempt} for: {text}")
        response = agent.run_sync(text)
        raw_output = response.output.strip()
        print("🧠 Raw AI output:", raw_output)

        try:
            data = json.loads(raw_output)
            result = schema(**data)
            print("✅ Valid structured output:", result.model_dump())
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            print("⚠️ JSON parse failed:", e)


            if attempt < max_retries:
                print("🤖 Asking AI to fix invalid JSON...")
                fix_prompt = (
                    f"The following text is not valid JSON for {schema.__name__}: {raw_output}\n\n. "
                    f"Please reformat it as valid JSON followinf this structure:\n"
                    f'{{"summary": "...", "sentiment": "Bullish/Bearish/Neutral"}}'
                )
                text = fix_prompt
            else:
                print("❌ Could not repair after retries.")
                return None

if __name__ == "__main__":
    bad_prompt = "Bitcoin is on fire! Bulls are back. the sentiment is very positive."
    result = run_with_retry(bad_prompt, CryptoSummary)

    if result:
        print("\n🧩 Final structured dict:", result.model_dump())


