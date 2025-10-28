from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
import json

# Define your schema (but enforcement may be limited in v1.7.0)
class CryptoSummary(BaseModel):
    summary: str
    sentiment: str

model = OpenAIChatModel("gpt-4o-mini")

agent = Agent[CryptoSummary](
    model,
    system_prompt=(
        "You are a crypto analyst. "
        "Return output **only as JSON** in this format: "
        '{"summary": "...", "sentiment": "Bullish/Bearish/Neutral"} '
        "based on the message given."
    ),
)

response = agent.run_sync("Bitcoin rose 3% while Ethereum fell 1% today.")
raw_output = response.output

print("\n🧠 Raw AI output:", raw_output)

try:
    data = json.loads(raw_output)
    summery = CryptoSummary(**data)
    print("✅ Parsed output:", summery)
    print("🧩 As dict:", summery.model_dump())
except (json.JSONDecodeError, ValidationError) as e:
    print("❌ Failed to parse output as CryptoSummary:", e)