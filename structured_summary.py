from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

class CryptoSummary(BaseModel):
    summary: str
    sentiment: str

model = OpenAIChatModel("gpt-4o-mini")

agent = Agent(
    model,
    system_prompt=(
        "You are an AI analyst that summarizes crypto trends clearly "
        "and tags the market sentiment as Bullish, Bearish, or Neutral."
    ),
)

user_message = "Bitcoin went up 5% while Ethereum dropped 2% today."

response = agent.run_sync(user_message)

print("✅ Structured response:", response.output)
if isinstance(response.output, BaseModel):
    print("\n🧩 As dictionary:", response.output.model_dump())
else:
    print("\n⚠️ Model returned plain text:", response.output)
