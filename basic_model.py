from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

model = OpenAIChatModel("gpt-4o-mini")

agent = Agent(model, system_prompt="You are a helpful assistant that summerizes text clearly.")

user_message = "Summerize: Bitcoiin and Etherium both rose today as traders reacted to ETF approval news."

response = agent.run_sync(user_message)

print("🧠 AI Summery:", response.output)

