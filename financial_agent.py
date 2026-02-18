import os
from dotenv import load_dotenv
load_dotenv(override=True)

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

groq_api_key = os.getenv("GROQ_API_KEY")

agent = Agent(
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct", api_key=groq_api_key),
    tools=[
        DuckDuckGoTools(),
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        ),
    ],
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=False,
    markdown=True,
)

if __name__ == "__main__":
    import sys, io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=False)
