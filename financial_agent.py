from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import os 
from dotenv import load_dotenv
load_dotenv()

# Only GROQ_API_KEY is needed - it will be automatically picked up from environment

##Web search agent
web_search_agent=Agent(
    name="Web search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()], 
    instructions=["Always include source"],
    show_tool_calls=True,
    markdown=True,
)

##Financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_search_agent,finance_agent],
    instructions=["Always include sources", "Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Sumarize analysis recomendation and share the latest news for NVDA", stream=True)
    
