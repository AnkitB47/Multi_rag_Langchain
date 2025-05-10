from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from langgraphagenticai.agentic.common import *

web_agent = Agent(
    name="Web Search Agent",
    role="Searches the web using DuckDuckGo",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Finance Agent",
    role="Gathers financial data via yFinance",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)
    ],
    show_tool_calls=True,
    markdown=True
)

phi_team = Agent(
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables"],
    show_tool_calls=True,
    markdown=True
)
