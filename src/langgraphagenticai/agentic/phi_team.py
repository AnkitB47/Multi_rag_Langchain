import os
import logging
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

# Load .env
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_phi_team():
    try:
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
            role="Fetches market data and stock insights",
            model=Groq(id="llama3-70b-8192"),
            tools=[YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True
            )],
            show_tool_calls=True,
            markdown=True
        )

        team = Agent(
            team=[web_agent, finance_agent],
            instructions=["Always include sources", "Use tables"],
            show_tool_calls=True,
            markdown=True
        )
        return team
    except Exception as e:
        logger.exception("Failed to initialize Phi team")
        raise e
