from agno.agent import Agent
from agno.models.groq import Groq
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.tools.duckduckgo import DuckDuckGoTools
from langgraphagenticai.agentic.common import *

agent = Agent(
    model=Groq(id="mixtral-8x7b-32768"),
    description="An agent that can read PDFs and search the web",
    instructions=["Prioritize PDF content but use web if needed."],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://arxiv.org/pdf/2405.04231"],  # example
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="multi_agent_knowledge",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)
