import os
import logging
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.tools.duckduckgo import DuckDuckGoTools

# Load env and setup logger
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_agno_team():
    try:
        agent = Agent(
            model=Groq(id="mixtral-8x7b-32768"),
            description="Reads PDFs and searches web",
            instructions=["Prioritize PDF content but use web if needed."],
            knowledge=PDFUrlKnowledgeBase(
                urls=["https://arxiv.org/pdf/2405.04231"],  # You can update dynamically if needed
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
        return agent
    except Exception as e:
        logger.exception("Failed to initialize Agno team")
        raise e
