from langchain_community.tools import ArxivQueryRun
from langchain.agents import Tool

def load_arxiv_tool():
    arxiv_tool_instance = ArxivQueryRun()
    return Tool(
        name="Arxiv Search",
        func=arxiv_tool_instance.run,
        description="Search academic papers from arXiv.org using keywords."
    )
