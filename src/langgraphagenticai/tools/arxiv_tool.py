from langchain_community.tools import ArxivQueryRun
from langchain.agents import Tool

def load_arxiv_tool():
    return Tool(
        name="Arxiv Search",
        func=ArxivQueryRun().run,
        description="Search academic papers from arXiv.org using keywords."
    )
