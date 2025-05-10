from langchain_community.tools import DuckDuckGoSearchRun

def query_search(query: str) -> str:
    return DuckDuckGoSearchRun().run(query)
