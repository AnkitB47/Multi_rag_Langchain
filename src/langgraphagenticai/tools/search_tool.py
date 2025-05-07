from langchain_community.tools import DuckDuckGoSearchRun

def query_search(query):
    tool = DuckDuckGoSearchRun()
    return tool.run(query)
