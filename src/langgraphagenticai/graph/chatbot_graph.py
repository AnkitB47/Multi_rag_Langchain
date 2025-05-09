from langgraph.graph import StateGraph
from langgraphagenticai.state.state import GraphState
from langgraphagenticai.nodes.node_runners import (
    run_query_pdf, run_query_image, run_query_search, run_translation
)

def create_graph():
    builder = StateGraph(GraphState)

    builder.add_node("query_pdf", run_query_pdf)
    builder.add_node("query_image", run_query_image)
    builder.add_node("query_search", run_query_search)
    builder.add_node("translate", run_translation)

    builder.set_entry_point("query_pdf")
    builder.add_edge("query_pdf", "query_image")
    builder.add_edge("query_image", "query_search")
    builder.add_edge("query_search", "translate")
    builder.set_finish_point("translate")

    return builder.compile()
