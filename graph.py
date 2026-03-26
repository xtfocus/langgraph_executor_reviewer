"""Compile the LangGraph workflow."""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import (
    agent_node,
    context_analyzer_node,
    finalize_node,
    reviewer_node,
    route_after_reviewer,
    route_main_after_agent,
    stream_answer_node,
)
from .state import SessionState
from .tools import TOOLS


def build_graph():
    g = StateGraph(SessionState)

    g.add_node("context_analyzer", context_analyzer_node)
    g.add_node("main_agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("reviewer", reviewer_node)
    g.add_node("stream_answer", stream_answer_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "context_analyzer")
    g.add_edge("context_analyzer", "main_agent")
    g.add_conditional_edges(
        "main_agent",
        route_main_after_agent,
        ["tools", "reviewer", "stream_answer"],
    )
    g.add_edge("tools", "main_agent")
    g.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        ["main_agent", "stream_answer"],
    )
    g.add_edge("stream_answer", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=None, debug=False)


# Default compiled graph for import by main
graph = build_graph()
