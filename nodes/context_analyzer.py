"""Query enhancement and KB-search necessity."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from langgraph.types import Command

from ..config import MAIN_AGENT_MAX_TOOL_ROUNDS, get_callbacks_config
from ..llm import llm
from ..prompts import CONTEXT_ANALYZER_PROMPT
from ..state import SessionState
from ..utils import render_history_for_prompt


@observe(name="context_analyzer")
def context_analyzer_node(state: SessionState) -> Command:
    query = state["current_query"]

    # Render prior conversation history only; the prompt includes `{query}` separately.
    history_str = render_history_for_prompt(
        state["messages"], current_user_turn_in_messages=False
    )

    print(f'\n🔍 [Context Analyzer] Evaluating: "{query}"')

    resp = llm.invoke(
        [SystemMessage(content=CONTEXT_ANALYZER_PROMPT.format(query=query, history=history_str))],
        config=get_callbacks_config(),
    )

    raw = resp.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "enhanced_query": {"query": query, "keywords": query.split()},
            "requires_search": True,
            "ambiguity": {"query_is_ambiguous": False, "query_suggestions": []},
        }

    eq = result.get("enhanced_query", {})
    keywords = eq.get("keywords", [])
    requires_search = result.get("requires_search", True)
    is_ambig = result.get("ambiguity", {}).get("query_is_ambiguous", False)

    print(f'   ✏️  Enhanced query : "{eq.get("query", query)}"')
    if not requires_search:
        print("   ✨ No KB search needed — will answer directly")
    elif is_ambig:
        print(f"   ⚠️  Ambiguous — search angles: {result['ambiguity'].get('query_suggestions', [])}")
    else:
        print(f"   🏷️  Keywords       : {keywords}")

    update = {
        "context_analyzer_result": result,
        "messages": [HumanMessage(content=query)],
        "tool_calls_at_start_of_current_attempt": len(state.get("tool_calling_history", [])),
        "agent_tool_rounds_remaining": MAIN_AGENT_MAX_TOOL_ROUNDS,
    }

    # Use Command to decide the next node immediately (skip main_agent when no KB search is needed).
    goto = "main_agent" if requires_search else "stream_answer"
    return Command(update=update, goto=goto)
