"""Post-agent review loop routing and reviewer node."""

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from ..config import MAIN_AGENT_MAX_TOOL_ROUNDS, MAX_REVIEWS_PER_TURN, get_callbacks_config
from ..llm import llm
from ..prompts import REVIEWER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT_EXPLORATIVE
from ..state import SessionState
from ..utils import render_history_for_prompt, format_collab_attempts_for_prompt


def _search_tools_used_this_turn(state: SessionState) -> bool:
    start_idx = state.get("tool_calls_at_turn_start", 0)
    history = state.get("tool_calling_history", [])
    new_calls = history[start_idx:]
    search_tools = {"hybrid_search", "semantic_search", "keyword_search"}
    return any(h.get("tool") in search_tools for h in new_calls)


def route_main_after_agent(
    state: SessionState,
) -> Literal["tools", "reviewer", "stream_answer"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    search_used = _search_tools_used_this_turn(state)
    review_count = state.get("review_count", 0)
    last_approved = state.get("last_review_approved", True)

    if not search_used:
        return "stream_answer"
    if review_count == 0:
        return "reviewer"
    if last_approved or review_count >= MAX_REVIEWS_PER_TURN:
        return "stream_answer"
    return "reviewer"


def _scratch_from_collab_attempts(state: SessionState) -> str:
    """Chronological log: main agent KB work + synthesis + reviewer (this turn)."""
    attempts = state.get("collab_attempts") or []
    return format_collab_attempts_for_prompt(attempts)


@observe(name="reviewer")
def reviewer_node(state: SessionState) -> dict:
    user_query = state.get("current_query", "")
    explorative = bool(state.get("reviewer_explorative", False))
    conversation_history = render_history_for_prompt(
        state["messages"], current_user_turn_in_messages=True
    )
    scratch_display = _scratch_from_collab_attempts(state)
    if not scratch_display.strip():
        scratch_display = "(No attempts yet.)"

    hist_display = (
        conversation_history
        if conversation_history.strip() and conversation_history != "None"
        else "empty"
    )
    base_prompt = (
        REVIEWER_SYSTEM_PROMPT_EXPLORATIVE
        if explorative
        else REVIEWER_SYSTEM_PROMPT
    )
    prompt = (
        base_prompt
        + "\n\n## Recent conversation history (prior turns only)\n"
        + hist_display
        + "\n\n## User's current message\n"
        + user_query
        + "\n\n## What we tried so far (this turn, chronological)\n\n"
        + scratch_display
    )

    print("\n📝 [Reviewer] Evaluating main agent findings...")
    resp = llm.invoke([SystemMessage(content=prompt)], config=get_callbacks_config())
    raw = resp.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "approved": False,
            "feedback": " - Unable to parse reviewer output reliably. Ask the agent to double-check its reasoning and consider re-running key searches with clearer queries.",
        }

    approved = bool(result.get("approved", False))
    feedback = result.get("feedback", "").strip()
    new_review_count = state.get("review_count", 0) + 1

    # Update the last attempt in collab_attempts with this reviewer's decision
    attempts = list(state.get("collab_attempts") or [])
    out: dict = {
        "reviewer_result": result,
        "review_count": new_review_count,
        "last_review_approved": approved,
    }
    if attempts:
        last = dict(attempts[-1])
        last["reviewer_approved"] = approved
        last["reviewer_feedback"] = feedback
        out["collab_attempts"] = attempts[:-1] + [last]
    # When sending main_agent for a retry, set start index for next attempt's tool calls
    if not approved:
        out["tool_calls_at_start_of_current_attempt"] = len(state.get("tool_calling_history", []))
        out["agent_tool_rounds_remaining"] = MAIN_AGENT_MAX_TOOL_ROUNDS

    return out


def route_after_reviewer(state: SessionState) -> Literal["main_agent", "stream_answer"]:
    res = state.get("reviewer_result") or {}
    if bool(res.get("approved", False)):
        return "stream_answer"
    return "main_agent"
