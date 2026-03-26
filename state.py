"""Graph state schema and initial state factory."""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages

from .config import MAIN_AGENT_MAX_TOOL_ROUNDS, REVIEWER_EXPLORATIVE_DEFAULT


# One snapshot per main-agent conclusion + reviewer round.
# Filled when main_agent produces a conclusion; reviewer_* set when reviewer runs.
CollabAttemptSnapshot = dict  # attempt_index, tools_called, conclusion, reviewer_approved?, reviewer_feedback?


class SessionState(TypedDict):
    messages: Annotated[list, add_messages]
    conversation_summary: str
    context_analyzer_result: Optional[dict]
    user_account: dict
    num_messages: int
    tool_calling_history: list
    file_system: dict
    current_query: str
    collab_attempts: list  # chronological rounds: tools+results, synthesis, reviewer
    review_count: int
    tool_calls_at_turn_start: int
    tool_calls_at_start_of_current_attempt: int
    reviewer_result: Optional[dict]
    last_review_approved: bool
    agent_tool_rounds_remaining: int
    # When True, reviewer uses a more search-demanding / explorative prompt.
    reviewer_explorative: bool


def make_initial_state() -> SessionState:
    return {
        "messages": [],
        "conversation_summary": "",
        "context_analyzer_result": None,
        "user_account": {"id": "user_001", "name": "Alice", "plan": "pro"},
        "num_messages": 0,
        "tool_calling_history": [],
        "file_system": {"docs/": [], "uploads/": []},
        "current_query": "",
        "collab_attempts": [],
        "review_count": 0,
        "tool_calls_at_turn_start": 0,
        "tool_calls_at_start_of_current_attempt": 0,
        "reviewer_result": None,
        "last_review_approved": True,
        "agent_tool_rounds_remaining": MAIN_AGENT_MAX_TOOL_ROUNDS,
        "reviewer_explorative": REVIEWER_EXPLORATIVE_DEFAULT,
    }
