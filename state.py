"""Graph state schema and initial state factory."""

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages

from .config import MAIN_AGENT_MAX_TOOL_ROUNDS, REVIEWER_EXPLORATIVE_DEFAULT


# One snapshot per main-agent conclusion + reviewer round.
# Filled when main_agent produces a conclusion; reviewer_* set when reviewer runs.
CollabAttemptSnapshot = dict  # attempt_index, tools_called, conclusion, reviewer_approved?, reviewer_feedback?

# Sentinel for `merge_turn_tool_calling_history`: replace the list with `right[1:]`
# (usually `[]`). This is needed because `turn_tool_calling_history` is updated via
# list-reducer semantics (append), so we still need a safe "clear" mechanism at the
# start of each user turn.
RESET_TURN_TOOL_HISTORY = "__lg_reset_turn_tool_history__"


def merge_turn_tool_calling_history(
    left: list | None,
    right: list | None,
) -> list:
    """Merge turn-scoped tool history updates.

    - Normal update: append new rows from `right` onto existing `left`.
    - Clear update: when `right` begins with `RESET_TURN_TOOL_HISTORY`, return
      an empty list (or `right[1:]` if present).

    This lets `context_analyzer_node` reset turn-scoped history without needing
    a separate cursor or message-history slicing.
    """
    left = left or []
    right = right or []
    if right and right[0] == RESET_TURN_TOOL_HISTORY:
        return list(right[1:])
    return left + list(right)


class SessionState(TypedDict):
    messages: Annotated[list, add_messages]
    conversation_summary: str
    context_analyzer_result: Optional[dict]
    user_account: dict
    num_messages: int
    tool_calling_history: Annotated[list, operator.add]
    file_system: dict
    current_query: str
    collab_attempts: list  # chronological rounds: tools+results, synthesis, reviewer
    review_count: int
    tool_calls_at_start_of_current_attempt: int
    reviewer_result: Optional[dict]
    last_review_approved: bool
    agent_tool_rounds_remaining: int
    # When True, reviewer uses a more search-demanding / explorative prompt.
    reviewer_explorative: bool
    # Current user turn only: KB tool calls + results (filled by Command-returning tools).
    turn_tool_calling_history: Annotated[list, merge_turn_tool_calling_history]


def make_initial_state() -> SessionState:
    return {
        "messages": [],
        "conversation_summary": "",
        "context_analyzer_result": None,
        "user_account": {"id": "user_001", "name": "Alice", "plan": "pro"},
        "num_messages": 0,
        "tool_calling_history": [],
        "turn_tool_calling_history": [],
        "file_system": {"docs/": [], "uploads/": []},
        "current_query": "",
        "collab_attempts": [],
        "review_count": 0,
        "tool_calls_at_start_of_current_attempt": 0,
        "reviewer_result": None,
        "last_review_approved": True,
        "agent_tool_rounds_remaining": MAIN_AGENT_MAX_TOOL_ROUNDS,
        "reviewer_explorative": REVIEWER_EXPLORATIVE_DEFAULT,
    }
