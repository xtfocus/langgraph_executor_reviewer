"""Shared message-history helpers."""

import json
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .config import PROMPT_HISTORY_ROLE_LABEL, PROMPT_HISTORY_WINDOW


def build_message_history(
    messages: list,
    window: int = 10,
    role_label: str = "Agent",
    debug_removed_types: bool = False,
) -> str:
    """Format recent conversation, excluding tool messages and tool-call-only AI messages."""
    # Important: apply the "window" after filtering.
    # Otherwise tool/internal messages can consume the window and push the latest
    # HumanMessage out of the slice, which makes the analyzer context look incomplete.
    selected: list[Any] = []
    for m in reversed(messages):
        cls_name = m.__class__.__name__

        # Exclude tool outputs from history context.
        if cls_name == "ToolMessage":
            continue

        # Exclude AI messages that only contain tool call instructions.
        if cls_name == "AIMessage" and hasattr(m, "tool_calls") and m.tool_calls:
            continue

        # Exclude internal synthesis that should not be shown to the analyzer.
        if cls_name == "AIMessage":
            add_kwargs = getattr(m, "additional_kwargs", {}) or {}
            internal = add_kwargs.get("internal", False) if isinstance(add_kwargs, dict) else False
            if internal:
                continue

        # Count only entries that would actually appear in the rendered history.
        if isinstance(m, HumanMessage):
            selected.append(m)
        elif isinstance(m, AIMessage) and m.content:
            selected.append(m)

        if len(selected) >= window:
            break

    filtered = list(reversed(selected))

    if debug_removed_types:
        selected_types = [m.__class__.__name__ for m in filtered]
        print(f"\n[History Debug] window={window} selected={selected_types}")

    history_lines = []
    for m in filtered:
        if isinstance(m, HumanMessage):
            history_lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            history_lines.append(f"{role_label}: {m.content}")

    return "\n".join(history_lines) if history_lines else "None"


def render_history_for_prompt(
    messages: list,
    window: int = PROMPT_HISTORY_WINDOW,
    role_label: str = PROMPT_HISTORY_ROLE_LABEL,
    current_user_turn_in_messages: bool = True,
) -> str:
    """
    Render conversation history for a prompt that *also* includes the current user message separately.

    Contract:
    - If `current_user_turn_in_messages=True`, exclude the last `HumanMessage` turn.
    - If `current_user_turn_in_messages=False`, render history as-is (do not exclude the last `HumanMessage`).
    """
    if current_user_turn_in_messages:
        last_human_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        prior = messages[:last_human_idx] if last_human_idx >= 0 else []
        return build_message_history(prior, window=window, role_label=role_label)

    # Current user turn isn't present yet; don't exclude anything.
    return build_message_history(messages, window=window, role_label=role_label)


def _turn_slice_from_last_human(messages: list) -> list:
    """Messages from last HumanMessage to end (current turn)."""
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    return messages[last_human_idx:] if last_human_idx >= 0 else messages[-20:]


def get_turn_tool_calls_with_results(messages: list) -> list[dict[str, Any]]:
    """
    From the current turn (last HumanMessage to end), collect (tool, args, result) in order.
    Used to slice by attempt via tool_calls_at_start_of_current_attempt.
    """
    turn = _turn_slice_from_last_human(messages)
    results_by_id: dict[str, str] = {}
    for m in turn:
        if isinstance(m, ToolMessage):
            tid = getattr(m, "tool_call_id", "") or ""
            content = getattr(m, "content", "") or ""
            # Some tool implementations may emit multiple ToolMessages per id.
            # Concatenate to avoid losing later chunks.
            if tid in results_by_id and results_by_id[tid]:
                results_by_id[tid] = results_by_id[tid] + "\n" + content
            else:
                results_by_id[tid] = content

    out: list[dict[str, Any]] = []
    for m in turn:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                result = results_by_id.get(tid, "<no result>")
                out.append({"tool": name, "args": args, "result": result})
    return out


def _fence_block_lines(text: str, indent: str = "     ") -> list[str]:
    """
    Markdown fenced block lines for prompt text. Grows fence length if `text` contains ```.
    """
    body = text if text is not None else ""
    fence = "```"
    while fence in body:
        fence = fence + "`"
    lines = [indent + fence]
    if body:
        for line in body.splitlines():
            lines.append(indent + line)
    lines.append(indent + fence)
    return lines


# Placeholder until main agent returns synthesis (no tool calls).
COLLAB_PENDING_CONCLUSION = (
    "(Pending — write internal synthesis from the KB results above.)"
)


def sync_collab_attempts_at_main_agent_start(state: dict) -> list[dict[str, Any]]:
    """
    After tool results arrive, ensure the current round exists in collab_attempts with
    tools_called filled and conclusion pending. Same round can get more tools before synthesis.
    After reviewer rejection, the next tool batch starts a new round.
    """
    attempts: list[dict[str, Any]] = [dict(a) for a in (state.get("collab_attempts") or [])]
    start_global = int(state.get("tool_calls_at_start_of_current_attempt") or 0)
    turn_start_global = int(state.get("tool_calls_at_turn_start") or 0)
    all_t = get_turn_tool_calls_with_results(state.get("messages") or [])
    # all_t is current-turn-local, while state indices are global over session history.
    # Convert to per-turn offset to avoid empty slices on later turns.
    start_local = max(0, start_global - turn_start_global)
    tools_slice = all_t[start_local:] if start_local <= len(all_t) else []

    if not tools_slice:
        return attempts

    def new_row(attempt_index: int) -> dict[str, Any]:
        return {
            "attempt_index": attempt_index,
            "tools_called": [dict(t) for t in tools_slice],
            "conclusion": COLLAB_PENDING_CONCLUSION,
            "reviewer_approved": None,
            "reviewer_feedback": "",
        }

    if not attempts:
        return [new_row(1)]

    last = attempts[-1]
    ra = last.get("reviewer_approved")
    conc = (last.get("conclusion") or "").strip()

    if ra is False:
        return attempts + [new_row(len(attempts) + 1)]

    if ra is None:
        if conc == COLLAB_PENDING_CONCLUSION or not conc:
            attempts[-1] = {
                **last,
                "tools_called": [dict(t) for t in tools_slice],
            }
        return attempts

    return attempts


def build_current_attempt_snapshot(
    messages: list,
    tool_calling_history: list,
    tool_calls_at_start_of_current_attempt: int,
    conclusion: str,
    tool_calls_at_turn_start: int = 0,
) -> dict[str, Any]:
    """
    Build one CollabAttemptSnapshot for the current attempt (tools + conclusion).
    reviewer_approved / reviewer_feedback left None/empty for reviewer to fill.
    """
    all_calls = get_turn_tool_calls_with_results(messages)
    # This attempt's calls are from start index to end of current turn's calls.
    start_local = max(0, tool_calls_at_start_of_current_attempt - tool_calls_at_turn_start)
    tools_called = all_calls[start_local:] if start_local <= len(all_calls) else all_calls
    return {
        "attempt_index": 0,  # caller sets from len(collab_attempts) + 1
        "tools_called": tools_called,
        "conclusion": conclusion or "<no conclusion>",
        "reviewer_approved": None,
        "reviewer_feedback": "",
    }


def format_collab_attempts_for_prompt(collab_attempts: list[dict]) -> str:
    """
    Temporal log for prompts: what the main agent and reviewer did this turn, in order.
    Per round: KB tool calls + results → main agent synthesis → reviewer (if any).
    """
    if not collab_attempts:
        return (
            "(Nothing recorded yet — call the KB tools, then results will appear here step by step.)"
        )
    lines: list[str] = [
        "Below is everything that happened this turn, **in time order** (oldest first). These are the paths we already attempted, which means we must not repeat them (don't try the same tool call with the same arguments).",
        "",
    ]
    for a in collab_attempts:
        idx = a.get("attempt_index", 0)
        lines.append(f"### Round {idx}")
        step = 1

        tools = a.get("tools_called") or []
        lines.append(f"{step}. **Main agent → KB search** (tool calls, then results)")
        step += 1
        if tools:
            for t in tools:
                args_str = json.dumps(t.get("args", {}), ensure_ascii=False)
                lines.append(f"   - `{t.get('tool', '?')}` {args_str}")
                lines.append("     →")
                lines.extend(_fence_block_lines(t.get("result") or ""))
        else:
            lines.append("   (no tool calls recorded for this round yet)")
        lines.append("")

        conc = a.get("conclusion", "") or ""
        if conc.strip() != COLLAB_PENDING_CONCLUSION:
            lines.append(f"{step}. **Main agent → internal synthesis**")
            step += 1
            lines.append(conc or "(none yet)")
            lines.append("")

        approved = a.get("reviewer_approved")
        feedback = (a.get("reviewer_feedback") or "").strip()
        if approved is True:
            lines.append(f"{step}. **Reviewer**")
            lines.append("   Approved.")
            lines.append("")
        elif approved is False and feedback:
            lines.append(f"{step}. **Reviewer**")
            lines.append(f"   Not approved. Feedback:\n   {feedback}")
            lines.append("")
        elif approved is False:
            lines.append(f"{step}. **Reviewer**")
            lines.append("   Not approved. (no feedback text)")
            lines.append("")

    return "\n".join(lines).strip()


def serialize_turn_for_agent_system_prompt(turn_messages: list) -> str:
    """
    Flatten one user-turn message slice for the main agent system prompt.
    Pairs each tool call with its output in one block (no duplicate tool name / id noise).
    """
    blocks: list[str] = []
    i = 0
    n = len(turn_messages)
    while i < n:
        m = turn_messages[i]
        if isinstance(m, HumanMessage):
            blocks.append("## User message\n" + (str(m.content) if m.content else ""))
            i += 1
            continue
        if isinstance(m, ToolMessage):
            blocks.append(
                "## Unmatched tool output\n"
                + (str(m.content) if m.content else "")
            )
            i += 1
            continue
        if isinstance(m, AIMessage):
            tcs = getattr(m, "tool_calls", None) or []
            if tcs:
                j = i + 1
                results_by_id: dict[str, str] = {}
                while j < n and isinstance(turn_messages[j], ToolMessage):
                    tm = turn_messages[j]
                    tid = getattr(tm, "tool_call_id", "") or ""
                    results_by_id[tid] = getattr(tm, "content", "") or ""
                    j += 1
                lines = ["## Tool calls and outputs"]
                for tc in tcs:
                    name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    tid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    args_str = json.dumps(args, ensure_ascii=False)
                    out = results_by_id.get(tid, "<no result>")
                    lines.append(f"Called: {name} {args_str}")
                    lines.append("Output:")
                    lines.append(str(out))
                    lines.append("")
                blocks.append("\n".join(lines).rstrip())
                i = j
            else:
                i += 1
            if m.content:
                blocks.append("## Assistant internal notes\n" + str(m.content))
            continue
        i += 1
    return "\n\n".join(blocks) if blocks else "## Turn context\n(empty)"
