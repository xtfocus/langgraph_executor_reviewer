"""Main tool-using research agent."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse import observe

from ..config import MAIN_AGENT_MAX_TOOL_ROUNDS, get_callbacks_config
from ..llm import llm, llm_with_tools
from ..prompts import AGENT_FORCE_SYNTHESIS_SUFFIX, AGENT_SYSTEM_PROMPT
from ..state import SessionState
from ..utils import (
    COLLAB_PENDING_CONCLUSION,
    build_current_attempt_snapshot,
    render_history_for_prompt,
    format_collab_attempts_for_prompt,
    sync_collab_attempts_at_main_agent_start,
)


@observe(name="main_agent")
def agent_node(state: SessionState) -> SessionState:
    ctx = state.get("context_analyzer_result", {})
    ambiguity = ctx.get("ambiguity", {})
    enhanced = ctx.get("enhanced_query", {})
    requires_search = ctx.get("requires_search", True)
    is_ambig = ambiguity.get("query_is_ambiguous", False)
    suggestions = ambiguity.get("query_suggestions", [])
    keywords = enhanced.get("keywords", [])
    eq = enhanced.get("query", state["current_query"])

    if is_ambig and suggestions:
        instruction = (
            f'Enhanced query: "{eq}"\n'
            f"Ambiguous — distinct search angles:\n"
            + "\n".join(f"  - {s}" for s in suggestions)
            + "\n\nRun hybrid_search for EACH angle in parallel. Synthesize into one answer. "
            "If results require scoping, call ask_user_question with options grounded in results."
        )
    else:
        instruction = (
            f'Enhanced query: "{eq}"\n'
            f"Keywords: {keywords}\n"
            "Run hybrid_search. If result is partial and points to another entity needed "
            "to complete the answer, search that entity immediately (multi-hop). "
            "Keep chaining until you have the complete answer."
        )

    collab = sync_collab_attempts_at_main_agent_start(state)
    collab_history_str = format_collab_attempts_for_prompt(collab)
    print(
        f"\n[Main Agent Debug] messages={len(state.get('messages') or [])} "
        f"collab_attempts={len(collab)}"
    )
    preview = collab_history_str[:300].replace("\n", "\\n")
    print(f"[Main Agent Debug] collab_history_preview={preview}")

    system_base = AGENT_SYSTEM_PROMPT.format(
        user_account=json.dumps(state.get("user_account", {})),
        summary=state.get("conversation_summary", "No summary yet."),
    )

    last_human_idx = -1
    for i in range(len(state["messages"]) - 1, -1, -1):
        if isinstance(state["messages"][i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx >= 0:
        hm = state["messages"][last_human_idx]
        turn_context = "### User message\n" + (str(hm.content) if hm.content else "")
    else:
        turn_context = "### User message\n(empty)"

    # Recent conversation from prior turns only (excluding the current user message).
    recent_history = render_history_for_prompt(
        state["messages"], current_user_turn_in_messages=True
    )

    remaining = int(state.get("agent_tool_rounds_remaining", MAIN_AGENT_MAX_TOOL_ROUNDS))

    if remaining <= 0:
        combined_system = (
            system_base
            + "\n\n## Recent conversation history (prior turns only)\n\n"
            + (recent_history or "None")
            + "\n\n## Turn context\n\n"
            + turn_context
            + "\n\n## What we tried so far (this turn, chronological)\n\n"
            + collab_history_str
            + AGENT_FORCE_SYNTHESIS_SUFFIX
        )
        print("\n🤖 [Main Agent] Tool budget exhausted — synthesis only (no tools).")
        response = llm.invoke(
            [SystemMessage(content=combined_system)],
            config=get_callbacks_config(),
        )
    else:
        combined_system = (
            system_base
            + "\n\n## Recent conversation history (prior turns only)\n\n"
            + (recent_history or "None")
            + "\n\n## Turn context\n\n"
            + turn_context
            + "\n\n## Output of context analysis\n\n"
            + instruction
            + "\n\n## What we tried so far (this turn, chronological)\n\n"
            + collab_history_str
            + "\n\n## What to do next: Try something new or finalize/give up\n"
            + "The **What we tried so far** section are the paths we already explored, which means we must not repeat them.\n"
            "- Be **adaptive and exploratory**: if the current approach failed, change the query formulation, add missing identifiers, "
            "or pivot to a different plausible interpretation grounded in the user message.\n"
            "- If the latest round was not approved, follow the reviewer feedback shown in that round.\n\n"
            "If you still need KB facts, call the search tools. When done, respond with internal analysis only (no tool calls)."
            "What you must not do: repeat the same tool call with the same arguments in the section **What we tried so far**."
        )
        print("\n🤖 [Main Agent] Thinking...")
        response = llm_with_tools.invoke(
            [SystemMessage(content=combined_system)],
            config=get_callbacks_config(),
        )
        for tc in getattr(response, "tool_calls", []):
            print(f"   🔧 {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:110]})")
            state["tool_calling_history"].append({"tool": tc["name"], "args": tc["args"]})
        if getattr(response, "tool_calls", None):
            return {
                "messages": [response],
                "collab_attempts": collab,
                "agent_tool_rounds_remaining": remaining - 1,
            }

    out: dict = {"messages": [response], "collab_attempts": collab}

    # Conclusion: fill pending row created after tool results, or append (e.g. no tools used).
    final_collab = [dict(a) for a in collab]
    content = (response.content or "").strip()

    # Prevent internal main-agent synthesis from leaking into user-visible history.
    # Keep it in messages for downstream nodes (e.g. stream_answer), but tag it as internal.
    if isinstance(response, AIMessage) and not getattr(response, "tool_calls", None) and response.content:
        out["messages"] = [
            AIMessage(content=response.content, additional_kwargs={"internal": True})
        ]
    if final_collab and final_collab[-1].get("conclusion") == COLLAB_PENDING_CONCLUSION:
        final_collab[-1] = {
            **final_collab[-1],
            "conclusion": content or "(no content)",
        }
    else:
        tool_calls_at_start = state.get("tool_calls_at_start_of_current_attempt", 0)
        snapshot = build_current_attempt_snapshot(
            state["messages"],
            state.get("tool_calling_history", []),
            tool_calls_at_start,
            content,
            int(state.get("tool_calls_at_turn_start", 0)),
        )
        snapshot["attempt_index"] = len(final_collab) + 1
        final_collab.append(snapshot)
    out["collab_attempts"] = final_collab
    return out
