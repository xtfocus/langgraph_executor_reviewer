"""User-facing streamed answer and periodic summarization."""

import json

from langchain_core.messages import AIMessage, SystemMessage
from langfuse import observe
from langgraph.config import get_stream_writer

from ..config import get_callbacks_config
from ..llm import llm
from ..prompts import FINALIZE_SUMMARY_PROMPT, STREAM_ANSWER_PROMPT
from ..state import SessionState
from ..utils import render_history_for_prompt


@observe(name="stream_answer")
def stream_answer_node(state: SessionState) -> dict:
    writer = get_stream_writer()

    agent_analysis = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not (
            hasattr(msg, "tool_calls") and msg.tool_calls
        ):
            agent_analysis = msg.content
            break

    user_query = state.get("current_query", "")
    prior_hist = render_history_for_prompt(
        state["messages"], current_user_turn_in_messages=True
    )
    if not prior_hist or prior_hist.strip() in ("", "None"):
        prior_display = "empty"
    else:
        prior_display = prior_hist

    user_account = json.dumps(state.get("user_account", {}), ensure_ascii=False)
    summary = (state.get("conversation_summary") or "").strip() or "None yet."

    research_block = (
        agent_analysis.strip()
        if agent_analysis.strip()
        else "None — no KB research block for this turn (answer from context, account, and summary)."
    )

    answer_body = f"""## User account
{user_account}

## Prior conversation summary
{summary}

## Conversation history (prior turns only)
{prior_display}

## Current question
{user_query}

## Internal research notes (not shown to end user; use for grounding when not "None")
{research_block}

## Your task
Answer the current question directly for the user. If research notes contain facts, ground your answer in them; otherwise rely on account, summary, and prior history appropriately."""

    merged_prompt = STREAM_ANSWER_PROMPT + "\n\n" + answer_body
    messages = [SystemMessage(content=merged_prompt)]

    full_content = ""
    for chunk in llm.stream(messages, config=get_callbacks_config()):
        if hasattr(chunk, "content") and chunk.content:
            full_content += chunk.content
            writer({"token": chunk.content})
    return {"messages": [AIMessage(content=full_content)]}


@observe(name="finalize")
def finalize_node(state: SessionState) -> dict:
    new_count = state["num_messages"] + 1
    updates: dict = {"num_messages": new_count}

    if new_count % 5 == 0:
        conversation_text = render_history_for_prompt(
            state["messages"],
            window=10,
            role_label="Agent",
            current_user_turn_in_messages=False,
        )
        summary_resp = llm.invoke(
            [
                SystemMessage(
                    content=FINALIZE_SUMMARY_PROMPT.format(
                        conversation_text=conversation_text
                    )
                )
            ],
            config=get_callbacks_config(),
        )
        updates["conversation_summary"] = summary_resp.content

    return updates
