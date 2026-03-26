"""LangChain tools backed by the simulated knowledge base."""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from .knowledge_base import SIMULATED_KB


def _kb_search(query: str, keywords: list[str]) -> list[tuple]:
    """Scored matching; threshold filtered."""
    q_tokens = [w.lower() for w in query.split() if len(w) > 3]
    kw_tokens = []
    for kw in keywords:
        kw_tokens.extend([t.lower() for t in kw.split() if len(t) > 2])

    results = []
    for k, v in SIMULATED_KB.items():
        text = (k + " " + v).lower()
        q_hits = sum(1 for t in q_tokens if t in text)
        kw_hits = sum(1 for t in kw_tokens if t in text)
        total_tokens = len(q_tokens) + len(kw_tokens)
        if total_tokens == 0:
            continue
        score = round((q_hits * 0.4 + kw_hits * 0.6) / max(len(q_tokens), 1), 3)
        exact_kw_match = any(kw.lower() in text for kw in keywords)
        if (q_hits + kw_hits >= 2) or exact_kw_match:
            results.append((k, v, score))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def _kb_tool_command(
    *,
    tool_name: str,
    args: dict,
    result_str: str,
    tool_call_id: str,
) -> Command:
    """Build a Command update for a KB search tool.

    This is what enables prompt rendering to use state instead of reconstructing
    tool-call/result pairs from `state["messages"]`.

    The update does 3 things:
    - emits a `ToolMessage` so the LLM receives the tool output
    - appends `{tool, args, result}` into `turn_tool_calling_history`
    - appends the same row into `tool_calling_history` for CLI `/history`
    """
    row = {"tool": tool_name, "args": args, "result": result_str}
    return Command(
        update={
            "messages": [ToolMessage(content=result_str, tool_call_id=tool_call_id)],
            "turn_tool_calling_history": [row],
            "tool_calling_history": [row],
        }
    )


@tool
def hybrid_search(
    query: str,
    keywords: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Hybrid search combining semantic + keyword-style matching."""
    results = _kb_search(query, keywords)
    if not results:
        return _kb_tool_command(
            tool_name="hybrid_search",
            args={"query": query, "keywords": keywords},
            result_str="NO_RESULTS",
            tool_call_id=tool_call_id,
        )
    out = [f"[Hybrid Search: '{query}']"]
    for i, (k, v, score) in enumerate(results[:3], 1):
        out.append(f"{i}. [{k}] (score:{score}) — {v}")
    result_str = "\n".join(out)
    print(f"   📄 Result:\n{result_str}")
    return _kb_tool_command(
        tool_name="hybrid_search",
        args={"query": query, "keywords": keywords},
        result_str=result_str,
        tool_call_id=tool_call_id,
    )


@tool
def semantic_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Semantic-style search over KB text."""
    hits = []
    for k, v in SIMULATED_KB.items():
        if any(w in v.lower() for w in query.lower().split() if len(w) > 3):
            hits.append((k, v))
    if not hits:
        return _kb_tool_command(
            tool_name="semantic_search",
            args={"query": query},
            result_str="NO_RESULTS",
            tool_call_id=tool_call_id,
        )
    out = [f"[Semantic Search: '{query}']"]
    for i, (k, v) in enumerate(hits[:3], 1):
        out.append(f"{i}. [{k}] — {v}")
    result_str = "\n".join(out)
    return _kb_tool_command(
        tool_name="semantic_search",
        args={"query": query},
        result_str=result_str,
        tool_call_id=tool_call_id,
    )


@tool
def keyword_search(
    keywords: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Keyword search over KB keys and values."""
    tokens = []
    for kw in keywords:
        tokens.extend(kw.lower().split())
    hits = []
    for k, v in SIMULATED_KB.items():
        if any(tok in k.lower() or tok in v.lower() for tok in tokens if len(tok) > 2):
            hits.append((k, v))
    if not hits:
        return _kb_tool_command(
            tool_name="keyword_search",
            args={"keywords": keywords},
            result_str="NO_RESULTS",
            tool_call_id=tool_call_id,
        )
    out = [f"[Keyword Search: {keywords}]"]
    for i, (k, v) in enumerate(hits[:3], 1):
        out.append(f"{i}. [{k}] — {v}")
    result_str = "\n".join(out)
    return _kb_tool_command(
        tool_name="keyword_search",
        args={"keywords": keywords},
        result_str=result_str,
        tool_call_id=tool_call_id,
    )


TOOLS = [hybrid_search, semantic_search, keyword_search]
