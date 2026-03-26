"""LangChain tools backed by the simulated knowledge base."""

from langchain_core.tools import tool

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


@tool
def hybrid_search(query: str, keywords: list[str]) -> str:
    """Hybrid search combining semantic + keyword-style matching."""
    results = _kb_search(query, keywords)
    if not results:
        return "NO_RESULTS"
    out = [f"[Hybrid Search: '{query}']"]
    for i, (k, v, score) in enumerate(results[:3], 1):
        out.append(f"{i}. [{k}] (score:{score}) — {v}")
    result_str = "\n".join(out)
    print(f"   📄 Result:\n{result_str}")
    return result_str


@tool
def semantic_search(query: str) -> str:
    """Semantic-style search over KB text."""
    hits = []
    for k, v in SIMULATED_KB.items():
        if any(w in v.lower() for w in query.lower().split() if len(w) > 3):
            hits.append((k, v))
    if not hits:
        return "NO_RESULTS"
    out = [f"[Semantic Search: '{query}']"]
    for i, (k, v) in enumerate(hits[:3], 1):
        out.append(f"{i}. [{k}] — {v}")
    return "\n".join(out)


@tool
def keyword_search(keywords: list[str]) -> str:
    """Keyword search over KB keys and values."""
    tokens = []
    for kw in keywords:
        tokens.extend(kw.lower().split())
    hits = []
    for k, v in SIMULATED_KB.items():
        if any(tok in k.lower() or tok in v.lower() for tok in tokens if len(tok) > 2):
            hits.append((k, v))
    if not hits:
        return "NO_RESULTS"
    out = [f"[Keyword Search: {keywords}]"]
    for i, (k, v) in enumerate(hits[:3], 1):
        out.append(f"{i}. [{k}] — {v}")
    return "\n".join(out)


@tool
def ask_user_question(question: str, suggestions: list[str]) -> str:
    """
    Ask the user a clarifying question with grounded suggestions.
    Only call after searching when needed (NO_RESULTS or scoping).
    """
    print(f"\n🤖 {question}")
    for i, s in enumerate(suggestions, 1):
        print(f"   {i}. {s}")
    raw = input("   Enter number or type your own: ").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        answer = suggestions[idx] if 0 <= idx < len(suggestions) else raw
    else:
        answer = raw
    return f"User clarification: {answer}"


TOOLS = [hybrid_search, semantic_search, keyword_search]
