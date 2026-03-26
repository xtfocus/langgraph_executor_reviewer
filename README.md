# LangGraph RAG Agent

Refactored, modular version of `main_langgraph.py`: a **LangGraph** workflow that analyzes the user query, runs a **tool-using research agent** against a **simulated knowledge base**, optionally **reviews** retrieval-grounded reasoning, then **streams** the final answer.

## Flow

```
START → context_analyzer → main_agent ⇄ tools → reviewer? → stream_answer → finalize → END
```

- **Context analyzer**: Decides if KB search is needed, enriches query / keywords.
- **Main agent**: Calls `hybrid_search`, `semantic_search`, `keyword_search` (multi-hop supported). After **`MAIN_AGENT_MAX_TOOL_ROUNDS`** tool-using replies (default `4`, env override), the next visit uses **`llm` without tools** and must synthesize so the **reviewer** can run.
- **Reviewer**: When search tools were used, validates grounding vs tool outputs (retry loop, capped).
- **Stream answer**: User-facing reply; tokens streamed via LangGraph custom mode.
- **Finalize**: Bump turn counter; every 5 turns refreshes `conversation_summary`.

## Layout

| File / package | Role |
|----------------|------|
| `config.py` | `OPENAI_*`, Langfuse, `MAX_REVIEWS_PER_TURN`, `MAIN_AGENT_MAX_TOOL_ROUNDS` |
| `state.py` | `SessionState` TypedDict, `make_initial_state()` |
| `knowledge_base.py` | `SIMULATED_KB` — dummy documents for retrieval demos |
| `tools.py` | KB search tools + `ask_user_question` (not in default `TOOLS` list) |
| `llm.py` | Shared `ChatOpenAI` + `llm_with_tools` |
| `prompts.py` | System prompts for analyzer, agent, reviewer, answer |
| `nodes/context_analyzer.py` | Query enhancement node |
| `nodes/main_agent.py` | Research agent node |
| `nodes/reviewer.py` | Reviewer node + routing after agent / after reviewer |
| `nodes/stream_finalize.py` | Streamed final answer + summarization |
| `graph.py` | `build_graph()`, compiled `graph` |
| `main.py` | CLI (`/exit`, `/history`, `/summary`) |

## Requirements

Same stack as the parent project, e.g.:

- `langgraph`, `langchain-openai`, `langchain-core`
- `langfuse` (optional, for tracing)

Set at least:

```bash
export OPENAI_API_KEY="..."
# optional
export OPENAI_MODEL="gpt-4o"
```

Optional Langfuse:

```bash
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_BASE_URL="http://localhost:3001"
```

Graph step budget (default **64**): context + many tool loops + up to 3 reviewer rounds + stream/finalize. Raise if you still hit limits:

```bash
export GRAPH_RECURSION_LIMIT=100
```

## Run

From the parent directory (`agentic_sys`), with dependencies installed:

```bash
python -m langgraph_rag_agent
```

(equivalent: `python -m langgraph_rag_agent.main`)

**Wrong:** `python langgraph_rag_agent.main` — that looks for a *file* named `langgraph_rag_agent.main`, not a module.

Or programmatically:

```python
from langgraph_rag_agent import graph, make_initial_state

state = make_initial_state()
state["current_query"] = "Who is Priya Nair?"
# Optional: make reviewer more search-demanding (useful for multi-round review testing)
# state["reviewer_explorative"] = True
# graph.invoke(state, config) or graph.stream(..., stream_mode=["custom", "values"], version="v2")
```

## Extending

- **Real RAG**: Replace or wrap tools in `tools.py` to hit a vector store / API; keep the same tool names or update `llm.bind_tools(TOOLS)` and reviewer search-tool detection in `nodes/reviewer.py`.
- **Checkpointer**: In `graph.py`, pass a checkpointer to `compile()` for thread persistence.
- **Prompts**: Edit `prompts.py` only to tune behavior without touching graph wiring.
