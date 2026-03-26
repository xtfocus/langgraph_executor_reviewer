"""Environment-driven settings and optional Langfuse tracing."""

import os

# Langfuse (optional) — e.g. local Docker at http://localhost:3001
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3001")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")

langfuse_client = None
langfuse_handler = None

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL,
        )
        langfuse_client.auth_check()
        langfuse_handler = CallbackHandler()
    except Exception as e:
        print(f"⚠️  Failed to initialize Langfuse: {e}")
        langfuse_client = None
        langfuse_handler = None

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

MAX_REVIEWS_PER_TURN = 3

# Default reviewer mode. You can override per run by setting:
#   state["reviewer_explorative"] = True/False
REVIEWER_EXPLORATIVE_DEFAULT = True

# How many prior turns to include when rendering "conversation history" into prompts.
# Can be tuned via env for token/cost tradeoffs.
PROMPT_HISTORY_WINDOW = int(os.environ.get("PROMPT_HISTORY_WINDOW", "10"))

# Label used when formatting AI messages inside rendered history blocks.
# (Human messages are always shown as "User: ...".)
PROMPT_HISTORY_ROLE_LABEL = os.environ.get("PROMPT_HISTORY_ROLE_LABEL", "Assistant")

# Main agent may emit tools this many times per phase before a mandatory no-tools synthesis
# (phase = until reviewer; reset at context_analyzer and after each reviewer rejection).
MAIN_AGENT_MAX_TOOL_ROUNDS = int(os.environ.get("MAIN_AGENT_MAX_TOOL_ROUNDS", "2"))

# LangGraph: one step per node run. Per user turn that includes:
#   context_analyzer (1)
#   + main_agent ⇄ tools (2 steps per tool round; multi-hop adds up fast)
#   + reviewer (1) per review, and each rejection repeats main_agent ⇄ tools
#   + up to MAX_REVIEWS_PER_TURN reviews
#   + stream_answer + finalize
# Default leaves headroom; override with GRAPH_RECURSION_LIMIT if needed.
GRAPH_RECURSION_LIMIT = int(os.environ.get("GRAPH_RECURSION_LIMIT", "64"))


def get_callbacks_config():
    """Return callbacks config; includes Langfuse handler when configured."""
    if langfuse_handler:
        return {"callbacks": [langfuse_handler]}
    return {}
