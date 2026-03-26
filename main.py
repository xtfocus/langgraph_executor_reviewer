"""CLI entry: interactive RAG agent loop."""

import json

from .config import (
    GRAPH_RECURSION_LIMIT,
    LANGFUSE_BASE_URL,
    get_callbacks_config,
    langfuse_handler,
)
from .graph import graph
from .state import make_initial_state


def main():
    print("=" * 60)
    print("  LangGraph RAG Agent — ToolNode + reviewer")
    print("  Type your query. Special: /exit  /history  /summary")
    print("=" * 60)

    if langfuse_handler:
        print(f"\n✅ Langfuse tracing enabled ({LANGFUSE_BASE_URL})")
    else:
        print("\n⚠️  Langfuse not configured. To enable:")
        print(f"   1. Create API key at {LANGFUSE_BASE_URL}")
        print("   2. Set environment variables:")
        print("      export LANGFUSE_PUBLIC_KEY='your-public-key'")
        print("      export LANGFUSE_SECRET_KEY='your-secret-key'")
        print("      export LANGFUSE_BASE_URL='http://localhost:3001' (optional)")
        print()

    print("\n📊 Graph structure:")
    print(graph.get_graph().draw_mermaid())

    state = make_initial_state()

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "/exit":
            print("👋 Goodbye!")
            break
        if user_input.lower() == "/history":
            print(f"\n📜 Tool history ({len(state['tool_calling_history'])} calls):")
            for i, h in enumerate(state["tool_calling_history"], 1):
                print(f"  {i}. {h['tool']} ← {json.dumps(h['args'])[:80]}")
            continue
        if user_input.lower() == "/summary":
            print(f"\n📝 {state['conversation_summary'] or 'No summary yet.'}")
            continue

        state["current_query"] = user_input
        state["collab_attempts"] = []
        state["review_count"] = 0
        state["tool_calls_at_turn_start"] = len(state.get("tool_calling_history", []))
        state["tool_calls_at_start_of_current_attempt"] = 0
        state["reviewer_result"] = None
        state["last_review_approved"] = True

        if langfuse_handler:
            langfuse_handler.session_id = f"session_{state['user_account']['id']}"

        graph_config = {"recursion_limit": GRAPH_RECURSION_LIMIT}
        graph_config.update(get_callbacks_config())

        try:
            print("\n🤖 Agent: ", end="", flush=True)
            final_state = None

            for chunk in graph.stream(
                state,
                graph_config,
                stream_mode=["custom", "values"],
                version="v2",
            ):
                if chunk["type"] == "custom":
                    if isinstance(chunk["data"], dict) and "token" in chunk["data"]:
                        print(chunk["data"]["token"], end="", flush=True)
                elif chunk["type"] == "values":
                    final_state = chunk["data"]

            print()
            if final_state:
                state = final_state

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
