"""Shared ChatOpenAI instances."""

from langchain_openai import ChatOpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL
from .tools import TOOLS

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY or None,
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOLS)
