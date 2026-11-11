"""
Assignment 3: the agent harness (`main.py ask "question"` runs `python agent.py "question"`).

LangChain `create_agent` on Gemini, with the tools from server.py over MCP and
AGENTS.md as the system prompt.

Guardrails
1. Approval: add_document never runs without a human "yes" (HumanInTheLoopMiddleware).
   In the terminal you are asked; with no terminal (evaluation, check.py) it is rejected.
2. Limits: at most 6 tool calls and 10 model calls per question.
3. Citation check (extra guardrail): every citation in the answer must match a
   passage that `search` actually returned in this run, and an answer that is not
   a refusal must cite at least one. On a violation the agent gets one chance to
   rewrite; if it still fails, the answer is withheld and replaced by the refusal.

`answer(question)` returns the answer plus the tool trace, tokens, time and
guardrail events, for evaluate.py.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import uuid
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-3.5-flash")
MAX_TOOL_CALLS = 6
MAX_MODEL_CALLS = 10

# Must be the exact sentence AGENTS.md tells the agent to use when the corpus
# does not answer the question (evaluate.py checks refusals with it).
REFUSAL = "The documents I have do not cover this."


# --------------------------------------------------------------------------
# Citation check (guardrail 3)
# --------------------------------------------------------------------------
RE_CITE = re.compile(r"\[\s*([a-z][a-z0-9_]*)\s*,\s*(Article|Annex|Page)\s+([0-9]+[a-z]?|[IVXLC]+)\b[^\]]*\]")
RE_RESULT = re.compile(r"^Result \d+ \[([^\]]+)\]", re.M)


def cited_passages(text: str) -> list[tuple[str, str]]:
    """[gdpr, Article 22(1)] -> ('gdpr', 'Article 22')."""
    return [(s, f"{k} {n}") for s, k, n in RE_CITE.findall(text)]


def retrieved_passages(tool_outputs: list[str]) -> set[tuple[str, str]]:
    found = set()
    for out in tool_outputs:
        for inner in RE_RESULT.findall(out):
            found.update(cited_passages(f"[{inner}]"))
    return found


def is_refusal(text: str) -> bool:
    return REFUSAL.lower().rstrip(".") in text.lower()


def check_citations(answer_text: str, tool_outputs: list[str], managing_documents: bool = False) -> list[str]:
    """Problems with the answer's citations; empty list = passes.
    A reply about adding a document (managing_documents) makes no legal claim,
    so it does not need a citation; any citation it does give is still checked."""
    cited = cited_passages(answer_text)
    retrieved = retrieved_passages(tool_outputs)
    problems = []
    if not cited and not is_refusal(answer_text) and not managing_documents:
        problems.append("the answer makes claims but cites no passage")
    for s, label in cited:
        if (s, label) not in retrieved:
            problems.append(f"[{s}, {label}] was not returned by the search tool in this conversation")
    return problems


# --------------------------------------------------------------------------
# MCP -> LangChain: each server tool becomes a LangChain tool that forwards
# the call over the open MCP session (name, description and argument schema
# come from server.py, so the model reads the same docstrings as your editor).
# --------------------------------------------------------------------------
def to_langchain_tool(session: ClientSession, tool) -> StructuredTool:
    async def call(**kwargs) -> str:
        result = await session.call_tool(tool.name, kwargs)
        return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    return StructuredTool(name=tool.name, description=tool.description or "",
                          args_schema=tool.input_schema, coroutine=call)


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
def build_agent(tools):
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0, api_key=load_api_key())
    return create_agent(
        model,
        tools,
        system_prompt=(ROOT / "AGENTS.md").read_text(encoding="utf-8"),
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on={"add_document": True}),
            ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS),
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS),
        ],
        checkpointer=InMemorySaver(),
    )


def ask_approval(request: dict, approve: bool | None) -> dict:
    """One decision per pending tool call. approve=None means ask in the terminal."""
    decisions = []
    for action in request.get("action_requests", []):
        if approve is None and sys.stdin.isatty():
            print(f"\nThe agent wants to run {action['name']} with {action['args']}", file=sys.stderr)
            ok = input("Approve? [y/N] ").strip().lower() in {"y", "yes", "s", "si", "sí"}
        else:
            ok = bool(approve)
        decisions.append({"type": "approve"} if ok else
                         {"type": "reject", "message": "The user did not approve adding this document."})
    return {"decisions": decisions}


async def _run_turn(agent, payload, config, approve):
    result = await agent.ainvoke(payload, config)
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        result = await agent.ainvoke(Command(resume=ask_approval(request, approve)), config)
    return result


def _used_tool(messages, name: str) -> bool:
    return any(tc["name"] == name for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls)


def _final_text(msg) -> str:
    content = msg.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return (content or "").strip()


async def answer_async(question: str, approve: bool | None = False) -> dict:
    start = time.time()
    server = StdioServerParameters(command=sys.executable, args=[str(ROOT / "server.py")], cwd=str(ROOT))
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = [to_langchain_tool(session, t) for t in (await session.list_tools()).tools]
        agent = build_agent(tools)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = await _run_turn(agent, {"messages": [HumanMessage(question)]}, config, approve)
        events = []
        tool_outputs = [str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
        text = _final_text(result["messages"][-1])
        managing = _used_tool(result["messages"], "add_document")
        problems = check_citations(text, tool_outputs, managing)

        if problems:                                   # guardrail 3: one rewrite
            events.append({"guardrail": "citation_check", "attempt": 1, "problems": problems})
            fix = ("Your answer failed the citation check: " + "; ".join(problems) + ". "
                   "Rewrite it using only passages returned by the search tool, cite every claim "
                   f"in the same [source, Article N] form, or reply exactly: {REFUSAL}")
            result = await _run_turn(agent, {"messages": [HumanMessage(fix)]}, config, approve)
            tool_outputs = [str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
            text = _final_text(result["messages"][-1])
            problems = check_citations(text, tool_outputs, managing)
            if problems:
                events.append({"guardrail": "citation_check", "attempt": 2, "problems": problems,
                               "withheld_answer": text})
                text = f"{REFUSAL} (An answer was drafted but withheld because its citations could not be verified.)"

    messages = result["messages"]
    trace = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                trace.append({"tool": tc["name"], "args": tc["args"]})
    usage = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    return {
        "question": question,
        "answer": text,
        "citations": sorted({f"[{s}, {l}]" for s, l in cited_passages(text)}),
        "refused": is_refusal(text),
        "tool_calls": trace,
        "retrieved": [inner for out in tool_outputs for inner in RE_RESULT.findall(out)],
        "retrieved_text": "\n\n".join(tool_outputs),
        "guardrail_events": events,
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "seconds": round(time.time() - start, 1),
    }


def answer(question: str, approve: bool | None = False) -> dict:
    return asyncio.run(answer_async(question, approve))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python agent.py "your question"')
    res = answer(" ".join(sys.argv[1:]), approve=None)   # None: ask in the terminal
    print(res["answer"])
    if res["citations"]:
        print("\nCitations: " + ", ".join(res["citations"]))
    for e in res["guardrail_events"]:
        print(f"\n[guardrail {e['guardrail']}, attempt {e['attempt']}] " + "; ".join(e["problems"]), file=sys.stderr)
    print(f"\n({len(res['tool_calls'])} tool calls, {res['input_tokens']}+{res['output_tokens']} tokens, "
          f"{res['seconds']} s)", file=sys.stderr)