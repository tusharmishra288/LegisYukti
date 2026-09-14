"""
Assembling the user-visible answer out of a LangGraph message stream.

`graph.stream(..., stream_mode="messages")` yields token DELTAS, so the answer has
to be accumulated. Two things make that less trivial than it sounds:

  * Deltas from every node arrive on the same stream, not just the answer node.
  * When the evaluator sends the graph back for another attempt, the answer node
    runs a SECOND time in a later superstep. Those deltas are a replacement draft,
    not a continuation - concatenating them shows the user two answers glued
    together. `langgraph_step` distinguishes the attempts.

Shared by app.py and eval/run_eval.py so the UI and the measurements can never
disagree about what the answer was.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

# Nodes whose tokens are part of the answer the user reads.
ANSWER_NODES = ("generate_response", "final_answer")

# Internal bookkeeping text that must never reach the user.
NOISE_MARKERS = ("context stored", "penalty:")


def iter_answer(events: Iterable[tuple[Any, dict]]) -> Iterator[str]:
    """Yield the answer-so-far after each accepted delta.

    Each yielded value is the complete text of the CURRENT generation attempt, so
    a caller can render it directly. A retry restarts the text from empty.
    """
    parts: list[str] = []
    current_step: Any = None

    for chunk, metadata in events:
        metadata = metadata or {}
        if metadata.get("langgraph_node", "") not in ANSWER_NODES:
            continue

        text = getattr(chunk, "content", "") or ""
        if not text or any(marker in text.lower() for marker in NOISE_MARKERS):
            continue

        step = metadata.get("langgraph_step")
        if step != current_step:
            current_step, parts = step, []   # a retry supersedes the previous draft

        parts.append(text)
        yield "".join(parts)


def collect_answer(events: Iterable[tuple[Any, dict]]) -> str:
    """Consume a stream and return only the final answer text."""
    answer = ""
    for partial in iter_answer(events):
        answer = partial
    return answer.strip()


def final_answer_from_messages(messages: Iterable[Any]) -> str:
    """The delivered answer as stored in the checkpoint.

    verify_citations_node runs after synthesis and may REPLACE the answer, so the
    last non-empty AIMessage is what the user should end up seeing - not
    necessarily what was streamed.
    """
    try:
        from langchain_core.messages import AIMessage
    except ImportError:  # pragma: no cover
        return ""
    for message in reversed(list(messages or [])):
        if isinstance(message, AIMessage) and (getattr(message, "content", "") or "").strip():
            return message.content.strip()
    return ""
