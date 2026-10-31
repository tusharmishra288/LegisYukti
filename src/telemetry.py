"""
Per-query latency instrumentation for the LegisYukti pipeline.

Answers one question: for a single user query, where did the wall-clock time go?

The pipeline makes between 6 and ~13 sequential LLM round-trips per legal query,
plus 2-3 hybrid Qdrant searches and a CPU cross-encoder rerank. Optimising it
without measurement is guesswork, so this module records:

  * per-node wall time (LangGraph nodes, via @timed_node)
  * every LLM call - model, duration, token counts - attributed to its node
  * retriever spans, nested, so rerank time can be separated from vector search
  * arbitrary manual spans (via `span("name")`)

Output is a human-readable summary in the log plus one JSON object per query
appended to logs/query_traces.jsonl, so timings can be aggregated across runs
rather than eyeballed one at a time.

Design notes:
  * Zero behaviour change. If no trace is active, every hook is a no-op, so the
    instrumented functions behave exactly as before.
  * Never raises. Instrumentation that can break the request path is worse than
    no instrumentation, so every hook is defensive.
  * Set LEGISYUKTI_TELEMETRY=0 to disable entirely.
"""

from __future__ import annotations

import os
import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Optional

from loguru import logger

from .config import LOG_DIR

ENABLED = os.getenv("LEGISYUKTI_TELEMETRY", "1").lower() not in ("0", "false", "no", "off")

TRACE_FILE = LOG_DIR / "query_traces.jsonl"

# Nodes run sequentially in the caller's thread, so a ContextVar correctly scopes
# the active trace to one query without leaking between concurrent Streamlit sessions.
_current_trace: ContextVar[Optional["QueryTrace"]] = ContextVar("legisyukti_trace", default=None)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class LLMCall:
    """One LLM round-trip."""
    model: str
    duration: float
    node: str
    node_order: int = 0  # which execution of that node, so retries stay distinguishable
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errored: bool = False
    in_retrieval: bool = False  # nested inside a retriever span (e.g. multi-query expansion)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "duration": round(self.duration, 3),
            "node": self.node,
            "node_order": self.node_order,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "errored": self.errored,
            "in_retrieval": self.in_retrieval,
        }


@dataclass
class NodeTiming:
    """Wall time for one LangGraph node execution (a node may run twice on retry)."""
    name: str
    duration: float
    order: int

    def as_dict(self) -> dict:
        return {"name": self.name, "duration": round(self.duration, 3), "order": self.order}


@dataclass
class SpanRecord:
    """A named block of work. `depth` > 0 means it ran inside another span.

    `kind` is "retriever" for spans derived from LangChain callbacks and "manual"
    for hand-placed ones. The two overlap in wall time - a manual span around a
    retriever call contains that retriever's own span - so only one kind is
    counted when computing totals.
    """
    name: str
    duration: float
    depth: int = 0
    node: str = "-"
    kind: str = "manual"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "duration": round(self.duration, 3),
            "depth": self.depth,
            "node": self.node,
            "kind": self.kind,
        }


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #

class QueryTrace:
    """Collects timings for a single user query."""

    def __init__(self, query: str):
        self.trace_id = uuid.uuid4().hex[:12]
        self.query = (query or "")[:300]
        self.started_at = time.time()
        self._t0 = time.perf_counter()
        self.total: float = 0.0

        self.nodes: list[NodeTiming] = []
        self.llm_calls: list[LLMCall] = []
        self.spans: list[SpanRecord] = []

        self.current_node: str = "-"
        self.current_node_order: int = 0
        self._node_counter = 0

    # -- node timing -------------------------------------------------------- #

    @contextmanager
    def node(self, name: str):
        previous, previous_order = self.current_node, self.current_node_order
        self._node_counter += 1
        order = self._node_counter
        self.current_node = name
        self.current_node_order = order
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.nodes.append(NodeTiming(name, time.perf_counter() - t0, order))
            self.current_node, self.current_node_order = previous, previous_order

    # -- collection --------------------------------------------------------- #

    def add_llm_call(self, call: LLMCall) -> None:
        self.llm_calls.append(call)

    def add_span(self, name: str, duration: float, depth: int = 0,
                 node: str = "-", kind: str = "manual") -> None:
        self.spans.append(SpanRecord(name, duration, depth, node, kind))

    def finish(self) -> "QueryTrace":
        self.total = time.perf_counter() - self._t0
        return self

    # -- derived metrics ---------------------------------------------------- #

    @property
    def llm_wall(self) -> float:
        return sum(c.duration for c in self.llm_calls)

    @property
    def retrieval_wall(self) -> float:
        """Top-level retrieval time, without double counting nested or manual spans.

        Prefers callback-derived retriever spans; falls back to manual spans if the
        callback never fired (e.g. callbacks not propagated into a nested invoke).
        """
        tops = [s for s in self.spans if s.depth == 0 and s.kind == "retriever"]
        if tops:
            return sum(s.duration for s in tops)
        return sum(s.duration for s in self.spans if s.depth == 0 and s.kind == "manual")

    @property
    def llm_in_retrieval(self) -> float:
        """LLM time that ran *inside* a retriever span.

        MultiQueryRetriever calls the LLM from within ContextualCompressionRetriever,
        so this time appears in both llm_wall and retrieval_wall. It must be
        subtracted once when reconciling against the query total.
        """
        return sum(c.duration for c in self.llm_calls if c.in_retrieval)

    @property
    def accounted(self) -> float:
        return self.llm_wall + self.retrieval_wall - self.llm_in_retrieval

    def by_model(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for c in self.llm_calls:
            e = out.setdefault(c.model, {"calls": 0, "duration": 0.0, "in": 0, "out": 0})
            e["calls"] += 1
            e["duration"] += c.duration
            e["in"] += c.prompt_tokens
            e["out"] += c.completion_tokens
        return out

    # -- rendering ---------------------------------------------------------- #

    def _pct(self, value: float) -> str:
        return f"{(value / self.total * 100):5.1f}%" if self.total > 0 else "    -"

    def summary(self) -> str:
        lines: list[str] = []
        add = lines.append

        add("")
        add("=" * 78)
        add(f" QUERY TRACE {self.trace_id}   total {self.total:6.2f}s")
        add(f" query: {self.query[:66]}")
        add("=" * 78)

        # Nodes, in execution order. A node appearing twice means the retry loop fired.
        if self.nodes:
            add(" NODE BREAKDOWN (execution order)")
            for n in sorted(self.nodes, key=lambda x: x.order):
                calls = [c for c in self.llm_calls if c.node_order == n.order]
                tag = f"{len(calls)} LLM" if calls else ""
                add(f"   {n.order:>2}. {n.name:<26} {n.duration:7.2f}s  {self._pct(n.duration)}  {tag}")
            seen: dict[str, int] = {}
            for n in self.nodes:
                seen[n.name] = seen.get(n.name, 0) + 1
            repeated = [k for k, v in seen.items() if v > 1]
            if repeated:
                add(f"   !! re-executed (retry loop): {', '.join(repeated)}")
            add("")

        # LLM calls, aggregated by model.
        if self.llm_calls:
            add(f" LLM CALLS: {len(self.llm_calls)} calls, {self.llm_wall:.2f}s wall ({self._pct(self.llm_wall).strip()})")
            for model, e in sorted(self.by_model().items(), key=lambda kv: -kv[1]["duration"]):
                add(
                    f"   {model:<28} x{e['calls']:<3} {e['duration']:7.2f}s"
                    f"   tok in/out {e['in']}/{e['out']}"
                )
            errors = [c for c in self.llm_calls if c.errored]
            if errors:
                add(f"   !! {len(errors)} call(s) errored")
            if self.llm_in_retrieval > 0:
                nested = sum(1 for c in self.llm_calls if c.in_retrieval)
                add(f"   of which {nested} call(s) run inside retrieval: {self.llm_in_retrieval:.2f}s")
            add("")

        # Retriever and manual spans, nested.
        if self.spans:
            add(f" RETRIEVAL / SPANS: {self.retrieval_wall:.2f}s at top level ({self._pct(self.retrieval_wall).strip()})")
            for s in self.spans:
                indent = "  " * s.depth
                marker = "" if s.kind == "retriever" else " *"
                add(f"   {indent}{s.name:<{max(4, 26 - len(indent))}} {s.duration:7.2f}s   [{s.node}]{marker}")
            if any(s.kind == "manual" for s in self.spans):
                add("   (* hand-placed span; overlaps the retriever spans it wraps)")

            # A top-level retriever span minus its deepest child is, in this pipeline,
            # dominated by the FlashRank cross-encoder. Labelled as an estimate because
            # it also absorbs multi-query fan-out overhead.
            tops = [s for s in self.spans if s.depth == 0 and s.kind == "retriever"]
            children = [s for s in self.spans if s.depth > 0 and s.kind == "retriever"]
            if tops and children:
                overhead = sum(s.duration for s in tops) - max(s.duration for s in children)
                if overhead > 0:
                    add(f"   ~ rerank + fan-out overhead (est.)   {overhead:7.2f}s")
            add("")

        # Whatever is left is Postgres checkpointing, Streamlit, and Python overhead.
        other = self.total - self.accounted
        if other < 0:
            # Should not happen now that nested LLM time is subtracted; if it does,
            # say so rather than printing a misleading negative.
            add(f" UNACCOUNTED: overlap detected ({other:.2f}s) - timings overlap, treat as ~0")
        else:
            add(f" UNACCOUNTED (checkpointing / app / overhead): {other:6.2f}s  {self._pct(other)}")
        add("=" * 78)
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "query": self.query,
            "total": round(self.total, 3),
            "llm_wall": round(self.llm_wall, 3),
            "retrieval_wall": round(self.retrieval_wall, 3),
            "llm_in_retrieval": round(self.llm_in_retrieval, 3),
            "accounted": round(self.accounted, 3),
            "llm_call_count": len(self.llm_calls),
            "nodes": [n.as_dict() for n in self.nodes],
            "llm_calls": [c.as_dict() for c in self.llm_calls],
            "spans": [s.as_dict() for s in self.spans],
            "by_model": {
                m: {"calls": e["calls"], "duration": round(e["duration"], 3),
                    "in": e["in"], "out": e["out"]}
                for m, e in self.by_model().items()
            },
        }


# --------------------------------------------------------------------------- #
# LangChain callback handler
# --------------------------------------------------------------------------- #

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:  # pragma: no cover - langchain always present in this app
    BaseCallbackHandler = object  # type: ignore[assignment, misc]


def _extract_model_name(serialized: Optional[dict], response: Any = None) -> str:
    """Best-effort model name. Providers disagree on where they put it."""
    for source in (
        lambda: (response.llm_output or {}).get("model_name"),
        lambda: (response.llm_output or {}).get("model"),
        lambda: (serialized or {}).get("kwargs", {}).get("model_name"),
        lambda: (serialized or {}).get("kwargs", {}).get("model"),
        lambda: (serialized or {}).get("name"),
    ):
        try:
            value = source()
            if value:
                return str(value)
        except Exception:
            continue
    return "unknown-model"


def _extract_token_usage(response: Any) -> tuple[int, int]:
    """Returns (prompt_tokens, completion_tokens); zeros when unavailable."""
    try:
        usage = (response.llm_output or {}).get("token_usage") or {}
        if usage:
            return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    except Exception:
        pass
    # Streaming responses carry usage on the message instead of llm_output.
    try:
        for generations in (response.generations or []):
            for gen in generations:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if meta:
                    return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    except Exception:
        pass
    return 0, 0


class TelemetryCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """Records LLM and retriever timings onto the active QueryTrace.

    Attached once via the LangGraph config, which propagates it to every nested
    LLM and retriever call - including the ones inside MultiQueryRetriever and
    ContextualCompressionRetriever that are otherwise invisible from the nodes.
    """

    raise_error = False  # never let instrumentation break the request path

    def __init__(self, trace: "QueryTrace"):
        self.trace = trace
        self._llm_runs: dict[Any, tuple[float, str, Optional[dict]]] = {}
        self._retriever_runs: dict[Any, tuple[float, str, Any]] = {}

    # -- LLM ---------------------------------------------------------------- #

    def _llm_start(self, serialized, run_id, **kwargs):
        try:
            self._llm_runs[run_id] = (
                time.perf_counter(),
                self.trace.current_node,
                self.trace.current_node_order,
                serialized,
                bool(self._retriever_runs),  # open retriever => this call is nested
            )
        except Exception:
            pass

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._llm_start(serialized, run_id, **kwargs)

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._llm_start(serialized, run_id, **kwargs)

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        try:
            started = self._llm_runs.pop(run_id, None)
            if started is None:
                return
            t0, node, node_order, serialized, in_retrieval = started
            prompt_tokens, completion_tokens = _extract_token_usage(response)
            self.trace.add_llm_call(LLMCall(
                model=_extract_model_name(serialized, response),
                duration=time.perf_counter() - t0,
                node=node,
                node_order=node_order,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                in_retrieval=in_retrieval,
            ))
        except Exception:
            pass

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        try:
            started = self._llm_runs.pop(run_id, None)
            if started is None:
                return
            t0, node, node_order, serialized, in_retrieval = started
            self.trace.add_llm_call(LLMCall(
                model=_extract_model_name(serialized),
                duration=time.perf_counter() - t0,
                node=node,
                node_order=node_order,
                errored=True,
                in_retrieval=in_retrieval,
            ))
        except Exception:
            pass

    # -- Retriever ---------------------------------------------------------- #

    def on_retriever_start(self, serialized, query, *, run_id=None, parent_run_id=None, **kwargs):
        try:
            name = (serialized or {}).get("name") or kwargs.get("name") or "retriever"
            self._retriever_runs[run_id] = (time.perf_counter(), str(name), parent_run_id)
        except Exception:
            pass

    def _retriever_end(self, run_id):
        try:
            started = self._retriever_runs.pop(run_id, None)
            if started is None:
                return
            t0, name, parent_run_id = started
            # Depth = how many ancestors are also retriever runs we are tracking.
            depth, cursor, guard = 0, parent_run_id, 0
            while cursor is not None and guard < 10:
                if cursor in self._retriever_runs:
                    depth += 1
                    cursor = self._retriever_runs[cursor][2]
                else:
                    break
                guard += 1
            self.trace.add_span(name, time.perf_counter() - t0, depth,
                                self.trace.current_node, kind="retriever")
        except Exception:
            pass

    def on_retriever_end(self, documents, *, run_id=None, **kwargs):
        self._retriever_end(run_id)

    def on_retriever_error(self, error, *, run_id=None, **kwargs):
        self._retriever_end(run_id)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def start_trace(query: str) -> Optional[QueryTrace]:
    """Begin a trace for one user query and make it the active one."""
    if not ENABLED:
        return None
    trace = QueryTrace(query)
    _current_trace.set(trace)
    return trace


def get_trace() -> Optional[QueryTrace]:
    return _current_trace.get()


def get_callbacks() -> list:
    """Callbacks to hand to graph.stream(...) so nested calls are captured."""
    trace = _current_trace.get()
    return [TelemetryCallbackHandler(trace)] if trace is not None else []


def finish_trace(trace: Optional[QueryTrace] = None) -> Optional[QueryTrace]:
    """Close the trace, log the summary, and append one JSON line for aggregation."""
    trace = trace or _current_trace.get()
    if trace is None:
        return None
    try:
        trace.finish()
        logger.info(trace.summary())
    except Exception as e:  # pragma: no cover
        logger.warning(f"telemetry: could not render summary: {e}")
    try:
        with open(TRACE_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.as_dict(), ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover
        logger.warning(f"telemetry: could not persist trace: {e}")
    finally:
        _current_trace.set(None)
    return trace


def timed_node(fn):
    """Decorator recording wall time for a LangGraph node. No-op without a trace."""
    @wraps(fn)
    def wrapper(state, *args, **kwargs):
        trace = _current_trace.get()
        if trace is None:
            return fn(state, *args, **kwargs)
        with trace.node(fn.__name__):
            return fn(state, *args, **kwargs)
    return wrapper


@contextmanager
def span(name: str):
    """Time an arbitrary block. No-op without a trace."""
    trace = _current_trace.get()
    if trace is None:
        yield
        return
    t0 = time.perf_counter()
    node = trace.current_node
    try:
        yield
    finally:
        trace.add_span(name, time.perf_counter() - t0, 0, node)
