"""
nodes.py

LangGraph node implementations for the agentic search graph.

Nodes:
  understand_query_node  — rewrites query, extracts filters (gemini-3.1-flash-lite)
  retrieve_node          — runs hybrid_rerank via OpenSearchInference
  assess_results_node    — judges quality, decides action (gemini-3-flash-preview)
  decompose_node         — splits multi-constraint query, retrieves both, merges hits
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from product_search.search_pipeline import OpenSearchInference, SearchHit

from .llm import get_agent_llm
from .state import AgentState

if TYPE_CHECKING:
    pass

# ── Load prompts ──────────────────────────────────────────────────────────────

_PROMPTS_PATH = Path(__file__).parent / "prompts.toml"
with open(_PROMPTS_PATH, "rb") as _f:
    _PROMPTS = tomllib.load(_f)

_UQ = _PROMPTS["understand_query"]
_AR = _PROMPTS["assess_results"]
_HS = _PROMPTS["hits_summary_format"]


# ── Pydantic output schemas ────────────────────────────────────────────────────

class QueryUnderstanding(BaseModel):
    rewritten_query: str
    # Empty string means "no value" — avoids Optional[str]/null which Vertex AI protobuf rejects.
    filter_source: str = ""   # "ESCI" | "WANDS" | ""
    complexity: Literal["simple", "complex"]


class ResultAssessment(BaseModel):
    quality: Literal[
        "good",
        "low_coverage",
        "wrong_category",
        "semantic_drift",
        "multi_constraint_miss",
    ]
    action: Literal["return", "widen", "decompose"]
    reasoning: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_hits_summary(hits: List[SearchHit]) -> str:
    if not hits:
        return _HS["no_results"]
    top_n = _HS["top_n"]
    max_chars = _HS["snippet_max_chars"]
    lines = []
    for rank, hit in enumerate(hits[:top_n], start=1):
        product_class = hit.metadata.get("product_class", "Unknown")
        snippet = (hit.full_text or hit.encode_text or "")[:max_chars]
        lines.append(_HS["per_hit"].format(
            rank=rank, score=hit.score, product_class=product_class, snippet=snippet
        ))
    return "\n".join(lines)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def understand_query_node(state: AgentState) -> dict:
    llm = get_agent_llm("understand_query").with_structured_output(QueryUnderstanding)
    messages = [
        SystemMessage(content=_UQ["system"]),
        HumanMessage(content=_UQ["user_template"].format(raw_query=state["original_query"])),
    ]
    result: QueryUnderstanding = llm.invoke(messages)

    # Caller-provided filter_source takes precedence over LLM extraction.
    llm_source = result.filter_source.strip().upper() if result.filter_source else ""
    filter_source = state.get("filter_source") or (llm_source if llm_source in ("ESCI", "WANDS") else None)

    return {
        "rewritten_query": result.rewritten_query.strip() or state["original_query"],
        "filter_source": filter_source,
        "complexity": result.complexity,
    }


def retrieve_node(state: AgentState, inference: OpenSearchInference) -> dict:
    hits = inference.query_hybrid_rerank(
        query=state["rewritten_query"],
        k=state["k"],
        filter_source=state["filter_source"],
        candidate_pool_size=state["candidate_pool_size"],
        include_full_text=True,
    )
    return {
        "hits": hits,
        "iteration": state["iteration"] + 1,
    }


def assess_results_node(state: AgentState) -> dict:
    llm = get_agent_llm("assess_results").with_structured_output(ResultAssessment)
    hits_summary = _build_hits_summary(state["hits"])
    messages = [
        SystemMessage(content=_AR["system"]),
        HumanMessage(content=_AR["user_template"].format(
            raw_query=state["original_query"],
            rewritten_query=state["rewritten_query"],
            iteration=state["iteration"],
            hits_summary=hits_summary,
        )),
    ]
    result: ResultAssessment = llm.invoke(messages)
    return {
        "quality": result.quality,
        "action": result.action,
    }


def decompose_node(state: AgentState, inference: OpenSearchInference) -> dict:
    """
    Splits the rewritten query into two focused sub-queries using the LLM,
    retrieves each independently with hybrid_rerank, then merges by score.
    Used only when assess_results labels the action as 'decompose'.
    """
    from pydantic import BaseModel as _BM

    class _Decomposition(_BM):
        sub_query_1: str
        sub_query_2: str

    decompose_system = (
        "You are a search query decomposer. "
        "Split the following multi-constraint product query into exactly two focused sub-queries. "
        "Each sub-query should capture a distinct subset of the constraints. "
        "Return JSON: {\"sub_query_1\": string, \"sub_query_2\": string}"
    )
    llm = get_agent_llm("understand_query").with_structured_output(_Decomposition)
    messages = [
        SystemMessage(content=decompose_system),
        HumanMessage(content=f"Query: {state['rewritten_query']}"),
    ]
    decomp: _Decomposition = llm.invoke(messages)

    pool = state["candidate_pool_size"]
    hits_1 = inference.query_hybrid_rerank(
        query=decomp.sub_query_1, k=state["k"] * 2,
        filter_source=state["filter_source"], candidate_pool_size=pool, include_full_text=True,
    )
    hits_2 = inference.query_hybrid_rerank(
        query=decomp.sub_query_2, k=state["k"] * 2,
        filter_source=state["filter_source"], candidate_pool_size=pool, include_full_text=True,
    )

    # Merge: best score per product_id, then take top-k
    seen: dict[str, SearchHit] = {}
    for hit in hits_1 + hits_2:
        pid = hit.product_id
        if pid and (pid not in seen or hit.score > seen[pid].score):
            seen[pid] = hit
    merged = sorted(seen.values(), key=lambda h: h.score, reverse=True)[: state["k"]]

    return {"hits": merged}
