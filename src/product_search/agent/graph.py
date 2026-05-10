"""
graph.py

LangGraph StateGraph for the agentic search pipeline.

Flow:
  understand_query → retrieve ──┬── (simple)  → END
                                └── (complex) → assess_results
                                                    ├── "return" / iteration >= 2  → END
                                                    ├── "widen"  → widen_pool → retrieve
                                                    └── "decompose"               → decompose → END

Simple queries skip assess_results entirely — saves one LLM call and avoids
the risk of assess_results making a bad retry decision on well-specified queries.

Public entry point:
  run_agent(query, k, inference, filter_source) -> (List[SearchHit], rewritten_query)
"""

from __future__ import annotations

from typing import List, Optional

from langgraph.graph import END, StateGraph

from product_search.search_pipeline import OpenSearchInference, SearchHit

from .nodes import (
    assess_results_node,
    decompose_node,
    retrieve_node,
    understand_query_node,
)
from .state import AgentState


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_retrieve(state: AgentState) -> str:
    """Simple queries skip assess_results — one LLM call instead of two."""
    if state["complexity"] == "simple":
        return "end"
    return "assess_results"


def _route_after_assess(state: AgentState) -> str:
    if state["iteration"] >= 2 or state["action"] == "return":
        return "end"
    if state["action"] == "widen":
        return "widen_pool"
    if state["action"] == "decompose":
        return "decompose"
    return "end"


def _widen_pool(state: AgentState) -> dict:
    return {"candidate_pool_size": state["candidate_pool_size"] * 2}


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph(inference: OpenSearchInference):
    g = StateGraph(AgentState)

    g.add_node("understand_query", understand_query_node)
    g.add_node("retrieve", lambda s: retrieve_node(s, inference))
    g.add_node("assess_results", assess_results_node)
    g.add_node("widen_pool", _widen_pool)
    g.add_node("decompose", lambda s: decompose_node(s, inference))

    g.set_entry_point("understand_query")
    g.add_edge("understand_query", "retrieve")

    # Simple → END directly; complex → assess_results
    g.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"end": END, "assess_results": "assess_results"},
    )

    g.add_conditional_edges(
        "assess_results",
        _route_after_assess,
        {"end": END, "widen_pool": "widen_pool", "decompose": "decompose"},
    )
    g.add_edge("widen_pool", "retrieve")
    g.add_edge("decompose", END)

    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(
    query: str,
    k: int,
    inference: OpenSearchInference,
    filter_source: Optional[str] = None,
    initial_candidate_pool_size: int = 25,
) -> tuple[List[SearchHit], str]:
    """
    Run the full agentic search pipeline for a single query.

    Returns:
        (hits, rewritten_query) — hits ranked by the reranker, rewritten_query for display.
    """
    graph = _build_graph(inference)
    initial_state: AgentState = {
        "original_query": query,
        "rewritten_query": "",
        "filter_source": filter_source,
        "complexity": "simple",
        "hits": [],
        "quality": None,
        "action": None,
        "iteration": 0,
        "k": k,
        "candidate_pool_size": initial_candidate_pool_size,
    }
    final_state = graph.invoke(initial_state)
    return final_state["hits"], final_state.get("rewritten_query", query)
