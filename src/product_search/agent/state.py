from typing import List, Optional
from typing_extensions import TypedDict

from product_search.search_pipeline import SearchHit


class AgentState(TypedDict):
    original_query: str
    rewritten_query: str
    filter_source: Optional[str]     # "ESCI" | "WANDS" | None
    complexity: str                  # "simple" | "complex"
    hits: List[SearchHit]
    quality: Optional[str]           # good | low_coverage | wrong_category | semantic_drift | multi_constraint_miss
    action: Optional[str]            # return | widen | decompose
    iteration: int                   # starts 0; max 2 enforced by assess_results node
    k: int
    candidate_pool_size: int         # starts 25; doubled on "widen" (50 → 100)
