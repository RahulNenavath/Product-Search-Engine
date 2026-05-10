import os
from langchain_google_genai import ChatGoogleGenerativeAI

_logged_models: set = set()  # print each (node, model) pair once per process


def get_agent_llm(node: str) -> ChatGoogleGenerativeAI:
    """
    node: "understand_query" → AGENT_LLM_QUERY  (default: gemini-3.1-flash-lite-preview)
          "assess_results"   → AGENT_LLM_ASSESS  (default: gemini-3-flash-preview)

    Requires langchain-google-genai >= 4.0.0
    thinking_level is a native top-level field — no generation_config workaround needed.

    Thinking levels for Gemini 3:
      gemini-3-flash-preview    → supports: low | medium | high  (default: high)
      gemini-3.1-flash-lite     → supports: minimal | low | medium | high (default: minimal)

    AGENT_THINKING=false → "low" on both nodes (fastest available for Flash)
    AGENT_THINKING=true  → "low" on understand_query (slightly more reasoning, still fast)
    """
    model_map = {
        "understand_query": os.getenv("AGENT_LLM_QUERY", "gemini-3.1-flash-lite-preview"),
        "assess_results":   os.getenv("AGENT_LLM_ASSESS", "gemini-3-flash-preview"),
    }
    if node not in model_map:
        raise ValueError(f"Unknown node '{node}'. Must be 'understand_query' or 'assess_results'.")

    model_name = model_map[node]
    thinking = os.getenv("AGENT_THINKING", "false").lower() == "true"

    # thinking_level is a native top-level kwarg in langchain-google-genai >= 4.0.0
    # "low" is the minimum for gemini-3-flash-preview; Flash-Lite also supports "minimal"
    if thinking and node == "understand_query":
        thinking_level = "low"
    else:
        # AGENT_THINKING=false: use lowest available level for each model
        is_flash_lite = "lite" in model_name
        thinking_level = "minimal" if is_flash_lite else "low"

    key = (node, model_name, thinking_level)
    if key not in _logged_models:
        print(f"[LLM] node={node!r}  model={model_name!r}  thinking_level={thinking_level!r}")
        _logged_models.add(key)

    return ChatGoogleGenerativeAI(
        model=model_name,
        project=os.getenv("GCP_PROJECT"),
        location=os.getenv("GCP_LLM_REGION", "global"),
        vertexai=True,
        thinking_level=thinking_level,
        # temperature intentionally omitted: langchain-google-genai >= 4.0.0
        # auto-sets it to 1.0 for Gemini 3+ per Google's best practices
    )