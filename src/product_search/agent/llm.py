import os
from langchain_google_genai import ChatGoogleGenerativeAI


def get_agent_llm(node: str) -> ChatGoogleGenerativeAI:
    """
    node: "understand_query" → AGENT_LLM_QUERY  (default: gemini-3.1-flash-lite-preview)
          "assess_results"   → AGENT_LLM_ASSESS  (default: gemini-3-flash-preview)

    Uses the Vertex AI backend (vertexai=True) so ADC / GCP_PROJECT auth is preserved.
    Gemini 3 preview models require location="global".
    thinking_budget is deprecated for Gemini 3; use thinking_level instead.
    """
    model_map = {
        "understand_query": os.getenv("AGENT_LLM_QUERY", "gemini-3.1-flash-lite-preview"),
        "assess_results":   os.getenv("AGENT_LLM_ASSESS", "gemini-3-flash-preview"),
    }
    if node not in model_map:
        raise ValueError(f"Unknown node '{node}'. Must be 'understand_query' or 'assess_results'.")

    model_name = model_map[node]
    thinking = os.getenv("AGENT_THINKING", "false").lower() == "true"

    model_kwargs: dict = {}
    if thinking and node == "understand_query":
        model_kwargs["thinking_level"] = "low"

    return ChatGoogleGenerativeAI(
        model=model_name,
        project=os.getenv("GCP_PROJECT"),
        location=os.getenv("GCP_LLM_REGION", "global"),
        temperature=0.0,
        vertexai=True,
        **model_kwargs,
    )
