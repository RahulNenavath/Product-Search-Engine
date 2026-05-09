"""
ui/app.py

Streamlit frontend for the Product Search API.
Calls the FastAPI backend at API_BASE_URL (default: http://api:8000).
"""

import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

st.set_page_config(
    page_title="Product Search",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Search Settings")

    search_type = st.radio(
        "Search mode",
        options=["hybrid", "bm25", "hnsw", "hybrid_rerank"],
        format_func=lambda x: {
            "hybrid": "🔀 Hybrid (BM25 + HNSW)",
            "bm25": "📝 BM25 (Keyword)",
            "hnsw": "🧠 HNSW (Semantic)",
            "hybrid_rerank": "🏆 Hybrid + Reranker (best quality)",
        }[x],
        index=0,
    )

    if search_type == "hybrid_rerank":
        st.info(
            "**Cross-encoder reranking** scores every candidate with `BAAI/bge-reranker-v2-m3` "
            "running on CPU. Expect **30–90 s** per query depending on candidate pool size.",
            icon="⏳",
        )

    k = st.slider("Results to return (k)", min_value=1, max_value=50, value=10)

    source_filter = st.selectbox(
        "Filter by source",
        options=["All", "ESCI", "WANDS"],
        index=0,
    )

    st.divider()

    # Health check
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if resp.ok:
            info = resp.json()
            st.success("API: connected")
            cluster = info.get("opensearch", {}).get("cluster_name", "—")
            st.caption(f"OpenSearch cluster: `{cluster}`")
        else:
            st.error(f"API returned {resp.status_code}")
    except Exception as e:
        st.error(f"API unreachable: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

st.title("🔍 Product Search")
st.caption("Powered by OpenSearch · BM25 · HNSW · Hybrid RRF · Cross-encoder Reranking")

query = st.text_input(
    label="Search query",
    placeholder='e.g. "red running shoes for women"',
    label_visibility="collapsed",
)

search_clicked = st.button("Search", type="primary", use_container_width=True)

if search_clicked and query.strip():
    payload: dict = {
        "query": query.strip(),
        "k": k,
    }
    if source_filter != "All":
        payload["filter_source"] = source_filter

    if search_type == "hybrid_rerank":
        timeout = 300
    elif search_type in ("hnsw", "hybrid"):
        timeout = 120  # Cloud Run cold-start can take 30-90s
    else:
        timeout = 15

    if search_type == "hybrid_rerank":
        with st.status("Reranking in progress — this can take up to 90 s on CPU…", expanded=True) as status:
            st.write("Retrieving BM25 and HNSW candidates…")
            st.write("Running cross-encoder reranking on CPU (please wait)…")
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/search/{search_type}",
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                status.update(label="Reranking complete!", state="complete", expanded=False)
            except requests.exceptions.HTTPError as e:
                status.update(label="Search failed", state="error", expanded=False)
                st.error(f"Search failed ({e.response.status_code}): {e.response.text}")
                st.stop()
            except Exception as e:
                status.update(label="Request error", state="error", expanded=False)
                st.error(f"Request error: {e}")
                st.stop()
    else:
        with st.spinner(f"Searching via {search_type.upper()}… (first request may take up to 90 s if the embedding service is cold)" if search_type in ("hnsw", "hybrid") else f"Searching via {search_type.upper()}…"):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/search/{search_type}",
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.HTTPError as e:
                st.error(f"Search failed ({e.response.status_code}): {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Request error: {e}")
                st.stop()

    hits = data.get("hits", [])

    # ── Summary bar ───────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Results", len(hits))
    col_b.metric("Mode", search_type.upper())
    col_c.metric("Index", data.get("index", "—"))

    st.divider()

    if not hits:
        st.info("No results found. Try a different query or search mode.")
    else:
        for rank, hit in enumerate(hits, start=1):
            product_id = hit.get("product_id") or "—"
            score = hit.get("score", 0.0)
            source = hit.get("source") or "—"
            full_text = hit.get("full_text") or ""
            metadata = hit.get("metadata") or {}

            # Source badge colour
            badge = {"ESCI": "🟠", "WANDS": "🔵"}.get(source, "⚪")

            with st.expander(
                f"**#{rank}** · {badge} {source} · score: `{score:.4f}` · `{product_id}`",
                expanded=(rank <= 3),
            ):
                if metadata:
                    cols = st.columns(2)
                    for i, (k_meta, v_meta) in enumerate(metadata.items()):
                        cols[i % 2].markdown(f"**{k_meta}:** {v_meta}")
                    st.divider()

                if full_text:
                    st.markdown("**Full text:**")
                    st.text(full_text[:800] + ("…" if len(full_text) > 800 else ""))

elif search_clicked and not query.strip():
    st.warning("Please enter a search query.")
