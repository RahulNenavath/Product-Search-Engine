FROM python:3.12-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY api/main.py .

# Copy the full product_search package so `from product_search.agent.graph import ...` resolves.
COPY src/product_search/ ./product_search/

# Also copy search_pipeline.py flat into /app for the existing `from search_pipeline import ...` import.
COPY src/product_search/search_pipeline.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000", "--timeout-keep-alive=300"]
