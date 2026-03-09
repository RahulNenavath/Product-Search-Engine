FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

COPY embedding_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY embedding_service/main.py .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8001", "--timeout-keep-alive=120"]