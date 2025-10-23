FROM python:3.12.4-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps ----
COPY assistant-LLM/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- App code ----
COPY assistant-LLM/main.py /app/assistant-LLM/main.py
COPY outputs /app/outputs

# Env for Streamlit and Gemini
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    GOOGLE_API_KEY=""

EXPOSE 8501

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["streamlit","run","assistant-LLM/main.py","--server.port=8501","--server.address=0.0.0.0"]
