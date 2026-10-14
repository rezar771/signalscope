FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Ollama runs on the host; the container reaches it through Docker's host gateway.
    OLLAMA_URL=http://host.docker.internal:11434/v1 \
    CACHE_PATH=/app/.cache/http

WORKDIR /app

# Dependencies first: this layer is reused as long as requirements.txt does not change.
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 10001 app
COPY --chown=app:app signalscope/ signalscope/
COPY --chown=app:app assets/ assets/
COPY --chown=app:app .streamlit/ .streamlit/
COPY --chown=app:app app.py .
RUN mkdir -p /app/.cache && chown app:app /app/.cache
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
