# Build from the repository root:
#   docker build -f serving/Dockerfile -t playlist-ranker .
# Run (mount the trained checkpoint and the processed data):
#   docker run -p 8000:8000 \
#     -v "$PWD/checkpoints:/app/checkpoints" -v "$PWD/data/processed:/app/data/processed" \
#     playlist-ranker
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# All dependencies ship binary wheels, so no compiler is needed.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY setup.py README.md ./
COPY data ./data
COPY evaluation ./evaluation
COPY features ./features
COPY models ./models
COPY serving ./serving
COPY training ./training
RUN pip install --no-deps -e .

EXPOSE 8000

# Uses the Python standard library, so the image needs no curl/wget.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
