# Hugging Face Docker Space for the playlist service (serving/insights_app.py).
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH INSIGHTS_DATA_DIR=/tmp/insights PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements-service.txt .
RUN pip install --no-cache-dir --user -r requirements-service.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "serving.insights_app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
