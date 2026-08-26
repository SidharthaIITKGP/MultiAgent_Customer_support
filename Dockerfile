FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x start.sh

EXPOSE 8000 8001 7860

# Default: single-container boot (mock backend + api in one process group) —
# used by platforms with one exposed port, like Hugging Face Spaces.
# docker-compose.yml overrides this per-service for local multi-container dev.
CMD ["./start.sh"]
