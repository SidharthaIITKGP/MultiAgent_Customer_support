FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch FIRST, from PyTorch's own CPU wheel index. Plain
# `pip install sentence-transformers` (via requirements.txt) pulls in the
# default CUDA-enabled torch build — ~1.1GB with a much heavier import-time
# memory footprint — even though nothing here uses a GPU. Satisfying torch
# with the CPU build up front means the later requirements.txt install sees
# it already installed and skips pulling the CUDA one. No code/library
# change — same sentence-transformers, same model, just a lighter binary.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x start.sh

EXPOSE 8000 8001 7860

# Default: single-container boot (mock backend + api in one process group) —
# used by platforms with one exposed port, like Hugging Face Spaces.
# docker-compose.yml overrides this per-service for local multi-container dev.
CMD ["./start.sh"]
