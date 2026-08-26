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

# Ingest the knowledge base into the image at BUILD time, not boot time.
# Without this, every container start (every free-tier spin-down/wake, every
# redeploy) re-downloads the embedding model and re-embeds the whole doc set
# before the app can serve a single request — 60-90s+ of pure cold-start tax
# on top of the actual pipeline latency. Baking it in means that cost is paid
# once per image build, not once per visitor's first message.
RUN python knowledge_base/ingest.py

EXPOSE 8000 8001 7860

# Default: single-container boot (mock backend + api in one process group) —
# used by platforms with one exposed port, like Hugging Face Spaces.
# docker-compose.yml overrides this per-service for local multi-container dev.
CMD ["./start.sh"]
