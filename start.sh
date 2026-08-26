#!/bin/sh
# Single-container boot for platforms that only expose one port (e.g. HF Spaces):
# runs the mock backend in the background and the main API+chat UI in the
# foreground on $PORT (HF Spaces convention; defaults to 7860 to match the
# "docker" SDK's default app_port).
#
# Knowledge base ingestion happens at BUILD time (see Dockerfile), not here —
# it's already baked into the image, so boot doesn't pay that cost every time.
set -e

uvicorn mock_backend:app --host 0.0.0.0 --port 8000 &

exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-7860}"
