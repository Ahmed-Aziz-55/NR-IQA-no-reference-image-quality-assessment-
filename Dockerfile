# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

# opencv-python-headless still needs a couple of shared libs present on
# the OS even without GUI support (libgl/libglib) — this is the
# standard minimal set, not the full opencv build-dependency list.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (separate layer) so code-only changes
# don't invalidate the pip install cache layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY app/ ./app/

# Model weights baked into the image — keeps the container
# self-contained and portable (no runtime volume dependency). Re-build
# the image whenever these are updated (e.g. after fine-tuning).
COPY models/ ./models/

# Runs as a non-root user in production.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]