# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       git \
       libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
# Copy fallback model
COPY outputs/models/ outputs/models/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Create a non-root user
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# Copy dependency manifests first for better caching
COPY --chown=appuser:appuser requirements.txt ./
COPY --chown=appuser:appuser pyproject.toml ./

# Install dependencies
RUN python -m pip install --upgrade pip \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy project files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser config/ ./config/

# Expose common ports if needed (e.g., for APIs)
EXPOSE 8000

# Default command: print help and list available scripts
CMD ["python", "-c", "import os,glob; print('Project container ready. Example runs:'); print('- Run feature build: python scripts/build_features.py'); print('- Generate insights: python scripts/generate_insights.py'); print('- Train sentiment model: python scripts/train_sentiment_analysis_model.py'); print('- Run migrations: python scripts/run_migrations.py')"]
