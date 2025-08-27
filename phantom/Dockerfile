# Multi-stage build for production
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create non-root user
RUN useradd -m -s /bin/bash phantom
USER phantom
WORKDIR /home/phantom

# Copy requirements first for better caching
COPY --chown=phantom:phantom pyproject.toml .
COPY --chown=phantom:phantom README.md .

# Install PHANTOM
COPY --chown=phantom:phantom phantom_core phantom_core/
COPY --chown=phantom:phantom phantom_cli phantom_cli/
COPY --chown=phantom:phantom examples examples/

RUN pip install --user -e .

# Set environment variables
ENV PATH="/home/phantom/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create directories for reports and runs
RUN mkdir -p reports runs

# Default command
CMD ["phantom", "--help"]

# Production image with additional tools
FROM base AS production

# Install additional dependencies
RUN pip install --user plotly tabulate

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD phantom --version || exit 1

# Development image with test tools
FROM base AS development

# Install development dependencies
RUN pip install --user -e ".[dev,vision,docs]"

# Add test command
CMD ["pytest", "tests/", "-v"]