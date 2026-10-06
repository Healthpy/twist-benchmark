# Reproducibility image (CPU). For Snellius/HPC use apptainer.def instead.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 TWIST_NUM_THREADS=1
WORKDIR /opt/twist

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -e ".[dev]"

COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts
COPY Makefile ./

CMD ["make", "PY=python", "toy"]
