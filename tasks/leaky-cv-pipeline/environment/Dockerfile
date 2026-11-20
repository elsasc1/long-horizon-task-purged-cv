FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Pinned so that results are reproducible; the agent has no network access at run time.
RUN pip install numpy==2.2.6 pandas==2.3.0 pytest==8.3.5

WORKDIR /app
COPY leaklab /app/leaklab
COPY tests_public /app/tests_public
