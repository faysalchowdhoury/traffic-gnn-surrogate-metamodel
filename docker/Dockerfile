# Build after `gnnsurrogate train` so registry/ holds the PRODUCTION models the API loads.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU-only torch wheel (the default PyPI wheel pulls ~2 GB of CUDA libraries)
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY registry ./registry
RUN pip install --no-deps . && useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
CMD ["uvicorn", "gnnsurrogate.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
