FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY requirements.txt .

# CPU-only PyTorch – no NVIDIA bloat
RUN uv pip install --system --no-cache \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache -r requirements.txt

# Optional: for health checks or debugging
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]