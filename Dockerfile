# syntax=docker/dockerfile:1

# ---------- Stage 1 : Builder ----------
FROM python:3.10-slim AS builder

WORKDIR /app

# نصب ابزار build (برای بعضی پکیج‌ها مثل numpy / sklearn)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---------- Stage 2 : Runtime ----------
FROM python:3.10-slim

WORKDIR /app

# نصب runtime موردنیاز LightGBM (خیلی مهم)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# کپی پکیج‌ها از builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# کپی سورس پروژه
COPY . .

# ساخت یوزر امن
RUN useradd -m appuser
USER appuser

EXPOSE 5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 120 app:app"]