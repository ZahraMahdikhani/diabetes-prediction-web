# syntax=docker/dockerfile:1

# مرحله 1: Builder (اختیاری - برای کاهش اندازه نهایی ایمیج)
FROM python:3.10-slim AS builder

WORKDIR /app

# به‌روزرسانی و نصب حداقل وابستگی‌های سیستمی
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# جلوگیری از ایجاد فایل‌های __pycache__ و .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# نصب وابستگی‌ها اول (برای کش بهتر در لایه‌ها)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# مرحله 2: Runtime (ایمیج نهایی سبک)
FROM python:3.10-slim

WORKDIR /app

# کپی فقط پکیج‌های نصب‌شده از builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# کپی کد برنامه
COPY . .

# ایجاد کاربر غیرروت (امنیت بهتر)
RUN useradd -m appuser
USER appuser

# پورت (بیشتر جنبه مستندسازی دارد)
EXPOSE 5000

# اجرای برنامه با gunicorn – استفاده از متغیر PORT برای سازگاری با پلتفرم‌های ابری
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 120 app:app"]

RUN apt-get update && apt-get install -y libgomp1