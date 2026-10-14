FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DEVICE=cpu APP_ENV=production YOLO_AUTOINSTALL=false
WORKDIR /service
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY ai ./ai
COPY playbooks ./playbooks
COPY config ./config
RUN useradd --uid 10001 --create-home aiuser
USER aiuser
EXPOSE 8000
HEALTHCHECK --start-period=90s --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "8", "--no-access-log"]
