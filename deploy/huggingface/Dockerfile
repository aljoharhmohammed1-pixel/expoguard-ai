FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DEVICE=cpu APP_ENV=production YOLO_AUTOINSTALL=false AI_SERVICE_VERSION=0.1.0-integration
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 1000 --create-home user
ENV HOME=/home/user YOLO_CONFIG_DIR=/home/user/.config/Ultralytics MPLCONFIGDIR=/home/user/.cache/matplotlib
WORKDIR /home/user/app
RUN chown user:user /home/user/app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user service.zip ./service.zip
USER user
RUN python -m zipfile -e service.zip . && rm service.zip
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--limit-concurrency", "8", "--no-access-log"]
