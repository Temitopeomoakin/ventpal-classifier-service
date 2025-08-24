FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system tools for wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# install Python deps
COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY service /app/service

# env for model
ENV CKPT_DIR=/models/emo_ckpt
# Set your folder id in Cloud Run console as an env var (CKPT_FOLDER_ID)
# or uncomment the next line to bake it in (not recommended):
# ENV CKPT_FOLDER_ID=your_drive_folder_or_url

EXPOSE 8080
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
