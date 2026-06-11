FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLUXA_WHISPER_MODEL_SIZE=small
ENV FLUXA_WHISPER_DEVICE=cpu
ENV FLUXA_WHISPER_COMPUTE_TYPE=int8
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
