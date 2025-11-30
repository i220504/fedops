FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cpu


RUN pip install -e .
RUN pip install prometheus-client
RUN pip install soundfile
RUN pip install --no-cache-dir mlflow==2.12.2


RUN python3 - << 'EOF'

print("=== Running pre-build dependency test ===")
import torch
import torchaudio
print("Torch OK:", torch.__version__)
print("Torchaudio OK:", torchaudio.__version__)
EOF


CMD ["flwr", "run"]
