FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:$PATH"
ENV HF_HOME=/tmp/hf_cache
ENV TORCH_HOME=/tmp/torch_cache
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV XDG_CACHE_HOME=/tmp/cache
ENV TMPDIR=/tmp

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download BTC model weights before copying the rest — these are too large
# for git (HF rejects binaries) so we fetch them at build time.
RUN mkdir -p /app/btc_model/test && \
    curl -L -o /app/btc_model/test/btc_model.pt \
      "https://huggingface.co/jayg996/BTC-ISMIR19/resolve/main/test/btc_model.pt?download=true" && \
    curl -L -o /app/btc_model/test/btc_model_large_voca.pt \
      "https://huggingface.co/jayg996/BTC-ISMIR19/resolve/main/test/btc_model_large_voca.pt?download=true"

COPY --chown=user . /app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
