FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    ca-certificates \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:$PATH"
ENV HF_HOME=/tmp/hf_cache
ENV TORCH_HOME=/tmp/torch_cache
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV XDG_CACHE_HOME=/tmp/cache
ENV TMPDIR=/tmp
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Keep yt-dlp at the latest version (YouTube's anti-bot measures evolve
# constantly — old versions get blocked quickly).
RUN pip install --no-cache-dir --upgrade yt-dlp

# The .pt weight files are too large for git (HF rejects binaries) so they
# are gitignored and NOT present in the Docker build context.  Instead,
# load_detector() in engine/model.py auto-downloads them from Hugging Face
# Hub at first use.  The HF_TOKEN env-var is injected automatically by
# the HF Spaces runtime, so no extra auth configuration is needed.
COPY --chown=user . /app

# Fix ownership so the non-root user can read everything.
USER root
RUN chown -R user:user /app
USER user

# Startup script — reads YOUTUBE_COOKIES secret and writes it to /tmp/cookies.txt
COPY --chown=user start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
