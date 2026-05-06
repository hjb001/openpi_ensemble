# Agibot G02 real-robot client image.
#
# Build:    docker build -f examples/agibot_real/Dockerfile -t openpi/agibot_real .
# Run:      docker run --rm --net=host \
#               -v ~/.cache/agibot:/root/.cache/agibot:ro \
#               openpi/agibot_real --host=10.42.1.102 --port=8000
#
# Notes:
# - GDK requires libprotobuf and pybind11 at runtime; install via apt.
# - agibot_gdk wheel is built from `~/.cache/agibot/app/gdk/build_dep/python/pybind/`
#   and must be built on the same Python version as this image (3.10).
# - The image does NOT bundle agibot_gdk by default; mount the wheel or build
#   it once and copy it in via `--build-arg AGIBOT_GDK_WHEEL=path/to.whl`.

FROM python:3.10-slim

ARG AGIBOT_GDK_WHEEL

RUN apt-get update && apt-get install -y --no-install-recommends \
        libprotobuf-dev \
        protobuf-compiler \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Workspace layout matches `app/` checkout root.
COPY packages/openpi-client /app/packages/openpi-client
COPY examples/agibot_real /app/examples/agibot_real

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -e /app/packages/openpi-client \
    && pip install --no-cache-dir -r /app/examples/agibot_real/requirements.in

# Optional: bake in the GDK wheel if provided at build time.
COPY ${AGIBOT_GDK_WHEEL:-/dev/null} /tmp/agibot_gdk.whl
RUN if [ -s /tmp/agibot_gdk.whl ]; then pip install --no-cache-dir /tmp/agibot_gdk.whl; fi \
    && rm -f /tmp/agibot_gdk.whl

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "examples.agibot_real.main"]
