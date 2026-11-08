FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# GDAL для rasterio, laspy для LAS
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости из pyproject.toml
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . && pip uninstall -y expds

# Код и веса
COPY src/expds/ ./expds/
COPY inference/solution.py inference/config.yaml inference/weights.pt ./

ENTRYPOINT []
CMD ["python", "solution.py", "/data/input", "/data/output/result.geojson"]
