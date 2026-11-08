FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# GDAL для rasterio, GEOS для shapely
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev libgeos-dev && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Только зависимости. Проект НЕ ставим через pip:
# pyproject требует torch>=2.14.0, а в образе torch 2.5.1 —
# pip потянул бы второй torch. Код подключается через sys.path.
RUN pip install --no-cache-dir \
        numpy \
        pyyaml \
        laspy>=2.7.0 \
        lazrs>=0.8.2 \
        rasterio>=1.4.4 \
        pyproj>=3.7.2 \
        scikit-image>=0.26.0 \
        scipy

# Код пакета: _ensure_expds найдёт /app/expds
COPY src/expds/ ./expds/
COPY inference/solution.py inference/config.yaml inference/weights.pt ./

ENTRYPOINT []
CMD ["python", "solution.py", "/data/input", "/data/output/result.geojson"]
