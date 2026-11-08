FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

# GDAL для rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /app

# Зависимости ставятся явно: pyproject тянет второй torch и ломает образ.
# Полный список сторонних импортов src/expds (проверено grep по пакету):
#   numpy, pyyaml, laspy, lazrs, rasterio, pyproj, scipy,
#   scikit-image, segmentation-models-pytorch, torch (в базовом образе)
# --break-system-packages: в базовом образе Python управляется apt (PEP 668),
# иначе pip отказывается ставить пакеты.
RUN pip install --no-cache-dir --break-system-packages \
        numpy \
        pyyaml \
        laspy>=2.7.0 \
        lazrs>=0.8.2 \
        rasterio>=1.4.4 \
        pyproj>=3.7.2 \
        scipy \
        scikit-image \
        segmentation-models-pytorch

COPY src/expds/ ./expds/
COPY inference/solution.py inference/config.yaml inference/weights.pt ./

ENTRYPOINT []
CMD ["python", "solution.py", "/data/input", "/data/output/result.geojson"]
