# Slim AnnotationCrops gallery. Stdlib, Pillow, NumPy, and SciPy (inscribed circles).
# The converter (/converter/, off unless GALLERY_CONVERTER_ROOT is set) adds grpcio, protobuf and contourpy.
# Mask refresh loads mounted mask modules only: no imageregistration, CuPy, or torch.
FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GALLERY_HOST=0.0.0.0 \
    GALLERY_PORT=80 \
    GALLERY_VOLUME_DIR=/gallery-volumes \
    GALLERY_IDENTITY_MODE=stub

WORKDIR /app

COPY annotation-gallery/circle_update.py /app/
COPY annotation-gallery/server.py annotation-gallery/identity.py annotation-gallery/catalog.py annotation-gallery/thumbs.py annotation-gallery/refresh.py annotation-gallery/crops_path.py annotation-gallery/negatives_view.py annotation-gallery/cardmask.py annotation-gallery/update_all.py /app/
COPY annotation-gallery/static /app/static
COPY annotation-gallery/converter /app/converter
COPY annotation-gallery/gold /app/gold
COPY annotation-gallery/seg_proto /app/seg_proto
COPY annotation_crops /app/annotation_crops

RUN pip install --no-cache-dir "numpy>=2.2" pillow scipy grpcio protobuf contourpy \
    && groupadd --gid 10001 gallery \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin gallery

EXPOSE 80 443

USER 10001:10001

CMD ["python", "/app/server.py"]
