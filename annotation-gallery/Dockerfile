# Slim AnnotationCrops gallery. Stdlib, Pillow, NumPy, and SciPy (inscribed circles).
# The converter (/converter/, off unless GALLERY_CONVERTER_ROOT is set) adds grpcio, protobuf and contourpy, and Viking's
# shared segmentation prompt library (native, built in the first stage).
# Mask refresh loads mounted mask modules only: no imageregistration, CuPy, or torch.
#
# Build contexts (docker/compose.cursor-dev.yaml): the Sam2Trainer checkout, plus Viking's SegmentationPrompts and
# gRPC_Protos folders as the named contexts segmentation_prompts and grpc_protos, so nothing of Viking is copied here.

# The prompt library for linux-x64 (NativeAOT). Its csproj compiles the protos at ../../gRPC_Protos, so the folders keep
# Viking's layout. The vendored seg_proto must be the protos the library is built from.
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS prompts
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends clang zlib1g-dev && rm -rf /var/lib/apt/lists/*
COPY --from=grpc_protos . /src/gRPC_Protos
COPY --from=segmentation_prompts . /src/SegmentationPrompts
COPY annotation-gallery/seg_proto /vendored
RUN for pair in segmentation.proto:Segmentation/SAM2/segmentation.proto \
                prompt_input.proto:Segmentation/SAM2/prompt_input.proto \
                Common/geometry.proto:Common/geometry.proto; do \
      ours="${pair%%:*}"; theirs="${pair#*:}"; \
      if ! cmp -s <(tr -d '\r' < "/vendored/$ours") <(tr -d '\r' < "/src/gRPC_Protos/$theirs"); then \
        echo "annotation-gallery/seg_proto/$ours differs from gRPC_Protos/$theirs: run tools/gen_seg_proto.py" >&2; \
        exit 1; \
      fi; \
    done
# A Windows checkout's bin/obj would point restore at Windows paths.
RUN find /src/SegmentationPrompts -type d \( -name bin -o -name obj -o -name TestResults \) -prune -exec rm -rf {} + \
    && dotnet publish /src/SegmentationPrompts/Viking.SegmentationPrompts.Native -c Release -r linux-x64 -o /out \
    && mkdir -p /check \
    && cp /src/SegmentationPrompts/Viking.SegmentationPrompts.Native/check_native.py /check/ \
    && cp -r /src/SegmentationPrompts/Viking.SegmentationPrompts.Tests /check/Tests

# Runtime: trixie, so its glibc is not older than the SDK image's the library was linked against.
FROM python:3.14-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GALLERY_HOST=0.0.0.0 \
    GALLERY_PORT=80 \
    GALLERY_VOLUME_DIR=/gallery-volumes \
    GALLERY_IDENTITY_MODE=stub

WORKDIR /app

COPY annotation-gallery/circle_update.py /app/
COPY annotation-gallery/server.py annotation-gallery/identity.py annotation-gallery/catalog.py annotation-gallery/thumbs.py annotation-gallery/refresh.py annotation-gallery/crops_path.py annotation-gallery/negatives_view.py annotation-gallery/cardmask.py annotation-gallery/update_all.py annotation-gallery/score.py /app/
COPY annotation-gallery/static /app/static
COPY annotation-gallery/converter /app/converter
COPY annotation-gallery/gold /app/gold
COPY annotation-gallery/seg_proto /app/seg_proto
COPY annotation_crops /app/annotation_crops
COPY --from=prompts /out/Viking.SegmentationPrompts.Native.so /app/native/

# The image fails to build when its native library does not give the .NET tests' golden request.
RUN --mount=type=bind,from=prompts,source=/check,target=/tmp/check \
    pip install --no-cache-dir "numpy>=2.2" pillow scipy grpcio protobuf contourpy \
    && python /tmp/check/check_native.py /app/native/Viking.SegmentationPrompts.Native.so /tmp/check/Tests \
    && groupadd --gid 10001 gallery \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin gallery

EXPOSE 80 443

USER 10001:10001

CMD ["python", "/app/server.py"]
