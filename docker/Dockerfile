# SAM2 trainer cursor-dev image.
# Build from repository root:
#   docker build -f docker/Dockerfile -t sam2-trainer:dev .
# Or: docker/docker-build.ps1 from D:\Docker\Builds\sam2-trainer
#
# Pin facebookresearch/sam2 so image builds are reproducible.
# Notify James if https://github.com/facebookresearch/sam2 has commits newer than SAM2_GIT_SHA.

FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0+PTX"
ENV VIRTUAL_ENV=/opt/venv
ENV SAM2_BUILD_CUDA=1

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-transport-https ca-certificates \
        git curl wget ca-certificates \
        build-essential ninja-build \
        ffmpeg libsm6 libxext6 \
        software-properties-common \
        cifs-utils nfs-common keyutils libcap2-bin util-linux \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --set python3 /usr/bin/python3.13 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 \
    && python3.13 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# CUDA 13.2 wheels must be installed before SAM2 so pip does not pull a CPU/default torch.
# Match nvcc in this image to PyTorch's cu132 index (SAM2 builds a CUDA extension).
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu132

# Declare immediately before use so --build-arg SAM2_GIT_SHA does not bust the torch layer.
ARG SAM2_GIT_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4
RUN git clone https://github.com/facebookresearch/sam2.git /opt/sam2 \
    && git -C /opt/sam2 checkout "$SAM2_GIT_SHA" \
    && pip install --no-cache-dir -e /opt/sam2 -v \
    && pip install --no-cache-dir -e "/opt/sam2[dev]" \
    && pip install --no-cache-dir albumentations scipy

COPY docker/mount-network-shares.sh docker/drop-sys-admin-after-mounts.sh docker/cursor-dev-entry.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/mount-network-shares.sh \
        /usr/local/bin/drop-sys-admin-after-mounts.sh \
        /usr/local/bin/cursor-dev-entry.sh

WORKDIR /workspace

# TensorBoard listens on 6006 in-container; compose publishes host 8060:6006.
EXPOSE 6006

CMD ["/bin/bash"]
