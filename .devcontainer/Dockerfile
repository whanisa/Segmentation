# ===== Base: CUDA 11.8 + cuDNN on Ubuntu 22.04 (Python 3.10) =====
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ---- Global environment ----
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_GPU_ALLOCATOR=cuda_malloc_async \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    MPLBACKEND=Agg

# ---- System packages + Python 3.10 + pip ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-distutils python3-pip python3-venv \
        build-essential git ca-certificates \
        libopenexr-dev libglib2.0-0 ffmpeg sudo && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# ---- Core Python stack (pin albumentations to avoid '<' in spec) ----
RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 \
    tensorflow==2.14.1 \
    albumentations==2.0.8 \
    tqdm h5py scikit-image pandas imageio pillow matplotlib \
    scikit-learn pydicom opencv-python-headless typing_extensions tabulate \
    gdown==2.3.1 ffmpeg-python GPUtil \
    neptune neptune-tensorflow-keras torch-tb-profiler jupyter

# ---- PyTorch (CUDA 11.8 wheels) ----
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# ---- Create non-root user ----
ARG USERNAME=vscode
ARG USER_UID=1003
ARG USER_GID=${USER_UID}
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    usermod --shell /bin/bash ${USERNAME}

# ---- Switch to non-root user ----
USER ${USERNAME}
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
WORKDIR /workspaces/PhD

# (Optional) Quick GPU check:
# docker run --gpus all --rm <image> python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"















# # ===== Base: CUDA 11.8 + cuDNN on Ubuntu 22.04 (Python 3.10) =====
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive \
#     PIP_DISABLE_PIP_VERSION_CHECK=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     TF_CPP_MIN_LOG_LEVEL=1 \
#     TF_FORCE_GPU_ALLOW_GROWTH=true \
#     MPLBACKEND=Agg

# # ---- System packages + Python 3.10 + pip ----
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         python3 python3-distutils python3-pip python3-venv \
#         build-essential git ca-certificates \
#         libopenexr-dev libglib2.0-0 ffmpeg sudo && \
#     ln -s /usr/bin/python3 /usr/bin/python && \
#     python3 -m pip install --upgrade pip setuptools wheel && \
#     rm -rf /var/lib/apt/lists/*

# # ---- Core Python stack (pin albumentations to avoid '<' in spec) ----
# RUN python3 -m pip install --no-cache-dir \
#     numpy==1.26.4 \
#     tensorflow==2.14.1 \
#     albumentations==2.0.8 \
#     tqdm h5py scikit-image pandas imageio pillow matplotlib \
#     scikit-learn pydicom opencv-python-headless typing_extensions tabulate \
#     gdown==2.3.1 ffmpeg-python GPUtil \
#     neptune neptune-tensorflow-keras torch-tb-profiler jupyter

# # ---- PyTorch (CUDA 11.8 wheels) ----
# RUN python3 -m pip install --no-cache-dir \
#     --index-url https://download.pytorch.org/whl/cu118 \
#     torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# # ---- Create non-root user ----
# ARG USERNAME=vscode
# ARG USER_UID=1003
# ARG USER_GID=${USER_UID}
# RUN groupadd --gid ${USER_GID} ${USERNAME} && \
#     useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
#     echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
#     chmod 0440 /etc/sudoers.d/${USERNAME} && \
#     usermod --shell /bin/bash ${USERNAME}

# USER ${USERNAME}
# ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
# WORKDIR /workspaces/PhD

# # (Optional) Quick GPU check:
# # docker run --gpus all --rm <image> python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"



