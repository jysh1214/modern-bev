FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9
ENV PYTHON_BIN=python$PYTHON_VERSION
ENV PIP=pip$PYTHON_VERSION
ENV PIP_INSTALL="$PIP install --no-cache-dir --default-timeout=120"

RUN apt update -y && \
  apt upgrade -y && \
  apt install -y \
  build-essential \
  ninja-build \
  git \
  wget \
  libssl-dev \
  # for python3.9
  libffi-dev \
  zlib1g-dev \
  liblzma-dev \
  libbz2-dev \
  libsqlite3-dev \
  tk-dev \
  # for mmcv
  libgl1 \
  libglib2.0-0

# Install Python3.9 for torch1.9.0.
ENV CPYTHON_ROOT="/tmp/cpython"
WORKDIR $CPYTHON_ROOT
RUN git clone -b v3.9.21 https://github.com/python/cpython.git $CPYTHON_ROOT && \
  $CPYTHON_ROOT/configure --enable-optimizations --enable-loadable-sqlite-extensions && \
  make -C $CPYTHON_ROOT -j $(nproc) && \
  make -C $CPYTHON_ROOT install

# Install modern CMake for CUDA project.
ENV CMAKE_ROOT="/tmp/CMake"
WORKDIR $CMAKE_ROOT
RUN git clone https://github.com/Kitware/CMake.git $CMAKE_ROOT && \
  $CMAKE_ROOT/bootstrap && \
  make -C $CMAKE_ROOT -j $(nproc) && \
  make -C $CMAKE_ROOT install

# Install protobuf for onnx.
ENV PROTOBUF_ROOT="/tmp/protobuf"
RUN git clone -b v21.12 https://github.com/protocolbuffers/protobuf.git $PROTOBUF_ROOT && \
  git -C $PROTOBUF_ROOT submodule update --init --recursive && \
  cmake -B $PROTOBUF_ROOT/build -S $PROTOBUF_ROOT -G Ninja \
  -D protobuf_BUILD_SHARED_LIBS=OFF \
  -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
  -D protobuf_BUILD_TESTS=OFF \
  -D CMAKE_BUILD_TYPE=Release && \
  cmake --build $PROTOBUF_ROOT/build --target install

# Install the dependencies.
RUN $PIP_INSTALL \
  einops==0.8.1 \
  fvcore==0.1.5.post20221221 \
  seaborn==0.13.2 \
  wheel==0.45.1 \
  numpy==1.23.5 \
  iopath==0.1.9 \
  timm==0.6.13 \
  typing-extensions==4.5.0 \
  pylint==3.3.4 \
  plyfile==1.1 \
  trimesh==4.6.4 \
  ipython==8.12 \
  matplotlib==3.5.2 \
  numba==0.60.0 \
  pandas==1.4.4 \
  scikit-image==0.19.3 \
  setuptools==59.5.0 \
  lyft_dataset_sdk==0.0.8 \
  nuscenes-devkit==1.1.11 && \
  $PIP_INSTALL git+https://github.com/facebookresearch/detectron2.git && \
  $PIP_INSTALL torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html && \
  $PIP_INSTALL mmdet==2.14.0 mmsegmentation==0.14.1

# Clean up.
RUN rm -rf $CPYTHON_ROOT && \
  rm -rf $CMAKE_ROOT && \
  rm -rf $PROTOBUF_ROOT

WORKDIR /home
CMD ["/bin/sh"]
