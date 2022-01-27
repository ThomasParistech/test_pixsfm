# syntax=docker/dockerfile:1.3

ARG CUDA_VERSION=11.1
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG CPU_SIZE=4
ARG PYTHON_VERSION=3.9

# Fix Cuda Path
ENV CUDA_ROOT /usr/local/cuda
ENV PATH $PATH:$CUDA_ROOT/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda/lib$LIBRARY_PATH

ENV DEPENDENCIES="/dependencies"
WORKDIR ${DEPENDENCIES}

# basic packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update                && \
    apt-get -y upgrade               && \
    apt-get install -y                  \
    software-properties-common      \
    sudo                            \
    cmake                           \
    build-essential                 \
    wget                            \
    curl                            \
    git                             \
    swig

#-------------------------------------------------------------
#       Python Installation
#-------------------------------------------------------------
RUN add-apt-repository ppa:deadsnakes/ppa       && \
    apt-get -y update                           && \
    apt-get install -y                             \
    python${PYTHON_VERSION}                        \
    python${PYTHON_VERSION}-dev                    \
    libpython${PYTHON_VERSION}-dev                 \
    python${PYTHON_VERSION}-tk                     \
    python${PYTHON_VERSION}-distutils              \
    python3-setuptools

#----------------------------------------------
# Cmake, cpp
#----------------------------------------------

# CMAKE
RUN apt purge -y --auto-remove cmake && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && \
    apt-get install -y cmake

# C++ 20
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    update-alternatives --config gcc

#----------------------------------------------
# Pip binding
#----------------------------------------------

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python${PYTHON_VERSION} get-pip.py

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1


#-------------------------------------------------------------
#       Python3 Module
#-------------------------------------------------------------

# Deps for python modules
RUN apt-get update -y && \
    apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0 \
        ffmpeg

#----------------------------------------------
# TORCH
#----------------------------------------------

COPY .devcontainer/requirements_torch.txt .

RUN \
    --mount=type=cache,target=/root/.cache/pip \
    pip install \
    -r requirements_torch.txt \
    -f https://download.pytorch.org/whl/torch_stable.html

#----------------------------------------------
# COLMAP
#----------------------------------------------

RUN apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

# CERES
RUN apt-get install -y libcgal-qt5-dev
RUN apt-get install -y libatlas-base-dev libsuitesparse-dev

RUN cd ${DEPENDENCIES} && \
    git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j${CPU_SIZE} && \
    make install


# COLMAP
RUN cd ${DEPENDENCIES} && \
    git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout dev && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j${CPU_SIZE} && \
    make install


#----------------------------------------------
# PIXEL PERFECT
#----------------------------------------------

RUN apt-get install -y libhdf5-dev

RUN cd ${DEPENDENCIES} && \
    git clone https://github.com/cvg/pixel-perfect-sfm --recursive && \
    cd pixel-perfect-sfm && \
    pip install -r requirements.txt

# HLOC
RUN cd ${DEPENDENCIES} && \
    git clone --recursive https://github.com/cvg/Hierarchical-Localization/ && \
    cd Hierarchical-Localization/ && \
    pip install -e .

RUN cd ${DEPENDENCIES} && \
    cd pixel-perfect-sfm && \
    pip install -e .



#-------------------------------------------------------------
#       3D_Tools Installation
#-------------------------------------------------------------

ENV TOOLS_PATH="/workspaces/test_pixsfm"
ENV PYTHONPATH $PYTHONPATH:$TOOLS_PATH

WORKDIR $TOOLS_PATH
COPY . .

#-------------------------------------------------------------
#       Post Processing
#-------------------------------------------------------------

## Switch to non-root user
ENV USER=dock
ENV GROUP=sudo

RUN useradd -ms /bin/bash ${USER} && \
    usermod -aG ${GROUP} ${USER}

# Cleaning Part
USER root
RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*

# Resolve authorization problem
RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" \
    >/etc/sudoers.d/${USER} && \
    chmod 0440 /etc/sudoers.d/${USER} && \
    chmod 777 -R /dependencies/pixel-perfect-sfm/pixsfm/features/models

# Final User
USER ${USER}
