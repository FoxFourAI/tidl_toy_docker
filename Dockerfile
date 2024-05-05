ARG BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM $BASE_IMAGE

SHELL [ "/bin/bash", "-c"]
# Install required packages
ENV DEBIAN_FRONTEND=noninteractive

ENV FORCE_CUDA 1
ENV TORCH_CUDA_ARCH_LIST "6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"

RUN apt-get update \
    && apt-get install -yq software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt install -yq apt-transport-https \
    gcc \
    g++ \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libstdc++6 \
    zip \
    libcurl4-openssl-dev \
    tmux \
    tmuxinator \
    libxrender1 \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    libprotoc-dev \
    graphviz \
    swig \
    curl \
    vim \
    wget \
    gdb \
    pkg-config \
    libgtk-3-dev \
    libyaml-cpp-dev \
    python3 \
    python3-pip \
    python3-setuptools \
    && apt-get install -yq ffmpeg x264 libx264-dev \
    && apt-get install -yq libavformat-dev libavcodec-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/workdir
WORKDIR /home/workdir

# Changing parameter back to default
ENV DEBIAN_FRONTEND=newt
# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN mkdir installation
WORKDIR /home/workdir/installation
ADD installation/environment.yml environment.yml
RUN conda env create -f ./environment.yml

RUN echo "source activate tidl-py310" > ~/.bashrc
ENV PATH /opt/conda/envs/tidl-py310/bin:$PATH

ADD installation/requirements.txt requirements.txt
RUN /opt/conda/envs/tidl-py310/bin/pip install -r requirements.txt
RUN pip install git+https://github.com/NVIDIA/TensorRT@release/8.5#subdirectory=tools/onnx-graphsurgeon
RUN pip install opencv-python

ADD installation/setup_tidl.sh setup_tidl.sh
RUN echo "source /home/workdir/installation/setup_tidl.sh" > ~/.bashrc
WORKDIR /home/workdir

ADD tidl-onnx-model-optimizer tidl-onnx-model-optimizer

ENV SOC "am68pa"
ENV TIDL_PATH "/home/workdir/tidl_tools"
ENV TIDL_PYTHON_INTERPRETER_PATH "/opt/conda/envs/tidl-py310/bin"


# Set entrypoint
ENTRYPOINT [ "/bin/bash" ]
