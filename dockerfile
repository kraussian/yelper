# Build using: docker build --no-cache -t yelper .
# Initial run: docker run --gpus all -it --name yelper -v .:/workspace yelper
# After first run, start using Docker Desktop or: docker start -a yelper

# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set a fixed model cache directory.
ENV TORCH_HOME=/root/.cache/torch

# Set frontend mode as non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and necessary packages
RUN apt update && apt install software-properties-common -y --no-install-recommends
RUN add-apt-repository 'ppa:deadsnakes/ppa' && apt update \
    && apt install -y --no-install-recommends \
    build-essential wget git curl unzip cmake \
    python3.12 python3-pip python3.12-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA's CUDA toolkit if needed (only required for DeepSpeed compilation)
RUN apt update && apt install -y --no-install-recommends \
    cuda-toolkit-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Update pip and setuptools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install miniforge
ENV CONDA_DIR=/opt/conda
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    /bin/bash Miniforge3-$(uname)-$(uname -m).sh -b -p ${CONDA_DIR}

ENV PATH=$CONDA_DIR/bin:$PATH

# Install PyTorch with CUDA 12.1 support and other essential packages
# using a dedicated conda env 
RUN conda create --name unsloth_env python=3.10
RUN echo "source activate unsloth_env" > ~/.bashrc
ENV PATH=/opt/conda/envs/unsloth_env/bin:$PATH

# Install dependencies as described in the Unsloth.ai Github
RUN mamba install -n unsloth_env -y pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install seaborn matplotlib
RUN pip install --no-deps trl peft accelerate bitsandbytes
RUN pip install autoawq
RUN pip install datasets
RUN pip install ipython

# Set working directory
WORKDIR /workspace
