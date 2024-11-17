# Build using: docker build -t yelper .
# Run using: docker run --gpus all -it --name yelper -v .:/workspace yelper
# After first run, start using: docker start -a yelper

# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set a fixed model cache directory.
ENV TORCH_HOME=/root/.cache/torch

# Install Python and necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential python3.10 python3-pip python3.10-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# update pip and setuptools
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# install miniforge
ENV CONDA_DIR=/opt/conda
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    /bin/bash Miniforge3-$(uname)-$(uname -m).sh -b -p ${CONDA_DIR}

ENV PATH=$CONDA_DIR/bin:$PATH

# install PyTorch with CUDA 12.1 support and other essential packages
# use a dedicated conda env 
RUN conda create --name unsloth_env python=3.10
RUN echo "source activate unsloth_env" > ~/.bashrc
ENV PATH=/opt/conda/envs/unsloth_env/bin:$PATH

# as described in the Unsloth.ai Github
RUN mamba install -n unsloth_env -y pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install matplotlib
RUN pip install --no-deps trl peft accelerate bitsandbytes
RUN pip install autoawq
RUN pip install datasets

# copy the fine-tuning script into the container
COPY ./train_llm_lora.py /trainer/train_llm_lora.py

WORKDIR /trainer
