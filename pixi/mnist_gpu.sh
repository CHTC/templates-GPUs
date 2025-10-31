#!/usr/bin/env bash

# detailed logging to stderr
set -x

echo -e "# Hello CHTC from Job ${1} running on $(hostname)\n"
echo -e "# GPUs assigned: ${CUDA_VISIBLE_DEVICES}\n"

echo -e "# Installing Pixi"
curl -fsSL https://pixi.sh/install.sh | bash
. ~/.bashrc

echo -e "\n# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

# Executing a Pixi command also installs the environment.
# If you wanted to install the environment in advance (not needed) you can
# run `pixi install`.
echo -e "\n# Check if PyTorch can detect the GPU:\n"
curl -sLO https://raw.githubusercontent.com/matthewfeickert/nvidia-gpu-ml-library-test/5d6165b2222dde67c16550f6fb595907ce7b0ce6/torch_detect_GPU.py
time pixi run python ./torch_detect_GPU.py

echo -e "\n# Extract the training data:\n"
if [ -f "MNIST_data.tar.gz" ]; then
    tar -vxzf MNIST_data.tar.gz
else
    echo "The training data archive, MNIST_data.tar.gz, is not found."
    echo "Please transfer it to the worker node in the HTCondor jobs submission file."
    exit 1
fi

echo -e "\n# Train a PyTorch CNN classifier on the MNIST dataset:\n"
time pixi run train
