#!/usr/bin/env bash

# detailed logging to stderr
set -x

echo -e "# Hello CHTC from Job ${1} running on $(hostname)\n"
echo -e "# GPUs assigned: ${CUDA_VISIBLE_DEVICES}\n"

echo -e "# Activate Pixi environment\n"
# The last line of the entrypoint.sh file is 'exec "$@"'. If this shell script
# receives arguments, exec will interpret them as arguments to it, which is not
# intended. To avoid this, strip the last line of entrypoint.sh and source that
# instead.
. <(sed '$d' /app/entrypoint.sh)

echo -e "\n# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

echo -e "\n# Extract the training data:\n"
if [ -f "MNIST_data.tar.gz" ]; then
    tar -vxzf MNIST_data.tar.gz
else
    echo "The training data archive, MNIST_data.tar.gz, is not found."
    echo "Please transfer it to the worker node in the HTCondor jobs submission file."
    exit 1
fi

echo -e "\n# Train a PyTorch CNN classifier on the MNIST dataset:\n"
# As main.py is copied to the worker through transfer_input_files, it is not
# located in the same directory as the Pixi manifest (/app/pixi.toml).
# To avoid moving files around further or complicating the HTCondor submit
# description file have the command written out here instead of using a Pixi
# task.
time python ./main.py --epochs 20 --save-model
