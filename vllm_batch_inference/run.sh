#!/bin/bash

export HOME=$_CONDOR_SCRATCH_DIR
echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

echo "Setting up environment variables"
source .env

python batch_inference.py
echo "Job completed"
