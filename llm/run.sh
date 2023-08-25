#!/bin/bash

echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

export TRANSFORMERS_CACHE=$_CONDOR_SCRATCH_DIR/models
export HF_DATASETS_CACHE=$_CONDOR_SCRATCH_DIR/datasets
export HF_MODULES_CACHE=$_CONDOR_SCRATCH_DIR/modules
export HF_METRICS_CACHE=$_CONDOR_SCRATCH_DIR/metrics

# You can remove the --use_wandb flag if tracking is not needed.
python3 train.py demo_run --use_wandb
