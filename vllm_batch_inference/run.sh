#!/bin/bash

# Configure directory paths to writable _CONDOR_SCRATCH_DIR location
export HOME=$_CONDOR_SCRATCH_DIR
export HF_HOME=$_CONDOR_SCRATCH_DIR/huggingface
export TORCHINDUCTOR_CACHE_DIR=$_CONDOR_SCRATCH_DIR/torchinductor

# vllm expects GPU IDs as integers (0, 1, 2, ...), but in CHTC they are in UUID format (GPU-xxxxxx). This script converts UUIDs to integer IDs before running batch inference.
convert_cuda_visible_devices() {
  if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "CUDA_VISIBLE_DEVICES not set."
    return
  fi

  if [[ "$CUDA_VISIBLE_DEVICES" == *"GPU-"* ]]; then
    IFS=',' read -ra UUIDS <<< "$CUDA_VISIBLE_DEVICES"
    INDICES=()
    for uuid in "${UUIDS[@]}"; do
      idx=$(nvidia-smi -L | grep -n "$uuid" | cut -d: -f1)
      idx=$((idx - 1))
      INDICES+=("$idx")
    done
    export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${INDICES[*]}")
    echo "For vllm compatibility, converted CUDA_VISIBLE_DEVICES to: $CUDA_VISIBLE_DEVICES"
  else
    echo "CUDA_VISIBLE_DEVICES is already in index format: $CUDA_VISIBLE_DEVICES"
  fi
}


# Convert CUDA_VISIBLE_DEVICES from UUID to index format

echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

convert_cuda_visible_devices

echo "Setting up environment variables"
source .env

python3 batch_inference.py
echo "Job completed"
