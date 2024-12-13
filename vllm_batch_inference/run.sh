#!/bin/bash

export HOME=$_CONDOR_SCRATCH_DIR
echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

echo "Setting up environment variables"
source .env

# i and j represent job indices (job_id). We can't alter batch size during a run, but optimization significantly speeds up job execution, allowing more batches per job. This serves as a temporary workaround."
# Highly recommend using one batch per job, it can avoid many db connection issues.

i=$(($1*5))
j=$((i+5))
echo "Running job from job_id: $i to $j"

python3 -s -m preprocess_extraction_direct \
    --id_pickle geoarchive_paragraph_ids.pkl \
    --job_index_start "$i" \
    --job_index_end $(($j))

echo "Job completed"