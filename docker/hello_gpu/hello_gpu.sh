#!/bin/bash
echo "Hello CHTC from Job $1 running on `hostname`"
echo ""
echo "Trying to see if nvidia/cuda can access the GPU...."
nvidia-smi
