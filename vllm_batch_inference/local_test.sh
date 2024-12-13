#!/bin/bash

docker run --rm --runtime nvidia --gpus all \
 	--env-file .env \
    --ipc=host \
	vllm/vllm-openai:v0.6.4 \
	--model microsoft/Phi-3.5-mini-instruct
