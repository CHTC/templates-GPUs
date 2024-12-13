#!/bin/bash

docker run --rm -it --gpus all \
 	--env-file .env \
    --ipc=host \
	--entrypoint /bin/bash \
	vllm/vllm-openai:v0.6.4 \
