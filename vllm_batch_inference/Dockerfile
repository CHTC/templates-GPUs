FROM vllm/vllm-openai:v0.8.1
RUN chmod -R 777 /root/

ENV UV_PYTHON_INSTALL_DIR=/root/.local/share/uv/python
ENV UV_CACHE_DIR=/root/.cache/uv

ENTRYPOINT ["/bin/bash", "-cl"]
