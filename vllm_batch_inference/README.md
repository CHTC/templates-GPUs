# vLLM Batch Inference on CHTC

Below is an example workflow demonstrating how to set up, package, submit, and run batch open source LLM inference jobs using `vllm` on CHTC via HTCondor. This is useful for:

- Generating large volumes of synthetic data using open-source LLMs.
- Conducting large-scale, structured data extraction from text.
- Embedding large volumes of text.
- Running any LLM-driven tasks cost-effectively and at massive scale, without relying on expensive commercial alternatives.

## Prerequisites

- You already know the basics on CHTC, HTCondor, Docker universe job, and staging.
- You are familiar with OpenAI python API (we are using the same syntax to call open source models).

## Step-by-step guide

1. 


## FAQ

1. How to find supported open source model?

    Go to [Huggingface's model](https://huggingface.co/models), pick one, you can check if it supports vllm by clicking `use this model` ![hugging face vllm](img/hf-vllm.png)

1. Why you use vllm instead of Ollama?

    Currently, vllm has higher throughput. I think Ollama is better suited in small amount of real time inference.

1. Does this example allow multiple parallel jobs?

    Since I want to keep things relatively simple, I used `SQLite` in this example. However, it is not really design for too many concurrent write. You probably should use a proper database like `PostgreSQL` if you are running into database bottleneck issue.
