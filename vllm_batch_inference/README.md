# vLLM Batch Inference on CHTC

Below is an example workflow demonstrating how to set up, package, submit, and run batch open source LLM inference jobs using `vllm` on CHTC via HTCondor. This is useful for:

- Generating large volumes of synthetic data using open-source LLMs.
- Conducting large-scale, structured data extraction from text.
- Embedding large volumes of text.
- Running any LLM-driven tasks cost-effectively at massive scale, without relying on expensive commercial alternatives.

## Prerequisites

- You already know the basics on CHTC, HTCondor, Docker universe job.

## Step-by-step guide

1. Go to [huggingface](https://huggingface.co/settings/tokens) and get a token. You need it for downloading open source models.
1. `ssh` to a submit node
1. Make a `.env` file based on this [example](.env.example)
1. `condor submit job.sub`

## FAQ

1. How to find supported open source model?

    Go to [Huggingface's model](https://huggingface.co/models), pick one, you can check if it supports vllm by clicking `use this model` ![hugging face vllm](img/hf-vllm.png). Please read the model instruction as some model requires approval or signing an user agreement on their website.

1. Why vllm?

    Currently, vllm has the higher throughput that I know of for batch offline inference.

## About the author

Contributed by [Jason from Data Science Institute, UW-Madison](https://github.com/jasonlo).
