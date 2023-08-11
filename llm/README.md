# Personal CHTC submit template for LLM fine-tuning

Use case: Fine-tuning Large language models on CHTC.

## Used stacks

- Docker
- Github container registry (ghcr.io)
- Huggingface Transformers
- Weight & Biases (WANDB)

## Used CHTC/HTCondor features

- Docker universe
- Checkpointing
- Staging (for storing checkpoints)
- GPU

## Building a container

Example source for building a training container:

- [Dockerfile](Dockerfile)
- [requirements.txt](requirements.txt)
- [Helper script](build.sh)
- [.env](.env.example) for Github container registry credentials (`CR_PAT`)

User probably should build their own container to suit their needs.

Example container image

- [Link](ghcr.io/jasonlo/chtc_condor:latest)

## Quick start

1. Put your WANDB credentials and staging_dir path in a environment file: `.env`, see [example](.env.example)
1. Update your training script: `train.py`
1. Update `run_name` in `run.sh`, it will be used as wandb tracking id, and checkpoints will be created in `staging_dir/run_name`
1. Update `run.sub` if needed
1. SSH to submit node
1. Submit your job with `condor_submit run.sub`

## FAQ

1. Why not running `python run.py` directly in `run.sub`?

> I need to export HuggingFace cache dir to `_CONDOR_SCRATCH_DIR` to global scope, I don't know an easy way to do it in `python`. Let me know if you know how to do it.

1. Why `+GPUJobLength = "short"` in `run.sub`?

> Queueing time for `long` is too long, and we checkpoint anyway, so it is better to use `short`.

1. Can I use more GPUs?

> Yes, just change `request_gpus` in `run.sub` to the number you want.

## TODOs

- [ ] Somehow put all configs in one place? Now it is scattered in `.env`, `run.sh`, `run.sub`
- [ ] Add `wandb` hyperparameter sweep support
- [ ] Add `DeepSpeed` support
- [ ] Put docker image in staging? Will it be faster? Or even possible?
- [ ] Experiment with training optimized container like: [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
