# CUDA accelerated Pixi environment provided through Linux containers

> [!NOTE]
> This is the recommended approach for use of Pixi with HTCondor systems with no shared file system.
> As the container image is cached at the HTCondor facility, this can be scaled out efficiently across multiple copies of the job if needed.

The example uses [Pixi](https://pixi.sh/) to create a fully reproducible PyTorch environment with CUDA support that is containerized.
The environment is provided by the Linux container, so the execution script only needs to ensure the environment is activated before running the requested training scripts.
The values of `gpus_minimum_capability` and the `requirement` of `GPUs_DriverVersion` in the `mnist_gpu_docker.sub` HTCondor submit description file ensure that the worker node GPU will be compatible with the builds of PyTorch and CUDA defined in the environment.

The PyTorch example uses the `MNIST_data.tar.gz` MNIST dataset and `main.py` Python script from the [`shared/pytorch`](https://github.com/CHTC/templates-GPUs/tree/master/shared/pytorch) directory.

> [!IMPORTANT]
> The `MNIST_data.tar.gz` dataset and `main.py` are copied from the HTCondor system login node as `transfer_input_files` arguments in the `mnist_gpu_docker.sub` HTCondor submit description file.
> They do not exist in the built container image.

## Workspace creation

The Pixi workspace in this example was created with the equivalent of the following Pixi commands (run on a `linux-64` platform machine with an NVIDIA GPU and driver present (providing the [`__cuda` virtual package](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-virtual.html)))

```bash
pixi init
pixi workspace system-requirements add cuda 12.9
pixi add pytorch-gpu torchvision 'cuda-version 12.9.*'
pixi task add --description "Train a PyTorch CNN classifier on the MNIST dataset" train "python ./main.py --epochs 20 --save-model"
```

> [!IMPORTANT]
> If you are on a platform that does support CUDA (e.g., Linux or Windows), but do not have a NVIDIA GPU and driver present on the host machine, then you will need to provide the `CONDA_OVERRIDE_CUDA` override environment variable for the `__cuda` virtual package  to resolve the dependency requirements.
>
> ```bash
> pixi init
> pixi workspace system-requirements add cuda 12.9
> CONDA_OVERRIDE_CUDA=12.9 pixi add pytorch-gpu torchvision 'cuda-version 12.9.*'
> pixi task add --description "Train a PyTorch CNN classifier on the MNIST dataset" train "python ./main.py --epochs 20 --save-model"
> ```

> [!NOTE]
> If you were running these commands from an `osx-arm64` or `win-64` platform you would need to specify that `linux-64` is the target platform
>
> ```bash
> pixi init
> pixi workspace platform add linux-64
> pixi workspace system-requirements add cuda 12.9
> pixi add --platform linux-64 pytorch-gpu torchvision 'cuda-version 12.9.*'
> pixi task add --description "Train a PyTorch CNN classifier on the MNIST dataset" train "python ./main.py --epochs 20 --save-model"
> ```

## Docker

### Build Docker container image

Build a Docker container image from the `Dockerfile` that installs the Pixi environment `ENVIRONMENT` and CUDA version `CONDA_OVERRIDE_CUDA`.
The default values are those set in the `Dockerfile`.

```
docker build --file Dockerfile --platform linux/amd64 --tag ghcr.io/<your github org>/templates-gpu:mnist-gpu-noble-cuda-12.9 .
```

and then [authenticate](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry) and publish your built container image to your GitHub container registry

```
docker push ghcr.io/<your github org>/templates-gpu:mnist-gpu-noble-cuda-12.9
```

Note that this manual build and publish process is slow.
It is recommended to have this step be done through a [continuous integration and continuous delivery (CI/CD) workflow](https://carpentries-incubator.github.io/reproducible-ml-workflows/pixi-deployment.html#automation-with-github-actions-workflows).

### Use

* Log into an HTC submit node
* Clone this repository

```
git clone https://github.com/CHTC/templates-GPUs
```

* Navigate to this directory

```
cd templates-GPUs/containers/pixi
```

* Submit the example to HTCondor

```
condor_submit mnist_gpu_docker.sub
```

