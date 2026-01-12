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
docker build --file Dockerfile --platform linux/amd64 --tag ghcr.io/<your github org>/templates-gpus:mnist-gpu-noble-cuda-12.9 .
```

and then [authenticate](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry) and publish your built container image to your GitHub container registry

```
docker push ghcr.io/<your github org>/templates-gpus:mnist-gpu-noble-cuda-12.9
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

## Apptainer

### Build Apptainer container image

Build a Apptainer container image (`.sif` format) from the `apptainer.def` that installs the Pixi environment `ENVIRONMENT` and CUDA version `CONDA_OVERRIDE_CUDA`.
The default values are those set in the `apptainer.def`.

```
apptainer build mnist-gpu-noble-cuda-12.9.sif apptainer.def
```

and then [authenticate](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry) and publish your built container image to your GitHub container registry

```
apptainer push mnist-gpu-noble-cuda-12.9.sif oras://ghcr.io/<your github org>/templates-gpus:apptainer-mnist-gpu-noble-cuda-12.9
```

Note that this manual build and publish process is slow.
It is recommended to have this step be done through a [continuous integration and continuous delivery (CI/CD) workflow](https://carpentries-incubator.github.io/reproducible-ml-workflows/pixi-deployment.html#automation-with-github-actions-workflows-1).

### Transferring `.sif` files to HTCondor Execution Points

The transferring of `.sif` files can happen one of two ways.
Both involve setting the `container` universe in the submission file along with giving an endpoint for the `container_image`.

#### Transferring from CHTC `/staging`

Container `.sif` files on CHTC's `/staging/<username>` storage are cached with [ODSF](https://osg-htc.org/services/osdf), and transferred to HTCondor jobs using the

```
container_image = osdf:///chtc/staging/<username>/<container image name>.sif
```

HTCondor submission syntax.

**Example**:

```
container_image = osdf:///chtc/staging/<user name>/apptainer/mnist-gpu-noble-cuda-12.9-sha-80ec247.sif
```

> [!TIP]
> OSDF offers the advantage of [not limiting your jobs to running on locations where `/staging/` is mounted](https://chtc.cs.wisc.edu/uw-research-computing/scaling-htc#3-submitting-jobs-to-run-beyond-chtc).

> [!WARNING]
> OSDF and Pelican treat all cached files as immutable, so each file placed on `/staging/` should have a unique name, such as including information about the repository's Git commit SHA in the file name

#### Using a remote container registry

If the `.sif` file exists on a Linux container registry then the container `.sif` file can be provided to the `container_image` argument [using the `oras://` prefix](https://htcondor.readthedocs.io/en/latest/users-manual/env-of-job.html#container-universe-jobs).

```
universe = container
container_image = oras://<container registry domain>/<account>/<container name>:<tag>
```

**Example**:

```
universe = container
container_image = oras://ghcr.io/<your github org>/templates-gpus:apptainer-mnist-gpu-noble-cuda-12.9
```

> [!TIP]
> Using the container registry approach allows the container used to be automatically built and deployed through a [continuous integration and continuous delivery (CI/CD) workflow](https://carpentries-incubator.github.io/reproducible-ml-workflows/pixi-deployment.html#automation-with-github-actions-workflows-1).
> Use of Git commit SHAs in the container image tag can aid in ensuring the software environment provided by the Apptainer container corresponds to the correct version of the code the user is running.
> Use of a Linux container registry can also help with distribution of the software environment to collaborators external to CHTC.

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
condor_submit mnist_gpu_apptainer.sub
```
