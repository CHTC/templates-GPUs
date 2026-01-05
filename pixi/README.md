# CUDA accelerated Pixi environment

> [!CAUTION]
> While this approach does work, as there is no shared file system cache that Pixi can leverage it is less efficient than building a Linux container with the Pixi environment and using a `container` universe job.
> Note also that as the jobs executable, `mnist_gpu.sh`, is installing all dependencies from a remote conda channel (conda-forge), multiple copies of the job should not be submitted to avoid intensive bandwidth demand.
>
> Look at the example in the [`containers/pixi`](https://github.com/CHTC/templates-GPUs/tree/master/containers/pixi) directory for the recommended approach for use of Pixi with HTCondor systems with no shared file system.

The example uses [Pixi](https://pixi.sh/) to create a fully reproducible PyTorch environment with CUDA support.
The environment is already fully resolved to the digest level in the `pixi.lock` Pixi lock file, so the execution script only installs Pixi on the worker and then installs the locked environment, before running the requested training scripts.
The values of `gpus_minimum_capability` and the `requirement` of `GPUs_DriverVersion` in the `mnist_gpu.sub` HTCondor submit description file ensure that the worker node GPU will be compatible with the builds of PyTorch and CUDA defined in the environment.

The PyTorch example uses the `MNIST_data.tar.gz` MNIST dataset and `main.py` Python script from the [`shared/pytorch`](https://github.com/CHTC/templates-GPUs/tree/master/shared/pytorch) directory.

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

## Use

* Log into an HTC submit node
* Clone this repository

```
git clone https://github.com/CHTC/templates-GPUs
```

* Navigate to this directory

```
cd templates-GPUs/pixi
```

* Submit the example to HTCondor

```
condor_submit mnist_gpu.sub
```

