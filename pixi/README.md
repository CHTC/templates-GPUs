# CUDA accelerated Pixi environment

> [!NOTE]
> While this approach does work, as there is no shared file system cache that Pixi can leverage it is less efficient than building a Linux container with the Pixi environment and using a `container` universe job.
> Note also that as the jobs executable, `mnist_gpu.sh`, is installing all dependencies from a remote conda channel (conda-forge), multiple copies of the job should not be submitted to avoid intensive bandwidth demand.

The example uses [Pixi](https://pixi.sh/) to create a fully reproducible PyTorch environment with CUDA support.
The environment is already fully resolved to the digest level in the `pixi.lock` Pixi lock file, so the execution script only installs Pixi on the worker and then installs the locked environment, before running the requested training scripts.
The values of `gpus_minimum_capability` and the `requirement` of `GPUs_DriverVersion` in the `mnist_gpu.sub` HTCondor submit description file ensure that the worker node GPU will be compatible with the builds of PyTorch and CUDA defined in the environment.

The PyTorch example uses the `MNIST_data.tar.gz` MNIST dataset and `main.py` Python script from the [`shared/pytorch`](https://github.com/CHTC/templates-GPUs/tree/master/shared/pytorch) directory.

## Workspace creation

The Pixi workspace in this example was created with the equivalent of the following Pixi commands (run on a `linux-64` platform machine)

```bash
pixi init
pixi workspace system-requirements add cuda 12.9
pixi add pytorch-gpu torchvision 'cuda-version 12.9.*'
pixi task add --description "Train a PyTorch CNN classifier on the MNIST dataset" train "python ./main.py --epochs 20 --save-model"
```

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
git clone https://github.com/CHTC/template-GPUs
```

* Navigate to this directory

```
cd template-GPUs/pixi
```

* Submit the example to HTCondor

```
condor_submit mnist_gpu.sub
```

