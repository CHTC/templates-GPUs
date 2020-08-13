## Convolutional Neural Network with PyTorch (NVIDIA GPU Cloud)
This example shows how to use containers from the [NVIDIA GPU Cloud](https://ngc.nvidia.com/catalog/containers) (NGC) to run GPU jobs.
It uses PyTorch and is very similar to the [PyTorch Docker example](../pytorch_python) that uses a PyTorch container from DockerHub.

### Submit file
The submit file is the same as the DockerHub PyTorch example except we use the container from NGC.
```
docker_image = nvcr.io/nvidia/pytorch:19.10-py3
```

This example uses version `19.10` of the container.
The available versions, also known as tags, are listed at the [NGC PyTorch site](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags).
The NVIDIA [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) shows which versions of PyTorch and other packages are available within each versioned container.
It also specifies the version of CUDA the container is based on.
This example uses CUDA 10.1.243, Python 3.6, and PyTorch 1.3.0.

We require a GPU server with a version of CUDA that matches the version used for the container.
```
Requirements = (Target.CUDADriverVersion >= 10.1)
```
This ensures that the server has a new enough NVIDIA driver to run the container.

The Python script and input data are located in the [`shared/pytorch`](../../shared/pytorch) directory
```
transfer_input_files = ../../shared/pytorch/main.py, ../../shared/pytorch/MNIST_data.tar.gz
```

The `pytorch_cnn.sh` executable is the same as the DockerHub PyTorch example.

We run the submit file with 
```shell
condor_submit pytorch_cnn.sub
```
