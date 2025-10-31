## conda environment with GPU support

The example uses Anaconda to create a Python environment with GPU support.
Specifically, it downloads Miniconda, initializes conda, creates an environment from the `environment.yml` file, and runs a PyTorch 1.5 example.
The `environment.yml` file specifies which conda packages to install into the new environment.
It also specifies that two of those packages should be obtained from the `pytorch` channel instead of the default channel.
See the [CHTC conda guide](https://chtc.cs.wisc.edu/uw-research-computing/conda-installation) for more information about alternative ways to use conda in CHTC jobs.

The PyTorch example uses the MNIST dataset and Python file from the [`shared/pytorch`](../../shared/pytorch) directory.

The submit file includes the requirement `CUDADriverVersion >= 10.2`.
The versions of CUDA available on the execute node are not used by PyTorch in this example.
The conda package `cudatoolkit` installs CUDA.
However, this requirement ensures that the execute node has a new enough NVIDIA driver to run `cudatoolkit` version 10.2.

CUDA 10.2 is not compatible with Ampere generation GPUs such as the A100.
Therefore the requirement `CUDACapability < 8` ensures the job will not run on this type of GPU.
[`CUDACapability`](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) corresponds to the GPU architecture, not a CUDA version.

### Usage
- log into an HTC submit node
- clone this repository: `git clone https://github.com/CHTC/template-GPUs`
- `cd` into this folder: `cd template-GPUs/conda/pytorch-1.5`
- submit the sample job: `condor_submit pytorch_cnn.sub`
