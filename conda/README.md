## conda environment with GPU support

The example uses Anaconda to create a Python environment with GPU support.
Specifically, it downloads Miniconda, initializes conda, creates an environment from the `environment.yml` file, and runs a PyTorch example.
The `environment.yml` file specifies which conda packages to install into the new environment.
It also specifies that two of those packages should be obtained from the `pytorch` channel instead of the default channel.
See the [CHTC conda guide](http://chtc.cs.wisc.edu/conda-installation.shtml) for more information about alternative ways to use conda in CHTC jobs.

The PyTorch example is derived from the [Docker PyTorch example](../docker/pytorch_python) and reuses the MNIST dataset and Python files from that directory.

The submit file includes the requirement `CUDADriverVersion >= 10.2`.
The verisons of CUDA available on the execute node are not used by PyTorch in this example.
The conda package `cudatoolkit` installs CUDA.
However, this requirement ensures that the execute node has a new enough NVIDIA driver to run `cudatoolkit` version 10.2.

### Usage
- log into the HTC system.
- clone this repository: `git clone https://github.com/CHTC/template-GPUs`
- `cd` into this folder: `cd template-GPUs/conda`
- submit the sample job: `condor_submit pytorch_cnn.sub`
