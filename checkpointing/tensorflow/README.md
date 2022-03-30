## Checkpointing in Tensorflow with conda environment and GPU support
The example uses a pre-configured conda environment to demonstrate how one might implement model checkpointing in Tensorflow.

The Tensorflow example downloads the MNIST dataset and trains a neural net on it for 20 epochs, making a "checkpoint" every fifth epoch. In the event of a runtime interruption, these checkpoints are preserved on the submit server and are used to resume training, preventing the loss of already-completed work.

Checkpointing can be useful when running jobs that take multiple days to train or when training on resources that may be interrupted, like GPUs on CHTC backfill servers, campus pools outside of CHTC, or OSG.
These additional GPU resources can be accessed by adding one or more of the following options to your submit file:
- `Is_resumable = true`: access CHTC backfill servers
- `wantFlocking = true`: access campus pools outside CHTC and CHTC backfill servers
- `wantGlideIn = true`: access OSG, campus pools outside CHTC, and CHTC backfill servers

The five files include:
- `checkpoint.sub` ## used to submit the job on HTCondor.
- `run.sh` ## the executable called by `checkpoint.sub`.
- `tf_checkpointing.py` ## the python program with an a checkpointing implementation.
- `tf_checkpointing.tar.gz` ## the conda environment used by HTCondor for tensorflow dependencies -- this is hosted on a squid web server and is not included in the repo.
- `environment.yml` ## The environment file used to create the conda env used in this example. This file is not used by the job. The environment is included in `tf_checkpointing.tar.gz`.

### Usage
- Log into the HTC system.
- Clone this repository: `git clone https://github.com/CHTC/templates-GPUs`.
- `cd` into this folder: `cd templates-GPUs/checkpointing/tensorflow`
- Because conda environments tend to be large, a Squid caching server is used to host the environment. You can view the environment `.tar.gz` file from the submit node at `/squid/gpu-examples`. For more information about using Squid, please review the CHTC guide:
[Large File Availability Via Squid](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-squid) guide.
- Submit the sample job: `condor_submit checkpoint.sub`.
- Upon completion, HTCondor will return a zipped model file, `model.tar.gz`, along with checkpoint files `checkpoint.h5` (checkpointed model) and `checkpoint.txt` (which epoch to resume training on).
- In the event of a runtime error, HTCondor will return only `checkpoint.h5` and `checkpoint.txt`, assuming it has reached at least the first checkpoint.
