## Checkpointing in Tensorflow with conda environment and GPU support
The example uses a pre-configured conda environment to demonstrate how one might implement model checkpointing in Tensorflow.

The Tensorflow example downloads the MNIST dataset and trains an neural net on it for 20 epochs, making a "checkpoint" every fifth epoch. In the event of a runtime interruption, these checkpoints are returned to the user and can be used to resume training, preventing the loss of already-completed work.

The four files include:
- checkpoint.sub ## used to submit the job on HTCondor
- run.sh ## the executable that is called in checkpoint.sub
- tf_checkpointing.py ## the python program with an implementation of checkpointing
- tf_checkpointing.tar.gz ## the conda environment used by HTCondor for tensorflow dependencies -- this is hosted on a squid server and is not included in the repo.

### Usage
- log into the HTC system.
- clone this repository: `git clone https://github.com/CHTC/template-GPUs`
- `cd` into this folder: `cd template-GPUs/checkpointing/conda/tensorflow`
- because conda environments tend to be large, a Squid caching server is used to host the environment. You can access the environment .tar.gz file from the submit node at /squid/gpu-examples. For more information about using Squid, please review the CHTC guide:
[Large File Availability Via Squid](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-squid) guide.
- submit the sample job: `condor_submit checkpoint.sub`
- upon completion, HTCondor will return a zipped model file, `model.tar.gz`, along with checkpoint files `checkpoint.h5` (checkpointed model) and `checkpoint.txt` (which epoch to resume training on)
- in the event of a runtime error, HTCondor will return only `checkpoint.h5` and `checkpoint.txt`, assuming it has reached at least the first checkpoint

