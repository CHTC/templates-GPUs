#!/bin/bash
set -e

# installation steps for Miniconda
export HOME=$PWD
sh Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
export PATH=$HOME/miniconda3/bin:$PATH
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no

# install environment
conda env create -f env.yml -n multigpu
conda activate multigpu

#wandb login <your api key>

# Edit the argument after -n to reflect how many GPUs were requested in the submit file
python3 model_parallel.py -n 1

