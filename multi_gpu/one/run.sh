#!/bin/bash
set -e

# installation steps for Miniconda
export HOME=$PWD
export PATH
sh Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/miniconda3
export PATH=$PWD/miniconda3/bin:$PATH

# install environment
conda env create -f env.yml -n env.yml
source activate base 
conda activate env.yml

#wandb login <your api key> 

python3 model.py