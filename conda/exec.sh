#!/bin/bash

echo 'Date: ' `date` 
echo 'Host: ' `hostname` 
echo 'System: ' `uname -spo` 
echo 'GPU: ' `lspci | grep NVIDIA`

# Following the example from http://chtc.cs.wisc.edu/conda-installation.shtml
# except here we download the installer instead of transferring it
# Download a specific version of Miniconda instead of latest to improve
# reproducibility
export HOME=$PWD
export PATH
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Install packages specified in the environment file
conda env create -f environment.yml

# Activate the environment and log all packages that were installed
conda activate pytorch-gpu
conda list

# Modify this line to run your desired Python script
python -c "import pytorch; print(pytorch.__version__)"
