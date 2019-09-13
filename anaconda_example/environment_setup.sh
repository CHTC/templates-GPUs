#!/usr/bin/env bash

# echo some HTCondor job information
echo _CONDOR_JOB_IWD $_CONDOR_JOB_IWD
echo Cluster $cluster
echo Process $process
echo RunningOn $runningon

# this makes it easier to set up the environments, since the PWD we are running in is not $HOME
export HOME=$PWD

# set up anaconda python environment
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3 > /dev/null
export PATH=~/miniconda3/bin:$PATH

# TODO: instead of using a whitelist, check cuda driver version and setup appropriate environment based on that
# to get cuda driver version: nvidia-smi | grep -oP "Driver Version: \K(\S*)"
# cuda driver & runtime compatibility table
# https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
# tensorflow & cuda/cudnn compatibility table
# https://www.tensorflow.org/install/source#linux


# set up appropriate cuda/cudnn/tensorflow environment depending on what server we are on
case $runningon in

  # gzk nodes only support tensorflow 1.4 due to cuda driver version
  (*gzk-1.chtc.wisc.edu|*gzk-2.chtc.wisc.edu|*gzk-3.chtc.wisc.edu|*gzk-4.chtc.wisc.edu)
    conda create --yes --name my_conda_env python=3.5
    source activate my_conda_env
    conda install -c anaconda cudatoolkit==8.0 --yes
    conda install -c anaconda cudnn==6.0.21 --yes
    pip install tensorflow-gpu==1.4

    # special workaround -- set up custom libc environment (see http://goo.gl/6iVTDZ)
    mkdir my_libc_env
    cd my_libc_env
    wget -nv http://proxy.chtc.wisc.edu/SQUID/sgelman2/libc6_2.17-0ubuntu5_amd64.deb
    wget -nv http://proxy.chtc.wisc.edu/SQUID/sgelman2/libc6-dev_2.17-0ubuntu5_amd64.deb
    wget -nv http://proxy.chtc.wisc.edu/SQUID/sgelman2/libstdc++6-4.8.2-3.2.mga4.x86_64.rpm
    ar p libc6_2.17-0ubuntu5_amd64.deb data.tar.gz | tar zx
    ar p libc6-dev_2.17-0ubuntu5_amd64.deb data.tar.gz | tar zx
    rpm2cpio libstdc++6-4.8.2-3.2.mga4.x86_64.rpm | cpio -idmv
    cd ~
    $HOME/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so --library-path $LD_LIBRARY_PATH:$HOME/my_libc_env/lib/x86_64-linux-gnu/:$HOME/my_libc_env/usr/lib64/ `which python` my_tensorflow_program.py

   ;;

  # other servers on our whitelist all have cuda >10 so can support newer versions of python, tensorflow, etc
  (*gitter0000.chtc.wisc.edu|*ahlquist0000.chtc.wisc.edu|*ahlquist2000.chtc.wisc.edu|*ahlquist2001.chtc.wisc.edu|*gpu2001.chtc.wisc.edu|*gpu2000.chtc.wisc.edu|*tekin0000.chtc.wisc.edu|*vetsigian0000.chtc.wisc.edu)
   conda create --yes --name my_conda_env python=3.6.8
   source activate my_conda_env
   conda install -c anaconda cudatoolkit==10.0.130 --yes
   conda install -c anaconda cudnn==7.6.0 --yes
   pip install tensorflow-gpu==1.14
   python my_tensorflow_program.py

  ;;

  (*)
   echo "Running on an unknown host"
   exit
  ;;

esac


