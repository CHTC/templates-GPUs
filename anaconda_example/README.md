# Introduction
This barebones example shows how to run TensorFlow GPU jobs on condor without using a Docker container. It uses Anaconda to install the necessary cuda libraries. Simply upload to a submit node and call `condor_submit gpu_job.sub` to run the example.

# Supported Servers
This example uses a whitelist and supports different environment setups depending on the specific server it is running on. Currently, the following servers are supported:
- ahlquist0000.chtc.wisc.edu
- ahlquist2000.chtc.wisc.edu
- ahlquist2001.chtc.wisc.edu
- gitter0000.chtc.wisc.edu
- gpu2001.chtc.wisc.edu
- gpu2000.chtc.wisc.edu
- gzk-1.chtc.wisc.edu
- gzk-2.chtc.wisc.edu
- gzk-3.chtc.wisc.edu
- gzk-4.chtc.wisc.edu
- tekin0000.chtc.wisc.edu
- vetsigian0000.chtc.wisc.edu

New servers can be added in [gpu_job.sub](gpu_job.sub) and [environment_setup.sh](environment_setup.sh). In the future, this example could be updated to support more general GPU requirements instead of a server whitelist. 

# GZK Workaround
The GZK servers require a special workaround due to an older OS version (see [environment_setup.sh](environment_setup.sh)). Additionally, they are only capable of running up to TensorFlow 1.4 due to older CUDA drivers. You can choose not to run on these servers by removing them from [gpu_job.sub](gpu_job.sub).