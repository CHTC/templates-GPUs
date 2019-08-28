
### Running Hello GPU via Docker

This is a simple example to pull the [Nvidia CUDA container](https://hub.docker.com/r/nvidia/cuda/) from Dockerhub. 


### Submit file 
We set the `universe` and the `docker_image` tags to make sure CHTC knows to
pull in the right images. 

```
universe = docker
docker_image = nvidia/cuda:10.1-base-ubuntu18.04
```

We require a machine with a modern version of the CUDA driver. CUDA drivers are usually backwards compatible. So a machine with CUDA Driver version 10.1 should be able to run containers built with older versions of CUDA. 
```
Requirements = (Target.CUDADriverVersion >= 10.1)
```

We should also request a `cpu` as well as a `gpu`. 
```
request_cpus = 1
request_gpus = 1
```
[The complete submit file is available here](./hello_gpu.sub). 

```shell
condor_submit hello_gpu.sub
```

### Execute file
We run `nvidia-smi` which gives us diagnostic information about the GPU. 


### Output
The output should be similar to below. You can see a complete list of files expected in the output in the [expected output directory](./expected_output/)
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.00    Driver Version: 418.87.00    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:3B:00.0 Off |                    0 |
| N/A   28C    P0    27W / 250W |     10MiB / 16280MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P100-PCIE...  Off  | 00000000:D8:00.0 Off |                    0 |
| N/A   26C    P0    25W / 250W |     10MiB / 16280MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

