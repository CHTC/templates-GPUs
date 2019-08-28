### Using GPUs on CHTC via Docker

Docker is software that helps bundle software programs, libraries and
dependencies in a package called a **container**. Once built, these containers
can be run on different machines that have the Docker Engine. Programs with
complex dependencies are often packaged with Docker and made availble for
download on [DockerHub](https://hub.docker.com).

The docker engine needs special configuration to give the software inside a
container access to a GPU. CHTC does this behind the scenes with
`nvidia-docker`. Any Docker container that wants to use `nvidia-docker` must
contain the Nvidia CUDA toolkit inside it. Here we have working examples and
also some pointers on how to find containers and how to build your own
containers. 


### Examples 

1. *Hello\_GPU* This is a simple example to see if we can access a GPU from
   inside a Docker container on CHTC. It uses the
[nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) Docker image which is a
tiny container that only contains the Nvidia CUDA toolkit. [Click here to
access this example](./hello_gpu/). 

2. *Matrix Multiplication with TensorFlow (Python)* This example uses a
   [TensorFlow](https://www.tensorflow.org) [Docker
container](https://hub.docker.com/r/tensorflow/tensorflow/) to benchmark matrix
multiplication on a GPU vs the same matrix multiplication on a CPU. [Click here
to access this example](./tensorflow_python/). 


3. *Convolutional Neural Network with PyTorch (Python)* This example shows how
   to send training and test data to the compute node along with the script.
After processing the trained network is returned to the submit node.  [Click
here to access this example](./pytorch_python/). 
 
### Finding containers
1. Pick a container that is built on a more modern version of CUDA Toolkit. Although the toolkits are backwards compatible, the more modern the toolkit, the less likely you are to run into problems. 
2. [Rocker](https://hub.docker.com/u/rocker) is a great place to look if you are  you are looking for machine learning GPU containers for the [R Project for Statistical Computing](https://www.r-project.org)
3. [Nvidia Catalog](https://ngc.nvidia.com/catalog/landing) has a good
   selection of containers that use the GPU for machine learning, inference,
visualization etc. These containers can be uploaded to your own account on
Dockerhub. Alternatively, they can be built directly on Dockerhub with the
Docker Automated Builder (see below).  


### Building containers
Building your own containers to access a GPU requires a bit of work and will
not be described fully here. It is best to start with a basic container that
can access the GPU and then build upon that container. The PyTorch Docker
container is built on top of Nvidia Cuda and is a [good example to follow](https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile).

```Dockerfile
FROM nvidia/cuda:10.1-base-ubuntu18.04
#....
```
or
```Dockerfile
# Pull from Nvidia's catalog
FROM nvcr.io/nvidia/pytorch:19.07-py3

# conda is already installed so just install packages
RUN conda install package_1 package_2 package_etc
```

Once you have the Dockerfile, you need to build a Docker container with the Docker app and then upload it to Dockerhub so that CHTC can access your container. Alternativelt, you can have the DockerHub Cloud service directly build it for youfrom a Github repository. 
