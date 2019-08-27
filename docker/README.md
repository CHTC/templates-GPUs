### Using GPUs on CHTC via Docker

Docker is software that helps bundle software programs, libraries and
dependencies in a package called a **container**. Once built, these containers
can be run on different machines that have the docker engine. HTCondor uses the
docker universe to run docker containers. These containers need to be
specifically configured to use the GPU and here we shall provide some examples
of how to do this. 

### Example Hello\_GPU

Using the [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) docker image to
see if we can talk to the gpu. Click [here](./hello_gpu) to access this example. 

### Example tensorflow (Python)

