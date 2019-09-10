
### Matrix Multiplication with TensorFlow (Python)

This example uses a [TensorFlow](https://www.tensorflow.org) [Docker
container](https://hub.docker.com/r/tensorflow/tensorflow/) to benchmark matrix
multiplication on a GPU vs the same matrix multiplication on a CPU. 

### Submit file

Here the submit file stays the same as that in the [Hello\_GPU](../hello_gpu/)
example with a few minor tweaks. 

We set the Docker image to TensorFlow that is GPU enabled. 
```
docker_image = tensorflow/tensorflow:1.14.0-gpu-py3
```

Also, we ask for our Python script to be transferred to the compute node. 
```
transfer_input_files = test_tensorflow.py
```

The rest of the submit file remains the same.  We run the submit file with 
```shell
condor_submit test_tensorflow.sub
```

### Execute script
The [Execute Shell script](./test_tensorflow.sh) just calls a python script
[test\_tensorflow.py](./test_tensorflow.py). This Python script that runs
matrix multiplication on the GPU and also runs it on the CPU and compares the
time difference between the GPU run and the CPU run. 

### Output
The output should be similar to below. 

``` 
 8192 x 8192 GPU matmul took: 0.12 sec, 8979.05 G ops/sec

 8192 x 8192 CPU matmul took: 1.15 sec, 952.13 G ops/sec
```

We can see that the GPU was almost 10x faster. These kinds of speed ups are not uncommon. 

You can see a complete list of files expected in the output in the [expected
output directory](./expected_output/).

### Attribution
This Python script was initially copied [from an
answer](https://stackoverflow.com/a/41810634) by [User:Yaroslav
Bulatov](https://stackoverflow.com/users/419116/yaroslav-bulatov) on the
[StackOverflow](https://stackoverflow.com/) website. The script was then further modified to run in the CHTC environment. 
