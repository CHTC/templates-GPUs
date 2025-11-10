### Convolutional Neural Network with PyTorch (Python)
This example shows how to send training and test data to the compute node along
with the script. After processing the trained network is returned to the
submit node.  

### Submit file

Here the submit file stays the same as that in the [Hello\_GPU](../hello_gpu/) example with a few minor tweaks. 

We set the Docker image to version of Pytorch that is build with CUDA. 
```
docker_image = pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
```

Also, we need the Python script as well as the data transferred to the compute node. 
These files are located in the [`shared/pytorch`](../../shared/pytorch) directory
```
transfer_input_files = ../../shared/pytorch/main.py, ../../shared/pytorch/MNIST_data.tar.gz
```

The rest of the submit file remains the same.  We run the submit file with 
```shell
condor_submit pytorch_cnn.sub
```

### Execute script
The [Execute Shell script](./pytorch_cnn.sh) extracts the data and then calls a
Python script [main.py](../../shared/pytorch/main.py) that figures out the network weights and saves it to disk.
Then the Execute script deletes the data directory so that it isn't returned to the submit node. 

```shell
tar zxf MNIST_data.tar.gz
python main.py --save-model --epochs 20
rm -r data
```
 
### Output 
We have the CNN Network that was trained returned to us as a file
[mnist\_cnn.pt](./expected_output/mnist_cnn.pt). The are also some output stats
on the training and test error in the [output
files](./expected_output/pytorch_cnn.out.txt).  
```
Test set: Average loss: 0.0278, Accuracy: 9909/10000 (99%)
```

You can see a complete list of files expected in the output in the [expected
output directory](./expected_output/).
