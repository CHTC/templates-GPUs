#!/bin/bash
echo "Hello CHTC from Job $1 running on `hostname`"

# untar the test and training data
tar zxf MNIST_data.tar.gz

# run the pytorch model
python main.py --save-model --epochs 20

# remove the data directory
rm -r data
