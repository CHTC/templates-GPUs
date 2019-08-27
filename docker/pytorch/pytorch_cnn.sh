#!/bin/bash
echo "Hello CHTC from Job $1 running on `hostname`"
python main.py --save-model --epochs 20
