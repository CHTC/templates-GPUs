#!/bin/bash

wd=$(pwd)
mkdir data
unzip images.zip -d data/
rm images.zip

cd /app/

python infer.py \
  --data-dir $wd/data/ \ 
  --model-path $wd/model.pth
