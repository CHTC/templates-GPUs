#!/bin/bash

wd=$(pwd)
mkdir data
mkdir $wd/output

unzip train.zip -d data/
rm train.zip

cd /app/

python train.py \
  --data-dir $wd/data/ \ 
  --checkpoint-dir $wd/output/
